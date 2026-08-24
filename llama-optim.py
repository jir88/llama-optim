import argparse
import csv
import datetime
import itertools
import json
import random
import subprocess
import time
from math import exp
from pathlib import Path

import requests


def load_config(path):
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def build_param_grid(optimize_dict, strategy, random_trials):
    keys = list(optimize_dict.keys())
    values = list(optimize_dict.values())

    if strategy == "grid":
        for combo in itertools.product(*values):
            yield dict(zip(keys, combo))

    elif strategy == "random":
        for _ in range(random_trials):
            combo = {k: random.choice(v) for k, v in optimize_dict.items()}
            yield combo

    else:
        raise ValueError(f"Unknown search strategy for grid/random: {strategy}")


def start_server(server_path, server_port, model_path, params):
    cmd = [server_path, "-m", model_path, "--port", str(server_port)]

    for key, value in params.items():
        flag = f"--{key.replace('_', '-')}"
        if isinstance(value, bool):
            if value:
                cmd.append(flag)
        else:
            cmd.extend([flag, str(value)])

    print("Launching:", " ".join(cmd))

    return subprocess.Popen(
        cmd,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE
    )


def wait_for_server_ready(server_port, timeout=20):
    start = time.time()
    while time.time() - start < timeout:
        try:
            r = requests.get("http://localhost:" + str(server_port) + "/health")
            if r.status_code == 200:
                return True
        except requests.HTTPError as error:
            print("Failed to wait for server ready:\n", error)
        time.sleep(0.5)
    return False


def benchmark(server_port, prompt, tokens, warmup_seconds, measure_seconds):
    start = time.time()
    r = requests.post(
        "http://localhost:" + str(server_port) + "/completion",
        json={"prompt": prompt, "n_predict": tokens, "ignore_eos": True}
    )
    end = time.time()

    if r.status_code != 200:
        print(str(r))
        return None

    data = r.json()
    tok_count = data.get("tokens_predicted", tokens)
    elapsed = end - start

    return {
        "tokens_per_second": tok_count / elapsed,
        "latency": elapsed
    }


def stop_server(proc):
    proc.terminate()
    try:
        proc.wait(timeout=5)
    except subprocess.TimeoutExpired:
        proc.kill()


def init_csv(path, optimize_keys):
    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        header = optimize_keys + ["tokens_per_second", "latency"]
        writer.writerow(header)


def append_csv(path, params, result, optimize_keys):
    with open(path, "a", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        row = [params[k] for k in optimize_keys] + [
            result["tokens_per_second"],
            result["latency"]
        ]
        writer.writerow(row)


def bayes_sample(optimize_params, history, beta=0.3):
    keys = list(optimize_params.keys())

    if not history:
        return {k: random.choice(v) for k, v in optimize_params.items()}

    scores = [h["result"]["tokens_per_second"] for h in history]
    max_s = max(scores)
    weights = [exp(beta * (s - max_s)) for s in scores]
    total_w = sum(weights)
    probs = [w / total_w for w in weights]

    r = random.random()
    cum = 0.0
    parent = history[-1]["params"]
    for p, h in zip(probs, history):
        cum += p
        if r <= cum:
            parent = h["params"]
            break

    child = dict(parent)
    num_mutations = max(1, len(keys) // 2)
    mutate_keys = random.sample(keys, num_mutations)

    for k in mutate_keys:
        values = optimize_params[k]
        current = child[k]
        idx = values.index(current)
        neighbors = [idx]
        if idx > 0:
            neighbors.append(idx - 1)
        if idx < len(values) - 1:
            neighbors.append(idx + 1)
        new_idx = random.choice(neighbors)
        child[k] = values[new_idx]

    return child


def run_single_config(server_path, server_port, model_path, static_params, combo,
                      benchmark_cfg, csv_path, optimize_keys, best, history):
    params = static_params.copy()
    params.update(combo)

    try:
        proc = start_server(server_path, server_port, model_path, params)

        print("Waiting for server to launch...")
        if not wait_for_server_ready(server_port=server_port, timeout=benchmark_cfg["warmup_seconds"]):
            print("Server failed to start, skipping configuration.")
            stop_server(proc)
            return best
        print("Server launched!")

        print("Running benchmark...")
        result = benchmark(
            server_port=server_port,
            prompt=benchmark_cfg["prompt"],
            tokens=benchmark_cfg["tokens"],
            warmup_seconds=benchmark_cfg["warmup_seconds"],
            measure_seconds=benchmark_cfg["measure_seconds"]
        )
    except KeyboardInterrupt:
        stop_server(proc=proc)
        print("Killed server before terminating.")

    stop_server(proc)

    if result is None:
        print("Benchmark failed.")
        return best

    print(f"Result for {combo}: {result}")
    append_csv(csv_path, combo, result, optimize_keys)

    entry = {"params": combo, "result": result}
    history.append(entry)

    if best is None or result["tokens_per_second"] > best["result"]["tokens_per_second"]:
        best = entry

    return best

def parse_args() -> argparse.Namespace:
    """
    Set up command line argument parsing.
    """
    parser = argparse.ArgumentParser(description=(
        "Semi-automatically optimize the parameters for a llama.cpp server."
    ))
    parser.add_argument("config", type=Path, help="Path to the configuration file to use.")
    parser.add_argument(
        "--output", type=Path, default="results.csv", required=False,
        help="Output file name to use.")
    parser.add_argument(
        "--overwrite", action="store_true",
        help=(
            "If enabled, output CSV file will be overwritten if it exists. Otherwise "
            "a timestamp will be added to the output file to avoid overwriting."
        )
    )
    return parser.parse_args()

def main():
    # automatically parse the command-line arguments
    args = parse_args()

    # try loading the configuration file
    config_path = args.config
    
    if not config_path.exists():
        raise FileNotFoundError(f"Error: Topic file not found: {config_path}")
    config = load_config(config_path)

    model_path = config["model_path"]
    server_port = config["server_port"]
    server_path = config["server_path"]
    static_params = config.get("static_params", {})
    optimize_params = config.get("optimize", {})
    benchmark_cfg = config["benchmark"]
    search_cfg = config["search"]

    # set up output CSV file with header
    csv_path = args.output
    if not args.overwrite and csv_path.exists():
        # put timestamp on input file name
        timestamp = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
        csv_path = csv_path.parent.joinpath(csv_path.stem + "_" + timestamp + ".csv")
    optimize_keys = list(optimize_params.keys())
    init_csv(csv_path, optimize_keys)

    strategy = search_cfg["strategy"]
    best = None
    history = []

    if strategy in ("grid", "random"):
        for combo in build_param_grid(
            optimize_params,
            strategy,
            search_cfg.get("random_trials", 10)
        ):
            best = run_single_config(
                server_path,
                server_port,
                model_path,
                static_params,
                combo,
                benchmark_cfg,
                csv_path,
                optimize_keys,
                best,
                history
            )

    elif strategy == "bayes":
        trials = search_cfg.get("bayes_trials", 30)
        for _ in range(trials):
            combo = bayes_sample(optimize_params, history)
            best = run_single_config(
                server_path,
                server_port,
                model_path,
                static_params,
                combo,
                benchmark_cfg,
                csv_path,
                optimize_keys,
                best,
                history
            )
    else:
        raise ValueError(f"Unknown search strategy: {strategy}")

    print("\n=== BEST CONFIGURATION ===")
    print(json.dumps(best, indent=2))


if __name__ == "__main__":
    main()
