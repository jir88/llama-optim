import argparse
import datetime
import json
import subprocess
import time
from pathlib import Path

import optuna
import requests


def load_config(path):
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


class InferenceObjective:

    server_path:str
    """File path to the llama-server executable we will be starting."""
    server_port:int
    """Port to have the server listen on."""
    model_path:str
    """Path to model file we will be running on the server."""
    static_params:dict
    """Server parameters that we are not optimizing, but want to specify."""
    optimize_params:dict
    """Server parameters that we will be optimizing."""
    benchmark_cfg:dict
    """Settings for the optimization itself."""

    def __init__(
        self, server_path:str, server_port:int, model_path:str, static_params:dict, 
        optimize_params:dict, benchmark_cfg:dict, search_cfg):
        # set up parameters and such
        self.server_path = server_path
        self.server_port = server_port
        self.model_path = model_path
        self.static_params = static_params
        self.optimize_params = optimize_params
        self.benchmark_cfg = benchmark_cfg


    def __call__(self, trial:optuna.Trial):
        """Run a single trial using class parameters."""

        params = self.static_params.copy()

        # select new set of parameters to test
        for k in self.optimize_params:
            # if this is too cumbersome or fails on mixed types, could also
            # consider selecting an index within the list of options
            params[k] = trial.suggest_categorical(
                name=k,
                choices=self.optimize_params[k]
            )

        try:
            proc = self.start_server(params)

            print("Waiting for server to launch...")
            if not self.wait_for_server_ready(timeout=self.benchmark_cfg["warmup_seconds"]):
                print("Server failed to start, skipping configuration.")
                self.stop_server(proc)
                return None
            print("Server launched!")

            print("Running benchmark...")
            result = self.benchmark(
                prompt=self.benchmark_cfg["prompt"],
                tokens=self.benchmark_cfg["tokens"],
                warmup_seconds=self.benchmark_cfg["warmup_seconds"],
                measure_seconds=self.benchmark_cfg["measure_seconds"]
            )
        except KeyboardInterrupt:
            self.stop_server(proc=proc)
            print("Killed server before terminating.")

        self.stop_server(proc)

        if result is None:
            print("Benchmark failed.")
            return None

        print(f"Result: {result}")
        trial.set_user_attr("prompt_per_second", result["prompt_per_second"])
        trial.set_user_attr("tokens_predicted", result["tokens_predicted"])
        trial.set_user_attr("draft_n", result["draft_n"])
        trial.set_user_attr("draft_n_accepted", result["draft_n_accepted"])
        # currently only optimizing TPS
        return result["tokens_per_second"]
    
    def start_server(self, params:dict):
        cmd = [self.server_path, "-m", self.model_path, "--port", str(self.server_port)]

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
    
    def wait_for_server_ready(self, timeout=20):
        start = time.time()
        while time.time() - start < timeout:
            try:
                r = requests.get("http://localhost:" + str(self.server_port) + "/health")
                if r.status_code == 200:
                    return True
            except requests.HTTPError as error:
                print("Failed to wait for server ready:\n", error)
            time.sleep(0.5)
        return False
    
    def stop_server(self, proc):
        proc.terminate()
        try:
            proc.wait(timeout=5)
        except subprocess.TimeoutExpired:
            proc.kill()
    
    def benchmark(self, prompt, tokens, warmup_seconds, measure_seconds):
        start = time.time()
        r = requests.post(
            "http://localhost:" + str(self.server_port) + "/completion",
            json={"prompt": prompt, "n_predict": tokens, "ignore_eos": True}
        )
        end = time.time()

        if r.status_code != 200:
            print(str(r))
            return None

        data = r.json()
        timing_data = data.get("timings")

        print(str(data))
        tok_count = data.get("tokens_predicted", tokens)
        elapsed = end - start

        return {
            "tokens_per_second": timing_data["predicted_per_second"],
            "tokens_predicted": tok_count,
            "prompt_per_second": timing_data.get("prompt_per_second"),
            "draft_n": timing_data.get("draft_n", -1),
            "draft_n_accepted": timing_data.get("draft_n_accepted", -1),
            "latency": elapsed
        }

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

    # set up output CSV file name
    csv_path = args.output
    if not args.overwrite and csv_path.exists():
        # put timestamp on input file name
        timestamp = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")  # noqa: DTZ005
        csv_path = csv_path.parent.joinpath(csv_path.stem + "_" + timestamp + ".csv")

    objective = InferenceObjective(
        server_path=server_path,
        server_port=server_port,
        model_path=model_path,
        static_params=static_params,
        optimize_params=optimize_params,
        benchmark_cfg=benchmark_cfg,
        search_cfg=search_cfg
    )

    optim_study = optuna.create_study(
        sampler=optuna.samplers.TPESampler(n_startup_trials=5),
        direction="maximize"
    )
    optim_study.optimize(
        func=objective,
        n_trials=search_cfg.get("n_trials", 10),
        n_jobs=1
    )

    print("\n=== BEST CONFIGURATION ===")
    best = optim_study.best_trials
    for t in best:
        print(t)
    
    df = optim_study.trials_dataframe()
    df.to_csv(csv_path)


if __name__ == "__main__":
    main()
