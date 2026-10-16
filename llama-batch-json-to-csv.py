import json
import sys

import pandas as pd


def json_to_csv(json_path, csv_path):
    # Load JSON list
    with open(json_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    df = pd.DataFrame(data=data)

    list_cols = [
        col for col in df.columns 
        if df[col].dtype == 'object' and isinstance(df[col].dropna().iloc[0], list)
    ]
    df = df.explode(list_cols)

    # Write CSV
    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        df.to_csv(f)

if __name__ == "__main__":
    if len(sys.argv) != 3:
        print("Usage: python llama-batch-json-to-csv.py input.json output.csv")
        sys.exit(1)

    json_to_csv(sys.argv[1], sys.argv[2])