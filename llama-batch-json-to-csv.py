import csv
import json
import sys


def json_to_tidy_csv(json_path, csv_path):
    # Load JSON list
    with open(json_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    # Identify list-valued keys (assume lists always same length per object)
    list_keys = []
    for obj in data:
        for k, v in obj.items():
            if isinstance(v, list):
                list_keys.append(k)
        break  # only need to inspect first object

    # Scalar keys = everything except list-valued keys
    scalar_keys = [k for k in data[0] if k not in list_keys]

    # CSV columns: scalar fields + one column per list key
    fieldnames = scalar_keys + list_keys

    rows = []

    for obj in data:
        # Determine number of rows to expand
        list_len = len(next(iter(obj[k] for k in list_keys)))
        
        for i in range(list_len):
            row = {}

            # Copy scalar fields
            for k in scalar_keys:
                row[k] = obj[k]

            # Insert list-indexed values
            for k in list_keys:
                row[k] = obj[k][i]

            rows.append(row)

    # Write CSV
    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


if __name__ == "__main__":
    if len(sys.argv) != 3:
        print("Usage: python llama-batch-json-to-csv.py input.json output.csv")
        sys.exit(1)

    json_to_tidy_csv(sys.argv[1], sys.argv[2])
