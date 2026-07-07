# deterministic_baseline.py
"""
Deterministic strongest-bond baseline.

Purpose
-------
Referee 3 asked whether the learning problem reduces to the known SDRG rule:
select the bond with the largest effective coupling.

This script evaluates the deterministic baseline directly from the saved JSON
dataset. Since the labels are generated using the strongest-bond rule and the
feature logJ = log(|J_ij|) is included in the full-feature dataset, this baseline
should achieve accuracy close to 1.0 on the full-feature dataset.

Important
---------
This script assumes FEATURE_MODE = "full", where edge_features are:

    [logJ, logR, rel_strength]

For other feature modes, logJ may not be present.
"""

import json
from pathlib import Path
import numpy as np


DATA_DIR = Path("data")


def collect_json_files(data_dir):
    files = []
    for rdir in sorted(data_dir.iterdir()):
        if not rdir.is_dir():
            continue
        files.extend(sorted(rdir.glob("*.json")))
    return files


def evaluate_strongest_logJ_baseline():
    files = collect_json_files(DATA_DIR)

    if not files:
        raise RuntimeError("No JSON files found. Run generate_data_train.py first.")

    correct = 0
    total = 0

    for path in files:
        with open(path, "r") as f:
            sample = json.load(f)

        edge_features = np.array(sample["edge_features"], dtype=float)
        edge_mask = np.array(sample["edge_mask"], dtype=bool)
        target = int(sample["target_edge"])

        # Canonical physical edges are the ones with edge_mask == True.
        physical_features = edge_features[edge_mask]

        # In full mode, column 0 is logJ.
        logJ = physical_features[:, 0]

        pred = int(np.argmax(logJ))

        correct += int(pred == target)
        total += 1

    acc = correct / total

    print("Deterministic strongest-logJ baseline")
    print("------------------------------------")
    print(f"Samples:  {total}")
    print(f"Correct:  {correct}")
    print(f"Accuracy: {acc:.6f}")


if __name__ == "__main__":
    evaluate_strongest_logJ_baseline()



"""
do this 

>> echo 'FEATURE_MODE = "full"' > feature_config.py
>> rm -rf data
>> python generate_data_train.py
>> python deterministic_baseline.py


"""