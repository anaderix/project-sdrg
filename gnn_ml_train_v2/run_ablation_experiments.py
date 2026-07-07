# run_ablation_experiments.py
"""
Run SDRG-GNN feature ablation experiments.

Why this script exists
----------------------
The JSON dataset depends on FEATURE_MODE because edge features are written
to disk during data generation. Therefore, for each ablation we must:

1. Set FEATURE_MODE.
2. Delete the old data directory.
3. Regenerate the dataset with the selected feature set.
4. Train a fresh GNN.
5. Save the checkpoint under a feature-specific name.
6. Save the final output log for reproducibility.

This script automates that workflow so that the ablation study can be
reproduced later and reported in the paper.
"""

import os
import shutil
import subprocess
from pathlib import Path


FEATURE_MODES = [
    "full",
    "no_logJ",
    "no_rel",
    "only_logJ",
    "only_logR",
    "only_rel",
]

RESULTS_DIR = Path("results/ablation")
LOG_DIR = RESULTS_DIR / "logs"
CHECKPOINT_DIR = RESULTS_DIR / "checkpoints"


def write_feature_config(feature_mode: str):
    """
    Write feature_config.py before each experiment.

    json_writer.py reads FEATURE_MODE from this file when generating data.
    train_with_validation.py also reads FEATURE_MODE to name checkpoints.
    """
    with open("feature_config.py", "w") as f:
        f.write(f'FEATURE_MODE = "{feature_mode}"\n')


def run_command(command, log_file):
    """
    Run shell command and stream output both to terminal and log file.
    """
    with open(log_file, "a") as log:
        process = subprocess.Popen(
            command,
            shell=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
        )

        for line in process.stdout:
            print(line, end="")
            log.write(line)

        process.wait()

        if process.returncode != 0:
            raise RuntimeError(f"Command failed: {command}")


def clean_previous_run(feature_mode: str):
    """
    Remove old generated data and old checkpoint for this feature mode.
    """
    if Path("data").exists():
        shutil.rmtree("data")

    checkpoint = Path(f"checkpoint_{feature_mode}.pt")
    if checkpoint.exists():
        checkpoint.unlink()


def move_checkpoint(feature_mode: str):
    """
    Move checkpoint into results directory after training.
    """
    checkpoint = Path(f"checkpoint_{feature_mode}.pt")

    if checkpoint.exists():
        target = CHECKPOINT_DIR / checkpoint.name
        shutil.move(str(checkpoint), str(target))
    else:
        print(f"Warning: checkpoint not found for {feature_mode}")


def main():
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    LOG_DIR.mkdir(parents=True, exist_ok=True)
    CHECKPOINT_DIR.mkdir(parents=True, exist_ok=True)

    summary_file = RESULTS_DIR / "summary.txt"

    with open(summary_file, "w") as f:
        f.write("SDRG-GNN feature ablation experiments\n")
        f.write("====================================\n\n")

    for feature_mode in FEATURE_MODES:
        print("\n" + "=" * 80)
        print(f"Running ablation: {feature_mode}")
        print("=" * 80)

        log_file = LOG_DIR / f"{feature_mode}.log"

        with open(log_file, "w") as log:
            log.write(f"Feature mode: {feature_mode}\n")
            log.write("=" * 80 + "\n\n")

        write_feature_config(feature_mode)
        clean_previous_run(feature_mode)

        run_command("python generate_data_train.py", log_file)
        run_command("python train_with_validation.py", log_file)

        move_checkpoint(feature_mode)

        with open(summary_file, "a") as f:
            f.write(f"{feature_mode}: see logs/{feature_mode}.log\n")

    print("\nAll ablation experiments finished.")
    print(f"Logs saved in: {LOG_DIR}")
    print(f"Checkpoints saved in: {CHECKPOINT_DIR}")


if __name__ == "__main__":
    main()