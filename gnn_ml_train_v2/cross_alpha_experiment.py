# cross_alpha_experiment.py
"""
Cross-alpha transfer experiment for GNN-assisted SDRG.

Purpose
-------
We fix N=80 and L=800, i.e. density N/L = 0.1.

We train separate GNNs at selected alpha values, then evaluate each trained
model on several test alpha values.

This answers:
1. Does a model trained at alpha=2.0 transfer to alpha=0.5?
2. Does a model trained at alpha=0.5 transfer to alpha=2.0?
3. Are the entanglement curves stable and comparable across alpha?
4. How does the final pairing accuracy r_P depend on train/test alpha?

Outputs
-------
results/cross_alpha/
    checkpoints/
    data_alpha*/
    json/
    figures/
    summary.csv
"""

import os
import csv
import json
import shutil
import subprocess
from pathlib import Path
import sys

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from sdrg_ground_state.sdrg_entropy import sdrg_pairing

import numpy as np
import torch
import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1.inset_locator import inset_axes
from torch_geometric.data import Data

from utils import generate_positions, initial_couplings
from sdrg import strongest_bond, decimate
from json_writer import build_step_json
from model import SDRGNet


# ============================================================
# Settings
# ============================================================

N = 80
L = 800
DENSITY = 0.1

ALPHA_TRAIN_LIST = [0.5, 2.0]
ALPHA_TEST_LIST = [0.5, 1.0, 2.0, 3.0]

N_TRAIN_REALIZATIONS = 200
N_EVAL_REALIZATIONS = 100

FEATURE_MODE = "full"

HIDDEN_DIM = 64
NUM_LAYERS = 3

RESULTS_DIR = Path("results/cross_alpha")
CHECKPOINT_DIR = RESULTS_DIR / "checkpoints"
JSON_DIR = RESULTS_DIR / "json"
FIG_DIR = RESULTS_DIR / "figures"
LOG_DIR = RESULTS_DIR / "logs"

for d in [CHECKPOINT_DIR, JSON_DIR, FIG_DIR, LOG_DIR]:
    d.mkdir(parents=True, exist_ok=True)


# ============================================================
# Config writers
# ============================================================

def write_feature_config():
    with open("feature_config.py", "w") as f:
        f.write(f'FEATURE_MODE = "{FEATURE_MODE}"\n')


def write_config(alpha, data_dir):
    """
    Overwrite config.py for training-data generation.

    This assumes your generate_data_train.py reads:
        DATA_DIR, N_SPINS, N_REALIZATIONS, LATTICE_SIZE, ALPHA
    """
    with open("config.py", "w") as f:
        f.write(f'DATA_DIR = "{data_dir}"\n')
        f.write(f"N_SPINS = {N}\n")
        f.write(f"N_REALIZATIONS = {N_TRAIN_REALIZATIONS}\n")
        f.write(f"LATTICE_SIZE = {L}\n")
        f.write(f"ALPHA = {alpha}\n")


def run_command(command, log_file):
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


# ============================================================
# Training
# ============================================================

def train_model_for_alpha(alpha_train):
    """
    Generate data at alpha_train and train one GNN.
    """
    print("\n" + "=" * 80)
    print(f"Training model at alpha_train={alpha_train}")
    print("=" * 80)

    write_feature_config()

    data_dir = f"data_alpha{alpha_train}"
    write_config(alpha_train, data_dir)

    if Path(data_dir).exists():
        shutil.rmtree(data_dir)

    checkpoint_name = f"checkpoint_full.pt"
    if Path(checkpoint_name).exists():
        Path(checkpoint_name).unlink()

    log_file = LOG_DIR / f"train_alpha{alpha_train}.log"
    if log_file.exists():
        log_file.unlink()

    run_command("python generate_data_train.py", log_file)

    # train_with_validation.py currently loads SDRGDataset(root='data')
    # Therefore temporarily symlink/copy generated alpha data to ./data.
    if Path("data").exists():
        shutil.rmtree("data")

    shutil.copytree(data_dir, "data")

    run_command("python train_with_validation.py", log_file)

    # Move checkpoint.
    ckpt_src_candidates = [
        Path("checkpoint_full.pt"),
        Path("checkpoint.pt"),
    ]

    ckpt_src = None
    for c in ckpt_src_candidates:
        if c.exists():
            ckpt_src = c
            break

    if ckpt_src is None:
        raise RuntimeError("No checkpoint found after training.")

    ckpt_dst = CHECKPOINT_DIR / f"checkpoint_train_alpha{alpha_train}.pt"
    if ckpt_dst.exists():
        ckpt_dst.unlink()

    shutil.move(str(ckpt_src), str(ckpt_dst))

    print(f"Saved checkpoint: {ckpt_dst}")

    return ckpt_dst


# ============================================================
# Model loading
# ============================================================

def load_model(checkpoint_path):
    model = SDRGNet(
        node_dim=1,
        edge_dim=3,
        hidden_dim=HIDDEN_DIM,
        num_layers=NUM_LAYERS,
        dropout=0.0,
    )

    checkpoint = torch.load(checkpoint_path, map_location="cpu")

    if isinstance(checkpoint, dict):
        if "model_state" in checkpoint:
            model.load_state_dict(checkpoint["model_state"])
        elif "model_state_dict" in checkpoint:
            model.load_state_dict(checkpoint["model_state_dict"])
        elif "state_dict" in checkpoint:
            model.load_state_dict(checkpoint["state_dict"])
        else:
            model.load_state_dict(checkpoint)
    else:
        raise RuntimeError("Unknown checkpoint format.")

    model.eval()
    return model


def sample_to_data(sample):
    return Data(
        x=torch.tensor(sample["node_features"], dtype=torch.float),
        edge_index=torch.tensor(sample["edge_index"], dtype=torch.long),
        edge_attr=torch.tensor(sample["edge_features"], dtype=torch.float),
        edge_mask=torch.tensor(sample["edge_mask"], dtype=torch.bool),
        y=torch.tensor(sample["target_edge"], dtype=torch.long),
    )


def physical_edges_from_active(active_spins):
    active_sorted = sorted(active_spins)
    edges = []

    for i in active_sorted:
        for j in active_sorted:
            if i < j:
                edges.append((i, j))

    return edges


# ============================================================
# SDRG / GNN rollout
# ============================================================

# def run_exact_sdrg(J_init, active_init):
#     J = dict(J_init)
#     active_spins = list(active_init)
#     pairs = []

#     while len(active_spins) > 1:
#         i, j = strongest_bond(J, active_spins)
#         pairs.append((i, j))
#         J, active_spins = decimate(J, active_spins, i, j)

#     return pairs

def run_exact_sdrg(positions, J_init):
    """
    Original SDRG entropy pipeline.
    """
    return sdrg_pairing(positions, J_init)


# def run_gnn_rollout(J_init, positions, active_init, model):
#     J = dict(J_init)
#     active_spins = list(active_init)
#     pairs = []

#     with torch.no_grad():
#         while len(active_spins) > 1:
#             placeholder_target = strongest_bond(J, active_spins)

#             sample = build_step_json(
#                 J=J,
#                 positions=positions,
#                 active_spins=active_spins,
#                 target_edge=placeholder_target,
#             )

#             data = sample_to_data(sample)
#             scores = model(data)

#             pred_idx = int(torch.argmax(scores).item())
#             physical_edges = physical_edges_from_active(active_spins)

#             i, j = physical_edges[pred_idx]

#             pairs.append((i, j))
#             J, active_spins = decimate(J, active_spins, i, j)

#     return pairs


def run_gnn_rollout(J_init, positions, active_init, model):
    """
    Original ML rollout used in the old entropy analysis.

    The GNN selects bonds sequentially, but no perturbative
    coupling renormalization is applied after decimation.
    """
    J = dict(J_init)
    active_spins = list(active_init)

    pairs = []

    with torch.no_grad():
        while len(active_spins) > 1:
            placeholder_target = strongest_bond(J, active_spins)

            sample = build_step_json(
                J=J,
                positions=positions,
                active_spins=active_spins,
                target_edge=placeholder_target,
            )

            data = sample_to_data(sample)
            scores = model(data)

            pred_idx = int(torch.argmax(scores).item())
            physical_edges = physical_edges_from_active(active_spins)

            i, j = physical_edges[pred_idx]

            # Store PHYSICAL POSITIONS (old convention)
            pairs.append((positions[i], positions[j]))

            # Remove spins
            active_spins.remove(i)
            active_spins.remove(j)

            # Keep only surviving couplings
            J = {
                (k, l): v
                for (k, l), v in J.items()
                if k in active_spins and l in active_spins
            }

    return pairs


def run_teacher_forced_step_accuracy(J_init, positions, active_init, model):
    J = dict(J_init)
    active_spins = list(active_init)

    correct = 0
    total = 0

    with torch.no_grad():
        while len(active_spins) > 1:
            target = strongest_bond(J, active_spins)

            sample = build_step_json(
                J=J,
                positions=positions,
                active_spins=active_spins,
                target_edge=target,
            )

            data = sample_to_data(sample)
            scores = model(data)

            pred_idx = int(torch.argmax(scores).item())
            physical_edges = physical_edges_from_active(active_spins)
            pred_edge = physical_edges[pred_idx]

            correct += int(pred_edge == target)
            total += 1

            J, active_spins = decimate(J, active_spins, target[0], target[1])

    return correct, total


# ============================================================
# Entropy and metrics
# ============================================================

# def pairing_accuracy(exact_pairs, ml_pairs, N):
#     exact_set = {tuple(sorted(p)) for p in exact_pairs}
#     ml_set = {tuple(sorted(p)) for p in ml_pairs}

#     matched = len(exact_set.intersection(ml_set))
#     return 2.0 * matched / N

def pairing_accuracy(exact_pairs, ml_pairs, N):
    def norm_pair(p):
        return tuple(sorted(p))

    exact_set = set(norm_pair(p) for p in exact_pairs)
    ml_set = set(norm_pair(p) for p in ml_pairs)

    matched = len(exact_set.intersection(ml_set))

    return 2.0 * matched / N

# def compute_entropy_from_pairs_index_cut(pairs, positions, L):
#     """
#     Entanglement entropy for physical cuts ell=1,...,L.

#     A pair contributes ln(2) if the two spin positions are on opposite sides
#     of the cut.
#     """
#     cuts = np.arange(1, L + 1)
#     S = np.zeros(L, dtype=float)

#     for i, j in pairs:
#         x1 = min(positions[i], positions[j])
#         x2 = max(positions[i], positions[j])

#         crossing = (cuts > x1) & (cuts <= x2)
#         S[crossing] += np.log(2.0)

#     return S


def compute_entropy_from_pairs(pairs, positions, L):
    """
    Original entropy convention used in previous figures.
    """
    S = np.zeros(L)

    for ell in range(L):
        crossings = 0

        for r1, r2 in pairs:
            if (r1 < ell < r2) or (r2 < ell < r1):
                crossings += 1

        S[ell] = np.log(2.0) * crossings

    return S

def sem(arr, axis=0):
    arr = np.asarray(arr)
    return np.std(arr, axis=axis, ddof=1) / np.sqrt(arr.shape[axis])


# ============================================================
# Plot
# ============================================================

def plot_result(json_path, out_png):
    with open(json_path, "r") as f:
        data = json.load(f)

    L = data["L"]
    ell = np.arange(1, L + 1)

    S_exact_mean = np.array(data["S_exact_mean"])
    S_exact_sem = np.array(data["S_exact_sem"])
    S_ml_mean = np.array(data["S_ml_mean"])
    S_ml_sem = np.array(data["S_ml_sem"])

    rP_all = np.array(data["r_P_all"])
    rP_mean = data["r_P_mean"]
    rP_std = data["r_P_std"]

    fig, ax = plt.subplots(figsize=(6, 4))

    ax.plot(ell, S_exact_mean, label="SDRG", linewidth=2)
    ax.fill_between(
        ell,
        S_exact_mean - S_exact_sem,
        S_exact_mean + S_exact_sem,
        alpha=0.25,
    )

    ax.plot(ell, S_ml_mean, "--", label="GNN-SDRG", linewidth=2)
    ax.fill_between(
        ell,
        S_ml_mean - S_ml_sem,
        S_ml_mean + S_ml_sem,
        alpha=0.25,
    )

    ax.set_xlabel(r"$\ell$")
    ax.set_ylabel(r"$S(\ell)$")
    ax.legend(loc="best")

    ax.text(
        0.05,
        0.95,
        rf"$r_P = {rP_mean:.3f} \pm {rP_std:.3f}$",
        transform=ax.transAxes,
        fontsize=11,
        verticalalignment="top",
        bbox=dict(boxstyle="round", facecolor="white", alpha=0.85),
    )

    ax_hist = inset_axes(
        ax,
        width="35%",
        height="35%",
        loc="center",
        borderpad=1,
    )

    ax_hist.hist(rP_all, bins=30, density=True, alpha=0.8)
    ax_hist.axvline(rP_mean, linestyle="--", linewidth=1)

    ax_hist.set_xlabel(r"$r_P$", fontsize=8)
    ax_hist.set_ylabel("PDF", fontsize=8)
    ax_hist.tick_params(axis="both", labelsize=8)
    ax_hist.set_xlim(0, 1)

    plt.tight_layout()
    plt.savefig(out_png, dpi=160)
    plt.close()


# ============================================================
# Evaluation
# ============================================================

def evaluate_model(alpha_train, alpha_test, checkpoint_path):
    print("\n" + "=" * 80)
    print(f"Evaluating train alpha={alpha_train} on test alpha={alpha_test}")
    print("=" * 80)

    model = load_model(checkpoint_path)

    exact_entropies = []
    ml_entropies = []
    rP_all = []

    total_correct = 0
    total_steps = 0

    for realization in range(N_EVAL_REALIZATIONS):
        if realization % 10 == 0:
            print(f"  realization {realization}/{N_EVAL_REALIZATIONS}")

        positions = generate_positions(N, L)
        J_init = initial_couplings(positions, alpha_test)
        active_init = list(range(N))

        # exact_pairs = run_exact_sdrg(J_init, active_init)
        ml_pairs = run_gnn_rollout(J_init, positions, active_init, model)

        exact_pairs = run_exact_sdrg(positions, J_init)

        correct, steps = run_teacher_forced_step_accuracy(
            J_init=J_init,
            positions=positions,
            active_init=active_init,
            model=model,
        )

        total_correct += correct
        total_steps += steps

        rP = pairing_accuracy(exact_pairs, ml_pairs, N)
        rP_all.append(rP)

        # exact_entropies.append(
        #     compute_entropy_from_pairs_index_cut(exact_pairs, positions, L)
        # )

        # ml_entropies.append(
        #     compute_entropy_from_pairs_index_cut(ml_pairs, positions, L)
        # )

        exact_entropies.append(
            compute_entropy_from_pairs(exact_pairs, positions, L)
        )

        ml_entropies.append(
            compute_entropy_from_pairs(ml_pairs, positions, L)
        )

    exact_entropies = np.array(exact_entropies)
    ml_entropies = np.array(ml_entropies)
    rP_all = np.array(rP_all)

    step_accuracy = total_correct / total_steps

    result = {
        "N": N,
        "L": L,
        "density": DENSITY,
        "alpha_train": alpha_train,
        "alpha_test": alpha_test,
        "n_eval_realizations": N_EVAL_REALIZATIONS,
        "step_accuracy": float(step_accuracy),
        "r_P_mean": float(np.mean(rP_all)),
        "r_P_std": float(np.std(rP_all, ddof=1)),
        "r_P_all": rP_all.tolist(),
        "S_exact_mean": np.mean(exact_entropies, axis=0).tolist(),
        "S_exact_sem": sem(exact_entropies, axis=0).tolist(),
        "S_ml_mean": np.mean(ml_entropies, axis=0).tolist(),
        "S_ml_sem": sem(ml_entropies, axis=0).tolist(),
    }

    json_path = JSON_DIR / f"train_alpha{alpha_train}_test_alpha{alpha_test}.json"
    fig_path = FIG_DIR / f"train_alpha{alpha_train}_test_alpha{alpha_test}.png"

    with open(json_path, "w") as f:
        json.dump(result, f, indent=2)

    plot_result(json_path, fig_path)

    print(f"Step accuracy: {step_accuracy:.3f}")
    print(f"r_P:           {result['r_P_mean']:.3f} ± {result['r_P_std']:.3f}")
    print(f"Saved JSON:    {json_path}")
    print(f"Saved figure:  {fig_path}")

    return result


# ============================================================
# Main
# ============================================================

def main():
    summary_rows = []

    for alpha_train in ALPHA_TRAIN_LIST:
        checkpoint_path = train_model_for_alpha(alpha_train)

        for alpha_test in ALPHA_TEST_LIST:
            result = evaluate_model(
                alpha_train=alpha_train,
                alpha_test=alpha_test,
                checkpoint_path=checkpoint_path,
            )

            summary_rows.append({
                "N": result["N"],
                "L": result["L"],
                "density": result["density"],
                "alpha_train": result["alpha_train"],
                "alpha_test": result["alpha_test"],
                "n_eval_realizations": result["n_eval_realizations"],
                "step_accuracy": result["step_accuracy"],
                "r_P_mean": result["r_P_mean"],
                "r_P_std": result["r_P_std"],
            })

    summary_path = RESULTS_DIR / "summary.csv"

    with open(summary_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=summary_rows[0].keys())
        writer.writeheader()
        writer.writerows(summary_rows)

    print("\nSaved summary:")
    print(summary_path)


if __name__ == "__main__":
    main()