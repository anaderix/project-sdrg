# generalization_benchmark.py
"""
Generalization benchmark for GNN-assisted SDRG.

Purpose
-------
This script addresses referee comments about:

1. Generalization:
   - train at one system size, test at unseen N
   - train at one alpha, test at unseen alpha

2. Error analysis:
   - compute disorder-averaged entanglement entropy
   - compute standard error of the mean

Outputs
-------
For each test case, the script saves:

results/generalization/json/generalization_N*_alpha*.json
results/generalization/figures/generalization_N*_alpha*.png
results/generalization/summary.csv

The JSON files contain:
- mean SDRG entropy
- SEM SDRG entropy
- mean GNN-SDRG entropy
- SEM GNN-SDRG entropy
- step accuracy
- r_P distribution
- r_P mean and std
"""


import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))


import csv
import json
from pathlib import Path
from xml.parsers.expat import model

import numpy as np
import torch
import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1.inset_locator import inset_axes
from torch_geometric.data import Data

#from gnn_ml_train_v2.cross_alpha_experiment import run_exact_sdrg
from utils import generate_positions, initial_couplings
from sdrg import strongest_bond, decimate
from json_writer import build_step_json
from model import SDRGNet

from sdrg_ground_state.sdrg_entropy import sdrg_pairing


# ============================================================
# User settings
# ============================================================

CHECKPOINT_PATH = "results/ablation/checkpoints/checkpoint_full.pt"

# Model was trained at N=80, alpha=2.0 in the main experiment.
# Here we test transfer to unseen N and unseen alpha.
TEST_CASES = [
    {"N": 80,  "alpha": 2.0},   # in-distribution reference
    {"N": 100, "alpha": 2.0},   # larger N
    {"N": 120, "alpha": 2.0},   # larger N
    {"N": 80,  "alpha": 0.5},   # unseen alpha
    {"N": 80,  "alpha": 1.0},   # unseen alpha
    {"N": 80,  "alpha": 3.0},   # unseen alpha
]

DENSITY = 0.1
N_REALIZATIONS = 100

HIDDEN_DIM = 64
NUM_LAYERS = 3

RESULTS_DIR = Path("results/generalization")
JSON_DIR = RESULTS_DIR / "json"
FIG_DIR = RESULTS_DIR / "figures"
SUMMARY_CSV = RESULTS_DIR / "summary.csv"

JSON_DIR.mkdir(parents=True, exist_ok=True)
FIG_DIR.mkdir(parents=True, exist_ok=True)


# ============================================================
# Utility functions
# ============================================================

def load_model(path):
    model = SDRGNet(
        node_dim=1,
        edge_dim=3,
        hidden_dim=HIDDEN_DIM,
        num_layers=NUM_LAYERS,
        dropout=0.0,
    )

    checkpoint = torch.load(path, map_location="cpu")

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

def compute_entropy_from_pairs(pairs, positions, L):
    """
    Old entropy convention.

    Here pairs are stored as physical positions, not spin indices.
    """
    S = np.zeros(L)

    for ell in range(L):
        crossings = 0

        for r1, r2 in pairs:
            if (r1 < ell < r2) or (r2 < ell < r1):
                crossings += 1

        S[ell] = np.log(2.0) * crossings

    return S

# def compute_entropy_from_pairs(pairs, positions, L):
#     """
#     T=0 entanglement entropy for all cuts ell = 1,...,L.

#     A singlet pair contributes ln(2) if it crosses the cut.
#     """
#     cuts = np.arange(1, L + 1)
#     S = np.zeros(L, dtype=float)

#     for i, j in pairs:
#         x1 = min(positions[i], positions[j])
#         x2 = max(positions[i], positions[j])

#         crossing = (cuts > x1) & (cuts <= x2)
#         S[crossing] += np.log(2.0)

#     return S


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
    Old reference calculation: use the original SDRG ground-state pairing routine.
    """
    return sdrg_pairing(positions, J_init)

def run_teacher_forced_step_accuracy(J_init, positions, active_init, model):
    """
    Evaluate step accuracy along the exact SDRG trajectory.

    At each step:
    - build graph from exact current J
    - compare model-selected edge to exact strongest bond
    - then advance using exact SDRG decimation
    """
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


# def run_gnn_rollout(J_init, positions, active_init, model):
#     """
#     Run full GNN-assisted SDRG rollout.

#     The model chooses the bond at each step, then the standard SDRG coupling
#     update is applied to the chosen bond.
#     """
#     J = dict(J_init)
#     active_spins = list(active_init)

#     pairs = []

#     with torch.no_grad():
#         while len(active_spins) > 1:
#             # Placeholder target is needed only because build_step_json expects it.
#             # It is not used for inference.
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
    Old ML-SDRG rollout:
    the GNN selects bonds, decimated spins are removed,
    but no perturbative coupling renormalization is applied.
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

            pairs.append((positions[i], positions[j]))

            active_spins.remove(i)
            active_spins.remove(j)

            J = {
                (k, l): v
                for (k, l), v in J.items()
                if k in active_spins and l in active_spins
            }

    return pairs


# def pairing_accuracy(exact_pairs, ml_pairs, N):
#     exact_set = {tuple(sorted(p)) for p in exact_pairs}
#     ml_set = {tuple(sorted(p)) for p in ml_pairs}

#     matched = len(exact_set.intersection(ml_set))
#     return 2.0 * matched / N

def pairing_accuracy(exact_pairs, ml_pairs, N):
    def norm(p):
        return tuple(sorted(p))

    exact_set = set(norm(p) for p in exact_pairs)
    ml_set = set(norm(p) for p in ml_pairs)

    matched = len(exact_set.intersection(ml_set))
    return 2.0 * matched / N

def sem(arr, axis=0):
    arr = np.asarray(arr)
    return np.std(arr, axis=axis, ddof=1) / np.sqrt(arr.shape[axis])


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
    ax.legend()

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
    plt.savefig(out_png, dpi=150)
    plt.close()


# ============================================================
# Main benchmark
# ============================================================

def evaluate_case(model, N, alpha):
    L = int(N / DENSITY)

    print("\n" + "=" * 80)
    print(f"Generalization test: N={N}, L={L}, alpha={alpha}")
    print("=" * 80)

    exact_entropies = []
    ml_entropies = []
    rP_all = []

    total_correct = 0
    total_steps = 0

    for realization in range(N_REALIZATIONS):
        if realization % 10 == 0:
            print(f"  realization {realization}/{N_REALIZATIONS}")

        positions = generate_positions(N, L)
        J_init = initial_couplings(positions, alpha)
        active_init = list(range(N))

        # exact_pairs = run_exact_sdrg(J_init, active_init)
        # ml_pairs = run_gnn_rollout(J_init, positions, active_init, model)

        exact_pairs = run_exact_sdrg(positions, J_init)
        ml_pairs = run_gnn_rollout(J_init, positions, active_init, model)

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
        "alpha": alpha,
        "n_realizations": N_REALIZATIONS,
        "step_accuracy": float(step_accuracy),
        "r_P_mean": float(np.mean(rP_all)),
        "r_P_std": float(np.std(rP_all, ddof=1)),
        "r_P_all": rP_all.tolist(),
        "S_exact_mean": np.mean(exact_entropies, axis=0).tolist(),
        "S_exact_sem": sem(exact_entropies, axis=0).tolist(),
        "S_ml_mean": np.mean(ml_entropies, axis=0).tolist(),
        "S_ml_sem": sem(ml_entropies, axis=0).tolist(),
    }

    json_path = JSON_DIR / f"generalization_N{N}_alpha{alpha}.json"
    png_path = FIG_DIR / f"generalization_N{N}_alpha{alpha}.png"

    with open(json_path, "w") as f:
        json.dump(result, f, indent=2)

    plot_result(json_path, png_path)

    print(f"Step accuracy: {step_accuracy:.3f}")
    print(f"r_P:           {np.mean(rP_all):.3f} ± {np.std(rP_all, ddof=1):.3f}")
    print(f"Saved JSON:    {json_path}")
    print(f"Saved figure:  {png_path}")

    return result


def main():
    model = load_model(CHECKPOINT_PATH)

    summary_rows = []

    for case in TEST_CASES:
        result = evaluate_case(
            model=model,
            N=case["N"],
            alpha=case["alpha"],
        )

        summary_rows.append({
            "N": result["N"],
            "L": result["L"],
            "density": result["density"],
            "alpha": result["alpha"],
            "n_realizations": result["n_realizations"],
            "step_accuracy": result["step_accuracy"],
            "r_P_mean": result["r_P_mean"],
            "r_P_std": result["r_P_std"],
        })

    with open(SUMMARY_CSV, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=summary_rows[0].keys())
        writer.writeheader()
        writer.writerows(summary_rows)

    print("\nSaved summary:")
    print(SUMMARY_CSV)


if __name__ == "__main__":
    main()


"""
>> echo 'FEATURE_MODE = "full"' > feature_config.py
>> python generalization_benchmark.py
"""