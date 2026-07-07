# density_generalization_benchmark.py
"""
Density-transfer benchmark for GNN-assisted SDRG.

Train setting:
    N = 80, L = 800, density = 0.1, alpha = 2.0

Test setting:
    N = 80 fixed, alpha = 2.0 fixed,
    density changed to 0.10, 0.15, 0.20, 0.25, 0.30.

This tests whether the trained GNN transfers to different spin densities.
"""

import os
import csv
import json
from pathlib import Path

import numpy as np
import torch
import matplotlib.pyplot as plt
from torch_geometric.data import Data
from mpl_toolkits.axes_grid1.inset_locator import inset_axes

from model import SDRGNet
from checkpoint import load_checkpoint
from utils import generate_positions, initial_couplings


# -------------------------
# Settings
# -------------------------
CHECKPOINT_PATH = "results/ablation/checkpoints/checkpoint_full.pt"

N = 80
ALPHA = 2.0
DENSITIES = [0.10, 0.15, 0.20, 0.25, 0.30]
N_REALIZATIONS = 1000

RESULTS_ROOT = Path("results/density_generalization")
RESULTS_ROOT.mkdir(parents=True, exist_ok=True)


def sdrg_pairing(positions, J):
    active = list(range(len(positions)))
    pairs = []
    J = J.copy()

    while len(active) > 1:
        i, j = max(
            [(i, j) for (i, j) in J if i in active and j in active],
            key=lambda x: J[x],
        )

        pairs.append((positions[i], positions[j]))

        active.remove(i)
        active.remove(j)

        J = {
            (k, l): v
            for (k, l), v in J.items()
            if k in active and l in active
        }

    return pairs


def build_graph_from_state(positions, J, active):
    active = sorted(active)
    spin_map = {s: i for i, s in enumerate(active)}

    node_features = torch.ones((len(active), 1), dtype=torch.float)

    edge_index = [[], []]
    edge_attr = []
    edge_list = []

    logJ_by_site = {i: [] for i in active}

    for i in active:
        for j in active:
            if i < j:
                logJ = np.log(abs(J[(i, j)]) + 1e-12)
                logJ_by_site[i].append(logJ)
                logJ_by_site[j].append(logJ)

    for i in active:
        for j in active:
            if i < j:
                edge_index[0].append(spin_map[i])
                edge_index[1].append(spin_map[j])

                logJ = np.log(abs(J[(i, j)]) + 1e-12)
                logR = np.log(abs(positions[i] - positions[j]) + 1e-12)

                neigh_i = sorted(logJ_by_site[i], reverse=True)[:4]
                neigh_j = sorted(logJ_by_site[j], reverse=True)[:4]
                local_mean = np.mean(neigh_i + neigh_j)

                rel_strength = logJ - local_mean

                edge_attr.append([logJ, logR, rel_strength])
                edge_list.append((i, j))

    data = Data(
        x=node_features,
        edge_index=torch.tensor(edge_index, dtype=torch.long),
        edge_attr=torch.tensor(edge_attr, dtype=torch.float),
    )

    return data, edge_list


def load_trained_model(sample_data):
    model = SDRGNet(
        node_dim=sample_data.x.shape[1],
        edge_dim=sample_data.edge_attr.shape[1],
        hidden_dim=64,
    )

    load_checkpoint(CHECKPOINT_PATH, model)
    model.eval()

    return model


def ml_sdrg_pairing(positions, J, model):
    active = list(range(len(positions)))
    pairs = []
    J = J.copy()

    while len(active) > 1:
        data, edge_list = build_graph_from_state(positions, J, active)

        with torch.no_grad():
            logits = model(data)
            pred_edge = logits.argmax().item()

        i, j = edge_list[pred_edge]
        pairs.append((positions[i], positions[j]))

        active.remove(i)
        active.remove(j)

        J = {
            (k, l): v
            for (k, l), v in J.items()
            if k in active and l in active
        }

    return pairs


def entanglement_entropy(pairs, L):
    S = np.zeros(L)

    for ell in range(L):
        crossings = 0

        for r1, r2 in pairs:
            if (r1 < ell < r2) or (r2 < ell < r1):
                crossings += 1

        S[ell] = np.log(2.0) * crossings

    return S


def pairing_accuracy(exact_pairs, ml_pairs):
    def norm(p):
        return tuple(sorted(p))

    exact_set = set(norm(p) for p in exact_pairs)
    ml_set = set(norm(p) for p in ml_pairs)

    matched = len(exact_set.intersection(ml_set))
    n_spins = 2 * len(exact_pairs)

    return 2.0 * matched / n_spins


def plot_entropy(
    S_exact_mean,
    S_exact_sem,
    S_ml_mean,
    S_ml_sem,
    rP_all,
    L,
    out_png,
):
    ell = np.arange(1, L + 1)

    rP_all = np.asarray(rP_all)
    rP_mean = np.mean(rP_all)
    rP_std = np.std(rP_all, ddof=1)

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


def evaluate_density(density):
    L = int(round(N / density))

    outdir = RESULTS_ROOT / f"N{N}_L{L}_rho{density:.2f}_alpha{ALPHA}"
    outdir.mkdir(parents=True, exist_ok=True)

    print("\n" + "=" * 80)
    print(f"Density transfer: N={N}, L={L}, rho={density:.2f}, alpha={ALPHA}")
    print("=" * 80)

    # Build sample graph to initialize model dimensions.
    positions = generate_positions(N, L)
    J = initial_couplings(positions, ALPHA)
    data, _ = build_graph_from_state(positions, J, list(range(N)))
    model = load_trained_model(data)

    S_exact_all = []
    S_ml_all = []
    rP_all = []

    for r in range(N_REALIZATIONS):
        if r % 50 == 0:
            print(f"  realization {r}/{N_REALIZATIONS}")

        positions = generate_positions(N, L)
        J = initial_couplings(positions, ALPHA)

        exact_pairs = sdrg_pairing(positions, J)
        ml_pairs = ml_sdrg_pairing(positions, J, model)

        rP_all.append(pairing_accuracy(exact_pairs, ml_pairs))
        S_exact_all.append(entanglement_entropy(exact_pairs, L))
        S_ml_all.append(entanglement_entropy(ml_pairs, L))

    S_exact_all = np.array(S_exact_all)
    S_ml_all = np.array(S_ml_all)
    rP_all = np.array(rP_all)

    S_exact_mean = np.mean(S_exact_all, axis=0)
    S_exact_std = np.std(S_exact_all, axis=0, ddof=1)
    S_exact_sem = S_exact_std / np.sqrt(N_REALIZATIONS)

    S_ml_mean = np.mean(S_ml_all, axis=0)
    S_ml_std = np.std(S_ml_all, axis=0, ddof=1)
    S_ml_sem = S_ml_std / np.sqrt(N_REALIZATIONS)

    rP_mean = float(np.mean(rP_all))
    rP_std = float(np.std(rP_all, ddof=1))
    rP_sem = float(rP_std / np.sqrt(N_REALIZATIONS))

    result = {
        "N": N,
        "L": L,
        "density": density,
        "alpha": ALPHA,
        "n_realizations": N_REALIZATIONS,
        "r_P_mean": rP_mean,
        "r_P_std": rP_std,
        "r_P_sem": rP_sem,
        "r_P_all": rP_all.tolist(),
        "S_exact_mean": S_exact_mean.tolist(),
        "S_exact_std": S_exact_std.tolist(),
        "S_exact_sem": S_exact_sem.tolist(),
        "S_ml_mean": S_ml_mean.tolist(),
        "S_ml_std": S_ml_std.tolist(),
        "S_ml_sem": S_ml_sem.tolist(),
    }

    json_path = outdir / "density_generalization.json"
    fig_path = outdir / f"entropy_density_rho{density:.2f}.png"

    with open(json_path, "w") as f:
        json.dump(result, f, indent=2)

    plot_entropy(
        S_exact_mean=S_exact_mean,
        S_exact_sem=S_exact_sem,
        S_ml_mean=S_ml_mean,
        S_ml_sem=S_ml_sem,
        rP_all=rP_all,
        L=L,
        out_png=fig_path,
    )

    print(f"r_P = {rP_mean:.3f} ± {rP_std:.3f} std")
    print(f"r_P = {rP_mean:.3f} ± {rP_sem:.3f} sem")
    print(f"Saved JSON:   {json_path}")
    print(f"Saved figure: {fig_path}")

    return {
        "N": N,
        "L": L,
        "density": density,
        "alpha": ALPHA,
        "n_realizations": N_REALIZATIONS,
        "r_P_mean": rP_mean,
        "r_P_std": rP_std,
        "r_P_sem": rP_sem,
        "json_path": str(json_path),
        "fig_path": str(fig_path),
    }


def main():
    summary_rows = []

    for density in DENSITIES:
        row = evaluate_density(density)
        summary_rows.append(row)

    summary_path = RESULTS_ROOT / "summary.csv"

    with open(summary_path, "w", newline="") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=[
                "N",
                "L",
                "density",
                "alpha",
                "n_realizations",
                "r_P_mean",
                "r_P_std",
                "r_P_sem",
                "json_path",
                "fig_path",
            ],
        )
        writer.writeheader()
        writer.writerows(summary_rows)

    print("\nSaved density-transfer summary:")
    print(summary_path)


if __name__ == "__main__":
    main()