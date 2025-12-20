
import os
import json
import torch
import numpy as np
import matplotlib.pyplot as plt

from torch_geometric.data import Data

from gnn_ml_train.model import SDRGNet
from gnn_ml_train.checkpoint import load_checkpoint

# Exact SDRG utilities
from sdrg_ground_state.sdrg_entropy import sdrg_pairing
from utils import generate_positions, initial_couplings

## ============================================================
## Graph construction (shared by ML-SDRG)
## ============================================================
def build_graph_from_state(positions, J, active):
    """
    Construct a fully-connected PyG graph for the current RG state.
    Nodes: active spins
    Edges: all pairs (i,j) with features [log|J|, log|r|, relative strength]
    """
    active = sorted(active)
    spin_map = {s: i for i, s in enumerate(active)}

    x = torch.ones((len(active), 1), dtype=torch.float)

    edge_index = [[], []]
    edge_attr = []
    edge_list = []

    # Precompute local logJ statistics
    logJ_by_site = {i: [] for i in active}
    for i in active:
        for j in active:
            if i < j:
                lj = np.log(abs(J[(i, j)]))
                logJ_by_site[i].append(lj)
                logJ_by_site[j].append(lj)

    for i in active:
        for j in active:
            if i < j:
                edge_index[0].append(spin_map[i])
                edge_index[1].append(spin_map[j])

                logJ = np.log(abs(J[(i, j)]))
                logR = np.log(abs(positions[i] - positions[j]))

                neigh_i = sorted(logJ_by_site[i], reverse=True)[:4]
                neigh_j = sorted(logJ_by_site[j], reverse=True)[:4]
                local_mean = np.mean(neigh_i + neigh_j)

                rel_strength = logJ - local_mean

                edge_attr.append([logJ, logR, rel_strength])
                edge_list.append((i, j))

    data = Data(
        x=x,
        edge_index=torch.tensor(edge_index, dtype=torch.long),
        edge_attr=torch.tensor(edge_attr, dtype=torch.float),
    )

    return data, edge_list

## ============================================================
## Load trained GNN model
## ============================================================
def load_trained_model(sample_data, checkpoint_path="../gnn_ml_train/checkpoint.pt"):
    model = SDRGNet(
        node_dim=sample_data.x.shape[1],
        edge_dim=sample_data.edge_attr.shape[1],
        hidden_dim=64,
    )
    load_checkpoint(checkpoint_path, model)
    model.eval()
    return model

## ============================================================
## Exact & ML SDRG with bond-length tracking (RG flow)
## ============================================================
def sdrg_pairing_with_lengths(positions, J):
    """
    Exact SDRG: return list of decimated bond lengths vs RG step.
    """
    active = list(range(len(positions)))
    J = J.copy()
    lengths = []

    while len(active) > 1:
        (i, j), _ = max(J.items(), key=lambda x: abs(x[1]))
        lengths.append(abs(positions[i] - positions[j]))

        active.remove(i)
        active.remove(j)

        J = {(k, l): v for (k, l), v in J.items()
             if k in active and l in active}

    return lengths

def ml_sdrg_pairing_with_lengths(positions, J, model):
    """
    ML-assisted SDRG: return list of decimated bond lengths vs RG step.
    """
    active = list(range(len(positions)))
    J = J.copy()
    lengths = []

    while len(active) > 1:
        data, edge_list = build_graph_from_state(positions, J, active)

        with torch.no_grad():
            scores = model(data)
            idx = scores.argmax().item()

        i, j = edge_list[idx]
        lengths.append(abs(positions[i] - positions[j]))

        active.remove(i)
        active.remove(j)

        J = {(k, l): v for (k, l), v in J.items()
             if k in active and l in active}

    return lengths

## ============================================================
## Collect RG-flow data over disorder
## ============================================================

def collect_rg_flow_data(N, L, alpha, n_realizations, model):
    exact_all = []
    ml_all = []

    for r in range(n_realizations):
        print(f"RG flow realization {r}")
        positions = generate_positions(N, L)
        J = initial_couplings(positions, alpha)

        exact_all.append(sdrg_pairing_with_lengths(positions, J))
        ml_all.append(ml_sdrg_pairing_with_lengths(positions, J, model))

    return np.array(exact_all), np.array(ml_all)

## ============================================================
## RG-flow heatmap
## ============================================================
def rg_flow_heatmap(lengths_all, L, n_bins=40):
    """
    Build normalized heatmap:
      x-axis: RG step
      y-axis: bond length (log bins)
    """
    n_real, n_steps = lengths_all.shape
    bins = np.logspace(np.log10(1), np.log10(L), n_bins + 1)

    heatmap = np.zeros((n_bins, n_steps))

    for k in range(n_steps):
        hist, _ = np.histogram(lengths_all[:, k], bins=bins)
        heatmap[:, k] = hist

    heatmap /= heatmap.sum(axis=0, keepdims=True) + 1e-12
    return heatmap, bins


def plot_rg_flow(exact_map, ml_map, outdir):
    fig, axes = plt.subplots(1, 2, figsize=(12, 4), sharey=True)

    for ax, data, title in zip(
        axes,
        [exact_map, ml_map],
        ["Exact SDRG", "GNN-SDRG"]
    ):
        im = ax.imshow(data, origin="lower", aspect="auto", cmap="viridis")
        ax.set_title(title)
        ax.set_xlabel("RG step")

    axes[0].set_ylabel("Bond length (log bins)")
    fig.colorbar(im, ax=axes, label="Probability")
    plt.tight_layout()
    plt.savefig(os.path.join(outdir, "rg_flow_heatmap.png"), dpi=200)
    plt.close()



## ============================================================
## Pairing accuracy r_P
## ============================================================
def pairing_accuracy(exact_pairs, ml_pairs):
    """
    r_P = fraction of identical (unordered) pairs.
    """
    def norm(p): return tuple(sorted(p))

    exact_set = set(norm(p) for p in exact_pairs)
    ml_set = set(norm(p) for p in ml_pairs)

    M_P = len(exact_set & ml_set)
    N = 2 * len(exact_pairs)

    return 2 * M_P / N




## ============================================================
## Main execution
## ============================================================


if __name__ == "__main__":
    N = 80
    L = 800
    alpha = 2.0
    n_realizations = 1000

    # Load model
    positions = generate_positions(N, L)
    J = initial_couplings(positions, alpha)
    data, _ = build_graph_from_state(positions, J, list(range(N)))
    model = load_trained_model(data)

    # RG-flow heatmaps
    exact_all, ml_all = collect_rg_flow_data(
        N, L, alpha, n_realizations, model
    )

    outdir = "rg_flow_results"
    os.makedirs(outdir, exist_ok=True)

    np.save(os.path.join(outdir, "exact_lengths.npy"), exact_all)
    np.save(os.path.join(outdir, "ml_lengths.npy"), ml_all)


    exact_map, _ = rg_flow_heatmap(exact_all, L)
    ml_map   , _ = rg_flow_heatmap(ml_all, L)

    np.save(os.path.join(outdir, "exact_heatmap.npy"), exact_map)
    np.save(os.path.join(outdir, "ml_heatmap.npy"), ml_map)


    plot_rg_flow(exact_map, ml_map, outdir)

