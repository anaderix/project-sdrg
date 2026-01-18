import torch
import numpy as np
import json
import os
import matplotlib.pyplot as plt

from gnn_ml_train.model import SDRGNet
from gnn_ml_train.checkpoint import load_checkpoint

# exact SDRG utilities
from sdrg_ground_state.sdrg_entropy import sdrg_pairing
from utils import initial_couplings, generate_positions

from torch_geometric.data import Data
from mpl_toolkits.axes_grid1.inset_locator import inset_axes


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
        x=node_features,
        edge_index=torch.tensor(edge_index, dtype=torch.long),
        edge_attr=torch.tensor(edge_attr, dtype=torch.float),
    )

    return data, edge_list


def load_trained_model(sample_data, checkpoint_path="gnn_ml_train/checkpoint.pt"):
    model = SDRGNet(
        node_dim=sample_data.x.shape[1],
        edge_dim=sample_data.edge_attr.shape[1],
        hidden_dim=64
    )
    load_checkpoint(checkpoint_path, model)
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





def plot_entropy(S_exact, S_ml, L, outdir, rP_all=None):
    l = np.arange(1, L + 1)

    fig, ax = plt.subplots(figsize=(6, 4))

    ax.plot(l, S_exact, label="Exact SDRG", linewidth=2)
    ax.plot(l, S_ml, "--", label="ML-SDRG", linewidth=2)

    ax.set_xlabel(r"$\ell$")
    ax.set_ylabel(r"$S(\ell)$")
    ax.legend()

    # ===============================
    # Inset 1: r_P mean ± std (text)
    # ===============================
    if rP_all is not None:
        rP_mean = np.mean(rP_all)
        rP_std = np.std(rP_all)

        text = rf"$r_P = {rP_mean:.3f} \pm {rP_std:.3f}$"

        ax.text(
            0.05, 0.95,
            text,
            transform=ax.transAxes,
            fontsize=11,
            verticalalignment="top",
            bbox=dict(boxstyle="round", facecolor="white", alpha=0.8)
        )

    # ===============================
    # Inset 2: histogram of r_P
    # ===============================
    if rP_all is not None:
        ax_hist = inset_axes(
            ax,
            width="35%",   # relative to parent
            height="35%",
            loc="lower center",
            borderpad=1
        )

        ax_hist.hist(
            rP_all,
            bins=25,
            density=True,
            alpha=0.8
        )

        ax_hist.set_xlabel(r"$r_P$", fontsize=8)
        ax_hist.set_ylabel("PDF", fontsize=8)
        ax_hist.tick_params(axis="both", labelsize=8)

        ax_hist.axvline(
            rP_mean,
            linestyle="--",
            linewidth=1
        )

    plt.tight_layout()
    plt.savefig(os.path.join(outdir, "entropy_linear.png"), dpi=150)
    plt.close()



def entanglement_entropy(pairs, L):
    S = np.zeros(L)
    for l in range(L):
        crossings = 0
        for r1, r2 in pairs:
            if (r1 < l < r2) or (r2 < l < r1):
                crossings += 1
        S[l] = np.log(2) * crossings
    return S




def pairing_accuracy(exact_pairs, ml_pairs):
    def norm(p):
        return tuple(sorted(p))

    exact_set = set(norm(p) for p in exact_pairs)
    ml_set = set(norm(p) for p in ml_pairs)

    M_P = len(exact_set.intersection(ml_set))
    N = 2 * len(exact_pairs)

    return 2 * M_P / N





def compare_entropy(
    N=80,
    L=800,
    alpha=2.0,
    n_realizations=200,
    outdir="entropy_results"
):
    os.makedirs(outdir, exist_ok=True)

    S_exact_all = []
    S_ml_all = []
    rP_all = []

    positions = generate_positions(N, L)
    J = initial_couplings(positions, alpha)
    data, _ = build_graph_from_state(positions, J, list(range(N)))
    model = load_trained_model(data)

    for r in range(n_realizations):
        print(f"Realization {r}")

        positions = generate_positions(N, L)
        J = initial_couplings(positions, alpha)

        exact_pairs = sdrg_pairing(positions, J)
        ml_pairs = ml_sdrg_pairing(positions, J, model)

        rP_all.append(pairing_accuracy(exact_pairs, ml_pairs))
        S_exact_all.append(entanglement_entropy(exact_pairs, L))
        S_ml_all.append(entanglement_entropy(ml_pairs, L))

    S_exact = np.mean(S_exact_all, axis=0)
    S_ml = np.mean(S_ml_all, axis=0)

    rP_mean = float(np.mean(rP_all))
    rP_std = float(np.std(rP_all))

    data_out = {
        "N": N,
        "L": L,
        "alpha": alpha,
        "n_realizations": n_realizations,
        "r_P_mean": rP_mean,
        "r_P_std": rP_std,
        "r_P_all": rP_all,          # ← ADD THIS
        "S_exact": S_exact.tolist(),
        "S_ml": S_ml.tolist(),
    }


    json_path = os.path.join(outdir, "entropy_ml_vs_exact.json")
    with open(json_path, "w") as f:
        json.dump(data_out, f, indent=2)

    print(f"\nPairing accuracy r_P = {rP_mean:.3f} ± {rP_std:.3f}")
    print(f"Saved entropy data to {json_path}")

    plot_entropy(
    S_exact,
    S_ml,
    L,
    outdir,
    rP_all=rP_all        # ← THIS WAS MISSING
)



if __name__ == "__main__":
    compare_entropy(
        N=80,
        L=800,
        alpha=0.5,
        n_realizations=1000
    )

