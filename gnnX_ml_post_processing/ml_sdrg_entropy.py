import torch
import numpy as np
import json
import os
import matplotlib.pyplot as plt

from gnn_ml_train.model import SDRGNet
from gnn_ml_train.checkpoint import load_checkpoint

from .utils import initial_couplings, generate_positions
from torch_geometric.data import Data


# ------------------------------------------------------------
# Graph construction (same as your current code)
# ------------------------------------------------------------
def build_graph_from_state(positions, J, active):
    active = sorted(active)
    spin_map = {s: i for i, s in enumerate(active)}

    node_features = torch.ones((len(active), 1), dtype=torch.float)

    edge_index = [[], []]
    edge_attr = []
    edge_list = []

    # Precompute neighborhood logJ for rel_strength
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
                dist = abs(positions[i] - positions[j])
                logR = np.log(dist) if dist > 0 else -1e9

                neigh_i = sorted(logJ_by_site[i], reverse=True)[:4]
                neigh_j = sorted(logJ_by_site[j], reverse=True)[:4]
                local_mean = np.mean(neigh_i + neigh_j)

                rel_strength = logJ - local_mean

                edge_attr.append([logJ, logR, rel_strength])
                edge_list.append((i, j))

    data = Data(
        x=torch.tensor(node_features, dtype=torch.float),
        edge_index=torch.tensor(edge_index, dtype=torch.long),
        edge_attr=torch.tensor(edge_attr, dtype=torch.float),
    )
    return data, edge_list


def load_trained_model(sample_data, checkpoint_path="gnn_ml_train_new/checkpoint.pt"):
    model = SDRGNet(
        node_dim=sample_data.x.shape[1],
        edge_dim=sample_data.edge_attr.shape[1],
        hidden_dim=64,
    )
    load_checkpoint(checkpoint_path, model)
    model.eval()
    return model


# ------------------------------------------------------------
# Greedy "exact" pairing (matches your dataset generation style)
# returns pairs with the bond strength at decimation
# ------------------------------------------------------------
def exact_greedy_pairing_with_J(positions, J):
    active = list(range(len(positions)))
    pairs = []
    J_work = J.copy()

    while len(active) > 1:
        # strongest bond among active spins
        i, j = max(
            [(a, b) for (a, b) in J_work if a in active and b in active],
            key=lambda x: abs(J_work[x]),
        )

        Jij = J_work[(i, j)]
        pairs.append((positions[i], positions[j], float(Jij)))

        active.remove(i)
        active.remove(j)

        # greedy decimation: remove bonds touching i or j
        J_work = {(k, l): v for (k, l), v in J_work.items() if k in active and l in active}

    return pairs


# ------------------------------------------------------------
# ML SDRG pairing using GNN (same flow), but keep Jij
# ------------------------------------------------------------
def ml_sdrg_pairing_with_J(positions, J, model):
    active = list(range(len(positions)))
    pairs = []
    J_work = J.copy()

    while len(active) > 1:
        data, edge_list = build_graph_from_state(positions, J_work, active)
        with torch.no_grad():
            logits = model(data)
            pred_edge = int(logits.argmax().item())

        i, j = edge_list[pred_edge]
        Jij = J_work[(i, j)]
        pairs.append((positions[i], positions[j], float(Jij)))

        active.remove(i)
        active.remove(j)

        # greedy decimation: remove bonds touching i or j
        J_work = {(k, l): v for (k, l), v in J_work.items() if k in active and l in active}

    return pairs


# ------------------------------------------------------------
# SDRG-X thermal sampling
# Convention matches your earlier code:
# s=0 singlet (entangled), s=1,2 (unentangled), s=3 (entangled triplet)
# ------------------------------------------------------------
def sample_pair_state(J, T, rng):
    if T <= 0:
        return 0

    beta = 1.0 / T
    energies = np.array([
        -J / 2.0,  # s=0
         0.0,      # s=1
         0.0,      # s=2
         J / 2.0,  # s=3
    ], dtype=float)

    w = np.exp(-beta * energies)
    p = w / np.sum(w)
    return int(rng.choice([0, 1, 2, 3], p=p))


def entanglement_entropy_sdrgx(pairs_with_J, L, T, rng):
    """
    Post-processing SDRG-X entropy:
    S(l) = ln(2) * (# entangled pairs crossing cut l),
    where entangled states are s in {0,3}.
    """
    S = np.zeros(L, dtype=float)

    # Sample s for each pair independently (valid for this greedy SDRG-X setup)
    sampled = []
    for r1, r2, Jij in pairs_with_J:
        s = sample_pair_state(Jij, T, rng)
        sampled.append((r1, r2, s))

    for l in range(L):
        crossings = 0
        for r1, r2, s in sampled:
            if s not in (0, 3):
                continue
            if (r1 < l < r2) or (r2 < l < r1):
                crossings += 1
        S[l] = np.log(2.0) * crossings

    return S


def pairing_accuracy(exact_pairs_with_J, ml_pairs_with_J):
    # compare only pairing structure, ignore J
    def norm_pair(r1, r2):
        return tuple(sorted((int(r1), int(r2))))

    exact_set = set(norm_pair(r1, r2) for (r1, r2, _) in exact_pairs_with_J)
    ml_set = set(norm_pair(r1, r2) for (r1, r2, _) in ml_pairs_with_J)

    M_P = len(exact_set.intersection(ml_set))
    N = 2 * len(exact_pairs_with_J)
    return 2.0 * M_P / N


# ------------------------------------------------------------
# Main comparison: ML vs exact, but entropy via SDRG-X postprocessing
# ------------------------------------------------------------
def compare_entropy_sdrgx(
    N=80,
    L=800,
    alpha=2.0,
    T_list=(0.0, 0.01, 0.1, 1.0),
    n_disorder=200,
    n_thermal=100,
    outdir="entropy_sdrgx_results",
    seed=0,
):
    os.makedirs(outdir, exist_ok=True)
    rng = np.random.default_rng(seed)

    # Load model once
    positions0 = generate_positions(N, L)
    J0 = initial_couplings(positions0, alpha)
    sample_data, _ = build_graph_from_state(positions0, J0, list(range(N)))
    model = load_trained_model(sample_data)

    # storage
    results = {
        "N": N,
        "L": L,
        "alpha": alpha,
        "T_list": [float(T) for T in T_list],
        "n_disorder": n_disorder,
        "n_thermal": n_thermal,
        "r_P_mean": None,
        "r_P_std": None,
        "S_exact_by_T": {},
        "S_ml_by_T": {},
    }

    rP_all = []
    S_exact_by_T_all = {float(T): [] for T in T_list}
    S_ml_by_T_all = {float(T): [] for T in T_list}

    for d in range(n_disorder):
        print(f"Disorder realization {d}")

        positions = generate_positions(N, L)
        J = initial_couplings(positions, alpha)

        exact_pairs = exact_greedy_pairing_with_J(positions, J)
        ml_pairs = ml_sdrg_pairing_with_J(positions, J, model)

        rP_all.append(pairing_accuracy(exact_pairs, ml_pairs))

        for T in T_list:
            T = float(T)

            # Thermal averaging
            nT = 1 if T <= 0 else n_thermal

            S_ex_list = []
            S_ml_list = []
            for _ in range(nT):
                S_ex_list.append(entanglement_entropy_sdrgx(exact_pairs, L, T, rng))
                S_ml_list.append(entanglement_entropy_sdrgx(ml_pairs, L, T, rng))

            S_exact_by_T_all[T].append(np.mean(S_ex_list, axis=0))
            S_ml_by_T_all[T].append(np.mean(S_ml_list, axis=0))

    # Final averages
    results["r_P_mean"] = float(np.mean(rP_all))
    results["r_P_std"] = float(np.std(rP_all))

    for T in [float(t) for t in T_list]:
        S_ex = np.mean(S_exact_by_T_all[T], axis=0)
        S_ml = np.mean(S_ml_by_T_all[T], axis=0)
        results["S_exact_by_T"][str(T)] = S_ex.tolist()
        results["S_ml_by_T"][str(T)] = S_ml.tolist()

        # quick plot per T
        plot_entropy(S_ex, S_ml, L, outdir, T=T, rP_mean=results["r_P_mean"], rP_std=results["r_P_std"])

    # Save combined json
    json_path = os.path.join(outdir, "entropy_sdrgx_ml_vs_exact.json")
    with open(json_path, "w") as f:
        json.dump(results, f, indent=2)

    print(f"\nPairing accuracy r_P = {results['r_P_mean']:.3f} ± {results['r_P_std']:.3f}")
    print(f"Saved SDRG-X entropy data to {json_path}")


def plot_entropy(S_exact, S_ml, L, outdir, T=0.0, rP_mean=None, rP_std=None):
    l = np.arange(1, L + 1)

    plt.figure(figsize=(6, 4))
    plt.plot(l, S_exact, label="Exact (greedy) + SDRG-X", linewidth=2)
    plt.plot(l, S_ml, "--", label="ML flow + SDRG-X", linewidth=2)
    plt.xlabel(r"$\ell$")
    plt.ylabel(r"$S(\ell)$")
    plt.title(rf"$T={T}$")
    plt.legend()

    if rP_mean is not None:
        text = rf"$r_P = {rP_mean:.3f}$"
        if rP_std is not None:
            text = rf"$r_P = {rP_mean:.3f} \pm {rP_std:.3f}$"
        plt.text(
            0.05, 0.95, text,
            transform=plt.gca().transAxes,
            fontsize=11,
            verticalalignment="top",
            bbox=dict(boxstyle="round", facecolor="white", alpha=0.8),
        )

    plt.tight_layout()
    fname = os.path.join(outdir, f"entropy_T_{float(T):.3f}.png")
    plt.savefig(fname, dpi=150)
    plt.close()


if __name__ == "__main__":
    compare_entropy_sdrgx(
        N=80,
        L=800,
        alpha=2.0,
        T_list=(0.0, 0.005, 0.01, 0.1, 1.0),
        n_disorder=500,
        n_thermal=200,
        outdir="entropy_sdrgx_results",
        seed=0,
    )
