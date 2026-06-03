# benchmark_runtime.py
"""
Runtime benchmark: SDRG vs GNN-assisted SDRG.

Purpose
-------
Referee 3 asked whether the GNN-assisted SDRG provides a computational
advantage over standard SDRG.

This script measures wall-clock runtime as a function of system size N.

Important interpretation
------------------------
The present GNN-assisted implementation still performs SDRG-style coupling
updates after every decimation step. Therefore, we do not expect a dramatic
speedup. The purpose of this benchmark is to report this honestly and clarify
that the main contribution is the learned graph-based surrogate policy, not
direct runtime acceleration.
"""

import time
import csv
from pathlib import Path
import tracemalloc # For memory profiling (optional, not used in current code)
import torch

from utils import generate_positions, initial_couplings
from sdrg import strongest_bond, decimate
from model import SDRGNet
from json_writer import build_step_json
from torch_geometric.data import Data


# -------------------------
# Benchmark settings
# -------------------------
N_VALUES = [20, 40, 60, 80, 100]
N_REALIZATIONS = 50

L_OVER_N = 10
ALPHA = 2.0

RESULTS_DIR = Path("results/runtime")
RESULTS_DIR.mkdir(parents=True, exist_ok=True)

OUTPUT_CSV = RESULTS_DIR / "runtime_scaling.csv"


def sample_to_data(sample):
    x = torch.tensor(sample["node_features"], dtype=torch.float)
    edge_index = torch.tensor(sample["edge_index"], dtype=torch.long)
    edge_attr = torch.tensor(sample["edge_features"], dtype=torch.float)
    edge_mask = torch.tensor(sample["edge_mask"], dtype=torch.bool)
    y = torch.tensor(sample["target_edge"], dtype=torch.long)

    return Data(
        x=x,
        edge_index=edge_index,
        edge_attr=edge_attr,
        edge_mask=edge_mask,
        y=y,
    )


def run_standard_sdrg(N, L, alpha):
    positions = generate_positions(N, L)
    J = initial_couplings(positions, alpha)
    active_spins = list(range(N))

    while len(active_spins) > 1:
        i, j = strongest_bond(J, active_spins)
        J, active_spins = decimate(J, active_spins, i, j)


def run_gnn_sdrg(N, L, alpha, model):
    positions = generate_positions(N, L)
    J = initial_couplings(positions, alpha)
    active_spins = list(range(N))

    model.eval()

    with torch.no_grad():
        while len(active_spins) > 1:
            # We build a temporary sample only to construct the PyG graph.
            # The target is not used during inference, so we use strongest_bond
            # only as a placeholder for JSON construction.
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

            active_sorted = sorted(active_spins)
            physical_edges = []
            for a in active_sorted:
                for b in active_sorted:
                    if a < b:
                        physical_edges.append((a, b))

            i, j = physical_edges[pred_idx]

            J, active_spins = decimate(J, active_spins, i, j)


def mean_runtime(fn, *args):
    times = []

    for _ in range(N_REALIZATIONS):
        start = time.perf_counter()
        fn(*args)
        end = time.perf_counter()
        times.append(end - start)

    mean = sum(times) / len(times)
    std = (sum((t - mean) ** 2 for t in times) / len(times)) ** 0.5

    return mean, std



def main():
    # Model used only for inference-time benchmark.
    # It does not need to be highly trained for measuring computational cost.
    model = SDRGNet(
        node_dim=1,
        edge_dim=3,
        hidden_dim=64,
        num_layers=3,
        dropout=0.0,
    )

    rows = []

    for N in N_VALUES:
        L = L_OVER_N * N

        print("\n" + "=" * 70)
        print(f"Benchmarking N={N}, L={L}, alpha={ALPHA}")
        print("=" * 70)

        sdrg_mean, sdrg_std = mean_runtime(run_standard_sdrg, N, L, ALPHA)
        gnn_mean, gnn_std = mean_runtime(run_gnn_sdrg, N, L, ALPHA, model)

        ratio = gnn_mean / sdrg_mean if sdrg_mean > 0 else float("nan")

        print(f"Standard SDRG:     {sdrg_mean:.6f} ± {sdrg_std:.6f} s")
        print(f"GNN-assisted SDRG: {gnn_mean:.6f} ± {gnn_std:.6f} s")
        print(f"Runtime ratio:     {ratio:.3f}")

        rows.append({
            "N": N,
            "L": L,
            "alpha": ALPHA,
            "n_realizations": N_REALIZATIONS,
            "sdrg_mean_s": sdrg_mean,
            "sdrg_std_s": sdrg_std,
            "gnn_mean_s": gnn_mean,
            "gnn_std_s": gnn_std,
            "gnn_over_sdrg": ratio,
        })

    with open(OUTPUT_CSV, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=rows[0].keys())
        writer.writeheader()
        writer.writerows(rows)

    print("\nSaved runtime results to:")
    print(OUTPUT_CSV)


if __name__ == "__main__":
    main()


"""

>> echo 'FEATURE_MODE = "full"' > feature_config.py
>> python benchmark_runtime.py
"""