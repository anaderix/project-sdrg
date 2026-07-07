# benchmark_runtime_memory.py
"""
Runtime and memory benchmark: SDRG vs GNN-assisted SDRG.

This script reports:
1. wall-clock runtime,
2. peak Python memory usage,

as a function of system size N.

The memory measurement uses tracemalloc, so it tracks Python-level memory
allocations. It is useful for comparing relative memory trends between the
two implementations.
"""

import time
import csv
import tracemalloc
from pathlib import Path

import numpy as np
import torch
from torch_geometric.data import Data

from utils import generate_positions, initial_couplings
from sdrg import strongest_bond, decimate
from model import SDRGNet
from json_writer import build_step_json


# -------------------------
# Benchmark settings
# -------------------------
N_VALUES = [20, 40, 60, 80, 100]
N_REALIZATIONS = 50

L_OVER_N = 10
ALPHA = 2.0

RESULTS_DIR = Path("results/runtime")
RESULTS_DIR.mkdir(parents=True, exist_ok=True)

OUTPUT_CSV = RESULTS_DIR / "runtime_memory_scaling.csv"


def sample_to_data(sample):
    return Data(
        x=torch.tensor(sample["node_features"], dtype=torch.float),
        edge_index=torch.tensor(sample["edge_index"], dtype=torch.long),
        edge_attr=torch.tensor(sample["edge_features"], dtype=torch.float),
        edge_mask=torch.tensor(sample["edge_mask"], dtype=torch.bool),
        y=torch.tensor(sample["target_edge"], dtype=torch.long),
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


def measure_once(fn, *args):
    """
    Measure one run.

    Returns:
        runtime_seconds
        peak_memory_mb
    """
    tracemalloc.start()

    start = time.perf_counter()
    fn(*args)
    end = time.perf_counter()

    current, peak = tracemalloc.get_traced_memory()
    tracemalloc.stop()

    runtime = end - start
    peak_mb = peak / (1024 ** 2)

    return runtime, peak_mb


def mean_runtime_memory(fn, *args):
    runtimes = []
    peak_memories = []

    for _ in range(N_REALIZATIONS):
        runtime, peak_mb = measure_once(fn, *args)

        runtimes.append(runtime)
        peak_memories.append(peak_mb)

    runtimes = np.array(runtimes)
    peak_memories = np.array(peak_memories)

    return {
        "time_mean": float(np.mean(runtimes)),
        "time_std": float(np.std(runtimes, ddof=1)),
        "mem_mean": float(np.mean(peak_memories)),
        "mem_std": float(np.std(peak_memories, ddof=1)),
    }


def main():
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

        print("\n" + "=" * 80)
        print(f"Benchmarking N={N}, L={L}, alpha={ALPHA}")
        print("=" * 80)

        sdrg = mean_runtime_memory(run_standard_sdrg, N, L, ALPHA)
        gnn = mean_runtime_memory(run_gnn_sdrg, N, L, ALPHA, model)

        time_ratio = (
            gnn["time_mean"] / sdrg["time_mean"]
            if sdrg["time_mean"] > 0
            else float("nan")
        )

        mem_ratio = (
            gnn["mem_mean"] / sdrg["mem_mean"]
            if sdrg["mem_mean"] > 0
            else float("nan")
        )

        print(
            f"Standard SDRG time:     "
            f"{sdrg['time_mean']:.6f} ± {sdrg['time_std']:.6f} s"
        )
        print(
            f"GNN-assisted SDRG time: "
            f"{gnn['time_mean']:.6f} ± {gnn['time_std']:.6f} s"
        )
        print(f"Runtime ratio:          {time_ratio:.3f}")

        print(
            f"Standard SDRG memory:   "
            f"{sdrg['mem_mean']:.3f} ± {sdrg['mem_std']:.3f} MB"
        )
        print(
            f"GNN-assisted memory:    "
            f"{gnn['mem_mean']:.3f} ± {gnn['mem_std']:.3f} MB"
        )
        print(f"Memory ratio:           {mem_ratio:.3f}")

        rows.append({
            "N": N,
            "L": L,
            "alpha": ALPHA,
            "n_realizations": N_REALIZATIONS,

            "sdrg_time_mean_s": sdrg["time_mean"],
            "sdrg_time_std_s": sdrg["time_std"],
            "gnn_time_mean_s": gnn["time_mean"],
            "gnn_time_std_s": gnn["time_std"],
            "gnn_over_sdrg_time": time_ratio,

            "sdrg_peak_memory_mean_mb": sdrg["mem_mean"],
            "sdrg_peak_memory_std_mb": sdrg["mem_std"],
            "gnn_peak_memory_mean_mb": gnn["mem_mean"],
            "gnn_peak_memory_std_mb": gnn["mem_std"],
            "gnn_over_sdrg_memory": mem_ratio,
        })

    with open(OUTPUT_CSV, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=rows[0].keys())
        writer.writeheader()
        writer.writerows(rows)

    print("\nSaved runtime and memory results to:")
    print(OUTPUT_CSV)


if __name__ == "__main__":
    main()


"""

echo 'FEATURE_MODE = "full"' > feature_config.py
python benchmark_runtime_memory.py
"""