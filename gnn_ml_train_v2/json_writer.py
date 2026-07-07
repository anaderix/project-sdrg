# json_writer.py
import numpy as np
from feature_config import FEATURE_MODE

def build_step_json(J, positions, active_spins, target_edge, k=4):
    active_spins = sorted(active_spins)
    spin_map = {s: i for i, s in enumerate(active_spins)}

    node_features = [[1.0] for _ in active_spins]

    physical_edges = []
    for i in active_spins:
        for j in active_spins:
            if i < j:
                physical_edges.append((i, j))

    logJ_by_site = {i: [] for i in active_spins}

    for i, j in physical_edges:
        Jij = J[(i, j)]
        logJ = np.log(abs(Jij) + 1e-12)
        logJ_by_site[i].append(logJ)
        logJ_by_site[j].append(logJ)

    edge_index = [[], []]
    edge_features = []
    edge_mask = []
    score_edge_indices = []

    for physical_idx, (i, j) in enumerate(physical_edges):
        Jij = J[(i, j)]

        logJ = np.log(abs(Jij) + 1e-12)
        logR = np.log(abs(positions[i] - positions[j]) + 1e-12)

        neigh_i = sorted(logJ_by_site[i], reverse=True)[:k]
        neigh_j = sorted(logJ_by_site[j], reverse=True)[:k]
        local_mean = np.mean(neigh_i + neigh_j)

        rel_strength = logJ - local_mean

        if FEATURE_MODE == "full":
            features = [
                float(logJ),
                float(logR),
                float(rel_strength),
            ]
        elif FEATURE_MODE == "no_logJ":
            features = [
                float(logR),
                float(rel_strength),
            ]
        elif FEATURE_MODE == "no_rel":
            features = [
                float(logJ),
                float(logR),
            ]
        elif FEATURE_MODE == "only_logJ":
            features = [
                float(logJ),
            ]
        elif FEATURE_MODE == "only_logR":
            features = [
                float(logR),
            ]
        elif FEATURE_MODE == "only_rel":
            features = [
                float(rel_strength),
            ]
        # i -> j
        edge_index[0].append(spin_map[i])
        edge_index[1].append(spin_map[j])
        edge_features.append(features)
        edge_mask.append(1)
        score_edge_indices.append(physical_idx)

        # j -> i
        edge_index[0].append(spin_map[j])
        edge_index[1].append(spin_map[i])
        edge_features.append(features)
        edge_mask.append(0)
        score_edge_indices.append(physical_idx)

    if target_edge[0] > target_edge[1]:
        target_edge = (target_edge[1], target_edge[0])

    target_edge_idx = physical_edges.index(target_edge)

    return {
        "num_nodes": len(active_spins),
        "node_features": node_features,
        "edge_index": edge_index,
        "edge_features": edge_features,
        "edge_mask": edge_mask,
        "score_edge_indices": score_edge_indices,
        "target_edge": target_edge_idx,
    }