import numpy as np

def build_step_json(J, positions, active_spins, target_edge, k=4):
    active_spins = sorted(active_spins)
    spin_map = {s: i for i, s in enumerate(active_spins)}

    node_features = [[1.0] for _ in active_spins]

    edge_index = [[], []]
    edge_features = []
    edge_mask = []

    # Precompute log couplings per site
    logJ_by_site = {i: [] for i in active_spins}
    for i in active_spins:
        for j in active_spins:
            if i < j:
                logJ_by_site[i].append(np.log(abs(J[(i, j)])))
                logJ_by_site[j].append(np.log(abs(J[(i, j)])))

    edges = []
    for i in active_spins:
        for j in active_spins:
            if i < j:
                edges.append((i, j))

    for (i, j) in edges:
        edge_index[0].append(spin_map[i])
        edge_index[1].append(spin_map[j])

        logJ = np.log(abs(J[(i, j)]))
        logR = np.log(abs(positions[i] - positions[j]))

        # local normalization (Ω-free)
        neigh_i = sorted(logJ_by_site[i], reverse=True)[:k]
        neigh_j = sorted(logJ_by_site[j], reverse=True)[:k]
        local_mean = np.mean(neigh_i + neigh_j)

        rel_strength = logJ - local_mean

        edge_features.append([
            float(logJ),
            float(logR),
            float(rel_strength)
        ])
        edge_mask.append(1)

    target_edge_idx = edges.index(target_edge)

    return {
        "num_nodes": len(active_spins),
        "node_features": node_features,
        "edge_index": edge_index,
        "edge_features": edge_features,
        "edge_mask": edge_mask,
        "target_edge": target_edge_idx
    }
