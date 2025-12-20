# sdrg_entropy.py

import numpy as np
import json
import os
from utils import generate_positions, initial_couplings



#def generate_positions(N, L):
#    """Randomly place N spins on a chain of length L."""
#    return np.sort(np.random.choice(L, size=N, replace=False))


#def initial_couplings(positions, alpha):
#    """Return dict {(i,j): J_ij}."""
#    J = {}
#    N = len(positions)
#    for i in range(N):
#        for j in range(i+1, N):
#            dist = abs(positions[i] - positions[j])
#            J[(i, j)] = dist ** (-alpha)
#    return J



def sdrg_pairing(positions, J):
    """
    Perform SDRG and return list of singlet pairs
    as tuples of real-space positions (r_i, r_j).
    """
    active = list(range(len(positions)))
    pairs = []

    J = J.copy()

    while len(active) > 1:
        # strongest bond
        i, j = max(
            [(i, j) for (i, j) in J if i in active and j in active],
            key=lambda x: J[x]
        )

        pairs.append((positions[i], positions[j]))

        # remove spins i, j
        active.remove(i)
        active.remove(j)

        # remove bonds touching i or j
        J = {
            (k, l): v
            for (k, l), v in J.items()
            if k in active and l in active
        }

    return pairs



def entanglement_entropy(pairs, L):
    """
    S(l) = ln(2) * number of singlets crossing cut l
    """
    S = np.zeros(L)

    for l in range(L):
        crossings = 0
        for r1, r2 in pairs:
            if (r1 < l < r2) or (r2 < l < r1):
                crossings += 1
        S[l] = np.log(2) * crossings

    return S



def run_sdrg_entropy(
    N=100,
    L=1000,
    alpha=2.0,
    n_realizations=10,
    outdir="sdrg_data"
):
    os.makedirs(outdir, exist_ok=True)

    all_S = []

    for r in range(n_realizations):
        print(f"Realization {r}")

        positions = generate_positions(N, L)
        J = initial_couplings(positions, alpha)
        pairs = sdrg_pairing(positions, J)

        S = entanglement_entropy(pairs, L)
        all_S.append(S)

    S_avg = np.mean(all_S, axis=0)

    data = {
        "N": N,
        "L": L,
        "alpha": alpha,
        "n_realizations": n_realizations,
        "S_l": S_avg.tolist()
    }

    with open(os.path.join(outdir, "S_l.json"), "w") as f:
        json.dump(data, f, indent=2)

    print("Saved entanglement entropy to", outdir)


if __name__ == "__main__":
    run_sdrg_entropy(
        N=100,
        L=1000,
        alpha=0.8,
        n_realizations=1000
    )


