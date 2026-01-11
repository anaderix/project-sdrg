import numpy as np
import json
import os
from utils import generate_positions, initial_couplings



# ============================================================
# Thermal sampling of pair eigenstates (SDRG-X)
# ============================================================

def sample_pair_state(J, T):
    """
    Sample one of the four eigenstates s = 0,1,2,3
    for a bond of strength J at temperature T.
    """
    if T == 0:
        return 0  # singlet only

    beta = 1.0 / T

    energies = np.array([
        -J / 2,  # s=0 (singlet, entangled)
         0.0,    # s=1
         0.0,    # s=2
         J / 2   # s=3 (entangled triplet)
    ])

    weights = np.exp(-beta * energies)
    probs = weights / np.sum(weights)

    return np.random.choice([0, 1, 2, 3], p=probs)


# ============================================================
# SDRG-X pairing
# ============================================================

def sdrg_pairing_finite_T(positions, J, T):
    """
    Perform SDRG-X and return list of pairs:
    (r_i, r_j, s)
    """
    active = list(range(len(positions)))
    pairs = []

    J = J.copy()

    while len(active) > 1:
        i, j = max(
            [(i, j) for (i, j) in J if i in active and j in active],
            key=lambda x: J[x]
        )

        Jij = J[(i, j)]
        s = sample_pair_state(Jij, T)

        pairs.append((positions[i], positions[j], s))

        active.remove(i)
        active.remove(j)

        J = {
            (k, l): v
            for (k, l), v in J.items()
            if k in active and l in active
        }

    return pairs


# ============================================================
# Entanglement entropy
# ============================================================

def entanglement_entropy_finite_T(pairs, L):
    """
    S(l) = ln(2) * number of entangled pairs crossing cut l
    Only s = 0 and s = 3 contribute.
    """
    S = np.zeros(L)

    for l in range(L):
        crossings = 0
        for r1, r2, s in pairs:
            if s not in (0, 3):
                continue
            if (r1 < l < r2) or (r2 < l < r1):
                crossings += 1
        S[l] = np.log(2) * crossings

    return S


# ============================================================
# Main driver: disorder + thermal + temperature sampling
# ============================================================

def run_sdrg_entropy_multi_T(
    N=100,
    L=1000,
    alpha=2.0,
    T_list=(0.0, 0.1, 0.2, 0.5),
    n_disorder=1000,
    n_thermal=100,
    outdir="sdrgX_data"
):
    os.makedirs(outdir, exist_ok=True)

    # storage for all temperatures
    S_by_T = {float(T): [] for T in T_list}

    for d in range(n_disorder):
        print(f"Disorder realization {d}")

        positions = generate_positions(N, L)
        J = initial_couplings(positions, alpha)

        for T in T_list:
            S_thermal = []

            for _ in range(n_thermal if T > 0 else 1):
                pairs = sdrg_pairing_finite_T(positions, J, T)
                S = entanglement_entropy_finite_T(pairs, L)
                S_thermal.append(S)

            # average over thermal sampling
            S_by_T[float(T)].append(np.mean(S_thermal, axis=0))

    # disorder average and save
    results = {}

    for T in T_list:
        T = float(T)
        S_avg = np.mean(S_by_T[T], axis=0)

        results[T] = S_avg.tolist()

        # save individual temperature file
        data_T = {
            "N": N,
            "L": L,
            "alpha": alpha,
            "T": T,
            "n_disorder": n_disorder,
            "n_thermal": n_thermal,
            "S_l": S_avg.tolist()
        }

        fname = f"S_l_T_{T:.3f}.json"
        with open(os.path.join(outdir, fname), "w") as f:
            json.dump(data_T, f, indent=2)

    # save combined file
    combined = {
        "N": N,
        "L": L,
        "alpha": alpha,
        "T_list": list(map(float, T_list)),
        "n_disorder": n_disorder,
        "n_thermal": n_thermal,
        "S_l_by_T": results
    }

    with open(os.path.join(outdir, "S_l_all_T.json"), "w") as f:
        json.dump(combined, f, indent=2)

    print("Saved entanglement entropy for all temperatures to", outdir)


# ============================================================
# Run
# ============================================================

if __name__ == "__main__":
    run_sdrg_entropy_multi_T(
        N=100,
        L=1000,
        alpha=3.0,
        T_list=[0.0, 0.005, 0.01, 0.1, 1.0],
        n_disorder=500,
        n_thermal=100
    )
