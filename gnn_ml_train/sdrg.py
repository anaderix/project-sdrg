# sdrg.py

def strongest_bond(J, active_spins):
    """
    Return (i, j) corresponding to the strongest bond among active spins.
    J is a dict {(i, j): J_ij} with i < j.
    """
    active = set(active_spins)

    return max(
        ((i, j) for (i, j) in J if i in active and j in active),
        key=lambda x: abs(J[x])
    )


def decimate(J, active_spins, i, j):
    """
    Perform SDRG decimation of the strongest bond (i, j).

    Spins i and j form a singlet and are removed.
    New effective couplings are generated perturbatively:
        J_eff = J_ik * J_jl / Omega   and   J_jk * J_il / Omega

    Returns:
        J_new : updated coupling dictionary
        active_new : updated list of active spins
    """
    # Ensure ordering
    if i > j:
        i, j = j, i

    Omega = J[(i, j)]

    # Remove decimated spins
    active_new = [s for s in active_spins if s not in (i, j)]

    J_new = {}

    for a in range(len(active_new)):
        for b in range(a + 1, len(active_new)):
            k = active_new[a]
            l = active_new[b]

            # Existing coupling
            Jkl = J.get((min(k, l), max(k, l)), 0.0)

            # Couplings to decimated spins
            Jik = J.get((min(i, k), max(i, k)), 0.0)
            Jjk = J.get((min(j, k), max(j, k)), 0.0)
            Jil = J.get((min(i, l), max(i, l)), 0.0)
            Jjl = J.get((min(j, l), max(j, l)), 0.0)

            # Second-order perturbative correction
            J_eff = 0.0
            if Jik != 0.0 and Jjl != 0.0:
                J_eff += Jik * Jjl / Omega
            if Jjk != 0.0 and Jil != 0.0:
                J_eff += Jjk * Jil / Omega

            J_total = Jkl + J_eff

            if J_total != 0.0:
                J_new[(min(k, l), max(k, l))] = J_total

    return J_new, active_new


# def strongest_bond(J, active_spins):
#     """Return (i,j) with largest |J_ij| among active spins."""
#     return max(
#         [(i, j) for (i, j) in J if i in active_spins and j in active_spins],
#         key=lambda x: abs(J[x])
#     )


# def decimate(J, active_spins, i, j):
#     """Remove spins i and j and all bonds touching them."""
#     active_spins.remove(i)
#     active_spins.remove(j)

#     J_new = {
#         (k, l): v
#         for (k, l), v in J.items()
#         if k in active_spins and l in active_spins
#     }
#     return J_new, active_spins
