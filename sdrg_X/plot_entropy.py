import json
import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import quad



def thermal_integral(zmin, zmax):
    integrand = lambda z: 1.0 / ((1.0 + z)**2 * np.log(z))
    val, _ = quad(integrand, zmin, zmax, limit=200)
    return val


def S_analytic_finite_L(l_vals, L, T, alpha, Omega0=1.0):
    """
    Finite-L, finite-T SDRG-X entanglement entropy.

    Implements Eq. (slL):
    S_l(T) = (ln2/6)[ ln f_L(l)
      - (2/alpha) ∫_{Omega0/(2T f_L(l)^alpha)}^{Omega0/(2T)}
        dz / ((1+z)^2 ln z) ]
    """
    S_th = []

    for l in l_vals:
        if l <= 0 or l >= L:
            S_th.append(0.0)
            continue

        fL = (L / np.pi) * np.sin(np.pi * l / L)

        # ---------- T = 0 ----------
        if T == 0:
            S_val = (np.log(2) / 6.0) * np.log(fL)
            S_th.append(S_val)
            continue

        # ---------- finite T ----------
        z_max = Omega0 / (2 * T)
        z_min = Omega0 / (2 * T * fL**alpha)

        # numerical safety
        z_max = max(z_max, 1.0001)
        z_min = max(z_min, 1.0001)

        corr = (2 / alpha) * thermal_integral(z_min, z_max)

        S_val = (np.log(2) / 6.0) * (np.log(fL) - corr)
        S_th.append(S_val)

    return np.array(S_th)






# Load data
with open("sdrgX_data_alpha0.8/S_l_all_T.json") as f:
    data = json.load(f)

L = data["L"]
alpha = data["alpha"]
l_vals = np.arange(1, L)

T_list = sorted(map(float, data["S_l_by_T"].keys()))

plt.figure(figsize=(6, 4))

for T in T_list:
    S = np.array(data["S_l_by_T"][str(T)])

    # ---- numerical SDRG-X ----
    plt.plot(
        l_vals,
        S[1:],
        label=rf"$T={T:g}$ (num)"
    )

    # ---- analytical curve ----
    S_th = S_analytic_finite_L(
        l_vals=l_vals,
        L=L,
        T=T,
        alpha=data["alpha"],
        Omega0=1.0
    )


    plt.plot(
        l_vals,
        S_th,
        "--",
        linewidth=1.5,
        label=rf"$T={T:g}$ (theory)"
    )

plt.xlabel(r"Cut position $l$")
plt.ylabel(r"Entanglement entropy $S(l)$")
plt.title(
    rf"SDRG-X entanglement entropy "
    rf"($N={data['N']},\,\alpha={alpha}$)"
)
plt.legend(fontsize=8, ncol=2)
plt.grid(True)
plt.tight_layout()

outname = (
    f"EE_SDRG_X_"
    f"N{data['N']}_"
    f"alpha{alpha}_"
    f"Tmulti_with_theory.png"
)
plt.savefig(outname, dpi=300, bbox_inches="tight")
plt.show()

print(f"Saved figure as {outname}")


# import json
# import numpy as np
# import matplotlib.pyplot as plt

# # Load data
# with open("sdrgX_data/S_l_all_T.json") as f:
#     data = json.load(f)

# L = data["L"]
# l_vals = np.arange(1, L)

# # Sort temperatures numerically
# T_list = sorted(map(float, data["S_l_by_T"].keys()))

# plt.figure(figsize=(6, 4))

# for T in T_list:
#     S = np.array(data["S_l_by_T"][str(T)])
#     plt.plot(
#         l_vals,
#         S[1:],
#         label=rf"$T={T:g}$"
#     )

# plt.xlabel(r"Cut position $l$")
# plt.ylabel(r"Entanglement entropy $S(l)$")
# plt.title(
#     rf"SDRG-X entanglement entropy "
#     rf"($N={data['N']},\,\alpha={data['alpha']}$)"
# )
# plt.legend(title=r"Temperature")
# plt.grid(True)
# plt.tight_layout()

# # 🔽 Save figure
# outname = (
#     f"EE_SDRG_X_"
#     f"N{data['N']}_"
#     f"alpha{data['alpha']}_"
#     f"Tmulti.png"
# )
# plt.savefig(outname, dpi=300, bbox_inches="tight")

# plt.show()

# print(f"Saved figure as {outname}")
