import json
import numpy as np
import matplotlib.pyplot as plt

# Load data
with open("sdrgX_data/S_l_all_T.json") as f:
    data = json.load(f)

L = data["L"]
l_vals = np.arange(1, L)

# Sort temperatures numerically
T_list = sorted(map(float, data["S_l_by_T"].keys()))

plt.figure(figsize=(6, 4))

for T in T_list:
    S = np.array(data["S_l_by_T"][str(T)])
    plt.plot(
        l_vals,
        S[1:],
        label=rf"$T={T:g}$"
    )

plt.xlabel(r"Cut position $l$")
plt.ylabel(r"Entanglement entropy $S(l)$")
plt.title(
    rf"SDRG-X entanglement entropy "
    rf"($N={data['N']},\,\alpha={data['alpha']}$)"
)
plt.legend(title=r"Temperature")
plt.grid(True)
plt.tight_layout()

# 🔽 Save figure
outname = (
    f"EE_SDRG_X_"
    f"N{data['N']}_"
    f"alpha{data['alpha']}_"
    f"Tmulti.png"
)
plt.savefig(outname, dpi=300, bbox_inches="tight")

plt.show()

print(f"Saved figure as {outname}")
