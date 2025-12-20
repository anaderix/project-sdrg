# plot_entropy.py

import json
import numpy as np
import matplotlib.pyplot as plt

with open("sdrg_data/S_l.json", "r") as f:
    data = json.load(f)

S = np.array(data["S_l"])
L = data["L"]

l = np.arange(L)


plt.figure(figsize=(6, 4))
plt.plot(l, S, linewidth=2)
plt.xlabel(r"Cut position $l$")
plt.ylabel(r"Entanglement entropy $S(l)$")
plt.title(
    rf"SDRG entanglement entropy ($N={data['N']},\, \alpha={data['alpha']}$)"
)
plt.grid(True)
plt.tight_layout()
plt.savefig("entanglement", dpi=300, bbox_inches="tight")

plt.show()

