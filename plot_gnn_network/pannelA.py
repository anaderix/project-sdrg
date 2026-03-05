import matplotlib.pyplot as plt
import networkx as nx
import numpy as np

fig, ax = plt.subplots(figsize=(6, 3))

# --- Spin chain (left) ---
x = np.linspace(0.1, 0.4, 5)
y = np.ones_like(x) * 0.7
ax.scatter(x, y, s=300)

for i in range(len(x) - 1):
    ax.plot([x[i], x[i+1]], [y[i], y[i+1]])

ax.text(0.18, 0.55, r"$J_{ij} \propto |r_i - r_j|^{-\alpha}$", fontsize=12)

# Arrow
ax.annotate("", xy=(0.55, 0.7), xytext=(0.45, 0.7),
            arrowprops=dict(arrowstyle="->", lw=2))

# --- Graph (right) ---
G = nx.complete_graph(5)
pos = {i: (0.65 + 0.25*np.cos(2*np.pi*i/5),
           0.7 + 0.25*np.sin(2*np.pi*i/5)) for i in range(5)}

nx.draw(G, pos, ax=ax, node_size=300, with_labels=False)

ax.set_xlim(0, 1)
ax.set_ylim(0, 1)
ax.axis("off")

plt.title("(a) Spin Chain → Graph Representation")
plt.tight_layout()
plt.show()