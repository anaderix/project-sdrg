import matplotlib.pyplot as plt
import networkx as nx
import numpy as np

fig, ax = plt.subplots(figsize=(4, 4))

G = nx.complete_graph(4)
pos = nx.circular_layout(G)

nx.draw(G, pos, ax=ax, node_size=600, arrows=True)

# Draw message arrows
for i in G.nodes():
    for j in G.nodes():
        if i != j:
            ax.annotate("", xy=pos[i], xytext=pos[j],
                        arrowprops=dict(arrowstyle="->", alpha=0.3))

ax.text(0, -1.3, "Aggregate\nΣ neighbor messages\n↓\nMLP\n↓\nUpdated embeddings",
        ha="center", fontsize=11)

ax.set_title("(b) GIN Message Passing")
ax.axis("off")
plt.tight_layout()
plt.show()
