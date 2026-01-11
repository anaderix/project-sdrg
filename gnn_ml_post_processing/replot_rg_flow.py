import os
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec

# --------------------------------------------------
# Load data
# --------------------------------------------------
alpha=2.0
outdir = f"rg_flow_results_alpha{alpha}"

exact_map = np.load(os.path.join(outdir, "exact_heatmap.npy"))
ml_map    = np.load(os.path.join(outdir, "ml_heatmap.npy"))

# --------------------------------------------------
# Plot
# --------------------------------------------------
fig = plt.figure(figsize=(6.8, 3.2))  # good for single-column papers
gs = GridSpec(
    nrows=1,
    ncols=3,
    width_ratios=[1, 1, 0.05],
    wspace=0.15
)

ax_exact = fig.add_subplot(gs[0, 0])
ax_ml    = fig.add_subplot(gs[0, 1])
cax      = fig.add_subplot(gs[0, 2])

vmin = 0.0
vmax = max(exact_map.max(), ml_map.max())

im0 = ax_exact.imshow(
    exact_map,
    origin="lower",
    aspect="auto",
    cmap="viridis",
    vmin=vmin,
    vmax=vmax,
)

im1 = ax_ml.imshow(
    ml_map,
    origin="lower",
    aspect="auto",
    cmap="viridis",
    vmin=vmin,
    vmax=vmax,
)

# --------------------------------------------------
# Axes labels & titles
# --------------------------------------------------
ax_exact.set_title("Exact SDRG")
ax_ml.set_title("GNN-SDRG")

ax_exact.set_xlabel("RG step")
ax_ml.set_xlabel("RG step")
ax_exact.set_ylabel("Bond length (log bins)")

# Remove duplicate y-axis ticks
ax_ml.set_yticks([])

# --------------------------------------------------
# Colorbar
# --------------------------------------------------
cbar = fig.colorbar(im0, cax=cax)
cbar.set_label("Probability")

# --------------------------------------------------
# Save
# --------------------------------------------------
plt.savefig(
    os.path.join(outdir, "rg_flow_heatmap_replot.png"),
    dpi=300,
    bbox_inches="tight"
)
plt.close()

print("Saved: rg_flow_heatmap_replot.png")
