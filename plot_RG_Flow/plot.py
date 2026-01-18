import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch

# =========================
# Helper: curved branch
# =========================
def draw_branch(x0, y0, x1, y1, color, lw=2, alpha=0.8, rad=0.15):
    patch = FancyArrowPatch(
        (x0, y0),
        (x1, y1),
        arrowstyle="-",
        connectionstyle=f"arc3,rad={rad}",
        linewidth=lw,
        color=color,
        alpha=alpha,
        zorder=1,
    )
    plt.gca().add_patch(patch)

# =========================
# SDRG-X parameters
# =========================
n_steps = 3
J = 1.0

# Four local pair states
pair_energies = [-J/2, 0.0, 0.0, +J/2]
pair_colors   = ["tab:blue", "black", "black", "tab:red"]
pair_rads     = [-0.3, -0.0, 0.0, 0.3]

# =========================
# Build SDRG-X energy tree
# =========================
energies = {0: [0.0]}

for n in range(1, n_steps + 1):
    energies[n] = []
    for E in energies[n - 1]:
        for dE in pair_energies:
            energies[n].append(E + dE)

# =========================
# Plot
# =========================
fig, ax = plt.subplots(figsize=(6 ,6))

# --- RG step separators (dashed) ---
for n in range(n_steps + 1):
    ax.axvline(
        n,
        color="black",
        lw=1.0,
        ls="--",
        alpha=0.5,
        zorder=0,
    )

# --- Horizontal energy levels (degenerate merged) ---
for n, Elist in energies.items():
    unique_E = sorted(set(Elist))   # merge degeneracies visually
    for E in unique_E:
        ax.hlines(
            E,
            n - 0.1,
            n + 0.1,
            color="gray",
            linewidth=1,
            zorder=3,
        )

# --- SDRG-X branching ---
for n in range(n_steps):
    for E in energies[n]:
        x0 = n + 0.12
        y0 = E
        x1 = (n + 1) - 0.12

        for dE, color, rad in zip(pair_energies, pair_colors, pair_rads):
            y1 = E + dE
            draw_branch(
                x0, y0,
                x1, y1,
                color=color,
                rad=rad,
            )

# =========================
# Axes & style
# =========================
ax.set_xlim(-0.1, n_steps + 0.1)
ax.set_xlabel(r"RG step $n$", fontsize=15)
ax.set_ylabel(r"Total energy $E$", fontsize=15)

ax.set_xticks(range(n_steps + 1))
ax.tick_params(axis="both", labelsize=13)
ax.set_yticks([])


ax.spines["top"].set_visible(False)
ax.spines["right"].set_visible(False)
ax.spines["left"].set_linewidth(1.5)
ax.spines["bottom"].set_linewidth(1.5)

# Legend (minimal & correct)
ax.plot([], [], color="tab:blue", lw=3, label=r"$E=-J/2$")
ax.plot([], [], color="black",   lw=3, label=r"$E=0$ (deg.)")
ax.plot([], [], color="tab:red", lw=3, label=r"$E=+J/2$")
ax.legend(frameon=False, fontsize=12, loc="upper left")

plt.tight_layout()
plt.savefig("sdrgx_schematic_tree_degenerate.png", dpi=300)
plt.show()


# import numpy as np
# import matplotlib.pyplot as plt
# from matplotlib.patches import FancyArrowPatch

# # =========================
# # Helper: curved RG branch
# # =========================
# def draw_branch(x0, y0, x1, y1, color, lw=1.6, alpha=0.85, rad=0.2):
#     patch = FancyArrowPatch(
#         (x0, y0),
#         (x1, y1),
#         arrowstyle="-",
#         connectionstyle=f"arc3,rad={rad}",
#         linewidth=lw,
#         color=color,
#         alpha=alpha,
#         zorder=2,
#     )
#     plt.gca().add_patch(patch)

# # =========================
# # RG parameters (schematic)
# # =========================
# n_steps = 4

# # Initial root energy
# E0 = [0.0]

# # Symmetric energy splitting size per RG step
# splitting = {
#     1: 1.0,
#     2: 0.6,
#     3: 0.4,
# }

# # =========================
# # Build symmetric RG tree
# # =========================
# energies = {0: E0}

# for n in range(1, n_steps):
#     energies[n] = []
#     for E in energies[n - 1]:
#         dE = splitting[n]
#         energies[n].append(E - dE)  # lower (blue)
#         energies[n].append(E + dE)  # upper (red)

# # =========================
# # Plot
# # =========================
# fig, ax = plt.subplots(figsize=(7.5, 5.5))

# # Horizontal spectrum (black)
# for n, Elist in energies.items():
#     for E in Elist:
#         ax.hlines(
#             E,
#             n - 0.15,
#             n + 0.15,
#             color="black",
#             linewidth=3,
#             zorder=3,
#         )

# # Symmetric branching
# for n in range(n_steps - 1):
#     dE = splitting[n + 1]
#     for E in energies[n]:
#         x0 = n + 0.15
#         y0 = E

#         # Children
#         y_low = E - dE
#         y_high = E + dE
#         x1 = (n + 1) - 0.15

#         # Blue = lower
#         draw_branch(
#             x0, y0,
#             x1, y_low,
#             color="tab:blue",
#             rad=-0.2,
#         )

#         # Red = higher
#         draw_branch(
#             x0, y0,
#             x1, y_high,
#             color="tab:red",
#             rad=0.2,
#         )

# # =========================
# # Axes & style
# # =========================
# ax.set_xlim(-0.5, n_steps - 0.5)
# ax.set_xlabel(r"RG step $n$", fontsize=15)
# ax.set_ylabel(r"Energy $E_n$", fontsize=15)

# ax.set_xticks(range(n_steps))
# ax.tick_params(axis="both", labelsize=13)

# ax.spines["top"].set_visible(False)
# ax.spines["right"].set_visible(False)
# ax.spines["left"].set_linewidth(1.5)
# ax.spines["bottom"].set_linewidth(1.5)

# # Legend (minimal, correct)
# ax.plot([], [], color="tab:blue", lw=3, label="Lower-energy branch")
# ax.plot([], [], color="tab:red", lw=3, label="Higher-energy branch")
# ax.plot([], [], color="black", lw=3, label="Two degenerate branches")
# ax.legend(frameon=False, fontsize=12, loc="upper left")

# plt.tight_layout()
# plt.savefig("rg_symmetric_binary_branching.png", dpi=300)
# plt.show()
