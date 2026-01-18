import json
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1.inset_locator import inset_axes


def plot_from_json(json_path, out_png="entropy_replot.png"):
    # ===============================
    # Load JSON
    # ===============================
    with open(json_path, "r") as f:
        data = json.load(f)

    L = data["L"]
    S_exact = np.array(data["S_exact"])
    S_ml = np.array(data["S_ml"])

    rP_all = np.array(data["r_P_all"])
    rP_mean = data["r_P_mean"]
    rP_std = data["r_P_std"]

    l = np.arange(1, L + 1)

    # ===============================
    # Main entropy plot
    # ===============================
    fig, ax = plt.subplots(figsize=(6, 4))

    ax.plot(l, S_exact, label="Exact SDRG", linewidth=2)
    ax.plot(l, S_ml, "--", label="ML-SDRG", linewidth=2)

    ax.set_xlabel(r"$\ell$")
    ax.set_ylabel(r"$S(\ell)$")
    ax.legend()

    # ===============================
    # Text inset: mean ± std
    # ===============================
    ax.text(
        0.05, 0.95,
        rf"$r_P = {rP_mean:.3f} \pm {rP_std:.3f}$",
        transform=ax.transAxes,
        fontsize=11,
        verticalalignment="top",
        bbox=dict(boxstyle="round", facecolor="white", alpha=0.85)
    )

    # ===============================
    # Histogram inset
    # ===============================
    ax_hist = inset_axes(
        ax,
        width="35%",
        height="35%",
        loc="center",
        borderpad=1
    )

    ax_hist.hist(
        rP_all,
        bins=30,
        density=True,
        alpha=0.8
    )

    ax_hist.axvline(
        rP_mean,
        linestyle="--",
        linewidth=1
    )

    ax_hist.set_xlabel(r"$r_P$", fontsize=8)
    ax_hist.set_ylabel("PDF", fontsize=8)
    ax_hist.tick_params(axis="both", labelsize=8)
    ax_hist.set_xlim(0, 1)

    # ===============================
    # Save
    # ===============================
    plt.tight_layout()
    plt.savefig(out_png, dpi=150)
    plt.close()

    print(f"Saved figure to {out_png}")


if __name__ == "__main__":
    alpha=2.0
    plot_from_json(
        json_path="entropy_ml_vs_exact.json",
        out_png=f"entropy_linear_alpha{alpha}.png"
    )
