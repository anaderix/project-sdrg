import json
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1.inset_locator import inset_axes


def plot_from_json(json_path, out_png="entropy_replot.png"):

    with open(json_path, "r") as f:
        data = json.load(f)

    L = data["L"]

    S_exact_mean = np.array(data["S_exact_mean"])
    S_exact_sem = np.array(data["S_exact_sem"])

    S_ml_mean = np.array(data["S_ml_mean"])
    S_ml_sem = np.array(data["S_ml_sem"])

    rP_all = np.array(data["r_P_all"])
    rP_mean = data["r_P_mean"]
    rP_std = data["r_P_std"]
    rP_sem = data["r_P_sem"]

    ell = np.arange(1, L + 1)

    fig, ax = plt.subplots(figsize=(6, 4))

    # ===============================
    # SDRG
    # ===============================
    ax.plot(
        ell,
        S_exact_mean,
        linewidth=2,
        label="SDRG",
    )

    ax.fill_between(
        ell,
        S_exact_mean - S_exact_sem,
        S_exact_mean + S_exact_sem,
        alpha=0.25,
    )

    # ===============================
    # GNN-SDRG
    # ===============================
    ax.plot(
        ell,
        S_ml_mean,
        "--",
        linewidth=2,
        label="GNN-SDRG",
    )

    ax.fill_between(
        ell,
        S_ml_mean - S_ml_sem,
        S_ml_mean + S_ml_sem,
        alpha=0.25,
    )

    ax.set_xlabel(r"$\ell$")
    ax.set_ylabel(r"$S(\ell)$")
    ax.legend()

    # ===============================
    # Text inset
    # use STD here because histogram
    # shows distribution
    # ===============================
    ax.text(
        0.05,
        0.95,
        rf"$r_P = {rP_mean:.3f} \pm {rP_std:.3f}$",
        transform=ax.transAxes,
        fontsize=11,
        verticalalignment="top",
        bbox=dict(
            boxstyle="round",
            facecolor="white",
            alpha=0.85,
        ),
    )

    # ===============================
    # Histogram inset
    # ===============================
    ax_hist = inset_axes(
        ax,
        width="35%",
        height="35%",
        loc="center",
        borderpad=1,
    )

    ax_hist.hist(
        rP_all,
        bins=30,
        density=True,
        alpha=0.8,
    )

    ax_hist.axvline(
        rP_mean,
        linestyle="--",
        linewidth=1,
        color="red",
    )

    ax_hist.set_xlabel(r"$r_P$", fontsize=8)
    ax_hist.set_ylabel("PDF", fontsize=8)
    ax_hist.tick_params(axis="both", labelsize=8)
    ax_hist.set_xlim(0, 1)

    plt.tight_layout()
    plt.savefig(out_png, dpi=150)
    plt.close()

    print(f"Saved figure to {out_png}")

    print(
        f"r_P = {rP_mean:.4f} ± {rP_std:.4f} (std)"
    )

    print(
        f"r_P = {rP_mean:.4f} ± {rP_sem:.4f} (sem)"
    )


if __name__ == "__main__":

    alpha = 2.0
    N = 100
    L = 1000

    plot_from_json(
        json_path="entropy_ml_vs_exact.json",
        out_png=f"entropy_linear_N{N}_L{L}_alpha{alpha}.png",
    )