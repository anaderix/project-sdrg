import json
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1.inset_locator import inset_axes
import os


def plot_entropy_all_T(
    json_path="entropy_sdrgx_results_alpha2.0/entropy_sdrgx_ml_vs_exact.json",
    outname="entropy_sdrgx_alpha2.0_T.png",
):
    with open(json_path, "r") as f:
        data = json.load(f)

    L = data["L"]
    T_list = [float(T) for T in data["T_list"]]
    rP_mean = data.get("r_P_mean", None)
    rP_std = data.get("r_P_std", None)

    l = np.arange(1, L + 1)

    fig, ax = plt.subplots(figsize=(7, 5))

    # color cycle
    cmap = plt.get_cmap("viridis")
    colors = cmap(np.linspace(0.1, 0.9, len(T_list)))

    # -------------------------
    # Main plot
    # -------------------------
    for color, T in zip(colors, T_list):
        S_exact = np.array(data["S_exact_by_T"][str(T)])
        S_ml = np.array(data["S_ml_by_T"][str(T)])

        ax.plot(
            l, S_exact,
            color=color,
            linewidth=0.5,
            label=rf"Exact, $T={T}$"
        )
        ax.plot(
            l, S_ml,
            "--",
            color=color,
            linewidth=1.5,
            label=rf"ML, $T={T}$"
        )

    ax.set_xlabel(r" $\ell$")
    ax.set_ylabel(r"$S(\ell)$")

    title = rf"SDRG-X entanglement entropy ($N={data['N']},\, \alpha={data['alpha']}$)"
    ax.set_title(title)

    if rP_mean is not None:
        txt = rf"$r_P = {rP_mean:.3f}$"
        if rP_std is not None:
            txt = rf"$r_P = {rP_mean:.3f} \pm {rP_std:.3f}$"
        ax.text(
            0.02, 0.98, txt,
            transform=ax.transAxes,
            va="top",
            ha="left",
            fontsize=11,
            bbox=dict(boxstyle="round", facecolor="white", alpha=0.9),
        )

    ax.legend(fontsize=9, ncol=2)

    # -------------------------
    # Inset: relative difference
    # -------------------------
    axins = inset_axes(
        ax,
        width="35%",
        height="25%",
        loc="upper right",
        borderpad=1.2,
    )


    eps = 1e-8
    for color, T in zip(colors, T_list):
        S_exact = np.array(data["S_exact_by_T"][str(T)])
        S_ml = np.array(data["S_ml_by_T"][str(T)])

        rel_diff = (S_ml - S_exact) / (S_exact + eps)

        axins.plot(l, rel_diff, color=color, linewidth=0.5)

    axins.axhline(0.0, color="black", lw=1, ls="--")
    axins.set_xlim(L * 0.25, L * 0.75)
    axins.set_ylabel(r"$\delta S / S$", fontsize=9)
    axins.set_xlabel(r"$\ell$", fontsize=9)
    axins.tick_params(axis="both", labelsize=8)

    axins.set_title("Relative difference (ML − Exact)", fontsize=9)

    plt.tight_layout()
    plt.savefig(outname, dpi=300)
    plt.show()

    print(f"Saved figure to {outname}")


if __name__ == "__main__":
    plot_entropy_all_T()
