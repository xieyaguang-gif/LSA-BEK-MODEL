"""Compare radial energy and harmonic growth for AF1, AF0.1 and AF0.01."""
from pathlib import Path
import json

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from scipy.signal import savgol_filter


ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "results/beta42_af1_steady/af1_af01_af001_harmonic_growth"
CASES = {
    "AF1": OUT / "af1/march.csv",
    "AF0.1": ROOT / "results/beta42_af1_steady/amplitude_reduction/spatial/factor_0p1/march.csv",
    "AF0.01": ROOT / "results/beta42_af1_steady/amplitude_reduction/spatial/factor_0p01_dr8/march.csv",
}
COLORS = {"AF1": "#1764a0", "AF0.1": "#D55E00", "AF0.01": "#009E73"}
HARMONIC_COLORS = ["#1764a0", "#D55E00", "#009E73", "#CC79A7"]


def load(path):
    return np.genfromtxt(path, delimiter=",", names=True)


def style():
    plt.rcParams.update({
        "font.family": "serif", "font.serif": ["Times New Roman", "STIXGeneral", "DejaVu Serif"],
        "mathtext.fontset": "stix", "font.size": 10.5, "axes.linewidth": 1,
        "xtick.direction": "in", "ytick.direction": "in", "xtick.top": True,
        "ytick.right": True, "legend.frameon": False, "pdf.fonttype": 42,
    })


def main():
    style()
    data = {name: load(path) for name, path in CASES.items()}
    summary = {}

    # Common radial comparison.  E1 is the primary-wave energy; the dashed
    # curves show the sum of the first four harmonics.
    fig, axes = plt.subplots(1, 2, figsize=(9.4, 3.8), constrained_layout=True)
    for name, d in data.items():
        re = d["Re"]
        e = np.column_stack([d[f"E{j}"] for j in range(1, 5)])
        axes[0].semilogy(re, e[:, 0], color=COLORS[name], lw=1.9, label=name)
        axes[1].semilogy(re, e.sum(axis=1), color=COLORS[name], lw=1.9, label=name)
        peak = int(np.argmax(e[:, 0]))
        summary[name] = {
            "inlet_E1": float(e[0, 0]), "primary_peak_Re": float(re[peak]),
            "primary_peak_E1": float(e[peak, 0]), "last_Re": float(re[-1]),
        }
    axes[0].set(title=r"(a) Primary energy", ylabel=r"$E_1$")
    axes[1].set(title=r"(b) First four harmonics", ylabel=r"$\sum_{j=1}^{4}E_j$")
    for ax in axes:
        ax.set(xlim=(320, 460), xlabel=r"$Re$")
        ax.legend(fontsize=9)
        ax.tick_params(length=3.5)
    for ext in ("png", "pdf"):
        fig.savefig(OUT / f"three_cases_radial_energy.{ext}", dpi=300, facecolor="white")
    plt.close(fig)

    # Follow only the first outward-growing branch, ending before dE1/dR=0.
    # In log-log space p_j=d ln(E_j)/d ln(E1) is the harmonic growth exponent.
    for name, d in data.items():
        e = np.column_stack([d[f"E{j}"] for j in range(1, 5)])
        peak = int(np.argmax(e[:, 0]))
        stop = max(peak + 1, 5)
        re, e = d["Re"][:stop], e[:stop]
        positive = np.all(e > 0, axis=1)
        re, e = re[positive], e[positive]
        loge = np.log(e)
        if len(re) >= 7:
            window = min(9, len(re) if len(re) % 2 else len(re)-1)
            loge_s = savgol_filter(loge, window, 2, axis=0, mode="interp")
        else:
            loge_s = loge
        derivative = np.gradient(loge_s, re, axis=0, edge_order=2)
        exponent = derivative / derivative[:, [0]]
        valid = np.abs(derivative[:, 0]) > 0.01

        fig, axes = plt.subplots(1, 2, figsize=(9.4, 3.8), constrained_layout=True)
        for j, color in enumerate(HARMONIC_COLORS, 1):
            axes[0].loglog(e[:, 0], e[:, j-1], color=color, lw=1.8, marker="o",
                           ms=2.5, label=rf"$E_{j}$")
            axes[1].plot(re[valid], exponent[valid, j-1], color=color, lw=1.8,
                         label=rf"$p_{j}$")
        axes[0].set(xlabel=r"primary-wave energy $E_1$", ylabel=r"harmonic energy $E_j$",
                    title=f"(a) {name}: growing branch")
        axes[1].set(xlabel=r"$Re$", ylabel=r"$p_j=\mathrm{d}\ln E_j/\mathrm{d}\ln E_1$",
                    title="(b) Relative growth exponent")
        for j, color in enumerate(HARMONIC_COLORS, 1):
            axes[1].axhline(j, color=color, lw=.75, ls=":", alpha=.7)
        axes[1].set_ylim(0, 5.5)
        for ax in axes:
            ax.legend(fontsize=8.5, ncol=2)
            ax.tick_params(length=3.5)
        tag = name.lower().replace(".", "p")
        for ext in ("png", "pdf"):
            fig.savefig(OUT / f"{tag}_harmonics_vs_primary.{ext}", dpi=300, facecolor="white")
        plt.close(fig)

        # Median exponent away from inlet transients and the primary peak.
        core = valid & (np.arange(len(re)) >= 2) & (np.arange(len(re)) < len(re)-2)
        summary[name]["median_relative_growth_exponent"] = {
            f"p{j}": float(np.nanmedian(exponent[core, j-1])) for j in range(1, 5)
        }

    (OUT / "harmonic_growth_summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
