"""Plot beta=42 nonlinear spatial evolution for four reduced inlet amplitudes."""
from pathlib import Path
import json

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "results/beta42_af1_steady/amplitude_reduction"
CASES = [
    (0.1, "spatial/factor_0p1/march.csv", "#1764a0"),
    (0.01, "spatial/factor_0p01_dr8/march.csv", "#D55E00"),
    (0.001, "spatial/factor_0p001_dr8/march.csv", "#009E73"),
    (0.0001, "spatial/factor_0p0001/march.csv", "#CC79A7"),
]


def main():
    plt.rcParams.update({
        "font.family": "serif", "font.serif": ["Times New Roman", "STIXGeneral", "DejaVu Serif"],
        "mathtext.fontset": "stix", "font.size": 11, "axes.linewidth": 1,
        "xtick.direction": "in", "ytick.direction": "in", "xtick.top": True,
        "ytick.right": True, "legend.frameon": False, "pdf.fonttype": 42,
    })
    fig, axes = plt.subplots(1, 2, figsize=(9.4, 3.8), constrained_layout=True)
    summary = []
    combined = []
    for factor, relative, color in CASES:
        data = np.genfromtxt(OUT / relative, delimiter=",", names=True)
        re = data["Re"]
        amp = data["A1_vtheta_over_disk_speed"]
        label = rf"$A_{{in}}={factor:g}A_{{in,AF1}}$"
        axes[0].plot(re, amp, lw=1.8, color=color, label=label)
        axes[1].semilogy(re, amp, lw=1.8, color=color, label=label)
        peak = int(np.argmax(amp))
        summary.append({
            "inlet_factor": factor,
            "inlet_A42": float(amp[0]),
            "peak_Re_in_computed_interval": float(re[peak]),
            "peak_A42_in_computed_interval": float(amp[peak]),
            "last_Re": float(re[-1]),
            "last_A42": float(amp[-1]),
            "radial_step": 8.0 if "dr8" in relative else 6.0,
        })
        combined.append(np.column_stack((np.full_like(re, factor), re, amp)))

    axes[0].set(xlim=(320, 460), ylim=(0, None), xlabel=r"$Re$", ylabel=r"$A_{42}$",
                title="(a) Nonlinear amplitude")
    axes[1].set(xlim=(320, 460), xlabel=r"$Re$", ylabel=r"$A_{42}$",
                title="(b) Logarithmic scale")
    axes[0].legend(loc="upper left", fontsize=8.5)
    axes[1].legend(loc="lower right", fontsize=8.5)
    for ax in axes:
        ax.tick_params(length=3.5)

    for ext in ("png", "pdf"):
        fig.savefig(OUT / f"beta42_reduced_inlet_A42.{ext}", dpi=300, facecolor="white")
    plt.close(fig)
    (OUT / "reduced_inlet_summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    np.savetxt(OUT / "reduced_inlet_A42.csv", np.vstack(combined), delimiter=",",
               header="inlet_factor,Re,A42", comments="")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
