"""Plot primary stationary-mode v_rms for AF1 through AF0.0001."""
from pathlib import Path
import json

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


ROOT = Path(__file__).resolve().parents[1]
BASE = ROOT / "results/beta42_af1_steady"
OUT = BASE / "amplitude_reduction/five_case_vrms"
CASES = [
    ("AF1", BASE / "af1_af01_af001_harmonic_growth/af1/march.csv", "#1764a0"),
    ("AF0.1", BASE / "amplitude_reduction/spatial/factor_0p1/march.csv", "#D55E00"),
    ("AF0.01", BASE / "amplitude_reduction/spatial/factor_0p01_dr8/march.csv", "#009E73"),
    ("AF0.001", BASE / "amplitude_reduction/first_peak_extension/af0p001/march.csv", "#CC79A7"),
    ("AF0.0001", BASE / "amplitude_reduction/first_peak_extension/af0p0001/march.csv", "#E69F00"),
]


def fitted_peak(re, vrms):
    i = int(np.argmax(vrms))
    if i == 0 or i == len(vrms)-1:
        return float(re[i]), float(vrms[i]), False
    c = np.polyfit(re[i-1:i+2], np.log(vrms[i-1:i+2]), 2)
    x = -c[1]/(2*c[0])
    return float(x), float(np.exp(np.polyval(c, x))), True


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    plt.rcParams.update({
        "font.family":"serif", "font.serif":["Times New Roman","STIXGeneral","DejaVu Serif"],
        "mathtext.fontset":"stix", "font.size":11, "axes.linewidth":1,
        "xtick.direction":"in", "ytick.direction":"in", "xtick.top":True,
        "ytick.right":True, "legend.frameon":False, "pdf.fonttype":42,
    })
    fig, axes = plt.subplots(1, 2, figsize=(9.6, 3.9), constrained_layout=True)
    report = []
    assembled = []
    for label, path, color in CASES:
        d = np.genfromtxt(path, delimiter=",", names=True)
        re = d["Re"]
        vrms = d["A1_vtheta_over_disk_speed"]
        rp, ap, bracketed = fitted_peak(re, vrms)
        axes[0].plot(re, vrms, color=color, lw=1.9, label=label)
        axes[1].semilogy(re, vrms, color=color, lw=1.9, label=label)
        if bracketed:
            axes[0].plot(rp, ap, "o", color=color, ms=4)
        report.append({"case":label, "inlet_vrms":float(vrms[0]),
                       "first_peak_Re":rp, "first_peak_vrms":ap,
                       "peak_bracketed":bracketed, "last_Re":float(re[-1])})
        assembled.append(np.column_stack((np.full_like(re, len(report)-1), re, vrms)))

    ylabel = r"$v_{\mathrm{rms},42}/(\Omega_d r)$"
    axes[0].set(xlim=(320, 500), ylim=(0, .135), xlabel=r"$Re$", ylabel=ylabel,
                title="(a) Primary stationary-mode RMS")
    axes[1].set(xlim=(320, 500), xlabel=r"$Re$", ylabel=ylabel,
                title="(b) Logarithmic scale")
    for ax in axes:
        ax.legend(fontsize=8.5, ncol=1)
        ax.tick_params(length=3.5)
    for ext in ("png", "pdf"):
        fig.savefig(OUT/f"five_inlet_vrms.{ext}", dpi=300, facecolor="white")
    plt.close(fig)
    (OUT/"five_inlet_vrms_summary.json").write_text(json.dumps(report, indent=2)+"\n")
    np.savetxt(OUT/"five_inlet_vrms.csv", np.vstack(assembled), delimiter=",",
               header="case_index,Re,v_rms42_over_disk_speed", comments="")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
