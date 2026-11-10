"""Locate the first A42 peaks for AF0.001 and AF0.0001."""
from pathlib import Path
import json

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "results/beta42_af1_steady/amplitude_reduction/first_peak_extension"
CASES = {
    "AF0.001": ("af0p001", "af0p001_dr10"),
    "AF0.0001": ("af0p0001", "af0p0001_dr10"),
}


def load(name):
    return np.genfromtxt(OUT/name/"march.csv", delimiter=",", names=True)


def peak(data):
    x, y = data["Re"], data["A1_vtheta_over_disk_speed"]
    i = int(np.argmax(y))
    if i == 0 or i == len(y)-1:
        raise RuntimeError("Peak is not bracketed")
    coefficient = np.polyfit(x[i-1:i+2], np.log(y[i-1:i+2]), 2)
    xp = -coefficient[1]/(2*coefficient[0])
    return float(xp), float(np.exp(np.polyval(coefficient, xp)))


def main():
    plt.rcParams.update({
        "font.family":"serif", "font.serif":["Times New Roman","STIXGeneral","DejaVu Serif"],
        "mathtext.fontset":"stix", "font.size":11, "axes.linewidth":1,
        "xtick.direction":"in", "ytick.direction":"in", "xtick.top":True,
        "ytick.right":True, "legend.frameon":False, "pdf.fonttype":42,
    })
    fig, axes = plt.subplots(1, 2, figsize=(9.2, 3.7), constrained_layout=True)
    report = {}
    for ax, (label, names) in zip(axes, CASES.items()):
        estimates = []
        for name, dr, style, color in zip(names, (8, 10), ("-", "--"), ("#1764a0", "#D55E00")):
            data = load(name)
            xp, yp = peak(data)
            estimates.append((xp, yp))
            ax.plot(data["Re"], data["A1_vtheta_over_disk_speed"], style,
                    color=color, lw=1.8, marker="o", ms=3, label=rf"$\Delta r={dr}$")
            ax.plot(xp, yp, "*", color=color, ms=9)
        xvalues = [v[0] for v in estimates]
        yvalues = [v[1] for v in estimates]
        report[label] = {
            "reported_peak_from_dr8": {"Re": estimates[0][0], "A42": estimates[0][1]},
            "dr10_check": {"Re": estimates[1][0], "A42": estimates[1][1]},
            "step_spread": {"Re": abs(xvalues[1]-xvalues[0]), "A42": abs(yvalues[1]-yvalues[0])},
            "recommended_rounded": {
                "Re": round(estimates[0][0]),
                "Re_uncertainty": round(abs(xvalues[1]-xvalues[0])+0.5),
                "A42": round(estimates[0][1], 3),
            },
        }
        ax.set(xlim=(430, 490), ylim=(0, .135), xlabel=r"$Re$", ylabel=r"$A_{42}$",
               title=label)
        ax.legend(fontsize=9)
        ax.tick_params(length=3.5)
    for ext in ("png", "pdf"):
        fig.savefig(OUT/f"weak_inlet_first_peaks.{ext}", dpi=300, facecolor="white")
    plt.close(fig)
    (OUT/"first_peak_summary.json").write_text(json.dumps(report, indent=2)+"\n")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
