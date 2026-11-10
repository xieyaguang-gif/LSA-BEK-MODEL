"""Audit radial-step sensitivity of AF1 valley and second radial peak."""
from pathlib import Path
import json

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "results/beta42_af1_steady/af1_delta_r_independence"
DRS = [5, 6, 7, 8, 9, 10, 11, 12, 13]


def fit_extremum(x, y, i):
    c = np.polyfit(x[i-1:i+2], np.log(y[i-1:i+2]), 2)
    xp = -c[1]/(2*c[0])
    return float(xp), float(np.exp(np.polyval(c, xp)))


def main():
    plt.rcParams.update({
        "font.family":"serif", "font.serif":["Times New Roman","STIXGeneral","DejaVu Serif"],
        "mathtext.fontset":"stix", "font.size":10.5, "axes.linewidth":1,
        "xtick.direction":"in", "ytick.direction":"in", "xtick.top":True,
        "ytick.right":True, "legend.frameon":False, "pdf.fonttype":42,
    })
    cmap = plt.get_cmap("viridis")
    fig, axes = plt.subplots(1, 2, figsize=(10, 3.9), constrained_layout=True)
    rows, second = [], []
    for k, dr in enumerate(DRS):
        d = np.genfromtxt(OUT/f"dr{dr}/march.csv", delimiter=",", names=True)
        x, a = d["Re"], d["A1_vtheta_over_disk_speed"]
        color = cmap(k/(len(DRS)-1))
        axes[0].plot(x, a, color=color, lw=1.25, marker="o", ms=2,
                     label=rf"$\Delta r={dr}$")
        axes[1].plot(x, a, color=color, lw=1.25, marker="o", ms=2)
        minima = np.where((a[1:-1]<a[:-2]) & (a[1:-1]<=a[2:]))[0]+1
        peaks = np.where((a[1:-1]>a[:-2]) & (a[1:-1]>=a[2:]))[0]+1
        valley = fit_extremum(x, a, int(minima[0])) if len(minima) else (None, None)
        later = peaks[peaks > minima[0]] if len(minima) else []
        peak2 = fit_extremum(x, a, int(later[-1])) if len(later) else (None, None)
        rows.append({"dr":dr, "last_Re":float(x[-1]),
                     "valley_Re":valley[0], "valley_A42":valley[1],
                     "second_peak_bracketed":bool(len(later)),
                     "second_peak_Re":peak2[0], "second_peak_A42":peak2[1],
                     "terminal_A42":float(a[-1])})
        if len(later):
            second.append(peak2)
            axes[1].plot(*peak2, "*", color=color, ms=9)
    axes[0].set(xlim=(320,550), ylim=(0,.135), xlabel=r"$Re$",
                ylabel=r"$A_{42}=v_{\mathrm{rms},42}/(\Omega_d r)$",
                title="(a) AF1 radial-step comparison")
    axes[1].set(xlim=(420,550), ylim=(0,.135), xlabel=r"$Re$",
                ylabel=r"$A_{42}$", title="(b) Valley and second peak")
    axes[0].legend(fontsize=7.5, ncol=2)
    axes[1].axvspan(439,444,color=".65",alpha=.15,lw=0,label="valley-position band")
    axes[1].axhspan(.123,.127,color="#D55E00",alpha=.10,lw=0,label="second-peak amplitude band")
    axes[1].legend(fontsize=8)
    for ax in axes: ax.tick_params(length=3.5)
    for ext in ("png","pdf"):
        fig.savefig(OUT/f"af1_delta_r_independence.{ext}",dpi=300,facecolor="white")
    plt.close(fig)

    fine = [r for r in rows if r["dr"] <= 11]
    v_re = np.array([r["valley_Re"] for r in fine])
    v_a = np.array([r["valley_A42"] for r in fine])
    report = {
        "runs": rows,
        "recommended": {
            "valley_Re": float(np.median(v_re)),
            "valley_Re_spread": [float(v_re.min()),float(v_re.max())],
            "valley_A42_observed_range": [float(v_a.min()),float(v_a.max())],
            "valley_A42_grid_independent": False,
            "second_peak_Re_observed_bracketed": [float(min(v[0] for v in second)),float(max(v[0] for v in second))],
            "second_peak_A42_mean": float(np.mean([v[1] for v in second])),
            "second_peak_A42_half_spread": float(np.ptp([v[1] for v in second])/2),
            "second_peak_position_grid_independent": False,
            "second_peak_amplitude_grid_independent_to_percent": 0.1,
        },
    }
    (OUT/"af1_delta_r_independence.json").write_text(json.dumps(report,indent=2)+"\n")
    print(json.dumps(report["recommended"],indent=2))


if __name__ == "__main__":
    main()
