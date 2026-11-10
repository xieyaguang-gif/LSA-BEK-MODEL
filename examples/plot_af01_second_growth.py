"""Plot the resolved AF0.1 decay minimum and second-growth branch."""
from pathlib import Path
import json

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "results/beta42_af1_steady/amplitude_reduction/second_growth"


def load(tag):
    return np.genfromtxt(OUT/tag/"march.csv", delimiter=",", names=True)


def extrema(d):
    x, a = d["Re"], d["A1_vtheta_over_disk_speed"]
    peaks = np.where((a[1:-1] > a[:-2]) & (a[1:-1] >= a[2:]))[0]+1
    valleys = np.where((a[1:-1] < a[:-2]) & (a[1:-1] <= a[2:]))[0]+1
    return int(peaks[0]), int(valleys[0])


def main():
    plt.rcParams.update({
        "font.family":"serif", "font.serif":["Times New Roman","STIXGeneral","DejaVu Serif"],
        "mathtext.fontset":"stix", "font.size":11, "axes.linewidth":1,
        "xtick.direction":"in", "ytick.direction":"in", "xtick.top":True,
        "ytick.right":True, "legend.frameon":False, "pdf.fonttype":42,
    })
    cases = [("af0p1_dr8", 8, "#777777", "--"),
             ("af0p1_dr12", 12, "#1764a0", "-")]
    fig, axes = plt.subplots(1, 2, figsize=(9.6, 3.8), constrained_layout=True)
    report = {}
    for tag, dr, color, style in cases:
        d = load(tag); re=d["Re"]; rd=d["r_delta"]
        amp=d["A1_vtheta_over_disk_speed"]
        ip, iv = extrema(d)
        axes[0].plot(re, amp, style, color=color, lw=1.9, marker="o", ms=2.8,
                     label=rf"$\Delta r={dr}$")
        axes[1].semilogy(re, amp, style, color=color, lw=1.9, marker="o", ms=2.8,
                         label=rf"$\Delta r={dr}$")
        axes[0].plot(re[ip],amp[ip],"o",color=color,ms=6)
        axes[0].plot(re[iv],amp[iv],"s",color=color,ms=6)
        fit = np.polyfit(rd[iv:], np.log(amp[iv:]), 1)
        report[f"dr{dr}"] = {
            "first_peak": {"Re":float(re[ip]), "v_rms":float(amp[ip])},
            "valley": {"Re":float(re[iv]), "v_rms":float(amp[iv])},
            "last": {"Re":float(re[-1]), "v_rms":float(amp[-1])},
            "second_growth_amplification": float(amp[-1]/amp[iv]),
            "second_growth_fit_dlnA_dr_delta": float(fit[0]),
        }
    ylabel=r"$v_{\mathrm{rms},42}/(\Omega_d r)$"
    axes[0].set(xlim=(320,580),ylim=(0,.12),xlabel=r"$Re$",ylabel=ylabel,
                title="(a) AF0.1 nonlinear evolution")
    axes[1].set(xlim=(320,580),xlabel=r"$Re$",ylabel=ylabel,
                title="(b) Decay and second growth")
    for ax in axes:
        ax.legend(fontsize=9)
        ax.tick_params(length=3.5)
        ax.axvspan(470, 490, color=".7", alpha=.12, lw=0)
    axes[0].annotate("second growth", xy=(545.3,.0523), xytext=(500,.095),
                     arrowprops=dict(arrowstyle="->",lw=1), fontsize=10)
    for ext in ("png","pdf"):
        fig.savefig(OUT/f"af01_second_growth.{ext}",dpi=300,facecolor="white")
    plt.close(fig)
    (OUT/"af01_second_growth_summary.json").write_text(json.dumps(report,indent=2)+"\n")
    print(json.dumps(report,indent=2))


if __name__ == "__main__":
    main()
