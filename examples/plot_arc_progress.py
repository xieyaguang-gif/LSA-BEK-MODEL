"""Plot accepted damping-arclength stations; exclude Newton checkpoints."""
from pathlib import Path
import json

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


ROOT = Path(__file__).resolve().parents[1]
RUN = ROOT / "results/radial_extension/arc740_quadratic64_resume_h5_to700"
BASE = ROOT / "results/radial_extension/release740_h5_to700/tau0.450000_try03"


def main():
    summary = json.loads((RUN / "summary.json").read_text())
    records = [{"release_fraction": 0.45, "folder": str(BASE), "status": "converged_auxiliary_damping"}]
    records += [r for r in summary["stations"] if r["status"] == "converged_auxiliary_damping"]
    records.sort(key=lambda r: r["release_fraction"])

    targets = np.array([0.45, 0.50, 0.545, 0.582, 0.609, records[-1]["release_fraction"]])
    chosen = []
    for target in targets:
        record = min(records, key=lambda r: abs(r["release_fraction"] - target))
        if record not in chosen:
            chosen.append(record)

    samples = {str(x): [] for x in (550, 600, 650, 700)}
    fractions = []
    for record in records:
        folder = Path(record["folder"])
        if not folder.is_absolute():
            folder = RUN / folder
        data = np.loadtxt(folder / "energy.csv", delimiter=",", comments="#")
        fractions.append(record["release_fraction"])
        for re in samples:
            samples[re].append(float(np.interp(float(re), data[:, 0], data[:, 1])))

    plt.rcParams.update({
        "font.family": "serif",
        "font.serif": ["Times New Roman", "STIXGeneral", "DejaVu Serif"],
        "mathtext.fontset": "stix", "font.size": 10,
        "axes.linewidth": 0.9, "xtick.direction": "in", "ytick.direction": "in",
        "xtick.top": True, "ytick.right": True,
    })
    fig, (ax, bx) = plt.subplots(1, 2, figsize=(9.2, 3.7), constrained_layout=True)
    colors = plt.cm.viridis(np.linspace(0.08, 0.92, len(chosen)))
    for color, record in zip(colors, chosen):
        folder = Path(record["folder"])
        if not folder.is_absolute():
            folder = RUN / folder
        data = np.loadtxt(folder / "energy.csv", delimiter=",", comments="#")
        ax.semilogy(data[:, 0], data[:, 1], color=color, lw=1.35,
                    label=rf'$\tau={record["release_fraction"]:.3f}$')
    ax.axvspan(535, 740, color="0.92", zorder=-10)
    ax.axvline(535, color="0.35", lw=0.9, ls="--")
    ax.text(538, 2e-7, "auxiliary damping region", color="0.35", fontsize=8,
            rotation=90, va="bottom")
    ax.set_xlim(350, 740)
    ax.set_xlabel(r"$Re$")
    ax.set_ylabel(r"$E_1$")
    ax.set_title("(a) Accepted steady solutions")
    ax.legend(frameon=False, ncol=2, fontsize=8, loc="upper right")

    markers = ["o", "s", "^", "D"]
    line_colors = ["#0072B2", "#D55E00", "#009E73", "#CC79A7"]
    for (re, values), marker, color in zip(samples.items(), markers, line_colors):
        bx.semilogy(fractions, values, marker=marker, ms=3.8, lw=1.15,
                    color=color, label=rf'$Re={re}$')
    bx.axvline(records[-1]["release_fraction"], color="0.45", lw=0.8, ls=":")
    bx.set_xlabel(r"auxiliary damping release $\tau$")
    bx.set_ylabel(r"$E_1$")
    bx.set_title("(b) Continuation progress")
    bx.legend(frameon=False, fontsize=8)
    bx.text(0.03, 0.03, r"Only $\tau=1$ is the target problem.",
            transform=bx.transAxes, fontsize=8, color="0.3")

    for axis in (ax, bx):
        axis.tick_params(length=3.5)
    fig.savefig(RUN / "accepted_progress.png", dpi=300, facecolor="white")
    fig.savefig(RUN / "accepted_progress.pdf", facecolor="white")
    plt.close(fig)

    report = {
        "status": "auxiliary_continuation_snapshot",
        "accepted_station_count": len(records),
        "latest_release_fraction": records[-1]["release_fraction"],
        "sample_Re": {key: values[-1] for key, values in samples.items()},
        "interpretation": "All curves contain auxiliary damping beyond Re=535; only tau=1 is the Re=700 target problem.",
    }
    (RUN / "accepted_progress.json").write_text(json.dumps(report, indent=2) + "\n")


if __name__ == "__main__":
    main()
