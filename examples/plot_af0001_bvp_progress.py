"""Plot the last converged AF0.001 full-BVP forcing-release checkpoint."""
from __future__ import annotations

import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from scipy.interpolate import BarycentricInterpolator

from llsa.secondary.radial_continuation import load_seed


ROOT = Path(__file__).resolve().parents[1]
RUN = ROOT / "results/beta42_af1_steady/lower_af_full_bvp/af0p001_s480_homotopy_resume"
CHECKPOINT = RUN / "checkpoint_lambda_0.109375.npz"
PSE = ROOT / "results/beta42_af1_steady/amplitude_reduction/first_peak_extension/af0p001/march.csv"
OUT = RUN / "progress_plot"
PROBE_Z = 1.1858025795710305


def probe_amplitudes(path: Path):
    config, radii, modes = load_seed(path)
    op = config.operators(radii[0])
    finite = np.isfinite(op.grid.z)
    interpolator = BarycentricInterpolator(op.grid.z[finite],
        modes.reshape(len(radii), 2*config.harmonics+1, 4, op.m)
             [:, config.harmonics+1:, 1, finite], axis=2)
    probe = interpolator(PROBE_Z)
    # A real Fourier pair q_j exp(ij beta theta)+c.c. has RMS sqrt(2)|q_j|.
    amplitude = np.sqrt(2)*np.abs(probe)/op.disk_rotation
    return abs(config.ro)*radii, amplitude


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    re, amp = probe_amplitudes(CHECKPOINT)
    pse = np.genfromtxt(PSE, delimiter=",", names=True)
    re_pse = pse["Re"]
    amp_pse = np.column_stack([pse[f"A{j}_vtheta_over_disk_speed"] for j in range(1, 6)])
    physical = re <= 480+1e-10
    physical_pse = re_pse <= 480+1e-10
    ipeak = np.flatnonzero(physical)[np.argmax(amp[physical, 0])]

    report = {
        "case": "AF0.001, beta=42, Ro=-0.751",
        "checkpoint": str(CHECKPOINT.relative_to(ROOT)),
        "forcing_fraction_lambda": 0.109375,
        "forcing_removed_fraction": 0.890625,
        "checkpoint_relative_residual": 3.9149875649712244e-10,
        "physical_interval_Re": [float(re[0]), 480.0],
        "sponge_interval_Re": [480.0, 520.0],
        "primary_peak_before_sponge": {"Re": float(re[ipeak]),
            "v_rms_over_disk_speed": float(amp[ipeak, 0])},
        "interpretation": "Converged auxiliary-forcing checkpoint; lambda=0 target is not reached."
    }
    (OUT/"summary.json").write_text(json.dumps(report, indent=2)+"\n")
    rows = np.column_stack((re, amp))
    np.savetxt(OUT/"checkpoint_probe_amplitudes.csv", rows, delimiter=",",
        header="Re,"+",".join(f"A{j}_vtheta_rms_over_disk_speed" for j in range(1,6)), comments="")

    plt.rcParams.update({
        "font.family": "serif", "font.serif": ["Times New Roman", "Nimbus Roman", "DejaVu Serif"],
        "mathtext.fontset": "stix", "font.size": 10.5, "axes.labelsize": 12,
        "axes.linewidth": 1.0, "xtick.direction": "in", "ytick.direction": "in",
        "xtick.top": True, "ytick.right": True, "legend.frameon": False,
        "pdf.fonttype": 42, "savefig.dpi": 300})
    fig, axes = plt.subplots(1, 2, figsize=(9.2, 3.75), layout="constrained")
    axes[0].semilogy(re_pse[physical_pse], amp_pse[physical_pse,0], color=".45", ls="--", lw=1.7,
                    label="PSE seed")
    axes[0].semilogy(re[physical], amp[physical,0], color="#0072B2", lw=2.0,
                    label=r"Full BVP checkpoint, $\lambda=0.109375$")
    axes[0].plot(re[ipeak], amp[ipeak,0], "o", color="#D55E00", ms=5,
                 label=rf"peak: $Re={re[ipeak]:.1f}$")
    axes[0].set_ylabel(r"$v_{\mathrm{rms},42}/(\Omega_d r)$")
    axes[0].set_title("(a) Primary stationary mode")
    colors = ["#0072B2", "#D55E00", "#009E73", "#CC79A7", "#E69F00"]
    for j, color in enumerate(colors):
        axes[1].semilogy(re[physical], amp[physical,j], color=color, lw=1.75,
                        label=rf"$j={j+1}$ ($\beta={42*(j+1)}$)")
    axes[1].set_title("(b) Nonlinear Fourier harmonics")
    axes[1].set_ylabel(r"$v_{j,\mathrm{rms}}/(\Omega_d r)$")
    for ax in axes:
        ax.set_xlabel(r"$Re$")
        ax.set_xlim(re[0], 480)
        ax.grid(color=".9", lw=.55)
        ax.legend(fontsize=8.2)
    axes[0].set_ylim(1e-7, .3)
    positive = amp[physical][amp[physical] > 0]
    axes[1].set_ylim(max(1e-14, positive.min()/2), .3)
    fig.suptitle(r"AF0.001 progress: 89.1% of auxiliary PSE forcing removed", fontsize=12.5)
    for ext in ("png", "pdf"):
        fig.savefig(OUT/f"af0001_bvp_progress.{ext}", facecolor="white")
    plt.close(fig)
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
