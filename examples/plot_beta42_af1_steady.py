"""Compare the converged steady AF1 primary wave with the DNS stationary mode."""
from pathlib import Path
import json

import numpy as np
from scipy.interpolate import BarycentricInterpolator
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from llsa.secondary.radial import RadialConfig


ROOT = Path(__file__).resolve().parents[1]
CASE = ROOT / "results/beta42_af1_steady/global_s460_end500_exact_inlet"
DNS = ROOT.parent / "semtex-gpu_work/cases/00_reference/af1/analysis/secondary_amplitude_sweep_20260914/af1_semtex-radial.csv"
DNS_RE_PER_R = 0.718373556
PROBE_Z = 1.1858025795710305


def main():
    with np.load(CASE / "fields.npz") as data:
        radii, modes = data["radii"], data["modes"]
        config = RadialConfig(**json.loads(str(data["config_json"])))
    op = config.operators(radii[0])
    v1 = modes[:, config.harmonics + 1, op.m:2*op.m]
    # The rational Chebyshev grid's last point represents z=infinity.  Its
    # velocity is a boundary value, but infinity cannot be passed as an
    # interpolation node.  The finite nodes already bracket this probe.
    finite_z = np.isfinite(op.grid.z)
    probe = BarycentricInterpolator(op.grid.z[finite_z], v1[:, finite_z], axis=1)(PROBE_Z)
    model_re = abs(config.ro)*radii
    model_a = np.sqrt(2)*np.abs(probe)/op.disk_rotation
    energy = np.loadtxt(CASE / "energy.csv", delimiter=",", comments="#")

    dns = np.genfromtxt(DNS, delimiter=",", names=True)
    dns_re = DNS_RE_PER_R*dns["r"]
    use = (dns_re >= 320) & (dns_re <= 500)
    dns_re = dns_re[use]
    dns_a = dns["stationary_primary_q1_RMS"][use]
    dns_temporal = dns["temporal_fraction"][use]

    physical = model_re <= 460 + 1e-12
    mi = np.flatnonzero(physical)[np.argmax(model_a[physical])]
    dns_physical = dns_re <= 460 + 1e-12
    di = np.flatnonzero(dns_physical)[np.argmax(dns_a[dns_physical])]
    f10 = np.flatnonzero((dns_temporal >= .1) & (dns["temporal_RMS"][use] >= .005))
    f10_re = float(dns_re[f10[0]]) if len(f10) else None
    common = np.linspace(320, 460, 561)
    dns_common = np.interp(common, dns_re, dns_a)
    model_common = np.interp(common, model_re, model_a)

    metrics = {
        "case": "Ro=-0.751, beta=42, disk-frame stationary, AF1 inlet",
        "inlet_amplitude": 0.001609634029544909,
        "inlet_calibration": "AF0.1 validated inlet multiplied by AF1/AF0.1 stationary-primary ratio at DNS coordinate corresponding to Re=320",
        "dns_coordinate": "Re=0.718373556*r_dns",
        "model_peak": {"Re": float(model_re[mi]), "A42": float(model_a[mi])},
        "dns_stationary_peak": {"Re": float(dns_re[di]), "A42": float(dns_a[di])},
        "peak_Re_difference_model_minus_dns": float(model_re[mi]-dns_re[di]),
        "mean_relative_A_error_Re320_460": float(np.mean(abs(model_common/dns_common-1))),
        "dns_temporal_fraction_10_percent_Re": f10_re,
        "model_relative_residual": json.loads((CASE/"metadata.json").read_text())["relative_residual"],
        "interpretation_limit_Re": 460.0,
        "limitation": "The model is a steady beta-harmonic base state; it intentionally excludes time-dependent secondary instability.",
    }
    (CASE / "comparison_metrics.json").write_text(json.dumps(metrics, indent=2) + "\n")
    np.savetxt(CASE / "model_probe.csv", np.column_stack((model_re, model_a)), delimiter=",",
               header="Re,A42_vtheta_over_disk_speed", comments="")

    plt.rcParams.update({
        "font.family": "serif", "font.serif": ["Times New Roman", "STIXGeneral", "DejaVu Serif"],
        "mathtext.fontset": "stix", "font.size": 10, "axes.linewidth": .9,
        "xtick.direction": "in", "ytick.direction": "in", "xtick.top": True,
        "ytick.right": True, "pdf.fonttype": 42,
    })
    fig, (ax, bx) = plt.subplots(1, 2, figsize=(9.3, 3.7), constrained_layout=True)
    ax.plot(dns_re, dns_a, color="black", lw=1.8, label=r"AF1 DNS: stationary $\beta=42$")
    ax.plot(model_re[physical], model_a[physical], color="#1764a0", lw=1.8,
            label="Steady global solution")
    ax.plot(dns_re[di], dns_a[di], "o", color="black", ms=4)
    ax.plot(model_re[mi], model_a[mi], "s", color="#1764a0", ms=4)
    ax.set(xlim=(320, 460), xlabel=r"$Re$", ylabel=r"$A_{42}$",
           title="(a) Primary-wave amplitude")
    ax.legend(frameon=False, fontsize=8, loc="upper left")

    colors = ["#1764a0", "#D55E00", "#009E73", "#CC79A7", "#E69F00"]
    for j, color in enumerate(colors, 1):
        bx.semilogy(energy[physical, 0], energy[physical, j], color=color, lw=1.35,
                    label=rf"$E_{j}$")
    bx.set(xlim=(320, 460), xlabel=r"$Re$", ylabel="integral harmonic energy",
           title="(b) Nonlinear harmonic transfer")
    bx.legend(frameon=False, fontsize=8, ncol=2)
    for axis in (ax, bx):
        axis.tick_params(length=3.5)
    fig.savefig(CASE / "af1_steady_comparison.png", dpi=300, facecolor="white")
    fig.savefig(CASE / "af1_steady_comparison.pdf", facecolor="white")
    plt.close(fig)
    print(json.dumps(metrics, indent=2))


if __name__ == "__main__":
    main()
