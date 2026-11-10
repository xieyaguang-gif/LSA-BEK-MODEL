"""Compare full-BVP and DNS beta=42 amplitudes and radial growth rates."""
from __future__ import annotations

import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from scipy.interpolate import BarycentricInterpolator
from scipy.signal import find_peaks

from llsa.secondary.radial_continuation import load_seed
from plot_dns_last_period_linear_growth import dns_case, filtered_curve

ROOT = Path(__file__).resolve().parents[1]
DNS_ROOT = ROOT.parent / "semtex-gpu_work/cases/00_reference/af1/analysis/secondary_amplitude_sweep_20260914"
OUT = ROOT / "results/beta42_bvp_dns_three_amplitudes"
PROBE_Z = 1.1858025795710305
CASES = {
    "AF1": {
        "dns": DNS_ROOT / "af1_nek.npz",
        "bvp": ROOT / "results/beta42_af1_steady/global_s460_end500_exact_inlet/fields.npz",
        "trusted_end": 460.0, "color": "#CC6677"},
    "AF0.1": {
        "dns": DNS_ROOT / "af0p1.npz",
        "bvp": ROOT / "results/radial_sponge_validation/h5_dr05_s520_w40_a10_end560_physicalseed/fields.npz",
        "trusted_end": 520.0, "color": "#D55E00"},
    "AF0.01": {
        "dns": DNS_ROOT / "af0p01.npz",
        "bvp": ROOT / "results/beta42_af1_steady/lower_af_full_bvp/af0p01_extend_s480/s480_try03/fields.npz",
        "trusted_end": 480.0, "color": "#0072B2"},
}


def bvp_case(path, trusted_end):
    config, radii, modes = load_seed(path)
    op = config.operators(radii[0])
    finite = np.isfinite(op.grid.z)
    v1 = modes[:, config.harmonics+1, op.m:2*op.m]
    probe = BarycentricInterpolator(op.grid.z[finite], v1[:, finite], axis=1)(PROBE_Z)
    re = abs(config.ro) * radii
    amplitude = np.sqrt(2) * np.abs(probe) / op.disk_rotation
    physical = re <= trusted_end + 1e-9
    x, smooth_a, growth = filtered_curve(re[physical], amplitude[physical])
    # Remove the 3-sigma derivative/filter endpoint margin from plotted growth.
    growth_valid = (x >= x[0]+6) & (x <= x[-1]-6)
    return dict(re=x, amplitude=smooth_a, growth=growth,
                growth_valid=growth_valid, raw_re=re[physical], raw_amplitude=amplitude[physical],
                trusted_end=trusted_end, residual_source=str(path))


def rmse_on_overlap(a, b, lo=330, hi=380):
    x = np.arange(max(lo, a["re"][0], b["re"][0]),
                  min(hi, a["re"][-1], b["re"][-1])+.001, .25)
    ga = np.interp(x, a["re"], a["growth"])
    gb = np.interp(x, b["re"], b["growth"])
    return float(np.sqrt(np.mean((ga-gb)**2)))


def first_peak(case):
    peaks = find_peaks(case["amplitude"], prominence=.005)[0]
    peaks = peaks[(case["re"][peaks] >= 330) & (case["amplitude"][peaks] >= .02)]
    i = int(peaks[0]) if len(peaks) else int(np.argmax(case["amplitude"]))
    return {"Re": float(case["re"][i]), "A42": float(case["amplitude"][i])}


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    result, summary, rows = {}, {"probe": {}, "cases": {}}, []
    summary["probe"] = {
        "BVP": "z_delta=1.1858025795710305; v_theta/(Omega_d r)",
        "DNS": "z=1.24 monitor; q=1 stationary coefficient over final 2*pi",
        "growth": "d ln(A42)/d r_delta; common dRe=0.25, Gaussian sigma_Re=2"}
    for name, info in CASES.items():
        dns = dns_case(info["dns"])
        bvp = bvp_case(info["bvp"], info["trusted_end"])
        result[name] = {"dns": dns, "bvp": bvp}
        summary["cases"][name] = {
            "DNS_source": str(info["dns"]), "BVP_source": str(info["bvp"]),
            "BVP_trusted_end_Re": info["trusted_end"],
            "DNS_time_window": [dns["time_start"], dns["time_end"]],
            "DNS_first_peak": first_peak(dns), "BVP_first_peak": first_peak(bvp),
            "growth_RMSE_Re330_380": rmse_on_overlap(dns, bvp)}
        for method, d in (("DNS", dns), ("BVP", bvp)):
            code = 0 if method == "DNS" else 1
            rows.append(np.column_stack((np.full(len(d["re"]), float(name[2:])),
                np.full(len(d["re"]), code), d["re"], d["amplitude"], d["growth"])))
    np.savetxt(OUT/"bvp_dns_curves.csv", np.vstack(rows), delimiter=",",
        header="AF,method_code_0_DNS_1_BVP,Re,A42_vrms,growth_dlnA_drdelta", comments="")
    (OUT/"summary.json").write_text(json.dumps(summary, indent=2)+"\n")

    plt.rcParams.update({"font.family": "serif",
        "font.serif": ["Times New Roman", "Nimbus Roman", "DejaVu Serif"],
        "mathtext.fontset": "stix", "font.size": 11, "axes.labelsize": 13,
        "axes.linewidth": 1.05, "xtick.direction": "in", "ytick.direction": "in",
        "xtick.top": True, "ytick.right": True, "legend.frameon": False,
        "legend.fontsize": 9, "pdf.fonttype": 42, "savefig.dpi": 300})
    fig, ax = plt.subplots(2, 1, figsize=(7.3, 6.9), sharex=True,
        gridspec_kw={"height_ratios": [1, 1.1]}, layout="constrained")
    for name, info in CASES.items():
        dns, bvp = result[name]["dns"], result[name]["bvp"]
        dm = (dns["re"] >= 315) & (dns["re"] <= 525)
        bm = (bvp["re"] >= 315) & (bvp["re"] <= info["trusted_end"])
        bg = bm & bvp["growth_valid"]
        ax[0].semilogy(dns["re"][dm], dns["amplitude"][dm], color=info["color"], lw=1.9,
                       label=f"{name} DNS")
        ax[0].semilogy(bvp["re"][bm], bvp["amplitude"][bm], color=info["color"], lw=1.8,
                       ls="--", label=f"{name} full BVP")
        ax[1].plot(dns["re"][dm], dns["growth"][dm], color=info["color"], lw=1.8)
        ax[1].plot(bvp["re"][bg], bvp["growth"][bg], color=info["color"], lw=1.7, ls="--")
    ax[0].set_ylabel(r"$v_{\mathrm{rms},42}/(\Omega_d r)$")
    ax[0].set_ylim(1e-6, .3)
    ax[0].legend(ncol=2, loc="lower right", fontsize=8.5)
    ax[0].set_title(r"Stationary $\beta=42$ primary wave: DNS and full radial BVP")
    ax[1].axhline(0, color=".65", lw=.8)
    ax[1].set_ylabel(r"$g_{42}=\mathrm{d}\ln v_{\mathrm{rms},42}/\mathrm{d}r_\delta$")
    ax[1].set_xlabel(r"$Re$")
    ax[1].set_ylim(-.11, .115)
    # Compact method key; color already identifies AF.
    ax[1].plot([], [], color=".25", lw=1.8, label="DNS")
    ax[1].plot([], [], color=".25", lw=1.8, ls="--", label="Full BVP")
    ax[1].legend(loc="lower left")
    for i, a in enumerate(ax):
        a.set_xlim(315, 525)
        a.grid(color=".9", lw=.55)
        a.text(.018, .965, f"({chr(97+i)})", transform=a.transAxes,
               ha="left", va="top", fontsize=13)
    for ext in ("png", "pdf"):
        fig.savefig(OUT/f"bvp_dns_vrms_growth.{ext}", facecolor="white")
    plt.close(fig)
    (OUT/"README.md").write_text(
        "# AF1, AF0.1 and AF0.01: DNS versus full BVP\n\n"
        "Solid curves are stationary beta=42 amplitudes from the final DNS disk period; "
        "dashed curves are converged full radial BVP solutions, truncated at each sponge start. "
        "Both growth rates use the same log-amplitude filter (dRe=0.25, sigma_Re=2).\n")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
