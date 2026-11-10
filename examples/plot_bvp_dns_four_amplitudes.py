"""Compare DNS and radial-BVP beta=42 evolution for four inlet amplitudes."""
from __future__ import annotations

import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import numpy as np
from scipy.interpolate import BarycentricInterpolator
from scipy.signal import find_peaks

from llsa.secondary.radial_continuation import load_seed
from plot_dns_last_period_linear_growth import dns_case, filtered_curve


ROOT = Path(__file__).resolve().parents[1]
DNS_ROOT = ROOT.parent / "semtex-gpu_work/cases/00_reference/af1/analysis/secondary_amplitude_sweep_20260914"
OUT = ROOT / "results/beta42_bvp_dns_three_amplitudes"
PROBE_Z = 1.1858025795710305
AF1_PSE = ROOT / "results/beta42_af1_steady/af1_delta_r_independence/dr8/march.csv"
CASES = {
    "AF1": {
        "af": 1.0, "dns": DNS_ROOT / "af1_nek.npz",
        "bvp": ROOT / "results/beta42_af1_steady/af1_full_bvp_target_s480_fine/s480_try05/fields.npz",
        "trusted_end": 480.0, "color": "#CC6677", "kind": "unforced full BVP"},
    "AF0.1": {
        "af": .1, "dns": DNS_ROOT / "af0p1.npz",
        "bvp": ROOT / "results/radial_sponge_validation/h5_dr05_s520_w40_a10_end560_physicalseed/fields.npz",
        "trusted_end": 520.0, "color": "#D55E00", "kind": "unforced full BVP"},
    "AF0.01": {
        "af": .01, "dns": DNS_ROOT / "af0p01.npz",
        "bvp": ROOT / "results/beta42_af1_steady/lower_af_full_bvp/af0p01_extend_s480/s480_try03/fields.npz",
        "trusted_end": 480.0, "color": "#0072B2", "kind": "unforced full BVP"},
    "AF0.001": {
        "af": .001, "dns": DNS_ROOT / "af0p001.npz",
        "bvp": ROOT / "results/beta42_af1_steady/lower_af_full_bvp/af0p001_phase_release/checkpoint_lambda_0.097656.npz",
        "trusted_end": 480.0, "color": "#009E73",
        "kind": "auxiliary-forcing approximation", "forcing_fraction": .09765625},
}


def bvp_case(path: Path, trusted_end: float):
    config, radii, modes = load_seed(path)
    op = config.operators(radii[0])
    finite = np.isfinite(op.grid.z)
    v1 = modes[:, config.harmonics+1, op.m:2*op.m]
    probe = BarycentricInterpolator(op.grid.z[finite], v1[:, finite], axis=1)(PROBE_Z)
    re = abs(config.ro)*radii
    amplitude = np.sqrt(2)*np.abs(probe)/op.disk_rotation
    physical = re <= trusted_end+1e-9
    x, smooth_a, growth = filtered_curve(re[physical], amplitude[physical])
    valid = (x >= x[0]+6) & (x <= x[-1]-6)
    return dict(re=x, amplitude=smooth_a, growth=growth, growth_valid=valid,
                raw_re=re[physical], raw_amplitude=amplitude[physical])


def pse_case(path: Path, display_end=524.272):
    raw = np.genfromtxt(path, delimiter=",", names=True)
    x, amplitude, growth = filtered_curve(raw["Re"], raw["A1_vtheta_over_disk_speed"])
    keep = x <= display_end+1e-9
    x, amplitude, growth = x[keep], amplitude[keep], growth[keep]
    valid = (x >= x[0]+6) & (x <= x[-1]-6)
    return dict(re=x, amplitude=amplitude, growth=growth, growth_valid=valid,
                raw_re=raw["Re"], raw_amplitude=raw["A1_vtheta_over_disk_speed"])


def first_peak(case):
    peaks = find_peaks(case["amplitude"], prominence=.005)[0]
    peaks = peaks[(case["re"][peaks] >= 330) & (case["amplitude"][peaks] >= .02)]
    i = int(peaks[0]) if len(peaks) else int(np.argmax(case["amplitude"]))
    return {"Re": float(case["re"][i]), "A42": float(case["amplitude"][i])}


def growth_rmse(a, b, lo=330., hi=380.):
    left, right = max(lo, a["re"][0], b["re"][0]), min(hi, a["re"][-1], b["re"][-1])
    x = np.arange(left, right+.001, .25)
    return float(np.sqrt(np.mean((np.interp(x, a["re"], a["growth"])
                                  - np.interp(x, b["re"], b["growth"]))**2)))


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    result, rows = {}, []
    summary = {
        "probe": {
            "BVP": "z_delta=1.1858025795710305; sqrt(2)|v_hat_1|/(Omega_d r)",
            "DNS": "z=1.24 monitor; q=1 stationary coefficient over final 2*pi",
            "growth": "d ln(A42)/d r_delta; dRe=0.25, Gaussian sigma_Re=2",
        },
        "cases": {},
        "integrity_note": ("AF1, AF0.1 and AF0.01 are unforced full-BVP fields. "
                           "AF0.001 retains 9.765625% of the auxiliary PSE forcing. "
                           "The AF1 nonlinear PSE curve uses the more accurate N=64, H=7, "
                           "Delta-r=8 data and stops at its last converged station."),
    }
    for name, info in CASES.items():
        dns, bvp = dns_case(info["dns"]), bvp_case(info["bvp"], info["trusted_end"])
        result[name] = {"dns": dns, "bvp": bvp}
        item = {
            "DNS_source": str(info["dns"]), "BVP_source": str(info["bvp"]),
            "BVP_status": info["kind"], "BVP_trusted_end_Re": info["trusted_end"],
            "DNS_time_window": [dns["time_start"], dns["time_end"]],
            "DNS_first_peak": first_peak(dns), "BVP_first_peak": first_peak(bvp),
            "growth_RMSE_Re330_380": growth_rmse(dns, bvp),
        }
        if "forcing_fraction" in info:
            item["BVP_auxiliary_forcing_fraction"] = info["forcing_fraction"]
        summary["cases"][name] = item
        for method, data in (("DNS", dns), ("BVP", bvp)):
            rows.append(np.column_stack((np.full(len(data["re"]), info["af"]),
                np.full(len(data["re"]), 0 if method == "DNS" else 1),
                data["re"], data["amplitude"], data["growth"])))
    af1_pse = pse_case(AF1_PSE)
    result["AF1"]["pse"] = af1_pse
    summary["cases"]["AF1"].update({
        "PSE_source": str(AF1_PSE),
        "PSE_status": "nonlinear PSE; N=64, H=7, Delta-r=8; stopped before Re=530.28",
        "PSE_source_last_Re": float(af1_pse["raw_re"][-1]),
        "PSE_display_end_Re": float(af1_pse["re"][-1]),
        "PSE_first_peak": first_peak(af1_pse),
    })
    rows.append(np.column_stack((np.full(len(af1_pse["re"]), 1.),
        np.full(len(af1_pse["re"]), 2), af1_pse["re"],
        af1_pse["amplitude"], af1_pse["growth"])))
    np.savetxt(OUT/"bvp_dns_four_curves.csv", np.vstack(rows), delimiter=",",
        header="AF,method_code_0_DNS_1_BVP_2_PSE,Re,A42_vrms,growth_dlnA_drdelta", comments="")
    (OUT/"four_case_summary.json").write_text(json.dumps(summary, indent=2)+"\n")

    plt.rcParams.update({"font.family": "serif",
        "font.serif": ["Times New Roman", "Nimbus Roman", "DejaVu Serif"],
        "mathtext.fontset": "stix", "font.size": 11, "axes.labelsize": 13,
        "axes.linewidth": 1.05, "xtick.direction": "in", "ytick.direction": "in",
        "xtick.top": True, "ytick.right": True, "legend.frameon": False,
        "legend.fontsize": 8.5, "pdf.fonttype": 42, "savefig.dpi": 300})
    fig, ax = plt.subplots(2, 1, figsize=(7.5, 7.1), sharex=True,
        gridspec_kw={"height_ratios": [1, 1.08]}, layout="constrained")
    for name, info in CASES.items():
        dns, bvp = result[name]["dns"], result[name]["bvp"]
        dm = (dns["re"] >= 315) & (dns["re"] <= 525)
        bm = (bvp["re"] >= 315) & (bvp["re"] <= info["trusted_end"])
        bg = bm & bvp["growth_valid"]
        bvp_style = ":" if name == "AF0.001" else "--"
        ax[0].semilogy(dns["re"][dm], dns["amplitude"][dm], color=info["color"], lw=1.9)
        ax[0].semilogy(bvp["re"][bm], bvp["amplitude"][bm], color=info["color"],
                       lw=1.9, ls=bvp_style)
        ax[1].plot(dns["re"][dm], dns["growth"][dm], color=info["color"], lw=1.8)
        ax[1].plot(bvp["re"][bg], bvp["growth"][bg], color=info["color"],
                   lw=1.8, ls=bvp_style)
    pse = result["AF1"]["pse"]
    pm = np.ones(len(pse["re"]), dtype=bool)
    pg = pm & pse["growth_valid"]
    pse_style = (0, (5, 1.5, 1.2, 1.5))
    ax[0].semilogy(pse["re"][pm], pse["amplitude"][pm], color=CASES["AF1"]["color"],
                   lw=1.8, ls=pse_style)
    ax[1].plot(pse["re"][pg], pse["growth"][pg], color=CASES["AF1"]["color"],
               lw=1.7, ls=pse_style)
    color_handles = [Line2D([0], [0], color=v["color"], lw=2, label=k) for k, v in CASES.items()]
    method_handles = [
        Line2D([0], [0], color=".25", lw=1.9, ls="-", label="DNS"),
        Line2D([0], [0], color=".25", lw=1.9, ls="--", label="Unforced full BVP"),
        Line2D([0], [0], color=".25", lw=1.9, ls=":",
               label=r"AF0.001 BVP, $\lambda=0.09765625$"),
        Line2D([0], [0], color=".25", lw=1.8, ls=pse_style, label="AF1 nonlinear PSE"),
    ]
    first_legend = ax[0].legend(handles=color_handles, ncol=4, loc="lower center",
                                bbox_to_anchor=(.5, .005), columnspacing=1.2, handlelength=2.0)
    ax[0].add_artist(first_legend)
    ax[0].legend(handles=method_handles, loc="upper left", fontsize=8.2)
    ax[0].set_ylabel(r"$v_{\mathrm{rms},42}/(\Omega_d r)$")
    ax[0].set_ylim(1e-7, .3)
    ax[0].set_title(r"Stationary $\beta=42$ primary wave: DNS and radial BVP")
    ax[1].axhline(0, color=".65", lw=.8)
    ax[1].set_ylabel(r"$g_{42}=\mathrm{d}\ln v_{\mathrm{rms},42}/\mathrm{d}r_\delta$")
    ax[1].set_xlabel(r"$Re$")
    ax[1].set_ylim(-.11, .115)
    for i, axis in enumerate(ax):
        axis.set_xlim(315, 550)
        axis.grid(color=".9", lw=.55)
        x, align = ((.985, "right") if i == 0 else (.018, "left"))
        axis.text(x, .965, f"({chr(97+i)})", transform=axis.transAxes,
                  ha=align, va="top", fontsize=13)
    for ext in ("png", "pdf"):
        fig.savefig(OUT/f"bvp_dns_four_vrms_growth.{ext}", facecolor="white")
    plt.close(fig)
    (OUT/"README_four_amplitudes.md").write_text(
        "# AF1, AF0.1, AF0.01 and AF0.001: DNS versus radial BVP\n\n"
        "Solid curves are stationary beta=42 amplitudes from the final DNS disk period. "
        "Dashed curves are unforced full radial BVP solutions. The dotted AF0.001 curve "
        "is the farthest converged forcing-release approximation and retains "
        "lambda=0.09765625 of the auxiliary PSE forcing. The dash-dot AF1 curve is the "
        "more accurate nonlinear PSE N=64, H=7, Delta-r=8 result; it stops at its "
        "last converged station near Re=524.27. "
        "BVP curves are truncated at each sponge start. All displayed growth rates use "
        "dRe=0.25 and Gaussian sigma_Re=2.\n")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
