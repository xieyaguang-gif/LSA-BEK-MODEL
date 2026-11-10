"""Compare last-period DNS beta=42 growth with local linear stability."""
from __future__ import annotations

import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from scipy.ndimage import gaussian_filter1d
from scipy.signal import find_peaks

ROOT = Path(__file__).resolve().parents[1]
DNS_ROOT = ROOT.parent / "semtex-gpu_work/cases/00_reference/af1/analysis/secondary_amplitude_sweep_20260914"
LINEAR = ROOT / "results/beta42_nonlinear_spatial/inputs/llsa_beta42_stationary.csv"
OUT = ROOT / "results/beta42_dns_last_period_linear_growth"
RE_FACTOR_DNS = 0.718373556
RO_ABS = 0.751
PERIOD = 2 * np.pi
SMOOTH_SIGMA_RE = 2.0
SAMPLE_DRE = 0.25
CASES = {
    "AF10": (10.0, DNS_ROOT / "af10.npz"),
    "AF1": (1.0, DNS_ROOT / "af1_nek.npz"),
    "AF0.1": (0.1, DNS_ROOT / "af0p1.npz"),
    "AF0.01": (0.01, DNS_ROOT / "af0p01.npz"),
    "AF0.001": (0.001, DNS_ROOT / "af0p001.npz"),
}


def filtered_curve(re, amplitude):
    """Apply the archived beta42 comparison filter and differentiate ln(A)."""
    order = np.argsort(re)
    re, amplitude = np.asarray(re)[order], np.asarray(amplitude)[order]
    x = np.arange(re[0], re[-1] + 1e-10, SAMPLE_DRE)
    raw_log = np.interp(x, re, np.log(np.maximum(amplitude, 1e-30)))
    loga = gaussian_filter1d(raw_log, SMOOTH_SIGMA_RE / SAMPLE_DRE,
                             mode="reflect", truncate=4)
    return x, np.exp(loga), RO_ABS * np.gradient(loga, x, edge_order=2)


def dns_case(path):
    with np.load(path) as data:
        r, time = data["r"].copy(), data["time"].copy()
        q1 = data["low_modes"][:, 0, :].copy()
    end = float(time[-1])
    mask = time >= end - PERIOD - 1e-10
    # Average the complex q=1 coefficient first to isolate the stationary wave.
    amplitude = np.sqrt(2) * np.abs(np.mean(q1[mask], axis=0))
    x, smooth_a, growth = filtered_curve(RE_FACTOR_DNS * r, amplitude)
    return dict(re_raw=RE_FACTOR_DNS * r, amplitude_raw=amplitude, re=x,
                amplitude=smooth_a, growth=growth,
                time_start=float(time[mask][0]), time_end=end,
                frames=int(mask.sum()), dt=float(np.median(np.diff(time))))


def interval_metrics(case, linear_re, linear_growth, lo, hi):
    mask = (case["re"] >= lo) & (case["re"] <= hi)
    x, g = case["re"][mask], case["growth"][mask]
    gl = np.interp(x, linear_re, linear_growth)
    return {"interval_Re": [lo, hi], "mean_DNS_growth": float(np.mean(g)),
            "mean_linear_growth": float(np.mean(gl)),
            "growth_bias_DNS_minus_linear": float(np.mean(g-gl)),
            "growth_RMSE": float(np.sqrt(np.mean((g-gl)**2)))}


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    cases = {name: dns_case(path) for name, (_, path) in CASES.items()}
    linear = np.genfromtxt(LINEAR, delimiter=",", names=True)
    lr, lg = linear["Re"], linear["growth_minus_alpha_i_paper"]
    summary = {
        "definition": "g=d ln[sqrt(2)|v_beta42/(Omega_d r)|]/d r_delta",
        "DNS_Re_mapping": "Re=0.718373556*r (case monitoring convention)",
        "derivative_conversion": "d/d r_delta = |Ro| d/dRe with |Ro|=0.751",
        "time_window": "last complete disk period: t in [t_end-2*pi,t_end]",
        "filter": "interpolate ln(A) to dRe=0.25; Gaussian sigma_Re=2; differentiate",
        "cases": {}}
    rows = []
    for name, d in cases.items():
        peaks = find_peaks(d["amplitude"], prominence=.005)[0]
        peaks = peaks[(d["re"][peaks] >= 320) & (d["amplitude"][peaks] >= .02)]
        peak_i = int(peaks[0]) if len(peaks) else int(np.argmax(d["amplitude"]))
        af, source = CASES[name]
        summary["cases"][name] = {
            "source": str(source),
            "time_window": [d["time_start"], d["time_end"]],
            "window_duration": d["time_end"]-d["time_start"], "frames": d["frames"],
            "smoothed_primary_peak": {"Re": float(d["re"][peak_i]),
                                        "A42": float(d["amplitude"][peak_i])},
            "comparison_300_360": interval_metrics(d, lr, lg, 300, 360),
            "comparison_330_380": interval_metrics(d, lr, lg, 330, 380)}
        rows.append(np.column_stack((np.full(len(d["re"]), af),
                                     d["re"], d["amplitude"], d["growth"],
                                     np.interp(d["re"], lr, lg))))
    np.savetxt(OUT / "dns_last_period_growth.csv", np.vstack(rows), delimiter=",",
               header="AF,Re,A42_smoothed,DNS_growth_dlnA_drdelta,local_linear_growth_minus_alpha_i",
               comments="")
    (OUT / "summary.json").write_text(json.dumps(summary, indent=2)+"\n")

    plt.rcParams.update({"font.family": "serif",
        "font.serif": ["Times New Roman", "Nimbus Roman", "DejaVu Serif"],
        "mathtext.fontset": "stix", "font.size": 11, "axes.labelsize": 13,
        "axes.linewidth": 1.05, "xtick.direction": "in", "ytick.direction": "in",
        "xtick.top": True, "ytick.right": True, "legend.frameon": False,
        "legend.fontsize": 9.5, "pdf.fonttype": 42, "savefig.dpi": 300})
    colors = {"AF10": "#7A5195", "AF1": "#CC6677", "AF0.1": "#D55E00",
              "AF0.01": "#0072B2", "AF0.001": "#009E73"}
    styles = {"AF10": (0, (3, 1, 1, 1)), "AF1": "--", "AF0.1": "-",
              "AF0.01": "-", "AF0.001": "-."}
    fig, axes = plt.subplots(2, 1, figsize=(7.1, 6.8), sharex=True,
        gridspec_kw={"height_ratios": [1, 1.15]}, layout="constrained")
    for name, d in cases.items():
        mask = (d["re"] >= 295) & (d["re"] <= 465)
        axes[0].semilogy(d["re"][mask], d["amplitude"][mask], color=colors[name],
                        ls=styles[name], lw=1.85,
                        label=fr"{name} DNS, last $2\pi$")
        axes[1].plot(d["re"][mask], d["growth"][mask], color=colors[name],
                     ls=styles[name], lw=1.65,
                     label=fr"{name} DNS, last $2\pi$")
    mask = (lr >= 295) & (lr <= 465)
    axes[1].plot(lr[mask], lg[mask], color="black", lw=1.7, ls="--",
                 label=r"Local linear stability, $\beta=42$")
    axes[0].set_ylabel(r"$A_{42}=\sqrt{2}|\widehat v_{42}/(\Omega_d r)|$")
    axes[0].set_ylim(1e-7, 3e-1)
    axes[0].legend(loc="lower right", ncol=2)
    axes[1].axhline(0, color="0.65", lw=0.8)
    axes[1].set_ylabel(r"$g_{42}=\mathrm{d}\ln A_{42}/\mathrm{d}r_\delta$")
    axes[1].set_xlabel(r"$Re$")
    axes[1].set_ylim(-0.10, 0.115)
    axes[1].legend(loc="lower left", ncol=2, fontsize=8.7)
    for i, ax in enumerate(axes):
        ax.set_xlim(295, 465)
        ax.text(.018, .965, f"({chr(97+i)})", transform=ax.transAxes,
                ha="left", va="top", fontsize=13)
        ax.grid(color="0.9", lw=.55)
    axes[0].set_title(r"Stationary $\beta=42$ mode from the final disk period")
    for ext in ("png", "pdf"):
        fig.savefig(OUT / f"dns_last_period_vs_linear_growth.{ext}", facecolor="white")
    plt.close(fig)

    readme = """# DNS last-period growth versus local linear stability

The stationary azimuthal Fourier mode `q=1` corresponds to `beta=42` in the
42-fold DNS sector. For each case, its complex coefficient is averaged over
the final complete disk period (`2*pi`) before the RMS amplitude is formed.
The radial logarithmic derivative uses the same `ln(A)` Gaussian filter as the
earlier PSE/DNS comparison (`dRe=0.25`, `sigma_Re=2`).

The DNS monitoring convention is `Re=0.718373556*r`. Growth is reported per
boundary-layer coordinate, `d ln(A)/d r_delta = 0.751 d ln(A)/dRe`, for direct
comparison with the local-stability value `-alpha_i`.
"""
    (OUT / "README.md").write_text(readme)
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
