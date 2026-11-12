"""Draw branch and family comparisons from frozen evidence (no flow solves).

The two physical reference flows are retained as independent samples in every
cross-Rossby comparison; connecting lines follow only the nine-point scan.
The two publication plates are 7 x 5.9 inch native Matplotlib layouts, using
10 pt Times New Roman/STIX text and 8 pt math scripts. The release contains the numerical inputs used in the paper.
"""
from pathlib import Path
import hashlib
import json
import os

os.environ.setdefault("MPLCONFIGDIR", "/tmp/prf_bek_mpl")
for _key in ("OPENBLAS_NUM_THREADS", "OMP_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ[_key] = "1"
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.font_manager as fm
from matplotlib import _mathtext
from matplotlib.lines import Line2D
from matplotlib.ticker import AutoMinorLocator, MultipleLocator, FormatStrFormatter

HERE = Path(__file__).resolve().parent
DATA_PATH = HERE / "data/evidence.json"
DIAGNOSTIC_PATH = HERE / "data/revision_diagnostics/common_diagnostics.json"
BENCHMARK_PATHS = [HERE / "data/benchmark_ro_m1.json", HERE / "data/benchmark_ro0751.json"]
D = json.loads(DATA_PATH.read_text())
DIAGNOSTICS = json.loads(DIAGNOSTIC_PATH.read_text())
BENCHMARKS = [json.loads(p.read_text()) for p in BENCHMARK_PATHS]
CORE_RO = [-1.0, -.751]
FIGURES = HERE / "figures"
FIGURES.mkdir(exist_ok=True)
BLUE, RED, GRAY = "#2166ac", "#b2182b", "#626262"
COLORS = {"local": BLUE, "linear": RED, "envelope": GRAY,
          "harmonic": BLUE, "frequency": RED, "fold": RED}
CASE_COLORS = {-.9: GRAY, -.5: RED, -.2: BLUE}
CASE_MARKERS = {-.9: "o", -.5: "s", -.2: "^"}
SIZE = 10.0
for FONT in ("Times New Roman", "Nimbus Roman", "STIXGeneral"):
    try:
        fm.findfont(fm.FontProperties(family=FONT), fallback_to_default=False)
        break
    except ValueError:
        continue
else:
    raise RuntimeError("Install Times New Roman, Nimbus Roman, or STIXGeneral.")
_mathtext.SHRINK_FACTOR = .8
_mathtext.GROW_FACTOR = 1.0 / _mathtext.SHRINK_FACTOR
plt.rcParams.update({
    "font.family": "serif", "font.serif": [FONT], "mathtext.fontset": "stix",
    "font.size": SIZE, "axes.labelsize": SIZE, "xtick.labelsize": SIZE,
    "ytick.labelsize": SIZE, "legend.fontsize": SIZE,
    "axes.linewidth": .85, "axes.edgecolor": "black", "text.color": "black",
    "axes.labelcolor": "black", "xtick.color": "black", "ytick.color": "black",
    "lines.linewidth": 1.35, "lines.markersize": 5.2,
    "lines.markeredgewidth": .9, "pdf.fonttype": 42,
    "savefig.dpi": 360, "figure.facecolor": "white", "axes.facecolor": "white",
    "axes.unicode_minus": True,
})
rows = sorted(D["rows"], key=lambda r:r["Ro"])
paired = sorted(DIAGNOSTICS["rows"] + [b["common_diagnostic"] for b in BENCHMARKS],
                key=lambda r:r["Ro"])
assert len(paired) == 11 and sum(r["N"] == 40 for r in paired) == 10
for record in paired:
    for state in ("linear", "nonlinear"):
        assert record[state]["unique_downcrossing"] and record[state]["no_upcrossing"]
local_by_ro = {r["Ro"]:r["R_ca"] for r in rows}
METADATA = {
    "diagnostic_definition": "common q_v RMS at z=1.3",
    "diagnostic_cases": [{"Ro":r["Ro"], "N":r["N"], "H":r["H"]} for r in paired],
    "figures": {}, "cross_Ro_series": [],
    "line_rule": "Only the nine regular scan cases are connected; Ro=-1 and -0.751 are independent symbols.",
    "font": FONT, "main_font_pt": 10, "minimum_math_script_pt": 8,
}


def setup(nrows=1):
    size = (7.0, 5.9) if nrows == 2 else (7.0, 3.15)
    fig, axes = plt.subplots(nrows, 2, figsize=size, squeeze=False)
    if nrows == 2:
        fig.subplots_adjust(left=.095, right=.985, bottom=.12, top=.945,
                            wspace=.40, hspace=.43)
    else:
        fig.subplots_adjust(left=.095, right=.985, bottom=.225, top=.91, wspace=.40)
    for index, ax in enumerate(axes.flat):
        ax.tick_params(which="major", direction="in", top=True, right=True,
                       length=4.0, width=.8, pad=3.0)
        ax.tick_params(which="minor", direction="in", top=True, right=True,
                       length=2.0, width=.65)
        ax.xaxis.set_minor_locator(AutoMinorLocator(2))
        ax.yaxis.set_minor_locator(AutoMinorLocator(2))
        ax.xaxis.labelpad = 5.5
        ax.yaxis.labelpad = 6.0
        ax.text(0, 1.025, "($" + chr(97 + index) + "$)", transform=ax.transAxes,
                ha="left", va="bottom", fontsize=SIZE)
    return fig, axes.ravel()


def rossby_axis(ax):
    ax.set_xlabel(r"$\mathrm{Ro}$")
    ax.set_xlim(-1.04, -.06)
    ax.set_xticks([-1., -.751, -.5, -.3, -.1])
    ax.set_xticklabels(["−1", "−0.751", "−0.5", "−0.3", "−0.1"])
    ax.xaxis.set_minor_locator(MultipleLocator(.1))


def legend(ax, loc, handles=None):
    return ax.legend(handles=handles, loc=loc, frameon=False, handlelength=1.8,
                     handletextpad=.55, labelspacing=.45, borderaxespad=.65)


def core_handle():
    return Line2D([], [], marker="o", linestyle="none", color="black",
                  markerfacecolor="none", markersize=7.2, markeredgewidth=.7,
                  label="Reference flows")


def series(ax, records, value, color, marker, linestyle, label, name, filled=False):
    """One scientific quantity: scan line, independent core symbols, core rings."""
    scan = [r for r in records if r["Ro"] not in CORE_RO]
    core = [r for r in records if r["Ro"] in CORE_RO]
    assert len(scan) == 9 and len(core) == 2
    ax.plot([r["Ro"] for r in scan], [value(r) for r in scan], color=color,
            marker=marker, linestyle=linestyle,
            markerfacecolor=color if filled else "white", label=label, zorder=3)
    ax.plot([r["Ro"] for r in core], [value(r) for r in core], color=color,
            marker=marker, linestyle="none", markerfacecolor=color if filled else "white", zorder=4)
    ax.plot([r["Ro"] for r in core], [value(r) for r in core], color="black",
            marker="o", markerfacecolor="none", markersize=7.2,
            markeredgewidth=.7, linestyle="none", zorder=5)
    METADATA["cross_Ro_series"].append({"name":name, "connected_Ro":[r["Ro"] for r in scan],
                                       "independent_Ro":[r["Ro"] for r in core],
                                       "sample_count":len(records)})


def add_core_legend(ax, loc):
    handles, _ = ax.get_legend_handles_labels()
    legend(ax, loc, handles + [core_handle()])


def draw_thresholds(ax):
    series(ax, rows, lambda r:r["Re_ca"], BLUE, "o", "-", "Local onset", "local_onset")
    series(ax, paired, lambda r:abs(r["Ro"])*r["linear"]["front"], RED, "s", "--",
           "Linear envelope", "linear_RMS")
    series(ax, paired, lambda r:abs(r["Ro"])*r["nonlinear"]["front"], GRAY, "^", "-.",
           "Nonlinear envelope", "nonlinear_RMS", filled=True)
    rossby_axis(ax)
    ax.set_ylabel(r"$\mathrm{Re}$")
    ax.set_ylim(195, 620)
    ax.yaxis.set_major_locator(MultipleLocator(100))
    add_core_legend(ax, "upper right")


def draw_local_gap(ax):
    series(ax, paired, lambda r:100*(r["linear"]["front"]/local_by_ro[r["Ro"]]-1),
           BLUE, "o", "-", "Survey", "linear_local_gap")
    rossby_axis(ax)
    ax.set_ylabel(r"$100(R_{f,\mathrm{L}}/R_{\mathrm{ca}}-1)$")
    ax.set_ylim(4.5, 19.5)
    ax.yaxis.set_major_locator(MultipleLocator(3))
    add_core_legend(ax, "lower left")


def draw_front_gap(ax):
    series(ax, paired, lambda r:r["diagnostic_position_difference"], GRAY, "o", "-",
           r"$R_{f,\mathrm{L}}-R_f$", "RMS_position_difference")
    series(ax, paired, lambda r:r["linear"]["front"]-r["second_harmonic_front"],
           BLUE, "s", "--", r"$R_{f,\mathrm{L}}-R_{f,2}$", "second_harmonic_position_difference")
    ax.axhline(0, color="black", ls=(0, (2, 2)), lw=.75, zorder=0)
    rossby_axis(ax)
    ax.set_ylabel("Radial position difference")
    ax.set_ylim(-12, 23)
    ax.yaxis.set_major_locator(MultipleLocator(10))
    add_core_legend(ax, "lower left")


def draw_frequency(ax):
    series(ax, paired, lambda r:100*(r["omega_nonlinear"]/r["omega_linear"]-1),
           RED, "o", "-", "Survey", "frequency_difference")
    ax.axhline(0, color="black", ls=(0, (2, 2)), lw=.75, zorder=0)
    rossby_axis(ax)
    ax.set_ylabel(r"$100(\omega_d^{\mathrm{NL}}/\omega_d^{\mathrm{L}}-1)$")
    ax.set_ylim(-.82, .045)
    ax.yaxis.set_major_locator(MultipleLocator(.2))
    ax.yaxis.set_major_formatter(FormatStrFormatter("%.1f"))
    add_core_legend(ax, "upper left")


def draw_representative_branches(ax):
    ax.axvspan(-9.5, 0, color="#f2f2f2", zorder=0)
    case_handles = []
    count = 0
    for branch in D["branches"]:
        record = next(r for r in rows if r["Ro"] == branch["Ro"])
        color, marker = CASE_COLORS[record["Ro"]], CASE_MARKERS[record["Ro"]]
        for h in sorted({p["H"] for p in branch["points"]}):
            points = sorted([p for p in branch["points"] if p["H"] == h], key=lambda p:p["A1"])
            assert len({(p["n"],p["intervals"]) for p in points}) == 1
            count += len(points)
            ax.plot([p["Rout"]-record["Rout_c_N32"] for p in points], [p["A1"] for p in points],
                    marker=marker, color=color, linestyle="-", markersize=3.4,
                    linewidth=1.05, markeredgewidth=.7, markerfacecolor="white" if h == 4 else color)
        case_handles.append(Line2D([], [], color=color, marker=marker, markersize=4.7,
                            markerfacecolor="white", linewidth=1.05, label=rf"$\mathrm{{Ro}}={record['Ro']:g}$"))
    assert count == 123
    ax.axvline(0, color="black", lw=.75, ls=(0, (2, 2)), zorder=1)
    ax.set(xlim=(-9.5, 17.), ylim=(0, .315), xlabel=r"$R_{\mathrm{out}}-R_{\mathrm{out},c}$", ylabel=r"$A_1$")
    ax.xaxis.set_major_locator(MultipleLocator(5))
    ax.yaxis.set_major_locator(MultipleLocator(.1))
    ax.yaxis.set_major_formatter(FormatStrFormatter("%.1f"))
    legend(ax, "upper right", case_handles)
    return count


def draw_folds(ax):
    folds = [r for r in rows if "Rout_f" in r and "Rout_c_N32" in r]
    widths = [b["neutral_N32"]["rout"]-b["fold"]["quadratic_Rout"] for b in BENCHMARKS]
    ax.plot([r["Ro"] for r in folds], [r["Rout_c_N32"]-r["Rout_f"] for r in folds],
            "s-", color=RED, markerfacecolor="white", label="Survey")
    ax.plot(CORE_RO, widths, "D", color="black", markerfacecolor="white", markersize=5.0,
            linestyle="none", label="Reference flows", zorder=5)
    assert all(r["Ro"] not in CORE_RO for r in folds) and len(folds) == 9
    rossby_axis(ax)
    ax.set_ylabel(r"$R_{\mathrm{out},c}-R_{\mathrm{out},\mathrm{fold}}$")
    ax.set_ylim(0, 11)
    ax.yaxis.set_major_locator(MultipleLocator(2))
    legend(ax, "lower left")
    return {"scan_fold_intervals":len(folds), "independent_core_fold_intervals":2,
            "core_fold_widths":dict(zip(map(str,CORE_RO),widths))}


def draw_core_branch(ax, benchmark):
    if benchmark["Ro"] == -1.0:
        points = benchmark["branch"]["points"] + benchmark["branch"]["fold_points"] + benchmark["upper_H5_extension"]
        limits = (-4.2, 5.8)
    else:
        points, limits = benchmark["points"], (-11.2,.7)
    assert len({p["id"] for p in points}) == len(points)
    n, intervals = benchmark["neutral_N32"].get("N",32), benchmark["neutral_N32"]["intervals"]
    assert all(p["n"] == n and p["intervals"] == intervals for p in points)
    critical = benchmark["neutral_N32"]["rout"]
    ax.axvspan(limits[0],0,color="#f2f2f2",zorder=0)
    for h,color,marker,linestyle in [(3,BLUE,"o","-"),(4,RED,"s","--"),(5,GRAY,"^","-.")]:
        group = sorted([p for p in points if p["H"] == h],key=lambda p:p["A1"])
        if group:
            ax.plot([p["Rout"]-critical for p in group],[p["A1"] for p in group],color=color,
                    marker=marker,markersize=4.8 if h==5 else 3.7,markeredgewidth=.7,
                    markerfacecolor="white" if h<5 else color,linewidth=1.1,
                    linestyle=linestyle if len(group)>1 else "none",label=rf"$H={h}$",
                    zorder=4 if h==5 else 2)
    ax.axvline(0,color="black",lw=.75,ls=(0,(2,2)),zorder=1)
    ax.set(xlim=limits,ylim=(0,.30),xlabel=r"$R_{\mathrm{out}}-R_{\mathrm{out},c}$",ylabel=r"$A_1$")
    ax.text(.05,.94,rf"$\mathrm{{Ro}}={benchmark['Ro']:g}$",transform=ax.transAxes,ha="left",va="top")
    ax.xaxis.set_major_locator(MultipleLocator(2))
    ax.yaxis.set_major_locator(MultipleLocator(.1))
    ax.yaxis.set_major_formatter(FormatStrFormatter("%.1f"))
    legend(ax,"lower right" if benchmark["Ro"] == -1. else "lower left")
    return {"Ro":benchmark["Ro"], "points":len(points), "N":n, "radial_intervals":intervals,
            "harmonic_counts":{str(h):sum(p["H"]==h for p in points) for h in sorted({p["H"] for p in points})},
            "neutral_reference":benchmark["neutral_N32"]}


def finish(fig,name):
    for ax in fig.axes:
        for line in ax.lines:
            if line.get_transform() != ax.transData:
                continue
            for values,limits in ((line.get_xdata(),ax.get_xlim()),(line.get_ydata(),ax.get_ylim())):
                values = np.asarray(values,dtype=float)
                assert np.all(np.isfinite(values))
                assert values.min() >= limits[0] and values.max() <= limits[1], (name,values,limits)
    METADATA["figures"].setdefault(name,{})["size_inches"] = list(fig.get_size_inches())
    fig.savefig(FIGURES/f"{name}.pdf")
    fig.savefig(FIGURES/f"{name}.png")
    plt.close(fig)


# Native four-panel publication plates.
fig,axes = setup(2)
counts = [draw_core_branch(ax,b) for ax,b in zip(axes[:2],BENCHMARKS)]
count = draw_representative_branches(axes[2])
folds = draw_folds(axes[3])
METADATA["figures"]["branches_all"] = {"core_branches":counts,"original_branch_samples":count,**folds}
finish(fig,"branches_all")
fig,axes = setup(2)
for drawer,ax in zip((draw_thresholds,draw_local_gap,draw_front_gap,draw_frequency),axes):
    drawer(ax)
METADATA["figures"]["family_comparison"] = {"panels":["local_and_envelope_stations","linear_local_gap",
                                                      "envelope_and_second_harmonic_differences","frequency_difference"],
                                            "samples_per_series":11,"connected_scan_samples":9,"independent_reference_samples":2}
finish(fig,"family_comparison")

METADATA["source_sha256"]={str(p.relative_to(HERE)):hashlib.sha256(p.read_bytes()).hexdigest()
                            for p in [DATA_PATH,DIAGNOSTIC_PATH,*BENCHMARK_PATHS]}
METADATA["plot_script_sha256"]=hashlib.sha256(Path(__file__).read_bytes()).hexdigest()
for series_record in METADATA["cross_Ro_series"]:
    assert not set(CORE_RO).intersection(series_record["connected_Ro"])
(FIGURES/"branch_family_metadata.json").write_text(json.dumps(METADATA,indent=2)+"\n")
print("Generated Figures 1 and 3.")
print("Core branch points:",[c["points"] for c in counts],"representative branch points:",count)
print("All cross-Ro series: 9 connected scan points, 2 independent physical reference points.")
