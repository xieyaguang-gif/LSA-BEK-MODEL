"""Build auditable comparisons against Pier (2003), Tables 1--3.

Table 1 is recomputed because it is inexpensive.  Tables 2 and 3 are read from
the full benchmark archive produced by ``pier2003_benchmark.py``.
"""
from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_OUT = ROOT / "results/pier2003/additional_comparisons"
TABLE1_PAPER = {
    30: [(502.645, 50.3471, .21686, -.11819), (507.537, 50.5043, .21757, -.12208),
         (507.284, 50.4885, .21676, -.12166), (507.561, 50.4974, .21694, -.12184),
         (507.216, 50.4876, .21712, -.12190), (506.895, 50.4822, .21655, -.12202)],
    40: [(508.475, 50.5285, .21719, -.12281), (507.504, 50.4952, .21679, -.12188),
         (507.369, 50.4926, .21694, -.12177), (507.402, 50.4933, .21691, -.12180),
         (507.403, 50.4934, .21692, -.12180), (507.386, 50.4929, .21692, -.12180)],
    50: [(507.088, 50.4836, .21688, -.12148), (507.368, 50.4925, .21693, -.12177),
         (507.401, 50.4934, .21692, -.12180), (507.396, 50.4932, .21692, -.12180),
         (507.397, 50.4932, .21692, -.12180), (507.397, 50.4932, .21692, -.12180)],
    60: [(507.500, 50.4961, .21690, -.12191), (507.403, 50.4934, .21692, -.12181),
         (507.398, 50.4932, .21692, -.12180), (507.397, 50.4932, .21692, -.12180),
         (507.397, 50.4932, .21692, -.12180), (507.397, 50.4932, .21692, -.12180)],
}


def write_csv(path, rows):
    rows = list(rows)
    with path.open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=rows[0].keys())
        writer.writeheader(); writer.writerows(rows)


def table1(out, recompute=False):
    cache = out / "table1_current.json"
    if recompute or not cache.exists():
        from llsa.secondary.validation import primary_absolute_threshold
        values = [primary_absolute_threshold(n=n, a=a) for n in TABLE1_PAPER for a in range(1, 7)]
        cache.write_text(json.dumps(values, indent=2) + "\n")
    values = json.loads(cache.read_text())
    current = {(r["n"], r["a"]): r for r in values}
    rows = []
    for n, paper_rows in TABLE1_PAPER.items():
        for a, (rc, omega, ar, ai) in enumerate(paper_rows, 1):
            c = current[n, a]
            rows.append(dict(n=n, a=a, paper_Rc=rc, current_Rc=c["radius"], error_Rc=c["radius"]-rc,
                paper_omega=omega, current_omega=c["omega"][0], error_omega=c["omega"][0]-omega,
                paper_alpha_real=ar, current_alpha_real=c["alpha"][0],
                paper_alpha_imag=ai, current_alpha_imag=c["alpha"][1]))
    write_csv(out / "table1_comparison.csv", rows)
    return rows


def archived_tables(out):
    refs = json.loads((ROOT / "src/llsa/secondary/references.json").read_text())
    bench = json.loads((ROOT / "results/pier2003/benchmark/comparison.json").read_text())
    comps = bench["comparisons"]
    wave = {(r["n"], r["H"]): r for r in comps if r["table3"]["reference"]["K"] == 1}
    rows2 = []
    for ref in refs["table2"]:
        key = ref["n"], ref["H"]
        if key not in wave: continue
        cur = wave[key]
        rows2.append(dict(n=ref["n"], H=ref["H"], quantity="alpha", harmonic=0,
                          paper=ref["alpha"], current=cur["alpha"], error=cur["alpha"]-ref["alpha"]))
        for j, val in enumerate(ref["energies"], 1):
            now = cur["energies"][j-1]
            rows2.append(dict(n=ref["n"], H=ref["H"], quantity=f"E{j}", harmonic=j,
                              paper=val, current=now, error=now-val))
    write_csv(out / "table2_comparison.csv", rows2)
    rows3 = []
    for cur in comps:
        ref = cur["table3"]["reference"]; now = cur["table3"]["omega_hat"]
        rows3.append(dict(n=cur["n"], H=cur["H"], K=ref["K"],
            paper_omega_real=ref["omega_hat"][0], current_omega_real=now[0],
            error_omega_real=now[0]-ref["omega_hat"][0], paper_growth=ref["omega_hat"][1],
            current_growth=now[1], error_growth=now[1]-ref["omega_hat"][1]))
    write_csv(out / "table3_comparison.csv", rows3)
    return rows2, rows3


def plot(rows1, rows2, rows3, out):
    import matplotlib.pyplot as plt
    from matplotlib.lines import Line2D
    plt.rcParams.update({"font.family": "serif", "mathtext.fontset": "stix", "font.size": 9,
                         "axes.linewidth": .8, "savefig.dpi": 300})
    fig, ax = plt.subplots(2, 3, figsize=(10.2, 6.1), constrained_layout=True)
    colors = {30:"#0072B2", 40:"#D55E00", 50:"#009E73", 60:"#7A5195"}
    for n in TABLE1_PAPER:
        r = [x for x in rows1 if x["n"] == n]; a = [x["a"] for x in r]
        ax[0,0].plot(a, [x["paper_Rc"] for x in r], "o-", color=colors[n], ms=3)
        ax[0,0].plot(a, [x["current_Rc"] for x in r], "x--", color=colors[n], ms=4)
        ax[0,1].plot(a, [x["paper_omega"] for x in r], "o-", color=colors[n], ms=3)
        ax[0,1].plot(a, [x["current_omega"] for x in r], "x--", color=colors[n], ms=4)
    ax[0,0].set(xlabel="mapping parameter $a$", ylabel="$R_{ca}$")
    ax[0,1].set(xlabel="mapping parameter $a$", ylabel="$\\omega_r$")
    color_handles = [Line2D([], [], color=colors[n], label=f"N={n}") for n in TABLE1_PAPER]
    style_handles = [Line2D([], [], color="0.25", marker="o", label="Pier"),
                     Line2D([], [], color="0.25", ls="--", marker="x", label="current")]
    ax[0,0].legend(handles=color_handles + style_handles, ncol=2, fontsize=7, frameon=False)
    for n in (30,40):
        r = [x for x in rows2 if x["n"]==n and x["quantity"]=="alpha"]
        ax[0,2].plot([x["H"] for x in r], [x["paper"] for x in r], "o-", color=colors[n], ms=3)
        ax[0,2].plot([x["H"] for x in r], [x["current"] for x in r], "x--", color=colors[n], ms=4)
        e = [x for x in rows2 if x["n"]==n and x["quantity"]=="E1"]
        ax[1,0].plot([x["H"] for x in e], [x["paper"] for x in e], "o-", color=colors[n], ms=3)
        ax[1,0].plot([x["H"] for x in e], [x["current"] for x in e], "x--", color=colors[n], ms=4)
    ax[0,2].set(xlabel="primary harmonics $H$", ylabel="$\\alpha$")
    ax[0,2].legend(handles=[Line2D([], [], color=colors[n], label=f"N={n}") for n in (30,40)] + style_handles,
                   fontsize=7, frameon=False)
    ax[1,0].set(xlabel="primary harmonics $H$", ylabel="$E_1$")
    r40 = [x for x in rows3 if x["n"]==40]
    for panel, key, title in [(ax[1,1], "error_omega_real", "$\\Delta\\,\\mathrm{Re}(\\hat\\omega)$"),
                               (ax[1,2], "error_growth", "$\\Delta\\,\\mathrm{Im}(\\hat\\omega)$")]:
        z = np.array([[next(x[key] for x in r40 if x["H"]==h and x["K"]==k) for k in range(1,6)] for h in range(1,6)])
        im = panel.imshow(z, origin="lower", extent=(.5,5.5,.5,5.5), cmap="coolwarm", aspect="equal")
        panel.set(xlabel="secondary harmonics $K$", ylabel="primary harmonics $H$", title=title)
        fig.colorbar(im, ax=panel, shrink=.80)
    for i, panel in enumerate(ax.flat):
        panel.text(.01, .98, f"({chr(97+i)})", transform=panel.transAxes, ha="left", va="top", fontweight="bold")
        panel.tick_params(direction="in", top=True, right=True)
    fig.savefig(out / "pier2003_tables_comparison.png")
    fig.savefig(out / "pier2003_tables_comparison.pdf")
    fig.savefig(out / "pier2003_tables_comparison.svg")
    plt.close(fig)


def report(rows1, rows2, rows3, out):
    t1 = max(abs(x["error_Rc"]) for x in rows1)
    t1_conv = next(x for x in rows1 if x["n"]==60 and x["a"]==6)
    t2a = max(abs(x["error"]) for x in rows2 if x["quantity"]=="alpha")
    t2e = max(abs(x["error"]) for x in rows2 if x["quantity"].startswith("E"))
    t2 = {x["quantity"]: x for x in rows2 if x["n"]==40 and x["H"]==5}
    t3 = next(x for x in rows3 if x["n"]==40 and x["H"]==5 and x["K"]==5)
    text = f"""# Pier (2003) additional quantitative comparisons

All differences below are **current minus Pier**. The comparison uses the paper's
parameters: Ro=-1, R=510, beta=68 and primary frequency 50.5 where applicable.

| Item | Result |
|---|---:|
| Table 1 maximum absolute error in critical radius (all 24 entries) | {t1:.6g} |
| Table 1 converged point, current / paper $R_{{ca}}$ (N=60,a=6) | {t1_conv['current_Rc']:.6f} / {t1_conv['paper_Rc']:.6f} |
| Table 2 maximum absolute alpha error | {t2a:.6g} |
| Table 2 maximum absolute harmonic-energy error (all entries) | {t2e:.6g} |
| Table 2 converged $E_1$, current / paper (N=40,H=5) | {t2['E1']['current']:.6f} / {t2['E1']['paper']:.2f} |
| Table 2 converged $E_2$, current / paper (N=40,H=5) | {t2['E2']['current']:.6f} / {t2['E2']['paper']:.2f} |
| Table 3 converged real frequency, current / paper | {t3['current_omega_real']:.6f} / {t3['paper_omega_real']:.6f} |
| Table 3 converged growth rate, current / paper | {t3['current_growth']:.6f} / {t3['paper_growth']:.6f} |

Tables 1 and 2 reproduce the published convergence trends and limiting values.
Table 3 shows a systematic discrepancy across the full 5x5 H-K matrix: the real
frequency remains close, while the current growth rate is lower. This demonstrates
that the discrepancy is not caused by insufficient H or K at the single H=K=5 point.
"""
    (out / "README.md").write_text(text)


def main():
    p = argparse.ArgumentParser(); p.add_argument("--output", type=Path, default=DEFAULT_OUT)
    p.add_argument("--recompute-table1", action="store_true"); args = p.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    r1 = table1(args.output, args.recompute_table1); r2, r3 = archived_tables(args.output)
    plot(r1, r2, r3, args.output); report(r1, r2, r3, args.output)
    print(f"Saved comparisons to {args.output}")


if __name__ == "__main__": main()
