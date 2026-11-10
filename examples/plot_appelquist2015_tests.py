#!/usr/bin/env python3
"""Manifest-only diagnostics for the Appelquist mechanism test campaign.

No solver imports or checkpoint discovery. Rendering requires --render and an
explicit root-approved manifest. Failed Newton iterates are never plotted.
Linear candidates are unconnected points, including when resolution is pending.
"""
from __future__ import annotations

import argparse
from dataclasses import dataclass
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys

for _key in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
    os.environ[_key] = "1"
os.environ.setdefault("MPLCONFIGDIR", "/tmp/appelquist2015_figure_mplconfig")
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "results/appelquist2015_tests_20261029"
COLORS = ("#0072B2", "#D55E00", "#009E73", "#CC79A7", "#555555")
MARKERS = ("o", "s", "^", "D", "v")


def read_json(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def digest(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024*1024), b""):
            h.update(block)
    return h.hexdigest()


def frozen(source):
    path = Path(source["path"]).expanduser()
    path = (path if path.is_absolute() else ROOT/path).resolve(strict=True)
    if digest(path) != source["sha256"]:
        raise ValueError(f"Source is no longer immutable: {path}")
    return path


def same(a, b, label):
    if not np.allclose(a, b, rtol=1e-10, atol=1e-12):
        raise ValueError(f"Inconsistent {label}")


@dataclass
class Nonlinear:
    spec: dict
    r: np.ndarray
    q: np.ndarray
    config: dict
    meta: dict
    mass: np.ndarray
    sponge: np.ndarray
    omega: float

    @property
    def h(self):
        return int(self.config["harmonics"])

    @property
    def m(self):
        return int(self.config["n"])+1

    def paper_energy(self):
        # E_paper,j = 2*pi*E_j/R^2. Use the original complete grid mass.
        v = self.q[:, self.h+1:, :3*self.m].reshape(len(self.r), self.h, 3, self.m)
        value = 2*np.pi*np.einsum("rhvi,ij,rhvj->rh", v.conj(), self.mass, v).real
        if np.any(~np.isfinite(value)) or np.any(value < 0):
            raise ValueError("Invalid harmonic energy")
        return value


def load_nonlinear(spec, residual_limit):
    if spec.get("accepted") is not True:
        raise ValueError("A nonlinear case requires explicit root acceptance")
    paths = {name: frozen(source) for name, source in spec["sources"].items()}
    status = read_json(paths["status"])
    if spec["source_format"] == "campaign":
        completion = read_json(paths["completion"])
        read_json(paths["configuration"])
        if status.get("classification") != "accepted_nonzero" or completion.get("accepted") is not True:
            raise ValueError("Campaign Newton/completion records do not accept a nonzero root")
        same(status["physical_residual_inf"], completion["validation"]["residual_inf"], "independent residual")
    elif spec["source_format"] == "legacy":
        if status.get("status") != "converged" or float(status.get("growth", np.inf)) != 0:
            raise ValueError("Legacy source is not a converged unforced root")
    else:
        raise ValueError("source_format must be legacy or campaign")
    residual = float(status["physical_residual_inf"])
    if not np.isfinite(residual) or not 0 <= residual <= residual_limit:
        raise ValueError("Physical residual fails the root-approved limit")
    with np.load(paths["fields"], allow_pickle=False) as data:
        r, q = data["radii"].copy(), data["modes"].copy()
        config = json.loads(str(data["config_json"].item()))
        meta = json.loads(str(data["metadata_json"].item()))
        omega = float(data["omega"])
        growth = float(data["auxiliary_growth"])
    if (meta.get("auxiliary") is not False or growth != 0
            or meta.get("amplitude_constraint") is not None
            or meta.get("homogeneous_inlet") is not True
            or "interior_forcing" not in meta or meta["interior_forcing"] is not None):
        raise ValueError("Nonlinear fields do not certify the unforced equations")
    same(omega, status["omega"], "omega")
    same(config["ro"], -1, "Ro"); same(config["beta"], 68, "beta")
    h, m = int(config["harmonics"]), int(config["n"])+1
    if (r.ndim != 1 or len(r) < 3 or np.any(~np.isfinite(r)) or np.any(np.diff(r) <= 0)
            or q.shape != (len(r), 2*h+1, 4*m) or np.any(~np.isfinite(q))):
        raise ValueError("Invalid nonlinear source array contract")
    if not np.any(np.abs(q[:, h+1:, :3*m]) > 0):
        raise ValueError("A nonzero harmonic root is required")
    same(q[:, :h], q[:, h+1:][:, ::-1].conj(), "conjugate symmetry")
    with np.load(paths["grid"], allow_pickle=False) as grid:
        mass = grid["mass"].copy()
        same(grid["n"], config["n"], "grid N")
        if mass.shape != (m, m) or grid["z"].shape != (m,):
            raise ValueError("Archived grid dimensions differ from fields")
    table = np.genfromtxt(paths["profile"], delimiter=",", names=True)
    columns = spec["profile_columns"]
    same(table[columns["radius"]], r, "original sponge sampling radii")
    profile = np.asarray(table[columns["value"]])
    if np.any(~np.isfinite(profile)) or np.any(profile < 0):
        raise ValueError("Invalid archived sponge profile")
    expected = spec["expected"]
    actual = dict(n=config["n"], h=h, dr=float(r[1]-r[0]), rmin=float(r[0]), rmax=float(r[-1]),
                  sponge_start=meta["sponge"]["start"], sponge_rise=meta["sponge"]["width"],
                  strength=meta["sponge"]["strength"], omega=omega)
    for key, value in actual.items():
        same(value, expected[key], key)
    same(np.diff(r), actual["dr"], "uniform radial grid")
    # Width is the rise distance. A plateau may continue to an independent outlet.
    if not r[0] < actual["sponge_start"] < r[-1] or actual["sponge_rise"] <= 0:
        raise ValueError("Invalid sponge onset/rise")
    if not spec.get("limitations"):
        raise ValueError("Each candidate needs explicit interpretation limits")
    return Nonlinear(spec, r, q, config, meta, mass, profile, omega)


def validate_comparison(group, cases):
    selected = [cases[key] for key in group["case_ids"]]
    if len(selected) < 2:
        raise ValueError("A controlled nonlinear comparison needs at least two accepted cases")
    first = selected[0]
    kind = group["kind"]
    if kind not in ("fixed_profile_outlet", "fixed_outlet_sponge", "refinement"):
        raise ValueError("Unknown controlled comparison kind")
    for other in selected[1:]:
        if kind != "refinement":
            same([first.config["n"], first.h, np.diff(first.r)[0]],
                 [other.config["n"], other.h, np.diff(other.r)[0]], "controlled N/H/dr")
        if kind in ("fixed_profile_outlet", "refinement"):
            same(first.r[0], other.r[0], "controlled inlet")
            for key in ("start", "width", "strength", "damp_mean"):
                same(first.meta["sponge"][key], other.meta["sponge"][key], f"fixed sponge {key}")
            common, ia, ib = np.intersect1d(first.r, other.r, return_indices=True)
            if len(common) < 3:
                raise ValueError("Too few shared original sponge nodes")
            same(first.sponge[ia], other.sponge[ib], "fixed actual sponge profile")
        if kind in ("fixed_outlet_sponge", "refinement"):
            same([first.r[0], first.r[-1]], [other.r[0], other.r[-1]], "controlled radial extent")
        if kind == "fixed_outlet_sponge":
            for key in ("width", "strength", "damp_mean"):
                same(first.meta["sponge"][key], other.meta["sponge"][key], f"controlled sponge {key}")
    return selected


def load_linear(spec):
    if spec.get("accepted") is not True:
        raise ValueError("Root must separately approve each linear source for its stated use")
    paths = {key: frozen(source) for key, source in spec["sources"].items()}
    summary = read_json(paths["summary"])
    selected = []
    with np.load(paths["eigenpairs"], allow_pickle=False) as data:
        for selection in spec["selected_modes"]:
            index = int(selection["index"])
            mode = summary["modes"][index]
            if mode.get("accepted") is not True:
                raise ValueError("Selected mode fails algebraic acceptance")
            state = selection["resolution_status"]
            if state not in ("unverified", "not_converged", "grid_checked"):
                raise ValueError("Explicit linear resolution_status required")
            if state == "grid_checked" and not selection.get("resolution_evidence"):
                raise ValueError("grid_checked needs frozen resolution evidence")
            for source in selection.get("resolution_evidence", []):
                frozen(source)
            same(mode["omega_lab"], data["omega_lab"][index], "linear omega")
            same(mode["sigma"], data["sigma"][index], "linear sigma")
            same(mode["omega_disk"], mode["omega_lab"]-68, "disk frequency conversion")
            if not np.isfinite([mode["omega_disk"], mode["sigma"]]).all():
                raise ValueError("Invalid complex frequency")
            selected.append(dict(mode, selection=selection))
    if not selected or not spec.get("limitations"):
        raise ValueError("Linear selections and interpretation limits are required")
    return dict(spec=spec, summary=summary, modes=selected)


def load_diagnostic(spec):
    if spec.get("accepted") is not True or spec.get("interpretation") != "iteration_diagnostic":
        raise ValueError("Iteration diagnostics need explicit root approval for diagnostic use")
    sources = {key: frozen(source) for key, source in spec["sources"].items()}
    status, completion = read_json(sources["status"]), read_json(sources["completion"])
    history = status[spec.get("history_key", "history")]
    iteration_key = spec.get("iteration_key", "iteration")
    residual_key = spec.get("residual_key", "physical_residual_inf")
    iteration = np.asarray([row[iteration_key] for row in history], dtype=float)
    residual = np.asarray([row[residual_key] for row in history], dtype=float)
    if (not len(iteration) or np.any(~np.isfinite(iteration)) or np.any(np.diff(iteration) <= 0)
            or np.any(~np.isfinite(residual)) or np.any(residual <= 0)):
        raise ValueError("Convergence diagnostic needs ordered iterations and finite positive residuals")
    return dict(spec=spec, status=status, completion=completion, iteration=iteration, residual=residual)


def load_manifest(path):
    manifest = read_json(path)
    if (manifest.get("schema_version") != 1 or manifest.get("approval", {}).get("status") != "accepted"
            or manifest["approval"].get("reviewer") != "root"):
        raise ValueError("An explicit root-approved manifest is required before reading accepted results")
    if not manifest.get("validation_note"):
        raise ValueError("A visible validation_note is required")
    nonlinear = {}
    for spec in manifest.get("nonlinear_cases", []):
        if spec["id"] in nonlinear:
            raise ValueError("Duplicate nonlinear case id")
        nonlinear[spec["id"]] = load_nonlinear(spec, float(manifest.get("physical_residual_limit", 1e-7)))
    for group in manifest.get("nonlinear_comparisons", []):
        validate_comparison(group, nonlinear)
    linear = {}
    for spec in manifest.get("linear_cases", []):
        if spec["id"] in linear:
            raise ValueError("Duplicate linear case id")
        linear[spec["id"]] = load_linear(spec)
    for group in manifest.get("linear_comparisons", []):
        if not group["case_ids"] or any(key not in linear for key in group["case_ids"]):
            raise ValueError("Invalid linear comparison ids")
    diagnostics = [load_diagnostic(spec) for spec in manifest.get("nonlinear_diagnostics", [])]
    return manifest, nonlinear, linear, diagnostics


def audit(script, args, output, source_contract=False):
    result = subprocess.run([sys.executable, str(script), *args], capture_output=True, text=True)
    report = json.loads(result.stdout) if result.stdout.strip() else dict(stderr=result.stderr)
    failed = {item["check_id"] for item in report.get("findings", []) if item.get("level") == "FAIL"}
    exempt = source_contract and result.returncode == 1 and failed == {"EXPORT-VECTOR"}
    if exempt:
        report["user_export_contract"] = "PDF/PNG only; editable PDF supplied; generic SVG requirement exempted."
    output.write_text(json.dumps(report, indent=2, ensure_ascii=False)+"\n")
    if result.returncode and not exempt:
        raise RuntimeError(f"QA blocked export: {output}")


def save_figure(fig, axes, stem, manifest_path, manifest, qa_dir):
    from audit_panel_alignment import require_matplotlib_panel_alignment
    fig.canvas.draw()
    require_matplotlib_panel_alignment(fig, axes=axes,
        json_out=stem.with_suffix(".alignment.json"), tolerance_pt=1.5,
        require_panel_labels=len(axes) > 1, strict=True)
    fig.savefig(stem.with_suffix(".pdf"))
    fig.savefig(stem.with_suffix(".png"), dpi=600)
    audit(qa_dir/"audit_pdf_text.py", [str(stem.with_suffix(".pdf")), "--min-pt", "5", "--json"],
          stem.with_suffix(".pdf-text-qa.json"))
    audit(qa_dir/"audit_figure_collisions.py", [str(stem.with_suffix(".pdf")), "--json"],
          stem.with_suffix(".collision-qa.json"))
    record = dict(manifest=manifest, manifest_sha256=digest(manifest_path),
                  plotting_source_sha256=digest(__file__), backend="python", width_mm=180,
                  normalization="Nonlinear E_paper,1=2*pi*E1/R^2; linear eigenmode amplitudes are arbitrary",
                  smoothing="none", interpolation="none", visual_review="PENDING", root_visual_review="PENDING")
    stem.with_suffix(".qa.json").write_text(json.dumps(record, indent=2, ensure_ascii=False)+"\n")


def render(manifest_path, manifest, nonlinear, linear, diagnostics, output, qa_dir):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    sys.path.insert(0, str(qa_dir))
    output.mkdir(parents=True, exist_ok=True)
    audit(qa_dir/"validate_figure.py", [str(Path(__file__)), "--json"],
          output/"plot_source.qa.json", source_contract=True)
    plt.rcParams.update({"font.family": "sans-serif", "font.sans-serif": ["Arial", "Liberation Sans", "DejaVu Sans"],
        "font.size": 9, "mathtext.fontset": "stix", "pdf.fonttype": 42, "svg.fonttype": "none",
        "axes.spines.right": False, "axes.spines.top": False, "axes.linewidth": 0.8, "legend.frameon": False})
    width_mm = 180
    groups = manifest.get("nonlinear_comparisons", [])
    if groups:
        fig, axes = plt.subplots(len(groups), 1, squeeze=False,
            figsize=(width_mm/25.4, (90+65*(len(groups)-1))/25.4))
        axes = list(axes[:, 0]); fig.subplots_adjust(left=0.14, right=0.97, top=0.89, bottom=0.17, hspace=0.55)
        colors = {key: COLORS[i % len(COLORS)] for i, key in enumerate(nonlinear)}
        for number, (axis, group) in enumerate(zip(axes, groups)):
            for case in validate_comparison(group, nonlinear):
                energy = case.paper_energy()[:, 0]
                physical = case.r < float(case.meta["sponge"]["start"])
                color = colors[case.spec["id"]]
                axis.plot(case.r, np.ma.masked_where(~physical, energy), color=color, lw=1.5,
                    label=f"{case.spec['label']}; ω_lab={case.omega:.5f}")
                tail = np.arange(len(case.r)) >= np.flatnonzero(physical)[-1]
                axis.plot(case.r, np.ma.masked_where(~tail, energy), color=color, ls="--", lw=1, alpha=0.4)
                axis.axvline(case.meta["sponge"]["start"], color=color, ls=":", lw=0.8)
            axis.set_title(group["title"], fontsize=9, loc="left")
            axis.set_ylabel(r"$E_{\mathrm{paper},1}$"); axis.set_xlabel(r"$R$"); axis.set_ylim(bottom=0)
            axis.legend(loc="upper left", fontsize=8)
            if len(axes) > 1:
                axis.annotate(chr(97+number), (0, 1), xycoords="axes fraction", xytext=(-30, 5),
                              textcoords="offset points", fontsize=10, fontweight="bold")
        fig.text(0.14, 0.035, manifest["validation_note"], fontsize=8)
        save_figure(fig, axes, output/"nonlinear_controls", manifest_path, manifest, qa_dir); plt.close(fig)
    linear_groups = manifest.get("linear_comparisons", [])
    if linear_groups:
        fig, array = plt.subplots(1, len(linear_groups), squeeze=False,
                                  figsize=(width_mm/25.4, 135/25.4))
        axes = list(array[0])
        fig.subplots_adjust(left=0.115, right=0.975, top=0.75, bottom=0.22, wspace=0.36)
        fig.text(0.115, 0.995, "Discrete global-frequency candidates", fontsize=10, ha="left", va="top")
        for panel, (axis, group) in enumerate(zip(axes, linear_groups)):
            handles = []
            for number, case_id in enumerate(group["case_ids"]):
                case = linear[case_id]
                n = int(case["summary"]["config"]["n"])
                # Smaller N40 symbols preserve near-coincident N32 points.
                size = 16 if n >= 40 else 38
                for j, mode in enumerate(case["modes"]):
                    verified = mode["selection"]["resolution_status"] == "grid_checked"
                    mark = axis.scatter(mode["omega_disk"], mode["sigma"], s=size,
                        marker=MARKERS[number % len(MARKERS)], edgecolors=COLORS[number % len(COLORS)],
                        linewidths=0.8, facecolors=COLORS[number % len(COLORS)] if verified else "none",
                        label=case["spec"]["label"] if j == 0 else None, zorder=3 if n >= 40 else 2)
                    if j == 0:
                        handles.append(mark)
            axis.axhline(0, color="#777777", lw=0.7, ls="--", zorder=1)
            axis.set_xlabel(r"$\omega_{\mathrm{disk}}$"); axis.set_ylabel(r"$\sigma$")
            axis.set_title(group["title"], fontsize=9, loc="left", pad=8)
            axis.legend(handles=handles, fontsize=8, loc="lower left", bbox_to_anchor=(0, 1.12),
                        borderaxespad=0, handletextpad=0.5)
            if len(axes) > 1:
                axis.annotate(chr(97+panel), (0, 1), xycoords="axes fraction", xytext=(-28, 9),
                    textcoords="offset points", fontsize=10, fontweight="bold", annotation_clip=False)
        fig.text(0.115, 0.09, "Filled: grid-checked representatives; open: unverified candidates. No lines connect modes.", fontsize=8)
        fig.text(0.115, 0.045, r"Finite spectral subsets; $R_{\mathrm{out}}=720$ growth sign unresolved; no full-spectrum stability claim.", fontsize=8)
        save_figure(fig, axes, output/"linear_candidates", manifest_path, manifest, qa_dir); plt.close(fig)
    if diagnostics:
        fig, axis = plt.subplots(figsize=(width_mm/25.4, 100/25.4))
        fig.subplots_adjust(left=0.14, right=0.97, top=0.86, bottom=0.22)
        for number, record in enumerate(diagnostics):
            axis.plot(record["iteration"], record["residual"], color=COLORS[number % len(COLORS)],
                marker=MARKERS[number % len(MARKERS)], markersize=3,
                label=f"{record['spec']['label']}: {record['status']['status']}")
        axis.set_yscale("log"); axis.set_xlabel("Newton iteration")
        axis.set_ylabel("Physical residual (infinity norm)")
        axis.set_title("Nonlinear convergence diagnostics", loc="left", fontsize=9)
        axis.legend(fontsize=8)
        fig.text(0.14, 0.04, "Iteration records, including failures, are not physical solution branches.", fontsize=8)
        save_figure(fig, [axis], output/"nonlinear_convergence", manifest_path, manifest, qa_dir); plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("manifest", type=Path, help="Only a root-approved explicit acceptance manifest")
    parser.add_argument("--render", action="store_true", help="Opt in to integration/rendering after root acceptance")
    parser.add_argument("--output-dir", type=Path, default=OUT)
    parser.add_argument("--qa-scripts", type=Path, default=Path.home()/".codex/skills/nature-figure/scripts")
    args = parser.parse_args()
    manifest, nonlinear, linear, diagnostics = load_manifest(args.manifest)
    if not args.render:
        print(f"Validated sources: nonlinear={len(nonlinear)}, linear={len(linear)}, diagnostics={len(diagnostics)}; no integration/render.")
        return
    render(args.manifest, manifest, nonlinear, linear, diagnostics, args.output_dir, args.qa_scripts)


if __name__ == "__main__":
    main()
