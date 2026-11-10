#!/usr/bin/env python3
"""Plot only explicitly accepted, immutable unforced free-frequency BVP cases.

The JSON input contract and interpretation limits are in FIGURE_BRIEF.md.
No globbing, solver calls, automatic phase alignment, or curve normalization.
Only PDF, PNG and machine-readable QA JSON files are exported.
"""
from __future__ import annotations

import argparse
from dataclasses import dataclass
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import subprocess
import sys

# This delegated postprocessing task is limited to one CPU thread.
for _name in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS",
              "NUMEXPR_NUM_THREADS", "VECLIB_MAXIMUM_THREADS"):
    os.environ[_name] = "1"

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_OUT = ROOT / "results/ro_minus1_beta68_free_bvp_20261029"
PALETTE = ("#0072B2", "#D55E00", "#009E73", "#CC79A7", "#555555", "#E69F00")
STYLES = ("-", "--", "-.", ":")
MARKERS = ("o", "s", "^", "D", "v", "P")


def read_json(path: Path) -> dict:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"Expected a JSON object: {path}")
    return value


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def resolve_path(value: str, root: Path = ROOT) -> Path:
    path = Path(value).expanduser()
    return (path if path.is_absolute() else root / path).resolve(strict=True)


def close(actual: float, expected: float, name: str) -> None:
    if not np.isclose(float(actual), float(expected), rtol=1e-9, atol=1e-10):
        raise ValueError(f"Mismatch for {name}: {actual!r} != {expected!r}")


@dataclass
class Case:
    spec: dict
    radii: np.ndarray
    q: np.ndarray
    config: dict
    metadata: dict
    status: dict
    omega: float
    identity: dict
    z: np.ndarray | None = None
    energies: np.ndarray | None = None

    @property
    def h(self) -> int:
        return int(self.config["harmonics"])

    @property
    def m(self) -> int:
        return int(self.config["n"]) + 1

    @property
    def sponge(self) -> float:
        return float(self.metadata["sponge"]["start"])


def load_case(spec: dict, residual_limit: float) -> Case:
    """Validate every identity and physical-status requirement before plotting."""
    for key in ("id", "label", "npz", "status_json", "npz_sha256",
                "status_sha256", "expected", "limitations"):
        if key not in spec:
            raise ValueError(f"Case lacks required key {key!r}")
    if spec.get("accepted") is not True:
        raise ValueError(f"Case {spec['id']!r} is not explicitly accepted")
    fields_path, status_path = resolve_path(spec["npz"]), resolve_path(spec["status_json"])
    identities = {}
    for name, path, hash_key in (("npz", fields_path, "npz_sha256"),
                                 ("status", status_path, "status_sha256")):
        actual = sha256(path)
        if actual != str(spec[hash_key]).lower():
            raise ValueError(f"Immutable source hash changed: {path}")
        identities[name] = {"path": str(path), "sha256": actual}
    status = read_json(status_path)
    with np.load(fields_path, allow_pickle=False) as data:
        radii, q = data["radii"].copy(), data["modes"].copy()
        config = json.loads(str(data["config_json"].item()))
        meta = json.loads(str(data["metadata_json"].item()))
        omega = float(data["omega"].item())
        growth = float(data["auxiliary_growth"].item())
    for origin, record in (("NPZ metadata", meta), ("status JSON", status)):
        if record.get("status") != "converged":
            raise ValueError(f"{spec['id']}: {origin} is not converged")
        if float(record.get("growth", np.inf)) != 0.0:
            raise ValueError(f"{spec['id']}: {origin} has nonzero auxiliary growth")
        residual = float(record.get("physical_residual_inf", np.inf))
        if not np.isfinite(residual) or residual < 0 or residual > residual_limit:
            raise ValueError(f"{spec['id']}: invalid/excessive physical residual")
        if float(record.get("max_E1", 0)) <= 0 or float(record.get("E1_integral", 0)) <= 1e-10:
            raise ValueError(f"{spec['id']}: a nonzero periodic root is required")
        close(record["omega"], omega, f"{origin} omega")
    if (growth != 0.0 or meta.get("auxiliary") is not False
            or meta.get("amplitude_constraint") is not None
            or meta.get("homogeneous_inlet") is not True
            or "interior_forcing" not in meta or meta["interior_forcing"] is not None):
        raise ValueError(f"{spec['id']}: fields do not certify an unforced system")
    close(meta["physical_residual_inf"], status["physical_residual_inf"], "residual")
    close(config["ro"], -1.0, "Ro")
    close(config["beta"], 68.0, "beta")
    if config.get("nonlinear") is not True:
        raise ValueError("Expected nonlinear equations")
    h, m = int(config["harmonics"]), int(config["n"]) + 1
    if (radii.ndim != 1 or len(radii) < 3 or np.any(~np.isfinite(radii))
            or np.any(np.diff(radii) <= 0) or q.shape != (len(radii), 2*h+1, 4*m)
            or np.any(~np.isfinite(q))):
        raise ValueError(f"{spec['id']}: invalid radius/mode arrays")
    if not np.allclose(q[:, :h], q[:, h+1:][:, ::-1].conj(), rtol=1e-8, atol=1e-10):
        raise ValueError(f"{spec['id']}: broken conjugate-harmonic symmetry")
    if np.max(np.abs(q[:, h].imag)) > 1e-10:
        raise ValueError(f"{spec['id']}: mean mode is not real")
    expected = spec["expected"]
    actual = {"n": config["n"], "h": h, "omega": omega,
              "rmin": radii[0], "rmax": radii[-1],
              "dr": float(np.diff(radii)[0]), "sponge_start": meta["sponge"]["start"]}
    for key, value in actual.items():
        if key not in expected:
            raise ValueError(f"{spec['id']}: expected.{key} is required")
        close(value, expected[key], f"{spec['id']} {key}")
    if not np.allclose(np.diff(radii), actual["dr"], rtol=1e-10, atol=1e-10):
        raise ValueError("Current data contract requires a uniform radial grid")
    start = float(meta["sponge"]["start"])
    if not radii[0] < start < radii[-1]:
        raise ValueError("Sponge onset must lie inside the finite domain")
    close(start + float(meta["sponge"]["width"]), radii[-1], "sponge end")
    if not isinstance(spec["limitations"], list) or not spec["limitations"]:
        raise ValueError("Each case must retain an explicit validation limitation")
    return Case(spec, radii, q, config, meta, status, omega, identities)


def load_manifest(path: Path) -> tuple[dict, list[Case]]:
    manifest = read_json(path)
    if manifest.get("schema_version") != 1:
        raise ValueError("Expected manifest schema_version=1")
    if not manifest.get("validation_note"):
        raise ValueError("A visible validation_note is required")
    cases = [load_case(spec, float(manifest.get("physical_residual_limit", 1e-7)))
             for spec in manifest["cases"]]
    ids = [case.spec["id"] for case in cases]
    if not ids or len(ids) != len(set(ids)) or manifest["reference_case"] not in ids:
        raise ValueError("Cases need unique ids and a valid reference_case")
    if manifest.get("energy_scale", "linear") not in ("linear", "log"):
        raise ValueError("energy_scale must be linear or log")
    if manifest.get("display_z_max") is not None and not manifest.get("height_crop_reason"):
        raise ValueError("A physical height crop requires height_crop_reason")
    if manifest.get("grid_archive"):
        grid_spec = manifest["grid_archive"]
        if sha256(resolve_path(grid_spec["npz"])) != grid_spec["sha256"]:
            raise ValueError("Immutable original-grid archive hash changed")
    return manifest, cases


def derive(case: Case, archive: dict | None = None) -> None:
    # Prefer the frozen original grid and mass. The legacy fallback constructs
    # only a cached grid: no base-flow solve, radial operators or BVP evaluation.
    if archive is not None:
        if int(archive["n"]) != int(case.config["n"]):
            raise ValueError("Grid archive N differs from the accepted case")
        case.z, mass = archive["z"], archive["mass"]
    else:
        sys.path.insert(0, str(ROOT / "src"))
        from llsa.secondary.discretization import pier_grid
        grid = pier_grid(n=int(case.config["n"]))
        case.z, mass = grid.z, grid.mass
    if case.z.shape != (case.m,) or mass.shape != (case.m, case.m):
        raise ValueError("Grid/mass dimensions do not match the accepted fields")
    velocity = case.q[:, case.h+1:, :3*case.m].reshape(len(case.radii), case.h, 3, case.m)
    energies = case.radii[:, None]**2 * np.einsum(
        "rhvi,ij,rhvj->rh", velocity.conj(), mass, velocity).real
    if np.any(~np.isfinite(energies)) or np.min(energies) < -1e-9:
        raise ValueError("Non-finite or negative harmonic energy")
    # Roundoff-only negative entries are rejected rather than silently clipped.
    if np.any(energies < 0):
        raise ValueError("Negative energy detected; inspect quadrature roundoff")
    case.energies = energies
    close(energies[:, 0].max(), case.status["max_E1"], "reconstructed max E1")


def run_json_audit(script: Path, arguments: list[str], target: Path,
                   contract_exemptions: dict | None = None) -> dict:
    result = subprocess.run([sys.executable, str(script), *arguments],
                            capture_output=True, text=True, check=False)
    if result.stdout.strip():
        try:
            report = json.loads(result.stdout)
        except json.JSONDecodeError:
            report = {"stdout": result.stdout, "stderr": result.stderr}
    else:
        report = {"stderr": result.stderr}
    allowed = contract_exemptions or {}
    failed_ids = {item["check_id"] for item in report.get("findings", [])
                  if item.get("level") == "FAIL"}
    resolved = bool(failed_ids) and failed_ids.issubset(allowed)
    if resolved:
        # Keep the raw validator verdict visible. Only the user's explicit
        # export contract can supersede the generic SVG requirement.
        report["contract_exemptions"] = {key: allowed[key] for key in sorted(failed_ids)}
        report["effective_ready_under_user_contract"] = True
    target.write_text(json.dumps(report, indent=2, ensure_ascii=False) + "\n")
    if result.returncode and not resolved:
        raise RuntimeError(f"QA failed ({result.returncode}); inspect {target}")
    return report


def render(manifest: dict, cases: list[Case], manifest_path: Path,
           out: Path, qa_dir: Path) -> None:
    os.environ.setdefault("MPLCONFIGDIR", "/tmp/ro1_beta68_mplconfig")
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib import font_manager
    from matplotlib.colors import LinearSegmentedColormap, TwoSlopeNorm
    from matplotlib.lines import Line2D
    from matplotlib.ticker import MaxNLocator

    if not qa_dir.is_dir():
        raise FileNotFoundError("Required Nature figure QA scripts directory is unavailable")
    for script in ("validate_figure.py", "audit_pdf_text.py", "audit_figure_collisions.py",
                   "audit_panel_alignment.py"):
        if not (qa_dir / script).is_file():
            raise FileNotFoundError(f"Required QA helper missing: {script}")
    if importlib.util.find_spec("fitz") is None:
        raise RuntimeError("Rendered collision QA requires PyMuPDF (fitz)")
    sys.path.insert(0, str(qa_dir))
    from audit_panel_alignment import require_matplotlib_panel_alignment

    out.parent.mkdir(parents=True, exist_ok=True)
    run_json_audit(qa_dir / "validate_figure.py", [str(Path(__file__)), "--json"],
                   out.with_suffix(".source-qa.json"),
                   {"EXPORT-VECTOR": "User explicitly requires PDF and PNG only; PDF is editable vector output, SVG excluded."})
    archive = None
    archive_identity = None
    if manifest.get("grid_archive"):
        archive_spec = manifest["grid_archive"]
        archive_path = resolve_path(archive_spec["npz"])
        archive_hash = sha256(archive_path)
        if archive_hash != archive_spec["sha256"]:
            raise ValueError("Immutable grid archive hash changed")
        with np.load(archive_path, allow_pickle=False) as data:
            archive = {key: data[key].copy() for key in ("n", "a", "b", "c", "z", "mass")}
        archive_identity = {"path": str(archive_path), "sha256": archive_hash,
                            "mapping": {key: float(archive[key]) for key in ("a", "b", "c")}}
    for case in cases:
        derive(case, archive)
    ref = next(case for case in cases if case.spec["id"] == manifest["reference_case"])
    installed = {font.name for font in font_manager.fontManager.ttflist}
    text_font = next((name for name in ("Arial", "Helvetica", "Liberation Sans", "DejaVu Sans")
                      if name in installed), "DejaVu Sans")
    plt.rcParams.update({"font.family": "sans-serif", "font.sans-serif": [text_font],
                         "font.size": 9, "mathtext.fontset": "stix", "pdf.fonttype": 42,
                         "svg.fonttype": "none",
                         "axes.linewidth": 0.8, "axes.spines.top": False,
                         "axes.spines.right": False, "legend.frameon": False,
                         "xtick.direction": "out", "ytick.direction": "out"})
    width_mm = 180.0
    height_mm = float(manifest.get("height_mm", 145.0))
    fig = plt.figure(figsize=(width_mm/25.4, height_mm/25.4))
    fig.text(0.13, 0.995, "Finite-domain periodic candidates: harmonic and outlet tests",
             ha="left", va="top", fontsize=10)
    grid = fig.add_gridspec(2, 1, height_ratios=(1.25, 1.0),
                           left=0.13, right=0.965, bottom=0.23, top=0.78, hspace=0.38)
    ax_a, ax_b = fig.add_subplot(grid[0]), fig.add_subplot(grid[1])
    log_scale = manifest.get("energy_scale", "linear") == "log"
    excluded_zeros = {}
    legend_handles = []
    common_rmin = max(case.radii[0] for case in cases)
    common_rmax = min(case.radii[case.radii < case.sponge][-1] for case in cases)
    if common_rmin >= common_rmax:
        raise ValueError("Accepted cases lack a common sponge-free comparison interval")
    ax_a.set_title(f"Common sponge-free: {common_rmin:g} ≤ R ≤ {common_rmax:g}; ω: inertial / disk rotation",
                   loc="left", fontsize=8, pad=6)
    for index, case in enumerate(cases):
        color, marker = PALETTE[index % len(PALETTE)], MARKERS[index % len(MARKERS)]
        energy = case.energies[:, 0]
        # Logarithms cannot represent exact zeros. Only those zeros are masked;
        # their count and radii are recorded in the QA JSON, never offset.
        displayed = np.ma.masked_equal(energy, 0.0) if log_scale else energy
        physical = case.radii < case.sponge
        ax_a.plot(case.radii, np.ma.masked_where(~physical, displayed), color=color,
                  ls="-", lw=1.5, marker=marker, markersize=2.5,
                  markerfacecolor="white", markevery=max(1, len(case.radii)//8))
        # Include one pre-sponge sample to join the raw curve without a gap.
        sponge_segment = case.radii >= case.radii[np.flatnonzero(physical)[-1]]
        ax_a.plot(case.radii, np.ma.masked_where(~sponge_segment, displayed),
                  color=color, ls="--", lw=1.0, alpha=0.45)
        excluded_zeros[case.spec["id"]] = case.radii[energy == 0].tolist() if log_scale else []
        ax_a.axvline(case.sponge, color=color, lw=0.9, ls=(2*index, (2, 4)), alpha=0.9)
        label = (f"{case.spec['label']}  N={case.config['n']}, H={case.h}, "
                 f"ΔR={np.diff(case.radii)[0]:g}, ω={case.omega:.5f}, sponge={case.sponge:g}")
        legend_handles.append(Line2D([], [], color=color, ls="-", marker=marker,
                                    markersize=3, markerfacecolor="white", lw=1.5, label=label))
    # All harmonics retained by the explicitly selected highest-H case appear;
    # absolute energy units remain shared with E1, without separate scaling.
    harmonic_id = manifest.get("harmonic_case")
    high = (next(case for case in cases if case.spec["id"] == harmonic_id)
            if harmonic_id else max(cases, key=lambda case: (case.h, case.config["n"])))
    if high.h != max(case.h for case in cases):
        raise ValueError("harmonic_case must have the highest accepted H")
    for harmonic in range(2, high.h+1):
        energy = high.energies[:, harmonic-1]
        displayed = np.ma.masked_equal(energy, 0.0) if log_scale else energy
        if log_scale:
            excluded_zeros[f"{high.spec['id']}:E{harmonic}"] = high.radii[energy == 0].tolist()
        physical = high.radii < high.sponge
        ax_a.plot(high.radii, np.ma.masked_where(~physical, displayed), color="#777777", lw=0.9,
                  ls=STYLES[(harmonic-1) % len(STYLES)], alpha=0.8)
        sponge_segment = high.radii >= high.radii[np.flatnonzero(physical)[-1]]
        ax_a.plot(high.radii, np.ma.masked_where(~sponge_segment, displayed),
                  color="#777777", lw=0.8, ls="--", alpha=0.35)
        legend_handles.append(Line2D([], [], color="#777777", lw=0.9,
                              ls=STYLES[(harmonic-1) % len(STYLES)],
                              label=f"E{harmonic}: {high.spec['label']}"))
    if log_scale:
        positive = [case.energies[case.energies > 0] for case in cases]
        if not any(value.size for value in positive):
            raise ValueError("No positive harmonic energy for a log scale")
        ax_a.set_yscale("log")
    else:
        ax_a.set_ylim(bottom=0)
        ax_a.ticklabel_format(axis="y", style="sci", scilimits=(-3, 4))
    ax_a.set_ylabel(r"$E_j(R)$")
    ax_a.xaxis.set_major_locator(MaxNLocator(6))
    fig.legend(handles=legend_handles, loc="upper left", bbox_to_anchor=(0.12, 0.965),
               fontsize=8, handlelength=2.5, ncol=1)

    phase = float(manifest.get("field_phase_radians", 0.0))
    js = np.arange(1, ref.h+1)
    vv = ref.q[:, :, ref.m:2*ref.m]
    field = vv[:, ref.h].real + 2*np.real(np.einsum(
        "rhj,h->rj", vv[:, ref.h+1:], np.exp(1j*js*phase)))
    finite = np.isfinite(ref.z)
    keep = finite.copy()
    if manifest.get("display_z_max") is not None:
        zmax = float(manifest["display_z_max"])
        if zmax <= 0:
            raise ValueError("display_z_max must be a positive physical height")
        keep &= ref.z <= zmax
    if np.count_nonzero(keep) < 2:
        raise ValueError("Field display requires at least two finite height nodes")
    displayed_field = field[:, keep]
    vmax = float(np.max(np.abs(displayed_field)))
    if not np.isfinite(vmax) or vmax <= 0:
        raise ValueError("Reference field is identically zero or invalid")
    cmap = LinearSegmentedColormap.from_list("signed_velocity", ["#2166AC", "white", "#B2182B"])
    cmap.set_bad("#BBBBBB")
    # Keep all finite original nodes in the cell geometry, then crop only the
    # axes window. Truncating the grid first would move its outer cell edge.
    mesh = ax_b.pcolormesh(ref.radii, ref.z[finite], field[:, finite].T,
                           shading="nearest", cmap=cmap,
                           norm=TwoSlopeNorm(vmin=-vmax, vcenter=0, vmax=vmax),
                           rasterized=True)
    ax_b.axvline(ref.sponge, color="#333333", lw=0.9, ls=(0, (2, 3)))
    upper_height = (float(manifest["display_z_max"]) if manifest.get("display_z_max") is not None
                    else float(ref.z[keep][-1]))
    ax_b.set_ylim(float(ref.z[keep][0]), upper_height)
    ax_b.set_xlabel(r"$R$")
    ax_b.set_ylabel(r"$z$ (BEK units)")
    ax_b.xaxis.set_major_locator(MaxNLocator(6))
    ax_b.yaxis.set_major_locator(MaxNLocator(4))
    common_xlim = (min(case.radii[0] for case in cases), max(case.radii[-1] for case in cases))
    if ref.radii[-1] < common_xlim[1]:
        ax_b.axvspan(ref.radii[-1], common_xlim[1], facecolor="#EEEEEE",
                     edgecolor="#CCCCCC", lw=0, zorder=3)
        ax_b.text((ref.radii[-1]+common_xlim[1])/2, 0.65*upper_height,
                   "Outside\nreference\ndomain", fontsize=8, ha="center", va="center",
                   color="#555555", zorder=4)
    for axis, letter in ((ax_a, "a"), (ax_b, "b")):
        axis.set_xlim(common_xlim)
        axis.annotate(letter, xy=(0, 1), xycoords="axes fraction", xytext=(-28, 5),
                      textcoords="offset points", fontsize=10, fontweight="bold",
                      ha="left", va="bottom", annotation_clip=False)
    ax_b.set_title(f"{ref.spec['label']}; phase={phase:g} rad; mean + all positive harmonics",
                   loc="left", fontsize=8, pad=5)
    cax = fig.add_axes([0.35, 0.12, 0.40, 0.018])
    colorbar = fig.colorbar(mesh, cax=cax, orientation="horizontal")
    fig.text(0.33, 0.129, r"$v'(R,z)/R$", ha="right", va="center", fontsize=9)
    colorbar.locator = MaxNLocator(5)
    colorbar.update_ticks()
    fig.text(0.13, 0.050, "Solid: sponge-free; pale dashed: sponge; vertical lines: sponge onset.",
             ha="left", va="bottom", fontsize=8)
    fig.text(0.13, 0.015, manifest["validation_note"], ha="left", va="bottom", fontsize=8)
    fig.canvas.draw()
    require_matplotlib_panel_alignment(fig, axes=[ax_a, ax_b], panel_ids=["a", "b"],
        json_out=str(out.with_suffix(".alignment.json")), tolerance_pt=1.5,
        gutter_tolerance_pt=1.5, require_panel_labels=True, strict=True)
    # Fixed physical canvas: no bbox_inches='tight' size changes.
    pdf_path, png_path = out.with_suffix(".pdf"), out.with_suffix(".png")
    fig.savefig(pdf_path, metadata={"Title": "Unforced finite-domain periodic BVP candidates"})
    fig.savefig(png_path, dpi=600)
    plt.close(fig)
    run_json_audit(qa_dir / "audit_pdf_text.py", [str(pdf_path), "--min-pt", "5", "--json"],
                   out.with_suffix(".pdf-text-qa.json"))
    run_json_audit(qa_dir / "audit_figure_collisions.py", [str(pdf_path), "--json"],
                   out.with_suffix(".collision-qa.json"))
    comparison = []
    comparison_id = manifest.get("comparison_reference_case", cases[0].spec["id"])
    comparison_ref = next(case for case in cases if case.spec["id"] == comparison_id)
    for radius in manifest.get("comparison_radii", []):
        radius = float(radius)
        if not common_rmin <= radius <= common_rmax:
            raise ValueError("Comparison probes must lie in the common sponge-free interval")
        reference_index = np.flatnonzero(np.isclose(comparison_ref.radii, radius, rtol=0, atol=1e-9))
        if len(reference_index) != 1:
            raise ValueError("Comparison probes must be original shared radial nodes")
        reference_energy = float(comparison_ref.energies[reference_index[0], 0])
        if reference_energy <= 0:
            raise ValueError("Comparison energy denominator must be positive")
        for case in cases:
            node = np.flatnonzero(np.isclose(case.radii, radius, rtol=0, atol=1e-9))
            if len(node) != 1:
                raise ValueError("Comparison probes must be original shared radial nodes")
            value = float(case.energies[node[0], 0])
            comparison.append({"R": radius, "case": case.spec["id"], "E1": value,
                "reference_case": comparison_id,
                "E1_difference_percent": 100*(value/reference_energy-1)})
    report = {"schema_version": 1, "manifest": {"path": str(manifest_path),
              "sha256": sha256(manifest_path)}, "backend": "python", "font": text_font,
              "figure_mm": [width_mm, height_mm], "exports": [str(pdf_path), str(png_path)],
              "energy_definition": "R^2 integral (|u_j|^2+|v_j|^2+|w_j|^2) dz; no factor 2",
              "energy_scale": manifest.get("energy_scale", "linear"),
              "log_zero_radii": excluded_zeros, "reference_case": ref.spec["id"],
              "field_phase_radians": phase, "field_color_limits": [-vmax, vmax],
              "field_height_nodes": {"total": len(ref.z), "finite": int(finite.sum()),
                  "displayed": int(keep.sum()), "infinity_endpoint_excluded": int((~finite).sum()),
                  "finite_nodes_cropped": int((finite & ~keep).sum()),
                  "crop_reason": manifest.get("height_crop_reason")},
              "field_geometry": "raw mapped nodes; nearest cells; independently scaled axes",
              "interpolation": "none", "smoothing": "none", "curve_normalization": "none",
              "frequency_units": "inertial / disk rotation (Ro=-1)",
              "common_sponge_free_interval": [float(common_rmin), float(common_rmax)],
              "common_region_original_node_comparison": comparison,
              "comparison_rule": "Compare common sponge-free interval; full-domain maxima have different support",
              "plotting_source": {"path": str(Path(__file__)), "sha256": sha256(Path(__file__))},
              "frozen_grid_archive": archive_identity,
              "grid_reconstruction": None if archive_identity else {"a": 2.0, "b": 0.6, "c": 0.5,
                  "source_sha256": sha256(ROOT/"src/llsa/secondary/discretization.py")},
              "cases": [{"id": case.spec["id"], "sources": case.identity,
                         "omega": case.omega, "omega_disk_signed": case.omega-68,
                         "pattern_speed_over_disk": case.omega/68,
                         "probe_z_from_metadata": case.metadata.get("probe_z"),
                         "limitations": case.spec["limitations"]} for case in cases],
              "visual_review": "PENDING: inspect every panel and assembled PNG at final size",
              "qa_scripts": {name: sha256(qa_dir/name) for name in
                             ("validate_figure.py", "audit_pdf_text.py", "audit_figure_collisions.py",
                              "audit_panel_alignment.py")}}
    out.with_suffix(".qa.json").write_text(json.dumps(report, indent=2, ensure_ascii=False)+"\n")
    print(f"Exported {pdf_path.name} and {png_path.name}; final visual review remains pending.")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("manifest", type=Path, help="Explicit accepted-cases JSON; paths relative to project root")
    parser.add_argument("--output-stem", type=Path, default=DEFAULT_OUT/"free_bvp_diagnostic")
    parser.add_argument("--qa-scripts", type=Path,
                        default=Path.home()/".codex/skills/nature-figure/scripts")
    parser.add_argument("--check-manifest", action="store_true",
                        help="Verify identities/metadata only; no integration or drawing")
    args = parser.parse_args()
    manifest_path = args.manifest.resolve(strict=True)
    manifest, cases = load_manifest(manifest_path)
    if args.check_manifest:
        print(f"Validated {len(cases)} explicit accepted case(s); no derived fields or render.")
        return
    render(manifest, cases, manifest_path, args.output_stem.resolve(), args.qa_scripts.resolve())


if __name__ == "__main__":
    main()
