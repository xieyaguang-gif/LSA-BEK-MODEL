#!/usr/bin/env python3
"""Recover sparse NL01 T=6 RMS isocontour crossings from the local source PDF.

This is deliberately not a reconstruction of a dense DNS radial trace.  The six
visible, unambiguous contour crossings are the entire exported reference.
"""
from __future__ import annotations

import csv
import hashlib
import json
import os
from pathlib import Path

os.environ.setdefault("OPENBLAS_NUM_THREADS", "2")
os.environ.setdefault("OMP_NUM_THREADS", "2")

import numpy as np
import pymupdf as fitz


ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "results/fullbvp_global_20261030/phaseB/reference_data"
SOURCE = Path("/home/lthpc/Desktop/data_bak/doctor-thesis/Transitional/2016 appelquist global nonlinear instability.pdf")
SOURCE_SHA256 = "7c9234c9f57ae9caf19ca572bf062eef307b816b78822d12b119d46c1cb51cce"
CSV = OUT / "nl01_T6_vrms.csv"
META = OUT / "nl01_T6_provenance.json"
OVERLAY = OUT / "nl01_T6_source_overlay.png"


def intersections(path, y):
    """Half-open crossing rule avoids duplicate line-segment vertices."""
    values = []
    for segment in path["items"]:
        if segment[0] != "l":
            raise ValueError("Expected straight vector segments in isocontour")
        a, b = segment[1:]
        if (a.y <= y < b.y) or (b.y <= y < a.y):
            values.append(a.x + (b.x - a.x) * (y - a.y) / (b.y - a.y))
    return sorted(values)


def calibration(drawings, indices, values):
    coords = np.array([drawings[i]["items"][0][1].x for i in indices])
    coefficients = np.polyfit(coords, values, 1)
    return coefficients, coords, float(np.max(np.abs(np.polyval(coefficients, coords) - values)))


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    if any(p.exists() for p in (CSV, META, OVERLAY)):
        raise SystemExit("Preserving existing outputs; use a new directory for another extraction.")
    source_hash = hashlib.sha256(SOURCE.read_bytes()).hexdigest()
    if source_hash != SOURCE_SHA256:
        raise ValueError("Source PDF hash changed; hard-coded source paths require review")
    source = fitz.open(SOURCE)
    page = source[11]
    drawings = page.get_drawings()
    radius_anchors = np.arange(500.0, 751.0, 50.0)
    coefficients, x_anchors, anchor_residual = calibration(
        drawings, [3024, 3026, 3028, 3030, 3032, 3034], radius_anchors
    )
    radius = lambda x: float(np.polyval(coefficients, x))
    y_t6 = float(drawings[3046]["items"][0][1].y)
    y_t5 = float(drawings[3044]["items"][0][1].y)
    pt_per_rotation = y_t5 - y_t6
    samples, crossing_audit = [], []
    expected_colors = [
        (0, 0.5019454956054688, 1), (0, 1, 1),
        (0.5019454956054688, 1, 0.5019454956054688),
        (1, 1, 0), (1, 0.5019454956054688, 0), (1, 0, 0),
    ]
    for index, log_level, expected_color in zip(range(50, 56), range(-7, -1), expected_colors):
        path = drawings[index]
        assert np.allclose(path["fill"], expected_color)
        xs = intersections(path, y_t6)
        assert len(xs) == 2 and abs(xs[-1] - 376.566) < 0.001
        x = xs[0]  # Second crossing is the closed polygon's R=700 domain edge.
        sample = dict(
            R=radius(x), vrms_normalized=10.0 ** log_level, log10_vrms=log_level,
            R_plot_reading_tolerance=2.5,
            vrms_plot_lower=10.0 ** (log_level - 0.5),
            vrms_plot_upper=10.0 ** (log_level + 0.5),
            pdf_x_pt=x, pdf_y_pt=y_t6,
        )
        samples.append(sample)
        near_y = y_t6 + 0.02 * pt_per_rotation
        near_x = intersections(path, near_y)[0]
        top_y = path["rect"].y0 + 0.0001
        top_x = intersections(path, top_y)[0]
        crossing_audit.append(dict(
            pdf_path_index_zero_based=index, fill_rgb=list(path["fill"]),
            log10_vrms=log_level, intersections_x_pt=xs,
            contour_top_y_pt=path["rect"].y0,
            R_at_T5_98=radius(near_x),
            R_at_polygon_top=radius(top_x),
            abs_delta_R_T5_98=float(abs(radius(near_x) - radius(x))),
            abs_delta_R_polygon_top=float(abs(radius(top_x) - radius(x))),
        ))
    assert np.all(np.diff([s["R"] for s in samples]) > 0)

    # Independent source-figure consistency check; never changes NL01 values.
    old_csv = ROOT / "results/appelquist2016_branch_20261029/literature/fig15_nl03_T6_vrms.csv"
    old = np.genfromtxt(old_csv, delimiter=",", names=True)
    c3, _, _ = calibration(drawings, [19919, 19921, 19923, 19925, 19927, 19929], radius_anchors)
    y3 = float(drawings[19945]["items"][0][1].y)
    checks = []
    for path_index, level in [(11260, -4), (11261, -3), (11262, -2)]:
        x3 = intersections(drawings[path_index], y3)[0]
        r3 = float(np.polyval(c3, x3))
        crossing_indices = np.flatnonzero(np.diff(np.signbit(old["log10_vrms"] - level)))
        assert len(crossing_indices) == 1
        k = crossing_indices[0]
        r15 = float(np.interp(level, old["log10_vrms"][k:k+2], old["R"][k:k+2]))
        checks.append(dict(log10_vrms=level, fig6e_R=r3, fig15_R=r15, fig15_minus_fig6e_R=r15-r3))

    # Source crop with exact crossing markers.  All content stays in a new PDF in
    # memory, so the original paper is never edited and no raster data are altered.
    audit = fitz.open()
    canvas = audit.new_page(width=940, height=500)
    canvas.insert_text((25, 26), "NL01 T = 6: six contour-derived RMS reference samples", fontsize=18)
    canvas.insert_text((25, 47), "Appelquist et al. (2016), Figure 6(b), PDF page 12 / printed page 343", fontsize=11)
    clip = fitz.Rect(245, 65, 422, 197)
    target = fitz.Rect(25, 90, 591.4, 512.4)
    # The lower crop margin is outside the page; panel labels end above y=495.
    canvas.show_pdf_page(target, source, 11, clip=clip)
    scale = target.width / clip.width
    for number, sample in enumerate(samples, 1):
        px = target.x0 + (sample["pdf_x_pt"] - clip.x0) * scale
        py = target.y0 + (sample["pdf_y_pt"] - clip.y0) * scale
        canvas.draw_circle((px, py), 5, color=(0.65, 0, 0.65), fill=(1, 1, 1), width=1.5)
        label_y = py - (21 if number % 2 else 39)
        canvas.draw_line((px, py - 5), (px, label_y + 3), color=(0.65, 0, 0.65), width=0.8)
        canvas.insert_text((px-3, label_y), str(number), fontsize=11, color=(0.65, 0, 0.65))
    canvas.insert_text((620, 108), "Markers intersect the T = 6 row", fontsize=13)
    canvas.insert_text((620, 137), "#       R          log10(v_rms)     v_rms", fontsize=10, fontname="cour")
    for number, sample in enumerate(samples, 1):
        text = f"{number}   {sample['R']:7.2f}          {sample['log10_vrms']:2d}        {sample['vrms_normalized']:.0e}"
        canvas.insert_text((620, 163 + 25*(number-1)), text, fontsize=10, fontname="cour")
    explanation = (
        "At z = 1.3; azimuthal RMS / local disk speed.\n\n"
        "Six samples only. No dense trace is inferred.\n"
        "Markers are the visible fill-color boundaries;\n"
        "the plot frame obscures their upper edges.\n\n"
        "Graphical comparison envelopes:\n"
        "R +/- 2.5; log10(v_rms) +/- 0.5.\n"
        "These are not statistical confidence limits.\n"
        "The -1 level has multiple turbulent crossings."
    )
    canvas.insert_textbox(fitz.Rect(620, 318, 930, 490), explanation, fontsize=10)
    canvas.get_pixmap(matrix=fitz.Matrix(1.5, 1.5), alpha=False).save(OVERLAY)

    with CSV.open("x", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(samples[0]))
        writer.writeheader()
        writer.writerows(samples)
    provenance = dict(
        schema_version=1, status="verified_sparse_vector_isocontour_samples",
        source_pdf=str(SOURCE), source_sha256=source_hash,
        paper_doi="10.1017/jfm.2016.506", pdf_page=12, pdf_page_index_zero_based=11,
        printed_page=343, figure="6", panel="b", case="NL01", T=6.0, z=1.3,
        quantity="Azimuthal velocity RMS normalized by local disk speed, equation (3.1)",
        normalization=dict(
            source_equation="3.1", source_pdf_page=11, source_printed_page=342,
            formula="sqrt(mean_theta((VN - mean_theta(VN))**2))", VN="V/(r*Omega)",
            project_formula="sqrt(2*sum(abs(q_v_j)**2 for j>=1))",
            condition="q_v_j are complex positive azimuthal Fourier coefficients already divided by R; exclude j=0; do not divide by R again",
        ),
        curve_identity="Visible color-band boundaries of NL01 panel (b), sampled where T=6 meets the top axis; not Figure 7 time histories or Figure 15 NL03",
        source_line_style="Filled vector isocontour boundaries; no radial line trace is present in this panel",
        extraction="Exact intersections of ordered PDF straight line segments with the calibrated T=6 row; six retained nominal levels, no smoothing, amplitude fit, rescaling, interpolation or extrapolation",
        calibration=dict(
            coordinate_system="PDF points, origin top-left, y increasing downward",
            R_of_x_slope_intercept=coefficients.tolist(), x_tick_points=x_anchors.tolist(),
            R_ticks=radius_anchors.tolist(), max_R_anchor_residual=anchor_residual,
            y_T6_pt=y_t6, y_T5_pt=y_t5, pt_per_rotation=pt_per_rotation,
            colorbar=dict(
                pdf_image_xref=1063, logarithm_base=10, lower=-8, upper=0,
                interval_decades=1,
                checked_color_order="blue, light blue, cyan, green, yellow, orange, red, dark red",
                method="Colorbar image and labeled ticks inspected; fill-color changes correspond to integer log10 levels",
            ),
        ),
        retained_points=len(samples), retained_log10_levels=list(range(-7, -1)),
        reliable_R_range=[samples[0]["R"], samples[-1]["R"]],
        recommended_label="NL01, Fig. 6(b), T=6 contour-derived samples",
        display="Use six markers with graphical envelopes. An optional log-linear visual guide may join adjacent samples only, explicitly identified as interpolation. Do not extrapolate or generate dense DNS samples.",
        uncertainty=dict(
            type="Conservative graphical comparison envelope, not DNS/statistical confidence limits",
            R_tolerance=2.5, log10_vrms_tolerance=0.5,
            vrms_multiplicative_factor=float(np.sqrt(10)),
            contour_level_spacing_decades=1,
            interpretation="Nominal isocontour levels are fixed by the published contour map. The +/-0.5-decade envelope explicitly represents the coarse graphical level resolution, not a statistical error in a known contour value.",
            R_tolerance_reason="Covers source coordinate/frame reading and the observed 1.6-1.7 R discrepancy between NL03 Fig.6(e) contours and the independently audited Fig.15 line",
            temporal_boundary="T=6 is exactly the top tick/frame. Filled paths extend 0.029 PDF point above it (~0.002 rotations), and the frame overlays the top endpoints. Crossings at the frame, polygon top, and T=5.98 were checked; no extrapolation beyond the displayed time is used.",
            max_abs_delta_R_T5_98=max(p["abs_delta_R_T5_98"] for p in crossing_audit),
            max_abs_delta_R_polygon_top=max(p["abs_delta_R_polygon_top"] for p in crossing_audit),
            excludes="DNS discretization, time sampling, statistical convergence, finite-time effects and modeling errors",
        ),
        exclusions=[
            "The -8 path is indistinguishable from the blue background by color and is not retained.",
            "The -1 dark-red level has multiple turbulent recrossings near the upper boundary and does not define a unique radial crossing; excluded.",
            "R=700 closure segments are polygon/domain edges, not RMS samples.",
            "Figure7 at fixed r=530 is not a radial reference curve. Figure15 contains NL03 only.",
        ],
        source_figure_consistency=dict(
            case="NL03", purpose="Independent audit of contour calibration/levels only; no transfer of NL03 amplitude to NL01",
            comparison_csv=str(old_csv.relative_to(ROOT)),
            comparison_csv_sha256=hashlib.sha256(old_csv.read_bytes()).hexdigest(),
            checks=checks, discrepancy_treatment="Reported, included in conservative R envelope, and not removed by shifting data",
        ),
        raw_crossing_audit=crossing_audit,
        visual_validation="Overlay generated for visual inspection; see supervisor review confirmation",
        csv=CSV.name, verification_overlay=OVERLAY.name,
        reproduce="OPENBLAS_NUM_THREADS=2 OMP_NUM_THREADS=2 python3 examples/fullbvp_phaseB_dns_reference.py (requires absent output files to preserve prior data)",
    )
    META.write_text(json.dumps(provenance, indent=2) + "\n")
    print(json.dumps(dict(csv=str(CSV.relative_to(ROOT)), rows=len(samples), R_range=provenance["reliable_R_range"], NL03_crosscheck=checks, temporal_checks=provenance["uncertainty"]["max_abs_delta_R_T5_98"]), indent=2))


if __name__ == "__main__":
    main()
