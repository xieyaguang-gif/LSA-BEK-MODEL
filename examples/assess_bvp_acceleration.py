"""Audit continuation cost and overlapping physical curves from saved BVP runs.

Uses the existing beta42 probe/filter definitions. A converged algebraic root
is reported separately from resolution and outlet-independence validation.
"""
import argparse
import json
from pathlib import Path

import numpy as np

from plot_bvp_dns_four_amplitudes import bvp_case


def cost(folder):
    meta = json.loads((folder / "metadata.json").read_text())
    steps = meta.get("steps", [])
    elapsed = meta.get("elapsed_seconds", 0.)
    factor = sum(s["factor_seconds"] for s in steps)
    return {
        "source": str(folder), "status": meta["status"],
        "sponge_start_Re": meta["sponge_start"],
        "points": meta["points"], "config": meta["config"],
        "elapsed_seconds": elapsed, "accepted_newton_updates": len(steps),
        "residual_history": meta["newton_history"],
        "recorded_factor_seconds": factor,
        "recorded_assembly_seconds": sum(s["assembly_seconds"] for s in steps),
        "recorded_linear_seconds_including_factor": sum(s["linear_seconds"] for s in steps),
        "recorded_krylov_iterations": sum(s["krylov_iterations"] for s in steps),
        "factor_ages": [s["factor_age"] for s in steps],
        "damping": [s["damping"] for s in steps],
        "step_paths": [s.get("step_path", "phase_continuity" if s.get("phase_continuity")
                              else "phase" if s.get("phase_update") else "cartesian") for s in steps],
        "last_factor_threads": meta.get("linear_stage", {}).get("threads"),
        "timing_note": "Accepted updates only; failed/unaccepted steps may add cost.",
    }


def compare(a, b, intervals):
    report = []
    for lo, hi in intervals:
        # Exclude filter-affected ends, as in the existing comparison figure.
        left = max(lo, a["re"][0]+6, b["re"][0]+6)
        right = min(hi, a["re"][-1]-6, b["re"][-1]-6)
        if right <= left:
            continue
        x = np.arange(left, right+1e-9, .25)
        aa, ab = [np.interp(x, c["re"], c["amplitude"]) for c in (a, b)]
        ga, gb = [np.interp(x, c["re"], c["growth"]) for c in (a, b)]
        report.append({
            "Re_interval": [float(x[0]), float(x[-1])],
            "amplitude_relative_max": float(np.max(abs(ab/aa-1))),
            "growth_rmse": float(np.sqrt(np.mean((gb-ga)**2))),
            "growth_absolute_max": float(np.max(abs(gb-ga))),
        })
    return report


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--baseline", type=Path, required=True)
    p.add_argument("--run", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True)
    args = p.parse_args()
    baseline = cost(args.baseline)
    trials = [cost(f) for f in sorted(args.run.glob("s*_try*"))
              if (f / "metadata.json").exists()]
    report = {
        "baseline": baseline, "trials": trials,
        "curve_definition": "A=sqrt(2)|v1(z_delta=1.1858025795710305)|/Omega_d; "
            "g=0.751*dln(A)/dRe, log interpolation dRe=0.25 and Gaussian sigma_Re=2",
        "validation_note": "A residual below tolerance certifies only the discrete equations. "
            "Overlap comparisons assess sensitivity to domain/sponge changes at fixed resolution; "
            "they do not establish grid or harmonic convergence.",
        "overlap_checks": [],
    }
    a = bvp_case(args.baseline / "fields.npz", baseline["sponge_start_Re"])
    for trial in trials:
        if trial["status"] != "converged_discrete_equations":
            continue
        path = Path(trial["source"]) / "fields.npz"
        if trial["sponge_start_Re"] == baseline["sponge_start_Re"]:
            trial["wall_speedup_vs_baseline"] = baseline["elapsed_seconds"] / trial["elapsed_seconds"]
            trial["same_initial_residual"] = bool(np.isclose(trial["residual_history"][0],
                baseline["residual_history"][0], rtol=1e-9, atol=1e-14))
            with np.load(args.baseline / "fields.npz") as old, np.load(path) as new:
                if (old["radii"].shape == new["radii"].shape
                        and np.allclose(old["radii"], new["radii"], rtol=0, atol=1e-12)
                        and old["modes"].shape == new["modes"].shape):
                    trial["field_relative_l2_difference"] = float(np.linalg.norm(new["modes"]-old["modes"])
                                                                 / np.linalg.norm(old["modes"]))
        b = bvp_case(path, trial["sponge_start_Re"])
        report["overlap_checks"].append({
            "source": str(path),
            "intervals": compare(a, b, [(330, 380), (380, 420), (420, 450), (450, 470), (470, 480)]),
        })
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2)+"\n")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
