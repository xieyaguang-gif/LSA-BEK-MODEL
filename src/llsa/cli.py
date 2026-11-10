"""Command line entry point. JSON records conventions; NPZ retains eigenvectors."""
import argparse
from dataclasses import asdict
import json
from pathlib import Path
import platform
from time import perf_counter

import numpy as np
import scipy

from . import LocalProblem, Parameters, load_baseflow, saddle, solve_bek, spatial, temporal, track_temporal


def _complex(s):
    return complex(s.replace("i", "j"))


def _json(value):
    if isinstance(value, complex):
        return {"real": value.real, "imag": value.imag}
    if isinstance(value, np.generic):
        return value.item()
    raise TypeError(type(value).__name__)


def main(argv=None):
    import sys
    arguments = sys.argv[1:] if argv is None else argv
    if arguments and arguments[0] == 'radial-arc':
        from .secondary.radial_arc_cli import main as arc_main
        return arc_main(arguments[1:])
    if arguments and arguments[0] == 'radial-release':
        from .secondary.radial_release import main as release_main
        return release_main(arguments[1:])
    if arguments and arguments[0] == 'radial-continue':
        from .secondary.radial_continuation import main as continuation_main
        return continuation_main(arguments[1:])
    if arguments and arguments[0] == 'radial-nonlinear':
        from .secondary.radial_cli import main as radial_main
        return radial_main(arguments[1:])
    if arguments and arguments[0] == 'beta42-spatial':
        from .secondary.spatial_cli import main as spatial_main
        return spatial_main(arguments[1:])
    if arguments and arguments[0].startswith('pier-'):
        from .secondary.cli import main as pier_main
        return pier_main(arguments)
    parser = argparse.ArgumentParser(description="Rotating-disk local stability (raw MATLAB alpha/omega convention)")
    parser.add_argument("command", choices=["baseflow", "temporal", "spatial", "saddle", "scan"])
    parser.add_argument("--re", type=float, default=500, help="Re=abs(Ro)*r_delta, not r_delta")
    parser.add_argument("--ro", type=float, default=-0.751)
    parser.add_argument("--beta", type=float, default=85, help="azimuthal mode number")
    parser.add_argument("--n", type=int, default=64, help="polynomial degree (N+1 points)")
    parser.add_argument("--zmax", type=float, default=20)
    parser.add_argument("--stretch", type=float, default=0.25)
    parser.add_argument("--boundary", choices=["matlab", "paper"], default="matlab")
    parser.add_argument("--flow", type=Path, help="six-column profile; omitted: solve BEK")
    parser.add_argument("--layout", choices=["temporal", "spatial"], default="temporal")
    parser.add_argument("--interpolation", choices=["pchip", "spline"], default="pchip")
    parser.add_argument("--alpha", type=_complex, default=0.26)
    parser.add_argument("--omega", type=_complex, default=0.0)
    parser.add_argument("--target", type=_complex, help="initial omega target for temporal/saddle/scan")
    parser.add_argument("--imag-limit", type=float, default=1)
    parser.add_argument("--alpha-min", type=float, default=0.1)
    parser.add_argument("--alpha-max", type=float, default=0.4)
    parser.add_argument("--count", type=int, default=31)
    parser.add_argument("--max-iter", type=int, default=100)
    parser.add_argument("--max-step", type=float, default=0.01)
    parser.add_argument("--tol", type=float, default=1e-10)
    parser.add_argument("--output", type=Path, default=Path("results/run"), help="output prefix")
    args = parser.parse_args(argv)
    start = perf_counter()
    try:
        flow = (load_baseflow(args.flow, args.layout, args.interpolation, args.ro) if args.flow
                else solve_bek(args.ro, args.zmax))
        args.output.parent.mkdir(parents=True, exist_ok=True)
        def output(suffix):
            return Path(str(args.output)+suffix)
        params = Parameters(args.re, args.ro, args.beta, args.n, args.zmax, args.stretch, args.boundary)
        metadata = dict(parameters=asdict(params), command=args.command, flow_source=flow.source,
                        flow_domain=[float(flow.z[0]), float(flow.z[-1])],
                        interpolation=flow.interpolation,
                        arguments={k: str(v) if isinstance(v, Path) else v for k, v in vars(args).items()},
                        versions=dict(python=platform.python_version(), numpy=np.__version__, scipy=scipy.__version__),
                        convention="A(alpha)q=omega*Bq; raw complex alpha/omega; spatial growth=-Im(alpha)")
        success = True
        if args.command == "baseflow":
            flow.save(output(".csv"))
        else:
            problem = LocalProblem(flow, params)
            if args.command in ("temporal", "spatial"):
                spec = temporal(problem, args.alpha) if args.command == "temporal" else spatial(problem, args.omega)
                np.savez_compressed(output(".npz"), eigenvalues=spec.values, coefficients=spec.vectors,
                                    residuals=spec.residuals, z=problem.grid.z)
                np.savetxt(output(".csv"), np.column_stack((spec.values.real, spec.values.imag, spec.residuals)),
                           delimiter=",", header="real,imag,relative_residual")
                metadata["finite_modes"] = len(spec.values)
                if args.command == "temporal":
                    k = spec.select(args.target, args.imag_limit)
                    metadata["selected_omega"] = complex(spec.values[k])
                    metadata["selected_residual"] = float(spec.residuals[k])
            elif args.command == "saddle":
                result = saddle(problem, args.alpha, args.target, args.tol, args.max_iter,
                                args.max_step, args.imag_limit)
                metadata["saddle"] = asdict(result)
                metadata["pinch_verified"] = False
                success = result.converged
            else:
                if args.count < 2 or args.alpha_min >= args.alpha_max:
                    raise ValueError("scan needs count>=2 and alpha-min<alpha-max")
                path = np.linspace(args.alpha_min, args.alpha_max, args.count)
                rows = track_temporal(problem, path, args.target, args.imag_limit)
                np.savetxt(output(".csv"), [[a.real, a.imag, w.real, w.imag, e] for a, w, e in rows],
                           delimiter=",", header="alpha_real,alpha_imag,omega_real,omega_imag,relative_residual")
        metadata["elapsed_seconds"] = perf_counter()-start
        output(".json").write_text(json.dumps(metadata, default=_json, indent=2)+"\n")
        summary = {k: v for k, v in metadata.items() if k in ("finite_modes", "selected_omega", "selected_residual", "elapsed_seconds")}
        if args.command == "saddle":
            summary.update(alpha=result.alpha, omega=result.omega, converged=success,
                           group_velocity=result.group_velocity, iterations=len(result.history))
        print(json.dumps(summary, default=_json, indent=2))
        print(f"Saved {output('.json')}")
        return 0 if success else 2
    except (ValueError, RuntimeError) as exc:
        parser.exit(2, f"llsa: {exc}\n")
