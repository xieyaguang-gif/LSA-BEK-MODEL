"""Separate entry point for the experimental two-layer radial framework."""
import argparse
from dataclasses import asdict
import json
from pathlib import Path
from time import perf_counter

import numpy as np

from .global_radial import GlobalRadial, OutletSponge
from .oneway import SpectralOneWay, causal_projector
from .radial import RadialConfig


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--method', choices=['global', 'oneway', 'compare'], default='compare')
    parser.add_argument('--ro', type=float, default=-.751)
    parser.add_argument('--beta', type=float, default=42.)
    parser.add_argument('--omega', type=float, help='Inertial frequency; default beta*Omega_disk/Omega_BEK')
    parser.add_argument('--re-start', type=float, default=320.)
    parser.add_argument('--re-end', type=float, default=326.)
    parser.add_argument('--points', type=int, default=33)
    parser.add_argument('--n', type=int, default=24)
    parser.add_argument('--harmonics', type=int, default=2)
    parser.add_argument('--amplitude', type=float, default=1e-5, help='sqrt(2)*abs(v1)/disk rotation at probe')
    parser.add_argument('--probe-z', type=float, default=1.1858)
    parser.add_argument('--target', type=complex, default=.4-.07j)
    parser.add_argument('--outlet-alpha', type=complex, help='Global Robin alpha; default local downstream eigenmode')
    parser.add_argument('--sponge-start', type=float,
                        help='Outlet sponge start in Re; disabled when omitted')
    parser.add_argument('--sponge-width', type=float, default=40., help='Outlet sponge width in Re')
    parser.add_argument('--sponge-strength', type=float, default=10., help='Maximum sponge damping')
    parser.add_argument('--sponge-damp-mean', action='store_true',
                        help='Also damp the nonlinear mean correction (normally leave it undamped)')
    parser.add_argument('--continuation-steps', type=int, default=64)
    parser.add_argument('--linear', action='store_true')
    parser.add_argument('--tol', type=float, default=1e-9)
    parser.add_argument('--max-iter', type=int, default=30)
    parser.add_argument('--output', type=Path, default=Path('results/radial_two_layer'))
    args = parser.parse_args(argv)
    if args.points < 5 or args.re_start <= 0 or args.re_end <= args.re_start:
        parser.error('Require points>=5 and 0<re-start<re-end')
    if args.probe_z <= 0 or args.amplitude <= 0 or args.continuation_steps < 8 or args.tol <= 0:
        parser.error('Positive probe-z/amplitude/tol and continuation-steps>=8 required')
    if args.sponge_start is not None and (args.sponge_start <= args.re_start
            or args.sponge_width <= 0 or args.sponge_strength < 0
            or args.sponge_start+args.sponge_width > args.re_end+1e-12):
        parser.error('Sponge must start after re-start, have positive width/nonnegative strength, and finish by re-end')
    config = RadialConfig(ro=args.ro, beta=args.beta, n=args.n, harmonics=args.harmonics,
                          omega=args.omega, nonlinear=not args.linear)
    radii = np.linspace(args.re_start, args.re_end, args.points)/abs(config.ro)
    args.output.mkdir(parents=True, exist_ok=True)
    # Refuse an accidental overwrite of an earlier result set.
    if (args.output/'metadata.json').exists():
        parser.error('Output already contains metadata.json; choose a new output directory')
    start = perf_counter()
    metadata = dict(schema_version=1, status='running', validation_status='experimental',
        config=asdict(config), arguments={k: str(v) if isinstance(v, (Path, complex)) else v for k, v in vars(args).items()},
        convention='Re=abs(Ro)*R; physical velocity=R*q; pressure=R^2*p; exp(i*j*(beta*theta-omega*t))',
        inlet='same downstream local full-radial eigenmode; mean/higher-harmonic inlet velocity zero; mean pressure solved',
        global_outlet='smooth DNS-form sponge followed by velocity Robin when enabled; physical data end at sponge start',
        solutions={})
    solutions = {}
    def save():
        metadata['elapsed_seconds'] = perf_counter()-start
        (args.output/'metadata.json').write_text(json.dumps(metadata, indent=2)+'\n')
    def save_solution(method, solution):
        energies = solution.energies(config)
        np.savez_compressed(args.output/f'{method}.npz', radii=solution.radii, modes=solution.modes,
            energies=energies, z=config.operators(radii[0]).grid.z, config_json=json.dumps(asdict(config)),
            method=solution.method)
        # Positive harmonic energies; the free mean w(infinity) is not integrable.
        np.savetxt(args.output/f'{method}.csv', np.column_stack((abs(config.ro)*radii, radii, energies)),
            delimiter=',', header='Re,R,'+','.join(f'E{j}' for j in range(1, config.harmonics+1)))
        metadata['solutions'][method] = dict(relative_algebraic_residual=solution.residual,
            newton_iterations=solution.iterations, residual_history=solution.history,
            energy_final=energies[-1].tolist())
        save()
    try:
        marcher = SpectralOneWay(config, continuation_steps=args.continuation_steps)
        inlet, projectors, alpha = marcher.initialize(radii[0], args.amplitude, args.probe_z, args.target)
        s = inlet.shape[1]//2
        metadata['inlet_alpha'] = [alpha.real, alpha.imag]
        metadata['inlet_projectors'] = {str(j): p.diagnostics for j, p in projectors.items()}
        if args.method in ('oneway', 'compare'):
            solution, diagnostics = marcher.march(radii, amplitude=args.amplitude, z=args.probe_z,
                target=args.target, tol=args.tol, max_iter=args.max_iter)
            solutions['oneway'] = solution
            metadata['oneway_diagnostics'] = diagnostics
            save_solution('oneway', solution)
        if args.method in ('global', 'compare'):
            outlet_alpha = args.outlet_alpha
            if outlet_alpha is None:
                p = causal_projector(config.operators(radii[-1]), 1, config.frequency(config.operators(radii[-1])),
                                     continuation_steps=args.continuation_steps)
                ix = np.flatnonzero(~p.upstream)
                selected = ix[np.argmin(abs(-1j*p.values[ix]-alpha))]
                outlet_alpha = -1j*p.values[selected]
            metadata['global_outlet_alpha'] = [outlet_alpha.real, outlet_alpha.imag]
            sponge = None if args.sponge_start is None else OutletSponge(
                args.sponge_start/abs(config.ro), args.sponge_width/abs(config.ro),
                args.sponge_strength, args.sponge_damp_mean)
            if sponge is not None:
                metadata['outlet_sponge'] = dict(
                    profile='C-infinity transition used by companion Semtex DNS',
                    start_Re=args.sponge_start, width_Re=args.sponge_width,
                    strength=args.sponge_strength, damp_mean=args.sponge_damp_mean,
                    interpretation_limit_Re=args.sponge_start)
            problem = GlobalRadial(radii, config, outlet_alpha=outlet_alpha, outlet_sponge=sponge)
            initial = solutions['oneway'].modes if 'oneway' in solutions else np.array([
                inlet[:, :s]*np.exp((1j*np.arange(-config.harmonics, config.harmonics+1)*alpha.real
                   -abs(np.arange(-config.harmonics, config.harmonics+1))*alpha.imag)[:, None]*(r-radii[0]))
                for r in radii])
            solutions['global'] = problem.solve(inlet[:, :s], initial=initial, tol=args.tol, max_iter=args.max_iter)
            save_solution('global', solutions['global'])
        metadata['status'] = 'converged_discrete_equations'
        save()
    except (ValueError, RuntimeError, np.linalg.LinAlgError) as exc:
        metadata.update(status='failed', error=str(exc))
        save()
        print(json.dumps(dict(status='failed', error=str(exc)), indent=2))
        return 2
    summary = {method: {key: value for key, value in data.items() if key != 'residual_history'}
               for method, data in metadata['solutions'].items()}
    print(json.dumps(dict(status=metadata['status'], validation_status='experimental',
                         solutions=summary, output=str(args.output)), indent=2))
    return 0
