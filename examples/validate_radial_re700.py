"""Long-domain validation driver; successful algebraic solves are not physical validation.

Use PYTHONPATH=src, OPENBLAS_NUM_THREADS=1. PSE fields supply a Newton initial
guess only; the new full radial equations determine the converged solution.
"""
import argparse
from dataclasses import asdict
import json
from pathlib import Path
from time import perf_counter

import numpy as np
from scipy.integrate import cumulative_trapezoid
from scipy.interpolate import BarycentricInterpolator, CubicSpline

from llsa.secondary.radial import RadialConfig
from llsa.secondary.global_radial import GlobalRadial, OutletSponge
from llsa.secondary.oneway import SpectralOneWay, causal_projector


def pse_seed(path, radii, config, inlet, match_tail_phase=False, seed_stop_re=None,
             tail_decay_length=50.):
    """Restore carrier phase; interpolate slow amplitudes and the z polynomial."""
    op = config.operators(radii[0])
    with np.load(path) as data:
        if 'config_json' in data:
            old_config = RadialConfig(**json.loads(str(data['config_json'])))
            old_r = data['radii']
            old_H = old_config.harmonics
            old_modes = data['modes']
            old_op = old_config.operators(old_r[0])
            if seed_stop_re is not None:
                keep = old_r <= seed_stop_re/abs(config.ro)+1e-10
                if np.count_nonzero(keep) < 10:
                    raise ValueError('seed-stop leaves fewer than 10 radial points')
                old_r, old_modes = old_r[keep], old_modes[keep]
            v = old_modes[:, old_H+1, :3*old_op.m].reshape(len(old_r), 3, old_op.m)
            overlaps = np.einsum('rvi,ij,rvj->r', v[:-1].conj(), old_op.grid.mass, v[1:])
            tail_phase_rate = float(np.median(np.angle(overlaps[-8:])/np.diff(old_r)[-8:]))
            # Remove a constant carrier before interpolation; this is only
            # initial-guess interpolation, not a change to the solved equations.
            ns = np.arange(-old_H, old_H+1)
            old_modes = old_modes*np.exp(-1j*(old_r-old_r[0])[:, None, None]*ns[None, :, None]*.35)
            old_alpha = np.full(len(old_r), .35+0j)
            old_xi = old_config.operators(old_r[0]).grid.xi
        else:
            old_r, old_modes, old_alpha = data['radius'], data['modes'], data['alpha']
            old_H = int(data['harmonics'])
            old_xi = data['xi']
            if seed_stop_re is not None:
                keep = old_r <= seed_stop_re/abs(config.ro)+1e-10
                if np.count_nonzero(keep) < 10:
                    raise ValueError('seed-stop leaves fewer than 10 radial points')
                old_r, old_modes, old_alpha = old_r[keep], old_modes[keep], old_alpha[keep]
            tail_phase_rate = old_alpha[-1].real
        remapped = BarycentricInterpolator(old_xi, old_modes.reshape(len(old_r), 2*old_H+1, 4, -1), axis=3)(op.grid.xi)
    H = config.harmonics
    slow = np.zeros((len(old_r), 2*H+1, op.size), complex)
    for j in range(-min(H, old_H), min(H, old_H)+1):
        slow[:, H+j] = remapped[:, old_H+j].reshape(len(old_r), op.size)
    # Align the complex phase and magnitude with the common full-radial inlet.
    a, b = slow[0, H+1, :3*op.m], inlet[H+1, :3*op.m]
    factor = np.vdot(a, b)/np.vdot(a, a)
    for j in range(-H, H+1):
        slow[:, H+j] *= abs(factor)**max(abs(j), 2 if j == 0 else 1)*np.exp(1j*j*np.angle(factor))
    phase = cumulative_trapezoid(old_alpha.real, old_r, initial=0)
    clamped = np.clip(radii, old_r[0], old_r[-1])
    values = CubicSpline(old_r, slow, axis=0)(clamped)
    phi = CubicSpline(old_r, phase)(clamped)+np.maximum(radii-old_r[-1], 0)*(
        tail_phase_rate if match_tail_phase else old_alpha[-1].real)
    for j in range(-H, H+1):
        decay = 0 if tail_decay_length == 0 else np.maximum(radii-old_r[-1], 0)*max(abs(j), 1)/tail_decay_length
        values[:, H+j] *= np.exp(1j*j*phi-decay)[:, None]
    # Only enforce prescribed inlet velocities. Pressure and the free mean
    # axial far-field value are solved variables, not extra inlet conditions.
    rows = np.concatenate([v*op.m+np.arange(1, op.m-1) for v in range(3)])
    values[0][:, rows] = inlet[:, rows]
    return values


def main():
    root = Path(__file__).resolve().parents[1]
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--method', choices=['global', 'oneway'], default='global')
    p.add_argument('--end', type=float, default=700.)
    p.add_argument('--n', type=int, default=24)
    p.add_argument('--harmonics', type=int, default=3)
    p.add_argument('--linear', action='store_true',
                   help='Disable harmonic convection; useful for a compatible global Newton seed')
    p.add_argument('--dr', type=float, default=1.)
    p.add_argument('--amplitude', type=float, default=.00016377687906989752)
    p.add_argument('--seed', type=Path, default=root/'results/beta42_nonlinear_spatial/primary/fields.npz')
    p.add_argument('--output', type=Path, required=True)
    p.add_argument('--tol', type=float, default=1e-9)
    p.add_argument('--max-iter', type=int, default=30)
    p.add_argument('--warm-mean', action='store_true', help='Use PSE mean as initial outer iterate only')
    p.add_argument('--outer-relaxation', type=float, default=1.)
    p.add_argument('--advection-derivative', choices=['auxiliary', 'bdf'], default='auxiliary')
    p.add_argument('--outlet-modal', action='store_true', help='Suppress all local finite upstream modes at the outlet')
    p.add_argument('--outlet-alpha', type=complex)
    p.add_argument('--linear-solver', choices=['direct', 'harmonic-gmres', 'pardiso'], default='direct')
    p.add_argument('--phase-update', action='store_true',
                   help='Split Newton corrections into radial phase and amplitude directions')
    p.add_argument('--phase-continuity', action='store_true',
                   help='Project phase-updated trial states back onto interior continuity')
    p.add_argument('--merit', choices=['inf', 'scaled-l2'], default='inf',
                   help='Newton line-search merit norm')
    p.add_argument('--match-tail-phase', action='store_true', help='Extrapolate a global seed using its measured terminal phase rate')
    p.add_argument('--seed-stop', type=float, help='Ignore seed data after this Re (e.g. its sponge start)')
    p.add_argument('--seed-tail-decay', type=float, default=50.,
                   help='Extrapolation decay length in R; zero disables artificial tail decay')
    p.add_argument('--sponge-start', type=float, help='Sponge start in Re; disabled when omitted')
    p.add_argument('--sponge-width', type=float, default=40., help='Sponge width in Re')
    p.add_argument('--sponge-strength', type=float, default=10.)
    p.add_argument('--sponge-damp-mean', action='store_true')
    args = p.parse_args()
    if args.output.exists():
        p.error('Choose a new output directory')
    if args.seed_tail_decay < 0:
        p.error('seed-tail-decay must be nonnegative')
    if args.sponge_start is not None and (args.sponge_start <= 320
            or args.sponge_width <= 0 or args.sponge_strength < 0
            or args.sponge_start+args.sponge_width > args.end+1e-12):
        p.error('Sponge must start after Re=320 and finish by --end')
    args.output.mkdir(parents=True)
    config = RadialConfig(n=args.n, harmonics=args.harmonics, nonlinear=not args.linear)
    nr = int(np.ceil((args.end-320)/abs(config.ro)/args.dr))+1
    radii = np.linspace(320/abs(config.ro), args.end/abs(config.ro), nr)
    report = dict(config=asdict(config), arguments={k:str(v) if isinstance(v, (Path, complex)) else v for k,v in vars(args).items()},
                  points=nr, step_r=float(radii[1]-radii[0]), status='running', validation_status='unvalidated', newton_history=[])
    started = perf_counter()
    def save():
        report['elapsed_seconds'] = perf_counter()-started
        (args.output/'metadata.json').write_text(json.dumps(report, indent=2)+'\n')
    save()
    try:
        march = SpectralOneWay(config, advection_derivative=args.advection_derivative)
        inlet, _, alpha = march.initialize(radii[0], args.amplitude, 1.1858025795710305)
        s = config.operators(radii[0]).size
        inlet = inlet[:, :s]
        report['inlet_alpha'] = [alpha.real, alpha.imag]
        if args.method == 'oneway':
            initial_mean = pse_seed(args.seed, radii, config, inlet)[:, config.harmonics] if args.warm_mean else None
            def progress(event, data):
                report['last_progress'] = dict(event=event, **data); save()
                print(json.dumps(report['last_progress']), flush=True)
            result, diagnostics = march.march(radii, amplitude=args.amplitude, z=1.1858025795710305,
                tol=args.tol, max_iter=args.max_iter, initial_mean=initial_mean,
                outer_relaxation=args.outer_relaxation, progress=progress)
            report['diagnostics'] = diagnostics
        else:
            proj = causal_projector(config.operators(radii[-1]), 1, config.frequency(config.operators(radii[-1])))
            valid = np.flatnonzero(~proj.upstream)
            ix = valid[np.argmin(abs(-1j*proj.values[valid]-alpha))]
            alpha_out = -1j*proj.values[ix] if args.outlet_alpha is None else args.outlet_alpha
            report['outlet_alpha'] = [alpha_out.real, alpha_out.imag]
            constraints = None
            if args.outlet_modal:
                constraints = {1: proj.upstream_constraints}
                for j in range(2, config.harmonics+1):
                    p = causal_projector(config.operators(radii[-1]), j, config.frequency(config.operators(radii[-1])))
                    constraints[j] = p.upstream_constraints
                report['outlet_type'] = 'local full finite-spectrum radiation; frozen laminar base'
            sponge = None if args.sponge_start is None else OutletSponge(
                args.sponge_start/abs(config.ro), args.sponge_width/abs(config.ro),
                args.sponge_strength, args.sponge_damp_mean)
            if sponge is not None:
                report['outlet_sponge'] = dict(
                    profile='companion Semtex DNS C-infinity transition',
                    start_Re=args.sponge_start, width_Re=args.sponge_width,
                    strength=args.sponge_strength, damp_mean=args.sponge_damp_mean,
                    interpretation_limit_Re=args.sponge_start)
            problem = GlobalRadial(radii, config, outlet_alpha=alpha_out,
                                   outlet_constraints=constraints, outlet_sponge=sponge)
            guess = pse_seed(args.seed, radii, config, inlet, args.match_tail_phase,
                             args.seed_stop, args.seed_tail_decay)
            def progress(it, residual):
                report['newton_history'].append(float(residual)); save()
                print(f'{args.method} end={args.end} n={args.n} H={args.harmonics} Newton {it}: {residual:.6g}', flush=True)
            result = problem.solve(inlet, initial=guess, tol=args.tol, max_iter=args.max_iter,
                                   progress=progress, linear_solver=args.linear_solver,
                                   phase_update=args.phase_update,
                                   phase_continuity=args.phase_continuity,
                                   merit=args.merit)
        energy = result.energies(config)
        np.savez_compressed(args.output/'fields.npz', radii=radii, modes=result.modes, config_json=json.dumps(asdict(config)))
        np.savetxt(args.output/'energy.csv', np.column_stack((abs(config.ro)*radii, energy)), delimiter=',',
            header='Re,'+','.join(f'E{j}' for j in range(1, config.harmonics+1)))
        report.update(status='converged_discrete_equations', relative_residual=result.residual,
                      final_energies=energy[-1].tolist(), maximum_E1=float(max(energy[:,0])),
                      maximum_E1_Re=float(abs(config.ro)*radii[np.argmax(energy[:,0])]))
    except (RuntimeError, ValueError, np.linalg.LinAlgError) as exc:
        report.update(status='failed', error=str(exc))
    save()
    print(json.dumps({k:v for k,v in report.items() if k not in ('diagnostics','newton_history')}, indent=2), flush=True)
    return 0 if report['status']=='converged_discrete_equations' else 2


if __name__ == '__main__':
    raise SystemExit(main())
