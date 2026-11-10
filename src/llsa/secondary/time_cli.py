"""Time-step and spatial checks with random initial data, then compare eigenvalues.

All times/frequencies use the BEK reference rotation (disk units at Ro=-1). The complex-alpha run is a
weighted-envelope check; it does not replace a localized impulse/pinch test.
"""
import argparse
import hashlib
import json
from pathlib import Path
from time import perf_counter

import numpy as np

from llsa.secondary import PrimaryWave
from llsa.secondary.time_domain import LinearTimeProblem, advance, fit_growth, assess_time_convergence, interpolate_primary


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--wave', type=Path, required=True)
    parser.add_argument('--ro', type=float, help="optional consistency check; actual Ro is loaded from --wave")
    parser.add_argument('--alpha-hat', type=complex, default=0.)
    parser.add_argument('--beta-hat', type=float, default=20.)
    parser.add_argument('--harmonics', type=int, default=5)
    parser.add_argument('--n', type=int, default=64)
    parser.add_argument('--zmax', type=float, default=320.)
    parser.add_argument('--scale', type=float, default=5.)
    parser.add_argument('--dt', type=float, nargs='+', default=[.001, .0005, .00025])
    parser.add_argument('--end-time', type=float, default=8.)
    parser.add_argument('--fit-start', type=float, nargs='+', default=None)
    parser.add_argument('--seed', type=int, default=7)
    parser.add_argument('--amplitude', type=float, default=1e-12)
    parser.add_argument('--no-spectrum', action='store_true', help="run with no eigenvalue solver at all")
    parser.add_argument('--output', type=Path, default=Path('results/pier_time/temporal'))
    args = parser.parse_args(argv)
    if not np.all(np.isfinite(args.dt)) or min(args.dt) <= 0 or np.any(np.diff(args.dt) >= 0):
        parser.error('Use positive, strictly decreasing time steps')
    args.output.mkdir(parents=True, exist_ok=True)
    wave = PrimaryWave.load(args.wave)
    if args.ro is not None and args.ro != wave.operators.ro:
        parser.error("--ro conflicts with saved primary; continue the primary before time integration")
    problem = LinearTimeProblem(wave, args.alpha_hat, args.beta_hat, args.harmonics,
                                args.n, args.zmax, args.scale)
    initial = problem.random_state(args.seed, args.amplitude)
    starts = args.fit_start or [args.end_time*.5, args.end_time*.75]
    if any(s < 0 or s >= args.end_time for s in starts):
        parser.error('Fitting windows must start inside the time interval')
    results = dict(
        method='physical-phase RHS; pressure elimination; CN/AB2; random initial data',
        convention='inertial frequency in Omega_BEK units; sigma=0.5*d(log E)/dt; disk units only at Ro=-1',
        interpretation='temporal Bloch mode' if args.alpha_hat.imag == 0 else 'spatially weighted envelope',
        independence='No Floquet pencil or eigenvalue enters the RHS, stepping, initial data, or fitting; primary and laminar BVP are shared',
        primary=dict(file=str(args.wave), sha256=hashlib.sha256(args.wave.read_bytes()).hexdigest(),
                     source_n=wave.operators.m-1, ro=wave.operators.ro,re=wave.operators.re,
                     disk_rotation=wave.operators.disk_rotation,external_rotation=wave.operators.external_rotation,
                     radius=problem.R, alpha=wave.alpha,
                     beta=wave.operators.beta, omega=wave.omega, H=wave.harmonics,
                     mean_mode=wave.operators.mean_mode),
        parameters={k: str(v) if isinstance(v, (Path, complex)) else v for k, v in vars(args).items()},
        runs=[])
    for index, dt in enumerate(args.dt):
        start = perf_counter()
        trace = advance(problem, initial, dt, args.end_time, sample_every=max(1, round(.01/dt)))
        fits = [fit_growth(trace, t) for t in starts]
        path = args.output/f'trace_{index}.npz'
        np.savez_compressed(path, **{k: v for k, v in trace.items() if k != 'constraints'}, z=problem.grid.z)
        row = dict(dt=trace['dt'], fits=fits, constraints=trace['constraints'],
                   final_shape_overlap=float(trace['shape_overlap'][-1]),
                   elapsed_seconds=perf_counter()-start, trace=path.name)
        results['runs'].append(row)
        print(json.dumps(row), flush=True)
        (args.output/'comparison.json').write_text(json.dumps(results, indent=2)+'\n')
    # Deliberately defer this import and calculation until time integration and
    # fitting have completed, so it cannot supply the mode or fitted frequency.
    if not args.no_spectrum:
        from llsa.secondary import FloquetProblem, secondary_spectrum
        measured = results['runs'][-1]['fits'][-1]
        target = measured['omega_real']+1j*measured['growth']
        reference = FloquetProblem(wave, args.beta_hat, args.harmonics, wall_condition='continuity')
        spectrum = secondary_spectrum(reference, args.alpha_hat, target=target+1e-4j, count=6)
        omega = complex(spectrum.omega[spectrum.nearest(target)])
        eigen_wave = PrimaryWave(wave.operators, wave.alpha, wave.omega,
                                 spectrum.vectors[:, spectrum.nearest(target)].reshape(2*args.harmonics+1, -1),
                                 0., [])
        spectral_velocity = interpolate_primary(eigen_wave, problem.grid.z)[:, :3]*problem.R
        time_velocity = problem.velocity(trace['final_state'])
        dot = np.einsum('hvi,ij,hvj->', spectral_velocity.conj(), problem.grid.mass, time_velocity)
        norm1 = np.einsum('hvi,ij,hvj->', spectral_velocity.conj(), problem.grid.mass, spectral_velocity).real
        norm2 = np.einsum('hvi,ij,hvj->', time_velocity.conj(), problem.grid.mass, time_velocity).real
        results['spectral_comparison'] = dict(
            omega_hat=[omega.real, omega.imag],
            velocity_shape_overlap=float(abs(dot)/np.sqrt(norm1*norm2)),
            note='Existing semi-infinite source grid and continuity wall closure; shared primary',
            growth_difference=measured['growth']-omega.imag,
            frequency_difference=measured['omega_real']-omega.real)
    results['published_table3_verified'] = False
    results['time_convergence'] = assess_time_convergence(results['runs'])
    (args.output/'comparison.json').write_text(json.dumps(results, indent=2)+'\n')
    print(f"Saved {args.output/'comparison.json'}")
    return 2 if results['time_convergence']['passed'] is False else 0


if __name__ == '__main__':
    raise SystemExit(main())
