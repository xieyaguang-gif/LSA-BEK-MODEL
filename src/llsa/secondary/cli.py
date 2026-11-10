"""Dedicated commands using Pier's inertial-frame conventions."""
import argparse
import json
from pathlib import Path
from time import perf_counter

import numpy as np

from .operators import PierOperators
from .primary import PrimaryWave, WaveSolver
from .floquet import FloquetProblem, secondary_pinch, secondary_saddle, secondary_spectrum
from .validation import compare_tables, primary_absolute_threshold
from .diagnostics import secondary_kinematic_defects
from .continuation import continue_ro
from .discretization import rotation_rates


def main(argv=None):
    if argv and argv[0] == 'pier-time':
        from .time_cli import main as time_main
        return time_main(argv[1:])
    parser = argparse.ArgumentParser(description="Pier 2003 nonlinear waves and secondary Floquet stability")
    parser.add_argument('command', choices=['pier-primary', 'pier-spectrum', 'pier-saddle',
                                          'pier-scan', 'pier-validate', 'pier-primary-neutral', 'pier-ro-scan'])
    parser.add_argument('--ro', type=float, help="BEK Rossby number; default: saved wave Ro or -1")
    radial = parser.add_mutually_exclusive_group()
    radial.add_argument('--radius', type=float, help="geometrical R=r*/sqrt(nu/Omega_BEK)")
    radial.add_argument('--re', type=float, help="positive Re=abs(Ro)*R")
    parser.add_argument('--beta', type=float)
    constraint = parser.add_mutually_exclusive_group()
    constraint.add_argument('--omega', type=float, help="inertial frequency in Omega_BEK units")
    constraint.add_argument('--alpha', type=float, help="prescribe alpha instead of omega for the nonlinear wave")
    constraint.add_argument('--energy', type=float, help="prescribe E1, solving alpha and omega")
    parser.add_argument('--n', type=int)
    parser.add_argument('--a', type=float)
    parser.add_argument('--b', type=float)
    parser.add_argument('--c', type=float)
    parser.add_argument('--harmonics', type=int, help="primary positive Fourier harmonics")
    parser.add_argument('--secondary-harmonics', type=int, default=5)
    parser.add_argument('--secondary-wall', choices=['pressure', 'continuity'], default='pressure',
                        help="secondary collocation wall closure; both require independent validation")
    parser.add_argument('--mean-mode', choices=['parallel', 'cylindrical'])
    parser.add_argument('--wave', type=Path, help="saved primary NPZ; source for secondary or seed for primary")
    parser.add_argument('--beta-hat', type=float, default=20)
    parser.add_argument('--alpha-hat', type=complex, default=-.0012-.0369j)
    parser.add_argument('--target', type=complex, default=8.52+1.16j)
    parser.add_argument('--count', type=int, default=6)
    parser.add_argument('--beta-hat-values', type=float, nargs='+', default=[20, 25, 30, 35, 40])
    parser.add_argument('--pinch', action='store_true')
    parser.add_argument('--ro-values', type=float, nargs='+', help="targets for pier-ro-scan")
    parser.add_argument('--ro-step', type=float, default=.025, help="maximum continuation Ro step")
    parser.add_argument('--fixed-re', action='store_true', help="pier-ro-scan: keep Re, varying geometrical R")
    parser.add_argument('--seed-alpha-bracket', type=float, nargs=2, default=[.5,.7])
    parser.add_argument('--seed-target', type=complex, default=65, help="target frequency for linear neutral seed")
    parser.add_argument('--output', type=Path, default=Path('results/pier2003/run'))
    args = parser.parse_args(argv)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    start = perf_counter()
    metadata = dict(command=args.command, convention="inertial frame; time=1/Omega_BEK; length=sqrt(nu/Omega_BEK); Re=abs(Ro)*R",
                    validation_status="experimental: secondary Table 3 discrepancy unresolved",
                    paper_doi="10.1017/S0022112003004981",
                    arguments={k: str(v) if isinstance(v, (Path, complex)) else v for k, v in vars(args).items()})

    def path(suffix):
        return Path(str(args.output)+suffix)

    try:
        source = PrimaryWave.load(args.wave) if args.wave else None
        given = dict(vars(args))
        args.ro = args.ro if args.ro is not None else (source.operators.ro if source else -1.)
        rotation_rates(args.ro)
        if args.ro == 0:
            raise ValueError("Ro=0 has zero differential rotation; finite-amplitude shear waves need nonzero Ro")
        defaults = dict(radius=510., beta=68., n=40, a=2., b=.6, c=.5, harmonics=5, mean_mode='cylindrical')
        if source:
            old = source.operators
            defaults = dict(radius=old.radius, beta=old.beta, n=old.m-1, a=old.grid.a,
                            b=old.grid.b, c=old.grid.c, harmonics=source.harmonics, mean_mode=old.mean_mode)
        for key, default in defaults.items():
            if getattr(args, key) is None:
                setattr(args, key, default)
        if args.re is not None:
            if not np.isfinite(args.re) or args.re <= 0:
                raise ValueError("--re must be positive and finite")
            args.radius = args.re/abs(args.ro)
        if source and args.command not in ('pier-primary', 'pier-ro-scan'):
            if args.ro != source.operators.ro or abs(args.radius-source.operators.radius)>1e-10:
                raise ValueError("Secondary analysis must use the saved wave Ro/R; generate or continue a new primary first")
            for key in ('beta','n','a','b','c','harmonics','mean_mode'):
                if given[key] is not None and getattr(args,key) != defaults[key]:
                    raise ValueError(f"--{key} conflicts with saved primary; rebuild it using pier-primary")
        if args.command == 'pier-ro-scan':
            if source is None or not args.ro_values:
                raise ValueError("pier-ro-scan requires --wave and --ro-values")
            if args.ro != source.operators.ro or any(getattr(args,k) != defaults[k] for k in defaults):
                raise ValueError("Use source model parameters for pier-ro-scan; change Ro with --ro-values")
            metadata['trajectory']=[]
            def checkpoint(wave,row):
                metadata['status']='running'
                metadata['trajectory'].append(row)
                metadata['checkpoint']=str(path('_latest_wave.npz'))
                wave.save(path('_latest_wave.npz'))
                path('.json').write_text(json.dumps(metadata,indent=2)+'\n')
                print(json.dumps(row),flush=True)
            try:
                waves, trajectory = continue_ro(source, args.ro_values, energy=args.energy, frequency=args.omega,
                    wavenumber=args.alpha, max_step=args.ro_step, fixed_re=args.fixed_re, callback=checkpoint)
            except (ValueError,RuntimeError,np.linalg.LinAlgError) as error:
                metadata.update(status='failed',error=str(error),elapsed_seconds=perf_counter()-start)
                path('.json').write_text(json.dumps(metadata,indent=2)+'\n')
                raise
            rows=[]
            for index, wave in enumerate(waves):
                saved=path(f'_ro_{index:03d}_wave.npz')
                wave.save(saved)
                rows.append(dict(wave=str(saved), **compare_tables(wave)))
            metadata.update(status='complete',results=rows, trajectory=trajectory, elapsed_seconds=perf_counter()-start,
                            constraint=dict(energy=args.energy if args.energy is not None else
                                (float(source.energies()[0]) if args.omega is args.alpha is None else None),
                                            frequency=args.omega,wavenumber=args.alpha),
                            fixed_re=args.fixed_re)
            path('.json').write_text(json.dumps(metadata,indent=2)+'\n')
            print(json.dumps([dict(ro=w.operators.ro,re=w.operators.re,radius=w.operators.radius,
                                  alpha=w.alpha,omega=w.omega,E1=float(w.energies()[0])) for w in waves],indent=2))
            print(f"Saved {path('.json')}")
            return 0
        if args.fixed_re or args.ro_values:
            raise ValueError("--fixed-re and --ro-values apply only to pier-ro-scan")
        if args.energy is not None:
            solve_constraint = dict(energy=args.energy)
        elif args.alpha is not None:
            solve_constraint = dict(wavenumber=args.alpha)
        else:
            if args.omega is None:
                args.omega = source.omega if source else 50.5
            solve_constraint = dict(frequency=args.omega)
        if args.command == 'pier-primary-neutral':
            if args.ro != -1:
                raise ValueError("pier-primary-neutral is the Ro=-1 Pier benchmark; use generalized spectra/continuation for other Ro")
            metadata['result'] = primary_absolute_threshold(args.n, args.a, args.b, args.c, args.beta)
        else:
            if source:
                wave = source
            else:
                op = PierOperators(args.radius, args.beta, args.n, args.a, args.b, args.c, args.mean_mode, ro=args.ro)
                solver = WaveSolver(op, 1)
                if args.energy is not None:
                    seed=solver.linear_seed(energy=min(.01,args.energy),bracket=args.seed_alpha_bracket,target=args.seed_target)
                    wave=solver.solve(seed,energy=min(.01,args.energy))
                    for energy in np.geomspace(min(.01,args.energy),args.energy,12):
                        wave=solver.solve(wave,energy=energy)
                else:
                    wave = solver.bootstrap(bracket=args.seed_alpha_bracket,target=args.seed_target)
                    wave = solver.solve(wave, **solve_constraint)
            if args.command == 'pier-primary' or not args.wave:
                if source and args.ro != source.operators.ro and args.radius == source.operators.radius:
                    wave = continue_ro(source,[args.ro],max_step=args.ro_step,**solve_constraint)[0][0]
                op = PierOperators(args.radius, args.beta, args.n, args.a, args.b, args.c, args.mean_mode, ro=args.ro)
                for h in range(1, args.harmonics+1):
                    solver = WaveSolver(op, h)
                    wave = solver.solve(wave, **solve_constraint)
                wave.save(path('_wave.npz'))
            metadata['primary'] = compare_tables(wave)
            if args.command == 'pier-primary':
                phases = np.linspace(0, 2*np.pi, 129)
                np.savez_compressed(path('_fields.npz'), phase=phases, z=wave.operators.grid.z,
                                    perturbation=wave.fields(phases), total=wave.fields(phases, total=True))
            elif args.command == 'pier-scan':
                results, alpha, target = [], args.alpha_hat, args.target
                for beta in args.beta_hat_values:
                    problem = FloquetProblem(wave, beta, args.secondary_harmonics, args.secondary_wall)
                    result = secondary_saddle(problem, alpha, target)
                    results.append([beta, result.alpha_hat.real, result.alpha_hat.imag,
                                    result.omega_hat.real, result.omega_hat.imag, abs(result.group_velocity),
                                    result.converged])
                    if not result.converged:
                        raise RuntimeError(f"Secondary saddle did not converge at beta_hat={beta}")
                    alpha, target = result.alpha_hat, result.omega_hat
                np.savetxt(path('.csv'), results, delimiter=',',
                           header='beta_hat,alpha_real,alpha_imag,omega_real,omega_imag,abs_group_velocity,converged')
                metadata['scan'] = results
            else:
                problem = FloquetProblem(wave, args.beta_hat, args.secondary_harmonics, args.secondary_wall)
                if args.command == 'pier-spectrum':
                    spec = secondary_spectrum(problem, args.alpha_hat, args.target, args.count)
                    np.savez_compressed(path('_spectrum.npz'), omega=spec.omega,
                                        vectors=spec.vectors, residuals=spec.residuals)
                    metadata['omega'] = [[v.real, v.imag] for v in spec.omega]
                else:
                    result = secondary_saddle(problem, args.alpha_hat, args.target)
                    metadata['kinematic_checks'] = secondary_kinematic_defects(
                        problem, result.alpha_hat, result.vector)
                    metadata['comparison'] = compare_tables(wave, (problem, result))
                    metadata['secondary'] = dict(alpha_hat=[result.alpha_hat.real, result.alpha_hat.imag],
                                                 omega_hat=[result.omega_hat.real, result.omega_hat.imag],
                                                 group_velocity=[result.group_velocity.real, result.group_velocity.imag],
                                                 converged=result.converged, history=result.history)
                    np.savez_compressed(path('_secondary.npz'), vector=result.vector,
                                        alpha_hat=result.alpha_hat, omega_hat=result.omega_hat,
                                        z=wave.operators.grid.z, harmonics=problem.K)
                    if args.pinch:
                        metadata['pinch_diagnostic'] = secondary_pinch(problem, result)
        metadata['elapsed_seconds'] = perf_counter()-start
        path('.json').write_text(json.dumps(metadata, indent=2)+'\n')
        summary = {k: v for k, v in metadata.items() if k in (
            'command', 'validation_status', 'result', 'omega', 'elapsed_seconds')}
        if 'primary' in metadata:
            summary['primary'] = {k: metadata['primary'][k] for k in (
                'ro', 're', 'radius', 'beta', 'n', 'H', 'alpha', 'omega', 'energies', 'mean_mode', 'nonlinear_residual',
                'paper_mean_constraints')}
        if 'secondary' in metadata:
            summary['secondary'] = {k: v for k, v in metadata['secondary'].items() if k != 'history'}
        if 'comparison' in metadata:
            summary['comparison'] = {k: metadata['comparison'][k] for k in ('table2', 'table3')}
        if 'pinch_diagnostic' in metadata:
            summary['opposite_half_planes_at_end'] = metadata['pinch_diagnostic']['opposite_half_planes_at_end']
        if 'kinematic_checks' in metadata:
            summary['kinematic_checks'] = metadata['kinematic_checks']
        print(json.dumps(summary, indent=2))
        print(f"Saved {path('.json')}")
        if args.command == 'pier-validate':
            passed = (all(metadata['comparison'][t]['passed'] is True for t in ('table2', 'table3'))
                      and metadata['kinematic_checks']['wall_continuity_passed']
                      and not metadata['primary']['paper_mean_constraints']['known_mismatch'])
            return 0 if passed else 2
        if 'secondary' in metadata and not metadata['secondary']['converged']:
            return 2
        return 0
    except (ValueError, RuntimeError, np.linalg.LinAlgError) as exc:
        parser.exit(2, f"pier: {exc}\n")
