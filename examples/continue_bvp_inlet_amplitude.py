"""Bounded continuation of unforced full BVPs in log inlet amplitude.

Every accepted station solves the original radial equations (forcing=None).
The source is a converged BVP; amplitudes may increase or decrease. The source
radial grid remains fixed and only the target field's inlet is used. Its outlet
condition is first corrected to the explicitly selected condition if necessary.
An LU factor is reused only as a GMRES preconditioner; Newton still uses the
current Jacobian and the original unforced residual controls acceptance.
"""
from __future__ import annotations

import argparse
from dataclasses import asdict
import json
from pathlib import Path
from time import perf_counter

import numpy as np

from llsa.secondary.global_radial import GlobalRadial, OutletSponge
from llsa.secondary.inlet_amplitude import InletAmplitudePath
from llsa.secondary.oneway import SpectralOneWay, causal_projector
from llsa.secondary.pardiso_backend import PardisoBackend
from llsa.secondary.radial import pack_field
from llsa.secondary.radial_continuation import load_seed, secant_seed, write_json


class StopBudget(Exception):
    pass


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--seed', type=Path, required=True)
    p.add_argument('--target-inlet-field', type=Path, required=True,
                   help='Only inlet data are read; its interior may be an auxiliary-forcing checkpoint')
    p.add_argument('--source-af', type=float, default=.01)
    p.add_argument('--target-af', type=float, default=.001)
    p.add_argument('--fit-inlet-ratio', action='store_true',
                   help='Use the measured inlet velocity ratio; AF values remain nominal labels')
    p.add_argument('--outlet-alpha', type=complex,
                   help='Explicit fixed outlet alpha; otherwise reconstruct the legacy radiation condition')
    p.add_argument('--sponge-start', type=float, default=480.)
    p.add_argument('--width', type=float, default=40.)
    p.add_argument('--strength', type=float, default=10.)
    p.add_argument('--log-step', type=float, default=.5)
    p.add_argument('--minimum-log-step', type=float, default=.125)
    p.add_argument('--max-iter', type=int, default=10)
    p.add_argument('--max-attempts', type=int, default=10)
    p.add_argument('--seconds', type=float, default=6600.)
    p.add_argument('--tol', type=float, default=1e-9)
    p.add_argument('--preconditioning-side', choices=['left', 'right'], default='left')
    p.add_argument('--preconditioner-refresh', type=int, default=3)
    p.add_argument('--gmres-restart', type=int, default=40)
    p.add_argument('--gmres-maxiter', type=int, default=3)
    p.add_argument('--cartesian-fallback', action='store_true')
    p.add_argument('--predictor', choices=['identity', 'phase', 'best'], default='phase',
                   help='Best chooses identity or phase secant by the actual equation residual')
    p.add_argument('--output', type=Path, required=True)
    a = p.parse_args()
    if (not np.all(np.isfinite([a.target_af, a.source_af, a.log_step, a.minimum_log_step,
                               a.seconds, a.tol]))
            or min(a.target_af, a.source_af) <= 0 or a.target_af == a.source_af
            or not 0 < a.minimum_log_step <= a.log_step
            or min(a.seconds, a.max_attempts, a.max_iter, a.tol, a.preconditioner_refresh,
                   a.gmres_restart, a.gmres_maxiter) <= 0):
        p.error('Require distinct positive amplitudes and positive continuation limits')
    if a.output.exists():
        p.error('Choose a new output directory')
    cfg, r, current = load_seed(a.seed)
    tc, tr, tq = load_seed(a.target_inlet_field)
    if cfg != tc or not np.isclose(r[0], tr[0], rtol=0, atol=1e-10):
        p.error('Source and target must have the same configuration and inlet radius')
    if not np.isclose(abs(cfg.ro)*r[-1], a.sponge_start+a.width, atol=1e-8):
        p.error('Source grid must end at the end of the requested sponge')
    a.output.mkdir(parents=True)
    started = perf_counter()
    summary = dict(status='initializing', method='unforced_log_inlet_continuation',
        config=asdict(cfg), arguments={k: str(v) if isinstance(v, (Path, complex)) else v for k, v in vars(a).items()},
        validation_status='discrete_equations_only',
        auxiliary_forcing_fraction=0., last_converged_af=None, attempts=[], stations=[])
    def save_summary():
        summary['elapsed_seconds'] = perf_counter()-started
        write_json(a.output/'summary.json', summary)
    save_summary()
    alpha_out = a.outlet_alpha
    if alpha_out is None:
        # Legacy default retained for the earlier decreasing-amplitude runs.
        _, _, alpha = SpectralOneWay(cfg).initialize(r[0], 1e-6, 1.1858025795710305)
        proj = causal_projector(cfg.operators(r[-1]), 1, cfg.frequency(cfg.operators(r[-1])))
        valid = np.flatnonzero(~proj.upstream)
        ix = valid[np.argmin(abs(-1j*proj.values[valid]-alpha))]
        alpha_out = -1j*proj.values[ix]
    problem = GlobalRadial(r, cfg, outlet_alpha=alpha_out,
        outlet_sponge=OutletSponge(a.sponge_start/abs(cfg.ro), a.width/abs(cfg.ro), a.strength, False))
    op, H = problem.ops[0], cfg.harmonics
    source_inlet, target_inlet = current[0].copy(), tq[0].copy()
    rows = problem.velocity_rows
    path = InletAmplitudePath.fit(source_inlet, target_inlet, H+1, rows,
        expected_ratio=None if a.fit_inlet_ratio else a.target_af/a.source_af)
    if np.sign(path.end) != np.sign(np.log(a.target_af/a.source_af)):
        raise ValueError('Measured amplitude and nominal AF changes must have the same direction')
    summary.update(outlet_alpha=[alpha_out.real, alpha_out.imag], target_inlet_fit_error=path.fit_error,
                   inlet_amplitude_ratio=path.ratio, nominal_af_ratio=a.target_af/a.source_af,
                   status='running', sponge_start=a.sponge_start, computational_end=float(abs(cfg.ro)*r[-1]))
    target = path.end
    coordinate, step, previous = 0., a.log_step, None
    source_error = np.linalg.norm(problem.evaluate(pack_field(current), source_inlet), np.inf)
    source_error /= max(np.max(abs(current)), np.max(abs(source_inlet)), 1e-30)
    summary['source_residual_under_target_outlet'] = float(source_error)
    revalidate = bool(source_error > a.tol)
    if not revalidate:
        summary['last_converged_af'] = a.source_af
    velocity = current[:, H+1, :3*op.m].reshape(len(r), 3, op.m)
    energy = r*r*np.einsum('rvi,ij,rvj->r', velocity.conj(), op.grid.mass, velocity).real
    eligible = (r > r[0]) & (abs(cfg.ro)*r <= a.sponge_start)
    pin = float(r[np.argmax(np.where(eligible, energy, -np.inf))])
    save_summary()
    try:
        with PardisoBackend() as backend:
            summary['threads'] = backend.threads
            if backend.threads > 32:
                raise ValueError('This bounded test requires at most 32 MKL threads')
            while revalidate or abs(coordinate-target) > 1e-12:
                if perf_counter()-started >= a.seconds:
                    raise StopBudget('Wall-clock budget reached')
                if len(summary['attempts']) >= a.max_attempts:
                    raise StopBudget('Attempt limit reached')
                trial = coordinate if revalidate else path.advance(coordinate, step)
                guess = current.copy()
                inlet = path.at(trial)
                predictor = 'identity'
                candidate_residuals = {}
                if previous is not None and not revalidate and a.predictor != 'identity':
                    old_coordinate, old_field = previous
                    predicted = secant_seed(current, old_field, (trial-coordinate)/(coordinate-old_coordinate),
                                        cfg, r[0], phase_aligned=True)
                    if a.predictor == 'best':
                        candidate_residuals = {name: float(np.linalg.norm(
                            problem.evaluate(pack_field(field), inlet), np.inf))
                            for name, field in [('identity', guess), ('phase', predicted)]}
                        predictor = min(candidate_residuals, key=candidate_residuals.get)
                    else:
                        predictor = 'phase'
                    if predictor == 'phase':
                        guess = predicted
                af = path.nominal_af(trial, a.source_af, a.target_af)
                folder = a.output/f'af{af:.8g}_try{len(summary["attempts"]):02d}'
                folder.mkdir()
                report = dict(status='running', af=af, log_coordinate=trial,
                    source_af=path.nominal_af(coordinate, a.source_af, a.target_af), source_correction=revalidate,
                    inlet_multiplier=float(np.exp(trial)), predictor=predictor,
                    candidate_residuals=candidate_residuals,
                    forcing_fraction=0., newton_history=[], steps=[])
                summary['current_folder'] = str(folder.relative_to(a.output))
                station_started = perf_counter()
                def save():
                    report['elapsed_seconds'] = perf_counter()-station_started
                    write_json(folder/'metadata.json', report)
                    save_summary()
                def guard():
                    if perf_counter()-started >= a.seconds:
                        raise StopBudget('Wall-clock budget reached')
                def progress(iteration, residual):
                    report['newton_history'].append(residual)
                    save()
                    print(f'AF={af:.8g} Newton {iteration}: {residual:.6g}', flush=True)
                    guard()
                    hist = report['newton_history']
                    if len(hist) >= 5 and hist[-1] > 100*a.tol and hist[-1] > .85*hist[-5]:
                        raise RuntimeError('Residual decreased less than 15% in four updates')
                def diagnostics(values):
                    report['steps'].append(values)
                    save()
                def stage_progress(values):
                    report['linear_stage'] = values
                    save()
                    guard()
                def linear_progress(iteration, count, residual):
                    report['linear_progress'] = dict(newton_iteration=iteration, krylov_iteration=count,
                                                     preconditioned_residual=residual)
                    save()
                    guard()
                def checkpoint(iteration, field, residual):
                    np.savez_compressed(folder/'checkpoint.tmp.npz', radii=r, modes=field,
                        config_json=json.dumps(asdict(cfg)), residual=residual, af=af, converged=False,
                        auxiliary_forcing_fraction=0.)
                    (folder/'checkpoint.tmp.npz').replace(folder/'checkpoint.npz')
                save()
                try:
                    result = problem.solve(inlet, initial=guess, forcing=None, tol=a.tol, max_iter=a.max_iter,
                        linear_solver='pardiso-gmres', pardiso_backend=backend,
                        preconditioner_refresh=a.preconditioner_refresh,
                        preconditioning_side=a.preconditioning_side,
                        gmres_restart=a.gmres_restart, gmres_maxiter=a.gmres_maxiter,
                        cartesian_fallback=a.cartesian_fallback,
                        phase_pin_radius=pin, phase_update=True, phase_continuity=True,
                        progress=progress, diagnostics=diagnostics, checkpoint=checkpoint,
                        stage_progress=stage_progress, linear_progress=linear_progress)
                    original = np.linalg.norm(problem.evaluate(pack_field(result.modes), inlet), np.inf)
                    original /= max(np.max(abs(result.modes)), np.max(abs(inlet)), 1e-30)
                    if original > a.tol:
                        raise RuntimeError(f'Original unforced equations failed: {original:.3e}')
                except StopBudget as exc:
                    report.update(status='stopped_at_budget', reason=str(exc)); save()
                    raise
                except KeyboardInterrupt:
                    report.update(status='interrupted'); save()
                    raise
                except (RuntimeError, ValueError, np.linalg.LinAlgError) as exc:
                    report.update(status='failed', error=str(exc)); save()
                    (folder/'checkpoint.npz').unlink(missing_ok=True)
                    summary['attempts'].append(dict(folder=folder.name, **report))
                    step *= .5
                    backend.close()
                    if revalidate or step < a.minimum_log_step:
                        summary['status'] = 'continuation_stalled'
                        break
                else:
                    np.savez_compressed(folder/'fields.npz', radii=r, modes=result.modes,
                        config_json=json.dumps(asdict(cfg)), residual=float(original), af=af,
                        inlet_multiplier=float(np.exp(trial)),
                        auxiliary_forcing_fraction=0., converged=True,
                        outlet_alpha=alpha_out, sponge_start_Re=a.sponge_start)
                    (folder/'checkpoint.npz').unlink(missing_ok=True)
                    np.savetxt(folder/'energy.csv', np.column_stack((abs(cfg.ro)*r, result.energies(cfg))),
                        delimiter=',', header='Re,'+','.join(f'E{j}' for j in range(1,H+1)))
                    report.update(status='converged_unforced_equations', residual=float(original),
                                  iterations=result.iterations); save()
                    summary['attempts'].append(dict(folder=folder.name, **report))
                    summary['stations'].append(dict(folder=folder.name, af=af, residual=float(original)))
                    summary['last_converged_af'] = af
                    previous = None if revalidate else (coordinate, current)
                    current, coordinate, revalidate = result.modes, trial, False
                    if result.iterations <= 5:
                        step = min(a.log_step, 1.5*step)
                save_summary()
            else:
                summary['status'] = 'target_reached_unforced_equations'
    except StopBudget as exc:
        summary.update(status='stopped_at_budget', reason=str(exc))
    except KeyboardInterrupt:
        summary.update(status='interrupted')
    except Exception as exc:
        summary.update(status='failed', error=str(exc))
    finally:
        save_summary()
    print(json.dumps({k:v for k,v in summary.items() if k not in ('attempts','arguments')}, indent=2), flush=True)
    return 0 if summary['status'] == 'target_reached_unforced_equations' else 2


if __name__ == '__main__':
    raise SystemExit(main())
