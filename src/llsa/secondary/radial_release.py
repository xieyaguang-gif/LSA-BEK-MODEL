"""Fixed-grid continuation from an old sponge to a prescribed target sponge.

Only the final release fraction solves the target problem. Intermediate
fields carry explicit auxiliary-damping metadata and are not final data.
"""
import argparse
from contextlib import nullcontext
from dataclasses import asdict
import json
from pathlib import Path
from time import perf_counter

import numpy as np

from .global_radial import BlendedSponge, GlobalRadial
from .global_radial import OutletSponge
from .radial_continuation import fixed_domain_seed, load_seed, secant_seed, write_json


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--seed', type=Path, required=True)
    p.add_argument('--old-start', type=float, required=True)
    p.add_argument('--target', type=float, required=True)
    p.add_argument('--fixed-end', type=float, required=True)
    p.add_argument('--width', type=float, default=40.)
    p.add_argument('--strength', type=float, default=10.)
    p.add_argument('--damp-mean', action='store_true')
    p.add_argument('--outlet-alpha', type=complex, default=0j)
    p.add_argument('--dr', type=float, default=.5)
    p.add_argument('--step', type=float, default=.1, help='Initial step in release fraction, not Re')
    p.add_argument('--max-step', type=float, default=.2)
    p.add_argument('--min-step', type=float, default=.00625)
    p.add_argument('--start-fraction', type=float, default=0.)
    p.add_argument('--predictor', choices=['none','linear','phase'], default='phase')
    p.add_argument('--initial-checkpoint', type=Path)
    p.add_argument('--trial-checkpoint', type=Path,
                   help='Initial guess for the first advanced fraction, with matching release metadata')
    p.add_argument('--phase-pin', action='store_true')
    p.add_argument('--phase-update', action='store_true')
    p.add_argument('--phase-continuity', action='store_true',
                   help='Use phase updates projected onto the affine interior continuity constraints')
    p.add_argument('--linear-solver', choices=['harmonic-gmres','direct','pardiso','pardiso-gmres'], default='harmonic-gmres')
    p.add_argument('--coupled-harmonics', type=int, default=1)
    p.add_argument('--preconditioner-refresh', type=int, default=5)
    p.add_argument('--gmres-restart', type=int, default=30)
    p.add_argument('--gmres-maxiter', type=int, default=2)
    p.add_argument('--tol', type=float, default=1e-9)
    p.add_argument('--max-iter', type=int, default=30)
    p.add_argument('--output', type=Path, required=True)
    args = p.parse_args(argv)
    if (not 0 < args.min_step <= args.step <= args.max_step <= 1
            or not 0 <= args.start_fraction < 1 or args.target <= args.old_start
            or min(args.dr,args.width,args.tol) <= 0 or args.strength < 0
            or args.fixed_end < args.target+args.width):
        p.error('Require valid release steps in (0,1], target>old-start and fixed-end>=target+width')
    if args.output.exists():
        p.error('Choose a new output directory')
    config, old_r, old_modes = load_seed(args.seed)
    release_settings = dict(old_start=args.old_start,target=args.target,width=args.width,
        strength=args.strength,damp_mean=args.damp_mean,fixed_end=args.fixed_end,
        outlet_alpha=str(args.outlet_alpha))
    if args.start_fraction:
        with np.load(args.seed) as data:
            if not {'release_json','converged'}.issubset(data.files) or not bool(data['converged']):
                p.error('A nonzero start-fraction requires a converged release seed')
            saved = json.loads(str(data['release_json']))
        saved_fraction = saved.pop('fraction', float('nan'))
        if (saved != release_settings
                or not np.isclose(saved_fraction,args.start_fraction,rtol=0,atol=1e-12)):
            p.error('Release seed metadata do not match the requested path/fraction')
    nr = int(np.ceil((args.fixed_end/abs(config.ro)-old_r[0])/args.dr))+1
    r = np.linspace(old_r[0],args.fixed_end/abs(config.ro),nr)
    modes = fixed_domain_seed(old_r,old_modes,r,config,damp_mean=args.damp_mean)
    inlet = old_modes[0].copy()
    if args.initial_checkpoint is not None:
        c, cr, cq = load_seed(args.initial_checkpoint)
        if c != config or cr.shape != r.shape or not np.allclose(cr,r,rtol=0,atol=1e-10):
            p.error('Checkpoint must match the fixed grid/config exactly')
        modes = cq
    trial_guess = None
    if args.trial_checkpoint is not None:
        c, cr, cq = load_seed(args.trial_checkpoint)
        with np.load(args.trial_checkpoint) as data:
            saved = json.loads(str(data['release_json'])) if 'release_json' in data else {}
        saved_fraction = saved.pop('fraction',float('nan'))
        expected_fraction = min(1.,args.start_fraction+args.step)
        if (c != config or cr.shape != r.shape or not np.allclose(cr,r,rtol=0,atol=1e-10)
                or saved != release_settings
                or not np.isclose(saved_fraction,expected_fraction,rtol=0,atol=1e-12)):
            p.error('Trial checkpoint must match the grid, path and first advanced fraction')
        trial_guess = cq
    op = config.operators(r[0])
    pin = None
    if args.phase_pin:
        velocity = modes[:,config.harmonics+1,:3*op.m].reshape(nr,3,op.m)
        energy = r*r*np.einsum('rvi,ij,rvj->r',velocity.conj(),op.grid.mass,velocity).real
        eligible = (r>r[0]) & (r<=args.old_start/abs(config.ro))
        pin = float(r[np.argmax(np.where(eligible,energy,-np.inf))])
    original_sponge = OutletSponge(args.old_start/abs(config.ro),args.width/abs(config.ro),args.strength,args.damp_mean)
    target_sponge = OutletSponge(args.target/abs(config.ro),args.width/abs(config.ro),args.strength,args.damp_mean)
    args.output.mkdir(parents=True)
    summary = dict(status='running',config=asdict(config),release=release_settings,
        arguments={k:str(v) if isinstance(v,(Path,complex)) else v for k,v in vars(args).items()},
        stations=[],attempts=[],last_converged_fraction=None,
        interpretation='Only fraction=1 is the target physical problem; earlier stations retain auxiliary damping.')
    write_json(args.output/'summary.json',summary)
    fraction, step, first = args.start_fraction,args.step,True
    previous = None
    started = perf_counter()
    backend_context = nullcontext(None)
    if args.linear_solver.startswith('pardiso'):
        from .pardiso_backend import PardisoBackend
        backend_context = PardisoBackend()
    with backend_context as backend:
        while first or fraction < 1-1e-12:
            trial = fraction if first else min(1.,fraction+step)
            guess = modes.copy()
            guess_kind = 'seed' if first else 'previous'
            if previous is not None and not first and args.predictor != 'none':
                old_fraction,old_field = previous
                guess = secant_seed(modes,old_field,(trial-fraction)/(fraction-old_fraction),config,r[0],
                                    phase_aligned=args.predictor=='phase')
                guess_kind = args.predictor
            if not first and trial_guess is not None:
                guess = trial_guess
                trial_guess = None
                guess_kind = 'trial_checkpoint'
            folder = args.output/f'tau{trial:.6f}_try{len(summary["attempts"]):02d}'
            folder.mkdir()
            scale = float(max(np.max(abs(inlet)),np.max(abs(guess)),1e-30))
            report = dict(status='running',release_fraction=trial,config=asdict(config),
                initial_guess=guess_kind,
                sponge_start=args.target if trial==1 else args.old_start,computational_end=args.fixed_end,
                fixed_end=args.fixed_end,points=nr,dr=float(r[1]-r[0]),residual_scale=scale,
                residual_definition='Newton residual / residual_scale; original target-frame equations checked on acceptance',
                phase_pin_Re=None if pin is None else pin*abs(config.ro),newton_history=[],steps=[])
            trial_started = perf_counter()
            def save():
                report['elapsed_seconds'] = perf_counter()-trial_started
                write_json(folder/'metadata.json',report)
            save()
            def progress(iteration,norm):
                report['newton_history'].append(norm);save()
                print(f'release={trial:.6g} Newton {iteration}: {norm:.6g}',flush=True)
            def diagnostics(values):
                report['steps'].append(values);save()
                print(' step '+json.dumps(values),flush=True)
            def checkpoint(iteration,field,norm):
                np.savez_compressed(folder/'checkpoint.tmp.npz',radii=r,modes=field,
                    config_json=json.dumps(asdict(config)),release_json=json.dumps(dict(**release_settings,fraction=trial)),
                    residual=norm,converged=norm<=args.tol)
                (folder/'checkpoint.tmp.npz').replace(folder/'checkpoint.npz')
            def linear_progress(iteration,count,residual):
                report['linear_progress']=dict(newton_iteration=iteration,krylov_iteration=count,preconditioned_residual=residual);save()
            def stage_progress(values):
                report['linear_stage']=values;save()
            problem = GlobalRadial(r,config,outlet_alpha=args.outlet_alpha,
                outlet_sponge=BlendedSponge(original_sponge,target_sponge,trial))
            try:
                solution = problem.solve(inlet,initial=guess,tol=args.tol,max_iter=args.max_iter,
                    linear_solver=args.linear_solver,coupled_harmonics=args.coupled_harmonics,
                    preconditioner_refresh=args.preconditioner_refresh,gmres_restart=args.gmres_restart,
                    gmres_maxiter=args.gmres_maxiter,phase_pin_radius=pin,
                    phase_update=args.phase_update or args.phase_continuity,phase_continuity=args.phase_continuity,
                    pardiso_backend=backend,progress=progress,diagnostics=diagnostics,checkpoint=checkpoint,
                    linear_progress=linear_progress,stage_progress=stage_progress)
            except KeyboardInterrupt:
                report['status']='interrupted';save()
                summary['status']='interrupted';write_json(args.output/'summary.json',summary)
                raise
            except (RuntimeError,ValueError,np.linalg.LinAlgError) as exc:
                report.update(status='failed',error=str(exc));save()
                summary['attempts'].append(dict(folder=folder.name,**report))
                if first:
                    summary['status']='initial_validation_failed'
                    break
                step *= .5
                if step < args.min_step:
                    summary['status']='release_continuation_failed'
                    break
            else:
                energy = solution.energies(config)
                np.savez_compressed(folder/'fields.npz',radii=r,modes=solution.modes,
                    config_json=json.dumps(asdict(config)),release_json=json.dumps(dict(**release_settings,fraction=trial)),
                    converged=True,residual=solution.residual)
                np.savetxt(folder/'energy.csv',np.column_stack((abs(config.ro)*r,energy)),delimiter=',',
                    header='Re,'+','.join(f'E{j}' for j in range(1,config.harmonics+1)))
                (folder/'checkpoint.npz').unlink()
                report.update(status='converged_discrete_equations' if trial==1 else 'converged_auxiliary_damping',
                    residual=solution.residual,E1_at_sponge_start=float(np.interp(report['sponge_start'],abs(config.ro)*r,energy[:,0])))
                save();summary['attempts'].append(dict(folder=folder.name,**report))
                summary['stations'].append(dict(folder=folder.name,**report))
                previous = None if first else (fraction,modes)
                modes,fraction = solution.modes,trial
                summary['last_converged_fraction']=fraction
                if not first and solution.iterations <= 6:
                    step=min(args.max_step,step*1.5)
                first=False
            summary['elapsed_seconds']=perf_counter()-started
            write_json(args.output/'summary.json',summary)
            del problem
        else:
            summary['status']='target_reached_discrete_equations'
    summary['elapsed_seconds']=perf_counter()-started
    write_json(args.output/'summary.json',summary)
    print(json.dumps({k:v for k,v in summary.items() if k not in ('stations','attempts')},indent=2),flush=True)
    return 0 if summary['status']=='target_reached_discrete_equations' else 2
