"""Follow the damping-release solution curve with a pseudo-arclength constraint."""
import argparse
from contextlib import nullcontext
from dataclasses import asdict
import json
from pathlib import Path
from time import perf_counter

import numpy as np

from .global_radial import GlobalRadial, OutletSponge
from .radial_arclength import DampingArc
from .radial_continuation import load_seed, write_json


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--previous-seed',type=Path,required=True)
    p.add_argument('--seed',type=Path,required=True)
    p.add_argument('--step',type=float,default=.1)
    p.add_argument('--max-step',type=float,default=.25)
    p.add_argument('--min-step',type=float,default=.0025)
    p.add_argument('--max-stations',type=int,default=20)
    p.add_argument('--min-fraction',type=float,default=-.25)
    p.add_argument('--parameter-weight',type=float,default=1.)
    p.add_argument('--predictor-order',type=int,choices=[0,1,2],default=0,
                   help='0: secant; 1: exact tangent; 2: tangent plus quadratic correction')
    p.add_argument('--phase-pin',action='store_true')
    p.add_argument('--phase-continuity',action='store_true')
    p.add_argument('--linear-solver',choices=['direct','pardiso','pardiso-gmres'],default='direct')
    p.add_argument('--preconditioner-refresh',type=int,default=5)
    p.add_argument('--gmres-restart',type=int,default=30)
    p.add_argument('--gmres-maxiter',type=int,default=2)
    p.add_argument('--tol',type=float,default=1e-9)
    p.add_argument('--max-iter',type=int,default=12)
    p.add_argument('--output',type=Path,required=True)
    args = p.parse_args(argv)
    if not (0 < args.min_step <= args.step <= args.max_step and args.max_stations >= 1
            and args.parameter_weight > 0 and args.tol > 0 and args.max_iter >= 1):
        p.error('Require positive continuation steps, weights, iteration limits and tolerance')
    if args.output.exists():
        p.error('Choose a new output directory')
    data = []
    for path in [args.previous_seed,args.seed]:
        config,r,q = load_seed(path)
        with np.load(path) as saved:
            if 'converged' not in saved or not bool(saved['converged']) or 'release_json' not in saved:
                p.error('Both seeds must be converged release/arc fields')
            settings = json.loads(str(saved['release_json']))
        fraction = settings.pop('fraction')
        data.append((config,r,q,settings,fraction))
    config,r,current,settings,fraction = data[1]
    pc,pr,previous,ps,previous_fraction = data[0]
    if pc != config or not np.array_equal(r,pr) or ps != settings or fraction == previous_fraction:
        p.error('Require distinct fractions with identical grid, configuration and release path')
    ro = abs(config.ro)
    old = OutletSponge(settings['old_start']/ro,settings['width']/ro,settings['strength'],settings['damp_mean'])
    target = OutletSponge(settings['target']/ro,old.width,old.strength,old.damp_mean)
    problem = GlobalRadial(r,config,outlet_sponge=old,outlet_alpha=complex(settings['outlet_alpha']))
    inlet = current[0].copy()
    pin = None
    if args.phase_pin:
        op = problem.ops[0]
        v = current[:,config.harmonics+1,:3*op.m].reshape(len(r),3,op.m)
        e = r*r*np.einsum('rvi,ij,rvj->r',v.conj(),op.grid.mass,v).real
        eligible = (r>r[0]) & (r<=old.start)
        pin = float(r[np.argmax(np.where(eligible,e,-np.inf))])
    arc = DampingArc(problem,target,inlet,previous,current,previous_fraction,fraction,args.step,
        phase_pin_radius=pin,parameter_weight=args.parameter_weight)
    for q,f in [(previous,previous_fraction),(current,fraction)]:
        error = np.linalg.norm(arc.physical_residual(arc.state(q,f)),np.inf)/max(np.max(abs(q)),1e-30)
        if error > max(10*args.tol,1e-8):
            p.error(f'Seed does not satisfy its original release equations: {error:.3e}')
    args.output.mkdir(parents=True)
    arguments = {k:str(v) if isinstance(v,Path) else v for k,v in vars(args).items()}
    arguments.update(settings)
    summary = dict(status='running',method='pseudo_arclength',config=asdict(config),release=settings,
        arguments=arguments,stations=[],attempts=[],last_converged_fraction=fraction,
        interpretation='Only the separate fraction=1 target correction is physical target data.')
    write_json(args.output/'summary.json',summary)
    backend_context = nullcontext(None)
    if args.linear_solver.startswith('pardiso'):
        from .pardiso_backend import PardisoBackend
        backend_context = PardisoBackend()
    started,step = perf_counter(),args.step
    with backend_context as backend:
        while len(summary['stations']) < args.max_stations or fraction >= 1:
            is_target = fraction >= 1
            if is_target:
                active = GlobalRadial(r,config,outlet_sponge=target,outlet_alpha=complex(settings['outlet_alpha']))
                arc,guess = None,current.copy()
            else:
                active = problem
                arc = DampingArc(problem,target,inlet,previous,current,previous_fraction,fraction,step,
                    phase_pin_radius=pin,parameter_weight=args.parameter_weight)
                guess = arc.physical_field(arc.predictor)
            folder = args.output/(f'target_try{len(summary["attempts"]):02d}' if is_target else
                f'arc{len(summary["stations"]):03d}_try{len(summary["attempts"]):02d}')
            folder.mkdir()
            report = dict(status='running',method='target_correction' if is_target else 'pseudo_arclength',
                config=asdict(config),release_fraction=1. if is_target else float(arc.predictor[-1]),
                arc_step=step,source_fraction=fraction,previous_fraction=previous_fraction,
                predicted_fraction=None if is_target else float(arc.predictor[-1]),
                sponge_start=settings['target'] if is_target else settings['old_start'],
                computational_end=settings['fixed_end'],points=len(r),dr=float(r[1]-r[0]),
                residual_scale=float(max(np.max(abs(inlet)),np.max(abs(guess)),1e-30)),
                newton_history=[],fraction_history=[],steps=[])
            trial_started = perf_counter()
            def save():
                report['elapsed_seconds']=perf_counter()-trial_started
                write_json(folder/'metadata.json',report)
            def progress(iteration,norm):
                value = 1. if is_target else arc.last_fraction
                report['release_fraction']=value
                report['newton_history'].append(norm);report['fraction_history'].append(value);save()
                print(f'arc Newton {iteration}: fraction={value:.9g} residual={norm:.6g}',flush=True)
            def diagnostics(values):
                report['steps'].append(values);save()
            def checkpoint(iteration,field,norm):
                np.savez_compressed(folder/'checkpoint.tmp.npz',radii=r,modes=field,
                    config_json=json.dumps(asdict(config)),residual=norm,converged=False,
                    release_json=json.dumps(dict(**settings,fraction=report['release_fraction'])))
                (folder/'checkpoint.tmp.npz').replace(folder/'checkpoint.npz')
            def stage_progress(values):
                report['linear_stage']=values;save()
            def linear_progress(iteration,count,residual):
                report['linear_progress']=dict(newton_iteration=iteration,krylov_iteration=count,
                    preconditioned_residual=residual);save()
            save()
            try:
                if arc is not None and args.predictor_order:
                    from scipy import sparse
                    from scipy.sparse.linalg import splu
                    report['job_phase']='predictor';report['release_fraction']=fraction;save()
                    print(f'Preparing order-{args.predictor_order} predictor at fraction={fraction:.9g}',flush=True)
                    predictor_started = perf_counter()
                    row_scale,local_factor = None,None
                    def predictor_solve(matrix,rhs):
                        nonlocal row_scale,local_factor
                        if row_scale is None:
                            row_scale = np.maximum(np.asarray(abs(matrix).max(axis=1).toarray()).ravel(),1e-30)
                            equilibrated = sparse.diags(1/row_scale)@matrix
                            if backend is None:
                                local_factor = splu(equilibrated.tocsc())
                            else:
                                backend.factorize(equilibrated,progress=stage_progress)
                        answer = (local_factor.solve(rhs/row_scale) if backend is None
                                  else backend.solve(rhs/row_scale))
                        error = np.linalg.norm(matrix@answer-rhs)/max(np.linalg.norm(rhs),1e-300)
                        if not np.isfinite(error) or error > 1e-7:
                            raise RuntimeError(f'Predictor linear residual too large: {error:.3e}')
                        return answer
                    report['predictor_diagnostics']=arc.prepare_predictor(predictor_solve,args.predictor_order)
                    report['predictor_seconds']=perf_counter()-predictor_started
                    print(f'Predictor diagnostics: {report["predictor_diagnostics"]}',flush=True)
                    local_factor = None
                    guess = arc.physical_field(arc.predictor)
                    report['predicted_fraction']=float(arc.predictor[-1])
                    report['release_fraction']=float(arc.predictor[-1])
                    report['residual_scale']=float(max(np.max(abs(inlet)),np.max(abs(guess)),1e-30))
                    report['job_phase']='corrector';save()
                result = active.solve(inlet,initial=guess,augmentation=arc,
                    phase_pin_radius=pin if is_target else None,
                    phase_update=args.phase_continuity,phase_continuity=args.phase_continuity,
                    tol=args.tol,max_iter=args.max_iter,linear_solver=args.linear_solver,
                    pardiso_backend=backend,preconditioner_refresh=args.preconditioner_refresh,
                    gmres_restart=args.gmres_restart,gmres_maxiter=args.gmres_maxiter,
                    progress=progress,diagnostics=diagnostics,checkpoint=checkpoint,
                    stage_progress=stage_progress,linear_progress=linear_progress)
            except KeyboardInterrupt:
                report['status']='interrupted';save();summary['status']='interrupted'
                write_json(args.output/'summary.json',summary)
                raise
            except (RuntimeError,ValueError,np.linalg.LinAlgError) as exc:
                if backend is not None and report.get('job_phase')=='predictor':
                    backend.close()
                report.update(status='failed',error=str(exc));save()
                summary['attempts'].append(dict(folder=folder.name,**report))
                step *= .5
                if is_target or step < args.min_step:
                    summary['status']='target_correction_failed' if is_target else 'arc_corrector_failed'
                    break
            else:
                value = 1. if is_target else float(arc.accepted_state[-1])
                energy = result.energies(config)
                np.savez_compressed(folder/'fields.npz',radii=r,modes=result.modes,
                    config_json=json.dumps(asdict(config)),residual=result.residual,converged=True,
                    release_json=json.dumps(dict(**settings,fraction=value)))
                np.savetxt(folder/'energy.csv',np.column_stack((ro*r,energy)),delimiter=',',
                    header='Re,'+','.join(f'E{j}' for j in range(1,config.harmonics+1)))
                (folder/'checkpoint.npz').unlink()
                report.update(status='converged_discrete_equations' if is_target else 'converged_auxiliary_damping',
                    release_fraction=value,residual=result.residual)
                save();summary['attempts'].append(dict(folder=folder.name,**report))
                summary['stations'].append(dict(folder=folder.name,**report))
                summary['last_converged_fraction']=value
                previous,current,previous_fraction,fraction = current,result.modes,fraction,value
                if is_target:
                    summary['status']='target_reached_discrete_equations'
                    break
                if fraction < args.min_fraction:
                    summary['status']='parameter_bound_reached'
                    break
                if result.iterations <= 5:
                    step = min(args.max_step,step*1.4)
            summary['elapsed_seconds']=perf_counter()-started
            write_json(args.output/'summary.json',summary)
        else:
            summary['status']='station_limit_reached'
    summary['elapsed_seconds']=perf_counter()-started
    write_json(args.output/'summary.json',summary)
    print(json.dumps({k:v for k,v in summary.items() if k not in ('stations','attempts')},indent=2),flush=True)
    return 0 if summary['status']=='target_reached_discrete_equations' else 2
