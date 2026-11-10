"""Restartable natural continuation of the steady full-radial equations.

Only the initial guess is transported. Every station still solves the full
equations on the original physical coordinate, including radial diffusion.
"""
import argparse
from dataclasses import asdict, replace
import json
from pathlib import Path
from time import perf_counter

import numpy as np
from scipy.interpolate import BarycentricInterpolator, CubicSpline

from llsa.secondary.radial import RadialConfig
from llsa.secondary.global_radial import GlobalRadial, OutletSponge


def transport_seed(r, modes, new_r, config, old_start, anchor_width=30., *,
                   mapping='linear', anchor=None):
    """Transport a seed, or keep its physical coordinates with mapping='physical'.

    All lengths except anchor_width are solver radii; anchor_width uses Re.
    The kink in the map belongs to the initial guess, never to the operator.
    """
    if mapping == 'physical':
        return fixed_domain_seed(r, modes, new_r, config)
    H = config.harmonics
    op = config.operators(r[0])
    active = np.flatnonzero(r <= old_start)
    tail = active[-min(12, len(active)):]
    v = modes[tail, H+1, :3*op.m].reshape(len(tail), 3, op.m)
    overlap = np.einsum('rvi,ij,rvj->r', v[:-1].conj(), op.grid.mass, v[1:])
    carrier = float(np.median(np.angle(overlap)/np.diff(r[tail])))
    anchor = max(r[0], old_start-anchor_width/abs(config.ro)) if anchor is None else float(anchor)
    if not r[0] <= anchor < min(r[-1], new_r[-1]):
        raise ValueError('Transport anchor must lie inside both domains')
    if mapping == 'linear':
        mapped = np.where(new_r <= anchor, new_r,
                          anchor+(new_r-anchor)*(r[-1]-anchor)/(new_r[-1]-anchor))
    elif mapping == 'smooth':
        t = np.clip((new_r-anchor)/(new_r[-1]-anchor), 0, 1)
        ramp = t**3*(10+t*(-15+6*t))
        mapped = new_r-(new_r[-1]-r[-1])*ramp
        if np.any(np.diff(mapped) <= 0):
            raise ValueError('Transport map is not monotone; reduce the continuation step')
    else:
        raise ValueError('Transport mapping must be linear, smooth or physical')
    ns = np.arange(-H, H+1)[None, :, None]
    slow = modes*np.exp(-1j*carrier*(r-r[0])[:, None, None]*ns)
    guess = CubicSpline(r, slow, axis=0)(mapped)
    guess *= np.exp(1j*carrier*(new_r-r[0])[:, None, None]*ns)
    # Eliminate interpolation roundoff in the real-field symmetry.
    guess = .5*(guess+guess[:, ::-1].conj())
    return guess


def secant_seed(guess, old_guess, ratio, config, radius, phase_aligned=False):
    """Extrapolate amplitude/profile and optionally the common spiral phase.

    A straight complex secant turns a phase change into artificial amplitude
    growth, especially for high harmonics. Aligning the fundamental velocity
    profiles first avoids that effect for a translating wave.
    """
    if not phase_aligned:
        return guess+ratio*(guess-old_guess)
    H = config.harmonics
    op = config.operators(radius)
    a, b = [q[:, H+1, :3*op.m].reshape(len(q), 3, op.m) for q in (old_guess, guess)]
    overlap = np.einsum('rvi,ij,rvj->r', a.conj(), op.grid.mass, b)
    phase = np.unwrap(np.angle(overlap))
    ns = np.arange(-H, H+1)[None, :, None]
    aligned = old_guess*np.exp(1j*phase[:,None,None]*ns)
    prediction = (guess+ratio*(guess-aligned))*np.exp(1j*ratio*phase[:,None,None]*ns)
    return .5*(prediction+prediction[:, ::-1].conj())


def write_json(path, data):
    temporary = path.with_suffix('.tmp')
    temporary.write_text(json.dumps(data, indent=2)+'\n')
    temporary.replace(path)


def fixed_domain_seed(r, modes, new_r, config, decay_width=10., damp_mean=False):
    """Keep the physical coordinate; append a damped tail only beyond the seed.

    This is an initial guess, not an equation or an imposed downstream field.
    The undamped mean is continued constantly. Damped harmonics have a Gaussian tail
    with zero envelope derivative at the old outlet.
    """
    sample = np.minimum(new_r, r[-1])
    guess = CubicSpline(r, modes, axis=0)(sample)
    distance = np.maximum(new_r-r[-1], 0)*abs(config.ro)
    envelope = np.exp(-(distance/decay_width)**2)
    guess[:, :config.harmonics] *= envelope[:, None, None]
    guess[:, config.harmonics+1:] *= envelope[:, None, None]
    if damp_mean:
        guess[:, config.harmonics] *= envelope[:, None]
    return .5*(guess+guess[:, ::-1].conj())


def transport_fixed_buffer(r, modes, config, old_start, new_start, width):
    """Move a buffer envelope on a fixed grid with a solenoidal velocity map.

    Starts and width use solver R. The radial/azimuthal Piola map preserves
    continuous cylindrical divergence, including the restored spiral carrier.
    This changes the initial guess only, never the solved equations.
    """
    if new_start == old_start:
        return modes.copy()
    if new_start < old_start or config.beta == 0:
        raise ValueError('Buffer transport requires downstream motion and nonzero beta')
    H = config.harmonics
    op = config.operators(r[0])
    active = np.flatnonzero(r <= old_start)
    tail = active[-min(12,len(active)):]
    v = modes[tail,H+1,:3*op.m].reshape(len(tail),3,op.m)
    overlap = np.einsum('rvi,ij,rvj->r',v[:-1].conj(),op.grid.mass,v[1:])
    carrier = float(np.median(np.angle(overlap)/np.diff(r[tail])))
    anchor = max(r[0],old_start-30/abs(config.ro))
    length = new_start+width-anchor
    t = np.clip((r-anchor)/length,0,1)
    ramp = t**3*(10+t*(-15+6*t))
    ramp_derivative = 30*t*t*(1-t)**2/length
    delta = new_start-old_start
    mapped = r-delta*ramp
    derivative = 1-delta*ramp_derivative
    if np.min(derivative) <= 0:
        raise ValueError('Fixed-buffer transport folded; reduce the parameter step')
    ns = np.arange(-H,H+1)[None,:,None]
    slow = modes*np.exp(-1j*carrier*(r-r[0])[:,None,None]*ns)
    values = CubicSpline(r,slow,axis=0)(mapped)
    values *= np.exp(1j*carrier*(r-r[0])[:,None,None]*ns)
    values = values.reshape(len(r),2*H+1,4,op.m)
    ratio = (mapped/r)[:,None,None]
    fp = derivative[:,None,None]
    u, v, w, pressure = [values[:,:,i,:].copy() for i in range(4)]
    values[:,:,0,:] = ratio**2*u
    values[:,:,1,:] = ratio*fp*v-ratio*(mapped[:,None,None]/config.beta)*carrier*(1-fp)*u
    # Axisymmetric azimuthal velocity does not enter continuity; transport
    # its physical value without adding the helical shear correction.
    values[:,H,1,:] = ratio[:,0,:]*v[:,H,:]
    values[:,:,2,:] = ratio**2*fp*w
    values[:,:,3,:] = ratio**2*pressure
    result = values.reshape(modes.shape)
    return .5*(result+result[:,::-1].conj())


def load_seed(path, n=None, harmonics=None):
    """Interpolate the wall-normal polynomial and pad/truncate Fourier modes."""
    with np.load(path) as data:
        old = RadialConfig(**json.loads(str(data['config_json'])))
        r, modes = data['radii'].copy(), data['modes'].copy()
    config = replace(old, n=old.n if n is None else n,
                     harmonics=old.harmonics if harmonics is None else harmonics)
    if config.n != old.n:
        old_grid, grid = old.operators(r[0]).grid, config.operators(r[0]).grid
        modes = BarycentricInterpolator(old_grid.xi,
            modes.reshape(len(r), 2*old.harmonics+1, 4, -1), axis=3)(grid.xi)
        modes = modes.reshape(len(r), 2*old.harmonics+1, -1)
    if config.harmonics != old.harmonics:
        q = np.zeros((len(r), 2*config.harmonics+1, modes.shape[-1]), complex)
        keep = min(old.harmonics, config.harmonics)
        q[:, config.harmonics-keep:config.harmonics+keep+1] = modes[:, old.harmonics-keep:old.harmonics+keep+1]
        modes = q
    return config, r, .5*(modes+modes[:, ::-1].conj())


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--seed', type=Path, required=True)
    p.add_argument('--previous-seed', type=Path, help='Previous converged station for a secant predictor')
    p.add_argument('--previous-start', type=float)
    p.add_argument('--predictor', choices=['none', 'linear', 'phase'], default='linear',
                   help='Transport alone, complex secant, or aligned-profile/phase secant')
    p.add_argument('--transport-map', choices=['linear','smooth','physical'], default='smooth',
                   help='Smooth/linear buffer mapping, or physical-coordinate interpolation with a damped appended tail')
    p.add_argument('--old-start', type=float, required=True, help='Old sponge start in Re')
    p.add_argument('--target', type=float, required=True, help='Target physical Re (sponge starts here)')
    p.add_argument('--revalidate', action='store_true', help='Solve the initial station first (e.g. after grid refinement)')
    p.add_argument('--fixed-end', type=float,
                   help='Fixed computational outlet in Re; requires revalidation and end>=target+width')
    p.add_argument('--fixed-predictor', choices=['identity','transport'], default='identity',
                   help='Initial guess on the fixed grid: unchanged field or divergence-preserving buffer transport')
    p.add_argument('--n', type=int, help='Override seed wall-normal order by polynomial interpolation')
    p.add_argument('--harmonics', type=int, help='Override seed truncation; added harmonics start at zero')
    p.add_argument('--step', type=float, default=5.)
    p.add_argument('--max-step', type=float, help='Allow step growth up to this value; default --step')
    p.add_argument('--min-step', type=float, default=.5)
    p.add_argument('--dr', type=float, default=.5, help='Maximum step in solver R')
    p.add_argument('--width', type=float, default=40., help='Sponge width in Re')
    p.add_argument('--strength', type=float, default=10.)
    p.add_argument('--damp-mean', action='store_true', help='Relax the mean correction too, as in the DNS sponge')
    p.add_argument('--outlet-alpha', type=complex, default=0j)
    p.add_argument('--coupled-harmonics', type=int, default=1)
    p.add_argument('--linear-solver', choices=['harmonic-gmres','direct','pardiso','pardiso-gmres'], default='harmonic-gmres')
    p.add_argument('--preconditioner-refresh', type=int, default=3)
    p.add_argument('--preconditioning-side', choices=['left', 'right'], default='left',
                   help='PARDISO-GMRES only: right minimizes the current row-scaled equation residual directly')
    p.add_argument('--preconditioner-sweep', choices=['diagonal','forward','symmetric'], default='diagonal')
    p.add_argument('--gmres-restart', type=int, default=60, help='Krylov subspace dimension before restart')
    p.add_argument('--gmres-maxiter', type=int, default=12, help='Maximum restart cycles before refactor/failure')
    p.add_argument('--phase-update', action='store_true', help='Retract the Newton phase tangent by rotation')
    p.add_argument('--phase-continuity', action='store_true',
                   help='Use phase updates projected onto affine interior continuity constraints')
    p.add_argument('--phase-pin', action='store_true',
                   help='Fix phase at the strongest physical fundamental; rotate back to the original inlet')
    p.add_argument('--cartesian-fallback', action='store_true',
                   help='Compare ordinary and phase Newton paths at each damping and use the lower residual merit')
    p.add_argument('--merit', choices=['inf','scaled-l2'], default='inf')
    p.add_argument('--initial-checkpoint', type=Path,
                   help='Resume the first trial on an identical grid; seed must remain the last converged station')
    p.add_argument('--max-iter', type=int, default=40)
    p.add_argument('--tol', type=float, default=1e-9)
    p.add_argument('--output', type=Path, required=True)
    args = p.parse_args(argv)
    max_step = args.step if args.max_step is None else args.max_step
    if args.output.exists():
        p.error('Choose a new output directory')
    if (min(args.step, max_step, args.min_step, args.dr, args.width, args.tol) <= 0
            or args.target < args.old_start or (args.target == args.old_start and not args.revalidate)):
        p.error('Require positive steps/width/tolerance and target>old-start, or --revalidate at equal target')
    if args.fixed_end is not None and (not args.revalidate or args.fixed_end < args.target+args.width):
        p.error('fixed-end requires --revalidate and fixed-end>=target+width')
    if args.cartesian_fallback and not (args.phase_update or args.phase_continuity):
        p.error('cartesian-fallback requires phase-update or phase-continuity')
    if args.preconditioning_side == 'right' and args.linear_solver != 'pardiso-gmres':
        p.error('right preconditioning requires pardiso-gmres')
    config, r, modes = load_seed(args.seed, args.n, args.harmonics)
    previous = None
    if args.previous_seed is not None:
        if args.previous_start is None or args.previous_start >= args.old_start:
            p.error('previous-start must be below old-start')
        previous_config, pr, pq = load_seed(args.previous_seed, args.n, args.harmonics)
        if previous_config != config:
            p.error('Secant seeds must have identical configs after interpolation')
        previous = (pr, pq, args.previous_start)
    args.output.mkdir(parents=True)
    summary = dict(status='running', validation_status='discrete_equations_only',
                   config=asdict(config), arguments={k:str(v) if isinstance(v,(Path,complex)) else v
                    for k,v in vars(args).items()}, stations=[], attempts=[])
    write_json(args.output/'summary.json', summary)
    started = perf_counter()
    current, step = args.old_start, args.step
    revalidate = args.revalidate
    # The inlet is identical to the seed, including its Fourier phase.
    inlet = modes[0].copy()
    while revalidate or current < args.target-1e-10:
        stop = current if revalidate else min(args.target, current+step)
        end = stop+args.width if args.fixed_end is None else args.fixed_end
        nr = int(np.ceil((end/abs(config.ro)-r[0])/args.dr))+1
        new_r = np.linspace(r[0], end/abs(config.ro), nr)
        anchor = max(r[0], (current-30.)/abs(config.ro))
        if args.fixed_end is None:
            guess = transport_seed(r, modes, new_r, config, current/abs(config.ro),
                                   mapping=args.transport_map, anchor=anchor)
        else:
            guess = fixed_domain_seed(r, modes, new_r, config, damp_mean=args.damp_mean)
            if args.fixed_predictor == 'transport' and not revalidate:
                guess = transport_fixed_buffer(new_r,guess,config,current/abs(config.ro),
                    stop/abs(config.ro),args.width/abs(config.ro))
        use_secant = previous is not None and not revalidate and args.predictor != 'none'
        if use_secant:
            pr, pq, ps = previous
            if args.fixed_end is None:
                old_guess = transport_seed(pr, pq, new_r, config, ps/abs(config.ro),
                                           mapping=args.transport_map, anchor=anchor)
            else:
                old_guess = fixed_domain_seed(pr, pq, new_r, config, damp_mean=args.damp_mean)
                if args.fixed_predictor == 'transport':
                    old_guess = transport_fixed_buffer(new_r,old_guess,config,ps/abs(config.ro),
                        stop/abs(config.ro),args.width/abs(config.ro))
            guess = secant_seed(guess, old_guess, (stop-current)/(current-ps), config,
                                new_r[0], phase_aligned=args.predictor == 'phase')
        if args.initial_checkpoint is not None:
            with np.load(args.initial_checkpoint) as data:
                if (json.loads(str(data['config_json'])) != asdict(config)
                        or data['radii'].shape != new_r.shape or not np.allclose(data['radii'],new_r,atol=1e-10,rtol=0)):
                    raise ValueError('Checkpoint must match the first trial config/grid exactly')
                guess = data['modes'].copy()
            args.initial_checkpoint = None
        folder = args.output/f's{stop:g}_try{len(summary["attempts"]):02d}'
        folder.mkdir()
        report = dict(status='running', sponge_start=stop, config=asdict(config),
                      points=nr, dr=float(new_r[1]-new_r[0]),
                      predictor='revalidation' if revalidate else ('secant_transport' if use_secant else 'transport'),
                      secant_type=args.predictor,
                      transport_map=args.transport_map,
                      computational_end=end, fixed_end=args.fixed_end,
                      fixed_predictor=args.fixed_predictor,
                      residual_scale=float(max(np.max(abs(inlet)), np.max(abs(guess)), 1e-30)),
                      residual_definition=('max(abs(packed real equation residual)) / residual_scale'
                          if not args.phase_pin else 'augmented phase-gauge residual / residual_scale; '
                          'acceptance additionally checks the original equations after rotating back'),
                      newton_history=[], steps=[])
        t0 = perf_counter()
        def save():
            report['elapsed_seconds'] = perf_counter()-t0
            write_json(folder/'metadata.json', report)
        save()
        def progress(it, norm):
            report['newton_history'].append(norm)
            save()
            print(f'Re_s={stop:g} H={config.harmonics} Newton {it}: {norm:.6g}', flush=True)
        def diagnostics(values):
            report['steps'].append(values)
            save()
            print(' step '+json.dumps(values), flush=True)
        def linear_progress(it, count, preconditioned_residual):
            report['linear_progress'] = dict(newton_iteration=it, krylov_iteration=count,
                preconditioned_residual=preconditioned_residual)
            save()
        def stage_progress(values):
            report['linear_stage'] = values
            save()
        def checkpoint(it, field, norm):
            np.savez_compressed(folder/'checkpoint.tmp.npz', radii=new_r, modes=field,
                config_json=json.dumps(asdict(config)), residual=norm, converged=norm<=args.tol)
            (folder/'checkpoint.tmp.npz').replace(folder/'checkpoint.npz')
        problem = GlobalRadial(new_r, config, outlet_alpha=args.outlet_alpha,
            outlet_sponge=OutletSponge(stop/abs(config.ro), args.width/abs(config.ro), args.strength, args.damp_mean))
        phase_pin_radius = None
        if args.phase_pin:
            m = problem.ops[0].m
            velocity = guess[:,config.harmonics+1,:3*m].reshape(nr,3,m)
            energies = new_r**2*np.einsum('rvi,ij,rvj->r',velocity.conj(),problem.ops[0].grid.mass,velocity).real
            eligible = (new_r > new_r[0]) & (new_r <= stop/abs(config.ro))
            phase_pin_radius = float(new_r[np.argmax(np.where(eligible,energies,-np.inf))])
            report['phase_pin_Re'] = phase_pin_radius*abs(config.ro)
            save()
        try:
            solution = problem.solve(inlet, initial=guess, max_iter=args.max_iter, tol=args.tol,
                linear_solver=args.linear_solver, coupled_harmonics=args.coupled_harmonics,
                progress=progress, diagnostics=diagnostics, checkpoint=checkpoint,
                preconditioner_refresh=args.preconditioner_refresh, merit=args.merit,
                preconditioner_sweep=args.preconditioner_sweep, gmres_restart=args.gmres_restart,
                gmres_maxiter=args.gmres_maxiter,
                linear_progress=linear_progress, stage_progress=stage_progress,
                phase_update=args.phase_update or args.phase_continuity,phase_continuity=args.phase_continuity,
                cartesian_fallback=args.cartesian_fallback,
                preconditioning_side=args.preconditioning_side,
                phase_pin_radius=phase_pin_radius)
        except KeyboardInterrupt:
            report.update(status='interrupted', reason='Explicit interruption; checkpoint is not a converged field')
            save()
            summary['attempts'].append(dict(folder=folder.name, **report))
            summary.update(status='interrupted', last_converged_start=current,
                           elapsed_seconds=perf_counter()-started)
            write_json(args.output/'summary.json', summary)
            raise
        except (RuntimeError, ValueError, np.linalg.LinAlgError) as exc:
            report.update(status='failed', error=str(exc))
            save()
            summary['attempts'].append(dict(folder=folder.name, **report))
            if revalidate:
                summary['status'] = 'revalidation_failed'
                break
            step *= .5
            if step < args.min_step:
                summary['status'] = 'continuation_failed'
                break
        else:
            energy = solution.energies(config)
            np.savez_compressed(folder/'fields.npz', radii=new_r, modes=solution.modes,
                                config_json=json.dumps(asdict(config)))
            np.savetxt(folder/'energy.csv', np.column_stack((abs(config.ro)*new_r, energy)),
                       delimiter=',', header='Re,'+','.join(f'E{j}' for j in range(1,config.harmonics+1)))
            (folder/'checkpoint.npz').unlink()
            physical = abs(config.ro)*new_r <= stop
            report.update(status='converged_discrete_equations', residual=solution.residual,
                          E1_at_sponge_start=float(np.interp(stop,abs(config.ro)*new_r,energy[:,0])),
                          tail_fraction_max=float(np.max(energy[physical,-1]/energy[physical].sum(axis=1))))
            save()
            summary['attempts'].append(dict(folder=folder.name, **report))
            summary['stations'].append(dict(folder=folder.name, **report))
            previous = None if revalidate else (r, modes, current)
            r, modes, current = new_r, solution.modes, stop
            revalidate = False
            if solution.iterations <= 6:
                step = min(max_step, step*1.5)
        summary['last_converged_start'] = current
        summary['elapsed_seconds'] = perf_counter()-started
        write_json(args.output/'summary.json', summary)
        del problem
    else:
        summary['status'] = 'target_reached_discrete_equations'
    summary['last_converged_start'] = current
    summary['elapsed_seconds'] = perf_counter()-started
    write_json(args.output/'summary.json', summary)
    print(json.dumps({k:v for k,v in summary.items() if k not in ('stations','attempts')},indent=2),flush=True)


if __name__ == '__main__':
    main()
