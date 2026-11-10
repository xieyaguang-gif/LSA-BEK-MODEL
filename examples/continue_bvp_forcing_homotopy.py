"""Continue a PSE field to an unforced full radial nonlinear BVP."""
import argparse
from dataclasses import asdict
import json
from pathlib import Path
from time import perf_counter

import numpy as np

from llsa.secondary.radial import RadialConfig, pack_field, unpack_field
from llsa.secondary.global_radial import GlobalRadial, OutletSponge
from llsa.secondary.oneway import SpectralOneWay, causal_projector
from llsa.secondary.pardiso_backend import PardisoBackend
from validate_radial_re700 import pse_seed


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--amplitude',type=float,required=True)
    p.add_argument('--seed',type=Path,required=True)
    p.add_argument('--sponge-start',type=float,required=True)
    p.add_argument('--end',type=float,required=True)
    p.add_argument('--output',type=Path,required=True)
    p.add_argument('--dr',type=float,default=.5)
    p.add_argument('--n',type=int,default=24)
    p.add_argument('--harmonics',type=int,default=5)
    p.add_argument('--initial-step',type=float,default=.25)
    p.add_argument('--minimum-step',type=float,default=.015625)
    p.add_argument('--max-iter',type=int,default=12)
    p.add_argument('--phase-pin',action='store_true',
                   help='Pin the azimuthal phase at the fundamental-energy maximum')
    p.add_argument('--phase-continuity',action='store_true',
                   help='Project phase updates onto the interior continuity constraints')
    p.add_argument('--start-lambda',type=float,default=1.,
                   help='Resume at this forcing fraction (default: 1)')
    p.add_argument('--start-field',type=Path,
                   help='Converged checkpoint at --start-lambda; forcing is still defined by the PSE seed')
    a=p.parse_args()
    if a.output.exists():p.error('output already exists')
    a.output.mkdir(parents=True)
    cfg=RadialConfig(n=a.n,harmonics=a.harmonics)
    nr=int(np.ceil((a.end-320)/abs(cfg.ro)/a.dr))+1
    r=np.linspace(320/abs(cfg.ro),a.end/abs(cfg.ro),nr)
    march=SpectralOneWay(cfg);inlet,_,alpha=march.initialize(r[0],a.amplitude,1.1858025795710305)
    op=cfg.operators(r[0]);inlet=inlet[:,:op.size]
    proj=causal_projector(cfg.operators(r[-1]),1,cfg.frequency(cfg.operators(r[-1])))
    allowed=np.flatnonzero(~proj.upstream);ix=allowed[np.argmin(abs(-1j*proj.values[allowed]-alpha))]
    alpha_out=-1j*proj.values[ix]
    sponge=OutletSponge(a.sponge_start/abs(cfg.ro),40/abs(cfg.ro),10,False)
    problem=GlobalRadial(r,cfg,outlet_alpha=alpha_out,outlet_sponge=sponge)
    q0=pse_seed(a.seed,r,cfg,inlet,match_tail_phase=True)
    residual=problem.evaluate(pack_field(q0),inlet)
    forcing=unpack_field(residual,nr,cfg.harmonics,op.size)
    forcing=.5*(forcing+forcing[:,::-1].conj())
    q=q0
    if a.start_field is not None:
        with np.load(a.start_field) as data:
            if data['radii'].shape != r.shape or not np.allclose(data['radii'],r,rtol=0,atol=1e-10):
                p.error('--start-field radial grid does not match this run')
            q=data['modes'].copy()
    elif a.start_lambda != 1.:
        p.error('--start-lambda other than 1 requires --start-field')
    if not 0 <= a.start_lambda <= 1:
        p.error('--start-lambda must lie in [0,1]')
    scale=max(np.max(abs(inlet)),np.max(abs(q)))
    pin=None
    if a.phase_pin:
        velocity=q[:,cfg.harmonics+1,:3*op.m].reshape(nr,3,op.m)
        energy=r*r*np.einsum('rvi,ij,rvj->r',velocity.conj(),op.grid.mass,velocity).real
        eligible=(r>r[0]) & (r<=a.sponge_start/abs(cfg.ro))
        pin=float(r[np.argmax(np.where(eligible,energy,-np.inf))])
    report={'arguments':{k:str(v) if isinstance(v,Path) else v for k,v in vars(a).items()},
            'config':asdict(cfg),'status':'running','initial_relative_residual':float(np.linalg.norm(residual,np.inf)/scale),
            'continuation':[]}
    started=perf_counter()
    backend=PardisoBackend()
    lam=a.start_lambda;step=a.initial_step
    try:
        while lam>0:
            trial=max(0.,lam-step);entry={'from':lam,'trial':trial,'step':step};print(json.dumps(entry),flush=True)
            try:
                result=problem.solve(inlet,initial=q,forcing=trial*forcing,tol=1e-9,max_iter=a.max_iter,
                    linear_solver='pardiso',pardiso_backend=backend,
                    phase_pin_radius=pin,
                    phase_update=a.phase_continuity,phase_continuity=a.phase_continuity,
                    progress=lambda it,res: print(f'lambda={trial:.6f} Newton {it}: {res:.6g}',flush=True))
            except RuntimeError as exc:
                entry.update(status='failed',error=str(exc));report['continuation'].append(entry)
                step*=.5
                if step<a.minimum_step:raise RuntimeError(f'homotopy minimum step reached after: {exc}')
                continue
            q=result.modes;lam=trial
            entry.update(status='converged',relative_residual=result.residual,iterations=result.iterations)
            report['continuation'].append(entry)
            np.savez_compressed(a.output/f'checkpoint_lambda_{lam:.6f}.npz',radii=r,modes=q,
                                config_json=json.dumps(asdict(cfg)))
            step=min(a.initial_step,step*1.5)
        energy=result.energies(cfg)
        np.savez_compressed(a.output/'fields.npz',radii=r,modes=q,config_json=json.dumps(asdict(cfg)))
        np.savetxt(a.output/'energy.csv',np.column_stack((abs(cfg.ro)*r,energy)),delimiter=',',
                   header='Re,'+','.join(f'E{j}' for j in range(1,cfg.harmonics+1)))
        report.update(status='converged_discrete_equations',relative_residual=result.residual,
                      elapsed_seconds=perf_counter()-started,final_lambda=lam)
    except Exception as exc:
        report.update(status='failed',error=str(exc),elapsed_seconds=perf_counter()-started,final_lambda=lam)
    finally:
        backend.close()
    (a.output/'metadata.json').write_text(json.dumps(report,indent=2)+'\n')
    print(json.dumps({k:v for k,v in report.items() if k!='continuation'},indent=2))
    return 0 if report['status']=='converged_discrete_equations' else 2


if __name__=='__main__':raise SystemExit(main())
