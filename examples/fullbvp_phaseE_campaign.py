"""Restartable temporal / saddle / pinch campaign. One BLAS thread per worker.

Every job has an input/code fingerprint, atomic JSON and its own DONE/FAILED.
Sparse results are searched maxima, never an assertion of spectral completeness.
"""
from __future__ import annotations
from fullbvp_phaseE_adapter import (ROOT, OUT, SOURCE_SHA, frozen_base, pair, write_json)
import argparse
import concurrent.futures
from dataclasses import replace
import hashlib
import json
import os
from pathlib import Path
import signal
import time
import traceback
import numpy as np
from scipy import linalg
from llsa.secondary.dns_floquet import DNSFloquetProblem
from llsa.secondary.floquet import (secondary_spectrum, secondary_saddle, secondary_pinch)


def code_hash():
    files = [Path(__file__), Path(__file__).with_name('fullbvp_phaseE_adapter.py')]
    files += [ROOT/'src/llsa/secondary'/name for name in
              ('floquet.py','dns_floquet.py','operators.py','discretization.py','time_domain.py')]
    return hashlib.sha256(b''.join(p.read_bytes() for p in files)).hexdigest()


def job_key(config):
    return hashlib.sha256(json.dumps(dict(config=config,code=code_hash(),source=SOURCE_SHA),
                                    sort_keys=True).encode()).hexdigest()[:20]


def make_problem(config):
    b = frozen_base(config['R'], n=config['N'], H=config['H'], zmax=config['zmax'])
    if config.get('amplitude', 1.) != 1.:
        # Diagnostic homotopy of all corrections, not a family of equilibria.
        b = replace(b, modes=b.modes*config['amplitude'])
    return DNSFloquetProblem(b, beta_hat=config['beta_hat'], harmonics=config['K'])


def mode_record(problem, omega, vector, residual):
    q = vector.reshape(2*problem.K+1,4,problem.base.operators.m)[:,:3]
    e = np.einsum('hvi,ij,hvj->h', q.conj(),problem.base.operators.grid.mass,q).real
    e = np.maximum(e,0); e /= max(e.sum(),1e-30)
    return dict(omega=pair(omega), growth=float(omega.imag), residual=float(residual),
                edge_energy=float(e[0]+e[-1]), harmonic_energy=e.tolist(),
                dominant_harmonic=int(np.argmax(e))-problem.K)


def temporal(config):
    p = make_problem(config)
    ah = config['alpha_hat']
    if ah == 0 and config['beta_hat'] == 0:
        raise ValueError('The exactly homogeneous sideband has a pressure nullspace; use +/-1e-4')
    records, vectors, failures = [], [], []
    dense_diagnostics=None
    if config.get('dense',False):
        A, B = p.matrix(ah).toarray(), p.B.toarray()
        homogeneous, vec = linalg.eig(A,B,check_finite=False,homogeneous_eigvals=True)
        numer,denom=homogeneous
        # Descriptor systems have infinite pressure eigenvalues. Testing only
        # isfinite(numer/denom) mislabels roundoff denominators as huge roots.
        keep=np.flatnonzero(abs(denom)>1e-10*np.hypot(abs(numer),abs(denom)))
        vals,vec = numer[keep]/denom[keep],vec[:,keep]
        # Use a declared physical frequency window, and reject the algebraic
        # pressure subspace. Descriptor infinities can otherwise masquerade as
        # enormous finite eigenvalues even with tiny backward residuals.
        physical=vals*p.radius
        mass_fraction=np.linalg.norm(B@vec,axis=0)/np.maximum(np.linalg.norm(vec,axis=0),1e-30)
        real_limit,imag_limit=config.get('dense_window',(250.,100.))
        resolved=(mass_fraction>1e-8)&(abs(physical.real)<=real_limit)&(abs(physical.imag)<=imag_limit)
        dense_diagnostics=dict(generalized_finite_candidates=len(vals),
            rejected_outside_window_or_pressure=int(np.sum(~resolved)),
            window=dict(real_abs_max=real_limit,imag_abs_max=imag_limit),mass_fraction_min=1e-8)
        vals,vec=vals[resolved],vec[:,resolved]
        residual = np.linalg.norm(A@vec-(B@vec)*vals,axis=0)/(
            (linalg.norm(A)+abs(vals)*linalg.norm(B))*np.linalg.norm(vec,axis=0))
        for j in np.flatnonzero(residual < 1e-8):
            records.append(mode_record(p,vals[j]*p.radius,vec[:,j],residual[j]))
            vectors.append(vec[:,j])
    else:
        for shift in config['shifts']:
            try:
                spec = secondary_spectrum(p,ah,target=complex(*shift),count=config.get('count',16))
                for j,w in enumerate(spec.omega):
                    if any(abs(w-complex(*row['omega'])) < 1e-6 for row in records):
                        continue
                    records.append(mode_record(p,w,spec.vectors[:,j],spec.residuals[j]))
                    vectors.append(spec.vectors[:,j])
            except Exception as exc:
                failures.append(dict(shift=shift,error=str(exc)))
    if not records:
        raise RuntimeError(f'No accepted temporal eigenvalues: {failures}')
    order = np.argsort([row['growth'] for row in records])[::-1]
    result = dict(base=p.base.metadata, config=config,
                  modes=[records[j] for j in order], shift_failures=failures,
                  search_complete=False,dense_diagnostics=dense_diagnostics,
                  coverage='resolved dense eigenvalues in declared frequency window' if config.get('dense',False)
                           else 'union of declared shift-invert searches; searched maximum only')
    # Save a few unstable branches for saddle seeds; tail modes remain visible.
    return result, dict(vectors=np.stack([vectors[j] for j in order[:8]],axis=1),
                        omega=np.array([complex(*records[j]['omega']) for j in order[:8]]))


def absolute(config):
    p = make_problem(config)
    source = OUT/'jobs'/config['temporal_job']
    with np.load(source/'vectors.npz') as data:
        q = data['vectors'][:,config['branch']]
        target = data['omega'][config['branch']]
    saddle = secondary_saddle(p,alpha_hat=complex(config['alpha_hat'],config['seed_imag']),
        target=target,reference_vector=q,count=16,max_iter=45,max_step=.02,tol=1e-6)
    result = dict(config=config,alpha_hat=pair(saddle.alpha_hat),omega_hat=pair(saddle.omega_hat),
        group_velocity=pair(saddle.group_velocity),converged=bool(saddle.converged),
        residual=saddle.residual,history=saddle.history,pinch=None,pinch_error=None)
    if saddle.converged:
        try:
            result['pinch'] = secondary_pinch(p,saddle,steps=config.get('pinch_steps',65),
                                            eta_min=1e-5,eta_max=config.get('eta_max',16.))
        except Exception as exc:
            result['pinch_error'] = str(exc)
    result['positive_finite_path_pinch'] = bool(saddle.converged and saddle.omega_hat.imag > 0
        and result['pinch'] and result['pinch']['opposite_half_planes_at_end'])
    return result, dict(vector=saddle.vector)


def execute(config):
    key = job_key(config); dest = OUT/'jobs'/key
    if (dest/'DONE').exists() and (dest/'result.json').exists() and (dest/'vectors.npz').exists():
        return key
    dest.mkdir(parents=True, exist_ok=True)
    write_json(dest/'request.json',dict(config=config,code_sha256=code_hash(),source_sha256=SOURCE_SHA))
    start = time.monotonic()
    try:
        def expired(signum,frame):
            raise TimeoutError('Per-job wall-time limit exceeded; restart or revise this job')
        signal.signal(signal.SIGALRM,expired)
        signal.alarm(900 if config['kind']=='absolute' else 180)
        result,arrays = (absolute(config) if config['kind']=='absolute' else temporal(config))
        signal.alarm(0)
        result.update(seconds=time.monotonic()-start,job=key)
        with (dest/'vectors.tmp').open('wb') as stream:
            np.savez_compressed(stream,**arrays)
        (dest/'vectors.tmp').replace(dest/'vectors.npz')
        write_json(dest/'result.json',result)
        (dest/'FAILED').unlink(missing_ok=True)
        (dest/'DONE').write_text('DONE\n')
    except Exception:
        signal.alarm(0)
        (dest/'FAILED').write_text(traceback.format_exc())
        raise
    return key


def run_jobs(configs,workers):
    keys,errors = [],[]
    with concurrent.futures.ProcessPoolExecutor(max_workers=workers) as pool:
        futures = {pool.submit(execute,c):c for c in configs}
        for future in concurrent.futures.as_completed(futures):
            try:
                keys.append(future.result())
            except Exception as exc:
                errors.append(dict(config=futures[future],error=str(exc)))
    return keys,errors


def crossings(rows, key):
    out=[]
    for left,right in zip(rows,rows[1:]):
        if left[key] <= 0 < right[key]:
            fraction=-left[key]/(right[key]-left[key])
            out.append(dict(bracket=[left['R'],right['R']],
                            linear_estimate=left['R']+fraction*(right['R']-left['R'])))
    return out


def summarize(keys, output):
    groups={}; absolute_rows=[]
    for key in keys:
        data=json.loads((OUT/'jobs'/key/'result.json').read_text());c=data['config']
        if c['kind']=='absolute':
            absolute_rows.append(data);continue
        group=(c['N'],c['H'],c['K'],c['zmax'],c['beta_hat'],c.get('amplitude',1.))
        groups.setdefault(group,{}).setdefault(c['R'],[]).append(data)
    result=[]
    for group,radii in sorted(groups.items()):
        rows=[]
        for R,data in sorted(radii.items()):
            best=max(data,key=lambda x:x['modes'][0]['growth']);mode=best['modes'][0]
            alpha_extent=max(abs(x['config']['alpha_hat']) for x in data)
            rows.append(dict(R=R,alpha_hat=best['config']['alpha_hat'],**mode,job=best['job'],
                maximum_at_search_boundary=bool(abs(best['config']['alpha_hat'])>=alpha_extent-1e-12),
                alpha_samples=len(data),shift_failures=sum(len(x['shift_failures']) for x in data)))
        result.append(dict(N=group[0],H=group[1],K=group[2],zmax=group[3],beta_hat=group[4],
            amplitude=group[5],sector_allowed=group[4]==0,rows=rows,
            zero_crossings=crossings(rows,'growth'),
            onset_status='left_censored' if rows[0]['growth']>0 else 'bracketed' if crossings(rows,'growth') else 'not_found',
            onset_upper_bound=rows[0]['R'] if rows[0]['growth']>0 else None))
    write_json(output,dict(groups=result,absolute=absolute_rows,
        interpretation='Positive Floquet growth may continue a laminar primary instability; no secondary-induced onset inferred without branch attribution.'))
    return result


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--stage',choices=('pilot','production','saddles','convergence','refine','absolute_refine'),required=True)
    parser.add_argument('--workers',type=int,default=8)
    args=parser.parse_args()
    if not 1<=args.workers<=32:parser.error('workers must be in 1..32')
    config=dict(kind='temporal',N=32,H=6,K=6,zmax=30.,beta_hat=0.,
        shifts=[[float(w),growth] for w in range(-160,161,20) for growth in (4.,12.)],count=16,amplitude=1.)
    radii=list(range(560,601,5));alphas=np.r_[np.linspace(-.6,-.05,12),-1e-4,1e-4,np.linspace(.05,.6,12)]
    configs=[]
    if args.stage=='pilot':
        config.update(N=32,K=3,shifts=[[float(w),3.] for w in (-80,-40,0,40,80)],count=12)
        alphas=np.r_[np.linspace(-.4,-.1,4),-1e-4,1e-4,np.linspace(.1,.4,4)]
        configs=[dict(config,R=float(r),alpha_hat=float(a)) for r in radii for a in alphas]
        # Laminar controls at the two ends expose inherited primary instabilities.
        configs += [dict(config,R=float(r),alpha_hat=float(a),amplitude=0.)
                    for r in (560,600) for a in alphas]
    elif args.stage=='production':
        configs=[dict(config,R=float(r),alpha_hat=float(a),beta_hat=bh)
                 for r in radii for bh in (0.,17.,68/3,34.) for a in alphas]
        configs += [dict(config,R=float(r),alpha_hat=float(a),amplitude=0.) for r in radii for a in alphas]
    else:
        manifest=json.loads((OUT/'production_manifest.json').read_text())
        temporal_keys=manifest['jobs']
        if (OUT/'refine_manifest.json').exists() and args.stage!='refine':
            temporal_keys += json.loads((OUT/'refine_manifest.json').read_text())['jobs']
        groups=summarize(temporal_keys,OUT/'production_summary.json')
        if args.stage=='saddles':
            for g in groups:
                if g['amplitude']!=1. or g['beta_hat']!=0:continue
                for row in g['rows']:
                    data=json.loads((OUT/'jobs'/row['job']/'result.json').read_text())
                    for branch in range(min(3,len(data['modes']))):
                        for imag in (0.,-.04):
                            configs.append(dict(data['config'],kind='absolute',temporal_job=row['job'],
                                branch=branch,seed_imag=imag,pinch_steps=65,eta_max=16.))
        elif args.stage=='absolute_refine':
            sm=json.loads((OUT/'saddles_manifest.json').read_text())
            qualified={}
            for key in sm['jobs']:
                s=json.loads((OUT/'jobs'/key/'result.json').read_text())
                if not s['converged'] or not s['pinch'] or not s['pinch']['opposite_half_planes_at_end']:continue
                r=s['config']['R']
                if r not in qualified or s['omega_hat'][1]>qualified[r]['omega_hat'][1]:qualified[r]=s
            ordered=sorted(qualified)
            for lo,hi in zip(ordered,ordered[1:]):
                if hi-lo>5.01:continue
                left,right=qualified[lo],qualified[hi]
                if not left['omega_hat'][1]<=0<right['omega_hat'][1]:continue
                for r in np.arange(lo+1,hi,1):
                    tc=dict(config,R=float(r),alpha_hat=right['alpha_hat'][0])
                    tk=execute(tc)
                    with np.load(OUT/'jobs'/tk/'vectors.npz') as d:
                        branch=int(np.argmin(abs(d['omega']-complex(*right['omega_hat']))))
                    configs.append(dict(tc,kind='absolute',temporal_job=tk,branch=branch,
                        seed_imag=right['alpha_hat'][1],pinch_steps=85,eta_max=24.))
        elif args.stage=='convergence':
            for g in groups:
                if g['amplitude']!=1.:continue
                for row in g['rows']:
                    if row['R'] not in (560,580,585,600):continue
                    best=json.loads((OUT/'jobs'/row['job']/'result.json').read_text())['config']
                    for update in (dict(N=48),dict(H=4),dict(K=8),dict(zmax=40.),dict(dense=True)):
                        configs.append(dict(best,**update))
                    for amp in (.25,.5,.75):configs.append(dict(best,amplitude=amp))
            # Absolute N/H/K/height sensitivity must include pinch, not only temporal roots.
            sm=OUT/'saddles_manifest.json'
            if sm.exists():
                for key in json.loads(sm.read_text())['jobs']:
                    result=json.loads((OUT/'jobs'/key/'result.json').read_text())
                    if not result['converged'] or result['config']['seed_imag']!=0.:continue
                    c=result['config'];seed_temporal=json.loads((OUT/'jobs'/c['temporal_job']/'result.json').read_text())['config']
                    for update in (dict(N=48),dict(H=4),dict(K=8),dict(zmax=40.)):
                        tc=dict(seed_temporal,**update)
                        tk=execute(tc)
                        # Fresh eigenspace has different size; seed its nearest frequency.
                        with np.load(OUT/'jobs'/tk/'vectors.npz') as d:
                            branch=int(np.argmin(abs(d['omega']-complex(*result['omega_hat']))))
                        configs.append(dict(tc,kind='absolute',temporal_job=tk,branch=branch,
                            seed_imag=result['alpha_hat'][1],alpha_hat=result['alpha_hat'][0],
                            pinch_steps=85,eta_max=24.))
        else:
            for g in groups:
                if g['amplitude']!=1.:continue
                for row in g['rows']:
                    best=json.loads((OUT/'jobs'/row['job']/'result.json').read_text())['config']
                    for a in np.linspace(row['alpha_hat']-.05,row['alpha_hat']+.05,11):
                        if abs(a)<1e-8 and g['beta_hat']==0:continue
                        configs.append(dict(best,alpha_hat=float(a)))
                for bracket in g['zero_crossings']:
                    for r in np.arange(bracket['bracket'][0]+1,bracket['bracket'][1],1):
                        configs += [dict(config,R=float(r),beta_hat=g['beta_hat'],alpha_hat=float(a)) for a in alphas]
    # Deduplicate before submitting; no two workers can write the same job.
    configs=list({job_key(c):c for c in configs}.values())
    start=time.monotonic();keys,errors=run_jobs(configs,args.workers)
    write_json(OUT/f'{args.stage}_manifest.json',dict(stage=args.stage,jobs=sorted(keys),errors=errors,
        requested=len(configs),workers=args.workers,threads_per_worker=1,seconds=time.monotonic()-start))
    groups=summarize(keys,OUT/f'{args.stage}_summary.json')
    print(json.dumps(dict(stage=args.stage,done=len(keys),failed=len(errors),seconds=time.monotonic()-start)))
    for g in groups:
        print('N/H/K/z/beta/amplitude',*[g[k] for k in ('N','H','K','zmax','beta_hat','amplitude')],
              'growth',[(row['R'],round(row['growth'],5)) for row in g['rows']])
    if errors:raise SystemExit(1)


if __name__=='__main__':main()
