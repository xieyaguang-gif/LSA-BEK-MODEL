#!/usr/bin/env python3
"""D2e: corrected coarse stationary fields and tightly certified local spectra.

This is a resolution study, not an N32/H6 certification or a full-spectrum test.
Campaign execution is intentionally separate from the bounded validator.
"""
from __future__ import annotations
import os
for key in ('OPENBLAS_NUM_THREADS','OMP_NUM_THREADS','MKL_NUM_THREADS','NUMEXPR_NUM_THREADS','BLIS_NUM_THREADS'):
    os.environ[key]='1'
os.environ['PYTHONDONTWRITEBYTECODE']='1'
import argparse
import fcntl
import json
import resource
import signal
import subprocess
import sys
import time
from pathlib import Path
import numpy as np
from scipy import sparse
from scipy.sparse.linalg import eigs, ArpackNoConvergence
import fullbvp_phaseD2d_precision as d2
import fullbvp_phaseB_solver as base
from fullbvp_phaseD2e_profile import TimedFactor

ROOT, OUT=d2.ROOT,d2.OUT
PAIR_ACCEPT=1e-8
LINEAR_ACCEPT=1e-10
STATIONARY_ACCEPT=1e-10
MEMORY_BYTES=100_000_000_000
POINT_BUDGET=4800.
TARGET_SECONDS=3600.
# Only this process sees the stricter inherited inverse gate; no source edits.
d2.ACCEPT=LINEAR_ACCEPT


def output_path(path):
    path=Path(path).absolute()
    if '..' in path.parts:
        raise ValueError('lexical traversal is forbidden')
    canonical=path.resolve()
    if canonical != path or any(p.is_symlink() for p in (path,*path.parents)):
        raise ValueError('symlink output alias is forbidden')
    if not any(canonical.is_relative_to(OUT/name) and canonical != OUT/name
               for name in ('D2e','production_v4')):
        raise ValueError('output must be a proper descendant of D2e or production_v4')
    return canonical


def write(path,value):
    path=output_path(path)
    d2.v1.write(path,value)


def code_fingerprints():
    paths=[Path(__file__),ROOT/'examples/fullbvp_phaseD2e_profile.py',
           OUT/'production_v4/commands.sh']
    return dict(d2.fingerprints(),**{str(p.resolve()):d2.v1.sha(p) for p in paths})


def parameters(source,n=24,H=4,dr=1.,shifts=(.05+0j,.05+2j),k=6,ncv=40):
    if source not in d2.v1.SOURCES or n not in (24,32) or H not in (4,6) or dr not in (.5,1.):
        raise ValueError('unsupported controlled resolution study')
    report=json.loads((d2.v1.SOURCES[source]/'summary.json').read_text())
    if not report.get('accepted'):
        raise ValueError('unaccepted source')
    intervals=int(round((report['Rout']-400.)/dr))
    return dict(source=source,n=n,H=H,intervals=intervals,requested_dr=dr,
                actual_dr=(report['Rout']-400.)/intervals,shifts=[d2.v1.zpair(s) for s in shifts],
                k=k,ncv=ncv,threads=16,budget_seconds=POINT_BUDGET,target_seconds=TARGET_SECONDS,
                memory_bytes=MEMORY_BYTES,stationary_accept=STATIONARY_ACCEPT,
                pair_accept=PAIR_ACCEPT,linear_accept=LINEAR_ACCEPT,
                code_sha256=code_fingerprints(),
                source_hashes={p.name:d2.v1.sha(p) for p in [d2.v1.SOURCES[source]/'fields.npz',
                                                           d2.v1.SOURCES[source]/'summary.json']})


class GuardedFactor(d2.PardisoComplex):
    """Reject a numerical factorization before it exceeds the memory envelope."""
    def _call(self,phase,rhs):
        if phase==22:
            rss=d2.memory().get('VmRSS',0)
            estimate=max(0,int(self.iparm[16]))*1024
            projected=rss+int(1.25*estimate)
            self.info['D2e_memory_guard']=dict(rss_bytes=rss,numerical_estimate_bytes=estimate,
                                              projected_bytes=projected,cap_bytes=MEMORY_BYTES)
            if projected>MEMORY_BYTES:
                if self.stats_path:write(self.stats_path,self.info)
                raise MemoryError('D2e numerical-factor estimate + RSS exceeds 100 GB')
        result=super()._call(phase,rhs)
        if phase!=-1 and d2.memory().get('VmRSS',0)>MEMORY_BYTES:
            raise MemoryError('D2e process RSS exceeds 100 GB')
        return result


class StrictCorrector(base.Corrector):
    def metrics(self,y):
        row=super().metrics(y)
        row['accepted']=bool(row['accepted'] and row['relative_field_inf']<1e-11)
        return row


def stationarity_gate(temporal):
    diag=temporal.diagnostics()
    if (not np.isfinite(diag['relative_field_inf']) or diag['relative_field_inf']>STATIONARY_ACCEPT
        or not np.isfinite(diag['phase_neutral_rate_residual']) or diag['phase_neutral_rate_residual']>1e-6):
        raise RuntimeError('coarse field fails original stationarity/phase-neutral gate: '+str(diag))
    return diag


def branch_gate(before,after):
    ratio=float(after['A1']/max(before['A1'],1e-300))
    frequency_change=float(after['omega']-before['omega'])
    # Necessary continuity diagnostics, not a proof of branch identity near a fold.
    accepted=bool(np.isfinite(ratio) and .2<ratio<5 and abs(frequency_change)<2.)
    record=dict(amplitude_ratio=ratio,omega_change=frequency_change,accepted=accepted,
                branch_identity_proven=False,
                limitation='fixed-Rout coarse Newton may change branch; continuation/matching still required')
    if not accepted:
        raise RuntimeError('coarse correction lost nonzero source branch: '+str(record))
    return record


def corrected_temporal(params,out,deadline):
    mapping=d2.v1.Mapping(n=params['n'],H=params['H'],intervals=params['intervals'])
    source=d2.v1.SOURCES[params['source']]
    source_summary=json.loads((source/'summary.json').read_text())
    y=base.transfer(source/'fields.npz',mapping)
    initial=y.copy()
    projected=d2.v1.unpack(y[:-2],mapping.nr,mapping.H,mapping.s)
    p,*_=mapping.at(y[-1],y[-2])
    before=dict(A1=float(np.sqrt(base.energies(p,projected)[1])),omega=float(y[-2]),
                energies=base.energies(p,projected).tolist())
    fingerprint=d2.v1.digest(params)
    cached=out/'corrected.json'
    if cached.exists():
        saved=json.loads(cached.read_text())
        fields=output_path(saved['fields'])
        if saved['fingerprint']!=fingerprint or d2.v1.sha(fields)!=saved['fields_sha256']:
            raise RuntimeError('corrected-state fingerprint/hash mismatch')
        with np.load(fields) as data:
            q=data['field'].copy(); R=float(data['Rout']); omega=float(data['omega'])
        temporal=d2.v1.Temporal(mapping,q,R,omega,deadline)
        saved['diagnostics']=stationarity_gate(temporal)
        saved['branch']=branch_gate(before,dict(A1=float(np.sqrt(base.energies(temporal.p,q)[1])),omega=omega))
        return temporal,saved
    attempts=out/'corrections';attempts.mkdir(exist_ok=True)
    attempt=attempts/f'attempt_{len(list(attempts.iterdir())):04d}'
    old=base.Corrector
    try:
        base.Corrector=StrictCorrector
        y,record=base.correct(mapping,projected[:,mapping.H+1],y,'Rout',float(y[-1]),attempt,
                              budget=max(1.,deadline.end-time.monotonic()),max_newton=12)
    finally:
        base.Corrector=old
    if not record.get('accepted'):
        raise RuntimeError('coarse stationarity correction failed; spectrum forbidden: '+str(record.get('failure')))
    q=d2.v1.unpack(y[:-2],mapping.nr,mapping.H,mapping.s)
    temporal=d2.v1.Temporal(mapping,q,float(y[-1]),float(y[-2]),deadline)
    diag=stationarity_gate(temporal)
    branch=branch_gate(before,record)
    fields=attempt/'fields.npz'
    saved=dict(fingerprint=fingerprint,fields=str(fields),fields_sha256=d2.v1.sha(fields),
               source={k:source_summary.get(k) for k in ('A1','omega','energies','Rout','n','H','intervals')},
               before=before,after={k:record[k] for k in ('A1','omega','energies','Rout')},
               projected_to_corrected_relative=float(np.linalg.norm(y[:-2]-initial[:-2])/max(np.linalg.norm(initial[:-2]),1e-300)),
               branch=branch,diagnostics=diag,correction_seconds=record['seconds'],
               original_resolution_certified=False)
    write(cached,saved)
    return temporal,saved


def run_center(temporal,J,sigma,out,fingerprint,deadline,k=6,ncv=40,backend='pardiso'):
    out=output_path(out);out.mkdir(parents=True,exist_ok=True)
    done=out/'DONE.json'
    if done.exists():
        result=json.loads(done.read_text())
        arrays=output_path(result['arrays'])
        if (result['fingerprint']!=fingerprint or result['sigma']!=d2.v1.zpair(sigma)
            or result['k']!=k or result['ncv']!=ncv or d2.v1.sha(arrays)!=result['arrays_sha256']):
            raise RuntimeError('center resume fingerprint/hash mismatch')
        with np.load(arrays) as data:
            checks=[d2.v1.certify(temporal.J,temporal.B,max(temporal.b),v,x,tol=PAIR_ACCEPT)
                    for v,x in zip(data['values'],data['vectors'].T)]
        if len(checks)!=k or not all(r['accepted'] for r in checks):
            raise RuntimeError('cached eigenpairs fail original recertification')
        return result
    attempts=out/'attempts';attempts.mkdir(exist_ok=True)
    attempt=attempts/f'attempt_{len(list(attempts.iterdir())):04d}';attempt.mkdir()
    start=time.monotonic();factor=None;inverse=None
    values=np.empty(0,complex);vectors=np.empty((J.shape[0],0),complex)
    result=dict(fingerprint=fingerprint,sigma=d2.v1.zpair(sigma),complete=False,k=k,ncv=ncv,
                original_resolution_certified=False,exhaustive_stability=False,
                residual_accept=PAIR_ACCEPT,linear_accept=LINEAR_ACCEPT)
    try:
        K=(J.astype(complex)+sigma*sparse.diags(temporal.b)).tocsr()
        factor=TimedFactor(GuardedFactor(K,threads=16,deadline=deadline,stats_path=attempt/'factor.json')
                           if backend=='pardiso' else d2.ScipyComplex(K,deadline=deadline))
        inverse=d2.PrecisionInverse(temporal.J,temporal.B,sigma,factor,deadline,
                                   attempt/'inverse.jsonl',assembled_K=K)
        rng=np.random.default_rng(20261030)
        seed=rng.normal(size=J.shape[0])+1j*rng.normal(size=J.shape[0])
        mu,vectors=eigs(inverse.T,k=k,ncv=ncv,which='LM',tol=1e-10,maxiter=100,v0=seed)
        values=sigma+1/mu
        result['eigenpairs']=[d2.v1.certify(temporal.J,temporal.B,max(temporal.b),v,x,
                              tol=PAIR_ACCEPT,phase=temporal.neutral()) for v,x in zip(values,vectors.T)]
        result['complete']=bool(len(values)==k and all(r['accepted'] for r in result['eigenpairs']))
        if not result['complete']:result['failure']='one or more original eigenpairs rejected'
    except Exception as exc:
        result['failure']=f'{type(exc).__name__}: {exc}'
    finally:
        if factor is not None:
            try:factor.close()
            except Exception as exc:result.update(complete=False,failure='factor cleanup: '+str(exc))
            result.update(factor=dict(factor.info),timing=factor.summary())
        result['arnoldi_callbacks']=0 if inverse is None else inverse.calls
        log=attempt/'inverse.jsonl'
        rows=[json.loads(s) for s in log.read_text().splitlines()] if log.exists() else []
        result.update(direct_corrections=sum(r['direct_corrections'] for r in rows),
                      gmres_calls=sum(r['krylov_used'] for r in rows),
                      gmres_iterations=sum(r.get('krylov_iterations',0) for r in rows),
                      gmres_preconditioner_calls=sum(r['krylov_preconditioner_calls'] for r in rows),
                      seconds=time.monotonic()-start,memory=d2.memory())
        arrays=attempt/'eigenpairs.npz';d2.v1.save(arrays,values=values,vectors=vectors)
        result.update(arrays=str(arrays),arrays_sha256=d2.v1.sha(arrays))
        write(attempt/'result.json',result)
        write(out/('DONE.json' if result['complete'] else 'FAILED.json'),result)
        if result['complete']:(out/'FAILED.json').unlink(missing_ok=True)
    return result


def worker(job):
    out=output_path(job['out']);params=job['params']
    current=parameters(params['source'],params['n'],params['H'],params['requested_dr'],
                       tuple(complex(*x) for x in params['shifts']),params['k'],params['ncv'])
    if current!=params:raise RuntimeError('worker code/source/parameter fingerprint changed')
    # A conservative address-space cap enforces the RSS ceiling as well.
    resource.setrlimit(resource.RLIMIT_AS,(params['memory_bytes'],params['memory_bytes']))
    deadline=d2.v1.Deadline(job['remaining'])
    start=time.monotonic()
    temporal,correction=corrected_temporal(params,out,deadline)
    J,assembly=d2.assemble(temporal,deadline);write(out/'assembly.json',assembly)
    results=[]
    fingerprint=d2.v1.digest(params)
    for i,pair in enumerate(params['shifts']):
        deadline.check()
        result=run_center(temporal,J,complex(*pair),out/f'center_{i:02d}',fingerprint,
                          deadline,k=params['k'],ncv=params['ncv'])
        results.append(result)
        if not result['complete']:break
    complete=len(results)==len(params['shifts']) and all(r['complete'] for r in results)
    result=dict(fingerprint=fingerprint,complete=complete,seconds=time.monotonic()-start,
                correction=correction,centers=results,original_resolution_certified=False,
                exhaustive_stability=False,perturbations='integer multiples of beta=68 only',
                peak_rss_bytes=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss*1024)
    write(out/'worker_result.json',result)
    return 0 if complete else 1


class Interrupted(RuntimeError):pass


def verify_worker_evidence(params,out,child):
    """Follow the persisted certificate hashes through to every numerical file."""
    fingerprint=d2.v1.digest(params)
    if not child.get('complete') or child.get('fingerprint')!=fingerprint:
        raise RuntimeError('worker certificate is incomplete or belongs to another point')
    correction=child['correction']
    fields=output_path(correction['fields'])
    if (not fields.is_relative_to(out/'corrections') or not fields.is_file()
        or correction['fingerprint']!=fingerprint
        or d2.v1.sha(fields)!=correction['fields_sha256']):
        raise RuntimeError('corrected-state evidence hash/path mismatch')
    cached=json.loads((out/'corrected.json').read_text())
    if any(cached.get(k)!=correction.get(k) for k in ('fingerprint','fields','fields_sha256')):
        raise RuntimeError('corrected-state manifest mismatch')
    centers=child['centers']
    if len(centers)!=len(params['shifts']):
        raise RuntimeError('incomplete center evidence')
    for i,center in enumerate(centers):
        center_dir=out/f'center_{i:02d}'
        arrays=output_path(center['arrays'])
        if (not center.get('complete') or center.get('fingerprint')!=fingerprint
            or center.get('sigma')!=params['shifts'][i]
            or center.get('k')!=params['k'] or center.get('ncv')!=params['ncv']
            or not arrays.is_relative_to(center_dir/'attempts') or not arrays.is_file()
            or d2.v1.sha(arrays)!=center['arrays_sha256']):
            raise RuntimeError('center evidence hash/path/configuration mismatch')
        if json.loads((center_dir/'DONE.json').read_text())!=center:
            raise RuntimeError('center DONE differs from worker certificate')
        pairs=center.get('eigenpairs',[])
        if len(pairs)!=params['k'] or not all(r.get('accepted') and
            np.isfinite(r.get('original_relative',np.nan)) and
            r['original_relative']<=PAIR_ACCEPT for r in pairs):
            raise RuntimeError('center eigenpair certificates fail the required tolerance')


def launch_point(params,out,command=None,lock_fd=None):
    """Own one process group. Atomic ledger charges interrupted reservations in full."""
    out=output_path(out);out.mkdir(parents=True,exist_ok=True)
    fingerprint=d2.v1.digest(params)
    manifest=out/'manifest.json'
    if manifest.exists():
        if json.loads(manifest.read_text())['fingerprint']!=fingerprint:
            raise RuntimeError('point fingerprint changed; choose a new output directory')
    else:write(manifest,dict(fingerprint=fingerprint,params=params))
    done=out/'DONE.json'
    if done.exists():
        result=json.loads(done.read_text());worker_file=out/'worker_result.json'
        if (result['fingerprint']!=fingerprint or not result.get('complete')
            or not worker_file.exists() or result['worker_result_sha256']!=d2.v1.sha(worker_file)):
            raise RuntimeError('invalid point DONE/hash')
        verify_worker_evidence(params,out,json.loads(worker_file.read_text()))
        return result
    ledger=out/'budget.json';active=out/'ACTIVE.json'
    used=json.loads(ledger.read_text())['used_seconds'] if ledger.exists() else 0.
    if active.exists():
        used+=json.loads(active.read_text())['reserved_seconds']
        write(ledger,dict(used_seconds=used,crash_reservation_charged=True));active.unlink()
    remaining=params['budget_seconds']-used
    if remaining<=0:
        result=dict(fingerprint=fingerprint,complete=False,failure='cumulative point budget exhausted',used_seconds=used)
        write(out/'FAILED.json',result);return result
    job=dict(params=params,out=str(out),remaining=remaining)
    write(out/'job.json',job)
    write(active,dict(reserved_seconds=remaining,started_unix=time.time()))
    command=command or [sys.executable,str(Path(__file__).resolve()),'worker','--job',str(out/'job.json')]
    start=time.monotonic();proc=None;failure=None
    def interrupted(signum,frame):raise Interrupted(f'signal {signum}')
    previous={s:signal.signal(s,interrupted) for s in (signal.SIGTERM,signal.SIGINT)}
    try:
        with (out/'worker.log').open('ab') as log:
            proc=subprocess.Popen(command,stdout=log,stderr=subprocess.STDOUT,start_new_session=True,cwd=ROOT,
                                  pass_fds=() if lock_fd is None else (lock_fd,))
            proc.wait(timeout=remaining)
            if proc.returncode:failure=f'worker exit {proc.returncode}'
    except BaseException as exc:
        failure=f'{type(exc).__name__}: {exc}'
        if proc is not None and proc.poll() is None:
            os.killpg(proc.pid,signal.SIGTERM)
            try:proc.wait(timeout=5)
            except subprocess.TimeoutExpired:
                os.killpg(proc.pid,signal.SIGKILL);proc.wait()
    finally:
        elapsed=time.monotonic()-start;used+=elapsed
        write(ledger,dict(used_seconds=used,hard_budget_seconds=params['budget_seconds']))
        active.unlink(missing_ok=True)
        for s,handler in previous.items():signal.signal(s,handler)
    result_file=out/'worker_result.json'
    child=json.loads(result_file.read_text()) if result_file.exists() else {}
    complete=bool(failure is None and child.get('complete') and child.get('fingerprint')==fingerprint
                  and used<=params['budget_seconds'] and child.get('peak_rss_bytes',MEMORY_BYTES+1)<=MEMORY_BYTES)
    if complete:
        try:verify_worker_evidence(params,out,child)
        except Exception as exc:
            complete=False;failure=f'evidence verification: {type(exc).__name__}: {exc}'
    result=dict(fingerprint=fingerprint,complete=complete,used_seconds=used,target_seconds=TARGET_SECONDS,
                time_target_met=bool(complete and used<=TARGET_SECONDS),hard_budget_seconds=params['budget_seconds'],
                peak_rss_bytes=child.get('peak_rss_bytes'),original_resolution_certified=False,
                failure=None if complete else failure or 'worker result, memory or budget gate failed')
    if result_file.exists():result['worker_result_sha256']=d2.v1.sha(result_file)
    write(out/('DONE.json' if complete else 'FAILED.json'),result)
    (out/('FAILED.json' if complete else 'DONE.json')).unlink(missing_ok=True)
    return result


def campaign(args):
    out=output_path(args.out)
    config={'baseline':(24,4,1.),'harmonic':(24,6,1.),'wallnormal':(32,4,1.),'radial':(24,4,.5)}
    n,H,dr=config[args.suite]
    # Workers inherit this descriptor, retaining exclusion even if the parent dies.
    # Lock busy/path rejection causes no output writes.
    with (OUT/'.production.lock').open('r') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        point_params=[parameters(source,n,H,dr) for source in args.sources]
        settings=dict(suite=args.suite,sources=args.sources,points=point_params)
        fingerprint=d2.v1.digest(settings)
        out.mkdir(parents=True,exist_ok=True)
        manifest=out/'manifest.json'
        if manifest.exists():
            if json.loads(manifest.read_text()).get('fingerprint')!=fingerprint:
                raise RuntimeError('campaign configuration changed; choose a new run directory')
        else:write(manifest,dict(fingerprint=fingerprint,settings=settings))
        results=[];failure=None
        try:
            for source,params in zip(args.sources,point_params):
                result=launch_point(params,out/source,lock_fd=lock.fileno());results.append(result)
                if 'Interrupted:' in str(result.get('failure','')):break
        except BaseException as exc:
            failure=f'{type(exc).__name__}: {exc}'
        complete=(failure is None and len(results)==len(args.sources)
                  and all(r['complete'] for r in results))
        write(out/('DONE.json' if complete else 'FAILED.json'),
              dict(fingerprint=fingerprint,complete=complete,points=results,failure=failure))
        (out/('FAILED.json' if complete else 'DONE.json')).unlink(missing_ok=True)
        return 0 if complete else 1


def main():
    p=argparse.ArgumentParser(description=__doc__);sub=p.add_subparsers(dest='mode',required=True)
    w=sub.add_parser('worker');w.add_argument('--job',type=Path,required=True)
    c=sub.add_parser('campaign');c.add_argument('--out',type=Path,required=True)
    c.add_argument('--suite',choices=['baseline','harmonic','wallnormal','radial'],default='baseline')
    c.add_argument('--sources',nargs='+',choices=list(d2.v1.SOURCES),default=['lower719p948669_H4','upper720_H6'])
    args=p.parse_args()
    if args.mode=='worker':
        job=json.loads(output_path(args.job).read_text());return worker(job)
    return campaign(args)


if __name__=='__main__':raise SystemExit(main())
