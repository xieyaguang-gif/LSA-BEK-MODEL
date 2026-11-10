#!/usr/bin/env python3
"""Short D2b actual coupled-operator/direct-complex-LU scientific validations."""
from __future__ import annotations
import os
for key in ('OPENBLAS_NUM_THREADS','OMP_NUM_THREADS','MKL_NUM_THREADS','NUMEXPR_NUM_THREADS','BLIS_NUM_THREADS'):
    os.environ[key] = '1'
os.environ['PYTHONDONTWRITEBYTECODE'] = '1'
import argparse
import gc
import json
import signal
import subprocess
import sys
import time
from pathlib import Path
from types import SimpleNamespace
import numpy as np
from scipy import linalg, sparse
from scipy.sparse.linalg import LinearOperator
import fullbvp_phaseD2b_direct as d2


def require(value, message):
    if not value:
        raise AssertionError(message)


def relative(a,b):
    return float(np.linalg.norm(a-b)/max(np.linalg.norm(a),np.linalg.norm(b),1e-300))


def actual_case(n,H,intervals,seed=26):
    mapping=d2.v1.Mapping(n=n,H=H,intervals=intervals)
    rng=np.random.default_rng(seed)
    x=rng.normal(size=mapping.nr*(2*H+1)*mapping.s)*1e-4
    q=d2.v1.unpack(x,mapping.nr,H,mapping.s)
    temporal=d2.v1.Temporal(mapping,q,720.,52.7)
    J,assembly=d2.assemble(temporal,d2.v1.Deadline(90))
    return temporal,J,assembly,rng


def actual_checks(out,backend,threads):
    start=time.monotonic()
    temporal,J,assembly,rng=actual_case(16,2,8)
    x=rng.normal(size=J.shape[0])+1j*rng.normal(size=J.shape[0])
    action=relative(J@x,temporal.J@x)
    require(action<1e-10,'N16 full coupled assembly failed general complex action check')
    sigma=.002+5j
    K=J.astype(complex)+sigma*sparse.diags(temporal.b)
    factor=d2.factorize(K,backend,threads,d2.v1.Deadline(90),out/'N16_factor.json')
    try:
        y=factor.solve(x)
        inverse_error=relative(temporal.J@y+sigma*(temporal.B@y),x)
        require(inverse_error<=5e-8,'actual coupled direct solve residual too large')
        complex_linearity=relative(factor.solve(1j*x),1j*y)
        require(complex_linearity<1e-10,'actual direct solve is not complex-linear')
        info=dict(factor.info)
    finally:
        factor.close()
    reduced,Jr,small_assembly,_=actual_case(8,1,4)
    dense=linalg.eigvals(Jr.toarray(),-np.diag(reduced.b))
    finite=dense[np.isfinite(dense)&(abs(dense)<1000)]
    center=d2.run_center(reduced,Jr,.037+.113j,out/'actual_QZ_center','actual-QZ-v1',
                         d2.v1.Deadline(90),backend=backend,threads=threads,k=4,ncv=20,outer_maxiter=120)
    accepted=[r for r in center['eigenpairs'] if r['accepted']]
    distances=[float(np.min(abs(complex(*r['value'])-finite))) for r in accepted]
    require(center['complete'] and len(accepted)==4 and max(distances)<1e-6,
            'actual coupled shift-invert spectrum differs from independent QZ')
    return dict(seconds=time.monotonic()-start,N16_size=J.shape[0],N16_H=2,N16_intervals=8,
                assembly=assembly,general_complex_action_relative=action,
                nonreal_sigma=d2.v1.zpair(sigma),direct_original_relative=inverse_error,
                direct_complex_linearity_relative=complex_linearity,factor=info,
                QZ_size=Jr.shape[0],QZ_assembly=small_assembly,QZ_finite_modes=len(finite),
                QZ_max_distance=max(distances),QZ_distances=distances,
                actual_center=center)


def manufactured_checks(out,backend,threads):
    start=time.monotonic()
    rng=np.random.default_rng(31)
    L=linalg.block_diag(np.diag([0.,-.4,-.6,-1.]),
                       np.array([[.03,-5.],[5.,.03]]),
                       np.array([[-.2,-2.],[2.,-.2]]))
    mass=np.diag(np.linspace(.4,1.1,8))
    C,E=rng.normal(size=(8,4)),rng.normal(size=(4,8))
    jd=np.block([[C@E-mass@L,C],[E,np.eye(4)]])
    bd=linalg.block_diag(mass,np.zeros((4,4)))
    phase=np.r_[np.eye(8)[:,0],-E[:,0]]
    J=sparse.csr_matrix(jd)
    temporal=SimpleNamespace(J=LinearOperator((12,12),matvec=lambda x:jd@x,dtype=complex),
                             B=LinearOperator((12,12),matvec=lambda x:bd@x,dtype=complex),
                             b=np.diag(bd),neutral=lambda:phase)
    finite=linalg.eigvals(jd,-bd)
    finite=finite[np.isfinite(finite)]
    center_path=out/'manufactured_sideband'
    center=d2.run_center(temporal,J,.002+5j,center_path,'sideband-v1',d2.v1.Deadline(30),
                         backend=backend,threads=threads,k=3,ncv=10)
    accepted=[r for r in center['eigenpairs'] if r['accepted']]
    target=min(abs(complex(*r['value'])-(.03+5j)) for r in accepted)
    require(center['complete'] and target<1e-9,'positive complex sideband was not recovered')
    distances=[float(np.min(abs(complex(*r['value'])-finite))) for r in accepted]
    require(max(distances)<1e-9,'manufactured direct spectrum disagrees with QZ')
    # Completed checkpoint must be reusable without entering factorization.
    real_factorize=d2.factorize
    def fail_factory(*args,**kwargs):
        raise RuntimeError('intentional factorization failure')
    try:
        d2.factorize=fail_factory
        resumed=d2.run_center(temporal,J,.002+5j,center_path,'sideband-v1',d2.v1.Deadline(30),
                              backend=backend,threads=threads,k=3,ncv=10)
        require(resumed==center,'completed center did not resume unchanged')
        failure_dir=out/'recovered_center'
        failed=d2.run_center(temporal,J,.002+5j,failure_dir,'recovery-v1',d2.v1.Deadline(30),
                             backend=backend,threads=threads,k=3,ncv=10)
        require(not failed['complete'] and (failure_dir/'FAILED.json').exists(),'factor failure marker missing')
    finally:
        d2.factorize=real_factorize
    recovered=d2.run_center(temporal,J,.002+5j,failure_dir,'recovery-v1',d2.v1.Deadline(30),
                            backend=backend,threads=threads,k=3,ncv=10)
    require(recovered['complete'] and (failure_dir/'DONE.json').exists()
            and not (failure_dir/'FAILED.json').exists(),'failed center did not resume/recover')
    bad=SimpleNamespace(solve=lambda rhs:np.zeros_like(rhs))
    direct=d2.DirectInverse(temporal.J,temporal.B,.002+5j,bad,d2.v1.Deadline(30),
                             out/'bad_direct.jsonl')
    rejected=False
    try:
        direct.T@np.ones(12)
    except d2.v1.InnerFailure:
        rejected=True
    require(rejected,'bad direct result escaped original matrix-free residual rejection')
    nan_factor=SimpleNamespace(solve=lambda rhs:np.full_like(rhs,np.nan))
    nan_inverse=d2.DirectInverse(temporal.J,temporal.B,.002+5j,nan_factor,d2.v1.Deadline(30),
                                  out/'nonfinite_direct.jsonl')
    nan_rejected=False
    try:
        nan_inverse.T@np.ones(12)
    except d2.v1.InnerFailure:
        nan_rejected=True
    require(nan_rejected,'nonfinite direct residual escaped explicit rejection')
    json.loads((out/'nonfinite_direct.jsonl').read_text())
    def release_failure_factory(*args,**kwargs):
        factor=real_factorize(*args,**kwargs)
        close=factor.close
        def fail_close():
            close()
            raise RuntimeError('intentional release failure after actual deallocation')
        factor.close=fail_close
        return factor
    try:
        d2.factorize=release_failure_factory
        release=d2.run_center(temporal,J,.002+5j,out/'release_failed_center','release-v1',
                              d2.v1.Deadline(30),backend=backend,threads=threads,k=3,ncv=10)
        require(not release['complete'] and release.get('release_failure')
                and (out/'release_failed_center/FAILED.json').exists(),
                'factor release failure lost center failure evidence')
    finally:
        d2.factorize=real_factorize
    real_eigs=d2.eigs
    try:
        d2.eigs=lambda *a,**kw:(np.zeros(1,complex),np.ones((12,1),complex))
        invalid=d2.run_center(temporal,J,.002+5j,out/'invalid_center','invalid-v1',
                              d2.v1.Deadline(30),backend=backend,threads=threads,k=1,ncv=4)
        require(not invalid['complete'] and invalid['rejected_count']==1
                and not (out/'invalid_center/DONE.json').exists(),'mu=0 center improperly completed')
    finally:
        d2.eigs=real_eigs
    return dict(size=12,known_unstable_sidebands=[[.03,5.],[.03,-5.]],
                positive_complex_mode_distance=float(target),QZ_distances=distances,
                accepted=center['eigenpairs'],resumed_without_refactor=True,
                failed_center_recovered=True,bad_direct_rejected=True,nonfinite_direct_rejected=True,nonfinite_candidate_failed=True,release_failure_recorded=True,
                seconds=time.monotonic()-start)


def termination_check(out):
    harness=r'''
import json,pathlib,signal,subprocess,sys
sys.path.insert(0,str(pathlib.Path(sys.argv[1])/'examples'))
import fullbvp_phaseD2b_direct as d2
folder=pathlib.Path(sys.argv[2]);source=folder/'source'
source.mkdir(parents=True,exist_ok=True)
(source/'fields.npz').write_bytes(b'bounded-dummy-only')
d2.v1.SOURCES={'dummy':source}
real=d2.subprocess.Popen
def dummy(cmd,**kw):
 p=real([sys.executable,'-B','-c','import signal;signal.alarm(30);signal.pause()'],**kw)
 print(json.dumps({'worker_pid':p.pid}),flush=True)
 return p
d2.subprocess.Popen=dummy
raise SystemExit(d2.campaign(folder/'run',budget=20))
'''
    p=subprocess.Popen([sys.executable,'-B','-c',harness,str(d2.ROOT),str(out/'termination')],
                         stdout=subprocess.PIPE,stderr=subprocess.PIPE,text=True,
                         env={**os.environ,'PYTHONDONTWRITEBYTECODE':'1'},start_new_session=True)
    try:
        ready=json.loads(p.stdout.readline())
        p.send_signal(signal.SIGTERM)
        stdout,stderr=p.communicate(timeout=5)
        require(p.returncode==143,'campaign termination return code wrong: '+stderr[-1000:])
        run=out/'termination/run'
        require((run/'FAILED.json').exists() and (run/'dummy/FAILED.json').exists(),
                'campaign termination failure propagation missing')
        require(not Path(f'/proc/{ready["worker_pid"]}').exists(),'owned worker was not reaped')
        ledger=json.loads((run/'dummy/budget.json').read_text())
        require(ledger['consumed_seconds']>0,'termination lost cumulative accounting')
        return dict(passed=True,returncode=p.returncode,worker_killed_and_reaped=True,
                    cumulative_budget_recorded=True,point_and_campaign_failed=True)
    finally:
        if p.returncode is None:
            os.killpg(p.pid,signal.SIGKILL)
            p.wait()


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--out',required=True,type=Path)
    parser.add_argument('--backend',choices=['auto','pardiso','scipy'],default='auto')
    parser.add_argument('--threads',type=int,default=16)
    args=parser.parse_args()
    require(args.out.resolve().is_relative_to(d2.OUT.resolve()),'D2 output restriction')
    require(not (args.out/'VALIDATION.json').exists(),'choose a new validation directory')
    args.out.mkdir(parents=True,exist_ok=True)
    result=dict(passed=False,production_started=False)
    start=time.monotonic()
    try:
        backend,probe=d2.select_backend(args.backend,args.threads)
        result.update(backend=backend,backend_probe=probe)
        result['actual']=actual_checks(args.out,backend,args.threads)
        result['manufactured']=manufactured_checks(args.out,backend,args.threads)
        result['termination']=termination_check(args.out)
        result['passed']=True
    except Exception as exc:
        result['failure']=f'{type(exc).__name__}: {exc}'
        raise
    finally:
        result['seconds']=time.monotonic()-start
        result['executed_code_sha256']=dict(d2.fingerprints(),**{str(Path(__file__).resolve()):d2.v1.sha(__file__)})
        d2.v1.write(args.out/'VALIDATION.json',result)
    print(json.dumps(dict(passed=result['passed'],seconds=result['seconds'],backend=result['backend'],
                          actual_size=result['actual']['N16_size'],QZ_size=result['actual']['QZ_size'])))


if __name__=='__main__':
    main()

