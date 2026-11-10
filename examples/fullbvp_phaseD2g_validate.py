#!/usr/bin/env python3
"""Bounded D2g numerical, cache, per-run lock and five-process dispatch checks."""
import os
for key in ('OPENBLAS_NUM_THREADS','OMP_NUM_THREADS','MKL_NUM_THREADS','NUMEXPR_NUM_THREADS','BLIS_NUM_THREADS'):
    os.environ[key]='1'
import argparse,fcntl,gc,json,sys,threading,time
from pathlib import Path
from types import SimpleNamespace
import numpy as np
from scipy import linalg,sparse
from scipy.sparse.linalg import aslinearoperator
import fullbvp_phaseD2g_campaign as g
import fullbvp_phaseD2e_validate as legacy


def require(ok,message):
    if not ok:raise AssertionError(message)


def gates(out):
    sigma=.05;K=sparse.diags([1.,1.05],format='csr');B=aslinearoperator(sparse.diags([0.,1.]))
    J=aslinearoperator(K-sigma*sparse.diags([0.,1.]))
    factor=SimpleNamespace(solve=lambda rhs:np.asarray(rhs)/K.diagonal()+np.array([1e-13,0]))
    inverse=g.Inverse(J,B,sigma,factor,g.old.d2.v1.Deadline(20),out/'component_one.jsonl',assembled_K=K)
    inverse.T@np.array([0.,-1.]);record=json.loads((out/'component_one.jsonl').read_text())
    require(record['accepted'] and record['componentwise_backward']==1. and not record['krylov_used'],
            'componentwise=1 still gates a normwise/original-relative accurate solve')
    K2=sparse.diags([1e12,1e-3],format='csr');B2=aslinearoperator(sparse.eye(2));J2=aslinearoperator(K2-sigma*sparse.eye(2))
    wrong=np.array([0.,1000.1],complex);factor2=SimpleNamespace(solve=lambda rhs:wrong.copy())
    false_success=lambda A,b,**kw:(wrong.copy(),0)
    inverse2=g.Inverse(J2,B2,sigma,factor2,g.old.d2.v1.Deadline(20),out/'rhs_reject.jsonl',assembled_K=K2,solver=false_success)
    require(legacy.rejected(lambda:inverse2.T@np.array([0.,-1.])), 'original relative residual is not enforced')
    rejected=json.loads((out/'rhs_reject.jsonl').read_text())
    require(rejected['normwise_backward']<g.BACKWARD_ACCEPT and rejected['original_relative']>g.RHS_ACCEPT,
            'RHS guard test did not isolate its intended criterion')
    pair=g.old.d2.v1.certify(J,B,1.,0.,np.ones(2),tol=1e-8)
    require(not pair['accepted'],'linear policy bypassed original eigenpair gate')
    return dict(component_one_accepted=record,original_rhs_rejected=rejected,bad_pair_rejected=True)


def numerical(out):
    source=g.OUT/'production_v4/run_D2f_baseline/upper720_H6/corrections/attempt_0000/fields.npz'
    mapping=g.old.base.Mapping(8,1,4);y=g.old.base.transfer(source,mapping)
    q=g.old.d2.v1.unpack(y[:-2],mapping.nr,mapping.H,mapping.s)
    t=g.old.d2.v1.Temporal(mapping,q,float(y[-1]),float(y[-2]))
    J,assembly=g.old.d2.assemble(t,g.old.d2.v1.Deadline(120))
    require(not np.iscomplexobj(J.data) or not np.any(J.data.imag),'packed Jacobian is not real')
    dense=linalg.eigvals(J.toarray(),-np.diag(t.b));dense=dense[np.isfinite(dense)]
    centers=[]
    for i,sigma in enumerate(g.SHIFTS):
        center=g.run_center(t,J,sigma,out/f'center_{i:02d}','D2g-all-centers',g.old.d2.v1.Deadline(60),k=6,ncv=40)
        require(center['complete'],'center failed: '+str(center.get('failure')))
        distances=[float(np.min(abs(complex(*r['value'])-dense))) for r in center['eigenpairs']]
        require(max(distances)<1e-6,'independent QZ mismatch')
        with np.load(center['arrays']) as data:
            conjugates=[g.old.d2.v1.certify(t.J,t.B,max(t.b),value.conjugate(),vector.conjugate(),tol=1e-8)
                        for value,vector in zip(data['values'],data['vectors'].T)]
        require(all(r['accepted'] for r in conjugates),'negative-frequency conjugates fail original certification')
        resumed=g.run_center(t,None,sigma,out/f'center_{i:02d}','D2g-all-centers',g.old.d2.v1.Deadline(30),k=6,ncv=40)
        require(resumed==center,'center resume changed the certificate or refactored')
        centers.append(dict(center=center,QZ_distances=distances,conjugate_certificates=conjugates))
    return dict(size=t.size,assembly=assembly,centers=centers,projected_field_not_stationary=True)


def cache_import(out):
    start=time.monotonic();point=out/'imported_lower';point.mkdir()
    params=g.parameters('lower719p948669_H4',run_name='baseline_rerun_lower')
    g.import_correction(params,point,g.old.d2.v1.Deadline(120))
    t,correction=g.prior.corrected_temporal(params,point,g.old.d2.v1.Deadline(120))
    target=point/'center_00';g.import_center(params,t,target,0,g.old.d2.v1.Deadline(120))
    require((target/'DONE.json').exists(),'completed D2f center could not be recertified')
    imported=json.loads((target/'DONE.json').read_text())
    require(imported['factorization_count_this_run']==0 and imported['reused_eigenpairs'],'import refactored')
    resumed=g.run_center(t,None,.05+0j,target,g.old.d2.v1.digest(params),g.old.d2.v1.Deadline(60),k=6,ncv=40)
    require(resumed==imported,'imported center did not resume without assembling/factoring J')
    return dict(passed=True,seconds=time.monotonic()-start,Rout=correction['after']['Rout'],
                original_pair_max=max(r['original_relative'] for r in imported['eigenpairs']),
                factorization_count=0,source_result_sha256=imported['source_result_sha256'])


def center_independence(out):
    saved=g.run_center;calls=[]
    params=dict(shifts=[g.old.d2.v1.zpair(x) for x in g.SHIFTS],k=6,ncv=40,center_budget_seconds=10.,import_from=None)
    def fake(t,J,sigma,path,fingerprint,deadline,**kw):
        i=len(calls);calls.append(sigma);result=dict(complete=i!=0,sigma=g.old.d2.v1.zpair(sigma),fingerprint=fingerprint)
        g.old.write(path/('DONE.json' if result['complete'] else 'FAILED.json'),result);return result
    try:
        g.run_center=fake
        result=g.scan_centers(None,None,params,out/'independent_centers',g.old.d2.v1.Deadline(20))
    finally:g.run_center=saved
    require(len(calls)==5 and not result[0]['complete'] and all(r['complete'] for r in result[1:]),
            'failed center prevented subsequent independent centers')
    return dict(first_failed_remaining_four_completed=True)


def run_locks(out):
    affinity=os.sched_getaffinity(0);saved_params,saved_launch=g.parameters,g.launch_point
    a=out/'locked_run';b=out/'other_run';a.mkdir();(a/'.run.lock').touch()
    try:
        g.parameters=lambda source,n,H,dr,**kw:dict(source=source,n=n,H=H,dr=dr)
        def fake(params,path,command=None,lock_fd=None):
            expected=os.stat(path.parent/'.run.lock');actual=os.fstat(lock_fd)
            require((expected.st_dev,expected.st_ino)==(actual.st_dev,actual.st_ino),'wrong lock descriptor inherited')
            return dict(complete=True)
        g.launch_point=fake
        with (a/'.run.lock').open('r+') as lock:
            fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
            try:g.campaign(SimpleNamespace(run='upper725',out=a))
            except BlockingIOError:pass
            else:raise AssertionError('same run lock was not exclusive')
            require(not (a/'manifest.json').exists(),'busy lock mutated the run manifest')
            require(g.campaign(SimpleNamespace(run='upper725',out=b))==0,'independent run blocked on another run lock')
        g.launch_point=lambda *a,**kw:dict(complete=False)
        require(g.campaign(SimpleNamespace(run='upper725',out=b))==1 and not (b/'DONE.json').exists(),'failure retains DONE')
        g.launch_point=fake
        require(g.campaign(SimpleNamespace(run='upper725',out=b))==0 and not (b/'FAILED.json').exists(),'success retains FAILED')
    finally:
        g.parameters,g.launch_point=saved_params,saved_launch;os.sched_setaffinity(0,affinity)
    return dict(same_run_exclusive=True,independent_runs_concurrent=True,per_run_lock_inherited=True,markers_consistent=True)


def parallel_dispatch(out):
    ready=out/'ready.fifo';release=out/'release.fifo';os.mkfifo(ready);os.mkfifo(release)
    ready_fd=os.open(ready,os.O_RDWR);release_fd=os.open(release,os.O_RDWR);barrier={}
    def coordinate():
        data=b''
        while len(data)<5:data+=os.read(ready_fd,5-len(data))
        barrier['arrivals']=len(data);os.write(release_fd,b'xxxxx')
    thread=threading.Thread(target=coordinate,daemon=True);thread.start();specs=[]
    for i in range(5):
        record=out/f'cpu_{i}.json'
        code=('import os,json,pathlib; cpus={int(x) for x in os.environ["D2G_CPUSET"].split(",")}; '
              'os.sched_setaffinity(0,cpus); pathlib.Path('+repr(str(record))+').write_text(json.dumps(sorted(os.sched_getaffinity(0)))); '
              'w=os.open('+repr(str(ready))+',os.O_WRONLY); os.write(w,b"x"); os.close(w); '
              'r=os.open('+repr(str(release))+',os.O_RDONLY); os.read(r,1); os.close(r)')
        specs.append((f'fixture_{i}',[sys.executable,'-c',code]))
    try:
        result=g.execute_parallel(specs,out/'parallel_dispatch',timeout_seconds=15)
        thread.join(timeout=1)
        require(result['complete'] and barrier.get('arrivals')==5,'all five workers were not started before waiting')
        sets=[set(json.loads((out/f'cpu_{i}.json').read_text())) for i in range(5)]
        require(all(len(x)==16 for x in sets) and len(set.union(*sets))==80,'CPU allocation is not five disjoint groups of 16')
    finally:os.close(ready_fd);os.close(release_fd)
    require(legacy.rejected(lambda:g.cpu_groups(6)),'dispatcher accepted more than 80 configured threads')
    failed=g.execute_parallel([('failure_fixture',[sys.executable,'-c','raise SystemExit(7)'])],out/'dispatch_failure',timeout_seconds=10)
    require(not failed['complete'] and (out/'dispatch_failure/FAILED.json').exists(),'dispatcher failure marker missing')
    return dict(five_started_concurrently=True,disjoint_16_cpu_groups=True,total_compute_threads=80,
                failure_marks_dispatch=True,used_only_fixture_processes=True)


def main(out):
    out=g.output_path(out);out.mkdir(parents=True,exist_ok=False);start=time.monotonic()
    numeric=numerical(out);gate=gates(out);imported=cache_import(out);gc.collect()
    result=dict(passed=True,numerical=numeric,gates=gate,cache_import=imported,
                centers=center_independence(out),locks=run_locks(out),dispatch=parallel_dispatch(out),
                lifecycle=legacy.lifecycle(out,dict(center=numeric['centers'][1]['center'])),
                seconds=time.monotonic()-start,code_sha256=g.fingerprints(),production_started=False)
    require((out/'exit_failure/center_00/FAILED.json').exists() and (out/'timeout/center_00/FAILED.json').exists(),
            'abrupt point failure did not leave an independent center failure marker')
    result['parent_records_unfinished_centers']=True
    g.old.write(out/'VALIDATION.json',result)
    print(json.dumps(dict(passed=True,seconds=result['seconds'],QZ_max=max(max(c['QZ_distances']) for c in numeric['centers']),
        pair_max=max(r['original_relative'] for c in numeric['centers'] for r in c['center']['eigenpairs']),
        component_one_accepted=gate['component_one_accepted']['accepted'],cache_import=imported,
        parallel=result['dispatch'])))


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--out',type=Path,required=True);a=p.parse_args();main(a.out)
