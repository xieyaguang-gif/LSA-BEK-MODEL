#!/usr/bin/env python3
"""Small real-operator QZ and resource/restart tests; no production work."""
import os
for key in ('OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS'):
    os.environ[key]='1'
import argparse
import copy
import fcntl
import shutil
import json
import signal
import subprocess
import sys
import time
from pathlib import Path
from types import SimpleNamespace
import numpy as np
from scipy import linalg, sparse
import fullbvp_phaseD2e_coarse as c
import fullbvp_phaseD2d_validate as fixtures


def require(ok,message):
    if not ok:raise AssertionError(message)


def rejected(action):
    try:action()
    except (RuntimeError,ValueError,MemoryError):return True
    return False


def numerical(out):
    t,J,assembly,rng=fixtures.actual_case(8,1,4)
    sigma=.05+2j
    center=c.run_center(t,J,sigma,out/'center','small-actual-QZ',c.d2.v1.Deadline(120),k=6,ncv=40)
    require(center['complete'],'D2e actual center did not complete: '+str(center))
    dense=linalg.eigvals(J.toarray(),-np.diag(t.b));dense=dense[np.isfinite(dense)]
    distances=[float(np.min(abs(complex(*r['value'])-dense))) for r in center['eigenpairs']]
    require(max(distances)<1e-6,'independent generalized QZ disagreement')
    require(max(r['original_relative'] for r in center['eigenpairs'])<=1e-8,'original eigenpair residual')
    probe=rng.normal(size=t.size)+1j*rng.normal(size=t.size)
    linearity=float(np.linalg.norm(t.J@(1j*probe)-1j*(t.J@probe))/np.linalg.norm(t.J@probe))
    require(linearity<1e-12,'complex Fourier packing is not complex-linear')
    resumed=c.run_center(t,J,sigma,out/'center','small-actual-QZ',c.d2.v1.Deadline(30),k=6,ncv=40)
    require(resumed==center,'resume result changed')
    require(rejected(lambda:c.run_center(t,J,sigma,out/'center','changed',c.d2.v1.Deadline(30),k=6,ncv=40)),
            'changed fingerprint resumed')
    require(rejected(lambda:c.stationarity_gate(t)),'nonstationary projected field passed')
    zero=c.d2.v1.Temporal(t.map,np.zeros_like(t.q),720.,52.7)
    laminar=c.stationarity_gate(zero)
    require(rejected(lambda:c.branch_gate(dict(A1=1,omega=52.7),dict(A1=0,omega=52.7))),
            'zero-field branch accepted')
    bad=[c.d2.v1.certify(t.J,t.B,max(t.b),v,x,tol=c.PAIR_ACCEPT) for v,x in
         [(complex('nan'),probe),(0,probe),(0,np.zeros_like(probe))]]
    require(not any(r['accepted'] for r in bad),'nonfinite/invalid/massless eigenpair accepted')
    # Real inherited recovery, deliberately inaccurate preconditioner, one LU.
    K=J.astype(complex)+sigma*sparse.diags(t.b)
    f=c.TimedFactor(c.GuardedFactor(1.03*K,threads=16,deadline=c.d2.v1.Deadline(30)))
    try:
        inv=c.d2.PrecisionInverse(t.J,t.B,sigma,f,c.d2.v1.Deadline(30),out/'recovery.jsonl',assembled_K=K)
        y=inv.T@probe
        recovery=json.loads((out/'recovery.jsonl').read_text().splitlines()[-1])
        require(recovery['krylov_used'] and recovery['original_relative']<=c.LINEAR_ACCEPT,
                'strict double-precision original recovery did not pass')
        recovery_timing=f.summary()
    finally:f.close()
    # Guard must reject after symbolic analysis and before numeric phase22.
    old_limit=c.MEMORY_BYTES
    try:
        c.MEMORY_BYTES=1
        require(rejected(lambda:c.GuardedFactor(K,threads=16,deadline=c.d2.v1.Deadline(30))),
                'memory guard did not reject')
    finally:c.MEMORY_BYTES=old_limit
    old_eigs=c.eigs
    try:
        def nonconvergent(*a,**kw):raise RuntimeError('controlled nonconvergence')
        c.eigs=nonconvergent
        failed=c.run_center(t,J,sigma,out/'failed_center','failure-case',c.d2.v1.Deadline(30),k=4,ncv=20)
        require(not failed['complete'] and not (out/'failed_center/DONE.json').exists(),'failed center left DONE')
    finally:c.eigs=old_eigs
    return dict(size=t.size,assembly=assembly,complex_linearity=linearity,center=center,QZ_distances=distances,
                recovery=recovery,recovery_timing=recovery_timing,zero_stationarity=laminar,
                nonstationary_rejected=True,zero_branch_rejected=True,memory_guard_rejected=True,
                nonconvergent_rejected=True,source_correction_convergence_tested=False)


def lifecycle(out,numerical):
    params=dict(budget_seconds=2.,test='bounded lifecycle',shifts=[[.05,2.]],k=6,ncv=40)
    failed=c.launch_point(params,out/'exit_failure',[sys.executable,'-c','raise SystemExit(7)'])
    require(not failed['complete'] and not (out/'exit_failure/DONE.json').exists(),'failed child left DONE')
    require(rejected(lambda:c.launch_point(dict(params,test='changed'),out/'exit_failure',[])),
            'point fingerprint not enforced')
    timeout=c.launch_point(dict(params,budget_seconds=.05),out/'timeout',
                           [sys.executable,'-c','import time; time.sleep(30)'])
    require(not timeout['complete'] and timeout['used_seconds']>=.05,'hard cumulative budget not enforced')
    resumed=c.launch_point(dict(params,budget_seconds=.05),out/'timeout',[sys.executable,'-c','raise SystemExit(0)'])
    require('exhausted' in resumed['failure'],'exhausted run was relaunched')
    crashed=out/'crash';crashed.mkdir()
    c.write(crashed/'ACTIVE.json',dict(reserved_seconds=2.))
    crash=c.launch_point(params,crashed,[sys.executable,'-c','raise SystemExit(0)'])
    require(not crash['complete'] and crash['used_seconds']==2.,'crash reservation not conservatively charged')
    # Owned child signals its parent after startup: no polling, no external PID.
    signal_result=c.launch_point(params,out/'signal',
       [sys.executable,'-c','import os,signal,time; os.kill(os.getppid(),signal.SIGTERM); time.sleep(30)'])
    require(not signal_result['complete'] and 'Interrupted' in signal_result['failure'],'SIGTERM cleanup failed')
    # This fixture tests artifact integrity, not physical stationarity.
    success=out/'success';success.mkdir()
    fingerprint=c.d2.v1.digest(params)
    correction_dir=success/'corrections'/'attempt_0000';correction_dir.mkdir(parents=True)
    fields=correction_dir/'fields.npz';fields.write_bytes(b'bounded integrity fixture')
    correction=dict(fingerprint=fingerprint,fields=str(fields),fields_sha256=c.d2.v1.sha(fields))
    c.write(success/'corrected.json',correction)
    center=copy.deepcopy(numerical['center']);center['fingerprint']=fingerprint
    center_dir=success/'center_00';arrays=center_dir/'attempts'/'attempt_0000'/'eigenpairs.npz'
    arrays.parent.mkdir(parents=True);shutil.copyfile(center['arrays'],arrays)
    center.update(arrays=str(arrays),arrays_sha256=c.d2.v1.sha(arrays))
    c.write(center_dir/'DONE.json',center)
    certificate=dict(fingerprint=fingerprint,complete=True,peak_rss_bytes=1024,
                     correction=correction,centers=[center])
    c.write(success/'fixture.json',certificate)
    lock_path=out/'owned_test.lock'
    with lock_path.open('w+') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        code=('import json,pathlib,os; p=pathlib.Path('+repr(str(success))+'); '
              'os.fstat('+str(lock.fileno())+'); '
              '(p/"worker_result.json").write_text((p/"fixture.json").read_text())')
        done=c.launch_point(params,success,[sys.executable,'-c',code],lock_fd=lock.fileno())
    require(done['complete'],'successful child/evidence/inherited lock not completed')
    require(c.launch_point(params,success,[sys.executable,'-c','raise SystemExit(99)'])==done,'DONE resumed worker')
    original_arrays=arrays.read_bytes();arrays.write_bytes(b'corrupt array')
    require(rejected(lambda:c.launch_point(params,success,[])),'corrupted eigenpair array reused')
    arrays.write_bytes(original_arrays)
    original_fields=fields.read_bytes();fields.write_bytes(b'corrupt corrected state')
    require(rejected(lambda:c.launch_point(params,success,[])),'corrupted corrected field reused')
    fields.write_bytes(original_fields)
    (success/'worker_result.json').write_text('{}')
    require(rejected(lambda:c.launch_point(params,success,[])),'corrupted DONE reused')
    forbidden=[c.OUT/'production_v3/new',c.OUT/'production_v2/new',c.OUT/'../escape',Path('/tmp/outside')]
    require(all(rejected(lambda p=p:c.output_path(p)) for p in forbidden),'output scope failure')
    alias=out/'alias';alias.symlink_to(c.OUT/'production_v3',target_is_directory=True)
    require(rejected(lambda:c.output_path(alias/'new')),'symlink alias accepted');alias.unlink()
    return dict(failed_child=failed,budget_timeout=timeout,budget_resume=resumed,crash=crash,
                signal=signal_result,done_resume=True,corrupt_done_rejected=True,path_checks=True,
                corrupt_arrays_rejected=True,corrupt_corrected_field_rejected=True,worker_inherits_lock=True)


def campaign_lifecycle(out):
    # Isolate all lock tests from the running production lock.
    fake=out/'isolated_D2';fake.mkdir();(fake/'.production.lock').touch()
    old_out,old_parameters,old_launch=c.OUT,c.parameters,c.launch_point
    try:
        c.OUT=fake
        c.parameters=lambda source,n,H,dr: dict(source=source,n=n,H=H,dr=dr)
        args=SimpleNamespace(out=fake/'D2e'/'campaign',suite='baseline',sources=['fixture'])
        c.launch_point=lambda *a,**kw: dict(complete=False)
        require(c.campaign(args)==1 and (args.out/'FAILED.json').exists(),'campaign failure marker absent')
        c.launch_point=lambda *a,**kw: dict(complete=True)
        require(c.campaign(args)==0 and (args.out/'DONE.json').exists()
                and not (args.out/'FAILED.json').exists(),'successful campaign retains FAILED')
        def fail(*a,**kw):raise RuntimeError('controlled campaign error')
        c.launch_point=fail
        require(c.campaign(args)==1 and (args.out/'FAILED.json').exists()
                and not (args.out/'DONE.json').exists(),'failed campaign retains DONE')
        args.suite='harmonic'
        require(rejected(lambda:c.campaign(args)),'campaign configuration mismatch was accepted')
        args.out=fake/'D2e'/'busy_lock'
        with (fake/'.production.lock').open('r') as lock:
            fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
            try:c.campaign(args)
            except BlockingIOError:pass
            else:raise AssertionError('busy campaign lock was acquired')
        require(not args.out.exists(),'busy lock wrote an output directory')
    finally:c.OUT,c.parameters,c.launch_point=old_out,old_parameters,old_launch
    return dict(configuration_binding=True,success_clears_failed=True,failure_clears_done=True,
                exception_marks_failure=True,busy_lock_no_writes=True,used_only_isolated_test_lock=True)


def main():
    p=argparse.ArgumentParser();p.add_argument('--out',type=Path,required=True);args=p.parse_args()
    out=c.output_path(args.out);out.mkdir(parents=True,exist_ok=False)
    start=time.monotonic()
    evidence=numerical(out)
    result=dict(numerical=evidence,lifecycle=lifecycle(out,evidence),campaign=campaign_lifecycle(out),
                seconds=time.monotonic()-start,passed=True,
                production_launched=False)
    c.write(out/'VALIDATION.json',result)
    print(json.dumps(dict(passed=True,seconds=result['seconds'],QZ_max=max(result['numerical']['QZ_distances']),
                         residual_max=max(r['original_relative'] for r in result['numerical']['center']['eigenpairs']))))


if __name__=='__main__':main()
