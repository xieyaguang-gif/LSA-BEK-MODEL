#!/usr/bin/env python3
"""Bounded D2d actual-operator, precision-recovery and lifecycle validation."""
from __future__ import annotations
import os
for key in ('OPENBLAS_NUM_THREADS','OMP_NUM_THREADS','MKL_NUM_THREADS','NUMEXPR_NUM_THREADS','VECLIB_MAXIMUM_THREADS','BLIS_NUM_THREADS'):
    os.environ[key] = '1'
os.environ['PYTHONDONTWRITEBYTECODE'] = '1'
import argparse
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
import fullbvp_phaseD2d_precision as d2


def require(value, message):
    if not value:
        raise AssertionError(message)


def relative(a, b):
    return float(np.linalg.norm(a-b)/max(np.linalg.norm(a), np.linalg.norm(b), 1e-300))


def records(path):
    return [json.loads(line) for line in path.read_text().splitlines()]


def actual_case(n, H, intervals):
    mapping = d2.v1.Mapping(n=n, H=H, intervals=intervals)
    rng = np.random.default_rng(26)
    x = rng.normal(size=mapping.nr*(2*H+1)*mapping.s)*1e-4
    q = d2.v1.unpack(x,mapping.nr,H,mapping.s)
    temporal = d2.v1.Temporal(mapping,q,720.,52.7)
    J, assembly = d2.assemble(temporal,d2.v1.Deadline(90))
    return temporal,J,assembly,rng


def actual_checks(out, backend, threads):
    temporal,J,assembly,rng = actual_case(16,2,8)
    sigma = .05+2j
    x = rng.normal(size=J.shape[0])+1j*rng.normal(size=J.shape[0])
    action = relative(J@x,temporal.J@x)
    linearity = relative(temporal.J@(1j*x),1j*(temporal.J@x))
    require(max(action,linearity)<1e-10,'actual coupled complex-linear action mismatch')
    K = J.astype(complex)+sigma*sparse.diags(temporal.b)
    factor = d2.factorize(K,backend,threads,d2.v1.Deadline(90),out/'actual_factor.json')
    try:
        inv = d2.PrecisionInverse(temporal.J,temporal.B,sigma,factor,d2.v1.Deadline(90),
                                   out/'actual_precision.jsonl',assembled_K=K)
        y = inv.T@x
        rhs = -(temporal.B@x)
        residual = float(np.linalg.norm(temporal.J@y+sigma*(temporal.B@y)-rhs)/np.linalg.norm(rhs))
        require(residual<=d2.ACCEPT,'actual original inverse sign/scaling failed')
        require(relative(inv.T@(1j*x),1j*y)<1e-8,'actual inverse is not complex-linear')
        info = dict(factor.info)
    finally:
        factor.close()
    small,Js,small_assembly,rng = actual_case(8,1,4)
    Ks = Js.astype(complex)+sigma*sparse.diags(small.b)
    # Deliberately nearby LU of 1.03*K; two direct corrections remain O(1e-5).
    inaccurate = d2.factorize(1.03*Ks,backend,threads,d2.v1.Deadline(90),out/'perturbed_actual_factor.json')
    try:
        inv = d2.PrecisionInverse(small.J,small.B,sigma,inaccurate,d2.v1.Deadline(90),
                                   out/'corrected_actual.jsonl',assembled_K=Ks)
        z = rng.normal(size=Js.shape[0])+1j*rng.normal(size=Js.shape[0])
        inv.T@z
        inv.T@(1j*z)
        history = records(out/'corrected_actual.jsonl')
        require(all(h['initial_original_relative']>d2.ACCEPT and h['direct_original_relative']>d2.ACCEPT
                    and h['krylov_used'] and h['original_relative']<=d2.ACCEPT for h in history),
                'controlled inaccurate actual LU did not exercise recovery')
        require(inaccurate.info['factorization_count']==1 and inaccurate.info['solve_calls']>2,
                'precision correction did not reuse one factor')
        recovery_info = dict(inaccurate.info)
    finally:
        inaccurate.close()
    dense = linalg.eigvals(Js.toarray(),-np.diag(small.b))
    finite = dense[np.isfinite(dense)&(abs(dense)<1000)]
    center = d2.run_center(small,Js,sigma,out/'actual_QZ_center','actual-QZ-D2d',
              d2.v1.Deadline(120),backend=backend,threads=threads,k=4,ncv=20,outer_maxiter=120)
    accepted = [r for r in center['eigenpairs'] if r['accepted']]
    distances = [float(np.min(abs(complex(*r['value'])-finite))) for r in accepted]
    require(center['complete'] and len(accepted)==4 and max(distances)<1e-6,
            'actual code path disagrees with dense generalized eigenvalues')
    return dict(size=J.shape[0],sigma=d2.v1.zpair(sigma),assembly=assembly,
                complex_action_relative=action,complex_linearity_relative=linearity,
                original_inverse_relative=residual,factor=info,
                controlled_recovery=history,controlled_factor=recovery_info,
                QZ_size=Js.shape[0],QZ_assembly=small_assembly,QZ_distances=distances,actual_center=center)


def manufactured_checks(out, backend, threads):
    rng = np.random.default_rng(31)
    L = linalg.block_diag(np.diag([0.,-.4,-.6,-1.]),
        np.array([[.03,-5.],[5.,.03]]),np.array([[-.2,-2.],[2.,-.2]]))
    mass = np.diag(np.linspace(.4,1.1,8))
    C,E = rng.normal(size=(8,4)),rng.normal(size=(4,8))
    jd = np.block([[C@E-mass@L,C],[E,np.eye(4)]])
    bd = linalg.block_diag(mass,np.zeros((4,4)))
    phase = np.r_[np.eye(8)[:,0],-E[:,0]]
    J = sparse.csr_matrix(jd)
    temporal = SimpleNamespace(J=LinearOperator((12,12),matvec=lambda x:jd@x,dtype=complex),
        B=LinearOperator((12,12),matvec=lambda x:bd@x,dtype=complex),b=np.diag(bd),neutral=lambda:phase)
    sigma, folder = .05+5j, out/'manufactured_sideband'
    center = d2.run_center(temporal,J,sigma,folder,'manufactured-D2d',d2.v1.Deadline(30),
                          backend=backend,threads=threads,k=3,ncv=10)
    accepted = [r for r in center['eigenpairs'] if r['accepted']]
    target = min(abs(complex(*r['value'])-(.03+5j)) for r in accepted)
    require(center['complete'] and target<1e-9,'positive-real complex mode not recovered/certified')
    real_factorize = d2.factorize
    def fail_factory(*args, **kwargs):
        raise RuntimeError('intentional factorization failure')
    try:
        d2.factorize = fail_factory
        resumed = d2.run_center(temporal,J,sigma,folder,'manufactured-D2d',d2.v1.Deadline(30),
                               backend=backend,threads=threads,k=3,ncv=10)
        require(resumed==center,'DONE resume refactorized or changed')
        failed = d2.run_center(temporal,J,sigma,out/'recovery','retry-D2d',d2.v1.Deadline(30),
                              backend=backend,threads=threads,k=3,ncv=10)
        require(not failed['complete'] and (out/'recovery/FAILED.json').exists(),'factor failure lost')
    finally:
        d2.factorize = real_factorize
    recovered = d2.run_center(temporal,J,sigma,out/'recovery','retry-D2d',d2.v1.Deadline(30),
                             backend=backend,threads=threads,k=3,ncv=10)
    require(recovered['complete'] and not (out/'recovery/FAILED.json').exists(),'failed center recovery lost')
    K = jd+sigma*bd
    def solve_exact(A, rhs, **kwargs):
        return np.linalg.solve(K,rhs),19
    def solve_wrong(A, rhs, **kwargs):
        kwargs['callback'](0.)
        return np.zeros_like(rhs),9
    def solve_nonfinite(A, rhs, **kwargs):
        return np.full_like(rhs,np.nan),0
    zero = SimpleNamespace(solve=lambda rhs:np.zeros_like(rhs))
    for label, solver in [('nonconvergent',solve_wrong),('krylov_nonfinite',solve_nonfinite)]:
        inv = d2.PrecisionInverse(temporal.J,temporal.B,sigma,zero,d2.v1.Deadline(30),
            out/(label+'.jsonl'),assembled_K=K,solver=solver)
        rejected = False
        try:
            inv.T@np.ones(12)
        except d2.v1.InnerFailure:
            rejected = True
        require(rejected and not records(out/(label+'.jsonl'))[-1]['accepted'],label+' escaped residual gate')
    nan = SimpleNamespace(solve=lambda rhs:np.full_like(rhs,np.nan))
    rejected = False
    try:
        d2.PrecisionInverse(temporal.J,temporal.B,sigma,nan,d2.v1.Deadline(30),
                           out/'direct_nonfinite.jsonl').T@np.ones(12)
    except d2.v1.InnerFailure:
        rejected = True
    require(rejected,'nonfinite initial LU escaped rejection')
    inv = d2.PrecisionInverse(temporal.J,temporal.B,sigma,zero,d2.v1.Deadline(30),
        out/'info_not_gate.jsonl',assembled_K=K,solver=solve_exact)
    inv.T@np.ones(12)
    require(records(out/'info_not_gate.jsonl')[-1]['accepted'],'solver info overrode certified residual')
    perturbed = d2.factorize(sparse.csr_matrix(K+.07*np.eye(12)),backend,threads,
                            d2.v1.Deadline(30),out/'perturbed_manufactured_factor.json')
    try:
        inv = d2.PrecisionInverse(temporal.J,temporal.B,sigma,perturbed,d2.v1.Deadline(30),
            out/'corrected_manufactured.jsonl',assembled_K=K,policy=dict(direct_steps=0))
        inv.T@np.ones(12)
        correction = records(out/'corrected_manufactured.jsonl')[-1]
        require(correction['krylov_used'] and correction['initial_original_relative']>d2.ACCEPT
                and correction['original_relative']<=d2.ACCEPT and correction['krylov_iterations']>1,
                'nonscalar perturbed LU recovery not exercised')
    finally:
        perturbed.close()
    worsening = d2.factorize(sparse.csr_matrix(K),backend,threads,d2.v1.Deadline(30),
                             out/'worsening_factor.json')
    try:
        actual_solve = worsening.solve
        worsening.solve = lambda rhs:-actual_solve(rhs)
        inv = d2.PrecisionInverse(temporal.J,temporal.B,sigma,worsening,d2.v1.Deadline(30),
                                 out/'worsening_direct.jsonl',assembled_K=K)
        inv.T@np.ones(12)
        worsening_record = records(out/'worsening_direct.jsonl')[-1]
        require(worsening_record['direct_early_stop'] and
                worsening_record['selected_best_direct_original_relative'] < worsening_record['direct_residuals'][-1]
                and worsening_record['original_relative'] <= d2.ACCEPT,
                'worsening direct refinement failed best-finite-iterate recovery')
    finally:
        worsening.close()
    real_eigs = d2.eigs
    try:
        d2.eigs = lambda *a,**kw:(np.zeros(1,complex),np.ones((12,1),complex))
        invalid = d2.run_center(temporal,J,sigma,out/'invalid_pair','invalid-D2d',d2.v1.Deadline(30),
                               backend=backend,threads=threads,k=1,ncv=4)
        require(not invalid['complete'] and invalid['rejected_count']==1,'invalid pencil pair accepted')
    finally:
        d2.eigs = real_eigs
    def fail_release(*args,**kwargs):
        factor = real_factorize(*args,**kwargs)
        close = factor.close
        def bad_close():
            close()
            raise RuntimeError('intentional release failure after deallocation')
        factor.close = bad_close
        return factor
    try:
        d2.factorize = fail_release
        release = d2.run_center(temporal,J,sigma,out/'release_failure','release-D2d',d2.v1.Deadline(30),
                               backend=backend,threads=threads,k=3,ncv=10)
        require(not release['complete'] and release.get('release_failure'),'release failure lost')
    finally:
        d2.factorize = real_factorize
    return dict(positive_mode_distance=float(target),eigenpairs=center['eigenpairs'],
        factorization_count=center['factor']['factorization_count'],resume_without_refactor=True,
        failure_recovered=True,nonconvergent_rejected=True,nonfinite_rejected=True,
        solver_info_not_acceptance_gate=True,invalid_original_pair_rejected=True,
        release_failure_recorded=True,nonscalar_correction=correction,worsening_direct=worsening_record)


def scope_checks(out):
    bad = [d2.OUT,d2.ROOT/'outside_D2d_refuse',d2.OUT/'production_v1/D2d_refuse',
           d2.OUT/'production_v2/D2d_refuse',d2.OUT/'production_v3/../production_v2/D2d_refuse']
    link = out/'legacy_alias'
    link.symlink_to(d2.OUT/'production_v2',target_is_directory=True)
    bad.append(link/'D2d_refuse')
    for path in bad:
        refused = False
        try:
            d2.output_path(path)
        except ValueError:
            refused = True
        require(refused,'canonical output scope accepted '+str(path))
    script = d2.OUT/'production_v3/commands.sh'
    for path in bad[1:]:
        proc = subprocess.run(['bash',str(script)],env={**os.environ,'D2D_RUN_DIR':str(path)},
                              capture_output=True,text=True,timeout=10)
        require(proc.returncode==2,'launcher scope preflight failed')
    return dict(canonical_legacy_and_outside_refused=True,symlink_refused=True,shell_preflight_refused=True)


def termination_check(out):
    harness = r'''
import json,pathlib,subprocess,sys
sys.path.insert(0,str(pathlib.Path(sys.argv[1])/'examples'))
import fullbvp_phaseD2d_precision as d2
folder=d2.output_path(sys.argv[2]);source=folder/'source'
source.mkdir(parents=True,exist_ok=True)
for name in ('fields.npz','summary.json'):(source/name).write_bytes(b'bounded-dummy-only')
d2.v1.SOURCES={'dummy':source}
real=d2.subprocess.Popen
class ReadyPopen(real):
 def wait(self,*a,**kw):
  if not getattr(self,'announced',False):
   self.announced=True
   print(json.dumps({'worker_pid':self.pid}),flush=True)
  return super().wait(*a,**kw)
def dummy(cmd,**kw):
 return ReadyPopen([sys.executable,'-B','-c','import signal;signal.alarm(20);signal.pause()'],**kw)
d2.subprocess.Popen=dummy
raise SystemExit(d2.campaign(folder/'run',budget=15))
'''
    p = subprocess.Popen([sys.executable,'-B','-c',harness,str(d2.ROOT),str(out/'termination')],
        stdout=subprocess.PIPE,stderr=subprocess.PIPE,text=True,
        env={**os.environ,'PYTHONDONTWRITEBYTECODE':'1'},start_new_session=True)
    try:
        ready = json.loads(p.stdout.readline())
        p.send_signal(signal.SIGTERM)
        stdout,stderr = p.communicate(timeout=10)
        require(p.returncode==143,'campaign termination return code: '+stderr[-1000:])
        run = out/'termination/run'
        require((run/'FAILED.json').exists() and (run/'dummy/FAILED.json').exists(),'termination markers missing')
        require(not Path(f'/proc/{ready["worker_pid"]}').exists(),'owned worker not reaped')
        ledger = json.loads((run/'dummy/budget.json').read_text())
        require(ledger['consumed_seconds']>0 and not (run/'dummy/ACTIVE.json').exists(),'termination ledger wrong')
        return dict(passed=True,returncode=p.returncode,owned_worker_reaped=True,budget_recorded=True)
    finally:
        if p.returncode is None:
            os.killpg(p.pid,signal.SIGKILL);p.wait()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--out',required=True,type=Path)
    parser.add_argument('--backend',choices=['auto','pardiso','scipy'],default='auto')
    parser.add_argument('--threads',type=int,default=16)
    args = parser.parse_args()
    out = d2.output_path(args.out)
    require(1<=args.threads<=16,'thread limit')
    require(not out.exists(),'choose a new validation directory')
    out.mkdir(parents=True)
    result = dict(passed=False,production_started=False,full_scale_convergence_tested=False)
    start = time.monotonic()
    try:
        backend,probe = d2.select_backend(args.backend,args.threads)
        result.update(backend=backend,backend_probe=probe)
        result['actual'] = actual_checks(out,backend,args.threads)
        result['manufactured'] = manufactured_checks(out,backend,args.threads)
        result['scope'] = scope_checks(out)
        result['termination'] = termination_check(out)
        result['passed'] = True
    except Exception as exc:
        result['failure'] = f'{type(exc).__name__}: {exc}'
        raise
    finally:
        result['seconds'] = time.monotonic()-start
        result['executed_code_sha256'] = dict(d2.fingerprints(),**{str(Path(__file__).resolve()):d2.v1.sha(__file__)})
        d2.v1.write(out/'VALIDATION.json',result)
    print(json.dumps(dict(passed=result['passed'],seconds=result['seconds'],backend=result['backend'],
                         actual_size=result['actual']['size'],QZ_size=result['actual']['QZ_size'])))


if __name__ == '__main__':
    main()
