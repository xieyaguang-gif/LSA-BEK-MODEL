#!/usr/bin/env python3
"""Bounded D2f original-pencil, backward-error and lifecycle validation."""
import os
for key in ('OPENBLAS_NUM_THREADS','OMP_NUM_THREADS','MKL_NUM_THREADS','NUMEXPR_NUM_THREADS','BLIS_NUM_THREADS'):
    os.environ[key]='1'
import argparse,json,time
from pathlib import Path
from types import SimpleNamespace
import numpy as np
from scipy import linalg,sparse
from scipy.sparse.linalg import aslinearoperator
import fullbvp_phaseD2f_repair as f
import fullbvp_phaseD2e_validate as legacy


def require(ok,message):
    if not ok:raise AssertionError(message)


def backward_case(out):
    sigma=.05
    K=sparse.csr_matrix(np.array([[1.,1.],[1.,1.+1e-12]],complex))
    J=aslinearoperator(K-sigma*sparse.eye(2));B=aslinearoperator(sparse.eye(2))
    rhs=np.array([.123456789,.987654321],complex)
    factor=f.EquilibratedFactor(K,deadline=f.old.d2.v1.Deadline(30))
    try:
        inv=f.BackwardInverse(J,B,sigma,factor,f.old.d2.v1.Deadline(30),out/'ill_conditioned.jsonl',assembled_K=K)
        y=inv.T@(-rhs);_,metric=inv.measure(y,rhs)
        require(metric['accepted'],'backward-stable ill-conditioned solve rejected')
        bad=f.old.d2.v1.certify(J,B,1.,10.,np.array([1.,0.]),tol=1e-8)
        require(not bad['accepted'],'linear acceptance bypassed original eigenpair gate')
        good=f.old.d2.v1.certify(J,B,1.,sigma-2.-.5e-12,np.array([1.,1.]),tol=1e-8)
        require(good['accepted'],'known original eigenpair rejected')
    finally:factor.close()
    broken=SimpleNamespace(solve=lambda r:np.zeros_like(r))
    fake_success=lambda A,b,**kw:(np.zeros_like(b),0)
    inv=f.BackwardInverse(J,B,sigma,broken,f.old.d2.v1.Deadline(30),out/'false_success.jsonl',assembled_K=K,solver=fake_success)
    require(legacy.rejected(lambda:inv.T@(-rhs)),'false-success GMRES passed backward gate')
    nan_factor=SimpleNamespace(solve=lambda r:np.full_like(r,np.nan))
    inv=f.BackwardInverse(J,B,sigma,nan_factor,f.old.d2.v1.Deadline(30),out/'nonfinite.jsonl',assembled_K=K)
    require(legacy.rejected(lambda:inv.T@(-rhs)),'nonfinite inverse passed')
    return dict(condition=float(np.linalg.cond(K.toarray())),metrics=metric,
                bad_pair_rejected=True,known_pair_accepted=True,false_success_rejected=True,nonfinite_rejected=True)


def recovery_case(out):
    rng=np.random.default_rng(261030)
    K=sparse.csr_matrix(np.diag(np.arange(1.,8.))+.05*(rng.normal(size=(7,7))+1j*rng.normal(size=(7,7))))
    sigma=.05+2j;J=aslinearoperator(K-sigma*sparse.eye(7));B=aslinearoperator(sparse.eye(7))
    factor=f.EquilibratedFactor(1.03*K,deadline=f.old.d2.v1.Deadline(30))
    try:
        inv=f.BackwardInverse(J,B,sigma,factor,f.old.d2.v1.Deadline(30),out/'recovery.jsonl',assembled_K=K)
        inv.T@(rng.normal(size=7)+1j*rng.normal(size=7))
        record=json.loads((out/'recovery.jsonl').read_text().splitlines()[-1])
        require(record['krylov_used'] and record['accepted'] and factor.info['factorization_count']==1,
                'same-LU backward-error recovery path not exercised')
        return dict(record=record,factorization_count=factor.info['factorization_count'])
    finally:factor.close()


def validate(out):
    out=f.old.output_path(out);out.mkdir(parents=True,exist_ok=False);start=time.monotonic()
    source=f.old.OUT/'production_v4/run_baseline/upper720_H6/corrections/attempt_0000/fields.npz'
    mapping=f.old.base.Mapping(12,1,4);y=f.old.base.transfer(source,mapping)
    q=f.old.d2.v1.unpack(y[:-2],mapping.nr,mapping.H,mapping.s)
    t=f.old.d2.v1.Temporal(mapping,q,float(y[-1]),float(y[-2]))
    J,assembly=f.old.d2.assemble(t,f.old.d2.v1.Deadline(120))
    dense=linalg.eigvals(J.toarray(),-np.diag(t.b));dense=dense[np.isfinite(dense)]
    centers=[]
    for i,sigma in enumerate((.05+0j,.05+2j,1.+0j)):
        center=f.run_center(t,J,sigma,out/f'center_{i:02d}','D2f-three-shifts',f.old.d2.v1.Deadline(90),k=6,ncv=40)
        require(center['complete'],'D2f center failed: '+str(center.get('failure')))
        distances=[float(np.min(abs(complex(*r['value'])-dense))) for r in center['eigenpairs']]
        require(max(distances)<1e-7,'independent original QZ disagrees')
        require(max(r['original_relative'] for r in center['eigenpairs'])<=1e-8,'original pair criterion changed')
        before_calls=center['factor']['solve_calls']
        resumed=f.run_center(t,J,sigma,out/f'center_{i:02d}','D2f-three-shifts',f.old.d2.v1.Deadline(30),k=6,ncv=40)
        require(resumed==center and resumed['factor']['solve_calls']==before_calls,'center resume refactored')
        log=Path(center['arrays']).parent/'inverse.jsonl';inverse=[json.loads(line) for line in log.read_text().splitlines()]
        centers.append(dict(center=center,QZ_distances=distances,shift_nearest_finite_eigenvalue=float(np.min(abs(dense-sigma))),
            backward_max=max(r['normwise_backward'] for r in inverse),componentwise_max=max(r['componentwise_backward'] for r in inverse),
            rhs_relative_max=max(r['original_relative'] for r in inverse)))
    result=dict(passed=True,seconds=None,projected_accepted_upper_field=True,field_is_stationary_on_test_grid=False,
        dimension=J.shape[0],assembly=assembly,centers=centers,backward_case=backward_case(out),
        controlled_recovery=recovery_case(out),
        lifecycle=legacy.lifecycle(out,dict(center=centers[1]['center'])),campaign=legacy.campaign_lifecycle(out),
        code_sha256=f.fingerprints(),production_launched=False)
    result['seconds']=time.monotonic()-start;f.old.write(out/'VALIDATION.json',result)
    print(json.dumps(dict(passed=True,seconds=result['seconds'],QZ_max=max(max(c['QZ_distances']) for c in centers),
        pair_residual_max=max(r['original_relative'] for c in centers for r in c['center']['eigenpairs']),
        backward_case=result['backward_case'],centers=[dict(sigma=c['center']['sigma'],calls=c['center']['arnoldi_callbacks'],
        gmres=c['center']['gmres_calls'],backward=c['backward_max'],componentwise=c['componentwise_max'],rhs_relative=c['rhs_relative_max']) for c in centers])))


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--out',type=Path,required=True);a=p.parse_args();validate(a.out)
