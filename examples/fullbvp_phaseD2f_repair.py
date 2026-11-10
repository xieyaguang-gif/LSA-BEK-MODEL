#!/usr/bin/env python3
"""D2f: equivalent equilibration, backward-stable inverse, nonzero lower control."""
import os
for key in ('OPENBLAS_NUM_THREADS','OMP_NUM_THREADS','MKL_NUM_THREADS','NUMEXPR_NUM_THREADS','BLIS_NUM_THREADS'):
    os.environ[key]='1'
import json,time
from pathlib import Path
import numpy as np
from scipy.sparse.linalg import LinearOperator,gmres
import fullbvp_phaseD2e_coarse as old
from fullbvp_phaseD2f_diagnose import equilibrate

BACKWARD_ACCEPT=1e-12
COMPONENT_ACCEPT=1e-10
POLICY=dict(name='original_operator_normwise_and_componentwise_backward_error',
            normwise_accept=BACKWARD_ACCEPT,componentwise_accept=COMPONENT_ACCEPT,
            rhs_relative_is_diagnostic=True,eigenpair_accept=1e-8,equilibration_steps=6)
_RawFactor=old.GuardedFactor
_original_parameters=old.parameters
_original_fingerprints=old.code_fingerprints
_original_launch=old.launch_point
_original_center=old.run_center


class EquilibratedFactor:
    """K y=b <=> (Dr K Dc) z=Dr b, y=Dc z; no equations are removed."""
    def __init__(self,K,threads=16,deadline=None,stats_path=None,policy=None):
        start=time.monotonic()
        matrix=K.tocsr(copy=False)
        self.singleton_rows=np.flatnonzero(np.diff(matrix.indptr)==1)
        indices=matrix.indptr[self.singleton_rows]
        self.singleton_columns=matrix.indices[indices]
        self.singleton_values=matrix.data[indices]
        E,self.rows,self.cols=equilibrate(K,POLICY['equilibration_steps'])
        equilibration_seconds=time.monotonic()-start
        self.factor=_RawFactor(E,threads=threads,deadline=deadline,stats_path=stats_path,policy=policy)
        self.info=self.factor.info
        self.info['equilibration']=dict(steps=POLICY['equilibration_steps'],
            seconds=equilibration_seconds,
            row_scale_range=[float(self.rows.min()),float(self.rows.max())],
            column_scale_range=[float(self.cols.min()),float(self.cols.max())],equivalent_original_system=True,
            exact_existing_singleton_constraints=len(self.singleton_rows))
    def solve(self,rhs):
        y=self.cols*self.factor.solve(self.rows*rhs)
        # Existing one-entry boundary rows prescribe these values exactly.
        # Enforce them instead of accepting roundoff contamination of homogeneous constraints.
        y[self.singleton_columns]=np.asarray(rhs)[self.singleton_rows]/self.singleton_values
        return y
    def close(self):self.factor.close()


class BackwardInverse:
    def __init__(self,J,B,sigma,factor,deadline,log_path,assembled_K=None,policy=None,solver=gmres):
        if assembled_K is None:raise ValueError('original assembled K is required for backward-error denominators')
        self.J,self.B,self.sigma,self.factor=J,B,sigma,factor
        self.deadline,self.log_path,self.solver=deadline,old.output_path(log_path),solver
        self.K=assembled_K;self.absK=abs(assembled_K)
        self.knorm=float(np.asarray(self.absK.sum(axis=1)).max());self.calls=0
        self.T=LinearOperator(J.shape,matvec=self.apply,dtype=complex)
        self.A=LinearOperator(J.shape,matvec=self.action,dtype=complex)
    def action(self,y):
        self.deadline.check();return self.J@y+self.sigma*(self.B@y)
    def measure(self,y,rhs):
        if not np.all(np.isfinite(y)):raise old.d2.v1.InnerFailure('nonfinite inverse iterate')
        residual=rhs-self.action(y)
        if not np.all(np.isfinite(residual)):raise old.d2.v1.InnerFailure('nonfinite original residual')
        scale=self.knorm*np.linalg.norm(y,np.inf)+np.linalg.norm(rhs,np.inf)
        denominator=self.absK@abs(y)+abs(rhs)
        if not np.isfinite(scale) or not np.all(np.isfinite(denominator)):
            raise old.d2.v1.InnerFailure('nonfinite backward-error denominator')
        normwise=float(np.linalg.norm(residual,np.inf)/max(scale,1e-300))
        component=float(np.max(abs(residual)/np.maximum(denominator,1e-300)))
        relative=float(np.linalg.norm(residual)/max(np.linalg.norm(rhs),1e-300))
        return residual,dict(original_relative=relative,normwise_backward=normwise,
            componentwise_backward=component,accepted=bool(normwise<=BACKWARD_ACCEPT and component<=COMPONENT_ACCEPT))
    def apply(self,x):
        self.deadline.check();self.calls+=1;start=time.monotonic()
        record=dict(call=self.calls,sigma=old.d2.v1.zpair(self.sigma),policy=POLICY,
            direct_corrections=0,krylov_used=False,krylov_preconditioner_calls=0,
            krylov_iterations=0,direct_residuals=[],backward_history=[],accepted=False)
        rhs=-(self.B@x);best_y=None;best_score=float('inf');best_measure=None
        try:
            if not np.all(np.isfinite(rhs)):raise old.d2.v1.InnerFailure('nonfinite inverse right-hand side')
            t=time.monotonic()
            y=np.zeros_like(rhs) if not np.any(rhs) else np.asarray(self.factor.solve(rhs),complex)
            for step in range(3):
                residual,metric=self.measure(y,rhs)
                record['direct_residuals'].append(metric['original_relative']);record['backward_history'].append(metric)
                score=max(metric['normwise_backward']/BACKWARD_ACCEPT,metric['componentwise_backward']/COMPONENT_ACCEPT)
                if score<best_score:best_y,best_score,best_measure=y.copy(),score,metric
                if metric['accepted'] or step==2:break
                y=y+self.factor.solve(residual);record['direct_corrections']+=1
            record['direct_seconds']=time.monotonic()-t
            if not best_measure['accepted']:
                record['krylov_used']=True
                def precondition(r):
                    self.deadline.check();record['krylov_preconditioner_calls']+=1
                    return self.factor.solve(r)
                def callback(value):
                    self.deadline.check();record['krylov_iterations']+=1
                candidate,info=self.solver(self.A,rhs,x0=best_y,M=LinearOperator(self.J.shape,matvec=precondition,dtype=complex),
                    rtol=1e-10,atol=0,restart=35,maxiter=5,callback=callback,callback_type='pr_norm')
                _,metric=self.measure(candidate,rhs);record['krylov_info']=int(info)
                score=max(metric['normwise_backward']/BACKWARD_ACCEPT,metric['componentwise_backward']/COMPONENT_ACCEPT)
                if score<best_score:best_y,best_score=candidate.copy(),score
            _,metric=self.measure(best_y,rhs);record.update(metric)
            record['solution_amplification']=float(np.linalg.norm(best_y)/max(np.linalg.norm(rhs),1e-300))
            if not metric['accepted']:raise old.d2.v1.InnerFailure('original backward-error criterion failed: '+str(metric))
            return best_y
        except Exception as exc:
            record['failure']=f'{type(exc).__name__}: {exc}';raise
        finally:
            record['seconds']=time.monotonic()-start;self.log_path.parent.mkdir(parents=True,exist_ok=True)
            with self.log_path.open('a') as f:f.write(json.dumps(record,allow_nan=False)+'\n')


class NonzeroCorrector(old.base.Corrector):
    def metrics(self,y):
        row=super().metrics(y)
        amplitude=float(np.max(abs(y[:-2])))
        if not np.isfinite(row['A1']) or row['A1']<1e-4*self.amplitude or amplitude<1e-100:
            raise RuntimeError('nonzero branch lost during correction')
        row['relative_field_inf']=row['residual_inf']/amplitude
        row['accepted']=bool(row['accepted'] and row['relative_field_inf']<1e-11)
        return row


def corrected_temporal(params,out,deadline):
    mapping=old.base.Mapping(params['n'],params['H'],params['intervals'])
    source=old.d2.v1.SOURCES[params['source']]
    y=old.base.transfer(source/'fields.npz',mapping);initial=y.copy()
    q=old.d2.v1.unpack(y[:-2],mapping.nr,mapping.H,mapping.s)
    p,*_=mapping.at(y[-1],y[-2]);energy=old.base.energies(p,q)
    before=dict(A1=float(np.sqrt(energy[1])),omega=float(y[-2]),Rout=float(y[-1]),energies=energy.tolist())
    fingerprint=old.d2.v1.digest(params);cached=out/'corrected.json'
    if cached.exists():
        saved=json.loads(cached.read_text());fields=old.output_path(saved['fields'])
        if saved['fingerprint']!=fingerprint or old.d2.v1.sha(fields)!=saved['fields_sha256']:
            raise RuntimeError('D2f corrected-state fingerprint/hash mismatch')
        with np.load(fields) as data:q=data['field'].copy();R=float(data['Rout']);omega=float(data['omega'])
        temporal=old.d2.v1.Temporal(mapping,q,R,omega,deadline)
        old.stationarity_gate(temporal);old.branch_gate(before,saved['after'])
        return temporal,saved
    mode='energy' if params['source'].startswith('lower') else 'Rout'
    target=before['A1'] if mode=='energy' else before['Rout']
    attempts=out/'corrections';attempts.mkdir(parents=True,exist_ok=True)
    attempt=attempts/f'attempt_{len(list(attempts.iterdir())):04d}'
    previous=old.base.Corrector
    try:
        old.base.Corrector=NonzeroCorrector
        y,record=old.base.correct(mapping,q[:,mapping.H+1],y,mode,target,attempt,
            budget=max(.001,deadline.remaining()),max_newton=12)
    finally:old.base.Corrector=previous
    if not record.get('accepted'):raise RuntimeError('D2f nonzero stationary correction failed: '+str(record.get('failure')))
    q=old.d2.v1.unpack(y[:-2],mapping.nr,mapping.H,mapping.s)
    temporal=old.d2.v1.Temporal(mapping,q,float(y[-1]),float(y[-2]),deadline)
    diagnostics=old.stationarity_gate(temporal);branch=old.branch_gate(before,record)
    fields=attempt/'fields.npz'
    saved=dict(fingerprint=fingerprint,fields=str(fields),fields_sha256=old.d2.v1.sha(fields),
        before=before,after={k:record[k] for k in ('A1','omega','energies','Rout')},branch=branch,
        diagnostics=diagnostics,correction_seconds=record['seconds'],mode=mode,target=target,
        Rout_change=float(y[-1]-initial[-1]),actual_dr=float((y[-1]-400)/mapping.intervals),
        projected_to_corrected_relative=float(np.linalg.norm(y[:-2]-initial[:-2])/np.linalg.norm(initial[:-2])),
        independent=record['independent'],original_resolution_certified=False,
        comparison='lower uses same nonzero amplitude with corrected Rout; upper uses fixed Rout')
    old.write(cached,saved);return temporal,saved


def fingerprints():
    paths=[Path(__file__),old.ROOT/'examples/fullbvp_phaseD2f_diagnose.py']
    return dict(_original_fingerprints(),**{str(p.resolve()):old.d2.v1.sha(p) for p in paths})


def parameters(*args,**kw):
    result=_original_parameters(*args,**kw)
    result.update(revision='D2f',linear_accept=None,linear_policy=POLICY,
                  lower_correction='fixed projected nonzero A1; Rout and omega free')
    return result


def run_center(*args,**kw):
    result=_original_center(*args,**kw)
    out=old.output_path(args[3] if len(args)>3 else kw['out'])
    result.update(linear_accept=None,linear_policy=POLICY)
    old.write(out/('DONE.json' if result['complete'] else 'FAILED.json'),result)
    arrays=Path(result['arrays']);old.write(arrays.parent/'result.json',result)
    return result


def launch_point(params,out,command=None,lock_fd=None):
    if command is None:command=[old.sys.executable,str(Path(__file__).resolve()),'worker','--job',str(old.output_path(out)/'job.json')]
    return _original_launch(params,out,command=command,lock_fd=lock_fd)


def install():
    old.GuardedFactor=EquilibratedFactor;old.d2.PrecisionInverse=BackwardInverse
    old.corrected_temporal=corrected_temporal;old.code_fingerprints=fingerprints
    old.parameters=parameters;old.run_center=run_center;old.launch_point=launch_point


install()
if __name__=='__main__':raise SystemExit(old.main())
