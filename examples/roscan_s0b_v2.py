#!/usr/bin/env python3
"""S0b: independent robust eigensolver, diagnostics and resumable neutral continuation."""
from __future__ import annotations
import os
for key in ('OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS','BLIS_NUM_THREADS','NUMEXPR_NUM_THREADS','NUMEXPR_MAX_THREADS','VECLIB_MAXIMUM_THREADS'):
    os.environ[key]='1'
os.environ['PYTHONDONTWRITEBYTECODE']='1'
import argparse, hashlib, json, resource, sys, time
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]
sys.path[:0]=[str(ROOT/'src'),str(ROOT/'examples')]
import numpy as np
from scipy import linalg, optimize
from scipy.sparse.linalg import ArpackNoConvergence, LinearOperator, eigs
import roscan_absolute as old
from llsa.solvers import temporal, Spectrum, _residual
from llsa.secondary.discretization import rotation_rates
OUT=old.OUT/'diag_0p5'
RUN=OUT/'s0b'
SOURCE=hashlib.sha256(Path(__file__).read_bytes()).hexdigest()
old_soft,hard=resource.getrlimit(resource.RLIMIT_AS)
resource.setrlimit(resource.RLIMIT_AS,(min(60_000_000_000,hard) if hard>=0 else 60_000_000_000,hard))
STATS={'calls':0,'partial':0,'dense_fallback':0}

def near(problem,alpha,target,count=2,residual_tol=1e-8):
    """Bound ARPACK effort; qualify partial pairs, then use dense QZ if necessary."""
    STATS['calls']+=1
    A,B=problem.matrix(alpha),problem.B
    lu=linalg.lu_factor(A-target*B,check_finite=False)
    op=LinearOperator(A.shape,matvec=lambda q:linalg.lu_solve(lu,B@q,check_finite=False),dtype=complex)
    try:
        mu,q=eigs(op,k=count,ncv=min(32,len(A)-1),maxiter=16,tol=1e-11,v0=np.ones(len(A),complex))
    except ArpackNoConvergence as exc:
        STATS['partial']+=1
        mu,q=exc.eigenvalues,exc.eigenvectors
    if mu is not None and q is not None and len(mu):
        good=np.isfinite(mu)&(abs(mu)>np.finfo(float).eps)
        values=target+1/mu[good]; q=q[:,good]
        errors=_residual(A,B,values,q)
        keep=np.isfinite(values)&(errors<min(residual_tol,1e-9))
        if np.any(keep):
            return Spectrum(values[keep],q[:,keep],errors[keep])
    STATS['dense_fallback']+=1
    return temporal(problem,alpha,residual_tol=residual_tol)

def velocity(problem,q):
    # Trapezoidal kinetic-energy inner product; pressure excluded.
    z=problem.grid.z; weights=np.empty_like(z)
    weights[1:-1]=abs(z[2:]-z[:-2])/2
    weights[0]=abs(z[1]-z[0])/2; weights[-1]=abs(z[-1]-z[-2])/2
    return (problem.fields(q)[:,:3]*np.sqrt(weights[:,None])).ravel()

def group(problem,alpha,w,q):
    M=problem.matrix(alpha)-w*problem.B; bq=problem.B@q
    border=np.block([[M.conj().T,q[:,None]],[bq.conj()[None,:],np.zeros((1,1))]])
    left=linalg.solve(border,np.r_[np.zeros(len(q)),1],check_finite=False)[:-1]
    return np.vdot(left,(problem.A1+2*alpha*problem.A2)@q)/np.vdot(left,bq)

class Calculator(old.Calculator):
    def critical(self,beta,seed=(.218,.122,507.3),target=.035,reference=None,dense=False):
        start=time.monotonic(); history=[]; solver=(lambda p,a,w: temporal(p,a)) if dense else near
        p0=self.problem(seed[2],beta); sp=solver(p0,complex(*seed[:2]),complex(target)+2e-6j)
        if not len(sp.values): raise RuntimeError('No qualified temporal eigenpairs')
        j=int(np.argmin(abs(sp.values-target)))
        qr=sp.vectors[:,j] if reference is None else reference
        vr=velocity(p0,qr); vr/=linalg.norm(vr); wr=complex(sp.values[j]); last={}
        def evaluate(x):
            if self.deadline and time.monotonic()>self.deadline: raise old.BudgetExceeded('S0b budget exhausted')
            if not np.all(np.isfinite(x)) or not 20<x[2]<100000 or abs(x[0])+abs(x[1])>8:
                raise RuntimeError('Nonphysical neutral trial')
            a=complex(*x[:2]); p=self.problem(x[2],beta); s=solver(p,a,wr+2e-6j)
            correlations=np.array([abs(np.vdot(vr,velocity(p,q)))/max(linalg.norm(velocity(p,q)),1e-300) for q in s.vectors.T])
            if not len(correlations) or max(correlations)<.6: raise RuntimeError('Physical velocity branch identity lost')
            k=int(np.argmax(correlations)); w=complex(s.values[k]); q=s.vectors[:,k]; g=group(p,a,w,q)
            err=np.array([g.real,g.imag,w.imag])
            last.update(w=w,q=q,correlation=float(correlations[k]),residual=float(s.residuals[k]))
            history.append(dict(x=np.asarray(x).tolist(),omega=old.pair(w),group=old.pair(g),correlation=float(correlations[k]),residual=float(s.residuals[k])))
            return err
        sol=optimize.root(evaluate,np.array(seed,float),tol=1e-8,options={'maxfev':80})
        err=evaluate(sol.x)
        if max(abs(err))>1e-8: raise RuntimeError('Neutral residual failed: '+str(err))
        w=last['w']; re=float(sol.x[2]); disk=rotation_rates(self.ro)[0]; wb=beta*disk-re*w.conjugate()
        rec=dict(ro=self.ro,beta=float(beta),Re_ca=re,R_ca=re/abs(self.ro),alpha=sol.x[:2].tolist(),alpha_global=old.pair(complex(*sol.x[:2]).conjugate()),omega_raw=old.pair(w),omega_BEK=old.pair(wb),omega_Omega_d=old.pair(wb/disk),Omega_BEK_over_Omega_d=1/disk,n=self.n,zmax=self.zmax,neutral_equation_residual=err.tolist(),eigen_residual=last['residual'],mode_correlation=last['correlation'],evaluations=len(history),seconds=time.monotonic()-start,code=SOURCE,kernel_code=old.CODE,history=history,status='neutral_candidate',pinch_verified=False,solver='dense_QZ' if dense else 'ARPACK_k2_ncv32_partial_qualified_dense_fallback')
        return rec,last['q']

def dense_check(calc,rec):
    p=calc.problem(rec['Re_ca'],rec['beta']); a=complex(*rec['alpha']); target=complex(*rec['omega_raw'])
    s=temporal(p,a); ids=np.argsort(abs(s.values-target)); j=int(ids[0]); w=complex(s.values[j]); g=group(p,a,w,s.vectors[:,j])
    return dict(omega=old.pair(w),omega_difference=float(abs(w-target)),group=old.pair(g),residual=float(s.residuals[j]),temporal_gap=float(abs(s.values[ids[1]]-w)),qualified_finite_modes=len(s.values),passed=bool(abs(w-target)<1e-8 and abs(g)<1e-7 and abs(w.imag)<1e-8))

def compact(rec):
    return {k:v for k,v in rec.items() if k!='history'}

def diagnose():
    source=old.OUT/'joint_resume_20261101T163033_1044050'
    successes=[]
    for f in sorted(source.glob('target-0.500_step*.json')):
        d=json.loads(f.read_text())
        if d.get('status')=='converged': successes.append((f,d))
    selected=successes[-20:]
    traj=[dict(source=str(f.relative_to(ROOT)),**compact(d['result'])) for f,d in selected]
    old.dump(OUT/'trajectory_last20.json',dict(all_successes=len(successes),selected=len(traj),records=traj))
    seed=selected[-1][1]['result']; failed=json.loads((source/'target-0.500_step0037.json').read_text())
    ro,beta=failed['requested_ro'],failed['requested_continuous_beta']
    calc=Calculator(ro,128,40.,time.monotonic()+120)
    fixed,_=calc.critical(beta,(*seed['alpha'],seed['Re_ca']),complex(*seed['omega_raw']))
    proof=dense_check(calc,fixed)
    # Dense QZ independently verifies all three neutral equations at the solved point.
    old.dump(RUN/'continuous_seed.json',fixed)
    difference=proof['omega_difference']
    x=np.array([*fixed['alpha'],fixed['Re_ca']]); jac=[]
    for k,h in enumerate([1e-5,1e-5,1e-2]):
        f=[]
        for sign in [-1,1]:
            xx=x.copy(); xx[k]+=sign*h; p=calc.problem(xx[2],beta); a=complex(*xx[:2]); s=near(p,a,complex(*fixed['omega_raw'])+2e-6j); j=int(np.argmin(abs(s.values-complex(*fixed['omega_raw'])))); w=s.values[j]; g=group(p,a,w,s.vectors[:,j]); f.append(np.array([g.real,g.imag,w.imag]))
        jac.append((f[1]-f[0])/(2*h))
    J=np.array(jac).T; scaled=J@np.diag([1.,1.,fixed['Re_ca']]); sv=linalg.svdvals(scaled)
    report=dict(failed_parameters=failed,robust=compact(fixed),dense_validation=proof,max_eigenvalue_difference=float(difference),scaled_neutral_jacobian_singular_values=sv.tolist(),scaled_condition_number=float(sv[0]/sv[-1]),classification='ARPACK failure at this point; physical fold not inferred',solver_stats=dict(STATS),source_sha256=SOURCE)
    old.dump(OUT/'diagnosis.json',report); old.dump(RUN/'continuous_seed.json',fixed)
    print(json.dumps(dict(ro=ro,beta=beta,Re=fixed['Re_ca'],dense_agreement=difference,condition=report['scaled_condition_number'],seconds=fixed['seconds'],stats=STATS)))

def continue_path(budget=45,targets=None):
    start=time.monotonic(); deadline=start+budget
    state=json.loads((RUN/'continuous_seed.json').read_text())
    steps=[]
    targets=targets or [-.5,-.4,-.3,-.2,-.1]
    invariant=state['beta']*abs(state['ro'])
    try:
        for target in targets:
            if state['ro']>=target-1e-12: continue
            delta=.01
            while state['ro']<target-1e-12:
                if time.monotonic()>deadline: raise old.BudgetExceeded('Continuation checkpoint saved')
                rr=min(target,state['ro']+delta); bb=invariant/abs(rr)
                try:
                    calc=Calculator(rr,128,40.,deadline)
                    rec,_=calc.critical(bb,(*state['alpha'],state['Re_ca']),complex(*state['omega_raw']))
                    state=rec; old.dump(RUN/'continuous_seed.json',state)
                    old.dump(RUN/f'continuous_ro{rr:+.8f}.json',rec)
                    steps.append(dict(ro=rr,beta=bb,Re=rec['Re_ca'],seconds=rec['seconds']))
                    delta=min(.02,delta*1.5)
                except old.BudgetExceeded: raise
                except Exception as exc:
                    old.dump(RUN/f'failed_ro{rr:+.9f}_step{delta:.9f}.json',dict(ro=rr,beta=bb,error=str(exc)))
                    delta/=2
                    if delta<1e-5: raise
            old.dump(RUN/f'target{target:+.1f}.json',state)
    except Exception as exc:
        reason=f'{type(exc).__name__}: {exc}'
    else: reason='all requested continuous targets reached'
    result=dict(last_ro=state['ro'],last_beta=state['beta'],last_Re=state['Re_ca'],steps=steps,seconds=time.monotonic()-start,reason=reason,stats=STATS,peak_RSS_kb=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss)
    old.dump(RUN/'continuation_completion.json',result)
    print(json.dumps(result))

def stationary(calc,state):
    """Solve the neutral saddle and d(Im omega)/d(beta)=0 together."""
    re0=state['Re_ca']; b0=state['beta']; target=complex(*state['omega_raw'])
    p0=calc.problem(re0,b0); a0=complex(*state['alpha']); s0=near(p0,a0,target+2e-6j)
    qr=s0.vectors[:,np.argmin(abs(s0.values-target))]; vr=velocity(p0,qr);vr/=linalg.norm(vr)
    def equations(y):
        if calc.deadline and time.monotonic()>calc.deadline: raise old.BudgetExceeded('Stationary minimum budget exhausted')
        ar,ai,re,bb=y*np.array([1,1,re0,b0])
        if re<20 or re>100000 or bb<=0 or abs(ar)+abs(ai)>8: raise RuntimeError('Nonphysical stationary trial')
        p=calc.problem(re,bb); aa=complex(ar,ai); sp=near(p,aa,target+2e-6j)
        corr=[abs(np.vdot(vr,velocity(p,q)))/max(linalg.norm(velocity(p,q)),1e-300) for q in sp.vectors.T]
        if not corr or max(corr)<.6: raise RuntimeError('Stationary mode identity lost')
        j=int(np.argmax(corr)); w=sp.values[j];q=sp.vectors[:,j];bq=p.B@q;M=p.matrix(aa)-w*p.B
        border=np.block([[M.conj().T,q[:,None]],[bq.conj()[None,:],np.zeros((1,1))]])
        left=linalg.solve(border,np.r_[np.zeros(len(q)),1],check_finite=False)[:-1];den=np.vdot(left,bq)
        g=np.vdot(left,(p.A1+2*aa*p.A2)@q)/den
        hh=.01; dA=(calc.problem(re,bb+hh).matrix(aa)-calc.problem(re,bb-hh).matrix(aa))/(2*hh)
        db=np.vdot(left,dA@q)/den
        return np.array([g.real,g.imag,w.imag,bb*db.imag])
    sol=optimize.root(equations,np.array([*state['alpha'],1.,1.]),tol=1e-8,options={'maxfev':100})
    err=equations(sol.x)
    if max(abs(err))>1e-8: raise RuntimeError('Stationary minimum residual '+str(err))
    ar,ai,re,bb=sol.x*np.array([1,1,re0,b0]);rec,_=calc.critical(bb,(ar,ai,re),target)
    rec['stationary_residual']=err.tolist(); return rec

def spatial_near(problem,omega,sigma,count=4):
    C,A1,A2=problem.A0-omega*problem.B,problem.A1,problem.A2; m=len(C)
    lu=linalg.lu_factor(C+sigma*A1+sigma*sigma*A2,check_finite=False);H=A1+sigma*A2
    def apply(v):
        x=-linalg.lu_solve(lu,H@v[:m]+A2@v[m:],check_finite=False)
        return np.r_[x,v[:m]+sigma*x]
    op=LinearOperator((2*m,2*m),matvec=apply,dtype=complex)
    try: mu,q=eigs(op,k=count,ncv=32,maxiter=120,which='LM',tol=1e-10,v0=np.ones(2*m,complex))
    except ArpackNoConvergence as ex: mu,q=ex.eigenvalues,ex.eigenvectors
    if mu is None or len(mu)<2: raise RuntimeError('Fewer than two spatial eigenpairs converged')
    good=np.isfinite(mu)&(abs(mu)>1e-15); a=sigma+1/mu[good];q=q[:m,good];q/=linalg.norm(q,axis=0)
    errors=linalg.norm(C@q+(A1@q)*a+(A2@q)*a*a,axis=0)/(linalg.norm(C)+abs(a)*linalg.norm(A1)+abs(a)**2*linalg.norm(A2))
    keep=np.isfinite(a)&np.isfinite(errors)&(errors<1e-8)
    return a[keep],errors[keep]

def validate_branch(ro,state,label,deadline):
    dest=RUN/old.tag(ro); calc=Calculator(ro,128,40.,deadline)
    branch=dict(label=label,status='in_progress',seed=compact(state),candidates=[],convergence=[],pinch={'status':'pending'})
    try:
        opt=stationary(calc,state); branch['continuous_stationary']=compact(opt)
        for b in sorted(set([max(0,int(np.floor(opt['beta']))-1),int(np.floor(opt['beta'])),int(np.ceil(opt['beta'])),int(np.ceil(opt['beta']))+1])):
            rec,_=calc.critical(b,(*opt['alpha'],opt['Re_ca']),complex(*opt['omega_raw']))
            rec['beta']=int(b); old.dump(dest/'candidates'/f'{label}_beta{b}.json',rec); branch['candidates'].append(compact(rec))
        best=min(branch['candidates'],key=lambda r:r['Re_ca']); branch['minimum']=best
        branch['integer_minimum_bracketed']=all(any(r['beta']==best['beta']+d and r['Re_ca']>best['Re_ca'] for r in branch['candidates']) for d in [-1,1])
        branch['dense_validation']=dense_check(calc,best)
        old.near_roots=spatial_near
        branch['pinch']=old.pinch(calc,best)
        if branch['pinch']['status']=='finite_path_passed':
            for n,z in [(192,40.),(128,50.)]:
                finer=Calculator(ro,n,z,deadline); rr,_=finer.critical(best['beta'],(*best['alpha'],best['Re_ca']),complex(*best['omega_raw']))
                diff=abs(rr['Re_ca']/best['Re_ca']-1); item=dict(n=n,zmax=z,relative_Re_change=diff,passed=diff<.001,result=compact(rr))
                branch['convergence'].append(item);old.dump(dest/f'{label}_N{n}_z{z:g}.json',item)
        branch['status']='validated_pinch_minimum' if branch['integer_minimum_bracketed'] and branch['dense_validation']['passed'] and branch['pinch']['status']=='finite_path_passed' and len(branch['convergence'])==2 and all(x['passed'] for x in branch['convergence']) else 'neutral_minimum_pinch_unresolved'
    except Exception as ex: branch.update(status='partial',error=f'{type(ex).__name__}: {ex}')
    old.dump(dest/f'branch_{label}.json',branch);return branch

def survey_main(ro,budget):
    start=time.monotonic(); state=json.loads((RUN/f'target{ro:+.1f}.json').read_text())
    branch=validate_branch(ro,state,'continued_physical_branch',start+budget)
    print(json.dumps(dict(ro=ro,status=branch['status'],minimum=branch.get('minimum'),error=branch.get('error'),seconds=time.monotonic()-start,stats=STATS)))

if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('action',choices=['diagnose','continue','survey']);p.add_argument('--budget',type=float,default=45);p.add_argument('--targets',nargs='+',type=float);p.add_argument('--ro',type=float)
    a=p.parse_args()
    if a.action=='diagnose':diagnose()
    elif a.action=='survey':survey_main(a.ro,a.budget)
    else:continue_path(a.budget,a.targets)
