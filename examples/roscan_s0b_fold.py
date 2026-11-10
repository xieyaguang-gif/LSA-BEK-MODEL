#!/usr/bin/env python3
"""Diagnose the secondary Ro=-0.3 neutral-Re fold and connect its two sides."""
import os
for k in ('OPENBLAS_NUM_THREADS','OMP_NUM_THREADS','MKL_NUM_THREADS'):os.environ[k]='1'
os.environ['PYTHONDONTWRITEBYTECODE']='1'
import json,time
import numpy as np
from scipy import linalg,optimize
import roscan_s0b_v2 as m
start=time.monotonic();dest=m.RUN/m.old.tag(-.3);b=json.loads((dest/'branch_legacy_integer_walk.json').read_text());seed=b['minimum'];calc=m.Calculator(-.3,128,40.,time.monotonic()+45);rr=seed['Re_ca'];bb=seed['beta'];target=complex(*seed['omega_raw']);p=calc.problem(rr,bb);s=m.near(p,complex(*seed['alpha']),target+2e-6j);qr=s.vectors[:,np.argmin(abs(s.values-target))];vr=m.velocity(p,qr);vr/=linalg.norm(vr)
def equations(y):
 if time.monotonic()>calc.deadline:raise m.old.BudgetExceeded('Neutral-fold budget')
 ar,ai,re,beta=y*np.array([1,1,rr,bb])
 if re<20 or re>100000 or beta<=0:raise RuntimeError('Nonphysical fold trial')
 p=calc.problem(re,beta);a=complex(ar,ai);s=m.near(p,a,target+2e-6j);corr=[abs(np.vdot(vr,m.velocity(p,q)))/linalg.norm(m.velocity(p,q)) for q in s.vectors.T]
 if max(corr)<.6:raise RuntimeError('Fold mode identity lost')
 j=int(np.argmax(corr));w=s.values[j];q=s.vectors[:,j];bq=p.B@q;M=p.matrix(a)-w*p.B
 L=np.block([[M.conj().T,q[:,None]],[bq.conj()[None,:],np.zeros((1,1))]]);left=linalg.solve(L,np.r_[np.zeros(len(q)),1],check_finite=False)[:-1];den=np.vdot(left,bq)
 g=np.vdot(left,(p.A1+2*a*p.A2)@q)/den;h=.01;dA=(calc.problem(re+h,beta).matrix(a)-calc.problem(re-h,beta).matrix(a))/(2*h);dw=np.vdot(left,dA@q)/den
 return np.array([g.real,g.imag,w.imag,re*dw.imag])
result={'ro':-.3,'scope':'Secondary neutral-Re fold, distinct from the original Ro=-0.50454 ARPACK failure','connections':[]}
try:
 sol=optimize.root(equations,np.array([*seed['alpha'],1,1.]),tol=1e-8,options={'maxfev':100});err=equations(sol.x)
 if max(abs(err))>1e-8:raise RuntimeError('Fold residual '+str(err))
 ar,ai,re,beta=sol.x*np.array([1,1,rr,bb]);result['fold']=dict(Re=re,beta=beta,alpha=[ar,ai],equations=err.tolist());m.old.dump(dest/'neutral_Re_fold.json',result)
 low=None
 for factor in [.85,1.15]:
  r,_=calc.critical(111,(ar,ai,re*factor),target);result['connections'].append(m.compact(r))
  if low is None or r['Re_ca']<low['Re_ca']:low=r
 main=json.loads((dest/'branch_continued_physical_branch.json').read_text())['minimum'];path=[]
 for beta in [115,121,130,140,150,157]:
  low,_=calc.critical(beta,(*low['alpha'],low['Re_ca']),complex(*low['omega_raw']));path.append(m.compact(low));m.old.dump(dest/'neutral_Re_fold_lower_path.json',path)
 result['lower_path']=path;result['connects_to_primary_minimum']=bool(abs(low['Re_ca']/main['Re_ca']-1)<1e-7 and abs(complex(*low['alpha'])-complex(*main['alpha']))<1e-5);result['status']='resolved' if result['connects_to_primary_minimum'] else 'partial'
except Exception as ex:result.update(status='partial',error=f'{type(ex).__name__}: {ex}')
result['seconds']=time.monotonic()-start;m.old.dump(dest/'neutral_Re_fold.json',result);print(json.dumps(result)[:4500])
