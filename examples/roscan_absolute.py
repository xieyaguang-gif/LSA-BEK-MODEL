#!/usr/bin/env python3
"""Resumable general-Ro absolute-neutral survey; all finite-search evidence retained."""
from __future__ import annotations
import argparse,hashlib,json,os,resource,sys,time,traceback
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];OUT=ROOT/'results/ro_scan_20261101/S0'
sys.path[:0]=[str(ROOT/'src'),str(ROOT/'examples')]
for k in ('OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS','BLIS_NUM_THREADS','NUMEXPR_NUM_THREADS','NUMEXPR_MAX_THREADS','VECLIB_MAXIMUM_THREADS'):os.environ[k]='1'
os.environ['PYTHONDONTWRITEBYTECODE']='1'
_,hard=resource.getrlimit(resource.RLIMIT_AS);resource.setrlimit(resource.RLIMIT_AS,(min(200_000_000_000,hard) if hard>=0 else 200_000_000_000,hard))
import numpy as np
from scipy import linalg,optimize
from llsa import LocalProblem,Parameters,solve_bek,temporal_near
from llsa.secondary.discretization import rotation_rates
from absolute_pinch_check import near_roots
ROS=[round(-1+i*.1,1) for i in range(10)]
CODE=hashlib.sha256(b''.join(p.read_bytes() for p in [Path(__file__),*sorted((ROOT/'src/llsa').rglob('*.py'))])).hexdigest()
def pair(z):return [float(z.real),float(z.imag)]
def dump(path,value):
 path=Path(path);path.parent.mkdir(parents=True,exist_ok=True);temp=path.with_suffix(path.suffix+'.tmp');temp.write_text(json.dumps(value,indent=2,allow_nan=False)+'\n');temp.replace(path)
def tag(ro):return f'ro_{ro:+.3f}'.replace('+','p').replace('-','m').replace('.','p')
class BudgetExceeded(RuntimeError):pass
class Calculator:
 def __init__(self,ro,n=128,zmax=40.,deadline=None):
  self.ro=float(ro);self.n=n;self.zmax=zmax;self.deadline=deadline;self.flow=solve_bek(ro,zmax=zmax,tol=1e-10,samples=16001);self.memo={}
 def problem(self,re,beta):
  key=(float(re),float(beta))
  if key not in self.memo:
   if len(self.memo)>3:self.memo.clear()
   self.memo[key]=LocalProblem(self.flow,Parameters(re=float(re),ro=self.ro,beta=float(beta),n=self.n,zmax=self.zmax,stretch=.1,boundary='matlab'))
  return self.memo[key]
 def critical(self,beta,seed=(.218,.122,507.3),target=.035,reference=None):
  begin=time.perf_counter();history=[]
  p0=self.problem(seed[2],beta);s0=temporal_near(p0,complex(*seed[:2]),complex(target)+1e-6j,count=6)
  if len(s0.values)==0:raise RuntimeError('No temporal seed eigenvalues')
  j=int(np.argmin(abs(s0.values-target)));qref=s0.vectors[:,j] if reference is None else reference
  qref=qref/linalg.norm(qref);wref=complex(s0.values[j]);last={}
  def evaluate(x):
   if self.deadline and time.monotonic()>self.deadline:raise BudgetExceeded('stage budget exhausted')
   if not np.all(np.isfinite(x)) or not 20<x[2]<100000 or abs(x[0])+abs(x[1])>8:raise RuntimeError('Nonphysical neutral trial')
   a=complex(*x[:2]);p=self.problem(x[2],beta);s=temporal_near(p,a,wref+2e-6j,count=6)
   if len(s.values)==0:raise RuntimeError('No residual-qualified eigenvalues')
   correlations=abs(qref.conj()@s.vectors)/np.maximum(linalg.norm(s.vectors,axis=0),1e-300)
   viable=np.flatnonzero(correlations>.35)
   if len(viable)==0:raise RuntimeError('Neutral branch identity lost')
   k=viable[np.argmax(correlations[viable])];w=complex(s.values[k]);q=s.vectors[:,k];bq=p.B@q;M=p.matrix(a)-w*p.B
   border=np.block([[M.conj().T,q[:,None]],[bq.conj()[None,:],np.zeros((1,1))]])
   left=linalg.solve(border,np.r_[np.zeros(len(q)),1],check_finite=False)[:-1]
   group=np.vdot(left,(p.A1+2*a*p.A2)@q)/np.vdot(left,bq)
   err=np.array([group.real,group.imag,w.imag]);last.update(omega=w,q=q,correlation=float(correlations[k]),residual=float(s.residuals[k]))
   history.append(dict(x=np.asarray(x).tolist(),omega=pair(w),group=pair(group),correlation=float(correlations[k]),residual=float(s.residuals[k])))
   return err
  sol=optimize.root(evaluate,np.asarray(seed,float),tol=1e-8);err=evaluate(sol.x)
  if max(abs(err))>1e-8:raise RuntimeError(f'Neutral equations failed: {err}; {sol.message}')
  w=last['omega'];re=float(sol.x[2]);R=re/abs(self.ro);disk,ext=rotation_rates(self.ro)
  # Raw local convention is exp(i alpha x+i beta theta-i omega t), time R/|Ro|.
  # Local V is disk-relative and the raw equations use U_BEK rather than Ro*U_BEK.
  # Inertial mapping follows conjugating the Ro<0 equations: alpha_g=conj(alpha), Omega=beta*disk-|Ro|*R*conj(omega).
  wb=beta*disk-abs(self.ro)*R*w.conjugate()
  rec=dict(ro=self.ro,beta=int(beta),Re_ca=re,R_ca=R,alpha=sol.x[:2].tolist(),alpha_global=pair(complex(*sol.x[:2]).conjugate()),omega_raw=pair(w),omega_BEK=pair(wb),omega_Omega_d=pair(wb/disk),Omega_BEK_over_Omega_d=1/disk,n=self.n,zmax=self.zmax,neutral_equation_residual=err.tolist(),eigen_residual=last['residual'],mode_correlation=last['correlation'],evaluations=len(history),seconds=time.perf_counter()-begin,code=CODE,history=history,status='neutral_candidate',pinch_verified=False)
  return rec,last['q']
def pinch(calc,rec):
 a0=complex(*rec['alpha']);w0=complex(*rec['omega_raw']);p=calc.problem(rec['Re_ca'],rec['beta']);path=[];prior=None
 from scipy.optimize import linear_sum_assignment
 for eta in np.geomspace(1e-7,.12,75):
  if calc.deadline and time.monotonic()>calc.deadline:raise BudgetExceeded('pinch budget exhausted')
  candidates=[]
  for shift in ([a0] if prior is None else prior):
   roots,errors=near_roots(p,w0+1j*eta,shift,count=6)
   for a,e in zip(roots,errors):
    if not any(abs(a-z[0])<1e-7 for z in candidates):candidates.append((a,e))
  roots=np.array([x[0] for x in candidates]);errors=np.array([x[1] for x in candidates])
  if len(roots)<2:raise RuntimeError('Pinch failed: less than two distinct roots')
  if prior is None:ids=np.argsort(abs(roots-a0))[:2]
  else:_,ids=linear_sum_assignment(abs(np.asarray(prior)[:,None]-roots[None,:]))
  chosen=roots[ids]
  if abs(chosen[0]-chosen[1])<1e-7:raise RuntimeError('Pinch branches collapsed')
  path.append(dict(eta=float(eta),omega=pair(w0+1j*eta),roots=[pair(x) for x in chosen],max_residual=float(max(errors[ids])),separation=float(abs(chosen[0]-chosen[1]))));prior=chosen
 opposite=bool(prior[0].imag*prior[1].imag<0)
 return dict(status='finite_path_passed' if opposite else 'undetermined',opposite_half_planes_at_end=opposite,path=path,scope='Two distinct local spatial roots tracked into upper raw-omega half-plane; finite eta<=0.12, not exhaustive global topology proof')
def compatible_codes():
 p=OUT/'compatible_kernel_sources.json'
 return {CODE,*json.loads(p.read_text()).get('codes',[])} if p.exists() else {CODE}
def load_candidates(ro):
 rows=[]
 for p in (OUT/tag(ro)/'candidates').glob('*.json'):
  row=json.loads(p.read_text())
  if row.get('code') in compatible_codes() and row.get('status')=='neutral_candidate':rows.append(row)
 return rows
def review_survey(rec):
 """Conservative evidence audit, without relabeling unresolved saddles as absolute."""
 branches=rec.get('branch_minima',[]);candidates=rec.get('found_branches',[]);unresolved=[];aliases=[]
 resolved={b['label'] for b in branches}
 for b in branches:resolved.update(b.get('independent_seed_labels',[]))
 for label in sorted(set(c.get('seed_label','unknown') for c in candidates)):
  if label in resolved:continue
  family=[c for c in candidates if c.get('seed_label')==label];minimum=min(family,key=lambda c:c['Re_ca']);matching=[]
  for other in candidates:
   if other.get('seed_label') in resolved and other['beta']==minimum['beta'] and abs(other['Re_ca']/minimum['Re_ca']-1)<1e-5 and abs(complex(*other['alpha'])-complex(*minimum['alpha']))<1e-4:matching.append(other.get('seed_label'))
  if matching:aliases.append(dict(seed_label=label,duplicate_of=matching[0],matched_at_beta=minimum['beta']))
  else:unresolved.append(dict(seed_label=label,minimum_Re=minimum['Re_ca'],beta=minimum['beta'],reason='Legacy or independent seed family has no verified walk/pinch minimum'))
 qualified=[];unresolved_branches=[]
 for b in branches:
  m=b['minimum'];label=b['label'];family=[c for c in candidates if c.get('seed_label')==label];neighbors=[1] if m['beta']==0 else [-1,1];bracket=all(any(c['beta']==m['beta']+d and c['Re_ca']>m['Re_ca'] for c in family) for d in neighbors);b['integer_minimum_bracketed']=bracket
  pin=b.get('pinch',{});path=pin.get('path',[]);pinok=bool(pin.get('status')=='finite_path_passed' and path and path[-1]['roots'][0][1]*path[-1]['roots'][1][1]<0 and all(x['separation']>1e-7 and x['max_residual']<1e-8 and x['eta']>0 for x in path))
  conv=b.get('convergence',[]);convok=len(conv)==2 and all(c.get('passed') and c['relative_Re_change']<.001 for c in conv) and any(c['n']>m['n'] and c['zmax']==m['zmax'] for c in conv) and any(c['n']==m['n'] and c['zmax']>m['zmax'] for c in conv)
  residualok=max(abs(x) for x in m.get('neutral_equation_residual',[1]))<1e-8 and m.get('eigen_residual',1)<1e-8
  if bracket and pinok and convok and residualok:qualified.append(b)
  else:unresolved_branches.append(dict(label=label,minimum_Re=m['Re_ca'],beta=m['beta'],bracket=bracket,pinch_verified=pinok,convergence_verified=convok,neutral_residual_verified=residualok,reason='Unresolved branch retained; no minimum-wide validation while unresolved'))
 selected=min(qualified,key=lambda b:b['minimum']['Re_ca']) if qualified else None
 if selected:rec.update(candidate_minimum=selected['minimum'],pinch=selected['pinch'],convergence=selected['convergence'],integer_minimum_bracketed=True,beta_coverage=selected['beta_coverage'])
 rec.update(status='numerically_validated' if selected and rec.get('search_complete',False) and not unresolved and not unresolved_branches else 'partial',unresolved_seed_families=unresolved,unresolved_branch_minima=unresolved_branches,duplicate_seed_aliases=aliases,producer_codes=sorted(set(c.get('code','missing') for c in candidates)),producer_survey_code=rec.get('producer_survey_code',rec.get('code')),validator_code=CODE,code=CODE)
 return rec

def survey(ro,n,deadline,anchor=None):
 dest=OUT/tag(ro);dest.mkdir(parents=True,exist_ok=True);calc=Calculator(ro,n,40.,deadline);candidates=load_candidates(ro);failures=[];coverage=[];branch_minima=[];search_complete=True
 # Cached eigenpairs retain their original computational-source hashes.
 def solve(beta,seed,target,label,calculator=calc):
  if calculator.deadline and time.monotonic()>calculator.deadline:raise BudgetExceeded('survey budget exhausted')
  same=[r for r in candidates if r['beta']==beta and r.get('seed_label')==label and r['n']==calculator.n]
  if same:return same[0]
  try:
   rec,q=calculator.critical(beta,seed,target);rec['seed_label']=label
   if calculator.n==n:candidates.append(rec);dump(dest/'candidates'/f'beta{beta}_{label}_N{n}.json',rec)
   else:dump(dest/'coarse'/f'beta{beta}_{label}_N{calculator.n}.json',rec)
   return rec
  except BudgetExceeded:raise
  except Exception as ex:
   fail=dict(ro=ro,beta=beta,n=calculator.n,seed=list(seed),target=pair(complex(target)),seed_label=label,error=str(ex));failures.append(fail);dump(dest/'failures'/f'beta{beta}_{label}_N{calculator.n}.json',fail);return None
 def walk(first,label,max_steps=400):
  family=[first]
  for direction in [-1,1]:
   previous=first;higher=0
   for step in range(1,max_steps+1):
    beta=first['beta']+direction*step
    if beta<0:break
    rec=solve(beta,(*previous['alpha'],previous['Re_ca']),complex(*previous['omega_raw']),label)
    if rec is None:break
    family.append(rec);higher=higher+1 if rec['Re_ca']>previous['Re_ca'] else 0;previous=rec
    if higher>=2:break
  best=min(family,key=lambda r:r['Re_ca']);bracket=all(any(r['beta']==best['beta']+d and r['Re_ca']>best['Re_ca'] for r in family) for d in [-1,1])
  if best['beta']==0:bracket=any(r['beta']==1 and r['Re_ca']>best['Re_ca'] for r in family)
  branch=dict(label=label,minimum=best,integer_minimum_bracketed=bracket,beta_coverage=sorted(set(r['beta'] for r in family)),pinch=dict(status='pending'),convergence=[])
  duplicate=next((b for b in branch_minima if b['minimum']['beta']==best['beta'] and abs(b['minimum']['Re_ca']/best['Re_ca']-1)<1e-5 and abs(complex(*b['minimum']['alpha'])-complex(*best['alpha']))<1e-4),None)
  if duplicate:duplicate.setdefault('independent_seed_labels',[]).append(label);return duplicate
  branch_minima.append(branch)
  try:
   pin=pinch(calc,best);branch['pinch']=pin;dump(dest/'branches'/f'{label}_pinch.json',dict(candidate_source_code=best['code'],candidate_beta=best['beta'],candidate_Re=best['Re_ca'],**pin))
   if pin['status']=='finite_path_passed':
    for nn,zz in [(max(192,n+32),40.),(n,50.)]:
     refined=Calculator(ro,nn,zz,deadline);rr,_=refined.critical(best['beta'],(*best['alpha'],best['Re_ca']),complex(*best['omega_raw']));change=abs(rr['Re_ca']/best['Re_ca']-1);item=dict(n=nn,zmax=zz,relative_Re_change=change,passed=change<.001,result=rr);branch['convergence'].append(item);dump(dest/'branches'/f'{label}_N{nn}_z{zz:g}.json',item)
  except BudgetExceeded:raise
  except Exception as ex:branch['validation_error']=str(ex)
  return branch
 def finish():
  qualified=[b for b in branch_minima if b['pinch']['status']=='finite_path_passed' and b['integer_minimum_bracketed'] and len(b['convergence'])==2 and all(c['passed'] for c in b['convergence'])]
  selected=min(qualified,key=lambda b:b['minimum']['Re_ca']) if qualified else None
  rec=dict(ro=ro,code=CODE,status='numerically_validated' if selected and search_complete and all(b['pinch']['status']!='pending' for b in branch_minima) else 'partial',candidate_minimum=selected['minimum'] if selected else (min(candidates,key=lambda r:r['Re_ca']) if candidates else {}),integer_minimum_bracketed=selected['integer_minimum_bracketed'] if selected else False,pinch=selected['pinch'] if selected else {},convergence=selected['convergence'] if selected else [],beta_coverage=selected['beta_coverage'] if selected else [],branch_minima=branch_minima,found_branches=candidates,failures=failures,coarse_probe_coverage=coverage,search_complete=search_complete,search_scope='Finite branch-specific integer continuation including beta=0 boundary; each distinct found minimum gets its own Briggs and convergence checks. Low-|Ro| coarse beta/frequency probes are finite and do not exclude undiscovered saddles. Negative beta follows real-coefficient conjugate symmetry, so beta>=0 is searched.')
  rec=review_survey(rec);dump(dest/'survey.json',rec);return rec
 try:
  existing=[r for r in candidates if r['n']==n and r.get('seed_label')=='typeI_continuation']
  if existing:first=min(existing,key=lambda r:r['Re_ca'])
  else:
   center=anchor['beta'] if anchor else 68;seed=(*anchor['alpha'],anchor['Re_ca']) if anchor else (.218,.122,507.3);target=complex(*anchor['omega_raw']) if anchor else .035
   # Small Ro steps preserve the same saddle family before beta minimisation.
   if anchor and abs(anchor['ro']-ro)>.025:
    for rmid in np.linspace(anchor['ro'],ro,int(np.ceil(abs(anchor['ro']-ro)/.025))+1)[1:-1]:
     check=Calculator(float(rmid),n,40.,deadline)
     try:hom,_=check.critical(center,seed,target)
     except BudgetExceeded:raise
     except Exception as ex:failures.append(dict(stage='Ro_homotopy',ro=float(rmid),error=str(ex)));break
     dump(dest/'homotopy'/f'Ro{rmid:.8f}.json',hom);seed=(*hom['alpha'],hom['Re_ca']);target=complex(*hom['omega_raw'])
   first=solve(center,seed,target,'typeI_continuation')
  if first:walk(first,'typeI_continuation')
  center=branch_minima[0]['minimum']['beta'] if branch_minima else (anchor['beta'] if anchor else 100)
  estimate=branch_minima[0]['minimum']['Re_ca'] if branch_minima else (anchor['Re_ca'] if anchor else 500.)
  # beta=0 is retained: axisymmetric disturbances were not excluded by TASK.
  beta_probes=sorted(set([0,center]+([] if branch_minima else [round(center*1.2),round(center*1.6)]))) if abs(ro)>.3 else sorted(set([0,20,50,100,200,400,center]))
  seeds=[(.12,-.08,.005),(.35,.25,.08)] if abs(ro)>.3 else [(.08,-.08,-.025),(.12,.04,.005),(.35,.25,.08)]
  coarse=Calculator(ro,64,40.,deadline)
  for beta in beta_probes:
   for j,(ar,ai,w) in enumerate(seeds):
    label=f'probe_beta{beta}_seed{j}';trial=solve(beta,(ar,ai,estimate),w,label,coarse);coverage.append(dict(beta=beta,alpha_seed=[ar,ai],omega_seed=w,n=64,found=trial is not None))
    if trial is None:continue
    fine=solve(beta,(*trial['alpha'],trial['Re_ca']),complex(*trial['omega_raw']),label)
    if fine is None:continue
    same=next((r for r in candidates if r['beta']==beta and r.get('seed_label')!=label and abs(r['Re_ca']/fine['Re_ca']-1)<1e-5 and abs(complex(*r['alpha'])-complex(*fine['alpha']))<1e-4),None)
    if same and any(b['label']==same['seed_label'] for b in branch_minima):continue
    walk(fine,label)
 except BudgetExceeded:search_complete=False
 except Exception as ex:search_complete=False;failures.append(dict(stage='survey',error=str(ex),traceback=traceback.format_exc()))
 return finish()

def summarize():
 rows=[]
 for ro in ROS+[-.751]:
  p=OUT/tag(ro)/'survey.json';s=json.loads(p.read_text()) if p.exists() else {}
  if s and s.get('code')!=CODE:s=dict(status='stale_source_requires_revalidation')
  best=s.get('candidate_minimum',{})
  row=dict(ro=ro,status=s.get('status','pending'),Re_ca=best.get('Re_ca'),R_ca=best.get('R_ca'),beta_c=best.get('beta'),omega_BEK=best.get('omega_BEK'),omega_Omega_d=best.get('omega_Omega_d'),omega_raw=best.get('omega_raw'),alpha=best.get('alpha'),pinch=s.get('pinch'),convergence=s.get('convergence',[]),beta_coverage=s.get('beta_coverage',[]),integer_minimum_bracketed=s.get('integer_minimum_bracketed',False),found_branches=s.get('found_branches',[]),source=str(p.relative_to(ROOT)),search_scope=s.get('search_scope'),branch_minima=s.get('branch_minima',[]),coarse_probe_coverage=s.get('coarse_probe_coverage',[]),search_complete=s.get('search_complete',False),unresolved_seed_families=s.get('unresolved_seed_families',[]),unresolved_branch_minima=s.get('unresolved_branch_minima',[]),producer_codes=s.get('producer_codes',[]),validator_code=s.get('validator_code'))
  rows.append(row)
 dump(OUT/'absolute_table.json',dict(schema='roscan-absolute-v1',code=CODE,rows=rows,completion='numerically_validated' if all(r['status']=='numerically_validated' for r in rows) else 'partial',frequency_mapping='Omega_BEK=beta*(Omega_disk/Omega_BEK)-Re*conj(omega_raw); alpha_global=conj(alpha_raw). Frequency scale ratio=1/rotation_rates(ro)[0]. Negative-Ro convention.',regression_reference_ro_minus1=507.3,regression_ro0751='Old phaseA did not compute local absolute critical point; external409.6 is comparison only'))
 lines=['# S0 absolute-neutral survey','','Finite search; each status refers to computed evidence. All candidates and failures are retained.','','| Ro | status | Re_ca | beta | bracket |','|---:|---|---:|---:|---|']
 for r in rows:lines.append(f"| {r['ro']} | {r['status']} | {r['Re_ca']} | {r['beta_c']} | {r['integer_minimum_bracketed']} |")
 lines+=['','A verified finite-path Briggs diagnostic does not exclude all undiscovered saddles.','Each minimum needs independently passed n and zmax checks (<0.1%).','Ro=-0.751 phaseA provides no local-absolute computed regression; 409.6 is an external comparison.']
 (OUT/'REPORT.md').write_text('\n'.join(lines)+'\n')
def main():
 p=argparse.ArgumentParser(description=__doc__);p.add_argument('stage',choices=['benchmark','scan','summary','evidence-review']);p.add_argument('--ro',type=float);p.add_argument('--n',type=int,default=128);p.add_argument('--budget',type=float,default=3000);p.add_argument('--all',action='store_true');a=p.parse_args();OUT.mkdir(parents=True,exist_ok=True);deadline=time.monotonic()+a.budget
 if a.stage=='benchmark':
  start=time.perf_counter();calc=Calculator(-1,a.n,40.,deadline);r,_=calc.critical(68);dump(OUT/'benchmark.json',dict(result=r,wall_seconds=time.perf_counter()-start,peak_rss_kb=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,threads=1));print(json.dumps({k:r[k] for k in ['ro','beta','Re_ca','seconds','evaluations']}))
 elif a.stage=='evidence-review':
  for ro in ROS+[-.751]:
   path=OUT/tag(ro)/'survey.json'
   if not path.exists():continue
   rec=json.loads(path.read_text())
   if rec.get('code') not in compatible_codes():continue
   dump(OUT/'validation_input_snapshot'/f'{tag(ro)}.json',rec);dump(path,review_survey(rec))
  summarize()
 elif a.stage=='scan':
  anchor=None
  for ro in (ROS+[-.751] if a.all else [a.ro]):
   try:
    available=[r for rr in ROS+[-.751] for r in load_candidates(rr) if r.get('seed_label')=='typeI_continuation']
    if available:anchor=min(available,key=lambda r:(abs(r['ro']-ro),r['Re_ca']))
    r=survey(ro,a.n,deadline,anchor)
    if r:anchor=r['candidate_minimum']
   except BudgetExceeded:break
   except Exception as ex:dump(OUT/tag(ro)/'stage_failure.json',dict(error=str(ex),traceback=traceback.format_exc()))
   summarize()
  summarize()
 else:summarize()
if __name__=='__main__':main()
