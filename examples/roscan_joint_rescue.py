#!/usr/bin/env python3
"""Budgeted joint Ro/continuous-beta seed rescue; canonical solver stays unchanged."""
from pathlib import Path
import argparse,hashlib,json,math,os,resource,sys,time,traceback
ROOT=Path(__file__).resolve().parents[1];sys.path[:0]=[str(ROOT/'src'),str(ROOT/'examples')]
for k in ('OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS','BLIS_NUM_THREADS','NUMEXPR_NUM_THREADS','NUMEXPR_MAX_THREADS','VECLIB_MAXIMUM_THREADS'):os.environ[k]='1'
resource.setrlimit(resource.RLIMIT_AS,(80_000_000_000,80_000_000_000))
import roscan_absolute as m
from llsa.secondary.discretization import rotation_rates
parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--budget',type=float,default=480);a=parser.parse_args();OUT=m.OUT/'joint_rescue';OUT.mkdir(exist_ok=True);started=time.monotonic();deadline=started+a.budget;helper_sha=hashlib.sha256(Path(__file__).read_bytes()).hexdigest();canonical=ROOT/'examples/roscan_absolute.py';canonical_sha=hashlib.sha256(canonical.read_bytes()).hexdigest();events=json.loads((m.OUT.parent/'supplement_trial_events.json').read_text());prior=next(x['cumulative_upper_seconds'] for x in events if x['stage']=='S0_cumulative_budget')+next(x['wall_seconds'] for x in events if x['stage']=='lowRo_supplement_finished')
if prior+a.budget+20>3600:raise SystemExit('S0 cumulative budget insufficient for joint rescue')
if not any(x['stage'] in ('Ro_minus0p9_trial_started','Ro_minus0p9_trial_finished','Ro_minus0p9_trial_deferred') for x in events):raise SystemExit('Shared summary writer has not finished')
proof_paths=[m.OUT/m.tag(ro)/'survey.json' for ro in [-1.,-.9]];proofs={str(p):hashlib.sha256(p.read_bytes()).hexdigest() for p in proof_paths};source_seed=m.OUT/m.tag(-.6)/'survey.json';seed_survey=json.loads(source_seed.read_text())
if seed_survey['status']!='numerically_validated':raise SystemExit('Ro=-0.6 source branch not verified')
state=dict(seed_survey['candidate_minimum']);state['beta_requested']=float(state['beta']);old_failures=[];reuses=[];steps=[];results=[]
for ro in [-.5,-.4]:
 for f in (m.OUT/m.tag(ro)/'failures').glob('*.json'):
  record=json.loads(f.read_text());archive=OUT/'failure_archive'/m.tag(ro)/f.name;archive.parent.mkdir(parents=True,exist_ok=True);archive.write_bytes(f.read_bytes());old_failures.append((record,str(archive)))
OriginalCalculator=m.Calculator
class FailureCachedCalculator(OriginalCalculator):
 def critical(self,beta,seed=(.218,.122,507.3),target=.035,reference=None):
  if self.n==64:
   for r,path in old_failures:
    if r.get('n')!=64 or abs(r.get('ro',9)-self.ro)>1e-12 or abs(r.get('beta',9e9)-beta)>1e-12:continue
    previous=r.get('seed',[]);w=complex(*r.get('target',[0,0]))
    if len(previous)==3 and max(abs(float(x)-float(y)) for x,y in zip(previous,seed))<1e-10 and abs(w-target)<1e-10:
     reuses.append(dict(ro=self.ro,beta=beta,seed=list(seed),target=m.pair(complex(target)),evidence=path));raise RuntimeError('Exact previously failed coarse seed reused without new eigensolve: '+path)
  return super().critical(beta,seed,target,reference)
m.Calculator=FailureCachedCalculator
try:
 for target_ro in [-.5,-.4]:
  if time.monotonic()>deadline:break
  reference_ro=state['ro'];reference_beta=state['beta_requested'];attempt=0
  while state['ro']<target_ro-1e-12:
   delta=min(.005,target_ro-state['ro']);success=False
   while delta>=1e-6:
    if time.monotonic()>deadline:raise m.BudgetExceeded('Joint continuation budget reached')
    rnew=min(target_ro,state['ro']+delta);bnew=reference_beta*abs(reference_ro)/abs(rnew);attempt+=1;record=dict(requested_ro=rnew,requested_continuous_beta=bnew,seed_ro=state['ro'],seed_true_beta=state['beta_requested'],step=delta,helper_sha256=helper_sha,kernel_code=m.CODE)
    try:
     calc=OriginalCalculator(rnew,128,40.,deadline);new,_=calc.critical(bnew,(*state['alpha'],state['Re_ca']),complex(*state['omega_raw']));new['calculator_integer_beta_field']=new['beta'];new['beta']=float(bnew);new['beta_requested']=float(bnew);new['status']='noninteger_neutral_seed_only';new['helper_sha256']=helper_sha;new['seed_source']=str(source_seed.relative_to(ROOT));expected=bnew*rotation_rates(rnew)[0]-new['Re_ca']*complex(*new['omega_raw']).conjugate()
     if abs(expected-complex(*new['omega_BEK']))>1e-9:raise RuntimeError('True-beta frequency mapping inconsistency')
     record['status']='converged';record['result']=new;state=new;success=True
    except m.BudgetExceeded:raise
    except Exception as ex:record['status']='failed_halve_step';record['error']=str(ex);delta/=2
    steps.append(record);m.dump(OUT/f'target{target_ro:+.3f}_step{attempt:04d}.json',record)
    if success:break
   if not success:raise RuntimeError('Joint continuation exhausted minimum Ro step')
  # The original calculator solves the neighboring INTEGER beta values afresh.
  for beta in sorted(set([math.floor(state['beta_requested']),math.ceil(state['beta_requested'])])):
   if time.monotonic()>deadline:raise m.BudgetExceeded('Integer seed budget reached')
   calc=OriginalCalculator(target_ro,128,40.,deadline);candidate,_=calc.critical(beta,(*state['alpha'],state['Re_ca']),complex(*state['omega_raw']));candidate.update(seed_label='typeI_continuation',helper_sha256=helper_sha,seed_source=str(source_seed.relative_to(ROOT)),true_integer_beta=beta);m.dump(m.OUT/m.tag(target_ro)/'candidates'/f'beta{beta}_typeI_continuation_N128.json',candidate)
  result=m.survey(target_ro,128,deadline,state);results.append(dict(ro=target_ro,status=result['status'],Re_ca=result.get('candidate_minimum',{}).get('Re_ca'),beta=result.get('candidate_minimum',{}).get('beta')));print(json.dumps(results[-1]),flush=True)
  if result['status']!='numerically_validated':break
  state=dict(result['candidate_minimum']);state['beta_requested']=float(state['beta']);source_seed=m.OUT/m.tag(target_ro)/'survey.json'
except Exception as ex:m.dump(OUT/'stopped.json',dict(error=str(ex),traceback=traceback.format_exc(),steps_completed=len(steps)))
finally:
 if hashlib.sha256(canonical.read_bytes()).hexdigest()!=canonical_sha:raise RuntimeError('Canonical source changed')
 for p,h in proofs.items():
  if hashlib.sha256(Path(p).read_bytes()).hexdigest()!=h:raise RuntimeError('S1 evidence survey changed')
 m.summarize();m.dump(OUT/'resources.json',dict(wall_seconds=time.monotonic()-started,peak_RSS_kb=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,address_space_cap=80000000000,S0_prior_upper_seconds=prior,S0_cumulative_upper_seconds=prior+time.monotonic()-started,helper_sha256=helper_sha,canonical_sha256=canonical_sha,kernel_code=m.CODE,source_survey_sha256=proofs,exact_failed_seed_reuses=reuses,results=results));print(json.dumps(dict(joint_rescue_results=results,wall_seconds=time.monotonic()-started)),flush=True)
