#!/usr/bin/env python3
"""Validate/resume a source-verified continuous-Ro/beta neutral seed. Single process."""
from pathlib import Path
import argparse,hashlib,json,math,os,resource,sys,time,traceback
ROOT=Path(__file__).resolve().parents[1];BASE=ROOT/'results/ro_scan_20261101';S0=BASE/'S0';sys.path[:0]=[str(ROOT/'src'),str(ROOT/'examples')]
for k in ('OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS','BLIS_NUM_THREADS','NUMEXPR_NUM_THREADS','NUMEXPR_MAX_THREADS','VECLIB_MAXIMUM_THREADS'):os.environ[k]='1'
p=argparse.ArgumentParser(description=__doc__);p.add_argument('--resume',required=True);p.add_argument('--budget',type=float,default=3000);p.add_argument('--validate-only',action='store_true');p.add_argument('--targets',nargs='+',type=float,default=[-.5,-.4]);p.add_argument('--max-ro-step',type=float,default=.005);a=p.parse_args()
if a.budget<=0 or not 0<a.max_ro_step<=.005:p.error('budget>0 and 0<max-ro-step<=.005 required')
resource.setrlimit(resource.RLIMIT_AS,(80_000_000_000,80_000_000_000))
import roscan_absolute as m
from llsa.secondary.discretization import rotation_rates
seed_path=Path(a.resume)
if not seed_path.exists():seed_path=BASE/seed_path
seed_path=seed_path.resolve()
if not seed_path.is_relative_to(S0):raise SystemExit('Resume seed must belong to task S0 output')
record=json.loads(seed_path.read_text());state=dict(record['result']);ro=float(record['requested_ro']);beta=float(record['requested_continuous_beta']);alpha=complex(*state['alpha']);omega=complex(*state['omega_raw']);re=float(state['Re_ca'])
producer_snapshot=ROOT/record['producer_snapshot'] if record.get('producer_snapshot') else S0/'joint_rescue/roscan_joint_rescue_producer.py'
producer_sha=hashlib.sha256(producer_snapshot.read_bytes()).hexdigest();declared=record.get('helper_sha256',record.get('producer_source_sha256'))
checks=dict(producer_snapshot_hash=producer_sha==declared,kernel_hash=record.get('kernel_code')==m.CODE and state.get('code')==m.CODE,true_Ro=abs(state['ro']-ro)<1e-12,true_beta=abs(state.get('beta_requested',state['beta'])-beta)<1e-10 and abs(state['beta']-beta)<1e-10,finite_parameters=all(math.isfinite(x) for x in [ro,beta,re,alpha.real,alpha.imag,omega.real,omega.imag]) and -1<=ro<0 and beta>=0 and re>0,neutral_residual=max(abs(x) for x in state['neutral_equation_residual'])<1e-8 and state['eigen_residual']<1e-8,frequency_mapping=abs(beta*rotation_rates(ro)[0]-re*omega.conjugate()-complex(*state['omega_BEK']))<1e-9,forward_targets=all(ro<t<0 for t in a.targets) and a.targets==sorted(a.targets))
validation=dict(validated=all(checks.values()),checks=checks,seed=str(seed_path.relative_to(ROOT)),seed_sha256=hashlib.sha256(seed_path.read_bytes()).hexdigest(),producer_snapshot=str(producer_snapshot.relative_to(ROOT)),producer_sha256=producer_sha,kernel_code=m.CODE,true_Ro=ro,true_beta=beta,Re=re,alpha=state['alpha'],omega_raw=state['omega_raw'],targets=a.targets,validate_only=a.validate_only,computation_started=False)
if not validation['validated']:print(json.dumps(validation));raise SystemExit(2)
if a.validate_only:
 m.dump(S0/'joint_resume_parameter_validation.json',validation);print(json.dumps(validation));raise SystemExit(0)
if a.budget>3600 and os.environ.get('ROSCAN_SUPERVISOR_LONG')!='1':raise SystemExit('Budget>1h requires supervisor launch with ROSCAN_SUPERVISOR_LONG=1')
started=time.monotonic();deadline=started+a.budget;job=S0/f'joint_resume_{time.strftime("%Y%m%dT%H%M%S",time.gmtime())}_{os.getpid()}';job.mkdir();self_source=Path(__file__).read_bytes();self_sha=hashlib.sha256(self_source).hexdigest();self_snapshot=job/'producer_source.py';self_snapshot.write_bytes(self_source);m.dump(job/'input_validation.json',validation);canonical_sha=hashlib.sha256((ROOT/'examples/roscan_absolute.py').read_bytes()).hexdigest();proofs={str(S0/m.tag(rr)/'survey.json'):hashlib.sha256((S0/m.tag(rr)/'survey.json').read_bytes()).hexdigest() for rr in [-1.,-.9]};state['beta_requested']=beta;steps=[];results=[]
try:
 for target in a.targets:
  start_ro=state['ro'];start_beta=state['beta_requested'];attempt=0
  while state['ro']<target-1e-12:
   delta=min(a.max_ro_step,target-state['ro']);success=False
   while delta>=1e-6:
    if time.monotonic()>deadline:raise m.BudgetExceeded('Resume budget reached')
    rnew=min(target,state['ro']+delta);bnew=start_beta*abs(start_ro)/abs(rnew);attempt+=1;rec=dict(requested_ro=rnew,requested_continuous_beta=bnew,seed_ro=state['ro'],seed_true_beta=state['beta_requested'],step=delta,helper_sha256=self_sha,kernel_code=m.CODE,producer_snapshot=str(self_snapshot.relative_to(ROOT)))
    try:
     calc=m.Calculator(rnew,128,40.,deadline);new,_=calc.critical(bnew,(*state['alpha'],state['Re_ca']),complex(*state['omega_raw']));new['calculator_integer_beta_field']=new['beta'];new.update(beta=float(bnew),beta_requested=float(bnew),status='noninteger_neutral_seed_only');rec.update(status='converged',result=new);state=new;success=True
    except m.BudgetExceeded:raise
    except Exception as ex:rec.update(status='failed_halve_step',error=str(ex));delta/=2
    steps.append(rec);m.dump(job/f'target{target:+.3f}_step{attempt:04d}.json',rec)
    if success:break
   if not success:raise RuntimeError('Minimum Ro step reached')
  for integer in sorted(set([math.floor(state['beta_requested']),math.ceil(state['beta_requested'])])):
   calc=m.Calculator(target,128,40.,deadline);candidate,_=calc.critical(integer,(*state['alpha'],state['Re_ca']),complex(*state['omega_raw']));candidate.update(seed_label='typeI_continuation',true_integer_beta=integer,helper_sha256=self_sha,seed_source=str(seed_path.relative_to(ROOT)),producer_snapshot=str(self_snapshot.relative_to(ROOT)));m.dump(S0/m.tag(target)/'candidates'/f'beta{integer}_typeI_continuation_N128.json',candidate)
  # Preserve original failures; continuation now supplies a new alpha/Re/omega seed.
  for failed in (S0/m.tag(target)/'failures').glob('*.json'):
   archived=job/'prior_failure_archive'/m.tag(target)/failed.name;archived.parent.mkdir(parents=True,exist_ok=True);archived.write_bytes(failed.read_bytes())
  survey=m.survey(target,128,deadline,state);results.append(dict(ro=target,status=survey['status'],beta=survey.get('candidate_minimum',{}).get('beta'),Re=survey.get('candidate_minimum',{}).get('Re_ca')))
  if survey['status']!='numerically_validated':break
  state=dict(survey['candidate_minimum']);state['beta_requested']=float(state['beta'])
except Exception as ex:m.dump(job/'stopped.json',dict(error=str(ex),traceback=traceback.format_exc()))
finally:
 if hashlib.sha256((ROOT/'examples/roscan_absolute.py').read_bytes()).hexdigest()!=canonical_sha:raise RuntimeError('Canonical source changed')
 for path,digest in proofs.items():
  if hashlib.sha256(Path(path).read_bytes()).hexdigest()!=digest:raise RuntimeError('S1 evidence survey changed')
 m.summarize();m.dump(job/'completion.json',dict(status='numerically_validated' if len(results)==len(a.targets) and all(r['status']=='numerically_validated' for r in results) else 'partial',results=results,seconds=time.monotonic()-started,peak_RSS_kb=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,true_last_beta=state['beta_requested'],last_Ro=state['ro'],producer_sha256=self_sha,kernel_code=m.CODE));print(json.dumps(dict(job=str(job.relative_to(ROOT)),results=results)))
