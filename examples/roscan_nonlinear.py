#!/usr/bin/env python3
"""Restartable S2 full-BVP campaigns. Production is launched by the supervisor."""
import os
for key in ('OPENBLAS_NUM_THREADS','OMP_NUM_THREADS','MKL_NUM_THREADS','NUMEXPR_NUM_THREADS','VECLIB_MAXIMUM_THREADS','BLIS_NUM_THREADS'):
    os.environ[key]='1'
import argparse, contextlib, fcntl, gc, hashlib, json, math, resource, subprocess, sys, time, traceback
from pathlib import Path
import numpy as np
from roscan_nonlinear_solver import ROOT, Mapping, Metric, pack, unpack, common, energies, transfer, write
from ro0751_phaseB_campaign import fold_fit
from roscan_nonlinear_fronts import diagnose, overlap, FRONTS
from roscan_nonlinear_resources import (memory_profile, worker_limit, worker_rss_reservation, acquire_numerical_locks, reserve_worker_bytes, MemoryQueueTimeout, resource_plan)
BASE=ROOT/'results/ro_scan_20261101'
OUT=BASE/'S2'
MEMORY=200_000_000_000

def read(path):return json.loads(Path(path).read_text())
def sha(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()
def slug(ro):return f'ro_m{abs(ro):.3f}'.replace('.','p')
def row_for(ro):
    source='S1b' if ro>=-.4000001 else 'S1'
    rows=read(BASE/source/'plan.json')['rows']
    row=next(x for x in rows if abs(x['ro']-ro)<1e-8).copy()
    row['linear_stage']=source
    return row

def inputs(row, required=True):
    b=BASE/row['linear_stage']/slug(row['ro'])
    paths={k:b/name for k,name in dict(seed='N32/neutral_state.npz',neutral='N32/neutral.json',curve='rend_N32.json',summary='summary.json').items()}
    if not all(p.is_file() for p in paths.values()):
        if required:raise RuntimeError(f"Await validated {row['linear_stage']} N32 neutral and local-growth results: {b}")
        return paths,None
    neutral,summary=read(paths['neutral']),read(paths['summary'])
    if summary.get('status')!='numerically_validated':
        if not required:return paths,None
        raise RuntimeError('Linear summary is not validated')
    curve=read(paths['curve'])
    if curve.get('status')!='numerically_validated' or abs(curve.get('ro',0)-row['ro'])>1e-10 or curve.get('beta')!=row['beta_c']:
        raise RuntimeError('S1 local-growth curve metadata mismatch')
    if neutral['n']!=32 or abs(neutral['ro']-row['ro'])>1e-10 or neutral['beta']!=row['beta_c']:
        raise RuntimeError('Linear neutral metadata mismatch')
    if abs(neutral['Rin']-row['Rin'])>1e-8 or abs(neutral['scale']-row['scale'])>1e-10:
        raise RuntimeError('S1 geometry differs from plan')
    if abs(neutral['sigma'])>2e-8 or neutral['residual']>1e-8:raise RuntimeError('Nonneutral or unresolved seed')
    with np.load(paths['seed']) as d:
        if (float(d['ro'])!=row['ro'] or float(d['beta'])!=row['beta_c'] or
            abs(float(d['sigma_residual']))>2e-8 or len(d['radii'])!=neutral['intervals']+1 or
            abs(float(d['radii'][-1])-neutral['rout'])>1e-8 or len(d['xi'])!=33 or
            abs(float(d['omega_c'])-neutral['omega'])>1e-9 or not np.isfinite(d['q']).all()):
            raise RuntimeError('Neutral state content mismatch')
    return paths,neutral

def quality(d):
    ratio=d.get('highest_to_fundamental')
    return bool(d.get('accepted') and isinstance(ratio,(int,float)) and math.isfinite(ratio) and
                0<=d['highest_to_fundamental']<=.01 and
                d.get('independent',{}).get('relative_field_inf',math.inf)<1e-7 and
                d.get('independent',{}).get('template_difference_relative',math.inf)<=1e-10)

class Pause(Exception):pass
class Campaign:
    def __init__(self,ro,point_budget=3600,wall_budget=86400):
        self.parallel=os.environ.get('S2_PARALLEL','0')=='1'
        self.memory_profile=memory_profile(ro) if self.parallel else None
        if self.parallel:
            resource.setrlimit(resource.RLIMIT_AS,(self.memory_profile['driver_AS_bytes'],self.memory_profile['driver_hard_AS_bytes']))
        else:
            resource.setrlimit(resource.RLIMIT_AS,(64_000_000_000,200_000_000_000))
        self.out=OUT/slug(ro)
        self.global_lock,self.lock=acquire_numerical_locks(self.out,self.parallel)
        self.row=row_for(ro);self.paths,self.neutral=inputs(self.row)
        checks=read(OUT/'short_checks/verification.json')
        if not checks.get('passed'):raise RuntimeError('Preparation checks missing or failed')
        sources=list((ROOT/'src/llsa').rglob('*.py'))+list((ROOT/'examples').glob('roscan_nonlinear*.py'))
        sources += [ROOT/'examples'/n for n in ['ro0751_phaseB_campaign.py','ro0751_phaseB_solver.py','appelquist2015_hopf_common.py','appelquist2015_hopf_block_corrector.py','appelquist2016_branch_continuation.py']]
        if checks.get('sources')!={str(p.relative_to(ROOT)):sha(p) for p in sorted(set(sources)) if p.name.startswith('roscan_nonlinear')}:
            raise RuntimeError('Preparation checks are stale for current S2 source')
        current=dict(files={str(p.relative_to(ROOT)):sha(p) for p in sorted(set(sources+list(self.paths.values())))},row=self.row)
        if self.parallel:current['parallel_memory_profile']=self.memory_profile
        contract=self.out/'source_contract.json'
        if contract.exists():
            if read(contract)!=current:raise RuntimeError('Source or S1 inputs changed since preparation; supervisor review required')
        else:write(contract,current)
        self.state=read(self.out/'campaign.json') if (self.out/'campaign.json').exists() else dict(
            schema='roscan-fullbvp-v1',ro=ro,beta=self.row['beta_c'],status='prepared',stage='seeds',points=[],attempts=[],
            branch_ids=[],anchors=[],overlaps=[],fold=None,mesh=None,next_step=.04,failures=[])
        self.start=time.monotonic();self.deadline=self.start+wall_budget;self.point_budget=point_budget
        self.mapping=None;self.metric=None
    def save(self):
        self.state['elapsed_current_invocation']=time.monotonic()-self.start
        write(self.out/'campaign.json',self.state)
    def point(self,pid):return next(p for p in self.state['points'] if p['id']==pid)
    def mapping_for(self,H=4,n=32,intervals=None):
        if n==40 and not any(g['n']==40 and g.get('execution_allowed',True) for g in self.row['grids']):
            raise RuntimeError('N40 blocked by S1b resource policy; supervisor/HPC review required; do not bypass the 200-GB cap')
        if intervals is None:intervals=self.neutral['intervals']
        spec=dict(n=n,H=H,intervals=intervals,ro=self.row['ro'],beta=self.row['beta_c'],
                  scale=self.row['scale'],Rin=self.row['Rin'],reference_R=self.neutral['rout'])
        return Mapping(**spec),spec
    def qref(self,mapping):
        with np.load(self.paths['seed']) as d:
            q=d['q'];r=d['radii'];xi=d['xi']
        from scipy.interpolate import BarycentricInterpolator,interp1d
        if len(xi)!=mapping.m:q=BarycentricInterpolator(xi,q,axis=-1)(mapping.op.grid.xi)
        q=interp1d(r,q,axis=0,kind='cubic',bounds_error=False,fill_value='extrapolate')(mapping.Rin+(self.neutral['rout']-mapping.Rin)*mapping.xi).reshape(mapping.nr,mapping.s)
        p,*_=mapping.at(self.neutral['rout'],self.neutral['omega'])
        return q/common.mass_norm(p,q)
    def attempt(self,mapping,y,mode,target,name,role,tangent=None,weak=False):
        prior=next((p for p in self.state['points'] if p['id']==name),None)
        if prior is not None and prior['accepted']:
            return transfer(prior['fields'],mapping),prior
        if self.deadline-time.monotonic()<20:raise Pause('wall_budget_before_next_point')
        spec=dict(n=mapping.n,H=mapping.H,intervals=mapping.intervals,ro=self.row['ro'],beta=self.row['beta_c'],
                  scale=mapping.scale,Rin=mapping.Rin,reference_R=mapping.reference_R)
        qref=self.qref(mapping)
        digest=hashlib.sha256(y.tobytes()+qref.tobytes()+(b'' if tangent is None else tangent.tobytes())).hexdigest()
        signature=hashlib.sha256(json.dumps([spec,mode,target,digest],sort_keys=True).encode()).hexdigest()
        folder=self.out/'attempts'/name;folder.mkdir(parents=True,exist_ok=True);result=folder/'result';summary=result/'summary.json'
        d=None
        if summary.exists():
            try:cached=read(summary)
            except (ValueError,OSError):cached={}
            if cached.get('request_sha256')==signature and cached.get('accepted') and (result/'fields.npz').exists():d=cached
        if d is None:
            if result.exists():result.rename(folder/f'archive_{time.time_ns()}')
            np.savez_compressed(folder/'input.npz',y=y,qref=qref,tangent=np.array([]) if tangent is None else tangent)
            budget=min(self.point_budget,max(1,self.deadline-time.monotonic()-5))
            request=dict(mapping=spec,mode=mode,target=float(target),input=str(folder/'input.npz'),output=str(result),
                         request_sha256=signature,budget=budget,weak_predictor=weak,rescale_energy=mode=='energy')
            parallel=getattr(self,'parallel',False)
            cap=worker_limit(self.memory_profile,mapping.n,mapping.H,mapping.intervals) if parallel else MEMORY
            reserved_rss=worker_rss_reservation(self.memory_profile,mapping.n,mapping.H,mapping.intervals) if parallel else None
            if parallel:request.update(worker_AS_bytes=cap,RSS_reserved_bytes=reserved_rss,resource_mode='parallel_RSS_tokens')
            write(folder/'request.json',request);began=time.monotonic()
            try:
                reservation=reserve_worker_bytes(reserved_rss,self.deadline) if parallel else contextlib.nullcontext(())
                with reservation as token_fds:
                    budget=min(self.point_budget,max(1,self.deadline-time.monotonic()-5))
                    request['budget']=budget;write(folder/'request.json',request)
                    with (folder/'worker.log').open('a') as log:
                        proc=subprocess.run([sys.executable,str(ROOT/'examples/roscan_nonlinear_solver.py'),'--request',str(folder/'request.json')],
                                            cwd=ROOT,stdout=log,stderr=log,timeout=budget+2,pass_fds=token_fds)
                d=read(summary) if summary.exists() else dict(accepted=False,failure=f'worker_exit_{proc.returncode}_without_summary')
                if proc.returncode:d.update(accepted=False,failure=f'worker_exit_{proc.returncode}')
            except subprocess.TimeoutExpired:d=dict(accepted=False,failure='hard_point_timeout',budget_stop=True)
            except MemoryQueueTimeout:d=dict(accepted=False,failure='wall_budget_memory_queue',budget_stop=True)
            d.update(seconds=time.monotonic()-began,request_sha256=signature);write(summary,d)
        equation=bool(d.get('accepted'));accepted=quality(d)
        self.state['attempts']=[x for x in self.state['attempts'] if x['id']!=name]+[dict(id=name,role=role,
              equations_accepted=equation,accepted=accepted,seconds=d.get('seconds'),failure=d.get('failure'),summary=str(summary.relative_to(ROOT)))]
        point=dict(id=name,role=role,accepted=accepted,equations_accepted=equation,fields=str(result/'fields.npz'),summary=str(summary))
        if equation:
            point.update({k:d[k] for k in ['Rout','omega','A1','n','H','intervals','highest_to_fundamental','relative_field_inf','independent']})
            point.update(diagnose(result/'fields.npz',self.paths['curve'],self.row['scale']))
            point['dr']=(point['Rout']-mapping.Rin)/mapping.intervals
            self.state['points']=[x for x in self.state['points'] if x['id']!=name]+[point]
        self.save();gc.collect()
        if d.get('budget_stop'):raise Pause(d.get('failure','point_budget'))
        return (transfer(result/'fields.npz',mapping) if equation else y.copy()),point
    def require(self,p):
        if not p['accepted']:raise RuntimeError('Corrector or harmonic truncation failed: '+p['id'])
    def seed(self):
        mapping,_=self.mapping_for();q1=self.qref(mapping)
        for k,A in enumerate((.0064,.009)):
            name=f'seed_{k}'
            if any(p['id']==name and p['accepted'] for p in self.state['points']):continue
            q=np.zeros((mapping.nr,2*mapping.H+1,mapping.s),complex)
            q[:,mapping.H+1]=A*q1;q[:,mapping.H-1]=q[:,mapping.H+1].conj()
            y=np.r_[pack(q),self.neutral['omega'],self.neutral['rout']]
            _,p=self.attempt(mapping,y,'energy',A,name,'weak_seed',weak=True);self.require(p)
        seeds=[self.point('seed_0'),self.point('seed_1')]
        fits={k:np.polyfit([p['A1']**2 for p in seeds],[p[k] for p in seeds],1).tolist() for k in ('Rout','omega')}
        check=dict(fits=fits,Rout_intercept_error=fits['Rout'][1]-self.neutral['rout'],
                   omega_intercept_error=fits['omega'][1]-self.neutral['omega'],subcritical=all(p['Rout']<self.neutral['rout'] for p in seeds))
        check['passed']=bool(max(abs(check['Rout_intercept_error']),abs(check['omega_intercept_error']))<.002)
        self.state['weak_seed_verification']=check;write(self.out/'weak_seed_verification.json',check)
        if not check['passed']:raise RuntimeError('Weak seed does not recover S1 neutral limit')
        self.state.update(stage='branch_fold',anchors=['seed_0','seed_1'],branch_ids=['seed_0','seed_1']);self.save()
    def upgrade(self):
        anchors=[self.point(k) for k in self.state['anchors']];H=anchors[-1]['H']
        if H>=6:raise RuntimeError('H6 truncation unresolved; no H7 and no relaxed tolerance')
        mapping,_=self.mapping_for(H+1);new=[]
        for low in anchors:
            name=f"overlap_{low['id']}_H{H+1}"
            y=transfer(low['fields'],mapping)
            _,high=self.attempt(mapping,y,'Rout',low['Rout'],name,'harmonic_overlap');self.require(high)
            test=overlap(low,high)
            self.state['overlaps']=[x for x in self.state['overlaps'] if x['high']!=name]+[test];self.save()
            if not test['passed']:raise RuntimeError('Same-Rout harmonic overlap failed')
            new.append(name)
        self.state['anchors']=new;self.save()
    def branch_step(self):
        previous,current=[self.point(k) for k in self.state['anchors']]
        mapping,_=self.mapping_for(current['H']);metric=Metric(mapping)
        old=transfer(previous['fields'],mapping);y=transfer(current['fields'],mapping);t=metric.tangent(old,y)
        ds=self.state['next_step'];idx=len(self.state['branch_ids'])
        for retry in range(5):
            trial,p=self.attempt(mapping,y+ds*t,'arc',0,f'branch_{idx:04d}_H{mapping.H}_try{retry}','primary_arclength',tangent=t)
            if p['equations_accepted'] and not p['accepted']:
                self.upgrade();return
            if p['accepted']:
                delta=trial-y
                forward=metric.dot(delta,t)/max(math.sqrt(metric.dot(delta,delta)),1e-30)
                if forward<=.1:
                    p.update(accepted=False,failure='arclength_orientation');self.save();ds*=.5;continue
                nt=metric.tangent(y,trial)
                p['upper_branch']=bool(current.get('upper_branch') or t[-1]<0<nt[-1])
                if current.get('upper_branch') and t[-1]>0>nt[-1]:raise RuntimeError('Upper-branch return turning point')
                self.state['branch_ids'].append(p['id']);self.state['anchors']=[current['id'],p['id']]
                self.state['next_step']=min(.16,ds*1.15);self.save();return
            ds*=.5
        raise RuntimeError('Five failed arclength corrections')
    def detect_fold(self):
        pts=[self.point(k) for k in self.state['branch_ids'] if self.point(k)['accepted']]
        fit=fold_fit(pts)
        if fit['Rout'] is None:return False
        ordered=sorted(pts,key=lambda p:p['A1']);i=min(range(len(ordered)),key=lambda k:ordered[k]['Rout'])
        if len(ordered)-i<3:return False
        self.state.update(stage='fold_refine',fold_initial=fit,fold_refine_H=ordered[i]['H']);self.save();return True
    def refine_fold(self):
        af=self.state['fold_initial']['A1'];H=self.state['fold_refine_H'];mapping,_=self.mapping_for(H)
        spacing=max(.002,min(.004,.04*af));amps=[af+k*spacing for k in (-2,-1,0,1,2)]
        for j in range(9):
            subset=[p for p in self.state['points'] if p['role']=='fold_refinement' and p['accepted']]
            if j>=5:
                fit=fold_fit(subset)
                if fit['Rout'] is not None:break
                ordered=sorted(subset,key=lambda p:p['A1']);i=min(range(len(ordered)),key=lambda k:ordered[k]['Rout'])
                A=ordered[0]['A1']-spacing if i==0 else ordered[-1]['A1']+spacing
            else:A=amps[j]
            candidates=[p for p in self.state['points'] if p['accepted'] and p['n']==32 and p['H']==H]
            source=min(candidates,key=lambda p:abs(p['A1']-A));y=transfer(source['fields'],mapping)
            _,p=self.attempt(mapping,y,'energy',A,f'fold_H{H}_{j:02d}','fold_refinement');self.require(p)
        subset=[p for p in self.state['points'] if p['role']=='fold_refinement' and p['accepted']]
        fit=fold_fit(subset)
        if fit['Rout'] is None:raise RuntimeError('Refined fold is not bracketed')
        fit.update(n=32,H=H,grid_validation='N40 only at upper endpoint; fold N40 not required')
        self.state.update(fold=fit,stage='upper');self.save()
    def stable(self):
        pts=[self.point(k) for k in self.state['branch_ids'] if self.point(k).get('upper_branch') and self.point(k)['accepted']]
        tail=pts[-3:];ranges={k:None for k in FRONTS}
        if len(tail)==3 and len({p['H'] for p in tail})==1 and tail[0]['Rout']>=self.state['fold']['Rout']+4:
            for k in FRONTS:
                vals=[p.get(k) for p in tail]
                if all(v is not None and math.isfinite(v) for v in vals):ranges[k]=max(vals)-min(vals)
        return all(v is not None and v<.2 for v in ranges.values()),ranges
    def upper_step(self):
        current=self.point(self.state['anchors'][-1]);mapping,_=self.mapping_for(current['H'])
        if current['Rout']>=self.state['fold']['Rout']+20 or self.stable()[0]:
            self.state.update(stage='mesh',endpoint=current['id'],front_converged=self.stable()[0],front_tail_ranges=self.stable()[1]);self.save();return
        step=self.state.get('upper_step',2.)
        for retry in range(5):
            target=min(current['Rout']+step,self.state['fold']['Rout']+20)
            y=transfer(current['fields'],mapping,Rout=target)
            _,p=self.attempt(mapping,y,'Rout',target,f"upper_{len(self.state['branch_ids']):04d}_H{mapping.H}_try{retry}",'upper_continuation')
            if p['equations_accepted'] and not p['accepted']:self.upgrade();return
            if p['accepted']:
                if p['A1']<current['A1']*.9:
                    p.update(accepted=False,branch_rejected='upper_branch_amplitude_drop_above_10_percent');self.save()
                    raise RuntimeError('Upper-branch amplitude drop exceeds 10 percent')
                p['upper_branch']=True
                self.state['anchors']=[current['id'],p['id']];self.state['branch_ids'].append(p['id']);self.state['upper_step']=min(2.,step*1.25);self.save();return
            step*=.5
        raise RuntimeError('Five failed fixed-Rout upper corrections')
    def mesh(self):
        source=self.point(self.state['endpoint']);H=source['H']
        for _ in range(3):
            intervals=int(math.ceil((source['Rout']-self.row['Rin'])/.25));mapping,_=self.mapping_for(H,40,intervals)
            y=transfer(source['fields'],mapping)
            _,point=self.attempt(mapping,y,'Rout',source['Rout'],f"N40_{source['Rout']:.9f}_H{H}",'mesh_check')
            if point['accepted']:
                if point['A1']<source['A1']*.9:
                    point.update(accepted=False,branch_rejected='N40_upper_branch_identity_lost');self.save()
                    raise RuntimeError('N40 lost upper-branch amplitude')
                diff={k:None if point.get(k) is None or source.get(k) is None else point[k]-source[k] for k in FRONTS}
                if any(v is None for v in diff.values()):raise RuntimeError('N40 or N32 front missing at endpoint')
                self.state.update(mesh=dict(source=source['id'],point=point['id'],same_H=H,front_differences=diff),
                                  status='complete',stage='done',stop=None);self.save();return
            if not point['equations_accepted'] or H>=6:raise RuntimeError('N40 corrector/truncation failure at H limit')
            H+=1;mapping,_=self.mapping_for(H);y=transfer(source['fields'],mapping)
            _,high=self.attempt(mapping,y,'Rout',source['Rout'],f"endpoint_N32_H{H}",'harmonic_overlap');self.require(high)
            test=overlap(source,high);self.state['overlaps'].append(test)
            if not test['passed']:raise RuntimeError('Endpoint harmonic overlap failed')
            source=high;self.state.update(endpoint=high['id'],front_converged=False,
                                         convergence_note='N40 required higher H; previous Rout stability measured at lower H');self.save()
    def run(self,trial=False,trial_steps=2):
        if self.state['stage']=='done' or (trial and self.state.get('trial_verification',{}).get('passed')):return
        self.state.update(status='running',stop=None);self.save();initial=len(self.state['branch_ids'])
        if self.state['stage']=='seeds':self.seed();initial=2
        if trial:
            while len(self.state['branch_ids'])<2+trial_steps:self.branch_step()
            self.state.update(status='trial_passed',stop='short_trial_complete',trial_verification=dict(passed=True,weak_seed_ids=['seed_0','seed_1'],branch_ids=self.state['branch_ids'][2:2+trial_steps]));self.save();return
        while self.state['stage']!='done':
            if self.state['stage']=='branch_fold':
                if not self.detect_fold():self.branch_step()
            elif self.state['stage']=='fold_refine':self.refine_fold()
            elif self.state['stage']=='upper':self.upper_step()
            elif self.state['stage']=='mesh':self.mesh()
            else:raise RuntimeError('Unknown campaign stage')

def prepare():
    rows=[]
    for ro in [-.9,-.8,-.7,-.6,-.5,-.4,-.3,-.2,-.1]:
        row=row_for(ro);paths,neutral=inputs(row,False)
        factor=(neutral['intervals'] if neutral else next(g['intervals_fixed_during_root'] for g in row['grids'] if g['n']==32))/730
        summary=read(paths['summary']) if neutral else {}
        rows.append(dict(row,seed=str(paths['seed'].relative_to(ROOT)),curve=str(paths['curve'].relative_to(ROOT)),
            execution_status='ready_pending_supervisor' if neutral else 'await_S1b',
            N40_policy='allowed_by_linear_resource_plan' if any(g['n']==40 and g.get('execution_allowed',True) for g in row['grids']) else 'blocked_by_S1b_resource_policy_supervisor_HPC_review',
            Re_ca=abs(ro)*row['R_ca'],Re_end_linear=summary.get('Re_end'),
            nonlinear=dict(N=32,H_initial=4,H_max=6,energy_ratio_limit=.01,upper_offset_R=20.,front_three_point_range_R=.2,N40_dr=.25),
            estimates=dict(N32_H4_point_minutes=[5*factor,15*factor],N40_H4_point_minutes=[13*factor,60*factor],
                N32_H6_peak_GB=[30*factor,100*factor],N40_H6_peak_GB=[70*factor,180*factor],
                campaign_hours=[6*factor,24*factor],basis='Ro=-0.751 timing anchor; linear interval scaling; H5/H6, retries and N40 may multiply time; memory is planning estimate, RLIMIT_AS is enforced')))
    plan=dict(schema='roscan-nonlinear-plan-v1',rows=rows,regressions=read(OUT/'regressions.json')['rows'],
        resources=dict(numerical_threads=1,serial_Ro=True,max_worker_AS_bytes=MEMORY,max_driver_AS_bytes=64_000_000_000,
                       task_AS_sum_bytes=264_000_000_000,parallel_numeric_processes=2,no_GPU=True),
        front_scaling='Fig8 [550*scale,min(650*scale,a)]; 2beta (420*scale,a); a=Rout-100*scale; z=1.3; physical threshold 1e-6 unchanged',
        source_templates=read(OUT/'template_hashes.json'))
    parallel=resource_plan()
    plan['resources'].update(parallel_available=True,parallel_policy='parallel_plan.json',legacy_ro09_must_remain_untouched=True,
        default_mode='serial',serial_Ro='default_mode_only',parallel_driver_AS_bytes=32_000_000_000,
        parallel_max_worker_AS_bytes=max(p['driver_hard_AS_bytes'] for p in parallel['profiles']),
        S2_RSS_budget_bytes=380_000_000_000,parallel_driver_RSS_reserved_bytes=4_000_000_000,parallel_numeric_processes_including_legacy=10,
        aggregate_Codex_cap_bytes=1_000_000_000_000,S1c_reserved_bytes=550_000_000_000)
    plan['resources'].pop('task_AS_sum_bytes',None)
    plan['resources']['accounting']='RSS budget; RLIMIT_AS independent'
    plan['parallel_resources']=parallel
    for row in plan['rows']:
        profile=next((p for p in parallel['profiles'] if abs(p['ro']-row['ro'])<1e-9),None)
        if profile is not None:
            row['parallel_memory_profile']=profile
            for n in (32,40):
                stage=profile['stages'][f'N{n}_H6']
                row['estimates'][f'N{n}_H6_peak_GB']=[stage['estimated_peak_RSS_bytes']/1e9,stage['RSS_reserved_bytes']/1e9]
            row['estimates']['parallel_memory_basis']=profile['model']
    write(OUT/'parallel_plan.json',parallel)
    write(OUT/'plan.json',plan);report(plan)

def report(plan=None):
    plan=read(OUT/'plan.json') if plan is None else plan
    lines=['# S2：Ro 扫描非线性全局模态','','本轮准备参数化 full BVP、独立数值检查及 Ro=-0.9 短试；生产长算由监工启动。',
      '原模板保持只读，SHA256 见 template_hashes.json。S1b 未验证的点保留 await_S1b。',
      '', '## 总表（R 为 BEK 半径；Re=|Ro|R）','',
      '| Ro | β | Re_ca | Re_end 线性 | Rout_f (N32) | Fig.8 R / Re | 2β R / Re | ω_BEK / ω_d | A₁ | 状态 |',
      '|---:|---:|---:|---:|---:|---|---|---|---:|---|']
    fmt=lambda x:'待算' if x is None else f'{x:.6f}'
    for reg in plan['regressions']:
        pair=lambda k:f"{fmt(reg.get(k))} / {fmt(reg.get(k+'_Re'))}"
        lines.append(f"| {reg['ro']} | {reg['beta']} | {fmt(reg['Re_ca'])} | {fmt(reg['Re_end_linear'])} | {fmt(reg['Rout_f'])} | {pair('front_fig8')} | {pair('front_2beta_1e6')} | {fmt(reg['omega_BEK'])} / {fmt(reg['omega_d'])} | {fmt(reg['A1'])} | 既有 N32/H{reg['H']}，Rout={reg['representative_Rout']:.6f} |")
    for row in plan['rows']:
        path=OUT/slug(row['ro'])/'campaign.json';state=read(path) if path.exists() else {}
        points=state.get('points',[]);endpoint=next((p for p in points if p['id']==state.get('endpoint')),None)
        if endpoint is None:
            accepted={p['id']:p for p in points if p['accepted'] and p.get('n')==32}
            candidate=next((accepted[k] for k in reversed(state.get('anchors',[])) if k in accepted),{})
            endpoint=candidate if candidate.get('upper_branch') else {}
        pair=lambda k:f"{fmt(endpoint.get(k))} / {fmt(endpoint.get(k+'_Re'))}"
        lines.append(f"| {row['ro']} | {row['beta_c']} | {row['Re_ca']:.4f} | {fmt(row['Re_end_linear'])} | {fmt((state.get('fold') or {}).get('Rout'))} | {pair('front_fig8')} | {pair('front_2beta_1e6')} | {fmt(endpoint.get('omega_BEK'))} / {fmt(endpoint.get('omega_d'))} | {fmt(endpoint.get('A1'))} | {state.get('status',row['execution_status'])} |")
    lines += ['', '## 定义与续算','',
      'Fig.8：z=1.3 处 d ln RMS(v)/dR 与已存 S1 N32 局部绝对增长率的首个下行交点，区间 [550s,min(650s,a)]。',
      '2β：|v₂| 随 R 增大首次上行到达 10⁻⁶，区间 (420s,a)。s 为各点 S1 scale，a=Rout−100s；阈值及 z 不缩放。',
      '两窗口只用局部曲线与全局径向网格的交集，缺失交点保留 null，不外推。ω_d=ω_BEK/[(2−Ro−Ro²)/2]。',
      'A₁=√E₁，为原 solver 的积分基频幅值；不是局部 v₁ 峰值。折点为有左右括点的三/五点二次拟合，非 DNS NL1 前沿。',
      '两弱种子 → H4 真伪弧长跨折点 → 五个固定能量点细化折点 → 上支固定 Rout 推进；到折点+20 或同 H 末三前沿全距均<0.2即止（首点至少折点+4）。',
      'E_H/E₁>0.01 时 H4→H5→H6，同 Rout 重叠验收：可用前沿差<0.1。两前沿齐全时以此前沿标准验收；下支前沿缺失时明确记录缺失，并要求公共谐波速度场质量范数相对差<5%。H6仍截断即停。',
      '末点固定 Rout、同 H 做 N40/ΔR≤0.25 复核。N40需升H时先补同H N32与重叠；Rout稳定性仍注明原来的H。',
      '', '## 时间与内存估计','',
      '每个 Ro 以下估计随线性区间数缩放；基准 H4/N32 5–15 min，N40 ≥13 min。高H、重试及 Krylov 可能显著增加时间。',
      '| Ro | N32 H4 单点 min | N40 H4 单点 min | N32 H6 GB | N40 H6 GB | 总时长 h |','|---:|---|---|---|---|---|']
    for r in plan['rows']:
        e=r['estimates'];ran=lambda k:'–'.join(f'{x:.1f}' for x in e[k])
        lines.append(f"| {r['ro']} | {ran('N32_H4_point_minutes')} | {ran('N40_H4_point_minutes')} | {ran('N32_H6_peak_GB')} | {ran('N40_H6_peak_GB')} | {ran('campaign_hours')} |")
    lines += ['', '内存预算按RSS记账，与AS虚拟地址空间上限分开。并行driver soft AS=32GB/hard AS=200GB；所有worker AS=200GB。RSS规划：旧-0.9预留100GB、四driver共16GB、worker令牌256GB、控制余量8GB，合计380GB；S1c550GB合计930GB。四Ro的H6内存列依次为RSS估计与含余量的RSS令牌额度，不是AS上限。',
      '默认串行持有全S2及每Ro锁；S2_PARALLEL=1只持有每Ro锁，按RSS令牌阻塞准入。run_parallel.sh后台启动四Ro并wait全部。退出0=完成、75=预算可续、2=需审查。',
      '', '并行增量规则与内存上限见 PARALLEL.md / parallel_plan.json；S2_PARALLEL=1只允许-0.8..-0.5，使用每Ro锁和阻塞内存令牌，旧-0.9保持原限制。',
      '', '启动：`bash results/ro_scan_20261101/S2/commands.sh -0.9`；其余 Ro 替换末参数。默认每次24h、单点1h，可用 S2_WALL_BUDGET/S2_POINT_BUDGET 调整。',
      '短试：同命令加 `--trial`，默认总预算3300s，只做两个弱种子及两步延拓。短试完成写 trial_passed，不冒称已求折点/上支。',
      'Ro=-1 采用既有 Rout=725/H5 的同一保存点填写前沿、频率和 A₁，故 Fig.8=587.576；交接的约589来自更短出口的历史点。尚未建立 Ro=-1 域收敛。Ro=-0.751 采用末个 N32/H6 点，折点≈754.66为历史采样最小值。Re_ca 若既有数据未提供则留空，不以 DNS 值代填。',
      '回归数据：results/fullbvp_global_20261030、results/ro0751_global_20261101/phaseC_rout/REPORT.md 和 SUMMARY_20261101.md；回归不重算。',
      '数值检查和短试实测见 short_checks/verification.json、ro_m0p900/campaign.json、command_status.json。']
    report_tmp=OUT/f'.REPORT.{os.getpid()}.tmp'
    report_tmp.write_text('\n'.join(lines)+'\n');os.replace(report_tmp,OUT/'REPORT.md')

def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--prepare',action='store_true');p.add_argument('--report',action='store_true')
    p.add_argument('--ro',type=float);p.add_argument('--trial',action='store_true');p.add_argument('--trial-steps',type=int,default=2)
    p.add_argument('--point-budget',type=float,default=3600);p.add_argument('--wall-budget',type=float)
    a=p.parse_args()
    if a.prepare:prepare();return 0
    if a.report:report();return 0
    if a.ro is None:p.error('--ro is required')
    if a.ro in (-1.,-.751):p.error('Regression points are read-only; no recomputation')
    c=None;code=2;reason=None
    try:
        c=Campaign(a.ro,a.point_budget,a.wall_budget or (3300 if a.trial else 86400))
        c.run(a.trial,a.trial_steps);code=0
    except Pause as exc:
        code=75;reason=str(exc)
        if c:c.state.update(status='pending_budget',stop=reason);c.save()
    except Exception as exc:
        reason=f'{type(exc).__name__}: {exc}'
        if c:c.state.update(status='needs_supervisor',stop=reason);c.save()
        traceback.print_exc()
    finally:
        if c:
            write(c.out/'command_status.json',dict(exit_code=code,reason=reason,status=c.state['status'],stage=c.state['stage']))
            pending=c.out/'PENDING_SUPERVISOR'
            if c.state['stage']!='done':pending.write_text(f"status={c.state['status']}; stage={c.state['stage']}; reason={reason}\nResume: bash results/ro_scan_20261101/S2/commands.sh {a.ro}\n")
            elif pending.exists():pending.unlink()
            report()
        print(json.dumps(dict(exit_code=code,reason=reason,ro=a.ro),ensure_ascii=False))
    return code
if __name__=='__main__':raise SystemExit(main())
