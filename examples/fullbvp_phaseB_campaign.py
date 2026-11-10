#!/usr/bin/env python3
"""Autonomous, restartable Phase B continuation and refinement campaign."""
import os
for key in ('OPENBLAS_NUM_THREADS','OMP_NUM_THREADS','MKL_NUM_THREADS','NUMEXPR_NUM_THREADS'):os.environ[key]='1'
import argparse, fcntl, gc, json, signal, time, traceback
from pathlib import Path
import numpy as np
from fullbvp_phaseB_solver import Mapping, Metric, ROOT, checks, correct, energies, pack, unpack, transfer, write, common

def load(path):return json.loads(Path(path).read_text())

class Campaign:
    def __init__(self,out,budget=1200):
        self.out=Path(out);self.out.mkdir(parents=True,exist_ok=True);self.budget=budget;self.start=time.monotonic()
        self.lock=(self.out/'numerical.lock').open('a');fcntl.flock(self.lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        self.state=load(self.out/'campaign.json') if (self.out/'campaign.json').exists() else dict(status='running',points=[],attempts=[],stage='seeds',stop=None,targets={},refinements=[],folds={})
        self.mapping=Mapping(32,3,800);self.metric=Metric(self.mapping)
    def save(self):
        self.state['elapsed_this_run_seconds']=time.monotonic()-self.start
        write(self.out/'campaign.json',self.state);write(self.out/'accepted_points.json',dict(points=self.state['points']))
    def attempt(self,mapping,y,qref,mode,target,name,role,tangent=None,metric=None):
        folder=self.out/'attempts'/name
        d=None
        if folder.exists():
            d=load(folder/'summary.json') if (folder/'summary.json').exists() else None
            if d:
                yy=transfer(folder/'fields.npz',mapping) if d['accepted'] else y.copy()
            else:
                # A killed point retains progress/checkpoint and is retried under a fresh name.
                i=1
                while folder.with_name(folder.name+f'_restart{i}').exists():i+=1
                folder=folder.with_name(folder.name+f'_restart{i}')
        if d is None:yy,d=correct(mapping,qref,y.copy(),mode,target,folder,self.budget,metric,tangent)
        record=dict(name=folder.name,summary=str((folder/'summary.json').relative_to(ROOT)),role=role,accepted=d['accepted'],failure=d.get('failure'),seconds=d['seconds'],Rout=d['Rout'],omega=d['omega'])
        self.state['attempts']=[p for p in self.state['attempts'] if p['name']!=record['name']]+[record]
        if d['accepted']:
            point={k:d[k] for k in ['Rout','omega','A1','energies','highest_to_fundamental','residual_inf','relative_field_inf','residual_over_fundamental_feedback','phase','branch_constraint','n','H','intervals']}
            point.update(id=folder.name,role=role,fields=str((folder/'fields.npz').relative_to(ROOT)),summary=record['summary'],a=d['Rout']-100,dr=(d['Rout']-400)/mapping.intervals,accepted=True,resolved=d['highest_to_fundamental']<=.01)
            if tangent is not None:point.update(predictor_tangent_Rout=float(tangent[-1]),predictor_tangent_omega=float(tangent[-2]))
            if point['resolved']:
                if not any(p['id']==point['id'] for p in self.state['points']):self.state['points'].append(point)
            else:
                record.update(accepted=False,converged_but_unresolved=True,highest_to_fundamental=point['highest_to_fundamental'])
                d['under_resolved']=True;d['accepted_branch_point']=False;write(folder/'summary.json',d)
        self.save();gc.collect();return yy,d,folder
    def primary(self):return [p for p in self.state['points'] if p['role'] in ['weak_seed','primary_arclength']]
    def getqref(self,y,mapping=None):
        mapping=mapping or self.mapping;p,*_=mapping.at(y[-1],y[-2]);q=unpack(y[:-2],mapping.nr,mapping.H,mapping.s)[:,mapping.H+1];return q/common.mass_norm(p,q)
    def seeds(self):
        base=ROOT/'results/appelquist2016_branch_20261029/continuation'
        for k,(name,A) in enumerate([('point_001_H2_A0.0064000',.0064),('point_002_H2_A0.0090000',.009)]):
            ident=f'seed_{k}'
            if any(p['id']==ident for p in self.primary()):continue
            src=base/name/'fields.npz';raw=np.load(src);R=float(raw['a'])+100
            y=transfer(src,self.mapping,R);qref=self.getqref(y)
            y,d,_=self.attempt(self.mapping,y,qref,'energy',A,ident,'weak_seed')
            if not d['accepted']:self.state.update(stop='weak_seed_failure',stage='postprocess');self.save();return
        self.state['stage']='branch';self.save()
    def branch(self):
        points=self.primary()
        if len(points)<2:return
        old=transfer(ROOT/points[-2]['fields'],self.mapping);y=transfer(ROOT/points[-1]['fields'],self.mapping)
        seed=transfer(ROOT/points[0]['fields'],self.mapping);qref=self.getqref(seed)
        tangent=self.metric.tangent(old,y);ds=self.state.get('next_step',.04)
        previous_signs=np.diff([p['Rout'] for p in points]);upper=bool(np.any(previous_signs<0) and previous_signs[-1]>0)
        while self.state['stop'] is None:
            idx=len(self.primary());successful=False
            for retry in range(4):
                pred=y+ds*tangent
                trial,d,folder=self.attempt(self.mapping,pred,qref,'arc',0.,f'branch_{idx:04d}_try{retry}','primary_arclength',tangent,self.metric)
                if d['accepted'] and d.get('under_resolved'):
                    self.state['stop']='highest_harmonic_ratio_above_1e-2';self.state['harmonic_stop']=dict(attempt=folder.name,ratio=d['highest_to_fundamental'],last_resolved_A1=self.primary()[-1]['A1']);break
                if d['accepted']:
                    newt=self.metric.tangent(y,trial,tangent);deltaR=trial[-1]-y[-1]
                    if tangent[-1]<0 and newt[-1]>0:upper=True
                    point=self.primary()[-1];point['tangent_Rout']=float(newt[-1]);point['tangent_omega']=float(newt[-2]);point['upper_branch']=upper
                    self.state['points']=[point if p['id']==point['id'] else p for p in self.state['points']]
                    np.savez_compressed(self.out/'continuation_state.npz',previous=y,current=trial,tangent=newt,step=ds,qref=qref)
                    old,y,tangent=y,trial,newt;successful=True
                    niter=len(d['iterations'])-1;ds=min(.22,ds*(1.25 if niter<=3 else 1.05))
                    self.state['next_step']=ds
                    if upper and y[-1]>=800:self.state['stop']='Rout_endpoint_reached'
                    self.save();break
                if d.get('budget_stop'):
                    self.state['budget_attempts']=self.state.get('budget_attempts',0)+1
                    self.state['stop']='per_point_budget_exceeded';break
                ds*=.5
            if self.state['stop'] is not None:break
            if not successful:self.state['stop']='four_repeated_Newton_failures';break
        self.state['stage']='folds';self.save()
    def fold_fit(self,points):
        pp=sorted(points,key=lambda p:p['A1']);A=np.array([p['A1'] for p in pp]);R=np.array([p['Rout'] for p in pp])
        if len(pp)<3:return dict(status='insufficient_points',Rout=None,A1=None)
        i=int(np.argmin(R));ids=np.argsort(abs(A-A[i]))[:min(5,len(A))];ids.sort();x=A[ids];z=R[ids];coef=np.polyfit(x-x.mean(),z,2);aa=x.mean()-coef[1]/(2*coef[0]);rr=np.polyval(coef,aa-x.mean())
        bracket=bool(coef[0]>0 and x[0]<aa<x[-1] and np.any(np.diff(z)<0) and np.any(np.diff(z)>0))
        three=np.argsort(abs(A-A[i]))[:3];three.sort();tx=A[three];tz=R[three];tc=np.polyfit(tx-tx.mean(),tz,2);ta=tx.mean()-tc[1]/(2*tc[0]);tr=np.polyval(tc,ta-tx.mean())
        return dict(status='bracketed_quadratic_estimate' if bracket else 'unbracketed',Rout=float(rr) if bracket else None,A1=float(aa) if bracket else None,fit_Rout=float(rr),fit_A1=float(aa),fit_rms=float(np.sqrt(np.mean((np.polyval(coef,x-x.mean())-z)**2))),samples=[p['id'] for p in pp],fitted_samples=[pp[k]['id'] for k in ids],A1_range=[float(x[0]),float(x[-1])],quadratic_coefficients=coef.tolist(),dRout_dA1_sign_change=bracket,three_point_A1=float(ta),three_point_Rout=float(tr),three_vs_five_Rout=float(tr-rr),three_vs_five_A1=float(ta-aa))
    def folds(self):
        points=self.primary();coarse=self.fold_fit(points);self.state['folds']['H3_coarse']=coarse
        if coarse['status']!='bracketed_quadratic_estimate':self.state['stage']='targets';self.save();return
        af=coarse['A1'];spacing=.004
        aa=[af+k*spacing for k in [-2,-1,0,1,2]]
        for H in [3,4]:
            mapping=self.mapping if H==3 else Mapping(32,4,800)
            for A in aa:
                candidates=[p for p in self.state['points'] if p['H']==H and p['n']==32 and p['intervals']==800]
                source=min(candidates or points,key=lambda p:abs(p['A1']-A));y=transfer(ROOT/source['fields'],mapping);qref=self.getqref(y,mapping)
                self.attempt(mapping,y,qref,'energy',A,f'fold_H{H}_A{A:.7f}','fold_refinement')
            subset=[p for p in self.state['points'] if p['role']=='fold_refinement' and p['H']==H]
            fit=self.fold_fit(subset)
            # H4 fold may move outside initial H3 bracket: extend in its indicated direction.
            for k in range(4):
                if fit['status']=='bracketed_quadratic_estimate':break
                if len(subset)<3:break
                A=(min(p['A1'] for p in subset)-spacing if fit['fit_A1']<min(p['A1'] for p in subset) else max(p['A1'] for p in subset)+spacing)
                source=min(subset,key=lambda p:abs(p['A1']-A));y=transfer(ROOT/source['fields'],mapping)
                self.attempt(mapping,y,self.getqref(y,mapping),'energy',A,f'fold_H{H}_extend{k}','fold_refinement')
                subset=[p for p in self.state['points'] if p['role']=='fold_refinement' and p['H']==H];fit=self.fold_fit(subset)
            self.state['folds'][f'H{H}']=fit;self.save()
            if H==4:del mapping;gc.collect()
        if all(self.state['folds'].get(f'H{H}',{}).get('Rout') is not None for H in [3,4]):self.state['folds']['H4_minus_H3_Rout']=self.state['folds']['H4']['Rout']-self.state['folds']['H3']['Rout']
        self.state['stage']='targets';self.save()
    def targets(self):
        upper=[p for p in self.primary() if p.get('upper_branch')]
        for R in [730.,750.,780.,800.]:
            if str(R) in self.state['targets']:continue
            if not upper or max(p['Rout'] for p in upper)<R:
                self.state['targets'][str(R)]=dict(status='unavailable',Rout=R,reason=self.state['stop']);continue
            source=min(upper,key=lambda p:abs(p['Rout']-R));y=transfer(ROOT/source['fields'],self.mapping);qref=self.getqref(y)
            yy,d,folder=self.attempt(self.mapping,y,qref,'Rout',R,f'target_R{R:g}','upper_target')
            self.state['targets'][str(R)]=dict(status='accepted' if d['accepted'] and not d.get('under_resolved') else 'failed',attempt=folder.name,Rout=R)
        self.state['stage']='mesh';self.save()
    def mesh(self):
        upper=[p for p in self.primary() if p.get('upper_branch')]
        if not upper:self.state['refinement_status']='no_reachable_upper_branch';self.state['stage']='postprocess';self.save();return
        comfortable=[p for p in upper if p['highest_to_fundamental']<=.006]
        source=(comfortable or upper)[-1];A=source['A1'];self.state['refinement_source']=source['id'];self.state['refinement_source_selection']='last upper point with E3/E1<=.006, fallback last reachable upper'
        tests=[('N40',40,3,800),('dr025',32,3,round((source['Rout']-400)/.25)),('H4',32,4,800)]
        for label,n,H,M in tests:
            mapping=Mapping(n,H,M);y=transfer(ROOT/source['fields'],mapping);qref=self.getqref(y,mapping)
            yy,d,folder=self.attempt(mapping,y,qref,'energy',A,f'refinement_{label}','mesh_refinement')
            self.state['refinements'].append(dict(label=label,status='accepted' if d['accepted'] and not d.get('under_resolved') else 'failed_or_unresolved',point=folder.name,source=source['id'],matching='fixed physical E1=integral |q1|^2 dz dR',A1=A,delta_Rout=float(yy[-1]-source['Rout']) if d['accepted'] else None,delta_omega=float(yy[-2]-source['omega']) if d['accepted'] else None,highest_to_fundamental=d.get('highest_to_fundamental')))
            self.save();del mapping;gc.collect()
        self.state['stage']='postprocess';self.save()
    def run(self):
        write(self.out/'run_policy.json',dict(per_point_budget_seconds=self.budget,budget_basis='Old H3/H4 correctors took 35–56 s, excluding build. Allow 1200 s for moving-grid and finer meshes.',CPU_threads=1,GPU=False,primary_intervals=800,dr_at_800=.5,no_total_point_cap=True,repeated_failure_limit=4,highest_harmonic_ratio_stop=.01,target_Rout=800,auxiliary_forcing=False,mean_damped=False,geometry='R=400+(Rout-400)xi',sponge='0.8 S((420-R)/10)+28 S((R-(Rout-100))/40) on nonzero harmonics',old_resolution_neutral=719.9581594895074,new_grid_seed_corrected=True))
        checks(self.out)
        try:
            for stage in ['seeds','branch','folds','targets','mesh']:
                if self.state['stage']==stage:getattr(self,stage)()
            self.state['status']='numerics_finished';self.save()
            from fullbvp_phaseB_report import finalize
            finalize(self.out)
            self.state.update(status='completed_with_documented_stop',stage='done');self.save()
            write(self.out/'DONE.json',dict(status=self.state['status'],stop=self.state['stop'],accepted_points=len(self.state['points']),visual_audit='pending human inspection'))
        except BaseException as exc:
            self.state.update(status='failed_exception',exception=f'{type(exc).__name__}: {exc}',traceback=traceback.format_exc());self.save()
            try:
                from fullbvp_phaseB_report import finalize
                finalize(self.out)
            except Exception:self.state['postprocess_exception']=traceback.format_exc();self.save()
            write(self.out/'FAILED.json',dict(exception=str(exc),stage=self.state['stage'],traceback=traceback.format_exc()));raise

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--out',default='results/fullbvp_global_20261030/phaseB');p.add_argument('--budget',type=float,default=1200);a=p.parse_args();Campaign(a.out,a.budget).run()
