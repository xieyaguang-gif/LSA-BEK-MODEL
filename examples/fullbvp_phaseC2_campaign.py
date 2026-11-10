#!/usr/bin/env python3
"""Prepared H6-baseline C2 campaign with nonrecursive higher-H recovery paths."""
from __future__ import annotations
import os
for _key in ('OPENBLAS_NUM_THREADS','OMP_NUM_THREADS','MKL_NUM_THREADS','NUMEXPR_NUM_THREADS','VECLIB_MAXIMUM_THREADS','BLIS_NUM_THREADS'):
    os.environ[_key]='1'
import argparse
import fcntl
import signal
import time
import traceback
from pathlib import Path
import numpy as np
import fullbvp_phaseC_campaign as prior
from fullbvp_phaseC_solver import (ROOT,Mapping,CoherentMetric,read,write,sha,fingerprint,
    mapped_transfer,phase_align,screen)
from fullbvp_phaseC_postprocess import diagnostic,finalize as phaseC_finalize
from fullbvp_phaseC2_evidence import validate_field,PHASE_C

DEFAULT_OUT=ROOT/'results/fullbvp_global_20261030/phaseC2'
SOURCE_ID='harmonic_audit_be43e3a56f551a7f_try0'
TARGETS=[730.,740.,750.,765.,780.,800.]
MAX_INCREMENT=2.5
ScientificStop=prior.ScientificStop
CampaignInterrupted=prior.CampaignInterrupted


def source_and_history():
    source=validate_field(PHASE_C/'attempts'/SOURCE_ID)
    for key,expected,tolerance in [('Rout',725.,1e-10),('omega',52.68143270236591,1e-10),
        ('A1',.2798737265153824,1e-9),('highest_to_fundamental',.00022563991710646954,1e-10)]:
        if abs(source[key]-expected)>tolerance:raise ValueError('725 H6 source scalar mismatch: '+key)
    if source['H']!=6:raise ValueError('C2 source must be a genuinely solved H6 field')
    history=[]
    for folder in sorted((PHASE_C/'attempts').iterdir()):
        if not (folder/'summary.json').exists() or not (folder/'screen.json').exists():continue
        row=read(folder/'summary.json')
        if not row.get('accepted') or row['Rout']>725.+1e-8:continue
        try: point=validate_field(folder)
        except (ValueError,KeyError,FileNotFoundError):continue
        point.update(role='imported_prior_phaseC_evidence',new_C2_target=False,source_id=SOURCE_ID)
        history.append(point)
    earlier=[p for p in history if p['H']==6 and abs(p['Rout']-720.)<1e-8]
    if len(earlier)!=1:raise ValueError('Expected one strict screened 720 H6 history point')
    source.update(role='imported_C2_source',new_C2_target=False,source_id=SOURCE_ID)
    return dict(id=SOURCE_ID,fields=source['fields'],fields_sha256=source['fields_sha256'],
        raw_record=source,audited_record=source,selection='independently strict 725 H6 audit field',
        import_requires_new_Newton=False),history


def configuration(targets=TARGETS,budget=4800.,minimum_step=.05,maximum_step=MAX_INCREMENT):
    if maximum_step>MAX_INCREMENT:raise ValueError('Every C2 increment must be <=2.5')
    if not targets or targets[0]<=725.:raise ValueError('C2 targets must be above the 725 H6 source')
    config=prior.configuration(list(targets),budget,minimum_step,maximum_step)
    config.update(campaign_family='phaseC2_H6_higher_harmonic_recovery',baseline_H=6,
        recovery_maximum_step=MAX_INCREMENT,source_id=SOURCE_ID,
        recovery_adaptation='fixed Hnext until requested Rout is reached; no recursive adaptation')
    config['code'].update({str(p.relative_to(ROOT)):sha(p) for p in sorted((ROOT/'examples').glob('fullbvp_phaseC2_*.py'))})
    return config


class Exclusive:
    def __init__(self,out):self.out=Path(out);self.stream=None
    def __enter__(self):
        self.out.mkdir(parents=True,exist_ok=True);self.stream=(self.out/'C2.lock').open('a')
        try:fcntl.flock(self.stream,fcntl.LOCK_EX|fcntl.LOCK_NB)
        except BlockingIOError:
            self.stream.close();raise RuntimeError('Another C2 process holds the exclusive campaign lock')
        return self
    def __exit__(self,*_):fcntl.flock(self.stream,fcntl.LOCK_UN);self.stream.close()


def marker(out,success,payload):
    out=Path(out);(out/('C2_FAILED.json' if success else 'C2_DONE.json')).unlink(missing_ok=True)
    write(out/('C2_DONE.json' if success else 'C2_FAILED.json'),payload)


class Campaign(prior.Campaign):
    def __init__(self,out,config,source,history):
        config=dict(config,imported_history_fingerprint=fingerprint([dict(id=p['id'],H=p['H'],Rout=p['Rout'],fields_sha256=p['fields_sha256']) for p in history]))
        super().__init__(out,config,source)
        self.state.setdefault('imported_history',history)
        self.state.setdefault('evidence',[])
        self.state.setdefault('recoveries',{})
        self.state.setdefault('refinement_outcomes',{})
        self.validate_resume_evidence()
        self.save()

    def validate_resume_evidence(self):
        for p in self.state.get('imported_history',[])+self.state.get('evidence',[]):
            if sha(p['fields'])!=p['fields_sha256']:raise ValueError('Immutable imported/candidate field hash mismatch')
        for path in self.state.get('recoveries',{}).values():
            for p in ([path['seed']] if path['seed'] is not None else [])+path['points']:
                if sha(p['fields'])!=p['fields_sha256']:raise ValueError('Immutable recovery checkpoint hash mismatch')
                if p['H']!=path['Hnext'] or p['highest_to_fundamental']>.01:
                    raise ValueError('Recovery ledger contains unresolved or wrong-H checkpoint')

    def candidate(self,mapping,y,previous,mode,target,role,tangent=None,refinement=False):
        yy,record,folder=self.attempt(mapping,y,mode,target,role,tangent)
        if not record['accepted']:return None
        yy,gate=screen(previous,yy,mapping,self.source['raw_record']['A1'],
            tangent=tangent,refinement=refinement)
        actual_dR=float(yy[-1]-previous[-1])
        if not refinement and actual_dR>MAX_INCREMENT+1e-7:
            gate['accepted']=False;gate['reasons'].append('actual_Rout_increment_exceeds_2.5')
        write(folder/'screen.json',gate)
        if not gate['accepted']:
            self.state['attempts'].append(dict(id=folder.name+'_C2_screen',role=role,
                numerical_accepted=True,branch_accepted=False,reasons=gate['reasons']))
            self.save();return None
        point={key:record[key] for key in ('Rout','omega','A1','energies','highest_to_fundamental',
            'relative_field_inf','residual_over_fundamental_feedback','n','H','intervals','phase','branch_constraint')}
        point.update(id=folder.name,role=role,accepted=True,resolved=record['highest_to_fundamental']<=.01,
            fields=str((folder/'fields.npz').resolve()),fields_sha256=sha(folder/'fields.npz'),
            summary=str((folder/'summary.json').resolve()),a=record['Rout']-100,
            dr=(record['Rout']-400)/mapping.intervals,screening=gate,source_id=SOURCE_ID,
            new_C2_target=False)
        point=diagnostic(point,self.out)
        if not any(p['id']==point['id'] for p in self.state['evidence']):self.state['evidence'].append(point)
        self.save();return point

    def seed(self):
        if self.state['points']:return
        point=dict(self.source['raw_record']);point.update(source_id=SOURCE_ID,
            role='imported_C2_source',new_C2_target=False,source_is_solved_H6=True,
            harmonic_comparison_status='imported_H6_with_existing_H5_comparison')
        point=diagnostic(point,self.out);self.commit(point)
        self.state['source_import']=dict(id=point['id'],H=6,Rout=725.,new_Newton_performed=False)
        self.save()

    def pool(self):
        points=self.state['imported_history']+self.state['points']+self.state['evidence']
        for recovery in self.state['recoveries'].values():points+=recovery['points']
        return list({p['id']:p for p in points}.values())

    def eligible(self,H,before):
        points=[p for p in self.pool() if p['H']==H and p['Rout']<before-1e-7
            and p['highest_to_fundamental']<=self.config['trustworthy_ratio']]
        return sorted(points,key=lambda p:p['Rout'])

    def history(self,mapping):
        current=self.state['points'][-1]
        eligible=self.eligible(mapping.H,current['Rout']+1e-6)
        # Only genuinely solved same-H states can seed a same-H secant.
        eligible=[p for p in eligible if p['Rout']<=current['Rout']+1e-8]
        at_current=next((p for p in eligible if p['id']==current['id']),None)
        if at_current is None and current['H']==mapping.H:eligible.append(current)
        eligible.sort(key=lambda p:p['Rout'])
        chosen=[]
        for point in eligible:
            if chosen and abs(chosen[-1]['Rout']-point['Rout'])<1e-7:chosen[-1]=point
            else:chosen.append(point)
        if not chosen:raise ScientificStop('No solved same-H continuation history')
        states=[mapped_transfer(p['fields'],mapping) for p in chosen[-2:]]
        if len(states)==2:states[0],_=phase_align(states[0],states[1],mapping)
        return states

    def refinement(self,base_point,role):
        H=base_point['H'];target=base_point['Rout'];outcomes=self.state['refinement_outcomes']
        key=fingerprint(dict(base=base_point['fields_sha256'],H=H,target=target))
        if key in outcomes and outcomes[key].get('point'):
            return next(p for p in self.pool() if p['id']==outcomes[key]['point'])
        if H>=self.config['maximum_H']:
            row=dict(source=base_point['id'],H=H,Hnext=None,Rout=target,status='H8_limit_no_H9',delta_omega=None,delta_front=None)
            self.state['refinements'].append(row);self.save();return None
        mapping,previous=self.load(base_point,H+1)
        direct=None
        if key in self.state['recoveries'] and self.state['recoveries'][key]['seed'] is not None:
            # Resume the accepted higher-H path before any costly failed lift retry.
            refined=self.recover_higher(base_point,H+1,key)
        else:
            direct=self.candidate(mapping,previous.copy(),previous,'Rout',target,role,refinement=True)
            refined=direct
            if refined is None:refined=self.recover_higher(base_point,H+1,key)
        row=dict(source=base_point['id'],H=H,Hnext=H+1,Rout=target,
            status='failed_or_branch_rejected' if refined is None else ('strict_direct' if direct else 'strict_recovered'),
            point=None if refined is None else refined['id'],
            delta_omega=None if refined is None else refined['omega']-base_point['omega'],
            delta_front=None if refined is None or refined['front_fig8'] is None or base_point['front_fig8'] is None else refined['front_fig8']-base_point['front_fig8'],
            next_ratio=None if refined is None else refined['highest_to_fundamental'])
        self.state['refinements']=[r for r in self.state['refinements'] if not (r['source']==base_point['id'] and r['Hnext']==H+1)]+[row]
        outcomes[key]=dict(status=row['status'],point=row['point']);self.save();return refined

    def recover_higher(self,base_point,Hnext,key):
        """Fixed-Hnext continuation, never calls adapt() or refinement()."""
        target=base_point['Rout'];recoveries=self.state['recoveries']
        if key not in recoveries:
            eligible=self.eligible(Hnext,target)
            if not eligible:
                recoveries[key]=dict(status='unavailable_no_prior_higher_H',Hnext=Hnext,target=target,
                    seed=None,points=[],events=[],step=min(MAX_INCREMENT,self.config['maximum_step']))
                self.save();return None
            seed=eligible[-1]
            recoveries[key]=dict(status='running',Hnext=Hnext,target=target,seed=seed,points=[],events=[],
                step=min(MAX_INCREMENT,self.config['maximum_step'],target-seed['Rout']),
                source_lower_H_target=base_point['id'])
            self.save()
        path=recoveries[key]
        if path['seed'] is None:return None
        mapping=Mapping(self.config['n'],Hnext,self.config['intervals'])
        fallback_used=path.get('fallback_used',False)
        while (path['points'][-1] if path['points'] else path['seed'])['Rout']<target-1e-7:
            old_point=path['points'][-1] if path['points'] else path['seed']
            old=mapped_transfer(old_point['fields'],mapping)
            earlier=self.eligible(Hnext,old_point['Rout'])
            tangent=None;predicted=old.copy();trial_R=min(target,old[-1]+path['step'])
            if earlier:
                before=mapped_transfer(earlier[-1]['fields'],mapping);before,_=phase_align(before,old,mapping)
                delta=old-before
                if delta[-1]>1e-7:
                    tangent=CoherentMetric(mapping).tangent(before,old)
                    predicted=old+(trial_R-old[-1])/delta[-1]*delta
            predicted[-1]=trial_R
            point=self.candidate(mapping,predicted,old,'Rout',trial_R,'recovery_Rout',tangent)
            if point is not None:
                if (point['Rout']-old_point['Rout']>MAX_INCREMENT+1e-7
                    or abs(point['Rout']-trial_R)>1e-7 or point['highest_to_fundamental']>.01):
                    path['events'].append(dict(status='rejected_recovery_checkpoint',point=point['id'],reason='step/target/truncation'))
                    point=None
            if point is not None:
                path['points'].append(point);path['step']=min(MAX_INCREMENT,self.config['maximum_step'],1.5*path['step'])
                path['events'].append(dict(status='accepted',point=point['id'],Rout=point['Rout']))
                path['status']='running';fallback_used=False;path['fallback_used']=False;self.save();continue
            path['events'].append(dict(status='natural_failed',requested_Rout=trial_R,step=path['step']))
            path['step']*=.5;self.save()
            if path['step']>=self.config['minimum_step']:continue
            arc=None
            if tangent is not None and tangent[-1]>1e-8 and not fallback_used:
                metric=CoherentMetric(mapping);ds=min(.05,.5*(target-old[-1])/tangent[-1],MAX_INCREMENT/tangent[-1])
                for _ in range(self.config['max_local_arc_steps']):
                    trial=old+ds*tangent
                    arc=self.candidate(mapping,trial,old,'arc',0.,'recovery_arc',tangent)
                    if arc is not None and (arc['Rout']>=target-1e-7 or arc['Rout']-old[-1]>MAX_INCREMENT+1e-7 or arc['highest_to_fundamental']>.01):arc=None
                    if arc is not None:break
                    ds*=.5
            else:path['events'].append(dict(status='arc_unavailable',reason='no justified forward same-H tangent or fallback already used'))
            if arc is None:
                path['status']='failed_recovery_subdivision_and_arc';self.save();return None
            path['points'].append(arc);path['step']=min(MAX_INCREMENT,max(self.config['minimum_step'],.5*(target-arc['Rout'])))
            path['events'].append(dict(status='accepted_arc',point=arc['id'],Rout=arc['Rout']))
            fallback_used=True;path['fallback_used']=True;self.save()
        final=path['points'][-1] if path['points'] else path['seed']
        if abs(final['Rout']-target)>1e-7 or final['H']!=Hnext or final['highest_to_fundamental']>.01:
            path['status']='failed_same_Rout_Hnext_trust_guard';self.save();return None
        lower=mapped_transfer(base_point['fields'],mapping);higher=mapped_transfer(final['fields'],mapping)
        _,gate=screen(lower,higher,mapping,self.source['raw_record']['A1'],refinement=True)
        path['cross_H_final_gate']=gate
        if not gate['accepted']:
            path['status']='failed_cross_H_same_Rout_screen';self.save();return None
        path.update(status='reached_strict_screened_Hnext_target',result=final['id'])
        self.save();return final

    def local_arc(self,mapping,target):
        """Parent fallback with an additional actual <=2.5 guard in candidate()."""
        return super().local_arc(mapping,target)

    def run(self):
        self.state.update(status='running',numerical_status='running',diagnostic_status='pending');self.save();self.seed()
        for target in self.config['targets']:
            target_key=str(target)
            if self.state['targets'][target_key]['status']=='accepted':continue
            step=min(self.config['maximum_step'],target-self.state['points'][-1]['Rout']);fallback_used=False
            while self.state['points'][-1]['Rout']<target-1e-7:
                current=self.state['points'][-1];mapping,_=self.load(current)
                trial_R=min(target,current['Rout']+step);old,predicted,tangent=self.predictor(mapping,trial_R)
                point=self.candidate(mapping,predicted,old,'Rout',trial_R,'C2_Rout_step',tangent)
                if point is not None:
                    if abs(point['Rout']-trial_R)>1e-7 or point['Rout']-current['Rout']>MAX_INCREMENT+1e-7:
                        raise ScientificStop('Natural point violated target or <=2.5 increment')
                    point=self.adapt(point);point['new_C2_target']=abs(point['Rout']-target)<1e-7
                    self.commit(point);step=min(self.config['maximum_step'],1.5*step);fallback_used=False;continue
                step*=.5
                if step>=self.config['minimum_step']:continue
                if not fallback_used and self.local_arc(mapping,target):
                    step=min(self.config['maximum_step'],max(self.config['minimum_step'],.5*(target-self.state['points'][-1]['Rout'])))
                    fallback_used=True;continue
                self.state['targets'][target_key].update(status='failed',last_accepted_Rout=self.state['points'][-1]['Rout']);self.save()
                raise ScientificStop(f'C2 Rout={target:g} unfinished after subdivision and arc')
            point=self.state['points'][-1]
            if abs(point['Rout']-target)>1e-7:raise ScientificStop('Off-target point cannot count as C2 target')
            self.state['targets'][target_key].update(status='accepted',point=point['id']);self.save()
        self.state.update(numerical_status='all_targets_accepted',status='postprocessing',diagnostic_status='running');self.save()
        finalize(self.out);self.state.update(status='complete',diagnostic_status='saved');self.save()


def finalize(out):
    summary=phaseC_finalize(out);path=Path(out)/'REPORT.md'
    text=path.read_text().splitlines();text[0]='# Phase C2 H6-baseline continuation evidence'
    text.insert(2,'Imported source/history are separate from newly achieved C2 targets. Higher-H recovery paths and failed direct promotions are preserved in campaign.json.')
    path.write_text('\n'.join(text)+'\n');return summary


def execute(out,config,source=None,history=None):
    out=Path(out).resolve()
    with Exclusive(out):
        campaign=None
        for name in ('C2_DONE.json','C2_FAILED.json'):
            path=out/name
            if path.exists():write(out/'marker_history'/f'{time.time_ns()}_{name}',read(path));path.unlink()
        try:
            if source is None:source,history=source_and_history()
            campaign=Campaign(out,config,source,history or [])
            if campaign.state.get('status')=='complete':
                campaign.state.update(status='postprocessing',diagnostic_status='running');campaign.save()
                finalize(out);campaign.state.update(status='complete',diagnostic_status='saved');campaign.save()
            else:campaign.run()
            done=out/'POSTPROCESS_DONE.json'
            if (campaign.state.get('numerical_status')!='all_targets_accepted'
                or any(t['status']!='accepted' for t in campaign.state['targets'].values())
                or not done.exists() or read(done).get('fingerprint')!=campaign.identity):
                raise ScientificStop('C2 completion guard: unfinished target/harmonics/diagnostics')
            marker(out,True,dict(status='all_requested_targets_and_diagnostics_saved',fingerprint=campaign.identity,
                target_count=len(config['targets']),domain_convergence='not_asserted',source_id=SOURCE_ID))
        except BaseException as exc:
            if campaign is not None:
                campaign.state.update(status='failed',failure=f'{type(exc).__name__}: {exc}')
                if campaign.state.get('numerical_status')=='all_targets_accepted':campaign.state['diagnostic_status']='failed'
                campaign.save()
            marker(out,False,dict(status='failed',failure=f'{type(exc).__name__}: {exc}',traceback=traceback.format_exc(),
                numerical_status=None if campaign is None else campaign.state.get('numerical_status'),
                diagnostic_status=None if campaign is None else campaign.state.get('diagnostic_status')))
            raise


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--out',type=Path,default=DEFAULT_OUT)
    parser.add_argument('--targets',type=float,nargs='+',default=TARGETS);parser.add_argument('--budget',type=float,default=4800.)
    parser.add_argument('--minimum-step',type=float,default=.05);parser.add_argument('--maximum-step',type=float,default=2.5)
    args=parser.parse_args()
    def interrupted(signum,_):raise CampaignInterrupted(f'C2 interrupted by signal {signum}; recovery checkpoint resume available')
    signal.signal(signal.SIGTERM,interrupted);signal.signal(signal.SIGINT,interrupted)
    execute(args.out,configuration(args.targets,args.budget,args.minimum_step,args.maximum_step))
