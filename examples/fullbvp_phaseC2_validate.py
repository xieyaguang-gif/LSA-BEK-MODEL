#!/usr/bin/env python3
"""Bounded C2 algebra/source/control-flow checks; no full-resolution Newton."""
from __future__ import annotations
import os
for _key in ('OPENBLAS_NUM_THREADS','OMP_NUM_THREADS','MKL_NUM_THREADS','NUMEXPR_NUM_THREADS','VECLIB_MAXIMUM_THREADS','BLIS_NUM_THREADS'):
    os.environ[_key]='1'
import argparse
import copy
import signal
import tempfile
import time
from pathlib import Path
from types import MethodType
import numpy as np
import fullbvp_phaseC2_campaign as driver
import fullbvp_phaseC2_evidence as evidence
from fullbvp_phaseC_solver import (Mapping,CoherentMetric,checkpoint,mapped_transfer,
    pack,qref_for,read,write,sha,energies,unpack)
from fullbvp_phaseC_validate import algebra,harmonic_policy,require


def make_point(root,H,R,name,amplitude=1.,ratio=.0002):
    """Synthetic fields for mocked control flow only; no solved-root claim."""
    folder=Path(root)/name;folder.mkdir(parents=True,exist_ok=True);mapping=Mapping(8,H,8)
    q=np.zeros((mapping.nr,2*H+1,mapping.s),complex)
    rng=np.random.default_rng(20261030)
    one=(rng.normal(size=(mapping.nr,mapping.s))+1j*rng.normal(size=(mapping.nr,mapping.s)))*1e-4*amplitude
    q[:,H+1]=one;q[:,H-1]=one.conj();y=np.r_[pack(q),52.7-.001*(R-725),R]
    checkpoint(folder/'fields.npz',mapping,y,qref_for(y,mapping),accepted=True)
    p,*_=mapping.at(R,y[-2]);e=energies(p,q)
    record=dict(accepted=True,Rout=R,omega=float(y[-2]),A1=float(np.sqrt(e[1])),energies=e.tolist(),
        highest_to_fundamental=ratio,relative_field_inf=1e-14,residual_over_fundamental_feedback=1e-9,
        n=8,H=H,intervals=8,phase=0.,branch_constraint=0.,independent=dict(relative_field_inf=1e-14))
    write(folder/'summary.json',record);write(folder/'screen.json',dict(accepted=True))
    return dict(record,id=name,fields=str(folder/'fields.npz'),fields_sha256=sha(folder/'fields.npz'),
        summary=str(folder/'summary.json'),front_fig8=588.,a=R-100,role='mock_control_only',accepted=True)


class Harness(driver.Campaign):
    def __init__(self,out,history):
        self.out=Path(out);self.out.mkdir(parents=True,exist_ok=True)
        self.identity='mock_control_only';self.source=dict(raw_record=dict(A1=1e-6));self.budget=1.
        self.config=dict(n=8,intervals=8,maximum_H=8,baseline_H=6,promotion_ratio=1e-3,
            trustworthy_ratio=.01,minimum_step=1.25,maximum_step=2.5,max_local_arc_steps=0)
        self.state=dict(points=[],attempts=[],refinements=[],refinement_outcomes={},recoveries={},
            imported_history=history,evidence=[],targets={'730.0':dict(status='pending',Rout=730.)})
        self.calls=[];self.interrupt_once=False;self.bad_final_amplitude=False;self.wrong_R=False;self.unresolved=False

    def candidate(self,mapping,y,previous,mode,target,role,tangent=None,refinement=False):
        self.calls.append(dict(role=role,target=float(target),previous_R=float(previous[-1])))
        if refinement:return None
        # An injected interruption happens only after 727.5 was atomically saved.
        if self.interrupt_once and target>=730.:
            self.interrupt_once=False;raise driver.CampaignInterrupted('injected recovery interruption')
        R=float(target)-(.1 if self.wrong_R else 0.)
        amp=.5 if self.bad_final_amplitude and target>=730. else 1.
        point=make_point(self.out/'mock_fields',mapping.H,R,f'mock_{len(self.calls)}_H{mapping.H}_{R}',
                         amplitude=amp,ratio=.02 if self.unresolved else .0002)
        self.state['evidence'].append(point);self.save();return point


def promotion_recovery(root):
    seed=make_point(root,6,725.,'mock_source_H6_725');earlier=make_point(root,6,720.,'mock_history_H6_720')
    lower=make_point(root,5,730.,'mock_lower_H5_730')
    harness=Harness(Path(root)/'recovery',[earlier,seed]);harness.interrupt_once=True
    try:harness.refinement(lower,'harmonic_promotion')
    except driver.CampaignInterrupted:pass
    else:raise AssertionError('Recovery interruption was swallowed')
    path=next(iter(harness.state['recoveries'].values()))
    require([p['Rout'] for p in path['points']]==[727.5],'Recovery checkpoint at 727.5 was not committed before interruption')
    calls=list(harness.calls)
    resumed=Harness(Path(root)/'recovery',[earlier,seed]);resumed.state=read(resumed.out/'campaign.json')
    resumed.validate_resume_evidence()
    result=resumed.refinement(lower,'harmonic_promotion')
    require(result is not None and abs(result['Rout']-730.)<1e-8 and result['H']==6,'Recovery did not reach exact Hnext target')
    require(resumed.calls[0]['role']=='recovery_Rout' and resumed.calls[0]['previous_R']==727.5
        and resumed.calls[0]['target']==730.,'Resume repeated direct lift or restarted from 725')
    require(sum(c['role']=='harmonic_promotion' for c in calls+resumed.calls)==1,'Known failed direct lift repeated on resume')
    path=next(iter(resumed.state['recoveries'].values()))
    require(path['cross_H_final_gate']['accepted'],'Recovered target not screened against lower-H same-Rout field')
    require(all(b['Rout']-a['Rout']<=2.5+1e-8 for a,b in zip([path['seed']]+path['points'][:-1],path['points'])),
            'Recovery accepted a >2.5 increment')
    require(resumed.state['targets']['730.0']['status']=='pending','Promotion recovery falsely committed a production target')
    # Committed internal checkpoint bytes are immutable on resume.
    field=Path(path['points'][0]['fields'])
    with field.open('ab') as stream:stream.write(b'injected corruption')
    try:resumed.validate_resume_evidence()
    except ValueError as exc:require('hash mismatch' in str(exc),'Recovery corruption error unclear')
    else:raise AssertionError('Corrupted recovery checkpoint accepted')
    none=Harness(Path(root)/'no_prior',[])
    require(none.refinement(lower,'harmonic_promotion') is None,'Zero-padded lift falsely treated as solved H6 history')
    require(next(iter(none.state['recoveries'].values()))['status']=='unavailable_no_prior_higher_H','No-prior-history reason absent')
    none.validate_resume_evidence()
    # Fresh sources avoid using the intentionally corrupted path field.
    for label,attribute in [('cross_H_rejection','bad_final_amplitude'),('same_Rout_rejection','wrong_R'),('unresolved_rejection','unresolved')]:
        h=Harness(Path(root)/label,[earlier,seed]);setattr(h,attribute,True)
        require(h.refinement(lower,'harmonic_promotion') is None,label+' failed')
        require(h.state['targets']['730.0']['status']=='pending',label+' marked target success')
    return dict(mocked_control_only=True,direct_failure_to_prior_Hnext=True,nearest_prior_seed_Rout=725.,
        accepted_intermediate_Rout=727.5,resume_first_new_Rout=730.,no_repeated_direct_lift=True,
        exact_Hnext_same_Rout_and_cross_H_screen=True,no_prior_history_explicit=True,
        wrong_Rout_rejected=True,unresolved_intermediate_rejected=True,
        cross_H_amplitude_rejected=True,immutable_recovery_fields=True,no_recursive_adaptation=True)


def actual_arc_cap(root):
    old=make_point(root,6,725.,'arc_previous');over=make_point(root,6,728.,'arc_actual_overrun')
    h=Harness(Path(root)/'actual_arc_cap',[old]);mapping=Mapping(8,6,8)
    previous=mapped_transfer(old['fields'],mapping);candidate=mapped_transfer(over['fields'],mapping)
    def attempt(self,*args,**kwargs):return candidate,read(over['summary']),Path(over['summary']).parent
    h.attempt=MethodType(attempt,h)
    result=driver.Campaign.candidate(h,mapping,candidate,previous,'arc',0.,'mock_arc_overrun')
    gate=read(Path(over['summary']).parent/'screen.json')
    require(result is None and 'actual_Rout_increment_exceeds_2.5' in gate['reasons'],
            'Actual corrected arc step >2.5 was admitted')
    return dict(actual_corrected_arc_increment_cap=True)


def marker_cases(root):
    out=Path(root)/'marker';out.mkdir(parents=True)
    write(out/'C2_DONE.json',dict(sentinel=True))
    with driver.Exclusive(out):
        try:driver.execute(out,{})
        except RuntimeError as exc:require('exclusive' in str(exc),'Busy lock error unclear')
        else:raise AssertionError('Duplicate C2 lock acquired')
        require(read(out/'C2_DONE.json')['sentinel'],'Busy duplicate altered markers')
    original=driver.Campaign;fail=['diagnostics']
    class Fake:
        def __init__(self,out,config,source,history):
            self.out=Path(out);self.identity='mock_C2_identity'
            self.state=dict(status='prepared',points=[],targets={'730.0':dict(status='pending')},
                numerical_status='running',diagnostic_status='pending')
        def save(self):write(self.out/'mock_state.json',self.state)
        def run(self):
            self.state['numerical_status']='all_targets_accepted'
            if fail[0]=='diagnostics':raise OSError('injected C2 diagnostic failure')
            if fail[0]=='unfinished':return
            self.state['targets']['730.0']['status']='accepted'
            write(self.out/'POSTPROCESS_DONE.json',dict(fingerprint=self.identity))
    try:
        driver.Campaign=Fake
        try:driver.execute(out,dict(targets=[730.]),source={'mock':True})
        except OSError:pass
        row=read(out/'C2_FAILED.json')
        require(row['numerical_status']=='all_targets_accepted' and row['diagnostic_status']=='failed'
            and not (out/'C2_DONE.json').exists(),'Diagnostic failure inherited C2 success')
        fail[0]='unfinished'
        try:driver.execute(out,dict(targets=[730.]),source={'mock':True})
        except driver.ScientificStop:pass
        else:raise AssertionError('Unfinished target received C2_DONE')
        fail[0]='none';driver.execute(out,dict(targets=[730.]),source={'mock':True})
        require((out/'C2_DONE.json').exists() and not (out/'C2_FAILED.json').exists(),'Success/failure markers coexist')
        require(not (out/'C_DONE.json').exists() and not (out/'C_FAILED.json').exists(),'C2 used inherited C marker names')
    finally:driver.Campaign=original
    return dict(exclusive_busy_preserves_active_markers=True,diagnostic_failure_distinguished=True,
        unfinished_target_refused=True,opposite_stale_marker_removed=True,C2_marker_names=True)


def source_checks(source,history):
    config=driver.configuration()
    require(config['baseline_H']==6 and config['budget_per_point']==4800.
        and config['targets']==[730.,740.,750.,765.,780.,800.]
        and config['maximum_step']<=2.5,'C2 source/target/budget/step configuration wrong')
    try:driver.configuration(maximum_step=3.)
    except ValueError:pass
    else:raise AssertionError('Maximum step >2.5 allowed')
    h6=[p for p in history if p['H']==6]
    require(any(abs(p['Rout']-720.)<1e-8 for p in h6),'Imported solved 720 H6 tangent history absent')
    old_sha=evidence.sha
    def mismatched(path):return '0'*64 if Path(path).name=='fields.npz' else old_sha(path)
    try:
        evidence.sha=mismatched
        try:evidence.validate_field(Path(source['fields']).parent)
        except ValueError as exc:require('provenance' in str(exc),'Provenance hash mismatch not identified')
        else:raise AssertionError('Mislabeled field hash inherited strict residual provenance')
    finally:evidence.sha=old_sha
    return dict(source_id=source['id'],source_H=6,source_Rout=725.,source_fields_sha256=source['fields_sha256'],
        independent_relative_field_inf=source['raw_record']['independent']['relative_field_inf'],
        solved_H6_imports=[dict(id=p['id'],Rout=p['Rout'],sha256=p['fields_sha256']) for p in h6],
        metadata_and_audited_field_hash_verified=True,injected_provenance_mismatch_refused=True,
        source_Newton_performed=False,target_sequence=config['targets'],maximum_step=2.5)



def integration_import(root,source,history):
    """Real Campaign constructor/seed/resume; any Newton entry is forbidden."""
    config=driver.configuration();out=Path(root)/'real_source_import'
    original_attempt=driver.Campaign.attempt
    original_correct=driver.prior.correct
    def forbidden(*args,**kwargs):raise AssertionError('Source import attempted a nonlinear solve')
    try:
        driver.Campaign.attempt=forbidden;driver.prior.correct=forbidden
        campaign=driver.Campaign(out,config,source,history);campaign.seed()
        require(len(campaign.state['points'])==1,'Source seed did not produce exactly one imported point')
        point=campaign.state['points'][0]
        require(point['H']==6 and point['Rout']==725. and point['source_id']==driver.SOURCE_ID
            and not point['new_C2_target'],'Source import H/Rout/id/target status mismatch')
        require(all(t['status']=='pending' for t in campaign.state['targets'].values())
            and len(campaign.state['targets'])==6,'Imported source incorrectly completed a C2 target')
        restored=driver.Campaign(out,config,source,history)
        require(restored.identity==campaign.identity and len(restored.state['points'])==1,
                'Real inherited verify_commits resume failed')
        old=read(driver.PHASE_C/'campaign.json')['fingerprint']
        require(restored.identity!=old,'C2 reused original C numerical fingerprint')
        try:driver.Campaign(out,dict(config,budget_per_point=4801.),source,history)
        except ValueError as exc:require('fingerprint' in str(exc),'Configuration mismatch error unclear')
        else:raise AssertionError('Changed C2 configuration resumed old checkpoints')
        code_changed=copy.deepcopy(config);code_changed['code']['injected_code_hash']='0'*64
        try:driver.Campaign(out,code_changed,source,history)
        except ValueError as exc:require('fingerprint' in str(exc),'Code mismatch error unclear')
        else:raise AssertionError('Changed C2 code identity resumed old checkpoints')
    finally:
        driver.Campaign.attempt=original_attempt;driver.prior.correct=original_correct
    return dict(real_Campaign_source_import_and_resume=True,source_id=driver.SOURCE_ID,
        imported_points=1,H=6,Rout=725.,all_six_targets_pending=True,
        distinct_C2_fingerprint=True,config_and_code_changes_refused=True,Newton_calls=0)


def run(out):
    out=Path(out).resolve();out.mkdir(parents=True,exist_ok=True);start=time.monotonic()
    def alarm(*_):raise driver.CampaignInterrupted('C2 validation hard budget 120 seconds')
    before=signal.signal(signal.SIGALRM,alarm);signal.alarm(120)
    try:
        source,history=driver.source_and_history();checks=dict(source=source_checks(source,history))
        m=Mapping(8,2,12);rng=np.random.default_rng(20261030)
        y=np.r_[rng.normal(size=m.nr*(2*m.H+1)*m.s)*1e-5,52.71,725.]
        checks['existing_algebra']=algebra(m,y);checks['harmonic_limits']=harmonic_policy()
        with tempfile.TemporaryDirectory(prefix='phaseC2_control_',dir='/tmp') as tmp:
            checks['promotion_recovery']=promotion_recovery(Path(tmp)/'recovery_tests')
            checks['actual_arc_cap']=actual_arc_cap(Path(tmp)/'arc_tests')
            checks['markers']=marker_cases(Path(tmp)/'marker_tests')
            checks['real_import_integration']=integration_import(Path(tmp)/'integration',source,history)
        result=dict(status='passed',wall_seconds=time.monotonic()-start,hard_budget_seconds=120,
            aggregate_ceiling_seconds=1200,CPU_threads=1,GPU=False,full_campaign_launched=False,
            full_N32_Newton_performed=False,mocked_paths_are_scientific_data=False,checks=checks)
        write(out/'VALIDATION.json',result);return result
    except BaseException as exc:
        write(out/'VALIDATION_FAILED.json',dict(status='failed',wall_seconds=time.monotonic()-start,
            failure=f'{type(exc).__name__}: {exc}',full_campaign_launched=False));raise
    finally:signal.alarm(0);signal.signal(signal.SIGALRM,before)

if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--out',type=Path,default=driver.DEFAULT_OUT)
    args=parser.parse_args();r=run(args.out);print(f'C2 validation passed in {r["wall_seconds"]:.3f} s; full campaign unlaunched.')
