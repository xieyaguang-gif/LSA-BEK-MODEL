#!/usr/bin/env python3
"""Short actual-operator and negative-path checks. Never runs a campaign."""
import os
os.environ['PYTHONDONTWRITEBYTECODE']='1'
import json
import time
from pathlib import Path
import numpy as np
from scipy import linalg, sparse
from scipy.sparse.linalg import LinearOperator
import fullbvp_phaseD2k_campaign as k


def main():
    started=time.monotonic();out=k.PREP/'validation'/str(time.time_ns());out.mkdir(parents=True)
    rng=np.random.default_rng(20261031);results={}
    for name in ('baseline','map_a3','sponge_wide_weak'):
        spec=dict(k.VARIANTS[name],n=8,H=1,dr=82.)
        mapping=k.mesh(spec,725.)
        y=np.r_[rng.normal(size=mapping.nr*3*mapping.s)*1e-4,52.65,725.]
        qref=k.base.unpack(y[:-2],mapping.nr,1,mapping.s)[:,2]
        system,K,t,assembly=k.system_matrix(mapping,y,qref,'Rout',725.,k.temporal_api.Deadline(30))
        v=np.r_[rng.normal(size=len(y)-2)*1e-4,.3,.7];eps=1e-5
        numerical=(system.residual(y+eps*v)-system.residual(y-eps*v))/(2*eps)
        analytic=K@v
        err=float(np.linalg.norm(numerical-analytic)/np.linalg.norm(analytic))
        assert err<2e-6,(name,err)
        independent=k.base.independent(mapping,y)
        assert independent['template_difference_relative']<1e-10,independent
        results[name]=dict(bordered_derivative_relative=err,independent_template_relative=independent['template_difference_relative'],assembly_relative=assembly)
        if name=='baseline': baseline=(mapping,y,qref,K,t)
    mapping,y,qref,K,t=baseline
    # Real Newton operator solve and the same D2g backward-error gates.
    fac=k.factor(K,k.temporal_api.Deadline(30),out/'bordered_factor.json')
    try:
        I=LinearOperator(K.shape,matvec=lambda x:x,dtype=complex)
        inverse=k.g.Inverse(K,I,0j,fac,k.temporal_api.Deadline(30),out/'newton_inverse.jsonl',assembled_K=K)
        rhs=rng.normal(size=K.shape[0]);solution=inverse.apply(-rhs)
        metrics=inverse.measure(solution,rhs)[1]
        assert metrics['accepted'],metrics
        results['bordered_solve']=metrics
        transposed=k.transpose_solve(fac,rhs)
        transpose_error=float(np.linalg.norm(K.T@transposed-rhs)/np.linalg.norm(rhs))
        assert transpose_error<1e-7,transpose_error
        results['single_LU_transpose_original_relative']=transpose_error
    finally:fac.close()
    # QZ is independent of the Arnoldi/inverse implementation and checks its sign.
    J,assembly=k.old.d2.assemble(t,k.temporal_api.Deadline(30))
    center=k.g.run_center(t,J,.05+2j,out/'center','D2k-small-actual-QZ',k.temporal_api.Deadline(30),k=6,ncv=40)
    assert center['complete'],center.get('failure')
    exact=linalg.eigvals(J.toarray(),-np.diag(t.b));exact=exact[np.isfinite(exact)]
    distances=[float(np.min(abs(complex(*p['value'])-exact))) for p in center['eigenpairs']]
    assert max(distances)<1e-6,distances
    results['actual_spectrum_QZ_max_distance']=max(distances)
    singular=k.stationary_diagnosis(mapping,y,out/'singular',k.temporal_api.Deadline(20),stationary=False)
    assert singular[0]['accepted'],singular
    exact_singular=linalg.svdvals(K.toarray())
    singular_error=abs(singular[0]['sigma_min']/exact_singular[-1]-1)
    assert singular_error<2e-5,singular_error
    results['smallest_singular_value_dense_relative']=singular_error
    # The phase-bordered arc Jacobian must differentiate the fixed hyperplane.
    metric=k.continuation.CoherentMetric(mapping)
    tangent=rng.normal(size=len(y));tangent/=metric.length(tangent)
    system,Ka,_,_=k.system_matrix(mapping,y,qref,'arc',725.,k.temporal_api.Deadline(20),tangent)
    v=np.r_[rng.normal(size=len(y)-2)*1e-4,.2,.4];eps=1e-5
    finite=(system.residual(y+eps*v)-system.residual(y-eps*v))/(2*eps)
    arc_error=np.linalg.norm(finite-Ka@v)/np.linalg.norm(Ka@v)
    assert arc_error<2e-6,arc_error
    results['arc_bordered_derivative_relative']=float(arc_error)

    resumed=k.g.run_center(t,J,.05+2j,out/'center','D2k-small-actual-QZ',k.temporal_api.Deadline(30),k=6,ncv=40)
    assert resumed==center
    try:k.g.run_center(t,J,.05+2j,out/'center','changed',k.temporal_api.Deadline(30),k=6,ncv=40)
    except RuntimeError:pass
    else:raise AssertionError('changed fingerprint resumed')
    # Phase exclusion requires both zero eigenvalue and neutral-direction overlap.
    assert k.phase_pair(dict(value=[0,0],phase_mass_overlap=1))
    assert not k.phase_pair(dict(value=[-.0001,0],phase_mass_overlap=1))
    assert not k.phase_pair(dict(value=[0,0],phase_mass_overlap=.5))
    old=[dict(value=[-.01,.2],shape='a'),dict(value=[-.02,.3],shape='b')]
    new=[dict(value=[-.01,.21],shape='b'),dict(value=[-.02,.29],shape='a')]
    match=k.match_modes(old,new,lambda a,b:1. if a==b else 0.)
    assert {(i,j) for i,j,_ in match}=={(0,1),(1,0)},match
    assert not k.match_modes(old,new,lambda a,b:.1)
    extrap=k.crossing_fit([dict(Rout=R,real=(R-729)*.01) for R in (727,728,728.5)])
    assert abs(extrap['Rout']-729)<1e-8 and not extrap['is_bifurcation_confirmation']
    assert k.crossing_fit([dict(Rout=727,real=0)])['status']=='insufficient_matched_points'
    for path in (k.OUT/'production_v8/unallowed.json',k.PREP/'../outside.json'):
        try:k.safe(path)
        except ValueError:pass
        else:raise AssertionError('output escape permitted')
    # Budget remains charged after a crash, and completion returns only unused time.
    budget=k.Budget(out);budget.begin('crash fixture',10.)
    budget2=k.Budget(out);budget2.begin('next fixture',10.);budget2.finish()
    ledger=k.read(out/'budget.json');assert ledger['charged']>=10 and ledger['events'][0]['status']=='ACTIVE'
    # Controlled continuation failure forces arc recovery. This exercises the
    # state machine and restart ledger without running any production solve.
    saved={name:getattr(k,name) for name in ('TARGETS','mesh','transfer','direct_newton','spectrum','stationary_diagnosis','point_diagnostics')}
    calls=[]
    try:
        k.TARGETS=[725.,727.]
        k.mesh=lambda spec,R:mapping
        def fake_transfer(point,mesh):
            initial=y.copy();initial[-1]=point['Rout'];return initial
        k.transfer=fake_transfer
        def fake_newton(mesh,initial,reference,mode,target,destination,deadline,tangent=None):
            calls.append(mode);destination.mkdir(parents=True)
            value=initial.copy();accepted=(target==725. or mode=='arc')
            if mode=='arc':value[-1]=727.
            k.save_field(destination/('fields.npz' if accepted else 'checkpoint.npz'),mesh,value,accepted)
            return value,dict(accepted=accepted,A1=.2,failure=None if accepted else 'controlled obstruction')
        k.direct_newton=fake_newton
        k.spectrum=lambda *args,**kw:dict(min_three_modes=True,requested_scan_finished=True)
        k.stationary_diagnosis=lambda *args,**kw:[dict(accepted=True)]
        k.point_diagnostics=lambda *args,**kw:dict(fixture=True)
        plan=dict(seeds=[dict(Rout=720.),dict(Rout=725.)]);fixture=out/'lifecycle'
        state=k.variant(fixture,'baseline',plan,k.Budget(fixture))
        assert state['status']=='complete' and 'arc' in calls,state
        old_calls=len(calls)
        resumed=k.variant(fixture,'baseline',plan,k.Budget(fixture))
        assert len(calls)==old_calls and resumed['status']=='complete'
        # An unfinished older spectrum must resume at that older state, without
        # launching a backwards continuation from the newest accepted point.
        state['points'][0]['diagnosed']=False
        k.write(fixture/'baseline/state.json',state)
        resumed=k.variant(fixture,'baseline',plan,k.Budget(fixture))
        assert len(calls)==old_calls and resumed['status']=='complete'
        Path(state['points'][-1]['fields']).write_bytes(b'controlled corruption')
        try:k.variant(fixture,'baseline',plan,k.Budget(fixture))
        except RuntimeError as exc:assert 'hash changed' in str(exc)
        else:raise AssertionError('corrupted restart admitted')
        results['continuation_lifecycle']=dict(natural_attempts=calls.count('Rout'),arc_attempts=calls.count('arc'),
            resume_without_recomputation=True,older_partial_diagnostics_resume=True,changed_field_rejected=True)
    finally:
        for name,value in saved.items():setattr(k,name,value)
    results.update(mode_matching='overlap beats nearest-frequency crossing; low overlap rejected',
        phase_exclusion='two criteria; near-zero nonphase retained',resume='same identity accepted; changed identity rejected',
        budget='crash reservation retained',output_isolation='v8 and traversal rejected',
        seconds=time.monotonic()-started,production_started=False)
    k.write(out/'VALIDATION.json',results);k.write(k.PREP/'VALIDATION.json',dict(path=str(out/'VALIDATION.json'),**results))
    print(json.dumps(dict(passed=True,seconds=results['seconds'],QZ_max_distance=max(distances),validation=str(out/'VALIDATION.json'))))


if __name__=='__main__':main()
