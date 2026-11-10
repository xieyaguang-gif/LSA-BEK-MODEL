#!/usr/bin/env python3
"""Bounded H3/H4 verification of the observed a(E1) direction change.

Imports the frozen continuation implementation; no previous archive is edited.
"""
from __future__ import annotations
import argparse
import gc
import json
import os
from pathlib import Path
import signal
import sys
import time
for key in ('OPENBLAS_NUM_THREADS','OMP_NUM_THREADS','MKL_NUM_THREADS','NUMEXPR_NUM_THREADS'):
    os.environ[key]='1'
import numpy as np
from scipy import sparse

ROOT=Path(__file__).resolve().parents[1]
sys.path[:0]=[str(ROOT/'src'),str(ROOT/'examples')]
from appelquist2016_branch_continuation import (System,BlockInverse,BudgetExpired,
    pack,unpack,transfer,constraints_row,correct_point,archive_point,predict,
    common,mass_row,norm,relative,rss,write,pack_field,unpack_field)


def check_h4(out,deadline):
    start=time.monotonic();rng=np.random.default_rng(201604)
    p=common.make_problem(8,30.,620.,harmonics=4,nonlinear=True,omega=53.)
    qref=rng.normal(size=(len(p.radii),p.s))+1j*rng.normal(size=(len(p.radii),p.s))
    qref/=common.mass_norm(p,qref);eps=1e-3;s=System(p,qref,eps**2,deadline)
    q=np.zeros((s.nr,9,s.s),complex);q[:,4]=eps**2*rng.normal(size=qref.shape)
    for j in range(1,5):
        q[:,4+j]=eps*qref if j==1 else eps**2*(rng.normal(size=qref.shape)+1j*rng.normal(size=qref.shape))
        q[:,4-j]=q[:,4+j].conj()
    y=np.r_[pack(q),53.013,619.96]
    p2=common.make_problem(8,30.,y[-1],harmonics=4,nonlinear=True,omega=y[-2])
    f,j=p2.evaluate(y[:-2],np.zeros_like(q[0]),jacobian=True);fw,fa=s.columns(q,y[-1])
    rows=np.stack([constraints_row(p,s.ell,True)/eps,constraints_row(p,2*mass_row(p,q[:,5])/s.energy)])
    exact=sparse.bmat([[j,sparse.csc_matrix(np.stack([pack(fw),pack(fa)],axis=1))],
        [sparse.csr_matrix(rows),sparse.csc_matrix((2,2))]],format='csc')
    v=rng.normal(size=len(y));v[:-2]*=eps;expected=(exact@v)/np.r_[s.fscale,1.,1.]
    inv=BlockInverse(s,-.005+.02j);rhs=rng.normal(size=len(y));delta=inv.at(y)@rhs
    dq=unpack(delta[:-2],s.nr,4,s.s)
    right=unpack(rhs[:-2]*s.fscale,s.nr,4,s.s)[:,8].ravel()
    left=inv.factors[4].a@dq[:,8].ravel()+delta[-2]*fw[:,8].ravel()+delta[-1]*fa[:,8].ravel()
    shift=np.r_[np.zeros(len(y)-2),.01,0]
    fd=(s.residual(y+shift,False)[:-2]-s.residual(y-shift,False)[:-2])/.02
    result=dict(kind='H4_algebra_only',n=8,dr=30.,H=4,pack_error=float(abs(pack(q)-pack_field(q)).max()),
        unpack_error=float(abs(unpack(pack(q),s.nr,4,s.s)-unpack_field(pack(q),s.nr,4,s.s)).max()),
        original_residual_relative=relative(s.residual(y,False)[:-2]-f,f),jv_explicit_relative=relative(s.jacobian(y)@v-expected,expected),
        fourth_harmonic_inverse_relative=relative(left-right,right),frequency_column_fd_relative=relative(fd-pack(fw),pack(fw)))
    result['passed']=bool(result['pack_error']==0 and result['unpack_error']==0 and result['original_residual_relative']<1e-10
        and result['jv_explicit_relative']<1e-10 and result['fourth_harmonic_inverse_relative']<1e-8 and result['frequency_column_fd_relative']<1e-7)
    result['seconds']=time.monotonic()-start;write(out/'algebra_H4.json',result)
    if not result['passed']:raise RuntimeError('H4 algebra gate failed')
    print(json.dumps(dict(event='H4_algebra',**result)),flush=True)
    return result


def load_state(point):
    with np.load(ROOT/point['fields']) as d:
        return dict(q=d['field'].copy(),a=float(d['a']),omega=float(d['omega']),E1=float(d['E1']),
                    id=point['id'],fields=point['fields'],H=int(d['H']))


def interpolate(s,states,target):
    nearest=sorted(states,key=lambda z:abs(z['E1']-target))[:2]
    if len(nearest)<2:return predict(s,nearest,target)
    # The predictor is linear in the normalized harmonic shape versus E1;
    # ordering the two nearest points is immaterial to interpolation.
    return predict(s,[nearest[1],nearest[0]],target)


def main():
    ap=argparse.ArgumentParser();ap.add_argument('--manifest',type=Path,required=True)
    ap.add_argument('--responses',type=Path,required=True);ap.add_argument('--out',type=Path,required=True)
    ap.add_argument('--budget',type=int,default=600);ap.add_argument('--reserve-H4',type=int,default=350)
    args=ap.parse_args();args.out=args.out.resolve();args.out.mkdir(parents=True,exist_ok=True)
    if (args.out/'completion.json').exists():raise RuntimeError('Refusing to overwrite frozen turning check')
    start=time.monotonic();deadline=start+args.budget;h3_deadline=deadline-args.reserve_H4
    def alarm(signum,frame):raise BudgetExpired('Turning-check hard budget exhausted')
    signal.signal(signal.SIGALRM,alarm);signal.alarm(args.budget)
    result=dict(status='running',trigger='Accepted H3 a rises between A=.144 and .204 after decreasing below .144',
        command=' '.join(sys.argv),source_manifest=str(args.manifest),budget_seconds=args.budget,
        reserved_H4_seconds=args.reserve_H4,accepted_limit=6,failure_limit=3,points=[],failed_attempts=[],comparisons=[])
    manifest=dict(frozen=False,points=result['points'],field_axis_order='radius,signed_harmonic,variable_major_z',variable_order=['ur','ut','uz','p'])
    try:
        result['algebra_H4']=check_h4(args.out,h3_deadline)
        old=json.loads(args.manifest.read_text());states=[load_state(p) for p in old['points'] if p['H']==3]
        with np.load(args.responses) as d:data={k:d[k].copy() for k in d.files}
        n,dr=int(data['n']),float(data['dr']);ac,wc=float(data['a_c']),float(data['omega_c']);qref=data['q1']
        p=common.make_problem(n,dr,ac,harmonics=3,nonlinear=True,omega=wc)
        s=System(p,qref,.16**2,h3_deadline);left=data['left_mode'].ravel()
        d=np.vdot(left,common.sponge_start_matrix(p,1)@qref.ravel())/np.vdot(left,s.b[1]@qref.ravel())
        inv=BlockInverse(s,d);result['H3_preconditioner']=inv.info
        for amplitude in [.16,.18]:
            if time.monotonic()>h3_deadline-20:break
            s.set_energy(amplitude**2);y=interpolate(s,states,s.energy);ident=f'H3_A{amplitude:.6f}'
            out=args.out/ident;y,rec=correct_point(s,inv,y,out,data['Cvec'],max_newton=6)
            print(json.dumps(dict(event='H3_point',id=ident,accepted=rec['accepted'],a=rec['a'],omega=rec['omega'],seconds=rec['seconds'],failure=rec.get('failure'))),flush=True)
            if rec['accepted']:
                point=archive_point(s,y,rec,out,ident,'turning_check');result['points'].append(point);states.append(load_state(point))
            else:
                result['failed_attempts'].append(dict(id=ident,summary=str((out/'summary.json').relative_to(ROOT)),reason=rec.get('failure')))
                if rec.get('budget_stop') or len(result['failed_attempts'])>=3:break
        del inv,s,p;gc.collect()
        p=common.make_problem(n,dr,ac,harmonics=4,nonlinear=True,omega=wc)
        s=System(p,qref,.144**2,deadline);inv=BlockInverse(s,d);result['H4_preconditioner']=inv.info
        targets=[.144,.18,.204];h4=[]
        while targets and len(result['points'])<6 and len(result['failed_attempts'])<3:
            if time.monotonic()>deadline-25:break
            amplitude=targets.pop(0);energy=amplitude**2;s.set_energy(energy)
            matched=[z for z in states if abs(z['E1']-energy)<1e-12]
            if matched:
                source=matched[0];q=transfer(source['q'],3,4);y=np.r_[pack(q),source['omega'],source['a']]
            else:
                # Interpolation is an initial guess only; acceptance is H4.
                near=sorted(states,key=lambda z:abs(z['E1']-energy))[:2]
                transformed=[dict(z,q=transfer(z['q'],3,4)) for z in near]
                y=interpolate(s,transformed,energy);source=None
            ident=f'H4_A{amplitude:.6f}';out=args.out/ident
            y,rec=correct_point(s,inv,y,out,data['Cvec'],max_newton=6)
            print(json.dumps(dict(event='H4_point',id=ident,accepted=rec['accepted'],a=rec['a'],omega=rec['omega'],seconds=rec['seconds'],failure=rec.get('failure'))),flush=True)
            if rec['accepted']:
                point=archive_point(s,y,rec,out,ident,'harmonic_check')
                if source:
                    point.update(paired_H3_id=source['id'],paired_H3_fields=source['fields'])
                    qh=unpack(y[:-2],s.nr,4,s.s)
                    comp=dict(amplitude=amplitude,E1=energy,H3_id=source['id'],H4_id=ident,
                        H3_a=source['a'],H4_a=float(y[-1]),delta_a=float(y[-1]-source['a']),
                        H3_omega=source['omega'],H4_omega=float(y[-2]),delta_omega=float(y[-2]-source['omega']),
                        q1_mass_relative=common.mass_norm(p,qh[:,5]-source['q'][:,4])/amplitude,
                        mean_mass_relative=common.mass_norm(p,qh[:,4]-source['q'][:,3])/max(common.mass_norm(p,source['q'][:,3]),1e-300),
                        H4_highest_harmonic_energy_fraction=point['harmonic_energy'][-1]/sum(point['harmonic_energy'][1:]))
                    result['comparisons'].append(comp)
                result['points'].append(point);h4.append(point)
            else:
                result['failed_attempts'].append(dict(id=ident,summary=str((out/'summary.json').relative_to(ROOT)),reason=rec.get('failure')))
                if rec.get('budget_stop'):break
            if not targets and len(h4)==3 and len(result['points'])<6 and time.monotonic()<deadline-70:
                ordered=sorted(h4,key=lambda z:z['amplitude'])
                if not ordered[0]['a']>ordered[1]['a']<ordered[2]['a']:
                    extra=.16 if ordered[1]['a']>=ordered[0]['a'] else .192
                    targets=[extra];result['sixth_point_reason']='Three H4 points did not bracket a local minimum; add an interior diagnostic amplitude'
        h4=sorted(h4,key=lambda z:z['amplitude'])
        result['H4_direction_samples']=[dict(amplitude=z['amplitude'],a=z['a'],omega=z['omega']) for z in h4]
        brackets=[]
        for first,middle,last in zip(h4,h4[1:],h4[2:]):
            if first['a']>middle['a']<last['a']:brackets.append([first['amplitude'],last['amplitude']])
        result['H4_turning_retained']=bool(brackets);result['turning_amplitude_brackets']=brackets
        result['precise_fold_computed']=False;result['Floquet_stability_computed']=False
        result['status']='completed_bounded_turning_check'
    except (BudgetExpired,RuntimeError,MemoryError,ValueError) as exc:
        result['status']='stopped_with_preserved_accepted_points';result['stop_reason']=type(exc).__name__+': '+str(exc)
    finally:
        signal.alarm(0);result.update(seconds=time.monotonic()-start,peak_rss_gib=rss())
        manifest.update(frozen=True,status=result['status'])
        write(args.out/'manifest.json',manifest);write(args.out/'summary.json',result);write(args.out/'completion.json',result)
        print(json.dumps(dict(event='completion',status=result['status'],accepted=len(result['points']),failed=len(result['failed_attempts']),seconds=result['seconds'],peak_rss_gib=rss())),flush=True)


if __name__=='__main__':main()
