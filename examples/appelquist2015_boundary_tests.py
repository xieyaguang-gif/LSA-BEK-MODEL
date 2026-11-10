#!/usr/bin/env python3
"""Separate sponge position/profile from outlet location in the free BVP.

All solves retain homogeneous inlet, real free omega, no forcing or auxiliary
growth/amplitude constraint.  Existing pilot files and driver are read-only.
"""
from __future__ import annotations
import argparse
from dataclasses import asdict
import importlib.util
import json
from pathlib import Path
import sys
import time
import numpy as np
from scipy.interpolate import BarycentricInterpolator

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'src'))
from llsa.secondary.global_radial import GlobalRadial,OutletSponge
from llsa.secondary.radial import RadialConfig,pack_field
from llsa.secondary.operators import PierOperators

spec=importlib.util.spec_from_file_location('free_pilot',ROOT/'examples/ro_minus1_beta68_free_bvp.py')
pilot=importlib.util.module_from_spec(spec);spec.loader.exec_module(pilot)
OUT=ROOT/'results/appelquist2015_tests_20261029/nonlinear'
OLD=ROOT/'results/ro_minus1_beta68_free_bvp_20261029'


def write(path,obj):
    path.write_text(json.dumps(obj,indent=2,allow_nan=False)+'\n')


def build(*,rmax=600.,sponge_start=550.,sponge_rise=30.,strength=10.,n=16,h=1,dr=2.,omega=50.5):
    r=np.linspace(450,rmax,round((rmax-450)/dr)+1)
    return GlobalRadial(r,RadialConfig(ro=-1,beta=68,n=n,harmonics=h,omega=omega),
        outlet_alpha=0j,outlet_sponge=OutletSponge(sponge_start,sponge_rise,strength,damp_mean=False))


def transfer(source,p):
    with np.load(source) as data:
        r,q,omega=data['radii'],data['modes'],float(data['omega'])
    oldh=(q.shape[1]-1)//2;oldm=q.shape[-1]//4
    oldgrid=PierOperators(n=oldm-1).grid
    shaped=q.reshape(len(r),2*oldh+1,4,oldm)
    values=BarycentricInterpolator(oldgrid.xi,shaped,axis=-1)(p.ops[0].grid.xi)
    out=np.zeros((len(p.radii),2*p.H+1,p.s),complex)
    for j in range(-min(oldh,p.H),min(oldh,p.H)+1):
        block=values[:,oldh+j].reshape(len(r),p.s)
        for k in range(p.s):
            out[:,p.H+j,k]=np.interp(p.radii,r,block[:,k].real)+1j*np.interp(p.radii,r,block[:,k].imag)
    return out,omega


def profiles():
    archived=np.genfromtxt(OLD/'amplitude_n16_mode4/fields.csv',delimiter=',',names=True)
    common=archived['R']
    old=OutletSponge(550,30,10)
    cases={}
    for name,end in [('B',600.),('C',620.)]:
        problem=build(rmax=end)
        if not np.array_equal(problem.radii[:len(common)],common):
            raise ValueError('Reference overlap grid differs')
        cases[name]=dict(rmax=end,start=550.,rise=30.,strength=10.,damp_mean=False,
            overlap_max_difference=float(abs(archived['sponge']-problem.sponge_strength[:len(common)]).max()),
            plateau_after=580.,plateau_length=end-580.)
    result=dict(cases=cases,exact_overlap_equal=all(x['overlap_max_difference']==0. for x in cases.values()),
        reference=str(OLD/'amplitude_n16_mode4/fields.csv'),
        explanation='Same OutletSponge(start=550,width=30,strength=10); only the constant plateau/outlet length changes.')
    OUT.mkdir(parents=True,exist_ok=True)
    write(OUT/'profile_preflight.json',result)
    rr=np.arange(450.,621.,2.)
    np.savetxt(OUT/'profile_preflight.csv',np.column_stack([rr,old.profile(rr),OutletSponge(570,30,10).profile(rr)]),
        delimiter=',',header='R,fixed_start550_rise30,old_moved_start570_rise30',comments='')
    return result


def independent(p,q,omega,initial_scale,tolerance):
    config=asdict(p.config)|{'omega':omega}
    original=GlobalRadial(p.radii,RadialConfig(**config),outlet_alpha=0j,outlet_sponge=p.outlet_sponge)
    F=original.evaluate(pack_field(q),np.zeros_like(q[0]))
    k=int(np.argmax(abs(F)));i=k//p.width;local=k%p.width
    part=local//p.s;field=local%p.s;m=p.ops[0].m
    z=float(p.ops[0].grid.z[field%m])
    kind=['u','v','w','continuity_or_pressure_boundary'][field//m]
    return dict(residual_inf=float(abs(F).max()),relative_to_initial_field_inf=float(abs(F).max()/initial_scale),
        relative_to_final_field_inf=float(abs(F).max()/max(abs(pack_field(q)).max(),1e-30)),
        tolerance=tolerance,field_scale=initial_scale,
        passes_original_equations=bool(abs(F).max()/initial_scale<tolerance),
        inlet_velocity_max=float(abs(q[0,:,:3*m]).max()),
        max_residual_location=dict(R=float(p.radii[i]),z=z if np.isfinite(z) else 'infinity',
            harmonic=0 if part==0 else (part+1)//2,part='real' if part==0 or part%2 else 'imag',equation=kind),
        pressure_reference='existing GlobalRadial wall/top closure')


def solve_case(args):
    target=OUT/args.case
    target.mkdir(parents=True,exist_ok=False)
    start=time.monotonic()
    params=dict(rmax=args.rmax,sponge_start=args.sponge_start,sponge_rise=args.sponge_rise,
        strength=args.strength,n=args.n,h=args.h,dr=args.dr)
    p=build(**params)
    q,omega=transfer(args.source,p)
    g=p.ops[0].grid
    np.savez_compressed(target/'grid_archive.npz',n=args.n,a=g.a,b=g.b,c=g.c,z=g.z,xi=g.xi,mass=g.mass,D=g.D,D2=g.D2)
    np.savetxt(target/'sponge_profile.csv',np.column_stack([p.radii,p.sponge_strength]),delimiter=',',header='R,lambda',comments='')
    write(target/'case.json',dict(parameters=params,source=str(args.source),omega_initial=omega,
        config=asdict(p.config),sponge=asdict(p.outlet_sponge),outlet_alpha=[0.,0.],
        homogeneous_inlet=True,forcing=None,auxiliary_growth=None,amplitude_constraint=None,
        tolerance=args.tol,max_iterations=args.max_iter,solver_budget=args.budget,
        mapping=dict(a=g.a,b=g.b,c=g.c),command=' '.join(sys.argv)))
    a=pilot.FreeFrequency(p,q)
    z,result=pilot.solve(a,a.initial_state(q,omega),target,'newton',
        max_iter=args.max_iter,tol=args.tol,budget=args.budget)
    final=a.physical_field(z)
    validation=independent(p,final,float(z[a.size]),result['field_scale'],args.tol)
    write(target/'independent_validation.json',validation)
    accepted=result['classification']=='accepted_nonzero' and validation['passes_original_equations']
    meta=dict(result)|dict(case=args.case,parameters=params,source_file=str(args.source),
        independently_accepted=accepted,independent_validation=validation,
        mathematical_state='unforced free-real-frequency finite-domain root' if accepted else 'failed/trivial iterate')
    if accepted:
        pilot.save_fields(target/'fields.npz',a,z,meta|{'history':None})
    # This includes failures and is the completion event for the background job.
    completion=dict(case=args.case,accepted=accepted,status=result['status'],classification=result['classification'],
        omega_lab=float(z[a.size]),omega_disk_signed=float(z[a.size]-68),
        omega_over_disk_beta=float(z[a.size]/68),seconds=result['seconds'],
        total_seconds=time.monotonic()-start,validation=validation,
        state_path=str(target/('fields.npz' if accepted else 'newton.npz')))
    write(target/'completion.json',completion)
    print(json.dumps(completion),flush=True)


def compare(first,second,common_end=540.):
    def load(path):
        with np.load(path) as data:
            return {k:data[k].copy() for k in data.files}
    aa,bb=load(first),load(second)
    rr=np.arange(450.,common_end+1e-10,2.)
    na=aa['modes'].shape[-1]//4-1;nb=bb['modes'].shape[-1]//4-1
    # When N differs, compare interpolants on the larger mapped Chebyshev grid.
    n=max(na,nb);grid=PierOperators(n=n).grid;m=n+1
    def sampled(data):
        r,q=data['radii'],data['modes'];h=(q.shape[1]-1)//2;oldm=q.shape[-1]//4
        oldgrid=PierOperators(n=oldm-1).grid
        vals=BarycentricInterpolator(oldgrid.xi,q.reshape(len(r),2*h+1,4,oldm),axis=-1)(grid.xi)
        val=vals[:,[h,h+1],:3].reshape(len(r),-1)
        result=np.array([np.interp(rr,r,val[:,k].real)+1j*np.interp(rr,r,val[:,k].imag) for k in range(val.shape[1])]).T
        return result.reshape(len(rr),2,3,m)
    q,qq=sampled(aa),sampled(bb)
    norm2=lambda x:float(np.einsum('vi,ij,vj->',x.reshape(-1,m).conj(),grid.mass,x.reshape(-1,m)).real)
    angle=float(np.angle(np.einsum('rvi,ij,rvj->',q[:,1].conj(),grid.mass,qq[:,1])))
    qq[:,1]*=np.exp(-1j*angle)
    rows=[]
    for R in [520.,530.,540.]:
        i=int(np.argmin(abs(rr-R)));e0=norm2(q[i,1]);e1=norm2(qq[i,1])
        rows.append(dict(R=R,E1_over_R2_first=e0,E1_over_R2_second=e1,E1_relative_change=e1/e0-1,
            complex_profile_relative_l2=np.sqrt(norm2(qq[i,1]-q[i,1])/e0),
            mean_first_l2=np.sqrt(norm2(q[i,0])),mean_second_l2=np.sqrt(norm2(qq[i,0])),
            mean_difference_l2=np.sqrt(norm2(qq[i,0]-q[i,0]))))
    return dict(first=str(first),second=str(second),common_region=[450.,common_end],
        uniform_alignment_phase=angle,omega_difference=float(bb['omega']-aa['omega']),
        omega_relative_change=float(bb['omega']/aa['omega']-1),
        q1_common_relative_l2=np.sqrt(norm2(qq[:,1]-q[:,1])/norm2(q[:,1])),
        mean_common_first_l2=np.sqrt(norm2(q[:,0])),mean_common_second_l2=np.sqrt(norm2(qq[:,0])),
        mean_common_difference_l2=np.sqrt(norm2(qq[:,0]-q[:,0])),stations=rows,
        comparison_grid_n=n,interpolation='mapped xi then linear radial; same dr grid is exact')


def check_jacobian(case):
    """Directional checks at an actual finite-amplitude checkpoint, not a toy grid."""
    target=OUT/case
    description=json.loads((target/'case.json').read_text())
    p=build(**description['parameters'])
    with np.load(target/'newton.npz') as data:
        q,ref,omega=data['modes'],data['reference'],float(data['omega'])
    a=pilot.FreeFrequency(p,ref)
    z=a.initial_state(q,omega)
    started=time.monotonic()
    F,J=a.evaluate(z,True)
    m=p.ops[0].m
    mean_v=np.concatenate([i*p.width+m+np.arange(1,m-1) for i in range(1,len(p.radii)-1)])
    rng=np.random.default_rng(2015)
    directions={}
    d=rng.normal(size=len(z));directions['all_unknowns']=d/np.linalg.norm(d)
    d=np.zeros_like(z);d[mean_v]=rng.normal(size=len(mean_v));directions['mean_v_only']=d/np.linalg.norm(d)
    rows=[]
    for label,direction in directions.items():
        exact=J@direction
        for step in [1e-4,1e-5,1e-6,1e-7]:
            fd=(a.evaluate(z+step*direction)-a.evaluate(z-step*direction))/(2*step)
            rows.append(dict(direction=label,step=step,
                global_relative_error=float(np.linalg.norm(fd-exact)/max(np.linalg.norm(exact),1e-30)),
                mean_v_relative_error=float(np.linalg.norm((fd-exact)[mean_v])/max(np.linalg.norm(exact[mean_v]),1e-30)),
                mean_v_absolute_error_inf=float(abs((fd-exact)[mean_v]).max())))
    result=dict(case=case,parameters=description['parameters'],checkpoint_omega=omega,
        augmented_residual_inf=float(abs(F).max()),checks=rows,seconds=time.monotonic()-started,
        max_global_relative_error=max(row['global_relative_error'] for row in rows),
        max_mean_v_relative_error=max(row['mean_v_relative_error'] for row in rows),
        interpretation='A directional-derivative consistency test; not a condition-number or existence proof.')
    write(target/'jacobian_directional_check.json',result)
    print(json.dumps(result),flush=True)


def summarize():
    cases=[]
    for path in sorted(OUT.glob('*/completion.json')):
        completion=json.loads(path.read_text())
        description=json.loads((path.parent/'case.json').read_text())
        result=json.loads((path.parent/'newton.json').read_text())
        cases.append(completion|dict(parameters=description['parameters'],source=description['source'],
            tolerance=description['tolerance'],max_E1=result['max_E1'],
            physical_region_max_E1=result['physical_region_max_E1'],
            phase_residual=result['phase_residual'],state_json=str(path.parent/'newton.json'),
            max_residual_location=completion['validation']['max_residual_location']))
    comparisons=[]
    candidates=[('old580_vs_B',OLD/'amplitude_n16_mode4/fields.npz',OUT/'B_r600_start550/fields.npz'),
        ('B_vs_old600_start570',OUT/'B_r600_start550/fields.npz',OLD/'domain_r600_n16_h1/fields.npz'),
        ('B_vs_C',OUT/'B_r600_start550/fields.npz',OUT/'C_r620_start550/fields.npz'),
        ('old580_vs_dr1',OLD/'amplitude_n16_mode4/fields.npz',OUT/'baseline_r580_dr1_n16_h1_resumed/fields.npz')]
    for name,first,second in candidates:
        if first.exists() and second.exists():
            data=compare(first,second);data['comparison_name']=name
            write(OUT/f'{name}.json',data);comparisons.append(data)
    result=dict(cases=cases,accepted_count=sum(row['accepted'] for row in cases),
        total_newton_seconds=sum(row['seconds'] for row in cases),comparisons=comparisons,
        acceptance='Free real omega, zero forcing and inlet velocities, no auxiliary growth/amplitude constraint; original equations relative to initial field <1e-7; nonzero branch required.',
        profile_preflight=json.loads((OUT/'profile_preflight.json').read_text()),
        unperformed='C and sponge-position comparison require an accepted B; priority switched to dr=1 after independent linear radial-resolution evidence.',
        limitations=['Uniform frequency is assumed by the rotating-wave ansatz.',
            'No dynamic attraction, group velocity, feedback direction, or absolute pinch has been established.'])
    write(OUT/'summary.json',result)
    print(json.dumps(dict(cases=len(cases),accepted=result['accepted_count'],newton_seconds=result['total_newton_seconds'])))


def main():
    ap=argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--preflight',action='store_true')
    ap.add_argument('--summarize',action='store_true')
    ap.add_argument('--check-jacobian',metavar='CASE')
    ap.add_argument('--case')
    ap.add_argument('--source',type=Path,default=OLD/'amplitude_n16_mode4/fields.npz')
    ap.add_argument('--rmax',type=float,default=600.)
    ap.add_argument('--sponge-start',type=float,default=550.)
    ap.add_argument('--sponge-rise',type=float,default=30.)
    ap.add_argument('--strength',type=float,default=10.)
    ap.add_argument('--n',type=int,default=16)
    ap.add_argument('--h',type=int,default=1)
    ap.add_argument('--dr',type=float,default=2.)
    ap.add_argument('--tol',type=float,default=1e-7)
    ap.add_argument('--budget',type=float,default=180.)
    ap.add_argument('--max-iter',type=int,default=24)
    args=ap.parse_args()
    if args.preflight:
        print(json.dumps(profiles()));return
    if args.summarize:
        summarize();return
    if args.check_jacobian:
        check_jacobian(args.check_jacobian);return
    if args.case is None:
        ap.error('--case required for an actual solve')
    solve_case(args)


if __name__=='__main__':
    main()
