#!/usr/bin/env python3
"""Independent small-grid operator/Jacobian and acceptance checks for S2."""
import os
for key in ('OPENBLAS_NUM_THREADS','OMP_NUM_THREADS','MKL_NUM_THREADS','NUMEXPR_NUM_THREADS','VECLIB_MAXIMUM_THREADS','BLIS_NUM_THREADS'):
    os.environ[key]='1'
import json,hashlib,math,tempfile,fcntl
from pathlib import Path
import numpy as np
from roscan_nonlinear_solver import ROOT,Mapping,Metric,Corrector,pack,unpack,common,energies,linear,convection,independent,norm,write
from roscan_nonlinear import row_for,quality,prepare,OUT
from roscan_nonlinear_fronts import crossing,diagnose,overlap

def main():
    records=[];rng=np.random.default_rng(20261102)
    for ro in [-1.,-.751,-.9,-.8,-.7,-.6,-.5,-.4,-.3,-.2,-.1]:
        row=row_for(ro) if ro not in (-1.,-.751) else dict(ro=ro,beta_c=68 if ro==-1. else 85,scale=1.,Rin=400.)
        scale=row['scale'];R=765*scale;w=70.;mapping=Mapping(8,4,120,ro,row['beta_c'],scale,row['Rin'],R)
        p,blocks,_,dblocks=mapping.at(R,w,True)
        q=unpack(rng.normal(size=mapping.nr*9*mapping.s)*1e-3,mapping.nr,4,mapping.s);y=np.r_[pack(q),w,R]
        separate=independent(mapping,y)
        derivative=pack(linear(dblocks,q,4)+mapping.convection_R(p,q));errors=[]
        for h in (.01*scale,.005*scale):
            pp,bp,*_=mapping.at(R+h,w);pm,bm,*_=mapping.at(R-h,w)
            fd=(pack(linear(bp,q,4)+convection(pp,q))-pack(linear(bm,q,4)+convection(pm,q)))/(2*h)
            errors.append(norm(fd-derivative)/norm(derivative))
        qref=q[:,5]/common.mass_norm(p,q[:,5]);metric=Metric(mapping)
        tangent=rng.normal(size=len(y));tangent/=math.sqrt(metric.dot(tangent,tangent));jerrors=[]
        for mode in ('energy','arc'):
            target=math.sqrt(energies(p,q)[1]) if mode=='energy' else 0.
            corrector=Corrector(mapping,qref,y,mode,target,metric=metric,tangent=tangent)
            J,*_=corrector.jacobian(y);v=rng.normal(size=len(y));v[:-2]*=1e-3;h=1e-5
            diff=(corrector.residual(y+h*v)-corrector.residual(y-h*v))/(2*h)
            jerrors.append(norm(diff-J@v)/norm(J@v))
        record=dict(ro=ro,scale=scale,template_vs_independent=separate['template_difference_relative'],
                    Rout_derivative=errors,energy_and_arc_Jv=jerrors,
                    passed=bool(separate['template_difference_relative']<1e-10 and max(errors)<1e-6 and max(jerrors)<1e-7))
        records.append(record)
    r=np.arange(5.)
    assert crossing(r,np.array([1.,-1.,-1.,1.,1.]),np.ones(5,bool),'up')==2.5
    assert crossing(r,np.array([1.,-1.,-1.,1.,1.]),np.ones(5,bool),'down')==.5
    d=dict(accepted=True,highest_to_fundamental=.01,independent=dict(relative_field_inf=1e-8,template_difference_relative=1e-12))
    assert quality(d)
    assert not quality(dict(d,highest_to_fundamental=.010001))
    assert not quality(dict(d,highest_to_fundamental=float('nan')))
    assert not quality(dict(d,independent={}))
    sources=sorted((ROOT/'examples').glob('roscan_nonlinear*.py'))
    result=dict(passed=all(x['passed'] for x in records),records=records,
        front_direction_checks='arrival up; Fig8 down; fail-closed truncation and independent-residual checks passed',
        scope='Small-grid coefficient/derivative tests at all Ro; regressions use random fields only, no historical branch recomputation',
        sources={str(p.relative_to(ROOT)):hashlib.sha256(p.read_bytes()).hexdigest() for p in sources})
    write(OUT/'short_checks/verification.json',result)
    if not result['passed']:raise RuntimeError(result)
    prepare()
    print(json.dumps(dict(passed=True,parameterized_operator_cases=len(records),max_Jv_error=max(max(x['energy_and_arc_Jv']) for x in records))))
if __name__=='__main__':main()
