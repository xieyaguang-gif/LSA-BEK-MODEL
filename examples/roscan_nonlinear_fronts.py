#!/usr/bin/env python3
"""Scaled S1 front diagnostics; Fig.8 downcrossing and second-harmonic arrival."""
import os
for key in ('OPENBLAS_NUM_THREADS','OMP_NUM_THREADS','MKL_NUM_THREADS','NUMEXPR_NUM_THREADS'):
    os.environ[key]='1'
import json
from pathlib import Path
import numpy as np
from scipy.interpolate import BarycentricInterpolator, PchipInterpolator
from scipy.optimize import brentq
from llsa.secondary.global_radial import radial_differences

FRONTS=('front_fig8','front_2beta_1e6')

def crossing(r, f, valid, direction):
    for i in range(len(r)-1):
        if not (valid[i] and valid[i+1] and np.isfinite(f[i:i+2]).all()):
            continue
        if (direction=='up' and f[i]<0<=f[i+1]) or (direction=='down' and f[i]>0>=f[i+1]):
            return float(r[i]-f[i]*(r[i+1]-r[i])/(f[i+1]-f[i]))
    return None

def diagnose(path, curve_path, scale):
    with np.load(path) as d:
        r=d['radii']; H=int(d['H']); ro=float(d['ro'])
        q=d['field'].reshape(len(r),2*H+1,4,len(d['xi']))
        transform=lambda x:.6*x+.4*(x**3+.5*(1-x*x))
        xi=brentq(lambda x:transform(x)-(1.3-2)/(1.3+2),-1,1)
        v=BarycentricInterpolator(d['xi'],q,axis=-1)(xi)[:,H+1:,1]
        omega=float(d['omega']); a=float(r[-1]-100*scale)
    curve=json.loads(Path(curve_path).read_text())
    rows=curve.get('local_curve',curve.get('rows',[]))
    if len(rows)<3:raise ValueError('Insufficient local absolute-growth curve')
    lr=np.array([x['r'] for x in rows]);lg=np.array([x['growth'] for x in rows])
    if not np.all(np.diff(lr)>0):raise ValueError('Local curve radii not increasing')
    local=PchipInterpolator(lr,lg,extrapolate=False)(r)
    D,_=radial_differences(r)
    rms=np.sqrt(2*np.sum(abs(v)**2,axis=1))
    growth=D@np.log(np.maximum(rms,1e-250))
    valid=(r>=550*scale)&(r<=min(650*scale,a))&(r>=lr[0])&(r<=lr[-1])
    fig=crossing(r,growth-local,valid,'down')
    two=crossing(r,abs(v[:,1])-1e-6,(r>420*scale)&(r<a),'up')
    return dict(front_fig8=fig,front_2beta_1e6=two,
                front_fig8_Re=None if fig is None else abs(ro)*fig,
                front_2beta_1e6_Re=None if two is None else abs(ro)*two,
                omega_BEK=omega,omega_d=omega/((2-ro-ro*ro)/2),a=a,
                front_windows=dict(fig8=[550*scale,min(650*scale,a)],two_beta=[420*scale,a]),
                local_curve_range=[float(lr[0]),float(lr[-1])],local_curve_grid=32,z=1.3)

def overlap(low,high):
    differences={k:None if low.get(k) is None or high.get(k) is None else abs(low[k]-high[k]) for k in FRONTS}
    present=[v for v in differences.values() if v is not None]
    same=abs(low['Rout']-high['Rout'])<1e-7 and low['n']==high['n']
    # On the lower branch fronts may not yet exist. Compare both available fronts
    # and a physical velocity-field norm at exactly the same parameter instead.
    with np.load(low['fields']) as dl, np.load(high['fields']) as dh:
        ql=dl['field'].reshape(len(dl['radii']),2*int(dl['H'])+1,4,len(dl['xi']))
        qh=dh['field'].reshape(len(dh['radii']),2*int(dh['H'])+1,4,len(dh['xi']))
        h=min(int(dl['H']),int(dh['H']))
        ql=ql[:,int(dl['H'])-h:int(dl['H'])+h+1,:3]
        qh=qh[:,int(dh['H'])-h:int(dh['H'])+h+1,:3]
        phase=np.vdot(qh[:,h+1].ravel(),ql[:,h+1].ravel())
        angle=np.angle(phase)
        for j in range(-h,h+1):qh[:,h+j]*=np.exp(1j*j*angle)
        mass=np.asarray(dl['mass'])
        rw=np.gradient(dl['radii']);rw[[0,-1]]*=.5
        def norm2(q):
            return float(np.einsum('rhvi,ij,rhvj,r->',q.conj(),mass,q,rw,optimize=True).real)
        field_difference=float(np.sqrt(max(norm2(qh-ql),0.)/max(norm2(ql),1e-300)))
    both_fronts=all(v is not None for v in differences.values())
    passed=bool(same and low['equations_accepted'] and high['accepted'] and
                (both_fronts or field_difference<.05) and all(v<.1 for v in present) and
                all((low.get(k) is None)==(high.get(k) is None) for k in FRONTS))
    return dict(low=low['id'],high=high['id'],Rout=low['Rout'],H_low=low['H'],H_high=high['H'],
                front_differences=differences,relative_velocity_field_difference=field_difference,
                missing_fronts=[k for k in FRONTS if differences[k] is None],
                field_fallback_limit=None if both_fronts else .05,passed=passed)
