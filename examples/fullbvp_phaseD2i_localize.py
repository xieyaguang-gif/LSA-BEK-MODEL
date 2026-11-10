#!/usr/bin/env python3
"""Original-pencil mode localization with positive finite-slab quadrature."""
import os
for key in ('OPENBLAS_NUM_THREADS','OMP_NUM_THREADS','MKL_NUM_THREADS','NUMEXPR_NUM_THREADS','BLIS_NUM_THREADS'):
    os.environ[key]='1'
import argparse,json,time
from pathlib import Path
import numpy as np
from numpy.polynomial.legendre import leggauss
from scipy.interpolate import BarycentricInterpolator
from scipy.optimize import brentq
import fullbvp_phaseD2_stability as d
import fullbvp_phaseB_solver as base
OUT=d.OUT/'D2i'
EDGES=(0.,1.,10.,40.,80.,160.)


def xi_of_z(z,a=2.,b=.6,c=.5):
    if np.isinf(z):return 1.
    return brentq(lambda x:b*x+(1-b)*(x**3+c*(1-x*x))-(z-a)/(z+a),-1.,1.,xtol=2e-15)


def quadrature(xi,order=96,a=2.):
    x,w=leggauss(order);matrices=[];all_x=[];all_w=[]
    for lo,hi in zip(EDGES[:-1],EDGES[1:]):
        l,r=xi_of_z(lo,a),xi_of_z(hi,a);xx=(l+r)/2+(r-l)*x/2
        t=.6*xx+.4*(xx**3+.5*(1-xx*xx));tp=.6+.4*(3*xx*xx-xx)
        ww=w*(r-l)/2*2*a*tp/(1-t)**2
        P=BarycentricInterpolator(xi,np.eye(len(xi)),axis=0)(xx)
        matrices.append(P.T@(ww[:,None]*P));all_x.append(xx);all_w.append(ww)
        assert abs(ww.sum()-(hi-lo))<1e-8
    return matrices,all_x,all_w


def inner(q,v,M,r):
    weights=base.common.radial_weights(r)*r**3
    return np.einsum('rhvi,ij,rhvj,r->',q[:,:,:3].conj(),M,v[:,:,:3],weights,optimize=True)


def radial_integral(r,density,lo,hi):
    xs=np.r_[lo,r[(r>lo)&(r<hi)],hi]
    return float(np.trapezoid(np.interp(xs,r,density),xs))


def measures(q,matrices,r):
    H=(q.shape[1]-1)//2;wr=base.common.radial_weights(r)*r**3
    density=np.stack([np.einsum('rhvi,ij,rhvj->rhv',q.conj(),M,q,optimize=True).real for M in matrices],axis=-1)
    if density.min() < -1e-10*max(density.max(),1e-300):raise RuntimeError('nonpositive quadrature energy')
    density=np.maximum(density,0);totals=np.einsum('rhvz,r->hvz',density,wr)
    hv=totals[:,:,:4].sum(axis=2);K=float(hv[:,:3].sum());P=float(hv[:,3].sum())
    kinetic=hv[:,:3].sum(axis=1);harmonics=[float(kinetic[H])]+[float(kinetic[H-j]+kinetic[H+j]) for j in range(1,H+1)]
    var=hv[:,:3].sum(axis=0);z=totals[:,:3,:4].sum(axis=(0,1));rd=density[:,:,:3,:4].sum(axis=(1,2,3))*r**3
    cut=float(r[-1]-100);regions=[(r[0],420.),(420.,cut),(cut,r[-1])]
    windows={}
    for cap,ns in ((40,3),(80,4),(160,5)):
        vals=totals[:,:,:ns].sum(axis=2);KK=float(vals[:,:3].sum())
        windows[str(cap)]=dict(kinetic_relative_to_Z80=KK/K,w0_fraction=float(vals[H,2]/KK),
                               far_above_10_fraction=float(totals[:,:3,2:ns].sum()/KK))
    result=dict(kinetic_norm_squared=K,harmonic_fraction=[x/K for x in harmonics],
        velocity_fraction=dict(zip(('u','v','w'),map(float,var/K))),w0_fraction=float(hv[H,2]/K),
        pressure_same_weight_norm_fraction=P/(K+P),pressure_over_kinetic=P/K,
        z_fraction=dict(wall_0_1=float(z[0]/K),boundary_layer_1_10=float(z[1]/K),far_10_80=float(z[2:].sum()/K)),
        radial_fraction=dict(zip(('inlet_400_420','interior_420_Rout_minus100','outlet_last100'),
                                 [radial_integral(r,rd,l,u)/K for l,u in regions])),
        front_band_570_600_fraction=radial_integral(r,rd,570.,600.)/K,
        finite_window_sensitivity=windows,
        w0_infinity_relative_to_Z80_velocity_rms=float(np.sqrt(80*np.sum(wr*abs(q[:,H,2,-1])**2)/K)))
    for vals in (result['harmonic_fraction'],list(result['velocity_fraction'].values()),list(result['z_fraction'].values()),list(result['radial_fraction'].values())):
        assert abs(sum(vals)-1)<1e-9
    return result


def load_point(point):
    manifest=json.loads((point/'manifest.json').read_text());correction=json.loads((point/'corrected.json').read_text())
    center=json.loads((point/'center_00/DONE.json').read_text());fields=Path(correction['fields']);arrays=Path(center['arrays'])
    assert center['fingerprint']==correction['fingerprint']==manifest['fingerprint']==d.digest(manifest['params'])
    assert d.sha(fields)==correction['fields_sha256'] and d.sha(arrays)==center['arrays_sha256']
    with np.load(fields) as data:q=data['field'].copy();r=data['radii'].copy();xi=data['xi'].copy();n=int(data['n']);H=int(data['H']);omega=float(data['omega'])
    mapping=base.Mapping(n,H,len(r)-1);t=d.Temporal(mapping,q,float(r[-1]),omega)
    with np.load(arrays) as data:values=data['values'].copy();vectors=data['vectors'].copy()
    assert all(r['accepted'] for r in center['eigenpairs'])
    return t,xi,r,values,vectors,{str(p):d.sha(p) for p in (point/'manifest.json',point/'corrected.json',point/'center_00/DONE.json',fields,arrays)}


def analyze_point(point,label,select):
    t,xi,r,values,vectors,hashes=load_point(point);matrices,_,_=quadrature(xi);M=sum(matrices[:4]);phase=t.neutral()
    phi=d.lift(phase,t.nr,t.H,t.s).reshape(t.nr,2*t.H+1,4,t.map.m);pp=inner(phi,phi,M,r).real
    records=[];retained=[]
    for idx in select(values):
        value=values[idx];v=vectors[:,idx];cert=d.certify(t.J,t.B,max(t.b),value,v,tol=1e-8,phase=phase)
        assert cert['accepted'];q=d.lift(v,t.nr,t.H,t.s).reshape(phi.shape)
        coefficient=inner(phi,q,M,r)/pp;remainder=q-coefficient*phi
        orth=abs(inner(phi,remainder,M,r))/np.sqrt(pp*inner(remainder,remainder,M,r).real)
        assert orth<1e-7
        raw=measures(q,matrices,r);projected=measures(remainder,matrices,r)
        phi_overlap=float(abs(inner(phi,q,M,r))/np.sqrt(pp*inner(q,q,M,r).real))
        vector_remainder=v-coefficient*phase;res=t.J@v+value*(t.B@v)
        fine,_,_=quadrature(xi,order=160);fine_stats=measures(remainder,fine,r)
        quadrature_difference=abs(fine_stats['kinetic_norm_squared']/projected['kinetic_norm_squared']-1)
        assert quadrature_difference<1e-7
        record=dict(label=label,index=int(idx),value=d.zpair(value),n=t.map.n,H=t.H,Rout=float(r[-1]),omega=float(t.p.config.omega),
            certificate=cert,phase_physical_overlap=phi_overlap,phase_coefficient=d.zpair(coefficient),
            orthogonal_kinetic_fraction=projected['kinetic_norm_squared']/raw['kinetic_norm_squared'],phase_orthogonality=orth,
            original_residual_per_phase_orthogonal_mass=float(np.linalg.norm(res)/np.linalg.norm(t.B@vector_remainder)),
            raw=raw,phase_orthogonal_shape=projected,quadrature_relative_difference=quadrature_difference,
            source_hashes=hashes,projection_is_shape_diagnostic_not_eigenvector=True)
        records.append(record);retained.append((q,remainder,xi,r))
    return records,retained


def pair_comparison(pair,target):
    xi=pair[0][2];xj=target[2];r=target[3]
    assert all(np.array_equal(x[3],r) for x in pair)
    _,xx,ww=quadrature(xi);x=np.concatenate(xx[:4]);w=np.concatenate(ww[:4])
    Pi=BarycentricInterpolator(xi,np.eye(len(xi)),axis=0)(x);Pj=BarycentricInterpolator(xj,np.eye(len(xj)),axis=0)(x)
    Mii=Pi.T@(w[:,None]*Pi);Mij=Pi.T@(w[:,None]*Pj);Mjj=Pj.T@(w[:,None]*Pj)
    answer={}
    for name,j in (('raw',0),('phase_orthogonal_shape',1)):
        A=[q[j] for q in pair];b=target[j];G=np.array([[inner(a,c,Mii,r) for c in A] for a in A]);v=np.array([inner(a,b,Mij,r) for a in A]);bb=inner(b,b,Mjj,r).real
        answer[name]=dict(pair_subspace_energy_fraction=float((v.conj()@np.linalg.pinv(G,rcond=1e-14)@v).real/bb),
            individual_overlap=[float(abs(inner(a,b,Mij,r))/np.sqrt(inner(a,a,Mii,r).real*bb)) for a in A],
            gram_condition=float(np.linalg.cond(G)))
    return answer


def main(out):
    out=Path(out);out.mkdir(parents=True,exist_ok=False);start=time.monotonic();all_records=[]
    upper24=d.OUT/'production_v4/run_D2f_baseline/upper720_H6';upper32=d.OUT/'production_v6/wallnormal/upper720_H6'
    pick_pair=lambda vals:[i for i,z in enumerate(vals) if -.0003<z.real<-.00005 and .0001<abs(z.imag)<.0005]
    a,aa=analyze_point(upper24,'upper720_N24_pair',pick_pair);assert len(a)==2
    b,bb=analyze_point(upper32,'upper720_N32_positive',lambda vals:[int(np.argmax(vals.real))]);all_records+=a+b
    comparison=pair_comparison(aa,bb[0])
    for folder,label in [('baseline_rerun_lower','lower_N24'),('wallnormal','lower_N32')]:
        rows,_=analyze_point(d.OUT/'production_v6'/folder/'lower719p948669_H4',label,
            lambda vals:[i for i,z in enumerate(vals) if z.real>1e-6 or (-1e-5<z.real<0 and abs(z.imag)>.0001)])
        all_records+=rows
    result=dict(seconds=time.monotonic()-start,modes=all_records,upper_pair_comparison=comparison,
        definitions=dict(kinetic='sum signed Fourier velocity |q_h|^2 integrated with R^3 dR dz on 0<=z<=80',
            pressure='same R^3-weighted stored p coefficient L2/(kinetic+p L2), diagnostic only; not physical energy',
            phase='orthogonal projection using finite-slab physical kinetic inner product; remainder is not an eigenvector',
            wall_normal_regions=['0..1','1..10','10..80'],radial_regions=['400..420','420..Rout-100','Rout-100..Rout'],
            front_band='fixed diagnostic 570..600, not an inferred front position',
            infinite_domain='w0 infinity is free in cylindrical closure, so a nonzero plateau has no finite whole-half-line kinetic norm'),
        production_started=False)
    d.write(out/'LOCALIZATION.json',result)
    print(json.dumps(dict(seconds=result['seconds'],modes=len(all_records),upper_pair_comparison=comparison,
        upper=[dict(label=x['label'],value=x['value'],remaining_fraction=x['orthogonal_kinetic_fraction'],
                    w0=x['phase_orthogonal_shape']['w0_fraction'],z=x['phase_orthogonal_shape']['z_fraction'],
                    harmonics=x['phase_orthogonal_shape']['harmonic_fraction']) for x in a+b])))

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--out',type=Path,required=True);args=p.parse_args();main(args.out)
