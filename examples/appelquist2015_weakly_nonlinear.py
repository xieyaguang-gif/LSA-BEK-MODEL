#!/usr/bin/env python3
"""Global Hopf responses from a matched, archived neutral mode.

Predictions are never labelled converged nonlinear BVP roots. The nonlinear
convolution is the existing full GlobalRadial convolution with nonlinear=True.
"""
from __future__ import annotations
import argparse
from dataclasses import replace
import importlib.util
import json
from pathlib import Path
import resource
import sys
import time
import numpy as np
from scipy import sparse
from scipy.sparse.linalg import splu,LinearOperator,onenormest

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'src'))
sys.path.insert(0,str(ROOT/'examples'))
import appelquist2015_hopf_common as common
from llsa.secondary.radial import pack_field

OUT=ROOT/'results/appelquist2015_hopf_20261029/weakly_nonlinear'


def write(path,value):
    path.write_text(json.dumps(value,indent=2,allow_nan=False)+'\n')


def pair(value):
    return dict(real=float(np.real(value)),imag=float(np.imag(value)))


def norm(value):
    return float(np.linalg.norm(value))


def relative(error,reference):
    return norm(error)/max(norm(reference),1e-300)


def rss_gib():
    return resource.getrusage(resource.RUSAGE_SELF).ru_maxrss/1024**2


class Factor:
    def __init__(self,A,label):
        start=time.monotonic()
        self.A=A.tocsc()
        self.scale=np.maximum(np.asarray(abs(A).max(axis=1).toarray()).ravel(),1e-30)
        self.scaled=(sparse.diags(1/self.scale)@A).tocsc()
        self.lu=splu(self.scaled)
        self.info=dict(label=label,shape=list(A.shape),nnz=int(A.nnz),
            factor_nnz=int(self.lu.L.nnz+self.lu.U.nnz),seconds=time.monotonic()-start,
            peak_rss_gib=rss_gib())

    def solve(self,rhs):
        return self.lu.solve(np.asarray(rhs)/self.scale)

    def condition(self):
        inverse=LinearOperator(self.A.shape,matvec=lambda x:self.lu.solve(x),
            rmatvec=lambda x:self.lu.solve(x,trans='H'),dtype=self.A.dtype)
        estimate=float(onenormest(inverse))
        a_norm=float(np.asarray(abs(self.scaled).sum(axis=0)).max())
        return a_norm*estimate


def field(p,q1,q0=None,q2=None,q13=None,epsilon=1.):
    q=np.zeros((len(p.radii),5,p.s),complex)
    q[:,3]=epsilon*q1
    if q13 is not None:q[:,3]+=epsilon**3*q13
    q[:,1]=q[:,3].conj()
    if q0 is not None:q[:,2]=epsilon**2*q0
    if q2 is not None:q[:,4]=epsilon**2*q2;q[:,0]=q[:,4].conj()
    return q


def convection(p,q):
    if not p.config.nonlinear:
        raise RuntimeError('Nonlinear feedback requires nonlinear=True')
    return p._convection_field(q,p.derivatives(q)[0])


def cubic_vectors(p,q1,q0,q2):
    mean=convection(p,field(p,q1,q0=q0))[:,3]
    second=convection(p,field(p,q1,q2=q2))[:,3]
    return mean,second


def block_residual(p,blocks,q):
    F=convection(p,q)
    for j,A in enumerate(blocks):
        F[:,2+j]+=(A@q[:,2+j].ravel()).reshape(len(p.radii),p.s)
        if j:F[:,2-j]=F[:,2+j].conj()
    return F


def radial_metrics(p,q):
    m=p.ops[0].m
    v=q.reshape(len(p.radii),4,m)[:,:3]
    density=np.einsum('rvi,ij,rvj->r',v.conj(),p.ops[0].grid.mass,v).real
    i=int(np.argmax(density))
    return dict(velocity_mass_norm=common.mass_norm(p,q),euclidean_norm=norm(q),
        max_abs_velocity=float(abs(v).max()),peak_R=float(p.radii[i]),
        peak_density=float(density[i]))


def run(args):
    out=OUT/args.name
    out.mkdir(parents=True,exist_ok=False)
    beginning=time.monotonic()
    with np.load(args.neutral) as data:
        archive={key:data[key].copy() for key in data.files}
    n,dr=int(archive['n']),float(archive['dr'])
    ac,wc=float(archive['a_c']),float(archive['omega_c'])
    q1=np.asarray(archive['q1'],complex);left=np.asarray(archive['left_mode'],complex)
    if abs(float(archive['sigma_residual']))>1e-7:
        raise ValueError('Neutral growth residual exceeds allowed input threshold')
    construction=time.monotonic()
    p=common.make_problem(n,dr,ac,harmonics=2,nonlinear=True,damp_mean=False,omega=wc)
    if not np.array_equal(p.radii,archive['radii']):raise ValueError('Mismatched neutral radial grid')
    if q1.shape!=(len(p.radii),p.s):raise ValueError('Mismatched neutral mode shape')
    build_info=dict(seconds=time.monotonic()-construction,peak_rss_gib=rss_gib(),
        full_real_field_unknowns=len(p.radii)*p.width,harmonic_complex_unknowns=q1.size)
    A0,B0=common.complex_block(p,0);A1,B1=common.complex_block(p,1);A2,B2=common.complex_block(p,2)
    Bq=(B1@q1.ravel()).reshape(q1.shape)
    denominator=np.vdot(left.ravel(),Bq.ravel())
    linear_defect=(A1@q1.ravel()).reshape(q1.shape)
    left_defect=A1.getH()@left.ravel()
    sigma_residual=float(archive['sigma_residual'])
    left_corrected=left_defect+sigma_residual*(B1.getH()@left.ravel())
    matrix_inf=float(np.asarray(abs(A1).sum(axis=1)).max())
    da=common.sponge_start_matrix(p,1)
    lambda_a=np.vdot(left.ravel(),da@q1.ravel())/denominator
    sigma_a=-lambda_a.real;omega_a=lambda_a.imag
    neutral_check=dict(a_c=ac,omega_c=wc,sigma_residual=float(archive['sigma_residual']),
        right_residual_inf=float(abs(linear_defect).max()),
        left_residual_inf=float(abs(left_defect).max()),
        left_corrected_residual_inf=float(abs(left_corrected).max()),
        left_relative_backward_error=norm(left_corrected)/max(matrix_inf*norm(left),1e-300),
        left_norm=norm(left),projection_condition=norm(left)*norm(Bq)/abs(denominator),
        q1_mass_norm=common.mass_norm(p,q1),left_B_q=pair(denominator),
        dOmega_da=pair(omega_a+1j*sigma_a),nonlinear_flag=p.config.nonlinear,
        harmonic2_frequency_input=wc,harmonic2_time_factor=2*wc)
    base=field(p,q1)
    quadratic=convection(p,base)
    N0,N2=quadratic[:,2],quadratic[:,4]
    boundary=max(float(abs(quadratic[:,:,p.ops[0].boundary_rows]).max()),
        float(abs(quadratic[[0,-1]][:,:,p.velocity_rows]).max()))
    quadratic_checks=dict(mean_imaginary_max=float(abs(N0.imag).max()),boundary_forcing_max=boundary,
        conjugacy_error=float(abs(quadratic-quadratic[:,::-1].conj()).max()))
    for scale in [0.5,1.7]:
        test=convection(p,field(p,scale*q1))
        quadratic_checks[f'scale_{scale}_relative']=relative(test-scale**2*quadratic,scale**2*quadratic)
    fac2=Factor(A2,'A2')
    q2=fac2.solve(-N2.ravel()).reshape(q1.shape)
    q2_check=dict(residual_relative=relative(A2@q2.ravel()+N2.ravel(),N2),
        condition_scaled_1norm=fac2.condition(),response=radial_metrics(p,q2),forcing=radial_metrics(p,N2))
    # The same bordered A1 factor is reused for both mean-damping controls.
    border=None
    if not args.no_q13:
        row=(left.ravel().conj()@B1)
        bordered=sparse.bmat([[A1,sparse.csc_matrix(Bq.ravel()[:,None])],
            [sparse.csr_matrix(row[None,:]),None]],format='csc')
        border=Factor(bordered,'bordered_A1')
    parameter_q=None
    if border is not None:
        rhs=-da@q1.ravel()+lambda_a*Bq.ravel()
        parameter_q=border.solve(np.r_[rhs,0j])[:-1].reshape(q1.shape)
    np.savez_compressed(out/'grid_archive.npz',radii=p.radii,z=p.ops[0].grid.z,xi=p.ops[0].grid.xi,
        mass=p.ops[0].grid.mass,radial_weights=common.radial_weights(p.radii),n=n,dr=dr,a=2.,b=.6,c=.5)
    results=[]
    for damp_mean in [False,True]:
        case=out/('mean_damped' if damp_mean else 'mean_undamped');case.mkdir()
        p.outlet_sponge=replace(p.outlet_sponge,damp_mean=damp_mean)
        p._linear_cache.clear()
        A0,_=common.complex_block(p,0)
        fac0=Factor(A0,'A0_mean_damped' if damp_mean else 'A0_mean_undamped')
        q0=fac0.solve(-N0.real.ravel()).reshape(q1.shape)
        mean_vector,second_vector=cubic_vectors(p,q1,q0,q2)
        Cmean=np.vdot(left.ravel(),mean_vector.ravel())/denominator
        C2=np.vdot(left.ravel(),second_vector.ravel())/denominator
        C=Cmean+C2;Cvec=mean_vector+second_vector
        check0=dict(residual_relative=relative(A0@q0.ravel()+N0.real.ravel(),N0),
            condition_scaled_1norm=fac0.condition(),response=radial_metrics(p,q0),forcing=radial_metrics(p,N0))
        # Perform phase and amplitude-coordinate tests using the actual quadratic solves.
        phase=.47;rotation=np.exp(1j*phase)
        rotated=convection(p,field(p,rotation*q1))
        q0r=fac0.solve(-rotated[:,2].real.ravel()).reshape(q1.shape)
        q2r=fac2.solve(-rotated[:,4].ravel()).reshape(q1.shape)
        cm,cs=cubic_vectors(p,rotation*q1,q0r,q2r)
        Cr=np.vdot((rotation*left).ravel(),(cm+cs).ravel())/denominator
        multiplier=1.7
        scaled=convection(p,field(p,multiplier*q1))
        q0s=fac0.solve(-scaled[:,2].real.ravel()).reshape(q1.shape)
        q2s=fac2.solve(-scaled[:,4].ravel()).reshape(q1.shape)
        cm,cs=cubic_vectors(p,multiplier*q1,q0s,q2s)
        Cs=np.vdot(left.ravel(),(cm+cs).ravel())/(multiplier*denominator)
        invariance=dict(phase_angle=phase,C_phase=pair(Cr),
            phase_relative_error=float(abs(Cr-C)/max(abs(C),1e-300)),
            q0_phase_relative=relative(q0r-q0,q0),q2_phase_relative=relative(q2r-rotation**2*q2,q2),
            normalization_multiplier=multiplier,C_scaled=pair(Cs),
            normalization_relative_error=float(abs(Cs-multiplier**2*C)/max(abs(multiplier**2*C),1e-300)))
        q13=None;q13_check=None
        if border is not None:
            response=border.solve(np.r_[-Cvec.ravel()+C*Bq.ravel(),0j])
            q13=response[:-1].reshape(q1.shape)
            q13_check=dict(equation_relative=relative(A1@q13.ravel()+Cvec.ravel()-C*Bq.ravel(),Cvec),
                projection=pair(np.vdot(left.ravel(),B1@q13.ravel())),border_multiplier=pair(response[-1]),
                response=radial_metrics(p,q13))
        hierarchy=max(common.mass_norm(p,q0),common.mass_norm(p,q2),1e-30)
        epsilon=min(.01,.025/hierarchy)
        if q13 is not None:epsilon=min(epsilon,np.sqrt(.025/max(common.mass_norm(p,q13),1e-30)))
        if abs(sigma_a)>1e-12:
            kappa=C.real/sigma_a
            epsilon=min(epsilon,np.sqrt(.1/max(abs(kappa),1e-30)))
        else:kappa=None
        nu=None if kappa is None else omega_a*kappa+C.imag
        q13_physical=None if parameter_q is None or kappa is None else q13+kappa*parameter_q
        if q13_physical is not None:
            epsilon=min(epsilon,np.sqrt(.025/max(common.mass_norm(p,q13_physical),1e-30)))
        scaling=[]
        blocks=[A0,A1,A2]
        for eps in [epsilon,epsilon/2,epsilon/4]:
            expansion=field(p,q1,q0,q2,epsilon=eps)
            F=block_residual(p,blocks,expansion)
            cubic=F[:,3]-eps*linear_defect
            entry=dict(epsilon=eps,raw_residual_inf=float(abs(F).max()),
                first_harmonic_nonlinear_norm=norm(cubic),
                cubic_identity_relative=relative(cubic-eps**3*Cvec,eps**3*Cvec),
                linear_to_cubic_ratio=norm(eps*linear_defect)/max(norm(eps**3*Cvec),1e-300))
            if q13 is not None:
                shaped=field(p,q1,q0,q2,q13,epsilon=eps)
                Fs=block_residual(p,blocks,shaped)
                # Numerical sign check for B*dq/dt + F = 0. The known
                # projected linear defect is retained in the linear rate.
                linear_rate=-np.vdot(left.ravel(),linear_defect.ravel())/denominator
                projected=np.vdot(left.ravel(),Fs[:,3].ravel())/denominator
                correct_rate=linear_rate*eps-C*eps**3
                wrong_rate=linear_rate*eps+C*eps**3
                entry.update(linear_amplitude_rate=pair(linear_rate),
                    temporal_sign_correct_normalized=float(abs(correct_rate+projected)/max(abs(C)*eps**3,1e-300)),
                    temporal_sign_wrong_normalized=float(abs(wrong_rate+projected)/max(abs(C)*eps**3,1e-300)))
                Fs[:,3]-=eps*linear_defect+eps**3*C*Bq
                Fs[:,1]=Fs[:,3].conj()
                entry.update(remainder_after_known_linear_and_cubic_norm=norm(Fs),
                    fundamental_remainder_norm=norm(Fs[:,3]))
            scaling.append(entry)
        for first,second in zip(scaling,scaling[1:]):
            second['measured_cubic_order']=float(np.log(first['first_harmonic_nonlinear_norm']/second['first_harmonic_nonlinear_norm'])/np.log(2))
            if q13 is not None:
                for key in ['remainder_after_known_linear_and_cubic_norm','fundamental_remainder_norm']:
                    second[key+'_order']=float(np.log(first[key]/second[key])/np.log(2))
        # Independent full residual comparison at this real field, without
        # removing any defect. This validates the separated-harmonic evaluation.
        sample=field(p,q1,q0,q2,epsilon=epsilon)
        full=p.evaluate(pack_field(sample),np.zeros_like(sample[0]))
        separated=pack_field(block_residual(p,blocks,sample))
        full_check=relative(full-separated,full)
        prediction=dict(status='weakly_nonlinear_prediction_only',epsilon=epsilon,
            q0_to_primary_ratio=epsilon*common.mass_norm(p,q0),q2_to_primary_ratio=epsilon*common.mass_norm(p,q2),
            q13_to_primary_ratio=None if q13 is None else epsilon**2*common.mass_norm(p,q13),
            delta_a=None if kappa is None else float(kappa*epsilon**2),
            omega_shift=None if nu is None else float(nu*epsilon**2),
            target_a630_reached=False,full_BVP_corrected=False,
            center_cubic_saturation_tendency=bool(C.real>0),
            small_branch_side='a<a_c; subcritical cubic prediction' if C.real<0 and sigma_a>0 else 'determined by Re(C)/sigma_a',
            center_radial_rate_at_predicted_branch=None if kappa is None else float(-2*sigma_a*kappa*epsilon**2),
            no_a630_extrapolation='Only a small local branch coordinate is reported, not a target saturated amplitude.')
        result=dict(damp_mean=damp_mean,C=pair(C),Cmean=pair(Cmean),C2=pair(C2),
            mean_response=check0,second_response=q2_check,invariance=invariance,
            q13=q13_check,scaling=scaling,prediction=prediction,
            full_residual_reconstruction_relative=full_check,
            A0_factor=fac0.info,peak_rss_gib=rss_gib())
        write(case/'summary.json',result)
        np.savez_compressed(case/'responses.npz',radii=p.radii,q1=q1,left_mode=left,
            q0=q0,q2=q2,q13=np.zeros_like(q1) if q13 is None else q13,
            q1_a=np.zeros_like(q1) if parameter_q is None else parameter_q,
            q13_physical=np.zeros_like(q1) if q13_physical is None else q13_physical,
            N0=N0,N2=N2,Cvec=Cvec,Cmean_vector=mean_vector,C2_vector=second_vector,
            C=C,Cmean=Cmean,C2=C2,a_c=ac,omega_c=wc,n=n,dr=dr,damp_mean=damp_mean,
            metadata_json=json.dumps(result),neutral_source=str(args.neutral))
        results.append(result)
        del fac0
    summary=dict(name=args.name,neutral_source=str(args.neutral),neutral=neutral_check,
        construction=build_info,quadratic_checks=quadratic_checks,A2_factor=fac2.info,
        bordered_A1_factor=None if border is None else border.info,results=results,
        wall_seconds=time.monotonic()-beginning,peak_rss_gib=rss_gib(),
        status='weakly_nonlinear_responses_no_corrected_full_BVP_root',command=' '.join(sys.argv))
    write(out/'summary.json',summary);write(out/'completion.json',summary)
    print(json.dumps(dict(name=args.name,seconds=summary['wall_seconds'],peak_rss_gib=rss_gib(),
        C=[row['C'] for row in results],full_BVP_corrected=False)),flush=True)


def main():
    ap=argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--neutral',type=Path)
    ap.add_argument('--name')
    ap.add_argument('--no-q13',action='store_true')
    ap.add_argument('--summarize',action='store_true')
    args=ap.parse_args()
    if args.summarize:
        summarize();return
    if args.neutral is None or args.name is None:ap.error('--neutral and --name are required for a response calculation')
    run(args)


def summarize():
    configurations=[];rows=[]
    for path in sorted(OUT.glob('N*/summary.json')):
        data=json.loads(path.read_text())
        with np.load(path.parent/'grid_archive.npz') as grid:
            n,dr=int(grid['n']),float(grid['dr'])
        configurations.append(dict(name=data['name'],n=n,dr=dr,neutral=data['neutral'],
            wall_seconds=data['wall_seconds'],peak_rss_gib=data['peak_rss_gib'],
            q13_computed=data['bordered_A1_factor'] is not None,path=str(path)))
        for result in data['results']:
            valid=(result['mean_response']['residual_relative']<1e-8 and
                result['second_response']['residual_relative']<1e-8 and
                result['invariance']['phase_relative_error']<1e-6 and
                result['invariance']['normalization_relative_error']<1e-6 and
                result['full_residual_reconstruction_relative']<1e-6 and
                all(abs(point.get('measured_cubic_order',3.)-3.)<.01 for point in result['scaling']))
            rows.append(dict(configuration=data['name'],n=n,dr=dr,damp_mean=result['damp_mean'],
                a_c=data['neutral']['a_c'],omega_c=data['neutral']['omega_c'],
                C=result['C'],Cmean=result['Cmean'],C2=result['C2'],
                validated_response_checks=valid,
                mean_response_mass_norm=result['mean_response']['response']['velocity_mass_norm'],
                q2_response_mass_norm=result['second_response']['response']['velocity_mass_norm'],
                mean_condition_estimate=result['mean_response']['condition_scaled_1norm'],
                q2_condition_estimate=result['second_response']['condition_scaled_1norm'],
                prediction=result['prediction'],full_BVP_corrected=False))
    comparisons=[]
    for first,second,label in [('N32_dr1','N32_dr0.5','radial_1_to_0.5'),
                              ('N32_dr0.5','N32_dr0.25','radial_0.5_to_0.25'),
                              ('N32_dr0.5','N40_dr0.5','wall_normal_32_to_40')]:
        for damping in [False,True]:
            a=next((r for r in rows if r['configuration']==first and r['damp_mean']==damping),None)
            b=next((r for r in rows if r['configuration']==second and r['damp_mean']==damping),None)
            if a is None or b is None:continue
            comparisons.append(dict(label=label,damp_mean=damping,
                relative_C_real_change=b['C']['real']/a['C']['real']-1,
                relative_C_imag_change=b['C']['imag']/a['C']['imag']-1))
    summary=dict(status='validated_weakly_nonlinear_feedback_no_full_BVP_root_in_this_archive',
        configurations=configurations,coefficients=rows,comparisons=comparisons,
        calculation_seconds=sum(c['wall_seconds'] for c in configurations),
        all_response_checks_pass=all(row['validated_response_checks'] for row in rows),
        negative_C_real_all_controls=all(row['C']['real']<0 for row in rows),
        amplitude_equation='dA/dt = [sigma_lin - i*(omega_lin-omega_c)]*A - C*abs(A)^2*A + ...',
        normalization='Integral q1_velocity^H mass_z q1_velocity dR = 1; algebraic p^H B1 q1 = 1.',
        interpretation='Negative Re(C) gives a subcritical small-amplitude branch and center-amplitude instability when sigma_a>0; it does not exclude a stable finite-amplitude branch after a fold or higher-order effects.',
        target_a630_reached=False,full_Floquet_or_time_attraction_validated=False)
    write(OUT/'summary.json',summary)
    print(json.dumps(dict(configurations=len(configurations),controls=len(rows),
        seconds=summary['calculation_seconds'],all_checks=summary['all_response_checks_pass'],
        negative_real_C=summary['negative_C_real_all_controls'])))


if __name__=='__main__':main()
