#!/usr/bin/env python3
"""Finite-energy continuation of the original H-harmonic rotating-wave BVP.

Generalizes the verified appelquist2015_hopf_block_corrector.py without editing
that frozen implementation. Sparse shifted factors and scalar Schur only.
"""
from __future__ import annotations
import argparse
import gc
import json
import math
import os
from pathlib import Path
import signal
import sys
import time
for _k in ('OPENBLAS_NUM_THREADS','OMP_NUM_THREADS','MKL_NUM_THREADS','NUMEXPR_NUM_THREADS'):
    os.environ[_k]='1'
import numpy as np
from scipy import sparse
from scipy.sparse.linalg import LinearOperator, gmres

ROOT=Path(__file__).resolve().parents[1]
sys.path[:0]=[str(ROOT/'src'),str(ROOT/'examples')]
import appelquist2015_hopf_common as common
from appelquist2015_hopf_block_corrector import Factor,mass_row,norm,relative,rss,write
from llsa.secondary.radial import pack_field,unpack_field


class BudgetExpired(RuntimeError):
    pass


def pack(q):
    h=(q.shape[1]-1)//2
    values=[q[:,h].real]
    for j in range(1,h+1):values.extend([q[:,h+j].real,q[:,h+j].imag])
    return np.stack(values,axis=1).ravel()


def unpack(x,nr,h,s):
    a=np.asarray(x).reshape(nr,2*h+1,s)
    q=np.empty_like(a,dtype=complex);q[:,h]=a[:,0]
    for j in range(1,h+1):
        q[:,h+j]=a[:,2*j-1]+1j*a[:,2*j]
        q[:,h-j]=q[:,h+j].conj()
    return q


def transfer(q,old_h,new_h):
    z=np.zeros((len(q),2*new_h+1,q.shape[2]),complex)
    take=min(old_h,new_h);z[:,new_h-take:new_h+take+1]=q[:,old_h-take:old_h+take+1]
    return z


def constraints_row(p,row,imaginary=False):
    z=np.zeros((len(p.radii),2*p.H+1,p.s));v=row.reshape(len(p.radii),p.s)
    z[:,1]=v.imag if imaginary else v.real
    z[:,2]=v.real if imaginary else -v.imag
    return z.ravel()


class System:
    def __init__(self,p,qref,energy,deadline):
        self.p=p;self.h=p.H;self.nr=len(p.radii);self.s=p.s
        self.qref=qref.copy();self.ell=mass_row(p,qref)/common.mass_inner(p,qref,qref).real
        self.wc=p.config.omega;self.ac=p.outlet_sponge.start
        self.blocks,self.b=zip(*(common.complex_block(p,j) for j in range(self.h+1)))
        self.bdiag=self.b[1].diagonal().reshape(self.nr,self.s)
        scales=[np.maximum(np.asarray(abs(a).max(axis=1).toarray()).ravel(),1e-30).reshape(self.nr,self.s) for a in self.blocks]
        values=[scales[0]]
        for j in range(1,self.h+1):values.extend([scales[j],scales[j]])
        self.row_scale=np.stack(values,axis=1).ravel()
        self.size=self.nr*(2*self.h+1)*self.s+2;self.jv_calls=0;self.deadline=deadline
        self.set_energy(energy)

    def check_budget(self):
        if time.monotonic()>self.deadline:raise BudgetExpired('Numerical phase deadline reached')

    def set_energy(self,energy):
        self.energy=float(energy);self.epsilon=np.sqrt(self.energy);self.fscale=self.epsilon*self.row_scale

    def conv(self,q):
        self.check_budget()
        qr=(self.p.D@q.reshape(self.nr,-1)).reshape(q.shape)
        return self.p._convection_field(q,qr)

    def linear(self,q,omega,a):
        out=np.zeros_like(q)
        delta=28*(common.smooth_step_and_derivative((self.p.radii-a)/40)[0]-common.smooth_step_and_derivative((self.p.radii-self.ac)/40)[0])
        ds=delta[:,None]*self.bdiag
        for j in range(self.h+1):
            out[:,self.h+j]=(self.blocks[j]@q[:,self.h+j].ravel()).reshape(self.nr,self.s)
            if j or self.p.outlet_sponge.damp_mean:out[:,self.h+j]+=ds*q[:,self.h+j]
            if j:
                out[:,self.h+j]-=1j*j*(omega-self.wc)*self.bdiag*q[:,self.h+j]
                out[:,self.h-j]=out[:,self.h+j].conj()
        return out

    def columns(self,q,a):
        fw,fa=np.zeros_like(q),np.zeros_like(q)
        da=-.7*common.smooth_step_and_derivative((self.p.radii-a)/40)[1]
        for j in range(self.h+1):
            if j or self.p.outlet_sponge.damp_mean:fa[:,self.h+j]=da[:,None]*self.bdiag*q[:,self.h+j]
            if j:
                fw[:,self.h+j]=-1j*j*self.bdiag*q[:,self.h+j]
                fw[:,self.h-j]=fw[:,self.h+j].conj();fa[:,self.h-j]=fa[:,self.h+j].conj()
        return fw,fa

    def residual(self,y,scaled=True):
        q=unpack(y[:-2],self.nr,self.h,self.s)
        f=pack(self.linear(q,y[-2],y[-1])+self.conv(q))
        phase=(self.ell@q[:,self.h+1].ravel()).imag/self.epsilon
        energy=common.mass_inner(self.p,q[:,self.h+1],q[:,self.h+1]).real/self.energy-1
        return np.r_[f/self.fscale if scaled else f,phase,energy]

    def jacobian(self,y):
        q=unpack(y[:-2],self.nr,self.h,self.s);qn=norm(q)
        fw,fa=self.columns(q,y[-1]);erow=2*mass_row(self.p,q[:,self.h+1])/self.energy
        def multiply(v):
            self.check_budget();self.jv_calls+=1
            dq=unpack(v[:-2],self.nr,self.h,self.s);vn=norm(dq)
            if vn:
                step=max(qn,1e-30)/vn;dn=(self.conv(q+step*dq)-self.conv(q-step*dq))/(2*step)
            else:dn=np.zeros_like(q)
            f=self.linear(dq,y[-2],y[-1])+dn+v[-2]*fw+v[-1]*fa
            return np.r_[pack(f)/self.fscale,(self.ell@dq[:,self.h+1].ravel()).imag/self.epsilon,(erow@dq[:,self.h+1].ravel()).real]
        return LinearOperator((self.size,self.size),matvec=multiply,dtype=float)


class BlockInverse:
    def __init__(self,s,derivative,shift=.05):
        self.s=s;self.derivative=complex(derivative)
        if abs(self.derivative.real)<1e-12:raise ValueError('Degenerate transversality')
        self.factors=[]
        for j,a in enumerate(s.blocks):
            s.check_budget();self.factors.append(Factor(a+shift*s.b[j] if j==1 else a,f'A{j}_shifted' if j==1 else f'A{j}'))
        self.u=s.b[1]@s.qref.ravel();self.z=self.factors[1].solve(self.u);self.schur=s.ell@self.z
        self.info=dict(factors=[f.info for f in self.factors],shift=shift,schur=[self.schur.real,self.schur.imag],derivative=[self.derivative.real,self.derivative.imag])

    def at(self,y):
        s=self.s;fw,fa=s.columns(unpack(y[:-2],s.nr,s.h,s.s),y[-1])
        def solve(rhs):
            s.check_budget();b=unpack(rhs[:-2]*s.fscale,s.nr,s.h,s.s)
            c=s.epsilon*(rhs[-1]/2+1j*rhs[-2]);first=self.factors[1].solve(b[:,s.h+1].ravel())
            eta=(s.ell@first-c)/self.schur;first-=self.z*eta
            da=eta.real/(s.epsilon*self.derivative.real);dw=self.derivative.imag*da-eta.imag/s.epsilon
            q=np.zeros_like(b);q[:,s.h+1]=first.reshape(s.nr,s.s);q[:,s.h-1]=q[:,s.h+1].conj()
            for j in range(s.h+1):
                if j==1:continue
                right=b[:,s.h+j]-dw*fw[:,s.h+j]-da*fa[:,s.h+j]
                if j==0:right=right.real
                q[:,s.h+j]=self.factors[j].solve(right.ravel()).reshape(s.nr,s.s)
                if j:q[:,s.h-j]=q[:,s.h+j].conj()
            return np.r_[pack(q),dw,da]
        return LinearOperator((s.size,s.size),matvec=solve,dtype=float)


def solve_step(s,inv,y,rhs):
    start=time.monotonic();j=s.jacobian(y);p=inv.at(y);history=[]
    operator=LinearOperator(j.shape,matvec=lambda v:j@(p@v),dtype=float)
    z,info=gmres(operator,rhs,rtol=1e-7,atol=0,restart=35,maxiter=2,
                 callback=lambda r:history.append(float(r)),callback_type='pr_norm')
    step=p@z;error=j@step-rhs;raw=error.copy();raw[:-2]*=s.fscale
    original=rhs.copy();original[:-2]*=s.fscale
    return step,dict(info=int(info),iterations=len(history),history=history,
        scaled_original_linear_relative=relative(error,rhs),unscaled_original_linear_relative=relative(raw,original),
        raw_field_linear_inf=float(abs(raw[:-2]).max()),seconds=time.monotonic()-start)


def checks(out,deadline):
    begin=time.monotonic();rng=np.random.default_rng(201629);p=common.make_problem(8,30.,620.,harmonics=3,nonlinear=True,omega=53.)
    qref=rng.normal(size=(len(p.radii),p.s))+1j*rng.normal(size=(len(p.radii),p.s));qref/=common.mass_norm(p,qref)
    eps=1e-3;s=System(p,qref,eps**2,deadline)
    q=np.zeros((s.nr,7,s.s),complex);q[:,3]=eps**2*rng.normal(size=qref.shape)
    for j in range(1,4):
        q[:,3+j]=eps*qref if j==1 else eps**2*(rng.normal(size=qref.shape)+1j*rng.normal(size=qref.shape));q[:,3-j]=q[:,3+j].conj()
    y=np.r_[pack(q),53.017,619.97];p2=common.make_problem(8,30.,y[-1],harmonics=3,nonlinear=True,omega=y[-2])
    f,j=p2.evaluate(y[:-2],np.zeros_like(q[0]),jacobian=True);fw,fa=s.columns(q,y[-1])
    rows=np.stack([constraints_row(p,s.ell,True)/eps,constraints_row(p,2*mass_row(p,q[:,4])/s.energy)])
    explicit=sparse.bmat([[j,sparse.csc_matrix(np.stack([pack(fw),pack(fa)],axis=1))],[sparse.csr_matrix(rows),sparse.csc_matrix((2,2))]],format='csc')
    v=rng.normal(size=len(y));v[:-2]*=eps
    exact=(explicit@v)/np.r_[s.fscale,1.,1.];jv=s.jacobian(y)@v
    result=dict(H=3,n=8,dr=30.,kind='algebra_only',pack_error=float(abs(pack(q)-pack_field(q)).max()),
        unpack_error=float(abs(unpack(pack(q),s.nr,3,s.s)-unpack_field(pack(q),s.nr,3,s.s)).max()),
        original_residual_relative=relative(s.residual(y,False)[:-2]-f,f),jv_explicit_relative=relative(jv-exact,exact))
    inv=BlockInverse(s,-.005+.02j);r=rng.normal(size=len(y));delta=inv.at(y)@r;dq=unpack(delta[:-2],s.nr,3,s.s)
    expected=unpack(r[:-2]*s.fscale,s.nr,3,s.s)[:,6].ravel()
    check=inv.factors[3].a@dq[:,6].ravel()+delta[-2]*fw[:,6].ravel()+delta[-1]*fa[:,6].ravel()
    result['third_harmonic_inverse_relative']=relative(check-expected,expected)
    result['frequency_column_fd_relative']=relative((s.residual(y+np.r_[np.zeros(len(y)-2),1e-4,0],False)[:-2]-s.residual(y-np.r_[np.zeros(len(y)-2),1e-4,0],False)[:-2])/2e-4-pack(fw),pack(fw))
    result['passed']=bool(result['pack_error']==0 and result['unpack_error']==0 and result['original_residual_relative']<1e-10 and result['jv_explicit_relative']<1e-10 and result['third_harmonic_inverse_relative']<1e-8 and result['frequency_column_fd_relative']<1e-7)
    result['seconds']=time.monotonic()-begin;write(out/'algebra_checks.json',result)
    if not result['passed']:raise RuntimeError('H3 generalization algebra checks failed')
    print(json.dumps(dict(event='algebra_checks',**result)),flush=True)
    return result


def energy_by_harmonic(p,q):
    return [float(common.mass_inner(p,q[:,p.H+j],q[:,p.H+j]).real) for j in range(p.H+1)]


def state_metrics(s,y,initial_scale,cubic_norm,cubic_inf):
    q=unpack(y[:-2],s.nr,s.h,s.s);n=s.conv(q);f=s.residual(y,False);raw=f[:-2]
    n1=n[:,s.h+1];feedback=norm(n1);feedback_inf=float(abs(n1).max())
    result=dict(a=float(y[-1]),omega=float(y[-2]),residual_inf=float(abs(raw).max()),residual_l2=norm(raw),
        residual_over_initial_field_inf=float(abs(raw).max()/initial_scale),
        residual_over_final_field_inf=float(abs(raw).max()/max(abs(y[:-2]).max(),1e-300)),
        phase=float(f[-2]),energy_constraint=float(f[-1]),nonlinear_packed_l2=norm(pack(n)),nonlinear_packed_inf=float(abs(pack(n)).max()),
        fundamental_nonlinear_l2=feedback,fundamental_nonlinear_inf=feedback_inf,
        residual_l2_over_fundamental_nonlinear=norm(raw)/max(feedback,1e-300),
        residual_inf_over_fundamental_nonlinear=float(abs(raw).max()/max(feedback_inf,1e-300)),
        residual_l2_over_weak_cubic_reference=norm(raw)/cubic_norm,residual_inf_over_weak_cubic_reference=float(abs(raw).max()/cubic_inf),
        weak_reference_is_accuracy_claim=False,harmonic_energy=energy_by_harmonic(s.p,q))
    result['acceptable']=bool(result['residual_over_initial_field_inf']<1e-7 and result['residual_over_final_field_inf']<1e-7
        and max(abs(f[-2:]))<1e-8 and result['residual_l2_over_fundamental_nonlinear']<1e-3
        and result['residual_inf_over_fundamental_nonlinear']<1e-3 and result['harmonic_energy'][1]>.5*s.energy)
    return result


def independent(s,y,initial_scale):
    s.check_budget();p=common.make_problem(s.p.config.n,s.p.radii[1]-s.p.radii[0],y[-1],harmonics=s.h,nonlinear=True,
        damp_mean=s.p.outlet_sponge.damp_mean,omega=y[-2])
    q=unpack(y[:-2],s.nr,s.h,s.s);f=p.evaluate(pack_field(q),np.zeros_like(q[0]));n=p._convection_field(q,p.derivatives(q)[0]);n1=n[:,s.h+1]
    con=s.residual(y,False)[-2:];energies=energy_by_harmonic(p,q)
    result=dict(original_residual_inf=float(abs(f).max()),original_residual_l2=norm(f),
        relative_initial_field=float(abs(f).max()/initial_scale),relative_final_field=float(abs(f).max()/abs(y[:-2]).max()),
        residual_l2_over_fundamental_nonlinear=norm(f)/max(norm(n1),1e-300),
        residual_inf_over_fundamental_nonlinear=float(abs(f).max()/max(abs(n1).max(),1e-300)),
        fundamental_nonlinear_l2=norm(n1),fundamental_nonlinear_inf=float(abs(n1).max()),nonlinear_packed_l2=norm(pack(n)),
        nonlinear_packed_inf=float(abs(pack(n)).max()),phase=float(con[0]),energy_constraint=float(con[1]),harmonic_energy=energies,
        neutral_defect_subtracted=False,auxiliary_forcing=False,rebuilt_at_final_parameters=True)
    result['accepted']=bool(result['relative_initial_field']<1e-7 and result['relative_final_field']<1e-7 and max(abs(con))<1e-8
        and result['residual_l2_over_fundamental_nonlinear']<1e-3 and result['residual_inf_over_fundamental_nonlinear']<1e-3 and energies[1]>.5*s.energy)
    return result


def correct_point(s,inv,y,out,cvec,max_newton=5):
    out.mkdir(parents=True,exist_ok=False);start=time.monotonic();initial_scale=float(abs(y[:-2]).max());iterations=[]
    c=s.epsilon**3*cvec;record=dict(status='failed_no_root',accepted=False,H=s.h,E1=s.energy,initial_field_inf=initial_scale,iterations=iterations)
    try:
        for k in range(max_newton+1):
            row=state_metrics(s,y,initial_scale,norm(c),float(abs(c).max()));row['iteration']=k;iterations.append(row)
            if row['acceptable']:
                check=independent(s,y,initial_scale);record['independent_validation']=check
                if check['accepted']:record.update(status='accepted_original_BVP_root',accepted=True)
                else:record['failure']='Independent original-equation validation failed'
                break
            if k==max_newton:record['failure']='Newton iteration limit';break
            f=s.residual(y);step,lin=solve_step(s,inv,y,-f);row['linear']=lin
            if lin['scaled_original_linear_relative']>.2:record['failure']='Insufficient original Krylov linear residual';break
            found=False
            for back in range(7):
                trial=y+2.**(-back)*step
                if not np.isfinite(trial[-2:]).all() or not 570<=trial[-1]<=650:continue
                ft=s.residual(trial)
                if norm(ft)<(1-1e-4*2.**(-back))*norm(f):
                    y=trial;row['step_length']=2.**(-back);found=True;break
            if not found:record['failure']='Original-equation line search or parameter bounds';break
    except BudgetExpired as exc:record['failure']=str(exc);record['budget_stop']=True
    record.update(seconds=time.monotonic()-start,peak_rss_gib=rss(),a=float(y[-1]),omega=float(y[-2]),jv_calls=s.jv_calls)
    write(out/'summary.json',record)
    if not record['accepted']:
        np.savez_compressed(out/'checkpoint.npz',field=unpack(y[:-2],s.nr,s.h,s.s),H=s.h,a=y[-1],omega=y[-2],E1=s.energy,accepted=False)
    return y,record


def archive_point(s,y,record,out,ident,role,paired=None):
    q=unpack(y[:-2],s.nr,s.h,s.s);p=s.p
    np.savez_compressed(out/'fields.npz',field=q,radii=p.radii,z=p.ops[0].grid.z,xi=p.ops[0].grid.xi,mass=p.ops[0].grid.mass,
        radial_weights=common.radial_weights(p.radii),H=s.h,n=p.config.n,dr=p.radii[1]-p.radii[0],a=y[-1],omega=y[-2],E1=s.energy,
        accepted=True,axis_order='radius,signed_harmonic,variable_major_z',variable_order='ur,ut,uz,p',signed_harmonics=np.arange(-s.h,s.h+1),
        ro=-1.,beta=68.,damp_mean=False,outer_sponge_strength=28.,outer_sponge_rise=40.)
    point=dict(id=ident,fields=str((out/'fields.npz').relative_to(ROOT)),summary=str((out/'summary.json').relative_to(ROOT)),
        status='accepted_original_BVP_root',accepted=True,n=p.config.n,dr=float(p.radii[1]-p.radii[0]),H=s.h,a=float(y[-1]),omega=float(y[-2]),
        E1=s.energy,amplitude=float(np.sqrt(s.energy)),harmonic_energy=record['independent_validation']['harmonic_energy'],role=role)
    if paired is not None:point['paired_H2_id']=paired
    return point


def predict(s,states,target):
    current=states[-1];old=states[-2] if len(states)>1 else None
    amp=np.sqrt(target);amp0=np.sqrt(current['E1']);q0=current['q'];q=np.empty_like(q0)
    if old is None:
        for j in range(-s.h,s.h+1):q[:,s.h+j]=q0[:,s.h+j]*(amp/amp0)**(2 if j==0 else abs(j))
        a=current['a'];w=current['omega']
    else:
        ampold=np.sqrt(old['E1']);alpha=(target-current['E1'])/(current['E1']-old['E1'])
        for j in range(-s.h,s.h+1):
            power=2 if j==0 else abs(j)
            shape=q0[:,s.h+j]/amp0**power
            shape_old=old['q'][:,s.h+j]/ampold**power
            q[:,s.h+j]=amp**power*(shape+alpha*(shape-shape_old))
        a=current['a']+alpha*(current['a']-old['a']);w=current['omega']+alpha*(current['omega']-old['omega'])
    q[:,s.h+1]*=amp/common.mass_norm(s.p,q[:,s.h+1]);q[:,s.h-1]=q[:,s.h+1].conj()
    rotation=np.exp(-1j*np.angle(s.ell@q[:,s.h+1].ravel()))
    for j in range(-s.h,s.h+1):q[:,s.h+j]*=rotation**j
    return np.r_[pack(q),w,a]


def main():
    ap=argparse.ArgumentParser();ap.add_argument('--input',type=Path,required=True);ap.add_argument('--responses',type=Path,required=True)
    ap.add_argument('--out',type=Path,required=True);ap.add_argument('--budget',type=int,default=1200);ap.add_argument('--reserve-harmonic',type=int,default=300)
    ap.add_argument('--H',type=int,default=2);ap.add_argument('--harmonic-check-H',type=int,default=3)
    ap.add_argument('--amplitudes',type=float,nargs='+',default=[.0064,.009,.0128,.018,.0255,.036,.051,.072,.102])
    ap.add_argument('--H3-extension-amplitudes',type=float,nargs='*',default=[.144,.204])
    args=ap.parse_args();args.out=args.out.resolve();args.out.mkdir(parents=True,exist_ok=True)
    if (args.out/'completion.json').exists():raise RuntimeError('Refusing to overwrite a completed campaign')
    started=time.monotonic();deadline=started+args.budget;branch_deadline=deadline-args.reserve_harmonic
    def alarm(signum,frame):raise BudgetExpired('Total 1200-second numerical budget exhausted')
    signal.signal(signal.SIGALRM,alarm);signal.alarm(args.budget)
    result=dict(status='running',command=' '.join(sys.argv),points=[],failed_attempts=[],budget_seconds=args.budget,
        reserve_harmonic_seconds=args.reserve_harmonic,accepted_limit=14,failure_limit=6,continuation_coordinate='actual E1 velocity mass energy')
    manifest=dict(frozen=False,points=result['points'],field_axis_order='radius,signed_harmonic,variable_major_z',variable_order=['ur','ut','uz','p'])
    s=inv=None
    try:
        result['algebra_checks']=checks(args.out,branch_deadline)
        with np.load(args.responses) as d:data={k:d[k].copy() for k in d.files}
        with np.load(args.input) as d:seed={k:d[k].copy() for k in d.files}
        n,dr=int(data['n']),float(data['dr']);ac,wc=float(data['a_c']),float(data['omega_c']);qref=data['q1']
        seed_h=(seed['field'].shape[1]-1)//2;q=transfer(seed['field'],seed_h,args.H)
        p=common.make_problem(n,dr,ac,harmonics=args.H,nonlinear=True,omega=wc)
        seed_energy=common.mass_inner(p,q[:,args.H+1],q[:,args.H+1]).real
        s=System(p,qref,seed_energy,branch_deadline);left=data['left_mode'].ravel()
        d=np.vdot(left,common.sponge_start_matrix(p,1)@qref.ravel())/np.vdot(left,s.b[1]@qref.ravel())
        inv=BlockInverse(s,d);result['branch_preconditioner']=inv.info
        y=np.r_[pack(q),float(seed['omega']),float(seed['a'])]
        out=args.out/'point_000_H2_seed';y,rec=correct_point(s,inv,y,out,data['Cvec'])
        if not rec['accepted']:raise RuntimeError('Archived seed did not pass original-equation revalidation')
        result['points'].append(archive_point(s,y,rec,out,'point_000_H2_seed','seed'))
        states=[dict(q=unpack(y[:-2],s.nr,s.h,s.s),a=y[-1],omega=y[-2],E1=s.energy,id='point_000_H2_seed')]
        targets=list(args.amplitudes);attempt=0;failed=0
        while targets and len(result['points'])<13 and failed<6:
            if time.monotonic()>branch_deadline-20:result['branch_stop']='Reserved H3 budget reached';break
            target=targets.pop(0)
            if target<=np.sqrt(states[-1]['E1'])*(1+1e-8):continue
            s.set_energy(target**2);y=predict(s,states,s.energy);attempt+=1
            ident=f'point_{attempt:03d}_H{s.h}_A{target:.7f}';out=args.out/ident
            y,rec=correct_point(s,inv,y,out,data['Cvec'])
            print(json.dumps(dict(event='point',id=ident,accepted=rec['accepted'],amplitude=target,a=rec['a'],omega=rec['omega'],seconds=rec['seconds'],failure=rec.get('failure'))),flush=True)
            if not rec['accepted']:
                failed+=1;result['failed_attempts'].append(dict(id=ident,summary=str((out/'summary.json').relative_to(ROOT)),reason=rec.get('failure')))
                if rec.get('budget_stop'):result['branch_stop']='Reserved H3 budget reached';break
                mid=(np.sqrt(states[-1]['E1'])+target)/2
                if mid-np.sqrt(states[-1]['E1'])<1e-4:result['branch_stop']='Minimum amplitude retry step';break
                targets[:0]=[mid,target];continue
            point=archive_point(s,y,rec,out,ident,'continuation');result['points'].append(point)
            states.append(dict(q=unpack(y[:-2],s.nr,s.h,s.s),a=y[-1],omega=y[-2],E1=s.energy,id=ident))
            if len(states)>2:states=states[-2:]
            harmonic_fraction=point['harmonic_energy'][-1]/sum(point['harmonic_energy'][1:])
            if harmonic_fraction>.03:result['branch_stop']='Highest H2 harmonic energy exceeds 3%; prioritize H3';break
        last=states[-1];h2=q2=last['q'].copy();h2_a=last['a'];h2_w=last['omega'];h2_e=last['E1'];h2_id=last['id']
        result.setdefault('branch_stop','Requested amplitude sequence, accepted-point limit, or failure cap reached')
        del inv,s,p;inv=s=p=None;gc.collect()
        if time.monotonic()<deadline-30:
            h=args.harmonic_check_H;p=common.make_problem(n,dr,ac,harmonics=h,nonlinear=True,omega=wc)
            s=System(p,qref,h2_e,deadline);inv=BlockInverse(s,d);result['harmonic_preconditioner']=inv.info
            y=np.r_[pack(transfer(h2,args.H,h)),h2_w,h2_a];ident=f'harmonic_H{h}_{h2_id}'
            out=args.out/ident;y,rec=correct_point(s,inv,y,out,data['Cvec'],max_newton=6)
            if rec['accepted']:
                point=archive_point(s,y,rec,out,ident,'harmonic_check',h2_id);result['points'].append(point)
                qh=unpack(y[:-2],s.nr,s.h,s.s)
                compare=dict(H_low=args.H,H_high=h,paired_H2_id=h2_id,E1=h2_e,delta_a=float(y[-1]-h2_a),delta_omega=float(y[-2]-h2_w),
                    q1_mass_relative=common.mass_norm(p,qh[:,h+1]-h2[:,args.H+1])/np.sqrt(h2_e),
                    mean_mass_relative=common.mass_norm(p,qh[:,h]-h2[:,args.H])/max(common.mass_norm(p,h2[:,args.H]),1e-300),
                    high_harmonic_energy_fraction=point['harmonic_energy'][-1]/sum(point['harmonic_energy'][1:]),accepted=True)
                result['harmonic_check']=compare
                # Additional finite-amplitude work follows the H3 equations;
                # a successful H3 comparison never licenses further H2-only
                # amplification. The total campaign limits remain unchanged.
                h3_states=[dict(q=qh,a=y[-1],omega=y[-2],E1=s.energy,id=ident)]
                if np.sqrt(h2_e)>=.1:
                    extension_targets=list(args.H3_extension_amplitudes)
                    while extension_targets and len(result['points'])<14 and len(result['failed_attempts'])<6:
                        if time.monotonic()>deadline-45:break
                        amplitude=extension_targets.pop(0)
                        if amplitude<=np.sqrt(h3_states[-1]['E1']):continue
                        s.set_energy(amplitude**2);trial=predict(s,h3_states,s.energy)
                        extid=f'extension_{len(result["points"])+len(result["failed_attempts"]):03d}_H{h}_A{amplitude:.7f}';extout=args.out/extid
                        trial,extrec=correct_point(s,inv,trial,extout,data['Cvec'],max_newton=6)
                        print(json.dumps(dict(event='H3_extension',id=extid,accepted=extrec['accepted'],a=extrec['a'],omega=extrec['omega'],seconds=extrec['seconds'],failure=extrec.get('failure'))),flush=True)
                        if not extrec['accepted']:
                            result['failed_attempts'].append(dict(id=extid,summary=str((extout/'summary.json').relative_to(ROOT)),reason=extrec.get('failure')))
                            if extrec.get('budget_stop'):break
                            middle=(amplitude+np.sqrt(h3_states[-1]['E1']))/2
                            if middle-np.sqrt(h3_states[-1]['E1'])<1e-4:break
                            extension_targets[:0]=[middle,amplitude];continue
                        extpoint=archive_point(s,trial,extrec,extout,extid,'continuation')
                        result['points'].append(extpoint)
                        h3_states.append(dict(q=unpack(trial[:-2],s.nr,s.h,s.s),a=trial[-1],omega=trial[-2],E1=s.energy,id=extid))
                        h3_states=h3_states[-2:]
                        if extpoint['harmonic_energy'][-1]/sum(extpoint['harmonic_energy'][1:])>.03:
                            result['H3_extension_stop']='Highest H3 harmonic exceeds 3% energy; H4 needed before further amplification';break
            else:
                result['harmonic_check']=dict(accepted=False,failure=rec.get('failure'),summary=str((out/'summary.json').relative_to(ROOT)))
                result['failed_attempts'].append(dict(id=ident,summary=str((out/'summary.json').relative_to(ROOT)),reason=rec.get('failure')))
            print(json.dumps(dict(event='harmonic_check',**result['harmonic_check'])),flush=True)
        else:result['harmonic_check']=dict(accepted=False,failure='Insufficient remaining hard budget to start H3')
        result['status']='completed_bounded_campaign'
    except (BudgetExpired,RuntimeError,MemoryError,ValueError) as exc:
        result['status']='stopped_with_preserved_accepted_points';result['stop_reason']=type(exc).__name__+': '+str(exc)
    finally:
        signal.alarm(0);result.update(seconds=time.monotonic()-started,peak_rss_gib=rss())
        manifest['frozen']=True;manifest['status']=result['status'];manifest['seconds']=result['seconds']
        write(args.out/'manifest.json',manifest);write(args.out/'summary.json',result);write(args.out/'completion.json',result)
        print(json.dumps(dict(event='completion',status=result['status'],accepted=len(result['points']),failed=len(result['failed_attempts']),seconds=result['seconds'],peak_rss_gib=rss())),flush=True)


if __name__=='__main__':main()
