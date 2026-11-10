#!/usr/bin/env python3
"""Exploratory unforced relative equilibrium with an unknown REAL frequency.

No core solver changes.  Run with OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1.
The optional amplitude/growth continuation is auxiliary; only growth=0 is
the requested physical system.  Failed and trivial Newton roots are retained.
"""
from __future__ import annotations

import argparse
from dataclasses import asdict
import json
from pathlib import Path
import sys
import time

import numpy as np
from scipy import sparse
from scipy.interpolate import BarycentricInterpolator
from scipy.sparse.linalg import LinearOperator, eigs, splu

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from llsa.secondary.global_radial import GlobalRadial, OutletSponge
from llsa.secondary.primary import PrimaryWave
from llsa.secondary.radial import RadialConfig, pack_field, unpack_field


ROOT = Path(__file__).resolve().parents[1]
SOURCE = ROOT / "results/pier2003/cyl_wave_n40_h5.npz"


def dump(path, value):
    path.write_text(json.dumps(value, indent=2, allow_nan=False) + "\n")


def build(n=12, h=1, dr=2., rmin=450., rmax=580., sponge_start=550.):
    radii = np.linspace(rmin, rmax, round((rmax-rmin)/dr)+1)
    return GlobalRadial(radii, RadialConfig(ro=-1., beta=68., n=n,
        harmonics=h, omega=50.5), outlet_alpha=0j,
        outlet_sponge=OutletSponge(sponge_start, rmax-sponge_start, 10.))


def seed(p, factor=1., center=515., alpha=None, global_phase=0.):
    wave = PrimaryWave.load(SOURCE)
    H, m = p.H, p.ops[0].m
    raw = wave.modes.reshape(2*wave.harmonics+1, 4, wave.operators.m)
    # Mapping parameters are identical, so xi interpolation is exact mapping
    # correspondence and avoids interpolation over the singular physical z.
    raw = BarycentricInterpolator(wave.operators.grid.xi, raw, axis=2)(p.ops[0].grid.xi)
    profile = np.zeros((2*H+1, 4, m), complex)
    common = min(H, wave.harmonics)
    profile[H-common:H+common+1] = raw[wave.harmonics-common:wave.harmonics+common+1]
    envelope = .5*(1+np.tanh((p.radii-center)/8.))
    envelope *= np.sin(np.pi*np.clip((p.radii-p.radii[0])/25., 0., 1.)/2)**2
    envelope *= 1-.98*p.sponge_strength/p.outlet_sponge.strength
    carrier = (wave.alpha if alpha is None else alpha)*(p.radii-center)+global_phase
    q = np.empty((len(p.radii), 2*H+1, 4*m), complex)
    for j in range(-H,H+1):
        power = 2 if j==0 else abs(j)
        q[:,H+j] = ((factor*envelope)**power*np.exp(1j*j*carrier))[:,None]*profile[H+j].ravel()
    q[:,:,np.r_[0,m-1,m,2*m-1,2*m,3*m-1]] = 0
    q[0] = 0
    return q


class FreeFrequency:
    def __init__(self, p, reference, amplitude=None):
        self.p, self.reference = p, reference.copy()
        self.size = len(p.radii)*p.width
        self.inlet = np.zeros((2*p.H+1,p.s),complex)
        self.amplitude = amplitude
        # Real packing: [q0, Re(q1), Im(q1), ...].  Boundary rows are
        # explicitly masked, including inlet/outlet replacement rows.
        tr, tc, tv, diagonal = [], [], [], np.zeros(self.size)
        for i in range(1,len(p.radii)-1):
            for j in range(1,p.H+1):
                a = i*p.width+(2*j-1)*p.s+p.velocity_rows
                b = a+p.s
                tr.extend(a); tc.extend(b); tv.extend(np.full(len(a),j/p.radii[i]))
                tr.extend(b); tc.extend(a); tv.extend(np.full(len(a),-j/p.radii[i]))
                diagonal[a] = j/p.radii[i]
                diagonal[b] = j/p.radii[i]
        self.T = sparse.csc_matrix((tv,(tr,tc)),shape=(self.size,self.size))
        self.S = sparse.diags(diagonal,format='csc')
        self.weights = np.zeros(self.size)
        self.row = np.zeros(self.size)
        # Integral of Im(conj(qref_1) . q1), divided by reference energy.
        m, mass = p.ops[0].m, p.ops[0].grid.mass
        wr = np.gradient(p.radii)
        self.energy_blocks = []
        denom = 0.
        for i in range(1,len(p.radii)-1):
            v = reference[i,p.H+1,:3*m].reshape(3,m)
            mv = (v@mass).ravel()*wr[i]
            a = i*p.width+p.s+np.arange(3*m)
            b = a+p.s
            self.row[a],self.row[b] = -mv.imag,mv.real
            denom += wr[i]*np.einsum('vi,ij,vj->',v.conj(),mass,v).real
            self.energy_blocks.append((a,b,wr[i]))
        self.ref_energy = float(denom)
        self.row /= max(denom,1e-30)
        self.phase_index = None
        self.accepted_state = None

    def energy(self,x,gradient=False):
        m, mass = self.p.ops[0].m,self.p.ops[0].grid.mass
        value, row = 0.,np.zeros(self.size)
        for a,b,weight in self.energy_blocks:
            for indices in (a,b):
                v=x[indices].reshape(3,m)
                value += weight*np.einsum('vi,ij,vj->',v,mass,v)
                if gradient:
                    row[indices]=2*weight*(v@mass).ravel()
        return (float(value),row) if gradient else float(value)

    def initial_state(self,field,omega=50.5,growth=0.):
        z=np.r_[pack_field(field),omega]
        return z if self.amplitude is None else np.r_[z,growth]

    def physical_field(self,state):
        return unpack_field(state[:self.size],len(self.p.radii),self.p.H,self.p.s)

    def physical_residual(self,state):
        x=state[:self.size]
        return self.p.evaluate(x,self.inlet)+(state[self.size]-50.5)*(self.T@x)

    def evaluate(self,state,jacobian=False):
        x,omega=state[:self.size],state[self.size]
        result=self.p.evaluate(x,self.inlet,jacobian=jacobian)
        F,J=result if jacobian else (result,None)
        tx=self.T@x
        F=F+(omega-50.5)*tx
        if jacobian:
            J=J+(omega-50.5)*self.T
        phase=float(self.row@x)
        if self.amplitude is None:
            F=np.r_[F,phase]
            if not jacobian:
                return F
            return F,sparse.bmat([[J,sparse.csc_matrix(tx[:,None])],
                [sparse.csr_matrix(self.row[None,:]),None]],format='csc')
        growth=state[-1]
        sx=self.S@x
        F=F+growth*sx
        e,erow=self.energy(x,True)
        F=np.r_[F,phase,e/self.amplitude-1]
        if not jacobian:
            return F
        J=J+growth*self.S
        return F,sparse.bmat([[J,sparse.csc_matrix(tx[:,None]),sparse.csc_matrix(sx[:,None])],
            [sparse.csr_matrix(self.row[None,:]),None,None],
            [sparse.csr_matrix((erow/self.amplitude)[None,:]),None,None]],format='csc')


def diagnostics(a,z):
    p=a.p
    q=a.physical_field(z)
    m=p.ops[0].m
    vel=q[:,p.H+1:,:3*m].reshape(len(p.radii),p.H,3,m)
    e=p.radii[:,None]**2*np.einsum('rhvi,ij,rhvj->rh',vel.conj(),p.ops[0].grid.mass,vel).real
    return dict(omega=float(z[a.size]),growth=float(z[-1]) if a.amplitude is not None else 0.,
        omega_inertial=float(z[a.size]),omega_disk_signed=float(z[a.size]-68.),
        pattern_speed_over_disk=float(z[a.size]/68.),
        physical_residual_inf=float(np.max(abs(a.physical_residual(z)))),
        augmented_residual_inf=float(np.max(abs(a.evaluate(z)))),
        phase_residual=float(a.row@z[:a.size]),max_E1=float(e[:,0].max()),
        physical_region_max_E1=float(e[p.radii<p.outlet_sponge.start,0].max()),
        physical_region_max_E1_R=float(p.radii[np.flatnonzero(p.radii<p.outlet_sponge.start)[np.argmax(e[p.radii<p.outlet_sponge.start,0])]]),
        E1_integral=float(a.energy(z[:a.size])),max_harmonic_velocity=float(np.max(abs(vel))),
        mean_velocity_max=float(np.max(abs(q[:,p.H,:3*m]))))


def save_fields(path,a,z,extra):
    p=a.p
    q=a.physical_field(z)
    iz=int(np.argmin(abs(p.ops[0].grid.z-1.)))
    metadata=dict(extra)|dict(rmin=float(p.radii[0]),rmax=float(p.radii[-1]),
        dr=float(p.radii[1]-p.radii[0]),n=p.config.n,h=p.H,
        sponge=asdict(p.outlet_sponge),probe_z=float(p.ops[0].grid.z[iz]),
        omega_inertial=float(z[a.size]),omega_disk_signed=float(z[a.size]-68.),
        pattern_speed_over_disk=float(z[a.size]/68.),
        auxiliary=a.amplitude is not None,amplitude_constraint=a.amplitude,
        homogeneous_inlet=True,interior_forcing=None)
    np.savez_compressed(path,radii=p.radii,modes=q,omega=z[a.size],
        auxiliary_growth=z[-1] if a.amplitude is not None else 0.,
        config_json=json.dumps(asdict(p.config)|{'omega':float(z[a.size])}),
        phase_definition='Im integral conj(qref_1) dot q_1 dz dR / integral abs(qref_1)^2 dz dR = 0',
        reference=a.reference,metadata_json=json.dumps(metadata),source=str(SOURCE))
    m=p.ops[0].m
    v=q[:,p.H+1:,:3*m].reshape(len(p.radii),p.H,3,m)
    E=p.radii[:,None]**2*np.einsum('rhvi,ij,rhvj->rh',v.conj(),p.ops[0].grid.mass,v).real
    probe=q[:,p.H+1,m+iz]
    ph=np.unwrap(np.angle(probe))
    alpha=np.gradient(ph,p.radii)
    alpha[abs(probe)<1e-5*max(abs(probe))]=np.nan
    mean=q[:,p.H,:3*m].reshape(len(p.radii),3,m)
    vals=np.column_stack((p.radii,E,abs(probe),np.sqrt(2)*abs(probe),ph,alpha,np.max(abs(mean),axis=(1,2)),
        p.sponge_strength,p.radii<p.outlet_sponge.start))
    np.savetxt(path.with_suffix('.csv'),vals,delimiter=',',header=','.join(
        ['R']+[f'E{j}' for j in range(1,p.H+1)]+['abs_v1_over_R','A1_rms_over_disk_speed','phase_v1','local_alpha','max_mean_velocity_over_R','sponge','physical_region']),comments='')


def solve(a,z,out,name,max_iter=22,tol=1e-9,budget=180.):
    if (out/f'{name}.json').exists() or (out/f'{name}.npz').exists():
        raise FileExistsError(f'Refusing to overwrite attempt {out/name}')
    start=time.monotonic()
    history=[]
    initial_energy=a.energy(z[:a.size])
    status='max_iterations'
    field_scale=max(np.max(abs(z[:a.size])),1e-5)
    for it in range(max_iter+1):
        F=a.evaluate(z)
        info=diagnostics(a,z)
        info.update(iteration=it,seconds=time.monotonic()-start,
                    scaled_residual=float(np.max(abs(F[:a.size]))/field_scale))
        history.append(info)
        if (info['scaled_residual']<tol and np.max(abs(F[a.size:]))<tol):
            status='trivial' if info['E1_integral']<max(1e-10,initial_energy*1e-7) else 'converged'
            break
        if time.monotonic()-start>budget:
            status='time_budget';break
        if it==max_iter:
            break
        F,J=a.evaluate(z,True)
        scale=np.maximum(np.asarray(abs(J).max(axis=1).toarray()).ravel(),1e-20)
        try:
            lu=splu((sparse.diags(1/scale)@J).tocsc())
            dz=lu.solve(-F/scale)
        except RuntimeError as e:
            status='factorization_failure';info['error']=str(e);break
        if not np.all(np.isfinite(dz)):
            status='nonfinite_step';break
        info['linear_relative_residual']=float(np.linalg.norm(J@dz+F)/max(np.linalg.norm(F),1e-30))
        merit=np.linalg.norm(F/scale)
        accepted=False
        for ls in range(13):
            damping=2.**(-ls)
            trial=z+damping*dz
            if abs(trial[a.size]-50.5)>60:
                continue
            Ft=a.evaluate(trial)
            if np.linalg.norm(Ft/scale)<merit*(1-1e-4*damping):
                z=trial;accepted=True;info['damping']=damping;break
        if not accepted:
            status='line_search_failure';break
    info=diagnostics(a,z)
    classification=('auxiliary_converged' if a.amplitude is not None else 'accepted_nonzero') if status=='converged' else ('trivial' if status=='trivial' else 'failed')
    result=dict(name=name,status=status,seconds=time.monotonic()-start,
        classification=classification,tolerance=tol,field_scale=float(field_scale),
        final_scaled_residual=float(info['augmented_residual_inf']/field_scale),
        initial_energy=initial_energy,history=history,**info)
    dump(out/f'{name}.json',result)
    save_fields(out/f'{name}.npz',a,z,result|{'history':None})
    print(json.dumps({k:result[k] for k in ('name','status','seconds','omega','growth','max_E1','physical_residual_inf')}),flush=True)
    return z,result


def checks(out):
    p=build(n=8,h=2,dr=4.,rmin=480.,rmax=512.,sponge_start=500.)
    q=seed(p,factor=.7,center=490.)
    a=FreeFrequency(p,q)
    z=a.initial_state(q,omega=52.3)
    F,J=a.evaluate(z,True)
    rng=np.random.default_rng(23)
    d=rng.normal(size=len(z));d/=np.linalg.norm(d)
    eps=1e-6
    fd=(a.evaluate(z+eps*d)-a.evaluate(z-eps*d))/(2*eps)
    dw=np.zeros_like(z);dw[-1]=1
    fw=(a.evaluate(z+eps*dw)-a.evaluate(z-eps*dw))/(2*eps)
    p2=build(n=8,h=2,dr=4.,rmin=480.,rmax=512.,sponge_start=500.)
    # Independently rebuilt baseline at omega=51.5, so T must match L1-L0.
    p2.config=RadialConfig(ro=-1,beta=68,n=8,harmonics=2,omega=51.5)
    from llsa.secondary.radial import radial_matrices
    p2.coefficients=[[radial_matrices(op,j,51.5) for j in range(-2,3)] for op in p2.ops]
    delta=p2._linear_matrix(False)-p._linear_matrix(False)-a.T
    ar=FreeFrequency(p,q,amplitude=a.energy(z[:a.size]))
    zz=ar.initial_state(q,omega=52.3,growth=.2)
    _,JJ=ar.evaluate(zz,True)
    dd=rng.normal(size=len(zz));dd/=np.linalg.norm(dd)
    ff=(ar.evaluate(zz+eps*dd)-ar.evaluate(zz-eps*dd))/(2*eps)
    r=dict(full_jacobian_relative=float(np.linalg.norm(fd-J@d)/np.linalg.norm(fd)),
        frequency_column_relative=float(np.linalg.norm(fw-J@dw)/np.linalg.norm(fw)),
        phase_row_absolute=float(abs(fd[-1]-(J@d)[-1])),
        independent_frequency_matrix_max=float(abs(delta).max()),
        auxiliary_jacobian_relative=float(np.linalg.norm(ff-JJ@dd)/np.linalg.norm(ff)))
    dump(out/'augmentation_checks.json',r)
    if max(r.values())>1e-6:
        raise RuntimeError(f'Augmentation derivative checks failed: {r}')
    print('CHECKS '+json.dumps(r),flush=True)


def global_mode(p,reference,out,k=8,select=0):
    a=FreeFrequency(p,reference)
    # Positive complex harmonic from its Re-Re and Im-Re real blocks.
    idx=np.concatenate([i*p.width+p.s+np.arange(p.s) for i in range(len(p.radii))])
    im=idx+p.s
    L=p._linear_matrix(False)
    A=(L[idx][:,idx]+1j*L[im][:,idx]).tocsc()
    B=a.S[idx][:,idx].tocsc()
    lu=splu(A)
    K=LinearOperator(A.shape,matvec=lambda v:lu.solve(B@v),dtype=complex)
    values,vectors=eigs(K,k=k,which='LM',tol=2e-9,maxiter=1200)
    lam=1/values
    omega=50.5+lam.imag
    growth=-lam.real
    order=np.argsort(-growth)
    records=[]
    for kk in order:
        v=vectors[:,kk]
        records.append(dict(omega=float(omega[kk]),growth=float(growth[kk]),
            eigen_residual=float(np.linalg.norm(A@v-lam[kk]*(B@v))/np.linalg.norm(A@v))))
    dump(out/'linear_modes.json',records)
    np.savez_compressed(out/'linear_eigenpairs.npz',radii=p.radii,
        vectors=vectors[:,order],omega=omega[order],growth=growth[order])
    chosen=order[select]
    v=vectors[:,chosen].reshape(len(p.radii),p.s)
    q=np.zeros_like(reference)
    q[:,p.H+1]=v
    q[:,p.H-1]=v.conj()
    q*=.003/max(abs(q[:,:,:3*p.ops[0].m]).ravel())
    print('LINEAR '+json.dumps(records),flush=True)
    return q,float(omega[chosen]),float(growth[chosen])


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output',type=Path,required=True)
    parser.add_argument('--stage',choices=['direct','linear','amplitude','restart'],default='direct')
    parser.add_argument('--n',type=int,default=12)
    parser.add_argument('--h',type=int,default=1)
    parser.add_argument('--dr',type=float,default=2.)
    parser.add_argument('--rmin',type=float,default=450.)
    parser.add_argument('--rmax',type=float,default=580.)
    parser.add_argument('--sponge-start',type=float,default=550.)
    parser.add_argument('--restart',type=Path)
    parser.add_argument('--max-iter',type=int,default=22)
    parser.add_argument('--budget',type=float,default=180.)
    parser.add_argument('--mode-index',type=int,default=0)
    parser.add_argument('--tol',type=float,default=1e-7)
    parser.add_argument('--aux-tol',type=float,default=1e-6)
    parser.add_argument('--amplitudes',type=float,nargs='+',default=[.003,.009,.018,.03,.05,.065,.08,.10])
    parser.add_argument('--free-growth-trigger',type=float,default=.15)
    parser.add_argument('--seed-multiplier',type=float,default=1.)
    parser.add_argument('--seed-phase',type=float,default=0.)
    parser.add_argument('--omega-offset',type=float,default=0.)
    args=parser.parse_args()
    out=args.output
    out.mkdir(parents=True,exist_ok=False)
    begin=time.monotonic()
    checks(out)
    p=build(args.n,args.h,args.dr,args.rmin,args.rmax,args.sponge_start)
    q=seed(p)
    dump(out/'problem.json',dict(config=asdict(p.config),radii=p.radii.tolist(),
        sponge=asdict(p.outlet_sponge),outlet_alpha=[0.,0.],homogeneous_inlet=True,
        forcing=None,phase='global reference first harmonic velocity inner product',
        source=str(SOURCE),arguments=vars(args)|{'output':str(out),'restart':str(args.restart)}))
    attempts=[]
    if args.stage=='direct':
        for factor,center,alpha in [(1.,510.,None),(.4,500.,.22),(1.4,520.,.28)]:
            q=seed(p,factor,center,alpha)
            a=FreeFrequency(p,q)
            z,r=solve(a,a.initial_state(q),out,f'direct_{len(attempts)}',args.max_iter,tol=args.tol,budget=args.budget)
            attempts.append(r)
            if r['status']=='converged':
                save_fields(out/'fields.npz',a,z,r|{'history':None})
                break
    elif args.stage in ('linear','amplitude'):
        q,omega,growth=global_mode(p,q,out,select=args.mode_index)
        a=FreeFrequency(p,q)
        save_fields(out/'linear_seed.npz',a,a.initial_state(q,omega),{'linear_growth':growth})
        if args.stage=='amplitude':
            for target_max in args.amplitudes:
                current_max=max(abs(q[:,p.H+1,:3*p.ops[0].m]).ravel())
                factor=target_max/current_max
                for j in range(1,p.H+1):
                    q[:,p.H+j]*=factor**j
                q[:,p.H]*=factor**2
                q[:,:p.H]=q[:,p.H+1:][:,::-1].conj()
                aa=FreeFrequency(p,q)
                a=FreeFrequency(p,q,amplitude=aa.energy(pack_field(q)))
                z,r=solve(a,a.initial_state(q,omega,growth),out,f'amplitude_{len(attempts)}',args.max_iter,tol=args.aux_tol,budget=args.budget)
                attempts.append(r)
                q,omega,growth=a.physical_field(z),float(z[a.size]),float(z[-1])
                if r['status']!='converged':
                    break
                if abs(growth)<args.free_growth_trigger or growth<0:
                    free=FreeFrequency(p,q)
                    zz,rr=solve(free,free.initial_state(q,omega),out,'physical_free',args.max_iter+10,tol=args.tol,budget=args.budget)
                    attempts.append(rr)
                    if rr['status']=='converged':
                        save_fields(out/'fields.npz',free,zz,rr|{'history':None})
                        break
    else:
        d=np.load(args.restart)
        qr=d['modes']
        oldn=(qr.shape[2]//4)-1
        oldh=(qr.shape[1]-1)//2
        from llsa.secondary.operators import PierOperators
        oldgrid=PierOperators(n=oldn).grid
        shaped=qr.reshape(len(qr),2*oldh+1,4,oldn+1)
        interp=BarycentricInterpolator(oldgrid.xi,shaped,axis=-1)(p.ops[0].grid.xi)
        q=np.zeros((len(p.radii),2*p.H+1,p.s),complex)
        common=min(oldh,p.H)
        for j in range(-common,common+1):
            block=interp[:,oldh+j].reshape(len(qr),p.s)
            for k in range(p.s):
                q[:,p.H+j,k]=np.interp(p.radii,d['radii'],block[:,k].real)+1j*np.interp(p.radii,d['radii'],block[:,k].imag)
        for j in range(-p.H,p.H+1):
            q[:,p.H+j]*=args.seed_multiplier**(2 if j==0 else abs(j))*np.exp(1j*j*args.seed_phase)
        a=FreeFrequency(p,q)
        z,r=solve(a,a.initial_state(q,float(d['omega'])+args.omega_offset),out,'restart_free',args.max_iter,tol=args.tol,budget=args.budget)
        attempts.append(r)
        if r['status']=='converged':
            save_fields(out/'fields.npz',a,z,r|{'history':None})
    dump(out/'summary.json',dict(seconds=time.monotonic()-begin,attempts=attempts))


if __name__=='__main__':
    main()
