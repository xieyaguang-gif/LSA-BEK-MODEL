#!/usr/bin/env python3
"""Moving-domain full BVP and genuine bordered pseudo-arclength continuation.

Ro=-0.751, beta=85 fork; legacy source and results remain read-only.
Only original homogeneous rotating-wave equations are evaluated. The sparse
template assembly is exact in powers of 1/R, and is verified independently.
"""
from __future__ import annotations
import os
for _key in ('OPENBLAS_NUM_THREADS','OMP_NUM_THREADS','MKL_NUM_THREADS','NUMEXPR_NUM_THREADS','VECLIB_MAXIMUM_THREADS'):
    os.environ[_key]='1'
import copy, gc, json, time
from pathlib import Path
import numpy as np
from scipy import sparse
from scipy.interpolate import BarycentricInterpolator, interp1d
from scipy.sparse.linalg import LinearOperator, gmres
import appelquist2015_hopf_common as common
from appelquist2015_hopf_block_corrector import Factor, mass_row, norm
from appelquist2016_branch_continuation import pack, unpack, constraints_row, BudgetExpired
from llsa.secondary.global_radial import GlobalRadial, radial_differences
from llsa.secondary.radial import RadialConfig, radial_matrices

ROOT=Path(__file__).resolve().parents[1]

def write(path,data):
    path=Path(path);path.parent.mkdir(parents=True,exist_ok=True)
    tmp=path.with_suffix(path.suffix+'.tmp');tmp.write_text(json.dumps(data,indent=2,ensure_ascii=False,allow_nan=False)+'\n');tmp.replace(path)

class Mapping:
    """R=400+L*xi. Fixed M, D_R=D_xi/L, D_RR=D_xixi/L²."""
    def __init__(self,n=32,H=3,intervals=730,ro=-.751,beta=85.):
        self.n,self.H,self.intervals=int(n),int(H),int(intervals)
        self.xi=np.linspace(0,1,intervals+1);self.nr=len(self.xi)
        self.config=RadialConfig(ro=ro,beta=beta,n=n,harmonics=H,omega=0.,nonlinear=True)
        self.op=self.config.operators(500.);self.s=self.op.size;self.m=self.op.m
        self.Dx,self.Dxx=radial_differences(400+320*self.xi);self.Dx*=320;self.Dxx*=320**2
        self.rows=np.concatenate([v*self.m+np.arange(1,self.m-1) for v in range(3)])
        ts=np.array([.8,1.,1.2]);va=np.stack([ts*0+1,ts,ts**3],axis=1)
        self.templates=[]
        for j in range(H+1):
            abc=[radial_matrices(self.config.operators(500/t),j,0.) for t in ts]
            aa=np.linalg.solve(va,np.stack([x[0] for x in abc]).reshape(3,-1)).reshape(3,self.s,self.s)
            bb=np.linalg.lstsq(np.stack([ts*0+1,ts**2],axis=1),np.stack([x[1] for x in abc]).reshape(3,-1),rcond=None)[0].reshape(2,self.s,self.s)
            cc=abc[1][2]
            # Numerical fitting roundoff must not generate dense sparse blocks.
            for arr in (aa,bb):
                arr[np.abs(arr)<2e-14*np.maximum(1,np.max(np.abs(arr),axis=0))[None]]=0
            self.templates.append(tuple(sparse.csr_matrix(a) for a in (*aa,*bb,cc)))
        mask=np.ones((self.nr,self.s))
        mask[0,self.rows]=0;mask[-1,self.rows]=0
        self.mask=sparse.diags(mask.ravel(),format='csr')
        bc=np.zeros_like(mask);bc[0,self.rows]=1;self.inlet=sparse.diags(bc.ravel(),format='csr')
        bc*=0;bc[-1,self.rows]=1;self.outlet=sparse.diags(bc.ravel(),format='csr')
        mass=np.zeros(self.s);mass[self.rows]=1;self.mass=sparse.diags(mass,format='csr')

    def at(self,Rout,omega,derivative=False):
        L=Rout-400.;r=400+L*self.xi;t=500/r;td=-t*self.xi/r
        D=self.Dx/L;D2=self.Dxx/L**2
        K=sparse.kron(D,sparse.eye(self.s),format='csr');K2=sparse.kron(D2,sparse.eye(self.s),format='csr')
        outer,ds=common.smooth_step_and_derivative((r-Rout+100)/40)
        inner,di=common.smooth_step_and_derivative((420-r)/10)
        damping=28*outer+.8*inner;dd=.7*ds*(self.xi-1)-.08*di*self.xi
        bdiag=np.zeros((self.nr,self.s));bdiag[1:-1,self.rows]=1/r[1:-1,None]
        blocks=[];derivatives=[]
        kron=lambda v,a:sparse.kron(sparse.diags(v),a,format='csr')
        for j,(a0,a1,a3,b0,b2,c1) in enumerate(self.templates):
            A=kron(t*0+1,a0)+kron(t,a1)+kron(t**3,a3)
            B=kron(t*0+1,b0)+kron(t*t,b2);C=kron(t,c1)
            if j:A+=kron((damping-1j*j*omega)/r,self.mass)
            mat=(self.mask@(A+B@K+C@K2)+self.inlet+self.outlet@K).tocsc();mat.eliminate_zeros();blocks.append(sparse.csc_matrix(mat.real.copy()) if j==0 else mat)
            if derivative:
                Ad=kron(td,a1)+kron(3*t*t*td,a3)
                if j:Ad+=kron(dd/r-(damping-1j*j*omega)*self.xi/r**2,self.mass)
                Bd=kron(2*t*td,b2);Cd=kron(td,c1)
                der=(self.mask@(Ad+(Bd-B/L)@K+(Cd-2*C/L)@K2)-self.outlet@K/L).tocsc()
                derivatives.append(sparse.csc_matrix(der.real.copy()) if j==0 else der)
        p=GlobalRadial.__new__(GlobalRadial);p.radii=r;p.config=RadialConfig(ro=self.config.ro,beta=self.config.beta,n=self.n,harmonics=self.H,omega=float(omega),nonlinear=True)
        p.ops=[]
        for R in r:
            op=copy.copy(self.op);op.radius=float(R);op.re=abs(self.config.ro)*R;p.ops.append(op)
        p.D,p.D2,p.H,p.s=D,D2,self.H,self.s;p.velocity_rows=self.rows
        p.outlet_sponge=common.HopfSponge(Rout-100,40.,28.);p.sponge_strength=damping
        return p,blocks,bdiag,derivatives

    def convection_R(self,p,q):
        H,m=self.H,self.m;L=p.radii[-1]-400
        qq=q.reshape(self.nr,2*H+1,4,m);qr=-(p.D@q.reshape(self.nr,-1)).reshape(qq.shape)/L
        drinv=(-self.xi/p.radii**2)[:,None,None];out=np.zeros_like(qq)
        for j in range(-H,H+1):
            for k in range(max(-H,j-H),min(H,j+H)+1):
                a,b=qq[:,j-k+H],qq[:,k+H]
                out[:,j+H,:3]+=a[:,0,None]*(qr[:,k+H,:3]+b[:,:3]*drinv)+1j*k*self.config.beta*drinv*a[:,1,None]*b[:,:3]
                out[:,j+H,0]-=a[:,1]*b[:,1]*drinv[:,0]
                out[:,j+H,1]+=a[:,0]*b[:,1]*drinv[:,0]
        out=out.reshape(q.shape);out[:,:,self.op.boundary_rows]=0;out[0][:,self.rows]=0;out[-1][:,self.rows]=0
        return out

def linear(blocks,q,H):
    out=np.zeros_like(q)
    for j,A in enumerate(blocks):
        out[:,H+j]=(A@q[:,H+j].ravel()).reshape(q.shape[0],q.shape[2])
        if j:out[:,H-j]=out[:,H+j].conj()
    return out

def convection(p,q):return p._convection_field(q,(p.D@q.reshape(len(q),-1)).reshape(q.shape))

def energies(p,q):return np.array([common.mass_inner(p,q[:,p.H+j],q[:,p.H+j]).real for j in range(p.H+1)])

class Metric:
    def __init__(self,mapping,reference_R=764.7635065091705,field_scale=.1,omega_scale=1.,R_scale=10.):
        p,*_=mapping.at(reference_R,80.31112435889007);self.p=p;self.H=p.H;self.nr=len(p.radii);self.s=p.s
        self.fs,self.ws,self.rs=field_scale,omega_scale,R_scale
    def row(self,v):
        q=unpack(v[:-2],self.nr,self.H,self.s);out=np.zeros((self.nr,2*self.H+1,self.s))
        for j in range(self.H+1):
            a=mass_row(self.p,q[:,self.H+j]).reshape(self.nr,self.s)/self.fs**2
            if j==0:out[:,0]=a.real
            else:out[:,2*j-1]=2*a.real;out[:,2*j]=-2*a.imag
        return np.r_[out.ravel(),v[-2]/self.ws**2,v[-1]/self.rs**2]
    def dot(self,u,v):return float(self.row(u)@v)
    def tangent(self,old,new,previous=None):
        t=new-old;t/=np.sqrt(self.dot(t,t))
        if previous is not None and self.dot(t,previous)<0:t=-t
        return t

class Corrector:
    def __init__(self,mapping,qref,y,mode,target,budget=1200.,metric=None,tangent=None):
        self.map=mapping;self.h=mapping.H;self.nr=mapping.nr;self.s=mapping.s;self.mode=mode;self.target=target
        self.deadline=time.monotonic()+budget;self.metric=metric;self.tangent=tangent;self.center=y.copy();self.jv_calls=0
        self.p,self.blocks,self.bdiag,_=mapping.at(y[-1],y[-2]);self.R=y[-1];self.w=y[-2]
        self.amplitude=max(np.sqrt(energies(self.p,unpack(y[:-2],self.nr,self.h,self.s))[1]),1e-5)
        self.ell=mass_row(self.p,qref)/common.mass_inner(self.p,qref,qref).real
        self.phase_row=constraints_row(self.p,self.ell,True)/self.amplitude
        scales=[np.maximum(np.asarray(abs(A).max(axis=1).toarray()).ravel(),1e-20).reshape(self.nr,self.s) for A in self.blocks]
        self.fscale=self.amplitude*np.stack([scales[0]]+[a for a in scales[1:] for _ in range(2)],axis=1).ravel()
        self.arcrow=metric.row(tangent) if mode=='arc' else None
    def check(self):
        if time.monotonic()>self.deadline:raise BudgetExpired('per-point runtime budget')
    def update(self,y,derivative=False):
        self.check()
        if derivative or y[-1]!=self.R or y[-2]!=self.w:
            self.p,self.blocks,self.bdiag,self.dblocks=self.map.at(y[-1],y[-2],derivative);self.R,self.w=y[-1],y[-2]
    def residual(self,y,scaled=True):
        self.update(y);q=unpack(y[:-2],self.nr,self.h,self.s);raw=pack(linear(self.blocks,q,self.h)+convection(self.p,q))
        phase=float(self.phase_row@y[:-2])
        if self.mode=='energy':con=energies(self.p,q)[1]/self.target**2-1
        elif self.mode=='Rout':con=(y[-1]-self.target)/10
        else:con=self.arcrow@(y-self.center)
        return np.r_[raw/self.fscale if scaled else raw,phase,con]
    def jacobian(self,y):
        self.update(y,True);p=self.p;blocks=self.blocks;q=unpack(y[:-2],self.nr,self.h,self.s);qn=norm(q)
        fw=np.zeros_like(q)
        for j in range(1,self.h+1):fw[:,self.h+j]=-1j*j*self.bdiag*q[:,self.h+j];fw[:,self.h-j]=fw[:,self.h+j].conj()
        fr=linear(self.dblocks,q,self.h)+self.map.convection_R(p,q)
        cols=np.stack([pack(fw),pack(fr)],axis=1)
        rows=np.zeros((2,len(y)));rows[0,:-2]=self.phase_row
        if self.mode=='energy':
            rows[1,:-2]=constraints_row(p,2*mass_row(p,q[:,self.h+1])/self.target**2)
            rows[1,-1]=energies(p,q)[1]/self.target**2/(y[-1]-400)
        elif self.mode=='Rout':rows[1,-1]=.1
        else:rows[1]=self.arcrow
        def mv(v):
            self.check();self.jv_calls+=1;dq=unpack(v[:-2],self.nr,self.h,self.s);vn=norm(dq)
            if vn:
                step=max(qn,1e-30)/vn;dn=(convection(p,q+step*dq)-convection(p,q-step*dq))/(2*step)
            else:dn=np.zeros_like(q)
            return np.r_[(pack(linear(blocks,dq,self.h)+dn)+cols@v[-2:])/self.fscale,rows@v]
        return LinearOperator((len(y),len(y)),matvec=mv,dtype=float),cols,rows
    def preconditioner(self,cols,rows):
        self.check();factors=[]
        for j,A in enumerate(self.blocks):
            shift=.02*sparse.diags(self.bdiag.ravel()) if j==1 else 0
            factors.append(Factor(A+shift,f'H{j}'));self.check()
        def solve_fields(rhs):
            q=unpack(rhs,self.nr,self.h,self.s);v=np.zeros_like(q)
            for j,f in enumerate(factors):
                z=q[:,self.h+j].ravel();v[:,self.h+j]=f.solve(z.real if j==0 else z).reshape(self.nr,self.s)
                if j:v[:,self.h-j]=v[:,self.h+j].conj()
            return pack(v)
        Z=np.stack([solve_fields(cols[:,k]) for k in range(2)],axis=1)
        schur=rows[:,-2:]-rows[:,:-2]@Z
        def mv(rhs):
            self.check();x=solve_fields(rhs[:-2]*self.fscale);c=np.linalg.solve(schur,rhs[-2:]-rows[:,:-2]@x)
            return np.r_[x-Z@c,c]
        return LinearOperator((len(self.center),len(self.center)),matvec=mv,dtype=float),dict(factors=[f.info for f in factors],schur=schur.tolist(),condition=float(np.linalg.cond(schur)))
    def metrics(self,y):
        f=self.residual(y,False);q=unpack(y[:-2],self.nr,self.h,self.s);nn=convection(self.p,q);e=energies(self.p,q)
        rel=float(np.max(abs(f[:-2]))/max(np.max(abs(y[:-2])),1e-30));feedback=max(norm(nn[:,self.h+1]),1e-30)
        ratio=norm(f[:-2])/feedback;cons=float(max(abs(f[-2:])))
        return dict(Rout=float(y[-1]),a=float(y[-1]-100),omega=float(y[-2]),A1=float(np.sqrt(e[1])),energies=e.tolist(),highest_to_fundamental=float(e[-1]/e[1]),residual_inf=float(max(abs(f[:-2]))),residual_l2=norm(f[:-2]),relative_field_inf=rel,residual_over_fundamental_feedback=ratio,phase=float(f[-2]),branch_constraint=float(f[-1]),accepted=bool(rel<1e-7 and ratio<1e-3 and cons<1e-8))

def correct(mapping,qref,y,mode,target,out,budget=1200.,metric=None,tangent=None,max_newton=9):
    out=Path(out);out.mkdir(parents=True,exist_ok=False);start=time.monotonic();record=dict(accepted=False,mode=mode,target=float(target) if np.isscalar(target) else None,budget_seconds=budget,iterations=[])
    s=None
    try:
        s=Corrector(mapping,qref,y,mode,target,budget,metric,tangent)
        for k in range(max_newton+1):
            row=s.metrics(y);row['iteration']=k;record['iterations'].append(row)
            write(out/'progress.json',record)
            if row['accepted']:record.update(row);break
            if k==max_newton:record['failure']='Newton iteration limit';break
            f=s.residual(y);J,cols,rows=s.jacobian(y);P,pre=s.preconditioner(cols,rows);row['preconditioner']=pre
            history=[];op=LinearOperator(J.shape,matvec=lambda v:J@(P@v),dtype=float)
            z,info=gmres(op,-f,rtol=2e-7,atol=0,restart=45,maxiter=4,callback=lambda r:history.append(float(r)),callback_type='pr_norm')
            step=P@z;lin=norm(J@step+f)/norm(f);row['linear']=dict(info=int(info),history=history,relative_original=lin)
            if lin>.1:record['failure']='Krylov original linear residual';break
            success=False
            for back in range(9):
                trial=y+2.**(-back)*step
                if not np.isfinite(trial).all() or not 600<trial[-1]<880:continue
                if norm(s.residual(trial))<(1-1e-4*2.**(-back))*norm(f):
                    y=trial;success=True;row['step_length']=2.**(-back);break
            if not success:record['failure']='Newton line search';break
            del J,P,op,cols,rows;gc.collect()
        if record['accepted']:
            # Independent physical-grid original local coefficients (no templates).
            check=independent(mapping,y);record['independent']=check
            if check['relative_field_inf']>=1e-7 or check['template_difference_relative']>1e-10:record.update(accepted=False,failure='Independent unforced physical equations')
    except Exception as exc:
        record.update(failure=f'{type(exc).__name__}: {exc}',budget_stop=isinstance(exc,BudgetExpired))
    record.update(seconds=time.monotonic()-start,Rout=float(y[-1]),omega=float(y[-2]),n=mapping.n,H=mapping.H,intervals=mapping.intervals,jv_calls=0 if s is None else s.jv_calls)
    p,*_=mapping.at(y[-1],y[-2]);q=unpack(y[:-2],mapping.nr,mapping.H,mapping.s)
    np.savez_compressed(out/('fields.npz' if record['accepted'] else 'checkpoint.npz'),field=q,radii=p.radii,xi=p.ops[0].grid.xi,z=p.ops[0].grid.z,mass=p.ops[0].grid.mass,H=mapping.H,n=mapping.n,omega=y[-2],Rout=y[-1],a=y[-1]-100,accepted=record['accepted'],ro=mapping.config.ro,beta=mapping.config.beta,qref=qref,tangent=np.array([]) if tangent is None else tangent)
    write(out/'summary.json',record);return y,record

def independent(mapping,y):
    p,blocks,_,_=mapping.at(y[-1],y[-2]);q=unpack(y[:-2],mapping.nr,mapping.H,mapping.s)
    D,D2=radial_differences(p.radii);qr=(D@q.reshape(mapping.nr,-1)).reshape(q.shape);qrr=(D2@q.reshape(mapping.nr,-1)).reshape(q.shape)
    out=np.zeros_like(q)
    for i,R in enumerate(p.radii):
        op=p.config.operators(R)
        for j in range(mapping.H+1):
            A,B,C=radial_matrices(op,j,y[-2]);v=A@q[i,mapping.H+j]+B@qr[i,mapping.H+j]+C@qrr[i,mapping.H+j]
            if j:v[mapping.rows]+=p.sponge_strength[i]/R*q[i,mapping.H+j,mapping.rows]
            if i==0:v[mapping.rows]=q[i,mapping.H+j,mapping.rows]
            elif i==mapping.nr-1:v[mapping.rows]=qr[i,mapping.H+j,mapping.rows]
            out[i,mapping.H+j]=v
            if j:out[i,mapping.H-j]=v.conj()
    out+=p._convection_field(q,qr);raw=pack(out);ours=pack(linear(blocks,q,mapping.H)+convection(p,q))
    values=out.reshape(mapping.nr,2*mapping.H+1,4,mapping.m)
    return dict(residual_inf=float(max(abs(raw))),relative_field_inf=float(max(abs(raw))/max(abs(y[:-2]))),template_difference_relative=norm(raw-ours)/max(norm(pack(linear(blocks,q,mapping.H))),norm(pack(convection(p,q))),1e-30),velocity_inf=float(np.max(abs(values[:,:,:3,1:-1]))),continuity_inf=float(np.max(abs(values[1:-1,:,3,1:-1]))),wall_far_boundary_inf=float(np.max(abs(out[:,:,mapping.op.boundary_rows]))),radial_boundary_inf=float(np.max(abs(out[[0,-1]][:,:,mapping.rows]))),forcing=False,physical_grid_reassembled=True)

def transfer(path,mapping,Rout=None):
    d=np.load(path)
    if float(d['ro'])!=mapping.config.ro or float(d['beta'])!=mapping.config.beta:
        raise ValueError('Restart Ro/beta mismatch')
    q=d['field'];oldh=int(d['H']);r=d['radii'];Rout=float(r[-1]) if Rout is None else Rout
    pp,*_=mapping.at(Rout,float(d['omega']));q=q.reshape(len(r),2*oldh+1,4,len(d['xi']))
    if len(d['xi'])!=mapping.m:q=BarycentricInterpolator(d['xi'],q,axis=-1)(mapping.op.grid.xi)
    q=interp1d(r,q,axis=0,kind='cubic',bounds_error=False,fill_value='extrapolate')(pp.radii).reshape(mapping.nr,2*oldh+1,mapping.s)
    new=np.zeros((mapping.nr,2*mapping.H+1,mapping.s),complex);h=min(oldh,mapping.H);new[:,mapping.H-h:mapping.H+h+1]=q[:,oldh-h:oldh+h+1]
    return np.r_[pack(new),float(d['omega']),Rout]

def checks(out):
    rng=np.random.default_rng(20261030);mapping=Mapping(8,3,80);R,w=764.7635065091705,80.31112435889007;p,blocks,bdiag,dblocks=mapping.at(R,w,True)
    q=unpack(rng.normal(size=mapping.nr*7*mapping.s)*1e-3,mapping.nr,3,mapping.s);y=np.r_[pack(q),w,R]
    original=GlobalRadial(p.radii,RadialConfig(ro=-.751,beta=85,n=8,harmonics=3,omega=w,nonlinear=True),outlet_alpha=0j,outlet_sponge=common.HopfSponge(R-100,40,28))
    raw=pack(linear(blocks,q,3)+convection(p,q));reference=original.evaluate(pack(q),np.zeros_like(q[0]))
    derivative=pack(linear(dblocks,q,3)+mapping.convection_R(p,q));fd=[]
    for h in (.01,.005):
        pp,bb,*_=mapping.at(R+h,w);pm,bm,*_=mapping.at(R-h,w)
        df=(pack(linear(bb,q,3)+convection(pp,q))-pack(linear(bm,q,3)+convection(pm,q)))/(2*h)
        fd.append(norm(df-derivative)/norm(derivative))
    qref=q[:,4]/common.mass_norm(p,q[:,4]);metric=Metric(mapping);t=rng.normal(size=len(y));t/=np.sqrt(metric.dot(t,t))
    cs=Corrector(mapping,qref,y,'energy',np.sqrt(energies(p,q)[1]),metric=metric)
    J,*_=cs.jacobian(y);v=rng.normal(size=len(y));v[:-2]*=1e-3;h=1e-5
    jrel=norm((cs.residual(y+h*v)-cs.residual(y-h*v))/(2*h)-J@v)/norm(J@v)
    cs2=Corrector(mapping,qref,y,'arc',0,metric=metric,tangent=t);J2,*_=cs2.jacobian(y)
    j2rel=norm((cs2.residual(y+h*v)-cs2.residual(y-h*v))/(2*h)-J2@v)/norm(J2@v)
    result=dict(original_residual_relative=norm(raw-reference)/norm(reference),Rout_derivative_relative=fd,energy_bordered_jv_relative=jrel,arclength_bordered_jv_relative=j2rel,metric_tangent_norm=metric.dot(t,t),derivative_scaling_D=float(abs((p.D-original.D).data).max()),physical_measure=float(common.radial_weights(p.radii).sum()),expected_measure=R-400,inner_transition_nodes=p.radii[(p.radii>410)&(p.radii<420)].tolist(),analytic_derivative_includes=['geometry','D_R','D_RR','inner_sponge','outer_sponge','outlet_BC','nonlinear_advection','energy_measure'],arclength='weighted secant predictor with fixed-metric hyperplane; field q velocity norm scale .1, omega scale1, Rout scale10, Rref764.7635065091705')
    result['passed']=bool(result['original_residual_relative']<1e-11 and max(fd)<1e-6 and max(jrel,j2rel)<1e-7)
    write(Path(out)/'mapping_checks.json',result)
    if not result['passed']:raise RuntimeError(result)
    return result


def request_worker(path):
    """One isolated corrector; the campaign enforces a hard wall-clock timeout."""
    import resource
    resource.setrlimit(resource.RLIMIT_AS, (200*1024**3, 200*1024**3))
    request = json.loads(Path(path).read_text())
    mapping = Mapping(**request['mapping'])
    with np.load(request['input']) as data:
        y, qref = data['y'].copy(), data['qref'].copy()
        tangent = data['tangent'].copy() if data['tangent'].size else None
    p, blocks, *_ = mapping.at(y[-1], y[-2])
    q = unpack(y[:-2], mapping.nr, mapping.H, mapping.s)
    if request.get('rescale_energy'):
        scale = request['target']/np.sqrt(energies(p, q)[1])
        for j in range(mapping.H+1):
            q[:, mapping.H+j] *= scale**(2 if j == 0 else j)
            if j:
                q[:, mapping.H-j] = q[:, mapping.H+j].conj()
    if request.get('weak_predictor'):
        from scipy.sparse.linalg import splu
        # Mean and 2 beta are the quadratic response, never auxiliary forcing.
        quadratic = convection(p, q)
        for j in (0, 2):
            rhs = -quadratic[:, mapping.H+j].ravel()
            factor = splu(blocks[j])
            q[:, mapping.H+j] = factor.solve(rhs.real if j == 0 else rhs).reshape(mapping.nr, mapping.s)
            if j:
                q[:, mapping.H-j] = q[:, mapping.H+j].conj()
        if mapping.H >= 3:
            rhs = -convection(p, q)[:, mapping.H+3].ravel()
            q[:, mapping.H+3] = splu(blocks[3]).solve(rhs).reshape(mapping.nr, mapping.s)
            q[:, mapping.H-3] = q[:, mapping.H+3].conj()
    y[:-2] = pack(q)
    metric = Metric(mapping) if request['mode'] == 'arc' else None
    y, result = correct(mapping, qref, y, request['mode'], request['target'],
                        request['output'], request['budget'], metric, tangent)
    result.update(ro=mapping.config.ro, beta=mapping.config.beta,
                  request_sha256=request['request_sha256'])
    write(Path(request['output'])/'summary.json', result)


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument('--check')
    group.add_argument('--request')
    args = parser.parse_args()
    if args.check:
        print(json.dumps(checks(args.check)))
    else:
        request_worker(args.request)

