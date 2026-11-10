"""Explicit, experimental frozen-DNS-base interface to the LLSA Floquet core.

DNSFrozenBase is not PrimaryWave: no nonlinear-equilibrium residual is invented
or waived. The model freezes radius and represents primary radial derivatives
by i*n*alpha, retaining the existing laminar self-similar terms. Eigenvalues
describe this specified mean-flow linearization, not an exact DNS equilibrium.
"""
from dataclasses import dataclass,field
from pathlib import Path
import numpy as np
from scipy import sparse
from scipy.interpolate import BarycentricInterpolator
from llsa.secondary.operators import PierOperators
from llsa.secondary.discretization import rotation_rates,sample_inertial_base
from llsa.secondary.time_domain import time_grid


def native_p6_interpolate(z,values,query):
    z=np.asarray(z);query=np.asarray(query)
    if (len(z)-1)%6 or values.shape[-1]!=len(z) or np.any(np.diff(z)<=0):
        raise ValueError('Require complete, increasing P6 native height grid')
    if np.min(query)<z[0]-1e-12 or np.max(query)>z[-1]+1e-12:
        raise ValueError('DNS extrapolation is forbidden; request a covered height')
    query=np.clip(query,z[0],z[-1]);out=np.empty(values.shape[:-1]+query.shape,dtype=values.dtype)
    elem=np.clip(np.searchsorted(z[::6],query,side='right')-1,0,(len(z)-1)//6-1)
    for e in np.unique(elem):
        sl=slice(6*e,6*e+7);sel=elem==e
        out[...,sel]=BarycentricInterpolator(z[sl],values[...,sl],axis=-1)(query[sel])
    return out


class FinitePierOperators(PierOperators):
    """Same cylindrical algebra on a finite rational Chebyshev height grid."""
    def __init__(self,*,radius,beta,ro,n,zmax,scale=2.):
        super().__init__(radius=radius,beta=beta,ro=ro,n=n)
        self.grid=time_grid(n=n,zmax=zmax,scale=scale)
        self.base=sample_inertial_base(self.grid.z,ro)


@dataclass
class DNSFrozenBase:
    operators: FinitePierOperators
    alpha: float
    omega: float
    modes: np.ndarray
    metadata: dict=field(default_factory=dict)

    @property
    def harmonics(self):return (len(self.modes)-1)//2

    @classmethod
    def from_file(cls,path,*,Re,alpha,n=48,harmonics=8,zmax_disk=35.,ro=-.751,mean_block=None):
        if not np.isfinite(alpha) or not np.isreal(alpha):raise ValueError('Real finite carrier alpha required')
        if int(harmonics)!=harmonics or not 1<=harmonics<=16:raise ValueError('Require H=1..16')
        with np.load(path) as data:
            i=int(np.argmin(abs(data['Re']-Re)))
            if abs(data['Re'][i]-Re)>1e-5:raise ValueError('Requested Re is not a native saved section')
            disk,external=rotation_rates(ro);scale=1/np.sqrt(disk)
            if zmax_disk>data['z'][-1]+1e-10:raise ValueError('Truncation exceeds saved DNS height')
            R=float(data['r'][i])*scale
            op=FinitePierOperators(radius=R,beta=float(data['beta']),ro=ro,n=n,zmax=zmax_disk*scale)
            source=(data['mean'][i] if mean_block is None else data['block_means'][int(mean_block),i])
            raw=native_p6_interpolate(data['z']*scale,source[:3,:harmonics+1],op.grid.z)
            total=disk*raw[[1,2,0]];total[1,0]+=disk
            base=np.stack([op.base[0],op.base[1],op.base[2]/R])
            correction=total.copy();correction[:,0]-=base
            if np.max(abs(correction[:,0].imag))>1e-12:raise ValueError('Mean Fourier mode must be real')
            modes=np.zeros((2*harmonics+1,4,op.m),complex)
            modes[harmonics:,:3]=correction.transpose(1,0,2)
            modes[:harmonics]=modes[harmonics+1:][::-1].conj()
            assert np.isfinite(modes).all()
            metadata={'schema':'DNSFrozenBase-v1','source':str(Path(path).resolve()),'Re_label':float(data['Re'][i]),
                'Re_exact':abs(ro)*R,'zmax_disk':zmax_disk,'mean_block':mean_block,
                'time_range':data['time'][[0,-1]].tolist(),'equilibrium_verified':False,
                'primary_radial_derivative':'i*n*alpha; laminar self-similar terms retained',
                'pressure':'not used in velocity linearization; padded pressure channels are not a reconstructed base pressure',
                'normalization':'inertial BEK velocity / R; base is laminar plus all retained DNS mean harmonics'}
        return cls(op,float(alpha),float(op.beta*disk),modes.reshape(2*harmonics+1,op.size),metadata)

    def continuity_defect(self):
        op=self.operators;H=self.harmonics;v=self.modes.reshape(2*H+1,4,op.m)[:,:3]
        j=np.arange(-H,H+1);u=v[:,0];w=v[:,2]
        terms=np.array([(1j*j[:,None]*self.alpha+1/op.radius)*u,
                        1j*j[:,None]*op.beta/op.radius*v[:,1],w@op.grid.D.T])
        # Interior diagnostic; this deliberately does not report a DNS divergence error.
        den=np.sum(np.linalg.norm(terms[:,:,1:-1],axis=2),axis=0)
        rel=np.linalg.norm(terms.sum(axis=0)[:,1:-1],axis=1)/np.maximum(den,1e-30)
        return {'harmonics':j.tolist(),'relative_interior_frozen_defect':rel.tolist(),
                'interpretation':'local frozen correction consistency, not DNS incompressibility error'}


class DNSFloquetProblem:
    """Frozen DNS-base pencil compatible with secondary_spectrum().

    Boundary choices are explicit. Wall continuity replaces pressure closure.
    top='decay': all perturbation velocities and p zero at the finite top.
    top='dns': u_r=u_theta=p=0 and d_z u_h=0, for the actual DNS core boundary.
    """
    def __init__(self,base,*,beta_hat=0.,harmonics=6,top='decay'):
        if not isinstance(base,DNSFrozenBase):raise TypeError('Use the explicit DNSFrozenBase type')
        if int(harmonics)!=harmonics or harmonics<1 or not np.isfinite(beta_hat):raise ValueError('Invalid Floquet controls')
        if top not in ('decay','dns'):raise ValueError('Unknown top boundary')
        self.wave=base;self.base=base;self.radius=base.operators.radius
        self.K=int(harmonics);self.beta_hat=float(beta_hat);self.top=top;self.wall_condition='continuity'
        self.B=sparse.block_diag([1j*base.operators.M]*(2*self.K+1),format='csc')
        self._cached_alpha=None;self._cached_matrix=None

    def matrix(self,alpha_hat):
        if not np.isfinite(alpha_hat):raise ValueError('Finite alpha_hat required')
        if alpha_hat==self._cached_alpha:return self._cached_matrix
        op,H=self.base.operators,self.base.harmonics;rows=[]
        for j in range(-self.K,self.K+1):
            row=[]
            for k in range(-self.K,self.K+1):
                l=j-k
                if abs(l)>H and j!=k:row.append(None);continue
                kr=alpha_hat+k*self.base.alpha;kt=self.beta_hat+k*op.beta
                C=(op.coupling(self.base.modes[l+H],l,self.base.alpha,kr,kt)
                   if abs(l)<=H else np.zeros((op.size,op.size),complex))
                if j==k:
                    C+=op.linear(kr,kt,mean_closure=False)-1j*k*self.base.omega/op.radius*op.M
                    C[3*op.m]=0;C[3*op.m,2*op.m:3*op.m]=op.grid.D[0]
                    if self.top=='dns':
                        C[3*op.m-1]=0;C[3*op.m-1,2*op.m:3*op.m]=op.grid.D[-1]
                row.append(sparse.csc_matrix(C))
            rows.append(row)
        self._cached_matrix=sparse.bmat(rows,format='csc');self._cached_alpha=alpha_hat
        return self._cached_matrix

    def fields(self,vector,phases):
        modes=np.asarray(vector).reshape(2*self.K+1,4,self.base.operators.m)
        return np.einsum('ph,hvz->pvz',np.exp(1j*np.asarray(phases)[:,None]*np.arange(-self.K,self.K+1)),modes)

    @property
    def A0(self):
        if not hasattr(self,'_A0'):self._A0=self.matrix(0)
        return self._A0

    @property
    def A1(self):
        if hasattr(self,'_A1'):return self._A1
        op,H=self.base.operators,self.base.harmonics;m=op.m;rows=[]
        for j in range(-self.K,self.K+1):
            row=[]
            for k in range(-self.K,self.K+1):
                l=j-k
                if abs(l)>H and j!=k:row.append(None);continue
                C=np.zeros((op.size,op.size),complex)
                if abs(l)<=H:
                    ur=self.base.modes[l+H,:m]
                    for v in range(3):C[v*m:(v+1)*m,v*m:(v+1)*m]=np.diag(1j*ur)
                if j==k:
                    scalar=np.diag(1j*op.base[0]+(2*k*self.base.alpha-1j/op.radius)/op.radius)
                    for v in range(3):C[v*m:(v+1)*m,v*m:(v+1)*m]+=scalar
                    C[:m,3*m:]=1j*op.I;C[3*m:,:m]=1j*op.I
                C[op.boundary_rows]=0;row.append(sparse.csc_matrix(C))
            rows.append(row)
        self._A1=sparse.bmat(rows,format='csc');return self._A1

    @property
    def A2(self):
        if not hasattr(self,'_A2'):
            op=self.base.operators
            self._A2=sparse.block_diag([op.M/op.radius]*(2*self.K+1),format='csc')
        return self._A2


def observed_shape(path,*,Re,frequency,grid,ro=-.751):
    """Full signed spatial Fourier shape at a positive NumPy temporal frequency."""
    with np.load(path) as data:
        j=int(np.argmin(abs(data['Re']-Re)));f=data['frequencies']
        kp=int(np.argmin(abs(f-frequency)));km=int(np.argmin(abs(f+f[kp])))
        disk,_=rotation_rates(ro);scale=1/np.sqrt(disk)
        positive=data['fourier'][kp,j,:3];negative=data['fourier'][km,j,:3,1:][:,::-1].conj()
        c=np.concatenate([negative,positive],axis=1)[[1,2,0]].transpose(1,0,2)*disk
        shape=native_p6_interpolate(data['z']*scale,c,grid.z)
        return float(f[kp]),shape


def shape_overlap(vector,problem,observed):
    """Common physical phase; optimize temporal complex phase ONLY, not theta."""
    op=problem.base.operators;K=problem.K;H=(len(observed)-1)//2
    if K>H:raise ValueError('Observed spatial bandwidth too small')
    modeled=np.zeros_like(observed)
    modeled[H-K:H+K+1]=np.asarray(vector).reshape(2*K+1,4,op.m)[:,:3]
    def dot(a,b):return np.einsum('qci,ij,qcj->',a.conj(),op.grid.mass,b)
    den=np.sqrt(dot(modeled,modeled).real*dot(observed,observed).real)
    return float(abs(dot(modeled,observed))/den)


def eigen_continuity(problem,alpha_hat,vector):
    """Independent between-node continuity check on the finite mapped grid."""
    op=problem.base.operators;grid=op.grid;ns=np.arange(-problem.K,problem.K+1)
    v=np.asarray(vector).reshape(len(ns),4,op.m)[:,:3].copy();v/=np.max(abs(v))
    kr=alpha_hat+ns*problem.base.alpha;kt=problem.beta_hat+ns*op.beta
    div=(1j*kr[:,None]+1/op.radius)*v[:,0]+1j*kt[:,None]/op.radius*v[:,1]+v[:,2]@grid.D.T
    xi=np.linspace(-1,1,4*(op.m-1)+3)
    interp=BarycentricInterpolator(grid.xi,v,axis=2);fine=interp(xi)
    den=1-xi+grid.scale/grid.zmax*(1+xi);dx_dz=den**2/(2*grid.scale)
    dw=interp.derivative(xi)[:,2]*dx_dz
    fine_div=(1j*kr[:,None]+1/op.radius)*fine[:,0]+1j*kt[:,None]/op.radius*fine[:,1]+dw
    return {'velocity_normalization':'max nodal harmonic velocity = 1',
         'wall_velocity':float(np.max(abs(v[:,:,0]))),'wall_continuity':float(np.max(abs(div[:,0]))),
         'interior_nodal_continuity':float(np.max(abs(div[:,1:-1]))),
         'between_node_continuity':float(np.max(abs(fine_div[:,1:-1])))}
