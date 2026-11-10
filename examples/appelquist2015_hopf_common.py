"""Shared fixed-domain Hopf model; research helpers, no core modifications."""
from pathlib import Path
import sys
import numpy as np
from scipy import sparse

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'src'))
from llsa.secondary.global_radial import GlobalRadial,OutletSponge
from llsa.secondary.radial import RadialConfig


def smooth_step_and_derivative(x):
    x=np.asarray(x,float)
    value=np.zeros_like(x); derivative=np.zeros_like(x)
    value[x>=1]=1
    inside=(x>0)&(x<1)
    z=x[inside]
    with np.errstate(over='ignore',divide='ignore'):
        exponent=1/(z-1)+1/z
    # Log-domain product avoids cancellation near S=1 and overflow at endpoints.
    log_s=-np.logaddexp(0.,exponent)
    log_one_minus_s=-np.logaddexp(0.,-exponent)
    value[inside]=np.exp(log_s)
    log_factor=np.logaddexp(-2*np.log(z),-2*np.log1p(-z))
    derivative[inside]=np.exp(log_s+log_one_minus_s+log_factor)
    return value,derivative


class HopfSponge(OutletSponge):
    def profile(self,radii):
        r=np.asarray(radii,float)
        return self.strength*smooth_step_and_derivative((r-self.start)/self.width)[0]+.8*smooth_step_and_derivative((420-r)/10)[0]

    def start_derivative(self,radii):
        return -self.strength/self.width*smooth_step_and_derivative((np.asarray(radii)-self.start)/self.width)[1]


def make_problem(n,dr,outer_start,harmonics=1,nonlinear=False,damp_mean=False,omega=50.5):
    radii=np.linspace(400.,730.,round(330./dr)+1)
    if abs(np.diff(radii)[0]-dr)>1e-10: raise ValueError('dr must divide the fixed radial interval 330')
    return GlobalRadial(radii,RadialConfig(ro=-1.,beta=68.,n=n,harmonics=harmonics,
        omega=float(omega),nonlinear=bool(nonlinear)),outlet_alpha=0j,
        outlet_sponge=HopfSponge(float(outer_start),40.,28.,damp_mean=bool(damp_mean)))


def complex_block(p,j):
    """Return Aj and physical Bj; dAj/domega=-i*j*Bj, including all BC masks."""
    if not 0<=j<=p.H: raise ValueError('Require 0 <= j <= harmonics')
    aa,bb,cc=[],[],[]
    diagonal=np.zeros((len(p.radii),p.s))
    for i,op in enumerate(p.ops):
        A,B,C=[m.copy() for m in p.coefficients[i][j+p.H]]
        rows=p.velocity_rows
        if j or p.outlet_sponge.damp_mean:
            A[rows,rows]+=p.sponge_strength[i]/op.radius
        if i in (0,len(p.radii)-1):
            A[rows]=B[rows]=C[rows]=0
            if i==0:
                A[rows,rows]=1
            elif j in p.outlet_constraints:
                A[rows]=p.outlet_constraints[j][:,:p.s]
                B[rows]=p.outlet_constraints[j][:,p.s:]
            else:
                lam=1j*j*p.outlet_alpha.real-j*p.outlet_alpha.imag
                A[rows,rows]=-lam;B[rows,rows]=1
        else:
            diagonal[i,rows]=1/op.radius
        aa.append(sparse.csr_matrix(A.real if j==0 else A))
        bb.append(sparse.csr_matrix(B.real if j==0 else B))
        cc.append(sparse.csr_matrix(C.real if j==0 else C))
    eye=sparse.eye(p.s)
    A=(sparse.block_diag(aa)+sparse.block_diag(bb)@sparse.kron(p.D,eye)
       +sparse.block_diag(cc)@sparse.kron(p.D2,eye)).tocsc()
    A.eliminate_zeros()
    return A,sparse.diags(diagonal.ravel(),format='csc')


def sponge_profile_and_derivative(p):
    """Total damping and derivative with respect to outer_start; inner is fixed."""
    return p.sponge_strength.copy(),p.outlet_sponge.start_derivative(p.radii)


def sponge_start_matrix(p,j):
    """dAj/da with all replaced BC rows zero; j=0 only if damp_mean=True."""
    diagonal=np.zeros((len(p.radii),p.s))
    if j or p.outlet_sponge.damp_mean:
        da=p.outlet_sponge.start_derivative(p.radii)
        diagonal[1:-1,p.velocity_rows]=da[1:-1,None]/p.radii[1:-1,None]
    return sparse.diags(diagonal.ravel(),format='csc')


def radial_weights(r):
    w=np.gradient(r);w[[0,-1]]*=.5
    return w


def mass_inner(p,q,v):
    m=p.ops[0].m;nr=len(p.radii)
    u=np.asarray(q).reshape(nr,4,m)[:,:3]
    v=np.asarray(v).reshape(nr,4,m)[:,:3]
    return np.einsum('rvi,ij,rvj,r->',u.conj(),p.ops[0].grid.mass,v,radial_weights(p.radii),optimize=True)


def mass_norm(p,q):
    return float(np.sqrt(max(mass_inner(p,q,q).real,0.)))


def mass_correlation(p,q,v):
    return float(abs(mass_inner(p,q,v))/(mass_norm(p,q)*mass_norm(p,v)))


def case_name(n,dr):
    return f'N{n}_dr{dr:g}'
