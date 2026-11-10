#!/usr/bin/env python3
"""Phase A physical-coordinate frozen dispersion, pinch continuation and r_end."""
from pathlib import Path
import argparse,json,time
import numpy as np
from scipy import linalg,optimize
from scipy.interpolate import BarycentricInterpolator,CubicSpline,PchipInterpolator
from scipy.optimize import linear_sum_assignment
from llsa.secondary.operators import PierOperators
from llsa.secondary.radial import radial_matrices
from fullbvp_phaseA_global import OUT,dump,load_state

pair=lambda z:[float(z.real),float(z.imag)]

class Frozen:
    def __init__(self,r,n,kind='global'):
        self.r=float(r);self.n=n;self.kind=kind
        self.op=op=PierOperators(radius=r,beta=68,n=n,ro=-1.)
        if kind=='global':self.A,self.B,self.C=radial_matrices(op,1,0.)
        else:
            self.A=op.linear(0.,68.)
            plus,minus=op.linear(1.,68.),op.linear(-1.,68.)
            self.B=(plus-minus)/(2j)
            self.C=-(plus+minus-2*self.A)/2
        self.M=op.M;self.mass=op.grid.mass
    def matrix(self,a):return self.A+1j*a*self.B-a*a*self.C
    def correlation(self,q,v):
        q=q.reshape(4,-1)[:3];v=v.reshape(4,-1)[:3]
        inner=lambda x,y:np.einsum('vi,ij,vj->',x.conj(),self.mass,y)
        return float(abs(inner(q,v))/np.sqrt(inner(q,q).real*inner(v,v).real))
    def temporal(self,a,target):
        T=-1j*self.r*self.matrix(a)
        vals,left,right=linalg.eig(T,self.M,left=True,right=True,check_finite=False)
        keep=np.flatnonzero(np.isfinite(vals)&(abs(vals)<2000))
        k=keep[np.argmin(abs(vals[keep]-target))];w=complex(vals[k]);q=right[:,k];p=left[:,k]
        group=-1j*self.r*np.vdot(p,(1j*self.B-2*a*self.C)@q)/np.vdot(p,self.M@q)
        residual=linalg.norm(T@q-w*(self.M@q))/(linalg.norm(T@q)+abs(w)*linalg.norm(self.M@q))
        return w,complex(group),q,float(residual)
    def saddle(self,a=.2-.13j,target=53+.6j):
        history=[]
        def evaluate(x):
            alpha=complex(*x);w,g,q,res=self.temporal(alpha,target)
            history.append(dict(alpha=pair(alpha),omega=pair(w),group=pair(g),residual=res))
            return [g.real,g.imag]
        sol=optimize.root(evaluate,[a.real,a.imag],tol=2e-10)
        a=complex(*sol.x);w,g,q,res=self.temporal(a,target)
        if abs(g)>2e-6 or res>1e-8:raise RuntimeError(f'Saddle failed {self.r,self.n,self.kind}: {sol.message}, group={g}, residual={res}')
        return dict(r=self.r,n=self.n,kind=self.kind,alpha=pair(a),omega=pair(w),growth=-a.imag,
            group=pair(g),group_abs=abs(g),residual=res,evaluations=len(history),history=history),q
    def spatial(self,w):
        Q0=self.A-1j*w/self.r*self.M;Q1=1j*self.B;Q2=-self.C
        m=len(Q0);I=np.eye(m);Z=np.zeros_like(Q0)
        L=np.block([[Z,I],[-Q0,-Q1]]);M=np.block([[I,Z],[Z,Q2]])
        vals,vec=linalg.eig(L,M,check_finite=False)
        good=np.isfinite(vals)&(abs(vals)<20)
        vals=vals[good];q=vec[:m,good];q/=linalg.norm(q,axis=0)
        num=linalg.norm(Q0@q+(Q1@q)*vals+(Q2@q)*vals**2,axis=0)
        den=linalg.norm(Q0)+abs(vals)*linalg.norm(Q1)+abs(vals)**2*linalg.norm(Q2)
        errors=num/den;keep=errors<1e-8
        return vals[keep],q[:,keep],errors[keep]
    def pinch(self,rec,steps=90,eta_max=12.):
        a0=complex(*rec['alpha']);w0=complex(*rec['omega']);previous=None;oldq=None;path=[]
        for eta in np.geomspace(1e-6,eta_max,steps):
            roots,q,res=self.spatial(w0+1j*eta)
            if previous is None:ids=np.argsort(abs(roots-a0))[:2];corr=[1.,1.]
            else:
                cost=np.array([[abs(a-b)+.025*(1-self.correlation(oldq[:,j],q[:,k])) for k,b in enumerate(roots)] for j,a in enumerate(previous)])
                ii,jj=linear_sum_assignment(cost);ids=jj[np.argsort(ii)]
                corr=[self.correlation(oldq[:,j],q[:,ids[j]]) for j in range(2)]
            previous=roots[ids];oldq=q[:,ids]
            if abs(previous[0]-previous[1])<1e-7:raise RuntimeError('Spatial roots collapsed')
            path.append(dict(eta=float(eta),roots=[pair(x) for x in previous],correlations=corr,
                residuals=res[ids].tolist(),separation=float(abs(previous[0]-previous[1])),candidates=len(roots)))
        crosses=[any(row['roots'][j][1]*path[0]['roots'][j][1]<0 for row in path) for j in range(2)]
        return dict(r=self.r,n=self.n,kind=self.kind,omega0=pair(w0),alpha0=pair(a0),path=path,
            eta_range=[1e-6,eta_max],opposite_half_planes_at_end=bool(previous[0].imag*previous[1].imag<0),
            crosses_real_axis=crosses,min_step_correlation=min(min(x['correlations']) for x in path),
            max_residual=max(max(x['residuals']) for x in path),
            interpretation='One-to-one branch continuation in upper Omega half-plane; finite-path numerical Briggs-Bers evidence.')

def curve(n,kind):
    rows={};states={};center=594.
    for r in [center]+list(np.arange(596.,632.,2.))+list(np.arange(592.,548.,-2.)):
        f=Frozen(r,n,kind)
        if states:
            old=min(states,key=lambda x:abs(x-r));prev=rows[old]
            rec,q=f.saddle(complex(*prev['alpha']),complex(*prev['omega']))
            rec['previous_r']=old;rec['correlation']=f.correlation(q,states[old])
            if rec['correlation']<.9:raise RuntimeError('Saddle branch correlation failed')
        else:rec,q=f.saddle();rec['correlation']=1.
        rows[r]=rec;states[r]=q
    return [rows[r] for r in sorted(rows)]

def crossings(r,y):
    out=[]
    for i in range(len(r)-1):
        if y[i]*y[i+1]<0:out.append(float(r[i]-y[i]*(r[i+1]-r[i])/(y[i+1]-y[i])))
    return out

def field_diagnostics(state,n):
    op=PierOperators(radius=594,beta=68,n=n,ro=-1.)
    r=state['radii'];v=state['q'][:,:3];mass=op.grid.mass
    energy=np.einsum('rvi,ij,rvj->r',v.conj(),mass,v).real
    t=(1.3-op.grid.a)/(1.3+op.grid.a)
    xi=optimize.brentq(lambda x:op.grid.b*x+(1-op.grid.b)*(x**3+op.grid.c*(1-x*x))-t,-1,1)
    vz=BarycentricInterpolator(op.grid.xi,v[:,1],axis=1)(xi)
    logE=np.log(np.maximum(energy,1e-300));logV=np.log(np.maximum(abs(vz),1e-150))
    gE=.5*np.gradient(logE,r,edge_order=2);gq=np.gradient(logV,r,edge_order=2)
    gEcs=.5*CubicSpline(r,logE).derivative()(r)
    return dict(r=r,energy=energy,gE=gE,gq=gq,gphys=gq+1/r,gE_cubic=gEcs,
        definition='E=integral sum|q_velocity|^2 dz; q=u_physical/R; z=1.3 interpolation in mapped Chebyshev coordinate')

def causal_global_branches(kind,wtarget,pinch):
    """Identify k-/k+ at high Im(Omega), then retain labels to global frequency and R."""
    f=Frozen(594.,32,kind);w0=complex(*pinch['omega0']);top=w0+12j
    aa=np.array([complex(*x) for x in pinch['path'][-1]['roots']])
    vals,q,res=f.spatial(top)
    ii,jj=linear_sum_assignment(abs(aa[:,None]-vals[None,:]));ids=jj[np.argsort(ii)]
    aa=vals[ids];oldq=q[:,ids]
    order=np.argsort(aa.imag);aa=aa[order];oldq=oldq[:,order]
    if not aa[0].imag<0<aa[1].imag:raise RuntimeError('High-frequency branches do not have causal half-plane labels')
    path=[]
    realpath=[complex(x,top.imag) for x in np.linspace(top.real,wtarget.real,12)[1:]]
    downpath=[complex(wtarget.real,wtarget.imag+y) for y in np.geomspace(top.imag-wtarget.imag,1e-6,140)[1:]]+[wtarget]
    for w in realpath+downpath:
        vals,q,res=f.spatial(w)
        cost=np.array([[abs(a-b)+.025*(1-f.correlation(oldq[:,j],q[:,k])) for k,b in enumerate(vals)] for j,a in enumerate(aa)])
        ii,jj=linear_sum_assignment(cost);ids=jj[np.argsort(ii)]
        corr=[f.correlation(oldq[:,j],q[:,ids[j]]) for j in range(2)]
        aa=vals[ids];oldq=q[:,ids]
        path.append(dict(omega=pair(w),roots=[pair(x) for x in aa],correlations=corr,residuals=res[ids].tolist()))
    rows={};states={594.:(aa,oldq)}
    def record(r,f,aa,qq,res,corr):
        branches=[]
        for j,label in enumerate(['k_minus','k_plus']):
            w,g,q,err=f.temporal(aa[j],wtarget)
            branches.append(dict(label=label,alpha=pair(aa[j]),spatial_growth=float(-aa[j].imag),group_velocity=pair(g),
                residual=float(res[j]),omega_recovered=pair(w),continuation_correlation=float(corr[j])))
        rows[r]=dict(r=r,kind=kind,omega_global=pair(wtarget),branches=branches,
            label_origin='k_minus: lower alpha half-plane; k_plus: upper alpha half-plane at Omega0+12i, then unique continuation')
    record(594.,f,aa,oldq,np.zeros(2),[1.,1.])
    for r in list(np.arange(592.,568.,-2.))+list(np.arange(596.,602.,2.)):
        f=Frozen(r,32,kind);previous=min(states,key=lambda x:abs(x-r));aa,oldq=states[previous]
        vals,q,res=f.spatial(wtarget)
        cost=np.array([[abs(a-b)+.025*(1-f.correlation(oldq[:,j],q[:,k])) for k,b in enumerate(vals)] for j,a in enumerate(aa)])
        ii,jj=linear_sum_assignment(cost);ids=jj[np.argsort(ii)]
        corr=[f.correlation(oldq[:,j],q[:,ids[j]]) for j in range(2)]
        aa=vals[ids];oldq=q[:,ids];states[r]=(aa,oldq)
        record(r,f,aa,oldq,res[ids],corr)
    return dict(kind=kind,high_frequency_omega=pair(top),frequency_path=path,
        min_frequency_correlation=min(min(x['correlations']) for x in path),
        radial_rows=[rows[r] for r in sorted(rows)])

def selfcheck():
    rows=[]
    for kind in ['global','pier']:
        f=Frozen(594.,32,kind);rec,q=f.saddle();pinch=f.pinch(rec)
        rows.append(dict(saddle=rec,pinch=pinch))
        print(json.dumps(dict(kind=kind,alpha=rec['alpha'],omega=rec['omega'],group=rec['group_abs'],
            pinch=pinch['opposite_half_planes_at_end'],last_roots=pinch['path'][-1]['roots'],correlation=pinch['min_step_correlation'])),flush=True)
    dump(OUT/'A3'/'selfcheck.json',rows)

def preflight():
    from fullbvp_phaseA_global import seed_for
    dest=OUT/'A3';summary=[]
    checked=json.loads((dest/'selfcheck.json').read_text())
    state=seed_for(32,.5);field=field_diagnostics(state,32);mask=(field['r']>=550)&(field['r']<=630)
    for kind in ['global','pier']:
        rows=curve(32,kind);dump(dest/f'curve_N32_{kind}.json',rows)
        cg=np.interp(field['r'][mask],[x['r'] for x in rows],[x['growth'] for x in rows])
        inter=crossings(field['r'][mask],field['gE'][mask]-cg)
        pin=next(x['pinch'] for x in checked if x['saddle']['kind']==kind)
        causal=causal_global_branches(kind,complex(state['omega'],state['sigma']),pin)
        dump(dest/f'legacy_causal_preflight_{kind}.json',causal)
        summary.append(dict(kind=kind,r_end_legacy=inter,min_correlation=causal['min_frequency_correlation'],
            sample=[r for r in causal['radial_rows'] if r['r'] in [570.,594.,600.]]))
    dump(dest/'legacy_preflight.json',summary)
    print(json.dumps(summary),flush=True)

def main():
    start=time.perf_counter();dest=OUT/'A3';dest.mkdir(parents=True,exist_ok=True)
    curves={};pinches=[];point_comparisons=[]
    for n in [32,40]:
        for kind in ['global','pier']:
            key=f'N{n}_{kind}';path=dest/f'curve_{key}.json'
            if path.exists():rows=json.loads(path.read_text())
            else:rows=curve(n,kind);dump(path,rows)
            curves[key]=rows
            mid=next(x for x in rows if x['r']==594.)
            f=Frozen(594.,n,kind);pinch=f.pinch(mid);pinches.append(pinch)
            dump(dest/f'pinch_{key}.json',pinch)
            if not pinch['opposite_half_planes_at_end']:raise RuntimeError(f'Pinch not established: {key}')
        fg=Frozen(594.,n,'global');fp=Frozen(594.,n,'pier');alpha=complex(*next(x for x in curves[f'N{n}_global'] if x['r']==594.)['alpha'])
        D=fg.matrix(alpha)-fp.matrix(alpha)
        point_comparisons.append(dict(n=n,r=594.,alpha=pair(alpha),matrix_max=float(abs(D).max()),
            matrix_relative_frobenius=float(linalg.norm(D)/linalg.norm(fp.matrix(alpha))),
            omega_at_same_alpha_global=pair(fg.temporal(alpha,53+.6j)[0]),omega_at_same_alpha_pier=pair(fp.temporal(alpha,53+.6j)[0]),
            reason='Frozen radial matrices retain derivatives of u_physical=R*q_velocity and p_physical=R^2*q_pressure; Pier.linear freezes those scales.'))
    grids=[]
    for n,dr in [(32,.5),(40,.5),(32,.25),(40,.25)]:
        path=OUT/f'neutral/N{n}_dr{dr:g}'/'neutral_state.npz';state=load_state(path);f=field_diagnostics(state,n);r=f['r']
        use=(r>=550)&(r<=630);rr=r[use];rec=dict(n=n,dr_requested=dr,dr_actual=float(r[1]-r[0]),source=str(path),omega=state['omega'],definition=f['definition'])
        for kind in ['global','pier']:
            c=curves[f'N{n}_{kind}'];cg=np.interp(rr,[x['r'] for x in c],[x['growth'] for x in c])
            rec[f'rend_{kind}']=crossings(rr,f['gE'][use]-cg)
            rec[f'rend_{kind}_cubic']=crossings(rr,f['gE_cubic'][use]-cg)
            rec[f'rend_{kind}_v13']=crossings(rr,f['gq'][use]-cg)
            cr=np.array([x['r'] for x in c]);cv=np.array([x['growth'] for x in c])
            rec[f'rend_{kind}_local_pchip']=crossings(rr,f['gE'][use]-PchipInterpolator(cr,cv)(rr))
            rec[f'rend_{kind}_local_cubic']=crossings(rr,f['gE'][use]-CubicSpline(cr,cv)(rr))
            rec[f'rend_{kind}_local_step4']=crossings(rr,f['gE'][use]-np.interp(rr,cr[::2],cv[::2]))
        sel=(r>=570)&(r<=600)
        rec['growth_R570_600']={name:dict(min=float(f[name][sel].min()),max=float(f[name][sel].max()),mean=float(f[name][sel].mean())) for name in ['gE','gq','gphys']}
        rec['sample_growth']=[dict(r=float(x),**{name:float(np.interp(x,r,f[name])) for name in ['gE','gq','gphys']}) for x in [570,580,590,594,600]]
        grids.append(rec)
        np.savez_compressed(dest/f'global_growth_N{n}_dr{dr:g}.npz',r=r,energy=f['energy'],gE=f['gE'],gq=f['gq'],gphys=f['gphys'])
    spaces=[];state=load_state(OUT/'neutral/N32_dr0.5/neutral_state.npz')
    for kind in ['global','pier']:
        pin=next(p for p in pinches if p['n']==32 and p['kind']==kind)
        causal=causal_global_branches(kind,complex(state['omega'],state['sigma']),pin)
        dump(dest/f'causal_global_frequency_{kind}.json',causal)
        spaces.extend(row for row in causal['radial_rows'] if row['r'] in [570.,580.,590.,594.,600.])
    dump(dest/'summary.json',dict(grids=grids,frozen_model_comparisons=point_comparisons,spatial_at_global_frequency=spaces,
        pinch_summary=[{k:v for k,v in p.items() if k!='path'} for p in pinches],
        r_end_definition='Intersection of 0.5*d ln(integral|q|^2 dz)/dR with -Im(alpha0(R)); alpha0 from local physical-coordinate pinch.',
        reference_r_end_2015=594.,reference_dns_growth_2016_approx=.106,
        reference_dns_note='0.106 is the user supplied approximate DNS comparison; Eq.(3.1) uses azimuthal velocity divided by local disk speed at z=1.3.',
        convention='exp(i alpha R + i beta theta - i Omega t); Omega=omega_lab+i sigma; temporal (-i R [A+i alpha B-alpha^2 C], M).',
        local_radius_range=[550.,630.],local_radius_step=2.,seconds=time.perf_counter()-start))
    print(json.dumps(dict(grids=[{k:v for k,v in g.items() if k.startswith('rend') or k in ['n','dr_actual']} for g in grids],seconds=time.perf_counter()-start)),flush=True)

if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--selfcheck',action='store_true');parser.add_argument('--preflight',action='store_true');args=parser.parse_args()
    (OUT/'A3').mkdir(parents=True,exist_ok=True)
    selfcheck() if args.selfcheck else preflight() if args.preflight else main()
