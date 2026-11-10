#!/usr/bin/env python3
"""Phase A: rebuilt-domain global linear eigenbranch, roots, and finite spectral search."""
from pathlib import Path
import sys, json, time, argparse, gc
import numpy as np
from scipy import sparse
from scipy.interpolate import interp1d, BarycentricInterpolator
from scipy.sparse.linalg import splu, eigs, LinearOperator
from scipy.optimize import brentq
from appelquist2015_hopf_common import (ROOT, HopfSponge, complex_block, mass_norm,
    radial_weights, mass_correlation)
from llsa.secondary.global_radial import GlobalRadial
from llsa.secondary.radial import RadialConfig
OUT=ROOT/'results/fullbvp_global_20261030/phaseA'
REF=50.5

def dump(path,data):
    path.parent.mkdir(parents=True,exist_ok=True)
    path.write_text(json.dumps(data,indent=2,ensure_ascii=False)+'\n')

def artifact(path,ext):return path.parent/(path.name+ext)

def build_problem(n,dr,rout,intervals=None,a=None,omega=REF):
    """Rout changes rebuild every radial stencil and boundary; intervals fixed in root calls."""
    intervals=round((rout-400)/dr) if intervals is None else int(intervals)
    r=np.linspace(400.,float(rout),intervals+1)
    return GlobalRadial(r,RadialConfig(ro=-1.,beta=68.,n=n,harmonics=1,
        omega=float(omega),nonlinear=False),outlet_alpha=0j,
        outlet_sponge=HopfSponge(float(rout-100 if a is None else a),40.,28.))

def interpolate_mode(state,p,common=False):
    q=np.asarray(state['q']).reshape(len(state['radii']),4,len(state['xi'])); r0=state['radii']; xi0=state['xi']
    if len(xi0)!=p.ops[0].m:
        q=BarycentricInterpolator(xi0,q,axis=2)(p.ops[0].grid.xi)
    r=p.radii
    if common:r=r[(r>=r0[0])&(r<=r0[-1])]
    # Extrapolation is used only to form an initial Arnoldi vector, never in correlation.
    return interp1d(r0,q,axis=0,kind='cubic',bounds_error=False,fill_value='extrapolate')(r)

def common_correlation(state,p,q):
    r=p.radii[(p.radii>=state['radii'][0])&(p.radii<=state['radii'][-1])]
    old=interpolate_mode(state,p,common=True)[:,:3]
    mask=(p.radii>=state['radii'][0])&(p.radii<=state['radii'][-1])
    new=q.reshape(len(p.radii),4,p.ops[0].m)[mask,:3]
    mass=p.ops[0].grid.mass; w=radial_weights(r)
    inner=lambda u,v:np.einsum('rvi,ij,rvj,r->',u.conj(),mass,v,w,optimize=True)
    return float(abs(inner(old,new))/np.sqrt(inner(old,old).real*inner(new,new).real))

def load_state(path):
    raw=np.load(path)
    q=raw['q'] if 'q' in raw else raw['q1']
    nr=len(raw['radii']);m=len(raw['xi'])
    if q.size!=nr*4*m:raise ValueError(f'Invalid mode shape {q.shape}: nr={nr}, m={m}')
    q=np.asarray(q,dtype=complex).reshape(nr,4,m)
    if not np.all(np.isfinite(q)):raise ValueError('Nonfinite seed mode')
    w=float(raw['omega_c']);s=float(raw['sigma_residual'])
    return dict(q=q,radii=raw['radii'],xi=raw['xi'],omega=w,sigma=s,
        source=str(path),rout=float(raw['radii'][-1]))

def save_state(path,p,q,omega,sigma,**kw):
    np.savez_compressed(path,q=q.reshape(len(p.radii),4,p.ops[0].m),
        radii=p.radii,xi=p.ops[0].grid.xi,z=p.ops[0].grid.z,
        mass=p.ops[0].grid.mass,omega_c=omega,sigma_residual=sigma,**kw)

def seed_for(n,dr):
    name=f'N{n}_dr{dr:g}'
    path=ROOT/'results/appelquist2015_hopf_20261029/neutral'/name/'neutral_state.npz'
    if not path.exists():
        path=ROOT/'results/appelquist2015_hopf_20261029/neutral/N40_dr0.5/neutral_state.npz'
    return load_state(path)

def solve_mode(n,dr,rout,state,intervals=None,a=None,path=None,k=4):
    if path and artifact(path,'.npz').exists() and artifact(path,'.json').exists():
        rec=json.loads(artifact(path,'.json').read_text())
        if rec.get('n')==n and abs(rec['rout']-rout)<1e-10 and abs(rec['a']-(rout-100 if a is None else a))<1e-10 and (intervals is None or rec['intervals']==intervals):
            state=load_state(artifact(path,'.npz'))
            print('RESUME '+str(path),flush=True);return rec,state
    start=time.perf_counter();p=build_problem(n,dr,rout,intervals,a);A,B=complex_block(p,1)
    target=complex(-state['sigma'],state['omega']-REF);shift=target+.055+.07j
    lu=splu(A-shift*B)
    K=LinearOperator(A.shape,matvec=lambda v:lu.solve(B@v),dtype=complex)
    v0=interpolate_mode(state,p).ravel();v0/=np.linalg.norm(v0)
    mu,vec=eigs(K,k=k,ncv=max(20,2*k+2),which='LM',tol=3e-11,maxiter=1200,v0=v0)
    values=shift+1/mu;candidates=[]
    for ix,lam in enumerate(values):
        q=vec[:,ix];corr=common_correlation(state,p,q)
        res=float(np.linalg.norm(A@q-lam*(B@q))/(np.linalg.norm(A@q)+abs(lam)*np.linalg.norm(B@q)))
        candidates.append(dict(index=ix,omega=float(REF+lam.imag),sigma=float(-lam.real),
            correlation=corr,residual=res,distance=float(abs(lam-target))))
    # Mode identity is selected by common-physical-domain velocity mass overlap.
    accepted=[c for c in candidates if c['residual']<1e-7]
    if not accepted:raise RuntimeError(f'No residual-accepted candidates: {candidates}')
    # Nonnormal neighbouring radial modes may have a slightly larger overlap.
    # Require local spectral continuity before comparing physical mode overlap.
    gated=[c for c in accepted if c['distance']<=.25]
    best=max(gated or accepted,key=lambda c:c['correlation'])
    if best['correlation']<.85 or best['distance']>.25:
        if path:dump(artifact(path,'_rejected.json'),dict(rout=rout,reference=state['source'],candidates=candidates))
        raise ContinuationError(f'Continuation needs smaller step: Rout={rout}, {candidates}')
    q=vec[:,best['index']];q/=mass_norm(p,q)
    v=q.reshape(len(p.radii),4,p.ops[0].m)[:,:3];q*=np.exp(-1j*np.angle(v.ravel()[np.argmax(abs(v))]))
    rec=dict(n=n,dr_requested=dr,dr_actual=float(p.radii[1]-p.radii[0]),
        intervals=len(p.radii)-1,nodes=len(p.radii),rout=float(rout),a=float(p.outlet_sponge.start),
        omega=best['omega'],omega_disk=best['omega']-68,sigma=best['sigma'],
        correlation=best['correlation'],residual=best['residual'],seconds=time.perf_counter()-start,
        candidates=candidates,reference=state['source'],spectral_distance_gate=.25,selection='maximum common-domain mass correlation inside local spectral-distance gate',k=k,ncv=max(20,2*k+2),shift=[shift.real,shift.imag])
    if path:
        dump(artifact(path,'.json'),rec)
        save_state(artifact(path,'.npz'),p,q,best['omega'],best['sigma'])
    result=dict(q=q.reshape(len(p.radii),4,p.ops[0].m),radii=p.radii.copy(),xi=p.ops[0].grid.xi,
        omega=best['omega'],sigma=best['sigma'],rout=float(rout),source=str(path))
    print('MODE '+json.dumps({x:rec[x] for x in ['n','dr_actual','rout','a','omega','sigma','correlation','residual','seconds']}),flush=True)
    del p,A,B,lu,K,vec;gc.collect()
    return rec,result

class ContinuationError(RuntimeError):pass

def scan():
    path=OUT/'A1';path.mkdir(parents=True,exist_ok=True);states={};records={};bridges=[]
    def step(r,seed,depth=0):
        try:
            rec,state=solve_mode(32,.5,r,seed,path=path/f'R{r:g}')
        except ContinuationError:
            if depth>=6 or abs(r-seed['rout'])<.08:raise
            mid=.5*(r+seed['rout'])
            _,seed=step(mid,seed,depth+1);bridges.append(mid)
            rec,state=step(r,seed,depth+1)
        states[r]=state;records[r]=rec;return rec,state
    requested=[720.,715.,710.,700.,725.,730.,740.]
    for r in [720.,715.,710.,705.,700.,725.,730.,735.,740.]:
        seed=min(states.values(),key=lambda s:abs(s['rout']-r)) if states else seed_for(32,.5)
        step(r,seed)
    dump(path/'all_points.json',[records[r] for r in sorted(records)])
    dump(path/'summary.json',[records[r] for r in sorted(requested)])
    dump(path/'continuation.json',dict(requested=requested,bridges=sorted(set(bridges+[705.,735.])),correlation_threshold=.85))

def neutral(n,dr):
    path=OUT/f'neutral/N{n}_dr{dr:g}';path.mkdir(parents=True,exist_ok=True)
    intervals=round(320/dr);history=[];states=[];seed=seed_for(n,dr)
    def evaluate(r):
        nonlocal seed
        for rec,st in zip(history,states):
            if abs(rec['rout']-r)<1e-11:return rec,st
        ref=min(states,key=lambda s:abs(s['rout']-r)) if states else seed
        rec,st=solve_mode(n,dr,r,ref,intervals=intervals,path=path/f'eval_{len(history):02}')
        history.append(rec);states.append(st);return rec,st
    lo,ls=evaluate(719.);hi,hs=evaluate(722.)
    if lo['sigma']*hi['sigma']>=0:raise RuntimeError('Root not bracketed')
    root=brentq(lambda r:evaluate(r)[0]['sigma'],719.,722.,xtol=1e-9,rtol=1e-12,maxiter=20)
    ix=min(range(len(history)),key=lambda i:abs(history[i]['rout']-root));rec,st=history[ix],states[ix]
    if abs(rec['sigma'])>2e-8:rec,st=evaluate(root)
    if abs(rec['sigma'])>2e-8:raise RuntimeError('Neutral defect too large')
    dump(path/'history.json',history)
    report=dict(rec,root_sigma_tolerance=2e-9,bracket=[lo['rout'],hi['rout']],
        topology='Fixed interval count at 320/dr; Rout and sponge start move continuously',
        total_seconds=sum(x['seconds'] for x in history))
    dump(path/'summary.json',report)
    np.savez_compressed(path/'neutral_state.npz',q=st['q'],radii=st['radii'],xi=st['xi'],
        omega_c=rec['omega'],sigma_residual=rec['sigma'])
    (path/'DONE').write_text('neutral branch accepted\n')

def controls():
    out=OUT/'A4';out.mkdir(parents=True,exist_ok=True);rows=[]
    for r in [720.,730.]:
        rec,_=solve_mode(32,.5,r,seed_for(32,.5),a=620.,path=out/f'a620_R{r:g}')
        rows.append(rec)
    dump(out/'summary.json',dict(rows=rows,matched_parameters=['N=32','actual dR=0.5','a=620','rise=40','strength=28'],
        legacy_neutral_a=[619.9581617759198,619.954056],legacy_dr=[.5,.25],legacy_omega_approx=53.0857))

def search():
    base=OUT/'neutral/N32_dr0.5';state=load_state(base/'neutral_state.npz')
    n=32;dr=.5;rout=state['rout'];p=build_problem(n,dr,rout,640);A,B=complex_block(p,1)
    rows=[];suspects=[];centers=[46.,49.,52.,55.,58.];k=10;ncv=36
    candidate_dir=OUT/'spectrum_candidates';candidate_dir.mkdir(parents=True,exist_ok=True)
    for center in centers:
        start=time.perf_counter();shift=.05+1j*(center-REF);lu=splu(A-shift*B)
        K=LinearOperator(A.shape,matvec=lambda v:lu.solve(B@v),dtype=complex)
        rng=np.random.default_rng(int(center*2015))
        mu,vec=eigs(K,k=k,ncv=ncv,which='LM',tol=2e-10,maxiter=2000,
            v0=rng.normal(size=A.shape[0])+1j*rng.normal(size=A.shape[0]))
        for i,lam in enumerate(shift+1/mu):
            q=vec[:,i];res=float(np.linalg.norm(A@q-lam*(B@q))/(np.linalg.norm(A@q)+abs(lam)*np.linalg.norm(B@q)))
            row=dict(center_omega=center,center_sigma=-.05,omega=float(REF+lam.imag),
                sigma=float(-lam.real),residual=res,tracked_correlation=common_correlation(state,p,q),accepted=res<1e-7)
            rows.append(row)
            if row['accepted'] and abs(complex(row['omega']-state['omega'],row['sigma']-state['sigma']))>2e-4 and row['sigma']>-.002:
                if not any(abs(complex(row['omega']-v['record']['omega'],row['sigma']-v['record']['sigma']))<2e-4 for v in suspects):
                    number=len(suspects);candidate_path=candidate_dir/f'candidate_{number:02}.npz'
                    save_state(candidate_path,p,q,row['omega'],row['sigma'])
                    suspects.append(dict(record=row,path=candidate_path))
        print('SPECTRUM '+json.dumps(dict(center=center,seconds=time.perf_counter()-start)),flush=True)
        del lu,K,vec;gc.collect()
    result=dict(n=n,dr_actual=float(p.radii[1]-p.radii[0]),rout=rout,k=k,ncv=ncv,
        centers=[[x,-.05] for x in centers],rows=rows,
        frequency_range=[min(x['omega'] for x in rows if x['accepted']),max(x['omega'] for x in rows if x['accepted'])],
        growth_range=[min(x['sigma'] for x in rows if x['accepted']),max(x['sigma'] for x in rows if x['accepted'])],
        finite_search=True,residual_tolerance=1e-7)
    result['independent_verifications']=[]
    result['suspect_criterion']='accepted residual, complex-frequency distance from target >2e-4, sigma>-0.002 regardless of mode correlation; deduplicated at 2e-4 in complex frequency'
    dump(OUT/'spectrum.json',result)
    del p,A,B;gc.collect()
    for ix,candidate in enumerate(suspects):
        seed=load_state(candidate['path']);record=dict(candidate=candidate['record'],source=str(candidate['path']))
        for label,nn in [('same_grid',32),('N40',40)]:
            try:
                rec,_=solve_mode(nn,.5,rout,seed,intervals=640,path=candidate_dir/f'verify_{ix:02}_{label}')
                record[label]=rec
            except Exception as exc:record[label]=dict(verified=False,error=str(exc))
        result['independent_verifications'].append(record);dump(OUT/'spectrum.json',result)
        print('INDEPENDENT_MODE '+json.dumps(dict(candidate=ix,omega=seed['omega'],sigma=seed['sigma'],same_grid=record['same_grid'].get('sigma'),N40=record['N40'].get('sigma'))),flush=True)

def checks():
    from appelquist2015_linear_tests import independent_complex_matrix
    p=build_problem(8,4.,580.);A,B=complex_block(p,1)
    independent=independent_complex_matrix(p,REF)
    pp=build_problem(8,4.,580.,omega=REF+.125);Ap,_=complex_block(pp,1)
    report=dict(complex_vs_independent_max=float(abs(A-independent).max()),
        omega_derivative_defect=float(abs(Ap-A+.125j*B).max()),
        sign='Omega=50.5+Im(lambda)-i Re(lambda); dA/domega=-iB')
    assert max(report[x] for x in ['complex_vs_independent_max','omega_derivative_defect'])<1e-11
    dump(OUT/'operator_checks.json',report);print('CHECKS '+json.dumps(report),flush=True)

def lightweight():
    rows=[]
    for n,dr in [(32,.5),(40,.5),(32,.25),(40,.25)]:
        st=seed_for(n,dr)
        for nn,rr in [(n,st['rout']),(40 if n==32 else 32,st['rout']),(n,720.)]:
            p=build_problem(nn,dr,rr);q=interpolate_mode(st,p);corr=common_correlation(st,p,q)
            assert q.shape==(len(p.radii),4,p.ops[0].m)
            assert abs(corr-1)<1e-10
            rows.append(dict(seed=st['source'],shape=list(st['q'].shape),target_n=nn,rout=rr,correlation=corr))
    dump(OUT/'lightweight_checks.json',rows)
    print(json.dumps(dict(cases=len(rows),minimum_correlation=min(x['correlation'] for x in rows))),flush=True)

if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('task',choices=['scan','neutral','controls','search','checks','lightweight'])
    parser.add_argument('--n',type=int,default=32);parser.add_argument('--dr',type=float,default=.5)
    args=parser.parse_args();OUT.mkdir(parents=True,exist_ok=True)
    if args.task=='neutral':neutral(args.n,args.dr)
    else:globals()[args.task]()
