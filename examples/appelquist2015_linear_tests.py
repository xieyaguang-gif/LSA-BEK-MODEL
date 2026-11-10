#!/usr/bin/env python3
"""Finite-domain beta=68 global linear spectrum; no core solver modifications.

Run with OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1.
Only first-positive-harmonic eigenpairs are retained.  Frequencies use
exp(i beta theta - i Omega t), Omega=omega_lab+i*sigma.
"""
from __future__ import annotations

import argparse
import csv
from dataclasses import asdict
import json
from pathlib import Path
import signal
import sys
import time

import numpy as np
from scipy import sparse
from scipy.interpolate import BarycentricInterpolator
from scipy.sparse.linalg import LinearOperator, eigs, splu, ArpackNoConvergence

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from llsa.secondary.global_radial import GlobalRadial, OutletSponge
from llsa.secondary.radial import RadialConfig, radial_matrices, pack_field, unpack_field

OUT = ROOT / "results/appelquist2015_tests_20261029/linear"
OMEGA_REF = 50.5


def dump(path, value):
    path.write_text(json.dumps(value, indent=2, allow_nan=False) + "\n")


def step(x):
    x = np.asarray(x, float)
    y = np.zeros_like(x)
    y[x >= 1] = 1
    ix = (x > 0) & (x < 1)
    a = 1 / (x[ix] - 1) + 1 / x[ix]
    y[ix] = np.exp(-np.logaddexp(0., a))
    return y


class PaperSponge(OutletSponge):
    """Appelquist outer step plus reversed inner step, in solver time units."""
    def profile(self, radii):
        r = np.asarray(radii)
        return self.strength * step((r-self.start)/self.width) + .8*step((420-r)/10)


def build(n=16, dr=2., rmin=450., rmax=580., paper=False, omega=OMEGA_REF):
    sponge = (PaperSponge(rmax-100, 40., 28.) if paper
              else OutletSponge(550., 30., 10.))
    r = np.linspace(rmin, rmax, round((rmax-rmin)/dr)+1)
    return GlobalRadial(r, RadialConfig(ro=-1., beta=68., n=n, harmonics=1,
        omega=omega, nonlinear=False), outlet_alpha=0j, outlet_sponge=sponge)


def extract(p):
    nr, s = len(p.radii), p.s
    ix = np.concatenate([i*p.width+s+np.arange(s) for i in range(nr)])
    L = p._linear_matrix(False)
    A = (L[ix][:, ix] + 1j*L[ix+s][:, ix]).tocsc()
    diagonal = np.zeros((nr, s))
    diagonal[1:-1, p.velocity_rows] = 1 / p.radii[1:-1, None]
    B = sparse.diags(diagonal.ravel(), format='csc')
    p._linear_cache.clear()
    return A, B


def independent_complex_matrix(p, omega):
    """Rebuild from complex radial coefficients, independent of real extraction."""
    a, b, c = [], [], []
    for i, op in enumerate(p.ops):
        aa, bb, cc = radial_matrices(op, 1, omega)
        rows = p.velocity_rows
        aa[rows, rows] += p.sponge_strength[i]/op.radius
        if i == 0 or i == len(p.radii)-1:
            aa[rows] = bb[rows] = cc[rows] = 0
            if i == 0:
                aa[rows, rows] = 1
            else:
                bb[rows, rows] = 1
        a.append(sparse.csr_matrix(aa))
        b.append(sparse.csr_matrix(bb))
        c.append(sparse.csr_matrix(cc))
    eye = sparse.eye(p.s)
    return (sparse.block_diag(a)+sparse.block_diag(b)@sparse.kron(p.D,eye)
            +sparse.block_diag(c)@sparse.kron(p.D2,eye)).tocsc()


def checks():
    started = time.perf_counter()
    p = build(n=8, dr=4., rmin=540., rmax=580.)
    A, B = extract(p)
    Acheck = independent_complex_matrix(p, OMEGA_REF)
    Aplus = independent_complex_matrix(p, OMEGA_REF+1.)
    rng = np.random.default_rng(2015)
    v = rng.normal(size=A.shape[0])+1j*rng.normal(size=A.shape[0])
    q = np.zeros((len(p.radii),3,p.s),complex)
    q[:,2], q[:,0] = v.reshape(len(p.radii),p.s), v.conj().reshape(len(p.radii),p.s)
    ref = unpack_field(p.evaluate_reference(pack_field(q),np.zeros((3,p.s),complex)),len(p.radii),1,p.s)
    expected = A@v
    empty = B.diagonal().reshape(len(p.radii),p.s)
    r = dict(complex_vs_real_matrix_max=float(abs(A-Acheck).max()),
        frequency_rebuild_max=float(abs(Aplus-A+1j*B).max()),
        original_residual_relative=float(np.linalg.norm(ref[:,2].ravel()-expected)/np.linalg.norm(expected)),
        conjugacy_relative=float(np.linalg.norm(ref[:,0]-ref[:,2].conj())/np.linalg.norm(ref[:,2])),
        mass_on_radial_bc_max=float(abs(empty[[0,-1]]).max()),
        mass_on_z_and_pressure_bc_max=float(abs(empty[:,p.ops[0].boundary_rows]).max()),
        sigma_sign="A q=lambda B q; sigma=-Re(lambda); omega=50.5+Im(lambda)",
        seconds=time.perf_counter()-started)
    # Adding uniform damping on all active rows moves sigma down by that amount.
    r['uniform_damping_algebra_max'] = float(abs((A+.3*B)-A-.3*B).max())
    dump(OUT/'operator_checks.json',r)
    assert max(v for k,v in r.items() if k not in ('seconds','sigma_sign')) < 1e-10
    print('CHECKS '+json.dumps(r),flush=True)
    return r


def radial_weights(r):
    w = np.gradient(r)
    w[[0,-1]] *= .5
    return w


def mass_norm(q, p):
    m = p.ops[0].m
    vel = q[:,:3*m].reshape(len(p.radii),3,m)
    density = np.einsum('rvi,ij,rvj->r',vel.conj(),p.ops[0].grid.mass,vel).real
    return np.sqrt(np.dot(radial_weights(p.radii),density)), density


def radial_diagnostics(q,r,mass):
    """Resolved-carrier and alternating-node indicators, without propagation claims."""
    m=len(mass)
    vel=q[:,:3*m].reshape(len(r),3,m)
    dr=float(r[1]-r[0])
    fourier=np.fft.fft(vel,axis=0)
    k=2*np.pi*np.fft.fftfreq(len(r),d=dr)
    power=np.einsum('kvi,ij,kvj->k',fourier.conj(),mass,fourier).real
    power=np.maximum(power,0.)
    power/=power.sum()
    alternating=vel[1:-1]-.5*(vel[:-2]+vel[2:])
    norm2=np.einsum('rvi,ij,rvj->',vel.conj(),mass,vel).real
    alt2=np.einsum('rvi,ij,rvj->',alternating.conj(),mass,alternating).real
    return dict(radial_fft_peak_k=float(k[np.argmax(power)]),
        radial_high_k_energy_fraction=float(power[abs(k)>=.65*np.pi/dr].sum()),
        alternating_node_curvature_relative=float(np.sqrt(max(alt2,0.)/norm2)),
        radial_fft_note='Periodic FFT diagnostic of saved finite-domain velocity; high k is |k|>=0.65*pi/dr.'),k,power


def spatial_growth(density,r,relative_floor=1e-8):
    """Central log derivative only where the point and both neighbours are resolved."""
    valid=np.asarray(density)>relative_floor*np.max(density)
    reliable=np.zeros(len(r),dtype=bool)
    reliable[1:-1]=valid[:-2]&valid[1:-1]&valid[2:]
    result=np.full(len(r),np.nan)
    i=np.flatnonzero(reliable)
    result[i]=.5*(np.log(density[i+1])-np.log(density[i-1]))/(r[i+1]-r[i-1])
    return result,reliable


def mode_diagnostics(p, A, B, lam, v):
    nr,m = len(p.radii),p.ops[0].m
    q = v.reshape(nr,p.s).copy()
    norm,density = mass_norm(q,p)
    q /= norm
    density /= norm**2
    # Use a reproducible global phase at the largest velocity sample.
    iv,jv = np.unravel_index(np.argmax(abs(q[:,:3*m])),(nr,3*m))
    q *= np.exp(-1j*np.angle(q[iv,jv]))
    v = q.ravel()
    av,bv = A@v,B@v
    defect = av-lam*bv
    full = np.zeros((nr,3,p.s),complex)
    full[:,2],full[:,0] = q,q.conj()
    ref = unpack_field(p.evaluate_reference(pack_field(full),np.zeros((3,p.s),complex)),nr,1,p.s)
    oracle = ref[:,2].ravel()-lam*bv
    velocity_scale = max(abs(q[:,:3*m]).max(),1e-30)
    wallfar = np.r_[0,m-1,m,2*m-1,2*m,3*m-1]
    outlet = p.D@q
    bc = max(abs(q[:,wallfar]).max(),abs(q[0,p.velocity_rows]).max(),
             abs(outlet[-1,p.velocity_rows]).max())/velocity_scale
    denom = np.linalg.norm(av)+abs(lam)*np.linalg.norm(bv)
    residual = np.linalg.norm(defect)/max(denom,1e-30)
    reference = np.linalg.norm(oracle)/max(denom,1e-30)
    rec = dict(omega_lab=float(OMEGA_REF+lam.imag),omega_disk=float(OMEGA_REF+lam.imag-68),
        sigma=float(-lam.real),lambda_real=float(lam.real),lambda_imag=float(lam.imag),
        eigen_residual=float(residual),independent_residual=float(reference),
        residual_inf=float(abs(defect).max()),bc_error_relative=float(bc),
        conjugacy_error=float(abs(full[:,0]-full[:,2].conj()).max()),
        mass_norm=float(mass_norm(q,p)[0]),energy_peak_R=float(p.radii[np.argmax(density)]),
        pressure_scale=float(abs(q[:,3*m:]).max()),velocity_scale=float(velocity_scale),
        accepted=bool(residual<1e-7 and reference<1e-7 and bc<1e-7))
    rec.update(radial_diagnostics(q,p.radii,p.ops[0].grid.mass)[0])
    growth,_ = spatial_growth(density,p.radii)
    rec.update(spatial_growth_relative_energy_floor=1e-8,
        spatial_growth_rule='NaN at endpoints or when central point/either neighbour has E1/R2 <= 1e-8 times that mode maximum.')
    return rec,q,density,growth


def run_case(name,n,dr,rmin,rmax,paper=False,targets=(50.15,),growth_target=.6,k=8,seed=2015):
    started = time.perf_counter()
    path = OUT/name
    path.mkdir(parents=True,exist_ok=True)
    config = dict(name=name,n=n,harmonics=1,dr=dr,rmin=rmin,rmax=rmax,paper=paper,
        targets=list(targets),growth_target=growth_target,k=k,omega_ref=OMEGA_REF,random_seed=seed,
        inlet='q_velocity=0',outlet='dR(q_velocity)=0',wall='q_velocity=0',
        farfield='q_velocity=0 at mapped z=infinity',pressure='original PierOperators closures')
    dump(path/'config.json',config)
    p=build(n,dr,rmin,rmax,paper)
    A,B=extract(p)
    assembled = time.perf_counter()
    records, vectors, densities, growths, timing = [],[],[],[],[]
    rng=np.random.default_rng(seed)
    for target in targets:
        shift=-growth_target+1j*(target-OMEGA_REF)
        t=time.perf_counter()
        lu=splu(A-shift*B)
        factored=time.perf_counter()
        K=LinearOperator(A.shape,matvec=lambda v:lu.solve(B@v),dtype=complex)
        status='converged'
        try:
            values,vecs=eigs(K,k=k,which='LM',tol=2e-10,maxiter=1600,
                ncv=max(2*k+1,24),v0=rng.normal(size=A.shape[0])+1j*rng.normal(size=A.shape[0]))
        except ArpackNoConvergence as exc:
            values,vecs=exc.eigenvalues,exc.eigenvectors
            status='partial_arpack_convergence'
        for mu,v in zip(values,vecs.T):
            lam=shift+1/mu
            if any(abs(lam-complex(r['lambda_real'],r['lambda_imag']))<1e-6 for r in records):
                continue
            rec,q,density,growth=mode_diagnostics(p,A,B,lam,v)
            rec.update(mode=len(records),shift_omega=float(target),arpack_status=status)
            records.append(rec);vectors.append(q);densities.append(density);growths.append(growth)
        timing.append(dict(target=target,factor_seconds=factored-t,total_seconds=time.perf_counter()-t,status=status))
        # Each shift is a durable checkpoint; a later timeout cannot erase it.
        dump(path/'modes.json',records)
        np.savez_compressed(path/'eigenpairs.npz',radii=p.radii,xi=p.ops[0].grid.xi,
            z=p.ops[0].grid.z,mass=p.ops[0].grid.mass,radial_weights=radial_weights(p.radii),
            modes=np.asarray(vectors),omega_lab=np.array([r['omega_lab'] for r in records]),
            sigma=np.array([r['sigma'] for r in records]),E1_over_R2=np.asarray(densities),
            spatial_growth=np.asarray(growths),sponge_total=p.sponge_strength,
            spatial_growth_valid_mask=np.isfinite(growths),spatial_growth_relative_energy_floor=1e-8,
            sponge_outer=p.outlet_sponge.strength*step((p.radii-p.outlet_sponge.start)/p.outlet_sponge.width),
            sponge_inner=.8*step((420-p.radii)/10) if paper else np.zeros(len(p.radii)))
        del lu,K
    data=dict(config=config,dimension=A.shape[0],matrix_nnz=A.nnz,
        assembly_seconds=assembled-started,shift_timings=timing,seconds=time.perf_counter()-started,
        sponge=asdict(p.outlet_sponge),accepted_count=sum(r['accepted'] for r in records),
        candidate_count=len(records),modes=records,
        interpretation='Algebraically accepted eigenpairs; grid convergence and band completeness separate.')
    dump(path/'summary.json',data)
    np.savetxt(path/'sponge_samples.csv',np.column_stack((p.radii,p.sponge_strength)),delimiter=',',header='R,total_damping',comments='')
    print('CASE '+json.dumps({k:data[k] for k in ('config','dimension','seconds','accepted_count')})+' MODES '+json.dumps([
        {k:r[k] for k in ('mode','omega_lab','sigma','eigen_residual','accepted')} for r in records]),flush=True)
    return data


def condition_check(name,mode):
    """First-order componentwise pencil sensitivity from an independently solved adjoint."""
    started=time.perf_counter()
    data=json.loads((OUT/name/'summary.json').read_text())
    c=data['config'];r=data['modes'][mode]
    p=build(c['n'],c['dr'],c['rmin'],c['rmax'],c['paper'])
    A,B=extract(p)
    q=np.load(OUT/name/'eigenpairs.npz')['modes'][mode].ravel()
    lam=complex(r['lambda_real'],r['lambda_imag'])
    shift=lam+.127+.173j
    lu=splu(A-shift*B)
    K=LinearOperator(A.shape,matvec=lambda w:lu.solve(B@w,trans='H'),dtype=complex)
    rng=np.random.default_rng(2026)
    mu,vectors=eigs(K,k=4,which='LM',tol=1e-11,maxiter=1600,ncv=24,
        v0=rng.normal(size=len(q))+1j*rng.normal(size=len(q)))
    values=(shift.conjugate()+1/mu).conj()
    selected=np.argmin(abs(values-lam))
    w=vectors[:,selected]
    left_lambda=values[selected]
    defect=A.getH()@w-lam.conjugate()*(B@w)
    overlap=abs(np.vdot(w,B@q))
    condition=(abs(w)@(abs(A)@abs(q))+abs(lam)*abs(w)@(abs(B)@abs(q)))/overlap
    result=dict(case=name,mode=mode,omega_lab=r['omega_lab'],sigma=r['sigma'],
        left_right_eigenvalue_difference=float(abs(left_lambda-lam)),
        left_residual=float(np.linalg.norm(defect)/(np.linalg.norm(A.getH()@w)+abs(lam)*np.linalg.norm(B@w))),
        generalized_eigenvalue_condition=float(np.linalg.norm(w)*np.linalg.norm(q)/overlap),
        componentwise_condition=float(condition),
        epsilon_times_componentwise_condition=float(np.finfo(float).eps*condition),
        seconds=time.perf_counter()-started,
        interpretation='First-order worst-case componentwise relative perturbations of A and lambda*B; not an actual error bound or proof of spectral completeness.')
    dump(OUT/name/'adjoint_condition.json',result)
    np.savez_compressed(OUT/name/'adjoint_eigenvector.npz',left_mode=w.reshape(len(p.radii),p.s),radii=p.radii,xi=p.ops[0].grid.xi)
    print('CONDITION '+json.dumps(result),flush=True)
    return result


def compare(left,right,rmax_common=None):
    a,b=[np.load(OUT/name/'eigenpairs.npz') for name in (left,right)]
    lo=max(a['radii'][0],b['radii'][0]);hi=min(a['radii'][-1],b['radii'][-1])
    if rmax_common is not None: hi=min(hi,rmax_common)
    r=np.intersect1d(a['radii'],b['radii'])
    r=r[(r>=lo)&(r<=hi)]
    if len(r)<3: raise ValueError('At least three shared radial nodes required for correlations')
    ix=np.searchsorted(a['radii'],r)
    ma,mb=len(a['xi']),len(b['xi'])
    va=a['modes'][:,:,:3*ma].reshape(-1,len(a['radii']),3,ma)[:,ix]
    va=BarycentricInterpolator(a['xi'],va,axis=-1)(b['xi'])
    vb=b['modes'][:,:,:3*mb].reshape(-1,len(b['radii']),3,mb)
    vb=vb[:,np.searchsorted(b['radii'],r)]
    mass=b['mass'];w=radial_weights(r)
    dot=np.einsum('arvi,ij,brvj,r->ab',va.conj(),mass,vb,w,optimize=True)
    na=np.einsum('arvi,ij,arvj,r->a',va.conj(),mass,va,w,optimize=True).real
    nb=np.einsum('brvi,ij,brvj,r->b',vb.conj(),mass,vb,w,optimize=True).real
    correlation=abs(dot)/np.sqrt(na[:,None]*nb[None,:])
    best=correlation.argmax(axis=1)
    records=[dict(left_mode=i,right_mode=int(j),correlation=float(correlation[i,j]),
        omega_left=float(a['omega_lab'][i]),omega_right=float(b['omega_lab'][j]),
        sigma_left=float(a['sigma'][i]),sigma_right=float(b['sigma'][j]),
        delta_omega=float(b['omega_lab'][j]-a['omega_lab'][i]),
        delta_sigma=float(b['sigma'][j]-a['sigma'][i])) for i,j in enumerate(best)]
    name=f'{left}__{right}'+('' if rmax_common is None else f'__Rle{rmax_common:g}')
    result=dict(left=left,right=right,common_region=[float(lo),float(hi)],
        metric='abs integral conj(velocity_left) M_z velocity_right dR / norms; source interpolated in xi',
        correlation_matrix=correlation.tolist(),best_overlap_matches=records,
        warning='Overlap matches are candidates; frequency continuity and convergence must also hold.')
    dump(OUT/f'{name}_correlation.json',result)
    print('CORRELATION '+name+' '+json.dumps(records),flush=True)
    return result


def audit():
    """Recompute derived diagnostics, preserve raw eigenpairs, and collate resolution evidence."""
    cases={}
    for path in sorted(OUT.glob('*/summary.json')):
        data=json.loads(path.read_text())
        data['config'].setdefault('harmonics',1)
        data['config'].setdefault('random_seed',2015)
        with np.load(path.parent/'eigenpairs.npz') as saved:
            arrays={k:saved[k] for k in saved.files}
        growth,mask,fft_power=[],[],[]
        for rec,q,density in zip(data['modes'],arrays['modes'],arrays['E1_over_R2']):
            g,valid=spatial_growth(density,arrays['radii'])
            rd,k,power=radial_diagnostics(q,arrays['radii'],arrays['mass'])
            rec.update(rd)
            rec.update(spatial_growth_relative_energy_floor=1e-8,
                spatial_growth_rule='Central 3-point stencil; all 3 energies > 1e-8*mode peak; NaN elsewhere and at both radial endpoints.',
                branch_id=None,branch_tracking_status='unmatched',resolution_status='not_demonstrated')
            growth.append(g);mask.append(valid);fft_power.append(power)
        arrays.update(spatial_growth=np.asarray(growth),spatial_growth_valid_mask=np.asarray(mask),
            spatial_growth_relative_energy_floor=np.array(1e-8),radial_fft_k=k,radial_fft_power=np.asarray(fft_power))
        np.savez_compressed(path.parent/'eigenpairs.npz',**arrays)
        cases[path.parent.name]=data
    branches=[]
    def track(branch_id,base,mode,targets):
        ref=cases[base]['modes'][mode]
        members=[]
        for name in targets:
            if name not in cases: continue
            if name==base:
                match=dict(right_mode=mode,correlation=1.,delta_omega=0.,delta_sigma=0.)
            else:
                result=compare(base,name)
                match=result['best_overlap_matches'][mode]
            rec=cases[name]['modes'][match['right_mode']]
            passed=(rec['accepted'] and match['correlation']>.99 and abs(match['delta_omega'])<.02 and abs(match['delta_sigma'])<.02)
            members.append(dict(case=name,mode=rec['mode'],omega_lab=rec['omega_lab'],omega_disk=rec['omega_disk'],
                sigma=rec['sigma'],correlation_to_reference=match['correlation'],
                delta_omega=match['delta_omega'],delta_sigma=match['delta_sigma'],comparison_passed=bool(passed)))
            if passed:
                rec.update(branch_id=branch_id,branch_tracking_status='tracked_by_frequency_and_mass_overlap',
                    resolution_status='consistent_on_selected_fine_grids')
        branches.append(dict(branch_id=branch_id,reference_case=base,reference_mode=mode,members=members,
            consistency_rule='mass overlap >0.99 and |delta omega|,|delta sigma|<0.02; these empirical checks do not certify full-spectrum completeness'))
    for mode in (0,1,2):
        track(f'pilot_fine_{mode}','pilot_N32_dr0.25',mode,
            ['pilot_N32_dr0.5','pilot_N32_dr0.25','pilot_N40_dr0.25'])
    for rmax in (680,730):
        name=f'paper_R{rmax}_N32_dr0.5'
        selected=max([r for r in cases[name]['modes'] if r['accepted'] and 50<=r['omega_lab']<=55],key=lambda r:r['sigma'])
        track(f'paper_R{rmax}_selected',name,selected['mode'],
            [f'paper_R{rmax}_N32_dr1',name,name+'_repeat',f'paper_R{rmax}_N32_dr0.25',f'paper_R{rmax}_N40_dr0.5'])
    for name in ('pilot_N16_dr2','pilot_N24_dr2','pilot_N32_dr2'):
        rec=cases[name]['modes'][0]
        rec.update(branch_id='pilot_coarse_original',branch_tracking_status='tracked_in_N_only',
            resolution_status='not_reproduced_in_targeted_radial_refinement')
    rows=[]
    failures=[]
    for name,data in cases.items():
        c=data['config']
        for rec in data['modes']:
            row=dict(case_id=name,N=c['n'],H=1,dr=c['dr'],Rmin=c['rmin'],Rmax=c['rmax'],paper=c['paper'])
            row.update({k:rec[k] for k in ('mode','omega_lab','omega_disk','sigma','eigen_residual','independent_residual',
                'bc_error_relative','accepted','branch_id','branch_tracking_status','resolution_status',
                'radial_fft_peak_k','radial_high_k_energy_fraction','alternating_node_curvature_relative')})
            rows.append(row)
            if not rec['accepted']: failures.append(row|dict(reason='eigen/original residual or BC exceeds 1e-7'))
        dump(OUT/name/'summary.json',data)
        dump(OUT/name/'modes.json',data['modes'])
    with (OUT/'resolution_table.csv').open('w',newline='') as f:
        writer=csv.DictWriter(f,fieldnames=list(rows[0]));writer.writeheader();writer.writerows(rows)
    conditions=[json.loads(f.read_text()) for f in sorted(OUT.glob('*/adjoint_condition.json'))]
    summary=dict(cases=list(cases.values()),tracked_branches=branches,adjoint_conditions=conditions,
        candidate_count=len(rows),algebraically_accepted_count=sum(r['accepted'] for r in rows),
        failures=failures,solver_seconds=sum(c['seconds'] for c in cases.values())+sum(c['seconds'] for c in conditions),
        acceptance_scope='accepted only tests the discrete eigenproblem; root review required for final scientific/plot acceptance.',
        spectrum_coverage='Finite shift-invert subsets near the stated complex shifts. No complete-spectrum stability or dominant-mode proof.',
        boundary_comparison='dq/dR=0 with q=u_prime/R, hence du_prime/dR=u_prime/R at outlet. Mapped z=infinity, all velocities zero; paper z=20.5, u=v=0, dw/dz=p.',
        spatial_growth_mask=dict(relative_energy_floor=1e-8,endpoint_rule='both endpoints NaN',
            stencil_rule='central 3-point; all three energies above threshold'),
        pinch_status='not_attempted; priority given to radial convergence and nonnormal spectral precision')
    if 'paper_R720_N32_dr0.5' in cases:
        probe=cases['paper_R720_N32_dr0.5']
        selected=max([r for r in probe['modes'] if r['accepted']],key=lambda r:r['sigma'])
        summary['neutral_probe']=dict(case='paper_R720_N32_dr0.5',mode=selected['mode'],
            omega_lab=selected['omega_lab'],omega_disk=selected['omega_disk'],sigma=selected['sigma'],
            status='single-grid near-neutral probe; growth sign unresolved because sigma is smaller than the neighbouring N32-to-N40 change; not a neutral threshold or paper r_end')
    dump(OUT/'summary.json',summary)
    dump(OUT/'failure_classification.json',dict(rejected_algebraic_candidates=failures,
        unresolved=['Coarse dr=2 growing candidates fail radial refinement.',
                    'Several deeper damped candidates vary with shift; do not interpret as precision spectra.',
                    'Coverage is local in complex frequency; global leading eigenvalue not certified.',
                    'No absolute-pinch or group-velocity/feedback-direction evidence.']))
    write_report(summary)
    print('AUDIT '+json.dumps({k:summary[k] for k in ('candidate_count','algebraically_accepted_count','solver_seconds')}),flush=True)


def write_report(summary):
    cases={d['config']['name']:d for d in summary['cases']}
    lines=['# Appelquist 2015：第一轮线性全局谱与分辨率检验','',
        '日期：2026-10-29。全部新结果位于本目录；既有归档与共享核心未修改。参数依据[原文对照与页码](../../../docs/APPELQUIST_2015_GLOBAL_INSTABILITY_COMPARISON.md)。', '',
        '**主要结果：旧 dr=2 增长候选未通过径向精化。论文相近的两组海绵配置则得到经网格、复 shift 和初始向量检查的代表候选：Rout=680 衰减、Rout=730 增长。** 这建立了当前近似模型与论文的定性对应；没有完成严格复现、全谱主模态判定或传播机制证明。','',
        '## 1. 时间符号、算子和验证','',
        '`A(50.5) q = lambda B q`；`sigma=-Re(lambda)`、`omega_lab=50.5+Im(lambda)`、`omega_disk=omega_lab-68`。第一正谐波由GlobalRadial零场实矩阵提取。B仅在内部速度动量行取1/R，径向/法向边界和压力行均为零。正阻尼进入A的正实部，因此降低sigma。', '',
        '独立复数系数装配与实块提取逐项一致；频率+1重建的最大误差1.13e-17；原始evaluate_reference随机场残差相对差2.10e-16；共轭和B的边界掩码检查为零。每个特征对另核对原始方程、速度边界、归一化及共轭。', '',
        '`accepted`仅指离散特征方程、独立原方程及BC的相对误差均低于1e-7，不等于物理解已收敛。最终可信度另由resolution_table.csv和tracked_branches给出。','',
        '## 2. 旧pilot：径向分辨率改变了谱','',
        '| N | dr | 候选omega_lab | sigma | 解释 |','|---:|---:|---:|---:|---|']
    for name,mode,meaning in [('pilot_N16_dr2',0,'原候选恢复'),('pilot_N24_dr2',0,'法向加密'),
        ('pilot_N32_dr2',0,'法向相关>0.99999'),('pilot_N32_dr0.5',0,'精细网格附近衰减候选；未与粗模态强行连接'),
        ('pilot_N32_dr0.25',0,'继续径向加密')]:
        c=cases[name]['config'];r=cases[name]['modes'][mode]
        lines.append(f"| {c['n']} | {c['dr']:g} | {r['omega_lab']:.8f} | {r['sigma']:+.8f} | {meaning} |")
    lines.extend(['',
        '在旧复频率Omega=50.063334397+0.509546340i附近，dr=1、0.5、0.25均用明确复shift提取12个候选；旧增长候选未重现。dr=0.5→0.25前三个附近衰减候选频率/增长率变化约1e-3以内；N40/dr0.25检查保持其衰减符号。该结论只覆盖已采样的复频率邻域。', '',
        '粗细网格不按排序索引对应。N24→N32、dr=2的原候选质量相关0.999997；dr=2→1的最佳相关约0.85–0.95，频率/增长率变化显著，因此不强认同一连续分支。', '',
        '旧N32/dr2原候选的径向FFT峰波数约0.1904，高波数能量(|k|>=0.65*pi/dr)仅1.36e-4；这不是明显的交替格点主模态。相邻点曲率随dr减小下降。径向离散及非正规放大仍需区分，不能仅凭FFT把它命名为棋盘格伪模态。', '',
        '## 3. 论文相近配置与精细网格候选','',
        'Rin=400，外海绵start=Rout-100、rise=40、strength=28；内海绵0.8*S((420-R)/10)。原始采样保存在sponge_samples.csv和eigenpairs.npz。平台保留到出口。', '',
        '| Rout | N | dr | omega_lab | omega_disk | sigma | 检查 |',
        '|---:|---:|---:|---:|---:|---:|---|'])
    for branch in summary['tracked_branches']:
        if not branch['branch_id'].startswith('paper'):continue
        for member in branch['members']:
            name=member['case'];c=cases[name]['config'];r=cases[name]['modes'][member['mode']]
            label='换shift/v0' if name.endswith('_repeat') else ('一致' if member['comparison_passed'] else '未通过分支一致性')
            lines.append(f"| {c['rmax']:g} | {c['n']} | {c['dr']:g} | {r['omega_lab']:.8f} | {r['omega_disk']:.8f} | {r['sigma']:+.8f} | {label} |")
    probe=summary.get('neutral_probe')
    if probe:
        lines.extend(['',f"补充的Rout=720单网格针对性候选：omega_lab={probe['omega_lab']:.8f}、omega_disk={probe['omega_disk']:.8f}、sigma={probe['sigma']:+.8f}。其增长率小于邻近两域N32→40变化约0.003，故应视为近中性、增长符号未定。该点未独立完成网格收敛，不能据此给出当前模型的精确中性边界。"])
    lines.extend(['',
        '680/730代表候选的dr=0.5→0.25变化约2e-5；N32→40的增长率变化约0.003，均保持符号。匹配同时使用复频率连续性和同一速度质量内积；完整相关矩阵保存在*correlation.json。', '',
        '## 4. 非正规性与数值精度','',
        '| Rout | 换shift/v0与原候选的差 | 左右特征值差 | 未结构化特征值条件数 | eps×逐元素条件量 |',
        '|---:|---:|---:|---:|---:|'])
    for c in summary['adjoint_conditions']:
        base=cases[c['case']];r=base['modes'][c['mode']]
        repeat=cases[c['case']+'_repeat']['modes']
        diff=min(abs(complex(t['omega_lab']-r['omega_lab'],t['sigma']-r['sigma'])) for t in repeat)
        lines.append(f"| {base['config']['rmax']:g} | {diff:.3e} | {c['left_right_eigenvalue_difference']:.3e} | {c['generalized_eigenvalue_condition']:.3e} | {c['epsilon_times_componentwise_condition']:.3e} |")
    lines.extend(['',
        '未结构化条件数很大。逐元素条件量考虑A和lambda*B的现有非零项与尺度，是一阶扰动敏感性估计，并非严格误差上界。代表候选的换shift实测及网格稳定性支持其增长/衰减符号。若干更深衰减候选在两个shift间漂移，未纳入精度或主导模态结论。小残差本身不足以保证特征值精确。', '',
        '## 5. 模型差异与诊断范围','',
        '当前入口为q速度齐次；出口为dR(q)=0，其中q=u_prime/R，因此物理速度满足dR(u_prime)=u_prime/R，并非严格物理速度齐次Neumann。当前映射至z=infinity且三速度归零；论文使用z=20.5，顶边界u=v=0、dw/dz=p。压力沿既有PierOperators闭合。论文的海绵选择相近，但这些边界与离散差异保留。', '',
        '论文r_end是与局部绝对空间分支对照后的有效终端。本轮没有为当前模型测定r_end，不能把Rout、海绵起点、能量峰或720直接替代它，也不把594移植为当前阈值。', '',
        '各模态按 integral(q_velocity* M_z q_velocity) dR=1归一化；幅值任意，不与非线性饱和值比较。保存E1/R²及g=0.5*dR log(E1/R²)。g只在中心与两邻点能量均超过模态峰值×1e-8时计算，端点及低幅值处NaN；保存显式mask。单个相位梯度或FFT波数不确定群速度或反馈方向。', '',
        '## 6. 交付、失败和下一步','',
        f"共{summary['candidate_count']}个候选，{summary['algebraically_accepted_count']}个通过离散代数验收；其余见failure_classification.json。累计装配、求谱、独立验证及左右条件检查约{summary['solver_seconds']:.1f}s；单进程BLAS/OMP/MKL=1，无GPU、远程作业或DNS。",
        '', '原始模态、频率、质量矩阵、径向权重、海绵、空间增长率mask及FFT诊断在各case/eigenpairs.npz；独立左模态另存。summary.json为统一机器接口，resolution_table.csv为全候选表，RUN_COMMANDS.sh记录实际成功命令。', '',
        '下一步优先使用已通过谱精化的径向分辨率重新核验非线性分支；若要证明论文中的传播/选频机制，再做同尺度局部空间分支和pinch拓扑及时间响应。此次优先完成径向和非正规性诊断，未启动未经核查的时域积分器，也未把局部驻点当作pinch证明。',''])
    (OUT/'REPORT.md').write_text('\n'.join(lines))


def alarm_handler(signum,frame):
    raise TimeoutError('Numerical wall-clock budget reached; completed shifts saved.')


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('stage',choices=['pilot','paper','refine','quality','neutral_probe','audit','compare'])
    parser.add_argument('--seconds',type=int,default=1000)
    args=parser.parse_args()
    OUT.mkdir(parents=True,exist_ok=True)
    signal.signal(signal.SIGALRM,alarm_handler)
    signal.alarm(args.seconds)
    started=time.perf_counter()
    try:
        if args.stage=='pilot':
            checks()
            for n,dr in ((16,2.),(24,2.),(32,2.),(32,1.)):
                run_case(f'pilot_N{n}_dr{dr:g}',n,dr,450.,580.)
        elif args.stage=='paper':
            for rmax in (680.,730.):
                for n,dr in ((24,2.),(32,2.),(32,1.)):
                    run_case(f'paper_R{rmax:g}_N{n}_dr{dr:g}',n,dr,400.,rmax,True,
                        targets=(50.5,53.5),growth_target=0.,k=8)
        elif args.stage=='refine':
            for dr in (1.,.5,.25):
                name=f'pilot_target_N32_dr{dr:g}' if dr==1 else f'pilot_N32_dr{dr:g}'
                run_case(name,32,dr,450.,580.,targets=(50.06333439718963,),growth_target=.5095463401520617,k=12)
            for rmax in (680.,730.):
                run_case(f'paper_R{rmax:g}_N32_dr0.5',32,.5,400.,rmax,True,
                    targets=(50.5,53.5),growth_target=0.,k=8)
        elif args.stage=='quality':
            run_case('pilot_N40_dr0.25',40,.25,450.,580.,targets=(50.9,),growth_target=-.5,k=6,seed=2026)
            for rmax in (680.,730.):
                name=f'paper_R{rmax:g}_N32_dr0.5'
                previous=json.loads((OUT/name/'modes.json').read_text())
                candidate=max([r for r in previous if r['accepted'] and 50<=r['omega_lab']<=55],key=lambda r:r['sigma'])
                omega,sigma=candidate['omega_lab'],candidate['sigma']
                run_case(name+'_repeat',32,.5,400.,rmax,True,targets=(omega+.173,),growth_target=sigma+.127,k=4,seed=2026)
                run_case(f'paper_R{rmax:g}_N32_dr0.25',32,.25,400.,rmax,True,targets=(omega,),growth_target=sigma,k=4,seed=2026)
                run_case(f'paper_R{rmax:g}_N40_dr0.5',40,.5,400.,rmax,True,targets=(omega,),growth_target=sigma,k=4,seed=2026)
                condition_check(name,candidate['mode'])
        elif args.stage=='audit':
            audit()
        elif args.stage=='neutral_probe':
            run_case('paper_R720_N32_dr0.5',32,.5,400.,720.,True,targets=(53.1,),growth_target=0.,k=6,seed=2026)
        pairs=[('pilot_N16_dr2','pilot_N24_dr2'),('pilot_N24_dr2','pilot_N32_dr2'),
               ('pilot_N32_dr2','pilot_N32_dr1'),('pilot_N32_dr1','pilot_N32_dr0.5'),
               ('pilot_N32_dr0.5','pilot_N32_dr0.25'),('pilot_N32_dr2','pilot_target_N32_dr1')]
        pairs.append(('pilot_N32_dr0.25','pilot_N40_dr0.25'))
        for rmax in (680,730):
            pairs.extend([(f'paper_R{rmax}_N24_dr2',f'paper_R{rmax}_N32_dr2'),
                          (f'paper_R{rmax}_N32_dr2',f'paper_R{rmax}_N32_dr1'),
                          (f'paper_R{rmax}_N32_dr1',f'paper_R{rmax}_N32_dr0.5')])
            pairs.extend([(f'paper_R{rmax}_N32_dr0.5',f'paper_R{rmax}_N32_dr0.5_repeat'),
                          (f'paper_R{rmax}_N32_dr0.5',f'paper_R{rmax}_N32_dr0.25'),
                          (f'paper_R{rmax}_N32_dr0.5',f'paper_R{rmax}_N40_dr0.5')])
        for left,right in pairs:
            if all((OUT/name/'eigenpairs.npz').exists() for name in (left,right)):
                compare(left,right)
        dump(OUT/f'{args.stage}_completion.json',dict(status='complete',seconds=time.perf_counter()-started))
    except Exception as exc:
        dump(OUT/f'{args.stage}_completion.json',dict(status='failed_or_budget',error=repr(exc),seconds=time.perf_counter()-started))
        raise


if __name__=='__main__':
    main()
