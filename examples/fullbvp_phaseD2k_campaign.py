#!/usr/bin/env python3
"""D2k: supervisor-launched continuation and temporal-spectrum diagnosis.

Preparation and validation never launch a production solve. All new writes are
confined to D2k/ or production_k/. Existing campaigns are read-only inputs.
"""
import os
for _key in ('OPENBLAS_NUM_THREADS','OMP_NUM_THREADS','MKL_NUM_THREADS','NUMEXPR_NUM_THREADS','BLIS_NUM_THREADS'):
    os.environ[_key] = '1'
import argparse
import csv
import fcntl
import gc
import hashlib
import json
import math
import resource
import signal
import sys
import time
from pathlib import Path
from types import MethodType
import numpy as np
from scipy import sparse
from scipy.interpolate import BarycentricInterpolator, PchipInterpolator, interp1d
from scipy.optimize import linear_sum_assignment
from scipy.sparse.linalg import LinearOperator, svds
import fullbvp_phaseD2i_campaign as inherited
import fullbvp_phaseD2i_localize as loc
import fullbvp_phaseC_solver as continuation
from fullbvp_phaseB_report import crossings

old, g = inherited.old, inherited.g
base, temporal_api = old.base, old.d2.v1
ROOT, OUT = inherited.ROOT, inherited.OUT
PREP = OUT / 'D2k'
PRODUCTION = OUT / 'production_k'
THREADS, MEMORY = 16, 400_000_000_000
WORKER_MEMORY = 398_000_000_000
TOTAL_BUDGET, STAGE_BUDGET = 172800., 21600.
TARGETS = [725., 727., 728., 728.5, 729., 729.2, 729.21875, 730., 740., 750., 760.]
VARIANTS = {
    'baseline': dict(n=24, H=4, dr=1., a=2., gap=100., width=40., strength=28.),
    'H6': dict(n=24, H=6, dr=1., a=2., gap=100., width=40., strength=28.),
    'radial_half': dict(n=24, H=4, dr=.5, a=2., gap=100., width=40., strength=28.),
    'map_a3': dict(n=24, H=4, dr=1., a=3., gap=100., width=40., strength=28.),
    'sponge_wide_weak': dict(n=24, H=4, dr=1., a=2., gap=100., width=60., strength=14.),
    'N40': dict(n=40, H=4, dr=1., a=2., gap=100., width=40., strength=28.),
}


def safe(path):
    path = Path(path).absolute()
    if path != path.resolve() or '..' in path.parts or not any(path.is_relative_to(p) for p in (PREP, PRODUCTION)):
        raise ValueError('D2k output escapes its two exclusive roots: ' + str(path))
    return path


def read(path):
    return json.loads(Path(path).read_text())


def write(path, value):
    path = safe(path); path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + '.tmp')
    tmp.write_text(json.dumps(value, indent=2, ensure_ascii=False, allow_nan=False) + '\n')
    tmp.replace(path)


def sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as f:
        for chunk in iter(lambda: f.read(1024*1024), b''):
            h.update(chunk)
    return h.hexdigest()


def install():
    old.output_path = safe; g.output_path = safe; inherited.h.output_path = safe
    old.MEMORY_BYTES = WORKER_MEMORY
    inherited.h.MEMORY_BYTES = WORKER_MEMORY


install()


def fingerprints():
    paths = {Path(__file__).resolve()}
    paths.update(p for p in (ROOT/'examples/fullbvp_phaseD2k_validate.py',ROOT/'examples/fullbvp_phaseD2k_audit.py',PRODUCTION/'commands.sh') if p.exists())
    for module in tuple(sys.modules.values()):
        p = getattr(module, '__file__', None)
        if p and Path(p).suffix == '.py' and any(Path(p).resolve().is_relative_to(ROOT/d) for d in ('examples','src')):
            paths.add(Path(p).resolve())
    return {str(p): sha(p) for p in sorted(paths)}


def seed_bundle(R):
    point = OUT / ('production_v4/run_D2f_baseline/upper720_H6' if R == 720 else 'production_v6/upper725/upper725_H6')
    manifest = read(point/'manifest.json'); record = read(point/'corrected.json')
    fields = Path(record['fields'])
    if not fields.is_absolute(): fields = ROOT / fields
    if sha(fields) != record['fields_sha256'] or record['fingerprint'] != manifest['fingerprint']:
        raise RuntimeError('source hash or manifest mismatch')
    params = manifest['params']
    if temporal_api.digest(params) != manifest['fingerprint']:raise RuntimeError('source manifest digest mismatch')
    with np.load(fields) as data:
        if not bool(data['accepted']) or (int(data['n']),int(data['H']),len(data['radii'])-1)!=(24,4,R-400) or abs(float(data['Rout'])-R)>1e-9:
            raise RuntimeError('source field metadata mismatch')
    if (params['n'],params['H'],params['intervals']) != (24,4,R-400):
        raise RuntimeError('source resolution mismatch')
    return dict(fields=str(fields), mapping=dict(a=2.,b=.6,c=.5), Rout=float(R),
                A1=record['after']['A1'], omega=record['after']['omega'],
                sha256={str(p):sha(p) for p in (fields,point/'manifest.json',point/'corrected.json')})


def audit_history():
    rows = []
    for family in ('phaseC','phaseC2'):
        for path in sorted((ROOT/'results/fullbvp_global_20261030'/family/'attempts').glob('*/summary.json')):
            d = read(path)
            if not 726 <= d.get('Rout',0) <= 731: continue
            it = d.get('iterations', [])
            linear = [r['linear']['relative_original'] for r in it if 'linear' in r and 'relative_original' in r['linear']]
            screen = path.with_name('screen.json')
            rows.append(dict(path=str(path),sha256=sha(path),Rout=d['Rout'],n=d.get('n'),H=d.get('H'),
                mode=d.get('mode'),target=d.get('target'),accepted=d.get('accepted'),failure=d.get('failure'),
                seconds=d.get('seconds'),iterations=len(it),linear_relative=linear,
                line_search_steps=[r['step_length'] for r in it if 'step_length' in r],
                final_metrics=d.get('final_metrics'),independent=d.get('independent'),
                screen=read(screen) if screen.exists() else None))
    write(PREP/'historical_attempts.json',rows)
    campaigns={}
    for family in ('phaseC','phaseC2'):
        path=ROOT/'results/fullbvp_global_20261030'/family/'campaign.json'
        data=read(path);campaigns[family]=dict(status=data.get('status'),failure=data.get('failure'),path=str(path),sha256=sha(path))
    write(PREP/'historical_campaign_status.json',campaigns)
    return rows


def prepare():
    seeds = [seed_bundle(R) for R in (720,725)]
    history = audit_history()
    plan = dict(revision='D2k',targets=TARGETS,variants=VARIANTS,seeds=seeds,code_sha256=fingerprints(),
        threads=THREADS,memory_bytes=MEMORY,worker_address_space_bytes=WORKER_MEMORY,total_budget_seconds=TOTAL_BUDGET,stage_budget_seconds=STAGE_BUDGET,
        initial_centers=[0.,.5,2.,5.,10.,20.],max_centers=10,center_budget_seconds=3600.,
        eigenpair_accept=1e-8,linear_policy=g.POLICY,stationary_accept=1e-11,
        minimum_Rout_step=.03125,maximum_Rout_step=2.5,maximum_arc_steps_per_target=24,
        production_started=False,coverage='local sampled centers; no exhaustive imaginary-axis exclusion',
        diagnostic='row-scaled phase-bordered stationary Jacobian, fixed-Rout and arc borders; finite-window vector localization')
    write(PREP/'plan.json',plan)
    print(json.dumps(dict(prepared=True,historical_attempts=len(history),production_started=False)))


def mesh(spec, Rout):
    mapping = inherited.mapping(spec['n'],spec['H'],int(math.ceil((Rout-400)/spec['dr'])),spec['a'])
    original = mapping.at
    def at(self, Rout, omega, derivative=False):
        p, blocks, bd, derivatives = original(Rout,omega,derivative)
        r=p.radii; x=self.xi
        original_shape, original_ds = base.common.smooth_step_and_derivative((r-Rout+100)/40)
        shape, ds = base.common.smooth_step_and_derivative((r-Rout+spec['gap'])/spec['width'])
        delta=spec['strength']*shape-28*original_shape
        delta_R=(spec['strength']/spec['width']*ds-.7*original_ds)*(x-1)
        change=self.mask @ sparse.kron(sparse.diags(delta/r),self.mass,format='csr')
        change_R=self.mask @ sparse.kron(sparse.diags(delta_R/r-delta*x/r**2),self.mass,format='csr') if derivative else None
        blocks=[blocks[0]]+[(A+change).tocsc() for A in blocks[1:]]
        if derivative: derivatives=[derivatives[0]]+[(A+change_R).tocsc() for A in derivatives[1:]]
        p.sponge_strength=p.sponge_strength+delta
        p.outlet_sponge=base.common.HopfSponge(Rout-spec['gap'],spec['width'],spec['strength'])
        return p,blocks,bd,derivatives
    mapping.at=MethodType(at,mapping)
    mapping.d2k_spec=spec
    return mapping


def transfer(point, mapping):
    return inherited.transfer_physical(point['fields'],mapping,point.get('mapping',dict(a=2.,b=.6,c=.5)))


def save_field(path,mapping,y,accepted,qref=None):
    path=safe(path); path.parent.mkdir(parents=True,exist_ok=True)
    temp=path.with_suffix('.tmp.npz'); spec=mapping.d2k_spec
    np.savez_compressed(temp,field=base.unpack(y[:-2],mapping.nr,mapping.H,mapping.s),y=y,
        radii=400+(y[-1]-400)*mapping.xi,xi=mapping.op.grid.xi,z=mapping.op.grid.z,
        mass=mapping.op.grid.mass,H=mapping.H,n=mapping.n,intervals=mapping.intervals,Rout=y[-1],omega=y[-2],
        a=y[-1]-spec['gap'],mapping_a=spec['a'],accepted=accepted,
        qref=np.array([]) if qref is None else qref)
    temp.replace(path)


def system_matrix(mapping,y,qref,mode,target,deadline,tangent=None):
    metric=continuation.CoherentMetric(mapping)
    system=g.prior.NonzeroCorrector(mapping,qref,y,mode,target,deadline.remaining(),metric,tangent)
    _,cols,rows=system.jacobian(y)
    q=base.unpack(y[:-2],mapping.nr,mapping.H,mapping.s)
    temporal=temporal_api.Temporal(mapping,q,y[-1],y[-2],deadline)
    J,stats=old.d2.assemble(temporal,deadline)
    K=sparse.bmat([[sparse.diags(1/system.fscale)@J,sparse.csr_matrix(cols/system.fscale[:,None])],
                  [sparse.csr_matrix(rows[:,:-2]),sparse.csr_matrix(rows[:,-2:])]],format='csr')
    return system,K,temporal,stats


def factor(K,deadline,path):
    return g.prior.EquilibratedFactor(K.astype(complex),threads=THREADS,deadline=deadline,stats_path=path)


def transpose_solve(fac,rhs):
    """Real K transpose with the SAME LU, undoing all three diagonal scalings.

    E=Dr K Dc; raw PARDISO stores A=S^-1 E. Thus K^T x=b gives
    A^T z=Dc b and x=Dr S^-1 z. The installed pypardiso CSC route
    uses iparm(12)=1; conjugate transpose equals transpose for this real K.
    """
    raw=fac.factor
    if np.any(raw.data.imag):raise ValueError('transpose helper requires a real stationary Jacobian')
    previous=int(raw.iparm[11]);raw.iparm[11]=1
    try:
        z=raw._call(33,fac.cols*np.asarray(rhs).reshape(-1))
        raw.info['transpose_solve_calls']=raw.info.get('transpose_solve_calls',0)+1
    finally:raw.iparm[11]=previous
    return (fac.rows*z/raw.scale).real.reshape(np.shape(rhs))


def direct_newton(mapping,initial,reference,mode,target,out,deadline,tangent=None):
    out=safe(out);out.mkdir(parents=True,exist_ok=False)
    y=initial.copy();qref=base.unpack(reference[:-2],mapping.nr,mapping.H,mapping.s)[:,mapping.H+1]
    metric=continuation.CoherentMetric(mapping)
    system=g.prior.NonzeroCorrector(mapping,qref,initial,mode,target,min(4800.,deadline.remaining()),metric,tangent)
    record=dict(accepted=False,mode=mode,target=float(target),iterations=[],initial_Rout=float(initial[-1]))
    started=time.monotonic()
    try:
        for step in range(13):
            deadline.check();row=system.metrics(y);row['iteration']=step;record['iterations'].append(row)
            save_field(out/'checkpoint.npz',mapping,y,False,qref);write(out/'progress.json',record)
            if row['accepted']:
                independent=base.independent(mapping,y)
                if independent['relative_field_inf']>1e-11 or independent['template_difference_relative']>1e-10:
                    raise RuntimeError('independent original-equation check failed')
                old.stationarity_gate(temporal_api.Temporal(mapping,base.unpack(y[:-2],mapping.nr,mapping.H,mapping.s),y[-1],y[-2],deadline))
                record.update(row,independent=independent);break
            if step==12: raise RuntimeError('Newton iteration limit')
            _,K,_,assembly=system_matrix(mapping,y,qref,mode,target,deadline,tangent)
            row['assembly']=assembly
            residual=system.residual(y);fac=None
            try:
                fac=factor(K,deadline,out/f'factor_{step:02d}.json')
                eye=LinearOperator(K.shape,matvec=lambda v:v,dtype=complex)
                inverse=g.Inverse(K,eye,0j,fac,deadline,out/f'linear_{step:02d}.jsonl',assembled_K=K)
                correction=inverse.apply(residual)
                if np.linalg.norm(correction.imag)>1e-8*max(np.linalg.norm(correction.real),1e-300):
                    raise RuntimeError('complex contamination of real Newton update')
                correction=correction.real
                row['linear']=inverse.measure(correction,-residual)[1]
            finally:
                if fac is not None: fac.close()
            f0=np.linalg.norm(residual)
            for back in range(13):
                trial=y+2.**(-back)*correction
                if not np.all(np.isfinite(trial)) or not 660<trial[-1]<850: continue
                if np.linalg.norm(system.residual(trial))<(1-1e-4*2.**(-back))*f0:
                    y=trial;row['step_length']=2.**(-back);break
            else: raise RuntimeError('Newton line search')
            del K,fac,inverse;gc.collect()
        if record['accepted']:
            aligned,screen=continuation.screen(reference,y,mapping,system.amplitude,tangent,
                refinement=abs(reference[-1]-y[-1])<1e-8,require_forward=False)
            if mode=='arc':
                # A fold can reverse R and amplitude while preserving the connected branch.
                reasons=[r for r in screen['reasons'] if r not in ('amplitude_ratio','upper_frequency_direction')]
                if not .8<=screen['amplitude_ratio']<=1.3:reasons.append('arc_amplitude_ratio')
                if abs(screen['delta_omega'])>.25:reasons.append('arc_frequency_jump')
                screen.update(reasons=reasons,accepted=not reasons,arc_allows_parameter_and_amplitude_turns=True,
                              arc_amplitude_bounds=[.8,1.3],arc_abs_frequency_change_max=.25)
            record['branch_screen']=screen
            if not screen['accepted']:record.update(accepted=False,failure='branch continuity screen')
            else:y=aligned
    except Exception as exc:
        record.update(accepted=False,failure=f'{type(exc).__name__}: {exc}')
    record.update(Rout=float(y[-1]),omega=float(y[-2]),seconds=time.monotonic()-started,
                  n=mapping.n,H=mapping.H,intervals=mapping.intervals)
    save_field(out/('fields.npz' if record['accepted'] else 'checkpoint.npz'),mapping,y,record['accepted'],qref)
    write(out/'summary.json',record)
    return y,record


def localize_vector(temporal,vector,a=2.):
    r=temporal.p.radii;xi=temporal.map.op.grid.xi
    matrices,_,_=loc.quadrature(xi,a=a);M=sum(matrices[:4])
    q=temporal_api.lift(vector,temporal.nr,temporal.H,temporal.s).reshape(temporal.nr,2*temporal.H+1,4,temporal.map.m)
    phi=temporal_api.lift(temporal.neutral(),temporal.nr,temporal.H,temporal.s).reshape(q.shape)
    coefficient=loc.inner(phi,q,M,r)/loc.inner(phi,phi,M,r)
    rem=q-coefficient*phi
    raw=loc.measures(q,matrices,r)
    remaining=float(loc.inner(rem,rem,M,r).real)
    result=dict(raw=raw,phase_coefficient=temporal_api.zpair(coefficient),
        orthogonal_fraction=remaining/raw['kinetic_norm_squared'],
        phase_orthogonal_shape=loc.measures(rem,matrices,r) if remaining>1e-20*raw['kinetic_norm_squared'] else None,
        projection_is_shape_diagnostic_not_eigenvector=True,window='z<=80; sensitivity to 40 and 160 retained')
    fine,_,_=loc.quadrature(xi,order=160,a=a)
    result['quadrature_relative_difference']=abs(loc.inner(q,q,sum(fine[:4]),r).real/raw['kinetic_norm_squared']-1)
    return result,rem


def stationary_diagnosis(mapping,y,out,deadline,tangent=None,stationary=False):
    out=safe(out);out.mkdir(parents=True,exist_ok=True)
    qref=base.unpack(y[:-2],mapping.nr,mapping.H,mapping.s)[:,mapping.H+1]
    results=[]
    for mode in (('Rout','arc') if tangent is not None else ('Rout',)):
        fac=None;result=dict(mode=mode,stationary=stationary,accepted=False,
            scaling='Newton field row scaling plus phase and continuation borders; Euclidean singular values',
            condition_is_scaling_dependent=True,single_LU_reused_for_adjoint=True)
        try:
            _,K,temporal,assembly=system_matrix(mapping,y,qref,mode,y[-1],deadline,tangent)
            fac=factor(K,deadline,out/f'{mode}_factor.json')
            def solve(f,b):
                deadline.check();shape=np.shape(b);x=f.solve(np.asarray(b).reshape(-1));return np.asarray(x).real.reshape(shape)
            inv=LinearOperator(K.shape,matvec=lambda b:solve(fac,b),rmatvec=lambda b:transpose_solve(fac,b),dtype=float)
            u,s,vh=svds(inv,k=1,which='LM',tol=1e-8,maxiter=160,random_state=20261031)
            # K^-1 u_K = v_K/sigma_K: inverse's left vector is K's right vector.
            right=u[:,0];left=vh[0];small=float(1/s[0])
            residual=max(np.linalg.norm(K@right-small*left),np.linalg.norm(K.T@left-small*right))/max(small,1e-300)
            largest=float(svds(K,k=1,which='LM',return_singular_vectors=False,tol=1e-6,maxiter=160,random_state=20261031)[0])
            localization,_=localize_vector(temporal,right[:-2],mapping.d2k_spec['a'])
            result.update(sigma_min=small,sigma_max=largest,condition_2_estimate=largest/small,
                singular_pair_relative_residual=float(residual),accepted=bool(np.isfinite(residual) and residual<=1e-4),
                localization=localization,omega_component=float(right[-2]),Rout_component=float(right[-1]),assembly=assembly)
            np.savez_compressed(out/f'{mode}_singular_vectors.npz',right=right,left=left,sigma_min=small)
        except Exception as exc: result['failure']=f'{type(exc).__name__}: {exc}'
        finally:
            if fac is not None:fac.close()
        results.append(result);write(out/'singular_diagnostics.json',results);gc.collect()
    return results


def phase_pair(pair):
    return abs(complex(*pair['value']))<=1e-8 and pair.get('phase_mass_overlap',0)>=1-1e-10


def candidate_pairs(centers,limit=5):
    pairs=[]
    for center in centers:
        if not center.get('complete'):continue
        for idx,p in enumerate(center['eigenpairs']):
            z=complex(*p['value'])
            if not p['accepted'] or phase_pair(p) or z.imag < -1e-7: continue
            if any(abs(z-complex(*q['value']))<2e-6*max(1,abs(z)) for q in pairs):continue
            pairs.append(dict(p,arrays=center['arrays'],index=idx))
    ordered=sorted(pairs,key=lambda p:abs(p['value'][0]))
    return ordered if limit is None else ordered[:limit]


def frequencies(value,omega,H):
    imag=value[1];disk=omega-68
    return dict(omega_lab=omega,omega_disk=disk,modulation_frequency=imag,
        sidebands=[dict(j=j,lab=j*omega-imag,disk=j*disk-imag) for j in range(-H,H+1)],
        convention='exp(lambda*t) exp(i*j*(68*theta-omega*t)); sideband=j*omega-Im(lambda)')


def spectrum(mapping,y,out,deadline,previous=None):
    out=safe(out);out.mkdir(parents=True,exist_ok=True)
    t=temporal_api.Temporal(mapping,base.unpack(y[:-2],mapping.nr,mapping.H,mapping.s),y[-1],y[-2],deadline)
    old.stationarity_gate(t);J,assembly=old.d2.assemble(t,deadline)
    fp=temporal_api.digest(dict(field=sha(out.parent/'fields.npz'),spec=mapping.d2k_spec,code=fingerprints()))
    required_shifts=[.05+1j*x for x in (0,.5,2,5,10,20)]
    shifts=list(required_shifts)
    if previous:
        shifts.extend(complex(max(.0001,p['value'][0]+.01),max(0,p['value'][1])) for p in previous.get('pairs',[])[:3] if p['value'][1]<=70)
    centers=[];used=[];index=0
    while shifts and index<10:
        sigma=shifts.pop(0)
        if any(abs(sigma-z)<.02 for z in used):continue
        if deadline.remaining()<60:break
        used.append(sigma)
        center_deadline=temporal_api.Deadline(min(3600.,deadline.remaining()))
        t.deadline=center_deadline
        result=g.run_center(t,J,sigma,out/f'center_{index:02d}',fp,center_deadline,k=8,ncv=48)
        centers.append(result);index+=1
        if result.get('complete'):
            for p in candidate_pairs([result])[:2]:
                z=complex(*p['value'])
                if abs(z.real)<.1 and z.imag>0 and z.imag<70:
                    shifts.append(complex(max(.0001,z.real+.01),z.imag))
        write(out/'centers.json',centers)
    t.deadline=deadline
    all_pairs=candidate_pairs(centers,limit=None)
    pairs=all_pairs[:5]+[p for p in all_pairs[5:] if p['value'][0]>0 and abs(p['value'][1])>1e-6]
    for idx,p in enumerate(pairs):
        with np.load(p['arrays']) as data: vector=data['vectors'][:,p['index']]
        p['localization'],rem=localize_vector(t,vector,mapping.d2k_spec['a'])
        p['frequency_relation']=frequencies(p['value'],float(y[-2]),mapping.H)
        shape_path=out/f'mode_{idx}.npz'
        np.savez_compressed(shape_path,field=rem,radii=t.p.radii,xi=mapping.op.grid.xi,H=mapping.H,a=mapping.d2k_spec['a'])
        p['shape']=str(shape_path);p['shape_sha256']=sha(shape_path)
        p['arrays_sha256']=sha(p['arrays'])
    result=dict(Rout=float(y[-1]),omega=float(y[-2]),pairs=pairs,assembly=assembly,
                centers_complete=sum(c.get('complete',False) for c in centers),centers_requested=len(used),
                requested_scan_finished=all(any(abs(complex(*c['sigma'])-z)<1e-8 and c.get('complete') for c in centers) for z in required_shifts),
                adaptive_centers_deferred=[temporal_api.zpair(z) for z in shifts],min_three_modes=len(pairs)>=3,
                positive_complex_candidates=[p for p in pairs if p['value'][0]>0 and abs(p['value'][1])>1e-6],
                stability_exhaustive=False,coverage='adaptive local windows; unsampled frequencies remain unresolved')
    write(out/'spectrum.json',result);return result


def overlap(a,b):
    with np.load(a) as da, np.load(b) as db:
        if int(da['H']) != int(db['H']) or float(da['a']) != float(db['a']):return 0.
        qa=da['field'];qb=db['field'];ra=da['radii'];rb=db['radii']
        qb=BarycentricInterpolator(db['xi'],qb,axis=-1)(da['xi'])
        xb=(rb-rb[0])/(rb[-1]-rb[0]);xa=(ra-ra[0])/(ra[-1]-ra[0])
        qb=interp1d(xb,qb,axis=0,kind='cubic')(xa)
        matrices,_,_=loc.quadrature(da['xi'],a=float(da['a']));M=sum(matrices[:4])
        aa=loc.inner(qa,qa,M,ra).real;bb=loc.inner(qb,qb,M,ra).real
        return float(min(1,abs(loc.inner(qa,qb,M,ra))/max(np.sqrt(aa*bb),1e-300)))


def match_modes(previous,current,overlap_fn=overlap):
    if not previous or not current:return []
    correlations=np.array([[overlap_fn(a['shape'],b['shape']) for b in current] for a in previous])
    distances=np.array([[abs(complex(*a['value'])-complex(*b['value']))/max(1,abs(complex(*a['value']))) for b in current] for a in previous])
    ii,jj=linear_sum_assignment(1-correlations+distances)
    return [(int(i),int(j),float(correlations[i,j])) for i,j in zip(ii,jj) if correlations[i,j]>=.5 and distances[i,j]<=.5]


def crossing_fit(rows):
    if len(rows)<3:return dict(status='insufficient_matched_points')
    x=np.array([p['Rout'] for p in rows[-4:]]);y=np.array([p['real'] for p in rows[-4:]])
    if np.ptp(x)<.2:return dict(status='insufficient_parameter_span')
    coef=np.polyfit(x-x.mean(),y,1);pred=np.polyval(coef,x-x.mean())
    quality=float(np.linalg.norm(y-pred)/max(np.linalg.norm(y-y.mean()),1e-30))
    if abs(coef[0])<1e-12 or quality>.25 or np.any(np.diff(y)*coef[0]<0):
        return dict(status='no_reliable_local_linear_extrapolation',relative_fit_residual=quality)
    root=float(x.mean()-coef[1]/coef[0])
    return dict(status='bracketed_interpolation' if y.min()<=0<=y.max() else 'unverified_local_extrapolation',
        Rout=root,slope=float(coef[0]),relative_fit_residual=quality,
        sampled_Rout=x.tolist(),outside_two_span=bool(root<x.min()-2*np.ptp(x) or root>x.max()+2*np.ptp(x)),
        is_bifurcation_confirmation=False)


def summarize(root,plot=False):
    root=safe(root);summary={};allrows=[]
    for name in VARIANTS:
        state_path=root/name/'state.json'
        if not state_path.exists():continue
        state=read(state_path);tracks={};last=[];next_id=0;previous_R=None
        for point in state['points']:
            p=Path(point['directory'])/'spectrum'/'spectrum.json'
            if not p.exists():continue
            spec=read(p);current=spec['pairs']
            # R-turns start a new segment; never regress a multivalued branch as lambda(R).
            matches=[] if previous_R is not None and point['Rout']<=previous_R else match_modes(last,current)
            ids={j:last[i]['track'] for i,j,_ in matches}
            for j,pair in enumerate(current):
                if j not in ids:ids[j]=next_id;next_id+=1
                pair['track']=ids[j]
                row=dict(variant=name,track=ids[j],Rout=point['Rout'],real=pair['value'][0],imag=pair['value'][1],
                         residual=pair['original_relative'],point=point['directory'])
                tracks.setdefault(ids[j],[]).append(row);allrows.append(row)
            last=current;previous_R=point['Rout']
        summary[name]=dict(status=state['status'],max_Rout=max((p['Rout'] for p in state['points']),default=None),
            tracks=tracks,crossings={str(k):crossing_fit(v) for k,v in tracks.items()},
            positive_complex_candidates=[p for p in allrows if p['variant']==name and p['real']>0 and abs(p['imag'])>1e-6],
            physical_points=[dict(Rout=p['Rout'],physical=p.get('physical_diagnostics'),singular=p.get('singular_diagnostics'),diagnosed=p.get('diagnosed'),directory=p['directory']) for p in state['points']],
            event_log=str(state_path))
    write(root/'SUMMARY.json',dict(variants=summary,exhaustive_stability=False))
    if allrows:
        with (root/'eigenvalue_tracks.csv').open('w',newline='') as f:
            writer=csv.DictWriter(f,fieldnames=list(allrows[0]));writer.writeheader();writer.writerows(allrows)
    lines=['# D2k production results','',
           'Status refers to completed calculations, not proof of a bifurcation. Spectral coverage is local.',
           'The fixed N32 (N40 for N40 fields) Phase A pinch reference is not recomputed at nonlinear omega.','']
    def number(value):return 'unavailable' if value is None else f'{value:.8g}'
    for name,data in summary.items():
        lines += [f'## {name}: {data["status"]}', '', '| Rout | omega | Fig.8 | max abs(qv1) before sponge | sigma_min (fixed Rout) | cond2 estimate |',
                  '|---:|---:|---:|---:|---:|---:|']
        for point in data['physical_points']:
            physical=point['physical']
            if not physical:continue
            singular=next((p for p in point['singular'] or [] if p['mode']=='Rout' and p['accepted']),{})
            lines.append('| '+' | '.join(number(v) for v in (point['Rout'],physical['omega'],physical['front_fig8'],
                physical['main_wave_z1p3_max_before_sponge'],singular.get('sigma_min'),singular.get('condition_2_estimate')))+' |')
        lines += ['',f'Matched tracks and conditional crossing fits: SUMMARY.json → variants → {name}.',
                  f'Positive complex candidates: {len(data["positive_complex_candidates"])}. Original residuals, harmonic/R/z localization and sidebands are in each point/spectrum/spectrum.json.',
                  'Compare fixed-Rout and arc singular diagnostics before interpreting a turn as a fold; inspect resolution and sponge controls before interpreting a temporal crossing as physical Hopf.','']
    (root/'RESULTS.md').write_text('\n'.join(lines)+'\n')
    if plot and allrows:
        import matplotlib
        matplotlib.use('Agg')
        import matplotlib.pyplot as plt
        plt.rcParams.update({'font.family':'serif','font.serif':['STIXGeneral'],'mathtext.fontset':'stix','font.size':10})
        for name,data in summary.items():
            fig,axes=plt.subplots(2,1,figsize=(6.3,5.2),sharex=True)
            fig.subplots_adjust(left=.16,right=.97,bottom=.12,top=.82,hspace=.13)
            for track,rows in data['tracks'].items():
                for ax,key in zip(axes,('real','imag')):
                    ax.plot([r['Rout'] for r in rows],[r[key] for r in rows],linestyle='-',marker=['o','s','^','D','v'][int(track)%5],ms=3,lw=1,label=f'mode {track}')
            axes[0].axhline(0,color='black',lw=.7);axes[0].set_ylabel(r'$\mathrm{Re}\,\lambda$')
            axes[1].set_ylabel(r'$\mathrm{Im}\,\lambda$');axes[1].set_xlabel(r'$R_{\mathrm{out}}$')
            axes[0].set_title(name+'; matched sampled modes',pad=30);axes[0].legend(fontsize=7,ncol=5,loc='lower center',bbox_to_anchor=(.5,1.01))
            for ax,key in zip(axes,('real','imag')):
                magnitudes=[abs(r[key]) for rows in data['tracks'].values() for r in rows if abs(r[key])>1e-9]
                if magnitudes and max(magnitudes)/min(magnitudes)>100:
                    ax.set_yscale('symlog',linthresh=1e-3)
                    values=[r[key] for rows in data['tracks'].values() for r in rows]
                    ax.set_ylim(min(-1e-4,min(values)*1.15),max(1e-4,max(values)*1.15))
            for ext in ('pdf','png'):fig.savefig(root/f'{name}_eigenvalue_tracks.{ext}',dpi=200)
            plt.close(fig)
    return summary


def point_diagnostics(mapping,y,point,out):
    spec=mapping.d2k_spec
    q=base.unpack(y[:-2],mapping.nr,mapping.H,mapping.s).reshape(mapping.nr,2*mapping.H+1,4,mapping.m)
    probe=BarycentricInterpolator(mapping.op.grid.xi,q,axis=-1)(loc.xi_of_z(1.3,spec['a']))
    r=400+(y[-1]-400)*mapping.xi;cut=y[-1]-spec['gap'];mask=r<cut
    v=probe[:,mapping.H+1:,1];fund=abs(v[:,0]);rms=np.sqrt(2*np.sum(abs(v)**2,axis=1))
    D,_=base.radial_differences(r)
    growth=D@np.log(np.maximum(rms,1e-250))
    # There is no N24 local-pinch curve. Keep the fixed N32 reference explicit.
    reference_n=40 if mapping.n==40 else 32
    path=ROOT/f'results/fullbvp_global_20261030/phaseA/A3/curve_N{reference_n}_global.json'
    curve=read(path);xr=np.array([p['r'] for p in curve]);gr=np.array([p['growth'] for p in curve])
    local=PchipInterpolator(xr,gr,extrapolate=False)(r)
    valid=(r>=max(550.,xr.min()))&(r<=min(650.,cut,xr.max()))&(fund>fund.max()*1e-8)
    found=crossings(r,growth-local,valid)
    front=next((p['R'] for p in found if p['direction']=='down'),None)
    result=dict(Rout=float(y[-1]),omega=float(y[-2]),sponge_onset=float(cut),front_fig8=front,
        front_status='found' if front is not None else 'no_downcrossing_in_reference_coverage',
        front_definition='z=1.3 d log(v_rms)/dR equals local absolute-pinch growth; first downstream downcrossing',
        reference=str(path),reference_sha256=sha(path),reference_n=reference_n,field_n=mapping.n,
        reference_frequency_status='fixed Phase A curve; not recomputed at nonlinear omega',
        front_search_interval=[max(550.,float(xr.min())),min(650.,float(cut),float(xr.max()))],
        main_wave_z1p3_max_before_sponge=float(fund[mask].max()),
        amplitude_definition='max |complex Fourier q_v,1(z=1.3)|, R<sponge onset; already normalized by R*Omega',
        main_wave_z1p3_real_harmonic_peak_before_sponge=float(2*fund[mask].max()),
        mapping_a=spec['a'],H=mapping.H,actual_dr=float((y[-1]-400)/mapping.intervals))
    np.savez_compressed(out/'physical_profiles.npz',radii=r,probe_v=v,vrms=rms,growth=growth,reference_growth=local)
    write(out/'physical_diagnostics.json',result)
    return result


def complete_diagnostics(mapping,y,point,state,deadline,tangent):
    out=Path(point['directory'])
    point['physical_diagnostics']=point_diagnostics(mapping,y,point,out)
    previous_spectrum=None
    predecessors=[]
    for predecessor in state['points']:
        if predecessor['directory']==point['directory']:break
        predecessors.append(predecessor)
    for predecessor in reversed(predecessors):
        path=Path(predecessor['directory'])/'spectrum/spectrum.json'
        if path.exists():previous_spectrum=read(path);break
    result=spectrum(mapping,y,out/'spectrum',deadline,previous_spectrum)
    singular=[]
    if 727<=point['Rout']<=730:
        singular=stationary_diagnosis(mapping,y,out/'stationary_diagnosis',deadline,tangent,stationary=True)
    point['singular_diagnostics']=singular
    point['spectrum_complete']=bool(result['min_three_modes'] and result['requested_scan_finished'])
    point['diagnosed']=bool(point['spectrum_complete'] and all(p['accepted'] for p in singular))
    if not point['diagnosed']:state['events'].append(dict(target=point['Rout'],status='partial_diagnostics'))


class Budget:
    def __init__(self,root):self.path=safe(root/'budget.json')
    def begin(self,label,seconds):
        d=read(self.path) if self.path.exists() else dict(limit=TOTAL_BUDGET,charged=0.,events=[])
        if d['charged']+seconds>d['limit']:raise RuntimeError('cumulative budget exhausted; no automatic extension')
        event=dict(label=label,reserved=seconds,status='ACTIVE',started=time.time());d['events'].append(event)
        d['charged']+=seconds;write(self.path,d)
        self.started=time.monotonic();return temporal_api.Deadline(seconds)
    def finish(self):
        d=read(self.path);event=d['events'][-1];elapsed=time.monotonic()-self.started
        d['charged']+=elapsed-event['reserved'];event.update(status='FINISHED',seconds=elapsed);write(self.path,d)


def variant(root,name,plan,budget):
    spec=VARIANTS[name];out=safe(root/name);out.mkdir(parents=True,exist_ok=True)
    state_path=out/'state.json'
    state=read(state_path) if state_path.exists() else dict(status='running',points=[],events=[])
    for point in state['points']:
        if sha(point['fields']) != point['fields_sha256']:raise RuntimeError('accepted restart field hash changed')
    target_list=TARGETS if name in ('baseline','sponge_wide_weak') else TARGETS[:8]+[760.]
    for target in target_list:
        accepted_target=next((p for p in reversed(state['points']) if abs(p['Rout']-target)<1e-7),None)
        if accepted_target is not None and accepted_target.get('diagnosed'):continue
        deadline=budget.begin(f'{name}:Rout={target}',min(STAGE_BUDGET,TOTAL_BUDGET))
        try:
            if accepted_target is not None:
                # Resume diagnostics at their own saved state, even if continuation
                # has subsequently advanced to a larger Rout.
                point=accepted_target
                with np.load(point['fields']) as data:y=data['y'].copy();intervals=int(data['intervals'])
                mapping=mesh(spec,400+spec['dr']*intervals)
                index=state['points'].index(point)
                predecessor=state['points'][index-1] if index else plan['seeds'][0]
                previous=transfer(predecessor,mapping);previous,_=continuation.phase_align(previous,y,mapping)
                metric=continuation.CoherentMetric(mapping)
                tangent=metric.tangent(previous,y) if metric.length(y-previous)>1e-10 else None
                complete_diagnostics(mapping,y,point,state,deadline,tangent)
                write(state_path,state);continue
            current=state['points'][-1] if state['points'] else plan['seeds'][1]
            prior=state['points'][-2] if len(state['points'])>1 else plan['seeds'][0]
            distance=max(0,target-current['Rout']);step=min(2.5,distance)
            arc=False;arc_steps=0;attempt_count=0;arc_scale=1.
            while True:
                deadline.check();attempt_count+=1
                if attempt_count>80:raise RuntimeError('target attempt bound')
                trial_R=min(target,current['Rout']+step)
                mapping=mesh(spec,max(trial_R,current['Rout']))
                reference=transfer(current,mapping);previous=transfer(prior,mapping)
                previous,_=continuation.phase_align(previous,reference,mapping)
                metric=continuation.CoherentMetric(mapping)
                tangent=metric.tangent(previous,reference) if metric.length(reference-previous)>1e-10 else None
                if arc and tangent is None:raise RuntimeError('no same-branch secant for arc')
                if arc:
                    arc_steps+=1
                    if arc_steps>24:raise RuntimeError('arc step bound; branch may turn away from target')
                    ds=arc_scale*min(.15,max(.005,step/max(abs(tangent[-1]),.05)))
                    initial=reference+ds*tangent;mode='arc'
                else:
                    initial=reference.copy();mode='Rout'
                    if tangent is not None and abs(tangent[-1])>.05:
                        initial+=(trial_R-reference[-1])/tangent[-1]*tangent
                    initial[-1]=trial_R
                attempts=out/'attempts';attempts.mkdir(exist_ok=True)
                attempt=attempts/f'attempt_{len(list(attempts.iterdir())):04d}'
                y,record=direct_newton(mapping,initial,reference,mode,trial_R,attempt,deadline,tangent)
                state['events'].append(dict(attempt=str(attempt),requested_Rout=trial_R,mode=mode,step=step,
                    actual_Rout=float(y[-1]),accepted=record['accepted'],failure=record.get('failure'),
                    predictor_Rout=float(initial[-1]),arc_length=ds if arc else None,
                    tangent_Rout=None if tangent is None else float(tangent[-1])))
                write(state_path,state)
                if not record['accepted']:
                    if not arc and step>.03125:
                        step/=2;continue
                    stationary_diagnosis(mapping,y,attempt/'failure_diagnosis',deadline,tangent,stationary=False)
                    if arc:
                        if arc_scale>.0625:arc_scale/=2;continue
                        raise RuntimeError('arc corrector failed at minimum step; see failure_diagnosis')
                    arc=True;step=.05;continue
                point=dict(fields=str(attempt/'fields.npz'),fields_sha256=sha(attempt/'fields.npz'),
                    mapping=dict(a=spec['a'],b=.6,c=.5),Rout=float(y[-1]),omega=float(y[-2]),
                    H=mapping.H,n=mapping.n,intervals=mapping.intervals,A1=record['A1'],directory=str(attempt),diagnosed=False)
                prior,current=current,point;state['points'].append(point);arc_scale=min(1.,arc_scale*1.25);write(state_path,state)
                at_target=abs(y[-1]-target)<1e-7
                if at_target:
                    complete_diagnostics(mapping,y,point,state,deadline,tangent)
                    write(state_path,state)
                    break
                if y[-1]>=target:
                    arc=False;step=target-y[-1]
                elif not arc:step=min(2.5,max(step,1.5*step),target-y[-1])
            state['status']='running';write(state_path,state)
        except Exception as exc:
            state.update(status='incomplete',failure=f'{type(exc).__name__}: {exc}',failed_target=target)
            write(state_path,state);break
        finally:budget.finish()
    else:
        state['status']='complete' if all(any(abs(p['Rout']-t)<1e-7 and p.get('diagnosed') for p in state['points']) for t in target_list) else 'partial_spectrum'
        write(state_path,state)
    return state


def run(root):
    root=safe(root);root.mkdir(parents=True,exist_ok=True)
    if root==PREP or not root.is_relative_to(PRODUCTION):raise ValueError('production requires production_k root')
    with (PRODUCTION/'.dispatch.lock').open('a') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        plan=read(PREP/'plan.json')
        for path,value in plan['code_sha256'].items():
            if sha(path)!=value:raise RuntimeError('code differs from prepared plan: '+path)
        for seed in plan['seeds']:
            for path,value in seed['sha256'].items():
                if sha(path)!=value:raise RuntimeError('source changed: '+path)
        manifest=root/'manifest.json';identity=temporal_api.digest(plan)
        if manifest.exists() and read(manifest)['fingerprint']!=identity:raise RuntimeError('resume identity changed')
        write(manifest,dict(fingerprint=identity,plan=plan))
        try:
            resource.setrlimit(resource.RLIMIT_AS,(WORKER_MEMORY,WORKER_MEMORY))
            affinity=sorted(os.sched_getaffinity(0));os.sched_setaffinity(0,set(affinity[:THREADS]))
            budget=Budget(root);states={}
            states['baseline']=variant(root,'baseline',plan,budget)
            summary=summarize(root)
            near=any(abs(p['real'])<1e-3 for tr in summary.get('baseline',{}).get('tracks',{}).values() for p in tr)
            # Harmonic and radial verification are always attempted; map/sponge alternatives
            # are conditional on obstruction. No automatic physical branch switch is claimed.
            names=['H6','radial_half']
            if states['baseline']['status']!='complete':names+=['map_a3','sponge_wide_weak']
            if near:names+=['N40']
            for name in names:
                try:states[name]=variant(root,name,plan,budget)
                except Exception as exc:
                    write(root/'STOPPED.json',dict(reason=str(exc),next_variant=name));break
            result=summarize(root,plot=True)
            success=all(s['status']=='complete' for s in states.values()) and len(states)==1+len(names)
            write(root/('DONE.json' if success else 'INCOMPLETE.json'),dict(computations_complete=success,
                variants={k:v['status'] for k,v in states.items()},scientific_classification='requires review of tracks, singular diagnostics and resolution checks',
                no_automatic_Hopf_claim=True,summary=str(root/'SUMMARY.json')))
            for stale in ('DONE.json','INCOMPLETE.json','FAILED.json'):
                if stale != ('DONE.json' if success else 'INCOMPLETE.json'):(root/stale).unlink(missing_ok=True)
            print(json.dumps(dict(computations_complete=success,root=str(root))))
            return 0 if success else 2
        except BaseException as exc:
            write(root/'FAILED.json',dict(failure=f'{type(exc).__name__}: {exc}',resume='same command; crash reservations remain charged'))
            raise



def main():
    parser=argparse.ArgumentParser(description=__doc__);sub=parser.add_subparsers(dest='mode',required=True)
    sub.add_parser('prepare')
    sub.add_parser('check')
    p=sub.add_parser('run');p.add_argument('--root',type=Path,default=PRODUCTION/'run')
    p=sub.add_parser('summarize');p.add_argument('--root',type=Path,default=PRODUCTION/'run');p.add_argument('--plot',action='store_true')
    args=parser.parse_args()
    if args.mode=='prepare':prepare();return 0
    if args.mode=='check':
        plan=read(PREP/'plan.json')
        for records in [plan['code_sha256']]+[p['sha256'] for p in plan['seeds']]:
            for path,value in records.items():
                if sha(path)!=value:raise RuntimeError('prepared input changed: '+path)
        print(json.dumps(dict(ready=True,production_started=False,threads=THREADS,memory_bytes=MEMORY)));return 0
    if args.mode=='summarize':summarize(args.root,args.plot);return 0
    return run(args.root)


if __name__=='__main__':raise SystemExit(main())
