#!/usr/bin/env python3
"""S1b: isolated low-Ro window correction; legacy S1 is read-only evidence."""
from __future__ import annotations
import argparse
import hashlib
import json
import math
import os
from pathlib import Path
import sys
import time
import traceback

for _key in ('OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS',
             'BLIS_NUM_THREADS', 'NUMEXPR_NUM_THREADS', 'NUMEXPR_MAX_THREADS',
             'VECLIB_MAXIMUM_THREADS'):
    os.environ[_key] = '1'
os.environ['PYTHONDONTWRITEBYTECODE'] = '1'
sys.path.insert(0, str(Path(__file__).resolve().parent))
import roscan_linear as legacy
import numpy as np

ROOT = legacy.ROOT
OLD = legacy.OUT
OUT = OLD.parent / 'S1b'
ROS = (-.4, -.3, -.2, -.1)
SCALE_REFERENCE = 507.389019046
LEGACY_CODE = legacy.CODE
CODE = hashlib.sha256(Path(__file__).read_bytes() + bytes.fromhex(LEGACY_CODE)).hexdigest()
tag, dump = legacy.tag, legacy.dump
CONFIG_FIELDS = ('ro', 'beta', 'n', 'intervals', 'Rin', 'Rout', 'scale',
                 'boundary', 'sponge', 'inlet', 'code', 'S0_code',
                 'S0_evidence_sha256', 'frequency_convention',
                 'seed_source', 'seed_omega', 'seed_sigma')


def sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(4 * 1024 * 1024), b''):
            h.update(block)
    return h.hexdigest()


def s0_input(ro):
    row = legacy.s0row(ro)
    if not row or row.get('status') != 'numerically_validated':
        raise RuntimeError(f'Ro={ro}: current validated S0 input required')
    return row, sha(ROOT / row['source'])


def fit_reference():
    points = []
    for ro in (-1., -.9, -.8, -.7, -.6, -.5):
        s0, proof = s0_input(ro)
        path = OLD / tag(ro) / 'summary.json'
        result = json.loads(path.read_text())
        if (result['status'] != 'numerically_validated'
                or result['S0_code'] != legacy.S0CODE
                or result['S0_evidence_sha256'] != proof
                or result['beta'] != s0['beta_c']):
            raise RuntimeError(f'Invalid fit evidence: {path}')
        points.append(dict(ro=ro, R_ca=s0['R_ca'], Rout_c=result['rout_c'],
                           ratio=result['rout_c'] / s0['R_ca'],
                           source=str(path.relative_to(ROOT)), sha256=sha(path)))
    x = np.array([abs(p['ro']) for p in points])
    y = np.array([p['ratio'] for p in points])
    coeff = np.polyfit(x, y, 2)
    return dict(model='quadratic least-squares Rout_c/R_ca versus abs(Ro)',
                coefficients_descending=coeff.tolist(), points=points,
                training_ratio_RMSE=float(np.sqrt(np.mean((np.polyval(coeff, x)-y)**2))),
                purpose='window/initial guess only; each neutral point requires a new solve')


def calibration():
    summary = OLD / tag(-.5) / 'summary.json'
    result = json.loads(summary.read_text())
    reference = json.loads((OLD / tag(-.5) / 'N40/neutral.json').read_text())
    return dict(ro=-.5, peak_RSS_GB=max(50., result['peak_RSS_kb']*1024/1e9),
                pipeline_seconds=max(70.*60, result['seconds']),
                N40_intervals=reference['intervals'], N40=40,
                source=str(summary.relative_to(ROOT)), sha256=sha(summary),
                time_v_source='results/ro_scan_20261101/S1/logs/-0.5_all.resources.txt',
                time_v_sha256=sha(OLD/'logs/-0.5_all.resources.txt'),
                original_continuation_steps=82,
                memory_upper_factor=1.5, time_upper_factor=1.65,
                correction='replace the old 9.34-GiB/51.65-s anchor with measured 50-GB/70-min pipeline')


def geometry(radius, scale):
    inlet_end, sponge_start = 420*scale, radius-100*scale
    return dict(Rout=radius, Rin=400*scale, inlet_damping_end=inlet_end,
                sponge_start=sponge_start,
                Rin_to_sponge_start=sponge_start-400*scale,
                undamped_gap=sponge_start-inlet_end,
                undamped_gap_in_inlet_widths=(sponge_start-inlet_end)/(10*scale),
                passed=bool(sponge_start-inlet_end >= 20*scale-1e-8))


def plan_row(ro):
    if not any(abs(ro-r) < 1e-9 for r in ROS):
        raise ValueError('S1b only accepts Ro=-0.4,-0.3,-0.2,-0.1')
    ro = round(ro, 1)
    s0, proof = s0_input(ro)
    rca = s0['R_ca']; scale = rca/SCALE_REFERENCE
    fit, anchor = fit_reference(), calibration()
    ratio = float(np.polyval(fit['coefficients_descending'], abs(ro)))
    center = ratio*rca; rin = 400*scale
    floor = 540*scale  # outlet onset >=20s beyond zero inlet damping at 420s
    lower, upper = max(.85*center, floor), 1.15*center
    if not floor < center < upper:
        raise RuntimeError('Extrapolated window has no safe inlet/sponge separation')
    predecessor = {-.4:(-.4,127), -.3:(-.3,157), -.2:(-.3,157), -.1:(-.2,218)}[ro]
    steps = max(1, math.ceil(abs(ro-predecessor[0])/.025),
                math.ceil(abs(s0['beta_c']-predecessor[1])/.5))
    homotopy_factor = max(1., steps/anchor['original_continuation_steps'])
    grids = []
    for n, dr in ((32,.5), (40,.25)):
        intervals = math.ceil((upper-rin)/dr)
        length_ratio = intervals/anchor['N40_intervals']
        rss = anchor['peak_RSS_GB']*length_ratio*((n+1)/41)**2
        seconds = anchor['pipeline_seconds']*length_ratio*((n+1)/41)**3*homotopy_factor
        upper_gb = anchor['memory_upper_factor']*rss
        grids.append(dict(n=n, dr_requested=dr,
                          intervals_fixed_during_root=intervals,
                          dr_actual_range=[(r-rin)/intervals for r in (lower, upper)],
                          maximum_Rout_at_requested_dr=rin+intervals*dr,
                          complex_dof=(intervals+1)*4*(n+1),
                          estimated_peak_RSS_GB=rss,
                          conservative_address_space_GB=upper_gb,
                          estimated_pipeline_seconds=seconds,
                          conservative_pipeline_seconds=seconds*anchor['time_upper_factor'],
                          execution_allowed=upper_gb < 200,
                          blocked_reason=None if upper_gb < 200 else 'N40 estimate exceeds 200-GB per-process cap'))
    active = [g for g in grids if g['execution_allowed']]
    seconds = sum(g['estimated_pipeline_seconds'] for g in active)
    upper_seconds = seconds*anchor['time_upper_factor']
    signature = hashlib.sha256(json.dumps(dict(code=CODE, S0_sha256=proof,
                                  fit=fit), sort_keys=True).encode()).hexdigest()
    return dict(ro=ro, status='ready', code=CODE, beta_c=s0['beta_c'],
                S0_source=s0['source'], S0_evidence_sha256=proof,
                input_signature=signature, R_ca=rca, scale=scale, Rin=rin,
                R_ca_is_estimate=False, Rout_initial=center,
                predicted_Rout_c_over_R_ca=ratio, Rout_scan_range=[lower,upper],
                absolute_lower_Rout=floor,
                window_policy='0.85..1.15 predicted root; lower bound extends automatically to 540s if sigma remains positive; fixed intervals',
                geometry_at_predicted_root=geometry(center,scale),
                geometry_at_lower_window=geometry(lower,scale),
                sponge=dict(distance_to_outlet=100*scale,width=40*scale,strength=28),
                inlet_damping=dict(strength=.8,zero_damping_radius=420*scale,width=10*scale),
                grids=active, all_grid_estimates=grids,
                convergence='two-grid required' if len(active)==2 else 'N32 only; N40 blocked; no grid-convergence claim',
                continuation_source_assumption=dict(ro=predecessor[0], beta=predecessor[1], steps=steps,
                    policy='same-Ro legacy checkpoint first; otherwise nearest completed lower Ro; do not skip an unfinished predecessor'),
                estimated_total_seconds=seconds, conservative_total_seconds=upper_seconds,
                launch_by_executor=upper_seconds<=3600,
                frequency_convention='negative-Ro raw-to-inertial-v1: Omega=beta*disk-Re*conj(raw); alpha=conj(raw)',
                memory_policy='single thread; shared S1 lock; process AS <=200 GB, task <=400 GB; serial Ro order',
                resource_estimate_limit='empirical mesh/bandwidth scaling; extrapolated cost and allocator/address-space uncertainty use 1.5x memory, 1.65x time; wall budget and RLIMIT_AS enforced')


def read_checkpoint(path, row, producer_code, exact_grid=None):
    path = Path(path); meta_path = path.with_name('neutral.json') if path.name=='neutral_state.npz' else path.with_suffix('.json')
    rec = json.loads(meta_path.read_text())
    config = {k:rec[k] for k in CONFIG_FIELDS}
    if hashlib.sha256(json.dumps(config,sort_keys=True).encode()).hexdigest() != rec['cache_key']:
        raise ValueError('Checkpoint configuration hash mismatch')
    if (rec['code'] != producer_code or rec['S0_code'] != legacy.S0CODE
            or rec['S0_evidence_sha256'] != row['S0_evidence_sha256']
            or abs(rec['ro']-row['ro'])>1e-10 or rec['beta']!=row['beta_c']
            or abs(rec['scale']/row['scale']-1)>1e-10
            or abs(rec['Rin']-row['Rin'])>1e-7
            or rec['residual']>=1e-7 or rec['correlation']<.85):
        raise ValueError('Checkpoint physics/provenance gate failed')
    if exact_grid and (rec['n']!=exact_grid['n'] or rec['intervals']!=exact_grid['intervals_fixed_during_root']):
        raise ValueError('Checkpoint grid mismatch')
    if not np.allclose(rec['sponge'],[rec['Rout']-100*row['scale'],40*row['scale'],28]):
        raise ValueError('Checkpoint sponge mismatch')
    if not np.allclose(rec['inlet'],[420*row['scale'],10*row['scale'],.8]):
        raise ValueError('Checkpoint inlet damping mismatch')
    with np.load(path,allow_pickle=False) as raw:
        if str(raw['code'])!=producer_code or str(raw['cache_key'])!=rec['cache_key']:
            raise ValueError('JSON/NPZ producer hash mismatch')
    state = legacy.load_state(path)
    if (state['q'].shape!=(rec['intervals']+1,4,rec['n']+1)
            or not np.all(np.isfinite(state['q']))
            or not np.all(np.diff(state['radii'])>0)
            or not np.allclose(state['radii'],np.linspace(rec['Rin'],rec['Rout'],rec['intervals']+1))
            or state['ro']!=row['ro'] or state['beta']!=row['beta_c']
            or not np.allclose([state['omega'],state['sigma'],state['scale']],
                               [rec['omega'],rec['sigma'],row['scale']])):
        raise ValueError('Checkpoint state metadata mismatch')
    return rec,state


def legacy_seed(row):
    candidates = []
    for path in (OLD/tag(row['ro'])/'N32').glob('Rout_*.json'):
        rec = json.loads(path.read_text())
        if rec.get('code')==LEGACY_CODE and 'sigma' in rec and path.with_suffix('.npz').exists():
            candidates.append((abs(rec['Rout']-row['Rout_initial']), path.with_suffix('.npz')))
    rejected = []
    for _,path in sorted(candidates):
        try:
            rec,state = read_checkpoint(path,row,LEGACY_CODE)
            return rec,state,dict(path=str(path.relative_to(ROOT)),
                npz_sha256=sha(path), json_sha256=sha(path.with_suffix('.json')),
                decision='initial guess only; new code, grid and root are recomputed',
                old_intervals=rec['intervals'], new_intervals=row['grids'][0]['intervals_fixed_during_root'],
                sigma=rec['sigma'], omega=rec['omega'], residual=rec['residual'],
                correlation=rec['correlation'], Rout=rec['Rout'], rejected=rejected)
        except (ValueError,KeyError,OSError) as exc:
            rejected.append(dict(path=str(path),reason=str(exc)))
    return None,None,dict(decision='no compatible same-Ro checkpoint',rejected=rejected)


def seed_stage(row):
    dest = OUT/tag(row['ro'])/'seed'; dest.mkdir(parents=True,exist_ok=True)
    _,state,audit = legacy_seed(row)
    if state is not None:
        dump(dest/'source_audit.json',audit)
        return state
    progress = dest/'progress.json'
    if progress.exists():
        saved=json.loads(progress.read_text())
        if saved.get('code')==CODE and saved.get('input_signature')==row['input_signature']:
            state=legacy.load_state(saved['state_path'])
    if state is None:
        for prior in (-.3,-.2):
            if prior>=row['ro'] or prior<row['continuation_source_assumption']['ro']-1e-9:
                continue
            prior_row=plan_row(prior)
            path=OUT/tag(prior)/'N32/neutral_state.npz'
            if path.exists():
                _,state=read_checkpoint(path,prior_row,CODE,prior_row['grids'][0])
        if state is None:
            raise RuntimeError('Complete the preceding Ro N32 neutral checkpoint before continuing')
    start=dict(state); target_ro=row['ro']; target_beta=row['beta_c']; target_s=row['scale']
    initial_normalized=start['rout']/start['scale']
    target_normalized=row['Rout_initial']/target_s
    steps=max(1,math.ceil(abs(target_ro-start['ro'])/.025),math.ceil(abs(target_beta-start['beta'])/.5))
    intervals=row['grids'][0]['intervals_fixed_during_root']; records=[]
    def bridge(old,ro,beta,scale,rout,depth=0):
        legacy.check_budget()
        seed=dict(old); seed['radii']=old['radii']*(scale/old['scale'])
        seed['rout']=old['rout']*(scale/old['scale'])
        old_disk=legacy.rotation_rates(old['ro'])[0]; disk=legacy.rotation_rates(ro)[0]
        seed['omega']=beta*disk+(old['omega']-old['beta']*old_disk)*(abs(ro)*scale)/(abs(old['ro'])*old['scale'])
        seed['source']=old['source']+f'|S1b_transform_{ro:.9f}_{beta:.9f}_{scale:.9f}'
        prefix=dest/f'ro{ro:.9f}_beta{beta:.9f}_scale{scale:.9f}_Rout{rout:.9f}'
        try:
            rec,new=legacy.solve(ro,beta,32,intervals,rout,scale,seed,prefix)
        except legacy.TrackingError:
            if depth>=10: raise
            mid=bridge(old,(old['ro']+ro)/2,(old['beta']+beta)/2,
                       (old['scale']+scale)/2,(old['rout']+rout)/2,depth+1)
            return bridge(mid,ro,beta,scale,rout,depth+1)
        records.append(dict(ro=ro,beta=beta,Rout=rout,source=new['source']))
        dump(progress,dict(code=CODE,input_signature=row['input_signature'],state_path=new['source']))
        return new
    for t in np.linspace(0,1,steps+1)[1:]:
        ro=start['ro']+t*(target_ro-start['ro']); beta=start['beta']+t*(target_beta-start['beta'])
        scale=start['scale']+t*(target_s-start['scale'])
        rout=(initial_normalized+t*(target_normalized-initial_normalized))*scale
        state=bridge(state,ro,beta,scale,rout)
    dump(dest/'summary.json',dict(code=CODE,input_signature=row['input_signature'],records=records,
        method='nearest completed predecessor; radius/disk-frequency transform; adaptive Ro/beta continuation; unchanged correlation/residual/competitor gates'))
    return state


def find_bracket(evaluate,row,trace):
    """Walk toward a sign change; a positive lower endpoint expands downward."""
    center=row['Rout_initial']; lower,upper=row['Rout_scan_range']
    floor=row['absolute_lower_Rout']; scale=row['scale']
    seen={}
    def sample(radius):
        value=float(evaluate(float(radius))); seen[float(radius)]=value
        trace.append(dict(Rout=float(radius),sigma=value,
                          below_planned_lower=bool(radius<lower-1e-9)))
        if abs(value)<=1e-8: return (radius,radius)
        keys=sorted(seen)
        changes=[(a,b) for a,b in zip(keys[:-1],keys[1:]) if seen[a]*seen[b]<0]
        if len(changes)>1: raise RuntimeError('Multiple neutral brackets; branch review required')
        return changes[0] if changes else None
    bracket=sample(center)
    if bracket: return bracket
    direction=-1 if seen[center]>0 else 1
    step=max(2*scale,.01*center); radius=center
    for _ in range(40):
        bound=(lower if radius>lower+1e-9 else floor) if direction<0 else upper
        candidate=max(bound,radius-step) if direction<0 else min(bound,radius+step)
        if abs(candidate-radius)<1e-9:
            raise RuntimeError('No sign change before safe inlet/sponge floor or resolved upper bound; evidence retained')
        bracket=sample(candidate)
        if bracket: return bracket
        radius=candidate; step*=1.6
    raise RuntimeError('Bounded adaptive bracket exhausted; evidence retained')


def neutral_stage(row,seed):
    dest=OUT/tag(row['ro']); grids=[]; states=[]
    for grid in row['grids']:
        if not grid['execution_allowed']: raise RuntimeError('Blocked grid cannot run')
        n=grid['n']; intervals=grid['intervals_fixed_during_root']
        gdest=dest/f'N{n}'; gdest.mkdir(parents=True,exist_ok=True)
        cache={}; trace=[]
        for meta in sorted(gdest.glob('Rout_*.json')):
            if not meta.with_suffix('.npz').exists(): continue
            try:
                rec,_=read_checkpoint(meta.with_suffix('.npz'),row,CODE,grid)
                cache[rec['Rout']]=rec
            except (ValueError,KeyError,OSError): pass
        def follow_radius(initial,target,depth=0):
            prefix=gdest/f'Rout_{target:.9f}'
            try: return legacy.solve(row['ro'],row['beta_c'],n,intervals,target,row['scale'],initial,prefix)
            except legacy.TrackingError:
                if depth>=10 or abs(target-initial['rout'])<.02*row['scale']: raise
                mid=.5*(initial['rout']+target)
                _,middle=follow_radius(initial,mid,depth+1)
                return follow_radius(middle,target,depth+1)
        def evaluate(rout):
            nonlocal seed
            legacy.check_budget(); rout=float(rout)
            if not geometry(rout,row['scale'])['passed']:
                raise RuntimeError('Inlet/outlet damping separation gate failed')
            if rout>grid['maximum_Rout_at_requested_dr']+1e-8:
                raise RuntimeError('Requested radius exceeds resolved fixed grid')
            if rout in cache: return cache[rout]['sigma']
            near=min(cache,key=lambda r:abs(r-rout)) if cache else None
            initial=legacy.load_state(gdest/f'Rout_{near:.9f}.npz') if near is not None else seed
            rec,state=follow_radius(initial,rout); cache[rout]=rec; seed=state
            return rec['sigma']
        try:
            bracket=find_bracket(evaluate,row,trace)
            root=bracket[0] if bracket[0]==bracket[1] else legacy.optimize.brentq(evaluate,*bracket,xtol=2e-6)
            evaluate(root); rec=cache[root]; state=legacy.load_state(gdest/f'Rout_{root:.9f}.npz')
            legacy.save_state(gdest/'neutral_state.npz',state,rec['cache_key'])
            dump(gdest/'neutral.json',rec); grids.append(rec); states.append(state); seed=state
        finally:
            dump(gdest/'bracket_trace.json',dict(code=CODE,input_signature=row['input_signature'],
                planned_range=row['Rout_scan_range'],floor=row['absolute_lower_Rout'],trace=trace))
    dump(dest/'neutral_summary.json',dict(code=CODE,grids=grids,
        topology='fixed radial intervals per grid; automatic downward extension with inlet-separation floor',
        convergence=row['convergence']))
    return grids,states


def report(rows=None):
    rows=rows or [plan_row(ro) for ro in ROS]
    text=['# S1b low-Ro window correction','',
          'Legacy S1 sources/results are preserved. All new evidence is written to S1b.',
          'Fit: quadratic Rout_c/R_ca versus |Ro| from six validated S1 points; it supplies initial guesses only.',
          'Memory/time: measured Ro=-0.5 50 GB / 70 min, with 1.5x memory and 1.65x time margins.',
          'Pipeline estimates include the next-predecessor beta-continuation steps; Ro order is mandatory.','',
          '| Ro | Rout initial | scan range | N32 GB upper | N40 GB upper | planned time h (upper) |',
          '|---:|---:|---|---:|---:|---:|']
    for row in rows:
        a,b=row['all_grid_estimates']
        text.append(f"| {row['ro']} | {row['Rout_initial']:.3f} | {row['Rout_scan_range'][0]:.3f}..{row['Rout_scan_range'][1]:.3f} | {a['conservative_address_space_GB']:.1f} | {b['conservative_address_space_GB']:.1f} | {row['estimated_total_seconds']/3600:.2f} ({row['conservative_total_seconds']/3600:.2f}) |")
    text += ['', 'Rin=400s and inlet damping ending at 420s are retained. The absolute floor 540s keeps outlet sponge onset at least 20s (two inlet widths) beyond inlet damping.',
             'At each predicted root the undamped separation exceeds 100s. This geometric check does not establish inlet-independence; changing Rin requires a separate sensitivity run.',
             'Ro=-0.1 N40 is excluded above 200 GB; N32 results remain a single-grid result.',
             'No nonlinear S2 work is included. See verification.json, checkpoint_audit.json, and ../PENDING_SUPERVISOR.']
    (OUT/'REPORT.md').write_text('\n'.join(text)+'\n')


def prepare():
    OUT.mkdir(parents=True,exist_ok=True); rows=[plan_row(ro) for ro in ROS]; audits=[]
    for row in rows:
        dest=OUT/tag(row['ro']); dest.mkdir(exist_ok=True)
        flow=legacy.solve_bek(row['ro'],zmax=40,tol=1e-10,samples=16001)
        np.savez_compressed(dest/'baseflow.npz',z=flow.z,state=flow.state,ro=row['ro'],source_hash=CODE)
        _,_,audit=legacy_seed(row); audits.append(dict(ro=row['ro'],**audit))
    dump(OUT/'plan.json',dict(schema='roscan-linear-s1b-v1',code=CODE,
        legacy_code=LEGACY_CODE,S0_code=legacy.S0CODE,rows=rows,
        fit=fit_reference(),resource_anchor=calibration()))
    dump(OUT/'checkpoint_audit.json',dict(code=CODE,rows=audits)); report(rows)


def run(ro,supervisor_long=False,trial=False):
    row=plan_row(ro); dest=OUT/tag(ro); dest.mkdir(parents=True,exist_ok=True)
    if trial:
        if ro not in (-.4,-.3): raise RuntimeError('Short trial requires compatible same-Ro seed')
        _,state,audit=legacy_seed(row)
        if state is None: raise RuntimeError('No compatible short-trial seed')
        old=json.loads((ROOT/audit['path']).with_suffix('.json').read_text())
        trial_upper=max(30.,old['seconds'])*64*row['grids'][0]['intervals_fixed_during_root']/old['intervals']
        if trial_upper>3300: raise RuntimeError('Short-trial estimate exceeds 55 minutes; supervisor required')
        row={**row,'grids':row['grids'][:1]}
    elif row['conservative_total_seconds']>3600 and not supervisor_long:
        raise RuntimeError('Long pipeline requires supervisor --supervisor-long')
    start=time.perf_counter(); seed=seed_stage(row); grids,states=neutral_stage(row,seed)
    if trial:
        result=dict(code=CODE,ro=ro,status='N32_neutral_trial_only',rout_c=grids[0]['Rout'],
            sigma=grids[0]['sigma'],omega=grids[0]['omega'],seconds=time.perf_counter()-start,
            scope='N32 neutral root; spectrum, r_end and N40 remain pending')
        dump(dest/'trial.json',result); print(json.dumps(result),flush=True); return
    spec=legacy.spectrum(row,states[-1]); ends=[]
    for grid,state in zip(row['grids'],states):
        end=legacy.radial_end(row,grid['n'],state); ends.append(end)
        dump(dest/f"rend_N{grid['n']}.json",end)
    result=legacy.make_summary(row,grids,ends,spec,time.perf_counter()-start)
    if len(grids)<2:
        result.update(status='partial_single_grid',grid_difference_r_end=None,
                      grid_difference_Rout_c=None,grid_difference_omega=None,
                      convergence='N32 only; N40 blocked by >200-GB estimate')
    dump(dest/'summary.json',result); report()


# Reuse the physical operator and its unchanged identity/spectrum/diagnostic gates.
legacy.OUT=OUT
legacy.CODE=CODE
legacy.plan_row=plan_row
legacy.seed_stage=seed_stage
legacy.neutral_stage=neutral_stage
legacy.report=report


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('stage',choices=['prepare','trial','all','report'])
    p.add_argument('--ro',type=float); p.add_argument('--budget',type=float,default=3300)
    p.add_argument('--supervisor-long',action='store_true'); a=p.parse_args()
    if a.budget<=0: p.error('budget must be positive')
    if a.stage=='trial' and a.budget>3300: p.error('short-trial budget must be <=3300 s')
    legacy.DEADLINE=time.monotonic()+a.budget
    OUT.mkdir(parents=True,exist_ok=True)
    if a.stage=='prepare': prepare()
    elif a.stage=='report': report()
    else:
        if a.ro is None: p.error('--ro required')
        try: run(a.ro,a.supervisor_long,a.stage=='trial')
        except Exception as exc:
            dump(OUT/tag(a.ro)/'stage_failure.json',dict(code=CODE,error=str(exc),traceback=traceback.format_exc()))
            raise


if __name__=='__main__': main()
