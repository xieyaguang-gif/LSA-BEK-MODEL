#!/usr/bin/env python3
"""Isolated S1c radial-resolution validation and independent resumable Ro jobs."""
from __future__ import annotations
import argparse, contextlib, ctypes, fcntl, gc, hashlib, json, math, os, resource
from pathlib import Path
import signal, sys, time, traceback
sys.dont_write_bytecode = True
# Initialize both BLAS libraries with four threads before legacy imports set env=1.
# Raising an already initialized one-thread pool segfaults in threaded zgemv/SuperLU.
THREADS = 4
for _key in ('OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS','BLIS_NUM_THREADS',
             'NUMEXPR_NUM_THREADS','NUMEXPR_MAX_THREADS','VECLIB_MAXIMUM_THREADS'):
    os.environ[_key] = str(THREADS)
import numpy as np
import scipy.linalg
import scipy.sparse.linalg
sys.path.insert(0, str(Path(__file__).resolve().parent))
import roscan_linear_s1b as previous
legacy = previous.legacy
ROOT, OLD = legacy.ROOT, previous.OLD
PREVIOUS_OUT = previous.OUT
OUT = OLD.parent / 'S1c'
ROS = (-.4, -.3, -.2, -.1)
PREVIOUS_CODE, LEGACY_CODE = previous.CODE, previous.LEGACY_CODE
CODE = hashlib.sha256(Path(__file__).read_bytes() + bytes.fromhex(PREVIOUS_CODE)).hexdigest()
dump, tag, sha = legacy.dump, legacy.tag, previous.sha
BASE_PLAN = previous.plan_row
BASELINE_PATH = OUT/'baseline_sha256.json'


def thread_limits():
    """Audit pools initialized at four; never resize legacy one-thread pools."""
    for key in ('OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS',
                'BLIS_NUM_THREADS','NUMEXPR_NUM_THREADS','NUMEXPR_MAX_THREADS',
                'VECLIB_MAXIMUM_THREADS'):
        os.environ[key] = str(THREADS)
    pools = []
    paths = {line.split()[-1] for line in Path('/proc/self/maps').read_text().splitlines()
             if 'openblas' in line and '/' in line}
    for path in sorted(paths):
        lib = ctypes.CDLL(path)
        for prefix, suffix in [('openblas_',''),('openblas_','64_'),
                               ('scipy_openblas_',''),('scipy_openblas_','64_')]:
            setter, getter = prefix+'set_num_threads'+suffix, prefix+'get_num_threads'+suffix
            if hasattr(lib,setter) and hasattr(lib,getter):
                actual = int(getattr(lib,getter)())
                if actual != THREADS: raise RuntimeError('BLAS was not initialized at four threads; restart this process')
                pools.append(dict(library=path,threads=actual)); break
        else: raise RuntimeError('Unknown BLAS thread setter: '+path)
    if not pools: raise RuntimeError('No auditable BLAS pool found')
    return pools


def allowed_producers():
    ledger = json.loads((OLD/'source_compatibility.json').read_text())
    if sha(ROOT/ledger['producer_source']) != ledger['producer_source_sha256']:
        raise RuntimeError('Legacy producer snapshot hash mismatch')
    return {LEGACY_CODE, PREVIOUS_CODE, ledger['producer_code'], CODE}


def base_row(ro):
    s0, proof = previous.s0_input(ro)
    scale = s0['R_ca']/previous.SCALE_REFERENCE
    return dict(ro=ro, beta_c=s0['beta_c'], S0_evidence_sha256=proof,
                S0_source=s0['source'], R_ca=s0['R_ca'], scale=scale,
                Rin=400*scale, wavelength=2*math.pi/abs(s0['alpha'][0]),
                absolute_lower_Rout=540*scale, code=CODE, status='ready',
                frequency_convention='negative-Ro raw-to-inertial-v1')


def grid(row,n,dr,anchor,steps=1):
    lower, upper = row['Rout_scan_range']
    intervals = math.ceil((upper-row['Rin'])/dr)
    factor = intervals/anchor['N40_intervals']
    rss = anchor['peak_RSS_GB']*factor*((n+1)/41)**2
    seconds = anchor['pipeline_seconds']*factor*((n+1)/41)**3*max(1,steps/82)
    return dict(n=n, dr_requested=dr, intervals_fixed_during_root=intervals,
                dr_actual_range=[(v-row['Rin'])/intervals for v in (lower,upper)],
                maximum_Rout_at_requested_dr=row['Rin']+intervals*dr,
                minimum_points_per_wavelength=row['wavelength']/dr,
                complex_dof=(intervals+1)*4*(n+1),
                estimated_peak_RSS_GB=rss, conservative_address_space_GB=1.5*rss,
                estimated_pipeline_seconds=seconds, conservative_pipeline_seconds=1.65*seconds,
                execution_allowed=1.5*rss<200)


def checkpoint(path, final_row=None, exact_grid=None):
    path = Path(path)
    meta = path.with_name('neutral.json') if path.name=='neutral_state.npz' else path.with_suffix('.json')
    rec = json.loads(meta.read_text())
    if rec.get('code') not in allowed_producers(): raise ValueError('Unregistered producer')
    if final_row is None:
        row = dict(ro=rec['ro'], beta_c=rec['beta'], scale=rec['scale'], Rin=rec['Rin'],
                   S0_evidence_sha256=rec['S0_evidence_sha256'])
        rounded = round(rec['ro'],1)
        if abs(rounded-rec['ro'])<1e-8:
            _, proof = previous.s0_input(rounded)
            if proof != row['S0_evidence_sha256']: raise ValueError('Stale S0 seed')
    else: row=final_row
    rec,state = previous.read_checkpoint(path,row,rec['code'],exact_grid)
    if not previous.geometry(rec['Rout'],rec['scale'])['passed']:
        raise ValueError('Checkpoint inlet/sponge geometry')
    return rec,state


def source_candidates(ro):
    candidates=[]
    for stage in [OLD, PREVIOUS_OUT]:
        for donor in [-1.,-.5,-.4,-.3,-.2]:
            dest=stage/tag(donor)
            for directory in ['N32','N40','seed']:
                for meta in (dest/directory).glob('*.json'):
                    path=meta.with_name('neutral_state.npz') if meta.name=='neutral.json' else meta.with_suffix('.npz')
                    if not path.exists(): continue
                    try:
                        rec=json.loads(meta.read_text())
                        if rec.get('code') not in allowed_producers() or 'residual' not in rec: continue
                        s0=legacy.s0row(ro)
                        steps=max(abs(rec['ro']-ro)/.025,abs(rec['beta']-s0['beta_c'])/.5)
                        # Prefer exact-Ro N32 or a more advanced homotopy; interpolation always re-solves.
                        candidates.append((steps, 0 if rec['n']==32 else 1,
                                           abs(rec['Rout']/rec['scale']-680),str(path)))
                    except (ValueError,KeyError,OSError): continue
    return [Path(c[-1]) for c in sorted(candidates)]


def seed_stage(row):
    dest=legacy.OUT/tag(row['ro'])/'seed';dest.mkdir(parents=True,exist_ok=True)
    progress=dest/'progress.json'; state=None; rejected=[]
    if progress.exists():
        saved=json.loads(progress.read_text())
        if saved.get('code')==CODE and saved.get('input_signature')==row['input_signature']:
            rec,state=checkpoint(saved['state_path'])
            if rec['code']!=CODE: raise RuntimeError('Resume producer mismatch')
    if state is None:
        for path in source_candidates(row['ro']):
            try:
                rec,state=checkpoint(path)
                dump(dest/'source_audit.json',dict(path=str(path.relative_to(ROOT)),
                     json_sha256=sha(path.with_name('neutral.json') if path.name=='neutral_state.npz' else path.with_suffix('.json')),
                     npz_sha256=sha(path),producer=rec['code'],ro=rec['ro'],beta=rec['beta'],
                     decision='initial guess only; new code/grid/root are solved',rejected=rejected))
                break
            except (ValueError,KeyError,OSError) as exc: rejected.append(dict(path=str(path),error=str(exc)))
    if state is None: raise RuntimeError('No validated independent continuation seed')
    if abs(state['ro']-row['ro'])<1e-9 and state['beta']==row['beta_c']: return state
    start=dict(state);target_s=row['scale']; dest_r=row['Rout_initial']/target_s
    start_r=start['rout']/start['scale']
    steps=max(1,math.ceil(abs(row['ro']-start['ro'])/.025),math.ceil(abs(row['beta_c']-start['beta'])/.5))
    intervals=row['grids'][0]['intervals_fixed_during_root'];n=row['grids'][0]['n']
    def bridge(old,ro,beta,scale,rout,depth=0):
        legacy.check_budget(); guess=dict(old)
        guess['radii']=old['radii']*(scale/old['scale']);guess['rout']=old['rout']*(scale/old['scale'])
        old_disk=legacy.rotation_rates(old['ro'])[0];disk=legacy.rotation_rates(ro)[0]
        guess['omega']=beta*disk+(old['omega']-old['beta']*old_disk)*(abs(ro)*scale)/(abs(old['ro'])*old['scale'])
        guess['source']=old['source']+f'|S1c_transform_{ro:.9f}_{beta:.9f}_{scale:.9f}'
        prefix=dest/f'ro{ro:.9f}_beta{beta:.9f}_scale{scale:.9f}_Rout{rout:.9f}'
        try: rec,new=legacy.solve(ro,beta,n,intervals,rout,scale,guess,prefix)
        except legacy.TrackingError:
            if depth>=10: raise
            mid=bridge(old,(old['ro']+ro)/2,(old['beta']+beta)/2,(old['scale']+scale)/2,(old['rout']+rout)/2,depth+1)
            return bridge(mid,ro,beta,scale,rout,depth+1)
        dump(progress,dict(code=CODE,input_signature=row['input_signature'],state_path=new['source']))
        return new
    for t in np.linspace(0,1,steps+1)[1:]:
        ro=start['ro']+t*(row['ro']-start['ro']);beta=start['beta']+t*(row['beta_c']-start['beta'])
        scale=start['scale']+t*(target_s-start['scale']);rout=(start_r+t*(dest_r-start_r))*scale
        state=bridge(state,ro,beta,scale,rout)
    return state


def validation_row(ro,dr,n):
    row=base_row(ro); reference=json.loads((OLD/tag(ro)/f'N{n}/neutral.json').read_text())
    center=reference['Rout'];scale=row['scale']
    row.update(Rout_initial=center,Rout_scan_range=[max(540*scale,center-3*scale),center+3*scale],
               convergence='single-grid radial validation against existing N32 and N40',
               input_signature=hashlib.sha256(f'{CODE}:{ro}:{dr}:{n}:{row["S0_evidence_sha256"]}'.encode()).hexdigest())
    row['grids']=[grid(row,n,dr,previous.calibration())]
    row['grids'][0]['process_address_space_cap_GB']=100
    return row


def compare_spectrum(ro,result):
    ref=json.loads((OLD/tag(ro)/'spectrum.json').read_text())
    target=[]
    for c in result['candidates']:
        if c['residual']>=1e-7: continue
        nearest=min(ref['candidates'],key=lambda a:abs(complex(a['omega']-c['omega'],a['sigma']-c['sigma'])))
        target.append(dict(omega=c['omega'],sigma=c['sigma'],
                           nearest_reference_omega=nearest['omega'],nearest_reference_sigma=nearest['sigma'],
                           delta_omega=abs(nearest['omega']-c['omega']),delta_sigma=abs(nearest['sigma']-c['sigma'])))
    return dict(scope='nearest finite-band eigenvalues at each grid neutral radius; reference uses N40; not an exhaustive spectral claim',
                reference_sha256=sha(OLD/tag(ro)/'spectrum.json'),matches=target,
                max_delta_sigma=max((x['delta_sigma'] for x in target),default=None),
                max_delta_omega=max((x['delta_omega'] for x in target),default=None))


def validate(ro,dr,n):
    row=validation_row(ro,dr,n);start=time.perf_counter()
    seed=seed_stage(row)
    try:
        grids,states=previous.neutral_stage(row,seed)
    except legacy.TrackingError as exc:
        # A failed identity/competitor gate defines no trustworthy neutral root.
        # Retain the failure and independently evaluate the spectrum at the known
        # fine-grid neutral radius, without inventing a coarse neutral/r_end.
        dest=legacy.OUT/tag(ro)
        reference=json.loads((OLD/tag(ro)/'summary.json').read_text())
        _,fine=checkpoint(OLD/tag(ro)/f'N{n}/neutral_state.npz')
        spec=legacy.spectrum(row,fine)
        reliable=[c for c in spec['candidates'] if c['residual']<1e-7]
        near=min(reliable,key=lambda c:abs(c['omega']-fine['omega']))
        spacing=(fine['rout']-row['Rin'])/row['grids'][0]['intervals_fixed_during_root']
        result=dict(code=CODE,ro=ro,n=n,dr_requested=dr,dr_actual=spacing,
            points_per_wavelength=row['wavelength']/spacing,rout_c=None,r_end=None,
            omega_Omega_d=None,omega_BEK=None,sigma=near['sigma'],residual=near['residual'],
            correlation=near['correlation'],delta_to_refined=dict(Rout_c=None,r_end=None,omega_Omega_d=None),
            delta_r_end_to_N32=None,spectrum_passed=spec['passed'],
            diagnostic_status='rejected_unresolved_neutral_identity',passed=False,
            rejection_reason=str(exc),spectrum_evaluated_Rout=fine['rout'],
            spectrum_max_sigma=max(c['sigma'] for c in reliable),
            sigma_definition='nearest-frequency eigenvalue at the refined reference neutral radius; not a coarse neutral residual',
            spectrum_comparison=compare_spectrum(ro,spec),seconds=time.perf_counter()-start,
            peak_RSS_GB=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss*1024/1e9,
            reference_sha256=sha(OLD/tag(ro)/'summary.json'))
        dump(dest/'validation.json',result)
        print(json.dumps({k:v for k,v in result.items() if k!='spectrum_comparison'}),flush=True)
        return
    spec=legacy.spectrum(row,states[-1]);end=legacy.radial_end(row,n,states[-1])
    dest=legacy.OUT/tag(ro); dump(dest/f'rend_N{n}.json',end)
    reference=json.loads((OLD/tag(ro)/'summary.json').read_text())
    coarse=json.loads((OLD/tag(ro)/'rend_N32.json').read_text())
    delta=dict(Rout_c=abs(grids[0]['Rout']-reference['rout_c']),
               r_end=None if end['r_end'] is None else abs(end['r_end']-reference['r_end']),
               omega_Omega_d=abs(grids[0]['omega_Omega_d']-reference['global_omega_Omega_d']))
    passed=(delta['Rout_c']<.5 and delta['r_end'] is not None and delta['r_end']<.5
            and delta['omega_Omega_d']<.05 and abs(grids[0]['sigma'])<1e-6
            and grids[0]['residual']<1e-7 and spec['passed'] and end['status']=='numerically_validated')
    record=dict(code=CODE,ro=ro,n=n,dr_requested=dr,dr_actual=grids[0]['dr_actual'],
                points_per_wavelength=row['wavelength']/grids[0]['dr_actual'],
                rout_c=grids[0]['Rout'],r_end=end['r_end'],omega_Omega_d=grids[0]['omega_Omega_d'],
                omega_BEK=grids[0]['omega'],sigma=grids[0]['sigma'],residual=grids[0]['residual'],
                correlation=grids[0]['correlation'],delta_to_refined=delta,
                delta_r_end_to_N32=abs(end['r_end']-coarse['r_end']) if end['r_end'] is not None else None,
                spectrum_passed=spec['passed'],spectrum_comparison=compare_spectrum(ro,spec),
                diagnostic_status=end['status'],passed=bool(passed),seconds=time.perf_counter()-start,
                peak_RSS_GB=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss*1024/1e9,
                reference_sha256=sha(OLD/tag(ro)/'summary.json'))
    dump(dest/'validation.json',record)
    print(json.dumps({k:v for k,v in record.items() if k!='spectrum_comparison'}),flush=True)


def selection():
    p=OUT/'resolution_selection.json'
    result=json.loads(p.read_text())
    if result.get('code')!=CODE or not result['passed']: raise RuntimeError('Resolution selection stale/incomplete')
    for source in result['evidence']:
        if sha(ROOT/source['path'])!=source['sha256']: raise RuntimeError('Resolution evidence changed')
    return result


def plan_row(ro):
    row=base_row(ro);old=BASE_PLAN(ro);sel=selection()
    for key in ['Rout_initial','Rout_scan_range','absolute_lower_Rout','sponge','inlet_damping']:
        row[key]=old[key]
    candidates=source_candidates(ro)
    if not candidates: raise RuntimeError('No independent seed available')
    rec,_=checkpoint(candidates[0]); steps=max(1,math.ceil(abs(rec['ro']-ro)/.025),math.ceil(abs(rec['beta']-row['beta_c'])/.5))
    row.update(continuation_source=dict(path=str(candidates[0].relative_to(ROOT)),ro=rec['ro'],beta=rec['beta'],estimated_steps=steps),
               points_per_wavelength=sel['minimum_points_per_wavelength'],
               convergence='N32 coarse and N40 half-dr; both required',
               geometry_at_predicted_root=previous.geometry(row['Rout_initial'],row['scale']),
               input_signature=hashlib.sha256(json.dumps([CODE,ro,row['S0_evidence_sha256'],sel],sort_keys=True).encode()).hexdigest())
    dr=row['wavelength']/row['points_per_wavelength']
    row['grids']=[grid(row,32,dr,previous.calibration(),steps),grid(row,40,dr/2,previous.calibration(),1)]
    row['estimated_total_seconds']=sum(g['estimated_pipeline_seconds'] for g in row['grids'])
    row['conservative_total_seconds']=sum(g['conservative_pipeline_seconds'] for g in row['grids'])
    row['launch_by_executor']=False
    row['memory_policy']='per-Ro lock; BLAS 4; task AS caps sum <=550 GB, each <=200 GB; S2 reservation 400 GB'
    return row


def prepare():
    OUT.mkdir(parents=True,exist_ok=True)
    trials=[]
    for dr in [.75,1.,1.5]:
        p=OUT/'validation'/f'dr_{dr:g}'/tag(-.5)/'validation.json'
        d=json.loads(p.read_text());
        if d['code']!=CODE: raise RuntimeError('Stale validation output')
        trials.append((dr,d,p))
    regpath=OUT/'validation/regression_dr_1'/tag(-1.)/'validation.json';reg=json.loads(regpath.read_text())
    if reg['code']!=CODE or not reg['passed']: raise RuntimeError('Ro=-1 dr=1 regression did not meet tolerance')
    good=[(dr,d,p) for dr,d,p in trials if d['passed']]
    if not good: raise RuntimeError('No tested radial spacing satisfies all thresholds')
    dr,d,chosen=max(good,key=lambda x:x[0]);wavelength=base_row(-.5)['wavelength']
    evidence=[dict(path=str(p.relative_to(ROOT)),sha256=sha(p)) for p in [*(p for _,_,p in trials),regpath]]
    dump(OUT/'resolution_selection.json',dict(code=CODE,passed=True,minimum_points_per_wavelength=wavelength/dr,
        selected_reference_dr=dr,reference_ro=-.5,criterion='Rout_c<0.5; r_end<0.5; omega_d<0.05; neutral/residual, branch and finite-spectrum gates',
        regression_n=40,regression_reason='594.19 is the N40 reference; pre-existing N32 vertical error is 0.71',evidence=evidence))
    rows=[plan_row(ro) for ro in ROS]
    # Limits constrain allocated virtual address space, conservatively above estimated peak RSS.
    caps=[min(200,max(60,math.ceil(1.15*max(g['conservative_address_space_GB'] for g in row['grids'])))) for row in rows]
    if any(cap>200 for cap in caps) or sum(caps)>550 or sum(caps)+400>1000:
        raise RuntimeError(f'Parallel hard caps exceed resource envelope: {caps}')
    for row,cap in zip(rows,caps):
        row['process_address_space_cap_GB']=cap
        for g in row['grids']:
            g['process_address_space_cap_GB']=cap
            if not g['execution_allowed']: raise RuntimeError('Grid exceeds process memory envelope')
        dest=OUT/tag(row['ro']);dest.mkdir(exist_ok=True)
        # A separate cache avoids any writes into S1/S1b.
        flow=legacy.solve_bek(row['ro'],zmax=40,tol=1e-10,samples=16001)
        np.savez_compressed(dest/'baseflow.npz',z=flow.z,state=flow.state,ro=row['ro'],source_hash=CODE)
    plan=dict(schema='roscan-linear-s1c-v1',code=CODE,rows=rows,resource_anchor=previous.calibration(),
              parallel_threads=16,per_process_threads=4,total_process_AS_caps_GB=sum(caps),S2_reservation_GB=400,
              max_S1c_GB=550,max_all_computations_GB=1000,
              estimated_parallel_seconds=max(r['estimated_total_seconds'] for r in rows),
              conservative_parallel_seconds=max(r['conservative_total_seconds'] for r in rows),
              estimate_note='single-thread S1b empirical anchor retained conservatively; no assumed 4x speedup; Ro homotopy uses existing independent partial seeds')
    dump(OUT/'plan.json',plan);report(plan,trials,reg)
    pending=(f'S1c short validation complete; long pipelines are prepared and MUST be launched by supervisor.\n'
             f'Working directory: {ROOT}\nLaunch: bash results/ro_scan_20261101/S1c/run_parallel.sh\n'
             f'Four independent Ro processes, BLAS 4 each (total 16); per-Ro flock and restart checkpoints.\n'
             f'Hard address-space caps GB: {caps}; total {sum(caps)} <=550; S2 reserve 400 GB; combined <=1000.\n'
             f'Resolution: at least {wavelength/dr:.3f} points per local S0 wavelength; N40 dr=N32 dr/2.\n'
             f'No S1c long calculation was launched by this executor.\n'
             f'Resume this dedicated S1c session after all jobs finish, or on failure; inspect S1c/parallel_completion.json,\n'
             f'ro_*/summary.json, stage_failure.json, spectrum.json, rend_N*.json, logs/*.resources.txt.\n'
             f'run_parallel.sh joins all four PIDs once, retains failures, and exits nonzero unless every result passes.\n'
             f'S0b is complete; no S2 calculation was launched. Original scripts/results remain read-only.\n')
    dump(OUT/'pending_superseded.json',dict(previous_text=(OUT.parent/'PENDING_SUPERVISOR').read_text()))
    (OUT.parent/'PENDING_SUPERVISOR').write_text(pending)


def report(plan,trials,reg):
    lines=['# S1c radial-grid validation and parallel launch plan','',
           '| Ro | N | requested dr | points/wavelength | Rout_c | r_end | omega_d | dRout | dr_end | domega_d | pass |',
           '|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|:---:|']
    def fmt(value): return 'unresolved' if value is None else f'{value:.6f}'
    for d in [*(d for _,d,_ in trials),reg]:
        delta=d['delta_to_refined']
        lines.append(f"| {d['ro']} | {d['n']} | {d['dr_requested']} | {d['points_per_wavelength']:.2f} | {fmt(d['rout_c'])} | {fmt(d['r_end'])} | {fmt(d['omega_Omega_d'])} | {fmt(delta['Rout_c'])} | {fmt(delta['r_end'])} | {fmt(delta['omega_Omega_d'])} | {d['passed']} |")
    sel=selection();lines+=['',f"Selected minimum {sel['minimum_points_per_wavelength']:.3f} points per local S0 wavelength (2*pi/alpha_r). N40 uses half the N32 spacing.",
        'Ro=-0.5 trials use N32 and compare to the existing N40/dr~0.25 refined reference. The N32 baseline r_end=712.42849 already differs by 0.41264 from N40=712.84113.',
        'Ro=-1 regression uses N40 to compare to the N40 594.19288 reference: the existing N32 value 594.90131 has a separate wall-normal discretization error.',
        'Accepted trials recompute the neutral root, independent five-center spectrum and full cylindrical saddle/Briggs r_end diagnostic. Rejected grids retain the failed branch gate and a five-center spectrum at the refined reference radius; no unsupported neutral or r_end value is assigned. Finite spectrum matching is reported in validation.json.',
        '', '| Ro | N32/N40 intervals | coarse/fine dr | AS cap GB | nominal/upper hours | seed continuation steps |',
        '|---:|---:|---:|---:|---:|---:|']
    for row in plan['rows']:
        a,b=row['grids'];lines.append(f"| {row['ro']} | {a['intervals_fixed_during_root']}/{b['intervals_fixed_during_root']} | {a['dr_requested']:.4f}/{b['dr_requested']:.4f} | {row['process_address_space_cap_GB']} | {row['estimated_total_seconds']/3600:.2f}/{row['conservative_total_seconds']/3600:.2f} | {row['continuation_source']['estimated_steps']} |")
    lines+=['',f"Total enforced AS caps {plan['total_process_AS_caps_GB']} GB; S2 reservation 400 GB. Four workers x four BLAS threads =16.",
            'Domains, inlet and outlet damping follow the unchanged physical scaling; only radial interval count changes. Fixed intervals within each root search.',
            'Each Ro starts from an existing independently checked S1/S1b checkpoint; no worker waits for another new Ro result. Resuming preserves source hashes and exact mesh configuration.',
            'The cost anchor is measured ~50 GB/70 min at Ro=-0.5, N40; no 4x BLAS speedup is assumed. Homotopy continuation and allocator fill-in remain extrapolation risks. The 1.5x conservative memory estimate must lie below each enforced AS cap; extra 15% headroom is used where it fits under 200 GB.',
            'Long jobs have not been launched. Supervisor command: bash results/ro_scan_20261101/S1c/run_parallel.sh']
    (OUT/'REPORT.md').write_text('\n'.join(lines)+'\n')


def run_all(ro):
    plan=json.loads((OUT/'plan.json').read_text())
    if plan['code']!=CODE: raise RuntimeError('Plan source changed')
    row=next(r for r in plan['rows'] if r['ro']==ro)
    # Never launch from a dynamic plan that differs from the supervisor-approved artifact.
    now=plan_row(ro)
    if row['input_signature']!=now['input_signature'] or row['grids'][0]['intervals_fixed_during_root']!=now['grids'][0]['intervals_fixed_during_root']:
        raise RuntimeError('Prepared physics/grid plan changed')
    start=time.perf_counter();seed=seed_stage(row);grids,states=previous.neutral_stage(row,seed)
    spec=legacy.spectrum(row,states[-1]);ends=[]
    for g,state in zip(row['grids'],states):
        end=legacy.radial_end(row,g['n'],state);ends.append(end)
        dump(OUT/tag(ro)/f"rend_N{g['n']}.json",end)
    result=legacy.make_summary(row,grids,ends,spec,time.perf_counter()-start)
    result['grid_difference_omega_Omega_d']=abs(grids[1]['omega_Omega_d']-grids[0]['omega_Omega_d'])
    diffs=[result['grid_difference_Rout_c'],result['grid_difference_r_end'],result['grid_difference_omega_Omega_d']]
    convergence=all(v is not None and v<t for v,t in zip(diffs,[.5,.5,.05]))
    result['two_grid_convergence_passed']=convergence
    if not convergence: result['status']='partial_grid_convergence'
    result['points_per_wavelength']=row['points_per_wavelength']
    result['threads']=THREADS
    dump(OUT/tag(ro)/'summary.json',result)
    if result['status']!='numerically_validated': raise RuntimeError('Pipeline evidence/convergence unresolved')


def baseline(create=False):
    if create:
        paths=[*OLD.rglob('*'),*PREVIOUS_OUT.rglob('*'),*legacy.S0.rglob('*'),
               ROOT/'examples/roscan_linear.py',ROOT/'examples/roscan_linear_s1b.py']
        records={str(p.relative_to(ROOT)):sha(p) for p in paths if p.is_file() and p.suffix not in ['.lock','.log'] and not p.name.endswith('.pid')}
        dump(BASELINE_PATH,records);return len(records)
    records=json.loads(BASELINE_PATH.read_text());changed=[]
    for rel,expected in records.items():
        p=ROOT/rel
        if not p.exists() or sha(p)!=expected: changed.append(rel)
    return dict(files=len(records),changed=changed,passed=not changed)


# Reuse the unchanged physical operator and S1b bounded root search.
previous.CODE=CODE
previous.OUT=OUT
legacy.CODE=CODE
legacy.OUT=OUT
legacy.plan_row=base_row
previous.read_checkpoint_original=previous.read_checkpoint


def alarm_handler(*_): raise legacy.BudgetExceeded('Hard wall budget reached; restart checkpoints retained')


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('stage',choices=['validate','prepare','all','baseline','audit'])
    p.add_argument('--ro',type=float);p.add_argument('--dr',type=float);p.add_argument('--n',type=int,default=32)
    p.add_argument('--budget',type=int,default=3300);p.add_argument('--supervisor-long',action='store_true')
    a=p.parse_args();OUT.mkdir(parents=True,exist_ok=True)
    pools=thread_limits()
    if a.stage=='baseline': print('baseline_files',baseline(True));return
    if a.stage=='audit': print(json.dumps(baseline()));return
    if a.stage=='prepare': prepare();return
    if a.ro is None: p.error('--ro required')
    if a.stage=='validate':
        if a.budget>3300 or a.budget<=0 or a.dr is None or a.dr<=0:p.error('validation: positive dr, budget <=3300')
        legacy.OUT=OUT/'validation'/('regression_dr_1' if a.ro==-1 else f'dr_{a.dr:g}')
        previous.OUT=legacy.OUT
        if a.ro not in [-.5,-1.]:p.error('validation Ro must be -0.5 or -1')
        cap=100
    else:
        if not a.supervisor_long or a.ro not in ROS:p.error('all requires supervisor-long and low Ro')
        data=json.loads((OUT/'plan.json').read_text())
        row=next(r for r in data['rows'] if r['ro']==a.ro);cap=row['process_address_space_cap_GB']
        if a.budget<=0:p.error('positive budget required')
    _,hard=resource.getrlimit(resource.RLIMIT_AS)
    resource.setrlimit(resource.RLIMIT_AS,(min(int(cap*1e9),hard) if hard>0 else int(cap*1e9),hard))
    dest=legacy.OUT/tag(a.ro);dest.mkdir(parents=True,exist_ok=True)
    with (dest/'pipeline.lock').open('a') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        dump(dest/'runtime.json',dict(code=CODE,pid=os.getpid(),blas=pools,AS_cap_GB=cap,budget=a.budget))
        signal.signal(signal.SIGALRM,alarm_handler);signal.alarm(a.budget)
        legacy.DEADLINE=time.monotonic()+a.budget
        try:
            if a.stage=='validate':validate(a.ro,a.dr,a.n)
            else:run_all(a.ro)
        except Exception as exc:
            dump(dest/'stage_failure.json',dict(code=CODE,error=str(exc),traceback=traceback.format_exc()))
            raise
        finally:signal.alarm(0)


if __name__=='__main__':main()
