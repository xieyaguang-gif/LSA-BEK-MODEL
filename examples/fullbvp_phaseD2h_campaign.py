#!/usr/bin/env python3
"""D2h: verified v5 reuse, real-axis-first completion, bounded v6 resources."""
import os
for key in ('OPENBLAS_NUM_THREADS','OMP_NUM_THREADS','MKL_NUM_THREADS','NUMEXPR_NUM_THREADS','BLIS_NUM_THREADS'):
    os.environ[key]='1'
import argparse,copy,fcntl,json,resource,shutil,sys,time,uuid
from contextlib import contextmanager
from pathlib import Path
import numpy as np
import fullbvp_phaseD2g_campaign as g
old=g.old
ROOT,OUT=g.ROOT,g.OUT
RUNS=g.RUNS
SHIFTS=g.SHIFTS
MEMORY_BYTES=400_000_000_000
CONTROLLER_BYTES=2_000_000_000
TOTAL_BYTES=1_000_000_000_000
POINT_BUDGET=14400.
CENTER_BUDGET=4800.
CONCURRENT_RUNS=2
THREADS=16
_saved_parameters=g._saved_parameters
_saved_fingerprints=g.fingerprints
_saved_launch=g.launch_point


def output_path(path,allow_root=False):
    path=Path(path).absolute();canonical=path.resolve()
    if '..' in path.parts or canonical!=path or any(p.is_symlink() for p in (path,*path.parents)):
        raise ValueError('unsafe output path')
    roots=[OUT/'D2e',OUT/'production_v6']
    if allow_root and canonical==roots[1]:return canonical
    if not any(canonical.is_relative_to(r) and canonical!=r for r in roots):
        raise ValueError('D2h writes only descendants of D2e or production_v6')
    return canonical


def fingerprints():
    paths=[Path(__file__),OUT/'production_v6/commands.sh']
    return dict(_saved_fingerprints(),**{str(p.resolve()):old.d2.v1.sha(p) for p in paths})


def source_file(path,point):
    path=Path(path).absolute()
    if path.resolve()!=path or not path.is_relative_to(point) or not path.is_file():
        raise RuntimeError('v5 evidence path escapes its point')
    return path


def v5_bundle(run_name,source):
    point=OUT/'production_v5'/run_name/source
    manifest=json.loads((point/'manifest.json').read_text())
    if manifest['fingerprint']!=old.d2.v1.digest(manifest['params']):
        raise RuntimeError('v5 manifest digest mismatch')
    for path,h in manifest['params']['code_sha256'].items():
        if old.d2.v1.sha(path)!=h:raise RuntimeError('v5 code dependency changed: '+path)
    correction=json.loads((point/'corrected.json').read_text())
    fields=source_file(correction['fields'],point)
    if correction['fingerprint']!=manifest['fingerprint'] or old.d2.v1.sha(fields)!=correction['fields_sha256']:
        raise RuntimeError('v5 corrected field fingerprint/hash mismatch')
    paths=[point/'manifest.json',point/'corrected.json',fields];centers={}
    for i,sigma in enumerate(SHIFTS):
        marker=point/f'center_{i:02d}/DONE.json'
        if not marker.exists():continue
        saved=json.loads(marker.read_text());arrays=source_file(saved['arrays'],point)
        if (not saved.get('complete') or saved['fingerprint']!=manifest['fingerprint']
            or saved['sigma']!=old.d2.v1.zpair(sigma) or saved['k']!=6 or saved['ncv']!=40
            or old.d2.v1.sha(arrays)!=saved['arrays_sha256']):
            raise RuntimeError('v5 completed center settings/hash mismatch')
        paths.extend([marker,arrays]);centers[str(i)]=str(marker)
    return dict(point=str(point),centers=centers,sha256={str(p):old.d2.v1.sha(p) for p in paths})


def parameters(source,n=24,H=4,dr=1.,shifts=SHIFTS,k=6,ncv=40,run_name=None):
    if run_name not in RUNS:raise ValueError('unknown v6 run')
    config=RUNS[run_name]
    if source not in config['sources'] or (n,H,dr)!=(config['n'],config['H'],config['dr']):
        raise ValueError('run resolution/source mismatch')
    if tuple(shifts)!=SHIFTS or (k,ncv)!=(6,40):raise ValueError('fixed five-center k6/ncv40 study')
    p=_saved_parameters(source,n,H,dr,shifts,k,ncv)
    p.update(revision='D2h',run_name=run_name,linear_policy=g.POLICY,linear_accept=g.RHS_ACCEPT,
             center_budget_seconds=CENTER_BUDGET,budget_seconds=POINT_BUDGET,memory_bytes=MEMORY_BYTES,
             positive_half_plane=True,threads=THREADS,import_from=None,reuse_v5=v5_bundle(run_name,source),
             center_order='real-axis first, then +2i,+5i,+10i,+20i',concurrent_runs=CONCURRENT_RUNS)
    p['source_projection_dr']=p.pop('actual_dr')
    previous=json.loads((Path(p['reuse_v5']['point'])/'manifest.json').read_text())['params']
    for key in ('source','n','H','intervals','requested_dr','source_hashes','shifts','k','ncv','linear_policy','pair_accept'):
        if p[key]!=previous[key]:raise RuntimeError('v5/v6 scientific settings mismatch: '+key)
    return p


def checked_copy(source,destination,sha):
    destination=output_path(destination);destination.parent.mkdir(parents=True,exist_ok=True)
    temporary=destination.with_name(destination.name+'.importing')
    shutil.copyfile(source,temporary)
    if old.d2.v1.sha(temporary)!=sha:raise RuntimeError('copied v5 evidence hash mismatch')
    temporary.replace(destination)


def prepare_correction(params,out,deadline):
    out=output_path(out);out.mkdir(parents=True,exist_ok=True);bundle=params['reuse_v5']
    for path,sha in bundle['sha256'].items():
        deadline.check()
        if old.d2.v1.sha(path)!=sha:raise RuntimeError('v5 import evidence changed')
    if not (out/'corrected.json').exists():
        saved=json.loads((Path(bundle['point'])/'corrected.json').read_text())
        fields=Path(saved['fields']);destination=out/'corrections/import_v5/fields.npz'
        with np.load(fields) as data:
            if (int(data['n'])!=params['n'] or int(data['H'])!=params['H'] or not bool(data['accepted'])
                or len(data['radii'])!=params['intervals']+1):raise RuntimeError('v5 corrected NPZ metadata mismatch')
        checked_copy(fields,destination,saved['fields_sha256'])
        saved.update(fingerprint=old.d2.v1.digest(params),fields=str(destination),reused_from_v5=bundle,
                     reused_correction_seconds=saved['correction_seconds'])
        old.write(out/'corrected.json',saved)
    return g.prior.corrected_temporal(params,out,deadline)


def prepare_centers(params,temporal,out,deadline):
    """Recertify every existing vector before deciding which centers need a solve."""
    accepted={};fingerprint=old.d2.v1.digest(params)
    for i,sigma in enumerate(SHIFTS):
        target=out/f'center_{i:02d}';done=target/'DONE.json';deadline.check()
        if done.exists():
            accepted[i]=g.run_center(temporal,None,sigma,target,fingerprint,deadline,k=6,ncv=40)
            continue
        marker=params['reuse_v5']['centers'].get(str(i))
        if marker is None:continue
        start=time.monotonic();previous=json.loads(Path(marker).read_text())
        arrays=Path(previous['arrays'])
        if old.d2.v1.sha(arrays)!=previous['arrays_sha256']:raise RuntimeError('v5 eigenpair hash changed')
        with np.load(arrays) as data:values=data['values'].copy();vectors=data['vectors'].copy()
        if values.shape!=(6,) or vectors.shape!=(temporal.size,6):raise RuntimeError('v5 eigenpair dimensions mismatch')
        if not np.array_equal(values,np.array([complex(*r['value']) for r in previous['eigenpairs']])):
            raise RuntimeError('v5 stored eigenvalues disagree with certificate')
        phase=temporal.neutral();checks=[]
        for value,vector in zip(values,vectors.T):
            deadline.check();checks.append(old.d2.v1.certify(temporal.J,temporal.B,max(temporal.b),value,vector,
                                                           tol=old.PAIR_ACCEPT,phase=phase))
        if not all(r['accepted'] for r in checks):raise RuntimeError('v5 eigenpair fails original-operator recertification')
        destination=target/'attempts/import_v5/eigenpairs.npz'
        checked_copy(arrays,destination,previous['arrays_sha256'])
        result=dict(fingerprint=fingerprint,sigma=params['shifts'][i],complete=True,k=6,ncv=40,
                    arrays=str(destination),arrays_sha256=previous['arrays_sha256'],eigenpairs=checks,
                    residual_accept=old.PAIR_ACCEPT,linear_policy=g.POLICY,linear_accept=g.RHS_ACCEPT,
                    componentwise_is_gate=False,original_resolution_certified=False,exhaustive_stability=False,
                    seconds=time.monotonic()-start,arnoldi_callbacks=0,factorization_count_this_run=0,
                    reused_eigenpairs=True,reused_from_v5=marker,source_result_sha256=old.d2.v1.sha(marker),
                    reused_previous_seconds=previous['seconds'])
        old.write(destination.parent/'result.json',result);old.write(done,result)
        (target/'FAILED.json').unlink(missing_ok=True);accepted[i]=result
    return accepted


def scan_centers(temporal,J,params,out,deadline,accepted):
    results=[];fingerprint=old.d2.v1.digest(params)
    for i,pair in enumerate(params['shifts']):
        if i in accepted:results.append(accepted[i]);continue
        target=out/f'center_{i:02d}'
        try:
            deadline.check()
            limit=old.d2.v1.Deadline(min(params['center_budget_seconds'],deadline.remaining()))
            result=g.run_center(temporal,J,complex(*pair),target,fingerprint,limit,k=params['k'],ncv=params['ncv'])
        except Exception as exc:
            result=dict(fingerprint=fingerprint,sigma=pair,complete=False,failure=f'{type(exc).__name__}: {exc}')
            old.write(target/'FAILED.json',result)
        results.append(result)
    return results


class ResourceFactor(old.d2.PardisoComplex):
    def _call(self,phase,rhs):
        if phase==22:
            rss=old.d2.memory().get('VmRSS',0);estimate=max(0,int(self.iparm[16]))*1024
            projected=rss+int(1.25*estimate)
            self.info['D2h_memory_guard']=dict(rss_bytes=rss,numerical_estimate_bytes=estimate,
                                              projected_bytes=projected,cap_bytes=MEMORY_BYTES)
            if projected>MEMORY_BYTES:
                if self.stats_path:old.write(self.stats_path,self.info)
                raise MemoryError('D2h numerical-factor estimate + RSS exceeds 400 GB')
        result=super()._call(phase,rhs)
        if phase!=-1 and old.d2.memory().get('VmRSS',0)>MEMORY_BYTES:
            raise MemoryError('D2h process RSS exceeds 400 GB')
        return result


def control_limit():
    # Soft controller cap; workers raise it only after acquiring a memory slot.
    resource.setrlimit(resource.RLIMIT_AS,(CONTROLLER_BYTES,MEMORY_BYTES))


@contextmanager
def memory_slot(root=None,blocking=True):
    directory=output_path(root or OUT/'production_v6/.memory_slots');directory.mkdir(parents=True,exist_ok=True)
    fd=None;slot=None
    try:
        for i in range(CONCURRENT_RUNS):
            candidate=os.open(output_path(directory/f'slot_{i}.lock'),os.O_RDWR|os.O_CREAT|os.O_CLOEXEC|os.O_NOFOLLOW,0o600)
            try:fcntl.flock(candidate,fcntl.LOCK_EX|fcntl.LOCK_NB)
            except BlockingIOError:os.close(candidate);continue
            fd=candidate;slot=i;break
        if fd is None:
            if not blocking:raise BlockingIOError('both 400 GB memory slots are occupied')
            fd=os.open(directory/'slot_0.lock',os.O_RDWR|os.O_CLOEXEC|os.O_NOFOLLOW)
            fcntl.flock(fd,fcntl.LOCK_EX);slot=0  # Blocking admission, no status polling.
        yield slot
    finally:
        if fd is not None:os.close(fd)


def worker(job,lock_fd):
    out=output_path(job['out']);params=job['params']
    actual=os.fstat(lock_fd);expected=os.stat(out.parent/'.run.lock')
    if (actual.st_dev,actual.st_ino)!=(expected.st_dev,expected.st_ino):raise RuntimeError('wrong inherited run lock')
    control_limit();start=time.monotonic();deadline=old.d2.v1.Deadline(job['remaining'])
    current=parameters(params['source'],params['n'],params['H'],params['requested_dr'],run_name=params['run_name'])
    if current!=params:raise RuntimeError('worker fingerprint changed')
    with memory_slot() as slot:
        deadline.check();resource.setrlimit(resource.RLIMIT_AS,(MEMORY_BYTES,MEMORY_BYTES))
        old.write(out/'resources.json',dict(memory_slot=slot,worker_address_space_bytes=MEMORY_BYTES,
            controller_address_space_bytes=CONTROLLER_BYTES,concurrent_workers=CONCURRENT_RUNS,
            dispatch_total_address_space_bound=2*MEMORY_BYTES+3*CONTROLLER_BYTES,total_cap_bytes=TOTAL_BYTES))
        temporal,correction=prepare_correction(params,out,deadline)
        accepted=prepare_centers(params,temporal,out,deadline)
        J=None
        if len(accepted)!=len(SHIFTS):
            J,assembly=old.d2.assemble(temporal,deadline);old.write(out/'assembly.json',assembly)
            if np.iscomplexobj(J.data) and np.any(J.data.imag):raise RuntimeError('non-real packed pencil')
        centers=scan_centers(temporal,J,params,out,deadline,accepted)
        result=dict(fingerprint=old.d2.v1.digest(params),complete=all(r['complete'] for r in centers),
            seconds=time.monotonic()-start,correction=correction,centers=centers,
            peak_rss_bytes=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss*1024,
            original_resolution_certified=False,exhaustive_stability=False,positive_half_plane=True,
            perturbations='integer multiples of beta=68 only',reused_centers=sorted(accepted),
            attempted_centers=[i for i in range(len(SHIFTS)) if i not in accepted],assembly_skipped=J is None)
        old.write(out/'worker_result.json',result)
        return 0 if result['complete'] else 1


def launch_point(params,out,command=None,lock_fd=None):
    if command is None:
        if lock_fd is None:raise RuntimeError('worker requires inherited run lock')
        command=[sys.executable,str(Path(__file__).resolve()),'worker','--job',str(output_path(out)/'job.json'),
                 '--lock-fd',str(lock_fd)]
    return _saved_launch(params,out,command=command,lock_fd=lock_fd)


def campaign(args):
    control_limit();return g.campaign(args)


def execute_waves(specs,out,timeout_seconds=28920.):
    if not 1<=len(specs)<=5 or len({name for name,cmd in specs})!=len(specs):raise ValueError('one to five unique runs')
    out=output_path(out);out.mkdir(parents=True,exist_ok=False)
    bound=2*MEMORY_BYTES+3*CONTROLLER_BYTES
    if bound>TOTAL_BYTES:raise RuntimeError('aggregate memory bound exceeds 1 TB')
    old.write(out/'PLAN.json',dict(runs=[name for name,cmd in specs],simultaneous_runs=CONCURRENT_RUNS,
        worker_memory_bytes=MEMORY_BYTES,aggregate_address_space_bound_bytes=bound,total_cap_bytes=TOTAL_BYTES,
        threads_per_run=THREADS,total_compute_threads=THREADS*CONCURRENT_RUNS,
        point_budget_seconds=POINT_BUDGET,center_budget_seconds=CENTER_BUDGET))
    waves=[]
    for i in range(0,len(specs),CONCURRENT_RUNS):
        result=g.execute_parallel(specs[i:i+CONCURRENT_RUNS],out/f'wave_{i//CONCURRENT_RUNS:02d}',timeout_seconds)
        waves.append(result)
        if 'Interrupted' in str(result.get('failure')):break
    result=dict(complete=len(waves)==(len(specs)+1)//2 and all(r['complete'] for r in waves),waves=waves,
                max_concurrent_runs=CONCURRENT_RUNS,aggregate_address_space_bound_bytes=bound)
    old.write(out/('DONE.json' if result['complete'] else 'FAILED.json'),result);return result


def dispatch(args):
    control_limit();root=output_path(args.root,allow_root=True);root.mkdir(parents=True,exist_ok=True)
    lock_path=output_path(OUT/'production_v6/.dispatch.lock');lock_path.parent.mkdir(parents=True,exist_ok=True)
    fd=os.open(lock_path,os.O_RDWR|os.O_CREAT|os.O_CLOEXEC|os.O_NOFOLLOW,0o600)
    with os.fdopen(fd,'r+') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        # Two 400 GB slots: three bounded waves, no log/status polling.
        order=('harmonic','wallnormal','radial','upper725','baseline_rerun_lower')
        specs=[(name,[sys.executable,str(Path(__file__).resolve()),'campaign','--run',name,'--out',str(root/name)]) for name in order]
        attempt=root/'dispatch_attempts'/f'{time.time_ns()}_{uuid.uuid4().hex[:8]}'
        result=execute_waves(specs,attempt)
        print(json.dumps(dict(dispatch=str(attempt),complete=result['complete'])))
        return 0 if result['complete'] else 1


def install():
    old.MEMORY_BYTES=MEMORY_BYTES;old.POINT_BUDGET=POINT_BUDGET
    old.output_path=output_path;old.parameters=parameters;old.code_fingerprints=fingerprints;old.launch_point=launch_point
    g.output_path=output_path;g.parameters=parameters;g.launch_point=launch_point;g.MAX_RUNS=CONCURRENT_RUNS
    g.prior._RawFactor=ResourceFactor


install()
if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);sub=p.add_subparsers(dest='mode',required=True)
    d=sub.add_parser('dispatch');d.add_argument('--root',type=Path,required=True)
    c=sub.add_parser('campaign');c.add_argument('--run',choices=list(RUNS),required=True);c.add_argument('--out',type=Path,required=True)
    w=sub.add_parser('worker');w.add_argument('--job',type=Path,required=True);w.add_argument('--lock-fd',type=int,required=True)
    a=p.parse_args()
    if a.mode=='dispatch':raise SystemExit(dispatch(a))
    if a.mode=='campaign':raise SystemExit(campaign(a))
    raise SystemExit(worker(json.loads(output_path(a.job).read_text()),a.lock_fd))
