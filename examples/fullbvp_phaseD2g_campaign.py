#!/usr/bin/env python3
"""D2g: independent run locks, resumable positive-half-plane scans, five-run dispatch."""
import os
for key in ('OPENBLAS_NUM_THREADS','OMP_NUM_THREADS','MKL_NUM_THREADS','NUMEXPR_NUM_THREADS','BLIS_NUM_THREADS'):
    os.environ[key]='1'
if os.environ.get('D2G_CPUSET'):
    os.sched_setaffinity(0,{int(x) for x in os.environ['D2G_CPUSET'].split(',')})
import argparse,copy,fcntl,json,resource,shutil,signal,subprocess,sys,time,uuid
from pathlib import Path
import numpy as np
from scipy.sparse.linalg import LinearOperator,gmres
import fullbvp_phaseD2f_repair as prior
old=prior.old
ROOT,OUT=old.ROOT,old.OUT
SHIFTS=tuple(.05+1j*x for x in (0,2,5,10,20))
RHS_ACCEPT=1e-5
BACKWARD_ACCEPT=1e-12
CENTER_BUDGET=1200.
THREADS=16
MAX_RUNS=5
POLICY=dict(name='normwise_backward_and_original_rhs_relative',normwise_accept=BACKWARD_ACCEPT,
            original_rhs_relative_accept=RHS_ACCEPT,componentwise_diagnostic_only=True,eigenpair_accept=1e-8,
            equilibration_steps=6)
RUNS={
    'baseline_rerun_lower':dict(n=24,H=4,dr=1.,sources=['lower719p948669_H4']),
    'harmonic':dict(n=24,H=6,dr=1.,sources=['lower719p948669_H4','upper720_H6']),
    'wallnormal':dict(n=32,H=4,dr=1.,sources=['lower719p948669_H4','upper720_H6']),
    'radial':dict(n=24,H=4,dr=.5,sources=['lower719p948669_H4','upper720_H6']),
    'upper725':dict(n=24,H=4,dr=1.,sources=['upper725_H6']),
}
_saved_parameters=old.parameters
_saved_fingerprints=old.code_fingerprints
_saved_center=old.run_center
_saved_launch=old.launch_point


def output_path(path,allow_root=False):
    path=Path(path).absolute()
    if '..' in path.parts:raise ValueError('lexical traversal forbidden')
    canonical=path.resolve()
    if canonical!=path or any(p.is_symlink() for p in (path,*path.parents)):
        raise ValueError('symlink output alias forbidden')
    roots=[OUT/'D2e',OUT/'production_v5']
    if allow_root and canonical==roots[1]:return canonical
    if not any(canonical.is_relative_to(r) and canonical!=r for r in roots):
        raise ValueError('D2g writes only descendants of D2e or production_v5')
    return canonical


def fingerprints():
    paths=[Path(__file__),OUT/'production_v5/commands.sh']
    return dict(_saved_fingerprints(),**{str(p.resolve()):old.d2.v1.sha(p) for p in paths})


def legacy_file(path,point):
    path=Path(path).resolve()
    if not path.is_relative_to(point.resolve()) or not path.is_file():
        raise RuntimeError('legacy evidence escapes its source point')
    return path


def import_bundle(source):
    point=OUT/'production_v4/run_D2f_baseline'/source
    manifest=json.loads((point/'manifest.json').read_text())
    correction=json.loads((point/'corrected.json').read_text())
    fields=legacy_file(correction['fields'],point)
    if (correction['fingerprint']!=manifest['fingerprint'] or old.d2.v1.sha(fields)!=correction['fields_sha256']):
        raise RuntimeError('legacy corrected field fingerprint/hash mismatch')
    center=json.loads((point/'center_00/DONE.json').read_text())
    arrays=legacy_file(center['arrays'],point)
    if (not center.get('complete') or center['fingerprint']!=manifest['fingerprint']
        or old.d2.v1.sha(arrays)!=center['arrays_sha256']):
        raise RuntimeError('legacy completed center fingerprint/hash mismatch')
    paths=[point/'manifest.json',point/'corrected.json',fields,point/'center_00/DONE.json',arrays]
    return dict(point=str(point),sha256={str(p):old.d2.v1.sha(p) for p in paths})


def parameters(source,n=24,H=4,dr=1.,shifts=SHIFTS,k=6,ncv=40,run_name=None):
    p=_saved_parameters(source,n,H,dr,shifts,k,ncv)
    p.update(revision='D2g',run_name=run_name,linear_policy=POLICY,center_budget_seconds=CENTER_BUDGET,
             positive_half_plane=True,threads=THREADS,import_from=None)
    p['source_projection_dr']=p.pop('actual_dr')
    if run_name=='baseline_rerun_lower':p['import_from']=import_bundle(source)
    return p


class Inverse(prior.BackwardInverse):
    def measure(self,y,rhs):
        residual,metric=super().measure(y,rhs)
        if not np.isfinite(metric['componentwise_backward']):metric['componentwise_backward']=None
        metric['accepted']=bool(metric['normwise_backward']<=BACKWARD_ACCEPT
                                and np.isfinite(metric['original_relative'])
                                and metric['original_relative']<=RHS_ACCEPT)
        return residual,metric
    def score(self,metric):
        return max(metric['normwise_backward']/BACKWARD_ACCEPT,metric['original_relative']/RHS_ACCEPT)
    def apply(self,x):
        self.deadline.check();self.calls+=1;start=time.monotonic()
        record=dict(call=self.calls,sigma=old.d2.v1.zpair(self.sigma),policy=POLICY,accepted=False,
            direct_corrections=0,krylov_used=False,krylov_preconditioner_calls=0,krylov_iterations=0,
            direct_residuals=[],backward_history=[])
        best=None;score=float('inf');rhs=-(self.B@x)
        try:
            if not np.all(np.isfinite(rhs)):raise old.d2.v1.InnerFailure('nonfinite original RHS')
            t=time.monotonic()
            y=np.zeros_like(rhs) if not np.any(rhs) else np.asarray(self.factor.solve(rhs),complex)
            for step in range(3):
                residual,metric=self.measure(y,rhs)
                record['direct_residuals'].append(metric['original_relative']);record['backward_history'].append(metric)
                candidate_score=self.score(metric)
                if candidate_score<score:best,score=y.copy(),candidate_score
                if metric['accepted'] or step==2:break
                y=y+self.factor.solve(residual);record['direct_corrections']+=1
            record['direct_seconds']=time.monotonic()-t
            _,metric=self.measure(best,rhs)
            if not metric['accepted']:
                record['krylov_used']=True
                def precondition(r):
                    self.deadline.check();record['krylov_preconditioner_calls']+=1
                    return self.factor.solve(r)
                def callback(value):
                    self.deadline.check();record['krylov_iterations']+=1
                t=time.monotonic()
                candidate,info=self.solver(self.A,rhs,x0=best,M=LinearOperator(self.J.shape,matvec=precondition,dtype=complex),
                    rtol=1e-10,atol=0,restart=35,maxiter=5,callback=callback,callback_type='pr_norm')
                _,metric=self.measure(candidate,rhs);record.update(krylov_info=int(info),krylov_seconds=time.monotonic()-t)
                if self.score(metric)<score:best,score=candidate.copy(),self.score(metric)
            _,metric=self.measure(best,rhs);record.update(metric)
            record['solution_amplification']=float(np.linalg.norm(best)/max(np.linalg.norm(rhs),1e-300))
            if not metric['accepted']:raise old.d2.v1.InnerFailure('normwise/original-relative gate failed: '+str(metric))
            return best
        except Exception as exc:
            record['failure']=f'{type(exc).__name__}: {exc}';raise
        finally:
            record['seconds']=time.monotonic()-start;self.log_path.parent.mkdir(parents=True,exist_ok=True)
            with self.log_path.open('a') as stream:stream.write(json.dumps(record,allow_nan=False)+'\n')


def run_center(*args,**kw):
    result=_saved_center(*args,**kw)
    out=output_path(args[3] if len(args)>3 else kw['out'])
    result.update(linear_policy=POLICY,linear_accept=RHS_ACCEPT,componentwise_is_gate=False)
    old.write(out/('DONE.json' if result['complete'] else 'FAILED.json'),result)
    old.write(Path(result['arrays']).parent/'result.json',result)
    return result


def import_correction(params,out,deadline):
    bundle=params.get('import_from')
    if not bundle or (out/'corrected.json').exists():return
    for path,h in bundle['sha256'].items():
        if old.d2.v1.sha(path)!=h:raise RuntimeError('legacy import evidence changed')
    point=Path(bundle['point']);saved=json.loads((point/'corrected.json').read_text())
    manifest=json.loads((point/'manifest.json').read_text())
    previous=manifest['params']
    if any(previous[k]!=params[k] for k in ('source','n','H','intervals','source_hashes')):
        raise RuntimeError('legacy import discretization/source mismatch')
    fields=legacy_file(saved['fields'],point)
    with np.load(fields) as data:
        if int(data['n'])!=params['n'] or int(data['H'])!=params['H'] or len(data['radii'])!=params['intervals']+1 or not bool(data['accepted']):
            raise RuntimeError('legacy corrected NPZ metadata mismatch')
    destination=out/'corrections/import_D2f/fields.npz';destination.parent.mkdir(parents=True,exist_ok=True)
    temporary=destination.with_name(destination.name+'.importing');shutil.copyfile(fields,temporary)
    if old.d2.v1.sha(temporary)!=saved['fields_sha256']:raise RuntimeError('copied corrected field hash mismatch')
    temporary.replace(destination)
    saved.update(fingerprint=old.d2.v1.digest(params),fields=str(destination),fields_sha256=old.d2.v1.sha(destination),
                 reused_from_D2f=bundle,reused_correction_seconds=saved['correction_seconds'])
    old.write(out/'corrected.json',saved)
    # Recheck original stationarity and nonzero branch through the ordinary loader.
    prior.corrected_temporal(params,out,deadline)


def import_center(params,temporal,out,index,deadline):
    if not params.get('import_from') or index!=0 or (out/'DONE.json').exists():return
    point=Path(params['import_from']['point']);previous=json.loads((point/'center_00/DONE.json').read_text())
    if (previous['sigma']!=params['shifts'][index] or previous['k']!=params['k'] or previous['ncv']!=params['ncv']):
        raise RuntimeError('legacy center settings mismatch')
    arrays=legacy_file(previous['arrays'],point)
    if old.d2.v1.sha(arrays)!=previous['arrays_sha256']:raise RuntimeError('legacy eigenpair hash mismatch')
    start=time.monotonic()
    with np.load(arrays) as data:values=data['values'].copy();vectors=data['vectors'].copy()
    if vectors.shape!=(temporal.size,params['k']) or len(values)!=params['k']:
        raise RuntimeError('legacy eigenpair dimensions mismatch')
    checks=[]
    for value,vector in zip(values,vectors.T):
        deadline.check();checks.append(old.d2.v1.certify(temporal.J,temporal.B,max(temporal.b),value,vector,
                                                       tol=old.PAIR_ACCEPT,phase=temporal.neutral()))
    if not all(r['accepted'] for r in checks):
        old.write(out/'IMPORT_REJECTED.json',dict(eigenpairs=checks));return
    attempt=out/'attempts/import_D2f';attempt.mkdir(parents=True,exist_ok=True)
    destination=attempt/'eigenpairs.npz';temporary=destination.with_name(destination.name+'.importing')
    shutil.copyfile(arrays,temporary)
    if old.d2.v1.sha(temporary)!=previous['arrays_sha256']:raise RuntimeError('copied eigenpair hash mismatch')
    temporary.replace(destination)
    result=dict(fingerprint=old.d2.v1.digest(params),sigma=params['shifts'][index],complete=True,
        k=params['k'],ncv=params['ncv'],arrays=str(destination),arrays_sha256=old.d2.v1.sha(destination),
        eigenpairs=checks,residual_accept=old.PAIR_ACCEPT,linear_policy=POLICY,linear_accept=RHS_ACCEPT,
        componentwise_is_gate=False,original_resolution_certified=False,exhaustive_stability=False,
        seconds=time.monotonic()-start,arnoldi_callbacks=0,factorization_count_this_run=0,
        reused_eigenpairs=True,source_result_sha256=old.d2.v1.sha(point/'center_00/DONE.json'),
        reused_previous_seconds=previous['seconds'],historical_linear_policy=previous.get('linear_policy'))
    old.write(attempt/'result.json',result);old.write(out/'DONE.json',result)
    (out/'FAILED.json').unlink(missing_ok=True)


def scan_centers(temporal,J,params,out,deadline):
    results=[];fingerprint=old.d2.v1.digest(params)
    for index,pair in enumerate(params['shifts']):
        target=out/f'center_{index:02d}'
        try:
            if not (target/'DONE.json').exists():deadline.check()
            import_center(params,temporal,target,index,deadline)
            center_deadline=old.d2.v1.Deadline(min(params['center_budget_seconds'],max(.001,deadline.remaining())))
            result=run_center(temporal,J,complex(*pair),target,fingerprint,center_deadline,k=params['k'],ncv=params['ncv'])
        except Exception as exc:
            result=dict(fingerprint=fingerprint,sigma=pair,complete=False,failure=f'{type(exc).__name__}: {exc}')
            done=target/'DONE.json'
            if done.exists() and isinstance(exc,old.d2.v1.BudgetExpired):
                result['cached_certificate_retained']=True;old.write(target/'DEFERRED.json',result)
            else:
                if done.exists():done.replace(target/f'INVALIDATED_DONE_{time.time_ns()}.json')
                old.write(target/'FAILED.json',result)
        results.append(result)
    return results


def worker(job,lock_fd):
    out=output_path(job['out']);params=job['params']
    info=os.fstat(lock_fd);expected=os.stat(out.parent/'.run.lock')
    if (info.st_dev,info.st_ino)!=(expected.st_dev,expected.st_ino):raise RuntimeError('worker did not inherit its run lock')
    current=parameters(params['source'],params['n'],params['H'],params['requested_dr'],
                       tuple(complex(*x) for x in params['shifts']),params['k'],params['ncv'],run_name=params['run_name'])
    if current!=params:raise RuntimeError('worker fingerprint changed')
    resource.setrlimit(resource.RLIMIT_AS,(params['memory_bytes'],params['memory_bytes']))
    start=time.monotonic();deadline=old.d2.v1.Deadline(job['remaining'])
    import_correction(params,out,deadline)
    temporal,correction=prior.corrected_temporal(params,out,deadline)
    J,assembly=old.d2.assemble(temporal,deadline);old.write(out/'assembly.json',assembly)
    if np.iscomplexobj(J.data) and np.any(J.data.imag):raise RuntimeError('positive-half-plane symmetry requires a real pencil')
    centers=scan_centers(temporal,J,params,out,deadline)
    result=dict(fingerprint=old.d2.v1.digest(params),complete=all(r['complete'] for r in centers),
        seconds=time.monotonic()-start,correction=correction,centers=centers,
        peak_rss_bytes=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss*1024,
        original_resolution_certified=False,exhaustive_stability=False,positive_half_plane=True,
        symmetry='real packed J and real B: (lambda,v) implies (conj(lambda),conj(v))',
        perturbations='integer multiples of beta=68 only',reused_center_seconds=sum(r.get('reused_previous_seconds',0) for r in centers))
    old.write(out/'worker_result.json',result);return 0 if result['complete'] else 1


def launch_point(params,out,command=None,lock_fd=None):
    if command is None:
        if lock_fd is None:raise RuntimeError('D2g worker requires an inherited per-run lock')
        command=[sys.executable,str(Path(__file__).resolve()),'worker','--job',str(output_path(out)/'job.json'),'--lock-fd',str(lock_fd)]
    result=_saved_launch(params,out,command=command,lock_fd=lock_fd)
    if not result['complete']:
        point=output_path(out)
        for index,pair in enumerate(params['shifts']):
            center=point/f'center_{index:02d}'
            if not (center/'DONE.json').exists() and not (center/'FAILED.json').exists():
                old.write(center/'FAILED.json',dict(fingerprint=old.d2.v1.digest(params),sigma=pair,
                    complete=False,failure='point stopped before center completion: '+str(result.get('failure')),
                    parent_recorded_failure=True))
    return result


def campaign(args):
    out=output_path(args.out);config=RUNS[args.run]
    cpus=sorted(os.sched_getaffinity(0))
    if len(cpus)>THREADS:os.sched_setaffinity(0,set(cpus[:THREADS]))
    out.mkdir(parents=True,exist_ok=True)
    lock_path=output_path(out/'.run.lock')
    fd=os.open(lock_path,os.O_RDWR|os.O_CREAT|os.O_CLOEXEC|os.O_NOFOLLOW,0o600)
    with os.fdopen(fd,'r+') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        point_params=[parameters(source,config['n'],config['H'],config['dr'],run_name=args.run) for source in config['sources']]
        settings=dict(run=args.run,points=point_params,threads=THREADS,shifts=[old.d2.v1.zpair(x) for x in SHIFTS])
        fingerprint=old.d2.v1.digest(settings);manifest=out/'manifest.json'
        if manifest.exists():
            if json.loads(manifest.read_text()).get('fingerprint')!=fingerprint:
                raise RuntimeError('run configuration changed; select a new output directory')
        else:old.write(manifest,dict(fingerprint=fingerprint,settings=settings))
        results=[];failure=None
        try:
            for source,params in zip(config['sources'],point_params):
                result=launch_point(params,out/source,lock_fd=lock.fileno());results.append(result)
                if 'Interrupted:' in str(result.get('failure','')):break
        except BaseException as exc:failure=f'{type(exc).__name__}: {exc}'
        complete=failure is None and len(results)==len(point_params) and all(r['complete'] for r in results)
        old.write(out/('DONE.json' if complete else 'FAILED.json'),dict(fingerprint=fingerprint,complete=complete,
            points=results,failure=failure,run_lock=str(lock_path),cpu_affinity=sorted(os.sched_getaffinity(0))))
        (out/('FAILED.json' if complete else 'DONE.json')).unlink(missing_ok=True)
        return 0 if complete else 1


def cpu_groups(count):
    if not 1<=count<=MAX_RUNS:raise ValueError('at most five simultaneous runs')
    available=sorted(os.sched_getaffinity(0))
    if len(available)<count*THREADS:raise RuntimeError('not enough allowed CPUs for requested disjoint 16-CPU groups')
    return [available[i*THREADS:(i+1)*THREADS] for i in range(count)]


def execute_parallel(specs,out,timeout_seconds=9720.):
    out=output_path(out);groups=cpu_groups(len(specs))
    if len({name for name,command in specs})!=len(specs):raise ValueError('duplicate run names')
    out.mkdir(parents=True,exist_ok=False)
    old.write(out/'PLAN.json',dict(total_compute_threads=len(specs)*THREADS,
        runs=[dict(name=name,command=command,cpus=cpus,threads=THREADS) for (name,command),cpus in zip(specs,groups)]))
    processes=[];logs=[];statuses=[];failure=None;start=time.monotonic()
    def interrupted(signum,frame):raise old.Interrupted(f'signal {signum}')
    previous={s:signal.signal(s,interrupted) for s in (signal.SIGINT,signal.SIGTERM)}
    try:
        for (name,command),cpus in zip(specs,groups):
            log=(out/f'{name}.log').open('ab');logs.append(log)
            env=dict(os.environ,D2G_CPUSET=','.join(map(str,cpus)))
            processes.append((name,subprocess.Popen(command,stdout=log,stderr=subprocess.STDOUT,
                                                     start_new_session=True,cwd=ROOT,env=env)))
        # All workers are launched before any blocking completion wait.
        for name,proc in processes:
            rc=proc.wait(timeout=max(.001,timeout_seconds-(time.monotonic()-start)))
            statuses.append(dict(name=name,returncode=rc))
    except BaseException as exc:
        failure=f'{type(exc).__name__}: {exc}'
        for name,proc in processes:
            if proc.poll() is None:
                try:os.killpg(proc.pid,signal.SIGTERM)
                except ProcessLookupError:pass
        for name,proc in processes:
            try:proc.wait(timeout=10)
            except subprocess.TimeoutExpired:
                try:os.killpg(proc.pid,signal.SIGKILL)
                except ProcessLookupError:pass
                proc.wait()
    finally:
        for log in logs:log.close()
        for signum,handler in previous.items():signal.signal(signum,handler)
    complete=failure is None and len(statuses)==len(specs) and all(r['returncode']==0 for r in statuses)
    result=dict(complete=complete,statuses=statuses,failure=failure,seconds=time.monotonic()-start,
                total_compute_threads=len(specs)*THREADS,cpu_groups=groups)
    old.write(out/('DONE.json' if complete else 'FAILED.json'),result)
    return result


def dispatch(args):
    root=output_path(args.root,allow_root=True)
    specs=[(name,[sys.executable,str(Path(__file__).resolve()),'campaign','--run',name,'--out',str(root/name)]) for name in RUNS]
    attempt=root/'dispatch_attempts'/f'{time.time_ns()}_{uuid.uuid4().hex[:8]}'
    result=execute_parallel(specs,attempt)
    print(json.dumps(dict(dispatch=str(attempt),complete=result['complete'],statuses=result['statuses'])))
    return 0 if result['complete'] else 1


def install():
    old.output_path=output_path;old.code_fingerprints=fingerprints;old.parameters=parameters
    old.d2.PrecisionInverse=Inverse;old.run_center=run_center;old.launch_point=launch_point


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
