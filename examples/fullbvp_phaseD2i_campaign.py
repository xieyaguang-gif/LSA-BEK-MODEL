#!/usr/bin/env python3
"""D2i: N40 single-real-center checks and a controlled wall-normal map test."""
import os
for key in ('OPENBLAS_NUM_THREADS','OMP_NUM_THREADS','MKL_NUM_THREADS','NUMEXPR_NUM_THREADS','BLIS_NUM_THREADS'):
    os.environ[key]='1'
import argparse,fcntl,json,resource,sys,time,uuid
from pathlib import Path
import numpy as np
from scipy.interpolate import BarycentricInterpolator,interp1d
import fullbvp_phaseD2h_campaign as h
from fullbvp_phaseD2i_localize import xi_of_z
from llsa.secondary.radial import RadialConfig
from llsa.secondary.operators import PierOperators
old,g=h.old,h.g
ROOT,OUT=h.ROOT,h.OUT
SHIFTS=(.05+0j,)
RUNS={
 'upper720_N40':dict(n=40,H=4,dr=1.,a=2.,sources=['upper720_H6'],seed_run='wallnormal'),
 'upper725_N40':dict(n=40,H=4,dr=1.,a=2.,sources=['upper725_H6'],seed_run='upper725'),
 'lower_N40':dict(n=40,H=4,dr=1.,a=2.,sources=['lower719p948669_H4'],seed_run='wallnormal'),
 'upper720_N40_a3':dict(n=40,H=4,dr=1.,a=3.,sources=['upper720_H6'],seed_run='wallnormal'),
}
_saved_fingerprints=h.fingerprints
_saved_launch=h.launch_point


def output_path(path,allow_root=False):
    path=Path(path).absolute();canonical=path.resolve()
    if '..' in path.parts or canonical!=path or any(p.is_symlink() for p in (path,*path.parents)):
        raise ValueError('unsafe output path')
    roots=[OUT/'D2i',OUT/'production_v7']
    if allow_root and canonical==roots[1]:return canonical
    if not any(canonical.is_relative_to(r) and canonical!=r for r in roots):
        raise ValueError('D2i writes only descendants of D2i or production_v7')
    return canonical


def fingerprints():
    paths=[Path(__file__),ROOT/'examples/fullbvp_phaseD2i_localize.py',OUT/'production_v7/commands.sh']
    return dict(_saved_fingerprints(),**{str(p.resolve()):old.d2.v1.sha(p) for p in paths})


def seed_bundle(run_name,source):
    point=OUT/'production_v6'/RUNS[run_name]['seed_run']/source
    manifest=json.loads((point/'manifest.json').read_text());c=json.loads((point/'corrected.json').read_text())
    fields=h.source_file(c['fields'],point)
    if (manifest['fingerprint']!=old.d2.v1.digest(manifest['params']) or c['fingerprint']!=manifest['fingerprint']
        or old.d2.v1.sha(fields)!=c['fields_sha256']):raise RuntimeError('v6 seed evidence mismatch')
    for p,sha in manifest['params']['code_sha256'].items():
        if old.d2.v1.sha(p)!=sha:raise RuntimeError('v6 seed code changed: '+p)
    return dict(point=str(point),fields=str(fields),A1=c['after']['A1'],Rout=c['after']['Rout'],
                mapping=dict(a=2.,b=.6,c=.5),
                sha256={str(p):old.d2.v1.sha(p) for p in [point/'manifest.json',point/'corrected.json',fields]})


def parameters(source,n=40,H=4,dr=1.,shifts=SHIFTS,k=6,ncv=40,run_name=None):
    if run_name not in RUNS:raise ValueError('unknown v7 run')
    cfg=RUNS[run_name]
    if source not in cfg['sources'] or (n,H,dr)!=(40,4,1.) or tuple(shifts)!=SHIFTS or (k,ncv)!=(6,40):
        raise ValueError('fixed N40/H4/dr1, real center only, k6/ncv40 study')
    seed=seed_bundle(run_name,source);source_path=old.d2.v1.SOURCES[source]
    # Keep the N32 comparison's number of radial intervals even if the lower Rout moves.
    previous=json.loads((Path(seed['point'])/'manifest.json').read_text())['params']
    return dict(revision='D2i',run_name=run_name,source=source,n=n,H=H,intervals=previous['intervals'],requested_dr=dr,
        source_projection_dr=(seed['Rout']-400)/previous['intervals'],shifts=[old.d2.v1.zpair(s) for s in SHIFTS],
        k=k,ncv=ncv,threads=16,budget_seconds=h.POINT_BUDGET,center_budget_seconds=h.CENTER_BUDGET,
        target_seconds=old.TARGET_SECONDS,memory_bytes=h.MEMORY_BYTES,stationary_accept=old.STATIONARY_ACCEPT,
        pair_accept=old.PAIR_ACCEPT,linear_accept=g.RHS_ACCEPT,linear_policy=g.POLICY,
        mapping=dict(a=cfg['a'],b=.6,c=.5),seed=seed,positive_half_plane=True,
        lower_correction='hold the v6 N32 nonzero A1 exactly; Rout and omega free',
        code_sha256=fingerprints(),source_hashes={p.name:old.d2.v1.sha(p) for p in [source_path/'fields.npz',source_path/'summary.json']})


def mapping(n,H,intervals,a):
    class Config(RadialConfig):
        def operators(self,radius):
            return PierOperators(radius=radius,beta=self.beta,n=self.n,ro=self.ro,a=a,b=.6,c=.5)
    class PhysicalMapping(old.base.Mapping):
        def at(self,Rout,omega,derivative=False):
            result=super().at(Rout,omega,derivative)
            # Independent physical-grid reassembly must use the same map.
            result[0].config=Config(ro=-1.,beta=68.,n=self.n,harmonics=self.H,
                                    omega=float(omega),nonlinear=True)
            return result
    saved=old.base.RadialConfig
    try:
        old.base.RadialConfig=Config
        result=PhysicalMapping(n,H,intervals)
    finally:old.base.RadialConfig=saved
    return result


def transfer_physical(path,target,source_mapping):
    with np.load(path) as data:
        r=data['radii'];xi=data['xi'];old_H=int(data['H']);q=data['field'].reshape(len(r),2*old_H+1,4,len(xi))
        omega=float(data['omega']);Rout=float(data['Rout'])
    # A map change must interpolate at equal physical z, not at equal Chebyshev xi.
    old_xi=np.array([xi_of_z(z,**source_mapping) for z in target.op.grid.z])
    q=BarycentricInterpolator(xi,q,axis=-1)(old_xi)
    radii=400+(Rout-400)*target.xi
    q=interp1d(r,q,axis=0,kind='cubic',bounds_error=True)(radii).reshape(target.nr,2*old_H+1,target.s)
    new=np.zeros((target.nr,2*target.H+1,target.s),complex);HH=min(old_H,target.H)
    new[:,target.H-HH:target.H+HH+1]=q[:,old_H-HH:old_H+HH+1]
    return np.r_[old.base.pack(new),omega,Rout]


def corrected_temporal(params,out,deadline):
    out=output_path(out);out.mkdir(parents=True,exist_ok=True);seed=params['seed']
    for path,sha in seed['sha256'].items():
        deadline.check()
        if old.d2.v1.sha(path)!=sha:raise RuntimeError('v6 seed changed')
    mesh=mapping(params['n'],params['H'],params['intervals'],params['mapping']['a'])
    fingerprint=old.d2.v1.digest(params);cache=out/'corrected.json'
    if cache.exists():
        saved=json.loads(cache.read_text());fields=output_path(saved['fields'])
        if saved['fingerprint']!=fingerprint or old.d2.v1.sha(fields)!=saved['fields_sha256']:
            raise RuntimeError('v7 corrected-state hash/fingerprint mismatch')
        with np.load(fields) as data:q=data['field'].copy();R=float(data['Rout']);omega=float(data['omega'])
        temporal=old.d2.v1.Temporal(mesh,q,R,omega,deadline)
        old.stationarity_gate(temporal);old.branch_gate(saved['before'],saved['after'])
        return temporal,saved
    y=transfer_physical(seed['fields'],mesh,seed['mapping']);initial=y.copy()
    q=old.d2.v1.unpack(y[:-2],mesh.nr,mesh.H,mesh.s);p,*_=mesh.at(y[-1],y[-2]);energy=old.base.energies(p,q)
    before=dict(A1=float(np.sqrt(energy[1])),omega=float(y[-2]),Rout=float(y[-1]),energies=energy.tolist())
    mode='energy' if params['source'].startswith('lower') else 'Rout'
    target=seed['A1'] if mode=='energy' else seed['Rout']
    attempts=out/'corrections';attempts.mkdir(exist_ok=True);attempt=attempts/f'attempt_{len(list(attempts.iterdir())):04d}'
    previous=old.base.Corrector
    try:
        old.base.Corrector=g.prior.NonzeroCorrector
        y,record=old.base.correct(mesh,q[:,mesh.H+1],y,mode,target,attempt,
                                  budget=max(.001,deadline.remaining()),max_newton=12)
    finally:old.base.Corrector=previous
    if not record.get('accepted'):raise RuntimeError('N40 nonzero stationary correction failed: '+str(record.get('failure')))
    q=old.d2.v1.unpack(y[:-2],mesh.nr,mesh.H,mesh.s)
    temporal=old.d2.v1.Temporal(mesh,q,float(y[-1]),float(y[-2]),deadline)
    diagnostics=old.stationarity_gate(temporal);branch=old.branch_gate(before,record);fields=attempt/'fields.npz'
    saved=dict(fingerprint=fingerprint,fields=str(fields),fields_sha256=old.d2.v1.sha(fields),
        before=before,after={k:record[k] for k in ('A1','omega','energies','Rout')},branch=branch,
        diagnostics=diagnostics,correction_seconds=record['seconds'],mode=mode,target=target,mapping=params['mapping'],
        seed=seed,Rout_change=float(y[-1]-initial[-1]),actual_dr=float((y[-1]-400)/mesh.intervals),
        projected_to_corrected_relative=float(np.linalg.norm(y[:-2]-initial[:-2])/np.linalg.norm(initial[:-2])),
        independent=record['independent'],original_resolution_certified=False)
    old.write(cache,saved);return temporal,saved


def worker(job,lock_fd):
    out=output_path(job['out']);params=job['params'];actual=os.fstat(lock_fd);expected=os.stat(out.parent/'.run.lock')
    if (actual.st_dev,actual.st_ino)!=(expected.st_dev,expected.st_ino):raise RuntimeError('wrong inherited run lock')
    h.control_limit();start=time.monotonic();deadline=old.d2.v1.Deadline(job['remaining'])
    if parameters(params['source'],run_name=params['run_name'])!=params:raise RuntimeError('worker fingerprint changed')
    with h.memory_slot(OUT/'production_v7/.memory_slots') as slot:
        deadline.check();resource.setrlimit(resource.RLIMIT_AS,(h.MEMORY_BYTES,h.MEMORY_BYTES))
        old.write(out/'resources.json',dict(slot=slot,worker_memory_bytes=h.MEMORY_BYTES,threads=16,
            aggregate_address_space_bound=806_000_000_000))
        temporal,correction=corrected_temporal(params,out,deadline);J=None;accepted={}
        target=out/'center_00'
        if (target/'DONE.json').exists():
            accepted[0]=g.run_center(temporal,None,SHIFTS[0],target,old.d2.v1.digest(params),deadline,k=6,ncv=40)
        else:
            J,assembly=old.d2.assemble(temporal,deadline);old.write(out/'assembly.json',assembly)
        centers=h.scan_centers(temporal,J,params,out,deadline,accepted)
        result=dict(fingerprint=old.d2.v1.digest(params),complete=all(c['complete'] for c in centers),
            seconds=time.monotonic()-start,correction=correction,centers=centers,
            peak_rss_bytes=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss*1024,
            original_resolution_certified=False,exhaustive_stability=False,mapping=params['mapping'],
            perturbations='integer multiples of beta=68 only; one sampled real-axis center')
        old.write(out/'worker_result.json',result);return 0 if result['complete'] else 1


def launch_point(params,out,command=None,lock_fd=None):
    if command is None:
        if lock_fd is None:raise RuntimeError('worker requires inherited run lock')
        command=[sys.executable,str(Path(__file__).resolve()),'worker','--job',str(output_path(out)/'job.json'),'--lock-fd',str(lock_fd)]
    return _saved_launch(params,out,command=command,lock_fd=lock_fd)


def campaign(args):
    h.control_limit();return g.campaign(args)


def dispatch(args):
    h.control_limit();root=output_path(args.root,allow_root=True);root.mkdir(parents=True,exist_ok=True)
    lock_path=output_path(OUT/'production_v7/.dispatch.lock');lock_path.parent.mkdir(parents=True,exist_ok=True)
    fd=os.open(lock_path,os.O_RDWR|os.O_CREAT|os.O_CLOEXEC|os.O_NOFOLLOW,0o600)
    with os.fdopen(fd,'r+') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        specs=[(name,[sys.executable,str(Path(__file__).resolve()),'campaign','--run',name,'--out',str(root/name)]) for name in RUNS]
        attempt=root/'dispatch_attempts'/f'{time.time_ns()}_{uuid.uuid4().hex[:8]}'
        result=h.execute_waves(specs,attempt,timeout_seconds=14520.)
        print(json.dumps(dict(dispatch=str(attempt),complete=result['complete'])))
        return 0 if result['complete'] else 1


def install():
    old.output_path=output_path;old.parameters=parameters;old.code_fingerprints=fingerprints;old.launch_point=launch_point
    g.output_path=output_path;g.parameters=parameters;g.launch_point=launch_point;g.RUNS=RUNS;g.SHIFTS=SHIFTS
    h.output_path=output_path


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
