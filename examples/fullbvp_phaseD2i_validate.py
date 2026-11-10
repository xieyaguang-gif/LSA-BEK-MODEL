#!/usr/bin/env python3
"""Bounded N40 transfer/operator checks; no production-size LU or eigensolve."""
import os
os.environ['PYTHONDONTWRITEBYTECODE']='1'
import argparse,copy,fcntl,gc,json,sys,time
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch
import numpy as np
from scipy import linalg
import fullbvp_phaseD2i_campaign as i
import fullbvp_phaseD2g_validate as gv
import fullbvp_phaseD2h_validate as hv
old=i.old

def require(ok,message):
    if not ok:raise AssertionError(message)

def transfers(out):
    source=i.mapping(12,1,4,2.);r=400+320*source.xi;z=source.op.grid.z
    t=.6*source.op.grid.xi+.4*(source.op.grid.xi**3+.5*(1-source.op.grid.xi**2))
    f=(1-t*t)/8;w=(1+t)/2
    q=np.zeros((5,3,4,13),complex);q[:,1,0]=f;q[:,1,2]=w
    path=out/'analytic_seed.npz';np.savez(path,field=q.reshape(5,3,-1),radii=r,xi=source.op.grid.xi,H=1,omega=53.,Rout=720.)
    errors=[]
    for a in (2.,3.):
        mesh=i.mapping(16,1,4,a);v=i.transfer_physical(path,mesh,dict(a=2.,b=.6,c=.5))
        qq=old.d2.v1.unpack(v[:-2],mesh.nr,mesh.H,mesh.s).reshape(5,3,4,17)
        zz=mesh.op.grid.z;expected_f=np.zeros_like(zz);expected_f[:-1]=zz[:-1]/(2+zz[:-1])**2
        expected_w=np.ones_like(zz);expected_w[:-1]=zz[:-1]/(2+zz[:-1])
        error=max(float(np.max(abs(qq[:,1,0]-expected_f))),float(np.max(abs(qq[:,1,2]-expected_w))))
        require(error<2e-12,'map-change interpolation used xi instead of physical z');errors.append(error)
    return dict(physical_interpolation_max=max(errors),free_w0_infinite_endpoint_preserved=True)

def operators(out):
    results=[]
    for name,cfg in i.RUNS.items():
        params=i.parameters(cfg['sources'][0],run_name=name)
        mesh=i.mapping(40,4,params['intervals'],cfg['a']);y=i.transfer_physical(params['seed']['fields'],mesh,params['seed']['mapping'])
        q=old.d2.v1.unpack(y[:-2],mesh.nr,mesh.H,mesh.s)
        original=old.base.independent(mesh,y)
        require(original['template_difference_relative']<1e-10,'N40 mapped templates differ from original physical equations')
        temporal=old.d2.v1.Temporal(mesh,q,float(y[-1]),float(y[-2]));rng=np.random.default_rng(20261031)
        v=rng.normal(size=temporal.size);eps=1e-6*max(abs(y[:-2]))/max(abs(v))
        exact=temporal.J@v;fd=(temporal.residual(y[:-2]+eps*v)-temporal.residual(y[:-2]-eps*v))/(2*eps)
        error=float(np.linalg.norm(exact-fd)/np.linalg.norm(exact));require(error<1e-7,'N40 Frechet derivative mismatch')
        before_energy=old.base.energies(temporal.p,q)
        record=dict(run=name,dimension=temporal.size,mapping=params['mapping'],single_center=params['shifts'],
            k=params['k'],ncv=params['ncv'],physical_template_relative=original['template_difference_relative'],
            centered_difference_relative=error,source_Rout=params['seed']['Rout'],source_A1=params['seed']['A1'],
            projected_A1=float(np.sqrt(before_energy[1])),fixed_lower_target=params['seed']['A1'] if name=='lower_N40' else None,
            largest_finite_z=float(mesh.op.grid.z[-2]),first_interior_z=float(mesh.op.grid.z[1]),
            seed_hashes=params['seed']['sha256'],not_yet_stationary=True,production_LU_started=False)
        results.append(record);del temporal,mesh,q,y,exact,fd,v;gc.collect()
    return results

def small_qz(out):
    results=[];fixture=None
    params=i.parameters('upper720_H6',run_name='upper720_N40')
    for a in (2.,3.):
        mesh=i.mapping(8,1,4,a);y=i.transfer_physical(params['seed']['fields'],mesh,params['seed']['mapping'])
        q=old.d2.v1.unpack(y[:-2],mesh.nr,mesh.H,mesh.s);t=old.d2.v1.Temporal(mesh,q,float(y[-1]),float(y[-2]))
        J,assembly=old.d2.assemble(t,old.d2.v1.Deadline(120))
        values=linalg.eigvals(J.toarray(),-np.diag(t.b));values=values[np.isfinite(values)]
        center=i.g.run_center(t,J,.05+0j,out/f'qz_a{int(a)}','D2i-map-test',old.d2.v1.Deadline(60),k=6,ncv=40)
        require(center['complete'],'mapped small center failed: '+str(center.get('failure')))
        distances=[float(np.min(abs(complex(*r['value'])-values))) for r in center['eigenpairs']]
        require(max(distances)<1e-6,'mapped spectrum disagrees with independent QZ')
        require(i.g.run_center(t,None,.05+0j,out/f'qz_a{int(a)}','D2i-map-test',old.d2.v1.Deadline(30),k=6,ncv=40)==center,
                'center cache refactored or changed')
        results.append(dict(a=a,center=center,QZ_max=max(distances),size=t.size,projected_field_not_stationary=True))
        if a==2:
            fixture=i.g.run_center(t,J,.05+2j,out/'lifecycle_spectrum','D2i-lifecycle',old.d2.v1.Deadline(60),k=6,ncv=40)
            require(fixture['complete'],'lifecycle spectrum fixture failed')
    return results,fixture

def locks(out):
    affinity=os.sched_getaffinity(0);a=out/'locked';a.mkdir();(a/'.run.lock').touch()
    def fake(params,path,command=None,lock_fd=None):
        actual=os.fstat(lock_fd);expected=os.stat(path.parent/'.run.lock')
        require((actual.st_dev,actual.st_ino)==(expected.st_dev,expected.st_ino),'run lock not inherited');return dict(complete=True)
    try:
        with patch.object(i.g,'parameters',lambda source,*a,**kw:dict(source=source)),patch.object(i.g,'launch_point',fake):
            with (a/'.run.lock').open('r+') as lock:
                fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
                try:i.g.campaign(SimpleNamespace(run='upper720_N40',out=a))
                except BlockingIOError:pass
                else:raise AssertionError('busy run entered')
                require(i.g.campaign(SimpleNamespace(run='upper725_N40',out=out/'other'))==0,'independent run blocked')
    finally:os.sched_setaffinity(0,affinity)
    return dict(per_run_exclusive=True,independent_runs=True,worker_inherits_lock=True)

def main(out):
    out=i.output_path(out);out.mkdir(parents=True,exist_ok=False);start=time.monotonic()
    transfer=transfers(out);physical=operators(out);numeric,fixture=small_qz(out)
    result=dict(passed=True,transfers=transfer,N40_operators=physical,small_QZ=numeric,gates=gv.gates(out),
        memory=hv.limits(out),dispatch=hv.waves(out),locks=locks(out),lifecycle=gv.legacy.lifecycle(out,dict(center=fixture)),
        code_sha256=i.fingerprints(),seconds=time.monotonic()-start,production_started=False)
    old.write(out/'VALIDATION.json',result)
    print(json.dumps(dict(passed=True,seconds=result['seconds'],physical_transfer=transfer,
        N40_Frechet_max=max(x['centered_difference_relative'] for x in physical),
        N40_template_max=max(x['physical_template_relative'] for x in physical),
        small_QZ_max=max(x['QZ_max'] for x in numeric),runs=[x['run'] for x in physical])))

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--out',type=Path,required=True);a=p.parse_args();main(a.out)
