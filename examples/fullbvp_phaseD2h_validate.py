#!/usr/bin/env python3
"""D2h bounded validation: actual v5 recertification plus scheduler/limit checks."""
import os
os.environ['PYTHONDONTWRITEBYTECODE']='1'
import argparse,copy,gc,json,sys,threading,time
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch
import numpy as np
import fullbvp_phaseD2h_campaign as h
import fullbvp_phaseD2g_validate as gv
old=h.old

def require(ok,message):
    if not ok:raise AssertionError(message)

def cache(out,runs):
    out=h.output_path(out);out.mkdir(parents=True,exist_ok=False);start=time.monotonic();points=[]
    for run in runs:
        cfg=h.RUNS[run]
        for source in cfg['sources']:
            params=h.parameters(source,cfg['n'],cfg['H'],cfg['dr'],run_name=run)
            point=out/run/source;deadline=old.d2.v1.Deadline(300)
            t,correction=h.prepare_correction(params,point,deadline)
            accepted=h.prepare_centers(params,t,point,deadline)
            require(len(accepted)==len(params['reuse_v5']['centers']),'lost completed v5 centers')
            # A real assembler or inverse call would fail: all cached centers must be replayable with J=None.
            if len(accepted)==5:
                result=h.scan_centers(t,None,params,point,deadline,accepted)
                require(all(c['complete'] for c in result),'fully cached point did not complete')
            resumed=h.prepare_centers(params,t,point,deadline)
            require(resumed==accepted,'cache resume changed/recomputed a center')
            points.append(dict(run=run,source=source,params=params,Rout=correction['after']['Rout'],
                corrected_field_sha256=correction['fields_sha256'],accepted=accepted,
                missing=[i for i in range(5) if i not in accepted],
                original_pair_max=max((r['original_relative'] for c in accepted.values() for r in c['eigenpairs']),default=None),
                factorization_count=0))
            del t,accepted,resumed;gc.collect()
    result=dict(passed=True,seconds=time.monotonic()-start,points=points,production_started=False)
    old.write(out/'VALIDATION.json',result)
    print(json.dumps(dict(passed=True,seconds=result['seconds'],points=len(points),
        reused_centers=sum(len(p['accepted']) for p in points),missing_centers=sum(len(p['missing']) for p in points))))


def limits(out):
    slotroot=out/'slots'
    with h.memory_slot(slotroot,blocking=False) as a:
        with h.memory_slot(slotroot,blocking=False) as b:
            require({a,b}=={0,1},'slots not independent')
            try:
                with h.memory_slot(slotroot,blocking=False):pass
            except BlockingIOError:pass
            else:raise AssertionError('third 400 GB worker admitted')
        with h.memory_slot(slotroot,blocking=False) as c:require(c==b,'released slot not reusable')
    obj=object.__new__(h.ResourceFactor);obj.iparm=np.zeros(64,dtype=int);obj.info={};obj.stats_path=None
    obj.iparm[16]=250_000_000_000//1024
    with patch.object(old.d2,'memory',return_value={'VmRSS':1_000_000_000}),patch.object(old.d2.PardisoComplex,'_call',return_value=37):
        require(obj._call(22,None)==37,'400 GB guard still rejects a 313.5 GB estimate')
        obj.iparm[16]=350_000_000_000//1024
        try:obj._call(22,None)
        except MemoryError:pass
        else:raise AssertionError('memory estimate above 400 GB accepted')
    return dict(two_slots_only=True,released_slot_reusable=True,old_100_GB_guard_removed=True,
                oversized_estimate_rejected=True,aggregate_address_space_bound_bytes=806_000_000_000)


def order(out):
    calls=[];accepted={1:dict(complete=True),3:dict(complete=True)}
    params=dict(shifts=[old.d2.v1.zpair(z) for z in h.SHIFTS],k=6,ncv=40,center_budget_seconds=h.CENTER_BUDGET)
    def fake(t,J,sigma,path,fingerprint,deadline,**kw):
        calls.append(dict(sigma=old.d2.v1.zpair(sigma),remaining=deadline.remaining()))
        if len(calls)==1:raise RuntimeError('injected real-axis failure')
        return dict(complete=True)
    with patch.object(h.g,'run_center',fake):
        result=h.scan_centers(None,None,params,out/'order',old.d2.v1.Deadline(h.POINT_BUDGET),accepted)
    require([r['sigma'] for r in calls]==[params['shifts'][i] for i in (0,2,4)],'cache was recomputed or real axis not first')
    require(all(4790<r['remaining']<=4800 for r in calls),'4800 second center budget not propagated')
    require(not result[0]['complete'] and all(r['complete'] for r in result[1:]),'failure prevented later centers')
    return dict(real_axis_first=True,only_missing_centers_called=True,failed_center_independent=True,center_budget=4800)


def waves(out):
    ready=out/'ready.fifo';release=out/'release.fifo';os.mkfifo(ready);os.mkfifo(release)
    rfd=os.open(ready,os.O_RDWR);wfd=os.open(release,os.O_RDWR);arrivals=[]
    def coordinate():
        for count in (2,2,1):
            data=b''
            while len(data)<count:data+=os.read(rfd,count-len(data))
            arrivals.append(data.decode());os.write(wfd,b'x'*count)
    thread=threading.Thread(target=coordinate,daemon=True);thread.start();specs=[]
    for i in range(5):
        record=out/f'cpu_{i}.json'
        code=('import os,json,pathlib; cpus={int(x) for x in os.environ["D2G_CPUSET"].split(",")}; '
              'os.sched_setaffinity(0,cpus); pathlib.Path('+repr(str(record))+').write_text(json.dumps(sorted(os.sched_getaffinity(0)))); '
              'w=os.open('+repr(str(ready))+',os.O_WRONLY); os.write(w,'+repr(str(i).encode())+'); os.close(w); '
              'r=os.open('+repr(str(release))+',os.O_RDONLY); os.read(r,1); os.close(r)')
        specs.append((f'fixture_{i}',[sys.executable,'-c',code]))
    try:
        result=h.execute_waves(specs,out/'waves',timeout_seconds=15);thread.join(timeout=1)
        require(result['complete'] and [set(x) for x in arrivals]==[set('01'),set('23'),set('4')],'runs did not execute in bounded waves')
        cpus=[set(json.loads((out/f'cpu_{i}.json').read_text())) for i in range(5)]
        require(all(len(s)==16 for s in cpus) and len(cpus[0]|cpus[1])==32 and len(cpus[2]|cpus[3])==32,'CPU bound incorrect')
    finally:os.close(rfd);os.close(wfd)
    return dict(two_concurrent_processes=True,wave_sizes=[2,2,1],threads_per_run=16,total_threads=32,lightweight_fixture_only=True)


def functional(out):
    out=h.output_path(out);out.mkdir(parents=True,exist_ok=False);start=time.monotonic()
    numeric=gv.numerical(out);gates=gv.gates(out)
    result=dict(passed=True,numerical=numeric,gates=gates,limits=limits(out),order=order(out),waves=waves(out),
                locks=gv.run_locks(out),lifecycle=gv.legacy.lifecycle(out,dict(center=numeric['centers'][1]['center'])),
                seconds=time.monotonic()-start,production_started=False)
    # Invalid imported metadata is rejected before any eigensolver is started.
    p=h.parameters('lower719p948669_H4',run_name='baseline_rerun_lower')
    bad=copy.deepcopy(p);key=next(iter(bad['reuse_v5']['sha256']));bad['reuse_v5']['sha256'][key]='0'*64
    try:h.prepare_correction(bad,out/'bad_hash',old.d2.v1.Deadline(30))
    except RuntimeError:result['corrupt_import_rejected']=True
    else:raise AssertionError('corrupt v5 hash accepted')
    old.write(out/'VALIDATION.json',result)
    print(json.dumps(dict(passed=True,seconds=result['seconds'],QZ_max=max(max(c['QZ_distances']) for c in numeric['centers']),
        pair_max=max(r['original_relative'] for c in numeric['centers'] for r in c['center']['eigenpairs']),
        limits=result['limits'],waves=result['waves'])))

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('mode',choices=['cache','functional']);p.add_argument('--out',type=Path,required=True)
    p.add_argument('--runs',nargs='+',choices=list(h.RUNS));a=p.parse_args()
    if a.mode=='cache':cache(a.out,a.runs)
    else:functional(a.out)
