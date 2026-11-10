#!/usr/bin/env python3
"""Independent AS limits and blocking RSS admission for the four authorized parallel Ro."""
import contextlib,ctypes,fcntl,json,math,os,select,time
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]
OUT=ROOT/'results/ro_scan_20261101/S2'
PARALLEL_RO=(-.8,-.7,-.6,-.5)
GB=1_000_000_000
DRIVER_AS=32*GB
DRIVER_RSS=4*GB
WORKER_AS=200*GB
LEGACY_RESERVE=100*GB
CONTROL_RESERVE=8*GB
WORKER_POOL=256*GB
TASK_CAP=380*GB
TOKEN_BYTES=2*GB

def memory_profile(ro):
    if not any(abs(ro-r)<1e-9 for r in PARALLEL_RO):
        raise ValueError('S2_PARALLEL=1 is restricted to Ro=-0.8,-0.7,-0.6,-0.5; active -0.9 remains untouched')
    rows=json.loads((ROOT/'results/ro_scan_20261101/S1/plan.json').read_text())['rows']
    row=next(r for r in rows if abs(r['ro']-ro)<1e-9)
    nr={g['n']:g['intervals_fixed_during_root'] for g in row['grids']}
    reference=json.loads((OUT/'short_checks/final_review.json').read_text())['trial_points']
    baseline=max(p['peak_RSS_kb'] for p in reference)*1024
    stages={}
    for n in (32,40):
        for H in (4,5,6):
            # Cubic wall-normal factor is deliberately conservative for N40.
            rss=baseline*(nr[n]/749)*((2*H+1)/9)*((n+1)/33)**3
            limit=math.ceil((rss*1.25+3*GB)/TOKEN_BYTES)*TOKEN_BYTES
            stages[f'N{n}_H{H}']=dict(estimated_peak_RSS_bytes=int(math.ceil(rss)),RSS_reserved_bytes=limit,worker_AS_bytes=WORKER_AS,intervals=nr[n])
    return dict(ro=ro,driver_AS_bytes=DRIVER_AS,driver_hard_AS_bytes=WORKER_AS,driver_RSS_reserved_bytes=DRIVER_RSS,
                stages=stages,reference_peak_RSS_bytes=baseline,reference_N32_intervals=749,
                model='RSS_ref*(intervals/749)*((2H+1)/9)*((N+1)/33)^3; RSS reservation=ceil_2GB(1.25*RSS+3GB); independent worker AS=200GB')

def worker_limit(profile,n,H,intervals):
    stage=profile['stages'][f'N{n}_H{H}']
    if intervals>stage['intervals']:
        raise MemoryError('Grid exceeds reviewed parallel memory envelope; supervisor review required')
    return stage['worker_AS_bytes']

def worker_rss_reservation(profile,n,H,intervals):
    worker_limit(profile,n,H,intervals)
    return profile['stages'][f'N{n}_H{H}']['RSS_reserved_bytes']

def acquire_numerical_locks(out,parallel):
    """Parallel mode opens only the per-Ro numerical lock."""
    out=Path(out);out.mkdir(parents=True,exist_ok=True)
    global_lock=None;local_lock=None
    try:
        if not parallel:
            global_lock=(out.parent/'S2_numerical.lock').open('a')
            fcntl.flock(global_lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        local_lock=(out/'numerical.lock').open('a')
        fcntl.flock(local_lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        return global_lock,local_lock
    except BaseException:
        if local_lock is not None:local_lock.close()
        if global_lock is not None:global_lock.close()
        raise

class MemoryQueueTimeout(TimeoutError):pass

@contextlib.contextmanager
def reserve_worker_bytes(cap,deadline,pool=None,pool_bytes=WORKER_POOL,token_bytes=TOKEN_BYTES):
    """Reserve estimated RSS, independently of RLIMIT_AS; block on inotify, never poll.

    Passed token descriptors stay open in the worker. If its driver disappears,
    the kernel retains reservations until the worker exits.
    """
    pool=Path(pool) if pool is not None else OUT/'parallel_memory_tokens'
    if cap<=0 or cap>pool_bytes:raise MemoryError('Estimated worker RSS reservation exceeds the reviewed RSS pool')
    pool.mkdir(parents=True,exist_ok=True)
    count=pool_bytes//token_bytes;needed=math.ceil(cap/token_bytes)
    mutex=(pool.parent/(pool.name+'.lock')).open('a');tokens=[];selected=[];watch=-1
    lib=ctypes.CDLL(None,use_errno=True)
    try:
        fcntl.flock(mutex,fcntl.LOCK_EX)
        try:
            for i in range(count):tokens.append(os.open(pool/f'{i:03d}.token',os.O_CREAT|os.O_RDWR,0o600))
        finally:fcntl.flock(mutex,fcntl.LOCK_UN)
        watch=lib.inotify_init1(os.O_CLOEXEC)
        if watch<0 or lib.inotify_add_watch(watch,os.fsencode(pool),0x8|0x4)<0:
            raise OSError(ctypes.get_errno(),'memory-pool inotify')
        while True:
            fcntl.flock(mutex,fcntl.LOCK_EX)
            try:
                for fd in tokens:
                    try:fcntl.flock(fd,fcntl.LOCK_EX|fcntl.LOCK_NB)
                    except BlockingIOError:continue
                    selected.append(fd)
                    if len(selected)==needed:break
                if len(selected)==needed:break
                for fd in selected:fcntl.flock(fd,fcntl.LOCK_UN)
                selected=[]
            finally:fcntl.flock(mutex,fcntl.LOCK_UN)
            remaining=deadline-time.monotonic()
            if remaining<=0 or not select.select([watch],[],[],remaining)[0]:
                raise MemoryQueueTimeout('wall budget while waiting for memory tokens')
            os.read(watch,65536)
        yield tuple(selected)
    finally:
        # Do not explicitly unlock: inherited descriptors must retain ownership
        # if a timed-out driver's worker has not exited yet.
        for fd in tokens:os.close(fd)
        if watch>=0:os.close(watch)
        mutex.close()

def resource_plan():
    profiles=[memory_profile(ro) for ro in PARALLEL_RO]
    n32=sum(p['stages']['N32_H6']['RSS_reserved_bytes'] for p in profiles)
    assert LEGACY_RESERVE+4*DRIVER_RSS+WORKER_POOL+CONTROL_RESERVE==TASK_CAP
    assert n32<=WORKER_POOL
    assert max(s['RSS_reserved_bytes'] for p in profiles for s in p['stages'].values())<=WORKER_POOL
    return dict(schema='S2-parallel-RSS-resources-v2',accounting='estimated_RSS_with_margin',authorized_Ro=list(PARALLEL_RO),profiles=profiles,
                legacy_ro09_RSS_reserved_bytes=LEGACY_RESERVE,new_driver_RSS_sum_bytes=4*DRIVER_RSS,
                worker_RSS_token_pool_bytes=WORKER_POOL,control_RSS_reserved_bytes=CONTROL_RESERVE,
                S2_total_RSS_budget_bytes=TASK_CAP,S1c_reserved_bytes=550*GB,combined_RSS_planning_bytes=930*GB,
                worker_AS_bytes=WORKER_AS,parallel_driver_soft_AS_bytes=DRIVER_AS,parallel_driver_hard_AS_bytes=WORKER_AS,
                N32_H6_all_four_worker_RSS_reserved_bytes=n32,
                mode='4 simultaneous campaigns; estimated RSS reservations govern admission; AS is a separate per-process limit',
                reservation_policy='legacy RSS100 + four driver RSS16 + worker RSS tokens256 + controls RSS8 =380GB; RSS estimates include safety margin')
