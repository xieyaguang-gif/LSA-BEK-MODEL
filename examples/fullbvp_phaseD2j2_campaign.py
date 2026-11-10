#!/usr/bin/env python3
"""D2j2: isolated v8 continuation; supervisor launches production separately."""
import os
os.environ['PYTHONDONTWRITEBYTECODE'] = '1'
for key in ('OPENBLAS_NUM_THREADS', 'OMP_NUM_THREADS', 'MKL_NUM_THREADS',
            'NUMEXPR_NUM_THREADS', 'BLIS_NUM_THREADS'):
    os.environ[key] = '1'
import argparse
import fcntl
import json
import resource
import selectors
import subprocess
import sys
import time
import uuid
from pathlib import Path
import fullbvp_phaseD2j_campaign as j

ROOT, OUT = j.ROOT, j.OUT
LEGACY = OUT / 'production_v8'
PROD = Path(os.environ.get('D2J2_RUN_ROOT', OUT / 'production_v8c')).absolute()
AUDIT = OUT / 'D2j2'
POINT_BUDGET = 172800.  # Includes the v8 run's already charged time.
CENTER_BUDGET = 14400.  # Outer wall-clock cap, including startup/assembly/certification.
CENTER_COOPERATIVE = 14280.
PARENT_MEMORY = 32_000_000_000
CENTER_MEMORY = 165_000_000_000
CONTROLLER_MEMORY = 2_000_000_000
WORKER_MEMORY = PARENT_MEMORY + CENTER_MEMORY
TASK_MEMORY = 2 * (WORKER_MEMORY + CONTROLLER_MEMORY) + CONTROLLER_MEMORY
ENTRY = Path(__file__).resolve()
_base_source_point = j.source_point
_base_parameters = j.parameters
_base_fingerprints = j.fingerprints
_base_verify = j.verify_worker_evidence
_base_save_spec = j.save_spec


def source_point(branch, n=24, a=2.):
    if n == 24:
        return LEGACY / (branch + '_N24') / (branch + '_H6')
    return _base_source_point(branch, n, a)


def fingerprints():
    return dict(_base_fingerprints(), **{
        str(p): j.old.d2.v1.sha(p) for p in
        (ENTRY, ROOT / 'examples/fullbvp_phaseD2j2_validate.py')})


def parameters(branch, n=24, a=2., shifts=None):
    params = _base_parameters(branch, n, a, shifts)
    params.update(revision='D2j3', center_cooperative_seconds=CENTER_COOPERATIVE,
                  parent_memory_bytes=PARENT_MEMORY, center_memory_bytes=CENTER_MEMORY,
                  carryover_used_seconds=0.)
    if n == 24:
        point = source_point(branch)
        if (point / 'ACTIVE.json').exists():
            raise RuntimeError('v8 source still has an ACTIVE reservation; supervisor must resolve it first')
        ledger = point / 'budget.json'
        used = float(j.read(ledger)['used_seconds'])
        if not 0 <= used < POINT_BUDGET:
            raise RuntimeError('invalid/exhausted source budget')
        params['carryover_used_seconds'] = used
        params['seed']['sha256'][str(ledger)] = j.old.d2.v1.sha(ledger)
    return params


def initialize_budget(params, point):
    ledger = point / 'budget.json'
    if ledger.exists():
        return  # The inherited launcher charges ACTIVE reservations; never reset this ledger.
    if (point / 'manifest.json').exists() or (point / 'ACTIVE.json').exists():
        raise RuntimeError('existing continuation lost its budget ledger')
    j.old.write(ledger, dict(used_seconds=params['carryover_used_seconds'],
                            hard_budget_seconds=params['budget_seconds'],
                            inherited_from=str(source_point(params['branch'])) if params['n'] == 24 else None))


def wait_once(proc, seconds):
    """One kernel completion wait; no status-query loop or timed waitpid polling."""
    fd = os.pidfd_open(proc.pid)
    try:
        with selectors.DefaultSelector() as selector:
            selector.register(fd, selectors.EVENT_READ)
            if not selector.select(max(0., seconds)):
                raise subprocess.TimeoutExpired(proc.args, seconds)
        return proc.wait()
    finally:
        os.close(fd)


def center_child(job):
    params, point = job['params'], j.output_path(job['point'])
    if params != parameters(params['branch'], params['n'], params['mapping']['a'],
                            [x[1] for x in params['shifts']]):
        raise RuntimeError('center source/configuration changed')
    j.h.control_limit()
    resource.setrlimit(resource.RLIMIT_AS, (CENTER_MEMORY, CENTER_MEMORY))
    deadline = j.old.d2.v1.Deadline(job['cooperative_seconds'])
    temporal, _ = j.prepare_state(params, point, deadline)
    matrix, assembly = j.old.d2.assemble(temporal, deadline)
    if j.np.iscomplexobj(matrix.data) and j.np.any(matrix.data.imag):
        raise RuntimeError('packed pencil must be real for positive-half-plane reduction')
    target = j.output_path(job['target'])
    j.old.write(target / 'assembly.json', assembly)
    result = j.g.run_center(temporal, matrix, complex(*job['sigma']), target,
                            j.old.d2.v1.digest(params), deadline, k=6, ncv=40)
    return 0 if result['complete'] else 1


def run_center(params, point, index, imag, deadline, lock_fd):
    target = point / f'center_{index:02d}'
    target.mkdir(parents=True, exist_ok=True)
    if (target / 'DONE.json').exists():
        raise RuntimeError('completed center must go through recertifying cache')
    (target / 'FAILED.json').unlink(missing_ok=True)
    launch = target / 'launches' / f'{time.time_ns()}_{uuid.uuid4().hex[:8]}'
    cap = min(CENTER_BUDGET, deadline.remaining())
    if cap <= 10:
        raise RuntimeError('insufficient remaining point budget for a new center')
    job = dict(params=params, point=str(point), target=str(target), sigma=[.05, imag],
               cooperative_seconds=min(CENTER_COOPERATIVE, cap - 10.))
    j.old.write(launch / 'job.json', job)
    command = [sys.executable, str(ENTRY), 'center', '--job', str(launch / 'job.json')]
    start, proc, failure = time.monotonic(), None, None
    try:
        with (launch / 'worker.log').open('ab') as log:
            # Same process group as the point worker: outer group cancellation includes this child.
            proc = subprocess.Popen(command, cwd=ROOT, stdout=log, stderr=subprocess.STDOUT,
                                    pass_fds=(lock_fd,))
            rc = wait_once(proc, cap - 5. - (time.monotonic() - start))
            if rc:
                failure = f'center child exit {rc}'
    except subprocess.TimeoutExpired:
        failure = 'BudgetExpired: hard center wall-clock limit'
    finally:
        if proc is not None and proc.returncode is None:
            try:
                proc.kill()  # Only this run's own center subprocess.
            except ProcessLookupError:
                pass
            proc.wait()
    elapsed = time.monotonic() - start
    marker = target / ('DONE.json' if failure is None else 'FAILED.json')
    result = j.read(marker) if marker.exists() else dict(
        fingerprint=j.old.d2.v1.digest(params), sigma=[.05, imag], complete=False, k=6, ncv=40)
    if failure is not None or elapsed > cap:
        reason = failure or 'center exceeded hard wall-clock budget'
        if result.get('failure'):
            reason += ': ' + result['failure']
        result.update(complete=False, failure=reason)
    result['center_execution'] = dict(seconds=elapsed, hard_limit_seconds=cap,
        cooperative_seconds=job['cooperative_seconds'], child_memory_bytes=CENTER_MEMORY,
        wait='single pidfd completion wait', launch=str(launch))
    marker = target / ('DONE.json' if result['complete'] else 'FAILED.json')
    j.old.write(marker, result)
    (target / ('FAILED.json' if result['complete'] else 'DONE.json')).unlink(missing_ok=True)
    if result.get('arrays'):
        j.old.write(Path(result['arrays']).parent / 'result.json', result)
    return result


def worker(job, lock_fd):
    params, out = job['params'], j.output_path(job['out'])
    actual, expected = os.fstat(lock_fd), os.stat(out.parent / '.run.lock')
    if (actual.st_dev, actual.st_ino) != (expected.st_dev, expected.st_ino):
        raise RuntimeError('wrong inherited run lock')
    if params != parameters(params['branch'], params['n'], params['mapping']['a'],
                            [x[1] for x in params['shifts']]):
        raise RuntimeError('worker source/configuration changed')
    j.h.control_limit()
    start = time.monotonic()
    deadline = j.old.d2.v1.Deadline(job['remaining'])
    centers, failure, localized = [], None, []
    with j.h.memory_slot(PROD / '.memory_slots') as slot:
        deadline.check()
        resource.setrlimit(resource.RLIMIT_AS, (PARENT_MEMORY, j.h.MEMORY_BYTES))
        j.old.write(out / 'resources.json', dict(slot=slot, threads=16,
            parent_memory_bytes=PARENT_MEMORY, center_memory_bytes=CENTER_MEMORY,
            controller_memory_bytes=CONTROLLER_MEMORY, route_bound_bytes=199_000_000_000,
            aggregate_address_space_bound=TASK_MEMORY))
        temporal, correction = j.prepare_state(params, out, deadline)
        frequencies = [x[1] for x in params['shifts']]
        try:
            index = 0
            while True:
                deadline.check()
                if index == len(frequencies):
                    if not params['adaptive'] or j.coverage(centers)['complete']:
                        break
                    new = j.next_center(centers)
                    if new is None or index >= j.MAX_CENTERS:
                        raise RuntimeError('coverage gaps remain at center/minimum-spacing limit')
                    frequencies.append(new)
                imag = frequencies[index]
                result = j.cached_center(params, temporal, out, index, imag, deadline)
                if result is None:
                    result = run_center(params, out, index, imag, deadline, lock_fd)
                centers.append(result)
                j.old.write(out / 'scan_progress.json', dict(
                    centers=[j.center_summary(c) for c in centers], coverage=j.coverage(centers)))
                if not result['complete']:
                    raise RuntimeError('center failed: ' + str(result.get('failure')))
                index += 1
        except Exception as exc:
            failure = f'{type(exc).__name__}: {exc}'
        refs = j.selected_pairs(centers, all_nonphase=params['n'] == 40)
        best = max(j.selected_pairs(centers, all_nonphase=True), key=lambda p: p['value'][0], default=None)
        if best is not None and not any(r['arrays'] == best['arrays'] and r['index'] == best['index'] for r in refs):
            refs.append(best)
        try:
            localized = j.localize(temporal, params, refs, deadline)
        except Exception as exc:
            failure = failure or f'localization: {type(exc).__name__}: {exc}'
        cov = j.coverage(centers)
        result = dict(fingerprint=j.old.d2.v1.digest(params), correction=correction, centers=centers,
            coverage=cov, complete=bool(failure is None and len(centers) >= len(params['shifts'])
                                       and (not params['adaptive'] or cov['complete'])),
            failure=failure, candidates=j.selected_pairs(centers), localization=localized,
            max_nonphase=best, frequency=None if best is None else j.frequency_relation(best['value'], temporal.p.config.omega),
            seconds=time.monotonic()-start, peak_rss_bytes=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss*1024,
            original_resolution_certified=False, exhaustive_stability=False)
        j.old.write(out / 'worker_result.json', result)
        return 0 if result['complete'] else 1


def verify_worker_evidence(params, out, child):
    _base_verify(params, out, child)
    for center in child['centers']:
        if center.get('reused_eigenpairs'):
            continue
        execution = center.get('center_execution', {})
        if not (0 < execution.get('seconds', float('inf')) <= execution.get('hard_limit_seconds', 0) <= CENTER_BUDGET):
            raise RuntimeError('missing/invalid hard center limit certificate')


def campaign(spec_path, run_out):
    j.h.control_limit()
    params, out = j.read(spec_path), j.output_path(run_out)
    os.sched_setaffinity(0, set(sorted(os.sched_getaffinity(0))[:16]))
    out.mkdir(parents=True, exist_ok=True)
    fd = os.open(out / '.run.lock', os.O_RDWR | os.O_CREAT | os.O_CLOEXEC | os.O_NOFOLLOW, 0o600)
    with os.fdopen(fd, 'r+') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        point = out / params['source']
        initialize_budget(params, point)
        command = [sys.executable, str(ENTRY), 'worker', '--job', str(point / 'job.json'), '--lock-fd', str(lock.fileno())]
        result = j._saved_launch(params, point, command=command, lock_fd=lock.fileno())
        return 0 if result['complete'] else 1


def save_spec(root, name, params):
    name, command = _base_save_spec(root, name, params)
    command[1] = str(ENTRY)
    return name, command


def cpu_groups(count):
    # D2k binds its single worker to the first 16 allowed CPUs. Leave that block idle here.
    available = sorted(os.sched_getaffinity(0))[16:]
    if not 1 <= count <= 2 or len(available) < count * 16:
        raise RuntimeError('need 32 CPUs in addition to the 16 reserved for D2k')
    return [available[i*16:(i+1)*16] for i in range(count)]


def install():
    # Mutate only this interpreter's imports. No D2k/shared source files are changed.
    j.PROD, j.AUDIT = PROD, AUDIT
    j.BASE_IMAG = tuple(sorted(set(range(0, 71, 5)) | {2., 7.5}))
    j.POINT_BUDGET, j.CENTER_BUDGET = POINT_BUDGET, CENTER_BUDGET
    j.source_point, j.parameters, j.fingerprints = source_point, parameters, fingerprints
    j.save_spec = save_spec
    j.h.POINT_BUDGET, j.h.CENTER_BUDGET = POINT_BUDGET, CENTER_BUDGET
    j.h.MEMORY_BYTES = j.old.MEMORY_BYTES = WORKER_MEMORY
    j.h.TOTAL_BYTES = TASK_MEMORY
    j.g.cpu_groups = cpu_groups
    j.old.TARGET_SECONDS = POINT_BUDGET
    j.old.verify_worker_evidence = verify_worker_evidence


install()
if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest='mode', required=True)
    for mode in ('dispatch', 'inventory', 'report'):
        sub.add_parser(mode).add_argument('--root', type=Path, required=True)
    p = sub.add_parser('campaign')
    p.add_argument('--spec', type=Path, required=True)
    p.add_argument('--out', type=Path, required=True)
    p = sub.add_parser('worker')
    p.add_argument('--job', type=Path, required=True)
    p.add_argument('--lock-fd', type=int, required=True)
    p = sub.add_parser('center')
    p.add_argument('--job', type=Path, required=True)
    args = parser.parse_args()
    if args.mode == 'dispatch':
        raise SystemExit(j.dispatch(args.root))
    if args.mode == 'inventory':
        j.inventory(args.root)
    if args.mode == 'report':
        j.report_results(j.output_path(args.root, allow_root=True))
    if args.mode == 'campaign':
        raise SystemExit(campaign(j.output_path(args.spec), args.out))
    if args.mode == 'worker':
        raise SystemExit(worker(j.read(j.output_path(args.job)), args.lock_fd))
    if args.mode == 'center':
        raise SystemExit(center_child(j.read(j.output_path(args.job))))
