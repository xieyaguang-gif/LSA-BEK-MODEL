#!/usr/bin/env python3
"""Supervisor-run D1 campaign. --prepare performs no production correction."""
from __future__ import annotations

import os
for _key in ('OPENBLAS_NUM_THREADS', 'OMP_NUM_THREADS', 'MKL_NUM_THREADS',
             'NUMEXPR_NUM_THREADS', 'VECLIB_MAXIMUM_THREADS', 'BLIS_NUM_THREADS'):
    os.environ[_key] = '1'

import argparse
import fcntl
import subprocess
import sys
import time
import traceback
from pathlib import Path

import numpy as np

import fullbvp_phaseD1_solver as solver
from fullbvp_phaseD1_solver import OUT, ROOT, SOURCE, BASE_PARAMS, Mapping, read, write, sha, fingerprint, savez
from fullbvp_phaseD1_diagnostics import diagnostic

BRANCHES = {'strength14': ('strength', 14.), 'strength56': ('strength', 56.),
            'width20': ('width', 20.), 'width80': ('width', 80.),
            'gap70': ('gap', 70.), 'gap140': ('gap', 140.)}
POINT_BUDGET = 4800.
INITIAL_STEP, MIN_STEP, MAX_STEP = .125, 1/1024, .25
TAIL_LIMIT = .01


class LockBusy(RuntimeError):
    pass


def lock(path):
    path.parent.mkdir(parents=True, exist_ok=True)
    handle = path.open('a')
    try:
        fcntl.flock(handle, fcntl.LOCK_EX|fcntl.LOCK_NB)
    except BlockingIOError:
        handle.close()
        raise LockBusy(f'Active process holds {path}')
    return handle


def marker(folder, accepted, value):
    """Write the terminal marker atomically; never leave contradictory markers."""
    folder = Path(folder)
    opposite = folder/('FAILED.json' if accepted else 'DONE.json')
    opposite.unlink(missing_ok=True)
    write(folder/('DONE.json' if accepted else 'FAILED.json'), value)


def manifest():
    source = solver.verified_source()
    files = [Path(__file__), Path(solver.__file__), ROOT/'examples/fullbvp_phaseD1_diagnostics.py',
             ROOT/'examples/fullbvp_phaseD1_validate.py',
             ROOT/'examples/fullbvp_phaseB_solver.py', ROOT/'examples/fullbvp_phaseC_solver.py',
             ROOT/'examples/appelquist2015_hopf_common.py', ROOT/'examples/fullbvp_phaseB_report.py',
             ROOT/'src/llsa/secondary/global_radial.py', ROOT/'src/llsa/secondary/radial.py',
             ROOT/'examples/appelquist2015_hopf_block_corrector.py',
             ROOT/'examples/appelquist2016_branch_continuation.py']
    files += [Path(module.__file__) for name, module in sorted(sys.modules.items())
              if name.startswith('llsa') and getattr(module, '__file__', None) and
              Path(module.__file__).is_relative_to(ROOT/'src/llsa')]
    files = sorted(set(files))
    return dict(version=1, source=source, fixed=dict(Ro=-1., beta=68., Rin=400., Rout=725., n=32, H=6, intervals=800),
                baseline=BASE_PARAMS, endpoints=BRANCHES, strict=solver.STRICT,
                continuity=solver.CONTINUITY, highest_to_fundamental_limit=TAIL_LIMIT,
                initial_step=INITIAL_STEP, min_step=MIN_STEP, max_step=MAX_STEP,
                single_point_hard_timeout_seconds=POINT_BUDGET,
                runtime=dict(python=sys.version, numpy=np.__version__, scipy=sys.modules['scipy'].__version__),
                numerical_threads=1, total_concurrent_workers=1, GPU=False,
                code_hashes={str(path.relative_to(ROOT)): sha(path) for path in files},
                pinch_curve_sha256=sha(ROOT/'results/fullbvp_global_20261030/phaseA/A3/curve_N32_global.json'))


def prepare():
    OUT.mkdir(parents=True, exist_ok=True)
    request = manifest()
    request_id = fingerprint(request)
    path = OUT/'request.json'
    if path.exists():
        existing = read(path)
        if existing['fingerprint'] != request_id:
            raise ValueError('D1 request/source/code fingerprint changed; retain outputs and review before a new campaign')
    else:
        write(path, dict(fingerprint=request_id, request=request))
    baseline_path = OUT/'diagnostics/baseline.json'
    baseline = read(baseline_path) if baseline_path.exists() else diagnostic(SOURCE/'fields.npz', BASE_PARAMS, 'baseline')
    state_path = OUT/'campaign.json'
    if state_path.exists():
        state = read(state_path)
        if state['request_fingerprint'] != request_id:
            raise ValueError('D1 state request fingerprint mismatch')
    else:
        state = dict(request_fingerprint=request_id, status='prepared', production_launched=False,
                     branches={name: dict(status='pending', fraction=0., step=INITIAL_STEP,
                                          accepted_fields=str(SOURCE/'fields.npz'),
                                          accepted_fields_sha256=solver.EXPECTED_SOURCE['fields.npz'],
                                          accepted_summary_sha256=solver.EXPECTED_SOURCE['summary.json'],
                                          attempts=[], inflight=None)
                               for name in BRANCHES})
        write(state_path, state)
    if not (OUT/'endpoints.json').exists():
        write(OUT/'endpoints.json', dict(baseline=baseline, endpoints={name: None for name in BRANCHES}))
    render(state, baseline)
    return state, baseline


def render(state, baseline):
    lines = ['# D1: independent outer-sponge homotopies', '',
             f"Status: {state['status']}. Fixed Ro=-1, beta=68, R=400–725, N32, H6, 800 intervals.",
             'Every branch starts independently at the pinned accepted Phase C baseline. No production run was launched during implementation/validation.', '',
             '| Endpoint | status | fraction reached | omega | Δomega | Fig.8 | ΔFig.8 | 2beta | Δ2beta | g_q | Δg_q | q / R³ sponge fractions | Δq / ΔR³ |',
             '|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|']
    fmt = lambda v: 'missing' if v is None else f'{v:.9g}'
    for name, branch in state['branches'].items():
        value = branch.get('endpoint')
        cells = ['pending']*10 if value is None else [cell for key in ('omega', 'front_fig8', 'front_2beta_absolute', 'g_q_570_600') for cell in (fmt(value[key]), fmt(value['endpoint_minus_baseline'][key]))]+[fmt(value['sponge_fraction_q'])+' / '+fmt(value['sponge_fraction_cylindrical_R3']), fmt(value['endpoint_minus_baseline']['sponge_fraction_q'])+' / '+fmt(value['endpoint_minus_baseline']['sponge_fraction_cylindrical_R3'])]
        lines.append('| '+ ' | '.join([name, branch['status'], f"{branch['fraction']:.6g}"]+cells)+' |')
    lines += ['', f"Verified baseline: omega={baseline['omega']:.14g}, Fig.8={baseline['front_fig8']:.14g}, 2beta={baseline['front_2beta_absolute']:.14g}; g_q={baseline['g_q_570_600']:.14g} (LS {baseline['g_q_least_squares_570_600']:.14g}); sponge fractions q={baseline['sponge_fraction_q']:.14g}, R³={baseline['sponge_fraction_cylindrical_R3']:.14g}.", '',
              'Source identity and code hashes: request.json. Validation evidence: validation.json. Only accepted, branch-continuous fields generate endpoints and endpoint-minus-baseline deltas in endpoints.json and diagnostics/*.json.',
              'Independent review and tool-call exceptions are recorded in [REVIEW.md](REVIEW.md); the original guard evidence is preserved.',
              'Front definitions match Phase C: Fig.8 uses the first downward RMS-growth/pinch crossing in the covered interval before actual a; 2beta uses direct q_v,2 crossing 1e-6, with its 5% convention also retained. q0 and pressure are excluded from coherent energies. The original end-to-end g_q and the additional least-squares fit are both retained.',
              'Sponge fractions are coherent energy fractions in the geometric support region R>=a, not damping dissipation rates. For gap changes the full-domain denominator uses the same domain while the integration start changes; fraction deltas therefore combine field changes and changes to the measured region.',
              'For gap=140 the sponge starts at a=585, inside R570–600 and upstream of the baseline Fig.8 front. Missing crossings in this shorter search are censored, not evidence that a physical front disappeared. The covered-pinch-domain crossing ignoring sponge is an auxiliary diagnostic only. Pinch coverage and sampled front-search bounds remain explicit in every diagnostic.', '',
              '## Supervisor launch and resume', '',
              f'Launch or resume with `bash {OUT}/commands.sh`. It detaches one sequential CPU process; logs are in production.log and the launch PID is in production.pid. The script passes --retry-failed, so failed branches are retried from their last accepted state, while completed endpoints are reused.',
              f'Foreground equivalent: `{ROOT}/.venv/bin/python {ROOT}/examples/fullbvp_phaseD1_campaign.py --run --retry-failed`.',
              'Each correction runs in a separate subprocess with a 4800-second hard timeout, including startup and blocking sparse factors. Every numerical-library thread variable is 1. At most one numerical worker runs; the controller is nonnumerical except lightweight continuity/diagnostics. No GPU is used.',
              'Campaign and worker flock locks prevent concurrent duplicate runs and survive stale lock files. Interrupted same-request Newton checkpoints resume; source/request/code changes are rejected. Failed steps halve the homotopy step down to 1/1024; a failed endpoint stays failed and is never interpolated. A fold or stricter-than-necessary continuity screen can therefore leave an endpoint unresolved.',
              'DONE.json exists only when all six endpoints are accepted. FAILED.json lists unresolved branches after a completed unsuccessful campaign. Crashes leave running/pending state for explicit supervisor resume. Do not infer endpoint values from accepted intermediate points.',
              'The six jobs can require many hours in total. No numerical result for the six target changes exists until supervisor execution completes.']
    (OUT/'REPORT.md').write_text('\n'.join(lines)+'\n')


def parameters(name, fraction):
    key, target = BRANCHES[name]
    params = BASE_PARAMS.copy()
    params[key] += fraction*(target-params[key])
    return params


def load_y(path):
    with np.load(path) as data:
        return data['y'].copy()


def accepted_fields(path, expected_sha=None, expected_summary_sha=None):
    """Refuse stale, modified or incorrectly labelled accepted checkpoints."""
    path = Path(path)
    if expected_sha is not None and sha(path) != expected_sha:
        raise ValueError('Committed accepted fields hash mismatch')
    if expected_summary_sha is not None and sha(path.parent/'summary.json') != expected_summary_sha:
        raise ValueError('Committed accepted summary hash mismatch')
    if path == SOURCE/'fields.npz':
        if sha(path) != solver.EXPECTED_SOURCE['fields.npz']:
            raise ValueError('Baseline field hash mismatch')
        return load_y(path)
    summary, request = read(path.parent/'summary.json'), read(path.parent/'request.json')
    request_id = request.pop('fingerprint')
    if (fingerprint(request) != request_id or not summary.get('accepted') or
            summary.get('request_fingerprint') != request_id or
            request['campaign'] != read(OUT/'request.json')['fingerprint'] or
            summary.get('fields_sha256') != sha(path)):
        raise ValueError('Accepted fields request/summary/hash mismatch')
    with np.load(path) as saved:
        if (not bool(saved['accepted']) or saved['request_fingerprint'].item() != request_id or
                not np.isfinite(saved['y']).all() or
                any(saved[key].item() != value for key, value in request['sponge'].items()) or
                any(saved[key].item() != request[key] for key in ('n', 'H', 'intervals', 'Rout')) or
                float(saved['omega']) != summary['omega'] or
                float(saved['a']) != request['Rout']-request['sponge']['gap'] or
                not np.array_equal(saved['y'][:-2], solver.pack(saved['field']))):
            raise ValueError('Accepted checkpoint metadata mismatch')
        return saved['y'].copy()


def worker(job):
    with lock(job/'worker.lock'):
        request = read(job/'request.json')
        request_id = request.pop('fingerprint')
        if fingerprint(request) != request_id:
            raise ValueError('Worker request fingerprint mismatch')
        initial_path = job/'predictor.npz'
        if sha(initial_path) != request['predictor_sha256']:
            raise ValueError('Predictor hash mismatch')
        if fingerprint(manifest()) != read(OUT/'request.json')['fingerprint']:
            raise ValueError('Source/code changed before worker dispatch')
        mapping = Mapping(**request['sponge'])
        resume = job/'checkpoint.npz'
        _, result = solver.correct(mapping, load_y(initial_path), job, request_id,
                                   budget=POINT_BUDGET, resume=resume if resume.exists() else None)
        marker(job, result['accepted'], result)
        return 0 if result['accepted'] else 1


def run_attempt(job):
    request = read(job/'request.json')
    request_id = request.pop('fingerprint')
    if (fingerprint(request) != request_id or
            request['campaign'] != read(OUT/'request.json')['fingerprint']):
        raise ValueError('Cached/inflight attempt request fingerprint mismatch')
    if (job/'summary.json').exists():
        record = read(job/'summary.json')
        if record.get('accepted'):
            accepted_fields(job/'fields.npz')
        return record
    # One lock check on restart; no polling or waiting for another worker.
    with lock(job/'worker.lock'):
        pass
    try:
        with (job/'worker.log').open('a') as log:
            # The OS timer survives a controller crash; the Python timeout also
            # bounds normal scheduling. timeout uses its own process group.
            result = subprocess.run(['/usr/bin/timeout', '--signal=KILL', str(POINT_BUDGET-1),
                                     sys.executable, str(Path(__file__).resolve()), '--worker', str(job)],
                                    cwd=ROOT, stdout=log, stderr=subprocess.STDOUT,
                                    timeout=POINT_BUDGET, check=False)
    except subprocess.TimeoutExpired:
        result = dict(accepted=False, failure='Hard single-point timeout',
                      hard_timeout_seconds=POINT_BUDGET, checkpoint_preserved=(job/'checkpoint.npz').exists())
        write(job/'summary.json', result)
        marker(job, False, result)
        return result
    if not (job/'summary.json').exists():
        failure = dict(accepted=False, failure='Worker exited without summary', returncode=result.returncode,
                       checkpoint_preserved=(job/'checkpoint.npz').exists())
        write(job/'summary.json', failure)
        marker(job, False, failure)
        return failure
    record = read(job/'summary.json')
    if record.get('accepted'):
        accepted_fields(job/'fields.npz')
    return record


def allocate_attempt(branch_dir, minimum):
    """Skip uncommitted orphan directories left before state's inflight write."""
    indices = [int(path.name[8:]) for path in branch_dir.iterdir()
               if path.is_dir() and path.name.startswith('attempt_') and path.name[8:].isdigit()]
    index = max([minimum]+[value+1 for value in indices])
    path = branch_dir/f'attempt_{index:04d}'
    path.mkdir(parents=True, exist_ok=False)
    return path


def run(retry_failed=False):
    state, baseline = prepare()
    for name in ('DONE.json', 'FAILED.json'):
        path = OUT/name
        if path.exists():
            path.replace(OUT/f'{path.stem}_previous_{time.time_ns()}.json')
    state.update(status='running', production_launched=True)
    write(OUT/'campaign.json', state)
    mapping = Mapping()
    for name, branch in state['branches'].items():
        if branch['status'] == 'accepted':
            accepted_fields(branch['accepted_fields'], branch['accepted_fields_sha256'], branch['accepted_summary_sha256'])
            if branch['endpoint']['sha256'] != sha(branch['accepted_fields']):
                raise ValueError('Saved endpoint diagnostic hash mismatch')
            continue
        if branch['status'] == 'failed' and not retry_failed:
            continue
        if branch['status'] == 'failed':
            branch.update(status='pending', step=INITIAL_STEP, failure=None)
        branch_dir = OUT/'branches'/name
        branch_dir.mkdir(parents=True, exist_ok=True)
        while branch['fraction'] < 1.-1e-12:
            prior = accepted_fields(branch['accepted_fields'], branch['accepted_fields_sha256'], branch['accepted_summary_sha256'])
            if branch['inflight'] is None:
                fraction = min(1., branch['fraction']+branch['step'])
                job = allocate_attempt(branch_dir, len(branch['attempts']))
                predictor = prior.copy()
                # A secant predictor uses accepted states from THIS branch only.
                if branch.get('previous_fields') and branch['fraction'] > branch.get('previous_fraction', 0.):
                    previous = accepted_fields(branch['previous_fields'], branch['previous_fields_sha256'], branch['previous_summary_sha256'])
                    previous, _ = solver.continuity(prior, previous, mapping,
                                                   read(OUT/'request.json')['request']['source']['A1'])
                    predictor += (fraction-branch['fraction'])/(branch['fraction']-branch['previous_fraction'])*(prior-previous)
                savez(job/'predictor.npz', y=predictor)
                request = dict(campaign=state['request_fingerprint'], branch=name, fraction=fraction,
                               sponge=parameters(name, fraction), n=32, H=6, intervals=800, Rout=725.,
                               previous_fields=str(branch['accepted_fields']), previous_sha256=sha(branch['accepted_fields']),
                               predictor_sha256=sha(job/'predictor.npz'), budget_seconds=POINT_BUDGET)
                write(job/'request.json', dict(request, fingerprint=fingerprint(request)))
                branch['inflight'] = dict(fraction=fraction, folder=str(job), fingerprint=fingerprint(request))
                write(OUT/'campaign.json', state)
            inflight = branch['inflight']
            job, fraction = Path(inflight['folder']), inflight['fraction']
            request = read(job/'request.json')
            if request['fingerprint'] != inflight['fingerprint']:
                raise ValueError('Inflight request mismatch')
            result = run_attempt(job)
            candidate, screen = None, dict(accepted=False, reasons=['nonlinear_correction_failed'])
            if result['accepted']:
                candidate, screen = solver.continuity(prior, load_y(job/'fields.npz'), mapping,
                                                      read(OUT/'request.json')['request']['source']['A1'])
                if result['highest_to_fundamental'] > TAIL_LIMIT:
                    screen['accepted'] = False
                    screen['reasons'].append('H6_energy_tail_unresolved')
                write(job/'screen.json', screen)
            good = bool(result['accepted'] and screen['accepted'])
            branch['attempts'].append(dict(folder=str(job), fraction=fraction, accepted=good,
                                           failure=result.get('failure'), screen=screen))
            branch['inflight'] = None
            if good:
                # Phase alignment only changes the gauge. Solver fields retain the
                # audited gauge; continuity/differences are always phase invariant.
                branch.update(previous_fields=branch['accepted_fields'], previous_fraction=branch['fraction'],
                              previous_fields_sha256=branch['accepted_fields_sha256'],
                              previous_summary_sha256=branch['accepted_summary_sha256'],
                              accepted_fields=str(job/'fields.npz'), fraction=fraction,
                              accepted_fields_sha256=sha(job/'fields.npz'), accepted_summary_sha256=sha(job/'summary.json'),
                              step=min(MAX_STEP, branch['step']*1.4), status='running')
                if fraction >= 1.-1e-12:
                    endpoint = diagnostic(job/'fields.npz', parameters(name, 1.), name, baseline=baseline)
                    endpoint.update(summary=str(job/'summary.json'), screen=str(job/'screen.json'))
                    branch.update(status='accepted', endpoint=endpoint)
                    marker(branch_dir, True, endpoint)
            else:
                marker(job, False, dict(accepted=False, nonlinear=result, screen=screen))
                branch['step'] *= .5
                if branch['step'] < MIN_STEP:
                    branch.update(status='failed', failure='Minimum homotopy step exhausted; endpoint unresolved')
                    marker(branch_dir, False, branch)
            write(OUT/'campaign.json', state)
            render(state, baseline)
            if branch['status'] == 'failed':
                break
    failed = [name for name, branch in state['branches'].items() if branch['status'] != 'accepted']
    state['status'] = 'failed_or_pending' if failed else 'done'
    write(OUT/'campaign.json', state)
    write(OUT/'endpoints.json', dict(baseline=baseline, endpoints={name: branch.get('endpoint') for name, branch in state['branches'].items()}))
    marker(OUT, not failed, dict(status=state['status'], unresolved=failed, request_fingerprint=state['request_fingerprint']))
    render(state, baseline)
    return 1 if failed else 0


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument('--prepare', action='store_true')
    group.add_argument('--run', action='store_true')
    group.add_argument('--worker', type=Path)
    parser.add_argument('--retry-failed', action='store_true')
    args = parser.parse_args()
    if args.worker:
        return worker(args.worker)
    with lock(OUT/'campaign.lock'):
        if args.prepare:
            state, _ = prepare()
            print(f"D1 prepared; six branches {state['status']}; no production corrections launched")
            return 0
        try:
            return run(args.retry_failed)
        except LockBusy:
            # A surviving worker owns its attempt. Leave it and its state alone.
            raise
        except Exception as exc:
            failure = dict(status='failed_exception', error=f'{type(exc).__name__}: {exc}',
                           traceback=traceback.format_exc(), resume='Review failure, then rerun the supervisor command; committed fields and inflight checkpoints are preserved')
            marker(OUT, False, failure)
            raise


if __name__ == '__main__':
    try:
        raise SystemExit(main())
    except LockBusy as exc:
        print(str(exc), file=sys.stderr)
        raise SystemExit(75)
