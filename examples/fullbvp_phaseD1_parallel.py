#!/usr/bin/env python3
"""D1b isolated gap120/gap140 controllers and source-based width diagnostic."""
from __future__ import annotations

import os
for _key in ('OPENBLAS_NUM_THREADS', 'OMP_NUM_THREADS', 'MKL_NUM_THREADS',
             'NUMEXPR_NUM_THREADS', 'VECLIB_MAXIMUM_THREADS', 'BLIS_NUM_THREADS'):
    os.environ[_key] = '1'
os.environ['PYTHONDONTWRITEBYTECODE'] = '1'

import argparse
import sys
import traceback
from pathlib import Path

import numpy as np

import fullbvp_phaseD1_campaign as shared
import fullbvp_phaseD1_diagnostics as diagnostic_module
import fullbvp_phaseD1_solver as solver

ROOT, OLD_OUT = solver.ROOT, solver.OUT
PARALLEL = OLD_OUT/'parallel'
ENTRY = Path(__file__).resolve()
OLD_ENTRY = Path(shared.__file__).resolve()
OLD_MANIFEST = shared.manifest
RAW_DIAGNOSTIC = diagnostic_module.diagnostic
TARGETS = {'gap120': 120., 'gap140': 140.}
CURRENT = {}
INTERMEDIATE = OLD_OUT/'branches/width80/attempt_0009'
INTERMEDIATE_SHA = 'b0d2b561da281cb56b515cc779246f289deb27af0bafeff1c37f92de98f31d9d'
INTERMEDIATE_OMEGA = 52.681834110595744


def checked_old_identity():
    """Read immutable identities only; never inspect or lock the live state."""
    request = solver.read(OLD_OUT/'request.json')
    if solver.fingerprint(request['request']) != request['fingerprint']:
        raise ValueError('Original D1 request identity mismatch')
    actual = {name: solver.sha(ROOT/name) for name in request['request']['code_hashes']}
    if actual != request['request']['code_hashes']:
        raise ValueError('Original pinned numerical source changed')
    return dict(request_fingerprint=request['fingerprint'], code_hashes=actual)


def routed_diagnostic(fields, params, point_id, *, out=None, baseline=None, save=True):
    target = CURRENT['out'] if out is None else Path(out).resolve()
    if not target.is_relative_to(PARALLEL):
        raise ValueError('D1b diagnostic output must remain under parallel/')
    # Explicit out defeats the shared function's captured original-D1 default.
    return RAW_DIAGNOSTIC(fields, params, point_id, out=target, baseline=baseline, save=save)


def parallel_manifest():
    original = checked_old_identity()
    request = OLD_MANIFEST()
    request['code_hashes'].update(original['code_hashes'])
    for path in (ENTRY, ROOT/'examples/fullbvp_phaseD1_parallel_validate.py', OLD_ENTRY):
        request['code_hashes'][str(path.relative_to(ROOT))] = solver.sha(path)
    request['D1b_routing'] = dict(branch=CURRENT['branch'], output=str(CURRENT['out']),
                                worker_entry=str(ENTRY), diagnostic_output_explicit=True,
                                original_request_fingerprint=original['request_fingerprint'],
                                old_controller_action='Supervisor stops original controller after gap70 and before old gap140; no automatic skip or signals')
    return request


def atomic_text(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name+'.tmp')
    with temporary.open('w') as stream:
        stream.write(value)
        stream.flush()
        os.fsync(stream.fileno())
    os.replace(temporary, path)


def branch_render(state, baseline):
    name, out = CURRENT['branch'], CURRENT['out']
    branch = state['branches'][name]
    endpoint = branch.get('endpoint')
    text = [f'# D1b independent {name}', '',
            f"Status: {branch['status']}; homotopy fraction: {branch['fraction']:.9g}.",
            f'Independent source: {solver.SOURCE}/fields.npz; target gap={TARGETS[name]:g}, strength=28, width=40.',
            f"Baseline omega={baseline['omega']:.14g}; accepted endpoint available={endpoint is not None}.",
            f'All state, locks, attempts, diagnostics and markers stay under {out}.',
            f'Worker entry: {ENTRY}; one numerical thread; 4800-second hard correction budget.',
            'A missing sponge-censored front is not a physical disappearance. Source pinch coverage and growth-window overlap remain visible in diagnostics.',
            'Refresh the durable combined D1b report with the --report command after completion.']
    if endpoint:
        text += ['', 'Endpoint values and baseline deltas: diagnostics/'+name+'.json.']
    atomic_text(out/'REPORT.md', '\n'.join(text)+'\n')


def bind(branch, out=None):
    if branch not in TARGETS:
        raise ValueError('Only independent gap120/gap140 entries are authorized')
    target = (PARALLEL/branch if out is None else Path(out)).resolve()
    if not target.is_relative_to(PARALLEL) or target == PARALLEL:
        raise ValueError('D1b output escaped isolated parallel directory')
    CURRENT.clear()
    CURRENT.update(branch=branch, out=target)
    shared.OUT = target
    shared.BRANCHES = {branch: ('gap', TARGETS[branch])}
    # Shared run_attempt resolves __file__ when constructing its worker command.
    # Changing this isolated process global never edits the original source file.
    shared.__file__ = str(ENTRY)
    shared.diagnostic = routed_diagnostic
    shared.manifest = parallel_manifest
    shared.render = branch_render
    return target


def infer_worker_branch(job):
    branch = solver.read(Path(job)/'request.json')['branch']
    if branch not in TARGETS:
        raise ValueError('Worker request is not a D1b branch')
    if not Path(job).resolve().is_relative_to(PARALLEL/branch):
        raise ValueError('Worker path escaped its independent branch directory')
    return branch


def intermediate_diagnostic():
    """An accepted intermediate checkpoint is never relabelled as width80."""
    identity = checked_old_identity()
    if solver.sha(INTERMEDIATE/'fields.npz') != INTERMEDIATE_SHA:
        raise ValueError('Pinned intermediate field hash mismatch')
    request = solver.read(INTERMEDIATE/'request.json')
    request_id = request.pop('fingerprint')
    summary, screen = solver.read(INTERMEDIATE/'summary.json'), solver.read(INTERMEDIATE/'screen.json')
    if (solver.fingerprint(request) != request_id or request['campaign'] != identity['request_fingerprint'] or
            summary.get('request_fingerprint') != request_id or not summary.get('accepted') or
            not screen.get('accepted') or screen.get('reasons') or summary.get('fields_sha256') != INTERMEDIATE_SHA):
        raise ValueError('Intermediate request/acceptance/screen identity mismatch')
    checks = [('relative_field_inf', solver.STRICT['relative_field_inf']),
              ('residual_over_fundamental_feedback', solver.STRICT['residual_over_fundamental_feedback']),
              ('highest_to_fundamental', shared.TAIL_LIMIT)]
    if any(not np.isfinite(summary[key]) or summary[key] >= limit for key, limit in checks):
        raise ValueError('Intermediate strict residual or H6 tail check failed')
    independent = summary['independent']
    if (not np.isfinite([independent['relative_field_inf'], independent['template_difference_relative']]).all() or
            independent['relative_field_inf'] >= solver.STRICT['independent_relative_field_inf'] or
            independent['template_difference_relative'] > solver.STRICT['template_difference_relative']):
        raise ValueError('Intermediate independent audit failed')
    with np.load(INTERMEDIATE/'fields.npz') as saved:
        metadata = {key: saved[key].item() for key in ('n', 'H', 'intervals', 'Rout', 'omega', 'a', 'strength', 'width', 'gap', 'accepted', 'request_fingerprint')}
        if (metadata['n'] != 32 or metadata['H'] != 6 or metadata['intervals'] != 800 or
                metadata['Rout'] != 725. or metadata['omega'] != INTERMEDIATE_OMEGA or metadata['a'] != 625. or
                metadata['strength'] != 28. or metadata['gap'] != 100. or not metadata['accepted'] or
                metadata['request_fingerprint'] != request_id or metadata['width'] != request['sponge']['width'] or
                not np.isfinite(saved['y']).all() or not np.isfinite(saved['field']).all()):
            raise ValueError('Intermediate saved metadata mismatch')
        params = {key: metadata[key] for key in ('strength', 'width', 'gap')}
    out = PARALLEL/'intermediate'
    baseline = RAW_DIAGNOSTIC(solver.SOURCE/'fields.npz', solver.BASE_PARAMS, 'baseline', out=out)
    result = RAW_DIAGNOSTIC(INTERMEDIATE/'fields.npz', params, 'width58_35775_INTERMEDIATE', out=out, baseline=baseline)
    result.update(point_role='INTERMEDIATE_POINT_NOT_WIDTH80_ENDPOINT', nominal_task_label='中间点 width=58.41',
                  actual_checkpoint_width=metadata['width'], actual_width_explanation='Nominal58.41 corresponds to rejected attempt_0014 at width58.410271875 (Newton line search); accepted diagnostic source is attempt_0009 at width58.35775 (~58.36).',
                  source_summary=str(INTERMEDIATE/'summary.json'), source_screen=str(INTERMEDIATE/'screen.json'),
                  source_hashes={name: solver.sha(INTERMEDIATE/name) for name in ('fields.npz', 'summary.json', 'request.json', 'screen.json')},
                  source_metadata=metadata, accepted_width80_endpoint=False)
    solver.write(out/'diagnostics/width58_35775_INTERMEDIATE.json', result)
    return result, baseline


def report():
    result, baseline = intermediate_diagnostic()
    fmt = lambda value: 'missing/censored' if value is None else f'{value:.12g}'
    lines = ['# D1b: isolated gap120/gap140 and width intermediate', '',
             '**INTERMEDIATE POINT — not an accepted width80 endpoint.**',
             'Requested label: 中间点 width=58.41. The nominal58.41 corresponds to the rejected attempt_0014 at width58.410271875 (Newton line search). This intermediate diagnostic uses the last accepted source attempt_0009 at actual width58.35775 (~58.36), strength=28, gap=100; it is not an accepted58.41 or width80 endpoint.',
             f'Pinned source: {INTERMEDIATE}/fields.npz; SHA256 {INTERMEDIATE_SHA}. Solver acceptance and branch screen are both verified; no width80 retry is requested.', '',
             '| Quantity | Baseline | Actual width58.35775 intermediate | Intermediate minus baseline |',
             '|---|---:|---:|---:|']
    labels = [('omega', 'omega'), ('front_fig8', 'Fig.8 front'), ('front_2beta_absolute', '2beta absolute front'),
              ('front_2beta_5percent', '2beta 5% front'), ('g_q_570_600', 'g_q end-to-end R570–600'),
              ('g_q_least_squares_570_600', 'g_q LS R570–600'), ('sponge_fraction_q', 'coherent q sponge fraction'),
              ('sponge_fraction_cylindrical_R3', 'coherent R³ sponge fraction')]
    for key, label in labels:
        lines.append('| '+' | '.join((label, fmt(baseline[key]), fmt(result[key]), fmt(result['endpoint_minus_baseline'][key])))+' |')
    lines += ['', f"Actual outer sponge starts at a={result['a']:g}; Fig.8 search={result['front_search_interval']}; sampled search={result['front_search_sampled_interval']}; pinch coverage={result['pinch_reference_coverage']}.",
              'These are the existing direct-harmonic, q0-excluded coherent and end-to-end/LS diagnostic definitions. Sponge fractions measure energy in geometric support R>=a, not dissipation; gap changes also change the integration region.', '',
              '| Independent branch | status | fraction reached | endpoint output |', '|---|---|---:|---|']
    for branch in TARGETS:
        path = PARALLEL/branch/'campaign.json'
        if path.exists():
            state = solver.read(path)
            point = state['branches'][branch]
            status, fraction = point['status'], point['fraction']
        else:
            status, fraction = 'not_prepared', 0.
        lines.append(f'| {branch} | {status} | {fraction:.9g} | {PARALLEL/branch}/endpoints.json |')
    lines += ['', 'Both independent branches start at the same pinned accepted 725/H6 baseline with strength28, width40, gap100. Each has separate state, campaign/worker locks, checkpoints, request/code/source identities, diagnostics and DONE/FAILED. No production was started or stopped during D1b preparation.',
              'Gap120 starts the sponge at a=605; gap140 starts at a=585. Gap140 overlaps the R570–600 growth window and truncates the upstream-sponge Fig.8 search before the baseline front. Missing crossings are censored; auxiliary covered-domain crossings remain labelled separately.', '',
              '## Supervisor actions', '',
              '**Chosen old-controller action:** the supervisor stops the original D1 controller after gap70 finishes and before it dispatches its queued gap140. The original controller does not automatically skip gap140. If dispatch races and an old gap140 worker already exists, the supervisor handles that old worker too to avoid duplicate work. This entry never signals or edits that controller, its workers, state or locks.',
              f'Launch/resume both new independent branches concurrently: `bash {PARALLEL}/commands.sh`.',
              'Do not restart the original D1/commands.sh: it passes --retry-failed and would retry width80. All subsequent launch/resume actions use parallel/commands.sh only.',
              'The launcher creates two detached processes, one numerical thread each, independent logs/PIDs and terminal markers. Added numerical CPU total=2; together with one existing worker the numerical total=3<=16. No GPU. Each correction retains the original 4800-second hard budget, strict unforced residuals, symmetric continuity, nonzero-amplitude and H6-tail checks.',
              'Completed branches are validated and reused on restart. Failed branches resume only through the same-request checkpoints and accepted branch history; missing endpoints stay missing.',
              f'Refresh durable diagnostics/report and append the latest bounded D1b snapshot after the original controller stops: `{ROOT}/.venv/bin/python {ENTRY} --report --merge-report`.',
              'The initial main REPORT addition is append-only. The active old controller can overwrite that appendix during its own progress updates. This parallel/REPORT.md is durable; the explicit refresh command appends the latest snapshot without replacing the original report body. There is no report watcher or renderer modification.',
              'The latest bounded D1b appendix supersedes earlier D1b snapshots. Numerical code and all pinned original files remain unchanged; validation.json records routing/isolation and source-hash checks.']
    text = '\n'.join(lines)+'\n'
    atomic_text(PARALLEL/'REPORT.md', text)
    return text


def merge_report(text):
    # One append-only write cannot replace the controller's existing report body.
    # Its renderer may subsequently remove this appendix; supervisor refreshes it.
    digest = solver.fingerprint(text)
    begin = '<!-- BEGIN D1b '+digest+' -->'
    target = OLD_OUT/'REPORT.md'
    if target.exists() and begin in target.read_text():
        return False
    appendix = '\n\n'+begin+'\n'+text+'<!-- END D1b '+digest+' -->\n'
    fd = os.open(target, os.O_WRONLY|os.O_APPEND|os.O_CREAT, 0o644)
    try:
        data = appendix.encode()
        if os.write(fd, data) != len(data):
            raise OSError('Short append writing D1b report')
        os.fsync(fd)
    finally:
        os.close(fd)
    return True


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--branch', choices=tuple(TARGETS))
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument('--prepare', action='store_true')
    group.add_argument('--run', action='store_true')
    group.add_argument('--worker', type=Path)
    group.add_argument('--report', action='store_true')
    parser.add_argument('--retry-failed', action='store_true')
    parser.add_argument('--merge-report', action='store_true')
    args = parser.parse_args(argv)
    if args.report:
        text = report()
        appended = merge_report(text) if args.merge_report else False
        print('D1b durable report refreshed; append-only main report update='+str(appended))
        return 0
    branch = infer_worker_branch(args.worker) if args.worker else args.branch
    if branch is None or args.branch is not None and args.branch != branch:
        parser.error('Use --branch gap120 or gap140; worker branch must match its request')
    out = bind(branch)
    if args.worker:
        return shared.worker(args.worker)
    with shared.lock(out/'campaign.lock'):
        if args.prepare:
            shared.prepare()
            print(f'D1b {branch} prepared; no production launched; output={out}')
            return 0
        try:
            return shared.run(args.retry_failed)
        except shared.LockBusy:
            raise
        except Exception as exc:
            shared.marker(out, False, dict(status='failed_exception', error=f'{type(exc).__name__}: {exc}', traceback=traceback.format_exc()))
            raise


if __name__ == '__main__':
    try:
        raise SystemExit(main())
    except shared.LockBusy as exc:
        print(str(exc), file=sys.stderr)
        raise SystemExit(75)
