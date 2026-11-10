#!/usr/bin/env python3
"""D3: independently refine N and radial intervals at fixed physical Rout.

Preparation never launches a corrector. --run is the resumable, CPU-only full
campaign; every correction has its own 4800 s subprocess deadline/checkpoints.
"""
from __future__ import annotations

import os
for _name in ('OPENBLAS_NUM_THREADS', 'OMP_NUM_THREADS', 'MKL_NUM_THREADS',
              'NUMEXPR_NUM_THREADS', 'VECLIB_MAXIMUM_THREADS', 'BLIS_NUM_THREADS'):
    os.environ[_name] = '1'
os.environ['OMP_THREAD_LIMIT'] = '1'
os.environ['CUDA_VISIBLE_DEVICES'] = ''
os.environ.setdefault('MPLCONFIGDIR', '/tmp/fullbvp_phaseD3_matplotlib')

import argparse
import contextlib
import fcntl
import hashlib
import io
import json
import signal
import subprocess
import sys
import threading
import time
import traceback
from pathlib import Path

sys.dont_write_bytecode = True
ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT/'src'), str(ROOT/'examples')]
DEFAULT_OUT = ROOT/'results/fullbvp_global_20261030/phaseD/D3'
SOURCE = ROOT/'results/fullbvp_global_20261030/phaseC/attempts/harmonic_audit_be43e3a56f551a7f_try0'
BUDGET = 4800
LOG_BYTES = 128*1024
MIN_STEP, MAX_STEP = .078125, 1.25
METRICS = ('omega', 'front_fig8', 'front_2beta_absolute', 'front_2beta_5percent',
           'g_q_570_600', 'g_q_fit_570_600', 'sponge_fraction_q',
           'sponge_fraction_cylindrical_R3', 'E1')
MATRIX = tuple(dict(key=f'R{R}_{kind}', Rout=float(R), n=n, H=6, intervals=M,
                    role=kind) for R in (725, 730) for kind, n, M in
               (('baseline', 32, 800), ('N40', 40, 800),
                ('radial', 32, 1300 if R == 725 else 1320)))


def read(path):
    return json.loads(Path(path).read_text())


def write(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name+'.tmp')
    with tmp.open('w') as stream:
        json.dump(value, stream, indent=2, ensure_ascii=False, allow_nan=False)
        stream.write('\n')
        stream.flush()
        os.fsync(stream.fileno())
    os.replace(tmp, path)


def sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(1024*1024), b''):
            h.update(block)
    return h.hexdigest()


def fingerprint(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, allow_nan=False).encode()).hexdigest()


def checked_out(path):
    path = Path(path).resolve()
    if not path.is_relative_to(DEFAULT_OUT.resolve()):
        raise ValueError('D3 outputs must stay within its own task directory')
    path.mkdir(parents=True, exist_ok=True)
    return path


class Exclusive:
    def __init__(self, out):
        self.path = checked_out(out)/'D3.lock'

    def __enter__(self):
        self.stream = self.path.open('a')
        try:
            fcntl.flock(self.stream, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError:
            self.stream.close()
            raise RuntimeError('Another D3 campaign holds D3.lock') from None
        return self

    def __exit__(self, *_):
        self.stream.close()  # inherited worker descriptor keeps ownership on parent SIGKILL


class TailLog(io.TextIOBase):
    """Keep a bounded tail; neither job output nor tracebacks grow logs forever."""
    def __init__(self, path):
        self.path = Path(path)
        self.tail = self.path.read_bytes()[-LOG_BYTES:] if self.path.exists() else b''

    def append(self, data):
        self.tail = (self.tail+data)[-LOG_BYTES:]
        self.path.write_bytes(self.tail)

    def write(self, text):
        self.append(text.encode('utf-8', errors='replace'))
        return len(text)

    def flush(self):
        pass


def numerical_gate(row):
    """Recheck all inherited thresholds, never trust an accepted Boolean alone."""
    import math
    from fullbvp_phaseC_solver import STRICT
    independent = row.get('independent', {})
    values = [row.get('relative_field_inf'), row.get('residual_over_fundamental_feedback'),
              row.get('phase'), row.get('branch_constraint'),
              independent.get('relative_field_inf'), independent.get('template_difference_relative'),
              row.get('highest_to_fundamental')]
    if any(v is None or not math.isfinite(v) for v in values):
        return False
    return bool(row.get('accepted') and values[0] < STRICT['relative_field_inf']
                and values[1] < STRICT['residual_over_fundamental_feedback']
                and max(abs(values[2]), abs(values[3])) < STRICT['constraint']
                and values[4] < STRICT['independent_relative_field_inf']
                and values[5] <= STRICT['template_difference_relative']
                and values[6] <= .01 and independent.get('forcing') is False
                and independent.get('physical_grid_reassembled') is True)


def actual_grid(fields, expected):
    import numpy as np
    with np.load(fields) as saved:
        r, xi = saved['radii'], saved['xi']
        expected_shape = (expected['intervals']+1, 2*expected['H']+1, 4*len(xi))
        if (not bool(saved['accepted']) or int(saved['H']) != expected['H']
                or int(saved['n']) != expected['n'] or len(r) != expected['intervals']+1
                or saved['field'].shape != expected_shape or len(xi) != expected['n']+1
                or ('intervals' in saved and int(saved['intervals']) != expected['intervals'])
                or abs(float(saved['Rout'])-expected['Rout']) > 1e-8
                or ('omega' in expected and abs(float(saved['omega'])-expected['omega']) > 1e-10)
                or not np.allclose(r, np.linspace(400., expected['Rout'], len(r)), rtol=0., atol=2e-10)):
            raise ValueError('Accepted checkpoint metadata/physical grid mismatch')
        if not np.isfinite(saved['field']).all():
            raise ValueError('Nonfinite accepted field')
        return dict(n=int(saved['n']), H=int(saved['H']), intervals=len(r)-1,
                    radial_nodes=len(r), wall_normal_nodes=len(xi), Rin=float(r[0]),
                    Rout=float(r[-1]), dR_min=float(np.diff(r).min()),
                    dR_max=float(np.diff(r).max()),
                    xi_sha256=hashlib.sha256(xi.tobytes()).hexdigest(),
                    radii_sha256=hashlib.sha256(r.tobytes()).hexdigest(),
                    wall_normal_z_max='infinity (mapped Chebyshev endpoint)')


def source_identity():
    from fullbvp_phaseC_solver import SCREEN
    row, gate = read(SOURCE/'summary.json'), read(SOURCE/'screen.json')
    if not numerical_gate(row) or not gate.get('accepted') or gate.get('reasons'):
        raise ValueError('725/H6 audit is not strictly accepted on the upper branch')
    if any(row[k] != v for k, v in (('n', 32), ('H', 6), ('intervals', 800), ('Rout', 725.))):
        raise ValueError('Unexpected source identity')
    return dict(id=SOURCE.name, fields=str(SOURCE/'fields.npz'),
                fields_sha256=sha(SOURCE/'fields.npz'), summary=str(SOURCE/'summary.json'),
                summary_sha256=sha(SOURCE/'summary.json'), screen=str(SOURCE/'screen.json'),
                screen_sha256=sha(SOURCE/'screen.json'),
                source_A=gate['source_amplitude_floor']/SCREEN['source_amplitude_floor'],
                actual_grid=actual_grid(SOURCE/'fields.npz', MATRIX[0]),
                status='strictly_accepted_phaseC_audit_not_committed_to_phaseC_targets')


def configuration():
    from fullbvp_phaseC_solver import STRICT, SCREEN
    paths = [Path(__file__), ROOT/'examples/fullbvp_phaseD3_checks.py']
    paths += [ROOT/'examples'/name for name in (
        'fullbvp_phaseB_solver.py', 'fullbvp_phaseB_upper_refinement.py',
        'fullbvp_phaseB_report.py', 'fullbvp_phaseC_solver.py',
        'fullbvp_phaseC_postprocess.py', 'appelquist2016_branch_continuation.py',
        'appelquist2015_hopf_common.py', 'appelquist2015_hopf_block_corrector.py')]
    # Include the complete local numerical library, not unrelated campaign files.
    paths += sorted((ROOT/'src/llsa').rglob('*.py'))
    paths += [ROOT/f'results/fullbvp_global_20261030/phaseA/A3/curve_N{n}_global.json' for n in (32, 40)]
    return dict(schema=1, matrix=list(MATRIX), comparison='fixed physical Rout; E1 free',
                budget_per_point_seconds=BUDGET, CPU_threads=1, GPU=False,
                minimum_Rout_step=MIN_STEP, maximum_Rout_step=MAX_STEP,
                equations=dict(Ro=-1., beta=68., Rin=400., inlet='0.8 S((420-R)/10)',
                               exterior='28 S((R-(Rout-100))/40)', inherited_BCs=True),
                strict=STRICT, screening=SCREEN, harmonic_trust_ratio=.01,
                code_and_reference_hashes={str(p.relative_to(ROOT)): sha(p) for p in paths})


def point_from_record(row, folder, key, role, source, out, imported=False):
    from fullbvp_phaseC_postprocess import diagnostic
    fields = Path(folder)/'fields.npz'
    point = {k: row[k] for k in ('Rout', 'omega', 'A1', 'energies', 'highest_to_fundamental',
             'relative_field_inf', 'residual_over_fundamental_feedback', 'n', 'H', 'intervals',
             'phase', 'branch_constraint')}
    point.update(id=key, role=role, accepted=True, resolved=True, fields=str(fields),
                 fields_sha256=sha(fields), summary=str(Path(folder)/'summary.json'),
                 summary_sha256=sha(Path(folder)/'summary.json'), source_id=source['id'],
                 E1=float(row['energies'][1]), a=row['Rout']-100,
                 dr=(row['Rout']-400)/row['intervals'], imported_source=imported,
                 actual_grid=actual_grid(fields, row), independent=row['independent'])
    return diagnostic(point, out)


def completion_gate(comparison, identity):
    rows = comparison.get('rows', [])
    return bool(len(rows) == 6 and {r.get('key') for r in rows} == {r['key'] for r in MATRIX}
                and comparison.get('fingerprint') == identity
                and all(r.get('status') == 'accepted' and r.get('metrics') is not None
                        and r.get('delta_from_same_Rout_H6_baseline') is not None
                        and r.get('actual_grid') is not None and r.get('fields_sha256')
                        for r in rows))


def worker(request_path):
    """Single numerical point. Only this child constructs production-sized maps."""
    def deadline(_signum, _frame):
        raise SystemExit('4800 s worker deadline; last atomic checkpoint preserved')
    signal.signal(signal.SIGALRM, deadline)
    signal.setitimer(signal.ITIMER_REAL, BUDGET)
    import numpy as np
    from fullbvp_phaseC_solver import (Mapping, CoherentMetric, mapped_transfer, phase_align,
                                      qref_for, correct, screen)
    request_path = Path(request_path)
    request = read(request_path)
    for item in request['history']:
        if sha(item['fields']) != item['fields_sha256']:
            raise ValueError('Worker source/history hash changed')
    grid = request['grid']
    mapping = Mapping(grid['n'], 6, grid['intervals'])
    previous = mapped_transfer(request['history'][-1]['fields'], mapping)
    predicted, tangent = previous.copy(), None
    if request['continuation']:
        if len(request['history']) == 2:
            older = mapped_transfer(request['history'][0]['fields'], mapping)
            older, _ = phase_align(older, previous, mapping)
            delta = previous-older
            if delta[-1] > 1e-8:
                tangent = CoherentMetric(mapping).tangent(older, previous)
                predicted += (grid['Rout']-previous[-1])/delta[-1]*delta
        predicted[-1] = grid['Rout']  # stretch xi only after interpolation at the old physical Rout
    elif abs(previous[-1]-grid['Rout']) > 1e-8:
        raise ValueError('Fixed-Rout refinement source mismatch')
    started = time.monotonic()
    y, row = correct(mapping, qref_for(predicted, mapping), predicted, 'Rout', grid['Rout'],
                     request_path.parent/'newton', BUDGET, CoherentMetric(mapping), tangent,
                     resume=request.get('resume'))
    gate = dict(accepted=False, reasons=['numerical_strict_gate'])
    if numerical_gate(row):
        _, gate = screen(previous, y, mapping, request['source_A'], tangent=tangent,
                         refinement=not request['continuation'])
    if abs(row['Rout']-grid['Rout']) >= 1e-8:
        gate = dict(accepted=False, reasons=['fixed_Rout_target_mismatch'])
    write(request_path.parent/'screen.json', gate)
    write(request_path.parent/'outcome.json', dict(accepted=numerical_gate(row) and gate['accepted'],
          numerical_accepted=numerical_gate(row), branch=gate, worker_seconds=time.monotonic()-started,
          summary=str(request_path.parent/'newton/summary.json'), failure=row.get('failure')))


class Campaign:
    def __init__(self, out, lock):
        self.out, self.lock = checked_out(out), lock
        self.source, self.config = source_identity(), configuration()
        self.identity = fingerprint(dict(source=self.source, config=self.config))
        statefile = self.out/'campaign.json'
        if statefile.exists():
            self.state = read(statefile)
            if self.state['fingerprint'] != self.identity:
                raise ValueError('Resume refused: source, code, reference or configuration changed')
            self.verify_commits()
        else:
            self.state = dict(schema=1, fingerprint=self.identity, source=self.source,
                              config=self.config, status='prepared', points={}, continuation=[],
                              matrix={r['key']:dict(r, status='pending') for r in MATRIX},
                              attempts=[], events=[])
            self.save()

    def save(self):
        write(self.out/'campaign.json', self.state)

    def verify_commits(self):
        for key, point in self.state['points'].items():
            if sha(point['fields']) != point['fields_sha256'] or sha(point['summary']) != point['summary_sha256']:
                raise ValueError(f'Committed field/summary changed: {key}')
            if not numerical_gate(read(point['summary'])):
                raise ValueError(f'Committed strict residual gate invalid: {key}')
            gate_path = SOURCE/'screen.json' if point['imported_source'] else Path(point['summary']).parent.parent/'screen.json'
            if sha(gate_path) != point['screen_sha256'] or not read(gate_path).get('accepted'):
                raise ValueError(f'Committed branch gate changed: {key}')
            actual_grid(point['fields'], point)
        for key, row in self.state['matrix'].items():
            if row['status'] == 'accepted' and key not in self.state['points']:
                raise ValueError('Accepted matrix row has no verified commit')

    def commit(self, point, gate_path, continuation=False):
        point['screen_sha256'] = sha(gate_path)
        self.state['points'][point['id']] = point
        if point['id'] in self.state['matrix']:
            self.state['matrix'][point['id']].update(status='accepted', point=point['id'])
        if continuation and point['id'] not in self.state['continuation']:
            self.state['continuation'].append(point['id'])
        self.save()

    def attempt(self, grid, history, continuation=False):
        immutable_history = [{k:p[k] for k in ('fields', 'fields_sha256')} for p in history]
        identity_request = dict(campaign=self.identity, grid=grid, history=immutable_history,
                                continuation=continuation, source_A=self.source['source_A'])
        key = fingerprint(identity_request)
        root = self.out/'attempts'
        root.mkdir(exist_ok=True)
        matches = sorted(root.glob(key[:20]+'_try*'))
        resume = None
        for folder in matches:
            request_path = folder/'request.json'
            if not request_path.exists():
                continue  # preserve an interrupted reservation, never infer its identity
            request = read(request_path)
            if {k:v for k,v in request.items() if k != 'resume'} != identity_request:
                raise ValueError('Attempt request hash mismatch')
            outcome = folder/'outcome.json'
            if outcome.exists() and read(outcome).get('accepted'):
                return folder, read(outcome)
            summary_path = folder/'newton/summary.json'
            if summary_path.exists() and numerical_gate(read(summary_path)):
                # A crash between atomic summary and outcome is recoverable without another Newton solve.
                # Re-evaluate screening by starting from the saved accepted checkpoint.
                resume = folder/'newton/fields.npz'
            elif (folder/'newton/checkpoint.npz').exists():
                resume = folder/'newton/checkpoint.npz'
        suffix = max([int(p.name.rsplit('try', 1)[1]) for p in matches] or [-1])+1
        folder = root/f'{key[:20]}_try{suffix:03d}'
        folder.mkdir()
        request = dict(identity_request, resume=None if resume is None else str(resume))
        write(folder/'request.json', request)
        logfile = TailLog(folder/'worker.log')
        process = subprocess.Popen([sys.executable, str(Path(__file__).resolve()), '--worker', str(folder/'request.json')],
                                   cwd=ROOT, stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                                   stdin=subprocess.DEVNULL, start_new_session=True,
                                   pass_fds=(self.lock.stream.fileno(),))
        def drain():
            for block in iter(lambda: process.stdout.read(8192), b''):
                logfile.append(block)
        reader = threading.Thread(target=drain, daemon=True)
        reader.start()
        deadline = False
        try:
            process.wait(timeout=BUDGET)
        except subprocess.TimeoutExpired:
            deadline = True
            os.killpg(process.pid, signal.SIGTERM)
            try:
                process.wait(timeout=5)
            except subprocess.TimeoutExpired:
                os.killpg(process.pid, signal.SIGKILL)
                process.wait()
        except BaseException:
            os.killpg(process.pid, signal.SIGKILL)
            process.wait()
            raise
        finally:
            reader.join(timeout=5)
            process.stdout.close()
        outcome_path = folder/'outcome.json'
        if deadline or process.returncode != 0 or not outcome_path.exists():
            outcome = dict(accepted=False, failure='4800 s worker deadline' if deadline else
                           f'worker exit {process.returncode}; missing or incomplete outcome',
                           returncode=process.returncode, checkpoint_preserved=True)
            write(folder/'supervisor_failure.json', outcome)
        else:
            outcome = read(outcome_path)
        self.state['attempts'].append(dict(folder=str(folder), grid=grid,
             continuation=continuation, accepted=outcome['accepted'], failure=outcome.get('failure')))
        self.save()
        return folder, outcome

    def corrected(self, grid, history, continuation=False):
        folder, outcome = self.attempt(grid, history, continuation)
        if not outcome['accepted']:
            return None
        record_path = folder/'newton/summary.json'
        row = read(record_path)
        gate_path = folder/'screen.json'
        if not numerical_gate(row) or not read(gate_path)['accepted']:
            raise ValueError('Reused outcome failed independent acceptance recheck')
        point = point_from_record(row, record_path.parent, grid['key'], grid['role'],
                                  self.source, self.out)
        self.commit(point, gate_path, continuation)
        return point

    def refinements(self, R):
        baseline = self.state['points'].get(f'R{R}_baseline')
        if baseline is None:
            return
        for grid in [r for r in MATRIX if r['Rout'] == R and r['role'] != 'baseline']:
            if self.state['matrix'][grid['key']]['status'] == 'accepted':
                continue
            point = self.corrected(grid, [baseline])
            if point is None:
                self.state['matrix'][grid['key']].update(status='pending', last_attempt='failed_or_branch_rejected')
                self.save()

    def continuation(self):
        step = MAX_STEP
        while True:
            history = [self.state['points'][k] for k in self.state['continuation'][-2:]]
            current = history[-1]
            if current['Rout'] >= 730.-1e-8:
                return True
            trial = min(730., current['Rout']+step)
            trial_step = trial-current['Rout']
            key = 'R730_baseline' if abs(trial-730.) < 1e-8 else f'continuation_R{trial:.9f}'
            grid = dict(key=key, Rout=trial, n=32, H=6, intervals=800, role='baseline' if key=='R730_baseline' else 'continuation')
            point = self.corrected(grid, history, continuation=True)
            self.state['events'].append(dict(Rout=trial, step=trial_step, accepted=point is not None))
            self.save()
            if point is not None:
                step = min(MAX_STEP, 1.5*trial_step)
            else:
                step = .5*trial_step
                if step < MIN_STEP:
                    self.state['matrix']['R730_baseline'].update(status='pending',
                        last_attempt='adaptive_step_floor', last_accepted_Rout=current['Rout'])
                    self.save()
                    return False

    def run(self):
        self.state.update(status='running', failure=None)
        self.save()
        if 'R725_baseline' not in self.state['points']:
            point = point_from_record(read(SOURCE/'summary.json'), SOURCE, 'R725_baseline',
                                      'baseline', self.source, self.out, imported=True)
            self.commit(point, SOURCE/'screen.json', continuation=True)
        self.refinements(725)  # retain independent 725 results even if 730 cannot be reached
        if self.continuation():
            self.refinements(730)
        complete = all(row['status'] == 'accepted' for row in self.state['matrix'].values())
        self.state['status'] = 'complete' if complete else 'incomplete'
        self.save()
        self.finalize()
        if not complete:
            raise RuntimeError('Six-row D3 matrix incomplete; pending states and checkpoints preserved')

    def finalize(self):
        points = self.state['points']
        comparisons = []
        for grid in MATRIX:
            row = dict(grid, status=self.state['matrix'][grid['key']]['status'])
            point = points.get(grid['key'])
            base = points.get(f'R{int(grid["Rout"])}_baseline')
            row['metrics'] = None if point is None else {k:point.get(k) for k in METRICS}
            row['delta_from_same_Rout_H6_baseline'] = None if point is None or base is None else {
                k:None if point.get(k) is None or base.get(k) is None else point[k]-base[k] for k in METRICS}
            row['actual_grid'] = None if point is None else point['actual_grid']
            row['fields_sha256'] = None if point is None else point['fields_sha256']
            comparisons.append(row)
        write(self.out/'comparisons.json', dict(fingerprint=self.identity, comparison='fixed physical Rout; E1 is free',
              rows=comparisons, inherited_front_reference='N-dependent fixed Phase A pinch curves; N40 delta includes reference discretization',
              convergence_established=False))
        report = ['# D3 fixed-Rout H6 resolution checks', '',
            f'Status: **{self.state["status"]}**. Six-row matrix required for DONE.json.', '',
            'The comparison fixes physical Rout (725 or 730), not E1. N40/M800 and N32/M1300 or M1320 change one resolution independently. All roots use H6.', '',
            '| Rout | N | intervals | status | omega | Fig8 front | 2-beta front | g_q end | g_q LS | sponge q | sponge R³ | E1 |',
            '|---:|---:|---:|---|---:|---:|---:|---:|---:|---:|---:|---:|']
        columns = ('omega', 'front_fig8', 'front_2beta_absolute', 'g_q_570_600', 'g_q_fit_570_600', 'sponge_fraction_q', 'sponge_fraction_cylindrical_R3', 'E1')
        fmt = lambda v:'pending/unavailable' if v is None else f'{v:.10g}'
        for row in comparisons:
            metrics = row['metrics'] or {}
            report.append(f'| {row["Rout"]:g} | {row["n"]} | {row["intervals"]} | {row["status"]} | '+' | '.join(fmt(metrics.get(k)) for k in columns)+' |')
        report += ['', 'Same-Rout H6 deltas (refined minus baseline; full precision in comparisons.json):', '',
                   '| check | Δomega | ΔFig8 | Δ2-beta | Δg_q end | Δg_q LS | Δsponge q | Δsponge R³ | ΔE1 |',
                   '|---|---:|---:|---:|---:|---:|---:|---:|---:|']
        for row in comparisons:
            if row['role'] != 'baseline':
                delta = row['delta_from_same_Rout_H6_baseline'] or {}
                report.append('| '+row['key']+' | '+' | '.join(fmt(delta.get(k)) for k in columns)+' |')
        report += ['', '725/H6 is imported from the accepted Phase C harmonic audit, with its original independent residual and branch evidence. 730/H6 is established only by D3 continuation from it; no 730/H5 field is relabelled.', '',
            'Ro=-1, beta=68, Rin=400; inlet sponge 0.8 S((420-R)/10); exterior sponge 28 S((R-(Rout-100))/40). Original wall/far-field, inlet and homogeneous outlet derivative conditions and strict unforced equations are inherited.', '',
            'Diagnostics call fullbvp_phaseC_postprocess.diagnostic directly: Fig8 is the most upstream downward crossing of total-RMS logarithmic growth with the fixed Phase A local pinch curve, at z=1.3 and upstream of the sponge. The separate 2-beta front is the first upward complex-coefficient crossing at 1e-6 (5% front is also saved). Both end-to-end and least-squares fundamental g_q on 570–600 and both coherent-velocity sponge fractions are retained; mean energy is excluded.', '',
            'The inherited Fig8 diagnostic uses curve_N32_global.json for N32 and curve_N40_global.json for N40. Thus its N-refinement delta includes the local-pinch reference discretization change. Neither curve is recomputed at each nonlinear omega; its coverage is recorded in each diagnostic. These finite checks alone do not establish asymptotic convergence or temporal stability.', '',
            'Each admitted point must satisfy relative original field residual <1e-10, residual/fundamental feedback <1e-5, phase/branch constraints <1e-10, independently reassembled physical-grid residual <1e-10, template difference <=1e-10, inherited coherent branch screening, and E6/E1<=1e-2. Actual grids, source/reference/code hashes and per-point evidence are durable.', '',
            'Run/resume: `bash '+str(self.out/'commands.sh')+'`. This launches sequential 4800 s workers, with one BLAS/OpenMP thread and no GPU, a D3 flock, immutable attempt folders, atomic Newton checkpoints and logs bounded to 128 KiB per file. A hard-deadline worker can restart from its last atomic checkpoint on the next invocation. Failed numerical states remain pending. DONE requires all six accepted rows and a matching comparisons.json; otherwise FAILED is written.', '',
            'Local preparation/verification evidence is in validation/. The full campaign was not started during implementation. After the supervisor finishes, review DONE.json or FAILED.json and comparisons.json; resume the same command if checkpoints can be advanced. Code/source changes intentionally refuse resumption under a different fingerprint.', '']
        (self.out/'REPORT.md').write_text('\n'.join(report))


def execute(out):
    with Exclusive(out) as lock:
        out = checked_out(out)
        for name in ('DONE.json', 'FAILED.json'):
            if (out/name).exists():
                write(out/'marker_history'/f'{time.time_ns()}_{name}', read(out/name))
                (out/name).unlink()
        campaign = None
        try:
            campaign = Campaign(out, lock)
            campaign.run()
            comparison = read(out/'comparisons.json')
            if not completion_gate(comparison, campaign.identity):
                raise RuntimeError('Completion guard: six accepted rows and comparisons required')
            write(out/'DONE.json', dict(status='all_six_resolution_points_and_comparisons_saved',
                fingerprint=campaign.identity, comparison_sha256=sha(out/'comparisons.json'),
                accepted_matrix_rows=6, convergence_established=False))
        except BaseException as exc:
            if campaign is not None:
                campaign.state.update(status='incomplete', failure=f'{type(exc).__name__}: {exc}')
                campaign.save()
                campaign.finalize()
            write(out/'FAILED.json', dict(status='incomplete', failure=f'{type(exc).__name__}: {exc}',
                  fingerprint=None if campaign is None else campaign.identity,
                  pending=list(MATRIX) if campaign is None else
                    [r for r in campaign.state['matrix'].values() if r['status'] != 'accepted']))
            raise


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument('--prepare', action='store_true')
    mode.add_argument('--run', action='store_true')
    mode.add_argument('--worker', type=Path)
    parser.add_argument('--out', type=Path, default=DEFAULT_OUT)
    args = parser.parse_args()
    if args.worker:
        worker(args.worker)
        return
    out = checked_out(args.out)
    if args.prepare:
        with Exclusive(out):
            source, config = source_identity(), configuration()
            write(out/'PREPARED.json', dict(status='prepared_not_launched', source=source, config=config,
                 fingerprint=fingerprint(dict(source=source, config=config)),
                 matrix=[dict(r, status='pending') for r in MATRIX]))
        print(json.dumps(dict(status='prepared_not_launched', matrix_rows=6)))
        return
    def interrupted(signum, _frame):
        raise KeyboardInterrupt(f'D3 interrupted by signal {signum}; checkpoints preserved')
    for sig in (signal.SIGTERM, signal.SIGINT, signal.SIGHUP):
        signal.signal(sig, interrupted)
    try:
        with contextlib.redirect_stdout(TailLog(out/'campaign.log')), contextlib.redirect_stderr(TailLog(out/'campaign_stderr.log')):
            execute(out)
    except BaseException:
        TailLog(out/'campaign_stderr.log').write(traceback.format_exc())
        print(json.dumps(dict(status='incomplete', marker=str(out/'FAILED.json'))))
        raise SystemExit(1)
    print(json.dumps(dict(status='complete', marker=str(out/'DONE.json'))))


if __name__ == '__main__':
    main()
