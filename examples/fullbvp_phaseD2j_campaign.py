#!/usr/bin/env python3
"""D2j: positive-frequency scan, adaptive coverage and triggered N40/map checks.

Production is launched by the supervisor, never by the preparation session.
The sampled Ritz disks are a coverage diagnostic, not a spectral-count theorem.
"""
import os
os.environ['PYTHONDONTWRITEBYTECODE'] = '1'
for key in ('OPENBLAS_NUM_THREADS', 'OMP_NUM_THREADS', 'MKL_NUM_THREADS',
            'NUMEXPR_NUM_THREADS', 'BLIS_NUM_THREADS'):
    os.environ[key] = '1'
import argparse
import copy
import fcntl
import json
import resource
import sys
import time
import uuid
from pathlib import Path
import numpy as np
import fullbvp_phaseD2i_campaign as previous
import fullbvp_phaseD2i_localize as loc

h, g, old = previous.h, previous.g, previous.old
ROOT, OUT = previous.ROOT, previous.OUT
PROD = OUT / 'production_v8'
AUDIT = OUT / 'D2j'
BASE_IMAG = tuple(sorted(set(range(0, 71, 5)) | {2}))
THRESHOLD = -1e-3
MAX_CENTERS = 96
MIN_SPACING = 1e-4
POINT_BUDGET = 172800.  # cumulative 48-hour hard limit, including restart reservations
CENTER_BUDGET = 4800.
_saved_fingerprints = previous.fingerprints
_saved_verify = old.verify_worker_evidence
_saved_launch = g.prior._original_launch


def output_path(path, allow_root=False):
    path = Path(path).absolute()
    if '..' in path.parts or path.resolve() != path or any(p.is_symlink() for p in (path, *path.parents)):
        raise ValueError('unsafe D2j output path')
    if allow_root and path == PROD:
        return path
    if not any(path.is_relative_to(r) and path != r for r in (PROD, AUDIT)):
        raise ValueError('D2j writes only below production_v8 or D2j')
    return path


def read(path):
    return json.loads(Path(path).read_text())


def fingerprints():
    paths = [Path(__file__), ROOT / 'examples/fullbvp_phaseD2j_validate.py', PROD / 'commands.sh']
    return dict(_saved_fingerprints(), **{str(p.resolve()): old.d2.v1.sha(p) for p in paths})


def source_point(branch, n=24, a=2.):
    source = branch + '_H6'  # historical folder name; metadata below checks actual H=4
    if n == 24:
        point = OUT / ('production_v4/run_D2f_baseline' if branch == 'upper720' else 'production_v6/upper725') / source
    else:
        suffix = '_a3' if a == 3 and branch == 'upper720' else ''
        point = OUT / 'production_v7' / (branch + '_N40' + suffix) / source
    return point


def bundle(point):
    """Read-only provenance snapshot; only accepted, metadata-matching centers."""
    point = Path(point).resolve()
    manifest, correction = read(point / 'manifest.json'), read(point / 'corrected.json')
    params = manifest['params']
    fp = old.d2.v1.digest(params)
    fields = h.source_file(correction['fields'], point)
    if manifest['fingerprint'] != fp or correction['fingerprint'] != fp:
        raise RuntimeError('source state fingerprint mismatch')
    if old.d2.v1.sha(fields) != correction['fields_sha256']:
        raise RuntimeError('source field hash mismatch')
    for path, sha in params['code_sha256'].items():
        if old.d2.v1.sha(path) != sha:
            raise RuntimeError('source code changed: ' + path)
    paths = [point / 'manifest.json', point / 'corrected.json', fields]
    centers = {}
    for marker in sorted(point.glob('center_*/DONE.json')):
        result = read(marker)
        arrays = h.source_file(result['arrays'], point)
        if (not result.get('complete') or result['fingerprint'] != fp or result['k'] != 6 or result['ncv'] != 40
                or old.d2.v1.sha(arrays) != result['arrays_sha256'] or len(result['eigenpairs']) != 6
                or not all(x['accepted'] and np.isfinite(x['original_relative']) and x['original_relative'] <= old.PAIR_ACCEPT
                           for x in result['eigenpairs'])):
            raise RuntimeError('source center certificate mismatch: ' + str(marker))
        if result['sigma'][0] != .05:
            continue
        centers[str(float(result['sigma'][1]))] = str(marker)
        paths.extend([marker, arrays])
    return dict(point=str(point), fields=str(fields), centers=centers,
                n=params['n'], H=params['H'], intervals=params['intervals'],
                mapping=params.get('mapping', dict(a=2., b=.6, c=.5)),
                A1=correction['after']['A1'], Rout=correction['after']['Rout'],
                sha256={str(p): old.d2.v1.sha(p) for p in paths})


def parameters(branch, n=24, a=2., shifts=None):
    if branch not in ('upper720', 'upper725') or n not in (24, 40) or a not in (2., 3.) or (n == 24 and a != 2.):
        raise ValueError('unsupported D2j case')
    seed = bundle(source_point(branch, n, a))
    if (seed['n'], seed['H'], seed['intervals']) != (n, 4, int(branch[-3:]) - 400):
        raise RuntimeError('source resolution or radial extent mismatch')
    imag = list(BASE_IMAG) if shifts is None else sorted(set(float(x) for x in shifts))
    if not imag or not all(np.isfinite(x) and 0 <= x <= 70 for x in imag):
        raise ValueError('centers must be in positive frequency interval [0,70]')
    return dict(revision='D2j', branch=branch, source=branch + '_H6', n=n, H=4,
                intervals=seed['intervals'], requested_dr=1., mapping=dict(a=a, b=.6, c=.5),
                seed=seed, import_state=(seed['mapping']['a'] == a),
                shifts=[[.05, x] for x in imag], k=6, ncv=40, threads=16,
                budget_seconds=POINT_BUDGET, center_budget_seconds=CENTER_BUDGET,
                memory_bytes=h.MEMORY_BYTES, stationary_accept=old.STATIONARY_ACCEPT,
                pair_accept=old.PAIR_ACCEPT, linear_policy=g.POLICY,
                adaptive=(n == 24), max_centers=MAX_CENTERS, min_spacing=MIN_SPACING,
                trigger_real_gt=THRESHOLD, coverage_real_lines=[.05, THRESHOLD],
                phase_rule=dict(abs_lambda_max=1e-8, mass_overlap_min=1 - 1e-10),
                code_sha256=fingerprints())


def is_phase(pair):
    return abs(complex(*pair['value'])) <= 1e-8 and pair.get('phase_mass_overlap', 0.) >= 1 - 1e-10


def center_summary(center):
    sigma = complex(*center['sigma'])
    pairs = center.get('eigenpairs', [])
    complete = bool(center.get('complete') and len(pairs) == 6 and all(p.get('accepted') for p in pairs))
    nonphase = [(i, p) for i, p in enumerate(pairs) if p.get('accepted') and not is_phase(p)]
    best = max(nonphase, key=lambda item: item[1]['value'][0], default=None)
    return dict(sigma=center['sigma'], complete=complete,
                radius=max((abs(complex(*p['value']) - sigma) for p in pairs), default=0.) if complete else 0.,
                max_nonphase=None if best is None else dict(index=best[0], **best[1]),
                excluded_phase=[p['value'] for p in pairs if is_phase(p)],
                reused=bool(center.get('reused_eigenpairs')), arrays=center.get('arrays'),
                arrays_sha256=center.get('arrays_sha256'), failure=center.get('failure'))


def coverage(centers):
    summaries = sorted((center_summary(c) for c in centers), key=lambda r: r['sigma'][1])
    good = [s for s in summaries if s['complete']]
    adjacent = [dict(left=x['sigma'][1], right=y['sigma'][1],
                     overlap=x['radius'] + y['radius'] - (y['sigma'][1] - x['sigma'][1]))
                for x, y in zip(good[:-1], good[1:])]
    lines = []
    for real in (.05, THRESHOLD):
        intervals = []
        for row in good:
            # Tiny inward margin avoids treating numerical tangency as covered.
            radius = row['radius'] * (1 - 1e-8)
            square = radius * radius - (real - row['sigma'][0]) ** 2
            if square >= 0:
                half = float(np.sqrt(square)); y = row['sigma'][1]
                if y + half >= 0 and y - half <= 70:
                    intervals.append((max(0., y - half), min(70., y + half)))
        cursor, gaps = 0., []
        for lo, hi in sorted(intervals):
            if lo > cursor + 1e-10:
                gaps.append([cursor, lo])
            cursor = max(cursor, hi)
        if cursor < 70 - 1e-10:
            gaps.append([cursor, 70.])
        lines.append(dict(real=real, gaps=gaps))
    return dict(complete=bool(len(good) == len(summaries) and good and
                             all(x['overlap'] > 0 for x in adjacent) and all(not x['gaps'] for x in lines)),
                adjacent=adjacent, lines=lines, sampled_centers=len(good),
                exhaustive_stability=False, interpretation='geometric coverage by returned k6 Ritz disks; no spectral-count certificate')


def next_center(centers):
    state = coverage(centers)
    if state['complete']:
        return None
    gaps = [gap for line in state['lines'] for gap in line['gaps']]
    gaps.extend([x['left'], x['right']] for x in state['adjacent'] if x['overlap'] <= 0)
    known = [float(c['sigma'][1]) for c in centers]
    for lo, hi in sorted(gaps, key=lambda x: x[1] - x[0], reverse=True):
        middle = (lo + hi) / 2
        if min(abs(middle - y) for y in known) >= MIN_SPACING:
            return middle
    return None


def selected_pairs(centers, all_nonphase=False):
    selected = []
    for center in centers:
        if not center.get('complete'):
            continue
        for index, pair in enumerate(center['eigenpairs']):
            if not pair['accepted'] or is_phase(pair) or (not all_nonphase and pair['value'][0] <= THRESHOLD):
                continue
            z = complex(*pair['value'])
            # Never merge two same-center Ritz pairs; be stricter near the phase root.
            tolerance = min(1e-7 * (1 + abs(z)), 1e-10) if abs(z) < 1e-3 else 1e-7 * (1 + abs(z))
            if any(p['arrays'] != center['arrays'] and abs(z - complex(*p['value'])) < tolerance for p in selected):
                continue
            selected.append(dict(index=index, arrays=center['arrays'], arrays_sha256=center['arrays_sha256'],
                                 sigma=center['sigma'], **pair))
    return selected


def frequency_relation(value, omega, H=4):
    imag = value[1]
    disk = omega - 68.  # ro=-1 => Omega_disk/Omega_BEK=1
    k = round(abs(imag) / abs(disk))
    return dict(imag_lambda=imag, wave_lab=omega, wave_disk=disk,
                ratio_to_abs_wave_disk=imag / abs(disk), nearest_abs_harmonic=k,
                residual_from_abs_harmonic=abs(imag) - k * abs(disk),
                convention='exp(lambda*t) exp(i*j*(beta*theta-omega_wave*t)); omega_j=j*omega_wave-Im(lambda)',
                signed_harmonics=[dict(j=j, disk_frequency=j * disk - imag, lab_frequency=j * omega - imag)
                                  for j in range(-H, H + 1)],
                interpretation='kinematic sideband family; harmonic proximity alone does not identify a resonance')


def prepare_state(params, out, deadline):
    seed = params['seed']
    for path, sha in seed['sha256'].items():
        deadline.check()
        if old.d2.v1.sha(path) != sha:
            raise RuntimeError('source evidence changed: ' + path)
    if params['import_state'] and not (out / 'corrected.json').exists():
        saved = read(Path(seed['point']) / 'corrected.json')
        fields = out / 'corrections/import/fields.npz'
        h.checked_copy(Path(seed['fields']), fields, saved['fields_sha256'])
        saved.update(fingerprint=old.d2.v1.digest(params), fields=str(fields), imported_from=seed['point'], mapping=params['mapping'])
        old.write(out / 'corrected.json', saved)
    temporal, saved = previous.corrected_temporal(params, out, deadline)
    with np.load(saved['fields']) as data:
        if (int(data['n']), int(data['H']), len(data['radii']) - 1) != (params['n'], 4, params['intervals']) or not bool(data['accepted']):
            raise RuntimeError('corrected field metadata mismatch')
    return temporal, saved


def recertify(temporal, result, deadline):
    arrays = Path(result['arrays'])
    if old.d2.v1.sha(arrays) != result['arrays_sha256']:
        raise RuntimeError('eigenvector hash mismatch')
    with np.load(arrays) as data:
        values, vectors = data['values'].copy(), data['vectors'].copy()
    if values.shape != (6,) or vectors.shape != (temporal.size, 6):
        raise RuntimeError('eigenpair dimensions mismatch')
    if not np.array_equal(values, np.array([complex(*p['value']) for p in result['eigenpairs']])):
        raise RuntimeError('certificate differs from stored eigenvalues')
    phase = temporal.neutral()
    checks = []
    for value, vector in zip(values, vectors.T):
        deadline.check()
        checks.append(old.d2.v1.certify(temporal.J, temporal.B, max(temporal.b), value, vector,
                                      tol=old.PAIR_ACCEPT, phase=phase))
    if not all(c['accepted'] for c in checks):
        raise RuntimeError('cached eigenpair fails original-operator recertification')
    return checks


def cached_center(params, temporal, out, index, imag, deadline):
    target = out / f'center_{index:02d}'
    marker = target / 'DONE.json'
    fp = old.d2.v1.digest(params)
    if marker.exists():
        result = read(marker)
        if result['fingerprint'] != fp or result['sigma'] != [.05, imag] or result['k'] != 6 or result['ncv'] != 40:
            raise RuntimeError('D2j center cache configuration mismatch')
        recertify(temporal, result, deadline)
        return result
    source = params['seed']['centers'].get(str(float(imag))) if params['import_state'] else None
    if source is None:
        return None
    previous_result = read(source)
    checks = recertify(temporal, previous_result, deadline)
    arrays = target / 'attempts/import/eigenpairs.npz'
    h.checked_copy(Path(previous_result['arrays']), arrays, previous_result['arrays_sha256'])
    result = dict(previous_result, fingerprint=fp, sigma=[.05, imag], arrays=str(arrays), eigenpairs=checks,
                  reused_eigenpairs=True, source_result=source, source_result_sha256=old.d2.v1.sha(source),
                  factorization_count_this_run=0, seconds=0., reused_seconds=previous_result.get('seconds'),
                  source_linear_policy=previous_result.get('linear_policy'))
    old.write(arrays.parent / 'result.json', result)
    old.write(marker, result)
    return result


def localize(temporal, params, refs, deadline):
    xi, r = temporal.map.op.grid.xi, temporal.p.radii
    matrices, _, _ = loc.quadrature(xi, a=params['mapping']['a'])
    fine, _, _ = loc.quadrature(xi, order=160, a=params['mapping']['a'])
    M = sum(matrices[:4])
    phase = temporal.neutral()
    shape = (temporal.nr, 2 * temporal.H + 1, 4, temporal.map.m)
    phi = old.d2.v1.lift(phase, temporal.nr, temporal.H, temporal.s).reshape(shape)
    pp = loc.inner(phi, phi, M, r).real
    records = []
    for ref in refs:
        deadline.check()
        if old.d2.v1.sha(ref['arrays']) != ref['arrays_sha256']:
            raise RuntimeError('localization array hash mismatch')
        with np.load(ref['arrays']) as data:
            value, vector = data['values'][ref['index']], data['vectors'][:, ref['index']].copy()
        cert = old.d2.v1.certify(temporal.J, temporal.B, max(temporal.b), value, vector, tol=old.PAIR_ACCEPT, phase=phase)
        if not cert['accepted']:
            raise RuntimeError('localization original-pencil residual failed')
        q = old.d2.v1.lift(vector, temporal.nr, temporal.H, temporal.s).reshape(shape)
        coefficient = loc.inner(phi, q, M, r) / pp
        perpendicular = q - coefficient * phi
        raw, projected = loc.measures(q, matrices, r), loc.measures(perpendicular, matrices, r)
        refined = loc.measures(perpendicular, fine, r)
        difference = abs(refined['kinetic_norm_squared'] / projected['kinetic_norm_squared'] - 1)
        orth = abs(loc.inner(phi, perpendicular, M, r)) / np.sqrt(pp * projected['kinetic_norm_squared'])
        if difference > 1e-7 or orth > 1e-7:
            raise RuntimeError('localization quadrature/phase-orthogonality check failed')
        records.append(dict(ref=ref, certificate=cert, phase_coefficient=old.d2.v1.zpair(coefficient),
                            raw=raw, phase_orthogonal_shape=projected,
                            orthogonal_kinetic_fraction=projected['kinetic_norm_squared'] / raw['kinetic_norm_squared'],
                            phase_orthogonality=float(orth), quadrature_relative_difference=difference,
                            projection_is_shape_diagnostic_not_eigenvector=True,
                            frequency=frequency_relation(ref['value'], temporal.p.config.omega)))
    return records


def verify_worker_evidence(params, out, child):
    centers = child.get('centers', [])
    count = len(params['shifts'])
    # Preserve the inherited file/hash, residual, stationary-field and base-center checks.
    base = dict(child, centers=centers[:count])
    _saved_verify(params, out, base)
    if not count <= len(centers) <= MAX_CENTERS:
        raise RuntimeError('invalid adaptive center count')
    for index, center in enumerate(centers):
        marker = out / f'center_{index:02d}/DONE.json'
        arrays = output_path(center['arrays'])
        if (read(marker) != center or center['fingerprint'] != old.d2.v1.digest(params)
                or not arrays.is_relative_to(marker.parent / 'attempts')
                or old.d2.v1.sha(arrays) != center['arrays_sha256']
                or center['k'] != 6 or center['ncv'] != 40 or not center.get('complete')
                or center['sigma'][0] != .05 or not 0 <= center['sigma'][1] <= 70
                or len(center['eigenpairs']) != 6
                or not all(p['accepted'] and np.isfinite(p['original_relative']) and p['original_relative'] <= old.PAIR_ACCEPT
                           for p in center['eigenpairs'])):
            raise RuntimeError('adaptive center evidence mismatch')
    if len({tuple(c['sigma']) for c in centers}) != len(centers):
        raise RuntimeError('duplicate center')
    if params['adaptive'] and (not coverage(centers)['complete'] or child['coverage'] != coverage(centers)):
        raise RuntimeError('unresolved or changed frequency coverage')
    expected = selected_pairs(centers, all_nonphase=params['n'] == 40)
    localized = {(x['ref']['arrays'], x['ref']['index']): x for x in child.get('localization', [])}
    for ref in expected:
        item = localized.get((ref['arrays'], ref['index']))
        if (item is None or item['ref'] != ref or not item['certificate']['accepted']
                or item['certificate']['original_relative'] > old.PAIR_ACCEPT
                or item['quadrature_relative_difference'] > 1e-7 or item['phase_orthogonality'] > 1e-7):
            raise RuntimeError('missing or invalid candidate localization')


def worker(job, lock_fd):
    params, out = job['params'], output_path(job['out'])
    actual, expected = os.fstat(lock_fd), os.stat(out.parent / '.run.lock')
    if (actual.st_dev, actual.st_ino) != (expected.st_dev, expected.st_ino):
        raise RuntimeError('wrong inherited run lock')
    if params != parameters(params['branch'], params['n'], params['mapping']['a'], [x[1] for x in params['shifts']]):
        raise RuntimeError('worker source/configuration changed')
    h.control_limit()
    start = time.monotonic()
    deadline = old.d2.v1.Deadline(job['remaining'])
    centers, failure, localized = [], None, []
    with h.memory_slot(PROD / '.memory_slots') as slot:
        deadline.check()
        resource.setrlimit(resource.RLIMIT_AS, (h.MEMORY_BYTES, h.MEMORY_BYTES))
        old.write(out / 'resources.json', dict(slot=slot, threads=16, worker_memory_bytes=h.MEMORY_BYTES,
                                             aggregate_address_space_bound=802_000_000_000))
        temporal, correction = prepare_state(params, out, deadline)
        J = None
        frequencies = [x[1] for x in params['shifts']]
        try:
            index = 0
            while True:
                deadline.check()
                if index == len(frequencies):
                    if not params['adaptive'] or coverage(centers)['complete']:
                        break
                    new = next_center(centers)
                    if new is None or index >= MAX_CENTERS:
                        raise RuntimeError('coverage gaps remain at center/minimum-spacing limit')
                    frequencies.append(new)
                imag = frequencies[index]
                result = cached_center(params, temporal, out, index, imag, deadline)
                if result is None:
                    if J is None:
                        J, assembly = old.d2.assemble(temporal, deadline)
                        old.write(out / 'assembly.json', assembly)
                        if np.iscomplexobj(J.data) and np.any(J.data.imag):
                            raise RuntimeError('packed pencil must be real for positive-half-plane reduction')
                    limit = old.d2.v1.Deadline(min(CENTER_BUDGET, deadline.remaining()))
                    result = g.run_center(temporal, J, .05 + 1j * imag, out / f'center_{index:02d}',
                                          old.d2.v1.digest(params), limit, k=6, ncv=40)
                centers.append(result)
                old.write(out / 'scan_progress.json', dict(centers=[center_summary(c) for c in centers], coverage=coverage(centers)))
                if not result['complete']:
                    raise RuntimeError('center failed: ' + str(result.get('failure')))
                index += 1
        except Exception as exc:
            failure = f'{type(exc).__name__}: {exc}'
        refs = selected_pairs(centers, all_nonphase=params['n'] == 40)
        best = max(selected_pairs(centers, all_nonphase=True), key=lambda p: p['value'][0], default=None)
        if best is not None and not any(r['arrays'] == best['arrays'] and r['index'] == best['index'] for r in refs):
            refs.append(best)
        try:
            localized = localize(temporal, params, refs, deadline)
        except Exception as exc:
            failure = failure or f'localization: {type(exc).__name__}: {exc}'
        cov = coverage(centers)
        result = dict(fingerprint=old.d2.v1.digest(params), correction=correction, centers=centers,
                      coverage=cov, complete=bool(failure is None and len(centers) >= len(params['shifts'])
                                                and (not params['adaptive'] or cov['complete'])),
                      failure=failure, candidates=selected_pairs(centers), localization=localized,
                      max_nonphase=best, frequency=None if best is None else frequency_relation(best['value'], temporal.p.config.omega),
                      seconds=time.monotonic() - start, peak_rss_bytes=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss * 1024,
                      original_resolution_certified=False, exhaustive_stability=False)
        old.write(out / 'worker_result.json', result)
        return 0 if result['complete'] else 1


def campaign(spec_path, run_out):
    h.control_limit()
    params, out = read(spec_path), output_path(run_out)
    cpus = sorted(os.sched_getaffinity(0))
    os.sched_setaffinity(0, set(cpus[:16]))
    out.mkdir(parents=True, exist_ok=True)
    fd = os.open(out / '.run.lock', os.O_RDWR | os.O_CREAT | os.O_CLOEXEC | os.O_NOFOLLOW, 0o600)
    with os.fdopen(fd, 'r+') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        point = out / params['source']
        command = [sys.executable, str(Path(__file__).resolve()), 'worker', '--job', str(point / 'job.json'), '--lock-fd', str(lock.fileno())]
        result = _saved_launch(params, point, command=command, lock_fd=lock.fileno())
        return 0 if result['complete'] else 1


def save_spec(root, name, params):
    path = root / 'specs' / (name + '.json')
    if path.exists() and read(path) != params:
        raise RuntimeError('immutable run specification changed; use a fresh D2J_RUN_ROOT')
    if not path.exists():
        old.write(path, params)
    return name, [sys.executable, str(Path(__file__).resolve()), 'campaign', '--spec', str(path), '--out', str(root / name)]


def checks_from_candidates(coarse):
    imag = set()
    for p in coarse.get('candidates', []):
        y = abs(p['value'][1])
        # The reused real center resolves the weak near-zero cluster; do not pretend high phase overlap removes it.
        if y <= .01:
            y = 0.
        if y <= 70:
            imag.add(float(y))
        else:
            # A returned Ritz root can lie outside the center interval; retain the original permitted center for checking it.
            imag.add(float(p['sigma'][1]))
    return sorted(imag)


def report_results(root):
    records = {}
    for path in sorted(root.glob('upper*/upper*/worker_result.json')):
        value = read(path)
        records[path.parent.parent.name] = dict(point=str(path.parent), result=value)
    comparisons = []
    for branch in ('upper720', 'upper725'):
        base = records.get(branch + '_N24', {}).get('result', {})
        for candidate in base.get('candidates', []):
            row = dict(branch=branch, N24=candidate, checks=[])
            for suffix in ('_N40', '_N40_a3'):
                target = records.get(branch + suffix, {}).get('result', {})
                pairs = selected_pairs(target.get('centers', []), all_nonphase=True)
                nearest = min(pairs, key=lambda p: abs(complex(*p['value']) - complex(*candidate['value'])), default=None)
                row['checks'].append(dict(run=branch + suffix, nearest=nearest,
                                          distance=None if nearest is None else abs(complex(*nearest['value']) - complex(*candidate['value'])),
                                          localization_available=bool(target.get('localization')),
                                          physical_mode_identity='requires shape/subspace review; nearest eigenvalue is only a candidate match'))
            comparisons.append(row)
    summary = dict(runs={name: dict(point=item['point'], complete=item['result']['complete'],
                                    coverage=item['result']['coverage'], centers=[center_summary(c) for c in item['result']['centers']],
                                    max_nonphase=item['result']['max_nonphase'], frequency=item['result']['frequency'],
                                    candidates=item['result']['candidates']) for name, item in records.items()},
                   candidate_comparisons=comparisons, exhaustive_stability=False,
                   physical_convergence='review N24/N40/a3 eigenvalues and localized shapes; not inferred from residuals')
    old.write(root / 'SUMMARY.json', summary)
    lines = ['# D2j production results', '', 'Returned k6 Ritz disks diagnose sampled coverage, not exhaustive spectral completeness.', '',
             '| Run | Im sigma | Radius | Max nonphase lambda | Source |', '|---|---:|---:|---|---|']
    for name, item in summary['runs'].items():
        for c in item['centers']:
            p = c['max_nonphase']
            value = 'FAILED' if p is None else f"{p['value'][0]:+.9g}{p['value'][1]:+.9g}i"
            lines.append(f"| {name} | {c['sigma'][1]:.9g} | {c['radius']:.9g} | {value} | {'reused' if c['reused'] else 'new'} |")
    lines += ['', 'Full frequency transformations, coverage gaps, root references and comparison candidates: `SUMMARY.json`.',
              'Harmonic, velocity, z-interval and radial-interval localization: each run’s `worker_result.json`.',
              'Phase-orthogonal fields are shape diagnostics, not eigenvectors. Physical mode convergence remains a review step.']
    path = output_path(root / 'RESULTS.md')
    path.write_text('\n'.join(lines) + '\n')
    return summary


def dispatch(root):
    h.control_limit()
    root = output_path(root, allow_root=True)
    root.mkdir(parents=True, exist_ok=True)
    fd = os.open(PROD / '.dispatch.lock', os.O_RDWR | os.O_CREAT | os.O_CLOEXEC | os.O_NOFOLLOW, 0o600)
    with os.fdopen(fd, 'r+') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        attempts = root / 'dispatch_attempts' / f'{time.time_ns()}_{uuid.uuid4().hex[:8]}'
        specs = [save_spec(root, branch + '_N24', parameters(branch)) for branch in ('upper720', 'upper725')]
        scan = h.execute_waves(specs, attempts / 'scan', timeout_seconds=POINT_BUDGET + 120.)
        # Freeze followup specifications only once both adaptive scans are complete.
        # A failed/partial scan keeps candidates in SUMMARY; resume cannot invalidate an existing check plan.
        if not scan['complete']:
            report_results(root)
            old.write(root / 'FAILED.json', dict(complete=False, scan=scan, checks_pending=True))
            (root / 'DONE.json').unlink(missing_ok=True)
            print(json.dumps(dict(root=str(root), complete=False, checks_pending=True)))
            return 1
        followups = []
        for branch in ('upper720', 'upper725'):
            path = root / (branch + '_N24') / (branch + '_H6') / 'worker_result.json'
            if not path.exists():
                continue
            shifts = checks_from_candidates(read(path))
            if shifts:
                for a in (2., 3.):
                    name = branch + ('_N40' if a == 2. else '_N40_a3')
                    followups.append(save_spec(root, name, parameters(branch, 40, a, shifts)))
        refinement = h.execute_waves(followups, attempts / 'checks', timeout_seconds=POINT_BUDGET + 120.) if followups else dict(complete=True)
        report_results(root)
        complete = bool(scan['complete'] and refinement['complete'])
        old.write(root / ('DONE.json' if complete else 'FAILED.json'), dict(complete=complete, scan=scan, checks=refinement,
                  physical_convergence_review_pending=True, exhaustive_stability=False))
        (root / ('FAILED.json' if complete else 'DONE.json')).unlink(missing_ok=True)
        print(json.dumps(dict(root=str(root), complete=complete, production_summary=str(root / 'SUMMARY.json'))))
        return 0 if complete else 1


def inventory(root):
    """Bounded read-only source audit: no operator construction, LU or Arnoldi."""
    rows = []
    for branch, n, a in [('upper720', 24, 2.), ('upper725', 24, 2.),
                         ('upper720', 40, 2.), ('upper725', 40, 2.), ('upper720', 40, 3.)]:
        seed = bundle(source_point(branch, n, a))
        centers = [read(p) for p in seed['centers'].values()]
        pairs = selected_pairs(centers, all_nonphase=True)
        best = max(pairs, key=lambda p: p['value'][0], default=None)
        omega = read(Path(seed['point']) / 'corrected.json')['after']['omega']
        rows.append(dict(branch=branch, n=n, a=a, source=seed,
                         centers=[center_summary(c) for c in centers], coverage=coverage(centers),
                         candidates=selected_pairs(centers), max_nonphase=best,
                         frequency=None if best is None else frequency_relation(best['value'], omega)))
    result = dict(records=rows, production_started=False, original_operators_recertified_this_audit=False,
                  validated='manifest, code, field and eigenpair-file hashes; saved acceptance certificates')
    old.write(output_path(root) / 'source_inventory.json', result)
    print(json.dumps(dict(source_points=len(rows), source_centers=sum(len(x['centers']) for x in rows),
                         coarse_reusable_centers=sum(len(x['centers']) for x in rows if x['n'] == 24),
                         production_started=False)))


def install():
    # Per route: 398 GB worker + 2 GB controller <= 400 GB; shared dispatcher adds 2 GB.
    h.MEMORY_BYTES = old.MEMORY_BYTES = 398_000_000_000
    old.output_path = g.output_path = h.output_path = previous.output_path = output_path
    old.verify_worker_evidence = verify_worker_evidence


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
    args = parser.parse_args()
    if args.mode == 'dispatch':
        raise SystemExit(dispatch(args.root))
    if args.mode == 'worker':
        raise SystemExit(worker(read(output_path(args.job)), args.lock_fd))
    if args.mode == 'campaign':
        raise SystemExit(campaign(output_path(args.spec), args.out))
    if args.mode == 'inventory':
        inventory(args.root)
    if args.mode == 'report':
        report_results(output_path(args.root, allow_root=True))
