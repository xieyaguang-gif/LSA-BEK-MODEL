#!/usr/bin/env python3
"""Bounded D2j checks: coverage, provenance, complex PARDISO, existing +20i vector recertification."""
import os
os.environ['PYTHONDONTWRITEBYTECODE'] = '1'
import argparse
import copy
import json
import time
from pathlib import Path
from types import SimpleNamespace
import numpy as np
from scipy import sparse
from scipy.optimize import linear_sum_assignment
import fullbvp_phaseD2j_campaign as j


def require(ok, message):
    if not ok:
        raise AssertionError(message)


def fake(imag, radius, complete=True):
    return dict(sigma=[.05, imag], complete=complete,
                eigenpairs=[dict(value=[.05-radius, imag], accepted=True, phase_mass_overlap=0.) for _ in range(6)])


def algorithms():
    require(not j.coverage([])['complete'], 'empty scan accepted')
    circles = [fake(0, .052), fake(5, 1.), fake(70, 1.)]
    require(not j.coverage(circles)['complete'], 'gapped scan accepted')
    midpoint = j.next_center(circles)
    require(6 < midpoint < 69, 'largest uncovered interval not selected')
    require(j.coverage([fake(0, 36), fake(70, 36)])['complete'], 'overlapping full interval rejected')
    require(not j.coverage([fake(0, 36), fake(70, 36, False)])['complete'], 'failed center counted as complete')
    zero_line = j.coverage([fake(0, .0500001), fake(70, 69.95)])
    require(not zero_line['complete'], 'threshold-line endpoint gap ignored')
    require(j.is_phase(dict(value=[1e-11, 1e-12], phase_mass_overlap=1.)), 'true phase root not marked')
    require(not j.is_phase(dict(value=[2e-5, 0.], phase_mass_overlap=.9999999999)), 'weak nonzero root wrongly removed')
    candidates = dict(candidates=[dict(value=[-1e-4, -.0003], sigma=[.05, 0.]),
                                  dict(value=[.01, 32.5], sigma=[.05, 30.]),
                                  dict(value=[.01, 71.], sigma=[.05, 70.])])
    require(j.checks_from_candidates(candidates) == [0., 32.5, 70.], 'candidate check scheduling failed')
    relation = j.frequency_relation([-.1, 2.], 52.7)
    require(abs(relation['wave_disk'] + 15.3) < 1e-12, 'disk-frame conversion failed')
    require(abs(relation['signed_harmonics'][5]['disk_frequency'] + 17.3) < 1e-12, 'sideband sign failed')
    rejected = 0
    for path in (j.ROOT / 'examples/bad.json', j.PROD / '../outside.json'):
        try:
            j.output_path(path)
        except ValueError:
            rejected += 1
    require(rejected == 2, 'output boundary checks failed')
    return dict(coverage=True, failed_center_rejection=True, weak_nonphase_preserved=True,
                trigger_scheduling=True, frame_sign=True, path_guard=True)


def complex_pardiso(out):
    expected = np.array([-.02*(i+1)+1j*(18+i) for i in range(6)])
    other = [-.3*(i+1)+1j*(i+1) for i in range(6)]
    blocks = [np.array([[-z.real, z.imag], [-z.imag, -z.real]]) for z in list(expected)+other]
    J = sparse.block_diag(blocks + [np.zeros((1, 1)), np.ones((1, 1))], format='csr')
    b = np.ones(J.shape[0]); b[-1] = 0
    B = sparse.diags(b, format='csr')
    phase = np.zeros(J.shape[0]); phase[-2] = 1
    temporal = SimpleNamespace(J=J, B=B, b=b, size=J.shape[0], neutral=lambda: phase)
    target = out / 'complex_20i'
    deadline = j.old.d2.v1.Deadline(90.)
    result = j.g.run_center(temporal, J, .05+20j, target, 'D2j-small-complex-test-v1', deadline, k=6, ncv=40)
    require(result['complete'], 'complex shift solve failed: '+str(result.get('failure')))
    actual = np.array([complex(*p['value']) for p in result['eigenpairs']])
    distances = abs(actual[:, None]-expected[None, :])
    rows, columns = linear_sum_assignment(distances)
    error = float(max(distances[rows, columns]))
    require(error < 1e-8, 'complex shift does not recover known spectrum')
    log = Path(result['arrays']).parent / 'inverse.jsonl'
    inverse = [json.loads(s) for s in log.read_text().splitlines()]
    require(inverse and all(x['accepted'] and x['normwise_backward'] <= 1e-12 and x['original_relative'] <= 1e-5 for x in inverse),
            'inverse backward/RHS gate failed')
    cached = j.g.run_center(temporal, None, .05+20j, target, 'D2j-small-complex-test-v1', deadline, k=6, ncv=40)
    require(cached['arrays_sha256'] == result['arrays_sha256'], 'completed center was recomputed')
    rejected = False
    try:
        j.g.run_center(temporal, None, .05+20j, target, 'wrong-fingerprint', deadline, k=6, ncv=40)
    except RuntimeError:
        rejected = True
    require(rejected, 'changed fingerprint accepted')
    return dict(max_eigenvalue_error=error, original_residual_max=max(x['original_relative'] for x in result['eigenpairs']),
                inverse_calls=len(inverse), cache_reused=True, bad_fingerprint_rejected=True)


def existing_20i():
    seed = j.bundle(j.source_point('upper725'))
    marker = seed['centers']['20.0']
    mesh = j.previous.mapping(24, 4, seed['intervals'], 2.)
    with np.load(seed['fields']) as data:
        q, R, omega = data['field'].copy(), float(data['Rout']), float(data['omega'])
    deadline = j.old.d2.v1.Deadline(120.)
    t = j.old.d2.v1.Temporal(mesh, q, R, omega, deadline)
    stationarity = j.old.stationarity_gate(t)
    checks = j.recertify(t, j.read(marker), deadline)
    return dict(source=marker, count=len(checks), original_residual_max=max(x['original_relative'] for x in checks),
                stationarity=stationarity, production_factorization_performed=False)



def reuse_and_localization(out):
    records = []
    cases = [('upper720', 24, 2.), ('upper725', 24, 2.),
             ('upper720', 40, 2.), ('upper725', 40, 2.), ('upper720', 40, 3.)]
    for branch, n, a in cases:
        params = j.parameters(branch, n, a, [0.])
        point = out / f'{branch}_N{n}_a{int(a)}' / params['source']
        point.mkdir(parents=True, exist_ok=True)
        j.old.write(point / 'manifest.json', dict(fingerprint=j.old.d2.v1.digest(params), params=params))
        deadline = j.old.d2.v1.Deadline(120.)
        t, correction = j.prepare_state(params, point, deadline)
        center = j.cached_center(params, t, point, 0, 0., deadline)
        require(center is not None and center['reused_eigenpairs'], 'existing real center not reused')
        again = j.cached_center(params, t, point, 0, 0., deadline)
        require(again == center, 'cache resume changed certificate')
        best = max(j.selected_pairs([center], all_nonphase=True), key=lambda p: p['value'][0])
        localized = j.localize(t, params, [best], deadline)[0]
        corrupt = copy.deepcopy(center); corrupt['arrays_sha256'] = '0'*64
        rejected = False
        try:
            j.recertify(t, corrupt, deadline)
        except RuntimeError:
            rejected = True
        require(rejected, 'bad eigenvector hash accepted')
        records.append(dict(branch=branch, n=n, a=a, center=center,
                            max_nonphase=best, localization=localized, bad_hash_rejected=True,
                            corrected_state_imported=True, factorization_count_this_run=0))
    j.old.write(out / 'existing_real_centers.json', dict(records=records, production_started=False))
    return dict(cases=len(records), all_original_pairs_recertified=6*len(records),
                maximum_nonphase_localizations=len(records), factorization_count=0,
                result=str(out / 'existing_real_centers.json'))


def main(out):
    start = time.monotonic()
    cpus = sorted(os.sched_getaffinity(0)); os.sched_setaffinity(0, set(cpus[:16]))
    out = j.output_path(out); out.mkdir(parents=True, exist_ok=True)
    result = dict(algorithms=algorithms(), complex_pardiso=complex_pardiso(out), existing_20i=existing_20i(), reuse=reuse_and_localization(out))
    # Validate all six prospective source choices; a3 upper725 must request physical-z interpolation + correction.
    configs = [j.parameters(b, n, a, [0.]) for b in ('upper720', 'upper725') for n, a in ((24, 2.), (40, 2.), (40, 3.))]
    require(sum(p['import_state'] for p in configs) == 5, 'unexpected state reuse/mapping correction plan')
    result.update(success=True, seconds=time.monotonic()-start, production_started=False,
                  existing_source_configurations=len(configs), fingerprints=j.fingerprints())
    j.old.write(out/'validation.json', result)
    print(json.dumps(dict(success=True, seconds=result['seconds'], max_complex_error=result['complex_pardiso']['max_eigenvalue_error'],
                         existing_20i_residual=result['existing_20i']['original_residual_max'], production_started=False)))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--out', type=Path, required=True)
    main(parser.parse_args().out)
