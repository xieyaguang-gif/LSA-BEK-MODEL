#!/usr/bin/env python3
"""Bounded D2 scientific implementation checks; never a production eigensolve."""
from __future__ import annotations
import os
for key in ('OPENBLAS_NUM_THREADS', 'OMP_NUM_THREADS', 'MKL_NUM_THREADS', 'NUMEXPR_NUM_THREADS', 'BLIS_NUM_THREADS'):
    os.environ[key] = '1'
os.environ['PYTHONDONTWRITEBYTECODE'] = '1'

import argparse
import gc
import json
import signal
import subprocess
import sys
import time
from pathlib import Path

import numpy as np
from scipy import linalg
from scipy.sparse.linalg import LinearOperator
import fullbvp_phaseD2_stability as d2


def require(value, message):
    if not value:
        raise AssertionError(message)


def relative(a, b):
    return float(np.linalg.norm(a - b) / max(np.linalg.norm(a), np.linalg.norm(b), 1e-300))


def reduced_checks(out):
    start = time.monotonic()
    rng = np.random.default_rng(20261030)
    mapping = d2.Mapping(n=8, H=1, intervals=4)
    size = mapping.nr * (2 * mapping.H + 1) * mapping.s
    x = rng.normal(size=size) * 1e-3
    q = d2.unpack(x, mapping.nr, mapping.H, mapping.s)
    op = d2.Temporal(mapping, q, 720., 53.)
    v, w = rng.normal(size=size), rng.normal(size=size)
    vc = v + 1j * w
    # All interior velocity rows, including the mean, carry 1/R.
    mass = op.b.reshape(mapping.nr, 3, mapping.s)
    expected = np.zeros_like(mass)
    for i in range(1, mapping.nr - 1):
        expected[i, :, mapping.rows] = 1 / op.p.radii[i]
    mass_error = float(np.max(abs(mass - expected)))
    zero_rows = np.flatnonzero(op.b == 0)
    require(mass_error == 0, 'temporal mass differs from 1/R on original interior velocity rows')
    require(np.count_nonzero((op.B @ vc)[zero_rows]) == 0, 'nonzero constraint-row mass')
    errors = []
    for h in (1e-3, 1e-4, 1e-5):
        errors.append(relative((op.residual(x + h * v) - op.residual(x - h * v)) / (2 * h), op.J @ v))
    require(max(errors) < 1e-9, 'fixed-parameter derivative inconsistency')
    cl = relative(op.J @ vc, op.J @ v + 1j * (op.J @ w))
    require(cl < 1e-12, 'complexification is not complex-linear')
    lift_error = relative(d2.lower(d2.lift(vc, mapping.nr, 1, mapping.s)), vc)
    require(lift_error < 1e-14, 'signed Fourier transform is not invertible')
    diagnostics = op.diagnostics()
    require(diagnostics['phase_equivariance_relative'] < 1e-10, 'symmetry tangent violates exact equivariance')
    # Real packed original residual is a separate implementation of the linear part.
    require(relative(op.linear_signed(op.q), d2.linear(op.blocks, op.q, 1)) < 1e-14,
            'signed operator differs from original real-packed solver')

    # Directly exercise complex-shift preconditioning with the ACTUAL Mapping
    # operator. q=0 makes the bare-block preconditioner the exact temporal inverse.
    zero = d2.Temporal(mapping, np.zeros_like(q), 720., 53.)
    sigma = .037 + .113j
    P, info = zero.preconditioner(sigma, d2.Deadline(30))
    solved = P @ vc
    inv_error = relative(zero.J @ solved + sigma * (zero.B @ solved), vc)
    inv_linear = relative(P @ vc, P @ v + 1j * (P @ w))
    require(inv_error < 1e-9 and inv_linear < 1e-11, 'nonreal shift Fourier inverse is wrong')

    # Independent dense generalized QZ solve of the actual reduced temporal pencil.
    eye = np.eye(size, dtype=complex)
    jd = np.column_stack([zero.J @ eye[:, i] for i in range(size)])
    bd = np.diag(zero.b)
    dense = linalg.eigvals(jd, -bd)
    finite = dense[np.isfinite(dense) & (abs(dense) < 1000)]
    center = d2.run_center(zero.J, zero.B, max(zero.b), sigma, out / 'actual_reduced_center',
                           'reduced-v1', d2.Deadline(60), zero.preconditioner,
                           k=4, ncv=14, outer_maxiter=40, v0=vc)
    accepted = [r for r in center['eigenpairs'] if r['accepted']]
    dense_distances = [float(np.min(abs(complex(*r['value']) - finite))) for r in accepted]
    require(center['complete'] and len(accepted) == 4, 'actual reduced shift-invert did not converge/certify')
    require(max(dense_distances) < 1e-6, 'actual reduced spectrum disagrees with independent QZ')
    return dict(size=size, n=8, H=1, intervals=4, seconds=time.monotonic() - start,
                mass_max_error=mass_error, constraint_zero_rows=len(zero_rows),
                derivative_relative=errors, complex_linearity_relative=cl,
                lift_roundtrip_relative=lift_error, symmetry=diagnostics,
                nonreal_sigma=d2.zpair(sigma), nonreal_inverse_relative=inv_error,
                nonreal_inverse_complex_linearity=inv_linear,
                sparse_factor_count=len(info), dense_finite_count=len(finite),
                accepted=center['eigenpairs'], dense_eigenvalue_distances=dense_distances)


def manufactured_checks(out):
    start = time.monotonic()
    rng = np.random.default_rng(18)
    # Real saddle-point pencil with nonidentity singular temporal mass.
    L = linalg.block_diag(np.diag([0., .03, -.2, -.7]),
                         np.array([[-.4, -.3], [.3, -.4]]),
                         np.array([[-1., -2.], [2., -1.]]))
    M = np.diag(np.linspace(.4, 1.1, 8))
    C, E = rng.normal(size=(8, 4)), rng.normal(size=(4, 8))
    jd = np.block([[C @ E - M @ L, C], [E, np.eye(4)]])
    bd = linalg.block_diag(M, np.zeros((4, 4)))
    J, B = (LinearOperator((12, 12), matvec=lambda x, a=a: a @ x, dtype=complex) for a in (jd, bd))
    dense = linalg.eigvals(jd, -bd)
    finite = dense[np.isfinite(dense)]
    require(len(finite) == 8, 'manufactured singular pencil finite/infinite count wrong')
    sigma = .02 + .05j
    calls = []

    def factory(shift, deadline):
        deadline.check()
        calls.append(shift)
        lu = linalg.lu_factor(jd + shift * bd)
        return LinearOperator((12, 12), matvec=lambda x: linalg.lu_solve(lu, x), dtype=complex), []

    center_dir = out / 'manufactured_center'
    center = d2.run_center(J, B, 1.1, sigma, center_dir, 'manufactured-v1',
                           d2.Deadline(20), factory, k=5, ncv=11)
    accepted = [r for r in center['eigenpairs'] if r['accepted']]
    distances = [float(np.min(abs(complex(*r['value']) - finite))) for r in accepted]
    require(center['complete'] and len(accepted) == 5 and max(distances) < 1e-9,
            'manufactured shift sign/scaling or original-pencil certification failed')
    call_count = len(calls)
    resumed = d2.run_center(J, B, 1.1, sigma, center_dir, 'manufactured-v1',
                            d2.Deadline(20), factory, k=5, ncv=11)
    require(resumed == center and len(calls) == call_count, 'completed checkpoint was recomputed')

    # An inaccurate solve claiming info=0 must fail ORIGINAL linear residual.
    bad_solver = lambda K, rhs, **kw: (np.zeros_like(rhs), 0)
    inverse = d2.ShiftInverse(J, B, sigma, None, d2.Deadline(20),
                              out / 'deliberately_bad_inner.jsonl', solver=bad_solver)
    rejected = False
    try:
        inverse.T @ np.ones(12)
    except d2.InnerFailure:
        rejected = True
    require(rejected, 'inaccurate inner solution escaped original-residual rejection')
    wrong = d2.certify(J, B, 1.1, .9, np.ones(12))
    constraint_only = d2.certify(J, B, 1.1, 0., np.r_[np.zeros(8), np.ones(4)])
    nonfinite = d2.certify(J, B, 1.1, complex(np.inf, np.nan), np.ones(12))
    json.dumps(nonfinite, allow_nan=False)
    require(not wrong['accepted'] and not constraint_only['finite_candidate'],
            'bad or singular-mass eigenpair was accepted')
    require(not nonfinite['accepted'], 'nonfinite eigenpair escaped rejection')

    failure_dir = out / 'recovered_center'
    def failure_factory(*args):
        raise RuntimeError('intentional failure to verify marker propagation')
    failed = d2.run_center(J, B, 1.1, sigma, failure_dir, 'recovery-v1',
                           d2.Deadline(20), failure_factory, k=5, ncv=11)
    require(not failed['complete'] and (failure_dir / 'FAILED.json').exists(), 'center failure marker missing')
    recovered = d2.run_center(J, B, 1.1, sigma, failure_dir, 'recovery-v1',
                              d2.Deadline(20), factory, k=5, ncv=11)
    require(recovered['complete'] and (failure_dir / 'DONE.json').exists()
            and not (failure_dir / 'FAILED.json').exists(), 'center resume recovery marker wrong')

    # Force an ARPACK-returned null transformed eigenvalue through the full
    # certification/write path: singular-mass modes must never produce DONE.
    real_eigs = d2.eigs
    try:
        d2.eigs = lambda *a, **kw: (np.zeros(1, complex), np.ones((12, 1), complex))
        rejected_center = d2.run_center(J, B, 1.1, sigma, out / 'rejected_center', 'rejected-v1',
                                       d2.Deadline(20), factory, k=1, ncv=4)
        require(not rejected_center['complete'] and rejected_center['rejected_count'] == 1
                and not (out / 'rejected_center/DONE.json').exists(), 'rejected center was marked DONE')
    finally:
        d2.eigs = real_eigs

    # Actual campaign launch-failure path, with tiny manufactured source files.
    # A nonexistent Python executable fails before any source preparation/work.
    source_dir = out / 'campaign_source'
    source_dir.mkdir(exist_ok=True)
    (source_dir / 'fields.npz').write_bytes(b'manufactured-source-hash-only')
    old_sources, old_python = d2.SOURCES, d2.sys.executable
    try:
        d2.SOURCES = {'manufactured': source_dir}
        d2.sys.executable = str(out / 'intentionally_missing_python')
        rc = d2.campaign(out / 'failed_campaign', [sigma], budget=5)
        require(rc == 1 and (out / 'failed_campaign/FAILED.json').exists()
                and (out / 'failed_campaign/manufactured/FAILED.json').exists(),
                'campaign/point failure propagation missing')
        exhausted = out / 'exhausted_campaign/manufactured'
        d2.write(exhausted / 'ACTIVE.json', dict(reserved_seconds=5.))
        rc2 = d2.campaign(out / 'exhausted_campaign', [sigma], budget=5)
        ledger = json.loads((exhausted / 'budget.json').read_text())
        require(rc2 == 1 and ledger['consumed_seconds'] == 5,
                'interrupted attempt improperly regained cumulative point budget')
    finally:
        d2.SOURCES, d2.sys.executable = old_sources, old_python
    merged = d2.deduplicate([dict(accepted[0], centers=[0]), dict(accepted[0], centers=[1])])
    require(len(merged) == 1 and merged[0]['centers'] == [0, 1], 'spectral deduplication failed')
    weak = [dict(accepted[0], value=[0., 0.], centers=[0], phase_mass_overlap=1.),
            dict(accepted[0], value=[1e-8, 0.], centers=[1], phase_mass_overlap=0.)]
    require(len(d2.deduplicate(weak)) == 2, 'weak growth mode was merged into phase neutral')
    return dict(size=12, finite_modes=8, infinite_modes=4, dense_eigenvalues=[d2.zpair(z) for z in finite],
                accepted=center['eigenpairs'], dense_eigenvalue_distances=distances,
                resumed_without_refactor=True, inaccurate_inner_rejected=rejected,
                wrong_eigenpair=wrong, zero_mass_eigenpair=constraint_only,
                nonfinite_eigenpair=nonfinite,
                center_recovered=True, campaign_failure_propagated=True,
                nonfinite_center_failed=True,
                interrupted_budget_conserved=True, deduplication=True,
                seconds=time.monotonic() - start)


def termination_check(out):
    """Real subprocess termination of a campaign-owned, bounded dummy worker."""
    harness = r'''
import json, os, pathlib, signal, subprocess, sys
sys.path.insert(0, str(pathlib.Path(sys.argv[1])/'examples'))
import fullbvp_phaseD2_stability as d2
folder=pathlib.Path(sys.argv[2])
source=folder/'source'
source.mkdir(parents=True,exist_ok=True)
(source/'fields.npz').write_bytes(b'bounded-dummy-only')
d2.SOURCES={'dummy':source}
real_popen=d2.subprocess.Popen
def spawn_dummy(cmd, **kw):
    proc=real_popen([sys.executable,'-B','-c','import signal; signal.alarm(30); signal.pause()'],**kw)
    print(json.dumps({'worker_pid':proc.pid}),flush=True)
    return proc
d2.subprocess.Popen=spawn_dummy
raise SystemExit(d2.campaign(folder/'run',[.02+.05j],budget=20))
'''
    proc = subprocess.Popen([sys.executable, '-B', '-c', harness, str(d2.ROOT), str(out / 'termination')],
                            stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True,
                            env={**os.environ, 'PYTHONDONTWRITEBYTECODE': '1'}, start_new_session=True)
    try:
        ready = json.loads(proc.stdout.readline())
        proc.send_signal(signal.SIGTERM)
        stdout, stderr = proc.communicate(timeout=5)
        require(proc.returncode == 143, f'campaign signal exit wrong: {proc.returncode}, {stderr}')
        folder = out / 'termination/run'
        require((folder / 'FAILED.json').exists() and (folder / 'dummy/FAILED.json').exists(),
                'termination failure markers missing')
        require(not Path(f'/proc/{ready["worker_pid"]}').exists(), 'owned worker was not killed and reaped')
        budget = json.loads((folder / 'dummy/budget.json').read_text())
        require(budget['consumed_seconds'] > 0, 'termination lost budget accounting')
        return dict(passed=True, exitcode=proc.returncode, owned_worker_killed_and_reaped=True,
                    campaign_and_point_failed=True, budget_accounted=budget['consumed_seconds'])
    finally:
        if proc.returncode is None:
            os.killpg(proc.pid, signal.SIGKILL)
            proc.wait()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--out', type=Path, required=True)
    parser.add_argument('--check-full-sources', action='store_true')
    args = parser.parse_args()
    require(args.out.resolve().is_relative_to(d2.OUT.resolve()), 'output must be in D2')
    require(not (args.out / 'VALIDATION.json').exists(), 'use a new validation directory')
    args.out.mkdir(parents=True, exist_ok=True)
    start = time.monotonic()
    result = dict(passed=False, production_spectra_computed=False,
                  thresholds=dict(derivative=1e-9, complex_linearity=1e-12,
                                  actual_nonreal_inverse=1e-9, actual_QZ_distance=1e-6,
                                  manufactured_QZ_distance=1e-9))
    try:
        result['reduced'] = reduced_checks(args.out)
        result['manufactured'] = manufactured_checks(args.out)
        result['termination'] = termination_check(args.out)
        result['full_sources'] = []
        if args.check_full_sources:
            for name in d2.SOURCES:
                t0 = time.monotonic()
                operator, metadata = d2.load_source(name, d2.Deadline(180))
                metadata['validation_seconds'] = time.monotonic() - t0
                result['full_sources'].append(metadata)
                del operator
                gc.collect()
        result['passed'] = True
    except Exception as exc:
        result['failure'] = f'{type(exc).__name__}: {exc}'
        raise
    finally:
        result['seconds'] = time.monotonic() - start
        d2.write(args.out / 'VALIDATION.json', result)
    print(json.dumps(dict(passed=result['passed'], seconds=result['seconds'],
                          reduced_size=result['reduced']['size'], full_sources=len(result['full_sources']))))


if __name__ == '__main__':
    main()
