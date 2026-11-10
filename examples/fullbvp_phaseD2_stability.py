#!/usr/bin/env python3
"""Fixed-parameter temporal pencil and resumable, sampled shift-invert spectra.

No continuation borders enter this operator. Production is deliberately opt-in.
The complex vector space is the complexification of the real Fourier packing.
"""
from __future__ import annotations

import os
for _name in ('OPENBLAS_NUM_THREADS', 'OMP_NUM_THREADS', 'MKL_NUM_THREADS',
              'NUMEXPR_NUM_THREADS', 'VECLIB_MAXIMUM_THREADS', 'BLIS_NUM_THREADS'):
    os.environ[_name] = '1'

import argparse
import gc
import hashlib
import json
import signal
import subprocess
import sys
import time
import traceback
from pathlib import Path

import numpy as np
from scipy import sparse
from scipy.sparse.linalg import LinearOperator, gmres, eigs, ArpackNoConvergence

from fullbvp_phaseB_solver import Mapping, ROOT, pack, unpack, linear, convection
from appelquist2015_hopf_block_corrector import Factor

OUT = ROOT / 'results/fullbvp_global_20261030/phaseD/D2'
SOURCES = {
    'upper720_H6': ROOT / 'results/fullbvp_global_20261030/phaseC/attempts/harmonic_audit_fc1397fc2cb12850_try0',
    'upper725_H6': ROOT / 'results/fullbvp_global_20261030/phaseC/attempts/harmonic_audit_be43e3a56f551a7f_try0',
    'lower719p948669_H4': ROOT / 'results/fullbvp_global_20261030/phaseB/attempts/refinement_H4',
}
DEFAULT_SHIFTS = ('0.002', '0.02', '-0.02', '0.002+0.1j', '0.002+1j', '0.002+5j')


def write(path, obj):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + '.tmp')
    tmp.write_text(json.dumps(obj, indent=2, allow_nan=False) + '\n')
    tmp.replace(path)


def save(path, **arrays):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + '.tmp')
    with tmp.open('wb') as f:
        np.savez(f, **arrays)
    tmp.replace(path)


def sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b''):
            h.update(chunk)
    return h.hexdigest()


def digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True).encode()).hexdigest()


def zpair(z):
    return [float(np.real(z)), float(np.imag(z))]


def lift(x, nr, H, s):
    """Complex-linear lift: negative modes use a-i*b, NEVER conjugation."""
    a = np.asarray(x).reshape(nr, 2 * H + 1, s)
    q = np.empty_like(a, dtype=complex)
    q[:, H] = a[:, 0]
    for j in range(1, H + 1):
        q[:, H + j] = a[:, 2 * j - 1] + 1j * a[:, 2 * j]
        q[:, H - j] = a[:, 2 * j - 1] - 1j * a[:, 2 * j]
    return q


def lower(q):
    H = (q.shape[1] - 1) // 2
    a = np.empty_like(q, dtype=complex)
    a[:, 0] = q[:, H]
    for j in range(1, H + 1):
        a[:, 2 * j - 1] = (q[:, H + j] + q[:, H - j]) / 2
        a[:, 2 * j] = (q[:, H + j] - q[:, H - j]) / (2j)
    return a.ravel()


class BudgetExpired(RuntimeError):
    pass


class Deadline:
    def __init__(self, seconds):
        self.end = time.monotonic() + seconds

    def check(self):
        if time.monotonic() >= self.end:
            raise BudgetExpired('total point or center time budget exhausted')

    def remaining(self):
        return max(0., self.end - time.monotonic())


class Temporal:
    """Analytic field-only J=dF/dq at fixed Rout and omega; B is unscaled."""
    def __init__(self, mapping, q, Rout, omega, deadline=None):
        self.map, self.q = mapping, np.asarray(q, complex)
        self.p, self.blocks, bd, _ = mapping.at(Rout, omega)
        self.nr, self.H, self.s = mapping.nr, mapping.H, mapping.s
        self.shape = self.q.shape
        self.size = self.q.size
        self.deadline = deadline
        self.b = np.broadcast_to(bd[:, None, :], self.shape).copy().ravel()
        self.B = LinearOperator((self.size, self.size), matvec=lambda x: self.b * x,
                                dtype=complex)
        self.J = LinearOperator((self.size, self.size), matvec=self.jv, dtype=complex)
        self.q4 = self.q.reshape(self.nr, 2 * self.H + 1, 4, mapping.m)
        self.qr = (self.p.D @ self.q.reshape(self.nr, -1)).reshape(self.q4.shape)
        self.qz = self.q4[:, :, :3] @ mapping.op.grid.D.T
        self.ri = (1 / self.p.radii)[:, None, None]

    def check(self):
        if self.deadline:
            self.deadline.check()

    def linear_signed(self, v):
        out = np.empty_like(v)
        for j, A in enumerate(self.blocks):
            out[:, self.H + j] = (A @ v[:, self.H + j].ravel()).reshape(self.nr, self.s)
            if j:
                out[:, self.H - j] = (A.conjugate() @ v[:, self.H - j].ravel()).reshape(self.nr, self.s)
        return out

    def nonlinear_derivative(self, v):
        """C(q,v)+C(v,q), using the exact original truncated convolution."""
        H, m = self.H, self.map.m
        d = v.reshape(self.q4.shape)
        dr = (self.p.D @ v.reshape(self.nr, -1)).reshape(d.shape)
        dz = d[:, :, :3] @ self.map.op.grid.D.T
        out = np.zeros_like(d)
        for j in range(-H, H + 1):
            for k in range(max(-H, j - H), min(H, j + H) + 1):
                ia, ib = j - k + H, k + H
                a, b = self.q4[:, ia], self.q4[:, ib]
                da, db = d[:, ia], d[:, ib]
                out[:, j + H, :3] += (
                    da[:, 0, None] * (self.qr[:, ib, :3] + b[:, :3] * self.ri)
                    + a[:, 0, None] * (dr[:, ib, :3] + db[:, :3] * self.ri)
                    + 1j * k * 68 * self.ri * (da[:, 1, None] * b[:, :3] + a[:, 1, None] * db[:, :3])
                    + da[:, 2, None] * self.qz[:, ib] + a[:, 2, None] * dz[:, ib])
                out[:, j + H, 0] -= (da[:, 1] * b[:, 1] + a[:, 1] * db[:, 1]) * self.ri[:, 0]
                out[:, j + H, 1] += (da[:, 0] * b[:, 1] + a[:, 0] * db[:, 1]) * self.ri[:, 0]
        out = out.reshape(self.shape)
        out[:, :, self.map.op.boundary_rows] = 0
        out[0][:, self.map.rows] = 0
        out[-1][:, self.map.rows] = 0
        return out

    def jv(self, x):
        self.check()
        v = lift(x, self.nr, self.H, self.s)
        return lower(self.linear_signed(v) + self.nonlinear_derivative(v))

    def residual(self, x=None):
        q = self.q if x is None else unpack(np.asarray(x).real, self.nr, self.H, self.s)
        return pack(linear(self.blocks, q, self.H) + convection(self.p, q))

    def neutral(self):
        q = self.q * (1j * np.arange(-self.H, self.H + 1))[None, :, None]
        return lower(q).real

    def diagnostics(self):
        x, f, phase = pack(self.q), self.residual(), self.neutral()
        jp, bp = self.J @ phase, self.B @ phase
        # Exact equivariance identity holds even away from a stationary state.
        sf = lower(lift(f, self.nr, self.H, self.s) *
                   (1j * np.arange(-self.H, self.H + 1))[None, :, None])
        return dict(size=self.size, n=self.map.n, H=self.H, intervals=self.map.intervals,
                    Rout=float(self.p.radii[-1]), omega=float(self.p.config.omega),
                    relative_field_inf=float(np.max(abs(f)) / max(np.max(abs(x)), 1e-300)),
                    residual_l2=float(np.linalg.norm(f)),
                    mass_nonzero=int(np.count_nonzero(self.b)), mass_max=float(max(self.b)),
                    phase_norm=float(np.linalg.norm(phase)),
                    phase_neutral_rate_residual=float(np.linalg.norm(jp) / max(np.linalg.norm(bp), 1e-300)),
                    phase_equivariance_absolute=float(np.linalg.norm(jp - sf)),
                    phase_equivariance_relative=float(np.linalg.norm(jp - sf) / max(np.linalg.norm(phase), 1e-300)),
                    symmetry='q_j -> exp(i*j*theta)*q_j; no phase border or deflation')

    def preconditioner(self, sigma, deadline=None):
        """Invert the bare harmonic blocks; nonlinear coupling remains in J.

        Complex sigma requires a DISTINCT conjugate-A factor on negative modes.
        Only real sigma permits conjugating the positive-mode solve.
        """
        M = sparse.diags(self.b.reshape(self.shape)[:, 0].ravel(), format='csc')
        plus, minus, infos = {}, {}, []
        for j, A in enumerate(self.blocks):
            if deadline:
                deadline.check()
            plus[j] = Factor((A + sigma * M).astype(complex), f'plus{j}')
            infos.append(plus[j].info)
            if j and sigma.imag != 0:
                minus[j] = Factor((A.conjugate() + sigma * M).astype(complex), f'minus{j}')
                infos.append(minus[j].info)

        def solve(x):
            if deadline:
                deadline.check()
            q, out = lift(x, self.nr, self.H, self.s), np.empty(self.shape, complex)
            for j in range(self.H + 1):
                out[:, self.H + j] = plus[j].solve(q[:, self.H + j].ravel()).reshape(self.nr, self.s)
                if j:
                    rhs = q[:, self.H - j].ravel()
                    z = minus[j].solve(rhs) if sigma.imag != 0 else plus[j].solve(rhs.conjugate()).conjugate()
                    out[:, self.H - j] = z.reshape(self.nr, self.s)
            return lower(out)
        return LinearOperator(self.J.shape, matvec=solve, dtype=complex), infos


class InnerFailure(RuntimeError):
    pass


class ShiftInverse:
    """T=-[J+sigma B]^-1 B; lambda=sigma+1/mu. Check every inner solve."""
    def __init__(self, J, B, sigma, preconditioner, deadline, log_path,
                 rtol=1e-9, accept=5e-8, restart=35, maxiter=5, solver=gmres):
        self.J, self.B, self.sigma, self.P = J, B, complex(sigma), preconditioner
        self.deadline, self.log_path = deadline, Path(log_path)
        self.rtol, self.accept = rtol, accept
        self.restart, self.maxiter, self.solver = restart, maxiter, solver
        self.calls = 0
        self.K = LinearOperator(J.shape, matvec=lambda v: J @ v + self.sigma * (B @ v), dtype=complex)
        self.T = LinearOperator(J.shape, matvec=self.apply, dtype=complex)

    def apply(self, x):
        self.deadline.check()
        rhs = -(self.B @ x)
        history = []
        start = time.monotonic()
        y, info = self.solver(self.K, rhs, M=self.P, rtol=self.rtol, atol=0.,
                              restart=self.restart, maxiter=self.maxiter,
                              callback=lambda r: history.append(float(r)), callback_type='pr_norm')
        rel = float(np.linalg.norm(self.K @ y - rhs) / max(np.linalg.norm(rhs), 1e-300))
        self.calls += 1
        self.log_path.parent.mkdir(parents=True, exist_ok=True)
        with self.log_path.open('a') as f:
            f.write(json.dumps(dict(call=self.calls, sigma=zpair(self.sigma), info=int(info),
                                    original_relative=rel, seconds=time.monotonic() - start,
                                    krylov_history=history)) + '\n')
        if not np.isfinite(rel) or rel > self.accept:
            raise InnerFailure(f'inner original residual {rel:.3e} > {self.accept:.3e}; info={info}')
        self.deadline.check()
        return y


def certify(J, B, bnorm, value, vector, tol=1e-7, max_abs=1000., phase=None):
    if not np.isfinite(value) or not np.all(np.isfinite(vector)):
        return dict(value=[None, None], raw_value=str(value), original_relative=None,
                    mass_participation=None, finite_candidate=False, accepted=False,
                    classification='nonfinite_eigenvalue_or_vector')
    jv, bv = J @ vector, B @ vector
    nv, nb = np.linalg.norm(vector), np.linalg.norm(bv)
    den = max(np.linalg.norm(jv) + max(abs(value), 1.) * nb, 1e-300)
    residual = float(np.linalg.norm(jv + value * bv) / den)
    participation = float(nb / max(bnorm * nv, 1e-300))
    finite = bool(np.isfinite(value) and abs(value) < max_abs and participation > 1e-10)
    result = dict(value=zpair(value), original_relative=residual, mass_participation=participation,
                finite_candidate=finite, accepted=bool(finite and residual <= tol),
                residual_scale='||Jv|| + max(|lambda|,1)*||Bv||',
                classification='finite_original_pencil_candidate' if finite else 'singular_mass_or_outside_cutoff')
    if phase is not None:
        phase_mass = max(float(np.vdot(phase, B @ phase).real), 0.)
        vector_mass = max(float(np.vdot(vector, bv).real), 0.)
        result['phase_mass_overlap'] = float(abs(np.vdot(phase, bv)) /
                                              max(np.sqrt(phase_mass * vector_mass), 1e-300))
    return result


def deduplicate(records, tol=2e-7):
    unique = []
    for rec in sorted(records, key=lambda r: (r['original_relative'], r['value'])):
        z = complex(*rec['value'])
        # Preserve distinct weak growth/phase modes. Near zero, use a much tighter
        # threshold; different phase overlaps or same-center Ritz pairs stay separate.
        threshold = min(tol * (1 + abs(z)), 1e-10) if abs(z) < 1e-4 else tol * (1 + abs(z))
        match = next((u for u in unique
                      if not set(u['centers']).intersection(rec['centers'])
                      and abs(z - complex(*u['value'])) <= threshold
                      and abs(u.get('phase_mass_overlap', 0.) - rec.get('phase_mass_overlap', 0.)) <= .1), None)
        if match:
            match['centers'] = sorted(set(match['centers'] + rec['centers']))
        else:
            unique.append(dict(rec))
    return sorted(unique, key=lambda r: (-r['value'][0], r['value'][1]))


def run_center(J, B, bnorm, sigma, out, fingerprint, deadline, preconditioner_factory,
               k=6, ncv=22, outer_tol=1e-8, outer_maxiter=80, v0=None, phase=None):
    out = Path(out)
    out.mkdir(parents=True, exist_ok=True)
    existing = out / 'result.json'
    if existing.exists():
        old = json.loads(existing.read_text())
        if old['fingerprint'] != fingerprint:
            raise RuntimeError(f'checkpoint fingerprint mismatch at {out}')
        if old['complete']:
            return old
    if (out / 'eigenpairs.npz').exists():
        with np.load(out / 'eigenpairs.npz') as d:
            if d['vectors'].shape[1]:
                v0 = d['vectors'][:, 0].copy()
    # Preserve all earlier partial spectra and certificates before retrying.
    if existing.exists():
        attempt = 0
        while (out / f'archive_attempt_{attempt:03d}').exists():
            attempt += 1
        archive = out / f'archive_attempt_{attempt:03d}'
        archive.mkdir()
        existing.replace(archive / existing.name)
        if (out / 'eigenpairs.npz').exists():
            (out / 'eigenpairs.npz').replace(archive / 'eigenpairs.npz')
    start, values, vectors = time.monotonic(), np.array([], complex), np.empty((J.shape[0], 0), complex)
    result = dict(fingerprint=fingerprint, sigma=zpair(sigma), complete=False,
                  shift_transform='T=-(J+sigma B)^-1 B; lambda=sigma+1/mu',
                  k=k, ncv=ncv, outer_tol=outer_tol, outer_maxiter=outer_maxiter,
                  coverage='local nearest-to-shift sample, no exhaustive half-plane coverage')
    try:
        deadline.check()
        P, result['preconditioner'] = preconditioner_factory(sigma, deadline)
        inverse = ShiftInverse(J, B, sigma, P, deadline, out / 'inner_solves.jsonl')
        try:
            mu, vectors = eigs(inverse.T, k=k, which='LM', ncv=ncv, tol=outer_tol,
                               maxiter=outer_maxiter, v0=v0)
            result['complete'] = True
        except ArpackNoConvergence as exc:
            mu, vectors = exc.eigenvalues, exc.eigenvectors
            result['failure'] = str(exc)
        with np.errstate(divide='ignore', invalid='ignore'):
            values = sigma + 1 / mu
        result['inner_calls'] = inverse.calls
    except Exception as exc:
        result['failure'] = f'{type(exc).__name__}: {exc}'
    finally:
        # Certification is mandatory, even for partially converged Ritz pairs.
        # Deadline does not bypass certification: after expiry no pairs are accepted.
        result['eigenpairs'] = []
        for val, vec in zip(values, vectors.T):
            try:
                deadline.check()
                result['eigenpairs'].append(certify(J, B, bnorm, val, vec, phase=phase))
            except Exception as exc:
                result['complete'] = False
                result['failure'] = f'certification: {type(exc).__name__}: {exc}'
                break
        result['accepted_count'] = sum(r['accepted'] for r in result['eigenpairs'])
        result['rejected_count'] = len(result['eigenpairs']) - result['accepted_count']
        if result['accepted_count'] < k or result['rejected_count']:
            result['complete'] = False
            result.setdefault('failure', 'fewer than k certified finite candidates, or a candidate was rejected')
        result['seconds'] = time.monotonic() - start
        save(out / 'eigenpairs.npz', values=values, vectors=vectors)
        write(existing, result)
        write(out / ('DONE.json' if result['complete'] else 'FAILED.json'), result)
        if result['complete']:
            (out / 'FAILED.json').unlink(missing_ok=True)
    return result


def load_source(name, deadline=None):
    source = SOURCES[name]
    fields, summary = source / 'fields.npz', source / 'summary.json'
    report = json.loads(summary.read_text())
    if not report.get('accepted'):
        raise RuntimeError('source summary is not accepted')
    with np.load(fields) as d:
        if not bool(d['accepted']):
            raise RuntimeError('source checkpoint is not accepted')
        q, Rout, omega = d['field'].copy(), float(d['Rout']), float(d['omega'])
        n, H = int(d['n']), int(d['H'])
        radii = d['radii'].copy()
    intervals = len(radii) - 1
    if n != 32 or intervals != 800 or (name.startswith('upper') and H != 6):
        raise RuntimeError('source truncation mismatch')
    if name.startswith('upper') and Rout != float(name[5:8]):
        raise RuntimeError('source Rout mismatch')
    expected = np.linspace(400., Rout, intervals + 1)
    if not np.allclose(radii, expected, rtol=0., atol=1e-10):
        raise RuntimeError('source does not use Mapping grid')
    mapping = Mapping(n, H, intervals)
    temporal = Temporal(mapping, q, Rout, omega, deadline)
    diagnostics = temporal.diagnostics()
    metadata = dict(name=name, fields=str(fields), fields_sha256=sha(fields),
                    summary=str(summary), summary_sha256=sha(summary), source_accepted=True,
                    acceptance_evidence={k: report.get(k) for k in
                                         ('relative_field_inf', 'residual_over_fundamental_feedback', 'independent')},
                    lower_control_resolution_limitation=(H != 6), diagnostics=diagnostics)
    if diagnostics['relative_field_inf'] > 1e-10 or diagnostics['phase_neutral_rate_residual'] > 1e-6:
        raise RuntimeError('source stationarity or neutral-mode check failed: ' + str(diagnostics))
    return temporal, metadata


def code_fingerprints():
    paths = [Path(__file__), ROOT / 'examples/fullbvp_phaseB_solver.py',
             ROOT / 'examples/appelquist2016_branch_continuation.py',
             ROOT / 'examples/appelquist2015_hopf_block_corrector.py',
             ROOT / 'examples/appelquist2015_hopf_common.py',
             ROOT / 'src/llsa/secondary/global_radial.py', ROOT / 'src/llsa/secondary/radial.py']
    return {str(p): sha(p) for p in paths}


def point(name, out, budget, shifts, center_seconds=700.):
    out = Path(out)
    out.mkdir(parents=True, exist_ok=True)
    start = time.monotonic()
    deadline = Deadline(budget)
    try:
        temporal, source = load_source(name, deadline)
        config = dict(source=source, code_sha256=code_fingerprints(), shifts=[zpair(s) for s in shifts],
                      k=6, ncv=22, inner_rtol=1e-9, inner_accept=5e-8, eigen_accept=1e-7,
                      finite_lambda_cutoff=1000., dedup_tol=2e-7,
                      near_zero_dedup_absolute=1e-10, center_seconds=center_seconds)
        fingerprint = digest(config)
        manifest = out / 'manifest.json'
        if manifest.exists() and json.loads(manifest.read_text())['fingerprint'] != fingerprint:
            raise RuntimeError('existing point fingerprint differs; choose a new run directory')
        write(manifest, dict(fingerprint=fingerprint, **config))
        write(out / 'source.json', source)
        rng = np.random.default_rng(20261030)
        v0 = rng.normal(size=temporal.size).astype(complex)
        phase = temporal.neutral()
        v0 /= np.linalg.norm(v0)
        v0 += phase / max(np.linalg.norm(phase), 1e-300)
        results, records = [], []
        for i, sigma in enumerate(shifts):
            deadline.check()
            local = Deadline(min(center_seconds, deadline.remaining()))
            # Temporal checks the point deadline; inner/preconditioner checks local too.
            temporal.deadline = local
            result = run_center(temporal.J, temporal.B, max(temporal.b), sigma,
                                out / f'center_{i:02d}', fingerprint, local, temporal.preconditioner,
                                v0=v0, phase=phase)
            results.append(dict(center=i, sigma=zpair(sigma), complete=result['complete'],
                                failure=result.get('failure')))
            for rec in result['eigenpairs']:
                if rec['accepted']:
                    records.append(dict(rec, centers=[i]))
            write(out / 'merged.json', dict(eigenpairs=deduplicate(records), centers=results,
                                          exhaustive_stability=False, symmetry_mode_retained=True))
            gc.collect()
        complete = all(r['complete'] for r in results) and bool(records)
        result = dict(complete=complete, seconds=time.monotonic() - start,
                      fingerprint=fingerprint, source=name, centers=results,
                      accepted_unique=len(deduplicate(records)), exhaustive_stability=False,
                      interpretation='finite local spectral samples only; decay is not established by incomplete coverage',
                      perturbations='retained integer beta=68 harmonics only; no subharmonics or other azimuthal families')
        write(out / ('DONE.json' if complete else 'FAILED.json'), result)
        if complete:
            (out / 'FAILED.json').unlink(missing_ok=True)
        return 0 if complete else 1
    except Exception as exc:
        write(out / 'FAILED.json', dict(complete=False, seconds=time.monotonic() - start,
                                       failure=f'{type(exc).__name__}: {exc}', traceback=traceback.format_exc()))
        return 1


def campaign(out, shifts, budget=4800.):
    """Serial points; cumulative per-point budget persists across all resumes."""
    out = Path(out)
    out.mkdir(parents=True, exist_ok=True)
    active_proc, active_folder = None, None

    def terminate_owned_worker(signum, frame):
        # Each worker has its own process group. Never inspect or signal other jobs.
        if active_proc is not None:
            try:
                os.killpg(active_proc.pid, signal.SIGKILL)
                active_proc.wait()
            except ProcessLookupError:
                pass
        failure = dict(complete=False, failure=f'campaign terminated by signal {signum}')
        if active_folder is not None:
            write(active_folder / 'FAILED.json', failure)
        write(out / 'FAILED.json', failure)
        raise SystemExit(128 + signum)

    signal.signal(signal.SIGTERM, terminate_owned_worker)
    signal.signal(signal.SIGINT, terminate_owned_worker)
    config = dict(code_sha256=code_fingerprints(), shifts=[zpair(s) for s in shifts], budget=budget,
                  source_hashes={n: sha(p / 'fields.npz') for n, p in SOURCES.items()})
    fp, manifest = digest(config), out / 'campaign_manifest.json'
    if manifest.exists() and json.loads(manifest.read_text())['fingerprint'] != fp:
        raise RuntimeError('campaign source/code/config changed; use a new output directory')
    write(manifest, dict(fingerprint=fp, **config))
    statuses = []
    for name in SOURCES:
        folder = out / name
        folder.mkdir(parents=True, exist_ok=True)
        if (folder / 'DONE.json').exists():
            statuses.append(dict(source=name, returncode=0, resumed_completed=True))
            continue
        account = folder / 'budget.json'
        used = json.loads(account.read_text())['consumed_seconds'] if account.exists() else 0.
        # An interrupted supervisor process leaves an active attempt. Conservatively
        # charge its entire reservation: resumption can never grant extra budget.
        active = folder / 'ACTIVE.json'
        if active.exists():
            used += json.loads(active.read_text())['reserved_seconds']
            write(account, dict(consumed_seconds=min(budget, used), interrupted_attempt_charged=True))
            active.unlink()
        remaining = budget - used
        if remaining <= 1:
            write(folder / 'FAILED.json', dict(failure='cumulative point budget exhausted', consumed_seconds=used))
            statuses.append(dict(source=name, returncode=1, reason='budget exhausted'))
            continue
        write(active, dict(reserved_seconds=remaining, started_unix=time.time()))
        start = time.monotonic()
        active_folder = folder
        cmd = [sys.executable, '-B', str(Path(__file__).resolve()), '--point', name,
               '--out', str(folder), '--budget', str(max(.1, remaining - 3)),
               '--shifts', *[str(s) for s in shifts]]
        rc = 1
        try:
            with (folder / 'run.log').open('a') as log:
                try:
                    os.fstat(9)
                    inherited_lock = (9,)
                except OSError:
                    inherited_lock = ()
                proc = subprocess.Popen(cmd, stdout=log, stderr=subprocess.STDOUT,
                                        env={**os.environ, 'PYTHONDONTWRITEBYTECODE': '1'},
                                        start_new_session=True, pass_fds=inherited_lock)
                active_proc = proc
                try:
                    rc = proc.wait(timeout=remaining)
                except subprocess.TimeoutExpired:
                    os.killpg(proc.pid, signal.SIGKILL)
                    proc.wait()
                    write(folder / 'FAILED.json', dict(failure='hard cumulative wall-time limit', budget=budget))
        except Exception as exc:
            write(folder / 'FAILED.json', dict(failure=f'launch: {type(exc).__name__}: {exc}'))
        finally:
            if active_proc is not None and active_proc.returncode is None:
                try:
                    os.killpg(active_proc.pid, signal.SIGKILL)
                    active_proc.wait()
                except ProcessLookupError:
                    pass
            consumed = min(budget, used + time.monotonic() - start)
            write(account, dict(consumed_seconds=consumed, limit_seconds=budget))
            active.unlink(missing_ok=True)
            active_proc, active_folder = None, None
        if rc != 0 or not (folder / 'DONE.json').exists():
            rc = rc or 1
            if not (folder / 'FAILED.json').exists():
                write(folder / 'FAILED.json', dict(failure='worker exited without DONE', returncode=rc))
        statuses.append(dict(source=name, returncode=rc))
    complete = all(s['returncode'] == 0 for s in statuses)
    write(out / ('DONE.json' if complete else 'FAILED.json'), dict(complete=complete, points=statuses,
                                                               exhaustive_stability=False))
    if complete:
        (out / 'FAILED.json').unlink(missing_ok=True)
    return 0 if complete else 1


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--point', choices=list(SOURCES))
    p.add_argument('--campaign', action='store_true')
    p.add_argument('--out', type=Path, required=True)
    p.add_argument('--budget', type=float, default=4800.)
    p.add_argument('--shifts', nargs='+', default=DEFAULT_SHIFTS)
    args = p.parse_args()
    shifts = [complex(s) for s in args.shifts]
    if not args.campaign and not args.point:
        p.error('select --campaign or --point')
    if not args.out.resolve().is_relative_to(OUT.resolve()):
        p.error('outputs must be confined to Phase D2')
    if args.budget > 4800 or args.budget <= 0:
        p.error('point/campaign budget must be in (0,4800] seconds')
    return campaign(args.out, shifts, args.budget) if args.campaign else point(args.point, args.out, args.budget, shifts)


if __name__ == '__main__':
    sys.exit(main())
