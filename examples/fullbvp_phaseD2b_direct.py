#!/usr/bin/env python3
"""D2b: full coupled fixed-parameter sparse temporal pencil, direct complex LU."""
from __future__ import annotations
import os
for name in ('OPENBLAS_NUM_THREADS','OMP_NUM_THREADS','MKL_NUM_THREADS',
             'NUMEXPR_NUM_THREADS','VECLIB_MAXIMUM_THREADS','BLIS_NUM_THREADS'):
    os.environ[name] = '1'
os.environ['PYTHONDONTWRITEBYTECODE'] = '1'

import argparse
import ctypes
import gc
import json
import resource
import signal
import subprocess
import sys
import time
import traceback
from pathlib import Path
import numpy as np
from scipy import sparse
from scipy.sparse.linalg import LinearOperator, splu, eigs, ArpackNoConvergence
import fullbvp_phaseD2_stability as v1
from llsa.secondary.radial import convection_blocks, real_blocks

ROOT, OUT = v1.ROOT, v1.OUT
IMAG_CENTERS = (0., 2., -2., 5., -5., 10., -10., 15., -15., 20., -20., 26., -26.)


def memory():
    stats = {'process_peak_rss_bytes': int(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss * 1024)}
    for line in Path('/proc/self/status').read_text().splitlines():
        if line.startswith(('VmRSS:', 'VmHWM:', 'Threads:')):
            key, val = line.split(':', 1)
            stats[key] = int(val.split()[0]) * (1024 if key != 'Threads' else 1)
    for line in Path('/proc/meminfo').read_text().splitlines():
        if line.startswith('MemAvailable:'):
            stats['system_mem_available_bytes'] = int(line.split()[1]) * 1024
    return stats


def sparse_bytes(A):
    return int(A.data.nbytes + A.indices.nbytes + A.indptr.nbytes)


def assemble(temporal, deadline):
    """Mapping linear blocks + exact existing local Frechet blocks, no borders."""
    start = time.monotonic()
    nr, H, s = temporal.nr, temporal.H, temporal.s
    width = (2 * H + 1) * s
    # Harmonic-major real representation, then radius-major packing permutation.
    components = [temporal.blocks[0].real.tocsr()]
    for A in temporal.blocks[1:]:
        deadline.check()
        components.append(sparse.bmat([[A.real, -A.imag], [A.imag, A.real]], format='csr'))
    harmonic = sparse.block_diag(components, format='csr')
    permutation = np.arange(temporal.size).reshape(nr, 2 * H + 1, s).transpose(1, 0, 2).ravel()
    inverse = np.argsort(permutation)
    L = harmonic[inverse][:, inverse].tocsr()
    del harmonic, components
    qr = temporal.qr.reshape(temporal.shape)
    local, derivative = [], []
    for i, op in enumerate(temporal.p.ops):
        deadline.check()
        if i in (0, nr - 1):
            local.append(sparse.csr_matrix((width, width)))
            derivative.append(sparse.csr_matrix((width, width)))
        else:
            blocks = {(j, k): convection_blocks(op, temporal.q[i], qr[i], j, k)
                      for j in range(H + 1) for k in range(-H, H + 1)}
            local.append(sparse.csr_matrix(real_blocks(lambda j, k: blocks[j, k][0], H, s)))
            derivative.append(sparse.csr_matrix(real_blocks(lambda j, k: blocks[j, k][1], H, s)))
    Q = sparse.block_diag(local, format='csr')
    T = sparse.block_diag(derivative, format='csr')
    del local, derivative
    D = sparse.kron(temporal.p.D, sparse.eye(width, format='csr'), format='csr')
    J = (L + Q + T @ D).tocsr()
    J.sum_duplicates()
    J.eliminate_zeros()
    J.sort_indices()
    deadline.check()
    rng = np.random.default_rng(20261030)
    probe = rng.normal(size=temporal.size) + 1j * rng.normal(size=temporal.size)
    reference = temporal.J @ probe
    error = float(np.linalg.norm(J @ probe - reference) / max(np.linalg.norm(reference), 1e-300))
    if error > 1e-10:
        raise RuntimeError(f'full coupled assembly disagrees with matrix-free J: {error:.3e}')
    stats = dict(shape=list(J.shape), matrix_nnz=int(J.nnz), matrix_storage_bytes=sparse_bytes(J),
                 linear_nnz=int(L.nnz), local_nonlinear_nnz=int(Q.nnz),
                 radial_derivative_coefficient_nnz=int(T.nnz),
                 assembly_seconds=time.monotonic() - start, complex_action_relative=error,
                 memory=memory(), representation='full coupled real-packed field Jacobian, fixed Rout/omega')
    return J, stats


class PardisoComplex:
    """Native MKL complex nonsymmetric PARDISO; pypardiso supplies library discovery."""
    def __init__(self, matrix, threads=16, deadline=None, stats_path=None):
        import pypardiso
        wrapper = pypardiso.PyPardisoSolver()
        self.lib = wrapper.libmkl
        self.deadline, self.stats_path = deadline, stats_path
        self.closed = False
        self.pt = np.zeros(64, dtype=np.uintp)
        # Prefer explicit 64-bit API; every integer argument and CSR array follows it.
        self.bits = 64 if hasattr(self.lib, 'pardiso_64') else 32
        self.integer = ctypes.c_int64 if self.bits == 64 else ctypes.c_int32
        self.npdtype = np.int64 if self.bits == 64 else np.int32
        self.fun = getattr(self.lib, 'pardiso_64' if self.bits == 64 else 'pardiso')
        P = ctypes.POINTER(self.integer)
        self.fun.argtypes = [ctypes.c_void_p, P, P, P, P, P, ctypes.c_void_p,
                             P, P, P, P, P, P, ctypes.c_void_p, ctypes.c_void_p, P]
        self.fun.restype = None
        set_threads = self.lib.MKL_Set_Num_Threads
        set_threads.argtypes, set_threads.restype = [ctypes.c_int], None
        self.lib.MKL_Set_Dynamic.argtypes = [ctypes.c_int]
        self.lib.MKL_Set_Dynamic.restype = None
        self.lib.MKL_Set_Dynamic(0)
        set_threads(threads)  # v1's import-time thread reset is explicitly overridden here.
        self.lib.MKL_Get_Max_Threads.argtypes = []
        self.lib.MKL_Get_Max_Threads.restype = ctypes.c_int
        actual_threads = int(self.lib.MKL_Get_Max_Threads())
        if actual_threads != threads or not 1 <= actual_threads <= 16:
            raise RuntimeError(f'MKL thread setting mismatch: {actual_threads} requested {threads}')
        os.environ['MKL_NUM_THREADS'] = str(threads)
        raw = sparse.csr_matrix(matrix, dtype=np.complex128)
        raw.sum_duplicates()
        raw.sort_indices()
        raw.eliminate_zeros()
        limit = np.iinfo(self.npdtype).max
        if max(raw.shape[0], raw.nnz, int(raw.indptr[-1]), int(raw.indices.max(initial=0))) > limit:
            raise OverflowError(f'matrix exceeds {self.bits}-bit PARDISO integer ABI')
        self.scale = np.maximum(np.asarray(abs(raw).max(axis=1).toarray()).ravel(), 1e-300)
        self.A = (sparse.diags(1 / self.scale) @ raw).tocsr()
        self.A.sum_duplicates()
        self.A.sort_indices()
        self.data = np.ascontiguousarray(self.A.data, dtype=np.complex128)
        self.ia = np.ascontiguousarray(self.A.indptr, dtype=self.npdtype)
        self.ja = np.ascontiguousarray(self.A.indices, dtype=self.npdtype)
        self.iparm = np.zeros(64, dtype=self.npdtype)
        self.perm = np.zeros(self.A.shape[0], dtype=self.npdtype)
        for index, value in {0: 1, 1: 2, 7: 2, 9: 13, 10: 1, 12: 1,
                             17: -1, 18: -1, 26: 1, 27: 0, 34: 1, 59: 0}.items():
            self.iparm[index] = value
        self.info = dict(backend='mkl_pardiso_complex', mtype=13, integer_bits=self.bits,
                         zero_based_csr=True, matrix_dtype='complex128', library=str(self.lib._name),
                         requested_threads=threads, mkl_max_threads=actual_threads,
                         matrix_nnz=int(raw.nnz), matrix_storage_bytes=sparse_bytes(raw),
                         factor_nnz=None, memory_before=memory(),
                         factor_memory_note='PARDISO iparm estimates/stats are separate from process peak RSS')
        dummy = np.zeros(self.A.shape[0], complex)
        start = time.monotonic()
        try:
            self._call(11, dummy)
            self.info['analysis_seconds'] = time.monotonic() - start
            self._stats()
            if stats_path:
                v1.write(stats_path, self.info)
            predicted = int(self.iparm[16]) * 1024
            available = memory()['system_mem_available_bytes']
            if predicted > .9 * available:
                raise MemoryError('PARDISO numerical-factor memory estimate exceeds 90% of currently available memory')
            start = time.monotonic()
            self._call(22, dummy)
            self.info['factorization_seconds'] = time.monotonic() - start
            self._stats()
            self.info['memory_after_factorization'] = memory()
            if stats_path:
                v1.write(stats_path, self.info)
        except BaseException:
            self.close()
            raise

    def _stats(self):
        self.info.update(factor_nnz=int(self.iparm[17]),
                         pardiso_symbolic_peak_kib=int(self.iparm[14]),
                         pardiso_symbolic_permanent_kib=int(self.iparm[15]),
                         pardiso_numerical_factor_kib=int(self.iparm[16]),
                         perturbed_pivots=int(self.iparm[13]))

    def _call(self, phase, rhs):
        if self.deadline and phase != -1:
            self.deadline.check()
        b = np.ascontiguousarray(rhs, dtype=np.complex128)
        x = np.zeros_like(b, dtype=np.complex128)
        I, P = self.integer, ctypes.POINTER(self.integer)
        scalars = [I(v) for v in (1, 1, 13, phase, self.A.shape[0], 1, 0, 0)]
        maxfct, mnum, mtype, ph, n, nrhs, msg, error = scalars
        self.fun(self.pt.ctypes.data_as(ctypes.c_void_p), ctypes.byref(maxfct), ctypes.byref(mnum),
                 ctypes.byref(mtype), ctypes.byref(ph), ctypes.byref(n),
                 self.data.ctypes.data_as(ctypes.c_void_p), self.ia.ctypes.data_as(P),
                 self.ja.ctypes.data_as(P), self.perm.ctypes.data_as(P), ctypes.byref(nrhs),
                 self.iparm.ctypes.data_as(P), ctypes.byref(msg),
                 b.ctypes.data_as(ctypes.c_void_p), x.ctypes.data_as(ctypes.c_void_p), ctypes.byref(error))
        if error.value:
            raise RuntimeError(f'PARDISO phase={phase} error={error.value}')
        if self.deadline and phase != -1:
            self.deadline.check()
        return x

    def solve(self, rhs):
        return self._call(33, np.asarray(rhs, complex) / self.scale)

    def close(self):
        if not self.closed:
            try:
                if np.any(self.pt):
                    self._call(-1, np.zeros(self.A.shape[0], complex))
            finally:
                self.closed = True


class ScipyComplex:
    def __init__(self, matrix, threads=1, deadline=None, stats_path=None):
        if deadline:
            deadline.check()
        start = time.monotonic()
        raw = sparse.csc_matrix(matrix, dtype=np.complex128)
        self.scale = np.maximum(np.asarray(abs(raw).max(axis=1).toarray()).ravel(), 1e-300)
        self.lu = splu((sparse.diags(1 / self.scale) @ raw).tocsc())
        lower_factor, upper_factor = self.lu.L, self.lu.U
        self.info = dict(backend='scipy_superlu_complex', matrix_dtype='complex128', threads=1,
                         matrix_nnz=int(raw.nnz), factor_nnz=int(lower_factor.nnz + upper_factor.nnz),
                         factor_sparse_storage_bytes=sparse_bytes(lower_factor) + sparse_bytes(upper_factor),
                         factorization_seconds=time.monotonic() - start, memory=memory(),
                         factor_memory_note='L/U sparse array storage excludes SuperLU workspace; RSS is process-wide')
        if stats_path:
            v1.write(stats_path, self.info)
        if deadline:
            deadline.check()

    def solve(self, rhs):
        return self.lu.solve(np.asarray(rhs, complex) / self.scale)

    def close(self):
        self.lu = None


def native_probe(threads):
    import pypardiso
    A = sparse.csr_matrix(np.array([[2+1j, .3-.6j, 0], [1j, 3+.2j, .5], [.2, 0, 1.7-1.1j]]))
    b = np.array([1+2j, -3+.7j, .1-4j])
    wrapper_rejects = False
    try:
        pypardiso.PyPardisoSolver(mtype=13).factorize(A)
    except TypeError:
        wrapper_rejects = True
    factor = PardisoComplex(A, threads=threads)
    try:
        x = factor.solve(b)
        residual = float(np.linalg.norm(A @ x - b) / np.linalg.norm(b))
        dense_error = float(np.linalg.norm(x - np.linalg.solve(A.toarray(), b)) / np.linalg.norm(x))
        if (not np.all(np.isfinite(x)) or not np.isfinite(residual) or not np.isfinite(dense_error)
                or max(residual, dense_error) > 1e-11 or not np.iscomplexobj(x)):
            raise RuntimeError('native PARDISO complex capability probe failed')
        return dict(supported=True, wrapper_complex_rejected=wrapper_rejects,
                    general_complex_matrix=True, general_complex_rhs=True,
                    relative_residual=residual, dense_relative_error=dense_error, backend=factor.info)
    finally:
        factor.close()


def select_backend(choice='auto', threads=16):
    if choice == 'scipy':
        return 'scipy', dict(selected='scipy', reason='explicit request')
    try:
        result = subprocess.run([sys.executable, '-B', str(Path(__file__).resolve()),
                                 '--probe-native', '--threads', str(threads)],
                                capture_output=True, text=True, timeout=60,
                                env={**os.environ, 'PYTHONDONTWRITEBYTECODE': '1'})
        lines = result.stdout.strip().splitlines()
        probe = json.loads(lines[-1]) if lines else {}
        if result.returncode or not probe.get('supported'):
            raise RuntimeError(f'exit={result.returncode}; stderr={result.stderr[-2000:]}; probe={probe}')
        return 'pardiso', probe
    except Exception as exc:
        if choice == 'pardiso':
            raise RuntimeError(f'required native complex PARDISO unavailable: {exc}') from exc
        return 'scipy', dict(selected='scipy', reason=f'native complex capability probe failed: {exc}')


def factorize(K, backend, threads, deadline, stats_path=None):
    return (PardisoComplex if backend == 'pardiso' else ScipyComplex)(
        K, threads=threads if backend == 'pardiso' else 1, deadline=deadline, stats_path=stats_path)


class DirectInverse:
    def __init__(self, temporal_J, B, sigma, factor, deadline, log_path):
        self.J, self.B, self.sigma, self.factor = temporal_J, B, sigma, factor
        self.deadline, self.log_path, self.calls = deadline, Path(log_path), 0
        self.T = LinearOperator(temporal_J.shape, matvec=self.apply, dtype=complex)

    def apply(self, x):
        self.deadline.check()
        start = time.monotonic()
        rhs = -(self.B @ x)
        y = np.asarray(self.factor.solve(rhs), dtype=np.complex128)
        residuals = []
        for attempt in range(3):
            r = rhs - (self.J @ y + self.sigma * (self.B @ y))
            rel = float(np.linalg.norm(r) / max(np.linalg.norm(rhs), 1e-300))
            residuals.append(rel)
            if not np.isfinite(rel):
                break
            if np.isfinite(rel) and rel <= 1e-10:
                break
            if attempt < 2:
                y += self.factor.solve(r)
        self.calls += 1
        self.log_path.parent.mkdir(parents=True, exist_ok=True)
        with self.log_path.open('a') as f:
            f.write(json.dumps(dict(call=self.calls, sigma=v1.zpair(self.sigma),
                                    original_relative=residuals[-1] if np.isfinite(residuals[-1]) else None,
                                    direct_residuals=[r if np.isfinite(r) else None for r in residuals],
                                    nonfinite_residual=not np.isfinite(residuals[-1]),
                                    seconds=time.monotonic()-start), allow_nan=False) + '\n')
        if not np.isfinite(residuals[-1]) or residuals[-1] > 5e-8:
            raise v1.InnerFailure(f'direct original linear residual {residuals[-1]:.3e} exceeds 5e-8')
        self.deadline.check()
        return y


def run_center(temporal, J, sigma, out, fingerprint, deadline, backend='pardiso',
               threads=16, k=6, ncv=22, outer_maxiter=80, seed=None):
    out = Path(out)
    out.mkdir(parents=True, exist_ok=True)
    result_path = out / 'result.json'
    if result_path.exists():
        old = json.loads(result_path.read_text())
        if old['fingerprint'] != fingerprint:
            raise RuntimeError('center fingerprint mismatch')
        if old['complete']:
            return old
        if (out/'eigenpairs.npz').exists():
            with np.load(out/'eigenpairs.npz') as d:
                if d['vectors'].shape[1]:
                    seed = d['vectors'][:, 0].copy()
        attempt = 0
        while (out/f'archive_attempt_{attempt:03d}').exists():
            attempt += 1
        archive = out/f'archive_attempt_{attempt:03d}'
        archive.mkdir()
        for name in ('result.json','eigenpairs.npz','factor.json','FAILED.json'):
            if (out/name).exists():
                (out/name).replace(archive/name)
    start, factor = time.monotonic(), None
    values, vectors = np.array([], complex), np.empty((J.shape[0], 0), complex)
    result = dict(fingerprint=fingerprint, sigma=v1.zpair(sigma), complete=False, k=k, ncv=ncv,
                  backend=backend, outer_tol=1e-8, outer_maxiter=outer_maxiter,
                  method='one full coupled complex sparse LU per center; direct backsolves',
                  exhaustive_stability=False)
    try:
        deadline.check()
        K = (J.astype(complex) + sigma*sparse.diags(temporal.b, format='csr')).tocsr()
        K.sum_duplicates()
        K.eliminate_zeros()
        K.sort_indices()
        factor = factorize(K, backend, threads, deadline, out/'factor.json')
        result['factor'] = factor.info
        direct = DirectInverse(temporal.J, temporal.B, sigma, factor, deadline, out/'direct_solves.jsonl')
        try:
            mu, vectors = eigs(direct.T, k=k, ncv=ncv, which='LM', tol=1e-8,
                               maxiter=outer_maxiter, v0=seed)
            result['complete'] = True
        except ArpackNoConvergence as exc:
            mu, vectors = exc.eigenvalues, exc.eigenvectors
            result['failure'] = str(exc)
        with np.errstate(divide='ignore', invalid='ignore'):
            values = sigma + 1/mu
        result['direct_inverse_calls'] = direct.calls
        phase = temporal.neutral()
        for value, vec in zip(values, vectors.T):
            deadline.check()
            result.setdefault('eigenpairs', []).append(v1.certify(
                temporal.J, temporal.B, max(temporal.b), value, vec, phase=phase))
    except Exception as exc:
        result['complete'] = False
        result['failure'] = f'{type(exc).__name__}: {exc}'
    finally:
        if factor is not None:
            try:
                factor.close()  # PARDISO phase=-1, also on rejection/budget failure.
            except Exception as exc:
                result['complete'] = False
                result['release_failure'] = f'{type(exc).__name__}: {exc}'
                result.setdefault('failure', 'factor release failed: ' + result['release_failure'])
        result.setdefault('eigenpairs', [])
        result['accepted_count'] = sum(r['accepted'] for r in result['eigenpairs'])
        result['rejected_count'] = len(result['eigenpairs']) - result['accepted_count']
        if result['accepted_count'] < k or result['rejected_count']:
            result['complete'] = False
            result.setdefault('failure', 'not all requested eigenpairs certified')
        result['seconds'] = time.monotonic()-start
        result['memory_at_completion'] = memory()
        v1.save(out/'eigenpairs.npz', values=values, vectors=vectors)
        v1.write(result_path, result)
        v1.write(out/('DONE.json' if result['complete'] else 'FAILED.json'), result)
        if result['complete']:
            (out/'FAILED.json').unlink(missing_ok=True)
    return result


def fingerprints():
    return dict(v1.code_fingerprints(), **{str(Path(__file__).resolve()): v1.sha(__file__)})


def point(name, out, budget, backend_choice, threads, real_shift, center_budget):
    out = Path(out)
    out.mkdir(parents=True, exist_ok=True)
    start, deadline = time.monotonic(), v1.Deadline(budget)
    try:
        temporal, source = v1.load_source(name, deadline)
        backend, probe = select_backend(backend_choice, threads)
        deadline.check()
        v1.write(out/'backend_probe.json', probe)
        J, assembly_stats = assemble(temporal, deadline)
        v1.write(out/'assembly.json', assembly_stats)
        shifts = [complex(real_shift, im) for im in IMAG_CENTERS]
        config = dict(source=source, code_sha256=fingerprints(), backend=backend, threads=threads,
                      shifts=[v1.zpair(s) for s in shifts], k=6, ncv=22,
                      center_budget=center_budget, direct_accept=5e-8, eigen_accept=1e-7)
        fp = v1.digest(config)
        manifest = out/'manifest.json'
        if manifest.exists() and json.loads(manifest.read_text())['fingerprint'] != fp:
            raise RuntimeError('point manifest changed; choose a new output directory')
        v1.write(manifest, dict(fingerprint=fp, **config))
        v1.write(out/'source.json', source)
        rng = np.random.default_rng(20261030)
        seed = rng.normal(size=temporal.size).astype(complex)
        phase = temporal.neutral()
        seed /= np.linalg.norm(seed)
        seed += phase/max(np.linalg.norm(phase), 1e-300)
        records, statuses = [], []
        for i, sigma in enumerate(shifts):
            deadline.check()
            local = v1.Deadline(min(center_budget or deadline.remaining(), deadline.remaining()))
            temporal.deadline = local
            r = run_center(temporal, J, sigma, out/f'center_{i:02d}', fp, local,
                           backend=backend, threads=threads, seed=seed)
            statuses.append(dict(center=i, sigma=v1.zpair(sigma), complete=r['complete'],
                                 failure=r.get('failure')))
            records.extend(dict(pair, centers=[i]) for pair in r['eigenpairs'] if pair['accepted'])
            v1.write(out/'merged.json', dict(eigenpairs=v1.deduplicate(records), centers=statuses,
                                           exhaustive_stability=False, sampled_imaginary_centers=list(IMAG_CENTERS)))
            if r.get('release_failure'):
                raise RuntimeError('factor release failed; aborting point to release all process-owned memory')
            gc.collect()
        complete = all(r['complete'] for r in statuses)
        result = dict(complete=complete, source=name, seconds=time.monotonic()-start,
                      centers=statuses, accepted_unique=len(v1.deduplicate(records)),
                      exhaustive_stability=False, perturbations='retained integer beta=68 harmonics only')
        v1.write(out/('DONE.json' if complete else 'FAILED.json'), result)
        if complete:
            (out/'FAILED.json').unlink(missing_ok=True)
        return 0 if complete else 1
    except Exception as exc:
        v1.write(out/'FAILED.json', dict(complete=False, failure=f'{type(exc).__name__}: {exc}',
                                        traceback=traceback.format_exc(), seconds=time.monotonic()-start))
        return 1


def campaign(out, budget=19200., backend='auto', threads=16, real_shift=.002, center_budget=0.):
    out = Path(out)
    out.mkdir(parents=True, exist_ok=True)
    active_proc, active_folder = None, None

    def terminate_owned(signum, frame):
        if active_proc is not None:
            try:
                os.killpg(active_proc.pid, signal.SIGKILL)
                active_proc.wait()
            except ProcessLookupError:
                pass
        failure = dict(complete=False, failure=f'campaign terminated by signal {signum}')
        if active_folder is not None:
            v1.write(active_folder/'FAILED.json', failure)
        v1.write(out/'FAILED.json', failure)
        raise SystemExit(128+signum)

    signal.signal(signal.SIGTERM, terminate_owned)
    signal.signal(signal.SIGINT, terminate_owned)
    config = dict(code_sha256=fingerprints(), budget=budget, backend=backend, threads=threads,
                  real_shift=real_shift, imag_centers=list(IMAG_CENTERS), center_budget=center_budget,
                  source_hashes={n: v1.sha(p/'fields.npz') for n,p in v1.SOURCES.items()})
    fp, manifest = v1.digest(config), out/'campaign_manifest.json'
    if manifest.exists() and json.loads(manifest.read_text())['fingerprint'] != fp:
        raise RuntimeError('campaign changed; choose a new output directory')
    v1.write(manifest, dict(fingerprint=fp, **config))
    statuses = []
    for name in v1.SOURCES:
        folder = out/name
        folder.mkdir(parents=True, exist_ok=True)
        if (folder/'DONE.json').exists():
            statuses.append(dict(source=name, returncode=0, resumed_completed=True))
            continue
        ledger, active = folder/'budget.json', folder/'ACTIVE.json'
        used = json.loads(ledger.read_text())['consumed_seconds'] if ledger.exists() else 0.
        if active.exists():
            used += json.loads(active.read_text())['reserved_seconds']
            used = min(budget, used)
            v1.write(ledger, dict(consumed_seconds=used, interrupted_attempt_charged=True))
            active.unlink()
        remaining = budget-used
        if remaining <= 1:
            v1.write(folder/'FAILED.json', dict(failure='cumulative point budget exhausted', consumed_seconds=used))
            statuses.append(dict(source=name, returncode=1, reason='budget exhausted'))
            continue
        v1.write(active, dict(reserved_seconds=remaining, started_unix=time.time()))
        active_folder, start, rc = folder, time.monotonic(), 1
        cmd = [sys.executable, '-B', str(Path(__file__).resolve()), '--point', name, '--out', str(folder),
               '--budget', str(max(.1,remaining-3)), '--backend', backend, '--threads', str(threads),
               '--real-shift', str(real_shift), '--center-budget', str(center_budget)]
        try:
            try:
                os.fstat(9)
                fds = (9,)
            except OSError:
                fds = ()
            with (folder/'run.log').open('a') as log:
                active_proc = subprocess.Popen(cmd, stdout=log, stderr=subprocess.STDOUT,
                                               env={**os.environ,'PYTHONDONTWRITEBYTECODE':'1'},
                                               start_new_session=True, pass_fds=fds)
                try:
                    rc = active_proc.wait(timeout=remaining)
                except subprocess.TimeoutExpired:
                    os.killpg(active_proc.pid, signal.SIGKILL)
                    active_proc.wait()
                    v1.write(folder/'FAILED.json', dict(failure='hard cumulative wall-time limit', budget=budget))
        except Exception as exc:
            v1.write(folder/'FAILED.json', dict(failure=f'launch: {type(exc).__name__}: {exc}'))
        finally:
            if active_proc is not None and active_proc.returncode is None:
                try:
                    os.killpg(active_proc.pid, signal.SIGKILL)
                    active_proc.wait()
                except ProcessLookupError:
                    pass
            v1.write(ledger, dict(consumed_seconds=min(budget,used+time.monotonic()-start), limit_seconds=budget))
            active.unlink(missing_ok=True)
            active_proc, active_folder = None, None
        if rc != 0 or not (folder/'DONE.json').exists():
            rc = rc or 1
            if not (folder/'FAILED.json').exists():
                v1.write(folder/'FAILED.json', dict(failure='worker exited without DONE', returncode=rc))
        statuses.append(dict(source=name, returncode=rc))
    complete = all(r['returncode']==0 for r in statuses)
    v1.write(out/('DONE.json' if complete else 'FAILED.json'),
             dict(complete=complete, points=statuses, exhaustive_stability=False))
    if complete:
        (out/'FAILED.json').unlink(missing_ok=True)
    return 0 if complete else 1


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--probe-native', action='store_true')
    parser.add_argument('--point', choices=list(v1.SOURCES))
    parser.add_argument('--campaign', action='store_true')
    parser.add_argument('--out', type=Path)
    parser.add_argument('--budget', type=float, default=19200.)
    parser.add_argument('--backend', choices=['auto','pardiso','scipy'], default='auto')
    parser.add_argument('--threads', type=int, default=16)
    parser.add_argument('--real-shift', type=float, default=.002)
    parser.add_argument('--center-budget', type=float, default=0.)
    args = parser.parse_args()
    if not 1 <= args.threads <= 16:
        parser.error('threads must lie in [1,16]')
    if args.probe_native:
        try:
            print(json.dumps(native_probe(args.threads), allow_nan=False))
            return 0
        except Exception as exc:
            print(json.dumps(dict(supported=False, failure=f'{type(exc).__name__}: {exc}')))
            return 1
    if not args.out:
        parser.error('an output subdirectory within D2 is required')
    args.out = args.out.resolve()
    scope = OUT.resolve()
    legacy = (scope/'production_v1').resolve()
    if args.out == scope or not args.out.is_relative_to(scope) or args.out.is_relative_to(legacy):
        parser.error('output must be a D2 subdirectory outside preserved production_v1')
    if not np.isfinite(args.budget) or args.budget <= 0 or args.center_budget < 0 or not np.isfinite(args.center_budget):
        parser.error('finite positive point budget and nonnegative center budget required')
    if not np.isfinite(args.real_shift) or args.real_shift <= 0:
        parser.error('center real part must be finite and positive')
    if args.campaign:
        return campaign(args.out,args.budget,args.backend,args.threads,args.real_shift,args.center_budget)
    if args.point:
        return point(args.point,args.out,args.budget,args.backend,args.threads,args.real_shift,args.center_budget)
    parser.error('select --campaign or --point')


if __name__ == '__main__':
    sys.exit(main())

