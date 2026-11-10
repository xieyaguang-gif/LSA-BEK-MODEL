#!/usr/bin/env python3
"""D2d: one full coupled complex LU per center, certified Krylov precision recovery."""
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
from scipy.sparse.linalg import LinearOperator, splu, eigs, gmres, ArpackNoConvergence
import fullbvp_phaseD2_stability as v1
from llsa.secondary.radial import convection_blocks, real_blocks

ROOT, OUT = v1.ROOT, v1.OUT
IMAG_CENTERS = (0., 2., -2., 5., -5., 10., -10., 15., -15., 20., -20., 26., -26.)


PILOT_CENTERS = (0., 2., -2.)
ACCEPT = 5e-8
DEFAULT_POLICY = dict(refinement=8, pivot_exponent=13, direct_steps=2,
                      krylov_restart=35, krylov_maxiter=5, krylov_rtol=1e-10)


def output_path(path):
    """Pure canonical validation: call before any output/lock/marker operation."""
    scope = OUT.resolve()
    path = Path(path).resolve()
    if path == scope or not path.is_relative_to(scope):
        raise ValueError('output must be a proper D2 descendant')
    for name in ('production_v1', 'production_v2'):
        legacy = scope / name
        if path.is_relative_to(legacy) or path.is_relative_to(legacy.resolve()):
            raise ValueError('refusing preserved ' + name)
    return path


def policy_checked(policy=None):
    p = dict(DEFAULT_POLICY, **(policy or {}))
    for key in ('refinement', 'direct_steps', 'krylov_restart', 'krylov_maxiter'):
        if not isinstance(p[key], int) or p[key] < (1 if key.startswith('krylov_') else 0):
            raise ValueError('invalid bounded solver policy: ' + key)
    if not 1 <= p['pivot_exponent'] <= 30 or not 0 < p['krylov_rtol'] <= 1e-9:
        raise ValueError('invalid pivot/refinement precision policy')
    return p


def policy_args(policy):
    return [value for key in DEFAULT_POLICY for value in
            ('--' + key.replace('_', '-'), str(policy[key]))]


def scan_centers(scan):
    if scan not in ('pilot', 'full'):
        raise ValueError('scan must be pilot or full')
    return PILOT_CENTERS if scan == 'pilot' else IMAG_CENTERS


def other_math_pools():
    """Enforce/query loaded OpenBLAS pools without adding a new dependency."""
    libraries = sorted({line.split()[-1] for line in Path('/proc/self/maps').read_text().splitlines()
                        if 'openblas' in line.lower() and '.so' in line and line.split()[-1].startswith('/')})
    records = []
    for path in libraries:
        lib = ctypes.CDLL(path)
        found = False
        for prefix in ('openblas_', 'scipy_openblas_'):
            for suffix in ('', '64_'):
                getter_name = prefix + 'get_num_threads' + suffix
                setter_name = prefix + 'set_num_threads' + suffix
                if hasattr(lib, getter_name) and hasattr(lib, setter_name):
                    setter, getter = getattr(lib,setter_name), getattr(lib,getter_name)
                    setter.argtypes, setter.restype = [ctypes.c_int], None
                    getter.argtypes, getter.restype = [], ctypes.c_int
                    setter(1)
                    actual = int(getter())
                    if actual != 1:
                        raise RuntimeError('non-MKL BLAS pool exceeded one thread')
                    records.append(dict(library=path,effective_threads=actual,getter=getter_name))
                    found = True
                    break
            if found:
                break
        if not found:
            raise RuntimeError('unable to verify OpenBLAS thread count: ' + path)
    return records

_OTHER_MATH_POOLS = other_math_pools()


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
    def __init__(self, matrix, threads=16, deadline=None, stats_path=None, policy=None):
        self.policy = policy_checked(policy)
        stats_path = output_path(stats_path) if stats_path else None
        import pypardiso
        wrapper = pypardiso.PyPardisoSolver()
        self.lib = wrapper.libmkl
        self.deadline, self.stats_path = deadline, stats_path
        self.closed = False
        self.pt = np.zeros(64, dtype=np.uintp)
        # Prefer explicit 64-bit API; every integer argument and CSR array follows it.
        if not hasattr(self.lib, 'pardiso_64'):
            raise RuntimeError('D2d requires PARDISO64; use the explicit SciPy fallback otherwise')
        self.bits = 64
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
        for index, value in {0: 1, 1: 2, 7: self.policy["refinement"], 9: self.policy["pivot_exponent"], 10: 1, 12: 1,
                             17: -1, 18: -1, 26: 1, 27: 0, 34: 1, 59: 0}.items():
            self.iparm[index] = value
        self.info = dict(backend='mkl_pardiso_complex', mtype=13, integer_bits=self.bits,
                         zero_based_csr=True, matrix_dtype='complex128', library=str(self.lib._name),
                         requested_threads=threads, mkl_max_threads=actual_threads,
                         matrix_nnz=int(raw.nnz), matrix_storage_bytes=sparse_bytes(raw),
                         factor_nnz=None, memory_before=memory(), solve_calls=0,
                         policy=self.policy, math_pools=other_math_pools(),
                         factorization_count=1, manual_row_scaling=True,
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
        requested = {7: self.policy['refinement'], 9: self.policy['pivot_exponent'], 10: 1, 12: 1}
        effective = {str(i + 1): int(self.iparm[i]) for i in requested}
        if any(int(self.iparm[i]) != value for i, value in requested.items()):
            raise RuntimeError('effective PARDISO numerical policy mismatch')
        self.info.update(pardiso_effective_fortran_iparm=effective,
                         scaling_matching_preexisting_in_v2=True,
                         pivot_policy='perturb pivots below 10^(-iparm(10)); no assumed causal diagnosis',
                         last_iterative_refinement_steps=int(self.iparm[6]))
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
        x = self._call(33, np.asarray(rhs, complex) / self.scale)
        self.info['solve_calls'] += 1
        self._stats()
        return x

    def close(self):
        if not self.closed:
            try:
                if np.any(self.pt):
                    self._call(-1, np.zeros(self.A.shape[0], complex))
            finally:
                self.closed = True
                self.info['phase_minus_one_cleanup_attempted'] = True


class ScipyComplex:
    def __init__(self, matrix, threads=1, deadline=None, stats_path=None, policy=None):
        if deadline:
            deadline.check()
        stats_path = output_path(stats_path) if stats_path else None
        start = time.monotonic()
        raw = sparse.csc_matrix(matrix, dtype=np.complex128)
        self.scale = np.maximum(np.asarray(abs(raw).max(axis=1).toarray()).ravel(), 1e-300)
        self.lu = splu((sparse.diags(1 / self.scale) @ raw).tocsc())
        lower_factor, upper_factor = self.lu.L, self.lu.U
        self.info = dict(factorization_count=1, solve_calls=0, policy=policy_checked(policy), backend='scipy_superlu_complex', matrix_dtype='complex128', threads=1,
                         matrix_nnz=int(raw.nnz), factor_nnz=int(lower_factor.nnz + upper_factor.nnz),
                         factor_sparse_storage_bytes=sparse_bytes(lower_factor) + sparse_bytes(upper_factor),
                         factorization_seconds=time.monotonic() - start, memory=memory(),
                         factor_memory_note='L/U sparse array storage excludes SuperLU workspace; RSS is process-wide')
        if stats_path:
            v1.write(stats_path, self.info)
        if deadline:
            deadline.check()

    def solve(self, rhs):
        self.info['solve_calls'] += 1
        return self.lu.solve(np.asarray(rhs, complex) / self.scale)

    def close(self):
        self.lu = None


def native_probe(threads, policy=None):
    import pypardiso
    A = sparse.csr_matrix(np.array([[2+1j, .3-.6j, 0], [1j, 3+.2j, .5], [.2, 0, 1.7-1.1j]]))
    b = np.array([1+2j, -3+.7j, .1-4j])
    wrapper_rejects = False
    try:
        pypardiso.PyPardisoSolver(mtype=13).factorize(A)
    except TypeError:
        wrapper_rejects = True
    factor = PardisoComplex(A, threads=threads, policy=policy)
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


def select_backend(choice='auto', threads=16, policy=None):
    if choice == 'scipy':
        return 'scipy', dict(selected='scipy', reason='explicit request')
    try:
        result = subprocess.run([sys.executable, '-B', str(Path(__file__).resolve()),
                                 '--probe-native', '--threads', str(threads)] + policy_args(policy_checked(policy)),
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


def factorize(K, backend, threads, deadline, stats_path=None, policy=None):
    return (PardisoComplex if backend == 'pardiso' else ScipyComplex)(
        K, threads=threads if backend == 'pardiso' else 1, deadline=deadline, stats_path=stats_path, policy=policy)



class PrecisionInverse:
    """Solve (J+sigma B)y=-Bx; certify the original matrix-free equation."""
    def __init__(self, temporal_J, B, sigma, factor, deadline, log_path,
                 assembled_K=None, policy=None, solver=gmres):
        self.J, self.B, self.sigma, self.factor = temporal_J, B, sigma, factor
        self.deadline, self.log_path = deadline, output_path(log_path)
        self.policy, self.solver = policy_checked(policy), solver
        self.assembled_K, self.calls = assembled_K, 0
        self.K = LinearOperator(temporal_J.shape, matvec=self.action, dtype=complex)
        self.T = LinearOperator(temporal_J.shape, matvec=self.apply, dtype=complex)

    def action(self, y):
        self.deadline.check()
        return self.J @ y + self.sigma * (self.B @ y)

    def apply(self, x):
        self.deadline.check()
        start = time.monotonic()
        self.calls += 1
        record = dict(call=self.calls, sigma=v1.zpair(self.sigma), accept=ACCEPT,
                      direct_residuals=[], assembled_residuals=[], solution_amplification=[],
                      direct_corrections=0, krylov_preconditioner_calls=0,
                      krylov_history=[], krylov_used=False, accepted=False)
        try:
            rhs = -(self.B @ x)
            if not np.all(np.isfinite(rhs)):
                raise v1.InnerFailure('nonfinite original right-hand side')
            denominator = max(float(np.linalg.norm(rhs)), 1e-300)

            def measure(y):
                if not np.all(np.isfinite(y)):
                    return np.full_like(rhs, np.nan), float('inf'), None, None
                residual = rhs - self.action(y)
                rel = float(np.linalg.norm(residual) / denominator)
                assembled = (float(np.linalg.norm(rhs - self.assembled_K @ y) / denominator)
                             if self.assembled_K is not None else None)
                amplification = float(np.linalg.norm(y) / denominator)
                return residual, rel, assembled, amplification

            def serial_number(value):
                return float(value) if value is not None and np.isfinite(value) else None

            t = time.monotonic()
            y = np.asarray(self.factor.solve(rhs), complex)
            best_y, best_rel, previous_rel = None, float('inf'), float('inf')
            for attempt in range(self.policy['direct_steps'] + 1):
                residual, rel, assembled, amplification = measure(y)
                record['direct_residuals'].append(serial_number(rel))
                record['assembled_residuals'].append(serial_number(assembled))
                record['solution_amplification'].append(serial_number(amplification))
                if not np.isfinite(rel):
                    if best_y is not None:
                        record['direct_early_stop'] = 'nonfinite correction; retained best finite iterate'
                        break
                    raise v1.InnerFailure('nonfinite initial/direct original solution or residual')
                if rel < best_rel:
                    best_y, best_rel = y.copy(), rel
                if rel <= ACCEPT or attempt == self.policy['direct_steps']:
                    break
                if attempt and rel >= previous_rel * .99:
                    record['direct_early_stop'] = 'less than 1 percent residual improvement'
                    break
                previous_rel = rel
                self.deadline.check()
                y += self.factor.solve(residual)
                record['direct_corrections'] += 1
            y, rel = best_y, best_rel
            record['selected_best_direct_original_relative'] = best_rel
            record.update(initial_original_relative=record['direct_residuals'][0],
                          direct_original_relative=serial_number(rel),
                          direct_seconds=time.monotonic() - t)
            if rel > ACCEPT:
                record['krylov_used'] = True

                def precondition(r):
                    self.deadline.check()
                    z = np.asarray(self.factor.solve(r), complex)
                    record['krylov_preconditioner_calls'] += 1
                    if not np.all(np.isfinite(z)):
                        raise v1.InnerFailure('nonfinite LU preconditioner action')
                    return z

                def callback(value):
                    self.deadline.check()
                    record['krylov_history'].append(serial_number(value))

                M = LinearOperator(self.J.shape, matvec=precondition, dtype=complex)
                t = time.monotonic()
                y, info = self.solver(self.K, rhs, x0=y, M=M,
                    rtol=self.policy['krylov_rtol'], atol=0.,
                    restart=self.policy['krylov_restart'], maxiter=self.policy['krylov_maxiter'],
                    callback=callback, callback_type='pr_norm')
                record.update(krylov_info=int(info), krylov_seconds=time.monotonic() - t,
                              krylov_iterations=len(record['krylov_history']))
            # Independent recomputation after either direct or Krylov path.
            residual, rel, assembled, amplification = measure(y)
            record.update(original_relative=serial_number(rel),
                          final_assembled_relative=serial_number(assembled),
                          final_solution_amplification=serial_number(amplification),
                          accepted=bool(np.all(np.isfinite(y)) and np.isfinite(rel) and rel <= ACCEPT))
            if not record['accepted']:
                raise v1.InnerFailure(f'original linear residual {rel:.3e} exceeds {ACCEPT:.3e}')
            self.deadline.check()
            return y
        except Exception as exc:
            record['failure'] = f'{type(exc).__name__}: {exc}'
            raise
        finally:
            record['seconds'] = time.monotonic() - start
            self.log_path.parent.mkdir(parents=True, exist_ok=True)
            with self.log_path.open('a') as f:
                f.write(json.dumps(record, allow_nan=False) + '\n')


def run_center(temporal, J, sigma, out, fingerprint, deadline, backend='pardiso',
               threads=16, k=6, ncv=22, outer_maxiter=80, seed=None, policy=None):
    out, policy = output_path(out), policy_checked(policy)
    out.mkdir(parents=True, exist_ok=True)
    result_path = out/'result.json'
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
        archive = output_path(out/f'archive_attempt_{attempt:03d}')
        archive.mkdir()
        for name in ('result.json','eigenpairs.npz','factor.json','FAILED.json','precision_solves.jsonl'):
            if (out/name).exists():
                (out/name).replace(archive/name)
    (out/'DONE.json').unlink(missing_ok=True)
    start, factor = time.monotonic(), None
    values, vectors = np.array([], complex), np.empty((J.shape[0], 0), complex)
    result = dict(fingerprint=fingerprint, sigma=v1.zpair(sigma), complete=False,
                  k=k, ncv=ncv, backend=backend, outer_tol=1e-8, outer_maxiter=outer_maxiter,
                  method='one full coupled complex LU; original-operator GMRES recovery as needed',
                  numerical_policy=policy, linear_accept=ACCEPT, exhaustive_stability=False)
    try:
        deadline.check()
        K = (J.astype(complex) + sigma*sparse.diags(temporal.b, format='csr')).tocsr()
        K.sum_duplicates(); K.eliminate_zeros(); K.sort_indices()
        factor = factorize(K, backend, threads, deadline, out/'factor.json', policy=policy)
        inverse = PrecisionInverse(temporal.J, temporal.B, sigma, factor, deadline,
                                   out/'precision_solves.jsonl', assembled_K=K, policy=policy)
        try:
            mu, vectors = eigs(inverse.T, k=k, ncv=ncv, which='LM', tol=1e-8,
                               maxiter=outer_maxiter, v0=seed)
            result['complete'] = True
        except ArpackNoConvergence as exc:
            mu, vectors = exc.eigenvalues, exc.eigenvectors
            result['failure'] = str(exc)
        with np.errstate(divide='ignore', invalid='ignore'):
            values = sigma + 1/mu
        result['inverse_calls'] = inverse.calls
        for value, vec in zip(values, vectors.T):
            deadline.check()
            result.setdefault('eigenpairs', []).append(v1.certify(
                temporal.J, temporal.B, max(temporal.b), value, vec, phase=temporal.neutral()))
    except Exception as exc:
        result.update(complete=False, failure=f'{type(exc).__name__}: {exc}')
    finally:
        if factor is not None:
            try:
                factor.close()
            except Exception as exc:
                result.update(complete=False, release_failure=f'{type(exc).__name__}: {exc}')
                result.setdefault('failure', 'factor release failed')
            result['factor'] = dict(factor.info)
            v1.write(out/'factor.json', factor.info)
        result.setdefault('eigenpairs', [])
        result['accepted_count'] = sum(r['accepted'] for r in result['eigenpairs'])
        result['rejected_count'] = len(result['eigenpairs']) - result['accepted_count']
        if result['accepted_count'] < k or result['rejected_count']:
            result['complete'] = False
            result.setdefault('failure', 'not all requested eigenpairs certified')
        result.update(seconds=time.monotonic()-start, memory_at_completion=memory())
        v1.save(out/'eigenpairs.npz', values=values, vectors=vectors)
        v1.write(result_path, result)
        v1.write(out/('DONE.json' if result['complete'] else 'FAILED.json'), result)
        if result['complete']:
            (out/'FAILED.json').unlink(missing_ok=True)
    return result


def fingerprints():
    paths = [Path(__file__).resolve(), ROOT/'examples/fullbvp_phaseD2b_direct.py']
    return dict(v1.code_fingerprints(), **{str(p): v1.sha(p) for p in paths})


def point(name, out, budget, backend_choice, threads, real_shift, center_budget,
          scan='pilot', policy=None):
    out, policy = output_path(out), policy_checked(policy)
    out.mkdir(parents=True, exist_ok=True)
    start, deadline = time.monotonic(), v1.Deadline(budget)
    try:
        temporal, source = v1.load_source(name, deadline)
        backend, probe = select_backend(backend_choice, threads, policy)
        deadline.check()
        v1.write(out/'backend_probe.json', probe)
        shifts = [complex(real_shift, im) for im in scan_centers(scan)]
        config = dict(source=source, code_sha256=fingerprints(), backend=backend, threads=threads,
                      shifts=[v1.zpair(s) for s in shifts], k=6, ncv=22, scan=scan,
                      coverage='limited retained-harmonic sampling', numerical_policy=policy,
                      center_budget=center_budget, linear_accept=ACCEPT, eigen_accept=1e-7)
        fp, manifest = v1.digest(config), out/'manifest.json'
        if manifest.exists() and json.loads(manifest.read_text())['fingerprint'] != fp:
            raise RuntimeError('point manifest changed; choose a new output directory')
        v1.write(manifest, dict(fingerprint=fp, **config))
        v1.write(out/'source.json', source)
        J, assembly_stats = assemble(temporal, deadline)
        v1.write(out/'assembly.json', assembly_stats)
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
                           backend=backend, threads=threads, seed=seed, policy=policy)
            statuses.append(dict(center=i, sigma=v1.zpair(sigma), complete=r['complete'], failure=r.get('failure')))
            records.extend(dict(pair, centers=[i]) for pair in r['eigenpairs'] if pair['accepted'])
            v1.write(out/'merged.json', dict(eigenpairs=v1.deduplicate(records), centers=statuses,
                exhaustive_stability=False, sampled_imaginary_centers=list(scan_centers(scan))))
            if r.get('release_failure'):
                raise RuntimeError('factor release failed; abort point to release process-owned memory')
            gc.collect()
        complete = all(r['complete'] for r in statuses)
        result = dict(complete=complete, source=name, seconds=time.monotonic()-start, scan=scan,
                      centers=statuses, accepted_unique=len(v1.deduplicate(records)),
                      exhaustive_stability=False, DONE_meaning='requested numerical sampling finished',
                      perturbations='retained integer beta=68 harmonics only')
        v1.write(out/('DONE.json' if complete else 'FAILED.json'), result)
        if complete:
            (out/'FAILED.json').unlink(missing_ok=True)
        return 0 if complete else 1
    except Exception as exc:
        (out/'DONE.json').unlink(missing_ok=True)
        v1.write(out/'FAILED.json', dict(complete=False, failure=f'{type(exc).__name__}: {exc}',
            traceback=traceback.format_exc(), seconds=time.monotonic()-start))
        return 1


def campaign(out, budget=19200., backend='auto', threads=16, real_shift=.05, center_budget=0.,
             scan='pilot', policy=None):
    out, policy = output_path(out), policy_checked(policy)
    out.mkdir(parents=True, exist_ok=True)
    active_proc, active_folder = None, None

    def cleanup_owned():
        if active_proc is not None and active_proc.returncode is None:
            try:
                os.killpg(active_proc.pid, signal.SIGKILL)
            except ProcessLookupError:
                pass
            active_proc.wait()

    def terminate_owned(signum, frame):
        cleanup_owned()
        failure = dict(complete=False, failure=f'campaign terminated by signal {signum}')
        if active_folder is not None:
            v1.write(active_folder/'FAILED.json', failure)
        v1.write(out/'FAILED.json', failure)
        raise SystemExit(128+signum)

    for sig in (signal.SIGTERM, signal.SIGINT, signal.SIGHUP):
        signal.signal(sig, terminate_owned)
    config = dict(code_sha256=fingerprints(), budget=budget, backend=backend, threads=threads,
                  real_shift=real_shift, scan=scan, imag_centers=list(scan_centers(scan)),
                  center_budget=center_budget, numerical_policy=policy,
                  source_hashes={n: {f: v1.sha(p/f) for f in ('fields.npz','summary.json')}
                                 for n,p in v1.SOURCES.items()})
    fp, manifest = v1.digest(config), out/'campaign_manifest.json'
    if manifest.exists() and json.loads(manifest.read_text())['fingerprint'] != fp:
        raise RuntimeError('campaign changed; choose a new output directory')
    v1.write(manifest, dict(fingerprint=fp, **config))
    (out/'DONE.json').unlink(missing_ok=True)
    statuses = []
    for name in v1.SOURCES:
        folder = output_path(out/name)
        folder.mkdir(parents=True, exist_ok=True)
        if (folder/'DONE.json').exists():
            statuses.append(dict(source=name, returncode=0, resumed_completed=True))
            continue
        ledger, active = folder/'budget.json', folder/'ACTIVE.json'
        used = json.loads(ledger.read_text())['consumed_seconds'] if ledger.exists() else 0.
        if active.exists():
            used = min(budget, used + json.loads(active.read_text())['reserved_seconds'])
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
               '--real-shift', str(real_shift), '--center-budget', str(center_budget), '--scan', scan] + policy_args(policy)
        try:
            try:
                os.fstat(9); fds = (9,)
            except OSError:
                fds = ()
            with (folder/'run.log').open('a') as log:
                active_proc = subprocess.Popen(cmd, stdout=log, stderr=subprocess.STDOUT,
                    env={**os.environ,'PYTHONDONTWRITEBYTECODE':'1'}, start_new_session=True, pass_fds=fds)
                try:
                    rc = active_proc.wait(timeout=remaining)
                except subprocess.TimeoutExpired:
                    cleanup_owned()
                    v1.write(folder/'FAILED.json', dict(failure='hard cumulative wall-time limit', budget=budget))
        except Exception as exc:
            v1.write(folder/'FAILED.json', dict(failure=f'launch: {type(exc).__name__}: {exc}'))
        finally:
            cleanup_owned()
            v1.write(ledger, dict(consumed_seconds=min(budget,used+time.monotonic()-start), limit_seconds=budget))
            active.unlink(missing_ok=True)
            active_proc, active_folder = None, None
        if rc != 0 or not (folder/'DONE.json').exists():
            rc = rc or 1
            if not (folder/'FAILED.json').exists():
                v1.write(folder/'FAILED.json', dict(failure='worker exited without DONE', returncode=rc))
        statuses.append(dict(source=name, returncode=rc))
    complete = all(r['returncode']==0 for r in statuses)
    v1.write(out/('DONE.json' if complete else 'FAILED.json'), dict(complete=complete, points=statuses,
        scan=scan, exhaustive_stability=False, DONE_meaning='requested numerical sampling finished'))
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
    parser.add_argument('--real-shift', type=float, default=.05)
    parser.add_argument('--center-budget', type=float, default=0.)
    parser.add_argument('--scan', choices=['pilot','full'], default='pilot')
    for key, value in DEFAULT_POLICY.items():
        parser.add_argument('--'+key.replace('_','-'), type=type(value), default=value)
    args = parser.parse_args()
    if not 1 <= args.threads <= 16:
        parser.error('threads must lie in [1,16]')
    policy = policy_checked({key:getattr(args,key) for key in DEFAULT_POLICY})
    if args.probe_native:
        try:
            print(json.dumps(native_probe(args.threads, policy), allow_nan=False)); return 0
        except Exception as exc:
            print(json.dumps(dict(supported=False, failure=f'{type(exc).__name__}: {exc}'))); return 1
    if not args.out:
        parser.error('an output subdirectory within D2 is required')
    args.out = output_path(args.out)
    if not np.isfinite(args.budget) or args.budget <= 0 or args.center_budget < 0 or not np.isfinite(args.center_budget):
        parser.error('finite positive point budget and nonnegative center budget required')
    if not np.isfinite(args.real_shift) or not .03 <= args.real_shift <= .05:
        parser.error('D2d center real part must lie in [0.03,0.05]')
    if args.campaign:
        return campaign(args.out,args.budget,args.backend,args.threads,args.real_shift,args.center_budget,args.scan,policy)
    if args.point:
        return point(args.point,args.out,args.budget,args.backend,args.threads,args.real_shift,args.center_budget,args.scan,policy)
    parser.error('select --campaign or --point')


if __name__ == '__main__':
    sys.exit(main())
