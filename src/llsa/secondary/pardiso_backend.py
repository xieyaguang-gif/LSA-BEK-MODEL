"""Optional parallel LU, retaining symbolic analysis for unchanged sparsity.

The supported PyPardiso version is recorded in requirements-pardiso.txt. Its
low-level phase interface is needed because factorize() always repeats both
symbolic and numerical analysis. All solves use the current matrix values.
"""
from time import perf_counter

import numpy as np
from scipy import sparse


class PardisoBackend:
    def __init__(self):
        try:
            from pypardiso import PyPardisoSolver
        except ImportError as exc:
            raise RuntimeError('Install optional requirements-pardiso.txt to use PARDISO') from exc
        self.solver = PyPardisoSolver()
        self.matrix = None
        self.pattern = None
        self.allocated = False
        self.analyses = 0
        self.last_stats = {}
        self.newton_steps = 0

    @property
    def threads(self):
        return int(self.solver.libmkl.mkl_get_max_threads())

    def factorize(self, matrix, progress=None):
        if np.iscomplexobj(matrix.data):
            raise ValueError('PARDISO backend expects a packed real Jacobian')
        # Factor A.T and request the transposed triangular solve. The CSC
        # orientation is substantially more robust for our constraint rows
        # with the tested oneMKL build; verify the actual Ax=b residual below.
        matrix = sparse.csc_matrix(matrix, dtype=np.float64, copy=True)
        matrix.sum_duplicates()
        matrix.sort_indices()
        if matrix.nnz >= np.iinfo(np.int32).max:
            raise ValueError('This PyPardiso interface requires fewer than 2^31 matrix entries')
        self.solver._check_A(matrix)
        same = (self.pattern is not None and self.pattern[0] == matrix.shape
                and np.array_equal(self.pattern[1], matrix.indptr)
                and np.array_equal(self.pattern[2], matrix.indices))
        zeros = np.zeros(matrix.shape[0])
        stats = dict(symbolic_reused=bool(same), analysis_seconds=0., threads=self.threads)
        if not same:
            if self.allocated:
                self.close()
            self.solver._check_A(matrix)
            if progress:
                progress(dict(stage='pardiso_analysis_start', threads=self.threads,
                              dimension=matrix.shape[0], nonzeros=matrix.nnz))
            started = perf_counter()
            self.solver.set_phase(11)
            self.allocated = True
            self.solver._call_pardiso(matrix, zeros)
            stats['analysis_seconds'] = perf_counter()-started
            self.analyses += 1
            self.pattern = (matrix.shape, matrix.indptr.copy(), matrix.indices.copy())
        if progress:
            progress(dict(stage='pardiso_numeric_start', **stats))
        started = perf_counter()
        self.solver.set_phase(22)
        self.solver._call_pardiso(matrix, zeros)
        stats['numeric_seconds'] = perf_counter()-started
        self.matrix = matrix
        self.newton_steps = 0
        self.last_stats = stats
        if progress:
            progress(dict(stage='pardiso_numeric_complete', **stats))
        return stats

    def solve(self, rhs):
        if self.matrix is None:
            raise RuntimeError('Factorize a matrix before solving')
        # The first native analysis initializes defaults, including iparm(12).
        # Restore the CSC transpose flag immediately before back substitution.
        self.solver._check_A(self.matrix)
        rhs = self.solver._check_b(self.matrix, np.asarray(rhs, dtype=np.float64))
        self.solver.set_phase(33)
        answer = self.solver._call_pardiso(self.matrix, rhs)
        rhs_norm = max(np.linalg.norm(rhs),1e-300)
        residual = rhs-self.matrix@answer
        error = np.linalg.norm(residual)/rhs_norm
        self.last_stats['initial_linear_relative_residual'] = float(error)
        refinements = 0
        # A perturbed native pivot can leave a larger forward residual even
        # after its internal refinement. Correct against the actual matrix;
        # never relax the Ax=b acceptance threshold to accommodate the factor.
        while np.isfinite(error) and error > 1e-8 and refinements < 3:
            self.solver._check_A(self.matrix)
            correction_rhs = self.solver._check_b(self.matrix,np.asarray(residual,dtype=np.float64))
            self.solver.set_phase(33)
            answer += self.solver._call_pardiso(self.matrix,correction_rhs)
            residual = rhs-self.matrix@answer
            error = np.linalg.norm(residual)/rhs_norm
            refinements += 1
        self.last_stats['external_refinement_steps'] = refinements
        self.last_stats['linear_relative_residual'] = float(error)
        if not np.isfinite(error) or error > 1e-8:
            raise RuntimeError(f'PARDISO failed the Ax=b residual check: {error:.3e}')
        return answer

    def close(self):
        if self.allocated:
            self.solver.free_memory(everything=True)
        self.allocated = False
        self.matrix = None
        self.pattern = None
        self.newton_steps = 0

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_value, traceback):
        self.close()
