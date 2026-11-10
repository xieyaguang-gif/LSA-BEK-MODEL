"""Generalized eigenproblems, branch continuation and complex saddle search."""
from dataclasses import dataclass

import numpy as np
from scipy import linalg
from scipy.sparse.linalg import LinearOperator, eigs


@dataclass
class Spectrum:
    values: np.ndarray
    vectors: np.ndarray
    residuals: np.ndarray
    left: np.ndarray | None = None

    def select(self, target=None, imag_limit=1.0):
        """Nearest frequency/wavenumber, or largest imaginary part inside a window.

        This is a numerical selector, not a classification of physical mode families.
        For spatial branches pass an explicit target.
        """
        indices = np.flatnonzero(np.abs(self.values.imag) < imag_limit)
        if not len(indices):
            raise RuntimeError("No eigenvalues in selection window; change imag_limit or resolution")
        if target is None:
            return indices[np.argmax(self.values[indices].imag)]
        return indices[np.argmin(abs(self.values[indices]-target))]


def _eig(A, B, left=False):
    result = linalg.eig(A, B, left=left, right=True, homogeneous_eigvals=True,
                       check_finite=False)
    h, vl, vr = result if left else (result[0], None, result[1])
    good = (np.abs(h[1]) > 1e-12*(np.abs(h[0])+np.abs(h[1])))
    values = h[0, good]/h[1, good]
    return values, vr[:, good], None if vl is None else vl[:, good]


def _residual(A, B, values, vectors):
    num = linalg.norm(A@vectors - (B@vectors)*values, axis=0)
    den = (linalg.norm(A) + abs(values)*linalg.norm(B))*linalg.norm(vectors, axis=0)
    return num/np.maximum(den, np.finfo(float).tiny)


def temporal(problem, alpha, left=False, residual_tol=1e-8):
    """Full finite omega spectrum; alpha may be complex for space–time analysis."""
    A, B = problem.matrix(alpha), problem.B
    values, vectors, vl = _eig(A, B, left)
    errors = _residual(A, B, values, vectors)
    keep = np.isfinite(values) & (errors <= residual_tol)
    return Spectrum(values[keep], vectors[:, keep], errors[keep],
                    None if vl is None else vl[:, keep])


def temporal_near(problem, alpha, target, count=6, residual_tol=1e-8):
    """A few omega modes near target using shift-invert Arnoldi.

    Factor A-target*B once; applying its inverse to B avoids inverting singular B.
    Use full temporal() for initial spectrum inspection or adjoint derivatives.
    Do not set target exactly at a known eigenvalue (the factorization is singular).
    """
    A, B = problem.matrix(alpha), problem.B
    if not np.isfinite(target) or not 1 <= count < len(A)-1 or int(count) != count:
        raise ValueError("Finite target and integer 1 <= count < matrix_size-1 required")
    lu = linalg.lu_factor(A-target*B, check_finite=False)
    op = LinearOperator(A.shape, matvec=lambda q: linalg.lu_solve(lu, B@q, check_finite=False),
                        dtype=complex)
    mu, vectors = eigs(op, k=count, which="LM", tol=1e-10,
                       v0=np.ones(len(A), complex))
    good = abs(mu) > np.finfo(float).eps
    values, vectors = target+1/mu[good], vectors[:, good]
    errors = _residual(A, B, values, vectors)
    keep = np.isfinite(values) & (errors <= residual_tol)
    return Spectrum(values[keep], vectors[:, keep], errors[keep])


def spatial(problem, omega, real_bounds=None, residual_tol=1e-8):
    """Solve the quadratic alpha problem with the SAME boundary rows as temporal.

    Companion vector is [q, alpha*q]. Singular A2 is handled by generalized QZ;
    no matrix inverse or artificial eigenvalue-dependent boundary penalty is used.
    Optional real_bounds=(0.06,1) matches the paper's spatial selection window.
    """
    if not np.isfinite(omega):
        raise ValueError("omega must be finite")
    C, A1, A2 = problem.A0-omega*problem.B, problem.A1, problem.A2
    n = len(C)
    I, Z = np.eye(n), np.zeros_like(C)
    L = np.block([[Z, I], [-C, -A1]])
    R = np.block([[I, Z], [Z, A2]])
    values, vectors, _ = _eig(L, R)
    vectors = vectors[:n]
    num = linalg.norm(C@vectors+(A1@vectors)*values+(A2@vectors)*values**2, axis=0)
    den = (linalg.norm(C)+abs(values)*linalg.norm(A1)+abs(values)**2*linalg.norm(A2))*linalg.norm(vectors, axis=0)
    errors = num/np.maximum(den, np.finfo(float).tiny)
    keep = np.isfinite(values) & (errors <= residual_tol)
    if real_bounds is not None:
        lo, hi = real_bounds
        if not lo < hi:
            raise ValueError("real_bounds must be increasing")
        keep &= (values.real > lo) & (values.real < hi)
    vectors = vectors[:, keep]
    vectors /= linalg.norm(vectors, axis=0)
    return Spectrum(values[keep], vectors, errors[keep])


def derivatives(problem, alpha, spectrum, index):
    """Analytic omega' and omega'' for a simple eigenvalue, paper (3.19–3.20).

    A bordered linear solve fixes q^H q'=0 and replaces the expensive pseudoinverse.
    scipy.linalg.eig returns matched left and right vectors in a single QZ solve.
    """
    if spectrum.left is None:
        raise ValueError("Call temporal(..., left=True) first")
    omega = spectrum.values[index]
    q, p = spectrum.vectors[:, index], spectrum.left[:, index]
    Bq = problem.B@q
    denom = np.vdot(p, Bq)
    if abs(denom) < 1e-12*linalg.norm(p)*linalg.norm(Bq):
        raise RuntimeError("Ill-conditioned or repeated eigenvalue; derivative is unreliable")
    Ap = problem.A1+2*alpha*problem.A2
    first = np.vdot(p, Ap@q)/denom
    K = Ap-first*problem.B
    M = problem.matrix(alpha)-omega*problem.B
    bordered = np.block([[M, -Bq[:, None]], [q.conj()[None, :], np.zeros((1, 1))]])
    dq = linalg.solve(bordered, np.r_[-K@q, 0], check_finite=False)[:-1]
    second = np.vdot(p, 2*K@dq+2*problem.A2@q)/denom
    return complex(first), complex(second)


@dataclass
class Saddle:
    alpha: complex
    omega: complex
    group_velocity: complex
    converged: bool
    history: list


def saddle(problem, alpha, omega_target=None, tol=1e-10, max_iter=100,
           max_step=0.01, imag_limit=1.0):
    """Capped complex Newton search; convergence tests abs(omega'), both components.

    Frequency prediction follows a branch between small steps. A converged saddle
    is a candidate only: Briggs–Bers pinching must be established separately.
    """
    if not 0 < tol or not np.isfinite(tol) or max_iter < 1 or not 0 < max_step or not np.isfinite(max_step):
        raise ValueError("Invalid saddle search controls")
    history = []
    target = omega_target
    for iteration in range(max_iter):
        spec = temporal(problem, alpha, left=True)
        k = spec.select(target, imag_limit)
        omega = complex(spec.values[k])
        first, second = derivatives(problem, alpha, spec, k)
        history.append(dict(iteration=iteration, alpha=complex(alpha), omega=omega,
                            group_velocity=first, residual=float(spec.residuals[k])))
        if abs(first) < tol:
            return Saddle(complex(alpha), omega, first, True, history)
        if abs(second) < 1e-14 or not np.isfinite(second):
            raise RuntimeError("Saddle curvature vanishes; choose another seed")
        if iteration == max_iter-1:
            break
        step = -first/second
        step *= min(1.0, max_step/abs(step))
        target = omega+first*step
        alpha += step
    return Saddle(complex(alpha), omega, first, False, history)


def track_temporal(problem, alphas, omega_target=None, imag_limit=1.0):
    """Continue one branch over an ordered alpha path; report the full complex data."""
    rows = []
    target = omega_target
    for alpha in alphas:
        spec = temporal(problem, alpha)
        k = spec.select(target, imag_limit)
        target = complex(spec.values[k])
        rows.append((complex(alpha), target, float(spec.residuals[k])))
    return rows
