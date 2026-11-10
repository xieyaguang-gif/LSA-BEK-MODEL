"""Sparse Fourier–Floquet pencil and complex secondary saddle search, Pier §7."""
from dataclasses import dataclass

import numpy as np
from scipy import sparse
from scipy.sparse import linalg as sla


class FloquetProblem:
    def __init__(self, wave, beta_hat=20.0, harmonics=3, wall_condition="pressure"):
        if wall_condition not in ("pressure", "continuity"):
            raise ValueError("wall_condition must be pressure or continuity")
        # These are equivalent continuum wall conditions, but the equal-order
        # collocation pressure closure can leave a large wall continuity defect.
        # Keep the original choice explicit for reproducibility of old results.
        self.wall_condition = wall_condition
        if int(harmonics) != harmonics or harmonics < 1 or not np.isfinite(beta_hat):
            raise ValueError("Require secondary harmonics>=1 and finite beta_hat")
        if not np.isfinite(wave.residual) or wave.residual > 1e-7:
            raise ValueError("A converged nonlinear primary wave is required")
        from .primary import WaveSolver
        checked = np.max(abs(WaveSolver(wave.operators, wave.harmonics).residual(
            wave.modes, wave.alpha, wave.omega)))
        if not np.isfinite(checked) or checked > 1e-7:
            raise ValueError(f"Primary wave fails nonlinear residual check: {checked}")
        self.wave, self.beta_hat, self.K = wave, float(beta_hat), int(harmonics)
        self.radius = wave.operators.radius
        nblocks = 2*self.K+1
        self.B = sparse.block_diag([1j*wave.operators.M]*nblocks, format='csc')
        self.A0 = self._matrix(0)
        # Build exact polynomial coefficients: finite-difference assembly would
        # leave dense roundoff entries in the otherwise diagonal quadratic term.
        op, H, m = wave.operators, wave.harmonics, wave.operators.m
        U = op.base[0]
        rows = []
        for j in range(-self.K, self.K+1):
            row = []
            for k in range(-self.K, self.K+1):
                l = j-k
                if abs(l) > H and j != k:
                    row.append(None)
                    continue
                C = np.zeros((op.size, op.size), complex)
                if abs(l) <= H:
                    ur = wave.modes[l+H, :m]
                    for v in range(3):
                        C[v*m:(v+1)*m, v*m:(v+1)*m] = np.diag(1j*ur)
                if j == k:
                    scalar = np.diag(1j*U+(2*k*wave.alpha-1j/op.radius)/op.radius)
                    for v in range(3):
                        C[v*m:(v+1)*m, v*m:(v+1)*m] += scalar
                    C[:m, 3*m:] = 1j*op.I
                    C[3*m:, :m] = 1j*op.I
                C[op.boundary_rows] = 0
                row.append(sparse.csc_matrix(C))
            rows.append(row)
        self.A1 = sparse.bmat(rows, format='csc')
        self.A2 = sparse.block_diag([op.M/op.radius]*nblocks, format='csc')

    def _matrix(self, alpha_hat):
        wave, K = self.wave, self.K
        op, H = wave.operators, wave.harmonics
        blocks = []
        for j in range(-K, K+1):
            row = []
            for k in range(-K, K+1):
                l = j-k
                if abs(l) > H and j != k:
                    row.append(None)
                    continue
                kr, kt = alpha_hat+k*wave.alpha, self.beta_hat+k*op.beta
                C = (op.coupling(wave.modes[l+H], l, wave.alpha, kr, kt)
                     if abs(l) <= H else np.zeros((op.size, op.size), complex))
                if j == k:
                    C += op.linear(kr, kt, mean_closure=False)-1j*k*wave.omega/op.radius*op.M
                    if self.wall_condition == "continuity":
                        m = op.m
                        C[3*m] = 0
                        C[3*m, 2*m:3*m] = op.grid.D[0]
                row.append(sparse.csc_matrix(C))
            blocks.append(row)
        return sparse.bmat(blocks, format='csc')

    def matrix(self, alpha_hat):
        if not np.isfinite(alpha_hat):
            raise ValueError("alpha_hat must be finite")
        return (self.A0+alpha_hat*self.A1+alpha_hat**2*self.A2).tocsc()

    def fields(self, vector, phases):
        """Complex secondary eigenfunction components (normalization arbitrary)."""
        modes = vector.reshape(2*self.K+1, 4, self.wave.operators.m)
        ns = np.arange(-self.K, self.K+1)
        return np.einsum('ph,hvz->pvz', np.exp(1j*np.asarray(phases)[:, None]*ns), modes)


@dataclass
class SecondarySpectrum:
    omega: np.ndarray
    vectors: np.ndarray
    residuals: np.ndarray

    def nearest(self, target):
        if not len(self.omega):
            raise RuntimeError("No secondary eigenvalue passed the residual filter")
        return int(np.argmin(abs(self.omega-target)))

    def tracked(self, target, reference_vector=None):
        """Select the spectral branch by frequency and eigenvector continuity.

        Eigenvalue proximity alone is unreliable when secondary branches pass
        close to one another.  A supplied reference vector therefore selects
        the candidate with the largest normalized overlap; frequency distance
        is used as a weak tie-breaker.
        """
        if reference_vector is None:
            return self.nearest(target)
        ref = np.asarray(reference_vector)
        if ref.shape != (self.vectors.shape[0],) or not np.all(np.isfinite(ref)):
            raise ValueError("reference_vector has the wrong shape or non-finite entries")
        den = np.linalg.norm(ref)*np.linalg.norm(self.vectors, axis=0)
        overlap = abs(ref.conj()@self.vectors)/np.maximum(den, np.finfo(float).tiny)
        distance = abs(self.omega-target)
        scale = max(1., float(np.median(distance)))
        score = overlap-1e-3*distance/scale
        return int(np.argmax(score))


def secondary_spectrum(problem, alpha_hat=0, target=8.5+1.2j, count=6, residual_tol=1e-8):
    """Targeted eigenvalues in Omega_BEK units (Pier's disk units at Ro=-1)."""
    A, B = problem.matrix(alpha_hat), problem.B
    if int(count) != count or not 1 <= count < A.shape[0]-1:
        raise ValueError("Invalid number of requested modes")
    sigma = target/problem.radius
    lu = sla.splu(A-sigma*B)
    operator = sla.LinearOperator(A.shape, matvec=lambda q: lu.solve(B@q), dtype=complex)
    mu, q = sla.eigs(operator, k=count, which='LM', tol=1e-10,
                     v0=np.ones(A.shape[0], complex))
    lam = sigma+1/mu
    residuals = np.linalg.norm(A@q-(B@q)*lam, axis=0)/(
        (sla.norm(A)+abs(lam)*sla.norm(B))*np.linalg.norm(q, axis=0))
    keep = np.isfinite(lam) & (residuals < residual_tol)
    return SecondarySpectrum(lam[keep]*problem.radius, q[:, keep], residuals[keep])


def secondary_derivatives(problem, alpha_hat, spectrum, index):
    """Paired adjoint derivatives of omega_hat with respect to alpha_hat."""
    omega, q = spectrum.omega[index], spectrum.vectors[:, index]
    A, B = problem.matrix(alpha_hat), problem.B
    M = A-omega/problem.radius*B
    Bq = B@q
    column = sparse.csc_matrix(q[:, None])
    row = sparse.csc_matrix(Bq.conj()[None, :])
    adjoint_border = sparse.bmat([[M.conj().T, column], [row, None]], format='csc')
    p = sla.spsolve(adjoint_border, np.r_[np.zeros(len(q)), 1])[:-1]
    den = np.vdot(p, Bq)
    if not np.all(np.isfinite(p)) or abs(den) < 1e-12:
        raise RuntimeError("Secondary eigenvalue has an ill-conditioned adjoint normalization")
    Ap = problem.A1+2*alpha_hat*problem.A2
    first = np.vdot(p, Ap@q)/den
    K = Ap-first*B
    border = sparse.bmat([[M, sparse.csc_matrix(-Bq[:, None])],
                          [sparse.csc_matrix(q.conj()[None, :]), None]], format='csc')
    dq = sla.spsolve(border, np.r_[-K@q, 0])[:-1]
    second = np.vdot(p, 2*K@dq+2*problem.A2@q)/den
    if not np.isfinite(first) or not np.isfinite(second):
        raise RuntimeError("Secondary eigenvalue derivatives are not finite")
    return complex(first*problem.radius), complex(second*problem.radius)


@dataclass
class SecondarySaddle:
    alpha_hat: complex
    omega_hat: complex
    group_velocity: complex
    residual: float
    converged: bool
    history: list
    vector: np.ndarray


def secondary_saddle(problem, alpha_hat=-.0012-.0369j, target=8.52+1.16j,
                     tol=1e-7, max_iter=25, max_step=.01, callback=None,
                     reference_vector=None, count=6):
    if tol <= 0 or max_iter < 1 or max_step <= 0:
        raise ValueError("Invalid saddle iteration controls")
    history = []
    for iteration in range(max_iter):
        spec = secondary_spectrum(problem, alpha_hat, target=target+1e-5j, count=count)
        k = spec.tracked(target, reference_vector)
        omega = complex(spec.omega[k])
        first, second = secondary_derivatives(problem, alpha_hat, spec, k)
        record = dict(iteration=iteration, alpha_hat=[alpha_hat.real, alpha_hat.imag],
                      omega_hat=[omega.real, omega.imag], group_velocity=abs(first),
                      residual=float(spec.residuals[k]))
        history.append(record)
        if callback:
            callback(record)
        if abs(first) < tol or iteration == max_iter-1:
            return SecondarySaddle(complex(alpha_hat), omega, first, float(spec.residuals[k]),
                                   abs(first) < tol, history, spec.vectors[:, k])
        if abs(second) < 1e-12:
            raise RuntimeError("Vanishing saddle curvature")
        step = -first/second
        step *= min(1, max_step/abs(step))
        target = omega+first*step
        alpha_hat += step
        reference_vector = spec.vectors[:, k]


def secondary_spatial_near(problem, omega_hat, alpha_shift, count=4):
    """Quadratic spatial Floquet problem, using an original-size sparse LU."""
    C, A1, A2 = problem.A0-omega_hat/problem.radius*problem.B, problem.A1, problem.A2
    m, sigma = C.shape[0], alpha_shift
    lu = sla.splu(C+sigma*A1+sigma*sigma*A2)
    H = A1+sigma*A2

    def apply(v):
        x = -lu.solve(H@v[:m]+A2@v[m:])
        return np.r_[x, v[:m]+sigma*x]

    op = sla.LinearOperator((2*m, 2*m), matvec=apply, dtype=complex)
    mu, q = sla.eigs(op, k=count, which='LM', tol=1e-10, v0=np.ones(2*m, complex))
    alpha = sigma+1/mu
    q = q[:m]
    q /= np.linalg.norm(q, axis=0)
    errors = np.linalg.norm(C@q+(A1@q)*alpha+(A2@q)*alpha**2, axis=0)/(
        sla.norm(C)+abs(alpha)*sla.norm(A1)+abs(alpha)**2*sla.norm(A2))
    keep = np.isfinite(alpha) & (errors < 1e-8)
    return alpha[keep], q[:, keep], errors[keep]


def secondary_pinch(problem, saddle, steps=45, eta_min=1e-5, eta_max=8):
    """Track two distinct roots from a saddle towards larger Im(omega_hat).

    Records a finite-path numerical pinch diagnostic, not an exhaustive topology
    proof. Positive saddle growth alone does not satisfy this diagnostic.
    """
    if not saddle.converged or steps < 3 or not 0 < eta_min < eta_max:
        raise ValueError("Converged saddle and a valid upward frequency path required")
    previous, records = None, []
    for eta in np.geomspace(eta_min, eta_max, steps):
        omega = saddle.omega_hat+1j*eta
        if previous is None:
            roots, _, errors = secondary_spatial_near(problem, omega, saddle.alpha_hat)
            if len(roots) < 2:
                raise RuntimeError("Cannot resolve two roots near the saddle")
            indices = np.argsort(abs(roots-saddle.alpha_hat))[:2]
            pair, errs = roots[indices], errors[indices]
        else:
            pair, errs = [], []
            for old in previous:
                roots, _, errors = secondary_spatial_near(problem, omega, old)
                if not len(roots):
                    raise RuntimeError("Spatial branch was lost")
                k = np.argmin(abs(roots-old))
                pair.append(roots[k])
                errs.append(errors[k])
            pair = np.array(pair)
        if abs(pair[0]-pair[1]) < 1e-8:
            raise RuntimeError("Both trackers selected the same root; refine frequency steps")
        records.append(dict(eta=float(eta), omega_hat=[omega.real, omega.imag],
                            alpha_branches=[[v.real, v.imag] for v in pair],
                            max_residual=float(max(errs))))
        previous = pair
    return dict(opposite_half_planes_at_end=bool(pair[0].imag*pair[1].imag < 0),
                path=records, classification="finite-path numerical diagnostic")
