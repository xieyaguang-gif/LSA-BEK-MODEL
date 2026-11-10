"""Experimental nonlinear PSE for disk-stationary, azimuthally periodic waves.

Radial envelopes are marched, not replaced by local saturated travelling waves.
The slow-envelope radial pressure derivative and second radial derivative are
discarded. This pressure regularization is a modelling approximation, so results
require step, wall-normal and Fourier convergence checks against DNS.
"""
from dataclasses import dataclass

import numpy as np
from scipy import linalg

from .operators import PierOperators
from .primary import pack, unpack


def spatial_mode(op, omega, target):
    """Quadratic spatial eigenproblem in the inertial exp(i alpha r-i omega t) convention."""
    A0 = op.linear(0, op.beta, mean_closure=False)-1j*omega/op.radius*op.M
    Ap = op.linear(1, op.beta, mean_closure=False)
    Am = op.linear(-1, op.beta, mean_closure=False)
    A1 = (Ap-Am)/2
    A2 = (Ap+Am)/2-op.linear(0, op.beta, mean_closure=False)
    s = op.size
    zero, eye = np.zeros((s, s)), np.eye(s)
    values, vectors = linalg.eig(np.block([[zero, eye], [-A0, -A1]]),
                                 np.block([[eye, zero], [zero, A2]]), check_finite=False)
    good = np.isfinite(values) & (np.abs(values) < 20)
    values, vectors = values[good], vectors[:s, good]
    idx = np.argmin(abs(values-target))
    alpha, q = values[idx], vectors[:, idx]
    residual = np.linalg.norm((A0+alpha*A1+alpha**2*A2)@q)/np.linalg.norm(q)
    return alpha, q, float(residual)


@dataclass
class SpatialStep:
    radius: float
    alpha: complex
    modes: np.ndarray
    residual: float
    iterations: int


class SpatialPSE:
    """Adaptive complex carrier, exponentially fitted BDF2 envelope marching.

    q velocities = physical velocities/r; q pressure = physical pressure/r**2.
    Therefore the extra scale derivatives must be included in addition to the
    frozen-radius Pier operator. Negative harmonics are exact conjugates.
    The mean axial velocity is free at infinity, as required by mean continuity.
    """
    def __init__(self, *, ro=-.751, beta=42., n=32, harmonics=3,
                 nonparallel=True, nonlinear=True):
        if harmonics < 1 or int(harmonics) != harmonics or ro == 0:
            raise ValueError("Require positive integer harmonics and nonzero Ro")
        self.ro, self.beta, self.n, self.H = ro, beta, n, int(harmonics)
        self.nonparallel, self.use_nonlinear = nonparallel, nonlinear

    def operators(self, radius):
        return PierOperators(radius=radius, beta=self.beta, ro=self.ro, n=self.n)

    def matrices(self, op, alpha, j):
        R, m, s = op.radius, op.m, op.size
        k = j*alpha.real+1j*abs(j)*alpha.imag
        omega = self.beta*op.disk_rotation
        A = op.linear(k, j*self.beta)-1j*j*omega/R*op.M
        B = np.zeros((s, s), complex)
        F = op.base[0]
        for v in range(3):
            sl = slice(v*m, (v+1)*m)
            B[sl, sl] = np.diag(F-(2j*k+1/R)/R)
            if self.nonparallel:
                A[sl, sl] += np.diag(F/R-2j*k/R**2-1/R**3)
                B[sl, sl] -= 2/R**2*op.I
        # Drop ONLY the slow pressure-envelope derivative; retain carrier pressure.
        B[3*m:, :m] = op.I
        if self.nonparallel:
            A[:m, 3*m:] += 2/R*op.I
            A[3*m:, :m] += op.I/R
        A[op.boundary_rows] = op.linear(k, j*self.beta)[op.boundary_rows]
        B[op.boundary_rows] = 0
        return A, B

    def initialize(self, radius, amplitude, z, target=.4-.07j):
        """Amplitude is sqrt(2)|v1/(Omega_disk*r)|, matching the DNS CSV."""
        if not np.isfinite(amplitude) or amplitude <= 0 or not np.isfinite(z) or z <= 0:
            raise ValueError("Positive finite inlet amplitude and probe height required")
        op = self.operators(radius)
        alpha, q, residual = spatial_mode(op, self.beta*op.disk_rotation, target)
        # Local polynomial evaluation in the computational coordinate.
        from scipy.interpolate import BarycentricInterpolator
        from scipy.optimize import brentq
        t = (z-op.grid.a)/(z+op.grid.a)
        xi = brentq(lambda x: op.grid.b*x+(1-op.grid.b)*(x**3+op.grid.c*(1-x*x))-t, -1, 1)
        v = BarycentricInterpolator(op.grid.xi, q.reshape(4, op.m)[1])(xi)
        factor = np.sqrt(2)*abs(v)/op.disk_rotation
        q *= amplitude/factor
        modes = np.zeros((2*self.H+1, op.size), complex)
        modes[self.H+1], modes[self.H-1] = q, q.conj()
        return SpatialStep(radius, alpha, modes, residual, 0)

    def advance(self, previous, radius, older=None, *, tol=1e-10, max_iter=40,
                check_jacobian=False):
        op, H = self.operators(radius), self.H
        if previous.modes.shape != (2*H+1, op.size):
            raise ValueError("Previous state has incompatible Fourier/wall-normal resolution")
        dr = radius-previous.radius
        if dr <= 0:
            raise ValueError("March toward increasing radius")
        alpha = previous.alpha
        ns = np.arange(-H, H+1)
        if older is None:
            d0 = 1/dr
        else:
            old_dr = previous.radius-older.radius
            if not np.isclose(old_dr, dr):
                raise ValueError("BDF2 currently requires uniform radial steps")
            d0 = 1.5/dr
        m, s, D = op.m, op.size, op.grid.D
        boundary = op.boundary_rows
        reference = previous.modes[H+1].reshape(4, m)[:3]
        weight = reference.conj()@op.grid.mass
        weight /= np.sqrt(np.sum(weight*reference).real)

        def evaluate(modes, carrier, jacobian=False):
            rates = -abs(ns)*carrier.imag
            predicted = np.exp(rates[:, None]*dr)*previous.modes
            history = (predicted/dr if older is None else
                (2*predicted-.5*np.exp(rates[:, None]*2*dr)*older.modes)/dr)
            As, Bs = zip(*(self.matrices(op, carrier, j) for j in ns))
            matrices = np.asarray(As)+d0*np.asarray(Bs)
            forcing = np.einsum('nij,nj->ni', Bs, history)
            derivative = 1j*ns*carrier.real+rates+d0
            q = modes.reshape(2*H+1, 4, m)
            radial = derivative[:, None, None]*q-history.reshape(2*H+1, 4, m)
            if self.nonparallel:
                radial = radial+q/op.radius
            vertical = q[:, :3]@D.T
            out = np.einsum('nij,nj->ni', matrices, modes)-forcing
            nonlin = np.zeros_like(q)
            if self.use_nonlinear:
                for j in ns:
                    for k in ns:
                        l = j-k
                        if abs(l) > H:
                            continue
                        u, v = q[l+H], q[k+H]
                        nonlin[j+H, :3] += u[0]*radial[k+H, :3]+1j*k*self.beta/op.radius*u[1]*v[:3]+u[2]*vertical[k+H]
                        nonlin[j+H, 0] -= u[1]*v[1]/op.radius
                        nonlin[j+H, 1] += u[0]*v[1]/op.radius
            nonlin = nonlin.reshape(2*H+1, s)
            nonlin[:, boundary] = 0
            out += nonlin
            slow = (d0*modes[H+1]-history[H+1]).reshape(4, m)[:3]
            gauge = np.sum(weight*slow)
            F = np.r_[pack(out), gauge.real, gauge.imag]
            if not jacobian:
                return F

            def block(j, k):
                C = matrices[j+H].copy() if j == k else np.zeros((s, s), complex)
                l = j-k
                if abs(l) <= H and self.use_nonlinear:
                    u = q[l+H]
                    deriv = derivative[k+H]+(1/op.radius if self.nonparallel else 0)
                    adv = np.diag(deriv*u[0]+1j*k*self.beta/op.radius*u[1])+u[2, :, None]*D
                    N = np.zeros((s, s), complex)
                    for v in range(3):
                        sl = slice(v*m, (v+1)*m)
                        N[sl, sl] += adv
                        N[sl, :m] += np.diag(radial[l+H, v])
                        N[sl, m:2*m] += np.diag(1j*l*self.beta/op.radius*u[v])
                        N[sl, 2*m:3*m] += np.diag(vertical[l+H, v])
                    N[:m, m:2*m] -= np.diag(2*u[1]/op.radius)
                    N[m:2*m, :m] += np.diag(u[1]/op.radius)
                    N[m:2*m, m:2*m] += np.diag(u[0]/op.radius)
                    N[boundary] = 0
                    C += N
                return C

            dimension = (2*H+1)*s
            J = np.zeros((dimension+2, dimension+2))
            for j in range(H+1):
                row = slice(0, s) if j == 0 else slice((2*j-1)*s, (2*j+1)*s)
                C = block(j, 0)
                J[row, :s] = C.real if j == 0 else np.vstack((C.real, C.imag))
                for k in range(1, H+1):
                    Cp, Cm = block(j, k), block(j, -k)
                    re, im = Cp+Cm, 1j*(Cp-Cm)
                    J[row, (2*k-1)*s:(2*k+1)*s] = (np.hstack((re.real, im.real)) if j == 0 else
                        np.block([[re.real, im.real], [re.imag, im.imag]]))
            c = weight.ravel()*d0
            J[-2, s:s+3*m] = c.real
            J[-2, 2*s:2*s+3*m] = -c.imag
            J[-1, s:s+3*m] = c.imag
            J[-1, 2*s:2*s+3*m] = c.real
            h = 1e-6
            J[:, -2] = (evaluate(modes, carrier+h)-evaluate(modes, carrier-h))/(2*h)
            J[:, -1] = (evaluate(modes, carrier+1j*h)-evaluate(modes, carrier-1j*h))/(2*h)
            return F, J

        predicted = np.exp((-abs(ns)*alpha.imag)[:, None]*dr)*previous.modes
        x = np.r_[pack(predicted), alpha.real, alpha.imag]
        # Relative residual: a small inlet must not pass Newton merely because
        # every dimensional residual is small before solving the equations.
        residual_scale = max(np.max(abs(previous.modes)), np.finfo(float).tiny)
        def evaluate_vector(xx, jacobian=False):
            return evaluate(unpack(xx[:-2], H, s), xx[-2]+1j*xx[-1], jacobian)

        for it in range(max_iter):
            modes = unpack(x[:-2], H, s)
            F, J = evaluate_vector(x, True)
            if check_jacobian and it == 0:
                direction = np.random.default_rng(42).normal(size=len(x))
                h = 1e-6
                fd = (evaluate_vector(x+h*direction)-evaluate_vector(x-h*direction))/(2*h)
                error = np.linalg.norm(fd-J@direction)/np.linalg.norm(fd)
                if error > 1e-7:
                    worst = np.argsort(abs(fd-J@direction))[-8:]
                    raise AssertionError(f"Spatial Jacobian directional error {error:.3e}: "
                                         f"{list(zip(worst, (fd-J@direction)[worst]))}")
            norm = np.linalg.norm(F, np.inf)
            if norm < tol*residual_scale:
                return SpatialStep(radius, x[-2]+1j*x[-1], modes, float(norm), it)
            change = linalg.solve(J, -F, check_finite=False)
            for scale in 2.**(-np.arange(10)):
                trial = x+scale*change
                if np.linalg.norm(evaluate_vector(trial), np.inf) < norm:
                    x = trial
                    break
            else:
                raise RuntimeError(f"PSE line search at Re={op.re:.4f}, residual={norm:.3e}")
        raise RuntimeError(f"PSE Newton failed at Re={op.re:.4f}, residual={norm:.3e}")
