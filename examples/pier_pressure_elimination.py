"""Independent pressure-elimination audit of a saved general-Ro parallel-mean wave.

Diagnostic only: this is a second formulation, not a fitted replacement for Pier's
published results. It reports an eigenfrequency at a prescribed complex alpha_hat;
no saddle or absolute-instability claim is made by this script.

Uses wall-normal velocity and eta = (i ell/R)u - (i k)v. For each harmonic,
q2 = k*k + ell*ell/R**2 - i*k/R, continuity gives
u = (i*k*D w - i*ell/R*eta)/q2,
v = (i*ell/R*D w + (i*k+1/R)*eta)/q2.
Pressure is eliminated by applying [-a D, -b D, -q2 I] and [b I, -g I, 0]
to the momentum equations, where g=i*k, a=g+1/R, b=i*ell/R.
The finite domain imposes w=Dw=eta=0 at both ends; increase its length and N.
Raw momentum blocks are assembled separately from FloquetProblem to cross-check
that solver, while the laminar BVP and nonlinear primary solver are shared.
The corrected analytic viscous projection is shared with the new time solver;
see docs/PIER_TIME_VALIDATION.md for the derivative correction and current results.
"""
import argparse
import json
from pathlib import Path
from scipy import sparse
from scipy.sparse.linalg import norm as sparse_norm
from scipy.sparse.linalg import splu, LinearOperator, eigs
import numpy as np
from scipy.interpolate import BarycentricInterpolator
from numpy.polynomial.legendre import leggauss
from numpy.polynomial import Polynomial
from llsa.secondary import PierOperators, PrimaryWave, WaveSolver
from llsa.secondary.discretization import PierGrid, sample_inertial_base
from llsa.secondary.time_domain import viscous_block

def finite_op(n, Z, ro=-1., radius=510., beta=68.):
    """Finite mapped domain for the independent pressure-elimination diagnostic."""
    if not np.isfinite(Z) or Z <= 40:
        raise ValueError('This far-field audit requires zmax > 40')
    op = PierOperators(n=n, mean_mode='parallel',ro=ro,radius=radius,beta=beta)
    xi = op.grid.xi
    (a, b, c) = (2, 0.6, 0.5)
    weights = (-1.0) ** np.arange(n + 1)
    weights[[0, -1]] *= 0.5
    dd = xi[:, None] - xi[None, :]
    np.fill_diagonal(dd, 1)
    Dx = weights[None, :] / weights[:, None] / dd
    np.fill_diagonal(Dx, 0)
    np.fill_diagonal(Dx, -Dx.sum(axis=1))

    def mp(x):
        t = b * x + (1 - b) * (x ** 3 + c * (1 - x * x))
        tp = b + (1 - b) * (3 * x * x - 2 * c * x)
        tpp = (1 - b) * (6 * x - 2 * c)
        den = 1 - t + a / Z * (1 + t)
        dp = (-1 + a / Z) * tp
        xp = den ** 2 / (2 * a * tp)
        xpp = xp * (den * dp / (a * tp) - den ** 2 * tpp / (2 * a * tp ** 2))
        return (a * (1 + t) / den, xp, xpp, 2 * a * tp / den ** 2)
    (z, xp, xpp, _) = mp(xi)
    D = xp[:, None] * Dx
    D2 = xp[:, None] ** 2 * (Dx @ Dx) + xpp[:, None] * Dx
    (xq, wq) = leggauss(2 * n + 80)
    I = BarycentricInterpolator(xi, np.eye(n + 1))(xq)
    mass = I.T @ ((wq * mp(xq)[3])[:, None] * I)
    op.grid = PierGrid(z, xi, D, D2, mass, a, b, c)
    # Repeated mapped D products lose the highest polynomial terms and can
    # create spurious growing viscous modes. Differentiate the rational mapping
    # exactly, then form the physical fourth derivative by the chain rule.
    transform = Polynomial([c*(1-b), b, -c*(1-b), 1-b])
    numerator = (Polynomial([1+a/Z])+(-1+a/Z)*transform)**2
    denominator = 2*a*transform.deriv()
    p = numerator(xi)/denominator(xi)
    dp = (numerator.deriv()(xi)-denominator.deriv()(xi)*p)/denominator(xi)
    ddp = (numerator.deriv(2)(xi)-denominator.deriv(2)(xi)*p-2*denominator.deriv()(xi)*dp)/denominator(xi)
    dddp = (numerator.deriv(3)(xi)-3*denominator.deriv(2)(xi)*dp-3*denominator.deriv()(xi)*ddp)/denominator(xi)
    op.fourth_derivative = (p[:, None]**4*(Dx@Dx@Dx@Dx)+6*(p**3*dp)[:, None]*(Dx@Dx@Dx)+
                            (7*p**2*dp**2+4*p**3*ddp)[:, None]*(Dx@Dx)+
                            (p*dp**3+4*p**2*dp*ddp+p**3*dddp)[:, None]*Dx)
    op.base = sample_inertial_base(z,ro)
    return op

def primary(old, n, Z):
    H = old.harmonics
    op = finite_op(n, Z,ro=old.operators.ro,radius=old.operators.radius,beta=old.operators.beta)
    z = op.grid.z
    t = (z - 2) / (z + 2)
    x = np.zeros_like(z)
    for (ii, tv) in enumerate(t):
        rr = np.roots([0.4, -0.2, 0.6, 0.2 - tv])
        x[ii] = [v.real for v in rr if abs(v.imag) < 1e-08 and -1.00001 <= v.real <= 1.00001][0]
    q = BarycentricInterpolator(old.operators.grid.xi, old.modes.reshape(2 * H + 1, 4, -1), axis=2)(x).reshape(2 * H + 1, -1)
    return WaveSolver(op, H).solve(PrimaryWave(op, old.alpha, old.omega, q, 1, []), frequency=old.omega)

def raw_linear(op, k, ell):
    (R, I, m) = (op.radius, op.I, op.m)
    D = op.grid.D
    D2 = op.grid.D2
    (U, V, W, dU, dV, dW) = op.base
    scalar = np.diag(1j * k * U + 1j * ell * V / R + (k * k + ell * ell / R ** 2 - 1j * k / R) / R) + W[:, None] * D / R - D2 / R
    A = np.zeros((3 * m, 3 * m), complex)
    for v in range(3):
        A[v * m:(v + 1) * m, v * m:(v + 1) * m] = scalar + np.diag(dW / R if v == 2 else U / R + 1 / R ** 3)
    A[:m, m:2 * m] = np.diag(-2 * V / R + 2j * ell / R ** 3)
    A[m:2 * m, :m] = np.diag(2 * V / R - 2j * ell / R ** 3)
    A[:m, 2 * m:] = np.diag(dU)
    A[m:2 * m, 2 * m:] = np.diag(dV)
    return A

def raw_coupling(op, q, h, alpha, k, ell):
    (R, m, D) = (op.radius, op.m, op.grid.D)
    u = q.reshape(4, m)[:3]
    du = u @ D.T
    C = np.zeros((3 * m, 3 * m), complex)
    adv = np.diag(1j * k * u[0] + 1j * ell / R * u[1]) + u[2, :, None] * D
    for v in range(3):
        row = slice(v * m, (v + 1) * m)
        C[row, row] += adv
        C[row, :m] += np.diag(1j * h * alpha * u[v])
        C[row, m:2 * m] += np.diag(1j * h * op.beta / R * u[v])
        C[row, 2 * m:] += np.diag(du[v])
    C[:m, m:2 * m] -= np.diag(2 * u[1] / R)
    C[m:2 * m, :m] += np.diag(u[1] / R)
    C[m:2 * m, m:2 * m] += np.diag(u[0] / R)
    return C

class Vorticity:

    def __init__(self, w, K, beta_hat=20.):
        (self.w, self.K) = (w, K)
        self.R = w.operators.radius
        self.beta_hat = float(beta_hat)
        if not np.isfinite(self.beta_hat):
            raise ValueError("Finite beta_hat required")

    def matrices(self, ahat):
        (w, K) = (self.w, self.K)
        op = w.operators
        m = op.m
        R = op.radius
        D = op.grid.D
        I = op.I
        Z = np.zeros_like(I)
        H = w.harmonics
        L = []
        T = []
        B = []
        for j in range(-K, K + 1):
            k = ahat + j * w.alpha
            ell = self.beta_hat + j * op.beta
            g = 1j * k
            a = g + 1 / R
            b = 1j * ell / R
            qs = -(a * g + b * b)
            if abs(qs)<1e-10:
                raise ValueError("Degenerate wavevector needs a separate pressure elimination")
            T.append(np.block([[g * D / qs, -b * I / qs], [b * D / qs, a * I / qs], [I, Z]]))
            L.append(np.block([[-a * D, -b * D, -qs * I], [b * I, -g * I, Z]]))
            mass = np.block([[op.grid.D2 - qs * I, Z], [Z, I]]).astype(complex)
            mass[[0, 1, m - 2, m - 1, m, 2 * m - 1]] = 0
            B.append(sparse.csc_matrix(1j * mass))
        blocks = []
        for j in range(-K, K + 1):
            row = []
            for k in range(-K, K + 1):
                h = j - k
                kr = ahat + k * w.alpha
                ell = 20 + k * op.beta
                if abs(h) > H and j != k:
                    row.append(None)
                    continue
                A = raw_coupling(op, w.modes[h + H], h, w.alpha, kr, ell) if abs(h) <= H else np.zeros((3 * m, 3 * m), complex)
                if j == k:
                    A += raw_linear(op, kr, ell) - 1j * k * w.omega / R * np.eye(3 * m)
                C = L[j + K] @ A @ T[k + K]
                if j == k:
                    g, a, b = 1j*kr, 1j*kr+1/R, 1j*ell/R
                    qs = -(a*g+b*b)
                    scalar = op.grid.D2-qs*I
                    lap = np.block([[scalar-I/R**2, -2j*ell/R**2*I, Z],
                                    [2j*ell/R**2*I, scalar-I/R**2, Z], [Z, Z, scalar]])
                    # Replace the old projected viscous product with its exact
                    # chain-rule form. Advection/coupling remains independently assembled.
                    C += (L[j+K]@lap@T[k+K]-viscous_block(
                        D, op.grid.D2, op.fourth_derivative, R, g, a, b, qs, ell))/R
                C[[0, 1, m - 2, m - 1, m, 2 * m - 1]] = 0
                if j == k:
                    C[0, 0] = 1
                    C[m - 1, m - 1] = 1
                    C[1, :m] = D[0]
                    C[m - 2, :m] = D[-1]
                    C[m, m] = 1
                    C[-1, -1] = 1
                row.append(sparse.csc_matrix(C))
            blocks.append(row)
        return (sparse.bmat(blocks, format='csc'), sparse.block_diag(B, format='csc'))

    def eig(self, a, target=8.8 + 1.17j):
        (A, B) = self.matrices(a)
        sig = target / self.R
        lu = splu(A - sig * B)
        mat = LinearOperator(A.shape, matvec=lambda v: lu.solve(B @ v), dtype=complex)
        (mu, q) = eigs(mat, k=4, which='LM', tol=1e-10, v0=np.ones(A.shape[0]))
        vals = (sig + 1 / mu) * self.R
        index = int(np.argmin(abs(vals - target)))
        lam = vals[index] / self.R
        mode = q[:, index]
        self.last_residual = float(np.linalg.norm(A @ mode - lam * (B @ mode)) / ((sparse_norm(A) + abs(lam) * sparse_norm(B)) * np.linalg.norm(mode)))
        if not np.isfinite(self.last_residual) or self.last_residual > 1e-08:
            raise RuntimeError('Pressure-eliminated eigenvalue failed the matrix residual check')
        return vals[index]

def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--wave', type=Path, required=True)
    parser.add_argument('--n', type=int, nargs='+', default=[80, 100])
    parser.add_argument('--zmax', type=float, nargs='+', default=[320, 640])
    parser.add_argument('--secondary-harmonics', type=int, default=5)
    parser.add_argument('--beta-hat', type=float, default=20.)
    parser.add_argument('--alpha-hat', type=complex, default=-0.00163221046 - 0.03698086207j)
    parser.add_argument('--target', type=complex, default=8.607 + 0.856j)
    parser.add_argument('--output', type=Path, default=Path('results/pier2003/pressure_elimination.json'))
    args = parser.parse_args()
    old = PrimaryWave.load(args.wave)
    if (old.operators.grid.a, old.operators.grid.b, old.operators.grid.c) != (2., .6, .5):
        parser.error('Source wave must use the reference semi-infinite mapping a=2, b=0.6, c=0.5')
    if old.operators.mean_mode != 'parallel':
        parser.error('This pressure-elimination diagnostic currently requires a parallel-mean primary')
    if len(args.n) != len(args.zmax):
        parser.error('Require one zmax for each N')
    records = []
    for (n, Z) in zip(args.n, args.zmax):
        wave = primary(old, n, Z)
        problem = Vorticity(wave, args.secondary_harmonics,args.beta_hat)
        omega = problem.eig(args.alpha_hat, args.target)
        record = dict(n=n, zmax=Z, H=wave.harmonics, K=args.secondary_harmonics, alpha=wave.alpha, alpha_hat=[args.alpha_hat.real, args.alpha_hat.imag], omega_hat=[float(omega.real), float(omega.imag)], matrix_residual=problem.last_residual)
        records.append(record)
        record.update(ro=wave.operators.ro,re=wave.operators.re,radius=wave.operators.radius,
                      beta=wave.operators.beta,beta_hat=args.beta_hat,primary_omega=wave.omega)
        print(json.dumps(record), flush=True)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(dict(status='independent pressure elimination; frequency at prescribed alpha_hat; not a saddle search', shared_components='laminar BVP and nonlinear primary solver', records=records), indent=2) + '\n')
    return 0
if __name__ == '__main__':
    raise SystemExit(main())
