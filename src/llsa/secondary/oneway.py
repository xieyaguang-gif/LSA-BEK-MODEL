"""Experimental dense spectral one-way solver coupled to a global mean BVP.

This is a reference projector implementation, NOT the recursive OWNS algorithm.
Finite upstream modes are removed from the descriptor equations; infinite
eigenvalues (pressure/continuity constraints) are not discarded. The zero
harmonic is solved globally by default. An optional boundary-layer mean
closure is provided for comparison, not recommended for a sign-changing F.
"""
from dataclasses import dataclass

import numpy as np
from scipy import linalg
from scipy.optimize import linear_sum_assignment

from .primary import pack, unpack
from .radial import radial_matrices, convection, convection_blocks, real_blocks
from .global_radial import RadialSolution, GlobalRadial


def descriptor(A, B, C):
    s = len(A)
    Z, I = np.zeros_like(A), np.eye(s)
    return np.block([[I, Z], [Z, C]]), np.block([[Z, I], [-A, -B]])


def finite_spectrum(E, L):
    ab, left, right = linalg.eig(L, E, left=True, homogeneous_eigvals=True, check_finite=False)
    finite = abs(ab[1]) > 1e-12*np.maximum(abs(ab[0])+abs(ab[1]), 1e-30)
    lam = ab[0, finite]/ab[1, finite]
    return lam, left[:, finite], right[:, finite]


def match_modes(old_values, old_vectors, values, vectors):
    if len(old_values) != len(values):
        raise RuntimeError("Finite pencil dimension changed; cannot continue causal branches")
    a = old_vectors/np.linalg.norm(old_vectors, axis=0)
    b = vectors/np.linalg.norm(vectors, axis=0)
    distance = abs(old_values[:, None]-values[None, :])/(1+abs(old_values[:, None]))
    cost = distance+.25*(1-np.minimum(abs(a.conj().T@b), 1))
    row, col = linear_sum_assignment(cost)
    return col[np.argsort(row)]


@dataclass
class SpectralProjector:
    E: np.ndarray
    L: np.ndarray
    equation: np.ndarray
    state: np.ndarray
    values: np.ndarray
    vectors: np.ndarray
    upstream: np.ndarray
    diagnostics: dict
    upstream_constraints: np.ndarray

    @classmethod
    def from_labels(cls, E, L, values, left, right, upstream):
        V, W = right[:, upstream], left[:, upstream].conj().T
        G = W@E@V
        # Normalize left rows before solving the biorthogonality matrix.
        norms = np.maximum(np.linalg.norm(G, axis=1), 1e-300)
        G, W = G/norms[:, None], W/norms[:, None]
        condition = float(np.linalg.cond(G)) if len(G) else 1.
        if not np.isfinite(condition) or condition > 1e10:
            raise RuntimeError(f"Unreliable upstream spectral basis: condition={condition:.3e}")
        dual = linalg.solve(G, W, check_finite=False) if len(G) else W
        Q = np.eye(len(E))-E@V@dual
        P = np.eye(len(E))-V@dual@E
        error = float(max(np.linalg.norm(P@P-P)/max(np.linalg.norm(P), 1),
                          np.linalg.norm(Q@Q-Q)/max(np.linalg.norm(Q), 1)))
        if error > 1e-7:
            raise RuntimeError(f"Projector idempotence failed: {error:.3e}")
        constraints = dual@E
        constraints /= np.maximum(np.linalg.norm(constraints, axis=1), 1e-300)[:, None]
        return cls(E, L, Q, P, values, right, np.asarray(upstream, bool),
                   dict(finite_modes=len(values), upstream_modes=int(sum(upstream)),
                        downstream_modes=int(sum(~upstream)), basis_condition=condition,
                        projector_error=error,
                        state_projector_frobenius_norm=float(np.linalg.norm(P))), constraints)


def causal_projector(op, j, omega, *, continuation_steps=64, previous=None):
    """Track from Im(omega)>0; never classify unstable waves by growth sign.

    Initial split requires separation in both of two high-frequency planes.
    This numerical continuation must still be checked under step refinement;
    it is not a proof of a Briggs pinch classification at an absolute saddle.
    """
    if j <= 0:
        raise ValueError("Project positive harmonics only; mean requires separate closure")
    def spectrum(eta):
        E, L = descriptor(*radial_matrices(op, j, omega+1j*eta))
        return E, L, *finite_spectrum(E, L)
    if previous is None:
        eta = max(100., 8*abs(omega))
        _, _, old, _, vectors = spectrum(2*eta)
        labels = old.real > 0
        if np.min(abs(old.real)) < 1e-8:
            raise RuntimeError("Unseparated high-frequency spatial spectrum")
        levels = np.r_[eta, np.geomspace(eta, 1e-5, continuation_steps)[1:], 0.]
        for step, level in enumerate(levels):
            E, L, values, left, right = spectrum(level)
            order = match_modes(old, vectors, values, right)
            values, left, right = values[order], left[:, order], right[:, order]
            if step == 0 and not np.array_equal(labels, values.real > 0):
                raise RuntimeError("High-frequency split changed; increase continuation height")
            old, vectors = values, right
    else:
        E, L, values, left, right = spectrum(0.)
        order = match_modes(previous.values, previous.vectors, values, right)
        values, left, right = values[order], left[:, order], right[:, order]
        labels = previous.upstream.copy()
    result = SpectralProjector.from_labels(E, L, values, left, right, labels)
    result.diagnostics['classification'] = 'frequency-continuation' if previous is None else 'radial-continuation'
    return result


class SpectralOneWay:
    """BDF1 startup/BDF2 nonlinear march, full diffusion in nonzero harmonics.

    Default mean closure is a global zero-harmonic BVP coupled by outer
    iteration. Optional boundary-layer closure omits physical mean radial
    pressure gradient and radial viscosity. A low-amplitude, locally linear
    inlet is supported; arbitrary strong nonlinear inlet compatibility is
    not established here.
    """
    def __init__(self, config, *, continuation_steps=64, mean_closure='global',
                 advection_derivative='auxiliary'):
        if mean_closure not in ('global', 'boundary-layer'):
            raise ValueError("Mean closure must be global or boundary-layer")
        if int(continuation_steps) != continuation_steps or continuation_steps < 8:
            raise ValueError("At least eight integer frequency continuation steps required")
        if advection_derivative not in ('auxiliary', 'bdf'):
            raise ValueError("Advection derivative must be auxiliary or bdf")
        self.config = config
        self.continuation_steps = continuation_steps
        self.mean_closure = mean_closure
        self.advection_derivative = advection_derivative

    def initialize(self, radius, amplitude=1e-5, z=1.1858, target=.4-.07j):
        from scipy.interpolate import BarycentricInterpolator
        from scipy.optimize import brentq
        if not np.isfinite(z) or z <= 0 or not np.isfinite(target):
            raise ValueError("Positive finite probe height and finite target required")
        op = self.config.operators(radius)
        if op.disk_rotation == 0:
            raise ValueError("Disk-speed-normalized inlet amplitude is undefined at Ro=1")
        H, s = self.config.harmonics, op.size
        projectors = {j: causal_projector(op, j, self.config.frequency(op),
                        continuation_steps=self.continuation_steps) for j in range(1, H+1)}
        p = projectors[1]
        allowed = np.flatnonzero(~p.upstream)
        if not len(allowed):
            raise RuntimeError("No downstream inlet mode")
        ix = allowed[np.argmin(abs(-1j*p.values[allowed]-target))]
        y = p.vectors[:, ix].copy()
        t = (z-op.grid.a)/(z+op.grid.a)
        xi = brentq(lambda x: op.grid.b*x+(1-op.grid.b)*(x**3+op.grid.c*(1-x*x))-t, -1, 1)
        v = BarycentricInterpolator(op.grid.xi, y[op.m:2*op.m])(xi)
        if not np.isfinite(amplitude) or amplitude <= 0 or abs(v) < 1e-14:
            raise ValueError("Positive finite amplitude and nonzero probe eigenfunction required")
        y *= amplitude*op.disk_rotation/(np.sqrt(2)*abs(v))
        state = np.zeros((2*H+1, 2*s), complex)
        state[H+1], state[H-1] = y, y.conj()
        return state, projectors, complex(-1j*p.values[ix])

    def _advance(self, radius, step, previous, older, projectors, *, tol, max_iter,
                 prescribed_mean=None, cached_projectors=None):
        op = self.config.operators(radius)
        H, s, m = self.config.harmonics, op.size, op.m
        d = 1/step if older is None else 1.5/step
        history = previous/step if older is None else (2*previous-.5*older)/step
        new_projectors = cached_projectors if cached_projectors is not None else {
            j: causal_projector(op, j, self.config.frequency(op), previous=projectors[j])
            for j in range(1, H+1)}
        # Parabolized mean; do not apply a wave projector at zero frequency.
        A, B, C = radial_matrices(op, 0, self.config.frequency(op))
        B[:m, 3*m:] = 0
        A[:m, 3*m:] -= 2*op.I/op.radius
        # Drop radial viscosity in physical variables, including derivatives
        # of the R scaling. Dropping only q_RR leaves a spurious upstream
        # mean advection coefficient F-3/R^2 in the far field.
        for v in range(3):
            sl = slice(v*m, (v+1)*m)
            B[sl, sl] += 3*op.I/op.radius**2
        A[2*m:3*m, 2*m:3*m] += op.I/op.radius**3
        A[op.boundary_rows] = op.linear(0, 0)[op.boundary_rows]
        B[op.boundary_rows] = 0
        E0, L0 = descriptor(A, B, np.zeros_like(C))
        equations, linear, forcing = {}, {}, {}
        for j in range(H+1):
            if j:
                p = new_projectors[j]
                E, L, Q = p.E, p.L, p.equation
            else:
                E, L, Q = E0, L0, np.eye(2*s)
            equations[j] = Q
            # Replace removed upstream equations by zero upstream amplitude.
            linear[j] = d*E-Q@L
            forcing[j] = Q@E@history[j+H]

        def evaluate(x, jacobian=False):
            y = unpack(x, H, 2*s)
            q, qr = y[:, :s], y[:, s:]
            if self.advection_derivative == 'bdf':
                qr = d*q-history[:, :s]
                if prescribed_mean is not None:
                    qr[H] = y[H, s:]
            N = convection(op, q, qr) if self.config.nonlinear else np.zeros_like(q)
            result = np.zeros_like(y)
            for j in range(H+1):
                result[j+H] = linear[j]@y[j+H]-forcing[j]+equations[j]@np.r_[np.zeros(s), N[j+H]]
                if j:
                    result[H-j] = result[H+j].conj()
            if prescribed_mean is not None:
                result[H] = y[H]-prescribed_mean
            F = pack(result)
            if not jacobian:
                return F
            def block(j, k):
                if j == 0 and prescribed_mean is not None:
                    return np.eye(2*s) if k == 0 else np.zeros((2*s, 2*s))
                J = linear[j].copy() if j == k else np.zeros((2*s, 2*s), complex)
                if self.config.nonlinear:
                    Qn, Tn = convection_blocks(op, q, qr, j, k)
                    if self.advection_derivative == 'bdf' and not (k == 0 and prescribed_mean is not None):
                        Qn, Tn = Qn+d*Tn, np.zeros_like(Tn)
                    J += equations[j]@np.block([[np.zeros((s, 2*s))], [np.hstack((Qn, Tn))]])
                return J
            return F, real_blocks(block, H, 2*s)

        x = pack(previous)
        scale = max(np.max(abs(previous)), 1e-30)
        for iteration in range(max_iter+1):
            F, J = evaluate(x, True)
            norm = np.linalg.norm(F, np.inf)/scale
            if norm < tol:
                return unpack(x, H, 2*s), new_projectors, float(norm), iteration
            if iteration == max_iter:
                break
            rows = np.maximum(np.max(abs(J), axis=1), 1e-30)
            change = linalg.solve(J/rows[:, None], -F/rows, check_finite=False)
            for damping in 2.**(-np.arange(14)):
                trial = x+damping*change
                if np.linalg.norm(evaluate(trial), np.inf) < norm*scale:
                    x = trial
                    break
            else:
                raise RuntimeError(f"One-way Newton line search failed at R={radius}: {norm:.3e}")
        raise RuntimeError(f"One-way Newton failed at R={radius}: {norm:.3e}")

    def march(self, radii, *, amplitude=1e-5, z=1.1858, target=.4-.07j, tol=1e-9,
              max_iter=30, max_outer=20, initial_mean=None, outer_relaxation=1., progress=None):
        if not np.isfinite(tol) or tol <= 0 or max_iter < 1 or max_outer < 1:
            raise ValueError("Positive finite tolerance and positive iteration limits required")
        if not np.isfinite(outer_relaxation) or not 0 < outer_relaxation <= 1:
            raise ValueError("Outer relaxation must lie in (0,1]")
        radii = np.asarray(radii, float)
        if radii.ndim != 1 or len(radii) < 2 or not np.all(np.isfinite(radii)) or radii[0] <= 0:
            raise ValueError("At least two positive finite radii required")
        h = np.diff(radii)
        if np.any(h <= 0) or not np.allclose(h, h[0], rtol=1e-9, atol=1e-12):
            raise ValueError("One-way BDF2 currently requires increasing uniform radii")
        initial, first_projectors, alpha = self.initialize(radii[0], amplitude, z, target)
        if self.mean_closure == 'global' and self.config.nonlinear:
            if len(radii) < 5:
                raise ValueError("Global mean coupling requires at least five radial nodes")
            problem = GlobalRadial(radii, self.config)
            s, H = problem.s, self.config.harmonics
            mean = np.zeros((len(radii), s), complex)
            if initial_mean is not None:
                mean = np.asarray(initial_mean, complex).copy()
                if mean.shape != (len(radii), s) or not np.all(np.isfinite(mean)) or np.any(abs(mean.imag) > 1e-14):
                    raise ValueError("Initial mean must be real, finite and have shape (radial nodes, field size)")
            cache, coupling_history = [], []
            for iteration in range(max_outer):
                if progress is not None:
                    progress('outer_start', dict(iteration=iteration))
                mean_r = problem.D@mean
                result, diagnostics, cache = self._march_given_mean(radii, initial, first_projectors,
                    alpha, tol, max_iter, np.concatenate((mean, mean_r), axis=1), cache, progress)
                updated, mean_residual = problem.solve_mean(result.modes, initial[:, :s], tol=tol*.1)
                relative_change = float(np.max(abs(updated-mean))/max(np.max(abs(updated)), amplitude**2, 1e-30))
                coupling_history.append(relative_change)
                if progress is not None:
                    progress('outer_update', dict(iteration=iteration, relative_change=relative_change,
                                                  mean_residual=mean_residual))
                if relative_change < tol:
                    result.modes[:, H] = updated
                    result.method = 'spectral-oneway-global-mean'
                    diagnostics.update(mean_closure='global full radial zero-harmonic BVP',
                        mean_relative_residual=mean_residual, coupling_history=coupling_history)
                    return result, diagnostics
                mean = (1-outer_relaxation)*mean+outer_relaxation*updated
            raise RuntimeError(f"One-way/global-mean coupling failed: {coupling_history[-1]:.3e}")
        prescribed = (np.zeros((len(radii), initial.shape[1]), complex)
                      if self.mean_closure == 'global' else None)
        result, diagnostics, _ = self._march_given_mean(radii, initial, first_projectors, alpha,
                                                        tol, max_iter, prescribed, progress=progress)
        if self.mean_closure == 'global':
            result.method = 'spectral-oneway-global-mean'
            diagnostics['mean_closure'] = 'zero mean in linear calculation'
        return result, diagnostics

    def _march_given_mean(self, radii, initial, first_projectors, alpha, tol, max_iter,
                          prescribed=None, cache=None, progress=None):
        y, projectors = initial.copy(), first_projectors
        if prescribed is not None:
            y[self.config.harmonics] = prescribed[0]
        s = y.shape[1]//2
        h = radii[1]-radii[0]
        cache = [] if cache is None else cache
        states, residuals, counts, diagnostics = [y[:, :s].copy()], [], [], []
        old = None
        for i, radius in enumerate(radii[1:]):
            new, projectors, residual, iterations = self._advance(radius, h, y, old,
                projectors, tol=tol, max_iter=max_iter,
                prescribed_mean=None if prescribed is None else prescribed[i+1],
                cached_projectors=cache[i] if i < len(cache) else None)
            if i == len(cache):
                cache.append(projectors)
            derivative = (new-y)/h if old is None else (1.5*new-2*y+.5*old)/h
            for j, p in projectors.items():
                k = self.config.harmonics+j
                # Projection need not preserve the auxiliary identity q_R=v
                # in a nonlinear/variable-coefficient problem. Report it.
                p.diagnostics['relative_kinematic_defect'] = float(
                    np.linalg.norm(derivative[k, :s]-new[k, s:])/max(np.linalg.norm(new[k]), 1e-30))
            old, y = y, new
            states.append(y[:, :s].copy())
            residuals.append(residual)
            counts.append(iterations)
            diagnostics.append({str(j): p.diagnostics.copy() for j, p in projectors.items()})
            if progress is not None and (i % 10 == 0 or i == len(radii)-2):
                progress('inner_step', dict(index=i+1, radius=float(radius), residual=residual))
        result = RadialSolution(radii.copy(), np.array(states), max(residuals), sum(counts),
                                residuals, 'spectral-oneway-parabolized-mean')
        return result, dict(inlet_alpha=[alpha.real, alpha.imag], steps=diagnostics,
            advection_derivative=self.advection_derivative,
            mean_closure='drop physical mean radial pressure gradient and full radial viscous operator',
            nonlinear_inlet_compatibility='linear inlet; not established at strong amplitude'), cache
