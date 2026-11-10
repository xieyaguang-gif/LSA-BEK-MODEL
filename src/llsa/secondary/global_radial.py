"""Experimental nonlinear radial BVP, without PSE/one-way truncation.

Finite differences in radius, mapped Chebyshev in z, exact Fourier convolution.
Inlet velocity and outlet velocity/Robin data are explicit modelling inputs.
An outlet Robin condition is NOT an exact nonreflecting boundary condition.
"""
from dataclasses import dataclass
from time import perf_counter

import numpy as np
from scipy import sparse
from scipy.sparse.linalg import spsolve, splu, gmres, LinearOperator

from .radial import (RadialConfig, radial_matrices, convection, convection_blocks,
                     real_blocks, pack_field, unpack_field)


def radial_differences(radii, stencil=5):
    r = np.asarray(radii, float)
    if r.ndim != 1 or len(r) < stencil or stencil < 3 or stencil % 2 != 1:
        raise ValueError("Require odd stencil>=3 and at least stencil radial nodes")
    if not np.all(np.isfinite(r)) or np.any(np.diff(r) <= 0) or r[0] <= 0:
        raise ValueError("Positive, finite, strictly increasing radii required")
    D, D2 = np.zeros((len(r), len(r))), np.zeros((len(r), len(r)))
    for i in range(len(r)):
        start = min(max(0, i-stencil//2), len(r)-stencil)
        ix = np.arange(start, start+stencil)
        scale = max(abs(r[ix]-r[i]))
        x = (r[ix]-r[i])/scale
        V = x[None, :]**np.arange(stencil)[:, None]
        for derivative, dest in ((1, D), (2, D2)):
            rhs = np.zeros(stencil)
            rhs[derivative] = 1 if derivative == 1 else 2
            dest[i, ix] = np.linalg.solve(V, rhs)/scale**derivative
    return sparse.csr_matrix(D), sparse.csr_matrix(D2)


@dataclass
class RadialSolution:
    radii: np.ndarray
    modes: np.ndarray
    residual: float
    iterations: int
    history: list
    method: str

    def energies(self, config):
        H = config.harmonics
        op = config.operators(self.radii[0])
        v = self.modes[:, H+1:].reshape(len(self.radii), H, 4, op.m)[:, :, :3]
        return self.radii[:, None]**2*np.einsum('rhvi,ij,rhvj->rh', v.conj(), op.grid.mass, v).real


@dataclass(frozen=True)
class OutletSponge:
    """Smooth velocity damping near the radial outlet.

    ``start`` and ``width`` use the solver radius R (Re=abs(Ro)*R).  The
    profile is the C-infinity step used by the companion Semtex DNS.  By
    default only non-zero Fourier harmonics are damped towards zero; this
    leaves the nonlinear mean-flow correction to satisfy its equations.
    """
    start: float
    width: float
    strength: float
    damp_mean: bool = False

    def __post_init__(self):
        if (not np.isfinite(self.start) or not np.isfinite(self.width)
                or not np.isfinite(self.strength) or self.start <= 0
                or self.width <= 0 or self.strength < 0):
            raise ValueError("Sponge start/width must be positive and strength nonnegative")

    def profile(self, radii):
        x = (np.asarray(radii, float)-self.start)/self.width
        value = np.zeros_like(x)
        value[x >= 1] = 1
        interior = (x > 0) & (x < 1)
        y = x[interior]
        exponent = 1/(y-1)+1/y
        # Algebraically identical to the overflow-safe DNS implementation.
        positive = exponent >= 0
        transition = np.empty_like(y)
        a = np.exp(-exponent[positive])
        transition[positive] = a/(1+a)
        transition[~positive] = 1/(1+np.exp(exponent[~positive]))
        value[interior] = transition
        return self.strength*value


@dataclass(frozen=True)
class BlendedSponge:
    """Auxiliary damping homotopy; only fraction=1 is the target sponge."""
    original: OutletSponge
    target: OutletSponge
    fraction: float

    def __post_init__(self):
        if not np.isfinite(self.fraction) or not 0 <= self.fraction <= 1:
            raise ValueError('Sponge release fraction must be in [0,1]')
        if self.original.damp_mean != self.target.damp_mean:
            raise ValueError('Sponge release must keep the same mean-damping choice')

    @property
    def start(self):
        return self.target.start if self.fraction == 1 else min(self.original.start,self.target.start)

    @property
    def width(self):
        return max(self.original.start+self.original.width,self.target.start+self.target.width)-self.start

    @property
    def damp_mean(self):
        return self.target.damp_mean

    def profile(self,radii):
        return (1-self.fraction)*self.original.profile(radii)+self.fraction*self.target.profile(radii)


class GlobalRadial:
    def __init__(self, radii, config=None, *, stencil=5, outlet_alpha=0j,
                 outlet_constraints=None, outlet_sponge=None):
        self.config = config or RadialConfig()
        self.radii = np.asarray(radii, float).copy()
        self.D, self.D2 = radial_differences(self.radii, stencil)
        self.ops = [self.config.operators(r) for r in self.radii]
        self.H, self.s = self.config.harmonics, self.ops[0].size
        self.width = (2*self.H+1)*self.s
        self.outlet_alpha = complex(outlet_alpha)
        if not np.isfinite(self.outlet_alpha):
            raise ValueError("Finite outlet alpha required")
        self.coefficients = [
            [radial_matrices(op, j, self.config.frequency(op)) for j in range(-self.H, self.H+1)]
            for op in self.ops]
        m = self.ops[0].m
        self.velocity_rows = np.concatenate([v*m+np.arange(1, m-1) for v in range(3)])
        self.outlet_constraints = {}
        for j, constraint in (outlet_constraints or {}).items():
            constraint = np.asarray(constraint, complex)
            if not 1 <= j <= self.H or constraint.shape != (len(self.velocity_rows), 2*self.s):
                raise ValueError("Modal outlet requires one upstream constraint per interior velocity row")
            if not np.all(np.isfinite(constraint)):
                raise ValueError("Finite outlet constraints required")
            self.outlet_constraints[j] = constraint.copy()
            self.outlet_constraints[-j] = constraint.conj().copy()
        if outlet_sponge is not None and not isinstance(outlet_sponge, (OutletSponge,BlendedSponge)):
            raise TypeError("outlet_sponge must be an OutletSponge or BlendedSponge")
        if outlet_sponge is not None and outlet_sponge.start+outlet_sponge.width > self.radii[-1]+1e-12:
            raise ValueError("Sponge must finish at or before the outlet")
        self.outlet_sponge = outlet_sponge
        self.sponge_strength = (np.zeros(len(self.radii)) if outlet_sponge is None
                                else outlet_sponge.profile(self.radii))
        self._linear_cache = {}
        self._radial_derivative = sparse.kron(self.D, sparse.eye(self.width), format='csr')

    def _linear_matrix(self, prescribed_outlet):
        """Cache the unchanged linear equations including boundary rows."""
        key = bool(prescribed_outlet)
        if key in self._linear_cache:
            return self._linear_cache[key]
        aa, bb, cc = [], [], []
        for i, op in enumerate(self.ops):
            real_a, real_b, real_c = [], [], []
            for j in range(self.H+1):
                A, B, C = [v.copy() for v in self.coefficients[i][j+self.H]]
                rows = self.velocity_rows
                if self.sponge_strength[i] and (j or self.outlet_sponge.damp_mean):
                    A[rows, rows] += self.sponge_strength[i]/op.radius
                if i in (0, len(self.radii)-1):
                    A[rows] = B[rows] = C[rows] = 0
                    if i == 0 or prescribed_outlet:
                        A[rows, rows] = 1
                    elif j in self.outlet_constraints:
                        A[rows] = self.outlet_constraints[j][:, :self.s]
                        B[rows] = self.outlet_constraints[j][:, self.s:]
                    else:
                        lam = 1j*j*self.outlet_alpha.real-j*self.outlet_alpha.imag
                        A[rows, rows] = -lam
                        B[rows, rows] = 1
                for M, dest in ((A, real_a), (B, real_b), (C, real_c)):
                    dest.append(sparse.csr_matrix(M.real if j == 0 else
                        np.block([[M.real, -M.imag], [M.imag, M.real]])))
            aa.append(sparse.block_diag(real_a, format='csr'))
            bb.append(sparse.block_diag(real_b, format='csr'))
            cc.append(sparse.block_diag(real_c, format='csr'))
        value = (sparse.block_diag(aa, format='csr')
            + sparse.block_diag(bb, format='csr')@self._radial_derivative
            + sparse.block_diag(cc, format='csr')@sparse.kron(self.D2, sparse.eye(self.width), format='csr'))
        value.eliminate_zeros()
        self._linear_cache[key] = value
        return value

    def _convection_field(self, modes, radial):
        """Exact convolution batched over radius; no change of equations."""
        H, m = self.H, self.ops[0].m
        q = modes.reshape(len(modes), 2*H+1, 4, m)
        qr = radial.reshape(q.shape)
        qz = q[:, :, :3]@self.ops[0].grid.D.T
        inverse_radius = 1/np.array([op.radius for op in self.ops])[:, None, None]
        out = np.zeros_like(q)
        for j in range(-H, H+1):
            for k in range(max(-H, j-H), min(H, j+H)+1):
                a, b = q[:, j-k+H], q[:, k+H]
                out[:, j+H, :3] += (a[:, 0, None]*(qr[:, k+H, :3]+b[:, :3]*inverse_radius)
                    + 1j*k*self.config.beta*inverse_radius*a[:, 1, None]*b[:, :3]
                    + a[:, 2, None]*qz[:, k+H])
                out[:, j+H, 0] -= a[:, 1]*b[:, 1]*inverse_radius[:, 0]
                out[:, j+H, 1] += a[:, 0]*b[:, 1]*inverse_radius[:, 0]
        out = out.reshape(modes.shape)
        out[:, :, self.ops[0].boundary_rows] = 0
        out[0][:, self.velocity_rows] = 0
        out[-1][:, self.velocity_rows] = 0
        return out

    def derivatives(self, q):
        shape = q.shape
        return ((self.D@q.reshape(len(q), -1)).reshape(shape),
                (self.D2@q.reshape(len(q), -1)).reshape(shape))

    def evaluate(self, x, inlet, *, outlet=None, forcing=None, jacobian=False):
        """Cached linear part plus the two local nonlinear Frechet blocks."""
        H, s, nr = self.H, self.s, len(self.radii)
        linear = self._linear_matrix(outlet is not None)
        residual = linear@x
        q = unpack_field(x, nr, H, s)
        qr = self.derivatives(q)[0]
        if self.config.nonlinear:
            residual += pack_field(self._convection_field(q, qr))
        # Forcing is replaced by velocity data on radial endpoint rows.
        rhs = np.zeros_like(q) if forcing is None else np.array(forcing, copy=True)
        rhs[0][:, self.velocity_rows] = inlet[:, self.velocity_rows]
        rhs[-1][:, self.velocity_rows] = 0 if outlet is None else outlet[:, self.velocity_rows]
        residual -= pack_field(rhs)
        if not jacobian:
            return residual
        if not self.config.nonlinear:
            return residual, linear.tocsc()
        local_blocks, derivative_blocks = [], []
        for i, op in enumerate(self.ops):
            if i in (0, nr-1):
                local_blocks.append(sparse.csr_matrix((self.width, self.width)))
                derivative_blocks.append(sparse.csr_matrix((self.width, self.width)))
                continue
            blocks = {(j, k): convection_blocks(op, q[i], qr[i], j, k)
                      for j in range(H+1) for k in range(-H, H+1)}
            local_blocks.append(sparse.csr_matrix(real_blocks(lambda j, k: blocks[j, k][0], H, s)))
            derivative_blocks.append(sparse.csr_matrix(real_blocks(lambda j, k: blocks[j, k][1], H, s)))
        J = (linear+sparse.block_diag(local_blocks, format='csr')
             + sparse.block_diag(derivative_blocks, format='csr')@self._radial_derivative)
        return residual, J.tocsc()

    def evaluate_reference(self, x, inlet, *, outlet=None, forcing=None, jacobian=False):
        """Original assembly retained as a small-problem verification oracle."""
        H, s, nr = self.H, self.s, len(self.radii)
        q = unpack_field(x, nr, H, s)
        qr, qrr = self.derivatives(q)
        out = np.zeros_like(q)
        blocks = [[None]*nr for _ in range(nr)] if jacobian else None
        for i, op in enumerate(self.ops):
            coeff = self.coefficients[i]
            for j in range(-H, H+1):
                A, B, C = coeff[j+H]
                out[i, j+H] = A@q[i, j+H]+B@qr[i, j+H]+C@qrr[i, j+H]
                if self.sponge_strength[i] and (j != 0 or self.outlet_sponge.damp_mean):
                    # The physical momentum equations contain +lambda*u' on
                    # the left after moving the DNS force -lambda*u' from the
                    # right.  They are divided by R^2 here, while u'=R*q.
                    out[i, j+H, self.velocity_rows] += (
                        self.sponge_strength[i]/op.radius*q[i, j+H, self.velocity_rows])
            if self.config.nonlinear:
                out[i] += convection(op, q[i], qr[i])
            if forcing is not None:
                out[i] -= forcing[i]
            # z boundary rows take precedence over radial endpoint conditions.
            if i == 0:
                out[i][:, self.velocity_rows] = (q[i]-inlet)[:, self.velocity_rows]
            elif i == nr-1:
                for j in range(-H, H+1):
                    if outlet is not None:
                        out[i, j+H, self.velocity_rows] = (q[i, j+H]-outlet[j+H])[self.velocity_rows]
                    elif j in self.outlet_constraints:
                        out[i, j+H, self.velocity_rows] = self.outlet_constraints[j]@np.r_[q[i, j+H], qr[i, j+H]]
                    else:
                        lam = 1j*j*self.outlet_alpha.real-abs(j)*self.outlet_alpha.imag
                        out[i, j+H, self.velocity_rows] = (qr[i, j+H]-lam*q[i, j+H])[self.velocity_rows]
            if not jacobian:
                continue
            links = sorted(set(self.D[i].indices) | set(self.D2[i].indices) | {i})
            nonlinear_blocks = {(j, k): convection_blocks(op, q[i], qr[i], j, k)
                for j in range(H+1) for k in range(-H, H+1)} if self.config.nonlinear else {}
            for t in links:
                d, d2, local = self.D[i, t], self.D2[i, t], float(t == i)
                def block(j, k):
                    J = np.zeros((s, s), complex)
                    if j == k:
                        A, B, C = coeff[j+H]
                        J += local*A+d*B+d2*C
                        if (self.sponge_strength[i] and local
                                and (j != 0 or self.outlet_sponge.damp_mean)):
                            J[self.velocity_rows, self.velocity_rows] += self.sponge_strength[i]/op.radius
                    if self.config.nonlinear:
                        Q, T = nonlinear_blocks[j, k]
                        J += local*Q+d*T
                    if i == 0 or i == nr-1:
                        J[self.velocity_rows] = 0
                        if j == k:
                            if i == nr-1 and outlet is None and j in self.outlet_constraints:
                                BC = self.outlet_constraints[j]
                                J[self.velocity_rows] = local*BC[:, :s]+d*BC[:, s:]
                                return J
                            value = local
                            if i == nr-1 and outlet is None:
                                lam = 1j*j*self.outlet_alpha.real-abs(j)*self.outlet_alpha.imag
                                value = d-lam*local
                            J[self.velocity_rows, self.velocity_rows] = value
                    return J
                blocks[i][t] = sparse.csr_matrix(real_blocks(block, H, s))
        residual = pack_field(out)
        if not jacobian:
            return residual
        return residual, sparse.bmat(blocks, format='csc')

    def project_interior_continuity(self, field, residual, target):
        """Correct v_j so interior continuity follows a prescribed affine step.

        Only nonzero harmonics and interior radial/normal nodes are changed.
        The coefficient of v_j in continuity is i*j*beta/R. Boundary and
        mean constraints are still assessed by the full residual/line search.
        """
        if self.config.beta == 0:
            raise ValueError('Continuity projection requires nonzero beta')
        nr, H, m = len(self.radii), self.H, self.ops[0].m
        q = np.array(field,copy=True).reshape(nr,2*H+1,4,m)
        defect = unpack_field(residual-target,nr,H,self.s).reshape(q.shape)
        for j in range(1,H+1):
            q[1:-1,H+j,1,1:-1] += (1j*self.radii[1:-1,None]/(j*self.config.beta)
                *defect[1:-1,H+j,3,1:-1])
            q[1:-1,H-j,1,1:-1] = q[1:-1,H+j,1,1:-1].conj()
        return q.reshape(field.shape)

    def solve(self, inlet, *, initial=None, outlet=None, forcing=None, tol=1e-9, max_iter=30,
              progress=None, linear_solver='direct', coupled_harmonics=1,
              checkpoint=None, diagnostics=None, preconditioner_refresh=1,
              complex_high_harmonics=True, merit='inf', preconditioner_sweep='diagonal',
              gmres_restart=60, gmres_maxiter=12, linear_progress=None, stage_progress=None,
              phase_update=False, pardiso_backend=None, phase_pin_radius=None,
              phase_continuity=False, augmentation=None, cartesian_fallback=False,
              preconditioning_side='left'):
        if not np.isfinite(tol) or tol <= 0 or max_iter < 1:
            raise ValueError("Positive finite tolerance and positive max_iter required")
        if phase_continuity and (not phase_update or self.config.beta == 0):
            raise ValueError('phase_continuity requires phase_update and nonzero beta')
        if cartesian_fallback and not phase_update:
            raise ValueError('cartesian_fallback requires phase_update')
        if linear_solver not in ('direct', 'harmonic-gmres', 'pardiso', 'pardiso-gmres'):
            raise ValueError("Linear solver must be direct, harmonic-gmres, pardiso or pardiso-gmres")
        if preconditioning_side not in ('left', 'right'):
            raise ValueError('preconditioning_side must be left or right')
        if preconditioning_side == 'right' and linear_solver != 'pardiso-gmres':
            raise ValueError('Right preconditioning currently requires pardiso-gmres')
        if not 1 <= coupled_harmonics <= self.H or int(coupled_harmonics) != coupled_harmonics:
            raise ValueError('coupled_harmonics must be an integer in [1,H]')
        if preconditioner_refresh < 1 or int(preconditioner_refresh) != preconditioner_refresh:
            raise ValueError('preconditioner_refresh must be a positive integer')
        if merit not in ('inf', 'scaled-l2'):
            raise ValueError('merit must be inf or scaled-l2')
        if preconditioner_sweep not in ('diagonal', 'forward', 'symmetric'):
            raise ValueError('preconditioner_sweep must be diagonal, forward or symmetric')
        if any(v < 1 or int(v) != v for v in (gmres_restart, gmres_maxiter)):
            raise ValueError('GMRES restart and maxiter must be positive integers')
        shape = (2*self.H+1, self.s)
        inlet = np.asarray(inlet, complex)
        if inlet.shape != shape:
            raise ValueError(f"Expected inlet shape {shape}")
        for label, value in (("inlet", inlet), ("outlet", outlet)):
            if value is not None:
                value = np.asarray(value)
                if value.shape != shape or not np.all(np.isfinite(value)) or not np.allclose(value, value[::-1].conj(), rtol=1e-12, atol=1e-15):
                    raise ValueError(f"{label} must be finite and Fourier conjugate symmetric")
        q = np.tile(inlet, (len(self.radii), 1, 1)) if initial is None else np.asarray(initial)
        if (q.shape != (len(self.radii), *shape) or not np.all(np.isfinite(q))
                or not np.allclose(q, q[:, ::-1].conj(), rtol=1e-12, atol=1e-15)):
            raise ValueError("Invalid initial field")
        x = pack_field(q)
        scale = max(np.max(abs(inlet)), np.max(abs(q)), 1e-30)
        if forcing is not None:
            forcing = np.asarray(forcing)
            if (forcing.shape != q.shape or not np.all(np.isfinite(forcing))
                    or not np.allclose(forcing, forcing[:, ::-1].conj(), rtol=1e-12, atol=1e-15)):
                raise ValueError("Forcing must match the field and be finite/conjugate symmetric")
            scale = max(scale, np.max(abs(forcing)))
        if augmentation is not None and phase_pin_radius is not None:
            raise ValueError('An augmentation must include its own phase gauge')
        field_size = len(self.radii)*self.width
        gauge = augmentation
        def evaluate_state(state, jacobian=False):
            return self.evaluate(state, inlet, outlet=outlet, forcing=forcing, jacobian=jacobian)
        def physical_field(state):
            return unpack_field(state, len(self.radii), self.H, self.s)
        if phase_pin_radius is not None:
            from .phase_gauge import PhaseGauge
            gauge = PhaseGauge(self, q, phase_pin_radius, inlet, outlet, forcing)
        if gauge is not None:
            x = gauge.initial_state(q)
            evaluate_state, physical_field = gauge.evaluate, gauge.physical_field
        history = []
        factors, factor_iteration = None, -1
        backend = pardiso_backend
        own_backend = backend is None
        if linear_solver.startswith('pardiso') and backend is None:
            from .pardiso_backend import PardisoBackend
            backend = PardisoBackend()
        elif not linear_solver.startswith('pardiso') and backend is not None:
            raise ValueError('An external PARDISO backend requires a pardiso linear solver')
        try:
            for iteration in range(max_iter+1):
                started = perf_counter()
                # A residual costs much less than a Jacobian. In particular, do
                # not assemble a final unused Jacobian after the root is reached.
                F = evaluate_state(x)
                norm = float(np.linalg.norm(F, np.inf)/scale)
                if gauge is not None and norm <= tol:
                    original = gauge.physical_residual(x)
                    norm = max(norm, float(np.linalg.norm(original, np.inf)/scale))
                history.append(norm)
                if progress is not None:
                    progress(iteration, norm)
                if checkpoint is not None:
                    checkpoint(iteration, physical_field(x), norm)
                if norm <= tol:
                    if gauge is not None:
                        gauge.accepted_state = x.copy()
                    return RadialSolution(self.radii.copy(), physical_field(x),
                        norm, iteration, history, 'global-hb')
                if iteration == max_iter:
                    break
                F, J = evaluate_state(x, jacobian=True)
                # Row equilibration leaves the Newton equations unchanged.
                row_scale = np.asarray(abs(J).max(axis=1).toarray()).ravel()
                row_scale = np.maximum(row_scale, 1e-30)
                equilibrated, rhs = sparse.diags(1/row_scale)@J, -F/row_scale
                assembled_at = perf_counter()
                factor_seconds = 0.
                krylov_iterations = []
                def monitor(value):
                    krylov_iterations.append(float(value))
                    if linear_progress is not None and (len(krylov_iterations) == 1 or len(krylov_iterations) % 20 == 0):
                        linear_progress(iteration, len(krylov_iterations), float(value))
                forcing_tolerance = min(.05, max(1e-10, .2*np.sqrt(norm)))
                if linear_solver == 'direct':
                    change = spsolve(equilibrated, rhs)
                elif linear_solver.startswith('pardiso'):
                    def direct_progress(values):
                        if stage_progress is not None:
                            stage_progress(dict(newton_iteration=iteration, **values))
                    try:
                        refreshed = (linear_solver == 'pardiso' or backend.matrix is None
                            or backend.matrix.shape != equilibrated.shape
                            or backend.newton_steps >= preconditioner_refresh)
                        if refreshed:
                            backend.factorize(equilibrated, progress=direct_progress)
                            factor_iteration = iteration
                        factor_seconds = perf_counter()-assembled_at
                        if linear_solver == 'pardiso':
                            change = backend.solve(rhs)
                        else:
                            def krylov_solve():
                                if preconditioning_side == 'right':
                                    # Krylov minimizes the current, equilibrated
                                    # equation residual instead of M^-1 times it.
                                    operator = LinearOperator(J.shape,
                                        matvec=lambda v: equilibrated@backend.solve(v), dtype=float)
                                    y, info = gmres(operator, rhs, rtol=forcing_tolerance, atol=0,
                                        restart=gmres_restart, maxiter=gmres_maxiter,
                                        callback=monitor, callback_type='pr_norm')
                                    return backend.solve(y), info
                                pre = LinearOperator(J.shape, matvec=backend.solve, dtype=float)
                                return gmres(equilibrated,rhs,M=pre,rtol=forcing_tolerance,atol=0,
                                    restart=gmres_restart,maxiter=gmres_maxiter,
                                    callback=monitor,callback_type='pr_norm')
                            change, info = krylov_solve()
                            if info != 0 and not refreshed:
                                refresh_started = perf_counter()
                                backend.factorize(equilibrated, progress=direct_progress)
                                factor_iteration = iteration
                                factor_seconds += perf_counter()-refresh_started
                                change, info = krylov_solve()
                            achieved = np.linalg.norm(equilibrated@change-rhs)/np.linalg.norm(rhs)
                            if not np.isfinite(achieved) or achieved > forcing_tolerance*(1+1e-6):
                                info = info or 1
                            if info != 0:
                                raise RuntimeError(f'Full-LU-preconditioned GMRES failed: residual {achieved:.3e}')
                    except Exception as exc:
                        # A failed native factorization/backsolve must not leave
                        # an apparently reusable factor for the next station.
                        backend.close()
                        raise RuntimeError(f'PARDISO factorization/solve failed: {exc}') from exc
                else:
                    # Couple mean and fundamental in the preconditioner: their
                    # feedback is essential near nonlinear saturation.
                    groups = [np.arange((2*coupled_harmonics+1)*self.s)]+[
                        np.arange((2*j-1)*self.s, (2*j+1)*self.s) for j in range(coupled_harmonics+1, self.H+1)]
                    def factorize():
                        answer = []
                        for group_number, local in enumerate(groups):
                            factor_started = perf_counter()
                            if stage_progress is not None:
                                stage_progress(dict(stage='factor_start', newton_iteration=iteration, group=group_number))
                            ix = np.concatenate([i*self.width+local for i in range(len(self.radii))])
                            if gauge is not None and group_number == 0:
                                ix = np.r_[ix, np.arange(field_size,len(x))]
                            harmonic = coupled_harmonics+group_number
                            if complex_high_harmonics and group_number and 2*harmonic > self.H:
                                # dN_j/dq_-j=0 if 2*j>H. This diagonal block is
                                # complex-linear, so a half-size complex LU is an
                                # exact substitute for the doubled real LU.
                                re = ix.reshape(len(self.radii), 2*self.s)[:, :self.s].ravel()
                                im = re+self.s
                                matrix = J[re][:, re]+1j*J[im][:, re]
                                answer.append((ix, splu(matrix.tocsc()),
                                               (re, im, row_scale[re].copy(), row_scale[im].copy())))
                            else:
                                answer.append((ix, splu(equilibrated[ix][:, ix].tocsc()), None))
                            if stage_progress is not None:
                                stage_progress(dict(stage='factor_complete', newton_iteration=iteration, group=group_number,
                                                    seconds=perf_counter()-factor_started))
                        return answer
                    if factors is None or iteration-factor_iteration >= preconditioner_refresh:
                        factors = None  # Release obsolete LU storage before allocating the next factors.
                        factors = factorize()
                        factor_iteration = iteration
                    # Optional block Gauss--Seidel sweeps retain inter-harmonic
                    # feedback in the preconditioner without increasing LU size.
                    couplings = []
                    if preconditioner_sweep != 'diagonal':
                        for g, (ix, _, _) in enumerate(factors):
                            lower = np.concatenate([f[0] for f in factors[:g]]) if g else np.array([], int)
                            upper = (np.concatenate([f[0] for f in factors[g+1:]])
                                     if g+1 < len(factors) else np.array([], int))
                            block_rows = equilibrated.tocsr()[ix]
                            couplings.append((lower, upper, block_rows[:, lower], block_rows[:, upper]))
                    def solve_block(entry, values):
                        _, factor, complex_rows = entry
                        if complex_rows is None:
                            return factor.solve(values)
                        _, _, sr, si = complex_rows
                        v = values.reshape(len(self.radii), 2*self.s)
                        value = factor.solve(sr*v[:, :self.s].ravel()+1j*si*v[:, self.s:].ravel())
                        value = value.reshape(len(self.radii), self.s)
                        return np.concatenate((value.real, value.imag), axis=1).ravel()
                    def precondition(vector):
                        answer = np.zeros_like(vector)
                        for g, entry in enumerate(factors):
                            ix = entry[0]
                            rhs_block = vector[ix]
                            if couplings:
                                lower, _, L, _ = couplings[g]
                                rhs_block = rhs_block-L@answer[lower]
                            answer[ix] = solve_block(entry, rhs_block)
                        if preconditioner_sweep == 'symmetric':
                            for g in range(len(factors)-2, -1, -1):
                                ix = factors[g][0]
                                lower, upper, L, U = couplings[g]
                                answer[ix] = solve_block(factors[g], vector[ix]-L@answer[lower]-U@answer[upper])
                        return answer
                    pre = LinearOperator(J.shape, matvec=precondition, dtype=float)
                    factor_seconds = perf_counter()-assembled_at
                    # Inexact Newton: loose linear solves far from the root,
                    # tightening as the *nonlinear* residual approaches tolerance.
                    change, info = gmres(equilibrated, rhs, M=pre, rtol=forcing_tolerance, atol=0,
                                         restart=gmres_restart, maxiter=gmres_maxiter, callback=monitor,
                                         callback_type='pr_norm')
                    if info != 0 and factor_iteration != iteration:
                        # Reusing an LU only changes the preconditioner. The new
                        # Jacobian equations and GMRES convergence test are exact.
                        refresh_started = perf_counter()
                        factors = None
                        factors = factorize()
                        factor_iteration = iteration
                        factor_seconds += perf_counter()-refresh_started
                        change, info = gmres(equilibrated, rhs, M=pre, rtol=forcing_tolerance, atol=0,
                            restart=gmres_restart, maxiter=gmres_maxiter, callback=monitor, callback_type='pr_norm')
                    if info != 0:
                        achieved = np.linalg.norm(equilibrated@change-rhs)/np.linalg.norm(rhs)
                        raise RuntimeError(f"Harmonic-preconditioned GMRES failed: relative linear residual {achieved:.3e}")
                if not np.all(np.isfinite(change)):
                    raise RuntimeError("Global Jacobian is singular or ill-conditioned")
                if linear_solver == 'pardiso-gmres':
                    factor_age = backend.newton_steps
                    backend.newton_steps += 1
                else:
                    factor_age = iteration-factor_iteration if linear_solver == 'harmonic-gmres' else 0
                solved_at = perf_counter()
                linear_action = J@change
                linear_error = float(np.linalg.norm(linear_action+F, np.inf)/max(np.linalg.norm(F, np.inf), 1e-30))
                def merit_value(residual):
                    return (np.linalg.norm(residual, np.inf) if merit == 'inf'
                            else np.linalg.norm(residual/row_scale))
                old_merit = merit_value(F)
                if phase_update:
                    field = unpack_field(x[:field_size], len(self.radii), self.H, self.s)
                    direction = unpack_field(change[:field_size], len(self.radii), self.H, self.s)
                    phase, remainder = phase_direction(field, direction, self.ops[0].grid.mass)
                    numbers = np.arange(-self.H, self.H+1)[None,:,None]
                line_search_trials = []
                for damping in 2.**(-np.arange(14)):
                    cartesian_F = None
                    if phase_update:
                        trial = pack_field((field+damping*remainder)*np.exp(1j*damping*phase[:,None,None]*numbers))
                        if gauge is not None:
                            trial = np.r_[trial, x[field_size:]+damping*change[field_size:]]
                    else:
                        trial = x+damping*change
                    trial_F = evaluate_state(trial)
                    step_path = 'phase' if phase_update else 'cartesian'
                    phase_comparison = {}
                    if phase_continuity or cartesian_fallback:
                        cartesian_trial = x+damping*change
                        cartesian_F = evaluate_state(cartesian_trial)
                        phase_comparison['cartesian_residual_inf'] = float(np.linalg.norm(cartesian_F,np.inf)/scale)
                    if phase_continuity:
                        phase_comparison['unprojected_phase_residual_inf'] = float(np.linalg.norm(trial_F,np.inf)/scale)
                        end = field_size
                        corrected = self.project_interior_continuity(
                            unpack_field(trial[:end],len(self.radii),self.H,self.s),
                            trial_F[:end],(F+damping*linear_action)[:end])
                        trial[:end] = pack_field(corrected)
                        trial_F = evaluate_state(trial)
                        step_path = 'phase_continuity'
                    trial_merit = merit_value(trial_F)
                    # Both paths have the same Newton tangent. Select using the
                    # actual nonlinear equations; Cartesian steps already obey
                    # the affine (linear) continuity equations. This avoids a
                    # large azimuthal correction from the phase projection when
                    # it harms the momentum equations.
                    if cartesian_fallback:
                        cartesian_merit = merit_value(cartesian_F)
                        if np.isfinite(cartesian_merit) and (not np.isfinite(trial_merit)
                                                             or cartesian_merit < trial_merit):
                            trial, trial_F, trial_merit = cartesian_trial, cartesian_F, cartesian_merit
                            step_path = 'cartesian'
                    line_search_trials.append(dict(damping=float(damping),
                        step_path=step_path,
                        residual_inf=float(np.linalg.norm(trial_F, np.inf)/scale),
                        residual_scaled_l2=float(np.linalg.norm(trial_F/row_scale)),**phase_comparison))
                    sufficient = old_merit if merit == 'inf' else old_merit*(1-1e-4*damping)
                    if trial_merit < sufficient:
                        x = trial
                        if diagnostics is not None:
                            diagnostics(dict(iteration=iteration, damping=float(damping),
                                merit=merit, merit_before=float(old_merit), merit_after=float(trial_merit),
                                linear_residual_inf=linear_error, residual_before=norm,
                                residual_after=float(np.linalg.norm(trial_F, np.inf)/scale),
                                assembly_seconds=assembled_at-started,
                                factor_seconds=factor_seconds,
                                factor_age=factor_age,
                                linear_solver=linear_solver,
                                preconditioning_side=preconditioning_side,
                                preconditioner_sweep=preconditioner_sweep,
                                linear_seconds=solved_at-assembled_at,
                                krylov_iterations=len(krylov_iterations),
                                gmres_restart=gmres_restart,
                                phase_update=phase_update,
                                phase_continuity=phase_continuity,
                                cartesian_fallback=cartesian_fallback,
                                step_path=step_path,
                                phase_pin_radius=phase_pin_radius,
                                inlet_phase=(float(x[gauge.phase_index]) if gauge is not None
                                    and gauge.phase_index is not None else None),
                                max_phase_direction=float(np.max(abs(phase))) if phase_update else None,
                                line_search_trials=line_search_trials,
                                line_search_seconds=perf_counter()-solved_at,
                                seconds=perf_counter()-started))
                        break
                else:
                    raise RuntimeError(f"Global Newton line search failed: relative residual {norm:.3e}")
            raise RuntimeError(f"Global Newton failed after {max_iter} iterations: {history[-1]:.3e}")
        finally:
            if own_backend and backend is not None:
                backend.close()

    def solve_mean(self, field, inlet, *, tol=1e-10, max_iter=20):
        """Solve the full zero harmonic with all nonzero harmonics held fixed.

        Used by the one-way/global-mean hybrid. Radial wave derivatives in the
        Reynolds stresses use this BVP's radial difference operator.
        """
        field = np.asarray(field)
        x = pack_field(field)
        indices = np.concatenate([i*self.width+np.arange(self.s) for i in range(len(self.radii))])
        scale = max(np.max(abs(field[:, self.H])), np.max(abs(field))**2, 1e-30)
        for iteration in range(max_iter+1):
            F, J = self.evaluate(x, inlet, jacobian=True)
            f, j = F[indices], J[indices][:, indices]
            norm = np.linalg.norm(f, np.inf)/scale
            if norm <= tol:
                result = unpack_field(x, len(self.radii), self.H, self.s)
                return result[:, self.H], float(norm)
            if iteration == max_iter:
                break
            rows = np.maximum(np.asarray(abs(j).max(axis=1).toarray()).ravel(), 1e-30)
            change = spsolve(sparse.diags(1/rows)@j, -f/rows)
            for damping in 2.**(-np.arange(14)):
                trial = x.copy()
                trial[indices] += damping*change
                trial_f = self.evaluate(trial, inlet)[indices]
                if np.linalg.norm(trial_f, np.inf) < norm*scale:
                    x = trial
                    break
            else:
                raise RuntimeError(f"Global mean Newton failed: {norm:.3e}")
        raise RuntimeError(f"Global mean Newton failed after {max_iter} iterations: {norm:.3e}")


def phase_direction(field, direction, mass):
    """Split a Newton tangent into spiral rotation and profile change.

    Retracting the rotation with exp(i*j*phase) has the same first derivative
    as the Cartesian Newton step. It changes the search path, not the root
    equations or the convergence criterion.
    """
    H = (field.shape[1]-1)//2
    m = mass.shape[0]
    q, dq = [a[:, H+1:].reshape(len(a), H, 4, m)[:, :, :3] for a in (field, direction)]
    j = np.arange(1, H+1)[None,:]
    numerator = np.sum(j*np.einsum('rhvi,ik,rhvk->rh', q.conj(), mass, dq).imag, axis=1)
    denominator = np.sum(j*j*np.einsum('rhvi,ik,rhvk->rh', q.conj(), mass, q).real, axis=1)
    phase = np.zeros(len(field))
    active = denominator > max(np.max(denominator)*1e-24, 1e-100)
    phase[active] = numerator[active]/denominator[active]
    # A nearly absent harmonic has no well-defined phase. Avoid arbitrarily
    # many rotations in a newly appended, strongly damped tail. The residual
    # tangent remains exactly direction because the remainder is recomputed.
    phase = np.clip(phase, -np.pi, np.pi)
    ns = np.arange(-H, H+1)[None,:,None]
    remainder = direction-1j*ns*phase[:,None,None]*field
    return phase, remainder
