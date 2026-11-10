"""Full complex global secondary Floquet pencil, with independent base/sidebands.

For exp(sigma*t+i*beta_hat*theta) sum_k v_k exp(ik(beta*theta-omega*t)),
the equations are (J + sigma*B)v=0.  J contains -ik*omega*B.  Arrays use
radius, signed harmonic, (u,v,w,p) collocation order; no real compression.
"""
from dataclasses import dataclass
from time import monotonic

import numpy as np
from scipy import sparse
from scipy.sparse.linalg import (LinearOperator, splu, gmres, eigs,
                                 ArpackNoConvergence)

from .radial import radial_matrices, convection_blocks


def convection_block(op, base, base_radial, j, k, beta_hat):
    """C(base,delta) uses beta_hat+k*beta; C(delta,base) uses (j-k)*beta."""
    Q, T = convection_blocks(op, base, base_radial, j, k)
    H = (len(base)-1)//2
    if abs(j-k) <= H and beta_hat:
        a = base[j-k+H].reshape(4, op.m)
        for v in range(3):
            ix = v*op.m+np.arange(op.m)
            Q[ix, ix] += 1j*beta_hat/op.radius*a[1]
        Q[op.boundary_rows] = 0
    return Q, T


@dataclass
class Spectrum:
    values: np.ndarray
    right_vectors: np.ndarray
    left_vectors: np.ndarray | None
    records: list
    stats: dict


class GlobalFloquet:
    """Sparse exact Galerkin linearization of an existing GlobalRadial field.

    ``problem`` supplies its actual D,D2, operators, endpoint and sponge data;
    this also accepts D2 Mapping.at's physical-grid problem. ``harmonics``
    defaults to H and ``indices`` optionally gives a contiguous shifted window.
    Nonzero beta_hat always retains every complex signed-mode degree of freedom.
    The sponge damps a sideband iff its physical azimuthal number is nonzero,
    or the original sponge explicitly damps the mean.
    """
    def __init__(self, problem, base, beta_hat=0., harmonics=None, *,
                 indices=None, prescribed_outlet=False, check=None):
        self.problem = problem
        self.base = np.asarray(base, complex)
        self.H, self.s = problem.H, problem.s
        self.nr = len(problem.radii)
        if self.base.shape != (self.nr, 2*self.H+1, self.s):
            raise ValueError('base shape must be (nr,2H+1,size)')
        K = self.H if harmonics is None else harmonics
        if int(K) != K or K < 0:
            raise ValueError('nonnegative integer perturbation harmonics required')
        self.indices = np.arange(-K, K+1) if indices is None else np.asarray(indices)
        if (self.indices.ndim != 1 or not len(self.indices) or
                np.any(self.indices != self.indices.astype(int)) or
                np.any(np.diff(self.indices) != 1)):
            raise ValueError('indices must be a nonempty contiguous integer window')
        self.indices = self.indices.astype(int)
        if len(self.indices)%2 != 1:
            raise ValueError('perturbation window must contain 2K+1 signed harmonics')
        if indices is not None:
            if harmonics is not None and len(self.indices) != 2*K+1:
                raise ValueError('explicit window length must agree with perturbation harmonics')
            K = (len(self.indices)-1)//2
        self.K, self.beta_hat = int(K), float(beta_hat)
        if not np.isfinite(self.beta_hat):
            raise ValueError('finite beta_hat required')
        self.beta = problem.config.beta
        self.omega = problem.config.frequency(problem.ops[0])
        self.nh = len(self.indices)
        self.shape = (self.nr, self.nh, self.s)
        self.size = int(np.prod(self.shape))
        self.width = self.nh*self.s
        self.prescribed_outlet = prescribed_outlet
        self.check = check or (lambda: None)
        constraints = getattr(problem, 'outlet_constraints', {})
        if constraints and self.beta_hat:
            raise ValueError('modal outlet constraints need an explicit noninteger-angle model')
        if not self.beta and getattr(problem, 'outlet_alpha', 0j):
            raise ValueError('nonzero Robin carrier requires nonzero base beta')
        self.base_radial = (problem.D@self.base.reshape(self.nr, -1)).reshape(self.base.shape)
        self.J, self.B = self._assemble()
        self.b = self.B.diagonal().real

    def _linear(self, op, i, k):
        p, mu = self.problem, self.beta_hat+k*self.beta
        A, B, C = radial_matrices(op, 0, 0.)
        A += op.linear(0, mu)-op.linear(0, 0)-1j*k*self.omega/op.radius*op.M
        rows = p.velocity_rows
        sponge = p.outlet_sponge
        if sponge is not None and (mu != 0 or getattr(sponge, 'damp_mean', False)):
            A[rows, rows] += p.sponge_strength[i]/op.radius
        if i in (0, self.nr-1):
            A[rows] = B[rows] = C[rows] = 0
            if i == 0 or self.prescribed_outlet:
                A[rows, rows] = 1
            elif k in getattr(p, 'outlet_constraints', {}):
                constraint = p.outlet_constraints[k]
                A[rows] = constraint[:, :self.s]
                B[rows] = constraint[:, self.s:]
            else:
                alpha = getattr(p, 'outlet_alpha', 0j)
                # Continue the original harmonic Robin carrier by physical angle.
                number = mu/self.beta if self.beta else k
                A[rows, rows] = -(1j*number*alpha.real-abs(number)*alpha.imag)
                B[rows, rows] = 1
        return A, B, C

    def _assemble(self):
        start = monotonic()
        aa, bb, cc, qq, tt = [], [], [], [], []
        for i, op in enumerate(self.problem.ops):
            self.check()
            abc = [self._linear(op, i, int(k)) for k in self.indices]
            for col, dest in enumerate((aa, bb, cc)):
                dest.append(sparse.block_diag([sparse.csr_matrix(t[col]) for t in abc], format='csr'))
            shape = (self.width, self.width)
            if i in (0, self.nr-1) or not self.problem.config.nonlinear:
                qq.append(sparse.csr_matrix(shape)); tt.append(sparse.csr_matrix(shape))
                continue
            blocks = [[convection_block(op, self.base[i], self.base_radial[i],
                       int(j), int(k), self.beta_hat) for k in self.indices] for j in self.indices]
            for col, dest in ((0, qq), (1, tt)):
                dest.append(sparse.bmat([[sparse.csr_matrix(t[col]) for t in row]
                                         for row in blocks], format='csr'))
        D = sparse.kron(self.problem.D, sparse.eye(self.width), format='csr')
        D2 = sparse.kron(self.problem.D2, sparse.eye(self.width), format='csr')
        J = (sparse.block_diag(aa, format='csr') + sparse.block_diag(qq, format='csr')
             +(sparse.block_diag(bb, format='csr')+sparse.block_diag(tt, format='csr'))@D
             +sparse.block_diag(cc, format='csr')@D2).tocsr()
        J.sum_duplicates(); J.eliminate_zeros(); J.sort_indices()
        mass = np.zeros(self.shape)
        mass[1:-1, :, self.problem.velocity_rows] = 1/self.problem.radii[1:-1, None, None]
        self.assembly_seconds = monotonic()-start
        return J, sparse.diags(mass.ravel(), format='csr')

    def matvec(self, vector):
        self.check()
        return self.J@np.asarray(vector).ravel()

    def residual(self, value, vector, *, left=False):
        if left:
            jv, bv = self.J.conj().T@vector, self.B.conj().T@vector
            value = np.conj(value)
        else:
            jv, bv = self.J@vector, self.B@vector
        return float(np.linalg.norm(jv+value*bv)/max(
            np.linalg.norm(jv)+max(abs(value), 1)*np.linalg.norm(bv), 1e-300))

    def energies(self, vector):
        """Physical velocity quadratic norm, R**2 plus cylindrical volume R*dR."""
        p = self.problem
        q = np.asarray(vector).reshape(self.nr, self.nh, 4, p.ops[0].m)[:, :, :3]
        density = np.einsum('rkvi,ij,rkvj->rk', q.conj(), p.ops[0].grid.mass, q).real
        density = np.maximum(density, 0)*p.radii[:, None]**2
        w = np.empty(self.nr)
        w[0] = (p.radii[1]-p.radii[0])/2
        w[-1] = (p.radii[-1]-p.radii[-2])/2
        w[1:-1] = (p.radii[2:]-p.radii[:-2])/2
        energy = density*(w*p.radii)[:, None]
        total = float(energy.sum())
        radial = density.sum(axis=1)
        split = self.nr//2
        return dict(total=total, harmonic=energy.sum(axis=0).tolist(),
                    harmonic_fraction=(energy.sum(axis=0)/max(total, 1e-300)).tolist(),
                    radial_density=radial.tolist(), radial_energy=energy.sum(axis=1).tolist(),
                    peak_R=float(p.radii[np.argmax(radial)]),
                    radial_left_fraction=float(energy[:split].sum()/max(total, 1e-300)),
                    radial_right_fraction=float(energy[split:].sum()/max(total, 1e-300)))

    def shift_invert(self, shift, *, k=6, ncv=None, tol=1e-9, maxiter=250,
                     factor_factory=None, left=True, residual_accept=1e-8,
                     inverse_accept=1e-8, v0=None, extended_refinement=False,
                     inverse_diagnostic=None, inverse_policy='backward',
                     normwise_accept=1e-12, componentwise_accept=1e-10,
                     adjoint_inverse_policy='normwise_rhs_relative',
                     adjoint_inverse_accept=1e-5, reused_right=None,
                     reused_right_complete=False):
        """D2-style LU, two best-iterate corrections, then bounded GMRES repair.

        Default right inverse certification is D2f: normwise backward error
        <=1e-12 and componentwise backward error <=1e-10. Default adjoint
        certification is D2g: normwise <=1e-12 and original RHS-relative error
        <=1e-5; its componentwise error is diagnostic because T^H has exact
        zero constraint rows. ``inverse_accept`` only gates the optional
        rhs_relative policy, and is diagnostic under the default right policy.
        Final right and left eigenpairs independently require the same original
        complex128 pencil residual <=residual_accept (default 1e-8).

        Factor protocol: solve(rhs), solve_adjoint(rhs), optional close()/info.
        Default uses row-scaled SuperLU. Left and right pencils share one LU.
        Every inverse action is certified against the original sparse pencil.
        ``reused_right=(values,vectors)`` skips right Arnoldi only after all
        supplied physical eigenpairs pass shape, finiteness and residual checks;
        the caller must explicitly preserve its original completion flag.
        """
        self.check()
        choices = ('backward', 'rhs_relative', 'normwise_rhs_relative')
        if inverse_policy not in choices or adjoint_inverse_policy not in choices:
            raise ValueError('unknown inverse certification policy')
        if reused_right is not None:
            reuse_values = np.asarray(reused_right[0], complex)
            reuse_vectors = np.asarray(reused_right[1], complex)
            if (reuse_values.shape!=(k,) or reuse_vectors.shape!=(self.size, k) or
                not np.all(np.isfinite(reuse_values)) or not np.all(np.isfinite(reuse_vectors))):
                raise ValueError('reused_right must contain finite k values and physical right vectors')
            if any(self.residual(z, reuse_vectors[:, i])>residual_accept for i,z in enumerate(reuse_values)):
                raise ValueError('reused right eigenpair fails current physical pencil certificate')
        start = monotonic()
        shift = complex(shift)
        matrix = (self.J+shift*self.B).tocsc()
        factor = (factor_factory or _SuperLU)(matrix)
        stats = dict(shift=[shift.real, shift.imag], inverse_calls=0,
                     maximum_inverse_relative=0., refinement_calls=0,
                     maximum_inverse_float64_relative=0.,
                     maximum_normwise_backward=0., maximum_componentwise_backward=0.,
                     inverse_policy='side_specific; see right_linear_policy and left_linear_policy',
                     right_inverse_policy=inverse_policy,
                     normwise_accept=normwise_accept, right_componentwise_accept=componentwise_accept,
                     maximum_componentwise_backward_is_diagnostic=True,
                     right_rhs_relative_gate=None if inverse_policy=='backward' else inverse_accept,
                     inverse_residual_precision='complex256' if extended_refinement else 'complex128',
                     krylov_repairs=0, arpack_complete=True, left_arpack_complete=True)
        recovery = {}

        def inverse(adjoint=False):
            side = 'left' if adjoint else 'right'
            policy = adjoint_inverse_policy if adjoint else inverse_policy
            relative_accept = adjoint_inverse_accept if adjoint else inverse_accept
            stats[side+'_linear_policy'] = dict(
                name='D2g' if policy=='normwise_rhs_relative' else ('D2f' if policy=='backward' else 'rhs_relative'),
                normwise_accept=normwise_accept if policy!='rhs_relative' else None,
                componentwise_accept=componentwise_accept if policy=='backward' else None,
                rhs_relative_accept=relative_accept if policy!='backward' else None,
                componentwise_diagnostic_only=policy!='backward')
            for key in ('normwise_backward', 'componentwise_backward', 'rhs_relative'):
                stats[side+'_maximum_'+key] = 0.
            A = matrix.conj().T if adjoint else matrix
            absA = abs(A).tocsr()
            anorm = float(np.max(np.asarray(absA.sum(axis=1))))
            csr = A.tocsr()
            singleton_rows = np.flatnonzero(np.diff(csr.indptr)==1)
            offsets = csr.indptr[singleton_rows]
            singleton_cols, singleton_vals = csr.indices[offsets], csr.data[offsets]
            extended = A.astype(np.clongdouble) if extended_refinement else A
            accurate = LinearOperator(A.shape,
                matvec=lambda y:np.asarray(extended@y, dtype=complex), dtype=complex)
            solve = factor.solve_adjoint if adjoint else factor.solve
            def projected_solve(rhs):
                y = np.asarray(solve(rhs), complex)
                y[singleton_cols] = rhs[singleton_rows]/singleton_vals
                return y

            def measure(y, rhs, denom):
                r = np.asarray(np.asarray(rhs, dtype=extended.dtype)-extended@y, dtype=complex)
                relative = float(np.linalg.norm(r)/denom)
                normwise = float(np.linalg.norm(r, np.inf)/max(
                    anorm*np.linalg.norm(y, np.inf)+np.linalg.norm(rhs, np.inf), 1e-300))
                componentwise = float(np.max(abs(r)/np.maximum(absA@abs(y)+abs(rhs), 1e-300)))
                if policy=='backward':
                    score = max(normwise/normwise_accept, componentwise/componentwise_accept)
                elif policy=='normwise_rhs_relative':
                    score = max(normwise/normwise_accept, relative/relative_accept)
                else:
                    score = relative/relative_accept
                return r, relative, normwise, componentwise, score
            def action(x, raw=False):
                self.check()
                # True adjoint of T=-K^-1 B is -B^H K^-H. Its Arnoldi
                # vectors y recover the physical pencil left mode K^-H y.
                rhs = np.asarray(x) if adjoint else -self.b*x
                denom = max(np.linalg.norm(rhs), 1e-300)
                y = projected_solve(rhs)
                best, err = y.copy(), np.inf
                best_score = np.inf
                direct = []
                krylov_info = None
                for attempt in range(3):
                    r, relative, normwise, componentwise, score = measure(y, rhs, denom)
                    direct.append(dict(accurate_relative=float(relative),
                        float64_relative=float(np.linalg.norm(rhs-A@y)/denom),
                        amplification=float(np.linalg.norm(y)/denom),
                        normwise_backward=normwise, componentwise_backward=componentwise))
                    if score < best_score:
                        best, err, best_score = y.copy(), relative, score
                    if score <= 1 or attempt == 2:
                        break
                    y += projected_solve(r)
                    y[singleton_cols] = rhs[singleton_rows]/singleton_vals
                    stats['refinement_calls'] += 1
                if best_score > 1:
                    stats['krylov_repairs'] += 1
                    def precondition(rhs):
                        self.check()
                        return projected_solve(rhs)
                    pre = LinearOperator(A.shape, matvec=precondition, dtype=complex)
                    candidate, krylov_info = gmres(accurate, rhs, x0=best, M=pre, rtol=1e-10, atol=0,
                                    restart=35, maxiter=5,
                                    callback=lambda value:self.check(), callback_type='pr_norm')
                    candidate[singleton_cols] = rhs[singleton_rows]/singleton_vals
                    _, rel, _, _, score = measure(candidate, rhs, denom)
                    if score < best_score:
                        best, err, best_score = candidate, rel, score
                self.check()
                _, err, normwise, componentwise, best_score = measure(best, rhs, denom)
                stats['inverse_calls'] += 1
                stats['maximum_inverse_relative'] = max(stats['maximum_inverse_relative'], err)
                float64_relative = float(np.linalg.norm(rhs-A@best)/denom)
                stats['maximum_inverse_float64_relative'] = max(
                    stats['maximum_inverse_float64_relative'], float64_relative)
                stats['maximum_normwise_backward'] = max(stats['maximum_normwise_backward'], normwise)
                stats['maximum_componentwise_backward'] = max(stats['maximum_componentwise_backward'], componentwise)
                for key, value in (('normwise_backward', normwise),
                    ('componentwise_backward', componentwise), ('rhs_relative', float64_relative)):
                    stats[side+'_maximum_'+key] = max(stats[side+'_maximum_'+key], value)
                if inverse_diagnostic is not None:
                    failed = not np.isfinite(best_score) or best_score > 1
                    record = dict(call=stats['inverse_calls'], adjoint=adjoint,
                        direct=direct, krylov_info=krylov_info,
                        accurate_relative=err, float64_relative=float64_relative,
                        rhs_norm=float(denom), amplification=float(np.linalg.norm(best)/denom),
                        residual_precision=stats['inverse_residual_precision'],
                        inverse_policy=policy,
                        inverse_accept=None if policy=='backward' else relative_accept,
                        normwise_backward=normwise, componentwise_backward=componentwise,
                        normwise_accept=normwise_accept,
                        componentwise_accept=componentwise_accept if policy=='backward' else None,
                        componentwise_is_diagnostic=policy!='backward',
                        accepted=not failed)
                    if failed or stats['inverse_calls']==1:
                        record['cancellation_ratio'] = float(np.linalg.norm(abs(A)@abs(best))/denom)
                    arrays = dict(rhs=rhs, solution=best) if failed else None
                    inverse_diagnostic(record, arrays)
                if not np.isfinite(best_score) or best_score > 1:
                    raise RuntimeError(f'uncertified inverse action: rhs={err:.3e}, normwise={normwise:.3e}, componentwise={componentwise:.3e}')
                return -self.b*best if adjoint and not raw else best
            if adjoint:
                recovery['left'] = lambda x:action(x, raw=True)
            return LinearOperator(matrix.shape, matvec=action, dtype=complex)

        def arnoldi(T, seed, side):
            try:
                return eigs(T, k=k, which='LM', ncv=ncv, tol=tol, maxiter=maxiter, v0=seed)
            except ArpackNoConvergence as exc:
                stats[side+'arpack_complete' if side else 'arpack_complete'] = False
                return exc.eigenvalues, exc.eigenvectors
        try:
            seed = np.random.default_rng(731).normal(size=self.size).astype(complex) if v0 is None else v0
            if reused_right is None:
                right_started = monotonic()
                mu, right = arnoldi(inverse(), seed, '')
                values = shift+1/mu
                stats['right_spectrum_reused'] = False
                stats['right_arnoldi_seconds'] = monotonic()-right_started
            else:
                values, right = reuse_values, reuse_vectors
                stats['right_spectrum_reused'] = True
                stats['arpack_complete'] = bool(reused_right_complete)
                stats['right_arnoldi_seconds'] = 0.
                stats['right_linear_policy'] = dict(name='supplied eigenpairs; original physical residuals recertified')
            order = np.argsort(-values.real)
            values, right = values[order], right[:, order]
            lv, left_vectors = None, None
            if left and len(values):
                left_started = monotonic()
                try:
                    lm, lv = arnoldi(inverse(True), seed.conj(), 'left_')
                    # lm=1/(conj(sigma)-conj(shift)); convert to sigma.
                    left_values = np.conj(np.conj(shift)+1/lm)
                    left_vectors = np.column_stack([recovery['left'](
                        lv[:, np.argmin(abs(left_values-z))]) for z in values]) if len(lm) else None
                except Exception as exc:
                    stats['left_arpack_complete'] = False
                    stats['left_error'] = type(exc).__name__+': '+str(exc)
                stats['left_arnoldi_seconds'] = monotonic()-left_started
            records = []
            for i, value in enumerate(values):
                rr = self.residual(value, right[:, i])
                record = dict(value=[float(value.real), float(value.imag)], residual=rr,
                              accepted=bool(rr <= residual_accept), right=self.energies(right[:, i]),
                              left_accepted=False if left else None)
                if left_vectors is not None:
                    lr = self.residual(value, left_vectors[:, i], left=True)
                    record.update(left_residual=lr, left=self.energies(left_vectors[:, i]),
                                  left_accepted=bool(lr <= residual_accept),
                                  left_eigenvalue_distance=float(np.min(abs(left_values-value))))
                records.append(record)
            stats.update(seconds=monotonic()-start, matrix_nnz=int(self.J.nnz),
                         assembly_seconds=self.assembly_seconds,
                         factor=getattr(factor, 'info', {}), exhaustive_spectrum=False)
            return Spectrum(values, right, left_vectors, records, stats)
        finally:
            if hasattr(factor, 'close'):
                factor.close()


class _SuperLU:
    def __init__(self, A):
        self.scale = np.maximum(np.asarray(abs(A).max(axis=1).toarray()).ravel(), 1e-300)
        self.factor = splu((sparse.diags(1/self.scale)@A).tocsc())
        self.info = dict(backend='row_scaled_superlu')

    def solve(self, rhs):
        return self.factor.solve(rhs/self.scale)

    def solve_adjoint(self, rhs):
        return self.factor.solve(rhs, trans='H')/self.scale
