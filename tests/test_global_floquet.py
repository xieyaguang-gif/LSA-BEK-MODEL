"""Independent numerical oracles for the full complex global Floquet pencil.

Small tests: .venv/bin/python -m unittest discover -s tests -p test_global_floquet.py
The opt-in actual N24/H4 D2 reference and dr comparison are exercised by the
pilot driver's --stage validate and stored in VALIDATION.json.
"""
import importlib.util
import hashlib
import os
import sys
from pathlib import Path
import unittest
sys.dont_write_bytecode = True
for _variable in ('OPENBLAS_NUM_THREADS', 'OMP_NUM_THREADS', 'MKL_NUM_THREADS',
                  'NUMEXPR_NUM_THREADS', 'VECLIB_MAXIMUM_THREADS', 'BLIS_NUM_THREADS'):
    os.environ[_variable] = '1'
sys.path.insert(0, str(Path(__file__).resolve().parents[1]/'src'))

import numpy as np
from scipy.linalg import eigvals

from llsa.secondary.global_floquet import GlobalFloquet, convection_block
from llsa.secondary.global_radial import GlobalRadial, OutletSponge
from llsa.secondary.radial import RadialConfig, pack_field, unpack_field


METRICS = {}


def fixture(H=1, nr=5, omega=2.3):
    p = GlobalRadial(np.linspace(400., 420., nr),
                     RadialConfig(ro=-1., beta=68., n=8, harmonics=H, omega=omega),
                     outlet_sponge=OutletSponge(405., 10., .7))
    rng = np.random.default_rng(741)
    q = (rng.normal(size=(nr, 2*H+1, p.s))+1j*rng.normal(size=(nr, 2*H+1, p.s)))*1e-4
    q[:, H] = q[:, H].real
    for k in range(1, H+1):
        q[:, H-k] = q[:, H+k].conj()
    return p, q


def relative(a, b):
    return float(np.linalg.norm(a-b)/max(np.linalg.norm(b), 1e-300))


def finite_spectrum(J, B):
    z = eigvals(J.toarray(), B.toarray())
    return z[np.isfinite(z) & (abs(z)<1000)]


def spectrum_distance(a, b):
    return max(float(np.max(np.min(abs(a[:, None]-b[None, :]), axis=1))),
               float(np.max(np.min(abs(b[:, None]-a[None, :]), axis=1))))


def lower_radial_validation():
    """Real unstable lower-branch spectra at the same controlled amplitude."""
    import json
    root = Path(__file__).resolve().parents[1]
    d2 = root/'results/fullbvp_global_20261030/phaseD/D2'
    points = [d2/'production_v6/baseline_rerun_lower/lower719p948669_H4',
              d2/'production_v6/radial/lower719p948669_H4']
    data = []
    for point in points:
        correction = json.loads((point/'corrected.json').read_text())
        field = Path(correction['fields'])
        digest = hashlib.sha256(field.read_bytes()).hexdigest()
        if digest != correction['fields_sha256']:
            raise AssertionError('radial reference field checksum mismatch')
        done = point/'center_00/DONE.json'
        spectrum = json.loads(done.read_text())
        accepted = [r for r in spectrum['eigenpairs'] if r['accepted'] and r['original_relative']<=1e-8]
        unstable = [r for r in accepted if r['value'][0]>1e-8]
        data.append(dict(dr=correction['actual_dr'], A1=correction['after']['A1'],
            Rout=correction['after']['Rout'], omega=correction['after']['omega'],
            unstable=unstable, accepted_count=len(accepted), spectrum=str(done),
            spectrum_sha256=hashlib.sha256(done.read_bytes()).hexdigest(),
            field=str(field), field_sha256=digest, search_shift=spectrum['sigma']))
    change = []
    for r in data[0]['unstable']:
        a = complex(*r['value'])
        b = min([complex(*r['value']) for r in data[1]['unstable']], key=lambda x:abs(x-a))
        change.append(dict(coarse=[a.real, a.imag], fine=[b.real, b.imag],
                           absolute=float(abs(a-b)), relative=float(abs(a-b)/abs(a))))
    amplitude_relative = abs(data[0]['A1']-data[1]['A1'])/data[0]['A1']
    return dict(passed=bool(change and amplitude_relative<5e-8),
        points=data, unstable_changes=change, amplitude_relative=amplitude_relative,
        interpretation='grid sensitivity at fixed lower-branch amplitude; Rout and omega are corrected BVP unknowns',
        convergence_established=False, exhaustive_spectrum=False,
        limitation='preserved independently corrected D2 spectra; one certified unstable sample at this shift; nonzero-beta_hat convergence untested')


def independent_projection(p, base, d, beta_hat, indices):
    """Physical-space direct products at independent phase quadrature points.

    The Bloch factor is kept explicitly in theta; only its product with the
    periodic base is demodulated before projection. No convolution-block calls.
    """
    nr, H, m = len(p.radii), p.H, p.ops[0].m
    q = base.reshape(nr, 2*H+1, 4, m)
    v = d.reshape(nr, len(indices), 4, m)
    qr = (p.D@base.reshape(nr, -1)).reshape(q.shape)
    vr = (p.D@d.reshape(nr, -1)).reshape(v.shape)
    qz, vz = q[:, :, :3]@p.ops[0].grid.D.T, v[:, :, :3]@p.ops[0].grid.D.T
    theta = np.arange(37)*2*np.pi/(37*p.config.beta)
    base_phase = np.exp(1j*np.outer(np.arange(-H, H+1)*p.config.beta, theta))
    physical = beta_hat+np.asarray(indices)*p.config.beta
    delta_phase = np.exp(1j*np.outer(physical, theta))
    expand = lambda a, e: np.einsum('rhvi,ht->rtvi', a, e)
    a, ar = expand(q, base_phase), expand(qr, base_phase)
    b, br = expand(v, delta_phase), expand(vr, delta_phase)
    az, bz = expand(qz, base_phase), expand(vz, delta_phase)
    at = expand(q[:, :, :3], 1j*np.arange(-H, H+1)[:, None]*p.config.beta*base_phase)
    bt = expand(v[:, :, :3], 1j*physical[:, None]*delta_phase)
    R = p.radii[:, None, None, None]
    prod = (a[:, :, 0, None]*(br[:, :, :3]+b[:, :, :3]/R)
            + b[:, :, 0, None]*(ar[:, :, :3]+a[:, :, :3]/R)
            + a[:, :, 1, None]*bt/R+b[:, :, 1, None]*at/R
            + a[:, :, 2, None]*bz+b[:, :, 2, None]*az)
    prod[:, :, 0] -= 2*a[:, :, 1]*b[:, :, 1]/p.radii[:, None, None]
    prod[:, :, 1] += (a[:, :, 0]*b[:, :, 1]+b[:, :, 0]*a[:, :, 1])/p.radii[:, None, None]
    out = np.zeros(v.shape, complex)
    out[:, :, :3] = np.einsum('rtvi,kt->rkvi', prod, delta_phase.conj())/len(theta)
    out = out.reshape(d.shape)
    out[:, :, p.ops[0].boundary_rows] = 0
    out[0][:, p.velocity_rows] = 0
    out[-1][:, p.velocity_rows] = 0
    return out


class TestGlobalFloquet(unittest.TestCase):
    def test_cosine_coordinates_keep_complex_degrees_and_adjoint(self):
        import sys
        root = Path(__file__).resolve().parents[1]
        sys.path.insert(0, str(root/'examples'))
        from global_floquet_pilot import CoordinateFactor, Deadline
        p, q = fixture()
        f = GlobalFloquet(p, q, 17.)
        A = f.J+.05*f.B
        factor = CoordinateFactor(A, f, Deadline(60.))
        try:
            rng = np.random.default_rng(942)
            rhs = rng.normal(size=f.size)+1j*rng.normal(size=f.size)
            right = factor.solve(rhs)
            left = factor.solve_adjoint(rhs)
            rr = relative(A@right, rhs)
            lr = relative(A.conj().T@left, rhs)
            self.assertLess(rr, 1e-8)
            self.assertLess(lr, 1e-8)
            self.assertEqual(factor.S.shape, (f.size, f.size))
            METRICS['cosine_complex_coordinate_adapter'] = dict(right_relative=rr,
                left_relative=lr, beta_hat=17., coordinates_complex=True,
                transformed_dimension=f.size, physical_dimension=f.size)
        finally:
            factor.close()

    def test_e_actual_unstable_lower_radial_refinement(self):
        report = lower_radial_validation()
        self.assertTrue(report['passed'])
        self.assertGreater(len(report['unstable_changes']), 0)
        METRICS['e_actual_radial'] = report

    def test_a_real_jacobian_and_full_nonlinear_finite_difference(self):
        p, q = fixture(H=2)
        p.outlet_alpha = .02+.003j
        f = GlobalFloquet(p, q)
        rng = np.random.default_rng(39)
        x = pack_field(q)
        v = rng.normal(size=len(x))
        d = unpack_field(v, len(q), p.H, p.s)
        _, J = p.evaluate(x, q[0], jacobian=True)
        error = relative(pack_field((f.J@d.ravel()).reshape(d.shape)), J@v)
        self.assertLess(error, 1e-10)
        # Full nonlinear residual, including all endpoint and z boundary rows.
        step = 2e-6
        fd = (p.evaluate(x+step*v, q[0])-p.evaluate(x-step*v, q[0]))/(2*step)
        fd_error = relative(fd, J@v)
        self.assertLess(fd_error, 1e-8)
        METRICS['a_b_real_jacobian'] = dict(relative=error, threshold=1e-10,
                                            full_nonlinear_fd_relative=fd_error,
                                            fd_step=step, fd_threshold=1e-8)

    def test_b_nonzero_bloch_independent_projection_K_not_H(self):
        p, q = fixture(H=1, nr=7)
        rng = np.random.default_rng(372)
        K, beta_hat = 2, 17.
        indices = np.arange(-K, K+1)
        d = rng.normal(size=(len(q), len(indices), p.s))+1j*rng.normal(size=(len(q), len(indices), p.s))
        dr = (p.D@d.reshape(len(q), -1)).reshape(d.shape)
        qr = (p.D@q.reshape(len(q), -1)).reshape(q.shape)
        out = np.zeros_like(d)
        for i in range(1, len(q)-1):
            for a, j in enumerate(indices):
                for b, k in enumerate(indices):
                    Q, T = convection_block(p.ops[i], q[i], qr[i], j, k, beta_hat)
                    out[i, a] += Q@d[i, b]+T@dr[i, b]
        expected = independent_projection(p, q, d, beta_hat, indices)
        error = relative(out, expected)
        self.assertLess(error, 1e-12)
        f = GlobalFloquet(p, q, beta_hat, harmonics=K)
        self.assertEqual(f.nh, 5)
        METRICS['b_nonzero_projection'] = dict(relative=error, threshold=1e-12,
                                                H=1, K=K, beta_hat=beta_hat, quadrature=37)

    def test_c_laminar_decoupling_and_linear_spectrum(self):
        p, q = fixture()
        q[:] = 0
        f = GlobalFloquet(p, q, 17.)
        # Independent existing GlobalRadial linear operator with beta+beta_hat.
        ref = GlobalRadial(p.radii, RadialConfig(ro=-1., beta=85., n=8, harmonics=1, omega=0.),
                           outlet_sponge=p.outlet_sponge)
        real = ref._linear_matrix(False)
        s, nr = p.s, len(q)
        re = (np.arange(nr)[:, None]*(3*s)+np.arange(s)[None, :]+s).ravel()
        im = re+s
        Aref = real[re][:, re]+1j*real[im][:, re]
        ix = (np.arange(nr)[:, None]*3*s+2*s+np.arange(s)[None, :]).ravel()
        A = f.J[ix][:, ix]+1j*p.config.omega*f.B[ix][:, ix]
        error = relative(A.toarray(), Aref.toarray())
        self.assertLess(error, 1e-12)
        mask = f.J.copy().tolil()
        for h in range(3):
            ixh = (np.arange(nr)[:, None]*3*s+h*s+np.arange(s)[None, :]).ravel()
            mask[np.ix_(ixh, ixh)] = 0
        self.assertEqual(mask.tocsr().nnz, 0)
        distance = spectrum_distance(finite_spectrum(A, f.B[ix][:, ix]),
                                     finite_spectrum(Aref, f.B[ix][:, ix]))
        self.assertLess(distance, 1e-7)
        METRICS['c_laminar'] = dict(matrix_relative=error, spectral_distance=distance,
                                    off_diagonal_nnz=0, physical_azimuthal_number=85.)

    def test_d_translation_time_gauge_and_conjugacy(self):
        p, q = fixture()
        p.outlet_alpha = .02+.003j
        a = GlobalFloquet(p, q, 17.)
        b = GlobalFloquet(p, q, 17.+68., indices=a.indices-1)
        translation = relative(b.J.toarray(), (a.J+1j*p.config.omega*a.B).toarray())
        self.assertLess(translation, 1e-12)
        c = GlobalFloquet(p, q, -17.)
        perm = np.arange(a.size).reshape(a.shape)[:, ::-1].ravel()
        conjugacy = relative(c.J[perm][:, perm].toarray(), a.J.conj().toarray())
        self.assertLess(conjugacy, 1e-12)
        za = finite_spectrum(a.J, a.B)
        zb = finite_spectrum(b.J, b.B)
        zc = finite_spectrum(c.J, c.B)
        # eigvals above returns eigenvalues of J v = z B v, sigma=-z.
        trans_spectrum = spectrum_distance(zb, za+1j*p.config.omega)
        conj_spectrum = spectrum_distance(zc, za.conj())
        self.assertLess(trans_spectrum, 1e-7)
        self.assertLess(conj_spectrum, 1e-7)
        METRICS['d_symmetries'] = dict(matrix_translation_relative=translation,
            matrix_conjugacy_relative=conjugacy, spectral_translation_distance=trans_spectrum,
            spectral_conjugacy_distance=conj_spectrum, sigma_shift_imag=-p.config.omega,
            condition='translate index window by -1; fixed symmetric finite windows are not isospectral')

    def test_shift_invert_right_and_left_original_pencil(self):
        p, q = fixture()
        f = GlobalFloquet(p, q, 17.)
        result = f.shift_invert(.05, k=3, ncv=24, tol=1e-10, maxiter=250,
                               extended_refinement=True)
        self.assertEqual(len(result.values), 3)
        self.assertIsNotNone(result.left_vectors)
        for record in result.records:
            self.assertTrue(record['accepted'])
            self.assertTrue(record['left_accepted'])
        METRICS['shift_invert'] = dict(right_max=max(r['residual'] for r in result.records),
            left_max=max(r['left_residual'] for r in result.records),
            inverse_rhs_relative_max=result.stats['maximum_inverse_relative'],
            inverse_normwise_max=result.stats['maximum_normwise_backward'],
            inverse_componentwise_max=result.stats['maximum_componentwise_backward'], threshold=1e-8)

    def test_reused_right_adjoint_and_original_pencil_recertification(self):
        p, q = fixture()
        f = GlobalFloquet(p, q, 6.8)
        right = f.shift_invert(.05, k=2, ncv=20, left=False)
        completed = f.shift_invert(.05, k=2, ncv=20,
            reused_right=(right.values, right.right_vectors),
            reused_right_complete=right.stats['arpack_complete'])
        self.assertTrue(completed.stats['right_spectrum_reused'])
        self.assertEqual(completed.stats['left_linear_policy']['name'], 'D2g')
        for r in completed.records:
            self.assertTrue(r['accepted'])
            self.assertTrue(r['left_accepted'])
        with self.assertRaises(ValueError):
            f.shift_invert(.05, k=2, reused_right=(right.values+1., right.right_vectors))


def run_small_validation():
    suite = unittest.defaultTestLoader.loadTestsFromTestCase(TestGlobalFloquet)
    result = unittest.TextTestRunner(verbosity=1).run(suite)
    return dict(passed=result.wasSuccessful(), tests_run=result.testsRun,
                failures=[str(x[0]) for x in result.failures],
                errors=[str(x[0]) for x in result.errors], metrics=dict(METRICS))


class TestActualD2Integration(unittest.TestCase):
    @unittest.skipUnless(os.environ.get('LLSA_FLOQUET_INTEGRATION') == '1',
                         'run driver --stage validate, then enable LLSA_FLOQUET_INTEGRATION=1')
    def test_actual_D2_eigenvalue_phase_and_grid_sensitivity(self):
        import json
        root = Path(__file__).resolve().parents[1]
        d = json.loads((root/'results/FULLBVP_PHASE2_20261104/global_floquet/VALIDATION.json').read_text())
        self.assertTrue(d['passed'])
        self.assertLessEqual(d['actual_D2_action_relative'], 1e-10)
        self.assertLessEqual(d['phase_mode_residual'], 1e-8)
        self.assertLessEqual(d['known_D2_error'], 1e-7)
        self.assertLessEqual(d['phase_eigenvalue_abs'], 1e-7)
        self.assertEqual(len(d['actual_spectrum']['records']), 4)
        self.assertTrue(d['actual_spectrum']['stats']['arpack_complete'])
        self.assertTrue(d['actual_spectrum']['stats']['left_arpack_complete'])
        for r in d['actual_spectrum']['records']:
            self.assertLessEqual(r['residual'], 1e-8)
            self.assertLessEqual(r['left_residual'], 1e-8)
        self.assertTrue(d['radial_convergence']['passed'])
        self.assertTrue(lower_radial_validation()['passed'])


if __name__ == '__main__':
    unittest.main()
