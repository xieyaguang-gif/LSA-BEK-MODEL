import numpy as np
import pytest
from scipy import linalg

from llsa.secondary.discretization import pier_grid
from llsa.secondary.operators import PierOperators
from llsa.secondary.primary import PrimaryWave, WaveSolver, pack, unpack
from llsa.secondary.floquet import (FloquetProblem, secondary_saddle, secondary_spectrum,
                                    secondary_derivatives, secondary_spatial_near)
from llsa.secondary.validation import compare_tables, primary_absolute_threshold
from llsa.secondary.diagnostics import secondary_kinematic_defects


@pytest.fixture(scope='module')
def wave():
    solver = WaveSolver(PierOperators(n=30, mean_mode='parallel'), 1)
    return solver.solve(solver.bootstrap(), frequency=50.5)


def test_rational_mapping():
    g = pier_grid(60)
    assert g.z[0] == 0 and np.isposinf(g.z[-1])
    assert np.all(np.diff(g.z) > 0)
    y = np.exp(-g.z)
    np.testing.assert_allclose(g.D@y, -y, atol=2e-6)
    np.testing.assert_allclose(g.D2@y, y, atol=1e-4)
    assert abs(y@g.mass@y-.5) < 1e-7


def test_table1_primary_absolute():
    result = primary_absolute_threshold(n=30)
    assert abs(result['radius']-507.537) < .02
    assert abs(result['omega'][0]-50.5043) < .002
    assert abs(complex(*result['alpha'])-(.21757-.12208j)) < 3e-5


def test_primary_newton_jacobian(wave):
    s = WaveSolver(wave.operators, 1)
    x = pack(wave.modes)
    v = np.random.default_rng(7).normal(size=len(x))
    h = 1e-7
    fd = (s.residual(unpack(x+h*v, 1, s.op.size), wave.alpha, wave.omega)-
          s.residual(unpack(x-h*v, 1, s.op.size), wave.alpha, wave.omega))/(2*h)
    np.testing.assert_allclose(s.jacobian(wave.modes, wave.alpha, wave.omega)@v,
                               fd, rtol=1e-7, atol=1e-8)


def test_cylindrical_mean_continuity(wave):
    op = PierOperators(n=30, mean_mode='cylindrical')
    w = WaveSolver(op, 1).solve(wave, frequency=50.5)
    q = w.modes[w.harmonics].reshape(4, op.m)
    np.testing.assert_allclose((op.grid.D@q[2]+q[0]/op.radius)[1:-1], 0, atol=1e-10)
    assert abs(q[2, -1]) > 1e-7


def test_table2_primary_and_harmonic_convergence(wave):
    assert compare_tables(wave)['table2']['passed']
    w = WaveSolver(wave.operators, 3).solve(wave, frequency=50.5)
    assert compare_tables(w)['table2']['passed']
    assert np.all(np.diff(w.energies()) < 0)


def test_primary_persistence_and_prescribed_alpha(wave, tmp_path):
    path = tmp_path/'wave.npz'
    wave.save(path)
    restored = PrimaryWave.load(path)
    np.testing.assert_array_equal(restored.modes, wave.modes)
    new = WaveSolver(restored.operators, 1).solve(restored, wavenumber=.35)
    assert new.alpha == .35 and abs(new.omega-50.5) > .1
    assert new.residual < 1e-9


def test_arclength_corrector(wave):
    solver = WaveSolver(wave.operators, 1)
    previous = solver.solve(wave, frequency=50.4)
    following = solver.arclength_step(previous, wave, step=.0005)
    assert following.alpha > wave.alpha
    assert following.omega > wave.omega
    assert following.residual < 1e-9


def test_energy_parseval_and_reality(wave):
    phases = np.linspace(0, 2*np.pi, 13, endpoint=False)
    vel = wave.fields(phases)[:, :3]
    vel -= vel.mean(axis=0)
    energy = .5*np.einsum('pvi,ij,pvj->', vel, wave.operators.grid.mass, vel)/len(phases)
    assert abs(energy-wave.energies().sum()) < 1e-8
    np.testing.assert_allclose(vel[:, :, [0, -1]], 0, atol=1e-8)


def test_fourier_conjugate_symmetry(wave):
    nlin = wave.operators.nonlinear(wave.modes, wave.alpha)
    np.testing.assert_allclose(nlin[0], nlin[2].conj(), atol=1e-14)
    np.testing.assert_allclose(nlin[1].imag, 0, atol=1e-14)


def test_floquet_translation_neutral_mode(wave):
    p = FloquetProblem(wave, beta_hat=0, harmonics=1)
    phase = (1j*np.arange(-1, 2)[:, None]*wave.modes).ravel()
    assert max(abs(p.matrix(0)@phase)) < 1e-9
    difference = p.matrix(.007+.02j)-p._matrix(.007+.02j)
    assert max(abs(difference.data), default=0) < 1e-10


def test_floquet_laminar_sidebands(wave):
    laminar = PrimaryWave(wave.operators, wave.alpha, wave.omega,
                          np.zeros_like(wave.modes), 0, [])
    p = FloquetProblem(laminar, beta_hat=20, harmonics=1)
    a, target = .006-.02j, 8+1j
    s = secondary_spectrum(p, a, target=target)
    independent = []
    op = wave.operators
    for j in (-1, 0, 1):
        vals = linalg.eig(op.linear(a+j*wave.alpha, 20+j*op.beta), 1j*op.M, right=False)
        independent.extend(vals[np.isfinite(vals)]*op.radius-j*wave.omega)
    for omega in s.omega:
        assert min(abs(np.asarray(independent)-omega)) < 1e-7


def test_floquet_polynomial_and_derivatives(wave):
    p = FloquetProblem(wave, harmonics=1)
    a, target = .004-.02j, 8.5+1j
    difference = p.matrix(a)-p._matrix(a)
    assert max(abs(difference.data), default=0) < 1e-10
    s = secondary_spectrum(p, a, target=target)
    k = s.nearest(target)
    first, second = secondary_derivatives(p, a, s, k)
    omega, h = s.omega[k], 1e-5
    vals = []
    for shift in (-h, h):
        spec = secondary_spectrum(p, a+shift, target=omega)
        vals.append(spec.omega[spec.nearest(omega)])
    np.testing.assert_allclose(first, (vals[1]-vals[0])/(2*h), rtol=1e-5, atol=2e-4)
    np.testing.assert_allclose(second, (vals[1]+vals[0]-2*omega)/h**2, rtol=1e-3, atol=.5)


def test_secondary_spatial_roundtrip(wave):
    p = FloquetProblem(wave, harmonics=1)
    alpha, target = .004-.02j, 8.5+1j
    s = secondary_spectrum(p, alpha, target=target)
    omega = s.omega[s.nearest(target)]
    roots, _, errors = secondary_spatial_near(p, omega, alpha+.001j)
    assert min(abs(roots-alpha)) < 1e-8
    assert max(errors) < 1e-9


def test_saddle_complex_stationarity(wave):
    result = secondary_saddle(FloquetProblem(wave, harmonics=1))
    assert result.converged and abs(result.group_velocity) < 1e-7
    assert result.residual < 1e-10


def test_secondary_spectrum_tracking_prefers_vector_continuity():
    from llsa.secondary.floquet import SecondarySpectrum
    vectors = np.array([[1, 0], [0, 1]], complex)
    spectrum = SecondarySpectrum(np.array([1.01, 1.2]), vectors, np.zeros(2))
    assert spectrum.nearest(1.0) == 0
    assert spectrum.tracked(1.0, np.array([0, 1], complex)) == 1


def test_reject_modified_primary(wave):
    bad = PrimaryWave(wave.operators, wave.alpha+.01, wave.omega,
                      wave.modes.copy(), wave.residual, [])
    with pytest.raises(ValueError, match='residual'):
        FloquetProblem(bad)


def test_secondary_wall_continuity_audit(wave):
    # The matrix residual alone misses the wall constraint replaced by the
    # pressure boundary row. This known defect must remain visible in diagnostics.
    original = FloquetProblem(wave, harmonics=1)
    r = secondary_saddle(original)
    defect = secondary_kinematic_defects(original, r.alpha_hat, r.vector)
    assert defect['interior_continuity'] < 1e-8
    assert defect['wall_continuity'] > .01
    assert not defect['wall_continuity_passed']
    constrained = FloquetProblem(wave, harmonics=1, wall_condition='continuity')
    r = secondary_saddle(constrained)
    defect = secondary_kinematic_defects(constrained, r.alpha_hat, r.vector)
    assert r.converged and defect['wall_continuity_passed']
    assert defect['wall_continuity'] < 1e-10


def test_floquet_against_physical_phase_equations(wave):
    """Direct physical-space evaluation of Pier (7.4), including Bloch factors.

    This uses dealiased phase products and physical velocity/pressure units,
    independently of the block coupling and internal frequency normalization.
    """
    op, H, K = wave.operators, wave.harmonics, 2
    R, m = op.radius, op.m
    ah, bh = .12-.03j, 20.
    p = FloquetProblem(wave, bh, K)
    rng = np.random.default_rng(141)
    q = rng.normal(size=(2*K+1, 4, m))+1j*rng.normal(size=(2*K+1, 4, m))
    q[:, :, [0, -1]] = 0
    vector = q.ravel().copy()
    q[:, :3] *= R
    q[:, 3] *= R**2
    primary = wave.modes.reshape(2*H+1, 4, m).copy()
    primary[:, :3] *= R
    phase = np.arange(25)*2*np.pi/25
    ns, hs = np.arange(-K, K+1), np.arange(-H, H+1)
    E = np.exp(1j*phase[:, None]*ns)
    F = np.exp(1j*phase[:, None]*hs)

    def synth(e, modes):
        return np.einsum('ph,hvz->pvz', e, modes)

    vel = synth(E, q[:, :3])
    dr = synth(E, 1j*(ah+ns*wave.alpha)[:, None, None]*q[:, :3])
    dt = synth(E, 1j*(bh+ns*op.beta)[:, None, None]*q[:, :3])
    dz = vel@op.grid.D.T
    base = synth(F, primary[:, :3])
    baser = synth(F, 1j*hs[:, None, None]*wave.alpha*primary[:, :3])
    baset = synth(F, 1j*hs[:, None, None]*op.beta*primary[:, :3])
    basez = base@op.grid.D.T
    U, V, W, dU, dV, dW = op.base
    rhs = ((R*U+base[:, 0])[:, None]*dr+
           (V+base[:, 1]/R)[:, None]*dt+(W+base[:, 2])[:, None]*dz)
    rhs += vel[:, 0, None]*baser+vel[:, 1, None]/R*baset+vel[:, 2, None]*basez
    rhs += vel[:, 2, None]*np.array([R*dU, R*dV, dW])[None]
    rhs[:, 0] += U*vel[:, 0]-2*V*vel[:, 1]-2*base[:, 1]*vel[:, 1]/R
    rhs[:, 1] += U*vel[:, 1]+2*V*vel[:, 0]+(base[:, 0]*vel[:, 1]+base[:, 1]*vel[:, 0])/R
    lap = vel@op.grid.D2.T+dr/R
    lap -= synth(E, ((ah+ns*wave.alpha)**2+(bh+ns*op.beta)**2/R**2)[:, None, None]*q[:, :3])
    lap[:, 0] += (-vel[:, 0]-2*dt[:, 1])/R**2
    lap[:, 1] += (-vel[:, 1]+2*dt[:, 0])/R**2
    rhs -= lap
    rhs[:, 0] += synth(E, 1j*(ah+ns*wave.alpha)[:, None, None]*q[:, 3:])[:, 0]
    rhs[:, 1] += synth(E, 1j*(bh+ns*op.beta)[:, None, None]*q[:, 3:])[:, 0]/R
    rhs[:, 2] += synth(E, q[:, 3:])[:, 0]@op.grid.D.T
    rhs -= synth(E, 1j*ns[:, None, None]*wave.omega*q[:, :3])
    divergence = dr[:, 0]+vel[:, 0]/R+dt[:, 1]/R+dz[:, 2]
    physical = np.concatenate((rhs/R**2, divergence[:, None]/R), axis=1)
    coefficients = np.einsum('ph,pvz->hvz', E.conj(), physical)/len(phase)
    matrix_result = (p.matrix(ah)@vector).reshape(2*K+1, 4, m)
    np.testing.assert_allclose(coefficients[:, :, 1:-1], matrix_result[:, :, 1:-1],
                               rtol=1e-10, atol=1e-9)
