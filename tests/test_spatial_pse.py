"""Physics checks for the experimental nonlinear spatial march."""
import numpy as np

from llsa.secondary.spatial import SpatialPSE, spatial_mode


def test_radius_scaling_against_physical_cylindrical_equations():
    """Independent physical-velocity substitution checks curvature/scale factors."""
    solver = SpatialPSE(n=24)
    R = 380/.751
    op = solver.operators(R)
    k, ell, omega = .3-.07j, 42, 42*op.disk_rotation
    rng = np.random.default_rng(19)
    q = rng.normal(size=(4, op.m))+1j*rng.normal(size=(4, op.m))
    velocity = R*q[:3]
    ur, vr, wr = (1/R+1j*k)*velocity
    urr, vrr, wrr = (2j*k/R-k*k)*velocity
    u, v, w = velocity
    uz, vz, wz = velocity@op.grid.D.T
    uzz, vzz, wzz = velocity@op.grid.D2.T
    pressure = R*R*q[3]
    pr = (2/R+1j*k)*pressure
    pz = op.grid.D@pressure
    F, G, W, Fz, Gz, Wz = op.base
    lap_u = urr+ur/R+uzz-(ell**2+1)*u/R**2-2j*ell*v/R**2
    lap_v = vrr+vr/R+vzz-(ell**2+1)*v/R**2+2j*ell*u/R**2
    lap_w = wrr+wr/R+wzz-ell**2*w/R**2
    expected = np.array([
        (R*F*ur+F*u+W*uz+R*Fz*w-2*G*v+1j*(ell*G-omega)*u+pr-lap_u)/R**2,
        (R*F*vr+F*v+2*G*u+W*vz+R*Gz*w+1j*(ell*G-omega)*v+1j*ell*pressure/R-lap_v)/R**2,
        (R*F*wr+W*wz+Wz*w+1j*(ell*G-omega)*w+pz-lap_w)/R**2,
        (ur+u/R+1j*ell*v/R+wz)/R])
    A, _ = solver.matrices(op, k, 1)
    actual = (A@q.ravel()).reshape(4, op.m)
    np.testing.assert_allclose(actual[:,1:-1], expected[:,1:-1], rtol=1e-11, atol=1e-11)


def test_frozen_linear_march_recovers_spatial_eigenmode():
    solver = SpatialPSE(n=24, harmonics=1, nonlinear=False, nonparallel=False)
    radius = 320/.751
    op = solver.operators(radius)
    solver.operators = lambda r: op
    first = solver.initialize(radius, 1e-7, 1.1858)
    alpha, q, residual = spatial_mode(op, 42*op.disk_rotation, first.alpha)
    assert residual < 1e-10
    second = solver.advance(first, radius+6, tol=1e-10)
    third = solver.advance(second, radius+12, first, tol=1e-10)
    expected = first.modes[2]*np.exp(-12*alpha.imag)
    np.testing.assert_allclose(third.modes[2], expected, rtol=1e-7, atol=1e-15)
    assert abs(third.alpha-alpha) < 1e-8


def test_nonlinear_mean_harmonic_generation_and_jacobian():
    solver = SpatialPSE(n=20, harmonics=2)
    radius = 320/.751
    results = []
    for amplitude in (1e-5, 2e-5):
        initial = solver.initialize(radius, amplitude, 1.1858)
        result = solver.advance(initial, radius+6, tol=1e-11, check_jacobian=True)
        np.testing.assert_allclose(result.modes[:2], result.modes[3:][::-1].conj())
        assert np.max(abs(result.modes[2].imag)) == 0
        results.append(result.modes)
    # Quadratic Navier--Stokes interactions generate mean and second harmonic.
    for j in (2, 4):
        assert np.linalg.norm(results[0][j]) > 0
        ratio = np.linalg.norm(results[1][j])/np.linalg.norm(results[0][j])
        assert abs(ratio-4) < 2e-3
