"""Independent equation, Jacobian, descriptor and nonlinear BVP checks."""
import numpy as np
import pytest
from scipy import linalg

from llsa.secondary.radial import (RadialConfig, radial_matrices, convection,
                                  convection_blocks, pack_field)
from llsa.secondary.global_radial import GlobalRadial, OutletSponge, radial_differences
from llsa.secondary.oneway import SpectralProjector, finite_spectrum, causal_projector, SpectralOneWay


def test_full_radial_operator_against_physical_ns():
    c = RadialConfig(n=12)
    op = c.operators(420.)
    R, m, ell, om = op.radius, op.m, 42., c.frequency(op)
    rng = np.random.default_rng(44)
    q, qr, qrr = rng.normal(size=(3, 4, m))+1j*rng.normal(size=(3, 4, m))
    u, v, w = R*q[:3]
    ur, vr, wr = q[:3]+R*qr[:3]
    urr, vrr, wrr = 2*qr[:3]+R*qrr[:3]
    uz, vz, wz = R*q[:3]@op.grid.D.T
    uzz, vzz, wzz = R*q[:3]@op.grid.D2.T
    p, pr, pz = R**2*q[3], 2*R*q[3]+R**2*qr[3], R**2*op.grid.D@q[3]
    F, G, W, Fz, Gz, Wz = op.base
    expected = np.array([
        (R*F*ur+F*u+W*uz+R*Fz*w-2*G*v+1j*(ell*G-om)*u+pr
         -(urr+ur/R+uzz-(ell**2+1)*u/R**2-2j*ell*v/R**2))/R**2,
        (R*F*vr+F*v+2*G*u+W*vz+R*Gz*w+1j*(ell*G-om)*v+1j*ell*p/R
         -(vrr+vr/R+vzz-(ell**2+1)*v/R**2+2j*ell*u/R**2))/R**2,
        (R*F*wr+W*wz+Wz*w+1j*(ell*G-om)*w+pz
         -(wrr+wr/R+wzz-ell**2*w/R**2))/R**2,
        (ur+u/R+1j*ell*v/R+wz)/R])
    A, B, C = radial_matrices(op, 1, om)
    actual = (A@q.ravel()+B@qr.ravel()+C@qrr.ravel()).reshape(4, m)
    np.testing.assert_allclose(actual[:, 1:-1], expected[:, 1:-1], rtol=1e-11, atol=1e-11)


def test_nonlinear_convolution_against_physical_phase_product():
    c = RadialConfig(n=12, harmonics=2)
    op = c.operators(420.)
    rng = np.random.default_rng(45)
    H, s = c.harmonics, op.size
    from llsa.secondary.primary import unpack
    q, qr = [unpack(rng.normal(size=(2*H+1)*s), H, s) for _ in range(2)]
    theta = 2*np.pi*np.arange(32)/32
    ns = np.arange(-H, H+1)
    phases = np.exp(1j*theta[:, None]*ns)
    vel = (phases@q).reshape(32, 4, op.m)[:, :3]*op.radius
    vel_r = (phases@(q/op.radius+qr)).reshape(32, 4, op.m)[:, :3]*op.radius
    vel_theta = (phases@(1j*ns[:, None]*op.beta*q)).reshape(32, 4, op.m)[:, :3]*op.radius
    vel_z = vel@op.grid.D.T
    physical = vel[:, 0, None]*vel_r+vel[:, 1, None]*vel_theta/op.radius+vel[:, 2, None]*vel_z
    physical[:, 0] -= vel[:, 1]**2/op.radius
    physical[:, 1] += vel[:, 0]*vel[:, 1]/op.radius
    expected = np.einsum('th,tvz->hvz', phases.conj(), physical)/32/op.radius**2
    actual = convection(op, q, qr).reshape(2*H+1, 4, op.m)[:, :3]
    np.testing.assert_allclose(actual[:, :, 1:-1], expected[:, :, 1:-1], atol=2e-12)


@pytest.mark.parametrize('linear_solver,coupled,refresh', [('direct', 1, 1), ('harmonic-gmres', 1, 1),
                                                        ('harmonic-gmres', 2, 1), ('harmonic-gmres', 1, 3),
                                                        ('pardiso', 1, 1), ('pardiso-gmres',1,3)])
@pytest.mark.parametrize('merit', ['inf', 'scaled-l2'])
@pytest.mark.parametrize('sweep', ['diagonal', 'symmetric'])
@pytest.mark.parametrize('phase_update', [False, True])
@pytest.mark.parametrize('phase_pin_radius', [None,423.])
def test_global_jacobian_and_manufactured_nonlinear_solution(linear_solver, coupled, refresh, merit, sweep, phase_update, phase_pin_radius, phase_continuity=False, cartesian_fallback=False, preconditioning_side='left'):
    if linear_solver.startswith('pardiso'):
        pytest.importorskip('pypardiso')
    c = RadialConfig(n=8, harmonics=2)
    r = np.linspace(420., 425., 7)
    g = GlobalRadial(r, c)
    rng = np.random.default_rng(46)
    from llsa.secondary.primary import unpack
    coefficients = [unpack(1e-4*rng.normal(size=g.width), c.harmonics, g.s) for _ in range(3)]
    x = r-r[0]
    q = np.array([coefficients[0]+t*coefficients[1]+t*t*coefficients[2] for t in x])
    qr = np.array([coefficients[1]+2*t*coefficients[2] for t in x])
    qrr = np.array([2*coefficients[2]]*len(r))
    forcing = np.zeros_like(q)
    for i, op in enumerate(g.ops):
        forcing[i] = convection(op, q[i], qr[i])
        for j in range(-c.harmonics, c.harmonics+1):
            A, B, C = radial_matrices(op, j, c.frequency(op))
            forcing[i, j+c.harmonics] += A@q[i, j+c.harmonics]+B@qr[i, j+c.harmonics]+C@qrr[i, j+c.harmonics]
    vector = pack_field(.9*q)
    F, J = g.evaluate(vector, q[0], outlet=q[-1], forcing=forcing, jacobian=True)
    direction = rng.normal(size=len(vector))
    h = 1e-6
    fd = (g.evaluate(vector+h*direction, q[0], outlet=q[-1], forcing=forcing)
          -g.evaluate(vector-h*direction, q[0], outlet=q[-1], forcing=forcing))/(2*h)
    np.testing.assert_allclose(J@direction, fd, rtol=2e-8, atol=1e-9)
    checkpoints, diagnostics = [], []
    initial = .9*q
    if phase_pin_radius is not None:
        initial = initial*np.exp(.7j*np.arange(-c.harmonics,c.harmonics+1))[None,:,None]
    result = g.solve(q[0], initial=initial, outlet=q[-1], forcing=forcing, tol=1e-10,
        linear_solver=linear_solver, coupled_harmonics=coupled,
        preconditioner_refresh=refresh,
        merit=merit,
        preconditioner_sweep=sweep,
        phase_update=phase_update,
        phase_continuity=phase_continuity,
        cartesian_fallback=cartesian_fallback,
        preconditioning_side=preconditioning_side,
        phase_pin_radius=phase_pin_radius,
        checkpoint=lambda it, field, norm: checkpoints.append((it, field, norm)),
        diagnostics=diagnostics.append)
    np.testing.assert_allclose(result.modes, q, atol=1e-9, rtol=1e-7)
    np.testing.assert_array_equal(checkpoints[-1][1], result.modes)
    assert checkpoints[-1][2] == result.residual
    assert all(d['merit_after'] < d['merit_before'] for d in diagnostics)


@pytest.mark.parametrize('solver',['direct','harmonic-gmres','pardiso-gmres'])
@pytest.mark.parametrize('pin',[None,423.])
def test_continuity_preserving_phase_recovers_manufactured_root(solver,pin):
    test_global_jacobian_and_manufactured_nonlinear_solution(
        solver,1,3,'inf','diagonal',True,pin,phase_continuity=True)


@pytest.mark.parametrize('solver', ['direct', 'pardiso-gmres'])
@pytest.mark.parametrize('merit', ['inf', 'scaled-l2'])
def test_cartesian_fallback_recovers_manufactured_root(solver, merit):
    test_global_jacobian_and_manufactured_nonlinear_solution(
        solver, 1, 3, merit, 'diagonal', True, 423.,
        phase_continuity=True, cartesian_fallback=True)


@pytest.mark.parametrize('merit', ['inf', 'scaled-l2'])
def test_right_lu_preconditioner_recovers_manufactured_root(merit):
    test_global_jacobian_and_manufactured_nonlinear_solution(
        'pardiso-gmres', 1, 6, merit, 'diagonal', True, 423.,
        phase_continuity=True, cartesian_fallback=True, preconditioning_side='right')


def test_phase_projection_preserves_affine_continuity_and_newton_tangent():
    from llsa.secondary.global_radial import phase_direction
    from llsa.secondary.radial import unpack_field
    r = np.array([420.,420.7,421.9,423.,424.2,425.,426.5])
    g = GlobalRadial(r,RadialConfig(n=8,harmonics=2))
    rng = np.random.default_rng(915)
    x = .001*rng.normal(size=len(r)*g.width)
    d = .002*rng.normal(size=len(x))
    q = unpack_field(x,len(r),g.H,g.s)
    direction = unpack_field(d,len(r),g.H,g.s)
    phase,remainder = phase_direction(q,direction,g.ops[0].grid.mass)
    F,J = g.evaluate(x,q[0],outlet=q[-1],jacobian=True)
    ns = np.arange(-g.H,g.H+1)[None,:,None]
    errors = []
    for eta in [.3,1e-4,5e-5]:
        trial = (q+eta*remainder)*np.exp(1j*eta*phase[:,None,None]*ns)
        Ft = g.evaluate(pack_field(trial),q[0],outlet=q[-1])
        target = F+eta*(J@d)
        corrected = g.project_interior_continuity(trial,Ft,target)
        Fc = g.evaluate(pack_field(corrected),q[0],outlet=q[-1])
        defect = unpack_field(Fc-target,len(r),g.H,g.s).reshape(len(r),2*g.H+1,4,-1)
        np.testing.assert_allclose(defect[1:-1,:,3,1:-1],0,atol=1e-14)
        np.testing.assert_array_equal(corrected[[0,-1]],trial[[0,-1]])
        np.testing.assert_array_equal(corrected[:,g.H],trial[:,g.H])
        errors.append(np.linalg.norm(pack_field(corrected)-x-eta*d))
    assert 3.95 < errors[-2]/errors[-1] < 4.05


@pytest.mark.parametrize('fail_first',[False,True])
def test_external_factor_age_and_failure_recovery_across_solves(fail_first):
    from scipy.sparse.linalg import splu
    from llsa.secondary.radial import unpack_field
    class Backend:
        def __init__(self):
            self.matrix = None
            self.newton_steps = self.factor_count = self.close_count = 0
            self.fail = fail_first
        def factorize(self,matrix,progress=None):
            self.matrix = matrix.tocsc()
            self.newton_steps = 0
            self.factor_count += 1
            if self.fail:
                self.fail = False
                raise RuntimeError('injected numerical factorization failure')
            self.lu = splu(self.matrix)
        def solve(self,rhs):
            return self.lu.solve(rhs)
        def close(self):
            self.matrix = None
            self.newton_steps = 0
            self.close_count += 1
    r = np.arange(420.,427.)
    g = GlobalRadial(r,RadialConfig(n=8,harmonics=1,nonlinear=False))
    q = unpack_field(1e-4*np.random.default_rng(92).normal(size=len(r)*g.width),len(r),g.H,g.s)
    forcing = unpack_field(g.evaluate(pack_field(q),q[0],outlet=q[-1]),len(r),g.H,g.s)
    backend = Backend()
    options = dict(initial=.9*q,outlet=q[-1],forcing=forcing,tol=1e-10,
        linear_solver='pardiso-gmres',pardiso_backend=backend,preconditioner_refresh=2)
    if fail_first:
        with pytest.raises(RuntimeError,match='injected numerical'):
            g.solve(q[0],**options)
        assert backend.matrix is None and backend.close_count==1
    ages = []
    for _ in range(3):
        diagnostics = []
        result = g.solve(q[0],diagnostics=diagnostics.append,**options)
        np.testing.assert_allclose(result.modes,q,atol=1e-10)
        assert len(diagnostics)==1
        ages.append(diagnostics[0]['factor_age'])
    assert ages == [0,1,0]
    assert backend.factor_count == 2+int(fail_first)


@pytest.mark.parametrize('factor_quality',[.999,0.])
def test_native_backsolve_refines_or_rejects_without_relaxing_accuracy(monkeypatch,factor_quality):
    pytest.importorskip('pypardiso')
    from scipy import sparse
    from scipy.sparse.linalg import spsolve
    from llsa.secondary.pardiso_backend import PardisoBackend
    A = sparse.csc_matrix([[3.,1.],[.2,2.]])
    rhs = np.array([1.,-.5])
    with PardisoBackend() as backend:
        backend.factorize(A)
        with monkeypatch.context() as patch:
            patch.setattr(backend.solver,'_call_pardiso',
                lambda matrix,b: factor_quality*spsolve(matrix,b))
            if factor_quality:
                answer = backend.solve(rhs)
                assert np.linalg.norm(A@answer-rhs)/np.linalg.norm(rhs) < 1e-8
                assert backend.last_stats['external_refinement_steps']==2
            else:
                with pytest.raises(RuntimeError,match='Ax=b residual check'):
                    backend.solve(rhs)
                assert backend.last_stats['external_refinement_steps']==3


@pytest.mark.parametrize('prescribed', [False,True])
def test_phase_gauge_jacobian_and_rotation_equivariance(prescribed):
    from llsa.secondary.phase_gauge import PhaseGauge
    from llsa.secondary.radial import unpack_field
    c = RadialConfig(n=8,harmonics=2)
    g = GlobalRadial(np.linspace(420.,425.,7),c)
    rng = np.random.default_rng(184)
    vector = rng.normal(size=7*g.width)*1e-4
    q = unpack_field(vector,7,g.H,g.s)
    forcing = .1*q
    outlet = q[-1] if prescribed else None
    gauge = PhaseGauge(g,q,423.,q[0],outlet,forcing)
    state = np.r_[vector,.37]
    f,j = gauge.evaluate(state,jacobian=True)
    direction = rng.normal(size=len(state))
    eps = 1e-6
    difference = (gauge.evaluate(state+eps*direction)-gauge.evaluate(state-eps*direction))/(2*eps)
    np.testing.assert_allclose(j@direction,difference,atol=2e-9,rtol=2e-8)
    physical = gauge.physical_field(state)
    original = g.evaluate(pack_field(physical),q[0],outlet=outlet,forcing=forcing)
    rotated = gauge.rotate(unpack_field(original,7,g.H,g.s),state[-1])
    np.testing.assert_allclose(pack_field(rotated),f[:-1],atol=1e-14,rtol=1e-12)


def test_projector_retains_algebraic_equation_and_unstable_downstream():
    # Finite eigenvalue +0.2 is downstream despite spatial amplification.
    E = np.diag([1., 1., 0.])
    L = np.diag([.2, 3., 1.])
    values, left, right = finite_spectrum(E, L)
    p = SpectralProjector.from_labels(E, L, values, left, right, values > 1)
    np.testing.assert_allclose(p.state, np.diag([1, 0, 1]), atol=1e-14)
    force = np.array([0., 4., 7.])
    # Algebraic constraint is 0=y_2+7, even with projection enabled.
    result = linalg.solve(E/.1-p.equation@L, p.equation@(E@np.array([1., 0, 0])/.1+force))
    np.testing.assert_allclose(result, [1/(1-.02), 0., -7.])


@pytest.mark.parametrize('advection_derivative', ['auxiliary', 'bdf'])
def test_causal_classification_refinement_and_weak_nonlinearity(advection_derivative):
    c = RadialConfig(n=12, harmonics=2)
    op = c.operators(320/.751)
    a = causal_projector(op, 1, c.frequency(op), continuation_steps=32)
    b = causal_projector(op, 1, c.frequency(op), continuation_steps=64)
    np.testing.assert_allclose(a.state, b.state, rtol=1e-7, atol=1e-7)
    solver = SpectralOneWay(c, continuation_steps=32, advection_derivative=advection_derivative)
    fields = []
    for amplitude in (1e-5, 2e-5):
        result, _ = solver.march(np.linspace(op.radius, op.radius+.5, 5), amplitude=amplitude)
        assert result.residual < 1e-9
        fields.append(result.modes[-1])
    for j in (c.harmonics, 2*c.harmonics):
        ratio = np.linalg.norm(fields[1][j])/np.linalg.norm(fields[0][j])
        assert abs(ratio-4) < 1e-3


def test_radial_differences_on_nonuniform_polynomial():
    r = np.array([1., 1.2, 1.7, 2.1, 2.8, 3.])
    D, D2 = radial_differences(r)
    np.testing.assert_allclose(D@r**4, 4*r**3, atol=1e-10)
    np.testing.assert_allclose(D2@r**4, 12*r**2, atol=1e-9)


def test_dns_form_outlet_sponge_profile_and_jacobian():
    sponge = OutletSponge(3., 2., 10.)
    values = sponge.profile(np.array([2., 3., 3.5, 4., 4.5, 5., 6.]))
    np.testing.assert_allclose(values, [0, 0, .64969169, 5, 9.35030831, 10, 10])
    c = RadialConfig(n=8, harmonics=2)
    g = GlobalRadial(np.linspace(2., 6., 7), c, outlet_sponge=sponge)
    rng = np.random.default_rng(49)
    q = rng.normal(size=(7, 2*c.harmonics+1, g.s))*1e-5
    q = q+q[:, ::-1].conj()
    x = pack_field(q)
    inlet = q[0]
    F, J = g.evaluate(x, inlet, jacobian=True)
    direction = rng.normal(size=len(x))
    h = 1e-6
    fd = (g.evaluate(x+h*direction, inlet)-g.evaluate(x-h*direction, inlet))/(2*h)
    np.testing.assert_allclose(J@direction, fd, rtol=2e-8, atol=2e-9)
    # The default sponge does not alter the zero harmonic directly.
    plain = GlobalRadial(np.linspace(2., 6., 7), c)
    delta = (g.evaluate(x, inlet)-plain.evaluate(x, inlet)).reshape(7, -1)
    mean = slice(0, g.s)
    np.testing.assert_allclose(delta[:, mean], 0, atol=1e-15)


@pytest.mark.parametrize('prescribed', [False, True])
@pytest.mark.parametrize('damp_mean', [False, True])
def test_cached_assembly_matches_original(prescribed, damp_mean):
    from llsa.secondary.radial import unpack_field
    c = RadialConfig(n=8, harmonics=3)
    r = np.array([420., 420.4, 421., 421.7, 422.5, 423.6, 425.])
    g = GlobalRadial(r, c, outlet_alpha=.3-.02j,
                     outlet_sponge=OutletSponge(421., 3., 10., damp_mean))
    rng = np.random.default_rng(71)
    x = rng.normal(size=len(r)*g.width)*1e-3
    q = unpack_field(x, len(r), g.H, g.s)
    f = .01*q
    kwargs = dict(outlet=q[-1] if prescribed else None, forcing=f, jacobian=True)
    new_f, new_j = g.evaluate(x, q[0], **kwargs)
    old_f, old_j = g.evaluate_reference(x, q[0], **kwargs)
    np.testing.assert_allclose(new_f, old_f, atol=1e-13, rtol=1e-12)
    diff = new_j-old_j
    assert np.max(abs(diff.data), initial=0) < 1e-12


def test_transport_seed_keeps_carrier_and_inlet():
    from examples.continue_radial_sponge import transport_seed
    c = RadialConfig(n=8, harmonics=2)
    r = np.linspace(420, 450, 91)
    op = c.operators(r[0])
    v = np.random.default_rng(17).normal(size=op.size)
    q = np.zeros((len(r), 5, op.size), complex)
    for j in (1, 2):
        q[:, 2+j] = v*np.exp(1j*j*.4*(r-r[0]))[:,None]
        q[:, 2-j] = q[:, 2+j].conj()
    new_r = np.linspace(420, 470, 151)
    new = transport_seed(r, q, new_r, c, 430., anchor_width=3.)
    for j in (1, 2):
        np.testing.assert_allclose(new[:, 2+j], v*np.exp(1j*j*.4*(new_r-r[0]))[:,None], atol=1e-12)
    same = transport_seed(r, q, r, c, 430.)
    np.testing.assert_allclose(same, q, atol=1e-12)


def test_smooth_transport_has_no_derivative_jump_at_anchor():
    from examples.continue_radial_sponge import transport_seed
    c = RadialConfig(n=8, harmonics=1)
    r = np.linspace(420., 460., 101)
    s = c.operators(r[0]).size
    q = np.zeros((len(r),3,s),complex)
    q[:,2] = ((1+.01*(r-r[0]))*np.exp(.4j*(r-r[0])))[:,None]
    q[:,0] = q[:,2].conj()
    h, anchor = .001, 430.
    new_r = np.r_[420.,anchor+h*np.arange(-3,4),480.]
    mapped = transport_seed(r,q,new_r,c,440.,mapping='smooth',anchor=anchor)
    amplitude = (mapped[:,2,0]*np.exp(-.4j*(new_r-r[0]))).real[1:-1]
    left = (3*amplitude[3]-4*amplitude[2]+amplitude[1])/(2*h)
    right = (-3*amplitude[3]+4*amplitude[4]-amplitude[5])/(2*h)
    assert abs(left-right) < 1e-8
    np.testing.assert_allclose(mapped[0],q[0],atol=1e-12)


@pytest.mark.parametrize('via_transport', [False, True])
def test_fixed_domain_seed_preserves_old_field_and_real_symmetry(via_transport):
    from examples.continue_radial_sponge import fixed_domain_seed, transport_seed
    c = RadialConfig(n=8, harmonics=2)
    r = np.linspace(420., 460., 21)
    s = c.operators(r[0]).size
    rng = np.random.default_rng(12)
    q = rng.normal(size=(len(r),5,s))+1j*rng.normal(size=(len(r),5,s))
    q = .5*(q+q[:,::-1].conj())
    new_r = np.r_[r, np.linspace(462., 500., 20)]
    result = (transport_seed(r, q, new_r, c, 440., mapping='physical') if via_transport
              else fixed_domain_seed(r, q, new_r, c))
    np.testing.assert_allclose(result[:len(r)], q, atol=1e-14)
    np.testing.assert_array_equal(result, result[:,::-1].conj())
    np.testing.assert_allclose(result[len(r):,2], np.broadcast_to(q[-1,2], (20,s)))
    assert np.linalg.norm(result[-1,3]) < .001*np.linalg.norm(q[-1,3])
    damped = fixed_domain_seed(r, q, new_r, c, damp_mean=True)
    np.testing.assert_allclose(damped[:len(r)], q, atol=1e-14)
    assert np.linalg.norm(damped[-1,2]) < .001*np.linalg.norm(q[-1,2])


def test_pardiso_numeric_reuse_and_changed_pattern():
    pytest.importorskip('pypardiso')
    from scipy import sparse
    from llsa.secondary.pardiso_backend import PardisoBackend
    rng = np.random.default_rng(45)
    a = sparse.diags([-np.ones(39), 5*np.ones(40), 2*np.ones(39)], [-1,0,1], format='csr')
    rhs = rng.normal(size=(40,2))
    with PardisoBackend() as solver:
        for scale in (1., 1.5):
            matrix = a*scale
            solver.factorize(matrix)
            answer = solver.solve(rhs)
            np.testing.assert_allclose(matrix@answer, rhs, atol=1e-12)
        assert solver.analyses == 1
        assert solver.last_stats['symbolic_reused']
        matrix = a+sparse.diags(np.ones(38), 2, shape=a.shape)
        solver.factorize(matrix)
        answer = solver.solve(rhs)
        np.testing.assert_allclose(matrix@answer, rhs, atol=1e-12)
        assert solver.analyses == 2


def test_fixed_buffer_transport_preserves_cylindrical_continuity():
    from llsa.secondary.radial_continuation import transport_fixed_buffer
    c = RadialConfig(n=8,harmonics=2)
    r = np.linspace(420.,520.,2001)
    m, k = c.n+1, .3
    field = np.zeros((len(r),5,4,m),complex)
    profile = np.linspace(.2,1.,m)
    for j in (1,2):
        u = np.exp(1j*j*k*(r-r[0]))[:,None]*profile
        field[:,2+j,0] = u
        field[:,2+j,1] = (2j/(j*c.beta)-k*r[:,None]/c.beta)*u
        field[:,2-j] = field[:,2+j].conj()
    original = field.reshape(len(r),5,-1)
    result = transport_fixed_buffer(r,original,c,450.,460.,30.).reshape(field.shape)
    h = r[1]-r[0]
    for j in (1,2):
        u, v = result[:,2+j,0],result[:,2+j,1]
        derivative = (u[:-4]-8*u[1:-3]+8*u[3:-1]-u[4:])/(12*h)
        divergence = derivative+2*u[2:-2]/r[2:-2,None]+1j*j*c.beta*v[2:-2]/r[2:-2,None]
        np.testing.assert_allclose(divergence,0,atol=2e-7)
    np.testing.assert_array_equal(result[0],field[0])
    np.testing.assert_array_equal(result,result[:,::-1].conj())
    np.testing.assert_array_equal(transport_fixed_buffer(r,original,c,450.,450.,30.),original)


def test_seed_refinement_preserves_polynomial_and_fourier_field(tmp_path):
    from dataclasses import asdict
    import json
    from examples.continue_radial_sponge import load_seed
    c = RadialConfig(n=8, harmonics=2)
    r = np.linspace(420, 450, 7)
    xi = c.operators(r[0]).grid.xi
    weights = np.arange(1, 21).reshape(5, 4)
    q = weights[None, :, :, None]*(1+xi**3)[None, None, None, :]
    q = np.repeat(q, len(r), axis=0).astype(complex)
    q = .5*(q+q[:, ::-1].conj())
    path = tmp_path/'seed.npz'
    np.savez(path, radii=r, modes=q.reshape(len(r), 5, -1), config_json=json.dumps(asdict(c)))
    refined, rr, modes = load_seed(path, n=12, harmonics=4)
    np.testing.assert_array_equal(rr, r)
    xnew = refined.operators(r[0]).grid.xi
    expected = .5*(weights+weights[::-1])[:, :, None]*(1+xnew**3)
    np.testing.assert_allclose(modes[:, 2:7].reshape(len(r), 5, 4, -1),
                               np.broadcast_to(expected, (len(r), *expected.shape)), atol=1e-12)
    assert not np.any(modes[:, [0, 1, 7, 8]])


def test_phase_secant_preserves_translating_fourier_wave():
    from examples.continue_radial_sponge import secant_seed
    c = RadialConfig(n=8, harmonics=3)
    s = c.operators(420.).size
    shape = np.random.default_rng(29).normal(size=s)
    phase = np.linspace(0, .9, 11)
    ns = np.arange(-3, 4)[None,:,None]
    old = shape[None,None,:]*np.exp(1j*ns*phase[:,None,None])
    current = 1.2*shape[None,None,:]*np.exp(2j*ns*phase[:,None,None])
    expected = 1.6*shape[None,None,:]*np.exp(4j*ns*phase[:,None,None])
    guess = secant_seed(current, old, 2., c, 420., phase_aligned=True)
    np.testing.assert_allclose(guess, expected, atol=2e-14)


def test_phase_retraction_has_newton_tangent_and_exact_rotations():
    from llsa.secondary.global_radial import phase_direction
    from llsa.secondary.radial import unpack_field
    c = RadialConfig(n=8, harmonics=3)
    op = c.operators(420.)
    rng = np.random.default_rng(137)
    q, d = [unpack_field(rng.normal(size=5*7*op.size), 5, 3, op.size) for _ in range(2)]
    phase, remainder = phase_direction(q, d, op.grid.mass)
    ns = np.arange(-3,4)[None,:,None]
    h = 1e-6
    plus = (q+h*remainder)*np.exp(1j*h*phase[:,None,None]*ns)
    minus = (q-h*remainder)*np.exp(-1j*h*phase[:,None,None]*ns)
    np.testing.assert_allclose((plus-minus)/(2*h), d, atol=6e-10)
    desired = np.linspace(-2, 3, 5)
    phase, remainder = phase_direction(q, 1j*ns*desired[:,None,None]*q, op.grid.mass)
    np.testing.assert_allclose(phase, desired, atol=1e-14)
    np.testing.assert_allclose(remainder, 0, atol=4e-14)
    tiny = q*1e-12
    phase, remainder = phase_direction(tiny, d, op.grid.mass)
    assert np.max(abs(phase)) <= np.pi
    np.testing.assert_allclose(remainder+1j*ns*phase[:,None,None]*tiny, d, atol=1e-14)


def test_modal_outlet_recovers_frozen_downstream_wave(monkeypatch):
    c = RadialConfig(n=12, harmonics=1, nonlinear=False)
    R = 320/.751
    op = c.operators(R)
    monkeypatch.setattr(RadialConfig, 'operators', lambda self, radius: op)
    initial, projectors, alpha = SpectralOneWay(c, continuation_steps=32).initialize(R, 1e-5)
    inlet = initial[:, :op.size]
    errors = []
    for count in (9, 17):
        r = np.linspace(R, R+2, count)
        expected = np.array([inlet*np.exp((1j*np.arange(-1, 2)*alpha.real
            -abs(np.arange(-1, 2))*alpha.imag)[:, None]*(radius-R)) for radius in r])
        solver = GlobalRadial(r, c, outlet_constraints={1:projectors[1].upstream_constraints})
        result = solver.solve(inlet, initial=expected, tol=1e-10)
        errors.append(np.linalg.norm(result.modes-expected)/np.linalg.norm(expected))
    assert errors[1] < 1e-6
    assert errors[1] < errors[0]/8
