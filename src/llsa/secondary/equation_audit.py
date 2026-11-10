"""Audit from full cylindrical Navier--Stokes, before base subtraction.

The nonlinear evaluator does not call PierOperators, its convolution, or its
coupling formula. Its centered directional derivative is algebraically exact
for this quadratic PDE, including complex directions. Boundary rows are audited
separately: replacing a row must not hide a physical constraint defect.
"""
import numpy as np
from scipy.interpolate import BarycentricInterpolator
from scipy.integrate import solve_bvp


ADVECTION = ('radial_transport', 'azimuthal_transport', 'normal_transport', 'metric_advection')


def navier_stokes_terms(state, radius):
    """Full total-flow NS left-hand side, in physical Pier units at r=R.

    All vector arrays end in (component,z). dr is evaluated BEFORE freezing
    coefficients, so the similarity base has dr(rU,rV,W)=(U,V,0).
    dp contains partial derivatives (p_r,p_theta,p_z), without metric factors.
    """
    v = state['v']
    u, azimuthal = v[..., 0, :], v[..., 1, :]
    zero = np.zeros_like(u)
    terms = {
        'time': state['dt'],
        'radial_transport': v[..., 0:1, :]*state['dr'],
        'azimuthal_transport': v[..., 1:2, :]*state['dtheta']/radius,
        'normal_transport': v[..., 2:3, :]*state['dz'],
        'metric_advection': np.stack((-azimuthal**2, u*azimuthal, zero), axis=-2)/radius,
        'pressure': state['dp']*np.array([1., 1/radius, 1.])[:, None],
        'scalar_diffusion': -(state['drr']+state['dtheta2']/radius**2+state['dzz']+state['dr']/radius),
        'vector_diffusion': -np.stack((-u-2*state['dtheta'][..., 1, :],
                                      -azimuthal+2*state['dtheta'][..., 0, :], zero), axis=-2)/radius**2,
    }
    divergence = (state['dr'][..., 0, :]+u/radius+
                  state['dtheta'][..., 1, :]/radius+state['dz'][..., 2, :])
    return terms, divergence


def combine(left, right, factor=1.):
    return {key: left[key]+factor*right[key] for key in left}


def linearized_terms(background, variation, radius, epsilon=1.):
    plus, dp = navier_stokes_terms(combine(background, variation, epsilon), radius)
    minus, dm = navier_stokes_terms(combine(background, variation, -epsilon), radius)
    return {key: (plus[key]-minus[key])/(2*epsilon) for key in plus}, (dp-dm)/(2*epsilon)


def laminar_state(op):
    U, V, W, dU, dV, dW = op.base
    R = op.radius
    zeros = np.zeros((3, op.m))
    ddU, ddV, ddW = U**2-V**2+op.external_rotation**2+W*dU, 2*U*V+W*dV, -2*dU
    return dict(v=np.array([R*U, R*V, W]), dr=np.array([U, V, np.zeros_like(U)]),
                dtheta=zeros.copy(), dz=np.array([R*dU, R*dV, dW]),
                drr=zeros.copy(), dtheta2=zeros.copy(), dzz=np.array([R*ddU, R*ddV, ddW]),
                dt=zeros.copy(), dp=np.array([np.full_like(U, R*op.external_rotation**2), np.zeros_like(U), ddW-W*dW]))


def fourier_state(modes, ns, radial, azimuthal, frequency, phases, D, D2):
    """Synthesize physical fields AND their coordinate derivatives, including time."""
    E = np.exp(1j*np.asarray(phases)[:, None]*np.asarray(ns))
    modes = np.asarray(modes)
    velocity, pressure = modes[:, :3], modes[:, 3]
    def synth(q):
        return np.einsum('ph,hvz->pvz', E, q)
    radial, azimuthal, frequency = map(np.asarray, (radial, azimuthal, frequency))
    return dict(v=synth(velocity), dr=synth(1j*radial[:, None, None]*velocity),
                dtheta=synth(1j*azimuthal[:, None, None]*velocity), dz=synth(velocity@D.T),
                drr=synth(-radial[:, None, None]**2*velocity),
                dtheta2=synth(-azimuthal[:, None, None]**2*velocity), dzz=synth(velocity@D2.T),
                dt=synth(-1j*frequency[:, None, None]*velocity),
                dp=synth(np.stack((1j*radial[:, None]*pressure,
                                   1j*azimuthal[:, None]*pressure, pressure@D.T), axis=1)))


def physical_modes(wave):
    modes = wave.modes.reshape(2*wave.harmonics+1, 4, wave.operators.m).copy()
    modes[:, :3] *= wave.operators.radius
    modes[:, 3] *= wave.operators.radius**2
    return modes


def floquet_full_ns_check(wave, harmonics=2, alpha_hat=.012-.031j, beta_hat=20., seed=413):
    """Differentiate FULL nonlinear NS about base+primary, then compare the pencil."""
    from .floquet import FloquetProblem
    op, R, H = wave.operators, wave.operators.radius, wave.harmonics
    K = harmonics
    phase = np.arange(2*(H+K)+3)*2*np.pi/(2*(H+K)+3)
    hs, ns = np.arange(-H, H+1), np.arange(-K, K+1)
    primary = fourier_state(physical_modes(wave), hs, hs*wave.alpha, hs*op.beta,
                            hs*wave.omega, phase, op.grid.D, op.grid.D2)
    background = combine(laminar_state(op), primary)
    rng = np.random.default_rng(seed)
    internal = rng.normal(size=(2*K+1, 4, op.m))+1j*rng.normal(size=(2*K+1, 4, op.m))
    physical = internal.copy()
    physical[:, :3] *= R
    physical[:, 3] *= R**2
    omega_hat = 7.3+.71j
    variation = fourier_state(physical, ns, alpha_hat+ns*wave.alpha,
                              beta_hat+ns*op.beta, omega_hat+ns*wave.omega,
                              phase, op.grid.D, op.grid.D2)
    terms, divergence = linearized_terms(background, variation, R)
    left = np.concatenate((sum(terms.values())/R**2, divergence[:, None]/R), axis=1)
    E = np.exp(1j*phase[:, None]*ns)
    direct = np.einsum('ph,pvz->hvz', E.conj(), left)/len(phase)
    pencil = FloquetProblem(wave, beta_hat, K, wall_condition='continuity')
    matrix = ((pencil.matrix(alpha_hat)-omega_hat/R*pencil.B)@internal.ravel()).reshape(internal.shape)
    error = np.max(abs((direct-matrix)[:, :, 1:-1]))
    scale = np.max(abs(direct[:, :, 1:-1]))
    return dict(max_absolute_error=float(error), relative_error=float(error/scale),
                passed=bool(error < 1e-8+1e-11*scale),
                note='Interior only; full NS differentiation includes pressure and complex inertial time')


def isolated_coefficient_checks(n=24):
    """Excite individual velocity/pressure components to expose small metric terms."""
    from .operators import PierOperators
    records = []
    for R in (17., 510.):
        op = PierOperators(radius=R, n=n, mean_mode='parallel')
        x = op.grid.xi
        f = (1-x*x)*(1+.3*x+.2*x*x)
        g = (1-x*x)*(1-.4*x+.1*x*x)
        for k, ell in ((.14, 23.), (-.013-.035j, -44.)):
            for component in range(4):
                q = np.zeros((1, 4, op.m), complex)
                q[0, component] = f*(.7+.4j)
                state = fourier_state(q, [0], [k], [ell], [0], [0], op.grid.D, op.grid.D2)
                terms, div = linearized_terms(laminar_state(op), state, R)
                expected = np.concatenate((sum(terms.values())[0]/R**2, div[0, None]/R))
                internal = q[0].copy()
                internal[:3] /= R
                internal[3] /= R**2
                actual = (op.linear(k, ell, mean_closure=False)@internal.ravel()).reshape(4, op.m)
                error = float(np.max(abs((expected-actual)[:, 1:-1])))
                records.append(dict(kind='laminar_linear', R=R, k=[complex(k).real, complex(k).imag],
                                    ell=ell, component=component, error=error))
            for h in (-2, 0, 3):
                for base_component in range(3):
                    for component in range(3):
                        p, q = np.zeros((1, 4, op.m), complex), np.zeros((1, 4, op.m), complex)
                        p[0, base_component], q[0, component] = g*(.5-.2j), f*(.7+.4j)
                        ps = fourier_state(p, [h], [h*.346], [h*op.beta], [0], [0], op.grid.D, op.grid.D2)
                        qs = fourier_state(q, [0], [k], [ell], [0], [0], op.grid.D, op.grid.D2)
                        terms, _ = linearized_terms(ps, qs, R)
                        expected = sum(terms[key] for key in ADVECTION)[0]/R**2
                        actual = (op.coupling((p[0]/R).ravel(), h, .346, k, ell)@(q[0]/R).ravel()).reshape(4, op.m)[:3]
                        error = float(np.max(abs((expected-actual)[:, 1:-1])))
                        records.append(dict(kind='primary_coupling', R=R, base_harmonic=h,
                                            base_component=base_component, component=component,
                                            k=[complex(k).real, complex(k).imag], ell=ell, error=error))
    maximum = max(r['error'] for r in records)
    return dict(cases=len(records), max_absolute_error=maximum, passed=bool(maximum < 1e-10), records=records)


def independent_laminar_check(ro=-1.):
    """Solve Pier (2.2) directly; do not reuse the BEK equation implementation."""
    from .discretization import rotation_rates, sample_inertial_base
    rotation_rates(ro)
    disk, external = rotation_rates(ro)
    def rhs(z, q):
        U, dU, V, dV, W = q
        return np.array([dU, U**2-V**2+external**2+W*dU, dV, 2*U*V+W*dV, -2*U])
    def bc(left, right):
        return np.array([left[0], left[2]-disk, left[4], right[0], right[2]-external])
    z = np.linspace(0, 40, 401)
    e, s, c = np.exp(-z), np.sin(z), np.cos(z)
    guess = np.array([-ro*e*s, -ro*e*(c-s), external-ro*e*c,
                      ro*e*(c+s), ro*(1-e*(s+c))])
    sol = solve_bvp(rhs, bc, z, guess, tol=1e-10, max_nodes=30000)
    if not sol.success:
        raise RuntimeError(sol.message)
    z = np.linspace(0, 40, 2001)
    independent = sol.sol(z)
    U, V, W, dU, dV, _ = sample_inertial_base(z, ro)
    reference = np.array([U, dU, V, dV, W])
    errors = np.max(abs(independent-reference), axis=1)
    return dict(ro=ro, component_order=['U', 'dU', 'V', 'dV', 'W'], max_errors=errors.tolist(),
                W_infinity=float(independent[-1, -1]), passed=bool(max(errors) < 1e-7))


def primary_equation_defects(wave):
    """Evaluate the frozen paper PDE, including constraints replaced by closures."""
    op, H, R = wave.operators, wave.harmonics, wave.operators.radius
    ns = np.arange(-H, H+1)
    q = physical_modes(wave)
    phase = np.arange(4*H+1)*2*np.pi/(4*H+1)
    perturbation = fourier_state(q, ns, ns*wave.alpha, ns*op.beta,
                                ns*wave.omega, phase, op.grid.D, op.grid.D2)
    base = laminar_state(op)
    total, divtotal = navier_stokes_terms(combine(base, perturbation), R)
    basic, divbasic = navier_stokes_terms(base, R)
    physical_residual = sum(total.values())-sum(basic.values())
    E = np.exp(1j*phase[:, None]*ns)
    momentum = np.einsum('ph,pvz->hvz', E.conj(), physical_residual)/len(phase)
    divergence = np.einsum('ph,pz->hz', E.conj(), divtotal-divbasic)/len(phase)
    nonzero = ns != 0
    xi = (op.grid.xi[:-1]+op.grid.xi[1:])/2
    interpolator = BarycentricInterpolator(op.grid.xi, q, axis=2)
    off = interpolator(xi)
    t = op.grid.b*xi+(1-op.grid.b)*(xi**3+op.grid.c*(1-xi**2))
    tp = op.grid.b+(1-op.grid.b)*(3*xi**2-2*op.grid.c*xi)
    dz = interpolator.derivative(xi)*(1-t)**2/(2*op.grid.a*tp)
    divoff = ((1j*ns[:, None]*wave.alpha+1/R)*off[:, 0]+
              1j*ns[:, None]*op.beta/R*off[:, 1]+dz[:, 2])
    wall = np.max(abs(divergence[nonzero, 0]))
    peak = np.max(abs(q[:, :3]))
    mean = np.max(abs(divergence[H, 1:-1]))
    mean_off = np.max(abs(divoff[H]))
    mean_wall = abs(divergence[H, 0])
    return dict(units='physical Pier velocity and length; absolute PDE residuals',
                mean_closure=op.mean_mode, source_n=op.m-1,
                interior_momentum_by_component=np.max(abs(momentum[:, :, 1:-1]), axis=(0, 2)).tolist(),
                mean_continuity=float(mean), mean_u_over_R=float(np.max(abs(q[H, 0]))/R),
                mean_off_node_continuity=float(mean_off), mean_wall_continuity=float(mean_wall),
                nonzero_interior_continuity=float(np.max(abs(divergence[nonzero, 1:-1]))),
                nonzero_off_node_continuity=float(np.max(abs(divoff[nonzero]))),
                nonzero_wall_continuity=float(wall), wall_continuity_over_peak_velocity=float(wall/peak),
                wall_pressure_momentum=float(np.max(abs(q[:, 3]@op.grid.D[0]-q[:, 2]@op.grid.D2[0]))),
                far_field_velocity_by_component=np.max(abs(q[:, :3, -1]), axis=0).tolist(),
                mean_continuity_at_nodes_passed=bool(mean < 1e-7),
                frozen_paper_mean_continuity_passed=bool(max(mean, mean_off, mean_wall) < 1e-7),
                note='Small Newton residual and matching harmonic energies do not override a failed paper-PDE constraint')
