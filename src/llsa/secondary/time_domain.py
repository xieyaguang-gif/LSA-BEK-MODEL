"""Independent linearized time advancement in physical Pier units.

The explicit RHS uses products in the primary-wave phase coordinate. It never
calls the Floquet matrix, its coupling routines, or an eigenvalue solver.
Pressure is eliminated using wall-normal velocity and a vorticity-like variable.
Only the harmonic-diagonal viscous operator is assembled for implicit stepping.
The supplied nonlinear primary wave and the laminar BVP remain shared inputs.
"""
from dataclasses import dataclass

import numpy as np
from numpy.polynomial.legendre import leggauss
from scipy import linalg
from scipy.interpolate import BarycentricInterpolator

from .discretization import sample_inertial_base


@dataclass
class TimeGrid:
    z: np.ndarray
    D: np.ndarray
    D2: np.ndarray
    D4: np.ndarray
    mass: np.ndarray
    xi: np.ndarray
    zmax: float
    scale: float


def time_grid(n=64, zmax=320., scale=5.):
    """Finite rational Chebyshev grid, distinct from the reference mapping."""
    if int(n) != n or n < 12 or not np.isfinite(zmax+scale) or not 0 < scale < zmax:
        raise ValueError("Require integer N>=12 and 0<scale<zmax")
    n = int(n)
    x = -np.cos(np.pi*np.arange(n+1)/n)
    bary = (-1.)**np.arange(n+1)
    bary[[0, -1]] *= .5
    delta = x[:, None]-x[None, :]
    np.fill_diagonal(delta, 1)
    Dx = (bary[None, :]/bary[:, None])/delta
    np.fill_diagonal(Dx, 0)
    np.fill_diagonal(Dx, -Dx.sum(axis=1))
    den = 1-x+scale/zmax*(1+x)
    z = scale*(1+x)/den
    xp = den**2/(2*scale)
    xpp = xp*den*(-1+scale/zmax)/scale
    D = xp[:, None]*Dx
    D2 = xp[:, None]**2*(Dx@Dx)+xpp[:, None]*Dx
    # Products of already-truncated mapped differentiation matrices are not
    # higher physical derivatives. Use the chain rule on the original polynomial.
    dp = den*(-1+scale/zmax)/scale
    ddp = (-1+scale/zmax)**2/scale
    Dx2, Dx3 = Dx@Dx, Dx@Dx@Dx
    D4 = (xp[:, None]**4*(Dx3@Dx)+6*(xp**3*dp)[:, None]*Dx3+
          (7*xp**2*dp**2+4*xp**3*ddp)[:, None]*Dx2+
          (xp*dp**3+4*xp**2*dp*ddp)[:, None]*Dx)
    xq, wq = leggauss(2*n+80)
    interpolation = BarycentricInterpolator(x, np.eye(n+1), axis=0)(xq)
    measure = wq*2*scale/(1-xq+scale/zmax*(1+xq))**2
    mass = interpolation.T@(measure[:, None]*interpolation)
    return TimeGrid(z, D, D2, D4, mass, x, float(zmax), float(scale))


def interpolate_primary(wave, z):
    """Interpolate a fixed saved semi-infinite wave; do not re-solve its branch."""
    grid = wave.operators.grid
    t = (z-grid.a)/(z+grid.a)
    x = t.copy()
    for _ in range(20):
        f = grid.b*x+(1-grid.b)*(x**3+grid.c*(1-x*x))-t
        df = grid.b+(1-grid.b)*(3*x*x-2*grid.c*x)
        x -= f/df
    if np.max(abs(grid.b*x+(1-grid.b)*(x**3+grid.c*(1-x*x))-t)) > 1e-12:
        raise RuntimeError("Cannot invert the source primary mapping")
    values = wave.modes.reshape(2*wave.harmonics+1, 4, wave.operators.m)
    return BarycentricInterpolator(grid.xi, values, axis=2)(x)


def viscous_block(D, D2, D4, R, g, a, b, q2, ell):
    """Analytic pressure-eliminated viscosity; avoid products of truncated derivatives."""
    I = np.eye(len(D))
    scalar = D2-q2*I
    fud, fue = (-g-2j*ell*b)/(R**2*q2), (b-2j*ell*a)/(R**2*q2)
    fvd, fve = (2j*ell*g-b)/(R**2*q2), (-2j*ell*b-a)/(R**2*q2)
    return np.block([
        [D4-2*q2*D2+q2**2*I-(a*fud+b*fvd)*D2, -(a*fue+b*fve)*D],
        [(b*fud-g*fvd)*D, scalar+(b*fue-g*fve)*I]])


class LinearTimeProblem:
    """Time-dependent Floquet envelope at a specified (alpha_hat,beta_hat).

    State shape is (2K+1,2,N+1), components w and eta=b*u-g*v,
    with g=i*k, a=g+1/R, b=i*ell/R. Velocities and time use BEK-reference
    physical nondimensional units, without the eigenvalue solver's R scaling.
    Complex alpha_hat describes a spatially weighted envelope, not a localized
    physical impulse. A single such calculation cannot establish a pinch point.
    """
    def __init__(self, wave, alpha_hat=0., beta_hat=20., harmonics=5,
                 n=64, zmax=320., scale=5., phase_points=None):
        if int(harmonics) != harmonics or harmonics < 1:
            raise ValueError("harmonics must be a positive integer")
        if not np.isfinite(alpha_hat) or not np.isfinite(beta_hat):
            raise ValueError("Finite Floquet wavenumbers required")
        if not np.isfinite(wave.residual) or wave.residual > 1e-7:
            raise ValueError("Use a converged saved primary wave")
        from .primary import WaveSolver
        checked = np.max(abs(WaveSolver(wave.operators, wave.harmonics).residual(
            wave.modes, wave.alpha, wave.omega)))
        if not np.isfinite(checked) or checked > 1e-7:
            raise ValueError(f"Primary does not satisfy its saved Ro/model equations: {checked}")
        self.wave, self.R, self.K = wave, wave.operators.radius, int(harmonics)
        self.alpha_hat, self.beta_hat = complex(alpha_hat), float(beta_hat)
        self.grid = time_grid(n, zmax, scale)
        self.m, self.ns = len(self.grid.z), np.arange(-self.K, self.K+1)
        self.k = self.alpha_hat+self.ns*wave.alpha
        self.ell = self.beta_hat+self.ns*wave.operators.beta
        self.g, self.a = 1j*self.k, 1j*self.k+1/self.R
        self.b = 1j*self.ell/self.R
        self.q2 = -(self.a*self.g+self.b**2)
        if np.min(abs(self.q2)) < 1e-10:
            raise ValueError("Degenerate horizontal wavevector needs a separate formulation")
        H = wave.harmonics
        count = 2*(H+self.K)+1 if phase_points is None else int(phase_points)
        if count <= H+2*self.K or count <= 2*H:
            raise ValueError("Too few phase points for unaliased retained products")
        self.phase = np.arange(count)*2*np.pi/count
        self.synthesis = np.exp(1j*self.phase[:, None]*self.ns)
        self.analysis = self.synthesis.conj().T/count
        primary = interpolate_primary(wave, self.grid.z)[:, :3]*self.R
        hs = np.arange(-H, H+1)
        E = np.exp(1j*self.phase[:, None]*hs)
        self.primary = np.einsum('ph,hvz->pvz', E, primary).real
        self.primary_r = np.einsum('ph,hvz->pvz', E, 1j*hs[:, None, None]*wave.alpha*primary).real
        self.primary_theta = np.einsum('ph,hvz->pvz', E, 1j*hs[:, None, None]*wave.operators.beta*primary).real
        self.primary_z = self.primary@self.grid.D.T
        self.ro = wave.operators.ro
        U, V, W, dU, dV, dW = sample_inertial_base(self.grid.z, self.ro)
        self.U, self.V = U, V
        self.adv_r = self.R*U+self.primary[:, 0]
        self.adv_theta = V+self.primary[:, 1]/self.R
        self.adv_z = W+self.primary[:, 2]
        self.total_z = self.primary_z+np.array([self.R*dU, self.R*dV, dW])[None]
        self.boundary_rows = np.array([0, 1, self.m-2, self.m-1, self.m, 2*self.m-1])
        self._prepare_viscosity()

    def velocity(self, state):
        w, eta = state[:, 0], state[:, 1]
        dw = w@self.grid.D.T
        u = (self.g[:, None]*dw-self.b[:, None]*eta)/self.q2[:, None]
        v = (self.b[:, None]*dw+self.a[:, None]*eta)/self.q2[:, None]
        return np.stack((u, v, w), axis=1)

    def project_force(self, force):
        horizontal = self.a[:, None]*force[:, 0]+self.b[:, None]*force[:, 1]
        normal = -self.q2[:, None]*force[:, 2]-horizontal@self.grid.D.T
        vorticity = self.b[:, None]*force[:, 0]-self.g[:, None]*force[:, 1]
        return np.stack((normal, vorticity), axis=1)

    def velocity_z(self, state):
        w, eta = state[:, 0], state[:, 1]
        ddw, deta = w@self.grid.D2.T, eta@self.grid.D.T
        du = (self.g[:, None]*ddw-self.b[:, None]*deta)/self.q2[:, None]
        dv = (self.b[:, None]*ddw+self.a[:, None]*deta)/self.q2[:, None]
        return np.stack((du, dv, w@self.grid.D.T), axis=1)

    def explicit_rhs(self, state):
        """Evaluate linearized physical advection/shear by dealiased products."""
        velocity = self.velocity(state)

        def synth(modes):
            return (self.synthesis@modes.reshape(len(self.ns), -1)).reshape(len(self.phase), 3, self.m)

        q = synth(velocity)
        dr = synth(1j*self.k[:, None, None]*velocity)
        dtheta = synth(1j*self.ell[:, None, None]*velocity)
        dz = synth(self.velocity_z(state))
        force = -(self.adv_r[:, None]*dr+self.adv_theta[:, None]*dtheta+self.adv_z[:, None]*dz)
        force -= q[:, 0, None]*self.primary_r+q[:, 1, None]/self.R*self.primary_theta+q[:, 2, None]*self.total_z
        force[:, 0] += -self.U*q[:, 0]+2*self.V*q[:, 1]+2*self.primary[:, 1]*q[:, 1]/self.R
        force[:, 1] += -self.U*q[:, 1]-2*self.V*q[:, 0]-(self.primary[:, 0]*q[:, 1]+self.primary[:, 1]*q[:, 0])/self.R
        modes = (self.analysis@force.reshape(len(self.phase), -1)).reshape(len(self.ns), 3, self.m)
        # Coordinates hold primary phase fixed during stepping. This term converts
        # the envelope frequency back to the inertial Floquet omega_hat.
        modes += 1j*self.ns[:, None, None]*self.wave.omega*velocity
        result = self.project_force(modes).reshape(len(self.ns), -1)
        result[:, self.boundary_rows] = 0
        return result

    def _prepare_viscosity(self):
        """Only diffusion and constraints are assembled, one uncoupled harmonic at a time."""
        m, D, D2, R = self.m, self.grid.D, self.grid.D2, self.R
        I, Z = np.eye(m), np.zeros((m, m))
        self.mass, self.diffusion, self.constraints = [], [], []
        for a, b, g, q2, ell in zip(self.a, self.b, self.g, self.q2, self.ell):
            scalar = D2-q2*I
            diffusion = viscous_block(D, D2, self.grid.D4, R, g, a, b, q2, ell)
            mass = np.block([[scalar, Z], [Z, I]]).astype(complex)
            mass[self.boundary_rows] = 0
            diffusion[self.boundary_rows] = 0
            constraints = np.zeros_like(mass)
            constraints[0, 0] = constraints[m-1, m-1] = 1
            constraints[1, :m] = D[0]
            constraints[m-2, :m] = D[-1]
            constraints[m, m] = constraints[-1, -1] = 1
            self.mass.append(mass)
            self.diffusion.append(diffusion)
            self.constraints.append(constraints)
        self.mass, self.diffusion, self.constraints = map(np.asarray, (self.mass, self.diffusion, self.constraints))

    def enforce_boundaries(self, state):
        state = np.array(state, dtype=complex, copy=True)
        state[:, :, [0, -1]] = 0
        D, m = self.grid.D, self.m
        indices = [1, m-2]
        state[:, 0, indices] = 0
        derivative = state[:, 0]@D[[0, -1]].T
        state[:, 0, indices] = linalg.solve(D[np.ix_([0, m-1], indices)], -derivative.T).T
        return state

    def random_state(self, seed=7, amplitude=1e-12):
        if not np.isfinite(amplitude) or amplitude <= 0:
            raise ValueError("Positive finite amplitude required")
        rng = np.random.default_rng(seed)
        z, end = self.grid.z, self.grid.zmax
        state = np.zeros((len(self.ns), 2, self.m), complex)
        for decay in (.5, 1., 2.):
            random = rng.normal(size=(len(self.ns), 2))+1j*rng.normal(size=(len(self.ns), 2))
            state[:, 0] += random[:, 0, None]*z**2*np.exp(-decay*z)*(1-z/end)**2
            state[:, 1] += random[:, 1, None]*z*np.exp(-decay*z)*(1-z/end)
        state = self.enforce_boundaries(state)
        state *= amplitude/np.sqrt(self.energy(state))
        return state

    def energy(self, state):
        q = self.velocity(state)
        return float(.5*np.einsum('hvi,ij,hvj->', q.conj(), self.grid.mass, q).real)

    def constraints_defect(self, state):
        q = self.velocity(state)
        norm = np.max(abs(q))
        divergence = self.a[:, None]*q[:, 0]+self.b[:, None]*q[:, 1]+q[:, 2]@self.grid.D.T
        return dict(continuity=float(np.max(abs(divergence))/norm),
                    boundary_velocity=float(np.max(abs(q[:, :, [0, -1]]))/norm))


def fit_growth(trace, start=None):
    """sigma=0.5*d(log E)/dt, omega_r=-d(arg probe)/dt, in Pier units."""
    t, energy, probe = trace['time'], trace['energy'], trace['probe']
    start = float(t[-1]*.5) if start is None else float(start)
    use = t >= start
    if use.sum() < 5 or np.any(energy <= 0) or not np.all(np.isfinite(energy)):
        raise ValueError("Insufficient finite positive time history for fitting")
    phase = np.unwrap(np.angle(probe))

    def fit(y):
        slope, intercept = np.polyfit(t[use], y[use], 1)
        residual = y[use]-slope*t[use]-intercept
        return float(slope), float(np.sqrt(np.mean(residual**2)))

    growth, amp_error = fit(.5*np.log(energy))
    phase_slope, phase_error = fit(phase)
    return dict(start=start, end=float(t[-1]), growth=growth, omega_real=-phase_slope,
                log_amplitude_fit_rms=amp_error, phase_fit_rms=phase_error)


def assess_time_convergence(runs, tolerance=1e-3):
    """Assess time-step, fitting-window and shape consistency independently of a paper target."""
    stable_windows = all(
        max(f['growth'] for f in r['fits'])-min(f['growth'] for f in r['fits']) < tolerance and
        max(f['omega_real'] for f in r['fits'])-min(f['omega_real'] for f in r['fits']) < tolerance and
        r['fits'][-1]['log_amplitude_fit_rms'] < tolerance and
        r['fits'][-1]['phase_fit_rms'] < tolerance and
        r['final_shape_overlap'] > .9999 for r in runs)
    result = dict(fitting_windows_and_shape_passed=bool(stable_windows),
                  tolerance=tolerance, passed=None,
                  note='Time convergence does not establish spatial convergence or published-table agreement')
    if len(runs) < 3:
        result['reason'] = 'Need at least three decreasing time steps'
        return result
    ordered = sorted(runs, key=lambda r: -r['dt'])[-3:]
    dts = np.array([r['dt'] for r in ordered])
    if not np.allclose(dts[:-1]/dts[1:], 2, rtol=1e-8):
        result['reason'] = 'Second-order extrapolation requires three successive halvings'
        return result
    sufficient = stable_windows
    for key in ('growth', 'omega_real'):
        values = np.array([r['fits'][-1][key] for r in ordered])
        delta = values[:-1]-values[1:]
        order = float(np.log2(abs(delta[0]/delta[1]))) if min(abs(delta)) > 1e-10 else None
        result[key] = dict(finest=float(values[-1]), last_step_difference=float(abs(delta[-1])),
                           observed_order=order, extrapolated=float(values[-1]-delta[-1]/3))
        sufficient &= abs(delta[-1]) < tolerance and (order is None or 1.5 < order < 2.5)
    result['passed'] = bool(sufficient)
    return result


def advance(problem, initial, dt=.001, end_time=6., sample_every=10, callback=None):
    """CN/AB2 with an IMEX Euler starter; velocities remain divergence-free.

    dt and end_time are in 1/Omega_BEK time units. No eigenvectors, eigenvalues
    or spectral matrices are used by this routine. Check at least three time
    steps and several late fitting windows before interpreting a measured rate.
    """
    if not np.isfinite(dt+end_time) or dt <= 0 or end_time <= 0 or sample_every < 1:
        raise ValueError("Positive dt, end_time and sample_every required")
    steps = int(np.ceil(end_time/dt))
    dt = end_time/steps
    state = problem.enforce_boundaries(initial)
    shape = state.shape
    mass, diff, bc = problem.mass, problem.diffusion, problem.constraints
    factors = [linalg.lu_factor(M-dt/2*L+C) for M, L, C in zip(mass, diff, bc)]
    starters = [linalg.lu_factor(M-dt*L+C) for M, L, C in zip(mass, diff, bc)]
    plus = mass+dt/2*diff
    records, previous_rhs = [], None
    probe_z = int(np.argmin(abs(problem.grid.z-1.6)))
    previous_velocity = None

    def record(step):
        nonlocal previous_velocity
        velocity = problem.velocity(state)
        energy = problem.energy(state)
        if not np.isfinite(energy) or energy <= 0:
            raise RuntimeError("Time integration diverged; reduce dt or inspect constraints")
        overlap = np.nan
        if previous_velocity is not None:
            dot = np.einsum('hvi,ij,hvj->', previous_velocity.conj(), problem.grid.mass, velocity)
            old_norm = np.einsum('hvi,ij,hvj->', previous_velocity.conj(), problem.grid.mass, previous_velocity).real
            overlap = float(abs(dot)/(np.sqrt(old_norm)*np.sqrt(2*energy)))
        records.append((step*dt, energy, velocity[problem.K, 1, probe_z], overlap))
        previous_velocity = velocity.copy()
        if callback:
            callback(step*dt, energy)

    record(0)
    for step in range(steps):
        rhs = problem.explicit_rhs(state)
        flat = state.reshape(len(problem.ns), -1)
        if previous_rhs is None:
            forcing = np.einsum('hij,hj->hi', mass, flat)+dt*rhs
            lu = starters
        else:
            forcing = np.einsum('hij,hj->hi', plus, flat)+dt*(1.5*rhs-.5*previous_rhs)
            lu = factors
        state = np.array([linalg.lu_solve(factor, f, check_finite=False)
                          for factor, f in zip(lu, forcing)]).reshape(shape)
        previous_rhs = rhs
        if (step+1) % sample_every == 0 or step == steps-1:
            record(step+1)
    time, energy, probe, overlap = map(np.asarray, zip(*records))
    return dict(time=time, energy=energy, probe=probe, shape_overlap=overlap,
                final_state=state, dt=dt, probe_z=float(problem.grid.z[probe_z]),
                constraints=problem.constraints_defect(state))
