"""Real Fourier travelling waves, constrained Newton solves and continuation."""
from dataclasses import dataclass
import json
from pathlib import Path

import numpy as np
from scipy import linalg, optimize

from .operators import PierOperators


def pack(modes):
    H = (len(modes)-1)//2
    return np.concatenate([modes[H].real]+[
        part for j in range(1, H+1) for part in (modes[H+j].real, modes[H+j].imag)])


def unpack(vector, harmonics, size):
    out = np.zeros((2*harmonics+1, size), complex)
    out[harmonics] = vector[:size]
    for j in range(1, harmonics+1):
        start = (2*j-1)*size
        out[harmonics+j] = vector[start:start+size]+1j*vector[start+size:start+2*size]
        out[harmonics-j] = out[harmonics+j].conj()
    return out


@dataclass
class PrimaryWave:
    operators: PierOperators
    alpha: float
    omega: float
    modes: np.ndarray
    residual: float
    history: list

    @property
    def harmonics(self):
        return (len(self.modes)-1)//2

    def energies(self):
        """Pier equation (5.5), one positive harmonic; no extra factor of two."""
        op = self.operators
        v = self.modes[self.harmonics+1:].reshape(self.harmonics, 4, op.m)[:, :3]
        return op.radius**2*np.einsum('hvi,ij,hvj->h', v.conj(), op.grid.mass, v).real

    def fields(self, phases, total=False):
        """Physical Pier velocities/pressure, shape=(phase,4,z), for real phases."""
        phases = np.asarray(phases)
        ns = np.arange(-self.harmonics, self.harmonics+1)
        out = (np.exp(1j*phases[:, None]*ns)@self.modes).reshape(len(phases), 4, self.operators.m).real
        out[:, :3] *= self.operators.radius
        out[:, 3] *= self.operators.radius**2
        if total:
            U, V, W, _, _, dW = self.operators.base
            out[:, 0] += self.operators.radius*U
            out[:, 1] += self.operators.radius*V
            out[:, 2] += W
            out[:, 3] += dW+(W[-1]**2-W**2)/2 + self.operators.external_rotation**2*self.operators.radius**2/2
        return out

    def save(self, path):
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        op = self.operators
        np.savez_compressed(path, modes=self.modes, alpha=self.alpha, omega=self.omega,
                            radius=op.radius, beta=op.beta, n=op.m-1,
                            a=op.grid.a, b=op.grid.b, c=op.grid.c,
                            mean_mode=op.mean_mode, ro=op.ro, schema_version=2,
                            convention="inertial; time=1/Omega_BEK; length=sqrt(nu/Omega_BEK)",
                            residual=self.residual, energies=self.energies(),
                            history_json=json.dumps(self.history))

    @classmethod
    def load(cls, path):
        with np.load(path, allow_pickle=False) as d:
            if 'mean_mode' not in d:
                raise ValueError("Primary file lacks mean_mode metadata; regenerate the wave")
            op = PierOperators(**{k: d[k].item() for k in ('radius', 'beta', 'n', 'a', 'b', 'c')},
                               mean_mode=str(d['mean_mode']), ro=float(d['ro']) if 'ro' in d else -1.)
            return cls(op, float(d['alpha']), float(d['omega']), d['modes'].copy(),
                       float(d['residual']), json.loads(str(d['history_json'])))


class WaveSolver:
    def __init__(self, operators=None, harmonics=3):
        if int(harmonics) != harmonics or harmonics < 1:
            raise ValueError("harmonics must be a positive integer")
        self.op = operators or PierOperators()
        self.H = int(harmonics)
        self.dimension = (2*self.H+1)*self.op.size

    def residual(self, modes, alpha, omega):
        out = self.op.nonlinear(modes, alpha)
        for j in range(-self.H, self.H+1):
            A = self.op.linear(j*alpha, j*self.op.beta)-1j*j*omega/self.op.radius*self.op.M
            out[j+self.H] += A@modes[j+self.H]
        return pack(out)

    def jacobian(self, modes, alpha, omega):
        """Analytic real Jacobian including both signs of every Fourier harmonic."""
        op, H, s = self.op, self.H, self.op.size
        J = np.zeros((self.dimension, self.dimension))
        for j in range(H+1):
            row = slice(0, s) if j == 0 else slice((2*j-1)*s, (2*j+1)*s)
            for k in range(H+1):
                blocks = []
                for kk in ([0] if k == 0 else [k, -k]):
                    base_n = j-kk
                    C = (op.coupling(modes[base_n+H], base_n, alpha, kk*alpha, kk*op.beta)
                         if abs(base_n) <= H else np.zeros((s, s), complex))
                    if j == kk:
                        C = C+op.linear(j*alpha, j*op.beta)-1j*j*omega/op.radius*op.M
                    blocks.append(C)
                if k == 0:
                    block = blocks[0].real if j == 0 else np.vstack((blocks[0].real, blocks[0].imag))
                    J[row, :s] = block
                else:
                    real, imag = blocks[0]+blocks[1], 1j*(blocks[0]-blocks[1])
                    block = (np.hstack((real.real, imag.real)) if j == 0 else
                             np.block([[real.real, imag.real], [real.imag, imag.imag]]))
                    J[row, (2*k-1)*s:(2*k+1)*s] = block
        return J

    def _constraints(self, modes, energy):
        op, s = self.op, self.op.size
        phase = modes[self.H+1, 3*op.m].imag
        phase_row = np.zeros(self.dimension)
        phase_row[2*s+3*op.m] = 1
        if energy is None:
            return [phase], [phase_row]
        v = modes[self.H+1].reshape(4, op.m)[:3]
        amp = np.einsum('vi,ij,vj->', v.conj(), op.grid.mass, v).real-energy/op.radius**2
        amp_row = np.zeros(self.dimension)
        grad = 2*v@op.grid.mass
        amp_row[s:s+3*op.m] = grad.real.ravel()
        amp_row[2*s:2*s+3*op.m] = grad.imag.ravel()
        return [phase, amp], [phase_row, amp_row]

    def solve(self, seed, *, energy=None, frequency=None, wavenumber=None, tol=1e-10, max_iter=30):
        """Fix E1 (solve alpha and omega) OR fix frequency (solve alpha).

        Energy continuation excludes the trivial zero-amplitude branch. Frequency
        solves require a nonzero seed on the desired branch. A phase condition is
        imposed as in Pier: Im(p1(z=0))=0.
        """
        if self.op.ro == 0:
            raise ValueError("Ro=0 is exact corotation in this inertial scaling; no finite-shear primary wave. Use nonzero Ro.")
        if sum(v is not None for v in (energy, frequency, wavenumber)) != 1:
            raise ValueError("Specify exactly one of energy, frequency or wavenumber")
        if energy is not None and (not np.isfinite(energy) or energy <= 0):
            raise ValueError("energy must be positive and finite")
        for value in (frequency, wavenumber):
            if value is not None and (np.iscomplexobj(value) or not np.isfinite(value)):
                raise ValueError("Travelling-wave frequency and wavenumber must be real and finite")
        oldH = seed.harmonics
        modes = np.zeros((2*self.H+1, self.op.size), complex)
        old_grid, new_grid = seed.operators.grid, self.op.grid
        if (seed.operators.m != self.op.m or
                (old_grid.a, old_grid.b, old_grid.c) != (new_grid.a, new_grid.b, new_grid.c)):
            from scipy.interpolate import BarycentricInterpolator
            values = seed.modes.reshape(2*oldH+1, 4, seed.operators.m)
            z = new_grid.z[:-1]
            transformed = (z-old_grid.a)/(z+old_grid.a)
            old_xi = np.r_[[optimize.brentq(
                lambda x: old_grid.b*x+(1-old_grid.b)*(x**3+old_grid.c*(1-x*x))-t,
                -1, 1) for t in transformed], 1]
            values = BarycentricInterpolator(old_grid.xi, values, axis=2)(old_xi)
            old_modes = values.reshape(2*oldH+1, self.op.size)
        else:
            old_modes = seed.modes
        common = min(oldH, self.H)
        modes[self.H-common:self.H+common+1] = old_modes[oldH-common:oldH+common+1]
        x = (np.r_[pack(modes), seed.alpha, seed.omega] if energy is not None else
             np.r_[pack(modes), seed.alpha if frequency is not None else seed.omega])
        history = []

        def evaluate(xx):
            q = unpack(xx[:self.dimension], self.H, self.op.size)
            alpha = xx[self.dimension] if wavenumber is None else wavenumber
            omega = xx[-1] if frequency is None else frequency
            F = self.residual(q, alpha, omega)
            cons, _ = self._constraints(q, energy)
            return np.r_[F, cons], q, alpha, omega

        for iteration in range(max_iter):
            F, modes, alpha, omega = evaluate(x)
            norm = float(np.linalg.norm(F, np.inf))
            history.append(dict(iteration=iteration, residual=norm, alpha=float(alpha), omega=float(omega)))
            if norm < tol:
                wave = PrimaryWave(self.op, float(alpha), float(omega), modes, norm, history)
                if wave.energies().sum() < 1e-6:
                    raise RuntimeError("Newton converged to the trivial laminar branch")
                return wave
            Jq = self.jacobian(modes, alpha, omega)
            da = 1e-6
            Ja = (self.residual(modes, alpha+da, omega)-self.residual(modes, alpha-da, omega))/(2*da)
            Jw = pack(np.array([-1j*j/self.op.radius*self.op.M@modes[j+self.H]
                                for j in range(-self.H, self.H+1)]))
            columns = [Jq, (Jw if wavenumber is not None else Ja)[:, None]]
            if energy is not None:
                columns.append(Jw[:, None])
            upper = np.hstack(columns)
            _, cons_jac = self._constraints(modes, energy)
            lower = np.pad(np.asarray(cons_jac), ((0, 0), (0, len(x)-self.dimension)))
            J = np.vstack((upper, lower))
            step = linalg.solve(J, -F, check_finite=False)
            scale = 1.0
            while scale >= 1/128:
                trial = x+scale*step
                ftrial, *_ = evaluate(trial)
                if np.linalg.norm(ftrial, np.inf) < norm:
                    x = trial
                    break
                scale *= .5
            else:
                raise RuntimeError(f"Wave Newton line search failed at residual {norm:.3e}")
        raise RuntimeError(f"Wave Newton did not converge after {max_iter} iterations")

    def linear_seed(self, energy=.01, bracket=(.5, .7), target=65):
        op = self.op

        def mode(alpha):
            values, vectors = linalg.eig(op.linear(alpha, op.beta), 1j*op.M)
            valid = np.isfinite(values)
            values, vectors = values[valid]*op.radius, vectors[:, valid]
            k = np.argmin(abs(values-target))
            return values[k], vectors[:, k]

        alpha = optimize.brentq(lambda a: mode(a)[0].imag, *bracket, xtol=1e-12)
        omega, q = mode(alpha)
        q *= np.exp(-1j*np.angle(q[3*op.m]))
        v = q.reshape(4, op.m)[:3]
        e = op.radius**2*np.einsum('vi,ij,vj->', v.conj(), op.grid.mass, v).real
        q *= np.sqrt(energy/e)
        modes = np.zeros((2*self.H+1, op.size), complex)
        modes[self.H+1], modes[self.H-1] = q, q.conj()
        return PrimaryWave(op, alpha, omega.real, modes, np.inf, [])

    def bootstrap(self, energies=(.01, .1, 1, 5, 20, 50, 100, 200, 400, 620), callback=None,
                  bracket=(.5, .7), target=65):
        wave = self.linear_seed(energy=energies[0], bracket=bracket, target=target)
        for energy in energies:
            wave = self.solve(wave, energy=energy)
            if callback:
                callback(wave)
        return wave

    def arclength_step(self, previous, current, step=.001, weight=.5, tol=1e-10, max_iter=25):
        """Secant predictor/corrector in (alpha,omega/R), following Pier (5.8).

        Unlike fixing alpha this can pass an alpha turning point. Both input waves
        must be on the same phase-fixed branch with identical discretization.
        step is measured in the weighted (alpha,omega/R) plane.
        """
        if not 0 < weight < 1 or step <= 0:
            raise ValueError("Require step>0 and 0<weight<1")
        op = self.op
        for wave in (previous, current):
            other = wave.operators
            if (wave.harmonics != self.H or other.m != op.m or
                    (other.ro, other.radius, other.beta, other.mean_mode, other.grid.a, other.grid.b, other.grid.c) !=
                    (op.ro, op.radius, op.beta, op.mean_mode, op.grid.a, op.grid.b, op.grid.c)):
                raise ValueError("Arclength inputs must use identical equations and discretization")
        delta = np.array([current.alpha-previous.alpha, (current.omega-previous.omega)/op.radius])
        metric = np.array([1-weight, weight])
        distance = np.sqrt(np.sum(metric*delta**2))
        if distance < 1e-12:
            raise ValueError("Arclength needs two distinct branch points")
        tangent = delta/distance
        old = np.r_[pack(previous.modes), previous.alpha, previous.omega]
        centre = np.r_[pack(current.modes), current.alpha, current.omega]
        x = centre+(centre-old)*step/distance
        history = []

        def evaluate(xx):
            modes = unpack(xx[:self.dimension], self.H, op.size)
            a, w = xx[-2:]
            phase, _ = self._constraints(modes, None)
            offset = np.array([a-current.alpha, (w-current.omega)/op.radius])
            arc = np.sum(metric*tangent*offset)-step
            return np.r_[self.residual(modes, a, w), phase, arc], modes, a, w

        for iteration in range(max_iter):
            F, q, a, w = evaluate(x)
            norm = float(np.linalg.norm(F, np.inf))
            history.append(dict(iteration=iteration, residual=norm, alpha=float(a), omega=float(w)))
            if norm < tol:
                return PrimaryWave(op, float(a), float(w), q, norm, history)
            Jq = self.jacobian(q, a, w)
            h = 1e-6
            Ja = (self.residual(q, a+h, w)-self.residual(q, a-h, w))/(2*h)
            Jw = pack(np.array([-1j*j/op.radius*op.M@q[j+self.H]
                                for j in range(-self.H, self.H+1)]))
            _, phase_jac = self._constraints(q, None)
            phase_row = np.r_[phase_jac[0], 0, 0]
            arc_row = np.r_[np.zeros(self.dimension), metric[0]*tangent[0], metric[1]*tangent[1]/op.radius]
            J = np.vstack((np.column_stack((Jq, Ja, Jw)), phase_row, arc_row))
            change = linalg.solve(J, -F, check_finite=False)
            scale = 1.0
            while scale >= 1/128:
                trial = x+scale*change
                if np.linalg.norm(evaluate(trial)[0], np.inf) < norm:
                    x = trial
                    break
                scale *= .5
            else:
                raise RuntimeError("Arclength corrector line search failed")
        raise RuntimeError("Arclength corrector did not converge")
