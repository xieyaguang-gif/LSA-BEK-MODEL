"""Equivalent phase-pinned equations for a convectively amplified wave.

Rotate all prescribed Fourier data by an unknown common phase and fix the
fundamental phase at an interior section. Rotating the answer back recovers
the original inlet, outlet and forcing. No physical phase freedom is added.
"""
import numpy as np
from scipy import sparse

from .radial import pack_field, unpack_field


class PhaseGauge:
    def __init__(self, problem, reference, radius, inlet, outlet=None, forcing=None):
        self.problem = problem
        self.inlet, self.outlet, self.forcing = inlet, outlet, forcing
        if not problem.radii[0] < radius < problem.radii[-1]:
            raise ValueError('Phase pin must be strictly inside the radial domain')
        self.index = int(np.argmin(abs(problem.radii-radius)))
        if self.index in (0, len(problem.radii)-1):
            raise ValueError('Phase pin must select an interior node')
        self.numbers = np.arange(-problem.H, problem.H+1)[:, None]
        self.size = len(problem.radii)*problem.width
        self.phase_index = self.size
        m, s = problem.ops[0].m, problem.s
        velocity = reference[self.index, problem.H+1, :3*m].reshape(3,m)
        mass = problem.ops[0].grid.mass
        weight = velocity.conj()@mass
        magnitude = float(np.sum(weight*velocity).real)
        if magnitude <= 1e-100:
            raise ValueError('Phase pin requires a nonzero fundamental velocity')
        weight = weight.ravel()/np.sqrt(magnitude)
        real_indices = self.index*problem.width+s+np.arange(3*m)
        columns = np.r_[real_indices, real_indices+s]
        self.row = sparse.csr_matrix((np.r_[weight.imag,weight.real],
            (np.zeros(len(columns),int),columns)), shape=(1,self.size))

    def rotate(self, field, angle):
        return None if field is None else field*np.exp(1j*self.numbers*angle)

    def initial_state(self, field):
        return np.r_[pack_field(field),0.]

    def physical_residual(self, state):
        return self.problem.evaluate(pack_field(self.physical_field(state)),self.inlet,
            outlet=self.outlet,forcing=self.forcing)

    def physical_field(self, state):
        p = self.problem
        field = unpack_field(state[:-1], len(p.radii), p.H, p.s)
        return self.rotate(field, -state[-1])

    def evaluate(self, state, jacobian=False):
        p, angle = self.problem, state[-1]
        inlet = self.rotate(self.inlet, angle)
        outlet = self.rotate(self.outlet, angle)
        forcing = self.rotate(self.forcing, angle)
        result = p.evaluate(state[:-1], inlet, outlet=outlet,
                            forcing=forcing, jacobian=jacobian)
        if jacobian:
            residual, matrix = result
        else:
            residual = result
        residual = np.r_[residual, (self.row@state[:-1])[0]]
        if not jacobian:
            return residual
        shape = (len(p.radii), 2*p.H+1, p.s)
        derivative = np.zeros(shape,complex) if forcing is None else 1j*self.numbers*forcing
        derivative[0][:,p.velocity_rows] = (1j*self.numbers*inlet)[:,p.velocity_rows]
        derivative[-1][:,p.velocity_rows] = (0 if outlet is None else
                                            (1j*self.numbers*outlet)[:,p.velocity_rows])
        column = sparse.csc_matrix(-pack_field(derivative)[:,None])
        return residual, sparse.bmat([[matrix,column],[self.row,None]],format='csc')
