"""Pseudo-arclength corrector for the fixed-grid damping-release equations.

The release fraction is an unknown, not a prescribed continuation coordinate.
Auxiliary points, including fractions outside [0,1], are never target data.
"""
import numpy as np
from scipy import sparse

from .phase_gauge import PhaseGauge
from .radial import pack_field, unpack_field


class DampingArc:
    def __init__(self, problem, target_sponge, inlet, previous, current,
                 previous_fraction, fraction, step, *, phase_pin_radius=None,
                 outlet=None, forcing=None, parameter_weight=1.):
        if not np.isfinite(step) or step <= 0 or not np.isfinite(parameter_weight) or parameter_weight <= 0:
            raise ValueError('Positive finite arclength step and parameter weight required')
        old = problem.outlet_sponge
        if old is None or old.damp_mean != target_sponge.damp_mean:
            raise ValueError('Require an original sponge with the same mean-damping setting')
        if target_sponge.start+target_sponge.width > problem.radii[-1]+1e-12:
            raise ValueError('Target sponge must finish inside the fixed domain')
        self.problem, self.inlet = problem, inlet
        self.outlet, self.forcing = outlet, forcing
        self.size = len(problem.radii)*problem.width
        self.gauge = (None if phase_pin_radius is None else
            PhaseGauge(problem,current,phase_pin_radius,inlet,outlet,forcing))
        self.phase_index = None if self.gauge is None else self.size
        self.reference = current
        difference = (target_sponge.profile(problem.radii)-old.profile(problem.radii))/problem.radii
        diagonal = np.zeros((len(problem.radii),2*problem.H+1,problem.s))
        for part in range(2*problem.H+1):
            if part or old.damp_mean:
                diagonal[1:-1,part][:,problem.velocity_rows] = difference[1:-1,None]
        self.diagonal = diagonal.ravel()
        padded = self.diagonal if self.gauge is None else np.r_[self.diagonal,0.]
        self.parameter_matrix = sparse.diags(padded,format='csc')
        self.current = self.state(current,fraction)
        before = self.state(previous,previous_fraction)
        metric = np.ones(len(self.current))
        metric[:self.size] = 1/max(np.linalg.norm(pack_field(current))**2,1e-30)
        metric[-1] = parameter_weight**2
        self.metric,self.step = metric,float(step)
        delta = self.current-before
        distance = np.sqrt(np.dot(metric*delta,delta))
        if not np.isfinite(distance) or distance <= 1e-14:
            raise ValueError('Arclength requires two distinct converged states')
        self.previous_distance = float(distance)
        self.tangent = delta/distance
        self.row = metric*self.tangent
        self.predictor = self.current+step*self.tangent
        self.last_fraction = float(self.predictor[-1])
        self.accepted_state = None

    def directional_second_derivative(self,state,direction):
        """Exact F_yy[t,t]: quadratic convection, tau*q and rotated data."""
        p = self.problem
        field = unpack_field(direction[:self.size],len(p.radii),p.H,p.s)
        second = np.zeros(self.size)
        if p.config.nonlinear:
            second += 2*pack_field(p._convection_field(field,p.derivatives(field)[0]))
        second += 2*direction[-1]*self.diagonal*direction[:self.size]
        if self.gauge is not None:
            angle = state[self.phase_index]
            given = (np.zeros_like(field) if self.forcing is None else self.gauge.rotate(self.forcing,angle))
            inlet = self.gauge.rotate(self.inlet,angle)
            outlet = self.gauge.rotate(self.outlet,angle)
            given[0][:,p.velocity_rows] = inlet[:,p.velocity_rows]
            given[-1][:,p.velocity_rows] = 0 if outlet is None else outlet[:,p.velocity_rows]
            second += direction[self.phase_index]**2*pack_field(self.gauge.numbers**2*given)
            second = np.r_[second,0.]
        return np.r_[second,0.]

    def prepare_predictor(self,solve,order=2):
        """Use one bordered factorization for a tangent and optional curvature.

        ``solve(A,b)`` must solve the actual matrix; both calls use the same A.
        The tangent orientation is inherited from the previous secant.
        """
        if order not in (1,2):
            raise ValueError('Exact predictor order must be 1 or 2')
        _,matrix = self.evaluate(self.current,jacobian=True)
        rhs = np.zeros(len(self.current));rhs[-1] = 1.
        tangent = np.asarray(solve(matrix,rhs))
        tangent /= np.sqrt(np.dot(self.metric*tangent,tangent))
        second = self.directional_second_derivative(self.current,tangent)
        curvature = np.zeros_like(tangent)
        if order==2:
            curvature = np.asarray(solve(matrix,-second))
            # Remove the component along the exact null tangent. This changes
            # the normalization row from the secant to the exact tangent,
            # without another factorization or changing the flow equations.
            curvature -= tangent*np.dot(self.metric*tangent,curvature)
        self.tangent = tangent
        self.row = self.metric*tangent
        linear = self.current+self.step*tangent
        self.predictor = linear
        linear_error = float(np.linalg.norm(self.evaluate(linear),np.inf))
        quadratic_error = None
        selected = 'exact_tangent'
        if order==2:
            quadratic = linear+.5*self.step**2*curvature
            quadratic_error = float(np.linalg.norm(self.evaluate(quadratic),np.inf))
            if np.isfinite(quadratic_error) and quadratic_error < linear_error:
                self.predictor = quadratic
                selected = 'quadratic'
        self.last_fraction = float(self.predictor[-1])
        return dict(selected_predictor=selected,linear_prediction_residual_inf=linear_error,
            quadratic_prediction_residual_inf=quadratic_error,
            tangent_flow_residual_inf=float(np.linalg.norm((matrix@tangent)[:-1],np.inf)),
            curvature_flow_residual_inf=(None if order==1 else
                float(np.linalg.norm((matrix@curvature+second)[:-1],np.inf))),
            curvature_metric_norm=float(np.sqrt(np.dot(self.metric*curvature,curvature))))

    def state(self, field, fraction):
        if self.gauge is None:
            return np.r_[pack_field(field),fraction]
        p,g = self.problem,self.gauge
        m = p.ops[0].m
        a = self.reference[g.index,p.H+1,:3*m].reshape(3,m)
        b = field[g.index,p.H+1,:3*m].reshape(3,m)
        overlap = np.sum((a.conj()@p.ops[0].grid.mass)*b)
        if abs(overlap) <= 1e-100:
            raise ValueError('Cannot align a zero/orthogonal fundamental at the phase pin')
        angle = -float(np.angle(overlap))
        return np.r_[pack_field(g.rotate(field,angle)),angle,fraction]

    def initial_state(self, field):
        return self.predictor.copy()

    def physical_field(self, state):
        if self.gauge is not None:
            return self.gauge.physical_field(state[:-1])
        p = self.problem
        return unpack_field(state[:self.size],len(p.radii),p.H,p.s)

    def physical_residual(self, state):
        q = pack_field(self.physical_field(state))
        return (self.problem.evaluate(q,self.inlet,outlet=self.outlet,forcing=self.forcing)
                +state[-1]*self.diagonal*q)

    def evaluate(self, state, jacobian=False):
        self.last_fraction = float(state[-1])
        if self.gauge is None:
            result = self.problem.evaluate(state[:-1],self.inlet,outlet=self.outlet,
                forcing=self.forcing,jacobian=jacobian)
        else:
            result = self.gauge.evaluate(state[:-1],jacobian=jacobian)
        residual,matrix = result if jacobian else (result,None)
        residual = residual.copy()
        derivative = self.diagonal*state[:self.size]
        residual[:self.size] += state[-1]*derivative
        result = np.r_[residual,np.dot(self.row,state-self.predictor)]
        if not jacobian:
            return result
        column = derivative if self.gauge is None else np.r_[derivative,0.]
        matrix = matrix+state[-1]*self.parameter_matrix
        return result,sparse.vstack([
            sparse.hstack([matrix,sparse.csc_matrix(column[:,None])],format='csc'),
            sparse.csr_matrix(self.row[None,:])],format='csc')
