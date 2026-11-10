"""Vectorized primitive-variable polynomial pencil A0 + alpha A1 + alpha² A2."""
from dataclasses import dataclass

import numpy as np

from .spectral import chebyshev


@dataclass(frozen=True)
class Parameters:
    re: float = 500.0
    ro: float = -0.751
    beta: float = 85.0
    n: int = 64
    zmax: float = 20.0
    stretch: float = 0.25
    boundary: str = "matlab"

    def __post_init__(self):
        if not np.isfinite(self.re) or self.re <= 0:
            raise ValueError("re must be positive: re=abs(ro)*r_delta")
        if not np.isfinite(self.ro) or not -1 <= self.ro <= 1 or self.ro == 0:
            raise ValueError("Stability scaling requires 0 < abs(ro) <= 1")
        if not np.isfinite(self.beta) or np.iscomplexobj(self.beta):
            raise ValueError("beta must be real (integer for a full annulus)")
        if self.boundary not in ("paper", "matlab"):
            raise ValueError("boundary must be paper or matlab")
        chebyshev(self.n, self.zmax, self.stretch)


class LocalProblem:
    """One prepared Re, Ro, beta; reuse flow interpolation and matrices across alpha.

    Coefficients interleave [u_n,v_n,w_n,p_n], as MATLAB cell2mat does.
    Temporal eigenvalues satisfy A q = omega B q, with B=i S on momentum rows.
    """

    def __init__(self, baseflow, parameters=None):
        self.parameters = p = parameters or Parameters()
        if baseflow.ro is not None and not np.isclose(baseflow.ro, p.ro, atol=1e-12, rtol=0):
            raise ValueError("Base-flow Ro and stability Ro differ")
        self.baseflow = baseflow
        self.grid = g = chebyshev(p.n, p.zmax, p.stretch)
        f = self.flow = baseflow.sample(g.z)
        m = g.size
        blocks = [np.zeros((m, 4, m, 4), complex) for _ in range(4)]
        a0, a1, a2, b = blocks
        S, D, D2 = g.S[1:-1], g.D[1:-1], g.D2[1:-1]
        re, ro, k = p.re, p.ro, p.beta*abs(p.ro)/p.re
        U, V, W, dU, dV, dW = (getattr(f, v)[1:-1, None] for v in ("U", "V", "W", "dU", "dV", "dW"))
        adv = (1j*k*V+k*k/re)*S + ro*W/re*D - D2/re
        cor = (2*ro*V+2-ro-ro*ro)/re*S
        for v in range(3):
            a0[1:-1, v, :, v] = adv + ro/re*(dW if v == 2 else U)*S
            a1[1:-1, v, :, v] = 1j*U*S
            a2[1:-1, v, :, v] = S/re
            b[1:-1, v, :, v] = 1j*S
        a0[1:-1, 0, :, 1] = -cor
        a0[1:-1, 1, :, 0] = cor
        a0[1:-1, 0, :, 2] = dU*S
        a0[1:-1, 1, :, 2] = dV*S
        a0[1:-1, 1, :, 3] = 1j*k*S
        a0[1:-1, 2, :, 3] = D
        a0[1:-1, 3, :, 0] = ro/re*S
        a0[1:-1, 3, :, 1] = 1j*k*S
        a0[1:-1, 3, :, 2] = D
        a1[1:-1, 0, :, 3] = 1j*S
        a1[1:-1, 3, :, 0] = 1j*S
        # g.z[0]=zmax, g.z[-1]=0. The legacy MATLAB endpoint order is reversed
        # relative to the boundary conditions printed in paper equation (3.10).
        for endpoint in (0, -1):
            for v in range(3):
                a0[endpoint, v, :, v] = g.S[endpoint]
        pressure_end = -1 if p.boundary == "paper" else 0
        derivative_end = 0 if p.boundary == "paper" else -1
        a0[pressure_end, 3, :, 3] = g.S[pressure_end]
        a0[derivative_end, 3, :, 2] = g.D[derivative_end]
        self.A0, self.A1, self.A2, self.B = [x.reshape(4*m, 4*m) for x in blocks]

    def matrix(self, alpha):
        if not np.isfinite(alpha):
            raise ValueError("alpha must be finite")
        return self.A0 + alpha*self.A1 + alpha*alpha*self.A2

    def fields(self, coefficients):
        """Nodal [u,v,w,p] values, in descending z order; complex phase retained."""
        return self.grid.S @ np.asarray(coefficients).reshape(self.grid.size, 4)
