"""Semi-infinite Chebyshev collocation, Pier (2003) equation (4.3)."""
from dataclasses import dataclass
from functools import lru_cache

import numpy as np
from scipy.interpolate import BarycentricInterpolator
from numpy.polynomial.legendre import leggauss

from ..baseflow import solve_bek


@dataclass(frozen=True)
class PierGrid:
    z: np.ndarray
    xi: np.ndarray
    D: np.ndarray
    D2: np.ndarray
    mass: np.ndarray
    a: float
    b: float
    c: float

    @property
    def size(self):
        return len(self.z)


@lru_cache(maxsize=16)
def pier_grid(n=40, a=2.0, b=0.6, c=0.5):
    if int(n) != n or n < 8 or a <= 0 or not 0 < b <= 1 or not 0 <= c <= 1:
        raise ValueError("Require integer n>=8, a>0, 0<b<=1, 0<=c<=1")
    n = int(n)
    xi = -np.cos(np.pi*np.arange(n+1)/n)
    weights = (-1.0)**np.arange(n+1)
    weights[[0, -1]] *= .5
    delta = xi[:, None]-xi[None, :]
    np.fill_diagonal(delta, 1)
    Dxi = (weights[None, :]/weights[:, None])/delta
    np.fill_diagonal(Dxi, 0)
    np.fill_diagonal(Dxi, -Dxi.sum(axis=1))

    def mapping(x):
        t = b*x+(1-b)*(x**3+c*(1-x*x))
        tp = b+(1-b)*(3*x*x-2*c*x)
        tpp = (1-b)*(6*x-2*c)
        return t, tp, tpp

    t, tp, tpp = mapping(xi)
    if np.any(tp <= 0):
        raise ValueError("Non-monotone mapping")
    z = np.full(n+1, np.inf)
    z[:-1] = a*(1+t[:-1])/(1-t[:-1])
    xp = (1-t)**2/(2*a*tp)
    xpp = xp*(-(1-t)/a-(1-t)**2*tpp/(2*a*tp**2))
    D = xp[:, None]*Dxi
    D2 = xp[:, None]**2*(Dxi@Dxi)+xpp[:, None]*Dxi
    # Positive quadrature on the interpolating nodal polynomial; quadrature nodes
    # avoid the singular endpoint. All velocity fields must vanish at infinity.
    xq, wq = leggauss(2*n+80)
    tq, tpq, _ = mapping(xq)
    interpolation = BarycentricInterpolator(xi, np.eye(n+1), axis=0)(xq)
    measure = wq*2*a*tpq/(1-tq)**2
    mass = interpolation.T @ (measure[:, None]*interpolation)
    for arr in (z, xi, D, D2, mass):
        arr.setflags(write=False)
    return PierGrid(z, xi, D, D2, mass, a, b, c)


def rotation_rates(ro):
    """(Omega_disk/Omega_BEK, Omega_external/Omega_BEK), including Ro=0."""
    if not np.isfinite(ro) or not -1 <= ro <= 1:
        raise ValueError("BEK requires finite -1 <= ro <= 1")
    disk = (2-ro-ro*ro)/2
    return float(disk), float(disk+ro)


def sample_inertial_base(z, ro=-1.):
    """General BEK -> inertial profiles; time unit is 1/Omega_BEK.

    F=Ro*U_BEK, G=Co/2+Ro*V_BEK, H=Ro*W_BEK.
    The nonzero external G must be retained beyond the BVP domain.
    """
    disk, external = rotation_rates(ro)
    z = np.asarray(z)
    if np.any(np.isnan(z)) or np.any(z < 0):
        raise ValueError("Nonnegative wall-normal coordinates required")
    bek = solve_bek(ro, zmax=40, tol=1e-10, samples=16001)
    values = bek.sample(np.minimum(z, 40))
    U, V, W = ro*values.U, disk+ro*values.V, ro*values.W
    dU, dV, dW = ro*values.dU, ro*values.dV, ro*values.dW
    far = z >= 40
    for field in (U, dU, dV, dW):
        field[far] = 0
    V[far] = external
    W[far] = ro*bek.state[0, -1]
    return U, V, W, dU, dV, dW


@lru_cache(maxsize=32)
def inertial_base(n=40, a=2.0, b=0.6, c=0.5, ro=-1.):
    fields = sample_inertial_base(pier_grid(n, a, b, c).z, ro)
    U, V, W, dU, dV, dW = fields
    for arr in (U, V, W, dU, dV, dW):
        arr.setflags(write=False)
    return U, V, W, dU, dV, dW
