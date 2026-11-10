"""BEK boundary-value solver and explicit legacy table formats."""
from dataclasses import dataclass
from pathlib import Path

import numpy as np
from scipy.integrate import solve_bvp
from scipy.interpolate import CubicSpline, PchipInterpolator


@dataclass(frozen=True)
class Flow:
    U: np.ndarray
    V: np.ndarray
    W: np.ndarray
    dU: np.ndarray
    dV: np.ndarray
    dW: np.ndarray


@dataclass
class BaseFlow:
    """Canonical state rows: W, U', U, V', V; z increases from the wall."""
    z: np.ndarray
    state: np.ndarray
    source: str
    ro: float | None = None
    interpolation: str = "pchip"

    def __post_init__(self):
        self.z = np.array(self.z, dtype=float, copy=True)
        self.state = np.array(self.state, dtype=float, copy=True)
        if (self.z.ndim != 1 or len(self.z) < 4 or
                self.state.shape != (5, len(self.z)) or
                not np.all(np.isfinite(self.z)) or
                not np.all(np.isfinite(self.state)) or np.any(np.diff(self.z) <= 0)):
            raise ValueError("Expected finite state (5, n) and strictly increasing z (n >= 4)")
        if self.interpolation not in ("pchip", "spline"):
            raise ValueError("interpolation must be pchip or spline")
        cls = PchipInterpolator if self.interpolation == "pchip" else CubicSpline
        self._interpolator = cls(self.z, self.state, axis=1, extrapolate=False)

    def sample(self, z):
        z = np.asarray(z, dtype=float)
        if (not np.all(np.isfinite(z)) or np.min(z) < self.z[0]-1e-12 or
                np.max(z) > self.z[-1]+1e-12):
            raise ValueError(f"Requested grid exceeds base-flow domain [{self.z[0]}, {self.z[-1]}]")
        W, dU, U, dV, V = self._interpolator(np.clip(z, self.z[0], self.z[-1]))
        return Flow(U, V, W, dU, dV, -2*U)

    def save(self, path):
        np.savetxt(path, np.column_stack((self.z, self.state.T)), delimiter=",",
                   header="z,W,dU,U,dV,V")


def load_baseflow(path, layout="temporal", interpolation="pchip", ro=None):
    """Read six-column comma/whitespace text; never infer velocity ordering.

    temporal: z,W,dU,U,dV,V; spatial: z,U,dU,V,dV,W.
    Extrapolation is forbidden (bundled IRSM profiles end at z=19.851...).
    """
    path = Path(path)
    lines = (x for x in path.read_text().splitlines() if x.strip() and not x.lstrip().startswith("#"))
    first = next(lines, "")
    a = np.loadtxt(path, delimiter="," if "," in first else None, ndmin=2)
    if a.shape[1] != 6:
        raise ValueError("Base-flow file must have exactly six columns")
    if layout == "temporal":
        state = a[:, [1, 2, 3, 4, 5]].T
    elif layout == "spatial":
        state = a[:, [5, 2, 1, 4, 3]].T
    else:
        raise ValueError("layout must be temporal or spatial")
    return BaseFlow(a[:, 0], state, str(path.resolve()), ro, interpolation)


def solve_bek(ro=-0.751, zmax=20.0, tol=1e-8, samples=2001):
    """diskBEK.m equations, continued from the analytic Ro=0 Ekman solution."""
    if not np.isfinite(ro) or not -1 <= ro <= 1:
        raise ValueError("BEK requires -1 <= ro <= 1")
    if not np.isfinite(zmax) or zmax <= 0 or not 0 < tol < 1:
        raise ValueError("Invalid zmax or BVP tolerance")
    if int(samples) != samples or samples < 4:
        raise ValueError("samples must be an integer >= 4")
    z = np.linspace(0, zmax, 201)
    e, s, c = np.exp(-z), np.sin(z), np.cos(z)
    U, V = -e*s, 1-e*c
    guess = np.array([1-e*(s+c), e*(s-c), U, e*(c+s), V])

    def bc(a, b):
        return np.array([a[0], a[2], a[4], b[2], b[4]-1])

    for r in np.linspace(0, ro, max(2, int(abs(ro)/0.1)+1)):
        co = 2-r-r*r

        def rhs(z, q):
            W, dU, U, dV, V = q
            return np.array([-2*U, r*(U*U+W*dU-V*V+1)-co*(V-1),
                             dU, r*(2*U*V+W*dV)+co*U, dV])

        sol = solve_bvp(rhs, bc, z, guess, tol=tol, max_nodes=50000)
        if not sol.success:
            raise RuntimeError(f"BEK BVP failed at Ro={r}: {sol.message}")
        z, guess = sol.x, sol.y
    z = np.linspace(0, zmax, int(samples))
    return BaseFlow(z, sol.sol(z), f"BEK(Ro={ro}, tol={tol})", ro)
