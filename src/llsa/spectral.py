"""Chebyshev coefficient basis and exponential mapping (paper §3.11–3.13)."""
from dataclasses import dataclass
from functools import lru_cache

import numpy as np


@dataclass(frozen=True)
class Grid:
    z: np.ndarray
    S: np.ndarray
    D: np.ndarray
    D2: np.ndarray

    @property
    def size(self):
        return len(self.z)


@lru_cache(maxsize=16)
def chebyshev(n=64, zmax=20.0, stretch=0.25):
    """N+1 nodes ordered far field -> wall, matching matrixcheby.m.

    S, D and D2 act on polynomial *coefficients*, not nodal values.
    stretch=0 selects a linear map. Cached arrays are read-only.
    """
    if isinstance(n, bool) or int(n) != n or n < 4:
        raise ValueError("n must be an integer >= 4")
    if not np.isfinite(zmax) or zmax <= 0 or not np.isfinite(stretch) or stretch < 0:
        raise ValueError("zmax must be positive; stretch must be nonnegative")
    n = int(n)
    y = np.cos(np.pi * np.arange(n + 1) / n)
    S = np.zeros((n + 1, n + 1))
    dy = np.zeros_like(S)
    dyy = np.zeros_like(S)
    S[:, 0], S[:, 1], dy[:, 1] = 1, y, 1
    for k in range(2, n + 1):
        S[:, k] = 2*y*S[:, k-1] - S[:, k-2]
        dy[:, k] = 2*S[:, k-1] + 2*y*dy[:, k-1] - dy[:, k-2]
        dyy[:, k] = 4*dy[:, k-1] + 2*y*dyy[:, k-1] - dyy[:, k-2]
    if stretch == 0:
        z = (y + 1)*zmax/2
        yp, ypp = np.full(n+1, 2/zmax), np.zeros(n+1)
    else:
        if stretch*zmax > 30:
            raise ValueError("stretch*zmax > 30 loses mapping precision")
        b = 2/np.expm1(-stretch*zmax)
        z = -np.log1p((y+1)/b)/stretch
        yp = -stretch*(y+1+b)
        ypp = -stretch*yp
    z[0], z[-1] = zmax, 0.0
    D = dy*yp[:, None]
    D2 = dyy*yp[:, None]**2 + dy*ypp[:, None]
    for a in (z, S, D, D2):
        a.setflags(write=False)
    return Grid(z, S, D, D2)
