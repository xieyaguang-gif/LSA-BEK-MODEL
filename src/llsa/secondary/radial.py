"""Shared, non-parabolized cylindrical radial Fourier equations.

q_j=(u_j,v_j,w_j,p_j), with physical velocities R*q[:3] and
pressure R**2*q[3].  The radial carrier is resolved, not factored out.
The phase convention is exp(i*j*(beta*theta-omega*t)).
"""
from dataclasses import dataclass

import numpy as np

from .operators import PierOperators
from .primary import pack, unpack


def real_blocks(block, harmonics, size):
    """Real representation of a conjugate-symmetric Fourier Jacobian."""
    H, s = harmonics, size
    out = np.zeros(((2*H+1)*s, (2*H+1)*s))
    for j in range(H+1):
        row = slice(0, s) if j == 0 else slice((2*j-1)*s, (2*j+1)*s)
        C = block(j, 0)
        out[row, :s] = C.real if j == 0 else np.vstack((C.real, C.imag))
        for k in range(1, H+1):
            Cp, Cm = block(j, k), block(j, -k)
            re, im = Cp+Cm, 1j*(Cp-Cm)
            out[row, (2*k-1)*s:(2*k+1)*s] = (np.hstack((re.real, im.real))
                if j == 0 else np.block([[re.real, im.real], [re.imag, im.imag]]))
    return out


@dataclass(frozen=True)
class RadialConfig:
    ro: float = -.751
    beta: float = 42.
    n: int = 24
    harmonics: int = 2
    omega: float | None = None  # inertial; None means disk-stationary
    nonlinear: bool = True

    def __post_init__(self):
        if not np.isfinite(self.ro) or self.ro == 0 or not np.isfinite(self.beta):
            raise ValueError("Finite nonzero Ro and finite beta required")
        if self.n < 8 or int(self.n) != self.n or self.harmonics < 1 or int(self.harmonics) != self.harmonics:
            raise ValueError("Integer n>=8 and harmonics>=1 required")
        if self.omega is not None and not np.isfinite(self.omega):
            raise ValueError("Finite real omega required")

    def operators(self, radius):
        return PierOperators(radius=radius, beta=self.beta, n=self.n, ro=self.ro)

    def frequency(self, op):
        return self.beta*op.disk_rotation if self.omega is None else self.omega


def radial_matrices(op, j, omega):
    """A*q + B*q_R + C*q_RR, including every radial scale derivative.

    Boundary rows in A enforce the existing wall/far-field closure. Their B,C
    rows are zero. C is singular: continuity and pressure are constraints.
    """
    R, m, s, I = op.radius, op.m, op.size, op.I
    A = op.linear(0, j*op.beta)-1j*j*omega/R*op.M
    B, C = np.zeros((s, s), complex), np.zeros((s, s), complex)
    for v in range(3):
        sl = slice(v*m, (v+1)*m)
        A[sl, sl] += np.diag(op.base[0]/R-1/R**3)
        B[sl, sl] = np.diag(op.base[0]-3/R**2)
        C[sl, sl] = -I/R
    A[:m, 3*m:] += 2*I/R
    A[3*m:, :m] += I/R
    B[:m, 3*m:] = I
    B[3*m:, :m] = I
    A[op.boundary_rows] = op.linear(0, j*op.beta)[op.boundary_rows]
    B[op.boundary_rows] = 0
    C[op.boundary_rows] = 0
    return A, B, C


def convection(op, modes, radial):
    """Exact Fourier convolution of nonlinear physical-velocity advection."""
    H = (len(modes)-1)//2
    q = modes.reshape(2*H+1, 4, op.m)
    qr = radial.reshape(q.shape)
    qz = q[:, :3]@op.grid.D.T
    out = np.zeros_like(q)
    for j in range(-H, H+1):
        for k in range(-H, H+1):
            l = j-k
            if abs(l) > H:
                continue
            a, b = q[l+H], q[k+H]
            out[j+H, :3] += (a[0]*(qr[k+H, :3]+b[:3]/op.radius)
                +1j*k*op.beta/op.radius*a[1]*b[:3]+a[2]*qz[k+H])
            out[j+H, 0] -= a[1]*b[1]/op.radius
            out[j+H, 1] += a[0]*b[1]/op.radius
    out = out.reshape(modes.shape)
    out[:, op.boundary_rows] = 0
    return out


def convection_blocks(op, modes, radial, j, k):
    """Frechet blocks dN_j/dq_k and dN_j/d(q_R)_k (signed harmonics)."""
    H, m, s, R = (len(modes)-1)//2, op.m, op.size, op.radius
    Q, T = np.zeros((s, s), complex), np.zeros((s, s), complex)
    l = j-k
    if abs(l) > H:
        return Q, T
    a = modes[l+H].reshape(4, m)
    ar = radial[l+H].reshape(4, m)
    az = a[:3]@op.grid.D.T
    adv = np.diag(a[0]/R+1j*k*op.beta/R*a[1])+a[2, :, None]*op.grid.D
    for v in range(3):
        sl = slice(v*m, (v+1)*m)
        Q[sl, sl] += adv
        Q[sl, :m] += np.diag(ar[v]+a[v]/R)
        Q[sl, m:2*m] += np.diag(1j*l*op.beta/R*a[v])
        Q[sl, 2*m:3*m] += np.diag(az[v])
        T[sl, sl] = np.diag(a[0])
    Q[:m, m:2*m] -= np.diag(2*a[1]/R)
    Q[m:2*m, :m] += np.diag(a[1]/R)
    Q[m:2*m, m:2*m] += np.diag(a[0]/R)
    Q[op.boundary_rows] = 0
    T[op.boundary_rows] = 0
    return Q, T


def pack_field(modes):
    return np.concatenate([pack(q) for q in modes])


def unpack_field(x, count, H, size):
    return np.array([unpack(v, H, size) for v in np.asarray(x).reshape(count, -1)])
