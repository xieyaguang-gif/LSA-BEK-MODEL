"""Accepted 725/H6 global field -> explicitly frozen, finite-height Floquet base.

The global file already uses inertial BEK (u,v,w)/R and p/R**2.  It stores
corrections to the laminar flow, with exp[i*h*(beta*theta-omega*t)].
No DNS disk-frame conversion, division by R, or frequency shift is appropriate.
"""
from __future__ import annotations

import os
for _key in ('OPENBLAS_NUM_THREADS', 'OMP_NUM_THREADS', 'MKL_NUM_THREADS',
             'NUMEXPR_NUM_THREADS', 'BLIS_NUM_THREADS', 'VECLIB_MAXIMUM_THREADS'):
    os.environ[_key] = '1'

import hashlib
import json
import sys
from functools import lru_cache
from pathlib import Path

import numpy as np
from scipy.interpolate import BarycentricInterpolator, CubicSpline
from scipy.optimize import brentq

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT/'src'))
from llsa.secondary.dns_floquet import DNSFrozenBase, FinitePierOperators

OUT = ROOT/'results/fullbvp_global_20261030/phaseE'
SOURCE = ROOT/'results/fullbvp_global_20261030/phaseC/attempts/harmonic_audit_be43e3a56f551a7f_try0/fields.npz'
SOURCE_SHA = '2dd79cc0d823ebc04fd0a9bd67a86c2a1e59df1f5f3302324401dfe7e8a2de1b'


def pair(z):
    return [float(np.real(z)), float(np.imag(z))]


def write_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_suffix(path.suffix+'.tmp')
    temp.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False)+'\n')
    temp.replace(path)


def source_xi(z):
    z = np.asarray(z)
    if np.any(np.isnan(z)) or np.any(z < 0):
        raise ValueError('Height must be nonnegative')
    def inverse(zz):
        if np.isinf(zz):
            return 1.
        target = (zz-2)/(zz+2)
        return brentq(lambda x: .6*x+.4*(x**3+.5*(1-x*x))-target, -1, 1)
    return np.array([inverse(zz) for zz in z.ravel()]).reshape(z.shape)


@lru_cache(maxsize=1)
def source():
    digest = hashlib.sha256(SOURCE.read_bytes()).hexdigest()
    if digest != SOURCE_SHA:
        raise ValueError('Accepted 725/H6 source hash changed')
    for name in ('summary.json', 'screen.json'):
        if not json.loads((SOURCE.parent/name).read_text())['accepted']:
            raise ValueError('Source is not accepted')
    with np.load(SOURCE) as data:
        if not data['accepted'] or int(data['H']) != 6 or float(data['Rout']) != 725.:
            raise ValueError('Wrong full BVP field')
        r, xi = data['radii'].copy(), data['xi'].copy()
        q = data['field'].reshape(len(r), 13, 4, len(xi)).copy()
        if not np.all(np.isfinite(q)) or not np.all(np.diff(r) > 0):
            raise ValueError('Invalid source array')
        if np.max(abs(q[:,:6]-q[:,7:][:,::-1].conj())) > 1e-12:
            raise ValueError('Source violates Fourier conjugacy')
        return r, xi, q, float(data['omega'])


def section(radius):
    """Local cubic interpolation; derivative is with respect to physical R."""
    r, xi, q, omega = source()
    if not r[0] <= radius <= r[-1]:
        raise ValueError('Radial extrapolation forbidden')
    i = int(np.clip(np.searchsorted(r, radius)-4, 0, len(r)-8))
    interp = CubicSpline(r[i:i+8], q[i:i+8], axis=0)
    return interp(radius), interp(radius, 1), xi, omega


def frozen_base(radius, *, n=32, H=6, zmax=30.):
    if int(H) != H or not 1 <= H <= 6:
        raise ValueError('Available base harmonics are H=1..6')
    q, dq, xi, omega = section(radius)
    probe = source_xi(np.array([1.3]))
    v, dv = [BarycentricInterpolator(xi, arr[7,1])(probe)[0] for arr in (q,dq)]
    if abs(v) < 1e-12:
        raise ValueError('Carrier phase is unresolved at z=1.3')
    logarithmic_derivative = dv/v
    alpha, growth = float(logarithmic_derivative.imag), float(logarithmic_derivative.real)
    op = FinitePierOperators(radius=radius, beta=68., ro=-1., n=n, zmax=zmax)
    values = BarycentricInterpolator(xi, q[6-H:7+H], axis=-1)(source_xi(op.grid.z))
    derivatives = BarycentricInterpolator(xi, dq[7,:3], axis=-1)(source_xi(op.grid.z))
    q1 = values[H+1,:3]
    inner = lambda a,b: np.einsum('vi,ij,vj->', a.conj(), op.grid.mass, b)
    projection = inner(q1,derivatives)/inner(q1,q1)
    frozen_error = derivatives-1j*alpha*q1
    derivative_defect = np.sqrt(max(0., inner(frozen_error,frozen_error).real)/
                                max(inner(derivatives,derivatives).real, 1e-30))
    metadata = dict(schema='FullBVPFrozen-v1', source=str(SOURCE), source_sha256=SOURCE_SHA,
        source_n=32, source_H=6, source_Rout=725., R=float(radius), beta=68., Ro=-1.,
        N=n, H=H, zmax=zmax, alpha=alpha, envelope_growth=growth,
        alpha_definition='Im[(d_R v_1)/v_1] at z=1.3; local cubic radial interpolation',
        probe_abs_q1=float(abs(v)), alpha_energy_projection=float(projection.imag),
        envelope_energy_projection=float(projection.real),
        physical_velocity_envelope_growth=growth+1/radius,
        omitted_radial_derivative_relative=float(derivative_defect),
        top_velocity_relative=float(np.max(abs(values[:,:3,-1]))/
                                    max(np.max(abs(values[:,:3])),1e-30)),
        equilibrium_verified=False,
        normalization='inertial BEK; corrections (u,v,w)/R and p/R**2; copied without rescaling',
        phase='exp[i h (alpha*(r-R)+68*theta-omega*t)]',
        primary_radial_derivative='i*h*alpha; envelope and shape gradients omitted',
        source_omega=omega)
    base = DNSFrozenBase(op, alpha, omega, values.reshape(2*H+1,op.size), metadata)
    metadata['continuity'] = base.continuity_defect()
    return base


if __name__ == '__main__':
    OUT.mkdir(parents=True, exist_ok=True)
    rows = [frozen_base(float(r)).metadata for r in range(560,601,5)]
    write_json(OUT/'adapter.json', rows)
    for row in rows:
        print(f"R={row['R']:g} alpha={row['alpha']:.7f} g={row['envelope_growth']:.7f} "
              f"|v1|={row['probe_abs_q1']:.6g} frozen_derivative_defect={row['omitted_radial_derivative_relative']:.3g}")
