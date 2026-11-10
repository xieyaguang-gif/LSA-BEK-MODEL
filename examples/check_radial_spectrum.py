"""Inspect windowed radial spectral tails; this is not a grid-convergence test."""
import argparse
import json
from pathlib import Path

import numpy as np

from llsa.secondary.radial_continuation import load_seed


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('field',type=Path)
    p.add_argument('--re-min',type=float,default=500.)
    p.add_argument('--re-max',type=float,default=600.)
    p.add_argument('--output',type=Path,required=True)
    args = p.parse_args()
    c,r,q = load_seed(args.field)
    if args.re_min >= args.re_max or args.re_min < abs(c.ro)*r[0] or args.re_max > abs(c.ro)*r[-1]+1e-8:
        p.error('Require an increasing interval inside the saved domain')
    use = (abs(c.ro)*r>=args.re_min) & (abs(c.ro)*r<=args.re_max)
    x = r[use]
    if len(x)<16 or not np.allclose(np.diff(x),np.diff(x)[0],rtol=1e-8,atol=1e-10):
        p.error('Require at least 16 uniformly spaced radial points')
    h = float(x[1]-x[0])
    op = c.operators(r[0])
    field = q[use].reshape(len(x),2*c.harmonics+1,4,op.m)
    window = np.hanning(len(x))
    k = 2*np.pi*np.fft.fftfreq(len(x),d=h)
    nyquist = np.pi/h
    records = []
    for j in range(c.harmonics+1):
        velocity = x[:,None,None]*field[:,c.harmonics+j,:3]
        transformed = np.fft.fft(window[:,None,None]*velocity,axis=0)
        power = np.einsum('kvi,ij,kvj->k',transformed.conj(),op.grid.mass,transformed).real
        power = np.maximum(power,0.)
        denominator = max(float(np.sum(power)),1e-300)
        records.append(dict(harmonic=j,peak_radial_wavenumber=float(k[np.argmax(power)]),
            power_fraction_above_06_nyquist=float(np.sum(power[abs(k)>.6*nyquist])/denominator),
            power_fraction_above_08_nyquist=float(np.sum(power[abs(k)>.8*nyquist])/denominator)))
    mean_advection = op.base[0]+field[:,c.harmonics,0].real-3/x[:,None]**2
    peclet = abs(mean_advection)*h*x[:,None]
    with np.load(args.field) as saved:
        converged = bool(saved['converged']) if 'converged' in saved else None
        release = json.loads(str(saved['release_json'])) if 'release_json' in saved else None
    report = dict(source=str(args.field),source_converged=converged,release=release,
        Re_interval=[float(abs(c.ro)*x[0]),float(abs(c.ro)*x[-1])],points=len(x),delta_R=h,
        nyquist_radial_wavenumber=nyquist,harmonics=records,
        maximum_mean_radial_cell_Peclet=float(np.max(peclet)),
        Peclet_definition='abs(F_base+u0-3/R**2)*delta_R/(1/R)',
        processing='Hann-windowed FFT of physical perturbation velocities R*q; Chebyshev mass matrix in z; no resampling.',
        limitation='Spectral leakage and real fine structure can enter the tail. Small tails do not exclude spurious Jacobian modes or establish grid convergence. Auxiliary damping fields are not target physical data.')
    args.output.write_text(json.dumps(report,indent=2)+'\n')
    print(json.dumps(report,indent=2))


if __name__=='__main__':
    main()
