#!/usr/bin/env python3
"""Reproducible stationary beta42 nonlinear PSE / AF0.1 comparison run.

Run with ./run.sh beta42-spatial --output results/my-spatial-march
No DNS growth rates enter the equations: only one upstream amplitude is used.
"""
import argparse
import json
from pathlib import Path
import time

import numpy as np
from scipy.interpolate import BarycentricInterpolator
from scipy.optimize import brentq

from llsa.secondary.spatial import SpatialPSE


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    reference = Path(__file__).resolve().parents[3]/'results/beta42_nonlinear_spatial/inputs/af01_semtex_turn4_vtheta.csv'
    p.add_argument('--dns', type=Path, default=reference, help='DNS CSV used only for the upstream amplitude')
    p.add_argument('--output', type=Path, required=True)
    p.add_argument('--n', type=int, default=48)
    p.add_argument('--harmonics', type=int, default=5)
    p.add_argument('--dr', type=float, default=6., help='Step in r_delta, not Re; smaller is not necessarily stable for PSE')
    p.add_argument('--re-start', type=float, default=320.)
    p.add_argument('--re-end', type=float, default=440.)
    p.add_argument('--amplitude-factor', type=float, default=1.)
    p.add_argument('--linear', action='store_true')
    p.add_argument('--no-fields', action='store_true', help='retain CSV/metadata only')
    a = p.parse_args(argv)
    if a.amplitude_factor <= 0 or a.dr <= 0 or a.re_end <= a.re_start:
        p.error('Positive amplitude, step and interval required')
    a.output.mkdir(parents=True, exist_ok=True)
    ro, beta = -.751, 42.
    dns = np.loadtxt(a.dns, delimiter=',')
    amplitude = np.interp(a.re_start, dns[:, 1], dns[:, 4])*a.amplitude_factor
    solver = SpatialPSE(ro=ro, beta=beta, n=a.n, harmonics=a.harmonics, nonlinear=not a.linear)
    op = solver.operators(a.re_start/abs(ro))
    z = 1.24/np.sqrt(op.disk_rotation)
    grid = op.grid
    t = (z-grid.a)/(z+grid.a)
    xi = brentq(lambda x: grid.b*x+(1-grid.b)*(x**3+grid.c*(1-x*x))-t, -1, 1)
    interpolation = BarycentricInterpolator(grid.xi, np.eye(op.m))(xi)
    previous = solver.initialize(op.radius, amplitude, z)
    older = None
    rows, all_modes = [], []
    started = time.monotonic()
    failure = None
    radii = op.radius+np.arange(int((a.re_end-a.re_start)/abs(ro)/a.dr)+1)*a.dr
    for i, radius in enumerate(radii):
        if i:
            try:
                new = solver.advance(previous, radius, older, tol=1e-12, check_jacobian=(i == 1))
            except (RuntimeError, np.linalg.LinAlgError) as exc:
                failure = str(exc)
                print(failure, flush=True)
                break
            older, previous = previous, new
        modes = previous.modes.reshape(2*a.harmonics+1, 4, op.m)
        amp = np.sqrt(2)*np.abs(modes[a.harmonics+1:, 1]@interpolation)/op.disk_rotation
        vel = modes[a.harmonics+1:, :3]
        energy = radius**2*np.einsum('hvi,ij,hvj->h', vel.conj(), grid.mass, vel).real
        mean = modes[a.harmonics, 0].real
        rows.append([abs(ro)*radius, radius, previous.alpha.real, -previous.alpha.imag,
                     *amp, *energy, previous.residual, previous.iterations,
                     np.min((op.base[0]+mean)[(grid.z > .1)&(grid.z < 10)])])
        all_modes.append(previous.modes)
        if i % 5 == 0:
            print(f'Re={abs(ro)*radius:.3f} A1={amp[0]:.6g} carrier_g={-previous.alpha.imag:.6g} residual={previous.residual:.2g}', flush=True)
    result = np.asarray(rows)
    growth = (np.gradient(np.log(result[:, 4]), result[:, 1], edge_order=min(2,len(result)-1))
              if len(result) > 1 else np.full(1, np.nan))
    result = np.column_stack((result, growth))
    columns = (['Re', 'r_delta', 'carrier_alpha_real', 'carrier_growth']+
               [f'A{j}_vtheta_over_disk_speed' for j in range(1, a.harmonics+1)]+
               [f'E{j}' for j in range(1, a.harmonics+1)]+
               ['residual', 'iterations', 'min_mean_radial_F_z01_10', 'growth_A1'])
    np.savetxt(a.output/'march.csv', result, delimiter=',', header=','.join(columns), comments='')
    if not a.no_fields:
        np.savez_compressed(a.output/'fields.npz', modes=np.asarray(all_modes), radius=result[:, 1],
                            alpha=result[:, 2]-1j*result[:, 3], z=grid.z, xi=grid.xi,
                            ro=ro, beta=beta, n=a.n, harmonics=a.harmonics, z_probe=z,
                            disk_rotation=op.disk_rotation)
    metadata = dict(parameters={k:str(v) if isinstance(v,Path) else v for k,v in vars(a).items()},
                    ro=ro, beta=beta, omega_inertial=beta*op.disk_rotation,
                    omega_disk_frame=0., z_probe=z, inlet_amplitude=amplitude,
                    status='complete' if failure is None else 'stopped', failure=failure,
                    last_Re=float(result[-1,0]), elapsed_seconds=time.monotonic()-started,
                    max_residual=float(np.max(result[:,4+2*a.harmonics])),
                    approximation='Nonlinear PSE; omit slow radial pressure derivative and second envelope derivative; adaptive carrier; BDF2')
    (a.output/'metadata.json').write_text(json.dumps(metadata, indent=2)+'\n')
    print(json.dumps(metadata, indent=2), flush=True)
    return 0 if failure is None else 2


if __name__ == '__main__':
    raise SystemExit(main())
