"""Reproducible short-domain checks; not validation of saturation to Re=700.

Run: OPENBLAS_NUM_THREADS=1 PYTHONPATH=src .venv/bin/python examples/validate_radial_two_layer.py
Only final curves and a compact metrics file are retained.
"""
from dataclasses import replace
import json
from pathlib import Path
from time import perf_counter

import numpy as np

from llsa.secondary.radial import RadialConfig
from llsa.secondary.global_radial import GlobalRadial
from llsa.secondary.oneway import SpectralOneWay, causal_projector


def main():
    root = Path(__file__).resolve().parents[1]
    out = root/'results/radial_two_layer_validation'
    out.mkdir(parents=True, exist_ok=True)
    started = perf_counter()
    config = RadialConfig(n=16, harmonics=2)
    records, curves = [], {}

    def run(label, points, *, end=326., amplitude=1e-5, c=config, outlet_shift=0j):
        r = np.linspace(320., end, points)/abs(c.ro)
        one = SpectralOneWay(c, continuation_steps=64)
        solution, diagnostics = one.march(r, amplitude=amplitude)
        p = causal_projector(c.operators(r[-1]), 1, c.frequency(c.operators(r[-1])))
        valid = np.flatnonzero(~p.upstream)
        alpha0 = complex(*diagnostics['inlet_alpha'])
        ix = valid[np.argmin(abs(-1j*p.values[valid]-alpha0))]
        alpha_out = -1j*p.values[ix]+outlet_shift
        global_solution = GlobalRadial(r, c, outlet_alpha=alpha_out).solve(solution.modes[0], initial=solution.modes)
        eo, eg = solution.energies(c), global_solution.energies(c)
        idx = int(np.argmin(abs(abs(c.ro)*r-323.)))
        record = dict(case=label, n=c.n, harmonics=c.harmonics, points=points, re_end=end,
            amplitude=amplitude, nonlinear=c.nonlinear, step_r=float(r[1]-r[0]),
            global_residual=global_solution.residual, oneway_residual=solution.residual,
            global_final_energies=eg[-1].tolist(), oneway_final_energies=eo[-1].tolist(),
            common_re=float(abs(c.ro)*r[idx]), global_common_E1=float(eg[idx, 0]), oneway_common_E1=float(eo[idx, 0]),
            inlet_alpha=diagnostics['inlet_alpha'], outlet_alpha=[alpha_out.real, alpha_out.imag])
        records.append(record)
        curves[label] = np.column_stack((abs(c.ro)*r, eg, eo))
        print(json.dumps(record), flush=True)
        return global_solution, solution

    for points in (17, 33, 65):
        run(f'linear_grid_{points}', points, c=replace(config, nonlinear=False))
    run('linear_extended_outlet', 129, end=332., c=replace(config, nonlinear=False))
    run('linear_robin_perturbed', 65, c=replace(config, nonlinear=False), outlet_shift=.02)
    gl, ow = run('nonlinear_reference', 65, amplitude=.001)
    run('nonlinear_harmonics3', 65, amplitude=.001, c=replace(config, harmonics=3))
    run('nonlinear_wallnormal24', 65, amplitude=.001, c=replace(config, n=24))
    # Final nonlinear reference fields; diagnostic trials retain curves only.
    np.savez_compressed(out/'nonlinear_reference.npz', radii=gl.radii, global_modes=gl.modes,
        oneway_modes=ow.modes, ro=config.ro, beta=config.beta, n=config.n, harmonics=config.harmonics)
    for label, curve in curves.items():
        H = (curve.shape[1]-1)//2
        np.savetxt(out/f'{label}.csv', curve, delimiter=',', header='Re,'+
            ','.join([f'global_E{j}' for j in range(1, H+1)]+[f'oneway_E{j}' for j in range(1, H+1)]))
    report = dict(status='short-domain numerical checks only', elapsed_seconds=perf_counter()-started,
        limitations=['No saturation or Re=700 validation', 'Finite radial outlet is approximate',
                     'Dense spectral projection, not recursive OWNS', 'Global mean BVP coupled to one-way waves'],
        cases=records)
    (out/'metrics.json').write_text(json.dumps(report, indent=2)+'\n')
    print(f'Saved {out}', flush=True)


if __name__ == '__main__':
    main()
