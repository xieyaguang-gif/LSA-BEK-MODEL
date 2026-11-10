"""Run from llsa_python: OPENBLAS_NUM_THREADS=1 PYTHONPATH=src .venv/bin/python examples/parameter_study.py"""
from pathlib import Path
import numpy as np

from llsa import LocalProblem, Parameters, solve_bek, track_temporal
from llsa.workflows import saddle_branch, spatial_branch

flow = solve_bek(-0.751)
params = Parameters(re=500, ro=-0.751, beta=85, n=64)
problem = LocalProblem(flow, params)
rows = track_temporal(problem, np.linspace(.2, .35, 16), omega_target=.02+.005j)
out = Path("results")
out.mkdir(exist_ok=True)
np.savetxt(out/"temporal_branch.csv", [[a.real, w.real, w.imag, e] for a, w, e in rows],
           delimiter=",", header="alpha,omega_real,omega_imag,residual")

# Seeds select a branch; inspect the spectrum before assigning type I/II/III.
for row in spatial_branch(flow, params, [498, 500, 502], omega=0, alpha_target=.3-.03j):
    print(row)
for re, result in saddle_branch(flow, params, [498, 500, 502], .26+.15j, omega_target=.02+.005j):
    print(re, result.alpha, result.omega)
