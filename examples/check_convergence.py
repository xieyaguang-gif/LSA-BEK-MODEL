"""Reproduce docs/convergence.json (outputs to results, preserves the audit record)."""
import json
from pathlib import Path
from time import perf_counter

from llsa import LocalProblem, Parameters, saddle, solve_bek

flow = solve_bek(-1)
rows = []
for n in (32, 48, 64, 96, 128):
    problem = LocalProblem(flow, Parameters(re=530, ro=-1, beta=67, n=n))
    start = perf_counter()
    result = saddle(problem, .213+.123j, .0311+.00035j)
    row = dict(n=n, alpha=[result.alpha.real, result.alpha.imag],
               omega=[result.omega.real, result.omega.imag],
               converged=result.converged, group_velocity_abs=abs(result.group_velocity),
               seconds=perf_counter()-start)
    print(row)
    rows.append(row)
Path("results").mkdir(exist_ok=True)
Path("results/convergence.json").write_text(json.dumps(rows, indent=2)+"\n")
