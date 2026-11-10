"""Track the two spatial branches away from the neutral saddle at Ro=-1.

OPENBLAS_NUM_THREADS=1 PYTHONPATH=src .venv/bin/python examples/absolute_pinch_check.py
This is a finite-resolution, finite-path numerical pinch diagnostic.
"""
import json
from pathlib import Path

import numpy as np
from scipy import linalg
from scipy.sparse.linalg import LinearOperator, eigs

from absolute_critical import NeutralCalculator


def near_roots(problem, omega, sigma, count=4):
    """Shift-invert the quadratic companion problem using one original-size LU."""
    C, A1, A2 = problem.A0-omega*problem.B, problem.A1, problem.A2
    m = len(C)
    lu = linalg.lu_factor(C+sigma*A1+sigma*sigma*A2, check_finite=False)
    H = A1+sigma*A2

    def apply(v):
        v1, v2 = v[:m], v[m:]
        x = -linalg.lu_solve(lu, H@v1+A2@v2, check_finite=False)
        return np.r_[x, v1+sigma*x]

    op = LinearOperator((2*m, 2*m), matvec=apply, dtype=complex)
    mu, q = eigs(op, k=count, which="LM", tol=1e-10, v0=np.ones(2*m, complex))
    alpha = sigma+1/mu
    q = q[:m]
    q /= linalg.norm(q, axis=0)
    errors = linalg.norm(C@q+(A1@q)*alpha+(A2@q)*alpha**2, axis=0)/(
        linalg.norm(C)+abs(alpha)*linalg.norm(A1)+abs(alpha)**2*linalg.norm(A2))
    if not np.all(np.isfinite(errors)) or max(errors) > 1e-8:
        raise RuntimeError(f"Spatial residual check failed: {errors}")
    return alpha, errors


def main():
    calc = NeutralCalculator(n=128, zmax=40, stretch=.1)
    critical = calc.critical(68)
    a0, w0 = complex(*critical["alpha"]), complex(*critical["omega"])
    problem = calc.problem(critical["re"], 68)
    path, previous = [], None
    for eta in np.geomspace(1e-7, .08, 100):
        if previous is None:
            roots, errors = near_roots(problem, w0+1j*eta, a0)
            pair = roots[np.argsort(abs(roots-a0))[:2]]
        else:
            pair, errors = [], []
            for last in previous:
                roots, residuals = near_roots(problem, w0+1j*eta, last)
                k = np.argmin(abs(roots-last))
                pair.append(roots[k])
                errors.append(residuals[k])
            pair = np.array(pair)
        if abs(pair[0]-pair[1]) < 1e-7:
            raise RuntimeError("Tracked branches collapsed onto one numerical root")
        previous = pair
        path.append(dict(eta=float(eta), roots=[[a.real, a.imag] for a in pair],
                         max_residual=float(max(errors))))
    result = dict(critical=critical, path=path,
                  opposite_half_planes_at_end=bool(previous[0].imag*previous[1].imag < 0),
                  interpretation="Numerical branch-tracking evidence, not a global search over all saddles")
    out = Path("results/absolute_ro_minus1/pinch_reproduced.json")
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(result, indent=2)+"\n")
    print(json.dumps({k: v for k, v in result.items() if k != "path"}, indent=2))
    print(f"Saved {out}")


if __name__ == "__main__":
    main()
