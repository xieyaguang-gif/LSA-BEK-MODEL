"""Ro=-1 absolute neutral point on the classical rotating-disk saddle branch.

From llsa_python:
OPENBLAS_NUM_THREADS=1 PYTHONPATH=src .venv/bin/python examples/absolute_critical.py

Solve Re(omega_alpha)=Im(omega_alpha)=Im(omega)=0 for alpha.real,
alpha.imag and Re. This finds a neutral saddle; use absolute_pinch_check.py
to check its spatial branches. beta is the integer azimuthal mode number.
"""
import argparse
import json
from pathlib import Path

import numpy as np
from scipy import linalg, optimize

from llsa import LocalProblem, Parameters, solve_bek, temporal_near


class NeutralCalculator:
    def __init__(self, n=192, zmax=40.0, stretch=0.1):
        self.n, self.zmax, self.stretch = n, zmax, stretch
        self.flow = solve_bek(-1, zmax=zmax, tol=1e-10, samples=16001)

    def problem(self, re, beta):
        return LocalProblem(self.flow, Parameters(re=re, ro=-1, beta=beta, n=self.n,
                            zmax=self.zmax, stretch=self.stretch, boundary="matlab"))

    def evaluate(self, x, beta):
        ar, ai, re = x
        alpha = complex(ar, ai)
        problem = self.problem(re, beta)
        spec = temporal_near(problem, alpha, target=.035+1e-5j, count=3)
        k = spec.select(target=.035)
        omega, q = spec.values[k], spec.vectors[:, k]
        bq = problem.B @ q
        M = problem.matrix(alpha)-omega*problem.B
        # M^H p=0, (Bq)^H p=1. The border removes the nullspace ambiguity;
        # it does not match a separately ordered adjoint spectrum.
        bordered = np.block([[M.conj().T, q[:, None]],
                             [bq.conj()[None, :], np.zeros((1, 1))]])
        left = linalg.solve(bordered, np.r_[np.zeros(len(q)), 1], check_finite=False)[:-1]
        first = np.vdot(left, (problem.A1+2*alpha*problem.A2)@q)/np.vdot(left, bq)
        return np.array([first.real, first.imag, omega.imag]), omega

    def critical(self, beta=68, seed=(.218, .122, 507.3)):
        root = optimize.root(lambda x: self.evaluate(x, beta)[0], seed, tol=1e-8)
        error, omega = self.evaluate(root.x, beta)
        if not np.all(np.isfinite(error)) or max(abs(error)) > 1e-8:
            raise RuntimeError(f"Neutral root failed: {root.message}; error={error}")
        return dict(ro=-1, beta=float(beta), re=float(root.x[2]),
                    alpha=root.x[:2].tolist(), omega=[float(omega.real), float(omega.imag)],
                    n=self.n, zmax=self.zmax, stretch=self.stretch, boundary="matlab",
                    neutral_equation_residual=error.tolist(), evaluations=root.nfev,
                    convention="Raw solver alpha and omega; arrays=[real,imag]",
                    pinch_verified=False)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--beta", type=float, default=68)
    parser.add_argument("--n", type=int, default=192)
    parser.add_argument("--zmax", type=float, default=40)
    parser.add_argument("--stretch", type=float, default=.1)
    parser.add_argument("--continuous-beta", action="store_true",
                        help="minimize on 66<beta<70; this relaxes integer periodicity")
    parser.add_argument("--output", type=Path, default=Path("results/absolute_ro_minus1/neutral.json"))
    args = parser.parse_args()
    calc = NeutralCalculator(args.n, args.zmax, args.stretch)
    beta = args.beta
    if args.continuous_beta:
        root = optimize.minimize_scalar(lambda b: calc.critical(b)["re"], bounds=(66, 70),
                                        method="bounded", options={"xatol": 1e-5})
        if not root.success:
            raise RuntimeError(root.message)
        beta = root.x
    result = calc.critical(beta)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2)+"\n")
    print(json.dumps(result, indent=2))
    print(f"Saved {args.output}")


if __name__ == "__main__":
    main()
