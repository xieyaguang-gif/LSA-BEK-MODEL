"""Reusable parameter studies, with explicit branch targets and bracketed roots."""
from dataclasses import replace

from scipy.optimize import brentq

from .operators import LocalProblem
from .solvers import saddle, spatial


def spatial_branch(baseflow, parameters, reynolds, omega, alpha_target):
    """Continue a spatial branch in Re; growth column is -Im(alpha)."""
    rows = []
    for re in reynolds:
        p = LocalProblem(baseflow, replace(parameters, re=float(re)))
        s = spatial(p, omega, real_bounds=(0.06, 1.0))
        k = s.select(alpha_target, imag_limit=float("inf"))
        alpha_target = complex(s.values[k])
        rows.append(dict(re=float(re), alpha=alpha_target, growth=-alpha_target.imag,
                         residual=float(s.residuals[k])))
    return rows


def spatial_neutral(baseflow, parameters, bracket, omega, alpha_target, xtol=1e-6):
    """Find -Im(alpha)=0 in a Re bracket for the branch nearest a supplied seed.

    Inspect spatial_branch first to ensure that the bracket contains one continuous
    branch. Fixed target avoids solver-call-order-dependent state inside brentq.
    """
    def growth(re):
        return spatial_branch(baseflow, parameters, [re], omega, alpha_target)[0]["growth"]
    return brentq(growth, *bracket, xtol=xtol)


def saddle_branch(baseflow, parameters, reynolds, alpha, omega_target=None, **kwargs):
    rows = []
    for re in reynolds:
        p = LocalProblem(baseflow, replace(parameters, re=float(re)))
        result = saddle(p, alpha, omega_target, **kwargs)
        if not result.converged:
            raise RuntimeError(f"Saddle search did not converge at Re={re}")
        rows.append((float(re), result))
        alpha, omega_target = result.alpha, result.omega
    return rows
