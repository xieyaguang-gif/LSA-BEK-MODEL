"""Check physical constraints beyond the rows retained in the matrix pencil."""
import numpy as np
from scipy.interpolate import BarycentricInterpolator


def secondary_kinematic_defects(problem, alpha_hat, vector):
    """Continuity on nodes, at the wall, and between collocation nodes.

    Normalize all velocities by the largest nodal harmonic velocity magnitude.
    Thus the reported divergence has units of inverse nondimensional length;
    it is independent of the arbitrary eigenvector normalization. The sampling
    is an additional diagnostic, not an error bound on the continuous solution.
    """
    op, K = problem.wave.operators, problem.K
    q = np.asarray(vector).reshape(2*K+1, 4, op.m)[:, :3].copy()
    scale = np.max(abs(q))
    if not np.isfinite(scale) or scale == 0:
        raise ValueError("A finite nonzero velocity eigenfunction is required")
    q /= scale
    ns = np.arange(-K, K+1)
    kr = alpha_hat+ns*problem.wave.alpha
    kt = problem.beta_hat+ns*op.beta
    divergence = ((1j*kr[:, None]+1/op.radius)*q[:, 0]+
                  1j*kt[:, None]/op.radius*q[:, 1]+q[:, 2]@op.grid.D.T)
    # Independent interpolation and differentiation in xi, evaluated away from
    # the original extrema. This catches defects suppressed at collocation nodes.
    xi = np.linspace(-1, 1, 4*(op.m-1)+2)[:-1]
    interpolation = BarycentricInterpolator(op.grid.xi, q, axis=2)
    u = interpolation(xi)
    dx = interpolation.derivative(xi)
    a, b, c = op.grid.a, op.grid.b, op.grid.c
    t = b*xi+(1-b)*(xi**3+c*(1-xi**2))
    tp = b+(1-b)*(3*xi**2-2*c*xi)
    dz = dx[:, 2]*(1-t)**2/(2*a*tp)
    fine = ((1j*kr[:, None]+1/op.radius)*u[:, 0]+
            1j*kt[:, None]/op.radius*u[:, 1]+dz)
    return dict(
        wall_condition=problem.wall_condition,
        normalization="max nodal harmonic velocity magnitude = 1",
        harmonics=ns.tolist(),
        wall_velocity=float(np.max(abs(q[:, :, 0]))),
        wall_continuity=float(np.max(abs(divergence[:, 0]))),
        wall_continuity_by_harmonic=np.abs(divergence[:, 0]).tolist(),
        interior_continuity=float(np.max(abs(divergence[:, 1:-1]))),
        sampled_continuity=float(np.max(abs(fine))),
        wall_continuity_passed=bool(np.max(abs(divergence[:, 0])) < 1e-7),
    )
