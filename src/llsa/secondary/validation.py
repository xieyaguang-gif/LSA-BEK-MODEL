"""Comparison with independently transcribed tables, with explicit pass/fail."""
import json
from pathlib import Path

import numpy as np
from scipy import linalg, optimize

from .operators import PierOperators


def primary_absolute_threshold(n=40, a=2, b=.6, c=.5, beta=68):
    def evaluate(x):
        ar, ai, radius = x
        op = PierOperators(radius=radius, beta=beta, n=n, a=a, b=b, c=c)
        alpha = complex(ar, ai)
        A, B = op.linear(alpha, beta), 1j*op.M
        vals, left, right = linalg.eig(A, B, left=True)
        good = np.isfinite(vals)
        vals, left, right = vals[good]*radius, left[:, good], right[:, good]
        k = np.argmin(abs(vals-50.5))
        q, p = right[:, k], left[:, k]
        h = 1e-5
        Ap = (op.linear(alpha+h, beta)-op.linear(alpha-h, beta))/(2*h)
        derivative = radius*np.vdot(p, Ap@q)/np.vdot(p, B@q)
        return np.array([derivative.real, derivative.imag, vals[k].imag]), vals[k]

    root = optimize.root(lambda x: evaluate(x)[0], [.21692, -.12180, 507.4], tol=1e-9)
    error, omega = evaluate(root.x)
    if max(abs(error)) > 1e-6:
        raise RuntimeError(f"Primary neutral saddle did not converge: {error}")
    return dict(n=n, a=a, b=b, c=c, beta=beta, radius=float(root.x[2]),
                alpha=root.x[:2].tolist(), omega=[omega.real, omega.imag],
                residual=error.tolist())


def compare_tables(wave, secondary=None, reference_path=None):
    if reference_path is None:
        reference_path = Path(__file__).with_name('references.json')
    references = json.loads(Path(reference_path).read_text())
    op = wave.operators
    common = dict(ro=op.ro, re=op.re, disk_rotation=op.disk_rotation, external_rotation=op.external_rotation,
                  radius=op.radius, beta=op.beta, n=op.m-1, H=wave.harmonics,
                  alpha=wave.alpha, omega=wave.omega, energies=wave.energies().tolist(),
                  mean_mode=op.mean_mode, nonlinear_residual=wave.residual)
    q = wave.modes.reshape(2*wave.harmonics+1, 4, op.m)
    divergence = []
    for j in range(-wave.harmonics, wave.harmonics+1):
        u, v, w, _ = q[j+wave.harmonics]
        div = (1j*j*wave.alpha+1/op.radius)*u+1j*j*op.beta/op.radius*v+op.grid.D@w
        divergence.append(float(np.max(abs(div[1:-1]))*op.radius))
    common['frozen_cylindrical_divergence_by_harmonic'] = divergence
    common['axial_mean_velocity_at_infinity'] = float(q[wave.harmonics, 2, -1].real*op.radius)
    mean_divergence = divergence[wave.harmonics]
    mean_far = abs(common['axial_mean_velocity_at_infinity'])
    common['paper_mean_constraints'] = dict(
        mean_continuity_at_nodes=mean_divergence,
        mean_far_axial_velocity=common['axial_mean_velocity_at_infinity'],
        tolerance=1e-7,
        known_mismatch=bool(mean_divergence > 1e-7 or mean_far > 1e-7),
        scope='Necessary mean constraints only. Table 2 passes compare alpha/energies, not the full paper PDE; off-node checks are also required.')
    eligible = (op.ro == -1 and abs(op.radius-510) < 1e-10 and abs(op.beta-68) < 1e-10 and
                abs(wave.omega-50.5) < 1e-8 and
                (op.grid.a, op.grid.b, op.grid.c) == (2., .6, .5))
    primary = [r for r in references['table2'] if r['n'] == op.m-1 and r['H'] == wave.harmonics]
    if eligible and primary:
        ref = primary[0]
        errors = wave.energies()-ref['energies']
        # Printed E has two decimals; use explicit scientific comparison bounds.
        energy_tolerance = .01+.001*np.asarray(ref['energies'])
        common['table2'] = dict(reference=ref, alpha_error=wave.alpha-ref['alpha'],
                                energy_errors=errors.tolist(), alpha_tolerance=3e-5,
                                energy_tolerance=energy_tolerance.tolist(),
                                passed=bool(abs(wave.alpha-ref['alpha']) < 3e-5 and
                                            np.all(abs(errors) < energy_tolerance)))
    else:
        common['table2'] = dict(passed=None, reason="No matching published case at these parameters")
    if secondary is not None:
        problem, result = secondary
        common['secondary_wall_condition'] = problem.wall_condition
        refs = [r for r in references['table3'] if r['n'] == op.m-1 and
                r['H'] == wave.harmonics and r['K'] == problem.K]
        if eligible and abs(problem.beta_hat-20) < 1e-10 and refs:
            ref = refs[0]
            diff = result.omega_hat-complex(*ref['omega_hat'])
            common['table3'] = dict(reference=ref, omega_hat=[result.omega_hat.real, result.omega_hat.imag],
                                    alpha_hat=[result.alpha_hat.real, result.alpha_hat.imag],
                                    frequency_error=[diff.real, diff.imag],
                                    growth_relative_error=abs(diff.imag/ref['omega_hat'][1]),
                                    component_tolerance=.02,
                                    passed=bool(result.converged and abs(diff.real) < .02 and abs(diff.imag) < .02))
        else:
            common['table3'] = dict(passed=None, reason="No matching published secondary case")
    return common
