"""Rossby continuation of nontrivial saturated primary waves."""
import numpy as np

from .operators import PierOperators
from .primary import WaveSolver
from .discretization import rotation_rates


def continue_ro(seed, values, *, energy=None, frequency=None, wavenumber=None,
                max_step=.025, min_step=1e-5, fixed_re=False, callback=None):
    """Return waves at requested Ro values, with adaptive internal Ro steps.

    Default holds E1 fixed, solving alpha and omega. This follows a selected
    branch, not necessarily the naturally selected primary or most unstable one.
    No continuation across zero shear is permitted.
    """
    values = list(values)
    if not values or not 0 < min_step <= max_step or not np.isfinite(max_step):
        raise ValueError("Require Ro targets and 0 < min_step <= max_step")
    for ro in values:
        rotation_rates(ro)
        if ro*seed.operators.ro <= 0:
            raise ValueError("Continue within one nonzero-Ro half-interval; do not cross Ro=0")
    if sum(v is not None for v in (energy, frequency, wavenumber)) > 1:
        raise ValueError("Choose at most one continuation constraint")
    if energy is frequency is wavenumber is None:
        energy = float(seed.energies()[0])
    constraint = dict(energy=energy) if energy is not None else (
        dict(frequency=frequency) if frequency is not None else dict(wavenumber=wavenumber))
    # Reject an input with stale/wrong-model residual metadata.
    original = seed.operators
    residual = np.max(abs(WaveSolver(original, seed.harmonics).residual(seed.modes, seed.alpha, seed.omega)))
    if not np.isfinite(residual) or residual > 1e-7:
        raise ValueError("Rossby continuation requires a converged source wave")
    current = WaveSolver(original, seed.harmonics).solve(seed, **constraint)
    output, path = [], []
    for target in values:
        step = max_step
        while abs(target-current.operators.ro) > 1e-12:
            before = current.operators.ro
            trial = before+np.sign(target-before)*min(step, abs(target-before))
            radius = original.re/abs(trial) if fixed_re else original.radius
            op = PierOperators(radius, original.beta, original.m-1, original.grid.a,
                               original.grid.b, original.grid.c, original.mean_mode, ro=trial)
            try:
                candidate = WaveSolver(op, current.harmonics).solve(current, **constraint)
            except (RuntimeError, np.linalg.LinAlgError):
                step *= .5
                if step < min_step:
                    raise RuntimeError(f"Ro continuation stopped at {before:.12g}; cannot reach {target:.12g}")
                continue
            current = candidate
            row = dict(ro=op.ro, radius=op.radius, re=op.re, alpha=current.alpha,
                       omega=current.omega, energy=float(current.energies()[0]), residual=current.residual)
            path.append(row)
            if callback:
                callback(current, row)
            step = min(max_step, step*1.5)
        output.append(current)
    return output, path
