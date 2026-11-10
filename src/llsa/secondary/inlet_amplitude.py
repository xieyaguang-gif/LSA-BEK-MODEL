"""A logarithmic inlet-amplitude path with exact prescribed endpoints."""
from dataclasses import dataclass

import numpy as np


@dataclass
class InletAmplitudePath:
    source: np.ndarray
    target: np.ndarray
    ratio: float
    fit_error: float

    @classmethod
    def fit(cls, source, target, harmonic, velocity_rows, expected_ratio=None):
        """Fit only prescribed fundamental velocity, excluding pressure and walls.

        Other prescribed inlet harmonics are interpolated between their actual
        endpoints. The fitted fundamental must have the same shape and phase.
        """
        source, target = np.asarray(source, complex), np.asarray(target, complex)
        if source.shape != target.shape or source.ndim != 2:
            raise ValueError('Inlet arrays must have the same two-dimensional shape')
        if not np.all(np.isfinite(source)) or not np.all(np.isfinite(target)):
            raise ValueError('Inlet arrays must be finite')
        a, b = source[harmonic, velocity_rows], target[harmonic, velocity_rows]
        if np.linalg.norm(a) == 0 or np.linalg.norm(b) == 0:
            raise ValueError('Nonzero fundamental inlet velocities are required')
        fitted = np.vdot(a, b)/np.vdot(a, a)
        ratio = float(fitted.real) if expected_ratio is None else float(expected_ratio)
        if not np.isfinite(ratio) or ratio <= 0 or abs(np.log(ratio)) < 1e-12:
            raise ValueError('The inlet amplitude ratio must be positive and different from one')
        error = float(np.linalg.norm(b-ratio*a)/np.linalg.norm(b))
        if error > 1e-9:
            raise ValueError(f'Fundamental inlet shape/phase or amplitude ratio mismatch: {error:.3e}')
        return cls(source.copy(), target.copy(), ratio, error)

    @property
    def end(self):
        return float(np.log(self.ratio))

    def at(self, coordinate):
        if coordinate == 0:
            return self.source.copy()
        if coordinate == self.end:
            return self.target.copy()
        fraction = coordinate/self.end
        if not np.isfinite(fraction) or not -1e-12 <= fraction <= 1+1e-12:
            raise ValueError('Coordinate must lie between the inlet path endpoints')
        weight = np.expm1(coordinate)/np.expm1(self.end)
        return (1-weight)*self.source+weight*self.target

    def advance(self, coordinate, step):
        if not np.isfinite(step) or step <= 0:
            raise ValueError('A positive finite log step is required')
        if step >= abs(self.end-coordinate):
            return self.end
        return float(coordinate+np.sign(self.end)*step)

    def nominal_af(self, coordinate, source_af, target_af):
        """AF labels interpolate geometrically; measured inlet ratios are separate."""
        return float(source_af*np.exp(coordinate/self.end*np.log(target_af/source_af)))
