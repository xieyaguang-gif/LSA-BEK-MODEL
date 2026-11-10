"""Pier (2003) finite-amplitude waves and secondary Floquet instability.

Experimental: quantitative secondary Table 3 validation has not passed.
See docs/PIER2003.md for the unresolved mean-flow closure and discrepancies.
This formulation uses an inertial frame and BEK reference-rotation time units (disk units at Ro=-1), unlike the
legacy BEK primary solver. Internal velocity and frequency scaling is by radius R.
"""
from .operators import PierOperators
from .primary import PrimaryWave, WaveSolver
from .floquet import (FloquetProblem, secondary_spectrum, secondary_derivatives,
                      secondary_saddle, secondary_spatial_near, secondary_pinch)
from .diagnostics import secondary_kinematic_defects
from .continuation import continue_ro

__all__ = ["PierOperators", "PrimaryWave", "WaveSolver", "FloquetProblem",
           "secondary_spectrum", "secondary_derivatives", "secondary_saddle",
           "secondary_spatial_near", "secondary_pinch", "secondary_kinematic_defects", "continue_ro"]
