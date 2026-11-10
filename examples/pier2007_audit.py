"""Revisit the secondary growth discrepancy using wall and resolution checks.

The 2007 paper uses the same inertial-frame Floquet convention as Pier (2003).
This audit therefore compares raw complex frequencies without an empirical shift.
"""
import argparse
import json
from pathlib import Path

from llsa.secondary import PrimaryWave, PierOperators, WaveSolver, FloquetProblem, secondary_saddle
from llsa.secondary.diagnostics import secondary_kinematic_defects
from llsa.secondary.validation import compare_tables


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--wave', type=Path, required=True)
    parser.add_argument('--n', type=int, nargs='+', default=[30, 40, 60])
    parser.add_argument('--secondary-harmonics', type=int, default=5)
    parser.add_argument('--output', type=Path, default=Path('results/pier2003/audit2007.json'))
    args = parser.parse_args()
    seed = PrimaryWave.load(args.wave)
    old = seed.operators
    records = []
    for n in args.n:
        op = PierOperators(old.radius, old.beta, n, old.grid.a, old.grid.b, old.grid.c, old.mean_mode)
        wave = WaveSolver(op, seed.harmonics).solve(seed, frequency=seed.omega)
        for boundary in ('pressure', 'continuity'):
            problem = FloquetProblem(wave, harmonics=args.secondary_harmonics, wall_condition=boundary)
            saddle = secondary_saddle(problem)
            record = compare_tables(wave, (problem, saddle))
            record['secondary'] = dict(alpha_hat=[saddle.alpha_hat.real, saddle.alpha_hat.imag],
                                       omega_hat=[saddle.omega_hat.real, saddle.omega_hat.imag],
                                       group_velocity=abs(saddle.group_velocity),
                                       residual=saddle.residual, converged=saddle.converged)
            record['kinematics'] = secondary_kinematic_defects(problem, saddle.alpha_hat, saddle.vector)
            records.append(record)
            print(f"N={n} H={seed.harmonics} K={args.secondary_harmonics} wall={boundary}: "
                  f"omega_hat={saddle.omega_hat:.8f}, "
                  f"wall divergence={record['kinematics']['wall_continuity']:.3e}", flush=True)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(dict(
        status='secondary published growth rate still not reproduced',
        paper_2003_doi='10.1017/S0022112003004981',
        paper_2007_doi='10.1007/s10665-006-9095-5',
        records=records), indent=2)+'\n')
    return 0 if all(r['secondary']['converged'] for r in records) else 2


if __name__ == '__main__':
    raise SystemExit(main())
