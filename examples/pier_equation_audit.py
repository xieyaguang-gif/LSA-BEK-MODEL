"""Audit paper equations, full-NS differentiation, and primary boundary sensitivity.

Diagnostic variants are temporary and do not replace or overwrite saved waves.
Run from the project root with OPENBLAS_NUM_THREADS=1 PYTHONPATH=src.
"""
import argparse
import hashlib
import json
from pathlib import Path

import numpy as np

from llsa.secondary import PrimaryWave, PierOperators, WaveSolver, FloquetProblem, secondary_saddle
from llsa.secondary.equation_audit import (independent_laminar_check, isolated_coefficient_checks,
                                          floquet_full_ns_check, primary_equation_defects)


class DiagnosticPrimaryOperators(PierOperators):
    def __init__(self, *args, primary_wall='continuity', literal_mean=False, **kwargs):
        super().__init__(*args, **kwargs)
        self.primary_wall, self.literal_mean = primary_wall, literal_mean

    def linear(self, k, ell, mean_closure=True):
        A = super().linear(k, ell, mean_closure)
        m = self.m
        if self.primary_wall == 'continuity' and (k != 0 or ell != 0):
            A[3*m] = 0
            A[3*m, 2*m:3*m] = self.grid.D[0]
        if self.literal_mean and mean_closure and k == 0 and ell == 0:
            # Restore the paper's stated w0(infinity)=0, retaining cylindrical
            # mean continuity. This can reveal unresolved collocation modes.
            A[3*m-1] = 0
            A[3*m-1, 3*m-1] = 1
        return A


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--wave', type=Path, default=Path('results/pier2003/wave_n60_h5.npz'))
    parser.add_argument('--primary-wall-grids', type=int, nargs='+', default=[40, 60])
    parser.add_argument('--probe-literal-mean', action='store_true')
    parser.add_argument('--output', type=Path, default=Path('results/pier_equation_audit/audit.json'))
    args = parser.parse_args(argv)
    wave = PrimaryWave.load(args.wave)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    result = dict(source_wave=str(args.wave), source_sha256=hashlib.sha256(args.wave.read_bytes()).hexdigest(),
                  source_parameters=dict(R=wave.operators.radius, beta=wave.operators.beta, omega=wave.omega,
                                         alpha=wave.alpha, H=wave.harmonics, mean_mode=wave.operators.mean_mode),
                  paper_doi='10.1017/S0022112003004981', published_table3_verified=False,
                  full_paper_model_verified=False, checks={}, primary_variants=[])

    def save():
        args.output.write_text(json.dumps(result, indent=2)+'\n')

    checks = [('independent_laminar', independent_laminar_check),
              ('isolated_coefficients', isolated_coefficient_checks),
              ('full_ns_floquet', lambda: floquet_full_ns_check(wave, harmonics=5)),
              ('current_primary_paper_pde', lambda: primary_equation_defects(wave))]
    for name, function in checks:
        result['checks'][name] = function()
        print(name, json.dumps({k:v for k,v in result['checks'][name].items() if k != 'records'}), flush=True)
        save()
    op = wave.operators
    for n in args.primary_wall_grids:
        choices = [('continuity_wall_parallel_mean', 'parallel', 'continuity', False)]
        if args.probe_literal_mean:
            choices += [('cylindrical_free_mean_far_boundary', 'cylindrical', 'pressure', False),
                        ('literal_paper_mean_boundaries', 'cylindrical', 'pressure', True)]
        for name, mean_mode, wall, literal in choices:
            row = dict(name=name, n=n, mean_mode=mean_mode, primary_wall=wall,
                       literal_mean_far_boundary=literal,
                       status='diagnostic only; source wave is unchanged')
            try:
                changed = DiagnosticPrimaryOperators(radius=op.radius, beta=op.beta, n=n,
                    a=op.grid.a, b=op.grid.b, c=op.grid.c, mean_mode=mean_mode,
                    primary_wall=wall, literal_mean=literal)
                new = WaveSolver(changed, wave.harmonics).solve(wave, frequency=wave.omega, max_iter=25)
                row.update(alpha=new.alpha, energies=new.energies().tolist(), newton_residual=new.residual,
                           physical_defects=primary_equation_defects(new))
                if not literal:
                    saddle = secondary_saddle(FloquetProblem(new, beta_hat=20, harmonics=5, wall_condition='continuity'),
                                              alpha_hat=-.00163221046-.03698086207j, target=8.6067+.8562j)
                    row['secondary'] = dict(alpha_hat=[saddle.alpha_hat.real, saddle.alpha_hat.imag],
                                            omega_hat=[saddle.omega_hat.real, saddle.omega_hat.imag],
                                            group_velocity=abs(saddle.group_velocity), converged=saddle.converged)
                else:
                    row['secondary'] = None
                    row['note'] = 'A small nodal Newton residual alone cannot validate this literal-boundary mean solution; inspect off-node continuity before using it.'
            except (ValueError, RuntimeError, np.linalg.LinAlgError) as exc:
                row.update(solve_failed=True, error=str(exc))
            result['primary_variants'].append(row)
            print('variant', json.dumps(row), flush=True)
            save()
    result['no_coefficient_mismatch_detected'] = all(result['checks'][key]['passed'] for key in
        ('independent_laminar', 'isolated_coefficients', 'full_ns_floquet'))
    result['conclusion'] = ('Bulk coefficient checks and primary physical constraints are separate. '
                           'Inspect both the nodal and off-node mean continuity and far boundary; '
                           'these checks do not certify the full paper model or establish an error in the published parameter.')
    save()
    print('Saved', args.output)
    return 0 if result['no_coefficient_mismatch_detected'] else 2


if __name__ == '__main__':
    raise SystemExit(main())
