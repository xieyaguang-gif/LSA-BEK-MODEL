"""Recompute selected entries of Pier (2003) Tables 1–3, never treating mismatch as pass.

Example:
OPENBLAS_NUM_THREADS=1 PYTHONPATH=src .venv/bin/python examples/pier2003_benchmark.py \
    --n 30 40 --harmonics 1 2 3 4 5 --secondary-harmonics 1 2 3 4 5 --mean-mode parallel
"""
import argparse
import json
from pathlib import Path
from time import perf_counter

from llsa.secondary import PierOperators, WaveSolver, FloquetProblem, secondary_saddle
from llsa.secondary.validation import compare_tables, primary_absolute_threshold


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--n', type=int, nargs='+', default=[30, 40])
    parser.add_argument('--harmonics', type=int, nargs='+', default=[1, 3, 5])
    parser.add_argument('--secondary-harmonics', type=int, nargs='+', default=[1, 3, 5])
    parser.add_argument('--mean-mode', choices=['parallel', 'cylindrical'], default='cylindrical')
    parser.add_argument('--output', type=Path, default=Path('results/pier2003/benchmark'))
    parser.add_argument('--fail-on-mismatch', action='store_true')
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    start = perf_counter()
    results = dict(mean_mode=args.mean_mode, primary_neutral=[], comparisons=[])
    for n in args.n:
        results['primary_neutral'].append(primary_absolute_threshold(n=n))
        op = PierOperators(n=n, mean_mode=args.mean_mode)
        solver = WaveSolver(op, 1)
        wave = solver.solve(solver.bootstrap(), frequency=50.5)
        for H in range(1, max(args.harmonics)+1):
            if H > 1:
                wave = WaveSolver(op, H).solve(wave, frequency=50.5)
            if H not in args.harmonics:
                continue
            wave.save(args.output/f'wave_n{n}_h{H}.npz')
            for K in args.secondary_harmonics:
                problem = FloquetProblem(wave, harmonics=K)
                secondary = secondary_saddle(problem)
                comparison = compare_tables(wave, (problem, secondary))
                results['comparisons'].append(comparison)
                results['elapsed_seconds'] = perf_counter()-start
                print(f"N={n} H={H} K={K}: alpha={wave.alpha:.8f}; "
                      f"omega_hat={secondary.omega_hat:.6f}; "
                      f"table3_passed={comparison['table3']['passed']}", flush=True)
                (args.output/'comparison.json').write_text(json.dumps(results, indent=2)+'\n')
    passed = all(c['table2']['passed'] is True and c['table3']['passed'] is True
                 for c in results['comparisons'])
    results['all_published_comparisons_passed'] = passed
    (args.output/'comparison.json').write_text(json.dumps(results, indent=2)+'\n')
    print(f"Published-table comparison passed: {passed}")
    return 2 if args.fail_on_mismatch and not passed else 0


if __name__ == '__main__':
    raise SystemExit(main())
