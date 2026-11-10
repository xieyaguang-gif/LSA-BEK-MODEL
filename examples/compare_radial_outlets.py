"""Compare accepted energy profiles at a common sponge onset (NumPy only)."""
import argparse
import json
from pathlib import Path

import numpy as np


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('short_case', type=Path)
    parser.add_argument('long_case', type=Path)
    parser.add_argument('--lower-re', type=float, default=350.)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    records = [json.loads((path/'metadata.json').read_text())
               for path in (args.short_case, args.long_case)]
    if any(r['status'] != 'converged_discrete_equations' for r in records):
        parser.error('Both cases must have converged to their physical equations')
    if records[0]['config'] != records[1]['config']:
        parser.error('Cases must have the same physical and spectral configuration')
    stop = records[0]['sponge_start']
    if stop != records[1]['sponge_start'] or args.lower_re >= stop:
        parser.error('Require a common sponge onset above lower-re')
    a, b = [np.loadtxt(path/'energy.csv', delimiter=',')
            for path in (args.short_case, args.long_case)]
    if b[-1, 0] <= a[-1, 0] or max(a[0, 0], b[0, 0]) > args.lower_re:
        parser.error('Require a longer second domain and a shared comparison interval')
    mask = (a[:, 0] >= args.lower_re) & (a[:, 0] <= stop)
    relative = np.interp(a[mask, 0], b[:, 0], b[:, 1])/a[mask, 1]-1
    result = dict(short_case=str(args.short_case), long_case=str(args.long_case),
                  physical_interval=[args.lower_re, stop],
                  interpolation='linear in Re onto short-domain nodes',
                  max_relative_E1_difference=float(np.max(abs(relative))),
                  short_outlet=dict(Re=float(a[-1, 0]), E1=float(a[-1, 1])),
                  long_outlet=dict(Re=float(b[-1, 0]), E1=float(b[-1, 1])),
                  caveat='Check sponge strength, width, mean damping and inlet data in case summaries; algebraic convergence is not grid convergence.')
    args.output.write_text(json.dumps(result, indent=2)+'\n')
    print(json.dumps(result, indent=2))


if __name__ == '__main__':
    main()
