"""Mesh/harmonic checks at the last accepted H4 upper point (supervisor-written).

Solves the same Rout with N40 (dr~0.25) at H=4 and N32 at H=5, transferring the H4 N32 upper point.
Outputs are written under phaseB/attempts/h4_check_* only.
"""
import argparse, json, resource, sys
from pathlib import Path
import numpy as np
HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import ro0751_phaseB_campaign as C  # noqa: E402
from ro0751_phaseB_h4_driver import H4Campaign  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--source', default='h4_branch_0023_try0')
    ap.add_argument('--which', choices=['N40', 'H5'], required=True)
    ap.add_argument('--budget', type=float, default=1800.)
    args = ap.parse_args()
    resource.setrlimit(resource.RLIMIT_AS, (200 * 1024**3, 200 * 1024**3))
    run = H4Campaign(C.OUT, args.budget, ['seed_0', 'branch_0027_try0', 'branch_0028_try0'])
    src = next(p for p in run.state['points'] if p['id'] == args.source)
    if args.which == 'N40':
        mapping = C.Mapping(40, 4, int(np.ceil((src['Rout']-400)/.25)))
        name = 'check_N40_dr025'
    else:
        mapping = C.Mapping(32, 5, run.mapping.intervals)
        name = 'check_H5'
    y = C.transfer(C.ROOT/src['fields'], mapping)
    y[-1] = src['Rout']
    _, d = run.attempt(mapping, y, run.qref(y, mapping), 'Rout', src['Rout'], name, 'mesh_check')
    keep = {k: d.get(k) for k in ('accepted', 'A1', 'Rout', 'omega', 'highest_to_fundamental', 'residual_inf', 'n', 'H', 'intervals', 'seconds', 'failure')}
    print(json.dumps(keep))


if __name__ == '__main__':
    main()
