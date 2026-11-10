"""Compare initial residuals of predictors on the same damping-release path."""
import argparse
import json
from pathlib import Path

import numpy as np

from llsa.secondary.global_radial import BlendedSponge, GlobalRadial, OutletSponge
from llsa.secondary.radial import pack_field
from llsa.secondary.radial_continuation import load_seed, secant_seed


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('previous', type=Path)
    p.add_argument('current', type=Path)
    p.add_argument('--fraction', type=float, required=True)
    p.add_argument('--output', type=Path, required=True)
    args = p.parse_args()
    config, r, q = load_seed(args.current)
    old_config, old_r, old_q = load_seed(args.previous)
    if old_config != config or not np.array_equal(r, old_r):
        p.error('Seeds must have identical configuration and grid')
    releases = []
    for path in [args.previous, args.current]:
        with np.load(path) as data:
            if not bool(data['converged']):
                p.error('Seeds must be converged')
            releases.append(json.loads(str(data['release_json'])))
    old_fraction = releases[0].pop('fraction')
    fraction = releases[1].pop('fraction')
    if releases[0] != releases[1] or not old_fraction < fraction < args.fraction <= 1:
        p.error('Require two increasing fractions on one path and a later target in (0,1]')
    settings = releases[1]
    ratio = (args.fraction-fraction)/(fraction-old_fraction)
    old = OutletSponge(settings['old_start']/abs(config.ro), settings['width']/abs(config.ro),
                       settings['strength'], settings['damp_mean'])
    new = OutletSponge(settings['target']/abs(config.ro), old.width, old.strength, old.damp_mean)
    problem = GlobalRadial(r, config, outlet_alpha=complex(settings['outlet_alpha']),
                           outlet_sponge=BlendedSponge(old, new, args.fraction))
    scale = float(np.max(abs(q)))
    answer = dict(previous=str(args.previous), current=str(args.current),
                  target_fraction=args.fraction, common_residual_scale=scale, predictors={})
    for method in ['none', 'linear', 'phase']:
        guess = q if method == 'none' else secant_seed(q, old_q, ratio, config, r[0], method=='phase')
        f = problem.evaluate(pack_field(guess), q[0]).reshape(len(r), 2*config.harmonics+1, 4, config.n+1)/scale
        answer['predictors'][method] = dict(residual_inf=float(np.max(abs(f))),
            continuity_interior_inf=float(np.max(abs(f[1:-1, :, 3, 1:-1]))))
    args.output.write_text(json.dumps(answer, indent=2)+'\n')
    print(json.dumps(answer, indent=2))


if __name__ == '__main__':
    main()
