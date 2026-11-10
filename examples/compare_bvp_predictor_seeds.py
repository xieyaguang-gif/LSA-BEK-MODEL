"""Compare full-BVP residuals of initial guesses before an expensive LU solve."""
import argparse
import json
from pathlib import Path

import numpy as np

from llsa.secondary.global_radial import GlobalRadial, OutletSponge
from llsa.secondary.radial import pack_field
from llsa.secondary.radial_continuation import (
    fixed_domain_seed, load_seed, secant_seed, transport_seed,
)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--seed', type=Path, required=True)
    p.add_argument('--old-start', type=float, required=True)
    p.add_argument('--target', type=float, required=True)
    p.add_argument('--previous-seed', type=Path, action='append', default=[])
    p.add_argument('--previous-start', type=float, action='append', default=[])
    p.add_argument('--dr', type=float, default=.5)
    p.add_argument('--width', type=float, default=40.)
    p.add_argument('--strength', type=float, default=10.)
    p.add_argument('--output', type=Path, required=True)
    a = p.parse_args()
    if (len(a.previous_seed) != len(a.previous_start) or a.target <= a.old_start
            or min(a.dr, a.width) <= 0 or any(v >= a.old_start for v in a.previous_start)):
        p.error('Require matching previous seeds/starts, target>old-start, and positive dr/width')
    c, r, q = load_seed(a.seed)
    end = (a.target+a.width)/abs(c.ro)
    rr = np.linspace(r[0], end, int(np.ceil((end-r[0])/a.dr))+1)
    problem = GlobalRadial(rr, c, outlet_sponge=OutletSponge(
        a.target/abs(c.ro), a.width/abs(c.ro), a.strength))
    anchor = max(r[0], (a.old_start-30)/abs(c.ro))
    mapped = transport_seed(r, q, rr, c, a.old_start/abs(c.ro), mapping='smooth', anchor=anchor)
    candidates = {'transport': mapped, 'fixed_physical': fixed_domain_seed(r, q, rr, c)}
    for path, start in zip(a.previous_seed, a.previous_start):
        pc, pr, pq = load_seed(path)
        if pc != c:
            p.error('Previous seeds must have the same configuration')
        old = transport_seed(pr, pq, rr, c, start/abs(c.ro), mapping='smooth', anchor=anchor)
        for phase in (False, True):
            candidates[f'secant_{start}_phase{phase}'] = secant_seed(mapped, old,
                (a.target-a.old_start)/(a.old_start-start), c, r[0], phase_aligned=phase)
    rows = []
    for name, guess in candidates.items():
        f = problem.evaluate(pack_field(guess), q[0])
        scale = float(np.max(abs(guess)))
        absolute = abs(f).reshape(len(rr), problem.width)/scale
        i, k = np.unravel_index(absolute.argmax(), absolute.shape)
        m, s = problem.ops[0].m, problem.s
        rows.append(dict(predictor=name, residual_inf_scaled=float(absolute[i, k]),
            residual_l2=float(np.linalg.norm(f)), max_field=scale,
            max_at_Re=float(rr[i]*abs(c.ro)), packed_harmonic_part=int(k//s),
            component=['u','v','w','continuity'][k%s//m],
            normal_z=float(problem.ops[0].grid.z[k%m]),
            old_physical_region_max=float(absolute[rr*abs(c.ro) <= a.old_start].max())))
    a.output.parent.mkdir(parents=True, exist_ok=True)
    a.output.write_text(json.dumps(rows, indent=2)+'\n')
    print(json.dumps(rows, indent=2))


if __name__ == '__main__':
    main()
