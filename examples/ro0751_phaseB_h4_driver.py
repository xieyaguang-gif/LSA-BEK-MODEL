"""H4 continuation of the Ro=-0.751, beta=85 subcritical branch (supervisor-written driver).

Starts from the accepted H3 points branch_0027/branch_0028 (and seed_0) transferred to H=4.
Writes only inside the Phase B directory, with names prefixed 'h4_' and state in campaign_h4.json.
The H3 campaign.json is read-only for this driver.
"""
import argparse, json, resource, sys, time
from pathlib import Path
import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import ro0751_phaseB_campaign as C  # noqa: E402


class H4Campaign(C.Campaign):
    def __init__(self, out, budget, start_ids):
        super().__init__(out, budget)
        h3 = {p['id']: p for p in self.state['points']}
        seeds = [h3[i] for i in start_ids]
        self.mapping = C.Mapping(32, 4, self.mapping.intervals)
        self.metric = C.Metric(self.mapping)
        if (self.out / 'campaign_h4.json').exists():
            self.state = C.read(self.out / 'campaign_h4.json')
            return
        self.state = dict(status='running', stage='branch_fold', points=[dict(p) for p in seeds],
                          attempts=[], targets={}, folds={}, refinements=[], stop=None,
                          next_step=0.02, source='H3 points ' + ','.join(start_ids) + ' transferred to H4')

    def save(self):
        self.state['elapsed_current_invocation'] = time.monotonic() - self.start
        C.write(self.out / 'campaign_h4.json', self.state)

    def attempt(self, mapping, y, qref, mode, target, name, role, tangent=None, weak=False):
        return super().attempt(mapping, y, qref, mode, target, 'h4_' + name, role, tangent, weak)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--budget', type=float, default=1200.)
    ap.add_argument('--upper', action='store_true', help='continue past the fold to the upper branch')
    args = ap.parse_args()
    resource.setrlimit(resource.RLIMIT_AS, (200 * 1024**3, 200 * 1024**3))
    run = H4Campaign(C.OUT, args.budget, ['seed_0', 'branch_0027_try0', 'branch_0028_try0'])
    try:
        run.branch(until_fold=not args.upper)
    except C.ScientificStop as exc:
        print('stopped:', exc)
    print(json.dumps({k: run.state.get(k) for k in ('stage', 'stop', 'next_step')}))


if __name__ == '__main__':
    main()
