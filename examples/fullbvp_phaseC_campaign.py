#!/usr/bin/env python3
"""Unlaunched Phase C: strict upper branch, Rout control, adaptive harmonics.

Run only through commands_C.sh after explicit campaign launch authorization.
Accepted checkpoints are immutable commits. Request-keyed attempt reuse recovers
the crash window between fields.npz and campaign.json without duplicating points.
"""
from __future__ import annotations

import os
for _key in ('OPENBLAS_NUM_THREADS', 'OMP_NUM_THREADS', 'MKL_NUM_THREADS',
             'NUMEXPR_NUM_THREADS', 'VECLIB_MAXIMUM_THREADS', 'BLIS_NUM_THREADS'):
    os.environ[_key] = '1'

import argparse
import fcntl
import signal
import time
import traceback
from pathlib import Path

import numpy as np

from fullbvp_phaseC_solver import (DEFAULT_OUT, Mapping, ROOT, SCREEN, SOURCE_ID,
    STRICT, CoherentMetric, correct, fingerprint, mapped_transfer, phase_align,
    qref_for, read, screen, sha, write)

TARGETS = [720., 725., 730., 740., 750., 765., 780., 800.]


class ScientificStop(RuntimeError):
    pass


class CampaignInterrupted(BaseException):
    """User/process cancellation bypasses ordinary failed-Newton retries."""


def source_identity():
    directory = ROOT/'results/fullbvp_global_20261030/phaseB'
    raw = next(p for p in read(directory/'campaign.json')['points'] if p['id'] == SOURCE_ID)
    accepted = next(p for p in read(directory/'accepted_points.json')['points'] if p['id'] == SOURCE_ID)
    audited = next(p for p in read(directory/'audited_points.json')['points'] if p['id'] == SOURCE_ID)
    summary = read(directory/'summary.json')
    if summary['strict_high_amplitude_point']['id'] != SOURCE_ID:
        raise ValueError('Audit strict-high representative is not first-pass point #36')
    expected = dict(A1=.2257879, Rout=718.931903, omega=52.710009,
                    front_fig8=588.3564, g_q_570_600=.1246564)
    for key, target in expected.items():
        if abs(audited[key]-target) > (6e-5 if key == 'front_fig8' else 6e-7):
            raise ValueError(f'Source identity mismatch: {key}')
    for record in (raw, accepted):
        if not record['accepted'] or not record.get('upper_branch'):
            raise ValueError('Source is not an accepted first-traversal upper point')
        for key in ('Rout', 'omega', 'A1'):
            if abs(record[key]-audited[key]) > 1e-11:
                raise ValueError('Accepted and audited source records disagree')
    fields = ROOT/raw['fields']
    with np.load(fields) as saved:
        if (int(saved['n']) != 32 or len(saved['radii']) != 801
                or int(saved['H']) != 3 or not bool(saved['accepted'])):
            raise ValueError('Source checkpoint is not the audited N32/H3/M800 root')
        if abs(float(saved['Rout'])-raw['Rout']) > 1e-10:
            raise ValueError('Source checkpoint Rout mismatch')
        if abs(float(saved['omega'])-raw['omega']) > 1e-10:
            raise ValueError('Source checkpoint frequency mismatch')
    diagnostic = read(directory/'diagnostics'/f'{SOURCE_ID}.json')
    digest = sha(fields)
    if diagnostic['sha256'] != digest:
        raise ValueError('Checkpoint bytes differ from audited diagnostic source')
    residual = read(ROOT/raw['summary'])
    if not residual['accepted'] or residual['independent']['relative_field_inf'] >= 1e-10:
        raise ValueError('Source independent residual is not strict')
    return dict(id=SOURCE_ID, fields=str(fields), fields_sha256=digest,
                raw_record=raw, audited_record=audited,
                summary_sha256=sha(ROOT/raw['summary']),
                diagnostic_sha256=sha(directory/'diagnostics'/f'{SOURCE_ID}.json'),
                selection='first traversal strict upper #36; terminal weak #81 excluded',
                H3_to_H4='requires same-Rout strict unforced correction before continuation')


class Exclusive:
    def __init__(self, out):
        self.out = Path(out)
        self.stream = None

    def __enter__(self):
        self.out.mkdir(parents=True, exist_ok=True)
        self.stream = (self.out/'C.lock').open('a')
        try:
            fcntl.flock(self.stream, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError:
            self.stream.close()
            raise RuntimeError('Another Phase C process holds the exclusive campaign lock')
        return self

    def __exit__(self, *_):
        fcntl.flock(self.stream, fcntl.LOCK_UN)
        self.stream.close()


def clear_markers(out):
    for name in ('C_DONE.json', 'C_FAILED.json'):
        (Path(out)/name).unlink(missing_ok=True)


def finish_marker(out, success, payload):
    """An interrupted or failed attempt cannot inherit a stale success marker."""
    other = Path(out)/('C_FAILED.json' if success else 'C_DONE.json')
    other.unlink(missing_ok=True)
    write(Path(out)/('C_DONE.json' if success else 'C_FAILED.json'), payload)


def configuration(targets, budget, minimum_step, maximum_step):
    if not targets or any(b <= a for a, b in zip(targets[:-1], targets[1:])):
        raise ValueError('Targets must be a nonempty strictly increasing sequence')
    if targets[0] <= 718.931903 or targets[-1] >= 1000:
        raise ValueError('Targets must lie above the source Rout and below 1000')
    if not 0 < minimum_step <= maximum_step or budget <= 0:
        raise ValueError('Positive budget and 0 < min_step <= max_step required')
    code = {str(p.relative_to(ROOT)): sha(p)
            for p in sorted((ROOT/'examples').glob('fullbvp_phaseC_*.py'))}
    return dict(schema=1, targets=targets, n=32, intervals=800, baseline_H=4,
                maximum_H=8, promotion_ratio=1e-3, trustworthy_ratio=1e-2,
                budget_per_point=budget, minimum_step=minimum_step,
                maximum_step=maximum_step, max_local_arc_steps=6,
                CPU_threads=1, GPU=False, strict=STRICT, screening=SCREEN,
                metric=dict(reference_R=720., field_scale=.1, omega_scale=1.,
                Rout_scale=10., coherent_velocity_only=True),
                equations=dict(Rin=400, Ro=-1, beta=68, outer_strength=28,
                               outer_width=40, outer_start='Rout-100'), code=code,
                equation_source_hashes={str(p.relative_to(ROOT)): sha(p) for p in (
                    ROOT/'examples/fullbvp_phaseB_solver.py',
                    ROOT/'examples/fullbvp_phaseB_upper_refinement.py',
                    ROOT/'examples/appelquist2016_branch_continuation.py',
                    ROOT/'examples/appelquist2015_hopf_common.py',
                    ROOT/'src/llsa/secondary/radial.py',
                    ROOT/'src/llsa/secondary/global_radial.py')})


class Campaign:
    def __init__(self, out, config, source):
        self.out, self.config, self.source = Path(out), config, source
        immutable_source = dict(id=source['id'], fields_sha256=source['fields_sha256'],
            scalars={k: source['raw_record'][k] for k in ('Rout', 'omega', 'A1', 'n', 'H', 'intervals')})
        self.identity = fingerprint(dict(config=config, source=immutable_source))
        statefile = self.out/'campaign.json'
        if statefile.exists():
            self.state = read(statefile)
            if self.state['fingerprint'] != self.identity:
                raise ValueError('Resume refused: source/config/code fingerprint changed')
            self.verify_commits()
        else:
            self.state = dict(schema=1, fingerprint=self.identity, config=config,
                              source=source, status='prepared', points=[], attempts=[],
                              refinements=[], fallback_events=[], targets={str(t):
                              dict(status='pending', Rout=t) for t in config['targets']})
            self.save()
        self.budget = config['budget_per_point']

    def verify_commits(self):
        seen = set()
        for point in self.state['points']:
            if point['id'] in seen:
                raise ValueError('Duplicate committed point id')
            seen.add(point['id'])
            if sha(ROOT/point['fields']) != point['fields_sha256']:
                raise ValueError('Committed checkpoint hash mismatch')
            numerical = read(ROOT/point['summary'])
            branch = read((ROOT/point['summary']).parent/'screen.json')
            if (not numerical['accepted'] or not branch['accepted']
                    or point['highest_to_fundamental'] > self.config['trustworthy_ratio']):
                raise ValueError('Resume ledger contains untrusted point')

    def save(self):
        write(self.out/'campaign.json', self.state)
        write(self.out/'accepted_points.json', dict(fingerprint=self.identity,
              points=self.state['points']))

    def attempt(self, mapping, y, mode, target, role, tangent=None):
        metric = CoherentMetric(mapping)
        request = dict(campaign=self.identity, n=mapping.n, H=mapping.H,
                       intervals=mapping.intervals, mode=mode, target=float(target),
                       role=role, predictor_sha256=__import__('hashlib').sha256(y.tobytes()).hexdigest(),
                       tangent_sha256=None if tangent is None else
                       __import__('hashlib').sha256(tangent.tobytes()).hexdigest())
        key = fingerprint(request)
        folderbase = self.out/'attempts'/f'{role}_{key[:16]}'
        resume = None
        index = 0
        while True:
            folder = folderbase.with_name(folderbase.name+f'_try{index}')
            if not folder.exists():
                break
            context = read(folder/'request.json') if (folder/'request.json').exists() else None
            if context is None:
                # A kill may occur between mkdir and the atomic request write.
                # Keep every byte as evidence and allocate a fresh verified try.
                orphan = dict(path=str(folder.resolve()), reason='unverified_request_reservation_preserved')
                records = self.state.setdefault('orphan_attempts', [])
                if orphan not in records:
                    records.append(orphan)
                    self.save()
                index += 1
                continue
            if context != request:
                raise ValueError('Attempt identity missing or mismatched; refuse unsafe reuse')
            unfinished = folder/'newton'
            if unfinished.exists() and ((unfinished/'summary.json').exists()
                                       or (folder/'summary.json').exists()):
                for artifact in unfinished.iterdir():
                    os.replace(artifact, folder/artifact.name)
                unfinished.rmdir()
            if (folder/'summary.json').exists():
                row = read(folder/'summary.json')
                if row['accepted']:
                    if not (folder/'fields.npz').exists():
                        raise ValueError('Accepted attempt has no committed field checkpoint')
                    return mapped_transfer(folder/'fields.npz', mapping), row, folder
                # A recorded failed attempt is evidence; restart explicitly on the
                # next campaign invocation, never overwrite its residual history.
                if (folder/'checkpoint.npz').exists():
                    resume = folder/'checkpoint.npz'
            elif (folder/'checkpoint.npz').exists():
                resume = folder/'checkpoint.npz'
            elif (unfinished/'checkpoint.npz').exists():
                resume = unfinished/'checkpoint.npz'
            index += 1
        # Reserve request metadata first. Each checkpoint and summary is atomic;
        # a crash during artifact moves is recovered by scanning both levels.
        folder.mkdir(parents=True)
        write(folder/'request.json', request)
        work = folder/'newton'
        yy, row = correct(mapping, qref_for(y, mapping), y, mode, target, work,
                          self.budget, metric, tangent, resume=resume)
        for artifact in work.iterdir():
            os.replace(artifact, folder/artifact.name)
        work.rmdir()
        attempt = dict(id=folder.name, role=role, numerical_accepted=row['accepted'],
                       summary=str((folder/'summary.json').resolve()),
                       failure=row.get('failure'), H=mapping.H, Rout=row['Rout'])
        self.state['attempts'] = [a for a in self.state['attempts'] if a['id'] != attempt['id']]+[attempt]
        self.save()
        return yy, row, folder

    def candidate(self, mapping, y, previous, mode, target, role, tangent=None,
                  refinement=False):
        yy, record, folder = self.attempt(mapping, y, mode, target, role, tangent)
        if not record['accepted']:
            return None
        yy, gate = screen(previous, yy, mapping, self.source['raw_record']['A1'],
                          tangent=tangent, refinement=refinement)
        write(folder/'screen.json', gate)
        if not gate['accepted']:
            self.state['attempts'].append(dict(id=folder.name+'_screen', role=role,
                numerical_accepted=True, branch_accepted=False, reasons=gate['reasons']))
            self.save()
            return None
        point = {key: record[key] for key in (
            'Rout', 'omega', 'A1', 'energies', 'highest_to_fundamental',
            'relative_field_inf', 'residual_over_fundamental_feedback',
            'n', 'H', 'intervals', 'phase', 'branch_constraint')}
        point.update(id=folder.name, role=role, accepted=True, resolved=record['highest_to_fundamental'] <= .01,
                     fields=str((folder/'fields.npz').resolve()),
                     fields_sha256=sha(folder/'fields.npz'),
                     summary=str((folder/'summary.json').resolve()),
                     a=record['Rout']-100, dr=(record['Rout']-400)/mapping.intervals,
                     screening=gate, source_id=SOURCE_ID)
        from fullbvp_phaseC_postprocess import diagnostic
        point = diagnostic(point, self.out)
        return point

    def load(self, point, H=None):
        mapping = Mapping(self.config['n'], H or point['H'], self.config['intervals'])
        return mapping, mapped_transfer(ROOT/point['fields'], mapping)

    def refinement(self, base_point, role):
        H = base_point['H']
        if H >= self.config['maximum_H']:
            row = dict(source=base_point['id'], H=H, Hnext=None, Rout=base_point['Rout'],
                       status='H8_limit_no_H9', delta_omega=None, delta_front=None)
            self.state['refinements'].append(row)
            self.save()
            return None
        mapping, previous = self.load(base_point, H+1)
        refined = self.candidate(mapping, previous.copy(), previous, 'Rout',
                                 base_point['Rout'], role, refinement=True)
        row = dict(source=base_point['id'], H=H, Hnext=H+1, Rout=base_point['Rout'],
                   status='failed_or_branch_rejected' if refined is None else 'strict_corrected',
                   point=None if refined is None else refined['id'],
                   delta_omega=None if refined is None else refined['omega']-base_point['omega'],
                   delta_front=None if refined is None or refined['front_fig8'] is None
                   or base_point['front_fig8'] is None else refined['front_fig8']-base_point['front_fig8'],
                   next_ratio=None if refined is None else refined['highest_to_fundamental'])
        self.state['refinements'] = [r for r in self.state['refinements']
            if not (r['source'] == row['source'] and r['Hnext'] == row['Hnext'])]+[row]
        self.save()
        return refined

    def adapt(self, point):
        while point['highest_to_fundamental'] > self.config['promotion_ratio']:
            if point['H'] >= self.config['maximum_H']:
                self.refinement(point, 'harmonic_limit')
                if point['highest_to_fundamental'] > self.config['trustworthy_ratio']:
                    raise ScientificStop('H8 highest/fundamental >1e-2: unresolved, no target success')
                point['harmonic_comparison_status'] = 'H8_limit_trustworthy_but_promotion_limited'
                return point
            refined = self.refinement(point, 'harmonic_promotion')
            if refined is None:
                raise ScientificStop('Required same-Rout harmonic promotion failed; evidence retained')
            point = refined
        if point['H'] < self.config['maximum_H']:
            audit = self.refinement(point, 'harmonic_audit')
            point['harmonic_comparison_status'] = 'H_to_Hplus1_corrected' if audit else 'Hplus1_failed_evidence_retained'
        else:
            self.refinement(point, 'harmonic_limit')
            point['harmonic_comparison_status'] = 'H8_limit_no_H9'
        return point

    def commit(self, point):
        if point['highest_to_fundamental'] > self.config['trustworthy_ratio']:
            raise ScientificStop('Attempt to commit unresolved harmonic truncation')
        if any(p['id'] == point['id'] for p in self.state['points']):
            return
        point['sequence'] = len(self.state['points'])
        self.state['points'].append(point)
        self.save()

    def seed(self):
        if self.state['points']:
            return
        mapping = Mapping(self.config['n'], 4, self.config['intervals'])
        old = mapped_transfer(self.source['fields'], mapping)
        point = self.candidate(mapping, old.copy(), old, 'Rout', old[-1],
                               'source_H4_lift', refinement=True)
        if point is None:
            raise ScientificStop('Strict same-Rout H4 source lift failed; no continuation launched')
        # Explicit H3-to-H4 comparison uses the same-Rout audited source.
        audited = self.source['audited_record']
        self.state['refinements'].append(dict(source=SOURCE_ID, point=point['id'],
            H=3, Hnext=4, Rout=point['Rout'], status='strict_corrected',
            delta_omega=point['omega']-audited['omega'],
            delta_front=None if point['front_fig8'] is None else point['front_fig8']-audited['front_fig8']))
        self.commit(self.adapt(point))

    def history(self, mapping):
        states = [mapped_transfer(ROOT/p['fields'], mapping) for p in self.state['points'][-2:]]
        if len(states) == 2:
            states[0], _ = phase_align(states[0], states[1], mapping)
        return states

    def predictor(self, mapping, target):
        states = self.history(mapping)
        old = states[-1]
        tangent = None
        if len(states) >= 2:
            delta = states[-1]-states[-2]
            if delta[-1] > 1e-7:
                tangent = CoherentMetric(mapping).tangent(states[-2], states[-1])
                y = old+(target-old[-1])/delta[-1]*delta
            else:
                y = old.copy()
        else:
            y = old.copy()
        y[-1] = target
        return old, y, tangent

    def local_arc(self, mapping, target):
        states = self.history(mapping)
        if len(states) < 2:
            self.state['fallback_events'].append(dict(status='unavailable_bootstrap',
                Rout=self.state['points'][-1]['Rout'], target=target,
                reason='Only one corrected C state; no justified coherent secant tangent'))
            self.save()
            return False
        metric = CoherentMetric(mapping)
        tangent = metric.tangent(states[-2], states[-1])
        if tangent[-1] <= 1e-8:
            self.state['fallback_events'].append(dict(status='unavailable_forward_tangent', target=target))
            self.save()
            return False
        ds = min(.05, .5*(target-states[-1][-1])/tangent[-1])
        for local in range(self.config['max_local_arc_steps']):
            # An arc is a corrected off-target local point, never target success.
            old = self.history(mapping)[-1]
            predicted = old+ds*tangent
            if predicted[-1] >= target:
                ds *= .5
                continue
            point = self.candidate(mapping, predicted, old, 'arc', 0., 'local_arc', tangent)
            if point is not None and point['Rout'] < target-1e-7:
                self.commit(self.adapt(point))
                self.state['fallback_events'].append(dict(status='accepted_local_arc',
                    point=point['id'], requested_target=target, ds=ds))
                self.save()
                return True
            ds *= .5
        self.state['fallback_events'].append(dict(status='local_arc_failed', target=target))
        self.save()
        return False

    def run(self):
        self.state.update(status='running', numerical_status='running', diagnostic_status='pending')
        self.save()
        self.seed()
        for target in self.config['targets']:
            target_key = str(target)
            if self.state['targets'][target_key]['status'] == 'accepted':
                continue
            step = min(self.config['maximum_step'], target-self.state['points'][-1]['Rout'])
            fallback_used = False
            while self.state['points'][-1]['Rout'] < target-1e-7:
                current = self.state['points'][-1]
                mapping, _ = self.load(current)
                trial_R = min(target, current['Rout']+step)
                old, predicted, tangent = self.predictor(mapping, trial_R)
                point = self.candidate(mapping, predicted, old, 'Rout', trial_R,
                                       'Rout_step', tangent)
                if point is not None:
                    self.commit(self.adapt(point))
                    step = min(self.config['maximum_step'], 1.5*step)
                    fallback_used = False
                    continue
                step *= .5
                if step >= self.config['minimum_step']:
                    continue
                if not fallback_used and self.local_arc(mapping, target):
                    step = min(self.config['maximum_step'], max(self.config['minimum_step'],
                               .5*(target-self.state['points'][-1]['Rout'])))
                    fallback_used = True
                    continue
                self.state['targets'][target_key].update(status='failed',
                    last_accepted_Rout=self.state['points'][-1]['Rout'])
                self.save()
                raise ScientificStop(f'Rout={target:g} unfinished after subdivision and local arc fallback')
            point = self.state['points'][-1]
            if abs(point['Rout']-target) >= 1e-7:
                raise ScientificStop('Target ledger cannot accept an off-target arc point')
            self.state['targets'][target_key].update(status='accepted', point=point['id'])
            self.save()
        self.state.update(numerical_status='all_targets_accepted', status='postprocessing')
        self.save()
        from fullbvp_phaseC_postprocess import finalize
        finalize(self.out)
        self.state.update(status='complete', diagnostic_status='saved')
        self.save()


def execute(out, config, source=None):
    """The lock covers numerical work, ledger commits, diagnostics and markers."""
    out = Path(out).resolve()
    with Exclusive(out):
        campaign = None
        # Preserve diagnostic evidence from previous outcomes in a small history.
        for name in ('C_DONE.json', 'C_FAILED.json'):
            if (out/name).exists():
                saved = read(out/name)
                write(out/'marker_history'/f'{time.time_ns()}_{name}', saved)
        clear_markers(out)
        try:
            campaign = Campaign(out, config, source or source_identity())
            if campaign.state.get('status') == 'complete':
                # Rebuild postprocessing after validating all committed fields.
                from fullbvp_phaseC_postprocess import finalize
                campaign.state.update(status='postprocessing', diagnostic_status='running')
                campaign.save()
                finalize(out)
                campaign.state.update(status='complete', diagnostic_status='saved')
                campaign.save()
            else:
                campaign.run()
            if (campaign.state.get('numerical_status') != 'all_targets_accepted'
                    or any(t['status'] != 'accepted' for t in campaign.state['targets'].values())
                    or not (out/'POSTPROCESS_DONE.json').exists()
                    or read(out/'POSTPROCESS_DONE.json').get('fingerprint') != campaign.identity):
                raise ScientificStop('Completion guard: unfinished targets or diagnostics')
            finish_marker(out, True, dict(status='all_requested_targets_and_diagnostics_saved',
                fingerprint=campaign.identity, target_count=len(config['targets']),
                domain_convergence='not_claimed_by_completion_marker',
                numerical_status=campaign.state['numerical_status'], diagnostic_status='saved'))
        except BaseException as exc:
            if campaign is not None:
                campaign.state.update(status='failed', failure=f'{type(exc).__name__}: {exc}')
                if campaign.state.get('numerical_status') == 'all_targets_accepted':
                    campaign.state['diagnostic_status'] = 'failed'
                campaign.save()
            finish_marker(out, False, dict(status='failed', failure=f'{type(exc).__name__}: {exc}',
                traceback=traceback.format_exc(),
                numerical_status=None if campaign is None else campaign.state.get('numerical_status'),
                diagnostic_status=None if campaign is None else campaign.state.get('diagnostic_status'),
                accepted_points=0 if campaign is None else len(campaign.state['points'])))
            raise


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--out', type=Path, default=DEFAULT_OUT)
    parser.add_argument('--targets', type=float, nargs='+', default=TARGETS)
    parser.add_argument('--budget', type=float, default=1200.)
    parser.add_argument('--minimum-step', type=float, default=.05)
    parser.add_argument('--maximum-step', type=float, default=5.)
    args = parser.parse_args()
    def interrupted(signum, _):
        raise CampaignInterrupted(f'Campaign interrupted by signal {signum}; checkpoint resume available')
    signal.signal(signal.SIGTERM, interrupted)
    signal.signal(signal.SIGINT, interrupted)
    execute(args.out, configuration(args.targets, args.budget, args.minimum_step, args.maximum_step))


if __name__ == '__main__':
    main()
