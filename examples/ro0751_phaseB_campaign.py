#!/usr/bin/env python3
"""Supervisor-launched, serial, restartable Ro=-.751/beta85 Phase B campaign."""
import os
for _key in ('OPENBLAS_NUM_THREADS', 'OMP_NUM_THREADS', 'MKL_NUM_THREADS',
             'NUMEXPR_NUM_THREADS', 'VECLIB_MAXIMUM_THREADS', 'BLIS_NUM_THREADS'):
    os.environ[_key] = '1'
os.environ.setdefault('MPLCONFIGDIR', '/tmp/ro0751_phaseB_matplotlib')
import argparse
import fcntl
import gc
import hashlib
import json
import resource
import subprocess
import sys
import time
import traceback
from pathlib import Path
import numpy as np
from ro0751_phaseB_solver import (ROOT, Mapping, Metric, common, energies,
                                  pack, transfer, unpack, write)

OUT = ROOT/'results/ro0751_global_20261101/phaseB'
PHASE_A = ROOT/'results/ro0751_global_20261101/phaseA'
BETA_A = PHASE_A/'beta85'
RO, BETA = -.751, 85.
TARGETS = (730., 750., 780.)


def read(path):
    return json.loads(Path(path).read_text())


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def prerequisites():
    markers = [PHASE_A/'NUMERICAL_DONE', BETA_A/'NUMERICAL_DONE',
               BETA_A/'status/neutral.done', BETA_A/'neutral.done']
    if not any(p.is_file() for p in markers):
        raise RuntimeError('Phase A beta85 completion marker absent; no numerical work allowed')
    seeds = {}
    for label, n in [('N32_dr0.5', 32), ('N40_dr0.25', 40)]:
        path = BETA_A/'neutral'/label
        d = read(path/'summary.json')
        if d['ro'] != RO or d['beta'] != BETA or d['n'] != n:
            raise ValueError('Phase A neutral metadata mismatch')
        with np.load(path/'neutral_state.npz') as s:
            if (float(s['ro']) != RO or float(s['beta']) != BETA
                    or abs(float(s['sigma_residual'])) > 2e-8
                    or abs(float(s['omega_c'])-d['omega']) > 1e-9
                    or abs(float(s['radii'][-1])-d['rout']) > 1e-9
                    or len(s['radii']) != d['intervals']+1
                    or len(s['xi']) != n+1 or not np.isfinite(s['q']).all()):
                raise ValueError('Phase A neutral state is inconsistent or nonneutral')
        seeds[label] = d
    a3 = read(BETA_A/'A3/summary.json')
    if a3['ro'] != RO or a3['beta'] != BETA or len(a3['grids']) != 2:
        raise ValueError('Phase A r_end diagnostics missing')
    for n in (32, 40):
        curve = read(BETA_A/f'A3/curve_N{n}_global.json')
        if curve['ro'] != RO or curve['beta'] != BETA or not curve['rows']:
            raise ValueError('Local pinch source mismatch')
        radii = [p['r'] for p in curve['rows']]
        if not np.all(np.diff(radii) > 0):
            raise ValueError('Local pinch radii must increase strictly')
    return dict(markers=[str(p.relative_to(ROOT)) for p in markers if p.is_file()],
                seeds=seeds, linear_r_end=a3['grids'])


def contract_paths():
    paths = list((ROOT/'src/llsa').rglob('*.py'))
    paths += list((ROOT/'examples').glob('ro0751_phaseB_*.py'))
    paths += [ROOT/'examples'/name for name in
              ('appelquist2015_hopf_common.py', 'appelquist2015_hopf_block_corrector.py',
               'appelquist2016_branch_continuation.py', 'fullbvp_phaseB_solver.py')]
    paths += [ROOT/'docs/tasks/RO0751_PHASE_B_NONLINEAR.md', OUT/'commands.sh']
    for label in ('N32_dr0.5', 'N40_dr0.25'):
        paths += [BETA_A/'neutral'/label/name for name in ('summary.json', 'neutral_state.npz')]
    paths += [BETA_A/f'A3/curve_N{n}_global.json' for n in (32, 40)]
    paths += [BETA_A/'A3/summary.json']
    return sorted(set(paths))


def verify_contract(out):
    contract = read(out/'source_contract.json')
    changed = [name for name, digest in contract['files'].items()
               if not (ROOT/name).is_file() or sha(ROOT/name) != digest]
    if changed:
        raise RuntimeError('Prepared inputs/source changed: '+', '.join(changed[:6]))


def fold_fit(points):
    """Bracketed minimum only; never extrapolate a missing fold."""
    points = sorted(points, key=lambda p: p['A1'])
    if len(points) < 3:
        return dict(status='insufficient_points', Rout=None, A1=None)
    amp = np.array([p['A1'] for p in points])
    rout = np.array([p['Rout'] for p in points])
    i = int(np.argmin(rout))
    if i == 0 or i == len(points)-1:
        return dict(status='minimum_not_bracketed', Rout=None, A1=None)
    fits = []
    for count in (3, 5):
        ids = np.sort(np.argsort(abs(amp-amp[i]))[:min(count, len(points))])
        x, z = amp[ids], rout[ids]
        if len(np.unique(x)) < 3:
            return dict(status='duplicate_amplitudes', Rout=None, A1=None)
        coef = np.polyfit(x-x.mean(), z, 2)
        af = x.mean()-coef[1]/(2*coef[0]) if coef[0] > 0 else np.inf
        valid = bool(x[0] < af < x[-1] and coef[0] > 0)
        fits.append(dict(A1=float(af) if valid else None,
                         Rout=float(np.polyval(coef, af-x.mean())) if valid else None,
                         samples=[points[k]['id'] for k in ids],
                         fit_rms=float(np.sqrt(np.mean((np.polyval(coef, x-x.mean())-z)**2)))))
    result = dict(fits[-1], status='bracketed_quadratic_estimate' if fits[-1]['Rout'] is not None else 'unbracketed')
    result['three_vs_five_Rout'] = (fits[0]['Rout']-fits[1]['Rout']
                                    if all(f['Rout'] is not None for f in fits) else None)
    return result


class ScientificStop(Exception):
    pass


class Campaign:
    def __init__(self, out, budget=1200.):
        self.out = Path(out).resolve()
        if self.out != OUT:
            raise ValueError('Writes are restricted to the designated Phase B directory')
        self.out.mkdir(parents=True, exist_ok=True)
        self.lock = (self.out/'numerical.lock').open('a')
        fcntl.flock(self.lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        self.inputs = prerequisites()
        verify_contract(self.out)
        if not read(self.out/'preparation_checks.json')['passed']:
            raise RuntimeError('Preparation checks have not passed')
        self.budget = float(budget)
        self.state = (read(self.out/'campaign.json') if (self.out/'campaign.json').exists()
                      else dict(status='running', stage='seeds', points=[], attempts=[],
                                targets={}, folds={}, refinements=[], stop=None))
        coarse = self.inputs['seeds']['N32_dr0.5']
        self.mapping = Mapping(32, 3, coarse['intervals'])
        self.metric = Metric(self.mapping)
        self.start = time.monotonic()

    def save(self):
        self.state['elapsed_current_invocation'] = time.monotonic()-self.start
        write(self.out/'campaign.json', self.state)
        write(self.out/'accepted_points.json', dict(points=self.state['points']))

    def primary(self):
        return [p for p in self.state['points'] if p['role'] in ('weak_seed', 'primary_arclength')]

    def qref(self, y, mapping):
        p, *_ = mapping.at(y[-1], y[-2])
        q = unpack(y[:-2], mapping.nr, mapping.H, mapping.s)[:, mapping.H+1]
        return q/common.mass_norm(p, q)

    def stop(self, reason):
        self.state.update(stop=reason, stage='postprocess')
        self.save()
        raise ScientificStop(reason)

    def attempt(self, mapping, y, qref, mode, target, name, role,
                tangent=None, weak=False):
        folder = self.out/'attempts'/name
        folder.mkdir(parents=True, exist_ok=True)
        spec = dict(n=mapping.n, H=mapping.H, intervals=mapping.intervals, ro=RO, beta=BETA)
        digest = hashlib.sha256(y.tobytes()+qref.tobytes()
                                +(b'' if tangent is None else tangent.tobytes())).hexdigest()
        signature = hashlib.sha256(json.dumps([spec, mode, target, digest], sort_keys=True).encode()).hexdigest()
        result_dir = folder/'result'
        result_path = result_dir/'summary.json'
        if result_path.exists():
            d = read(result_path)
            if d.get('request_sha256') != signature:
                raise ValueError('Restart request differs for '+name)
        else:
            if result_dir.exists():
                number = 1
                while folder.joinpath(f'interrupted_{number}').exists():
                    number += 1
                result_dir.rename(folder/f'interrupted_{number}')
            np.savez_compressed(folder/'input.npz', y=y, qref=qref,
                                tangent=np.array([]) if tangent is None else tangent)
            request = dict(mapping=spec, mode=mode, target=float(target),
                           input=str(folder/'input.npz'), output=str(result_dir),
                           request_sha256=signature, budget=self.budget,
                           weak_predictor=weak, rescale_energy=(mode == 'energy'))
            write(folder/'request.json', request)
            began = time.monotonic()
            try:
                with (folder/'worker.log').open('a') as log:
                    proc = subprocess.run([sys.executable, str(ROOT/'examples/ro0751_phaseB_solver.py'),
                                           '--request', str(folder/'request.json')],
                                          cwd=ROOT, stdout=log, stderr=log, timeout=self.budget)
                d = read(result_path) if result_path.exists() else dict(
                    accepted=False, failure=f'worker_exit_{proc.returncode}_without_result')
                if proc.returncode != 0:
                    d.update(accepted=False, failure=f'worker_exit_{proc.returncode}')
            except subprocess.TimeoutExpired:
                d = dict(accepted=False, failure='hard_per_point_timeout', budget_stop=True)
            d.update(seconds=time.monotonic()-began, request_sha256=signature)
            write(result_path, d)
        solved = bool(d['accepted'])
        yy = transfer(result_dir/'fields.npz', mapping) if solved else y.copy()
        forward = True
        if solved and tangent is not None:
            # y is the predictor; acceptance uses the full step from the previous root.
            previous = transfer(ROOT/self.primary()[-1]['fields'], mapping)
            delta = yy-previous
            forward = (self.metric.dot(delta, tangent) > 0
                       and self.metric.dot(delta, tangent)/max(np.sqrt(self.metric.dot(delta, delta)), 1e-30) > .1)
        d['equations_accepted'] = solved
        d['forward_orientation'] = bool(forward)
        if solved and not forward:
            d.update(accepted=False, failure='backward_or_orthogonal_branch_correction')
        record = dict(id=name, role=role, accepted=bool(d['accepted']),
                      equations_accepted=solved, failure=d.get('failure'),
                      seconds=d['seconds'], summary=str(result_path.relative_to(ROOT)))
        self.state['attempts'] = [p for p in self.state['attempts'] if p['id'] != name]+[record]
        if d['accepted']:
            point = {k:d[k] for k in ('Rout', 'omega', 'A1', 'energies', 'highest_to_fundamental',
                                      'relative_field_inf', 'residual_over_fundamental_feedback',
                                      'phase', 'branch_constraint', 'n', 'H', 'intervals')}
            point.update(id=name, role=role, fields=str((result_dir/'fields.npz').relative_to(ROOT)),
                         summary=record['summary'], a=d['Rout']-100,
                         dr=(d['Rout']-400)/mapping.intervals, accepted=True,
                         independent=d['independent'])
            if d['highest_to_fundamental'] > .01:
                record.update(accepted=False, failure='highest_harmonic_ratio_above_1e-2')
                d.update(accepted_branch_point=False, truncation_rejected=True)
                write(result_path, d)
                self.state['rejected_truncation'] = dict(id=name, ratio=d['highest_to_fundamental'])
                self.save()
                self.stop('highest_harmonic_ratio_above_1e-2')
            if not any(p['id'] == name for p in self.state['points']):
                self.state['points'].append(point)
        write(result_path, d)
        self.save()
        gc.collect()
        if d.get('budget_stop'):
            self.stop('per_point_budget_exceeded')
        return yy, d

    def seeds(self):
        coarse = self.inputs['seeds']['N32_dr0.5']
        with np.load(BETA_A/'neutral/N32_dr0.5/neutral_state.npz') as d:
            q1 = d['q'].reshape(self.mapping.nr, self.mapping.s).copy()
        p, *_ = self.mapping.at(coarse['rout'], coarse['omega'])
        q1 /= common.mass_norm(p, q1)
        for k, amplitude in enumerate((.0064, .009)):
            name = f'seed_{k}'
            if any(q['id'] == name for q in self.primary()):
                continue
            q = np.zeros((self.mapping.nr, 7, self.mapping.s), complex)
            q[:, 4] = amplitude*q1
            q[:, 2] = q[:, 4].conj()
            y = np.r_[pack(q), coarse['omega'], coarse['rout']]
            _, d = self.attempt(self.mapping, y, q1, 'energy', amplitude, name, 'weak_seed', weak=True)
            if not d['accepted']:
                self.stop('weak_seed_failure')
        points = self.primary()[:2]
        amp2 = np.array([p['A1']**2 for p in points])
        fits = {key: np.polyfit(amp2, [p[key] for p in points], 1).tolist()
                for key in ('Rout', 'omega')}
        verification = dict(linear_Rout=coarse['rout'], linear_omega=coarse['omega'],
                            quadratic_fits=fits,
                            intercept_Rout_error=fits['Rout'][1]-coarse['rout'],
                            intercept_omega_error=fits['omega'][1]-coarse['omega'],
                            subcritical=all(p['Rout'] < coarse['rout'] for p in points),
                            passed=(max(abs(fits['Rout'][1]-coarse['rout']),
                                        abs(fits['omega'][1]-coarse['omega'])) < .002))
        write(self.out/'weak_seed_verification.json', verification)
        self.state['weak_seed_verification'] = verification
        if not verification['passed']:
            self.stop('weak_seed_neutral_limit_mismatch')
        self.state['stage'] = 'branch_fold'
        self.save()

    def branch(self, until_fold=False, until_mesh=False):
        points = self.primary()
        old = transfer(ROOT/points[-2]['fields'], self.mapping)
        y = transfer(ROOT/points[-1]['fields'], self.mapping)
        seed = transfer(ROOT/points[0]['fields'], self.mapping)
        qref = self.qref(seed, self.mapping)
        tangent = self.metric.tangent(old, y)
        ds = self.state.get('next_step', .04)
        while True:
            points = self.primary()
            rout = [p['Rout'] for p in points]
            imin = int(np.argmin(rout))
            fold = (imin > 0 and len(points)-imin >= 3
                    and points[imin+1]['Rout'] > points[imin]['Rout'])
            if until_fold and fold:
                self.state['fold_bracket'] = [p['id'] for p in points[imin-1:imin+3]]
                self.state['stage'] = 'folds'
                self.save()
                return
            if until_mesh:
                linear = self.inputs['seeds']
                margin = max(.25, 2*abs(linear['N40_dr0.25']['rout']-linear['N32_dr0.5']['rout']))
                if (points[-1].get('upper_branch')
                        and y[-1] >= self.state['folds']['H3']['Rout']+margin):
                    self.state.update(stage='mesh', mesh_minimum_fold_margin=margin)
                    self.save()
                    return
            if not until_fold and not until_mesh and y[-1] >= max(TARGETS):
                self.state.update(stage='targets', stop='target_endpoint_reached')
                self.save()
                return
            success = False
            idx = len(points)
            for retry in range(4):
                trial, d = self.attempt(self.mapping, y+ds*tangent, qref, 'arc', 0.,
                                         f'branch_{idx:04d}_try{retry}', 'primary_arclength', tangent)
                if d['accepted']:
                    newt = self.metric.tangent(y, trial)
                    point = self.primary()[-1]
                    point.update(tangent_Rout=float(newt[-1]), tangent_omega=float(newt[-2]),
                                 upper_branch=bool(points[-1].get('upper_branch') or fold
                                                   or (tangent[-1] < 0 < newt[-1])))
                    upper_before = points[-1].get('upper_branch', False)
                    if upper_before and tangent[-1] > 0 > newt[-1]:
                        self.state['upper_return'] = point['id']
                        self.stop('upper_branch_return_turning_point')
                    np.savez_compressed(self.out/'continuation_state.npz', previous=y,
                                        current=trial, tangent=newt, step=ds, qref=qref)
                    y, tangent = trial, newt
                    ds = min(.16, ds*(1.2 if len(d['iterations']) <= 4 else 1.03))
                    self.state['next_step'] = ds
                    self.save()
                    success = True
                    break
                ds *= .5
            if not success:
                self.stop('four_consecutive_Newton_or_orientation_failures')

    def folds(self):
        coarse = fold_fit(self.primary())
        self.state['folds']['H3_coarse'] = coarse
        if coarse['Rout'] is None:
            self.stop('fold_not_bracketed')
        af = coarse['A1']
        spacing = max(.002, min(.004, af*.04))
        for H in (3, 4):
            mapping = self.mapping if H == 3 else Mapping(32, H, self.mapping.intervals)
            amplitudes = [af+k*spacing for k in (-2, -1, 0, 1, 2)]
            for k in range(9):
                if k < 5:
                    amp = amplitudes[k]
                else:
                    subset = [p for p in self.state['points'] if p['role'] == f'fold_H{H}']
                    fit = fold_fit(subset)
                    if fit['Rout'] is not None:
                        break
                    i = int(np.argmin([p['Rout'] for p in sorted(subset, key=lambda p:p['A1'])]))
                    amp = (min(p['A1'] for p in subset)-spacing if i == 0
                           else max(p['A1'] for p in subset)+spacing)
                if amp <= 0:
                    self.stop('fold_amplitude_bracket_reached_zero')
                candidates = [p for p in self.state['points'] if p['H'] == H] or self.primary()
                source = min(candidates, key=lambda p:abs(p['A1']-amp))
                y = transfer(ROOT/source['fields'], mapping)
                _, d = self.attempt(mapping, y, self.qref(y, mapping), 'energy', amp,
                                    f'fold_H{H}_{k:02d}', f'fold_H{H}')
                if not d['accepted']:
                    self.stop(f'fold_H{H}_correction_failure')
            self.state['folds'][f'H{H}'] = fold_fit(
                [p for p in self.state['points'] if p['role'] == f'fold_H{H}'])
            self.save()
        h3, h4 = [self.state['folds'][f'H{h}'] for h in (3, 4)]
        if h3['Rout'] is None or h4['Rout'] is None:
            self.stop('H3_or_H4_refined_fold_not_bracketed')
        self.state['folds']['H4_minus_H3_Rout'] = (h4['Rout']-h3['Rout']
                                                  if h4['Rout'] is not None else None)
        self.state['stage'] = 'branch_mesh'
        self.save()

    def mesh(self):
        af = self.state['folds']['H3']['A1']
        if af is None:
            self.stop('H3_refined_fold_not_bracketed')
        upper = [p for p in self.primary() if p.get('upper_branch') and p['A1'] > af]
        if not upper:
            self.stop('no_verified_upper_mesh_source')
        source = upper[-1]
        tests = [('N40_dr025', 40, 3, int(np.ceil((source['Rout']-400)/.25))),
                 ('H4_upper', 32, 4, self.mapping.intervals)]
        for label, n, H, intervals in tests:
            if any(p['label'] == label for p in self.state['refinements']):
                continue
            mapping = Mapping(n, H, intervals)
            y = transfer(ROOT/source['fields'], mapping)
            _, d = self.attempt(mapping, y, self.qref(y, mapping), 'Rout', source['Rout'],
                                f'mesh_{label}', 'mesh_refinement')
            reference_fold_A = self.state['folds'][f'H{H}']['A1']
            retained_upper = bool(d['accepted'] and d['A1'] > reference_fold_A)
            self.state['refinements'].append(dict(label=label, source=source['id'],
                                                   point=f'mesh_{label}', matching='fixed_Rout',
                                                   accepted=retained_upper,
                                                   branch_check='A1 exceeds matched coarse-grid harmonic fold; field mass difference reported'))
            self.save()
            if not retained_upper:
                self.stop(f'mesh_{label}_correction_failure')
        self.state['stage'] = 'branch_upper'
        self.save()

    def targets(self):
        af = self.state['folds'].get('H3', {}).get('A1')
        upper = [p for p in self.primary() if p.get('upper_branch')
                 and af is not None and p['A1'] > af]
        for target in TARGETS:
            label = f'{target:g}'
            if label in self.state['targets']:
                continue
            if not upper or not min(p['Rout'] for p in upper) <= target <= max(p['Rout'] for p in upper):
                self.state['targets'][label] = dict(status='unavailable', reason='outside_verified_upper_branch_range')
                continue
            source = min(upper, key=lambda p:abs(p['Rout']-target))
            y = transfer(ROOT/source['fields'], self.mapping)
            # Preserve xi when changing Rout: avoid extrapolation into an extended outlet.
            y[-1] = target
            name = f'target_R{target:g}'
            _, d = self.attempt(self.mapping, y, self.qref(y, self.mapping), 'Rout', target,
                                name, 'upper_target')
            accepted = bool(d['accepted'] and self.state['points'][-1]['A1'] > af)
            if not accepted:
                self.stop('target_did_not_retain_upper_branch')
            self.state['targets'][label] = dict(status='accepted', point=name)
            self.save()
        self.state['stage'] = 'postprocess'
        self.save()

    def run(self):
        if self.state['stage'] == 'done' and (self.out/'WORKFLOW_DONE.json').exists():
            return
        self.state['status'] = 'running'
        write(self.out/'NUMERICAL_STARTED.json', dict(pid=os.getpid(), point_budget=self.budget))
        for name in ('PENDING_SUPERVISOR', 'READY_FOR_SUPERVISOR'):
            (self.out/name).unlink(missing_ok=True)
        if (self.out/'FAILED.json').exists():
            (self.out/'FAILED.json').rename(self.out/f'FAILED_previous_{time.time_ns()}.json')
        self.save()
        try:
            for stage in ('seeds', 'branch_fold', 'folds', 'branch_mesh', 'mesh', 'branch_upper', 'targets'):
                if self.state['stage'] == stage:
                    if stage.startswith('branch_'):
                        self.branch(until_fold=(stage == 'branch_fold'), until_mesh=(stage == 'branch_mesh'))
                    else:
                        getattr(self, stage)()
        except ScientificStop:
            pass
        except BaseException:
            self.state.update(status='failed_exception', traceback=traceback.format_exc())
            self.save()
            try:
                from ro0751_phaseB_report import finalize
                finalize(self.out, self.inputs)
            finally:
                write(self.out/'FAILED.json', dict(stage=self.state['stage'], traceback=self.state['traceback']))
            raise
        complete = (all(self.state['folds'].get(f'H{h}', {}).get('Rout') is not None for h in (3, 4))
                    and all(self.state['targets'].get(f'{r:g}', {}).get('status') == 'accepted' for r in TARGETS)
                    and all(any(x['label'] == label and x['accepted'] for x in self.state['refinements'])
                            for label in ('N40_dr025', 'H4_upper')))
        self.state.update(stage='done', status='numerical_done' if complete else 'stopped_incomplete')
        self.save()
        from ro0751_phaseB_report import finalize
        # Scientific stops do not authorize additional solves. Existing points are still diagnosed.
        finalize(self.out, self.inputs)
        write(self.out/('NUMERICAL_DONE' if complete else 'NUMERICAL_STOPPED.json'),
              dict(status=self.state['status'], stop=self.state['stop'],
                   targets=self.state['targets'], human_figure_review='pending'))
        write(self.out/'WORKFLOW_DONE.json', dict(status=self.state['status'],
                                                report='REPORT.md', human_figure_review='pending'))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--out', type=Path, default=OUT)
    parser.add_argument('--budget', type=float, default=1200.)
    args = parser.parse_args()
    resource.setrlimit(resource.RLIMIT_AS, (200*1024**3, 200*1024**3))
    if not 60 <= args.budget <= 3600:
        raise ValueError('Per-point budget must be 60..3600 seconds')
    Campaign(args.out, args.budget).run()


if __name__ == '__main__':
    main()
