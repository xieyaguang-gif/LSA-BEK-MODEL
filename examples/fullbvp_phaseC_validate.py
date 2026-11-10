#!/usr/bin/env python3
"""Bounded Phase C checks; no full-size continuation is launched.

Algebraic tests use N8/H2/12 intervals. A genuine small, unforced nonlinear
correction starts from a coarsened audited field, with a 12-second Newton budget.
Failure of that coarse correction is retained as evidence, not reported as a
successful full-resolution upper root. Total numerical budget is <20 minutes.
"""
from __future__ import annotations

import os
for _key in ('OPENBLAS_NUM_THREADS', 'OMP_NUM_THREADS', 'MKL_NUM_THREADS',
             'NUMEXPR_NUM_THREADS', 'VECLIB_MAXIMUM_THREADS', 'BLIS_NUM_THREADS'):
    os.environ[_key] = '1'

import argparse
import copy
import signal
import tempfile
import time
from pathlib import Path
from types import SimpleNamespace

import numpy as np

import fullbvp_phaseC_campaign as driver
import fullbvp_phaseC_solver as solver
from fullbvp_phaseC_solver import (DEFAULT_OUT, Mapping, CoherentMetric, checkpoint,
    correct, mapped_transfer, pack, phase_align, qref_for, read, screen, unpack, write)
from fullbvp_phaseB_upper_refinement import TightCorrector
from appelquist2015_hopf_block_corrector import norm


def require(condition, message):
    if not condition:
        raise AssertionError(message)


def algebra(mapping, y):
    rng = np.random.default_rng(20261030)
    metric = CoherentMetric(mapping)
    nr, h, s = mapping.nr, mapping.H, mapping.s
    mean = np.zeros((nr, 2*h+1, s), complex)
    mean[:, h] = rng.normal(size=(nr, s))*1e8
    zero = np.zeros_like(y)
    zero[:-2] = pack(mean)
    require(metric.dot(zero, zero) == 0., 'Mean perturbation has nonzero metric')
    pressure = np.zeros_like(mean).reshape(nr, 2*h+1, 4, mapping.m)
    pressure[:, :, 3] = rng.normal(size=(nr, 2*h+1, mapping.m))
    pv = np.r_[pack(pressure.reshape(mean.shape)), 0., 0.]
    require(metric.dot(pv, pv) == 0., 'Pressure perturbation has nonzero metric')
    u, v = rng.normal(size=(2, len(y)))
    require(abs(metric.dot(u, v)-metric.dot(v, u)) < 1e-8*max(abs(metric.dot(u, v)), 1.),
            'Metric is not symmetric')
    q = unpack(y[:-2], nr, h, s)
    expected = 2*sum(metric.inner(q, q, j).real for j in range(1, h+1))/metric.fs**2
    require(abs(metric.dot(np.r_[y[:-2], 0., 0.], np.r_[y[:-2], 0., 0.])-expected) < 1e-10,
            'Coherent metric does not match velocity mass norm')
    rotated = q*np.exp(1j*np.arange(-h, h+1)*.73)[None, :, None]
    aligned, angle = phase_align(np.r_[pack(rotated), y[-2:]], y, mapping)
    require(norm(aligned-y)/max(norm(y), 1.) < 1e-12, 'Harmonic phase alignment failed')
    lower = y.copy()
    lower[:-2] *= .25
    _, gate = screen(y, lower, mapping, .00001, refinement=True)
    require(gate['fundamental_correlation'] > .999999 and not gate['accepted']
            and 'amplitude_ratio' in gate['reasons'], 'Scaled lower root passed correlation-only gate')
    same_energy = y.copy()
    same_energy[-1] += 5.
    same_energy[:-2] *= np.sqrt((y[-1]-400)/(same_energy[-1]-400))
    _, physical_gate = screen(y, same_energy, mapping, .00001)
    require(abs(physical_gate['amplitude_ratio']-1) < 1e-12 and physical_gate['accepted'],
            'Physical amplitude measure changed spuriously when domain stretched')
    qref = qref_for(y, mapping)
    direction = rng.normal(size=len(y))
    direction[:-2] *= 1e-4
    tangent = direction/metric.length(direction)
    errors = {}
    for mode in ('Rout', 'arc'):
        system = TightCorrector(mapping, qref, y, mode, y[-1] if mode == 'Rout' else 0.,
                                budget=20., metric=metric, tangent=tangent)
        J, _, _ = system.jacobian(y)
        epsilon = 1e-4
        fd = (system.residual(y+epsilon*direction)-system.residual(y-epsilon*direction))/(2*epsilon)
        error = norm(fd-J@direction)/max(norm(J@direction), 1e-30)
        require(error < 2e-7, f'{mode} bordered Jv relative error {error}')
        errors[mode] = error
    return dict(mean_metric=0., pressure_metric=0., coherent_mass=expected,
                phase_rotation_recovered=angle, bordered_Jv_relative=errors,
                lower_scaled_root_gate=gate, physical_measure_gate=physical_gate)


def harmonic_policy():
    config = dict(promotion_ratio=1e-3, trustworthy_ratio=1e-2, maximum_H=8)
    calls = []
    def refine(point, role):
        calls.append((point['H'], role))
        return dict(id=f'H{point["H"]+1}', H=point['H']+1, highest_to_fundamental=1e-4)
    fake = SimpleNamespace(config=config, refinement=refine)
    point = driver.Campaign.adapt(fake, dict(id='H4', H=4, highest_to_fundamental=.003))
    require(point['H'] == 5 and calls == [(4, 'harmonic_promotion'), (5, 'harmonic_audit')],
            'Harmonic promotion or same-Rout audit was skipped')
    stopped = False
    try:
        driver.Campaign.adapt(fake, dict(id='H8_bad', H=8, highest_to_fundamental=.02))
    except driver.ScientificStop:
        stopped = True
    require(stopped, 'Unresolved H8 truncation was not stopped')
    limited = driver.Campaign.adapt(fake, dict(id='H8_limited', H=8, highest_to_fundamental=.005))
    require('limited' in limited['harmonic_comparison_status'], 'H8 limit was hidden')
    fake.refinement = lambda *_: None
    accepted = driver.Campaign.adapt(fake, dict(id='H4_audit_failure', H=4, highest_to_fundamental=1e-4))
    require(accepted['harmonic_comparison_status'] == 'Hplus1_failed_evidence_retained',
            'Failed nonmandatory H+1 comparison was erased')
    return dict(promotion=True, same_Rout_audit=True, unresolved_H8_stops=True,
                H8_cap_and_failed_comparison_explicit=True)


def injected_operations(root, mapping, y, source):
    root.mkdir(parents=True)
    marker = root/'atomic.json'
    write(marker, dict(version=1))
    try:
        write(marker, dict(bad=object()))
    except TypeError:
        pass
    require(read(marker)['version'] == 1, 'Failed JSON replaced last atomic commit')
    write(root/'C_DONE.json', dict(sentinel=True))
    with driver.Exclusive(root):
        try:
            driver.execute(root, {}, source)
        except RuntimeError as exc:
            require('exclusive' in str(exc), 'Lock failure had no meaningful error')
        else:
            raise AssertionError('Duplicate campaign acquired exclusive lock')
        require(read(root/'C_DONE.json')['sentinel'], 'Busy invocation overwrote active markers')
    driver.finish_marker(root, False, dict(status='injected_failure'))
    require(not (root/'C_DONE.json').exists(), 'Failure inherited stale DONE marker')
    driver.finish_marker(root, True, dict(status='injected_success'))
    require(not (root/'C_FAILED.json').exists(), 'Success inherited stale FAILED marker')

    config = dict(n=mapping.n, intervals=mapping.intervals, budget_per_point=10.,
                  targets=[720.], trustworthy_ratio=.01)
    work = driver.Campaign(root/'resume', config, source)
    original_correct = driver.correct
    calls = []
    def interrupted(m, qref, initial, mode, target, out, *args, **kwargs):
        out.mkdir(parents=True)
        checkpoint(out/'checkpoint.npz', m, initial, qref)
        raise driver.CampaignInterrupted('injected interruption inside nested newton/')
    def recovered(m, qref, initial, mode, target, out, *args, **kwargs):
        saved = kwargs['resume']
        require(saved is not None and saved.parent.name == 'newton' and saved.exists(),
                'Nested interrupted checkpoint was not selected for resume')
        calls.append(str(saved))
        out.mkdir(parents=True)
        checkpoint(out/'fields.npz', m, initial, qref, accepted=True)
        record = dict(accepted=True, Rout=float(initial[-1]), omega=float(initial[-2]), H=m.H)
        write(out/'summary.json', record)
        return initial.copy(), record
    try:
        driver.correct = interrupted
        try:
            work.attempt(mapping, y, 'Rout', y[-1], 'injected')
        except driver.CampaignInterrupted:
            pass
        else:
            raise AssertionError('Cancellation was swallowed by failed-Newton handling')
        driver.correct = recovered
        _, _, accepted_folder = work.attempt(mapping, y, 'Rout', y[-1], 'injected')
        # Inject a crash halfway through flattening: summary already moved,
        # while fields.npz remains under newton/. No extra solve is allowed.
        nested = accepted_folder/'newton'
        nested.mkdir()
        os.replace(accepted_folder/'fields.npz', nested/'fields.npz')
        # Preserve a prior reservation whose request never committed.
        first_folder = accepted_folder.with_name(accepted_folder.name.rsplit('_try', 1)[0]+'_try0')
        (first_folder/'request.json').unlink()
        # This reuses the accepted attempt even before any point-ledger commit.
        work.attempt(mapping, y, 'Rout', y[-1], 'injected')
        require(len(calls) == 1, 'Crash-window recovery duplicated a numerical correction')
        require((accepted_folder/'fields.npz').exists() and not nested.exists(),
                'Partial artifact move was not consolidated')
        require(work.state['orphan_attempts'] and first_folder.exists(),
                'Unverified reservation orphan was not preserved and skipped')
    finally:
        driver.correct = original_correct
    altered_source = copy.deepcopy(source)
    altered_source['audited_record']['presentation_only_injected'] = True
    driver.Campaign(root/'resume', config, altered_source)
    bad = dict(config, intervals=mapping.intervals+1)
    try:
        driver.Campaign(root/'resume', bad, source)
    except ValueError as exc:
        require('fingerprint' in str(exc), 'Config mismatch was not identified')
    else:
        raise AssertionError('Changed configuration was resumed')
    committed = dict(id=accepted_folder.name, fields=str(accepted_folder/'fields.npz'),
        fields_sha256=solver.sha(accepted_folder/'fields.npz'),
        summary=str(accepted_folder/'summary.json'), highest_to_fundamental=.001)
    write(accepted_folder/'screen.json', dict(accepted=True))
    work.state['points'] = [committed]
    work.save()
    driver.Campaign(root/'resume', config, source)
    with (accepted_folder/'fields.npz').open('ab') as stream:
        stream.write(b'injected corruption')
    try:
        driver.Campaign(root/'resume', config, source)
    except ValueError as exc:
        require('hash mismatch' in str(exc), 'Committed corruption was not identified')
    else:
        raise AssertionError('Corrupted accepted checkpoint was resumed')

    # Test completion vs failure markers through the real execute() guard.
    real_campaign = driver.Campaign
    class FakeCampaign:
        def __init__(self, out, config, source):
            self.out, self.identity = Path(out), 'injected_identity'
            self.state = dict(status='prepared', points=[], targets={'720.0': dict(status='pending')},
                              numerical_status='running', diagnostic_status='pending')
        def save(self):
            write(self.out/'campaign.json', self.state)
        def run(self):
            self.state['numerical_status'] = 'all_targets_accepted'
            if fail_diagnostic[0]:
                raise OSError('injected plot/report failure after numerical targets')
            self.state['targets']['720.0']['status'] = 'accepted'
            write(self.out/'POSTPROCESS_DONE.json', dict(fingerprint=self.identity))
    fail_diagnostic = [True]
    try:
        driver.Campaign = FakeCampaign
        marker_out = root/'execution'
        try:
            driver.execute(marker_out, dict(targets=[720.]), source)
        except OSError:
            pass
        failed = read(marker_out/'C_FAILED.json')
        require(failed['numerical_status'] == 'all_targets_accepted'
                and failed['diagnostic_status'] == 'failed'
                and not (marker_out/'C_DONE.json').exists(), 'Diagnostic failure marked target success')
        fail_diagnostic[0] = False
        driver.execute(marker_out, dict(targets=[720.]), source)
        require((marker_out/'C_DONE.json').exists() and not (marker_out/'C_FAILED.json').exists(),
                'Successful retry did not atomically replace failure')
    finally:
        driver.Campaign = real_campaign
    return dict(atomic_JSON_preserves_last_commit=True, exclusive_busy_preserves_markers=True,
                stale_marker_removal=True, nested_checkpoint_resume=True,
                partial_artifact_move_recovered=True, unverified_orphan_preserved=True,
                accepted_attempt_reused_before_ledger=True, config_mismatch_refused=True,
                committed_checkpoint_corruption_refused=True,
                harmless_audit_refresh_allowed=True, interruption_propagates=True,
                diagnostic_failure_distinguished=True)


def transfers(out, mapping, y):
    checkpoint(out/'transfer_source.npz', mapping, y, qref_for(y, mapping), accepted=True)
    larger = Mapping(mapping.n, mapping.H+1, mapping.intervals)
    lifted = mapped_transfer(out/'transfer_source.npz', larger)
    q = unpack(lifted[:-2], larger.nr, larger.H, larger.s)
    original = unpack(y[:-2], mapping.nr, mapping.H, mapping.s)
    error = norm(q[:, 1:-1]-original)/max(norm(original), 1e-30)
    require(error < 1e-11 and norm(q[:, [0, -1]]) == 0., 'H lift did not preserve existing modes')
    stretched = mapped_transfer(out/'transfer_source.npz', mapping, 725.)
    require(norm(stretched[:-2]-y[:-2])/max(norm(y[:-2]), 1e-30) < 1e-11,
            'Rout stretch physically extrapolated the field')
    return dict(H_lift_relative_error=error, H_lift_zero_padding=True,
                Rout_change_preserves_mapped_field=True)


def arc_resume(out, mapping, y):
    metric = CoherentMetric(mapping)
    tangent = np.zeros_like(y)
    tangent[-1] = 10.
    shifted = y.copy()
    shifted[-1] += .01
    saved = out/'arc_resume_input.npz'
    checkpoint(saved, mapping, shifted, qref_for(y, mapping), tangent)
    _, result = correct(mapping, qref_for(y, mapping), y, 'arc', 0.,
                        out/f'arc_resume_{time.time_ns()}', budget=2.,
                        metric=metric, tangent=tangent, max_newton=0, resume=saved)
    measured = result['iterations'][0]['branch_constraint']
    expected = metric.row(tangent)@(shifted-y)
    require(abs(measured-expected) < 1e-12 and abs(measured) > 1e-5,
            'Arc resume moved the original prediction hyperplane')
    return dict(original_arc_center_preserved=True, branch_constraint=measured)


def source_diagnostic(out, source):
    from fullbvp_phaseC_postprocess import diagnostic, profile_difference
    target = out/'source_reference_only'
    target.mkdir(exist_ok=True)
    point = dict(source['raw_record'])
    point['fields'] = source['fields']
    result = diagnostic(point, target)
    require(abs(result['front_fig8']-source['audited_record']['front_fig8']) < 1e-10,
            'Fig.8 definition changed from audited Phase B reference')
    require(result['pinch_reference_coverage'] == [550., 630.]
            and result['front_search_truncated_by_pinch_coverage'] == (point['a'] > 630.),
            'Pinch coverage censoring does not account for the actual sponge cutoff')
    with np.load(target/'diagnostics'/f'{point["id"]}.npz') as new, np.load(
            solver.ROOT/'results/fullbvp_global_20261030/phaseB/diagnostics'/f'{point["id"]}.npz') as old:
        error = norm(new['vrms']-old['vrms'])/max(norm(old['vrms']), 1e-30)
    require(error < 1e-12, 'RMS normalization changed (possible extra division by R)')
    comparison = profile_difference(target, result, result)
    require(comparison['RMS_relative_L2'] == 0., 'Common physical-R identity comparison failed')
    return dict(reference_only_not_C_acceptance=True, Fig8_front=result['front_fig8'],
                RMS_relative_to_PhaseB=error, g_q_fit=result['g_q_fit_570_600'],
                pinch_coverage=result['pinch_reference_coverage'],
                common_R_profile_identity=comparison)


def assessment_policy():
    """Synthetic edge cases test flags only; these are not campaign data."""
    from fullbvp_phaseC_postprocess import assessment
    points = [dict(id=f'synthetic_flag_only_{i}', Rout=720.+5*i, a=620.+5*i, H=4,
        omega=52.7-.01*i, front_fig8=588.+i, g_q_fit_570_600=.125,
        sponge_fraction_q=.6, sponge_fraction_cylindrical_R3=.7) for i in range(3)]
    targets = {str(p['Rout']): dict(status='accepted', point=p['id'], Rout=p['Rout']) for p in points}
    flags = assessment(points, [], targets)['C4_persistent_sponge_and_outward_front']
    require(flags['boundary_sponge_support_hypothesis_indicator'] is True,
            'Persistent >=50% support plus outward fronts did not trigger C4 hypothesis')
    inward = copy.deepcopy(points)
    for i, p in enumerate(inward):
        p['front_fig8'] = 588.-i
    require(assessment(inward, [], targets)['C4_persistent_sponge_and_outward_front']
            ['boundary_sponge_support_hypothesis_indicator'] is False,
            'Inward front motion falsely triggered C4 outlet-tracking hypothesis')
    absent = copy.deepcopy(points)
    absent[-1]['front_fig8'] = None
    require(assessment(absent, [], targets)['C4_persistent_sponge_and_outward_front']
            ['boundary_sponge_support_hypothesis_indicator'] is None,
            'Missing front was treated as measured outward motion')
    once = copy.deepcopy(points)
    once[0]['sponge_fraction_q'] = .05
    require(assessment(once, [], targets)['C4_persistent_sponge_and_outward_front']
            ['persistent_q_fraction_ge_half'] is False, 'Nonpersistent sponge support triggered C4')
    return dict(synthetic_flags_only=True, persistent_half_and_outward=True,
                inward_motion_does_not_trigger=True, missing_front_unavailable=True,
                nonpersistent_support_does_not_trigger=True)


def run(out):
    out = Path(out).resolve()
    out.mkdir(parents=True, exist_ok=True)
    evidence = out/'validation'
    evidence.mkdir(exist_ok=True)
    for name in ('VALIDATION.json', 'VALIDATION_FAILED.json'):
        previous = out/name
        if previous.exists():
            history = evidence/'history'
            history.mkdir(exist_ok=True)
            os.replace(previous, history/f'{time.time_ns()}_{name}')
    started = time.monotonic()
    # Hard outer bound is well below the task's aggregate 20-minute allowance.
    def deadline(*_):
        raise driver.CampaignInterrupted('Validation hard wall-clock budget (120 seconds)')
    previous_handler = signal.signal(signal.SIGALRM, deadline)
    signal.alarm(120)
    try:
        source = driver.source_identity()
        mapping = Mapping(8, 2, 12)
        rng = np.random.default_rng(20261030)
        y = np.r_[rng.normal(size=mapping.nr*(2*mapping.H+1)*mapping.s)*1e-5, 52.71, 718.931903]
        checks = dict(algebra=algebra(mapping, y), harmonic_policy=harmonic_policy(),
                      transfers=transfers(evidence, mapping, y),
                      arc_resume=arc_resume(evidence, mapping, y),
                      source_diagnostic=source_diagnostic(evidence, source),
                      assessment_policy=assessment_policy())
        with tempfile.TemporaryDirectory(prefix='phaseC_injection_', dir='/tmp') as temp:
            checks['injected_operations'] = injected_operations(Path(temp)/'test', mapping, y, source)
        # Persist genuine nonlinear residual/Newton evidence. Coarsening is
        # deliberate and does not certify the full N32 upper branch.
        small = Mapping(8, 2, 20)
        initial = mapped_transfer(source['fields'], small)
        initial[-2] += .001
        real_out = evidence/f'small_nonlinear_{time.time_ns()}'
        _, real = correct(small, qref_for(initial, small), initial, 'Rout', initial[-1],
                          real_out, budget=12., max_newton=2)
        checks['small_real_nonlinear'] = dict(n=8, H=2, intervals=20,
            attempted=True, accepted=real['accepted'], failure=real.get('failure'),
            seconds=real['seconds'], jv_calls=real['jv_calls'],
            summary=str((real_out/'summary.json').resolve()),
            initial_relative_field_inf=real['iterations'][0]['relative_field_inf'] if real['iterations'] else None,
            final_relative_field_inf=real['final_metrics']['relative_field_inf'] if real['iterations'] else None,
            initial_scaled_residual_l2=real['iterations'][0]['scaled_residual_l2'] if real['iterations'] else None,
            final_scaled_residual_l2=real['final_metrics']['scaled_residual_l2'] if real['iterations'] else None,
            interpretation='Small original-equation nonlinear correction only; convergence is reported honestly and does not validate N32 upper continuation')
        result = dict(status='algebraic_and_operational_checks_passed',
            full_campaign_launched=False, numerical_wall_seconds=time.monotonic()-started,
            aggregate_numerical_budget_seconds=1200, hard_validation_budget_seconds=120,
            CPU_threads=1, GPU=False, source_id=source['id'],
            source_fields_sha256=source['fields_sha256'], checks=checks)
        write(out/'VALIDATION.json', result)
        return result
    except BaseException as exc:
        write(out/'VALIDATION_FAILED.json', dict(status='failed', failure=f'{type(exc).__name__}: {exc}',
            seconds=time.monotonic()-started, full_campaign_launched=False))
        raise
    finally:
        signal.alarm(0)
        signal.signal(signal.SIGALRM, previous_handler)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--out', type=Path, default=DEFAULT_OUT)
    args = parser.parse_args()
    result = run(args.out)
    print(f'{result["status"]}; numerical wall {result["numerical_wall_seconds"]:.2f}s; full campaign unlaunched')
