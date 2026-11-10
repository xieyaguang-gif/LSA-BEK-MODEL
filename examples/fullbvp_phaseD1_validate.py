#!/usr/bin/env python3
"""Short D1 verification only; no production Newton solve or factorization."""
from __future__ import annotations

import os
os.environ['PYTHONDONTWRITEBYTECODE'] = '1'
import subprocess
import sys
import tempfile
import time
from pathlib import Path
from unittest.mock import patch

import numpy as np

import fullbvp_phaseB_solver as old
import fullbvp_phaseD1_solver as solver
import fullbvp_phaseD1_campaign as campaign
from fullbvp_phaseD1_diagnostics import diagnostic
from fullbvp_phaseD1_solver import OUT, ROOT, Mapping, common, pack, unpack, norm, write
from llsa.secondary.global_radial import GlobalRadial
from llsa.secondary.radial import RadialConfig


def require(condition, message):
    if not condition:
        raise AssertionError(message)


def run():
    started = time.monotonic()
    evidence = dict(production_launched=False, GPU=False, threads=1, tests={})
    source = solver.verified_source()
    evidence['source'] = source
    n, H, intervals = 8, 2, 80
    baseline = Mapping(n, H, intervals)
    previous = old.Mapping(n, H, intervals)
    p, blocks, _, derivatives = baseline.at(725., 52.7, True)
    pp, original, _, original_d = previous.at(725., 52.7, True)
    difference = max(float(abs(a-b).max()) for a, b in zip(blocks+derivatives, original+original_d))
    require(difference < 1e-13, 'Baseline changed relative to original operator/derivative')
    evidence['tests']['baseline_operator_and_derivative_max_abs'] = difference
    rng = np.random.default_rng(124701)
    q = np.zeros((intervals+1, 2*H+1, baseline.s), complex)
    q[:, H] = rng.normal(size=(intervals+1, baseline.s))*1e-4
    for j in range(1, H+1):
        q[:, H+j] = (rng.normal(size=(intervals+1, baseline.s))+1j*rng.normal(size=(intervals+1, baseline.s)))*1e-4
        q[:, H-j] = q[:, H+j].conj()
    y = np.r_[pack(q), 52.7, 725.]
    endpoint_tests = {}
    for name in campaign.BRANCHES:
        params = campaign.parameters(name, 1.)
        mapping = Mapping(n, H, intervals, **params)
        point, matrices, _, _ = mapping.at(725., 52.7)
        expected = common.HopfSponge(725.-params['gap'], params['width'], params['strength']).profile(point.radii)
        sponge_error = float(np.max(abs(point.sponge_strength-expected)))
        mean_error = float(abs(matrices[0]-blocks[0]).max())
        oracle = GlobalRadial(point.radii, RadialConfig(ro=-1., beta=68., n=n, harmonics=H, omega=52.7, nonlinear=True),
                              outlet_alpha=0j, outlet_sponge=common.HopfSponge(725.-params['gap'], params['width'], params['strength']))
        raw = pack(old.linear(matrices, q, H)+old.convection(point, q))
        reference = oracle.evaluate_reference(pack(q), np.zeros_like(q[0]))
        relative = norm(raw-reference)/norm(reference)
        audit = solver.independent(mapping, y)
        require(sponge_error < 1e-14 and mean_error < 1e-14 and relative < 1e-10,
                f'Parameter propagation/original equation failure: {name}')
        require(audit['template_difference_relative'] < 1e-10, f'Independent audit failed: {name}')
        endpoint_tests[name] = dict(params=params, sponge_error=sponge_error, mean_matrix_error=mean_error,
                                    GlobalRadial_reference_relative=relative, independent=audit['template_difference_relative'])
    evidence['tests']['six_endpoint_propagation_and_original_equations'] = endpoint_tests
    mapping = Mapping(n, H, intervals, strength=56., width=20., gap=140.)
    point, matrices, _, dmatrices = mapping.at(725., 52.7, True)
    eps = .001
    plus, minus = mapping.at(725.+eps, 52.7)[1], mapping.at(725.-eps, 52.7)[1]
    derr = max(float(np.sqrt(abs((a-b)/(2*eps)-d).power(2).sum()))/
               max(float(np.sqrt(abs(d).power(2).sum())), 1e-30)
               for a, b, d in zip(plus, minus, dmatrices))
    qref = q[:, H+1]/common.mass_norm(point, q[:, H+1])
    corrector = solver.Corrector(mapping, qref, y, 'Rout', 725., budget=30.)
    J, _, _ = corrector.jacobian(y)
    direction = np.r_[rng.normal(size=len(y)-2)*1e-4, .03, .3]
    epsilon = 1e-4
    jv = J@direction
    fd = (corrector.residual(y+epsilon*direction)-corrector.residual(y-epsilon*direction))/(2*epsilon)
    jerr = norm(fd-jv)/norm(jv)
    require(derr < 1e-6 and jerr < 1e-6, 'Nonbaseline Rout derivative/JVP mismatch')
    evidence['tests']['nonbaseline_Rout_derivative_relative'] = derr
    evidence['tests']['nonbaseline_bordered_JVP_relative'] = jerr
    b = diagnostic(solver.SOURCE/'fields.npz', solver.BASE_PARAMS, 'baseline', save=False)
    expected = dict(omega=52.68143270236591, front_fig8=587.5735992526727,
                    front_2beta_absolute=580.0844129186014, g_q_570_600=.12397141950563219,
                    g_q_least_squares_570_600=.12434899093655402,
                    sponge_fraction_q=.6223279316921053, sponge_fraction_cylindrical_R3=.6401522770424705)
    errors = {key: abs(b[key]-value) for key, value in expected.items()}
    require(max(errors.values()) < 1e-11, 'Baseline diagnostics changed')
    censored = diagnostic(solver.SOURCE/'fields.npz', campaign.parameters('gap140', 1.), 'censor_fixture', save=False)
    require(censored['front_fig8'] is None and censored['front_search_truncated_by_sponge'] and
            censored['growth_fit_sponge_overlap'] == [585., 600.] and
            abs(censored['front_in_covered_pinch_domain_ignoring_sponge']-b['front_fig8']) < 1e-11,
            'Sponge-censored diagnostic lost coverage explanation')
    evidence['tests']['baseline_diagnostic_errors'] = errors
    evidence['tests']['gap140_baseline_field_fixture'] = dict(
        front_status=censored['front_status'], a=censored['a'],
        front_search_interval=censored['front_search_interval'],
        auxiliary_front=censored['front_in_covered_pinch_domain_ignoring_sponge'],
        growth_fit_sponge_overlap=censored['growth_fit_sponge_overlap'],
        interpretation='Unchanged source field used only as a definition/censoring fixture; no target solution')
    small = Mapping(n=8, H=2, intervals=80, gap=140.)
    # Reuse the random field: test gauge invariance and symmetric change bounds.
    rotated = unpack(y[:-2], small.nr, small.H, small.s)
    rotated *= np.exp(1j*np.arange(-H, H+1)*.37)[None, :, None]
    aligned, screen = solver.continuity(y, np.r_[pack(rotated), y[-2:]], small, source_A=1e-4)
    require(screen['accepted'] and norm(aligned-y)/norm(y) < 1e-12, 'Gauge continuity failed')
    zeroed = y.copy()
    zeroed[:-2] = 0
    require(not solver.continuity(y, zeroed, small, 1e-4)[1]['accepted'], 'Zero branch admitted')
    evidence['tests']['phase_and_nonzero_screen'] = 'passed'
    campaign.prepare()
    with tempfile.TemporaryDirectory(prefix='validation_', dir=OUT) as folder:
        folder = Path(folder)
        with campaign.lock(folder/'test.lock'):
            try:
                campaign.lock(folder/'test.lock')
            except RuntimeError:
                pass
            else:
                raise AssertionError('Duplicate process lock admitted')
        campaign.marker(folder, True, dict(accepted=True))
        campaign.marker(folder, False, dict(accepted=False))
        require((folder/'FAILED.json').exists() and not (folder/'DONE.json').exists(), 'Contradictory markers')
        job = folder/'checkpoint_test'
        _, result = solver.correct(small, y, job, 'fixture', budget=0.)
        require(not result['accepted'] and result.get('budget_stop') and (job/'checkpoint.npz').exists(), 'Budget checkpoint not preserved')
        with np.load(job/'checkpoint.npz') as saved:
            require(saved['a'].item() == 585. and saved['gap'].item() == 140., 'Checkpoint sponge metadata incorrect')
        _, resumed = solver.correct(small, y, folder/'resume_test', 'fixture', budget=0., resume=job/'checkpoint.npz')
        require(resumed['resumed_from'] == str(job/'checkpoint.npz') and not resumed['accepted'], 'Checkpoint resume failed')
        try:
            solver.correct(small, y, folder/'mismatch_test', 'wrong', budget=0., resume=job/'checkpoint.npz')
        except ValueError:
            pass
        else:
            raise AssertionError('Mismatched checkpoint admitted')
        timeout_job = folder/'timeout_test'
        timeout_job.mkdir()
        timeout_request = dict(campaign=solver.read(OUT/'request.json')['fingerprint'])
        write(timeout_job/'request.json', dict(timeout_request, fingerprint=solver.fingerprint(timeout_request)))
        solver.savez(timeout_job/'checkpoint.npz', preserved=True)
        with patch.object(campaign.subprocess, 'run', side_effect=subprocess.TimeoutExpired('fixture', 4800)) as mock:
            failure = campaign.run_attempt(timeout_job)
            require(mock.call_args.kwargs['timeout'] == 4800 and mock.call_args.args[0][0] == '/usr/bin/timeout', 'Hard timeout missing')
        require(not failure['accepted'] and failure['checkpoint_preserved'] and (timeout_job/'FAILED.json').exists(), 'Timeout terminal state incorrect')
        orphan = folder/'orphan_fixture'
        orphan.mkdir()
        (orphan/'attempt_0000').mkdir()
        (orphan/'attempt_0000'/'predictor.npz').write_bytes(b'uncommitted fixture')
        require(campaign.allocate_attempt(orphan, 0).name == 'attempt_0001' and
                campaign.allocate_attempt(orphan, 0).name == 'attempt_0002' and
                (orphan/'attempt_0000'/'predictor.npz').read_bytes() == b'uncommitted fixture',
                'Uncommitted orphan prevents restart or was overwritten')
        accepted = folder/'accepted_fixture'
        accepted.mkdir()
        req = dict(campaign=solver.read(OUT/'request.json')['fingerprint'],
                   sponge=small.params, n=small.n, H=small.H, intervals=small.intervals, Rout=725.)
        rid = solver.fingerprint(req)
        write(accepted/'request.json', dict(req, fingerprint=rid))
        solver.checkpoint(accepted/'fields.npz', small, y, qref, rid, accepted=True)
        rec = dict(accepted=True, request_fingerprint=rid, fields_sha256=solver.sha(accepted/'fields.npz'), omega=y[-2])
        write(accepted/'summary.json', rec)
        fields_sha, summary_sha = solver.sha(accepted/'fields.npz'), solver.sha(accepted/'summary.json')
        campaign.accepted_fields(accepted/'fields.npz', fields_sha, summary_sha)
        require(campaign.run_attempt(accepted)['accepted'], 'Valid accepted cache not reusable')
        field_bytes = (accepted/'fields.npz').read_bytes()
        (accepted/'fields.npz').write_bytes(field_bytes+b'alteration')
        try:
            campaign.accepted_fields(accepted/'fields.npz', fields_sha, summary_sha)
        except ValueError:
            pass
        else:
            raise AssertionError('Tampered committed field admitted')
        (accepted/'fields.npz').write_bytes(field_bytes)
        write(accepted/'summary.json', dict(rec, request_fingerprint='wrong'))
        try:
            campaign.run_attempt(accepted)
        except ValueError:
            pass
        else:
            raise AssertionError('Cached summary request mismatch admitted')
        write(accepted/'summary.json', rec)
        write(accepted/'request.json', dict(req, fingerprint='wrong'))
        try:
            campaign.run_attempt(accepted)
        except ValueError:
            pass
        else:
            raise AssertionError('Tampered request admitted')
        failure_root = folder/'controller_failure_fixture'
        with patch.object(campaign, 'OUT', failure_root), patch.object(campaign, 'run', side_effect=ValueError('fixture failure')), patch.object(sys, 'argv', ['fixture', '--run']):
            try:
                campaign.main()
            except ValueError:
                pass
            else:
                raise AssertionError('Controller fixture did not raise')
        require(solver.read(failure_root/'FAILED.json')['status'] == 'failed_exception', 'Controller exception missing terminal marker')
        previous_marker = solver.sha(failure_root/'FAILED.json')
        with campaign.lock(failure_root/'campaign.lock'):
            with patch.object(campaign, 'OUT', failure_root), patch.object(sys, 'argv', ['fixture', '--run']):
                try:
                    campaign.main()
                except campaign.LockBusy:
                    pass
                else:
                    raise AssertionError('Busy controller lock admitted')
        require(solver.sha(failure_root/'FAILED.json') == previous_marker, 'Lock conflict polluted terminal marker')
        require(not list(folder.rglob('*.tmp')), 'Atomic write debris')
    evidence['tests']['atomic_lock_checkpoint_resume_mismatch_timeout_markers'] = 'passed'
    evidence['tests']['uncommitted_orphan_attempt_recovery'] = 'passed'
    evidence['tests']['accepted_cache_field_summary_request_tamper_refusal'] = 'passed'
    evidence['tests']['controller_failure_marker_and_lock_conflict_preservation'] = 'passed'
    # Prepare is idempotent; changing the request refuses reuse without mutation.
    state, _ = campaign.prepare()
    checksum = solver.sha(OUT/'campaign.json')
    campaign.prepare()
    require(solver.sha(OUT/'campaign.json') == checksum, 'Prepare unexpectedly changed checkpoint state')
    original_manifest = campaign.manifest()
    with patch.object(campaign, 'manifest', return_value=dict(original_manifest, changed_algorithm=True)):
        try:
            campaign.prepare()
        except ValueError:
            pass
        else:
            raise AssertionError('Changed request fingerprint admitted')
    require(solver.sha(OUT/'campaign.json') == checksum, 'Mismatch refusal changed state')
    evidence['tests']['idempotent_prepare_and_code_request_mismatch_refusal'] = 'passed'
    evidence.update(passed=True, seconds=time.monotonic()-started,
                    code_hashes=campaign.manifest()['code_hashes'])
    write(OUT/'validation.json', evidence)
    print('D1 short validation passed; no production solve/factorization launched.')
    print(f'Baseline matrix difference {difference:.3g}; nonbaseline derivative {derr:.3g}; bordered JVP {jerr:.3g}.')
    print(f'Six parameter endpoints checked against original GlobalRadial; diagnostics and restart/lock/timeout fixtures passed in {evidence["seconds"]:.2f}s.')


if __name__ == '__main__':
    run()
