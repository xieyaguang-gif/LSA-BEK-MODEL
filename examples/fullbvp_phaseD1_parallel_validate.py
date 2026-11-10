#!/usr/bin/env python3
"""Short routing/restart/source diagnostics checks for D1b; no BVP solve."""
from __future__ import annotations

import os
os.environ['PYTHONDONTWRITEBYTECODE'] = '1'
import tempfile
import time
from pathlib import Path
from unittest.mock import patch

import fullbvp_phaseD1_parallel as parallel
from fullbvp_phaseD1_parallel import solver, shared, PARALLEL, OLD_OUT, ROOT


def require(test, message):
    if not test:
        raise AssertionError(message)


def run():
    started = time.monotonic()
    before = parallel.checked_old_identity()
    tests = {}
    writes, locks = [], []
    original_write, original_savez, original_lock = shared.write, shared.savez, shared.lock
    raw_diag = parallel.RAW_DIAGNOSTIC

    def record_write(path, value):
        writes.append(str(Path(path).resolve()))
        return original_write(path, value)

    def record_savez(path, **values):
        writes.append(str(Path(path).resolve()))
        return original_savez(path, **values)

    def record_lock(path):
        locks.append(str(Path(path).resolve()))
        return original_lock(path)

    with tempfile.TemporaryDirectory(prefix='validation_', dir=PARALLEL) as folder:
        folder = Path(folder)
        snapshots = {}
        with patch.object(shared, 'write', record_write), patch.object(shared, 'savez', record_savez), patch.object(shared, 'lock', record_lock):
            for branch in parallel.TARGETS:
                out = parallel.bind(branch, folder/branch)
                state, baseline = shared.prepare()
                require(list(state['branches']) == [branch], 'Cross-branch state contamination')
                require(state['branches'][branch]['accepted_fields'] == str(solver.SOURCE/'fields.npz'), 'Independent branch inherited a different endpoint')
                request = solver.read(out/'request.json')
                require(request['request']['D1b_routing']['output'] == str(out), 'Manifest output routing wrong')
                require(str(parallel.ENTRY.relative_to(ROOT)) in request['request']['code_hashes'], 'Adapter not in code identity')
                require((out/'diagnostics/baseline.json').exists() and not state['production_launched'], 'Captured diagnostic default wrote outside isolated output')
                snapshot = solver.sha(out/'campaign.json')
                shared.prepare()
                require(solver.sha(out/'campaign.json') == snapshot, 'Idempotent prepare changed state')
                snapshots[branch] = snapshot
                with shared.lock(out/'campaign.lock'):
                    try:
                        shared.lock(out/'campaign.lock')
                    except shared.LockBusy:
                        pass
                    else:
                        raise AssertionError('Duplicate controller lock admitted')
                orphan = out/'branches'/branch
                orphan.mkdir(parents=True)
                (orphan/'attempt_0000').mkdir()
                require(shared.allocate_attempt(orphan, 0).name == 'attempt_0001', 'Orphan blocks restart')
                # Worker dispatch is inspected without starting any process.
                job = shared.allocate_attempt(orphan, 0)
                request = dict(campaign=request['fingerprint'], branch=branch)
                solver.write(job/'request.json', dict(request, fingerprint=solver.fingerprint(request)))
                def fake_worker(command, **kwargs):
                    require(command[4] == str(parallel.ENTRY), 'Subprocess escaped to original D1 entry')
                    require(command[-2:] == ['--worker', str(job)] and kwargs['timeout'] == 4800., 'Worker path/hard budget wrong')
                    solver.write(job/'summary.json', dict(accepted=False, failure='isolated dispatch fixture'))
                    return type('Result', (), {'returncode': 1})()
                with patch.object(shared.subprocess, 'run', side_effect=fake_worker):
                    require(not shared.run_attempt(job)['accepted'], 'Dispatch fixture unexpectedly accepted')
                shared.marker(job, True, dict(accepted=True))
                shared.marker(job, False, dict(accepted=False))
                require((job/'FAILED.json').exists() and not (job/'DONE.json').exists(), 'Contradictory isolated terminal markers')
                with patch.object(parallel, 'RAW_DIAGNOSTIC', wraps=raw_diag) as wrapped:
                    parallel.routed_diagnostic(solver.SOURCE/'fields.npz', solver.BASE_PARAMS, 'default_capture_fixture', save=False)
                    require(wrapped.call_args.kwargs['out'] == out, 'Diagnostic out default capture not overridden')
                try:
                    parallel.routed_diagnostic(solver.SOURCE/'fields.npz', solver.BASE_PARAMS, 'escape', out=OLD_OUT, save=False)
                except ValueError:
                    pass
                else:
                    raise AssertionError('Original diagnostic output path admitted')
            require(solver.sha(folder/'gap120/campaign.json') == snapshots['gap120'], 'Preparing gap140 modified gap120 state')
        require(all(Path(path).is_relative_to(folder) for path in writes+locks), 'Fixture wrote or acquired a lock outside isolated paths')
        tests['isolated_writes_and_locks'] = dict(write_paths=len(writes), lock_paths=len(locks), all_under_validation_directory=True)
        # The real new-worker CLI infers branch from its own request; no numerical worker executes.
        fake = PARALLEL/'gap120/branches/gap120/routing_fixture'
        with patch.object(parallel.solver, 'read', return_value={'branch': 'gap120'}), patch.object(shared, 'worker', return_value=0) as worker:
            require(parallel.main(['--worker', str(fake)]) == 0 and worker.call_args.args[0] == fake, 'Worker CLI branch routing failed')
            require(shared.OUT == PARALLEL/'gap120' and shared.__file__ == str(parallel.ENTRY), 'Worker module globals leaked to original D1')
        tests['worker_subprocess_entry_and_default_argument_routing'] = 'passed'
        tests['independent_source_idempotence_orphans_locks_markers'] = 'passed'
    # All source-based outputs remain under parallel/intermediate.
    value, baseline = parallel.intermediate_diagnostic()
    require(value['actual_checkpoint_width'] == 58.357749999999996 and not value['accepted_width80_endpoint'], 'Intermediate was relabelled as width80 endpoint')
    require(abs(baseline['omega']-52.68143270236591) < 1e-12 and abs(baseline['front_fig8']-587.5735992526727) < 1e-11, 'Baseline diagnostic changed')
    tests['verified_actual_width_intermediate_and_baseline'] = {key: value[key] for key in ('omega', 'actual_checkpoint_width', 'front_fig8', 'front_2beta_absolute', 'g_q_570_600', 'sponge_fraction_q', 'sponge_fraction_cylindrical_R3', 'endpoint_minus_baseline')}
    after = parallel.checked_old_identity()
    require(before == after, 'Pinned old numerical source was edited')
    evidence = dict(passed=True, seconds=time.monotonic()-started, production_launched=False,
                    tests=tests, original_pinned_identity=after,
                    parent_state_hash_comparison='Not used: live controller may legitimately change its own state. Isolation was checked by intercepted write and lock paths.',
                    adapter_sha256=solver.sha(parallel.ENTRY), validation_sha256=solver.sha(Path(__file__)))
    solver.write(PARALLEL/'validation.json', evidence)
    print(f'D1b isolation/restart/worker-routing/source diagnostic validation passed in {evidence["seconds"]:.2f}s; no BVP solve.')
    print('Actual intermediate width='+str(value['actual_checkpoint_width'])+'; omega='+str(value['omega']))


if __name__ == '__main__':
    run()
