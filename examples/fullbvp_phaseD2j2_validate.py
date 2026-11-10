#!/usr/bin/env python3
"""Bounded D2j2 validation; no production factorization or scan."""
import os
os.environ['PYTHONDONTWRITEBYTECODE'] = '1'
import json
import subprocess
import sys
import tempfile
from unittest.mock import patch
from pathlib import Path
import fullbvp_phaseD2j2_campaign as c
import fullbvp_phaseD2j_validate as prior


def main():
    j = c.j
    result = dict(production_started=False, algorithms=prior.algorithms())
    plans = []
    for branch in ('upper720', 'upper725'):
        p = c.parameters(branch)
        shifts = [x[1] for x in p['shifts']]
        assert p['budget_seconds'] >= 28800 and p['center_budget_seconds'] <= 5400
        assert 7.5 in shifts and all(float(x) in shifts for x in range(45, 71, 5))
        assert len(p['seed']['centers']) == 10
        assert p['carryover_used_seconds'] == j.read(c.source_point(branch)/'budget.json')['used_seconds']
        cached = [j.read(x) for x in p['seed']['centers'].values()]
        circles = j.coverage(cached)
        pair = next(x for x in circles['adjacent'] if x['left'] == 5 and x['right'] == 10)
        assert pair['overlap'] > 0
        with tempfile.TemporaryDirectory(prefix='spec_', dir=c.AUDIT) as spec_tmp:
            name, command = c.save_spec(Path(spec_tmp), branch+'_N24', p)
        assert command[1] == str(c.ENTRY)
        plans.append(dict(branch=branch, reused_centers=10, shifts=shifts,
                          carryover_seconds=p['carryover_used_seconds'],
                          remaining_seconds=c.POINT_BUDGET-p['carryover_used_seconds'],
                          overlap_5i_10i=pair['overlap'], coverage=circles))
    assert c.TASK_MEMORY == 400_000_000_000
    groups = c.cpu_groups(2)
    reserved = set(sorted(os.sched_getaffinity(0))[:16])
    assert len(groups[0]) == len(groups[1]) == 16
    assert not set(groups[0]) & set(groups[1])
    assert not reserved & (set(groups[0]) | set(groups[1]))
    assert j.h.CONCURRENT_RUNS == 2 and j.h.THREADS == 16
    for forbidden in (c.LEGACY/'bad.json', c.OUT/'production_k/bad.json'):
        try:
            j.output_path(forbidden)
        except ValueError:
            pass
        else:
            raise AssertionError('legacy/D2k write allowed')
    with tempfile.TemporaryDirectory(prefix='D2j2_short_', dir=c.AUDIT) as tmp:
        point = Path(tmp)/'point'
        p = c.parameters('upper720')
        c.initialize_budget(p, point)
        first = j.read(point/'budget.json')
        c.initialize_budget(dict(p, carryover_used_seconds=0.), point)
        assert j.read(point/'budget.json') == first
        # The inherited launcher charges an interrupted reservation in full without spawning.
        exhausted = Path(tmp)/'exhausted'
        j.old.write(exhausted/'budget.json', dict(used_seconds=p['budget_seconds']-1.))
        j.old.write(exhausted/'ACTIVE.json', dict(reserved_seconds=2.))
        stopped = j._saved_launch(p, exhausted, command=['/must-not-spawn'])
        assert stopped['used_seconds'] == p['budget_seconds']+1. and not stopped['complete']
        assert not (exhausted/'ACTIVE.json').exists()
        proc = subprocess.Popen([sys.executable, '-c', 'pass'])
        assert c.wait_once(proc, 5.) == 0
        proc = subprocess.Popen([sys.executable, '-c', 'import signal; signal.pause()'])
        try:
            c.wait_once(proc, .05)
        except subprocess.TimeoutExpired:
            proc.kill()
            proc.wait()
        else:
            raise AssertionError('hard timeout did not fire')
    # Validate the same soft/hard limit inheritance used by a real center, in a tiny child.
    resource_probe = (
        "import resource; import fullbvp_phaseD2j2_campaign as c; "
        "resource.setrlimit(resource.RLIMIT_AS,(c.PARENT_MEMORY,c.WORKER_MEMORY)); "
        "c.j.h.control_limit(); "
        "resource.setrlimit(resource.RLIMIT_AS,(c.CENTER_MEMORY,c.CENTER_MEMORY)); "
        "assert resource.getrlimit(resource.RLIMIT_AS)==(c.CENTER_MEMORY,c.CENTER_MEMORY)"
    )
    env = dict(os.environ, PYTHONPATH=str(c.ROOT/'examples'))
    subprocess.run([sys.executable, '-B', '-c', resource_probe], check=True, env=env, timeout=10.)
    # Exercise publication, timeout cleanup, and retained prior centers through the real launcher.
    with tempfile.TemporaryDirectory(prefix='launcher_', dir=c.AUDIT) as tmp:
        point = Path(tmp)
        params = c.parameters('upper720')
        original_popen = subprocess.Popen
        children = []
        def fake_center(command, **kwargs):
            tiny = "import json,sys,pathlib; p=pathlib.Path(sys.argv[1]); j=json.loads(p.read_text()); r=dict(fingerprint='short-test',sigma=j['sigma'],complete=True,k=6,ncv=40); (pathlib.Path(j['target'])/'DONE.json').write_text(json.dumps(r))"
            child = original_popen([sys.executable, '-c', tiny, command[-1]], **kwargs)
            children.append(child)
            return child
        fd = os.open(point/'.lock', os.O_CREAT|os.O_RDWR, 0o600)
        try:
            with patch.object(c.subprocess, 'Popen', side_effect=fake_center):
                completed = c.run_center(params, point, 0, 7.5, j.old.d2.v1.Deadline(30.), fd)
            assert completed['complete'] and (point/'center_00/DONE.json').exists()
            assert completed['center_execution']['seconds'] <= 30.
            saved = (point/'center_00/DONE.json').read_bytes()
            def sleeper(command, **kwargs):
                child = original_popen([sys.executable, '-c', 'import signal; signal.pause()'], **kwargs)
                children.append(child)
                return child
            with patch.object(c.subprocess, 'Popen', side_effect=sleeper), patch.object(
                    c, 'wait_once', side_effect=subprocess.TimeoutExpired('short-test', .05)):
                failed = c.run_center(params, point, 1, 45., j.old.d2.v1.Deadline(30.), fd)
            assert not failed['complete'] and 'BudgetExpired' in failed['failure']
            assert (point/'center_01/FAILED.json').exists() and not (point/'center_01/DONE.json').exists()
            assert (point/'center_00/DONE.json').read_bytes() == saved
            assert all(child.returncode is not None for child in children)
        finally:
            os.close(fd)
    # Real saved +20i vectors are recertified on the original temporal operator.
    result['existing_20i'] = prior.existing_20i()
    result.update(plans=plans, ledger_no_reset=True, crash_reservation_charged=True,
                  hard_wait_success_and_timeout=True, center_launch_publication_and_cleanup=True, resource_limit_inheritance=True, isolated_output_paths=True,
                  task_memory_bound_bytes=c.TASK_MEMORY, cpu_groups=groups, reserved_for_D2k=sorted(reserved),
                  production_factorization_performed=False)
    j.old.write(c.AUDIT/'validation.json', result)
    print(json.dumps(dict(passed=True, reused_centers=sum(p['reused_centers'] for p in plans),
        recertified_pairs=result['existing_20i']['count'],
        original_residual_max=result['existing_20i']['original_residual_max'],
        production_started=False)))


if __name__ == '__main__':
    c.AUDIT.mkdir(parents=True, exist_ok=True)
    main()
