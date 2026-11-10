#!/usr/bin/env python3
"""Brief D3 validation: inherited equations, fixed Rout, source diagnostics and fault gates.

No production correction or full campaign is launched. Durable evidence is
written only inside D3/validation, under the same exclusive D3 lock.
"""
from __future__ import annotations

import os
for _key in ('OPENBLAS_NUM_THREADS', 'OMP_NUM_THREADS', 'MKL_NUM_THREADS',
             'NUMEXPR_NUM_THREADS', 'VECLIB_MAXIMUM_THREADS', 'BLIS_NUM_THREADS'):
    os.environ[_key] = '1'
os.environ['OMP_THREAD_LIMIT'] = '1'
os.environ['CUDA_VISIBLE_DEVICES'] = ''
os.environ['MPLCONFIGDIR'] = '/tmp/fullbvp_phaseD3_matplotlib'

import argparse
import ast
import copy
import io
import subprocess
import sys
import time
from pathlib import Path
from unittest.mock import patch

sys.dont_write_bytecode = True
ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT/'examples'), str(ROOT/'src')]
import numpy as np
import fullbvp_phaseD3_campaign as d3


def check(out):
    from fullbvp_phaseB_solver import Mapping, Corrector, common, pack, unpack, checks
    started = time.monotonic()
    out = d3.checked_out(out)
    evidence = dict(status='running', full_campaign_launched=False)
    protected_names = ('codex_stdout.log', 'rollout_path.txt', 'codex.pid', 'guard.json',
                       'IMPLEMENTATION_BRIEF.md', 'dispatch_audit.json')
    protected = {name:d3.sha(d3.DEFAULT_OUT/name) for name in protected_names
                 if (d3.DEFAULT_OUT/name).exists() and name not in ('codex_stdout.log', 'guard.json')}
    with d3.Exclusive(d3.DEFAULT_OUT) as lock:
        for path in (Path(__file__), Path(d3.__file__)):
            ast.parse(path.read_text())
        evidence['syntax'] = 'passed'
        subprocess.run(['/usr/bin/bash','-n',str(d3.DEFAULT_OUT/'commands.sh')],check=True)
        evidence['commands_shell_syntax'] = 'passed'
        evidence['inherited_small_grid_equations'] = checks(out)

        # Changing field amplitude changes E1 but cannot change the fixed-Rout border.
        mapping = Mapping(8, 2, 24)
        rng = np.random.default_rng(20261030)
        q = unpack(rng.normal(size=mapping.nr*5*mapping.s)*1e-3, mapping.nr, 2, mapping.s)
        y = np.r_[pack(q), 53., 725.]
        p, *_ = mapping.at(y[-1], y[-2])
        qref = q[:,3]/common.mass_norm(p, q[:,3])
        system = Corrector(mapping, qref, y, 'Rout', 725., budget=30.)
        border = float(system.residual(y)[-1])
        changed = y.copy()
        changed[:-2] *= 1.2
        scaled_border = float(system.residual(changed)[-1])
        displaced = y.copy()
        displaced[-1] += .1
        displaced_border = float(system.residual(displaced)[-1])
        assert border == scaled_border == 0. and abs(displaced_border-.01)<1e-12
        evidence['fixed_Rout_E1_free'] = dict(passed=True, original_border=border,
              scaled_field_border=scaled_border, displaced_Rout_border=displaced_border,
              imposed_energy_constraint=False)

        from fullbvp_phaseC_solver import checkpoint, mapped_transfer
        # A known polynomial exercises both interpolations and old-Rout-then-xi stretch.
        wall = mapping.op.grid.xi
        factor = (1+mapping.xi+mapping.xi**2)[:,None,None,None]
        wall_factor = (1+wall+.2*wall**2)[None,None,None,:]
        harmonic = np.exp(1j*np.arange(-2,3)*.17)[None,:,None,None]
        analytic = np.broadcast_to(factor*wall_factor*harmonic*1e-3,
                                   (mapping.nr,5,4,mapping.m)).copy().reshape(mapping.nr,5,mapping.s)
        analytic_y = np.r_[pack(analytic),53.,725.]
        fixture = out/'transfer_fixture.npz'
        checkpoint(fixture, mapping, analytic_y, qref, accepted=True)
        refined = Mapping(10,2,32)
        transferred = mapped_transfer(fixture,refined)
        stretched = mapped_transfer(fixture,refined,new_Rout=730.)
        expected_field = np.broadcast_to((1+refined.xi+refined.xi**2)[:,None,None,None]*
            (1+refined.op.grid.xi+.2*refined.op.grid.xi**2)[None,None,None,:]*harmonic*1e-3,
            (refined.nr,5,4,refined.m)).copy().reshape(refined.nr,5,refined.s)
        transfer_error = float(np.max(abs(unpack(transferred[:-2],refined.nr,2,refined.s)-expected_field)))
        stretch_error=float(np.max(abs(transferred[:-1]-stretched[:-1])))
        assert transfer_error<1e-13 and stretch_error<1e-13, (transfer_error,stretch_error)
        assert transferred[-1]==725. and stretched[-1]==730.
        evidence['grid_transfer'] = dict(passed=True,max_absolute_polynomial_error=transfer_error,
             max_stretch_field_change=stretch_error,
             old_Rout=725.,stretched_Rout=730.,physical_extrapolation_used=False)
        grid_expected=dict(n=8,H=2,intervals=24,Rout=725.,omega=53.)
        d3.actual_grid(fixture,grid_expected)
        with np.load(fixture) as saved:
            fixture_arrays={key:saved[key].copy() for key in saved.files}
        grid_faults=[]
        for label, value in (('field',analytic[:,:-1]),('intervals',25),('Rout',726.),('omega',54.)):
            mutated=dict(fixture_arrays);mutated[label]=value
            bad_file=out/f'bad_grid_{label}.npz'
            np.savez_compressed(bad_file,**mutated)
            try:
                d3.actual_grid(bad_file,grid_expected)
                raise AssertionError('Bad accepted grid admitted: '+label)
            except ValueError:
                grid_faults.append(label)
        evidence['grid_provenance_faults_rejected']=grid_faults

        source = d3.source_identity()
        source_row = d3.read(d3.SOURCE/'summary.json')
        rejected = []
        for label, location, value in (
            ('accepted_boolean_only', ('independent',), {}),
            ('relative_field', ('relative_field_inf',), 1e-9),
            ('feedback', ('residual_over_fundamental_feedback',), 1e-4),
            ('phase', ('phase',), 1e-9),
            ('branch_constraint', ('branch_constraint',), 1e-9),
            ('independent_residual', ('independent','relative_field_inf'), 1e-9),
            ('template_difference', ('independent','template_difference_relative'), 1e-9),
            ('nonfinite_independent', ('independent','relative_field_inf'), float('nan')),
            ('harmonic_unresolved', ('highest_to_fundamental',), .02),
            ('forced', ('independent','forcing'), True)):
            row = copy.deepcopy(source_row)
            owner = row if len(location)==1 else row[location[0]]
            owner[location[-1]] = value
            assert not d3.numerical_gate(row), label
            rejected.append(label)
        evidence['strict_acceptance_faults_rejected'] = rejected
        source_point = d3.point_from_record(source_row, d3.SOURCE, 'existing_725_H6_validation',
                                            'validation_existing_source', source, out, imported=True)
        expected = dict(omega=52.68143270236591, front_fig8=587.573599,
                        g_q_fit_570_600=.124348991)
        errors = {key:abs(source_point[key]-value) for key,value in expected.items()}
        assert errors['omega'] < 1e-10 and errors['front_fig8'] < 1e-6 and errors['g_q_fit_570_600'] < 1e-9
        evidence['existing_source_diagnostic_regression'] = dict(passed=True, errors=errors,
              metrics={k:source_point.get(k) for k in d3.METRICS}, source=source,
              interpretation='validation of existing accepted source only; no new refined root computed')

        assert len(d3.MATRIX)==6
        for R, radial in ((725.,1300),(730.,1320)):
            subset = [r for r in d3.MATRIX if r['Rout']==R]
            assert {(r['n'],r['intervals']) for r in subset} == {(32,800),(40,800),(32,radial)}
            assert (R-400)/radial == .25
        evidence['independent_resolution_matrix'] = list(d3.MATRIX)

        fake = dict(fingerprint='test', rows=[dict(r, status='accepted', metrics={},
                    delta_from_same_Rout_H6_baseline={}, actual_grid={}, fields_sha256='test') for r in d3.MATRIX])
        assert d3.completion_gate(fake, 'test')
        bad_cases = []
        for label in ('prepared_only', 'five_rows', 'duplicate_row', 'pending_refinement', 'missing_comparison', 'fingerprint_changed'):
            bad = copy.deepcopy(fake)
            if label=='prepared_only': bad['rows']=[]
            elif label=='five_rows': bad['rows'].pop()
            elif label=='duplicate_row': bad['rows'][5]=bad['rows'][4]
            elif label=='pending_refinement': bad['rows'][5]['status']='pending'
            elif label=='missing_comparison': bad['rows'][5]['delta_from_same_Rout_H6_baseline']=None
            elif label=='fingerprint_changed': bad['fingerprint']='changed'
            assert not d3.completion_gate(bad, 'test'), label
            bad_cases.append(label)
        evidence['false_completion_cases_rejected'] = bad_cases

        # Inject fast worker exits and deadlines through the real immutable-attempt code.
        lifecycle=out/f'lifecycle_{time.time_ns()}'
        lifecycle.mkdir(exist_ok=True)
        campaign=d3.Campaign.__new__(d3.Campaign)
        campaign.out,campaign.lock=lifecycle,lock
        campaign.identity='fault_injection_no_production_solver'
        campaign.source=source
        campaign.state=dict(attempts=[])
        launched=[]
        class FakeWorker:
            def __init__(self, command, **_kwargs):
                request_path=Path(command[-1]);request=d3.read(request_path)
                launched.append(request)
                work=request_path.parent/'newton';work.mkdir()
                np.savez_compressed(work/'checkpoint.npz',y=np.array([1.]),qref=np.array([1.]))
                self.stdout=io.BytesIO(b'synthetic worker failure; checkpoint durable\n')
                self.returncode=None;self.pid=123456789;self.calls=0
            def wait(self,timeout=None):
                self.calls+=1
                if len(launched)==2 and self.calls==1:
                    raise d3.subprocess.TimeoutExpired('synthetic_worker',timeout)
                self.returncode=-15 if len(launched)==2 else 1
                return self.returncode
        with patch.object(d3.subprocess,'Popen',FakeWorker),patch.object(d3.os,'killpg') as kill:
            first,first_outcome=campaign.attempt(dict(d3.MATRIX[0]),[source_point])
            second,second_outcome=campaign.attempt(dict(d3.MATRIX[0]),[source_point])
            assert not first_outcome['accepted'] and not second_outcome['accepted']
            assert launched[0]['resume'] is None
            assert launched[1]['resume']==str(first/'newton/checkpoint.npz')
            assert second_outcome['failure']=='4800 s worker deadline'
            assert kill.call_count==1 and (first/'newton/checkpoint.npz').exists()
        evidence['failed_and_resume_lifecycle']=dict(passed=True,
            immutable_attempts=2,previous_checkpoint_selected=True,hard_deadline_rejected=True,
            solver_subprocesses_launched=0,fixture_checkpoint_is_synthetic=True)

        # Drive the actual execute/marker path with a deliberately incomplete campaign.
        incomplete=out/'incomplete_marker_case'
        class IncompleteCampaign:
            def __init__(self,case_out,_lock):
                self.out=case_out;self.identity='synthetic_partial_matrix'
                self.state=dict(status='prepared',matrix={r['key']:dict(r,status='pending') for r in d3.MATRIX})
            def save(self): d3.write(self.out/'campaign.json',self.state)
            def finalize(self):
                d3.write(self.out/'comparisons.json',dict(fingerprint=self.identity,rows=fake['rows'][:1]))
            def run(self): self.save();self.finalize()
        with patch.object(d3,'Campaign',IncompleteCampaign):
            try:
                d3.execute(incomplete)
                raise AssertionError('Partial campaign received DONE')
            except RuntimeError as exc:
                assert 'Completion guard' in str(exc)
        assert (incomplete/'FAILED.json').exists() and not (incomplete/'DONE.json').exists()
        assert len(d3.read(incomplete/'FAILED.json')['pending'])==6
        evidence['partial_matrix_marker_lifecycle']=dict(passed=True,FAILED_written=True,DONE_written=False,
                                                        pending_rows=6,solver_subprocesses_launched=0)

        try:
            with d3.Exclusive(d3.DEFAULT_OUT):
                raise AssertionError('Nested D3 lock was admitted')
        except RuntimeError:
            evidence['exclusive_lock'] = 'passed; concurrent acquisition refused'
        log = d3.TailLog(out/'bounded_log_test.log')
        log.append(b'a'*(d3.LOG_BYTES+100))
        log.append(b'FINAL_TAIL')
        assert log.path.stat().st_size == d3.LOG_BYTES and log.path.read_bytes().endswith(b'FINAL_TAIL')
        evidence['bounded_log'] = dict(passed=True, bytes=log.path.stat().st_size)
        for name, original in protected.items():
            assert d3.sha(d3.DEFAULT_OUT/name)==original, name
        evidence['protected_artifacts_unchanged'] = list(protected)
        evidence.update(status='passed', seconds=time.monotonic()-started,
                        production_corrections_launched=0, CPU_threads=1, GPU=False)
        d3.write(out/'checks.json', evidence)
    return evidence


if __name__ == '__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--out',type=Path,default=d3.DEFAULT_OUT/'validation')
    args=parser.parse_args()
    result=check(args.out)
    print(f'D3 validation: {result["status"]}; {result["seconds"]:.2f}s; production corrections=0')
