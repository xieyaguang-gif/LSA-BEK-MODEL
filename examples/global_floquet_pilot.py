#!/usr/bin/env python3
"""Bounded, foreground global-Floquet validation and three-point pilot.

Use the project .venv Python. Each invocation is bounded to <=1770 seconds,
16 MKL threads, one BLAS thread and 100 GB address space. No production scan.
"""
import os
for _name in ('OPENBLAS_NUM_THREADS', 'OMP_NUM_THREADS', 'MKL_NUM_THREADS',
              'NUMEXPR_NUM_THREADS', 'VECLIB_MAXIMUM_THREADS', 'BLIS_NUM_THREADS'):
    os.environ[_name] = '1'
os.environ['PYTHONDONTWRITEBYTECODE'] = '1'

import argparse
import gc
import hashlib
import importlib.util
import json
from pathlib import Path
import resource
import sys
import time
import traceback
sys.dont_write_bytecode = True

import numpy as np
from scipy import sparse

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT/'src'))
sys.path.insert(0, str(ROOT/'examples'))
from llsa.secondary.global_floquet import GlobalFloquet
import fullbvp_phaseD2_stability as original
import fullbvp_phaseD2d_precision as precision
from fullbvp_phaseB_solver import Mapping
from fullbvp_phaseD2f_diagnose import equilibrate

OUT = ROOT/'results/FULLBVP_PHASE2_20261104/global_floquet'
D2 = ROOT/'results/fullbvp_global_20261030/phaseD/D2'
UPPER = D2/'production_v4/run_D2f_baseline/upper720_H6'
RADIAL = D2/'production_v6/radial/upper720_H6'
LIMIT = 100_000_000_000


def sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as f:
        for chunk in iter(lambda: f.read(1024*1024), b''):
            h.update(chunk)
    return h.hexdigest()


def write(name, data):
    dest = OUT/name
    tmp = dest.with_name(dest.name+'.new')
    tmp.write_text(json.dumps(data, indent=2, allow_nan=False)+'\n')
    os.replace(tmp, dest)


def save_modes(name, spectrum, floquet):
    target = OUT/name
    tmp = target.with_name(target.name+'.new')
    with tmp.open('wb') as stream:
        np.savez(stream, sigma=spectrum.values, right_vectors=spectrum.right_vectors,
                 left_vectors=(spectrum.left_vectors if spectrum.left_vectors is not None
                               else np.empty((floquet.size, 0), complex)),
                 indices=floquet.indices, radii=floquet.problem.radii)
    os.replace(tmp, target)
    return dict(path=str(target), sha256=sha(target))


class Deadline:
    def __init__(self, seconds):
        self.start = time.monotonic()
        self.end = self.start+seconds
    def check(self):
        if time.monotonic() > self.end:
            raise TimeoutError('foreground numerical deadline reached')
        if precision.memory()['VmRSS'] > LIMIT:
            raise MemoryError('100 GB process RSS limit reached')


class ReusedFactor(precision.PardisoComplex):
    """Unchanged D2 LU policy, with its same factors used for adjoint solves."""
    def __init__(self, matrix, deadline):
        super().__init__(matrix, threads=16, deadline=deadline,
                         policy=dict(precision.DEFAULT_POLICY))
    def _stats(self):
        super()._stats()
        mem = precision.memory()
        predicted = int(self.iparm[16])*1024
        if not getattr(self, '_estimate_checked', False):
            projected = mem['VmRSS']+1.3*max(predicted, 0)+2_000_000_000
            if projected > LIMIT:
                raise MemoryError(f'factor memory estimate {projected:.0f} exceeds 100 GB')
            self._estimate_checked = True
            self.info['floquet_memory_projection_bytes'] = projected
    def solve_adjoint(self, rhs):
        previous = int(self.iparm[11])
        try:
            self.iparm[11] = 1  # MKL complex conjugate-transposed solve.
            result = self._call(33, np.asarray(rhs, complex))/self.scale
            self.info['solve_calls'] += 1
            return result
        finally:
            self.iparm[11] = previous


class CoordinateFactor:
    """D2 cos/sin coordinates, still 2K+1 independent complex coordinates.

    S is deliberately the original (nonunitary) D2 change of basis. Physical
    operators and returned vectors retain the signed complex representation.
    """
    def __init__(self, matrix, floquet, deadline):
        nh, s, nr = floquet.nh, floquet.s, floquet.nr
        K = (nh-1)//2
        T = np.zeros((nh, nh), complex)
        T[K, 0] = 1
        for k in range(1, K+1):
            T[K+k, 2*k-1] = T[K-k, 2*k-1] = 1
            T[K+k, 2*k] = 1j
            T[K-k, 2*k] = -1j
        inverse = np.linalg.inv(T)
        self.S = sparse.kron(sparse.eye(nr), sparse.kron(sparse.csr_matrix(T), sparse.eye(s)), format='csr')
        self.inverse = sparse.kron(sparse.eye(nr), sparse.kron(sparse.csr_matrix(inverse), sparse.eye(s)), format='csr')
        transformed = (self.inverse@matrix@self.S).tocsr()
        transformed.eliminate_zeros()
        balanced, self.rows, self.cols = equilibrate(transformed, 6)
        self.factor = ReusedFactor(balanced, deadline)
        self.info = self.factor.info
        self.info.update(coordinates='invertible D2 cos/sin; every coordinate remains complex',
                         physical_matrix_nnz=int(matrix.nnz),
                         transformed_matrix_nnz=int(transformed.nnz),
                         equilibration_steps=6,
                         row_scale_range=[float(self.rows.min()), float(self.rows.max())],
                         column_scale_range=[float(self.cols.min()), float(self.cols.max())],
                         transformation_condition_number=float(np.linalg.cond(T)))
    def solve(self, rhs):
        return self.S@(self.cols*self.factor.solve(self.rows*(self.inverse@rhs)))
    def solve_adjoint(self, rhs):
        return self.inverse.conj().T@(self.rows*self.factor.solve_adjoint(self.cols*(self.S.conj().T@rhs)))
    def close(self):
        self.factor.close()


def load_source(point=UPPER, deadline=None):
    record = json.loads((point/'corrected.json').read_text())
    path = Path(record['fields'])
    digest = sha(path)
    if digest != record['fields_sha256']:
        raise RuntimeError('D2 source hash mismatch')
    with np.load(path) as d:
        q, r = d['field'].copy(), d['radii'].copy()
        n, H, omega = int(d['n']), int(d['H']), float(d['omega'])
        if not bool(d['accepted']) or (n, H) != (24, 4):
            raise RuntimeError('expected accepted actual N24/H4 D2 field')
    mapping = Mapping(n, H, len(r)-1)
    temporal = original.Temporal(mapping, q, r[-1], omega, deadline)
    p = temporal.p
    return p, q, temporal, dict(path=str(path), sha256=digest,
        corrected_json=str(point/'corrected.json'), corrected_sha256=sha(point/'corrected.json'),
        n=n, H=H, nr=len(r), dr=float(r[1]-r[0]), Rout=float(r[-1]), omega=omega,
        historical_directory_label_is_not_resolution=True,
        branch_identity_proven=record['branch'].get('branch_identity_proven', False))


def source_snapshot():
    paths = list((ROOT/'src/llsa/secondary').glob('*.py'))+list((ROOT/'examples').glob('*.py'))
    paths = [p for p in paths if p.name not in ('global_floquet.py', 'global_floquet_pilot.py')]
    return {str(p.relative_to(ROOT)):sha(p) for p in paths}


def check_snapshot():
    baseline = json.loads((OUT/'executor_immutable_baseline.json').read_text())
    changed = [p for p, h in baseline.items() if sha(ROOT/p) != h]
    if changed:
        raise RuntimeError('preexisting source changed: '+str(changed))
    return dict(files_checked=len(baseline), changed=changed)


def diagnostics(name):
    records = []
    def save(record, arrays):
        records.append(record)
        write(name+'.json', dict(records=records))
        if arrays is not None:
            target = OUT/(name+'_rejected.npz')
            tmp = target.with_name(target.name+'.new')
            with tmp.open('wb') as f:
                np.savez(f, **arrays)
            os.replace(tmp, target)
    return save


def small_tests(deadline):
    spec = importlib.util.spec_from_file_location('floquet_checks', ROOT/'tests/test_global_floquet.py')
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    report = module.run_small_validation()
    A = sparse.csr_matrix(np.array([[2+1j, .3-.2j, 0], [.2j, 3-.7j, 1], [.1, -.4j, 4+.8j]]))
    f = ReusedFactor(A, deadline)
    try:
        b = np.array([.3+1j, 2-.2j, -.4j])
        x = f.solve_adjoint(b)
        error = float(np.linalg.norm(A.conj().T@x-b)/np.linalg.norm(b))
        report['pardiso_shared_factor_adjoint_relative'] = error
        report['passed'] = report['passed'] and error < 1e-12
    finally:
        f.close()
    return report


def radial_comparison(coarse_values):
    # Actual independently corrected dr=.5 field at the same fixed Rout=720.
    corrected = json.loads((RADIAL/'corrected.json').read_text())
    fields = Path(corrected['fields'])
    if sha(fields) != corrected['fields_sha256']:
        raise RuntimeError('radial source hash mismatch')
    done = RADIAL/'center_00/DONE.json'
    records = json.loads(done.read_text())['eigenpairs']
    fine = np.array([complex(*r['value']) for r in records if r['accepted']])
    pairs = []
    for z in coarse_values:
        w = fine[np.argmin(abs(fine-z))]
        pairs.append(dict(dr1=[float(z.real), float(z.imag)], dr_half=[float(w.real), float(w.imag)],
                          absolute_change=float(abs(z-w)), unstable=bool(z.real>1e-8)))
    from_test = importlib.util.spec_from_file_location('floquet_radial_checks', ROOT/'tests/test_global_floquet.py')
    module = importlib.util.module_from_spec(from_test)
    from_test.loader.exec_module(module)
    lower = module.lower_radial_validation()
    return dict(passed=lower['passed'], unstable_lower_grid_sensitivity=lower,
        kind='actual corrected upper720 dr=1 versus dr=.5, plus unstable lower fixed-amplitude dr comparison',
        dr1=1., dr_half=float(corrected['actual_dr']), comparisons=pairs,
        fine_field=str(fields), fine_field_sha256=sha(fields),
        fine_spectrum_reference=str(done), fine_spectrum_sha256=sha(done),
        limitation='dr=.5 eigenvalues are preserved D2 solves; no unstable sample in the near-zero shift; no spatial convergence claim')


def validate(deadline):
    started = time.monotonic()
    report = dict(passed=False, small=small_tests(deadline), timestamp_utc=time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime()))
    write('VALIDATION.json', report)
    if not report['small']['passed']:
        raise RuntimeError('small independent tests failed; full solve and pilot gated')
    p, q, temporal, source = load_source(deadline=deadline)
    f = GlobalFloquet(p, q, check=deadline.check)
    rng = np.random.default_rng(924)
    real = rng.normal(size=temporal.size)
    signed = original.lift(real, temporal.nr, temporal.H, temporal.s)
    actual = original.lower((f.J@signed.ravel()).reshape(signed.shape))
    expected = temporal.J@real
    action_error = float(np.linalg.norm(actual-expected)/np.linalg.norm(expected))
    phase = q*(1j*np.arange(-p.H, p.H+1))[None, :, None]
    phase_residual = f.residual(0j, phase.ravel())
    report.update(source=source, actual_D2_action_relative=action_error,
                  actual_D2_action_threshold=1e-10, phase_mode_residual=phase_residual,
                  phase_mode_threshold=1e-8, matrix_shape=list(f.J.shape), matrix_nnz=int(f.J.nnz))
    write('VALIDATION.json', report)
    if action_error > 1e-10 or phase_residual > 1e-8:
        raise RuntimeError('actual D2 action or phase neutral check failed')
    spectrum = f.shift_invert(.05, k=4, ncv=40, tol=1e-10,
        maxiter=150, factor_factory=lambda A:CoordinateFactor(A, f, deadline), left=True,
        inverse_policy='backward', extended_refinement=False,
        inverse_diagnostic=diagnostics('inverse_validation_attempt5'))
    known = complex(-.00013544089779849583, .000219391806609919)
    nearest = min(abs(spectrum.values-known), default=float('inf'))
    phase_nearest = min(abs(spectrum.values), default=float('inf'))
    report.update(actual_spectrum=dict(records=spectrum.records, stats=spectrum.stats),
        known_D2_reference=[known.real, known.imag], known_D2_error=float(nearest),
        known_D2_threshold=1e-7, phase_eigenvalue_abs=float(phase_nearest),
        phase_eigenvalue_threshold=1e-7, radial_convergence=radial_comparison(spectrum.values))
    report['numerical_policy'] = dict(normwise_accept=1e-12, componentwise_accept=1e-10,
        rhs_relative_is_diagnostic=True, equilibration_steps=6,
        inverse_policy_source=str(UPPER/'center_00/DONE.json'), eigenpair_accept=1e-8,
        inverse_residual_precision='complex128, same-source D2f backward-error policy',
        eigenpair_residual_precision='complex128',
        action_accept=1e-10, coordinates='invertible D2 cos/sin, all complex freedoms retained',
        deadline_granularity='Python operations and before/after each native PARDISO phase; no native interruption')
    report['numerical_policy'].update(right_linear_policy='D2f',left_linear_policy='D2g',
        left_normwise_accept=1e-12,left_rhs_relative_accept=1e-5,left_componentwise_is_diagnostic=True)
    report['passed'] = bool(nearest<1e-7 and phase_nearest<1e-7 and
        report['radial_convergence']['passed'] and
        len(spectrum.values)==4 and spectrum.stats['arpack_complete'] and
        spectrum.stats['left_arpack_complete'] and
        all(r['accepted'] and r.get('left_accepted', False) for r in spectrum.records))
    report.update(seconds=time.monotonic()-started, memory=precision.memory(),
                  unchanged_sources=check_snapshot())
    write('VALIDATION.json', report)
    print(json.dumps(dict(stage='validate', passed=report['passed'], seconds=report['seconds'],
                         action_relative=action_error, known_error=nearest, phase_abs=phase_nearest)))
    if not report['passed']:
        raise RuntimeError('actual source spectrum failed numerical validation')


def pilot(fraction, deadline):
    validation = json.loads((OUT/'VALIDATION.json').read_text())
    if not validation['passed']:
        raise RuntimeError('all mandatory validation must pass before pilot')
    started = time.monotonic()
    p, q, _, source = load_source(deadline=deadline)
    beta_hat = fraction*p.config.beta
    f = GlobalFloquet(p, q, beta_hat, check=deadline.check)
    result = dict(fraction=fraction, beta_hat=beta_hat, source=source, centers=[],
                  complete=False, exhaustive_spectrum=False, physical_conclusion=False,
                  harmonics=f.indices.tolist(), shape=list(f.J.shape), matrix_nnz=int(f.J.nnz))
    name = 'PILOT_'+str(fraction).replace('.', 'p')+'.json'
    shifts = [.05+0j, .05-1j*fraction*f.omega]
    wanted = 2
    for center_index, shift in enumerate(shifts):
        record = dict(shift=[shift.real, shift.imag], complete=False)
        remaining = deadline.end-time.monotonic()
        if remaining <= 30:
            record.update(status='not_attempted_budget', error='global foreground budget exhausted')
            result['centers'].append(record)
            continue
        center_deadline = Deadline(min(780., remaining-10.))
        f.check = center_deadline.check
        try:
            s = f.shift_invert(shift, k=wanted, ncv=24, tol=1e-8, maxiter=60,
                              factor_factory=lambda A:CoordinateFactor(A, f, center_deadline), left=True,
                              inverse_policy='backward', extended_refinement=False,
                              inverse_diagnostic=diagnostics('inverse_pilot_'+str(fraction)+'_'+str(shift.imag)))
            complete = bool(len(s.records)==wanted and s.stats['arpack_complete'] and
                s.stats['left_arpack_complete'] and
                all(r['accepted'] and r.get('left_accepted', False) for r in s.records))
            record.update(complete=complete, status='converged' if complete else 'partial_or_nonconverged',
                          expected_k=wanted, ncv=24, center_budget_seconds=780.,
                          records=s.records, stats=s.stats,
                          modes=save_modes(name.replace('.json', '_center_'+str(center_index)+'.npz'), s, f),
                          accepted_right=sum(r['accepted'] for r in s.records),
                          accepted_left=sum(r.get('left_accepted', False) for r in s.records))
        except Exception as exc:
            record.update(status='failed_or_budget_expired',
                          error=type(exc).__name__+': '+str(exc), traceback=traceback.format_exc())
        result['centers'].append(record)
        result.update(seconds=time.monotonic()-started, memory=precision.memory(),
                      unchanged_sources=check_snapshot())
        write(name, result)
        gc.collect()
    samples = [r for c in result['centers'] for r in c.get('records', []) if r['accepted']]
    unique = []
    for r in sorted(samples, key=lambda r:-r['value'][0]):
        if not any(abs(complex(*r['value'])-complex(*other['value']))<2e-7 for other in unique):
            unique.append(r)
    samples = unique
    result.update(complete=len(result['centers'])==len(shifts) and all(c['complete'] for c in result['centers']),
                  sampled_largest_real=samples[0]['value'][0] if samples else None,
                  accepted_samples=len(samples), leading_samples=samples[:6],
                  seconds=time.monotonic()-started, memory=precision.memory())
    write(name, result)
    print(json.dumps(dict(stage='pilot', fraction=fraction, complete=result['complete'],
        samples=len(samples), sampled_largest_real=result['sampled_largest_real'], seconds=result['seconds'])))


def complete_left(fraction, deadline):
    """Keep the original right search; certify its saved vectors and add adjoints."""
    started = time.monotonic()
    if not json.loads((OUT/'VALIDATION.json').read_text())['passed']:
        raise RuntimeError('mandatory validation gate has not passed')
    name = 'PILOT_'+str(fraction).replace('.', 'p')+'.json'
    previous = json.loads((OUT/name).read_text())
    previous_sha = sha(OUT/name)
    tag = 'before_left_'+time.strftime('%Y%m%dT%H%M%S', time.gmtime())
    archive = name.replace('.json', '_'+tag+'.json')
    write(archive, previous)
    p, q, _, source = load_source(deadline=deadline)
    f = GlobalFloquet(p, q, fraction*p.config.beta, check=deadline.check)
    if (previous['source']['sha256']!=source['sha256'] or previous['beta_hat']!=f.beta_hat
            or previous['harmonics']!=f.indices.tolist() or previous['shape']!=list(f.J.shape)):
        raise RuntimeError('saved right spectrum source/beta_hat/K/shape mismatch')
    result = dict(previous)
    result['centers'] = []
    result.update(reused_right=True, previous_result=dict(path=str(OUT/archive),
        sha256=sha(OUT/archive), original_result_sha256=previous_sha),
        right_linear_policy='D2f', left_linear_policy='D2g')
    for index, original_center in enumerate(previous['centers']):
        shift = complex(*original_center['shift'])
        record = dict(shift=original_center['shift'], complete=False,
            records=original_center.get('records',[]), modes=original_center.get('modes'),
            accepted_right=original_center.get('accepted_right',0),
            accepted_left=original_center.get('accepted_left',0),
            original_stats=original_center.get('stats',{}),
            left_completion_policy=dict(name='D2g',normwise_accept=1e-12,
                rhs_relative_accept=1e-5,componentwise_is_diagnostic=True),
            original_center_seconds=original_center.get('stats',{}).get('seconds'),
            original_right_arnoldi_seconds=original_center.get('stats',{}).get('right_arnoldi_seconds'),
            original_right_time_note='earlier runs recorded center total, not isolated right time',
            reused_right_source=original_center.get('modes'),
            original_right_arpack_complete=original_center.get('stats',{}).get('arpack_complete',False))
        if original_center.get('complete'):
            result['centers'].append(original_center)
            continue
        remaining = deadline.end-time.monotonic()
        if remaining<=30:
            record.update(status='not_attempted_budget',error='left completion budget exhausted')
            result['centers'].append(record)
            continue
        center_deadline = Deadline(min(780., remaining-10.))
        f.check = center_deadline.check
        try:
            mode_path = Path(original_center['modes']['path'])
            if sha(mode_path)!=original_center['modes']['sha256']:
                raise RuntimeError('saved right mode checksum mismatch')
            with np.load(mode_path) as data:
                values, vectors = data['sigma'].copy(), data['right_vectors'].copy()
                if (not np.array_equal(data['indices'],f.indices) or
                    not np.array_equal(data['radii'],p.radii)):
                    raise RuntimeError('saved right mode grid/harmonic indices mismatch')
            if len(values)!=2 or len(original_center.get('records',[]))!=2:
                raise RuntimeError('saved center does not contain requested two right samples')
            recorded_values = np.array([complex(*r['value']) for r in original_center['records']])
            if not np.array_equal(values,recorded_values):
                raise RuntimeError('saved mode values disagree with original center record')
            s = f.shift_invert(shift, k=2, ncv=24, tol=1e-10, maxiter=60,
                factor_factory=lambda A:CoordinateFactor(A,f,center_deadline),
                reused_right=(values,vectors),
                reused_right_complete=record['original_right_arpack_complete'],
                adjoint_inverse_policy='normwise_rhs_relative', adjoint_inverse_accept=1e-5,
                inverse_diagnostic=diagnostics('inverse_left_repair_'+str(fraction)+'_'+str(shift.imag)))
            original_diagnostics = OUT/('inverse_pilot_'+str(fraction)+'_'+str(shift.imag)+'.json')
            if original_diagnostics.exists():
                original_calls=json.loads(original_diagnostics.read_text())['records']
                original_right=[r for r in original_calls if not r['adjoint']]
                s.stats['right_linear_policy']=dict(name='D2f',normwise_accept=1e-12,
                    componentwise_accept=1e-10,rhs_relative_is_diagnostic=True)
                for key,field in [('normwise_backward','normwise_backward'),
                    ('componentwise_backward','componentwise_backward'),('rhs_relative','float64_relative')]:
                    s.stats['right_maximum_'+key]=max((r[field] for r in original_right),default=0.)
                record['original_right_inverse_diagnostics']=dict(path=str(original_diagnostics),sha256=sha(original_diagnostics))
            complete=bool(len(s.records)==2 and s.stats['arpack_complete'] and
                s.stats['left_arpack_complete'] and all(r['accepted'] and r['left_accepted'] for r in s.records))
            record.update(complete=complete,status='converged' if complete else 'partial_or_nonconverged',
                expected_k=2,ncv=24,left_arnoldi_tol=1e-10,records=s.records,stats=s.stats,
                modes=save_modes(name.replace('.json','_'+tag+'_center_'+str(index)+'.npz'),s,f),
                accepted_right=sum(r['accepted'] for r in s.records),
                accepted_left=sum(r['left_accepted'] for r in s.records))
        except Exception as exc:
            record.update(status='failed_or_budget_expired',error=type(exc).__name__+': '+str(exc),traceback=traceback.format_exc())
        result['centers'].append(record)
        result.update(left_repair_seconds=time.monotonic()-started,
            seconds=previous['seconds']+time.monotonic()-started,memory=precision.memory())
        write(name,result)
        gc.collect()
    samples=[]
    for r in sorted([r for c in result['centers'] for r in c.get('records',[]) if r['accepted']],key=lambda x:-x['value'][0]):
        if not any(abs(complex(*r['value'])-complex(*a['value']))<2e-7 for a in samples):samples.append(r)
    mem=precision.memory()
    result.update(complete=len(result['centers'])==2 and all(c['complete'] for c in result['centers']),
        accepted_samples=len(samples),leading_samples=samples[:6],
        sampled_largest_real=samples[0]['value'][0] if samples else None,
        initial_seconds=previous['seconds'],left_repair_seconds=time.monotonic()-started,
        seconds=previous['seconds']+time.monotonic()-started,memory=mem,
        combined_peak_rss_bytes=max(previous.get('combined_peak_rss_bytes',0),
            previous['memory']['process_peak_rss_bytes'],mem['process_peak_rss_bytes']),
        unchanged_sources=check_snapshot())
    write(name,result)
    print(json.dumps(dict(stage='left',fraction=fraction,complete=result['complete'],
        samples=len(samples),left_repair_seconds=result['left_repair_seconds'],total_seconds=result['seconds'])))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--stage', choices=['small', 'validate', 'pilot', 'left'], required=True)
    parser.add_argument('--fraction', type=float, choices=[.1, .25, .5])
    parser.add_argument('--seconds', type=float, default=1700.)
    args = parser.parse_args()
    if not 0<args.seconds<=1770:
        raise ValueError('single foreground execution must be <=1770 seconds')
    resource.setrlimit(resource.RLIMIT_AS, (LIMIT, LIMIT))
    baseline = OUT/'executor_immutable_baseline.json'
    if not baseline.exists():
        write(baseline.name, source_snapshot())
    deadline = Deadline(args.seconds)
    try:
        if args.stage == 'small':
            report = small_tests(deadline)
            write('SMALL_VALIDATION.json', report)
            print(json.dumps(dict(stage='small', passed=report['passed'], tests_run=report['tests_run'])))
            if not report['passed']:
                raise RuntimeError('small tests failed')
        elif args.stage == 'validate':
            validate(deadline)
        else:
            if args.fraction is None:
                raise ValueError('--fraction required for pilot')
            if args.stage=='left':complete_left(args.fraction,deadline)
            else:pilot(args.fraction, deadline)
    except Exception as exc:
        write('EXECUTION_ERROR_'+args.stage+'.json', dict(error=repr(exc),
            traceback=traceback.format_exc(), seconds=time.monotonic()-deadline.start,
            memory=precision.memory(), unchanged_sources=check_snapshot()))
        raise


if __name__ == '__main__':
    main()
