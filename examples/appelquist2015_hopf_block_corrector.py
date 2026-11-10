#!/usr/bin/env python3
"""Bounded matrix-free H2 Hopf corrector; only original equations are accepted.

The large calculation never assembles a full real Jacobian or a dense border.
Small-grid explicit matrices are used exclusively as algebraic test oracles.
"""
from __future__ import annotations
import argparse
import json
import os
from pathlib import Path
import resource
import signal
import sys
import time

for _key in ('OPENBLAS_NUM_THREADS', 'OMP_NUM_THREADS', 'MKL_NUM_THREADS', 'NUMEXPR_NUM_THREADS'):
    os.environ[_key] = '1'
import numpy as np
from scipy import sparse
from scipy.sparse.linalg import LinearOperator, gmres, splu

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT/'src'), str(ROOT/'examples')]
import appelquist2015_hopf_common as common
from llsa.secondary.radial import pack_field, unpack_field

OUT = ROOT/'results/appelquist2015_hopf_20261029/corrector'


def write(path, value):
    path.write_text(json.dumps(value, indent=2, allow_nan=False)+'\n')


def norm(x):
    return float(np.linalg.norm(x))


def relative(x, reference):
    return norm(x)/max(norm(reference), 1e-300)


def rss():
    return resource.getrusage(resource.RUSAGE_SELF).ru_maxrss/1024**2


def pack(q):
    """Vectorized original packing: mean, Re1, Im1, Re2, Im2 per radius."""
    return np.stack((q[:, 2].real, q[:, 3].real, q[:, 3].imag,
                     q[:, 4].real, q[:, 4].imag), axis=1).ravel()


def unpack(x, nr, s):
    z = np.asarray(x).reshape(nr, 5, s)
    q = np.empty((nr, 5, s), complex)
    q[:, 2] = z[:, 0]
    q[:, 3] = z[:, 1]+1j*z[:, 2]
    q[:, 4] = z[:, 3]+1j*z[:, 4]
    q[:, 1] = q[:, 3].conj()
    q[:, 0] = q[:, 4].conj()
    return q


def mass_row(p, q):
    z = np.asarray(q).reshape(len(p.radii), 4, p.ops[0].m)
    row = np.zeros_like(z)
    row[:, :3] = (z[:, :3].conj()@p.ops[0].grid.mass)*common.radial_weights(p.radii)[:, None, None]
    return row.ravel()


def real_constraint(p, row, imaginary=False):
    z = np.zeros((len(p.radii), 5, p.s))
    row = row.reshape(len(p.radii), p.s)
    z[:, 1] = row.imag if imaginary else row.real
    z[:, 2] = row.real if imaginary else -row.imag
    return z.ravel()


class Factor:
    def __init__(self, a, label):
        start = time.monotonic()
        self.a = a.tocsc()
        self.scale = np.maximum(np.asarray(abs(a).max(axis=1).toarray()).ravel(), 1e-30)
        self.lu = splu((sparse.diags(1/self.scale)@a).tocsc())
        self.info = dict(label=label, size=a.shape[0], matrix_nnz=int(a.nnz),
                         factor_nnz=int(self.lu.L.nnz+self.lu.U.nnz),
                         seconds=time.monotonic()-start, peak_rss_gib=rss())

    def solve(self, b):
        return self.lu.solve(np.asarray(b)/self.scale)


class System:
    def __init__(self, p, qref, energy, epsilon):
        self.p = p
        self.nr, self.s = len(p.radii), p.s
        self.wc, self.ac = p.config.omega, p.outlet_sponge.start
        self.qref = qref.copy()
        self.ell = mass_row(p, qref)/common.mass_inner(p, qref, qref).real
        self.energy, self.epsilon = float(energy), float(epsilon)
        self.blocks, self.b = zip(*(common.complex_block(p, j) for j in range(3)))
        self.bdiag = self.b[1].diagonal().reshape(self.nr, self.s)
        scales = [np.maximum(np.asarray(abs(a).max(axis=1).toarray()).ravel(), 1e-30).reshape(self.nr, self.s)
                  for a in self.blocks]
        self.fscale = epsilon*np.stack((scales[0], scales[1], scales[1], scales[2], scales[2]), axis=1).ravel()
        self.size = self.nr*5*self.s+2
        self.jv_calls = 0

    def conv(self, q):
        qr = (self.p.D@q.reshape(self.nr, -1)).reshape(q.shape)
        return self.p._convection_field(q, qr)

    def delta_profile(self, a):
        return 28*(common.smooth_step_and_derivative((self.p.radii-a)/40)[0]
                   -common.smooth_step_and_derivative((self.p.radii-self.ac)/40)[0])

    def linear(self, q, omega, a):
        out = np.zeros_like(q)
        ds = self.delta_profile(a)[:, None]*self.bdiag
        for j in range(3):
            out[:, 2+j] = (self.blocks[j]@q[:, 2+j].ravel()).reshape(self.nr, self.s)
            if j or self.p.outlet_sponge.damp_mean:
                out[:, 2+j] += ds*q[:, 2+j]
            if j:
                out[:, 2+j] -= 1j*j*(omega-self.wc)*self.bdiag*q[:, 2+j]
                out[:, 2-j] = out[:, 2+j].conj()
        return out

    def columns(self, q, a):
        frequency, parameter = np.zeros_like(q), np.zeros_like(q)
        derivative = -.7*common.smooth_step_and_derivative((self.p.radii-a)/40)[1]
        for j in range(3):
            if j or self.p.outlet_sponge.damp_mean:
                parameter[:, 2+j] = derivative[:, None]*self.bdiag*q[:, 2+j]
            if j:
                frequency[:, 2+j] = -1j*j*self.bdiag*q[:, 2+j]
                frequency[:, 2-j] = frequency[:, 2+j].conj()
                parameter[:, 2-j] = parameter[:, 2+j].conj()
        return frequency, parameter

    def residual(self, y, scaled=True):
        q = unpack(y[:-2], self.nr, self.s)
        f = pack(self.linear(q, y[-2], y[-1])+self.conv(q))
        phase = (self.ell@q[:, 3].ravel()).imag/self.epsilon
        energy = common.mass_inner(self.p, q[:, 3], q[:, 3]).real/self.energy-1
        return np.r_[f/self.fscale if scaled else f, phase, energy]

    def jacobian(self, y):
        q = unpack(y[:-2], self.nr, self.s)
        qnorm = norm(q)
        fw, fa = self.columns(q, y[-1])
        energy_row = 2*mass_row(self.p, q[:, 3])/self.energy

        def multiply(v):
            self.jv_calls += 1
            dq = unpack(v[:-2], self.nr, self.s)
            vn = norm(dq)
            if vn:
                h = max(qnorm, 1e-30)/vn
                dn = (self.conv(q+h*dq)-self.conv(q-h*dq))/(2*h)
            else:
                dn = np.zeros_like(q)
            out = self.linear(dq, y[-2], y[-1])+dn+v[-2]*fw+v[-1]*fa
            phase = (self.ell@dq[:, 3].ravel()).imag/self.epsilon
            energy = (energy_row@dq[:, 3].ravel()).real
            return np.r_[pack(out)/self.fscale, phase, energy]
        return LinearOperator((self.size, self.size), matvec=multiply, dtype=float)


class BlockInverse:
    def __init__(self, system, derivative, shift=.05):
        self.system, self.derivative = system, complex(derivative)
        if abs(self.derivative.real) < 1e-12:
            raise ValueError('Nonzero Hopf transversality is required')
        self.shift = shift
        self.f0 = Factor(system.blocks[0], 'A0')
        self.f2 = Factor(system.blocks[2], 'A2')
        self.f1 = Factor(system.blocks[1]+shift*system.b[1], 'A1_shifted_sparse')
        self.u = system.b[1]@system.qref.ravel()
        self.z = self.f1.solve(self.u)
        self.schur = system.ell@self.z
        if abs(self.schur) < 1e-14:
            raise ValueError('Degenerate scalar Schur complement')
        self.info = dict(factors=[self.f0.info, self.f1.info, self.f2.info], shift=shift,
                         schur=[float(self.schur.real), float(self.schur.imag)],
                         derivative=[self.derivative.real, self.derivative.imag])

    def border(self, b, c):
        y = self.f1.solve(b)
        eta = (self.system.ell@y-c)/self.schur
        return y-self.z*eta, eta

    def at(self, y):
        s = self.system
        fw, fa = s.columns(unpack(y[:-2], s.nr, s.s), y[-1])

        def solve(rhs):
            b = unpack(rhs[:-2]*s.fscale, s.nr, s.s)
            c = s.epsilon*(rhs[-1]/2+1j*rhs[-2])
            first, eta = self.border(b[:, 3].ravel(), c)
            da = eta.real/(s.epsilon*self.derivative.real)
            dw = self.derivative.imag*da-eta.imag/s.epsilon
            q = np.zeros_like(b)
            q[:, 3] = first.reshape(s.nr, s.s)
            q[:, 1] = q[:, 3].conj()
            q[:, 2] = self.f0.solve((b[:, 2]-dw*fw[:, 2]-da*fa[:, 2]).real.ravel()).reshape(s.nr, s.s)
            q[:, 4] = self.f2.solve((b[:, 4]-dw*fw[:, 4]-da*fa[:, 4]).ravel()).reshape(s.nr, s.s)
            q[:, 0] = q[:, 4].conj()
            return np.r_[pack(q), dw, da]
        return LinearOperator((s.size, s.size), matvec=solve, dtype=float)


def solve_step(s, inverse, y, rhs, rtol=1e-8, restart=40, cycles=3):
    j, pre = s.jacobian(y), inverse.at(y)
    hist = []
    start = time.monotonic()
    right = LinearOperator(j.shape, matvec=lambda v: j@(pre@v), dtype=float)
    z, info = gmres(right, rhs, rtol=rtol, atol=0., restart=restart, maxiter=cycles,
                    callback=lambda r: hist.append(float(r)), callback_type='pr_norm')
    step = pre@z
    defect = j@step-rhs
    raw = defect.copy(); raw[:-2] *= s.fscale
    original_rhs = rhs.copy(); original_rhs[:-2] *= s.fscale
    report = dict(gmres_info=int(info), iterations=len(hist), residual_history=hist,
                  scaled_original_linear_relative=relative(defect, rhs),
                  unscaled_original_linear_relative=relative(raw, original_rhs),
                  unscaled_field_linear_inf=float(abs(raw[:-2]).max()),
                  seconds=time.monotonic()-start, jv_calls=s.jv_calls, peak_rss_gib=rss())
    return step, report


def checks():
    out = OUT/'algebra_checks'; out.mkdir(parents=True, exist_ok=True)
    if (out/'summary.json').exists():
        (out/'summary.json').rename(out/f'prior_summary_{time.time_ns()}.json')
    start = time.monotonic()
    rng = np.random.default_rng(20261029)
    p = common.make_problem(8, 30., 620., harmonics=2, nonlinear=True, omega=53.)
    nr, size = len(p.radii), p.s
    qref = rng.normal(size=(nr, size))+1j*rng.normal(size=(nr, size))
    qref /= common.mass_norm(p, qref)
    eps = 1e-3
    q = np.zeros((nr, 5, size), complex)
    q[:, 3] = eps*qref; q[:, 1] = q[:, 3].conj()
    q[:, 2] = eps**2*rng.normal(size=(nr, size))
    q[:, 4] = eps**2*(rng.normal(size=(nr, size))+1j*rng.normal(size=(nr, size)))
    q[:, 0] = q[:, 4].conj()
    s = System(p, qref, eps**2, eps)
    y = np.r_[pack(q), p.config.omega+.012, p.outlet_sponge.start-.03]
    report = dict(kind='algebra_only_not_physical_branch', n=8, dr=30., real_unknowns=len(y),
                  pack_error=float(abs(pack(q)-pack_field(q)).max()),
                  unpack_error=float(abs(unpack(pack(q), nr, size)-unpack_field(pack(q), nr, 2, size)).max()))
    independent = common.make_problem(8, 30., y[-1], harmonics=2, nonlinear=True, omega=y[-2])
    f, explicit = independent.evaluate(y[:-2], np.zeros_like(q[0]), jacobian=True)
    report['original_residual_error'] = relative(s.residual(y, False)[:-2]-f, f)
    fw, fa = s.columns(q, y[-1])
    rows = np.stack((real_constraint(p, s.ell, True)/eps,
                     real_constraint(p, 2*mass_row(p, q[:, 3])/s.energy)))
    augmented = sparse.bmat([[explicit, sparse.csc_matrix(np.stack((pack(fw), pack(fa)), axis=1))],
                             [sparse.csr_matrix(rows), sparse.csc_matrix((2, 2))]], format='csc')
    augmented_scaled = sparse.diags(np.r_[1/s.fscale, 1., 1.])@augmented
    j = s.jacobian(y)
    v = rng.normal(size=len(y)); v[:-2] *= eps
    report['matrix_free_vs_explicit_relative'] = relative(j@v-augmented_scaled@v, augmented_scaled@v)
    report['centered_difference_h'] = []
    for h in (1e-2, 1e-3, 1e-4):
        fd = (s.residual(y+h*v)-s.residual(y-h*v))/(2*h)
        report['centered_difference_h'].append(dict(h=h, relative=relative(j@v-fd, j@v)))
    inv = BlockInverse(s, -.005+.02j)
    b = rng.normal(size=qref.size)+1j*rng.normal(size=qref.size); c = .3+.2j
    x, eta = inv.border(b, c)
    border = sparse.bmat([[inv.f1.a, sparse.csc_matrix(inv.u[:, None])],
                          [sparse.csr_matrix(s.ell[None, :]), None]], format='csc')
    direct_border = splu(border).solve(np.r_[b, c])
    report['scalar_schur_vs_explicit_border'] = relative(np.r_[x, eta]-direct_border, direct_border)
    report['scalar_schur_original_residual'] = relative(border@np.r_[x, eta]-np.r_[b, c], np.r_[b, c])
    pre = inv.at(y)
    rr = rng.normal(size=len(y)); dz = pre@rr
    dq = unpack(dz[:-2], nr, size)
    expected_eta = eps*(inv.derivative*dz[-1]-1j*dz[-2])
    block1_residual = inv.f1.a@dq[:, 3].ravel()+inv.u*expected_eta
    expected_b = unpack(rr[:-2]*s.fscale, nr, size)[:, 3].ravel()
    report['complex_real_border_equation_relative'] = relative(block1_residual-expected_b, expected_b)
    report['complex_real_constraint_error'] = float(abs(s.ell@dq[:, 3].ravel()-eps*(rr[-1]/2+1j*rr[-2])))
    # A known exact step avoids conflating conditioning with the algebra test.
    rhs = augmented_scaled@v
    direct = splu(augmented_scaled.tocsc()).solve(rhs)
    step, linear = solve_step(s, inv, y, rhs, rtol=1e-13, restart=50, cycles=4)
    report['gmres'] = linear
    report['gmres_step_vs_direct_relative'] = relative(step-direct, direct)
    report['direct_original_linear_relative'] = relative(augmented@direct-augmented@v, augmented@v)
    report['factors'] = inv.info
    report['seconds'] = time.monotonic()-start
    report['passed'] = bool(report['pack_error'] == 0 and report['unpack_error'] == 0
        and report['original_residual_error'] < 1e-10 and report['matrix_free_vs_explicit_relative'] < 1e-10
        and report['scalar_schur_vs_explicit_border'] < 1e-8 and report['scalar_schur_original_residual'] < 1e-8
        and report['complex_real_border_equation_relative'] < 1e-8 and report['complex_real_constraint_error'] < 1e-8
        and linear['gmres_info'] == 0 and linear['scaled_original_linear_relative'] < 1e-8
        and report['gmres_step_vs_direct_relative'] < 1e-5)
    write(out/'summary.json', report)
    print(json.dumps({k: report[k] for k in ('passed', 'seconds', 'matrix_free_vs_explicit_relative',
                                          'scalar_schur_vs_explicit_border', 'gmres_step_vs_direct_relative')}), flush=True)
    return report


def run_real(args):
    check = json.loads((OUT/'algebra_checks/summary.json').read_text())
    if not check['passed']:
        raise RuntimeError('Small-grid checks did not pass; large calculation is forbidden')
    out = OUT/'N32_dr0.5'; out.mkdir(parents=True, exist_ok=False)
    start = time.monotonic()
    with np.load(args.responses) as d:
        data = {k: d[k].copy() for k in d.files}
    meta = json.loads(str(data['metadata_json']))
    pred = meta['prediction']
    eps = float(pred['epsilon'])
    ac, wc = float(data['a_c']), float(data['omega_c'])
    n, dr = int(data['n']), float(data['dr'])
    p = common.make_problem(n, dr, ac, harmonics=2, nonlinear=True,
                            damp_mean=bool(data['damp_mean']), omega=wc)
    q = np.zeros((len(p.radii), 5, p.s), complex)
    q[:, 3] = eps*data['q1']+eps**3*data['q13_physical']
    # Align the complete initial fundamental to the fixed velocity-mass phase.
    ell = mass_row(p, data['q1'])
    rotation = np.exp(-1j*np.angle(ell@q[:, 3].ravel()))
    q[:, 3] *= rotation; q[:, 1] = q[:, 3].conj()
    q[:, 2] = eps**2*data['q0']
    q[:, 4] = eps**2*data['q2']*rotation**2; q[:, 0] = q[:, 4].conj()
    energy = common.mass_inner(p, q[:, 3], q[:, 3]).real
    s = System(p, data['q1'], energy, eps)
    y = np.r_[pack(q), wc+pred['omega_shift'], ac+pred['delta_a']]
    initial = y.copy()
    scale = float(abs(y[:-2]).max())
    cubic = eps**3*data['Cvec']
    cubic_norm = norm(cubic)
    cubic_inf = float(abs(cubic).max())
    aa = common.sponge_start_matrix(p, 1)
    left = data['left_mode'].ravel()
    derivative = np.vdot(left, aa@data['q1'].ravel())/np.vdot(left, s.b[1]@data['q1'].ravel())
    report = dict(status='checkpoint_no_new_root', responses_source=str(args.responses),
                  n=n, dr=dr, epsilon=eps, target_energy=float(energy), nominal_epsilon_squared=eps**2,
                  energy_definition='integral q1_velocity^H Mz q1_velocity dR; target is actual predicted field energy',
                  reference_a=ac, reference_omega=wc, initial_a=float(y[-1]), initial_omega=float(y[-2]),
                  initial_field_inf=scale, cubic_feedback_l2=cubic_norm, cubic_feedback_inf=cubic_inf,
                  phase_rotation=[rotation.real, rotation.imag], iterations=[],
                  auxiliary_forcing=False, damp_mean=bool(data['damp_mean']), accepted=False,
                  numerical_budget_seconds=args.budget, prior_check_seconds=check['seconds'],
                  command=' '.join(sys.argv))
    def checkpoint():
        report.update(seconds=time.monotonic()-start, peak_rss_gib=rss(), jv_calls=s.jv_calls,
                      current_a=float(y[-1]), current_omega=float(y[-2]))
        write(out/'summary.json', report)
        np.savez_compressed(out/'checkpoint.npz', field=unpack(y[:-2], s.nr, s.s),
                            omega=y[-2], a=y[-1], initial=initial, target_energy=energy,
                            accepted=False, n=n, dr=dr)

    try:
        inverse = BlockInverse(s, derivative)
        report['preconditioner'] = inverse.info
        for iteration in range(args.newton+1):
            f = s.residual(y)
            raw = f[:-2]*s.fscale
            fq = unpack(raw, s.nr, s.s)
            residual_vs_cubic = norm(fq[:, 3])/cubic_norm
            row = dict(iteration=iteration, a=float(y[-1]), omega=float(y[-2]),
                       scaled_residual_l2=norm(f), original_residual_inf=float(abs(raw).max()),
                       original_residual_relative_initial=float(abs(raw).max()/scale),
                       phase=float(f[-2]), energy=float(f[-1]),
                       fundamental_residual_l2_over_cubic=residual_vs_cubic,
                       full_packed_residual_l2_over_cubic=norm(raw)/cubic_norm,
                       original_residual_inf_over_cubic=float(abs(raw).max()/cubic_inf))
            report['iterations'].append(row)
            checkpoint()
            print(json.dumps(row), flush=True)
            if (row['original_residual_relative_initial'] < 1e-7 and max(abs(f[-2:])) < 1e-8
                    and row['full_packed_residual_l2_over_cubic'] < 1e-3
                    and row['original_residual_inf_over_cubic'] < 1e-3):
                report['status'] = 'candidate_pending_independent_original_equation_validation'
                break
            if iteration == args.newton:
                report['stop_reason'] = 'Newton iteration cap'
                break
            step, linear = solve_step(s, inverse, y, -f, rtol=1e-7, restart=35, cycles=2)
            row['linear'] = linear
            if linear['scaled_original_linear_relative'] > .2:
                report['stop_reason'] = 'Krylov original linear residual insufficient for a Newton step'
                break
            accepted = False
            for back in range(6):
                trial = y+2.**(-back)*step
                if not (590. < trial[-1] < 650. and abs(trial[-2]-wc) < .5):
                    continue
                ft = s.residual(trial)
                if norm(ft) < (1-1e-4*2.**(-back))*norm(f):
                    y = trial; row['step_length'] = 2.**(-back); accepted = True
                    break
            if not accepted:
                report['stop_reason'] = 'Original-equation line search failed'
                break
        if report['status'] == 'candidate_pending_independent_original_equation_validation':
            independent = common.make_problem(n, dr, y[-1], harmonics=2, nonlinear=True,
                                               damp_mean=bool(data['damp_mean']), omega=y[-2])
            finalq = unpack(y[:-2], s.nr, s.s)
            original = independent.evaluate(pack_field(finalq), np.zeros_like(finalq[0]))
            constraints = s.residual(y)[-2:]
            check_final = dict(original_residual_inf=float(abs(original).max()),
                               relative_to_initial_field=float(abs(original).max()/scale),
                               packed_residual_l2_over_cubic=norm(original)/cubic_norm,
                               residual_inf_over_cubic=float(abs(original).max()/cubic_inf),
                               phase=float(constraints[0]), energy=float(constraints[1]),
                               fundamental_mass_norm=common.mass_norm(independent, finalq[:, 3]),
                               neutral_defect_subtracted=False, rebuilt_at_final_parameters=True)
            report['independent_validation'] = check_final
            good = (check_final['relative_to_initial_field'] < 1e-7
                    and check_final['packed_residual_l2_over_cubic'] < 1e-3
                    and check_final['residual_inf_over_cubic'] < 1e-3
                    and max(abs(constraints)) < 1e-8 and check_final['fundamental_mass_norm'] > .5*eps)
            report['accepted'] = bool(good)
            report['status'] = 'original_full_H2_BVP_small_branch_root' if good else 'independent_validation_failed_no_new_root'
            if good:
                np.savez_compressed(out/'fields.npz', field=finalq, omega=y[-2], a=y[-1],
                                    n=n, dr=dr, radii=p.radii, z=p.ops[0].grid.z,
                                    target_energy=energy, epsilon=eps, accepted=True)
    except (TimeoutError, MemoryError) as exc:
        report['stop_reason'] = type(exc).__name__+': '+str(exc)
        report['status'] = 'budget_stop_checkpoint_no_new_root'
    finally:
        checkpoint()
        write(out/'completion.json', report)
    return report


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('action', choices=['checks', 'real'])
    parser.add_argument('--responses', type=Path)
    parser.add_argument('--budget', type=int, default=540)
    parser.add_argument('--newton', type=int, default=3)
    args = parser.parse_args()
    def alarm(signum, frame):
        raise TimeoutError('Finite numerical wall-time budget exhausted')
    signal.signal(signal.SIGALRM, alarm)
    signal.alarm(args.budget)
    try:
        result = checks() if args.action == 'checks' else run_real(args)
    finally:
        signal.alarm(0)
    print(json.dumps(dict(action=args.action, status=result.get('status', 'algebra_checks'),
                          passed=result.get('passed'), accepted=result.get('accepted', False),
                          seconds=result['seconds'], peak_rss_gib=rss())), flush=True)


if __name__ == '__main__':
    main()
