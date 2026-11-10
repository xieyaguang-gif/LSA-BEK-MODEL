#!/usr/bin/env python3
"""Fixed-domain D1 sponge homotopies; original unforced BVP equations only."""
from __future__ import annotations

import os
for _key in ('OPENBLAS_NUM_THREADS', 'OMP_NUM_THREADS', 'MKL_NUM_THREADS',
             'NUMEXPR_NUM_THREADS', 'VECLIB_MAXIMUM_THREADS', 'BLIS_NUM_THREADS'):
    os.environ[_key] = '1'

import copy
import gc
import hashlib
import json
import time
from pathlib import Path

import numpy as np
from scipy import sparse
from scipy.sparse.linalg import LinearOperator, gmres

import fullbvp_phaseB_solver as base
import fullbvp_phaseC_solver as phaseC
from appelquist2015_hopf_block_corrector import norm
from fullbvp_phaseB_solver import ROOT, common, energies, pack, unpack
from llsa.secondary.global_radial import GlobalRadial, radial_differences
from llsa.secondary.radial import RadialConfig, radial_matrices

OUT = ROOT/'results/fullbvp_global_20261030/phaseD/D1'
SOURCE = ROOT/'results/fullbvp_global_20261030/phaseC/attempts/harmonic_audit_be43e3a56f551a7f_try0'
EXPECTED_SOURCE = {
    'fields.npz': '2dd79cc0d823ebc04fd0a9bd67a86c2a1e59df1f5f3302324401dfe7e8a2de1b',
    'request.json': 'a402afa3a71ce0337360dd2601f3e6679089cc883b4b4bdb4b14f258ba1f907b',
    'summary.json': '84d11812d2ed18f441252e7a93fe464667f34c15cbfcb3c7c00f0fea5f291416',
    'screen.json': 'a280ae272318338d69c938560d23a777dba6c3a96d50c9cc85323a9db6ace6cd',
}
BASE_PARAMS = dict(strength=28., width=40., gap=100.)
STRICT = phaseC.STRICT.copy()
CONTINUITY = dict(fundamental_correlation_min=.97, coherent_correlation_min=.95,
                  amplitude_ratio_min=.80, amplitude_ratio_max=1.25,
                  coherent_energy_ratio_min=.64, coherent_energy_ratio_max=1.5625,
                  omega_change_max=.5, source_amplitude_floor_fraction=.10,
                  absolute_amplitude_floor=1e-5)
write, savez, read, sha, fingerprint = phaseC.write, phaseC.savez, phaseC.read, phaseC.sha, phaseC.fingerprint


def verified_source():
    actual = {name: sha(SOURCE/name) for name in EXPECTED_SOURCE}
    if actual != EXPECTED_SOURCE:
        raise ValueError('Pinned Phase C source hash mismatch')
    summary, request, screen = (read(SOURCE/name) for name in ('summary.json', 'request.json', 'screen.json'))
    if not summary['accepted'] or not screen['accepted']:
        raise ValueError('Source not accepted by solver and branch screen')
    for key, expected in dict(n=32, H=6, intervals=800, Rout=725.).items():
        if summary[key] != expected:
            raise ValueError(f'Source summary {key} mismatch')
    if request['mode'] != 'Rout' or request['target'] != 725.:
        raise ValueError('Source request mismatch')
    numeric = [summary['relative_field_inf'], summary['residual_over_fundamental_feedback'],
               summary['independent']['relative_field_inf'], summary['omega'], summary['A1']]
    if (not np.isfinite(numeric).all() or
            summary['relative_field_inf'] >= STRICT['relative_field_inf'] or
            summary['residual_over_fundamental_feedback'] >= STRICT['residual_over_fundamental_feedback'] or
            summary['independent']['relative_field_inf'] >= STRICT['independent_relative_field_inf']):
        raise ValueError('Source fails strict residual thresholds')
    with np.load(SOURCE/'fields.npz') as data:
        for key, expected in dict(n=32, H=6, intervals=800, Rout=725., a=625., accepted=True).items():
            if data[key].item() != expected:
                raise ValueError(f'Source fields {key} mismatch')
        if (data['radii'].shape != (801,) or abs(data['radii'][0]-400) > 1e-12 or
                data['field'].shape != (801, 13, 132) or
                not np.isfinite(data['field']).all() or
                not np.isfinite(data['y']).all() or
                data['omega'].item() != summary['omega'] or
                data['y'][-2] != summary['omega'] or data['y'][-1] != 725. or
                not np.array_equal(data['y'][:-2], pack(data['field'])) or
                np.max(abs(data['radii']-np.linspace(400, 725, 801))) > 1e-12):
            raise ValueError('Source grid, field, or frequency mismatch')
    return dict(directory=str(SOURCE), hashes=actual, n=32, H=6, intervals=800,
                Rin=400., Rout=725., Ro=-1., beta=68., sponge=BASE_PARAMS,
                omega=summary['omega'], A1=summary['A1'])


class Mapping(base.Mapping):
    """Phase B operator with all outer-sponge coefficients parameterized.

    Rout derivatives hold strength, width and gap fixed, so da/dRout=1.
    No sponge is applied to the j=0 mean; inlet sponge remains unchanged.
    """
    def __init__(self, n=32, H=6, intervals=800, *, strength=28., width=40., gap=100.):
        if not (np.isfinite([strength, width, gap]).all() and strength > 0 and width > 0 and gap > 0):
            raise ValueError('Positive finite outer sponge parameters required')
        self.params = dict(strength=float(strength), width=float(width), gap=float(gap))
        super().__init__(n, H, intervals)

    def at(self, Rout, omega, derivative=False):
        strength, width, gap = (self.params[k] for k in ('strength', 'width', 'gap'))
        L = Rout-400.
        r = 400+L*self.xi
        if Rout-gap < 420.:
            raise ValueError('Outer sponge overlaps prescribed inlet region')
        t, td = 500/r, -500*self.xi/r**2
        D, D2 = self.Dx/L, self.Dxx/L**2
        K = sparse.kron(D, sparse.eye(self.s), format='csr')
        K2 = sparse.kron(D2, sparse.eye(self.s), format='csr')
        outer, ds = common.smooth_step_and_derivative((r-Rout+gap)/width)
        inner, di = common.smooth_step_and_derivative((420-r)/10)
        damping = strength*outer+.8*inner
        dd = strength/width*ds*(self.xi-1)-.08*di*self.xi
        bdiag = np.zeros((self.nr, self.s))
        bdiag[1:-1, self.rows] = 1/r[1:-1, None]
        blocks, derivatives = [], []
        kron = lambda v, a: sparse.kron(sparse.diags(v), a, format='csr')
        for j, (a0, a1, a3, b0, b2, c1) in enumerate(self.templates):
            A = kron(t*0+1, a0)+kron(t, a1)+kron(t**3, a3)
            B, C = kron(t*0+1, b0)+kron(t*t, b2), kron(t, c1)
            if j:
                A += kron((damping-1j*j*omega)/r, self.mass)
            mat = (self.mask@(A+B@K+C@K2)+self.inlet+self.outlet@K).tocsc()
            mat.eliminate_zeros()
            blocks.append(mat.real.tocsc() if j == 0 else mat)
            if derivative:
                Ad = kron(td, a1)+kron(3*t*t*td, a3)
                if j:
                    Ad += kron(dd/r-(damping-1j*j*omega)*self.xi/r**2, self.mass)
                Bd, Cd = kron(2*t*td, b2), kron(td, c1)
                der = (self.mask@(Ad+(Bd-B/L)@K+(Cd-2*C/L)@K2)-self.outlet@K/L).tocsc()
                derivatives.append(der.real.tocsc() if j == 0 else der)
        p = GlobalRadial.__new__(GlobalRadial)
        p.radii = r
        p.config = RadialConfig(ro=-1., beta=68., n=self.n, harmonics=self.H,
                               omega=float(omega), nonlinear=True)
        p.ops = []
        for R in r:
            op = copy.copy(self.op)
            op.radius = float(R)
            p.ops.append(op)
        p.D, p.D2, p.H, p.s, p.velocity_rows = D, D2, self.H, self.s, self.rows
        p.outlet_sponge = common.HopfSponge(Rout-gap, width, strength)
        p.sponge_strength = damping
        return p, blocks, bdiag, derivatives


def independent(mapping, y):
    """Reassemble physical local equations and sponge from the request itself."""
    p, blocks, *_ = mapping.at(y[-1], y[-2])
    q = unpack(y[:-2], mapping.nr, mapping.H, mapping.s)
    D, D2 = radial_differences(p.radii)
    qr = (D@q.reshape(mapping.nr, -1)).reshape(q.shape)
    qrr = (D2@q.reshape(mapping.nr, -1)).reshape(q.shape)
    params = mapping.params
    sponge = common.HopfSponge(float(y[-1])-params['gap'], params['width'], params['strength'])
    damping = sponge.profile(p.radii)
    out = np.zeros_like(q)
    for i, R in enumerate(p.radii):
        op = p.config.operators(R)
        for j in range(mapping.H+1):
            A, B, C = radial_matrices(op, j, y[-2])
            v = A@q[i, mapping.H+j]+B@qr[i, mapping.H+j]+C@qrr[i, mapping.H+j]
            if j:
                v[mapping.rows] += damping[i]/R*q[i, mapping.H+j, mapping.rows]
            if i == 0:
                v[mapping.rows] = q[i, mapping.H+j, mapping.rows]
            elif i == mapping.nr-1:
                v[mapping.rows] = qr[i, mapping.H+j, mapping.rows]
            out[i, mapping.H+j] = v
            if j:
                out[i, mapping.H-j] = v.conj()
    out += p._convection_field(q, qr)
    raw = pack(out)
    ours = pack(base.linear(blocks, q, mapping.H)+base.convection(p, q))
    return dict(residual_inf=float(max(abs(raw))),
                relative_field_inf=float(max(abs(raw))/max(float(max(abs(y[:-2]))), 1e-30)),
                template_difference_relative=norm(raw-ours)/max(norm(pack(base.linear(blocks, q, mapping.H))), norm(pack(base.convection(p, q))), 1e-30),
                forcing=False, physical_grid_reassembled=True,
                sponge_recomputed_from_parameters=True, sponge=mapping.params,
                a=float(y[-1]-params['gap']))


class Corrector(base.Corrector):
    def metrics(self, y):
        result = super().metrics(y)
        result.update(a=float(y[-1]-self.map.params['gap']), sponge=self.map.params.copy())
        result['accepted'] = bool(
            np.isfinite(y).all() and np.isfinite([result[k] for k in (
                'relative_field_inf', 'residual_over_fundamental_feedback', 'phase',
                'branch_constraint', 'A1', 'highest_to_fundamental')]).all() and
            result['relative_field_inf'] < STRICT['relative_field_inf'] and
            result['residual_over_fundamental_feedback'] < STRICT['residual_over_fundamental_feedback'] and
            max(abs(result['phase']), abs(result['branch_constraint'])) < STRICT['constraint'])
        return result


def checkpoint(path, mapping, y, qref, request_id, accepted=False):
    # Saving grid metadata must not reassemble seven full sparse matrices.
    savez(path, y=y, field=unpack(y[:-2], mapping.nr, mapping.H, mapping.s),
          radii=400+(y[-1]-400)*mapping.xi, xi=mapping.op.grid.xi,
          z=mapping.op.grid.z, mass=mapping.op.grid.mass, H=mapping.H,
          n=mapping.n, intervals=mapping.intervals, omega=y[-2], Rout=y[-1],
          a=y[-1]-mapping.params['gap'], accepted=accepted, qref=qref,
          strength=mapping.params['strength'], width=mapping.params['width'],
          gap=mapping.params['gap'], request_fingerprint=request_id)


def continuity(previous, candidate, mapping, source_A):
    """Symmetric change bounds: no increasing-Rout amplitude/frequency rule."""
    a = unpack(previous[:-2], mapping.nr, mapping.H, mapping.s)
    b = unpack(candidate[:-2], mapping.nr, mapping.H, mapping.s)
    p = GlobalRadial.__new__(GlobalRadial)
    p.radii, p.ops = 400+325*mapping.xi, [mapping.op]
    inner = lambda u, v, j: common.mass_inner(p, u[:, mapping.H+j], v[:, mapping.H+j])
    overlap = inner(a, b, 1)
    if abs(overlap) < 1e-30:
        return candidate, dict(accepted=False, reasons=['vanishing_fundamental_overlap'])
    phase = float(np.angle(overlap))
    b *= np.exp(-1j*np.arange(-mapping.H, mapping.H+1)*phase)[None, :, None]
    candidate = np.r_[pack(b), candidate[-2:]]
    aa = np.array([inner(a, a, j).real for j in range(1, mapping.H+1)])
    bb = np.array([inner(b, b, j).real for j in range(1, mapping.H+1)])
    ab = np.array([inner(a, b, j) for j in range(1, mapping.H+1)])
    qc = float(abs(ab[0])/max(np.sqrt(aa[0]*bb[0]), 1e-30))
    cc = float(ab.real.sum()/max(np.sqrt(aa.sum()*bb.sum()), 1e-30))
    ar, er = float(np.sqrt(bb[0]/max(aa[0], 1e-30))), float(bb.sum()/max(aa.sum(), 1e-30))
    physical_A = float(np.sqrt(max(bb[0], 0.)))
    floor = max(CONTINUITY['absolute_amplitude_floor'], source_A*CONTINUITY['source_amplitude_floor_fraction'])
    tests = dict(fundamental_correlation=qc >= CONTINUITY['fundamental_correlation_min'],
                 coherent_correlation=cc >= CONTINUITY['coherent_correlation_min'],
                 amplitude_ratio=CONTINUITY['amplitude_ratio_min'] <= ar <= CONTINUITY['amplitude_ratio_max'],
                 coherent_energy_ratio=CONTINUITY['coherent_energy_ratio_min'] <= er <= CONTINUITY['coherent_energy_ratio_max'],
                 frequency_jump=abs(candidate[-2]-previous[-2]) <= CONTINUITY['omega_change_max'],
                 nonzero_amplitude=physical_A >= floor, fixed_Rout=abs(candidate[-1]-725.) < 1e-9,
                 finite=bool(np.isfinite(candidate).all()))
    return candidate, dict(accepted=all(tests.values()), reasons=[k for k, good in tests.items() if not good],
                          phase_alignment=phase, fundamental_correlation=qc, coherent_correlation=cc,
                          amplitude_ratio=ar, coherent_energy_ratio=er, A1=physical_A,
                          amplitude_floor=floor, delta_omega=float(candidate[-2]-previous[-2]),
                          thresholds=CONTINUITY)


def correct(mapping, initial, out, request_id, budget=4800., resume=None):
    """One strict correction, supervised externally by a hard subprocess timeout."""
    out = Path(out)
    out.mkdir(parents=True, exist_ok=True)
    y = initial.copy()
    # Norm and qref do not require construction of full template matrices.
    q = unpack(initial[:-2], mapping.nr, mapping.H, mapping.s)
    p = GlobalRadial.__new__(GlobalRadial)
    p.radii, p.ops = 400+(initial[-1]-400)*mapping.xi, [mapping.op]
    qref = q[:, mapping.H+1]/max(common.mass_norm(p, q[:, mapping.H+1]), 1e-30)
    if resume is not None:
        with np.load(resume) as saved:
            if saved['request_fingerprint'].item() != request_id:
                raise ValueError('Checkpoint request mismatch')
            if (any(saved[key].item() != value for key, value in mapping.params.items()) or
                    int(saved['n']) != mapping.n or int(saved['H']) != mapping.H or
                    int(saved['intervals']) != mapping.intervals or
                    not np.isfinite(saved['y']).all() or not np.isfinite(saved['qref']).all() or
                    saved['y'].shape != initial.shape or abs(float(saved['Rout'])-725.) > 1e-9):
                raise ValueError('Checkpoint parameters/grid/finite-state mismatch')
            y, qref = saved['y'].copy(), saved['qref'].copy()
    record = dict(accepted=False, request_fingerprint=request_id, sponge=mapping.params.copy(),
                  budget_seconds=budget, iterations=[], strict_thresholds=STRICT,
                  resumed_from=None if resume is None else str(resume))
    started = time.monotonic()
    system = None
    checkpoint(out/'checkpoint.npz', mapping, y, qref, request_id)
    try:
        system = Corrector(mapping, qref, y, 'Rout', 725., budget)
        for k in range(13):
            row = system.metrics(y)
            row.update(iteration=k, scaled_residual_l2=norm(system.residual(y)))
            record['iterations'].append(row)
            record['final_metrics'] = row
            checkpoint(out/'checkpoint.npz', mapping, y, qref, request_id)
            write(out/'progress.json', record)
            if row['accepted']:
                record.update(row)
                break
            if k == 12:
                record['failure'] = 'Newton iteration limit'
                break
            f = system.residual(y)
            J, cols, rows = system.jacobian(y)
            P, pre = system.preconditioner(cols, rows)
            row['preconditioner'] = pre
            history = []
            op = LinearOperator(J.shape, matvec=lambda v: J@(P@v), dtype=float)
            z, info = gmres(op, -f, rtol=2e-7, atol=0., restart=45, maxiter=4,
                            callback=lambda r: history.append(float(r)), callback_type='pr_norm')
            step = P@z
            relative = norm(J@step+f)/max(norm(f), 1e-300)
            row['linear'] = dict(info=int(info), relative_original=relative, history=history)
            if relative > .1:
                record['failure'] = 'Krylov original linear residual'
                break
            for back in range(9):
                trial = y+2.**(-back)*step
                if not np.isfinite(trial).all() or abs(trial[-1]-725.) > 1e-8:
                    continue
                if norm(system.residual(trial)) < (1-1e-4*2.**(-back))*norm(f):
                    y = trial
                    row['step_length'] = 2.**(-back)
                    break
            else:
                record['failure'] = 'Newton line search'
                break
            del J, P, op, cols, rows
            gc.collect()
        if record['accepted']:
            system.check()
            audit = independent(mapping, y)
            record['independent'] = audit
            if (not np.isfinite([audit['residual_inf'], audit['relative_field_inf'], audit['template_difference_relative']]).all() or
                    audit['relative_field_inf'] >= STRICT['independent_relative_field_inf'] or
                    audit['template_difference_relative'] > STRICT['template_difference_relative']):
                record.update(accepted=False, failure='Independent original physical equations')
    except Exception as exc:
        record.update(accepted=False, failure=f'{type(exc).__name__}: {exc}',
                      budget_stop=isinstance(exc, base.BudgetExpired))
    record.update(seconds=time.monotonic()-started, n=mapping.n, H=mapping.H,
                  intervals=mapping.intervals, Rout=float(y[-1]), omega=float(y[-2]),
                  a=float(y[-1]-mapping.params['gap']), jv_calls=0 if system is None else system.jv_calls)
    checkpoint(out/('fields.npz' if record['accepted'] else 'checkpoint.npz'), mapping, y, qref, request_id, record['accepted'])
    if record['accepted']:
        record['fields_sha256'] = sha(out/'fields.npz')
    write(out/'summary.json', record)
    return y, record
