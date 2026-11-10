#!/usr/bin/env python3
"""Phase C coherent metric, branch safeguards and strict unforced corrector.

The Phase B moving-grid equations and sparse bordered Newton method are reused.
Only j >= 1 velocities contribute to continuation distance; q0 and pressure
remain unknowns of the original equations but contribute exactly zero distance.
"""
from __future__ import annotations

import os
for _key in ('OPENBLAS_NUM_THREADS', 'OMP_NUM_THREADS', 'MKL_NUM_THREADS',
             'NUMEXPR_NUM_THREADS', 'VECLIB_MAXIMUM_THREADS', 'BLIS_NUM_THREADS'):
    os.environ[_key] = '1'

import gc
import hashlib
import json
import time
from pathlib import Path

import numpy as np
from scipy.sparse.linalg import LinearOperator, gmres

import fullbvp_phaseB_solver as base
from fullbvp_phaseB_solver import Mapping, ROOT, common, energies, pack, unpack
from fullbvp_phaseB_upper_refinement import TightCorrector
from appelquist2015_hopf_block_corrector import mass_row, norm

DEFAULT_OUT = ROOT / 'results/fullbvp_global_20261030/phaseC'
SOURCE_ID = 'branch_0036_try0'
STRICT = dict(relative_field_inf=1e-10, residual_over_fundamental_feedback=1e-5,
              constraint=1e-10, independent_relative_field_inf=1e-10,
              template_difference_relative=1e-10)
SCREEN = dict(fundamental_correlation_min=.97, coherent_correlation_min=.95,
              amplitude_ratio_min=.995, amplitude_ratio_max=1.30,
              coherent_energy_ratio_min=.80, coherent_energy_ratio_max=1.80,
              source_amplitude_floor=.80, omega_increase_max=.05,
              direction_cos_min=0., refinement_amplitude_ratio_min=.90,
              refinement_omega_change_max=.25)


def read(path):
    return json.loads(Path(path).read_text())


def sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b''):
            h.update(block)
    return h.hexdigest()


def fingerprint(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, allow_nan=False).encode()).hexdigest()


def write(path, value):
    """Replace JSON atomically; the caller holds the campaign lock."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + '.tmp')
    with tmp.open('w') as stream:
        json.dump(value, stream, indent=2, ensure_ascii=False, allow_nan=False)
        stream.write('\n')
        stream.flush()
        os.fsync(stream.fileno())
    os.replace(tmp, path)


def savez(path, **values):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + '.tmp')
    with tmp.open('wb') as stream:
        np.savez_compressed(stream, **values)
        stream.flush()
        os.fsync(stream.fileno())
    os.replace(tmp, path)


class CoherentMetric:
    """Fixed Rref=720 mass measure: 2 sum(j>=1) ||velocity_j||².

    Scales: coherent field .1, omega 1, Rout 10. No mean, pressure, or
    amplitude-dependent normalization enters the metric or its hyperplane.
    """
    def __init__(self, mapping, reference_R=720., field_scale=.1,
                 omega_scale=1., R_scale=10.):
        self.map = mapping
        self.p, *_ = mapping.at(reference_R, 53.)
        self.fs, self.ws, self.rs = field_scale, omega_scale, R_scale

    def inner(self, u, v, j):
        return common.mass_inner(self.p, u[:, self.map.H + j], v[:, self.map.H + j])

    def row(self, vector):
        m = self.map
        q = unpack(vector[:-2], m.nr, m.H, m.s)
        out = np.zeros((m.nr, 2*m.H+1, m.s))
        for j in range(1, m.H+1):
            a = mass_row(self.p, q[:, m.H+j]).reshape(m.nr, m.s) / self.fs**2
            out[:, 2*j-1] = 2*a.real
            out[:, 2*j] = -2*a.imag
        return np.r_[out.ravel(), vector[-2]/self.ws**2, vector[-1]/self.rs**2]

    def dot(self, u, v):
        return float(self.row(u) @ v)

    def length(self, vector):
        return float(np.sqrt(max(self.dot(vector, vector), 0.)))

    def tangent(self, old, new, previous=None):
        tangent = new-old
        length = self.length(tangent)
        if length < 1e-13:
            raise ValueError('No coherent secant: mean/pressure changes give zero length')
        tangent /= length
        if previous is not None and self.dot(tangent, previous) < 0:
            tangent = -tangent
        return tangent


def phase_align(candidate, reference, mapping, metric=None):
    metric = metric or CoherentMetric(mapping)
    q = unpack(candidate[:-2], mapping.nr, mapping.H, mapping.s)
    ref = unpack(reference[:-2], mapping.nr, mapping.H, mapping.s)
    overlap = metric.inner(ref, q, 1)
    if abs(overlap) < 1e-30:
        raise ValueError('Vanishing coherent overlap; no reliable phase alignment')
    phase = float(np.angle(overlap))
    q *= np.exp(-1j*np.arange(-mapping.H, mapping.H+1)*phase)[None, :, None]
    return np.r_[pack(q), candidate[-2:]], phase


def qref_for(y, mapping):
    p, *_ = mapping.at(y[-1], y[-2])
    q = unpack(y[:-2], mapping.nr, mapping.H, mapping.s)[:, mapping.H+1]
    return q / max(common.mass_norm(p, q), 1e-30)


def screen(previous, candidate, mapping, source_A, tangent=None,
           refinement=False, require_forward=True):
    """Newton convergence alone is insufficient for branch admission."""
    metric = CoherentMetric(mapping)
    candidate, phase = phase_align(candidate, previous, mapping, metric)
    a = unpack(previous[:-2], mapping.nr, mapping.H, mapping.s)
    b = unpack(candidate[:-2], mapping.nr, mapping.H, mapping.s)
    aa = np.array([metric.inner(a, a, j).real for j in range(1, mapping.H+1)])
    bb = np.array([metric.inner(b, b, j).real for j in range(1, mapping.H+1)])
    ab = np.array([metric.inner(a, b, j) for j in range(1, mapping.H+1)])
    q1corr = float(abs(ab[0])/max(np.sqrt(aa[0]*bb[0]), 1e-30))
    corr = float(ab.real.sum()/max(np.sqrt(aa.sum()*bb.sum()), 1e-30))
    xi_aratio = float(np.sqrt(bb[0]/max(aa[0], 1e-30)))
    measure_ratio = (candidate[-1]-400)/(previous[-1]-400)
    aratio = xi_aratio*float(np.sqrt(measure_ratio))
    eratio = float(bb.sum()/max(aa.sum(), 1e-30))*measure_ratio
    physical_A = float(np.sqrt(bb[0]*(candidate[-1]-400)/320))
    delta = candidate-previous
    length = metric.length(delta)
    cosine = None if tangent is None or length < 1e-12 else float(
        metric.dot(delta, tangent)/max(length*metric.length(tangent), 1e-30))
    reasons = []
    if q1corr < SCREEN['fundamental_correlation_min']:
        reasons.append('fundamental_correlation')
    if corr < SCREEN['coherent_correlation_min']:
        reasons.append('coherent_correlation')
    amin = SCREEN['refinement_amplitude_ratio_min'] if refinement else SCREEN['amplitude_ratio_min']
    if not amin <= aratio <= SCREEN['amplitude_ratio_max']:
        reasons.append('amplitude_ratio')
    if not SCREEN['coherent_energy_ratio_min'] <= eratio <= SCREEN['coherent_energy_ratio_max']:
        reasons.append('coherent_energy_ratio')
    if physical_A < SCREEN['source_amplitude_floor']*source_A:
        reasons.append('source_upper_amplitude_floor')
    if refinement:
        if abs(candidate[-1]-previous[-1]) > 1e-7:
            reasons.append('refinement_changed_Rout')
        if abs(candidate[-2]-previous[-2]) > SCREEN['refinement_omega_change_max']:
            reasons.append('refinement_frequency_jump')
    else:
        if require_forward and candidate[-1] <= previous[-1]+1e-8:
            reasons.append('Rout_not_forward')
        if candidate[-2] > previous[-2]+SCREEN['omega_increase_max']:
            reasons.append('upper_frequency_direction')
        if cosine is not None and cosine < SCREEN['direction_cos_min']:
            reasons.append('coherent_secant_direction')
    return candidate, dict(accepted=not reasons, reasons=reasons, phase_alignment=phase,
                          fundamental_correlation=q1corr, coherent_correlation=corr,
                          amplitude_ratio=aratio, xi_amplitude_ratio=xi_aratio,
                          coherent_energy_ratio=eratio,
                          source_amplitude_floor=SCREEN['source_amplitude_floor']*source_A,
                          physical_A1=physical_A, direction_cos=cosine,
                          delta_Rout=float(delta[-1]), delta_omega=float(delta[-2]),
                          coherent_metric_step=length, thresholds=SCREEN.copy())


def mapped_transfer(path, mapping, new_Rout=None):
    """Change N/H/M at the old Rout; then stretch xi, without extrapolation."""
    y = base.transfer(path, mapping)
    if new_Rout is not None:
        y[-1] = float(new_Rout)
    return y


def checkpoint(path, mapping, y, qref, tangent=None, accepted=False):
    p, *_ = mapping.at(y[-1], y[-2])
    savez(path, field=unpack(y[:-2], mapping.nr, mapping.H, mapping.s),
          y=y, radii=p.radii, xi=mapping.op.grid.xi, z=mapping.op.grid.z,
          mass=mapping.op.grid.mass, H=mapping.H, n=mapping.n,
          intervals=mapping.intervals, omega=y[-2], Rout=y[-1], a=y[-1]-100,
          accepted=accepted, qref=qref,
          tangent=np.array([]) if tangent is None else tangent)


def correct(mapping, qref, initial, mode, target, out, budget=1200.,
            metric=None, tangent=None, max_newton=12, resume=None):
    """Strict Phase B Newton/Krylov algorithm, with atomic iteration checkpoints."""
    out = Path(out)
    out.mkdir(parents=True, exist_ok=False)
    y = initial.copy()
    if resume is not None:
        with np.load(resume) as saved:
            y = saved['y'].copy()
            qref = saved['qref'].copy()
    started = time.monotonic()
    record = dict(accepted=False, mode=mode, target=float(target),
                  budget_seconds=budget, strict_thresholds=STRICT, iterations=[],
                  resumed_from=None if resume is None else str(resume))
    system = None
    try:
        system = TightCorrector(mapping, qref, y, mode, target, budget, metric, tangent)
        # Restart the Newton iterate without moving the requested arc hyperplane.
        system.center = initial.copy()
        for k in range(max_newton+1):
            row = system.metrics(y)
            row['iteration'] = k
            row['scaled_residual_l2'] = norm(system.residual(y))
            record['iterations'].append(row)
            record['final_metrics'] = row
            checkpoint(out/'checkpoint.npz', mapping, y, qref, tangent)
            write(out/'progress.json', record)
            if row['accepted']:
                record.update(row)
                break
            if k == max_newton:
                record['failure'] = 'Newton iteration limit'
                break
            residual = system.residual(y)
            J, cols, rows = system.jacobian(y)
            P, pre = system.preconditioner(cols, rows)
            row['preconditioner'] = pre
            history = []
            op = LinearOperator(J.shape, matvec=lambda v: J @ (P @ v), dtype=float)
            z, info = gmres(op, -residual, rtol=2e-7, atol=0., restart=45, maxiter=4,
                            callback=lambda r: history.append(float(r)), callback_type='pr_norm')
            step = P @ z
            relative = norm(J @ step+residual)/max(norm(residual), 1e-300)
            row['linear'] = dict(info=int(info), relative_original=relative, history=history)
            if relative > .1:
                record['failure'] = 'Krylov original linear residual'
                break
            for back in range(9):
                trial = y+2.**(-back)*step
                if not np.isfinite(trial).all() or not 660. < trial[-1] < 1000.:
                    continue
                if norm(system.residual(trial)) < (1-1e-4*2.**(-back))*norm(residual):
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
            check = base.independent(mapping, y)
            record['independent'] = check
            if (check['relative_field_inf'] >= STRICT['independent_relative_field_inf']
                    or check['template_difference_relative'] > STRICT['template_difference_relative']):
                record.update(accepted=False, failure='Strict independent original physical equations')
    except Exception as exc:
        record.update(accepted=False, failure=f'{type(exc).__name__}: {exc}',
                      budget_stop=isinstance(exc, base.BudgetExpired))
    record.update(seconds=time.monotonic()-started, Rout=float(y[-1]), omega=float(y[-2]),
                  n=mapping.n, H=mapping.H, intervals=mapping.intervals,
                  jv_calls=0 if system is None else system.jv_calls)
    checkpoint(out/('fields.npz' if record['accepted'] else 'checkpoint.npz'),
               mapping, y, qref, tangent, record['accepted'])
    write(out/'summary.json', record)
    return y, record
