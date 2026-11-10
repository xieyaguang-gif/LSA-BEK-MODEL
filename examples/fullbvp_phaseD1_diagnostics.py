#!/usr/bin/env python3
"""D1 diagnostics with Phase C definitions and actual sponge support."""
from __future__ import annotations

import numpy as np
from scipy.interpolate import BarycentricInterpolator, PchipInterpolator
from scipy.optimize import brentq

from fullbvp_phaseB_report import crossings, integrate_from
from fullbvp_phaseD1_solver import ROOT, OUT, read, sha, savez, write
from llsa.secondary.global_radial import radial_differences

KEYS = ('omega', 'front_fig8', 'front_2beta_absolute', 'front_2beta_5percent',
        'g_q_570_600', 'g_q_least_squares_570_600',
        'sponge_fraction_q', 'sponge_fraction_cylindrical_R3')


def diagnostic(fields, params, point_id, *, out=OUT, baseline=None, save=True):
    with np.load(fields) as d:
        r, H, n = d['radii'].copy(), int(d['H']), int(d['n'])
        q = d['field'].reshape(len(r), 2*H+1, 4, len(d['xi']))
        transform = lambda x: .6*x+.4*(x**3+.5*(1-x*x))
        xi = brentq(lambda x: transform(x)-(1.3-2)/(1.3+2), -1, 1)
        v = BarycentricInterpolator(d['xi'], q, axis=-1)(xi)[:, H+1:, 1]
        density = np.einsum('rhvi,ij,rhvj->rh', q[:, H:, :3].conj(), d['mass'], q[:, H:, :3], optimize=True).real
        omega = float(d['omega'])
    a = float(r[-1]-params['gap'])
    rms, fund = np.sqrt(2*np.sum(abs(v)**2, axis=1)), abs(v[:, 0])
    D, _ = radial_differences(r)
    logq, logrms = np.log(np.maximum(fund, 1e-250)), np.log(np.maximum(rms, 1e-250))
    gq, grms = D@logq, D@logrms
    curvepath = ROOT/f'results/fullbvp_global_20261030/phaseA/A3/curve_N{n}_global.json'
    curve = read(curvepath)
    lr, lg = np.array([x['r'] for x in curve]), np.array([x['growth'] for x in curve])
    localg = PchipInterpolator(lr, lg, extrapolate=False)(r)
    coverage = (r >= max(550., lr.min())) & (r <= min(650., lr.max())) & (fund > fund.max()*1e-8)
    valid = coverage & (r <= a)
    xr, xq = crossings(r, grms-localg, valid), crossings(r, gq-localg, valid)
    covered = crossings(r, grms-localg, coverage)
    choose = lambda xs, direction: next((x['R'] for x in xs if x['direction'] == direction), None)
    threshold = 1e-6
    x2 = crossings(r, abs(v[:, 1])-threshold, (r > 420)&(r < a)) if H >= 2 else []
    xratio = crossings(r, abs(v[:, 1])/np.maximum(fund, 1e-250)-.05,
                       (r > 500)&(r < a)&(fund > fund.max()*1e-8)) if H >= 2 else []
    coherent = density[:, 1:].sum(axis=1)
    e = np.trapezoid(density[:, 1:], r, axis=0)
    region = (r >= 570)&(r <= 600)
    gend = float((np.interp(600, r, logq)-np.interp(570, r, logq))/30)
    gfit = float(np.polyfit(r[region], logq[region], 1)[0])
    theta = 2*np.pi*np.arange(2*H+3)/(2*H+3)
    realv = 2*np.real(np.einsum('rj,jt->rt', v, np.exp(1j*np.arange(1, H+1)[:, None]*theta)))
    parseval = float(np.max(abs(np.sqrt(np.mean(realv**2, axis=1))-rms))/max(rms.max(), 1e-30))
    front = choose(xr, 'down')
    hi, lo = float(min(650., a, lr.max())), float(max(550., lr.min()))
    truncated = bool(a < min(650., lr.max()))
    result = dict(id=point_id, fields=str(fields), sha256=sha(fields), Rout=float(r[-1]),
                  Rin=float(r[0]), H=H, n=n, intervals=len(r)-1, Ro=-1., beta=68.,
                  omega=omega, sponge=params.copy(), a=a,
                  outer_sponge_support=[a, float(r[-1])], transition_end=a+params['width'],
                  front_fig8=front, front_fundamental=choose(xq, 'down'),
                  front_all_crossings=xr, front_fundamental_crossings=xq,
                  front_status=('downward_crossing_found' if front is not None else
                                'unavailable_censored_by_sponge_search_limit' if truncated else
                                'unavailable_no_downcrossing_in_covered_search_interval'),
                  front_search_interval=[lo, hi],
                  front_search_sampled_interval=None if not valid.any() else [float(r[valid][0]), float(r[valid][-1])],
                  front_search_truncated_by_sponge=truncated,
                  front_search_truncated_by_pinch_coverage=bool(lr.max() < min(650., a) or lr.min() > 550.),
                  pinch_reference_coverage=[float(lr.min()), float(lr.max())],
                  front_in_covered_pinch_domain_ignoring_sponge=choose(covered, 'down'),
                  front_crossings_in_covered_pinch_domain_ignoring_sponge=covered,
                  front_definition='z=1.3 d ln(v_rms/(R Omega))/dR = fixed Phase A global-operator absolute-pinch -Im(alpha0); first downward crossing before actual outer sponge; no upward fallback',
                  front_2beta_absolute=choose(x2, 'up'), front_2beta_absolute_crossings=x2,
                  front_2beta_5percent=choose(xratio, 'up'),
                  front_2beta_5percent_crossings=xratio,
                  front_2beta_search_interval=[420., a],
                  front_2beta_status='upward_crossing_found' if choose(x2, 'up') is not None else 'unavailable_no_upcrossing_before_actual_sponge',
                  threshold_2beta=threshold,
                  threshold_2beta_units='complex Fourier q_v,2 normalized by R*Omega; direct harmonic, not inferred from fundamental',
                  g_q_570_600=gend, g_q_least_squares_570_600=gfit, g_q_fit_570_600=gfit,
                  g_rms_570_600=float((np.interp(600, r, logrms)-np.interp(570, r, logrms))/30),
                  growth_fit_window=[570., 600.], growth_fit_intersects_outer_sponge=bool(a < 600.),
                  growth_fit_sponge_overlap=None if a >= 600 else [max(570., a), 600.],
                  coherent_energy_shares=(e/e.sum()).tolist(), mean_all_height_energy=None,
                  mean_energy_status='excluded_nondecaying_w0_tail',
                  coherent_definition='positive j>=1 velocities only; common conjugate factor cancels in fractions; q0 and pressure excluded',
                  rms_normalization='sqrt(2 sum_j>=1 |q_v,j(z=1.3)|^2); no extra division by R',
                  sponge_fraction_q=integrate_from(r, coherent, a)/float(np.trapezoid(coherent, r)),
                  sponge_fraction_cylindrical_R3=integrate_from(r, coherent*r**3, a)/float(np.trapezoid(coherent*r**3, r)),
                  vrms_max=float(rms.max()), parseval_relative_error=parseval,
                  local_pinch_source=str(curvepath), local_pinch_sha256=sha(curvepath),
                  pinch_reference_frequency_status='fixed Phase A curve; not recomputed at each nonlinear omega')
    if baseline is not None:
        result['endpoint_minus_baseline'] = {
            key: None if result[key] is None or baseline[key] is None else result[key]-baseline[key]
            for key in KEYS}
    if save:
        target = out/'diagnostics'
        target.mkdir(parents=True, exist_ok=True)
        savez(target/f'{point_id}.npz', radii=r, probe_v=v, vrms=rms, gq=gq,
              grms=grms, local_growth=localg, coherent_energy_density=coherent)
        write(target/f'{point_id}.json', result)
    return result
