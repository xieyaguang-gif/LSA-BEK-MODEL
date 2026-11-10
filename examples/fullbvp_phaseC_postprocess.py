#!/usr/bin/env python3
"""Phase C diagnostics and future DNS/domain assessment from actual fields only.

No plotting or target fabrication occurs while preparing the campaign. The
Phase B numerical definitions are reused; non-decaying mean-tail energies are
explicitly excluded from every coherent energy interpretation.
"""
from __future__ import annotations

import os
for _key in ('OPENBLAS_NUM_THREADS', 'OMP_NUM_THREADS', 'MKL_NUM_THREADS',
             'NUMEXPR_NUM_THREADS', 'VECLIB_MAXIMUM_THREADS', 'BLIS_NUM_THREADS'):
    os.environ[_key] = '1'
os.environ.setdefault('MPLCONFIGDIR', '/tmp/fullbvp_phaseC_matplotlib')

import argparse
import csv
from pathlib import Path

import numpy as np

from fullbvp_phaseB_report import diagnostic as phaseB_diagnostic
from fullbvp_phaseC_solver import DEFAULT_OUT, ROOT, read, sha, write

NL03 = ROOT/'results/appelquist2016_branch_20261029/literature/fig15_nl03_T6_vrms.csv'
NL01 = ROOT/'results/fullbvp_global_20261030/phaseB/reference_data/nl01_T6_vrms.csv'


def diagnostic(point, out):
    out = Path(out)
    result = phaseB_diagnostic(point, out)
    # Phase B audit identified q0's far-w tail as nonlocalizing: never label
    # its quadrature norm as all-height kinetic energy or use it in a share.
    result.pop('harmonic_energy_shares_with_conjugates', None)
    result['mean_all_height_energy'] = None
    result['energy_labels'] = ['legacy_discrete_mean_norm'] + [f'positive_j{j}_velocity_mass' for j in range(1, point['H']+1)]
    result['mean_energy_status'] = 'excluded_nondecaying_w0_tail'
    result['rms_normalization'] = 'sqrt(2 sum_j>=1 |q_v,j(z=1.3)|^2); q_v already locally normalized; no extra division by R'
    curve = read(ROOT/result['local_pinch_source'])
    result['local_pinch_source'] = str((ROOT/result['local_pinch_source']).resolve())
    lo, hi = min(p['r'] for p in curve), max(p['r'] for p in curve)
    search = [max(550., lo), min(650., point['a'], hi)]
    result['front_search_interval'] = search
    result['pinch_reference_coverage'] = [lo, hi]
    result['front_search_truncated_by_pinch_coverage'] = hi < min(650., point['a']) or lo > 550.
    result['pinch_reference_frequency_status'] = 'fixed Phase A curve; not recomputed at each nonlinear omega'
    if result['front_fig8'] is None:
        result['front_status'] = 'unavailable_no_downcrossing_in_covered_search_interval'
    # Store both Phase B end-to-end growth and the requested least-squares fit.
    result['g_q_fit_570_600'] = result['g_q_least_squares_570_600']
    write(out/'diagnostics'/f'{point["id"]}.json', result)
    return result


def profile_difference(out, base, other):
    """Compare log-RMS on a common upstream physical R interval, never xi."""
    out = Path(out)
    with np.load(out/'diagnostics'/f'{base["id"]}.npz') as a, np.load(out/'diagnostics'/f'{other["id"]}.npz') as b:
        lower, upper = 550., min(650., base['a'], other['a'])
        R = a['radii'][(a['radii'] >= lower) & (a['radii'] <= upper)]
        if len(R) < 3:
            return dict(status='insufficient_common_upstream_support')
        av = np.interp(R, a['radii'], a['vrms'])
        bv = np.interp(R, b['radii'], b['vrms'])
        valid = (av > 1e-8) & (bv > 1e-8)
        if np.count_nonzero(valid) < 3:
            return dict(status='insufficient_signal_above_1e-8')
        R, av, bv = R[valid], av[valid], bv[valid]
        difference = np.log(bv/av)
        return dict(status='computed', common_R=[float(R[0]), float(R[-1])],
                    log_RMS_difference_max=float(np.max(abs(difference))),
                    log_RMS_difference_rms=float(np.sqrt(np.mean(difference**2))),
                    RMS_relative_L2=float(np.linalg.norm(bv-av)/max(np.linalg.norm(av), 1e-30)),
                    amplitude_floor=1e-8)


def assessment(points, refinements, targets, out=None):
    """Report measurable differences; no automatic asymptotic-convergence claim."""
    selected = []
    by_id = {p['id']: p for p in points}
    for target in targets.values():
        if target['status'] == 'accepted':
            selected.append(by_id[target['point']])
    selected.sort(key=lambda p: p['Rout'])
    differences = []
    for a, b in zip(selected[:-1], selected[1:]):
        dR = b['Rout']-a['Rout']
        differences.append(dict(Rout_pair=[a['Rout'], b['Rout']],
            delta_omega=b['omega']-a['omega'],
            delta_front=None if a['front_fig8'] is None or b['front_fig8'] is None else b['front_fig8']-a['front_fig8'],
            delta_g_q_fit=b['g_q_fit_570_600']-a['g_q_fit_570_600'],
            sponge_q_pair=[a['sponge_fraction_q'], b['sponge_fraction_q']],
            sponge_R3_pair=[a['sponge_fraction_cylindrical_R3'], b['sponge_fraction_cylindrical_R3']],
            domega_dRout=(b['omega']-a['omega'])/dR,
            dfront_dRout=None if a['front_fig8'] is None or b['front_fig8'] is None else (b['front_fig8']-a['front_fig8'])/dR,
            dg_q_fit_dRout=(b['g_q_fit_570_600']-a['g_q_fit_570_600'])/dR,
            common_R_profile=None if out is None else profile_difference(out, a, b)))
    last = selected[-1] if selected else None
    tail = differences[-3:]
    high_sponge = any(p['sponge_fraction_q'] > .05 or p['sponge_fraction_cylindrical_R3'] > .10 for p in selected[-3:])
    translation = any(d['dfront_dRout'] is not None and d['dfront_dRout'] > .20 for d in tail)
    terminal = selected[-3:]
    enough_terminal = len(terminal) == 3
    persistent_half_q = None if not enough_terminal else all(p['sponge_fraction_q'] >= .5 for p in terminal)
    persistent_half_R3 = None if not enough_terminal else all(p['sponge_fraction_cylindrical_R3'] >= .5 for p in terminal)
    if not enough_terminal:
        outward_status, outward = 'insufficient_three_target_tail', None
    elif any(p['front_fig8'] is None for p in terminal):
        outward_status, outward = 'unavailable_missing_front_in_tail', None
    else:
        outward = all(b['front_fig8'] > a['front_fig8'] for a, b in zip(terminal[:-1], terminal[1:]))
        outward_status = 'outward_at_both_tail_increments' if outward else 'outward_persistence_absent'
    C4_indicator = None if persistent_half_q is None or outward is None else persistent_half_q and outward
    tail_changes = dict(increments=tail,
        omega_slope_abs_decreasing=None if len(tail) < 2 else bool(np.all(np.diff([abs(t['domega_dRout']) for t in tail]) < 0)),
        front_slope_abs_decreasing=None if len(tail) < 2 or any(t['dfront_dRout'] is None for t in tail)
        else bool(np.all(np.diff([abs(t['dfront_dRout']) for t in tail]) < 0)))
    followups = ['At the largest trusted Rout, independently refine N and radial spacing at fixed physical parameters.',
                 'Repeat selected states with sponge start and width varied independently of Rout.']
    if high_sponge or translation:
        followups += ['Extend Rout while keeping the previously tested upstream sponge position fixed, then move the sponge separately.',
                      'Compare an independently justified outflow/radiation boundary model if coherent support continues to track the outlet.']
    if any(p['front_fig8'] is None for p in selected):
        followups.append('Extend the local absolute-pinch calculation with an independently verified saddle before quoting a front outside R550–630.')
    if C4_indicator:
        followups.append('Persistent >=50% coherent q sponge support and outward front motion jointly trigger a boundary/sponge-supported-branch hypothesis; test it by independent sponge-placement and outflow-model changes.')
    return dict(established=False,
        status='no_target_data' if not selected else 'finite_domain_sequence_requires_independent_resolution_and_sponge_assessment',
        accepted_target_count=len(selected), target_count=len(targets),
        missing_targets=[t['Rout'] for t in targets.values() if t['status'] != 'accepted'],
        consecutive_differences=differences,
        tail_trend=tail_changes,
        sponge_or_boundary_sensitivity_flags=dict(high_sponge_support=high_sponge,
            front_translates_with_Rout=translation, interpretation='follow-up indicators, not proofs of boundary causation',
            heuristic_thresholds=dict(coherent_q_fraction=.05, coherent_R3_fraction=.10, positive_dfront_dRout=.20)),
        C4_persistent_sponge_and_outward_front=dict(tail_target_count=len(terminal), required_count=3,
            persistent_q_fraction_ge_half=persistent_half_q,
            persistent_R3_fraction_ge_half=persistent_half_R3,
            outward_front=outward, outward_front_status=outward_status,
            boundary_sponge_support_hypothesis_indicator=C4_indicator,
            interpretation='Conditional follow-up hypothesis requires persistent q fraction >=0.5 and outward motion at both final increments; missing evidence is unavailable.'),
        suggested_model_and_boundary_followups=followups,
        sponge_support=[dict(Rout=p['Rout'], outer_start=p['a'],
            front_to_sponge=None if p['front_fig8'] is None else p['a']-p['front_fig8'],
            coherent_q_fraction=p['sponge_fraction_q'], coherent_R3_fraction=p['sponge_fraction_cylindrical_R3']) for p in selected],
        finest_target=None if last is None else dict(Rout=last['Rout'], H=last['H'],
            omega=last['omega'], front=last['front_fig8'],
            omega_minus_DNS=last['omega']-52.36,
            front_minus_DNS=None if last['front_fig8'] is None else last['front_fig8']-583,
            growth_fit_minus_DNS=last['g_q_fit_570_600']-.106,
            growth_fit_minus_linear=last['g_q_fit_570_600']-.132),
        harmonic_checks=refinements,
        limitations=['Moving Rout also moves a=Rout-100; this sequence does not separate outlet from sponge placement.',
                    'N32 and 800 radial intervals fixed; increasing Rout increases physical dr.',
                    'No independent Phase C N or dr refinement is supplied by this campaign.',
                    'Same-Rout H comparisons assess harmonic sensitivity; capped or failed checks remain explicit.',
                    'Static local-pinch coverage limits the front search; missing fronts are not inferred.',
                    'Frequency/front stabilization and small sponge fractions alone do not prove domain convergence.',
                    'Rotating waves and finite-time nonlinear DNS are distinct states; temporal stability is uncomputed.'])


def render(out, points, state):
    """Generate publication-style static plots only after actual checkpoints exist."""
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    plt.rcParams.update({'font.family': 'serif', 'font.serif': ['STIXGeneral', 'DejaVu Serif'],
                         'mathtext.fontset': 'stix', 'font.size': 10,
                         'axes.spines.top': False, 'axes.spines.right': False,
                         'savefig.bbox': 'tight', 'pdf.fonttype': 42})
    figs = out/'figures'
    figs.mkdir(exist_ok=True)
    artifacts = []
    def save(fig, name):
        for suffix in ('png', 'pdf'):
            path = figs/f'{name}.{suffix}'
            fig.savefig(path, dpi=220)
            artifacts.append(str(path.resolve()))
        plt.close(fig)
    by_id = {p['id']: p for p in points}
    selected = [by_id[t['point']] for t in state['targets'].values() if t['status'] == 'accepted']
    display = ([points[0]] if points else []) + selected
    # Keep a legible subset without changing saved numerical profiles.
    display = list({p['id']: p for p in display}.values())
    if len(display) > 5:
        display = [display[i] for i in sorted(set(np.linspace(0, len(display)-1, 5).astype(int)))]
    fig, ax = plt.subplots(figsize=(7.2, 4.3))
    for p in display:
        with np.load(out/'diagnostics'/f'{p["id"]}.npz') as profile:
            r, rms = profile['radii'], profile['vrms']
            upstream = r <= p['a']
            line, = ax.semilogy(r[upstream], rms[upstream], label=f'H{p["H"]}, Rout={p["Rout"]:.2f}')
            ax.semilogy(r[~upstream], rms[~upstream], '--', color=line.get_color(), alpha=.65)
    if NL03.exists():
        data = np.genfromtxt(NL03, delimiter=',', names=True)
        ax.semilogy(data['R'], data['vrms_normalized'], color='#a52b2b', label='NL03 T=6, Fig.15 digitization')
        ax.fill_between(data['R'], data['vrms_plot_lower'], data['vrms_plot_upper'], color='#a52b2b', alpha=.12)
    if NL01.exists():
        data = np.genfromtxt(NL01, delimiter=',', names=True)
        ax.errorbar(data['R'], data['vrms_normalized'], xerr=data['R_plot_reading_tolerance'],
                    yerr=np.stack([data['vrms_normalized']-data['vrms_plot_lower'],
                                   data['vrms_plot_upper']-data['vrms_normalized']]),
                    fmt='s', ms=3, color='#287840', label='NL01 T=6, sparse Fig.6(b) contours')
    ax.set(xlabel=r'$R$', ylabel=r'$v_{\rm rms}/(R\Omega)$ at $z=1.3$',
           xlim=(515, max([p['Rout'] for p in display], default=800)), ylim=(2e-8, .3))
    ax.legend(fontsize=7)
    ax.text(.02, .98, 'Dashed: sponge support', transform=ax.transAxes, va='top', fontsize=8)
    save(fig, 'vrms_DNS')
    fig, axes = plt.subplots(1, 2, figsize=(7.4, 3.3))
    for ax, key, label, reference in ((axes[0], 'front_fig8', 'Fig.8 RMS front', 583.),
                                     (axes[1], 'omega', r'$\omega$', 52.36)):
        valid = [p for p in selected if p[key] is not None]
        if valid:
            ax.plot([p['Rout'] for p in valid], [p[key] for p in valid], 'o-', ms=4)
        ax.axhline(reference, color='#a52b2b', ls='--', lw=1, label='2016 DNS reference')
        ax.set(xlabel=r'$R_{\rm out}$', ylabel=label)
        ax.legend(fontsize=7)
    fig.suptitle('Finite-domain upper branch; domain convergence not established', fontsize=10)
    fig.tight_layout()
    save(fig, 'front_omega_Rout')
    fig, axes = plt.subplots(1, 2, figsize=(7.4, 3.3))
    for ax, key, label in ((axes[0], 'sponge_fraction_q', 'Coherent mass fraction'),
                           (axes[1], 'sponge_fraction_cylindrical_R3', r'Coherent $R^3$ energy fraction')):
        ax.plot([p['Rout'] for p in selected], [p[key] for p in selected], 'o-', ms=4)
        ax.set(xlabel=r'$R_{\rm out}$', ylabel=label)
    fig.tight_layout()
    save(fig, 'sponge_support')
    write(figs/'visual_audit.json', dict(status='pending_human_review', artifacts=artifacts,
        profiles='Numerical NPZ profiles retained for every accepted point',
        missing_targets=[t['Rout'] for t in state['targets'].values() if t['status'] != 'accepted']))
    return artifacts


def finalize(out):
    out = Path(out).resolve()
    state = read(out/'campaign.json')
    if not state['points']:
        raise ValueError('No accepted Phase C fields; postprocessing cannot manufacture data')
    (out/'POSTPROCESS_DONE.json').unlink(missing_ok=True)
    points = [diagnostic(p, out) for p in state['points']]
    domain = assessment(points, state['refinements'], state['targets'], out)
    summary = dict(status=state['status'], fingerprint=state['fingerprint'],
        numerical_status=state.get('numerical_status'), points=points, targets=state['targets'],
        source=state['source'], domain_convergence=domain,
        DNS=dict(NL03=dict(path=str(NL03), sha256=sha(NL03)) if NL03.exists() else None,
                 NL01=dict(path=str(NL01), sha256=sha(NL01),
                           status='sparse_Fig6b_T6_contour_crossings') if NL01.exists() else None),
        normalization='locally normalized complex q_v; mean excluded; no extra /R',
        temporal_stability='not_computed')
    write(out/'summary.json', summary)
    columns = ['id', 'Rout', 'H', 'omega', 'A1', 'front_fig8', 'front_status',
               'front_2beta_absolute', 'front_2beta_5percent', 'g_q_fit_570_600',
               'g_q_570_600', 'sponge_fraction_q', 'sponge_fraction_cylindrical_R3',
               'highest_to_fundamental', 'harmonic_comparison_status']
    with (out/'accepted_diagnostics.csv').open('w') as stream:
        writer = csv.DictWriter(stream, fieldnames=columns, extrasaction='ignore')
        writer.writeheader()
        writer.writerows(points)
    artifacts = render(out, points, state)
    def val(x):
        return 'unavailable' if x is None else f'{x:.7g}'
    lines = ['# Phase C upper-branch evidence', '',
        f'Numerical status: {state.get("numerical_status", "unknown")}. Accepted targets: {domain["accepted_target_count"]}/{domain["target_count"]}.',
        'Domain convergence is not established by this finite sequence.', '',
        '| Rout | H | omega | Fig.8 front | growth fit R570–600 | q sponge fraction | R³ sponge fraction |',
        '|---|---|---|---|---|---|---|']
    for p in points:
        lines.append('| '+' | '.join(val(p[k]) for k in ('Rout', 'H', 'omega', 'front_fig8',
            'g_q_fit_570_600', 'sponge_fraction_q', 'sponge_fraction_cylindrical_R3'))+' |')
    lines += ['', 'Reference values: DNS front about 583, omega about 52.36, growth about 0.106; linear growth about 0.132.',
        'NL03 is the Fig.15 digitized profile. NL01 is a separate sparse T=6 Fig.6(b) contour sample set.',
        'q_v is already locally normalized: RMS = sqrt(2 sum |q_v,j|²), with no additional division by R.',
        'Fig.8 uses the total RMS log-growth downcrossing against the fixed Phase A local absolute pinch. Its available coverage and any missing crossing are recorded per point.', '',
        '## Assessment limits', '']
    lines += ['- '+text for text in domain['limitations']]
    lines += ['', '## Measured tail trend and follow-up', '']
    for row in domain['tail_trend']['increments']:
        profile = row['common_R_profile'] or {}
        lines.append(f'Rout {row["Rout_pair"]}: Δomega/ΔR={val(row["domega_dRout"])}, Δfront/ΔR={val(row["dfront_dRout"])}, Δg_fit/ΔR={val(row["dg_q_fit_dRout"])}, common-R RMS relative L2={val(profile.get("RMS_relative_L2"))}.')
    flags = domain['sponge_or_boundary_sensitivity_flags']
    lines += [f'High sponge support indicator: {flags["high_sponge_support"]}; front translation indicator: {flags["front_translates_with_Rout"]}. These are follow-up indicators, not causal proof.', '']
    C4 = domain['C4_persistent_sponge_and_outward_front']
    lines += [f'C4 persistent >=50% q sponge fraction: {C4["persistent_q_fraction_ge_half"]}; outward tail fronts: {C4["outward_front_status"]}; combined hypothesis indicator: {C4["boundary_sponge_support_hypothesis_indicator"]}.', '']
    lines += ['- '+text for text in domain['suggested_model_and_boundary_followups']]
    lines += ['', 'Same-Rout H-to-H+1 changes and failed or capped comparisons are retained in campaign.json and summary.json.',
              'The numerical target ledger, nonlinear failures, branch rejections, and diagnostic failures remain distinct. Figure visual review is pending.']
    (out/'REPORT.md').write_text('\n'.join(lines)+'\n')
    write(out/'POSTPROCESS_DONE.json', dict(status='diagnostics_and_figures_saved',
        fingerprint=state['fingerprint'], accepted_points=len(points),
        artifacts=artifacts, visual_review='pending_human_review', domain_convergence_established=False))
    return summary


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--out', type=Path, default=DEFAULT_OUT)
    args = parser.parse_args()
    # Standalone postprocessing shares the campaign lock and changes no target ledger.
    from fullbvp_phaseC_campaign import Exclusive
    with Exclusive(args.out):
        finalize(args.out)
