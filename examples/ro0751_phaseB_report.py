#!/usr/bin/env python3
"""Phase B evidence, Chinese report and data-linked Python PNG/PDF figures."""
import os
for _key in ('OPENBLAS_NUM_THREADS', 'OMP_NUM_THREADS', 'MKL_NUM_THREADS',
             'NUMEXPR_NUM_THREADS', 'VECLIB_MAXIMUM_THREADS', 'BLIS_NUM_THREADS'):
    os.environ[_key] = '1'
os.environ.setdefault('MPLCONFIGDIR', '/tmp/ro0751_phaseB_matplotlib')
import argparse
import functools
import hashlib
import json
import subprocess
import sys
from pathlib import Path
import numpy as np
from scipy.interpolate import BarycentricInterpolator, PchipInterpolator, interp1d
from scipy.optimize import brentq
from ro0751_phaseB_solver import ROOT, common, write
from llsa.secondary.global_radial import radial_differences

RO = -.751
BETA_A = ROOT/'results/ro0751_global_20261101/phaseA/beta85'
QA = Path('/home/lthpc/.codex/skills/nature-figure/scripts')


def crossing(r, f, valid):
    roots = []
    for i in range(len(r)-1):
        if not (valid[i] and valid[i+1]) or not np.isfinite(f[i:i+2]).all():
            continue
        down = f[i] >= 0 > f[i+1]
        up = f[i] <= 0 < f[i+1]
        if down or up:
            radius = r[i]-f[i]*(r[i+1]-r[i])/(f[i+1]-f[i])
            roots.append(dict(R=float(radius), direction='down' if down else 'up',
                              bracket=[float(r[i]), float(r[i+1])]))
    return roots


def integrate(r, f, low, high):
    mask = (r > low) & (r < high)
    x = np.r_[low, r[mask], high]
    y = np.r_[np.interp(low, r, f), f[mask], np.interp(high, r, f)]
    return float(np.trapezoid(y, x))


@functools.lru_cache(maxsize=2)
def pinch(n):
    path = BETA_A/f'A3/curve_N{n}_global.json'
    d = json.loads(path.read_text())
    if d['ro'] != RO or d['beta'] != 85.:
        raise ValueError('Wrong local-pinch Ro/beta')
    rows = d['rows']
    r = np.array([x['r'] for x in rows])
    g = np.array([x['growth'] for x in rows])
    if not np.all(np.diff(r) > 0) or not np.isfinite(g).all():
        raise ValueError('Invalid local-pinch data')
    return r, g, str(path.relative_to(ROOT)), hashlib.sha256(path.read_bytes()).hexdigest()


def diagnostic(point, out):
    source = ROOT/point['fields']
    with np.load(source) as d:
        if float(d['ro']) != RO or float(d['beta']) != 85. or not bool(d['accepted']):
            raise ValueError('Unaccepted or mismatched field')
        r, H, n = d['radii'], int(d['H']), int(d['n'])
        q = d['field'].reshape(len(r), 2*H+1, 4, len(d['xi']))
        transform = lambda x:.6*x+.4*(x**3+.5*(1-x*x))
        xi = brentq(lambda x:transform(x)-(1.3-2)/(1.3+2), -1, 1)
        probe = BarycentricInterpolator(d['xi'], q, axis=-1)(xi)
        v = probe[:, H+1:, 1]
        rms = np.sqrt(2*np.sum(abs(v)**2, axis=1))
        fundamental = abs(v[:, 0])
        D, _ = radial_differences(r)
        gq = D@np.log(np.maximum(fundamental, 1e-250))
        grms = D@np.log(np.maximum(rms, 1e-250))
        lr, lg, local_source, local_hash = pinch(n)
        local = PchipInterpolator(lr, lg, extrapolate=False)(r)
        a = float(r[-1]-100)
        valid = ((r >= max(550, lr.min())) & (r <= min(650, a, lr.max()))
                 & (fundamental > fundamental.max()*1e-8))
        roots = crossing(r, grms-local, valid)
        fundamental_roots = crossing(r, gq-local, valid)
        choose = lambda xs:next((p['R'] for p in xs if p['direction'] == 'down'), None)
        front = choose(roots)
        pchip_front = None
        if front is not None:
            bracket = next(p['bracket'] for p in roots if p['direction'] == 'down')
            nodes = valid & np.isfinite(grms-local)
            if np.count_nonzero(nodes) >= 3:
                fun = PchipInterpolator(r[nodes], (grms-local)[nodes], extrapolate=False)
                pchip_front = float(brentq(fun, *bracket))
        density = np.einsum('rhvi,ij,rhvj->rh', q[:, H:, :3].conj(),
                            d['mass'], q[:, H:, :3], optimize=True).real
        energy = np.trapezoid(density, r, axis=0)
        coherent = density[:, 1:].sum(axis=1)
        denominator = float(np.trapezoid(coherent, r))
        physical_denominator = float(np.trapezoid(coherent*r**3, r))
        shares = energy*np.r_[1, np.full(H, 2)]
        shares /= shares.sum()
        theta = 2*np.pi*np.arange(2*H+3)/(2*H+3)
        realv = 2*np.real(np.einsum('rj,jt->rt', v,
                          np.exp(1j*np.arange(1, H+1)[:, None]*theta)))
        parseval = float(np.max(abs(np.sqrt(np.mean(realv**2, axis=1))-rms))/max(rms.max(), 1e-30))
        x2 = crossing(r, abs(v[:, 1])-1e-6, (r > 420) & (r < a))
        result = dict(point, front_fig8=front, front_fundamental=choose(fundamental_roots),
                      front_all_crossings=roots, front_search_interval=[max(550., float(lr.min())), min(650., a, float(lr.max()))],
                      front_definition='z=1.3, d ln(v_rms/(R Omega))/dR = local -Im(alpha0); first downstream-sign-changing (positive-to-negative) crossing upstream of sponge, same Ro=-1 convention',
                      front_pchip=pchip_front,
                      front_interpolation_difference=(pchip_front-front if front is not None else None),
                      front_Re=abs(RO)*front if front is not None else None,
                      front_Re_minus_Xie471_8=abs(RO)*front-471.8 if front is not None else None,
                      front_2beta_absolute=next((p['R'] for p in x2 if p['direction'] == 'up'), None),
                      threshold_2beta=1e-6,
                      harmonic_energy_shares_with_conjugates=shares.tolist(),
                      coherent_energy_shares=(energy[1:]/energy[1:].sum()).tolist(),
                      sponge_fraction_q=integrate(r, coherent, a, r[-1])/denominator,
                      sponge_fraction_cylindrical_R3=integrate(r, coherent*r**3, a, r[-1])/physical_denominator,
                      inlet_sponge_fraction_q=integrate(r, coherent, r[0], 420)/denominator,
                      inlet_sponge_fraction_R3=integrate(r, coherent*r**3, r[0], 420)/physical_denominator,
                      g_q_570_600=float((np.interp(600, r, np.log(np.maximum(fundamental, 1e-250)))
                                         -np.interp(570, r, np.log(np.maximum(fundamental, 1e-250))))/30),
                      parseval_relative_error=parseval,
                      local_pinch_source=local_source, local_pinch_sha256=local_hash,
                      sha256=hashlib.sha256(source.read_bytes()).hexdigest())
        if parseval > 1e-12:
            raise RuntimeError('Fourier/real-sector Parseval check failed')
        target = out/'diagnostics'
        target.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(target/(point['id']+'.npz'), radii=r, probe_v=v,
                            vrms=rms, gq=gq, grms=grms, local_growth=local,
                            energy_density=density, front_valid=valid)
        write(target/(point['id']+'.json'), result)
    return result


def compare_fields(base, refined):
    with np.load(ROOT/base['fields']) as b, np.load(ROOT/refined['fields']) as c:
        r = b['radii']; mask = r <= c['radii'][-1]; r = r[mask]
        H1, H2 = int(b['H']), int(c['H'])
        h = min(H1, H2)
        q = b['field'][mask].reshape(len(r), 2*H1+1, 4, len(b['xi']))[:, H1:H1+h+1, :3]
        v = c['field'].reshape(len(c['radii']), 2*H2+1, 4, len(c['xi']))[:, H2:H2+h+1, :3]
        if len(b['xi']) != len(c['xi']):
            v = BarycentricInterpolator(c['xi'], v, axis=-1)(b['xi'])
        v = interp1d(c['radii'], v, axis=0, kind='cubic', bounds_error=True)(r)
        inner = lambda x,y:np.einsum('rvi,ij,rvj,r->', x.conj(), b['mass'], y,
                                     common.radial_weights(r), optimize=True)
        phase = np.angle(inner(q[:, 1], v[:, 1]))
        v *= np.exp(-1j*np.arange(h+1)*phase)[None, :, None, None]
        return dict(common_physical_domain=[float(r[0]), float(r[-1])],
                    phase_alignment=float(phase),
                    q1_relative_mass=float(np.sqrt(inner(v[:, 1]-q[:, 1], v[:, 1]-q[:, 1]).real
                                                   /inner(q[:, 1], q[:, 1]).real)))


def render(out, summary):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    sys.path.insert(0, str(QA))
    from audit_panel_alignment import require_matplotlib_panel_alignment
    plt.rcParams.update({'font.family':'serif', 'font.serif':['Times New Roman', 'Nimbus Roman', 'STIXGeneral', 'DejaVu Serif'],
                         'mathtext.fontset':'stix', 'font.size':10, 'axes.linewidth':.85,
                         'lines.linewidth':1.3, 'savefig.dpi':300, 'pdf.fonttype':42,
                         'legend.fontsize':8, 'legend.frameon':False})
    figs = out/'figures'; figs.mkdir(exist_ok=True)
    qa = figs/'qa'; qa.mkdir(exist_ok=True)
    points = summary['points']
    primary = [p for p in points if p['role'] in ('weak_seed', 'primary_arclength')]
    upper = [p for p in primary if p.get('upper_branch')]
    selected_ids = [p['id'] for p in points if p['role'] == 'upper_target']
    selected_ids += [x['source'] for x in summary['refinements']]
    selected_ids += [p['id'] for p in upper[-1:]]
    selected = [p for p in points if p['id'] in set(selected_ids)]
    audits = []

    def canvas(ylabel, xlabel):
        fig, ax = plt.subplots(figsize=(6.4, 3.8), layout='constrained')
        ax.set(xlabel=xlabel, ylabel=ylabel)
        ax.tick_params(direction='in', top=True, right=True)
        return fig, ax

    def finish(fig, ax, name):
        if not points:
            ax.text(.5, .5, 'No accepted numerical states', transform=ax.transAxes, ha='center')
        if ax.get_legend_handles_labels()[0]:
            ax.legend(loc='best')
        require_matplotlib_panel_alignment(fig, json_out=qa/f'{name}.alignment.json', strict=True)
        fig.savefig(figs/f'{name}.png')
        fig.savefig(figs/f'{name}.pdf')
        plt.close(fig)
        row = dict(figure=name)
        for kind, argv in (
            ('glyphs', [sys.executable, str(QA/'audit_pdf_text.py'), str(figs/f'{name}.pdf'), '--min-pt', '5', '--json']),
            ('collisions', [sys.executable, str(QA/'audit_figure_collisions.py'), str(figs/f'{name}.pdf'), '--json', '--json-out', str(qa/f'{name}.collisions.json')])):
            proc = subprocess.run(argv, text=True, capture_output=True, timeout=60)
            (qa/f'{name}.{kind}.log').write_text(proc.stdout+proc.stderr)
            row[kind+'_returncode'] = proc.returncode
        audits.append(row)

    for key, label, name in [('Rout', r'$R_{\rm out}$', 'branch_Rout_A1'),
                             ('omega', r'$\omega_{\rm lab}$', 'branch_omega_A1')]:
        fig, ax = canvas(label, r'$A_1=\sqrt{E_1}$')
        if primary:
            ax.plot([p['A1'] for p in primary], [p[key] for p in primary], '-o', color='#205b90', ms=3, label='H3 arclength')
        for H, color, marker in [(3, '#222222', 's'), (4, '#aa4b2c', '^')]:
            pp = [p for p in points if p['role'] == f'fold_H{H}']
            if pp:
                ax.plot([p['A1'] for p in pp], [p[key] for p in pp], marker, color=color, ms=4, label=f'H{H} fold checks')
        ax.axhline(summary['linear'][key], ls=':', color='.45', label='Linear neutral, N32')
        finish(fig, ax, name)
    fig, ax = canvas(r'$v_{\rm rms}/(R\Omega)$ at $z=1.3$', r'$R$')
    for point in selected:
        with np.load(out/'diagnostics'/(point['id']+'.npz')) as d:
            r, rms = d['radii'], d['vrms']; inside = r <= point['a']
            line, = ax.semilogy(r[inside], rms[inside], label=f'Rout={point["Rout"]:.2f}, H{point["H"]}')
            ax.semilogy(r[~inside], rms[~inside], '--', color=line.get_color(), alpha=.6)
    if not selected:
        ax.text(.5, .5, 'No verified upper-branch profile', transform=ax.transAxes, ha='center')
    finish(fig, ax, 'upper_vrms_z1p3')
    for key, label, name in [('front_fig8', r'$R_{\rm front}$', 'upper_front_Rout'),
                             ('omega', r'$\omega_{\rm lab}$', 'upper_omega_Rout'),
                             ('sponge_fraction_cylindrical_R3', 'Outer sponge energy fraction', 'upper_sponge_Rout')]:
        fig, ax = canvas(label, r'$R_{\rm out}$')
        pp = [p for p in upper if p.get(key) is not None]
        if pp:
            ax.plot([p['Rout'] for p in pp], [p[key] for p in pp], '-o', ms=3, color='#205b90', label='Verified H3 upper branch')
        if key == 'front_fig8':
            ax.axhline(471.8/abs(RO), ls=':', color='#aa4b2c', label='Xie: Re=471.8')
        finish(fig, ax, name)
    write(figs/'visual_audit.json', dict(automatic_audits=audits,
         automatic_passed=all(x['glyphs_returncode'] == 0 and x['collisions_returncode'] == 0 for x in audits),
         manual_review='pending: inspect every PNG and PDF at final size',
         profile_selection_ids=[p['id'] for p in selected],
         all_accepted_point_count=len(points), profile_point_count=len(selected)))


def number(value):
    return '未取得' if value is None else f'{value:.7g}'


def finalize(out, inputs):
    out = Path(out)
    state = json.loads((out/'campaign.json').read_text())
    points = [diagnostic(p, out) for p in state['points']]
    byid = {p['id']:p for p in points}
    refinements = []
    for item in state['refinements']:
        row = dict(item)
        if item['accepted'] and item['point'] in byid:
            base, refined = byid[item['source']], byid[item['point']]
            row.update(delta_Rout=refined['Rout']-base['Rout'], delta_omega=refined['omega']-base['omega'],
                       delta_front=(refined['front_fig8']-base['front_fig8']
                                    if all(p['front_fig8'] is not None for p in (base, refined)) else None),
                       field_comparison=compare_fields(base, refined))
        refinements.append(row)
    targets = {}
    for target in (730, 750, 780):
        entry = state['targets'].get(str(target), dict(status='unavailable', reason=state['stop'] or state['stage']))
        targets[str(target)] = dict(entry, diagnostics=byid.get(entry.get('point')))
    consecutive = []
    reached = [targets[str(t)]['diagnostics'] for t in (730, 750, 780) if targets[str(t)].get('diagnostics')]
    for left, right in zip(reached[:-1], reached[1:]):
        consecutive.append(dict(Rout_pair=[left['Rout'], right['Rout']], delta_omega=right['omega']-left['omega'],
                                 delta_front=(right['front_fig8']-left['front_fig8']
                                              if all(p['front_fig8'] is not None for p in (left, right)) else None),
                                 delta_sponge_R3=right['sponge_fraction_cylindrical_R3']-left['sponge_fraction_cylindrical_R3']))
    coarse = inputs['seeds']['N32_dr0.5']; fine = inputs['seeds']['N40_dr0.25']
    linear_end = next(p for p in inputs['linear_r_end'] if p['n'] == 40)['rend_global']['pchip'][0]
    summary = dict(status=state['status'], numerical_stage=state['stage'], stop=state['stop'], ro=RO, beta=85.,
                   linear=dict(Rout=coarse['rout'], omega=coarse['omega'], Rout_fine=fine['rout'],
                               omega_fine=fine['omega'], r_end_fine=linear_end, Re_end_fine=abs(RO)*linear_end,
                               grid_delta_Rout=fine['rout']-coarse['rout'], grid_delta_omega=fine['omega']-coarse['omega']),
                   folds=state['folds'], points=points, targets=targets, refinements=refinements,
                   consecutive_target_differences=consecutive,
                   external_reference=dict(provenance='User-supplied task: Xie 2025 Table 7; no digitized profile supplied',
                                           Re_linear=484.4, Re_nonlinear=471.8, Re_displacement=12.6,
                                           r_nonlinear=471.8/abs(RO), Re_absolute=409.6),
                   front_convergence_status=('three_targets_available_finite_domain_differences_only'
                                              if len(reached) == 3 else 'insufficient_target_domains'),
                   auxiliary_forcing=False, secondary_instability_computed=False)
    upper = [p for p in points if p.get('upper_branch')]
    latest_upper = upper[-1] if upper else {}
    summary['last_verified_upper_point'] = latest_upper or None
    write(out/'summary.json', summary)
    h3 = state['folds'].get('H3', {})
    lines = ['# Ro=−0.751、β=85 全局非线性：阶段 B', '',
             f'计算状态：{state["status"]}；停止原因：{state["stop"] or "尚未触发"}。', '',
             f'折点 Rout_f（H3）={number(h3.get("Rout"))}；H4−H3={number(state["folds"].get("H4_minus_H3_Rout"))}。',
             f'前沿和频率的域长收敛：{summary["front_convergence_status"]}。仅有已验收根和实际目标点参与比较。',
             f'最后一个已验证上支点：Rout={number(latest_upper.get("Rout"))}、前沿 R={number(latest_upper.get("front_fig8"))}、ω={number(latest_upper.get("omega"))}、Re_front−471.8={number(latest_upper.get("front_Re_minus_Xie471_8"))}。该点不是收敛外推值。',
             '与 Xie 的 Re=471.8 比较使用 Re_front=0.751 R_front；Rout 是海绵截断域的出口，不能直接作为论文的非线性前沿。', '',
             '## 上支目标与外部对照', '',
             '| Rout | 状态 | ω_lab | 前沿 R | 前沿 Re | Re−471.8 | 外海绵 R³ 能量占比 |',
             '|---:|---|---:|---:|---:|---:|---:|']
    for target, entry in targets.items():
        p = entry.get('diagnostics') or {}
        lines.append(f'| {target} | {entry["status"]} | {number(p.get("omega"))} | {number(p.get("front_fig8"))} | {number(p.get("front_Re"))} | {number(p.get("front_Re_minus_Xie471_8"))} | {number(p.get("sponge_fraction_cylindrical_R3"))} |')
    lines += ['', f'阶段 A 细网格 r_end={linear_end:.7g}、Re_end={abs(RO)*linear_end:.7g}。Xie 任务书数值：线性 484.4、非线性 471.8，位移 12.6。',
              '线性 r_end 使用高度积分能量增长率，非线性前沿使用 z=1.3 的 rms 增长率；二者的位移附带诊断定义差异。', '',
              '## 不确定度与分辨率', '',
              f'折点局部三点与五点插值差={number(h3.get("three_vs_five_Rout"))}；H3/H4 差为谐波敏感性，不能替代折点网格误差。',
              f'阶段 A 两网格中性出口差={fine["rout"]-coarse["rout"]:.7g}，频率差={fine["omega"]-coarse["omega"]:.7g}。', '',
              '| 上支检查（固定 Rout） | 验收 | Δω | Δ前沿 R |', '|---|---|---:|---:|']
    for row in refinements:
        lines.append(f'| {row["label"]} | {row["accepted"]} | {number(row.get("delta_omega"))} | {number(row.get("delta_front"))} |')
    lines += ['', '这些差值是离散与插值敏感性，不是统计置信区间。仅一个上支点的联合 N40/ΔR≤0.25 复核不能确定全分支或折点的渐近收敛阶。', '',
              '## 定义、验收与复现', '',
              'Ro=−0.751、β=85、Rin=400；入口海绵 0.8 S((420−R)/10)，外海绵 28 S((R−(Rout−100))/40)。均值修正不受海绵阻尼；无辅助体积强迫、无二次失稳。',
              f'主网格 N32、固定 {coarse["intervals"]} 个径向区间，随 Rout 伸缩；每点实际 ΔR 保存在 summary.json 中。ω 为惯性系实频率，时间单位与阶段 A 相同。',
              'A1²=∫|q1,velocity|² dz dR；谐波能量占比含共轭模态的因子 2。v_rms/(RΩ)=sqrt(2∑_{j≥1}|q_v,j(z=1.3)|²)，不含均值修正。',
              '前沿沿用 Ro=−1 的定义：在 R∈[550,min(650,a)] 且基频幅值大于其峰值 1e−8 的区间，寻找 d ln(v_rms/(RΩ))/dR 与同 Ro、同 β 的局部夹点 −Im α0 的第一个向下交点。无交点时保留缺失值；所有交点和搜索区间已保存。',
              '海绵能量分别报告 q 范数与物理柱坐标 R³ 权重；外海绵区间 [a,Rout]，入口海绵区间 [400,420]。均值不计入相干扰动海绵占比。',
              '每个接受根均检查原始无强迫物理方程、独立逐半径矩阵重组、相位和分支约束；最高谐波/基频超过 1e−2 的场不会列为可信分支点。',
              '两个弱非线性点的 A1² 外推及中性极限误差见 weak_seed_verification.json。dRout/ds 转向及正向割线条件用于识别上支；回返不作为新的域长收敛证据。',
              '单点硬预算含子进程启动、矩阵组装、Newton、独立复核和保存；默认 1200 秒。单线程串行执行，不用 GPU，地址空间上限 200 GiB。',
              'commands.sh 供主管后台启动；相同命令从 campaign.json 恢复。源文件和阶段 A 种子、夹点曲线的哈希见 source_contract.json。',
              'summary.json、accepted_points.json 和 diagnostics/*.npz 保存数据链；PNG/PDF 由所附 Python 脚本重建。图像自动检查见 figures/visual_audit.json，人工逐图审阅尚待主管完成。',
              'NUMERICAL_DONE 仅表示全部目标和复核完成；NUMERICAL_STOPPED.json 表示按已记录原因停止。WORKFLOW_DONE.json 表示报告生成结束，不能替代科学完成状态。',
              '任务书只给出 Xie 的数值，未给出可叠加的 DNS 径向曲线或非线性参考频率；图中仅显示已给定的 Re=471.8 标记。']
    (out/'REPORT.md').write_text('\n'.join(lines)+'\n')
    render(out, summary)
    return summary


if __name__ == '__main__':
    from ro0751_phaseB_campaign import OUT, prerequisites
    parser = argparse.ArgumentParser()
    parser.add_argument('--out', type=Path, default=OUT)
    args = parser.parse_args()
    if args.out.resolve() != OUT:
        raise ValueError('Unexpected output directory')
    finalize(args.out, prerequisites())
