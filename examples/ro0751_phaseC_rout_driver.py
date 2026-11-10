#!/usr/bin/env python3
"""Read-only Phase B seeds; resumable serial Phase C fixed-Rout continuation.

The inherited H4 implementation supplies the original phase reference. Numerical
workers use the unchanged original solver. New output is confined to Phase C.
Truncation is the solver's integrated energy ratio E_H/E_1, not a local amplitude.
"""
from __future__ import annotations
import os
for _k in ('OPENBLAS_NUM_THREADS', 'OMP_NUM_THREADS', 'MKL_NUM_THREADS',
           'NUMEXPR_NUM_THREADS', 'VECLIB_MAXIMUM_THREADS', 'BLIS_NUM_THREADS',
           'GOTO_NUM_THREADS'):
    os.environ[_k] = '1'
for _k in tuple(os.environ):
    if _k.endswith('_NUM_THREADS'):
        os.environ[_k] = '1'
import argparse
import fcntl
import gc
import hashlib
import json
import math
from pathlib import Path
import resource
import shutil
import signal
import subprocess
import sys
import time
sys.dont_write_bytecode = True
os.environ['PYTHONDONTWRITEBYTECODE'] = '1'

ROOT = Path(__file__).resolve().parents[1]
os.environ.setdefault('MPLCONFIGDIR', str(ROOT/'results/ro0751_global_20261101/phaseC_rout/mplconfig'))
sys.path[:0] = [str(ROOT/'examples'), str(ROOT/'src')]
import numpy as np
from ro0751_phaseB_h4_driver import H4Campaign
import ro0751_phaseB_campaign as C
from ro0751_phaseB_fronts import diagnose

OUT = ROOT/'results/ro0751_global_20261101/phaseC_rout'
OLD = ROOT/'results/ro0751_global_20261101/phaseB'
STATE = OUT/'campaign_rout.json'
ENERGY_LIMIT = .01
FRONT_LIMIT = .1
MEMORY_BYTES = 200_000_000_000


def read(path):
    return json.loads(Path(path).read_text())


def write(path, data):
    C.write(path, data)


def sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as f:
        for block in iter(lambda: f.read(1024*1024), b''):
            h.update(block)
    return h.hexdigest()


def admissible(d):
    """Fail closed: equation acceptance alone is insufficient for truncation."""
    ratio = d.get('highest_to_fundamental')
    return bool(d.get('accepted') and ratio is not None and
                math.isfinite(ratio) and 0 <= ratio <= ENERGY_LIMIT and
                d.get('independent', {}).get('relative_field_inf', math.inf) < 1e-7 and
                d.get('independent', {}).get('template_difference_relative', math.inf) <= 1e-10)


def overlap_result(lo, hi, allow_rejected_low=False):
    keys = ('front_fig8', 'front_2beta_1e6')
    finite = all(x.get(k) is not None and math.isfinite(x[k])
                 for x in (lo, hi) for k in keys)
    diff = {k: abs(hi[k]-lo[k]) if finite else None for k in keys}
    same = abs(lo['Rout']-hi['Rout']) < 1e-8 and lo['n'] == hi['n']
    low_quality = lo.get('quality_equations_accepted',False) if allow_rejected_low else lo['quality_accepted']
    return dict(allow_rejected_low=allow_rejected_low,low_curve_accepted=lo['quality_accepted'],low=lo['id'], high=hi['id'], Rout=lo['Rout'], n=lo['n'],
                H_low=lo['H'], H_high=hi['H'], front_absolute_differences=diff,
                threshold_strictly_less_than=FRONT_LIMIT,
                passed=bool(same and finite and all(v < FRONT_LIMIT for v in diff.values())
                            and low_quality and hi['quality_accepted']))


class CampaignPause(Exception):
    pass


class RoutCampaign(H4Campaign):
    """Override output/state/worker orchestration; never invoke Phase B init/save."""
    def __init__(self, point_budget=3600, wall_budget=86400, step=2.):
        self.out = OUT
        self.start = time.monotonic()
        self.budget = point_budget
        self.deadline = self.start+wall_budget
        self.domain = read(OUT/'xie_domain.json')
        self.baseline = read(OUT/'source_baseline.json')
        for p, digest in self.baseline.items():
            if sha(ROOT/p) != digest:
                raise RuntimeError('Protected source changed: '+p)
        self.mapping = None
        self.metric = None
        if STATE.exists():
            self.state = read(STATE)
            if self.state.get('scope') != 'n40_followup_20261102' and self.state['target_Rout'] != self.domain['target_Rout']:
                raise RuntimeError('Domain target changed; a reviewed state migration is required')
        else:
            old = read(OLD/'campaign_h4.json')
            pts = [dict(p, origin='phaseB_read_only') for p in old['points']
                   if p.get('upper_branch') or p['id'] in
                   ('h4_check_N40_dr025', 'h4_check_H5')]
            assert any(p['id'] == 'h4_branch_0023_try0' for p in pts)
            self.state = dict(schema=1, status='pending_supervisor', stage='prepared',
                stop='long_calculation_not_started', points=pts, attempts=[], overlaps=[],
                active_id='h4_branch_0023_try0',
                continuation_ids=[p['id'] for p in pts if p.get('upper_branch')],
                target_Rout=self.domain['target_Rout'],
                comparable_Rout=self.domain['comparable_Rout'],
                n40_targets=self.domain['n40_targets'], mesh_checks={},
                next_step=step, energy_ratio_limit=ENERGY_LIMIT,
                overlap_front_limit=FRONT_LIMIT, folds_N40='pending_low_priority',
                resource_contract=dict(threads=1, max_processes=2,
                    per_process_RLIMIT_AS_bytes=MEMORY_BYTES, total_memory_ceiling_bytes=600_000_000_000,
                    scheduling='one worker plus a single supervising driver; serial solves'),
                source_hashes=self.baseline, history=[])
        self.state.setdefault('references',[])
        self.rows = []
        self.reference_rows = []

    def save(self):
        self.state['elapsed_current_invocation'] = time.monotonic()-self.start
        self.state['driver_sha256'] = sha(Path(__file__))
        write(STATE, self.state)

    def point(self, point_id):
        return next(p for p in self.state['points'] if p['id'] == point_id)

    def row(self, point_id):
        return next(r for r in self.rows+self.reference_rows if r['id'] == point_id)

    def diagnostic(self, p):
        path = ROOT/p['fields']
        cache = OUT/'diagnostics'/f"{p['id']}.json"
        stat = path.stat()
        signature = dict(path=p['fields'], size=stat.st_size, mtime_ns=stat.st_mtime_ns,
                         diagnostic_source_sha256=sha(ROOT/'examples/ro0751_phaseB_fronts.py'))
        if cache.exists() and read(cache).get('source') == signature:
            d = read(cache)['diagnostic']
        else:
            raw = diagnose(path)
            raw['highest_local_v_amplitude_ratio'] = raw.pop('highest_to_fundamental')
            raw.pop('A1', None)
            d = raw
            write(cache, dict(source=signature, diagnostic=d))
        d = dict(d)
        d.update(id=p['id'], n=p['n'], H=p['H'], A1=p['A1'], omega=p['omega'],
                 omega_disk=p['omega']*.9145, a=p['Rout']-100,
                 intervals=p['intervals'], dr=(p['Rout']-400)/p['intervals'],
                 role=p['role'], fields=p['fields'], origin=p.get('origin', 'phaseC'),
                 highest_to_fundamental_energy_ratio=p['highest_to_fundamental'],
                 quality_accepted=admissible(p),
                 quality_equations_accepted=bool(p.get('equations_accepted',p.get('accepted')) and
                    p.get('independent',{}).get('relative_field_inf',math.inf)<1e-7 and
                    p.get('independent',{}).get('template_difference_relative',math.inf)<=1e-10))
        if abs(d['Rout']-p['Rout']) > 1e-7:
            raise RuntimeError('Saved field/domain mismatch for '+p['id'])
        return d

    def refresh(self):
        self.rows = [self.diagnostic(p) for p in self.state['points']]
        self.reference_rows = [self.diagnostic(p) for p in self.state['references']]
        for item in self.state['overlaps']:
            fresh = overlap_result(self.row(item['low']), self.row(item['high']),item.get('allow_rejected_low',False))
            item.update(fresh)
        comparable = [r for r in self.rows if r['n'] == 40 and
                      abs(r['Rout']-self.domain['comparable_Rout']) < 1e-8]
        continuation = [self.row(i) for i in self.state['continuation_ids']]
        tail = sorted(continuation, key=lambda r: r['Rout'])[-3:]
        tail_ranges = {k: (max(r[k] for r in tail)-min(r[k] for r in tail)
                          if len(tail) == 3 and all(r[k] is not None for r in tail) else None)
                       for k in ('front_fig8', 'front_2beta_1e6')}
        d = dict(status=self.state['status'], pending=self.state['status'] != 'complete',
                 stop=self.state.get('stop'), target_Rout=self.domain['target_Rout'],
                 comparable_Rout=self.domain['comparable_Rout'],
                 coordinate='R=Re/0.751; tabular rounded-Red convention',
                 reference_DNS_NL1_front_R=628.2, reference_DNS_NL1_front_Re=471.8,
                 criteria=dict(fig8='first downward crossing of dln RMS(v,z=1.3)/dR minus local absolute growth, 550<=R<=min(650,a)',
                               two_beta='first upward arrival |v2(z=1.3)|=1e-6, 420<R<a'),
                 truncation_metric='integrated harmonic kinetic energy E_H/E_1<=0.01; local v amplitude ratio is a different diagnostic',
                 continuation_ids=self.state['continuation_ids'], rows=self.rows,
                 rejected_truncation_references=self.reference_rows,
                 overlaps=self.state['overlaps'], n40_checks=self.state['mesh_checks'],
                 tail_last_three_front_ranges=tail_ranges,
                 convergence_status='pending_target_and_mesh_checks' if self.state['status'] != 'complete'
                 else 'finite_domain_sequence_measured; inspect_tail_and_mesh_differences',
                 comparable_N40_difference_from_628_2=[dict(id=r['id'],
                     fig8=None if r['front_fig8'] is None else r['front_fig8']-628.2,
                     two_beta=None if r['front_2beta_1e6'] is None else r['front_2beta_1e6']-628.2)
                     for r in comparable],
                 data_integrity='Only accepted computed fields are plotted; no extrapolated/predicted solution points.')
        write(OUT/'fronts_vs_rout.json', d)
        self.save()
        self.plot(d)
        self.report(d)
        self.pending()

    def report(self, d):
        active = self.point(self.state['active_id'])
        estimate = read(OUT/'runtime_estimate.json')
        lines = ['# Ro=-0.751, β=85：阶段 C 径向外边界续算', '',
          f"状态：**{d['status']}**；停止原因：`{d.get('stop')}`。",
          f"目标Rout={d['target_Rout']:.12f}；当前活动点Rout={active['Rout']:.9f}（N{active['n']}/H{active['H']}）。",
          '本轮仅准备及既有场诊断；数值完成状态由真实求解结果自动更新。' if not self.state['attempts'] else
          f"阶段C已经记录{len(self.state['attempts'])}次尝试（成功/失败均保留），不把求解失败作为已接受点。", '',
          '## NL1域与换算证据', '',
          '§5.2 DNS原域为盘面坐标500≤r≤900；NL1表4为rd=900、Red=646.5、rs=820、Res=589.1。',
          f"按既有前沿一致的Re=0.751R，精确执行的表格约定Rout={d['comparable_Rout']:.12f}；我们的a={d['comparable_Rout']-100:.12f}。",
          f"Xie表格海绵起点转换为R={self.domain['xie_outlet_start_R_from_rounded_Res']:.12f}。",
          '文献的Re=r|Ro|√(Ω/Ωd)及表4支持R≈0.956556 r_disk；任务旧写法r_disk=0.956556R互逆，',
          f"按旧字面式会得Rout={self.domain['conversion_conflict']['literal_outlet_R']:.6f}，与表4不相容，故保留为冲突分支并不用于主比较。",
          f"原始长度换算900×0.956556={self.domain['nl1_converted_from_disk']['outlet_R']:.6f}与表格主比较差{self.domain['conversion_conflict']['raw_length_minus_rounded_Red_R']:.6f}；Red一位小数对应R约±0.0666。",
          '入口：通用式在r≤410阻尼1、410–420过渡，文中又称所有case相同；DNS入口却为500。该公式海绵不在NL1域内，',
          'NL1入口海绵的实际域内位置未明确，不能捏造为500–520。Fig.12(c)仅对应NL1，不能挪用线性L5或NL2海绵。',
          'NL1出口系数10，rs=820、过渡40到860、平台40到900；本代码只阻尼非零谐波，HopfSponge(Rout−100,40,28)，',
          '平台60，入口400及0.8系数/410–420过渡；径向和法向截断、系数及均值阻尼均不等价。这里只匹配出口Red，不宣称复制DNS域/边界。',
          '原文逐行证据及行号约定见xie_domain.json。', '', '## 真实已接受点', '',
          '| id | Rout | a | 网格/谐波 | Fig.8前沿 | 2β前沿 | E_H/E1 | ω/Ωd |',
          '|---|---:|---:|---|---:|---:|---:|---:|']
        fmt = lambda x: '未检出' if x is None else f'{x:.6f}'
        for r in self.rows:
            lines.append(f"| {r['id']} | {r['Rout']:.6f} | {r['a']:.6f} | N{r['n']}/H{r['H']}, ΔR={r['dr']:.4f} | {fmt(r['front_fig8'])} | {fmt(r['front_2beta_1e6'])} | {r['highest_to_fundamental_energy_ratio']:.7f} | {r['omega_disk']:.6f} |")
        lines += ['', 'Fig.8为z=1.3的ln RMS(v)增长率与局部增长率首个下行交点；2β为|v₂|首次上行到达10⁻⁶；均限制在海绵起点a以内。',
          '最高谐波验收严格使用原solver能量比；局部v幅值比另存，禁止混作0.01阈值。局部绝对增长率基准仍为阶段A已保存N32曲线，N40复核改变全局场网格；若N40需升H，同Rout也计算N32的同H解并验收重叠，最终网格差取相同H。', '', '## 谐波重叠与N40复核', '']
        if self.state['overlaps']:
            for item in self.state['overlaps']:
                lines.append(f"- {item['low']} / {item['high']}，同Rout={item['Rout']:.9f}：{item['front_absolute_differences']}，通过={item['passed']}（两判据均严格<0.1）。")
        else:
            lines.append('尚未验收谐波重叠；已有同Rout的H4/H5场需真实诊断后才能切换。')
        for t in self.state['n40_targets']:
            ck = self.state['mesh_checks'].get(f'{t:.12f}')
            lines.append(f"- N40目标Rout={t:.12f}："+('pending' if ck is None else json.dumps(ck, ensure_ascii=False)))
        lines += ['', '## 收敛结论', '',
          f"末三延拓点的前沿全距：{d['tail_last_three_front_ranges']}；这只是已计算区间的变化量，不替代N40检验或无穷域收敛证明。"]
        if d['pending']:
            lines += ['**尚不能判断目标区间是否收敛**；可比Rout处的前沿及与628.2的差仍pending。图中只画真实已有点，目标和未计算区间明确标注。']
        else:
            lines += [f"目标及三处N40检查已经完成；可比Rout的N40前沿差为{d['comparable_N40_difference_from_628_2']}。",
                      '结合末段变化与N32/N40差报告有限域敏感性；不以仅完成续算自动宣称前沿收敛。']
        lines += ['', '## 耗时、资源与续接', '',
          f"已有N32/H5耗时{estimate['existing_H5_seconds']:.2f}s，N40/H4耗时{estimate['existing_N40_H4_seconds']:.2f}s；",
          f"ΔR=2至少{estimate['minimum_nominal_steps']}步，未计H6/重试的预算下界约{(estimate['nominal_H5_continuation_seconds']+estimate['n40_minimum_H4_seconds'])/3600:.2f}小时；本轮未启动长求解。",
          'commands.sh使用flock，串行子进程，所有数值线程1，driver及worker各受200,000,000,000字节RLIMIT_AS约束，最多两进程且总上限400GB<600GB。',
          '每次求解后刷新JSON/PDF/PNG/报告；失败与时间预算中断保留checkpoint和pending状态，同一命令可续接，旧失败结果归档不缓存卡死。',
          'H4→H5、必要H5→H6均在同Rout保留真解，前沿差≥0.1或H6仍超0.01时停止并保留PENDING_SUPERVISOR。',
          'N40也执行谐波阈值，必要时在相同N40网格做H提升重叠；超阈值低H仅作拒绝的诊断对照，绝不进入已接受点/曲线。折点N40为低优先级，本轮未启动。',
          '命令：`bash results/ro0751_global_20261101/phaseC_rout/commands.sh`。',
          '原solver及所有旧结果由source_baseline.json哈希保护；没有修改phaseB文件。', '',
          '## 本轮实际短检查', '',
          '语法compile（无pyc）、--help、验收逻辑self-check、既有21个真实场前沿诊断及同Rout H4/H5复核已完成；未启动生产求解。',
          'N8/H3/12 intervals真实worker入口在budget=0下返回BudgetExpired、accepted=false；实际RLIMIT_AS=200,000,000,000字节，所有线程环境1。',
          '隔离失败注入覆盖预算归档恢复、独立质量失败缓存重试、半写result恢复、H5截断升H6、overlap失败拒绝及H6上限；见short_checks/verification.json。',
          '渲染字体下限与碰撞审计见figure_qa_status.json；源验证器的sans/SVG/TIFF默认与本任务serif/PDF+PNG约定不一致，处置见figure_source_dispositions.json。']
        (OUT/'REPORT.md').write_text('\n'.join(lines)+'\n')

    def plot(self, d):
        import matplotlib
        matplotlib.use('Agg')
        import matplotlib.pyplot as plt
        plt.rcParams.update({'font.family':'serif', 'font.serif':['STIXGeneral', 'DejaVu Serif'],
            'mathtext.fontset':'stix', 'font.size':8, 'axes.linewidth':.8,
            'pdf.fonttype':42, 'svg.fonttype':'none', 'axes.spines.top':False, 'axes.spines.right':False,
            'xtick.direction':'in', 'ytick.direction':'in'})
        fig, ax = plt.subplots(figsize=(7.2, 3.7))
        fig.subplots_adjust(left=.10, right=.97, bottom=.16, top=.86)
        active = [self.row(i) for i in self.state['continuation_ids']]
        active = sorted(active, key=lambda x:x['Rout'])
        colors = ('#2166ac', '#b2182b')
        for k, marker, label, color in zip(('front_fig8','front_2beta_1e6'), ('o','s'),
                ('Fig. 8 criterion', r'$2\beta$, $|v_2|=10^{-6}$'), colors):
            vals = [(r['Rout'],r[k]) for r in active if r[k] is not None and r['quality_accepted']]
            if vals:
                xx, yy = zip(*vals)
                ax.plot(xx, yy, marker=marker, markersize=3, linewidth=1.2, color=color, label=label)
            fine = [(r['Rout'],r[k]) for r in self.rows if r['n']==40 and r[k] is not None and r['quality_accepted']]
            if fine:
                xx, yy = zip(*fine)
                ax.scatter(xx, yy, marker=marker, s=28, facecolors='none', edgecolors=color,
                           linewidths=1, zorder=4, label='N40, '+label)
        ax.axhline(628.2, color='.35', ls=':', lw=1, label='DNS NL1: 628.2')
        ax.axvline(d['comparable_Rout'], color='.5', ls='--', lw=.9, ymax=.56)
        ax.set_xlim(min(r['Rout'] for r in active)-1, d['target_Rout']+2)
        ax.set_xlabel(r'Outer boundary $R_{\mathrm{out}}$')
        ax.set_ylabel(r'Front position $R$')
        ax.legend(loc='upper center', bbox_to_anchor=(.50,1.), frameon=False, fontsize=8, ncol=2)
        status = 'PENDING: only computed points shown' if d['pending'] else 'Computed sequence; see mesh checks'
        fig.suptitle(status, x=.10, y=.97, ha='left', fontsize=9)
        fig.canvas.draw()
        ax.text(d['comparable_Rout']-2, ax.get_ylim()[0]+.5, 'NL1 outlet', ha='right', va='bottom', fontsize=7)
        fig.savefig(OUT/'fronts_vs_rout.png', dpi=300)
        fig.savefig(OUT/'fronts_vs_rout.pdf')
        plt.close(fig)
        qa = Path('/home/lthpc/.codex/skills/nature-figure/scripts')
        audits = {}
        for label, argv in (
            ('text', [str(qa/'audit_pdf_text.py'),str(OUT/'fronts_vs_rout.pdf'),'--json']),
            ('collisions', [str(qa/'audit_figure_collisions.py'),str(OUT/'fronts_vs_rout.pdf'),
                            '--json-out',str(OUT/'fronts_vs_rout.collision-audit.json')])):
            proc = subprocess.run([sys.executable,*argv],cwd=ROOT,capture_output=True,text=True,timeout=60)
            (OUT/f'figure_{label}_audit.log').write_text(proc.stdout+proc.stderr)
            audits[label] = dict(exit_code=proc.returncode,log=f'figure_{label}_audit.log')
        write(OUT/'figure_qa_status.json',audits)
        write(OUT/'figure_contract.json', dict(backend='Python/matplotlib', archetype='single quantitative comparison',
            conclusion='Only the computed part of the front-versus-Rout sequence is available; target/mesh work remains pending.' if d['pending'] else 'Measured finite-domain front sensitivity with N40 checks.',
            evidence='Two front definitions, true accepted N32 continuation and N40 checks; DNS reference and NL1 outlet.',
            dimensions_inches=[7.2,3.7], formats=['png','pdf'], transformations='No smoothing or extrapolation; lines connect genuine accepted continuation points only.',
            excluded='Lower-branch points and unaccepted truncation/failed solves; retained in source or attempt history.',
            multi_panel_alignment='not applicable: one plot area'))

    def pending(self):
        p = OUT/'PENDING_SUPERVISOR'
        if self.state['status'] == 'complete':
            p.unlink(missing_ok=True)
            return
        p.write_text('Status: '+self.state['status']+'\nReason: '+str(self.state.get('stop'))+
            '\nExpected duration exceeds one hour; agent did not launch numerical work.\n'
            'Supervisor launch/resume from repository root:\n'
            'bash results/ro0751_global_20261101/phaseC_rout/commands.sh\n'
            'Resume agent after commands.sh exits; inspect command_status.json and campaign_rout.json.\n'
            'Exit 0=target plus three N40 checks complete; 75=budget/interruption resumable; '
            '2=scientific/worker stop requiring review; 1=unexpected error.\n'
            'No polling required; outputs refresh after each completed/failed attempt.\n')

    def mapping_for(self, n, H, Rout):
        intervals = int(math.ceil((Rout-400)/(.25 if n == 40 else .5)))
        if self.mapping is None or (self.mapping.n,self.mapping.H,self.mapping.intervals) != (n,H,intervals):
            self.mapping = None
            gc.collect()
            self.mapping = C.Mapping(n, H, intervals)
        return self.mapping

    def attempt(self, source, Rout, n, H, role, previous=None):
        self.check_budget()
        name = f"r_N{n}_H{H}_R{Rout:.12f}".replace('.', 'p')
        folder = OUT/'attempts'/name
        folder.mkdir(parents=True, exist_ok=True)
        spec = dict(n=n, H=H, intervals=int(math.ceil((Rout-400)/(.25 if n == 40 else .5))), ro=-.751, beta=85.)
        descriptor = dict(mapping=spec, mode='Rout', target=float(Rout), source=source['fields'],
                          source_sha256=sha(ROOT/source['fields']))
        signature = hashlib.sha256(json.dumps(descriptor, sort_keys=True).encode()).hexdigest()
        result_dir = folder/'result'
        summary = result_dir/'summary.json'
        old_request = read(folder/'request.json') if (folder/'request.json').exists() else None
        if old_request and old_request.get('operation_signature') != signature:
            raise RuntimeError('Restart operation changed: '+name)
        cached = read(summary) if summary.exists() else None
        cached_quality = bool(cached and cached.get('accepted') and
            cached.get('independent',{}).get('relative_field_inf',math.inf)<1e-7 and
            cached.get('independent',{}).get('template_difference_relative',math.inf)<=1e-10)
        if cached_quality:
            d = cached
        else:
            mapping = self.mapping_for(n,H,Rout)
            y = C.transfer(ROOT/source['fields'], mapping)
            if previous is not None and abs(source['Rout']-previous['Rout']) > 1e-7:
                yp = C.transfer(ROOT/previous['fields'], mapping)
                mult = min(2., (Rout-source['Rout'])/(source['Rout']-previous['Rout']))
                if mult > 0:
                    y += mult*(y-yp)
            y[-1] = Rout
            if result_dir.exists():
                checkpoint = result_dir/'checkpoint.npz'
                if checkpoint.exists():
                    # Only reuse an interrupted/budget iterate at the exact requested Rout.
                    ck = np.load(checkpoint)
                    if abs(float(ck['Rout'])-Rout) < 1e-7 and np.isfinite(ck['field']).all():
                        y = C.transfer(checkpoint, mapping)
                    ck.close()
                archives = folder/'archive'
                archives.mkdir(exist_ok=True)
                index = len(list(archives.iterdir()))
                result_dir.rename(archives/f'result_{index:04d}')
            qref = self.qref(y,mapping)
            np.savez_compressed(folder/'input.npz', y=y,qref=qref,tangent=np.array([]))
            budget = min(self.budget, self.deadline-time.monotonic()-15)
            if budget < 30:
                raise CampaignPause('campaign_wall_budget_before_worker')
            request = dict(mapping=spec, mode='Rout', target=float(Rout), input=str(folder/'input.npz'),
                output=str(result_dir), budget=budget, request_sha256=signature,
                operation_signature=signature, weak_predictor=False, rescale_energy=False)
            write(folder/'request.json', request)
            self.state.update(stage='worker_running', current_operation=dict(id=name,source=source['id'],Rout=Rout,n=n,H=H,role=role))
            self.save()
            begin = time.monotonic()
            proc = None
            try:
                with (folder/'worker.log').open('a') as log:
                    proc = subprocess.Popen([sys.executable, str(Path(__file__).resolve()),
                        '--worker-request', str(folder/'request.json')], cwd=ROOT,stdout=log,stderr=log)
                    code = proc.wait(timeout=budget+10)
                d = read(summary) if summary.exists() else dict(accepted=False,failure=f'worker_exit_{code}_without_summary')
                if code != 0:
                    d.update(accepted=False,failure=f'worker_exit_{code}')
            except subprocess.TimeoutExpired:
                proc.terminate()
                try:
                    proc.wait(timeout=10)
                except subprocess.TimeoutExpired:
                    proc.kill()
                    proc.wait()
                d = read(summary) if summary.exists() else dict(accepted=False)
                d.update(accepted=False,failure='hard_per_point_timeout',budget_stop=True)
            except BaseException:
                if proc is not None and proc.poll() is None:
                    proc.terminate()
                    try:
                        proc.wait(timeout=10)
                    except subprocess.TimeoutExpired:
                        proc.kill()
                        proc.wait()
                raise
            d.update(seconds=time.monotonic()-begin,request_sha256=signature,
                     operation_signature=signature)
            write(summary,d)
        accepted_equations = bool(d.get('accepted'))
        ratio = d.get('highest_to_fundamental')
        truncation_rejected = bool(accepted_equations and (ratio is None or not math.isfinite(ratio) or ratio > ENERGY_LIMIT))
        branch_ok = not (role == 'continuation' and accepted_equations and
                         d.get('A1', 0) < .9*source['A1'])
        record = dict(id=name, role=role, Rout=Rout, n=n,H=H,
            accepted=bool(admissible(d) and branch_ok), equations_accepted=accepted_equations,
            truncation_rejected=truncation_rejected, highest_to_fundamental_energy_ratio=ratio,
            seconds=d.get('seconds'), failure=d.get('failure'), budget_stop=d.get('budget_stop',False),
            summary=str(summary.relative_to(ROOT)))
        if not branch_ok:
            record['failure'] = 'upper_branch_identity_amplitude_drop_above_10pct'
        if truncation_rejected:
            record['failure'] = 'highest_harmonic_energy_ratio_above_0.01'
        self.state['attempts'] = [r for r in self.state['attempts'] if r['id'] != name]+[record]
        self.state.setdefault('attempt_history',[]).append(dict(record))
        point = None
        if record['accepted']:
            if abs(d['Rout']-Rout) > 1e-7 or not (result_dir/'fields.npz').exists():
                raise RuntimeError('Accepted result does not match requested domain')
            point = {k:d[k] for k in ('Rout','omega','A1','energies','highest_to_fundamental',
                'relative_field_inf','residual_over_fundamental_feedback','phase','branch_constraint',
                'n','H','intervals','independent')}
            point.update(id=name,role=role,accepted=True,origin='phaseC',
                fields=str((result_dir/'fields.npz').relative_to(ROOT)),summary=str(summary.relative_to(ROOT)))
            if not any(p['id']==name for p in self.state['points']):
                self.state['points'].append(point)
        if truncation_rejected and (result_dir/'fields.npz').exists():
            reference = {k:d[k] for k in ('Rout','omega','A1','energies','highest_to_fundamental',
                'relative_field_inf','residual_over_fundamental_feedback','phase','branch_constraint',
                'n','H','intervals','independent')}
            reference.update(id=name,role='rejected_truncation_reference',accepted=False,
                equations_accepted=True,origin='phaseC_rejected_control',
                fields=str((result_dir/'fields.npz').relative_to(ROOT)),summary=str(summary.relative_to(ROOT)))
            self.state['references'] = [r for r in self.state['references'] if r['id']!=name]+[reference]
        self.state.update(stage='worker_finished',current_operation=None)
        self.refresh()
        if record['budget_stop']:
            raise CampaignPause('per_point_budget_exceeded; retry resumes checkpoint')
        return point,record

    def check_budget(self):
        if self.deadline-time.monotonic() < 45:
            raise CampaignPause('campaign_wall_budget')

    def add_overlap(self, low, high, allow_rejected_low=False):
        self.refresh()
        item = overlap_result(self.row(low['id']),self.row(high['id']),allow_rejected_low)
        self.state['overlaps'] = [r for r in self.state['overlaps']
            if (r['low'],r['high']) != (low['id'],high['id'])]+[item]
        self.refresh()
        if not item['passed']:
            raise RuntimeError('Harmonic overlap not accepted: '+json.dumps(item))

    def elevate(self, source):
        if source['H'] >= 6:
            raise RuntimeError('H6 truncation unresolved; no relaxation or H7 allowed')
        H = source['H']+1
        if source['id']=='h4_branch_0023_try0' and H==5:
            high = self.point('h4_check_H5')
        else:
            high,record = self.attempt(source,source['Rout'],source['n'],H,'harmonic_overlap')
            if high is None:
                raise RuntimeError('Higher harmonic overlap solve failed: '+json.dumps(record))
        self.add_overlap(source,high)
        return high

    def check_n40(self, coarse):
        key = f"{coarse['Rout']:.12f}"
        prior = self.state['mesh_checks'].get(key)
        if prior and prior.get('accepted') and all(
                prior.get('front_differences',{}).get(k) is not None and
                math.isfinite(prior['front_differences'][k])
                for k in ('front_fig8','front_2beta_1e6')):
            return
        self.state['mesh_checks'].pop(key,None)
        fine,record = self.attempt(coarse,coarse['Rout'],40,coarse['H'],'N40_check')
        if fine is None and record['truncation_rejected']:
            if coarse['H']>=6:
                raise RuntimeError('N40 H6 truncation unresolved; maximum H6 retained')
            low = next(p for p in self.state['references'] if p['id']==record['id'])
            fine,higher = self.attempt(low,low['Rout'],40,coarse['H']+1,'N40_harmonic_overlap')
            if fine is None:
                raise RuntimeError('N40 higher-H check failed: '+json.dumps(higher))
            self.add_overlap(low,fine,allow_rejected_low=True)
            # Isolate radial/normal-grid error with equal H on coarse and fine grids.
            coarse = self.elevate(coarse)
        if fine is None:
            raise RuntimeError('N40 check failed: '+json.dumps(record))
        self.refresh()
        cr,fr = self.row(coarse['id']),self.row(fine['id'])
        differences = {k:None if cr[k] is None or fr[k] is None else fr[k]-cr[k]
                       for k in ('front_fig8','front_2beta_1e6')}
        if not all(v is not None and math.isfinite(v) for v in differences.values()):
            raise RuntimeError('N40 front criterion missing; no completed check committed')
        self.state['mesh_checks'][key] = dict(coarse_id=coarse['id'],fine_id=fine['id'],
            Rout=coarse['Rout'],coarse_H=coarse['H'],fine_H=fine['H'],accepted=True,
            front_differences=differences)
        self.refresh()

    def run(self):
        if self.state.get('scope') == 'n40_followup_20261102':
            raise RuntimeError('Rout extension closed by supervisor; use N40 follow-up commands.sh')
        self.state.update(status='running',stop=None,stage='continuation')
        self.refresh()
        active = self.point(self.state['active_id'])
        if active['H']==4:
            active = self.elevate(active)
            self.state['active_id'] = active['id']
            if active['id'] not in self.state['continuation_ids']:
                self.state['continuation_ids'].append(active['id'])
            self.refresh()
        # Repair a completed coarse target whose N40 check was interrupted before registration.
        for t in self.state['n40_targets']:
            candidates = [p for p in self.state['points'] if p['n']==32 and abs(p['Rout']-t)<1e-8]
            if candidates:
                self.check_n40(max(candidates,key=lambda p:p['H']))
        step = self.state['next_step']
        while active['Rout'] < self.state['target_Rout']-1e-9:
            self.check_budget()
            remaining_targets = [t for t in self.state['n40_targets']+[self.state['target_Rout']]
                                 if t>active['Rout']+1e-9]
            target = min(active['Rout']+step,min(remaining_targets))
            prim = [self.point(i) for i in self.state['continuation_ids']
                    if self.point(i)['n']==32 and self.point(i)['H']==active['H'] and
                    self.point(i)['Rout']<active['Rout']-1e-7]
            previous = max(prim,key=lambda p:p['Rout']) if prim else None
            point,record = self.attempt(active,target,32,active['H'],'continuation',previous)
            if point is None:
                if record['truncation_rejected']:
                    active = self.elevate(active)
                    self.state['active_id'] = active['id']
                    if active['id'] not in self.state['continuation_ids']:
                        self.state['continuation_ids'].append(active['id'])
                    self.refresh()
                    continue
                step *= .5
                self.state['next_step'] = step
                self.refresh()
                if step < .125:
                    raise RuntimeError('Continuation corrector failed down to minimum step: '+json.dumps(record))
                continue
            if point['A1'] < .9*active['A1']:
                raise RuntimeError('Unexpected upper-branch amplitude decrease >10%; accepted equations retained for review')
            active = point
            self.state['active_id'] = active['id']
            if active['id'] not in self.state['continuation_ids']:
                self.state['continuation_ids'].append(active['id'])
            # Persist branch advancement before a potentially interrupted N40 check.
            step = min(2.,step*1.25)
            self.state['next_step'] = step
            self.refresh()
            if any(abs(active['Rout']-t)<1e-8 for t in self.state['n40_targets']):
                self.check_n40(active)
        if len(self.state['mesh_checks']) != 3:
            raise RuntimeError('Target reached without three complete N40 checks')
        if not all(x['passed'] for x in self.state['overlaps']):
            raise RuntimeError('Unaccepted harmonic overlap remains')
        self.state.update(status='complete',stage='done',stop=None)
        self.refresh()


def worker_entry(path):
    """Clamp the unchanged legacy worker's attempted AS-limit reset in this process."""
    import ro0751_phaseB_solver as solver
    real_setrlimit = resource.setrlimit
    def bounded_setrlimit(kind, limits):
        if kind == resource.RLIMIT_AS:
            _, current_hard = resource.getrlimit(kind)
            ceiling = min(MEMORY_BYTES, current_hard) if current_hard != resource.RLIM_INFINITY else MEMORY_BYTES
            limits = (min(limits[0], ceiling), min(limits[1], ceiling))
        return real_setrlimit(kind, limits)
    resource.setrlimit = bounded_setrlimit
    try:
        solver.request_worker(path)
    finally:
        request_path = Path(path)
        write(request_path.parent/'worker_entry_limits.json',dict(RLIMIT_AS=resource.getrlimit(resource.RLIMIT_AS),maximum_bytes=MEMORY_BYTES,legacy_solver_sha256=sha(ROOT/'examples/ro0751_phaseB_solver.py'),thread_environments={k:v for k,v in os.environ.items() if k.endswith('_NUM_THREADS')}))
        resource.setrlimit = real_setrlimit


def self_check():
    low=dict(id='low',Rout=780.,n=32,H=4,quality_accepted=True,front_fig8=630.,front_2beta_1e6=625.)
    high=dict(low,id='high',H=5,front_fig8=630.01,front_2beta_1e6=625.02)
    assert overlap_result(low,high)['passed']
    assert not overlap_result(low,dict(high,front_fig8=630.1))['passed']
    assert not overlap_result(low,dict(high,front_2beta_1e6=None))['passed']
    d=dict(accepted=True,highest_to_fundamental=.01,independent=dict(relative_field_inf=1e-8,template_difference_relative=1e-12))
    assert admissible(d)
    assert not admissible(dict(d,highest_to_fundamental=.01000001))
    assert not admissible(dict(d,highest_to_fundamental=float('nan')))
    print('Self-check passed: energy cutoff, strict two-front overlap, nonfinite/missing data fail closed.')


def main(campaign_class=None):
    ap=argparse.ArgumentParser(description=__doc__)
    mode=ap.add_mutually_exclusive_group(required=True)
    mode.add_argument('--prepare',action='store_true',help='Initialize pending state and diagnose existing fields only; no solves')
    mode.add_argument('--refresh',action='store_true',help='Refresh outputs from recorded points only; no solves')
    mode.add_argument('--run',action='store_true',help='Supervisor-only long campaign; resumable')
    mode.add_argument('--worker-request',metavar='JSON',help=argparse.SUPPRESS)
    mode.add_argument('--self-check',action='store_true',help='Short acceptance/overlap logic checks; no solves or output mutations')
    ap.add_argument('--point-budget',type=float,default=3600.)
    ap.add_argument('--wall-budget',type=float,default=86400.)
    args=ap.parse_args()
    _, old_hard = resource.getrlimit(resource.RLIMIT_AS)
    ceiling = MEMORY_BYTES if old_hard == resource.RLIM_INFINITY else min(MEMORY_BYTES,old_hard)
    resource.setrlimit(resource.RLIMIT_AS,(ceiling,ceiling))
    if args.worker_request:
        worker_entry(args.worker_request)
        return 0
    def interrupted(signum, frame):
        raise KeyboardInterrupt('signal '+str(signum))
    signal.signal(signal.SIGTERM,interrupted)
    if args.self_check:
        self_check()
        return 0
    OUT.mkdir(parents=True,exist_ok=True)
    with (OUT/'driver.lock').open('a') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        selected_class = campaign_class or RoutCampaign
        if campaign_class is None and STATE.exists() and read(STATE).get('scope') == 'n40_followup_20261102':
            sys.path.insert(0,str(OUT))
            from n40_followup import FinalMeshCampaign
            selected_class = FinalMeshCampaign
        run=selected_class(args.point_budget,args.wall_budget)
        code=0
        try:
            if args.run:
                run.run()
            else:
                run.refresh()
                # Validate the already available true H4/H5 overlap without launching any worker.
                low=run.point('h4_branch_0023_try0');high=run.point('h4_check_H5')
                run.add_overlap(low,high)
        except (CampaignPause,KeyboardInterrupt) as exc:
            run.state.update(status='paused_resumable',stage='pending_resume',stop=str(exc) or 'interrupted')
            code=75
        except Exception as exc:
            run.state.update(status='stopped_pending_review',stage='pending_review',stop=f'{type(exc).__name__}: {exc}')
            code=2
        finally:
            run.refresh()
        print(json.dumps(dict(status=run.state['status'],stop=run.state.get('stop'),
             points=len(run.state['points']),overlaps=run.state['overlaps'],exit_code=code)))
        return code


if __name__=='__main__':
    raise SystemExit(main())
