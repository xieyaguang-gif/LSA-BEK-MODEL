#!/usr/bin/env python3
"""Short numerical preflight and preparation only; never launches the campaign."""
import os
for _key in ('OPENBLAS_NUM_THREADS', 'OMP_NUM_THREADS', 'MKL_NUM_THREADS',
             'NUMEXPR_NUM_THREADS', 'VECLIB_MAXIMUM_THREADS', 'BLIS_NUM_THREADS'):
    os.environ[_key] = '1'
os.environ.setdefault('MPLCONFIGDIR', '/tmp/ro0751_phaseB_matplotlib')
import argparse
import datetime
import importlib.util
import json
import py_compile
import resource
import subprocess
import sys
import tempfile
from pathlib import Path
from unittest.mock import patch
import numpy as np
from ro0751_phaseB_solver import (ROOT, Mapping, Metric, Corrector, checks, common,
                                  convection, energies, independent, linear, norm,
                                  pack, unpack, write)
from ro0751_phaseB_campaign import (OUT, Campaign, ScientificStop, contract_paths,
                                    fold_fit, prerequisites, read, sha, verify_contract)


def extra_mapping_checks():
    rng = np.random.default_rng(75185)
    mapping = Mapping(8, 4, 80)
    R, w = 764.8, 80.31
    p, blocks, _, derivatives = mapping.at(R, w, True)
    q = unpack(rng.normal(size=mapping.nr*9*mapping.s)*1e-3, mapping.nr, 4, mapping.s)
    y = np.r_[pack(q), w, R]
    direct = independent(mapping, y)
    derivative = pack(linear(derivatives, q, 4)+mapping.convection_R(p, q))
    h = .002
    pp, bp, *_ = mapping.at(R+h, w)
    pm, bm, *_ = mapping.at(R-h, w)
    fd = (pack(linear(bp, q, 4)+convection(pp, q))-pack(linear(bm, q, 4)+convection(pm, q)))/(2*h)
    qref = q[:, 5]/common.mass_norm(p, q[:, 5])
    metric = Metric(mapping)
    tangent = rng.normal(size=len(y))
    tangent /= np.sqrt(metric.dot(tangent, tangent))
    corrector = Corrector(mapping, qref, y, 'arc', 0., metric=metric, tangent=tangent)
    J, *_ = corrector.jacobian(y)
    direction = rng.normal(size=len(y)); direction[:-2] *= 1e-3
    eps = 1e-5
    jfd = (corrector.residual(y+eps*direction)-corrector.residual(y-eps*direction))/(2*eps)
    result = dict(H4_template_difference=direct['template_difference_relative'],
                  H4_Rout_derivative_relative=norm(fd-derivative)/norm(derivative),
                  H4_arc_jacobian_relative=norm(jfd-J@direction)/norm(J@direction))
    # Test the full block fork against the unmodified Ro=-1 implementation.
    from fullbvp_phaseB_solver import Mapping as LegacyMapping
    old, new = LegacyMapping(8, 3, 50), Mapping(8, 3, 50, ro=-1., beta=68.)
    po, bo, _, do = old.at(720., 53., True)
    pn, bn, _, dn = new.at(720., 53., True)
    relative = lambda a,b:float(np.linalg.norm((a-b).data)/max(np.linalg.norm(a.data), 1e-30))
    result['Ro_minus1_block_regression'] = max(relative(a,b) for a,b in zip(bo+do, bn+dn))
    qq = unpack(rng.normal(size=51*7*old.s)*1e-3, 51, 3, old.s)
    result['Ro_minus1_nonlinear_Rout_regression'] = norm(old.convection_R(po, qq)-new.convection_R(pn, qq))/norm(old.convection_R(po, qq))
    # Controls: an unbracketed minimum must never be reported as a fold.
    fixture = [dict(id=f'fixture_{i}', A1=a, Rout=760+10*(a-.2)**2)
               for i,a in enumerate((.16, .18, .2, .22, .24))]
    fit = fold_fit(fixture)
    result['fold_bracket_control'] = bool(abs(fit['Rout']-760) < 1e-10 and abs(fit['A1']-.2) < 1e-10)
    result['fold_extrapolation_rejected'] = fold_fit(fixture[:3])['Rout'] is None
    result['passed'] = bool(result['H4_template_difference'] < 1e-10
                            and result['H4_Rout_derivative_relative'] < 1e-6
                            and result['H4_arc_jacobian_relative'] < 1e-7
                            and result['Ro_minus1_block_regression'] < 1e-12
                            and result['Ro_minus1_nonlinear_Rout_regression'] < 1e-12
                            and result['fold_bracket_control'] and result['fold_extrapolation_rejected'])
    return result


def workflow_checks():
    """Failure-path controls in a disposable directory, with no corrector run."""
    with tempfile.TemporaryDirectory(prefix='workflow_controls_', dir=OUT/'checks') as directory:
        campaign = Campaign.__new__(Campaign)
        campaign.out = Path(directory)
        campaign.budget = 1200.
        campaign.start = __import__('time').monotonic()
        campaign.state = dict(stage='seeds', points=[], attempts=[], stop=None)
        mapping = Mapping(8, 3, 8)
        p, *_ = mapping.at(764.8, 80.31)
        q = np.zeros((mapping.nr, 7, mapping.s), complex)
        q[:, 4, 1] = 1e-3; q[:, 2] = q[:, 4].conj()
        y = np.r_[pack(q), 80.31, 764.8]
        qref = q[:, 4]/common.mass_norm(p, q[:, 4])
        with patch('ro0751_phaseB_campaign.subprocess.run',
                   side_effect=subprocess.TimeoutExpired('test-only-mocked-worker', 1200)) as worker:
            try:
                campaign.attempt(mapping, y, qref, 'energy', .0064,
                                 'timeout_control', 'test_fixture')
                timeout_stopped = False
            except ScientificStop:
                timeout_stopped = campaign.state['stop'] == 'per_point_budget_exceeded'
            timeout_stopped = timeout_stopped and worker.call_count == 1 and not campaign.state['points']
        # Four failed primary corrections stop; no fifth solve is attempted.
        campaign.mapping, campaign.metric = mapping, Metric(mapping)
        old = y.copy(); old[-1] += .001
        campaign.state = dict(stage='branch_fold', points=[
            dict(id='fixture0', role='weak_seed', Rout=float(old[-1]), fields='fixture0'),
            dict(id='fixture1', role='weak_seed', Rout=float(y[-1]), fields='fixture1')],
            attempts=[], stop=None)
        with patch('ro0751_phaseB_campaign.transfer', side_effect=[old, y, old]), \
             patch.object(campaign, 'qref', return_value=qref), \
             patch.object(campaign, 'attempt', return_value=(y, dict(accepted=False))) as attempt:
            try:
                campaign.branch(until_fold=True)
                consecutive_stopped = False
            except ScientificStop:
                consecutive_stopped = campaign.state['stop'] == 'four_consecutive_Newton_or_orientation_failures'
            consecutive_stopped = consecutive_stopped and attempt.call_count == 4
        # A science stop bypasses every subsequent solve and still creates a report.
        campaign.inputs = {}
        campaign.state = dict(status='running', stage='seeds', points=[], attempts=[],
                              targets={}, folds={}, refinements=[], stop=None)
        with patch.object(campaign, 'seeds', side_effect=lambda:campaign.stop('control_stop')), \
             patch.object(campaign, 'branch') as branch, \
             patch('ro0751_phaseB_report.finalize', return_value={}) as report:
            campaign.run()
            stop_bypasses_solves = (branch.call_count == 0 and report.call_count == 1
                                    and (campaign.out/'NUMERICAL_STOPPED.json').is_file()
                                    and not (campaign.out/'NUMERICAL_DONE').exists())
    return dict(timeout_stops_without_retry=bool(timeout_stopped),
                four_failures_stop=bool(consecutive_stopped),
                scientific_stop_bypasses_later_solves=bool(stop_bypasses_solves),
                passed=bool(timeout_stopped and consecutive_stopped and stop_bypasses_solves),
                fixtures='Disposable synthetic control-flow fixtures; no production solves or figure data')


def preparation():
    if (OUT/'campaign.json').exists():
        raise RuntimeError('Numerical campaign exists; use --verify-prepared instead of overwriting preparation')
    OUT.mkdir(parents=True, exist_ok=True)
    for name in ('PREPARED', 'PREPARED.json', 'READY_FOR_SUPERVISOR'):
        (OUT/name).unlink(missing_ok=True)
    write(OUT/'preparation_checks.json', dict(passed=False, status='preparation_in_progress'))
    inputs = prerequisites()
    for source in sorted((ROOT/'examples').glob('ro0751_phaseB_*.py')):
        py_compile.compile(str(source), doraise=True)
    subprocess.run(['bash', '-n', str(OUT/'commands.sh')], check=True)
    for package in ('numpy', 'scipy', 'matplotlib', 'fitz'):
        if importlib.util.find_spec(package) is None:
            raise RuntimeError('Missing runtime package: '+package)
    qa_dir = Path('/home/lthpc/.codex/skills/nature-figure/scripts')
    for name in ('audit_panel_alignment.py', 'audit_pdf_text.py', 'audit_figure_collisions.py'):
        if not (qa_dir/name).is_file():
            raise RuntimeError('Missing figure QA tool: '+name)
    plot_env = dict(os.environ, NATURE_FIGURE_CONFIG=str(OUT/'plot_backend.json'))
    subprocess.run([sys.executable, str(qa_dir/'nature_figure_backend.py'), 'set', 'python'],
                   env=plot_env, check=True, stdout=subprocess.DEVNULL)
    h3 = checks(OUT/'checks/H3')
    extended = extra_mapping_checks()
    workflow = workflow_checks()
    result = dict(passed=bool(h3['passed'] and extended['passed'] and workflow['passed']), phaseA=inputs,
                  H3=h3, extended=extended, workflow=workflow, syntax_and_bash=True,
                  scientific_campaign_started=False, GPU_used=False,
                  CPU_threads=1, address_space_limit_GiB=200)
    write(OUT/'preparation_checks.json', result)
    if not result['passed']:
        raise RuntimeError('Numerical preflight failed')
    write(OUT/'run_policy.json', dict(ro=-.751, beta=85, Rin=400, CPU_threads=1,
          address_space_limit_GiB=200, GPU=False, per_point_hard_budget_seconds=1200,
          primary_N=32, primary_intervals=inputs['seeds']['N32_dr0.5']['intervals'],
          primary_dr='moving domain / fixed interval count, actual dr saved per point',
          arclength='fixed-metric secant hyperplane; explicit positive secant orientation',
          mesh='one verified upper point at least max(0.25,2*linear_grid_Rout_shift) above H3 fold: N40 and dr<=0.25 jointly, fixed Rout; H4 at same Rout',
          fold='H3/H4 five energy-controlled points around the arclength-bracketed minimum',
          targets=[730,750,780], repeated_failure_limit=4,
          highest_harmonic_stop=.01, auxiliary_forcing=False, mean_damped=False,
          stop_on_upper_return=True, supervisor_launch_only=True,
          terminal_markers=['WORKFLOW_DONE.json','NUMERICAL_DONE','NUMERICAL_STOPPED.json','FAILED.json']))
    write(OUT/'figure_contract.json', dict(backend='python', archetype='quantitative_grid',
          claim='Measure the finite-amplitude fold, then test upper-branch front/frequency/domain sensitivity using accepted roots',
          panels=['Rout vs A1: bracket and H3/H4 fold', 'omega vs A1: branch frequency',
                  'vrms(z=1.3): selected upper profiles', 'front vs Rout: Xie Re=471.8 comparison',
                  'omega vs Rout: domain sensitivity', 'outer R3 sponge energy vs Rout: truncation influence'],
          physical_inches=[6.4,3.8], formats=['PNG','PDF'], single_panel_files=True,
          data='all accepted roots diagnosed; profile display uses target points, mesh source and last verified upper state',
          interpolation='Barycentric in wall-normal xi; PCHIP local pinch, no extrapolation; linear front crossing with PCHIP sensitivity',
          uncertainty='H3/H4, three/five-point fold fit and joint N40/dr025 differences; no statistical CI',
          integrity='missing targets/fronts remain missing; no invented reference profiles or reference omega',
          qa='Parseval; alignment NA for single panels; exported PDF glyph and geometry audits; human PNG/PDF review pending'))
    files = {}
    for path in contract_paths()+[qa_dir/name for name in
            ('audit_panel_alignment.py','audit_pdf_text.py','audit_figure_collisions.py')]:
        key = str(path.relative_to(ROOT)) if path.is_relative_to(ROOT) else str(path)
        files[key] = sha(path)
    write(OUT/'source_contract.json', dict(files=files, ro=-.751, beta=85.,
                                          neutral_done_override='explicit user authorization'))
    verify_contract(OUT)
    timestamp = datetime.datetime.now(datetime.timezone.utc).isoformat()
    summary = dict(status='prepared_pending_supervisor', prepared_at=timestamp,
                   preparation_checks_passed=True, scientific_campaign_started=False,
                   Rout_f=None, upper_front=None, omega=None,
                   linear=inputs['seeds'], launch='bash commands.sh',
                   next_condition='Supervisor launches detached job; resume after a terminal marker')
    write(OUT/'PREPARED.json', summary)
    write(OUT/'summary.json', summary)
    (OUT/'PREPARED').write_text(timestamp+'; preparation only, no Phase B numerical run\n')
    (OUT/'READY_FOR_SUPERVISOR').write_text('Preparation checks passed. Launch commands.sh in a detached supervisor shell.\n')
    (OUT/'PENDING_SUPERVISOR').write_text('No continuation or finite-amplitude production solve has been launched.\n')
    (OUT/'REPORT.md').write_text('''# Ro=−0.751、β=85 全局非线性：阶段 B

状态：准备完成，待主管后台启动；尚无阶段 B 数值结论。
Rout_f、上支前沿收敛值、ω 及与 Xie Re=471.8 的偏差均待计算。

前置条件已核验：phaseA/beta85/status/neutral.done 存在。
N32 的 Rout_c=764.7635065091705、ω_c=80.31112435889007；
N40 的 Rout_c=765.6305810363921、ω_c=80.33665285837452。
中性场、参数元数据与 r_end、局部夹点曲线均已检查；阶段 A completion.json 的总状态无需替代用户授权的 neutral.done 条件。

独立的新求解器已覆盖 Ro、β、基流和移动域的非线性导数。
小网格原始方程、Rout 导数、能量约束/伪弧长 Jacobian、H4 和 Ro=−1 回归检查见 preparation_checks.json。
本轮未运行 B1–B4 正式计算，未生成或宣称最终 PNG/PDF。

流程：两个弱非线性点 → 伪弧长括住折点 → H3/H4 折点复核 →
一个上支点的 N40/ΔR≤0.25 联合复核及 H4 检查 → 继续上支与 Rout=730、750、780 目标。
正向割线和 Rout 转向用于识别上支；不把折点外推、回返或缺失目标当成收敛证据。
最高谐波/基频能量超过 1e−2、连续 Newton/方向检查失败或单点硬预算用尽时停止。

Re=0.751R。与任务书 Xie 471.8 对照的是物理前沿 Re_front，不是截断出口 Rout。
前沿与 Ro=−1 同定义：z=1.3 的 rms 对数增长率与局部夹点 −Im α0 首个向下交点，搜索区间 R∈[550,min(650,a)]。
无交点则记录缺失；两种海绵能量权重、谐波占比和 Parseval 检查均由报告脚本保存。

commands.sh、run_policy.json、source_contract.json、figure_contract.json 和 HANDOFF.md 给出启动、资源与复现约定。
执行过程中保持单线程、串行，无 GPU，地址空间上限 200 GiB；默认每点硬预算 1200 秒。
NUMERICAL_DONE 表示全目标和复核完成；NUMERICAL_STOPPED.json 表示科学停止；FAILED.json 表示执行失败；WORKFLOW_DONE.json 表示报告生成完毕。
PREPARED/READY_FOR_SUPERVISOR 仅是准备标记。正式数值运行结束后，本文件与 summary.json 自动更新，图像人工审阅另行记录。
''')
    (OUT/'HANDOFF.md').write_text('''# 阶段 B 主管交接

准备完成；正式长作业未启动。禁止派发 subagent，不要轮询。

主管在本机后台启动一次：
```bash
cd /home/lthpc/Desktop/data_bak/nonlinear_rotating_disk/llsa_python
setsid bash results/ro0751_global_20261101/phaseB/commands.sh > results/ro0751_global_20261101/phaseB/run.log 2>&1 < /dev/null &
```

commands.sh 用文件锁阻止重复启动，先复核源码/输入哈希，再恢复同一 campaign.json。
CPU 单线程串行，地址空间 200 GiB，不用 GPU。预算 1200 秒/点包含整个隔离进程。
全过程可能超过一小时，由主管后台等待完成后再续接本专用会话。
续接条件：WORKFLOW_DONE.json 或 FAILED.json；同时核对 NUMERICAL_DONE / NUMERICAL_STOPPED.json 的科学状态。
源文件改动会阻止启动。尚未开算时，主管可审查并重新执行 checks 脚本 --prepare；已有 campaign.json 时应先审查变更并另建隔离的续接任务，禁止覆盖原准备基线。
不要手改 source_contract.json 以绕过不匹配。

建议续接提示（无需重读任务书）：
“阶段 B 已出现终止标记。读取 phaseB/summary.json 与 REPORT.md；审阅每张 PNG/PDF 和自动 QA，核对上支方向、折点括区、停止原因及 H3/H4/细网格差。不要擅自重启或扩展预算。”

未完成目标保留缺失；无大域前沿/频率时不外推收敛值。原 Ro=−1 代码和结果、阶段 A 均保持只读。
''')
    print(json.dumps(dict(passed=True, prepared=True, scientific_campaign_started=False,
                          H3=h3['passed'], H4=extended['passed'], source_files=len(files))))


def main():
    parser = argparse.ArgumentParser()
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument('--prepare', action='store_true')
    group.add_argument('--verify-prepared', action='store_true')
    args = parser.parse_args()
    resource.setrlimit(resource.RLIMIT_AS, (200*1024**3, 200*1024**3))
    if args.prepare:
        preparation()
    else:
        prerequisites()
        verify_contract(OUT)
        if not read(OUT/'preparation_checks.json')['passed']:
            raise RuntimeError('Preparation checks failed')
        print(json.dumps(dict(prepared_inputs_verified=True, campaign_not_started_by_check=True)))


if __name__ == '__main__':
    main()
