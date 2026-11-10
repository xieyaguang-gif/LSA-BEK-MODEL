#!/usr/bin/env python3
"""Audit saved S0b evidence and publish the finite-search absolute table; no eigensolves."""
import os
for k in ('OPENBLAS_NUM_THREADS','OMP_NUM_THREADS','MKL_NUM_THREADS'):os.environ[k]='1'
os.environ['PYTHONDONTWRITEBYTECODE']='1'
import hashlib,json,time
from pathlib import Path
import roscan_s0b_v2 as m
ROOT=m.ROOT;OUT=m.OUT;S0=m.old.OUT;RUN=m.RUN

def load(p):return json.loads(Path(p).read_text())
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def rel(p):return str(Path(p).relative_to(ROOT))
def qualified(b):
    r=b['minimum'];p=b['pinch'];c=b['convergence'];path=p.get('path',[])
    return bool(b.get('integer_minimum_bracketed') and r['beta']==int(r['beta']) and max(abs(x) for x in r['neutral_equation_residual'])<1e-8 and r['eigen_residual']<1e-8 and b.get('dense_validation',{}).get('passed') and p['status']=='finite_path_passed' and path and path[-1]['roots'][0][1]*path[-1]['roots'][1][1]<0 and all(x['max_residual']<1e-8 and x['separation']>1e-7 for x in path) and len(c)==2 and all(x['passed'] and x['relative_Re_change']<.001 for x in c) and any(x['n']==192 and x['zmax']==40 for x in c) and any(x['n']==128 and x['zmax']==50 for x in c))

def main():
    before=OUT/'absolute_table_before_s0b.json'
    if not before.exists():before.write_bytes((S0/'absolute_table.json').read_bytes())
    baseline=load(before);table=json.loads(json.dumps(baseline));diagnosis=load(OUT/'diagnosis.json');fold=load(RUN/m.old.tag(-.3)/'neutral_Re_fold.json');ekman=load(OUT/'ekman_limit_refinement.json');boundary=load(OUT/'ekman_boundary_convention_check.json')
    assert diagnosis['dense_validation']['passed'] and diagnosis['scaled_condition_number']<100
    assert fold['status']=='resolved' and fold['connects_to_primary_minimum']
    sources={};evidence={};rows={r['ro']:r for r in table['rows']};checks=[];branch_lines=[];main_rows=[]
    for ro in [-.5,-.4,-.3,-.2,-.1]:
        dest=RUN/m.old.tag(ro);branch=load(dest/'branch_continued_physical_branch.json');search=load(dest/'branch_search.json');r=branch['minimum'];assert qualified(branch),(ro,'minimum acceptance');assert search['search_complete'] and len(search['coverage'])==20,(ro,'independent probes')
        assert r['kernel_code']==m.old.CODE,(ro,'kernel changed')
        families=[];candidate_rows=list(branch['candidates']);unverified=[]
        for f in sorted(dest.glob('branch_*.json')):
            if f.name=='branch_search.json':continue
            b=load(f)
            if not b.get('minimum'):continue
            scope='lowest verified pinching integer minimum' if b['label']=='continued_physical_branch' else 'additional neutral family'
            if ro==-.3 and b['label']=='legacy_integer_walk':scope='upper neutral sheet; connected through resolved Re fold to primary minimum'
            families.append(dict(label=b['label'],minimum=b['minimum'],pinch=b.get('pinch',{}),convergence=b.get('convergence',[]),integer_minimum_bracketed=b.get('integer_minimum_bracketed',False),status=b['status'],interpretation=scope,source=rel(f)))
            candidate_rows.extend(b.get('candidates',[]))
            if b['label']!='continued_physical_branch':candidate_rows.append(b['minimum'])
            if b.get('pinch',{}).get('status')!='finite_path_passed':unverified.append(dict(label=b['label'],Re=b['minimum']['Re_ca'],beta=b['minimum']['beta'],pinch=b.get('pinch',{}).get('status'),reason='Finite path ends in same alpha half-plane; not accepted as an absolute critical point.'))
            branch_lines.append(f"| {ro:.1f} | {b['label']} | {b['minimum']['Re_ca']:.6f} | {b['minimum']['beta']:g} | {b.get('pinch',{}).get('status')} | {scope} |")
            evidence[rel(f)]=sha(f)
        resolution=[]
        for u in search.get('unresolved',[]):
            resolution.append(dict(original=u,resolved_by=rel(dest/('neutral_Re_fold.json' if ro==-.3 else 'branch_legacy_integer_walk.json')),conclusion='same neutral family upper sheet' if ro==-.3 else 'integer-beta boundary minimum; pinch not confirmed'))
        rr=dict(ro=ro,status='numerically_validated',Re_ca=r['Re_ca'],R_ca=r['R_ca'],beta_c=int(r['beta']),omega_BEK=r['omega_BEK'],omega_Omega_d=r['omega_Omega_d'],omega_raw=r['omega_raw'],alpha=r['alpha'],alpha_global=r['alpha_global'],pinch=branch['pinch'],convergence=branch['convergence'],dense_validation=branch['dense_validation'],beta_coverage=sorted(x['beta'] for x in branch['candidates']),integer_minimum_bracketed=True,found_branches=candidate_rows,branch_minima=families,source=rel(dest/'branch_continued_physical_branch.json'),search_scope='Finite independent n64 probes at beta={0,0.5bc,bc,1.5bc,2bc}, four complex-alpha/frequency seeds; every legacy seed family audited at nominal n128. Continuous-beta stationary solve followed by independent neighboring integer recomputation.',coarse_probe_coverage=search['coverage'],search_complete=True,producer_codes=sorted(set(x['code'] for x in candidate_rows)),validator_code=sha(__file__),criticality_scope='Minimum among found branches qualified by neutral residuals, integer neighbors, dense QZ, finite-path Briggs check and two convergence refinements.',global_minimum_proven=False,secondary_pinch_unconfirmed=unverified,legacy_audit_resolutions=resolution)
        rows[ro].clear();rows[ro].update(rr);main_rows.append(rr);checks.append(dict(ro=ro,primary_minimum_accepted=True,independent_probes=len(search['coverage']),max_grid_relative_change=max(c['relative_Re_change'] for c in branch['convergence'])))
        evidence[rel(dest/'branch_search.json')]=sha(dest/'branch_search.json')
    table.update(schema='roscan-absolute-v2-s0b',completion='finite_search_complete',updated_by='TASK_S0b.md',operator_kernel_code=m.old.CODE,validator_code=sha(__file__),global_minimum_proven=False,criticality_scope='Each new row is the lowest verified pinching integer minimum in the documented finite search. Secondary neutral saddles without a passed finite-path Briggs check are retained and are not counted as absolute critical points.',prior_table_snapshot=rel(before),diagnostic_source=rel(OUT/'diagnosis.json'))
    # Previously validated rows are retained byte-for-byte at the JSON-object level.
    for ro in [-1.,-.9,-.8,-.7,-.6,-.751]:assert rows[ro]==next(r for r in baseline['rows'] if r['ro']==ro)
    for p in sorted((ROOT/'examples').glob('roscan_s0b*.py')):sources[rel(p)]=sha(p)
    for p in [OUT/'diagnosis_producer.py',OUT/'continuation_producer.py',OUT/'numerical_producer.py',OUT/'numerical_producer_v2.py']:sources[rel(p)]=sha(p)
    assert sha(OUT/'diagnosis_producer.py')==diagnosis['source_sha256']
    assert sha(OUT/'numerical_producer.py')==sha(ROOT/'examples/roscan_s0b.py')
    assert sha(OUT/'numerical_producer_v2.py')==sha(ROOT/'examples/roscan_s0b_v2.py')
    old=m.old
    old.dump(S0/'absolute_table.json',table)
    lines=['# S0b：绝对中性点卡顿诊断与有限分支扫描（2026-11-02）','',
      '原 Ro≈−0.50454 卡顿由 ARPACK 等待无关特征对造成。改用较少近目标特征对、残差合格的部分收敛结果及稠密回退后，同一物理模态已连续推进到 Ro=−0.1。以下为已找到、且通过中性残差、整数邻点、稠密 QZ、有限路径 Briggs 与两项网格检查的分支中最低值。搜索范围和未确认 pinch 的中性分支在下文列明。','',
      '| Ro | Re_ca | R_ca | β_c | α_r | α_i | ω_raw | ω_BEK | ω_Ωd | max ΔRe/Re |','|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|']
    for r in main_rows:
        lines.append(f"| {r['ro']:.1f} | {r['Re_ca']:.6f} | {r['R_ca']:.6f} | {r['beta_c']} | {r['alpha'][0]:.8f} | {r['alpha'][1]:.8f} | {r['omega_raw'][0]:.8f} | {r['omega_BEK'][0]:.8f} | {r['omega_Omega_d'][0]:.8f} | {max(c['relative_Re_change'] for c in r['convergence']):.3g} |")
    lines+=['', '## 失败点证据与轨迹','',
      f"在原失败参数 Ro={diagnosis['robust']['ro']:.12f}、β={diagnosis['robust']['beta']:.12f}，得到 Re={diagnosis['robust']['Re_ca']:.12f}。独立 scipy.linalg.eig 稠密特征值差 {diagnosis['max_eigenvalue_difference']:.3g}，稠密群速度和中性残差均合格；相邻有限时域特征值间隙 {diagnosis['dense_validation']['temporal_gap']:.6g}。尺度化中性 Jacobian 条件数 {diagnosis['scaled_condition_number']:.6g}，该点无局部折点或特征值合并证据。",
      '原续算人为固定 β|Ro|=58.2，因此 dβ/dRo=β/|Ro|，不是从 Re_ca(β) 优化中测得的导数。固定此路径到 Ro=−0.1 得到的非整数种子 Re=234.223316 仍高于整数优化后的 218.160799，不能把种子直接写入临界总表。',
      '最后20个成功步取自 joint_rescue 与 joint_resume 两段，按 Ro 去重后连接；没有平滑或插值。图中 α、ω 为原始局部约定。轨迹图：[PNG](diag_0p5/trajectory_last20.png)、[PDF](diag_0p5/trajectory_last20.pdf)；数据和图契约在同目录。',
      '新脚本以速度场的梯形积分内积跟踪模态，排除压力幅值对相关性的影响。近目标求解优先 k=2、ncv=32；部分收敛结果必须通过原广义特征值残差检查。每个正式最小值另用稠密 QZ 复核，并独立重算相邻整数 β。',
      '', '## 分支、折点与 pinch','',
      '| Ro | 分支 | 已计算最低 Re | β | 有限路径 pinch | 判读 |','|---:|---|---:|---:|---|---|',*branch_lines,'',
      f"Ro=−0.3 存在中性曲线的 Re 折点：β={fold['fold']['beta']:.9f}、Re={fold['fold']['Re']:.9f}。同一整数 β=111 有 Re={fold['connections'][0]['Re_ca']:.6f} 与 {fold['connections'][1]['Re_ca']:.6f} 两个中性解。由折点低 Re 侧经 β=115、121、130、140、150、157 连续跟踪，最终与主分支最小值的 Re 和 α 一致。旧 Re=527.322683 位于这条曲线的高 Re 侧，不能作为最低绝对临界值。折点方程、中间步骤和模态相关性保存在 s0b/ro_m0p300/neutral_Re_fold*.json。",
      'Ro=−0.2 的旧负 α_i 家族经整数 β 续算到 β=0，得到中性 Re≈139.159；该点及较高 Re≈563.692 的另一家族均未通过现有有限路径 pinch 检查。Ro=−0.1 的 Re≈498.243 家族也在路径末端保持同侧空间根。这些是中性鞍点证据，未列为已验证的绝对临界值。',
      'pinch 路径为原局部 ω 的上半平面 75 个对数间隔点，η=10⁻⁷…0.12；两根必须连续分离、谱残差合格，且末端位于不同 α 半平面。这里只作有限路径 Briggs 检查，不将同侧末端视为全拓扑排除。分支按可核查的模态连续性命名，未凭 α 或频率符号擅自赋予 type I/II 标签。',
      '每个 Ro 另有20组独立 n=64 复初值探测，β 取 0、约0.5β_c、β_c、1.5β_c、2β_c，覆盖四组复 α/频率初值；成功候选在 n=128 重新求解并按物理模态归类。完整失败与重复证据保存在各 Ro 的 branch_search.json。有限搜索不等价于遍历全部鞍点。',
      '', '## Lingwood (1997) 与小 Ro 极限','',
      'Lingwood 的原文在 p.425 给出 Ekman 绝对失稳阈值约198；这次已核查原文索引内容。文献：[Absolute instability of the Ekman layer and related rotating flows](https://electronicsandbooks.com/edt/manual/Magazine/J/Journal%20of%20Fluid%20Mechanics/1997%20Volume%20331/S0022112096004144.pdf)，[DOI](https://doi.org/10.1017/S0022112096004144)。',
      '辅助连续 β 最小值为 Ro=−0.05：208.390521；−0.01：200.711568；−0.001：198.997711；−0.00001：198.809475。趋势与文献约198一致。辅助点未加入正式10点扫描总表。',
      f"在 Ro=−0.001 固定 β 的 n/zmax 加密差最大 {max(abs(c['relative_difference']) for c in ekman['checks_at_ro_minus0p001']):.3g}，因此剩余偏差不是该网格未收敛。Ro=−0.00001 时，生产 matlab 端点约定给 Re={boundary['matlab_boundary']['Re_ca']:.6f}，已有 paper 端点约定给 {boundary['paper_boundary']['Re_ca']:.6f}，差 {boundary['difference']:.6f}。压力与法向速度导数的端点位置足以解释大部分小偏差；精确复现文献余差还需逐项对照其方程与远场条件。生产结果继续采用通过原回归的 matlab 约定，没有为凑198改动方程或边界。",'',
      '## 复现、审计与交付','',
      '新入口：`bash results/ro_scan_20261101/S0/diag_0p5/commands.sh <stage> [Ro] [budget]`。stage 可用 diagnose、continue、survey、probe、ekman、fold、finalize；锁与原 S0 local_pipeline.lock 共用。各阶段日志、源码 SHA256 和证据清单位于 diag_0p5。',
      '原 examples/roscan_absolute.py、LLSA 算子及旧 survey 未修改；原 Ro=−1…−0.6 和 −0.751 六行按 JSON 对象原样保留。旧表完整快照为 absolute_table_before_s0b.json。S1/S2 未运行。',
      '计算均为单线程，数值进程最大并发5；独立探测进程各限8GB，主验证进程各限60GB。没有使用 GPU 或向其他项目进程发送信号。没有超过1小时的待执行计算。',
      '执行审计说明：本会话命令工具有效 yield 上限30秒，完成回收等待限定在50秒内，无法满足任务书的1800000毫秒工具参数要求；事件守卫记录了短 yield 告警。未使用周期性 tail/ps/nvidia-smi 状态轮询，未更改用户配置。图的布局、PDF字形和碰撞检查通过；通用 Nature 源码检查对用户指定的衬线字体与仅PDF/PNG导出产生格式规则告警，详见 figure_qa_resolution.json。']
    (S0/'REPORT.md').write_text('\n'.join(lines)+'\n')
    old.dump(OUT/'source_manifest.json',dict(source_sha256=sources,evidence_sha256=evidence,canonical_source_sha256=sha(ROOT/'examples/roscan_absolute.py'),operator_kernel_code=m.old.CODE))
    old.dump(OUT/'verification.json',dict(passed=True,rows=checks,regression_rows_unchanged=True,original_failure_dense_qualified=True,neutral_Re_fold_connected=True,independent_probes=100,finite_search_complete=True,global_topology_exhaustive=False,figure_inspected=True,figure_alignment=load(OUT/'trajectory.alignment.json')['verdict'],figure_collision=load(OUT/'trajectory.collision_audit.json').get('verdict'),source_manifest='source_manifest.json'))
    (ROOT/'results/ro_scan_20261101/PENDING_SUPERVISOR').write_text('S0b: no long calculation pending. Five Ro=-0.5..-0.1 primary integer minima are validated in S0/absolute_table.json.\nThe documented finite branch search and secondary saddle classifications are complete; global Briggs topology is not claimed.\nReproduce or extend with S0/diag_0p5/commands.sh. See S0/REPORT.md and diag_0p5/verification.json.\nS1 prior outcomes are unchanged. S2 remains unstarted and requires separate supervisor authorization.\n')
    print(json.dumps(dict(passed=True,finite_search_complete=True,rows=[dict(ro=r['ro'],Re=r['Re_ca'],beta=r['beta_c']) for r in main_rows],unchanged_prior_rows=6)))

if __name__=='__main__':main()
