"""Generate the Chinese report from completed manifests, without launching work."""
from fullbvp_phaseE_adapter import ROOT,OUT,SOURCE,SOURCE_SHA,write_json
import json


def read(path):
    return json.loads(path.read_text())


def main():
    validation=read(OUT/'validation.json')
    stages={}
    for stage in ('pilot','production','refine','saddles','absolute_refine','convergence','short_audit','short_absolute'):
        path=OUT/f'{stage}_manifest.json'
        if path.exists():stages[stage]=read(path)
    temporal=[];absolute=[]
    for stage,manifest in stages.items():
        for key in manifest['jobs']:
            data=read(OUT/'jobs'/key/'result.json');data['stage']=stage
            (absolute if data['config']['kind']=='absolute' else temporal).append(data)
    chosen=[x for x in temporal if x['stage']=='production'] or [x for x in temporal if x['stage']=='pilot']
    if not any(x['stage']=='production' for x in chosen):
        chosen += [x for x in temporal if x['stage']=='short_audit' and x['config']['N']==32
            and x['config']['H']==6 and x['config']['K']==3 and x['config']['zmax']==30.
            and not x['config'].get('dense',False)]
    chosen += [x for x in temporal if x['stage']=='refine']
    # Strongest sampled growth for each identical numerical setup and radius.
    groups={}
    for row in chosen:
        c=row['config'];g=tuple(c[k] for k in ('N','H','K','zmax','beta_hat','amplitude'))
        if c['R'] not in groups.setdefault(g,{}) or row['modes'][0]['growth']>groups[g][c['R']]['modes'][0]['growth']:
            groups[g][c['R']]=row
    sector=[x for x in chosen if x['config']['beta_hat']==0 and x['config']['amplitude']==1.]
    at560=[x for x in sector if x['config']['R']==560]
    positive560=bool(at560 and max(x['modes'][0]['growth'] for x in at560)>0)
    production_done='production' in stages and not stages['production']['errors']
    abs_positive=[x for x in absolute if x.get('positive_finite_path_pinch')]
    abs560=[x for x in abs_positive if x['config']['R']==560. and x['config']['amplitude']==1.]
    laminar560=[x for x in abs_positive if x['config']['R']==560. and x['config']['amplitude']==0.]
    lines=['# Phase E：725/H6 全局场的冻结半径 Floquet 分析','',
        '**结论。** '+('当前 β̂=0 的搜索在 R=560 已找到正时间增长率；560–600 内不能给出首次时间失稳的内部零点。'
                       if positive560 else '时间谱扫描和临界点判定尚未完成。')+
        ' 正 Floquet 增长率还可能来自层流的一次失稳，不能单独解释为饱和主波诱发的二次失稳。', '',
        ('R=560 已有正增长的有限路径夹点，且去掉主波后的层流对照仍有该夹点；总谱的绝对不稳定也已在窗口左端存在。'
         '本窗口不能给出由主波诱发的新二次绝对失稳起点。' if abs560 and laminar560 else
         '已取得正增长的有限路径夹点候选，但 R_sa 仍须结合分支追踪及收敛结果判定。' if abs_positive else
         '**尚未确定二次绝对失稳起点 R_sa。** 当前没有可据以确认该起点的已收敛夹点证据。')+
        ' 因此尚不能把 587.6 与 583 的差异归因于二次绝对失稳。', '',
        '## 执行范围与续接','',
        ('生产时间谱已运行；以下数值仍受搜索范围及收敛结果限制。' if production_done else
         '本轮完成适配器、独立校验和短程数值诊断。完整多参数扫描及鞍点/夹点收敛预计超过任务书的 20 分钟界限，交由主管运行 `commands.sh`。'),'',
        '| 阶段 | 已完成作业 | 失败作业 | 耗时/s |','|---|---:|---:|---:|']
    for stage,m in stages.items():lines.append(f"| {stage} | {len(m['jobs'])} | {len(m.get('errors',[]))} | {m.get('seconds',0):.1f} |")
    if (OUT/'production_benchmark.json').exists():
        bench=read(OUT/'production_benchmark.json')
        hours=bench['estimated_production_worker_seconds']/3600
        lines += ['',f"N32/H6/K6 实测三个 shift 用时 {bench['seconds']:.2f} s。"
            f"按 1170 个生产时间谱点、每点 34 个 shift、8 workers 粗估，仅生产时间谱约 {hours:.1f} 小时，"
            '不含加密、鞍点与收敛；这是单点外推，实际耗时受算例及共享负载影响。完整任务因此未在本轮启动。']
    lines += ['', '启动命令（8 个单线程 worker，无 GPU）：','', '```bash',
        f"setsid bash {OUT}/commands.sh > {OUT}/campaign.log 2>&1 < /dev/null &",'```','',
        '每个作业有输入/代码哈希及 DONE/FAILED；重复启动复用相同输入的完成作业。总入口写 phaseE/DONE 或 FAILED。'
        'DONE 表示计算流程结束，不等于科学上已通过收敛或夹点判定。完成后读取本报告与各阶段 manifest；无需工具轮询。','',
        '## E1：基态、标度与局部校验','',
        f'源文件：`{SOURCE.relative_to(ROOT)}`；SHA-256：`{SOURCE_SHA}`。', '',
        '源数据为 Ro=−1、β=68、Rout=725、N_base=32、H_base=6 的严格接受场。'
        '速度系数已经除以当地 R，压力系数除以 R²，且只含相对层流的修正；无需再减层流或变换旋转坐标系。'
        '均值修正 q0 与正负谐波原样保留，频率为惯性系 ω=52.6814327024，采用 exp[ih(βθ−ωt)]。', '',
        '沿 R 用相邻八个节点构造局部三次样条。α(R)=Im[(∂R v1)/v1]、g(R)=Re[(∂R v1)/v1]，'
        '探针高度 z=1.3。法向经原有半无限 Chebyshev 映射逆变换后作重心插值。'
        'Floquet 基态使用实 α，并保留剖面当地复相位；g 和剖面导数只诊断、不加入冻结算子。', '',
        '| R | α | g=d ln(abs(v1))/dR | abs(v1(z=1.3)) | 被省略径向导数的相对范数 |',
        '|---:|---:|---:|---:|---:|']
    for b in read(OUT/'adapter.json'):
        lines.append(f"| {b['R']:g} | {b['alpha']:.6f} | {b['envelope_growth']:.6f} | {b['probe_abs_q1']:.4g} | {b['omitted_radial_derivative_relative']:.3f} |")
    local=validation['local_PrimaryWave'];sp=validation['spatial_comparison']['pier']
    lines += ['',f"独立校验全部通过：合成载波 α 误差 {validation['checks']['manufactured_alpha_error']:.2g}；"
        f"既有 PrimaryWave 与 DNSFrozenBase 的算子差 {validation['checks']['existing_PrimaryWave_matrix_max_error']:.2g}；"
        f"复 α 二次矩阵多项式差 {validation['checks']['matrix_polynomial_max_error']:.2g}。",'',
        f"R=560、同 ω 的局部空间模态 α={sp['alpha'][0]:.6f}{sp['alpha'][1]:+.6f}i，"
        f"与全局 q1 速度剖面的相关性 {sp['velocity_overlap']:.6f}，消除整体相位和幅值后的相对误差 {sp['phase_and_amplitude_aligned_error']:.4f}。"]
    if local['status']=='converged':
        lines += ['',f"同 R、同 E1 的 H1 中性 PrimaryWave 收敛至 α={local['alpha']:.6f}、ω={local['omega']:.6f}，"
            f"非线性残差 {local['residual']:.2g}；与全局 q1 的相关性 {local['velocity_overlap']:.4f}。"
            '它固定实 α 并求中性 ω，与固定 ω、沿 R 增长的全局场不是同一解。'
            '这一比较说明差异并非简单的符号或速度标度错误，不能将全局截面标记成已验证的局部平衡态。']
    else:lines += ['',f"同半径 H1 PrimaryWave 校验未收敛：{local.get('error','unknown')}。"]
    lines += ['', '## E2：时间谱','',
        'β̂=0 对应 2016 年 2π/68 扇区；β̂=17、68/3、34 的对照在该扇区不允许。'
        '其中 β̂=17、34 可满足整圆周整数方位模态条件；β̂=68/3 仅作形式上的连续 Bloch 参数对照，也不满足整圆周周期性。'
        'Im(ω̂)>0 为时间不稳定，单位为 Ω_BEK；Ro=−1 时与盘转速单位相同。', '',
        '以下为多个频率 shift 搜索所得最大值，不宣称穷尽连续算子的全谱。'
        '每个原始 result.json 保留频率、矩阵残差、扰动谐波能量及搜索失败。'
        '恰好 α̂=β̂=0 时现有连续性壁面闭合有压力零空间，扫描用 α̂=±10⁻⁴ 接近该点，未用奇异中心的伪谱判定失稳。','']
    for group,rows in sorted(groups.items()):
        N,H,K,z,bh,amp=group
        lines += [f'### N={N}, H={H}, K={K}, z_top={z:g}, β̂={bh:g}, 基态修正倍率={amp:g}','',
            '| R | 最大搜索增长率 | α̂ | Re(ω̂) | 两端扰动谐波能量占比 |', '|---:|---:|---:|---:|---:|']
        for R,row in sorted(rows.items()):
            m=row['modes'][0]
            lines.append(f"| {R:g} | {m['growth']:.6f} | {row['config']['alpha_hat']:.6f} | {m['omega'][0]:.6f} | {m['edge_energy']:.4g} |")
        if min(rows)==560 and rows[560]['modes'][0]['growth']>0:
            lines += ['', '**时间失稳区间判定：** 最左样点已为正，只能给出该离散冻结模型的 R_s≤560；无法由本窗口确定更上游的首次零点。']
    lines += ['', '基态修正倍率=0 的行是相同层流与同一 Floquet 载波的对照。'
        '若去掉全部主波后仍有正增长，则总谱的最强模态不提供独立的“二次诱发”起点。'
        '修正倍率 0.25/0.5/0.75 的对照只作分支识别，不能当作新的非线性平衡解。', '',
        '## E3：复鞍点和夹点','',
        '生产流程从每个 R 的最强时间点取前三个谱支，分别从 Im(α̂)=0、−0.04 搜索。'
        '要求 |∂ω̂/∂α̂|<10⁻⁶，再将 Im(ω̂) 向上延拓，检查两个不同空间根是否进入相反半平面。'
        '这属于有限路径数值夹点诊断；仅有正鞍点增长率不认定绝对失稳。','',
        '| R | N/H/K/z_top | 基态修正倍率 | α̂_s | ω̂_s | 群速度模 | 夹点路径 |', '|---:|---|---:|---|---|---:|---|']
    for x in absolute:
        c=x['config'];pinch=x.get('pinch')
        status=('相反半平面' if pinch['opposite_half_planes_at_end'] else '同半平面') if pinch else ('未收敛' if not x['converged'] else x.get('pinch_error','未检查'))
        lines.append(f"| {c['R']:g} | {c['N']}/{c['H']}/{c['K']}/{c['zmax']:g} | {c['amplitude']:g} | {complex(*x['alpha_hat']):.5g} | {complex(*x['omega_hat']):.5g} | {abs(complex(*x['group_velocity'])):.3g} | {status} |")
    if not absolute:lines += ['| — | — | — | — | — | — | 待主管计算 |']
    if abs560 and laminar560:
        full=abs560[0];lam=laminar560[0]
        lines += ['',f"R=560 的总谱夹点增长率为 {full['omega_hat'][1]:.6f}，层流对照为 {lam['omega_hat'][1]:.6f}。"
            '按这一有限路径诊断，总谱绝对失稳阈值的当前上界也是 560，而不是 583 附近的内部零点。'
            '完整新二次分支的归属和起点仍需额外搜索。']
    lines += ['', 'R_sa 必须由通过夹点检查的同一分支在 R 方向变号并经 N/H/K/顶部高度检验后确定。'
        '缺失的半径或失败的空间延拓不能插值成临界点。','',
        '## E4：收敛与覆盖','',
        '生产基准 N=32、H=6、K=6、z_top=30；独立变化为 N=48、H=4、K=8、z_top=40。'
        '法向基态始终来自同一个 N_base=32 的 H6 全局场，增加 N 只提高扰动求解分辨率。'
        'H=4 是截断同一基态的敏感性，不代表重新求解了 H4 全局平衡。'
        '有限高度尾部幅度、冻结连续性缺陷及省略包络导数的范数保存在 adapter.json 和各 result.json 中。', '',
        '完整谱交叉检查和各分辨率都必须对同一模态比较；边缘谐波能量较大或最大值落在 α̂ 搜索端点时需继续扩大范围。']
    audits=[x for x in temporal if x['stage']=='convergence' or (x['stage']=='short_audit'
        and x['config']['R']==585 and x['config']['alpha_hat']==.2 and x['config']['beta_hat']==0
        and x['config']['amplitude']==1)]
    if audits:
        lines += ['', '| R | N/H/K/z_top | β̂ | α̂ | 搜索最大 Im(ω̂) | 搜索方式 |','|---:|---|---:|---:|---:|---|']
        for x in audits:
            c=x['config'];m=x['modes'][0]
            method='密集广义谱/有限频窗' if c.get('dense',False) else '多 shift'
            lines.append(f"| {c['R']:g} | {c['N']}/{c['H']}/{c['K']}/{c['zmax']:g} | {c['beta_hat']:.5g} | {c['alpha_hat']:.5g} | {m['growth']:.6f} | {method} |")
        lines += ['', '该表是单点数值敏感性，不能替代各配置重新扫描后的临界点收敛。']
    else:lines += ['', '本轮尚无完整 N/H/K/顶部高度收敛结果；已写入主管脚本。']
    lines += ['', '密集广义谱采用 |Re(ω̂)|≤250、|Im(ω̂)|≤100 的显式交叉检查窗口，并排除速度质量范数低于 10⁻⁸ 的压力代数子空间。'
        '原始广义特征值中的近无穷压力根会给出极大伪增长率，即使矩阵后向残差很小也不能当作流体模态。'
        '被排除数量和窗口保存在 dense_diagnostics 中；此检查不宣称全频谱穷尽。', '',
        '试算还发现：靠近实轴的少量频率 shift 会受稳定连续谱拥挤影响漏掉更强模态。'
        '生产配置已增加 Im(shift)=12 的搜索层；N=32/K=3 的 R585、α̂=0.2 点用密集谱独立核对。','',
        '## 与 2016 年 583 的比较','',
        '项目既有 D4 文献审计将约 582.8/583 识别为有限时间趋势的渐近外推前沿，'
        '不是本文已经测得的局部 Floquet 临界半径。当前 H6 相干波的 Fig.8 前沿约 587.6。'
        '即使后续得到 R_sa≈583，仍需区分局部冻结模型、全局反馈及前沿定义，不能把数值接近直接当作因果证明。', '',
        '来源：[总报告](../final/REPORT.md)、[D4 原文与图形审计](../phaseD/D4/REPORT.md)、'
        '[DNS 冻结接口](../../../docs/DNS_FLOQUET.md)。', '',
        '## 可复现文件','',
        '- `adapter.json`：九个半径的基态与冻结误差诊断。',
        '- `validation.json`、`local_primary_H1_R560.npz`：独立校验及局部中性主波。',
        '- `jobs/<hash>/request.json,result.json,vectors.npz,DONE/FAILED`：逐点参数、谱、向量与状态。',
        '- `*_manifest.json`、`*_summary.json`：各阶段索引和扫描汇总。',
        '- `commands.sh`：绝对路径、线程上限、断点续算、完成/失败标志。','']
    if (OUT/'execution_audit.json').exists():
        lines += ['工具执行记录见 `execution_audit.json` 与 `tool_audit.json`。'
            '当前工具的 exec_command 等待参数上限为 30000 ms，无法接受任务书的 1800000 ms 下限；'
            '因此该项参数约束未满足，短程计算采用有限时长调用，完整长作业留给主管后台启动。','']
        if (OUT/'tool_audit.json').exists():
            audit=read(OUT/'tool_audit.json')
            lines += [f"守卫快照记录 {len(audit.get('short_or_unspecified_wait_calls',[]))} 次短等待参数调用、"
                      f"{len(audit.get('alerts',[]))} 条阈值报警；详情保留在原始审计 JSON。",'']
    (OUT/'REPORT.md').write_text('\n'.join(lines))
    write_json(OUT/'status.json',dict(stages=list(stages),temporal_Rs_upper_bound=560 if positive560 else None,
        secondary_induced_Rs=None,confirmed_Rsa=None,
        total_spectrum_finite_path_Rsa_upper_bound=560 if abs560 else None,
        production_temporal_executed=production_done,
        positive_finite_path_candidates=len(abs_positive),
        task_complete=False,remaining='full scan, branch attribution, absolute onset and convergence review'))
    print(f"Report written: {OUT/'REPORT.md'}; completed manifests: {', '.join(stages)}")


if __name__=='__main__':main()
