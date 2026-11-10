#!/usr/bin/env python3
"""Reproduce Phase A tables, uncertainty intervals, and publication figures."""
import json
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from scipy.interpolate import interp1d,BarycentricInterpolator
from llsa.secondary.operators import PierOperators
from appelquist2015_hopf_common import radial_weights
from fullbvp_phaseA_global import load_state,solve_mode
from fullbvp_phaseA_global import OUT,dump

def read(path):return json.loads((OUT/path).read_text())
def fmt(x,p=6):return f'{x:.{p}f}' if x is not None else '未识别'
def first(values):return values[0] if len(values)==1 else None

def cross_grid_correlations():
    reference=load_state(OUT/'neutral/N40_dr0.25/neutral_state.npz')
    mass=PierOperators(radius=594,n=40,beta=68,ro=-1.).grid.mass;rows=[]
    for n,dr in [(32,.5),(40,.5),(32,.25),(40,.25)]:
        state=load_state(OUT/f'neutral/N{n}_dr{dr:g}/neutral_state.npz')
        mask=(reference['radii']>=state['radii'][0])&(reference['radii']<=state['radii'][-1])
        r=reference['radii'][mask];old=state['q'][:,:3]
        if len(state['xi'])!=len(reference['xi']):old=BarycentricInterpolator(state['xi'],old,axis=2)(reference['xi'])
        old=interp1d(state['radii'],old,axis=0,kind='cubic')(r);new=reference['q'][mask,:3]
        weight=radial_weights(r)
        inner=lambda x,y:np.einsum('rvi,ij,rvj,r->',x.conj(),mass,y,weight,optimize=True)
        corr=float(abs(inner(old,new))/np.sqrt(inner(old,old).real*inner(new,new).real))
        if corr<.85:raise RuntimeError(f'Neutral cross-grid identity failed: {n,dr,corr}')
        rows.append(dict(n=n,dr_requested=dr,reference='N40_dr0.25',common_domain=[float(r[0]),float(r[-1])],correlation=corr))
    return rows

def main():
    scan=read('A1/summary.json')
    neutral=[read(f'neutral/N{n}_dr{dr:g}/summary.json') for n,dr in [(32,.5),(40,.5),(32,.25),(40,.25)]]
    a3=read('A3/summary.json');a4=read('A4/summary.json');spec=read('spectrum.json');checks=read('operator_checks.json')
    # Far spectral shifts can perturb a highly nonnormal target root despite a tiny residual.
    # Re-solve its most displaced near-zero repeat near the physical frequency.
    repeats=[x for x in spec['rows'] if x['accepted'] and x['sigma']>-.002 and
        2e-6<abs(complex(x['omega']-neutral[0]['omega'],x['sigma']-neutral[0]['sigma']))<2e-4]
    if repeats and 'target_repeat_refinement' not in spec:
        candidate=max(repeats,key=lambda x:abs(complex(x['omega']-neutral[0]['omega'],x['sigma']-neutral[0]['sigma'])))
        seed=load_state(OUT/'neutral/N32_dr0.5/neutral_state.npz')
        seed['omega']=candidate['omega'];seed['sigma']=candidate['sigma']
        destination=OUT/'spectrum_candidates';destination.mkdir(exist_ok=True)
        same,_=solve_mode(32,.5,neutral[0]['rout'],seed,intervals=640,path=destination/'target_repeat_same')
        fine,_=solve_mode(40,.5,neutral[0]['rout'],seed,intervals=640,path=destination/'target_repeat_N40')
        error=abs(complex(same['omega']-neutral[0]['omega'],same['sigma']-neutral[0]['sigma']))
        if error>1e-6:raise RuntimeError('Displaced near-zero spectrum repeat did not recover the target')
        spec['target_repeat_refinement']=dict(candidate=candidate,same_grid=same,N40_at_same_Rout=fine,
            distance_from_tracked_root=error,interpretation='Far-shift numerical sensitivity of the same highly nonnormal target mode; near-frequency repeat recovers the root.')
        dump(OUT/'spectrum.json',spec)
    finest=neutral[-1];cross_grid=cross_grid_correlations()
    rc=np.array([x['rout'] for x in neutral]);wc=np.array([x['omega'] for x in neutral])
    rend=np.array([first(x['rend_global']) for x in a3['grids']],dtype=float)
    rend_pier=np.array([first(x['rend_pier']) for x in a3['grids']],dtype=float)
    rend_all=np.array([v for g in a3['grids'] for key,vs in g.items() if key in ['rend_global','rend_global_cubic','rend_global_local_pchip','rend_global_local_cubic','rend_global_local_step4'] for v in vs],dtype=float)
    uncertainty=dict(method='Four-grid min/max envelope; spread is a resolution sensitivity, not an asymptotic error estimate.',
        rout_interval=[float(rc.min()),float(rc.max())],rout_spread=float(np.ptp(rc)),
        omega_interval=[float(wc.min()),float(wc.max())],omega_spread=float(np.ptp(wc)),
        r_end_global_grid_interval=[float(np.nanmin(rend)),float(np.nanmax(rend))] if np.isfinite(rend).any() else None,
        r_end_global_interval=[float(rend_all.min()),float(rend_all.max())] if len(rend_all) else None,
        r_end_sensitivity='Envelope includes four grids, unsmoothed gradient versus cubic derivative, local linear/PCHIP/cubic interpolation, and local step2 versus step4.',
        r_end_pier_interval=[float(np.nanmin(rend_pier)),float(np.nanmax(rend_pier))] if np.isfinite(rend_pier).any() else None)
    nominal_rend=first(a3['grids'][-1]['rend_global'])
    a3['nominal_r_end']=dict(n=40,dr_requested=.25,value=nominal_rend,
        unique_intersection=nominal_rend is not None,deviation_from_2015_594=None if nominal_rend is None else nominal_rend-594,
        uncertainty_interval=uncertainty['r_end_global_interval'],grid_interval=uncertainty['r_end_global_grid_interval'])
    samples=[]
    global_sample=a3['grids'][0]['sample_growth']
    for row in global_sample:
        rec=dict(row)
        for kind in ['global','pier']:
            local=next(x for x in a3['spatial_at_global_frequency'] if x['r']==row['r'] and x['kind']==kind)
            minus=next(x for x in local['branches'] if x['label']=='k_minus')
            rec[f'kminus_{kind}']=minus['spatial_growth']
            rec[f'group_real_{kind}']=minus['group_velocity'][0]
            rec[f'gq_minus_kminus_{kind}']=row['gq']-minus['spatial_growth']
        rec['gq_minus_DNS_approx']=row['gq']-.106;samples.append(rec)
    a3['growth_comparison_N32_dr0_5']=dict(samples=samples,
        gq_minus_DNS_approx_range=[min(x['gq_minus_DNS_approx'] for x in samples),max(x['gq_minus_DNS_approx'] for x in samples)],
        gq_minus_kminus_global_range=[min(x['gq_minus_kminus_global'] for x in samples),max(x['gq_minus_kminus_global'] for x in samples)])
    accepted=[x for x in spec['rows'] if x['accepted']]
    recovered=min(accepted,key=lambda x:abs(complex(x['omega']-neutral[0]['omega'],x['sigma']-neutral[0]['sigma'])))
    spec['target_recovery']=dict(recovered,omega_error=recovered['omega']-neutral[0]['omega'],sigma_error=recovered['sigma']-neutral[0]['sigma'])
    assert recovered['tracked_correlation']>.85 and abs(spec['target_recovery']['omega_error'])<1e-6 and abs(spec['target_recovery']['sigma_error'])<1e-6
    other=[x for x in accepted if abs(complex(x['omega']-neutral[0]['omega'],x['sigma']-neutral[0]['sigma']))>2e-4 and x['sigma']>-.002]
    verification=spec.get('independent_verifications',[])
    if other:
        first_statement='有限搜索中存在独立正增长或近零候选；其复核见下表。不能据此将目标分支称为最先失稳。'
    else:
        first_statement='已验收的有限搜索内，未识别出目标分支之外的正增长或近零模态；“最先失稳”仅限此次频率窗口、搜索中心与离散分辨率，未证明全谱排序。'
    summary=dict(phase='A, global linear only',parameters=dict(ro=-1,beta=68,rin=400,entry='0.8*S((420-R)/10)',sponge='a=Rout-100, rise=40, strength=28'),
        convention='Omega=omega_lab+i*sigma; disk frequency=omega_lab-68; omega_lab=50.5+Im(lambda), sigma=-Re(lambda)',
        A1=scan,neutral=neutral,nominal=finest,uncertainty=uncertainty,A3=a3,A4=a4,
        spectrum=spec,first_instability_statement=first_statement,operator_checks=checks,
        literature=dict(rout_2015_estimate=719,r_end_2015_estimate=594,neutral_omega_lab_graph_approx=53,
            neutral_omega_disk_graph_approx=-15,frequency_source='2015 Fig.7, graph reading only; Fig.8 -15+0.708i is not neutral global frequency.',
            growth_2016_approx=.106,growth_definition='Eq.(3.1), azimuthal RMS/local disk speed at z=1.3'),
        validation=dict(neutral_cross_grid_correlations=cross_grid,minimum_continuation_correlation=min(x['correlation'] for x in scan+neutral),
            maximum_eigen_residual=max(x['residual'] for x in scan+neutral),
            all_seven_requested_rout=[x['rout'] for x in scan],visual_review='Required after PNG and PDF rasterization.'))
    dump(OUT/'summary.json',summary)
    plt.rcParams.update({'font.family':'serif','font.serif':['STIXGeneral','DejaVu Serif'],'mathtext.fontset':'stix',
        'font.size':11,'axes.linewidth':.9,'xtick.direction':'in','ytick.direction':'in','savefig.dpi':300})
    fig,ax=plt.subplots(figsize=(5.4,3.6),layout='constrained')
    ax.axhline(0,color='0.25',lw=.8)
    ax.axvspan(rc.min(),rc.max(),color='#d99a28',alpha=.25,label='Four-grid critical interval')
    ax.axvline(719,color='0.5',ls='--',lw=.9,label='2015 estimate: 719')
    ax.plot([x['rout'] for x in scan],[x['sigma'] for x in scan],'-o',color='#1f5a92',lw=1.4,ms=4.5,label=r'$N=32,\ \Delta R=0.5$')
    ax.plot(finest['rout'],0,'D',color='#a6342f',ms=4.5,label=r'$N=40,\ \Delta R\simeq0.25$ neutral')
    ax.set(xlabel=r'$R_{\mathrm{out}}$',ylabel=r'$\sigma$',xlim=(698,742))
    ax.legend(frameon=False,fontsize=9,loc='upper left')
    ax.text(.98,.04,f'$R_{{c}}$ range: {rc.min():.4f}–{rc.max():.4f}',ha='right',transform=ax.transAxes,fontsize=9)
    for ext in ['png','pdf']:fig.savefig(OUT/f'sigma_Rout.{ext}')
    plt.close(fig)
    fig,ax=plt.subplots(figsize=(5.4,3.6),layout='constrained')
    growth=np.load(OUT/'A3/global_growth_N40_dr0.25.npz');mask=(growth['r']>=560)&(growth['r']<=625)
    ax.plot(growth['r'][mask],growth['gE'][mask],color='#1f5a92',lw=1.5,label=r'Global $\frac{1}{2}\,d\ln E/dR$')
    for kind,style,color,label in [('global','--','#a6342f','Frozen radial equations'),('pier',':','#3b753b','Classical frozen equations')]:
        c=read(f'A3/curve_N40_{kind}.json')
        ax.plot([x['r'] for x in c],[x['growth'] for x in c],style,color=color,lw=1.4,label=label)
    if uncertainty['r_end_global_interval']:ax.axvspan(*uncertainty['r_end_global_interval'],color='#d99a28',alpha=.2)
    ax.axvline(594,color='0.5',ls='-.',lw=.8,label='2015 estimate: 594')
    ax.set(xlabel=r'$R$',ylabel=r'Spatial growth rate',xlim=(560,625))
    ax.legend(frameon=False,fontsize=9)
    for ext in ['png','pdf']:fig.savefig(OUT/f'growth_intersection.{ext}')
    plt.close(fig)
    lines=['# Phase A：全局线性分支与中性出口','',
        '本阶段重建变化出口的完整径向离散，只计算零场全局线性模态。采用 '+summary['convention']+'。入口为 `0.8 S((420−R)/10)`，海绵参数为 `a=Rout−100, rise=40, strength=28`。','',
        '## A1：同一模态随出口变化','',
        '| Rout | 实际 ΔR | ω_lab | σ | 共同域速度质量相关 | 相对残差 |','|---:|---:|---:|---:|---:|---:|']
    lines += [f"| {x['rout']:g} | {x['dr_actual']:.6f} | {x['omega']:.6f} | {x['sigma']:.7f} | {x['correlation']:.6f} | {x['residual']:.2e} |" for x in scan]
    lines += ['', '相关追踪使用共同物理域、映射 Chebyshev 插值和速度质量内积，阈值 0.85，另要求相邻谱距离不超过0.25；高度非正交的相邻径向模态可有更高重叠，因此先用谱连续性门限再比较质量相关。705、735 及必要自动细分点仅作续接桥点。每次 Arnoldi 的全部候选在 A1/R*.json，失败候选单独保存。','',
        '## A2：中性出口与离散敏感性','',
        '| N | 请求 ΔR | 实际 ΔR | 区间数 | Rout_c | ω_c,lab | σ 根残差 |','|---:|---:|---:|---:|---:|---:|---:|']
    lines += [f"| {x['n']} | {x['dr_requested']:g} | {x['dr_actual']:.9f} | {x['intervals']} | {x['rout']:.7f} | {x['omega']:.7f} | {x['sigma']:.2e} |" for x in neutral]
    lines += ['', '四个中性解与N40/ΔR≈0.25参考模态的共同物理域速度质量相关为 '+str([(x['n'],x['dr_requested'],round(x['correlation'],8)) for x in cross_grid])+'，确认比较的是同一物理模态。', '',f"名义结果取 N40、ΔR≈0.25：Rout_c={finest['rout']:.7f}，ω_c,lab={finest['omega']:.7f}，ω_c,disk={finest['omega']-68:.7f}。四网格出口包络为 [{rc.min():.7f}, {rc.max():.7f}]，全幅 {np.ptp(rc):.5g}；频率包络为 [{wc.min():.7f}, {wc.max():.7f}]，全幅 {np.ptp(wc):.5g}。这给出分辨率敏感性；只有两个 N，不能宣称已进入渐近收敛阶。根求解固定径向区间数640或1280，避免节点跳变。",'',
        f"2015 印刷 p.623 给出 Rout_c≈719 的估计；本计算偏差 {finest['rout']-719:+.4f}。图7中性频率仅可图读为惯性系约53、盘系约−15；本结果相对约53的偏差为 {finest['omega']-53:+.4f}，该差异应与图读精度及入口/出口、海绵和离散实现差异一起判断。图8的 −15+0.708i 是局部理论对照，未用作中性真值。",'',
        '## 已搜索模态的失稳排序','',first_statement,'',
        f"多中心 shift-invert 的中心 (ω_lab, σ) 为 {spec['centers']}，每中心 k={spec['k']}、ncv={spec['ncv']}；实际验收频率范围 [{spec['frequency_range'][0]:.6f}, {spec['frequency_range'][1]:.6f}]，增长率范围 [{spec['growth_range'][0]:.6f}, {spec['growth_range'][1]:.6f}]，相对残差阈值 1e−7。完整候选与复核均保存在 spectrum.json。",'',f"独立多中心谱重获目标模态，质量相关 {recovered['tracked_correlation']:.9f}，频率差 {spec['target_recovery']['omega_error']:.3g}，增长率差 {spec['target_recovery']['sigma_error']:.3g}。",'']
    if 'target_repeat_refinement' in spec:
        rep=spec['target_repeat_refinement'];raw=rep['candidate'];same=rep['same_grid'];fine=rep['N40_at_same_Rout']
        lines += ['',f"远频移中心{raw['center_omega']:g}得到目标根的偏移重复值 ω={raw['omega']:.8f}、σ={raw['sigma']:.3g}，其目标质量相关为 {raw['tracked_correlation']:.11f}。窄频移复核恢复 ω={same['omega']:.8f}、σ={same['sigma']:.3g}，距已追踪根仅 {rep['distance_from_tracked_root']:.3g}；因此原谱范围中的该近零正值属于高度非正交目标根的远频移数值敏感性。相同Rout下N40复核 σ={fine['sigma']:.6g}，对应分辨率变化后的同一分支，不能当作N40自身的中性根。"]
    if verification:
        lines += ['| 候选 ω | 候选 σ | 同网格复核 ω/σ/相关 | N40复核 ω/σ/相关 |','|---:|---:|---|---|']
        for v in verification:
            same=v.get('same_grid',{});fine=v.get('N40',{})
            lines.append(f"| {v['candidate']['omega']:.6f} | {v['candidate']['sigma']:.6f} | {same.get('omega')}/{same.get('sigma')}/{same.get('correlation')} | {fine.get('omega')}/{fine.get('sigma')}/{fine.get('correlation')} |")
    lines += ['', '## A3：有效线性区末端与空间增长','',
        '采用2015图5(b)定义：全局 g_E=(1/2)d ln E/dR 与局部绝对夹点 −Im α₀(R) 的交点。E=∫|q_velocity|²dz；q_velocity=u_physical/R，已消去旧能量储存中可能存在的 R² 因子。绝对夹点直接来自物理坐标冻结 QEP，解析群速为零，并以相邻半径速度质量相关续接。','',
        '| N | 实际 ΔR | r_end：冻结径向方程 | r_end：经典冻结方程 | 三次样条导数诊断 |','|---:|---:|---:|---:|---:|']
    for g in a3['grids']:
        lines.append(f"| {g['n']} | {g['dr_actual']:.9f} | {fmt(first(g['rend_global']),4)} | {fmt(first(g['rend_pier']),4)} | {fmt(first(g['rend_global_cubic']),4)} |")
    lines += ['',f"名义 r_end（N40、ΔR≈0.25、唯一交点）={fmt(nominal_rend,4)}，相对2015约594的偏差为 {fmt(None if nominal_rend is None else nominal_rend-594,4)}。",'',f"冻结径向方程 r_end 综合包络：{uncertainty['r_end_global_interval']}；经典冻结方程包络：{uncertainty['r_end_pier_interval']}。2015 的约594是基于图5(b)交点的估计。两种冻结方式保留的径向缩放导数/非平行项不同，不能将其差值归入中性求根残差。局部半径采样550–630、步长2，交点以分段线性为主，并比较局部PCHIP/三次插值及抽稀至步长4；误差包络合并这些局部插值变化、四网格差和全局导数差。导数从未平滑的 log(E) 计算，另以三次样条导数检查敏感性。",'',
        '| N | R594 矩阵最大差 | 相对 Frobenius 差 | 同 α 的 Ω：冻结径向/经典 |','|---:|---:|---:|---|']
    for d in a3['frozen_model_comparisons']:lines.append(f"| {d['n']} | {d['matrix_max']:.4g} | {d['matrix_relative_frobenius']:.4g} | {d['omega_at_same_alpha_global']} / {d['omega_at_same_alpha_pier']} |")
    lines += ['', '每组夹点均将 Ω 从 Ω₀+i10⁻⁶ 向上续接至 Ω₀+12i；每一步使用唯一指派保持两支一一对应，并记录速度质量相关、空间残差与根间距。两支最终位于 α 平面相反半平面。该检验提供有限路径的 Briggs–Bers 因果夹点证据；未声称穷尽其他夹点。原始路径见 A3/pinch_*.json。','',
        '2016式(3.1)采用 z=1.3 的方位速度除以局部盘速，故主比较量为 g_q=d ln|q_v(1.3)|/dR；物理速度对应 g_phys=g_q+1/R。0.106 作为用户给定的近似DNS对照，不能与局部夹点 −Im α₀≈0.1246 直接等同。','',
        '| N | ΔR | g_q：R570–600 均值及范围 | g_phys均值 | g_E均值 |','|---:|---:|---|---:|---:|']
    for g in a3['grids']:
        z=g['growth_R570_600'];q=z['gq'];lines.append(f"| {g['n']} | {g['dr_actual']:.6f} | {q['mean']:.6f} [{q['min']:.6f}, {q['max']:.6f}] | {z['gphys']['mean']:.6f} | {z['gE']['mean']:.6f} |")
    finest_growth=a3['grids'][-1]['growth_R570_600'];qg=finest_growth['gq']
    lines += ['',f"最细网格的归一化方位速度增长率在R570–600均值为 {qg['mean']:.6f}，范围 [{qg['min']:.6f}, {qg['max']:.6f}]，相对DNS近似0.106仍高 {qg['min']-.106:.6f}–{qg['max']-.106:.6f}。物理速度换算项1/R约0.00167–0.00175，无法解释全部差距。当前线性中性全局模态、局部冻结空间支与2016 DNS的线性/非线性阶段、参数和取样设置尚未逐项匹配，这些来源尚未分离；不能将差距全部归因于速度归一化或定义差异。"]
    lines += ['', '| R | 全局 g_q(z=1.3) | 全局 g_E | 因果 k−：冻结径向 | 因果 k−：经典冻结 | g_q−0.106 |',
        '|---:|---:|---:|---:|---:|---:|']
    for row in samples:
        lines.append(f"| {row['r']:g} | {row['gq']:.6f} | {row['gE']:.6f} | {row['kminus_global']:.6f} | {row['kminus_pier']:.6f} | {row['gq_minus_DNS_approx']:+.6f} |")
    comp=a3['growth_comparison_N32_dr0_5']
    lines += ['',f"上述R570–600采样中，g_q相对用户给定DNS近似0.106的差值范围为 {comp['gq_minus_DNS_approx_range']}；相对冻结径向因果k−增长率的差值范围为 {comp['gq_minus_kminus_global_range']}。这些差别保留了全局场、多分量能量与局部冻结空间支之间的定义差异。"]
    bad=[(x['r'],kind,x[f'group_real_{kind}']) for x in samples for kind in ['global','pier'] if x[f'group_real_{kind}']>=0]
    if not bad:lines += ['', '两种模型所有列出采样点的因果k−支均有 Re(dΩ/dα)<0，与向内传播解释一致。']
    else:lines += ['',f'下列采样点的因果k−支复群速实部不为负：{bad}。k−标签由高ImΩ处的因果半平面和连续路径保留；复α上的复群速实部本身不能替代这一定义，因此这些点不宣称已由群速确认向内传播。']
    lines += ['', '局部 k−/k+ 从已验证夹点 Ω₀+12i 下的 α 下/上半平面分别标记，唯一续接至全局中性频率，再以步长2沿半径续接；负群速度另由复群速实部核实。R570、580、590、594、600 的 α、增长率与群速完整保存于 A3/summary.json。','',
        '## A4：出口长度与海绵参数化','',
        '以 N32、实际 ΔR=0.5、a=620、rise=40、strength=28 固定，仅改变出口，隔离出口闭合的影响。旧定域中性海绵起点 a_c=619.958162（ΔR=0.5）和619.954056（ΔR=0.25）作为历史对照，未修改旧结果。','',
        '| Rout | a | 实际 ΔR | ω_lab | σ |','|---:|---:|---:|---:|---:|']
    for x in a4['rows']:lines.append(f"| {x['rout']:g} | {x['a']:g} | {x['dr_actual']:.9f} | {x['omega']:.7f} | {x['sigma']:.7f} |")
    rows=a4['rows'];dw=rows[1]['omega']-rows[0]['omega'];ds=rows[1]['sigma']-rows[0]['sigma']
    outlet_small=abs(dw)<max(1e-7,.01*np.ptp(wc)) and abs(ds)<max(1e-7,.01*np.ptp(rc)*.006)
    outlet_statement='与四网格敏感性相比，该出口差异可忽略；a_c+100与移动出口临界值在当前精度内一致，额外阻尼平台影响可忽略。' if outlet_small else '该差异量化了可辨的出口闭合效应；旧定域a_c+100到移动出口中性根仅是近似映射。'
    lines += ['',f"相同海绵与网格下，Rout从720到730的频率差为 {dw:+.5g}，增长率差为 {ds:+.5g}。"+outlet_statement,'',
        '## 复现与验收','',
        '`commands.sh` 固定 BLAS/OpenMP 等六项线程数为1，同时最多五个 worker；逐一等待并记录退出码，任一失败写 FAILED，只有所有计算和报告成功才写 GLOBAL_DONE。失败的旧日志保留于 archive/shape_failure_20261030；首次高相关相邻模态误选的检查点与退出码保留于 archive/branch_mix_20261030。最终分支以谱距离门限加共同域质量相关重新验收。',
        f"独立复矩阵核验最大差 {checks['complex_vs_independent_max']:.3g}；频率导数缺陷 {checks['omega_derivative_defect']:.3g}。主分支最小相关 {summary['validation']['minimum_continuation_correlation']:.6f}，最大残差 {summary['validation']['maximum_eigen_residual']:.3g}。",'',
        '图件由本脚本直接读取 JSON/NPZ：sigma_Rout.png/.pdf 显示零增长线、临界网格包络和2015出口估计；growth_intersection.png/.pdf 显示全局增长与两个局部夹点曲线。画布5.4×3.6英寸，STIX衬线字体，不对数据平滑。PNG与PDF栅格化后的目检结果另见 visual_review.json。','',
        '文献：Appelquist et al. (2015), https://doi.org/10.1017/jfm.2015.2，p.623–624、图5(b)、图7；Appelquist et al. (2016), https://doi.org/10.1017/jfm.2016.506，式(3.1)、图8。']
    model_shift=None if nominal_rend is None else nominal_rend-first(a3['grids'][-1]['rend_pier'])
    lines += ['', '偏差来源判断：四网格量化的出口临界值敏感性为 '+f'{np.ptp(rc):.4f}'+
        '，频率敏感性为 '+f'{np.ptp(wc):.5f}'+
        '；A4将额外阻尼平台长度的影响单独隔离。'+
        ('该出口效应远小于上述分辨率差，不能解释主要临界值偏差。' if outlet_small else '出口效应应与分辨率差分别计入。')+
        '局部冻结方式使名义r_end改变 '+fmt(model_shift,4)+
        '，说明非平行/径向缩放约定比局部交点插值误差更敏感。2015的719与594是估计，频率约53来自图读；当前证据不足以把图读差异归为已证实的物理频移。']
    (OUT/'REPORT.md').write_text('\n'.join(lines)+'\n')
    print(json.dumps(dict(report=str(OUT/'REPORT.md'),rout_interval=uncertainty['rout_interval'],omega_interval=uncertainty['omega_interval'],rend_interval=uncertainty['r_end_global_interval'],figures=['sigma_Rout','growth_intersection'])),flush=True)

if __name__=='__main__':main()
