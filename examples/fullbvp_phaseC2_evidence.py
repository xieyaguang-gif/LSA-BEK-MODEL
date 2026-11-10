#!/usr/bin/env python3
"""Derive the achieved Phase C C3 evidence without changing numerical ledgers."""
from __future__ import annotations
import os
for _key in ('OPENBLAS_NUM_THREADS','OMP_NUM_THREADS','MKL_NUM_THREADS','NUMEXPR_NUM_THREADS','VECLIB_MAXIMUM_THREADS','BLIS_NUM_THREADS'):
    os.environ[_key]='1'
os.environ.setdefault('MPLCONFIGDIR','/tmp/fullbvp_phaseC2_matplotlib')
import argparse
from pathlib import Path
import numpy as np
from fullbvp_phaseC_solver import ROOT,STRICT,read,write,sha
from fullbvp_phaseC_postprocess import diagnostic,profile_difference,NL03,NL01

PHASE_C=ROOT/'results/fullbvp_global_20261030/phaseC'

def validate_field(folder):
    """Read independently strict, screened, trustworthy accepted checkpoint only."""
    folder=Path(folder).resolve()
    summary=read(folder/'summary.json'); gate=read(folder/'screen.json')
    independent=summary.get('independent',{})
    if not (summary.get('accepted') and gate.get('accepted')
        and summary['relative_field_inf']<STRICT['relative_field_inf']
        and summary['residual_over_fundamental_feedback']<STRICT['residual_over_fundamental_feedback']
        and max(abs(summary['phase']),abs(summary['branch_constraint']))<STRICT['constraint']
        and independent.get('relative_field_inf',float('inf'))<STRICT['independent_relative_field_inf']
        and independent.get('template_difference_relative',float('inf'))<=STRICT['template_difference_relative']
        and summary['highest_to_fundamental']<=.01):
        raise ValueError('Field fails independent strict/screen/truncation acceptance: '+str(folder))
    field=folder/'fields.npz'
    digest=sha(field)
    audited_diagnostic=folder.parent.parent/'diagnostics'/f'{folder.name}.json'
    if not audited_diagnostic.exists() or read(audited_diagnostic).get('sha256')!=digest:
        raise ValueError('Checkpoint SHA256 differs from independently diagnosed field provenance: '+str(field))
    with np.load(field) as saved:
        H=int(saved['H']);n=int(saved['n']);r=saved['radii'];q=saved['field']
        if (not bool(saved['accepted']) or n!=32 or summary['n']!=n
            or summary['intervals']!=800 or int(saved['intervals'])!=800 or len(r)!=801
            or abs(r[0]-400.)>1e-10 or abs(r[-1]-float(saved['Rout']))>1e-10
            or not np.all(np.diff(r)>0) or H!=summary['H']
            or abs(float(saved['omega'])-summary['omega'])>1e-10
            or abs(float(saved['Rout'])-summary['Rout'])>1e-10
            or not np.isfinite(q).all()):
            raise ValueError('Accepted field metadata mismatch: '+str(field))
        v=q.reshape(len(r),2*H+1,4,len(saved['xi']))[:,H+1:,:3]
        density=np.einsum('rhvi,ij,rhvj->rh',v.conj(),saved['mass'],v,optimize=True).real
        energy=np.trapezoid(density,r,axis=0)
        if (abs(np.sqrt(energy[0])-summary['A1'])>1e-9
            or abs(energy[-1]/energy[0]-summary['highest_to_fundamental'])>1e-9):
            raise ValueError('Velocity mass energy inconsistent with accepted summary')
    keys=('Rout','omega','A1','energies','highest_to_fundamental','relative_field_inf',
          'residual_over_fundamental_feedback','n','H','intervals','phase','branch_constraint')
    point={key:summary[key] for key in keys}
    point.update(id=folder.name,fields=str(field),fields_sha256=digest,
                 summary=str(folder/'summary.json'),summary_sha256=sha(folder/'summary.json'),
                 screen=str(folder/'screen.json'),screen_sha256=sha(folder/'screen.json'),
                 independent=independent,accepted=True,resolved=True,a=summary['Rout']-100,
                 dr=(summary['Rout']-400)/800,role='strict_attempt_evidence')
    return point


def coherent_discrepancy(base,other):
    with np.load(base['fields']) as a,np.load(other['fields']) as b:
        if int(a['H'])!=int(b['H']) or not np.allclose(a['radii'],b['radii'],atol=1e-9,rtol=0):
            raise ValueError('Dedup requires same H and physical radial coordinates')
        H=int(a['H']);m=len(a['xi']);r=a['radii'];rw=np.gradient(r);rw[[0,-1]]*=.5
        u=a['field'].reshape(len(r),2*H+1,4,m)[:,H+1:,:3]
        v=b['field'].reshape(len(r),2*H+1,4,m)[:,H+1:,:3].copy()
        inner=lambda x,y:np.einsum('rhvi,ij,rhvj,r->',x.conj(),a['mass'],y,rw,optimize=True)
        fundamental=np.einsum('rvi,ij,rvj,r->',u[:,0].conj(),a['mass'],v[:,0],rw,optimize=True)
        phase=float(np.angle(fundamental));v*=np.exp(-1j*np.arange(1,H+1)*phase)[None,:,None,None]
        rel=float(np.sqrt(max(inner(v-u,v-u).real,0)/max(inner(u,u).real,1e-30)))
        corr=float(inner(u,v).real/np.sqrt(max(inner(u,u).real*inner(v,v).real,1e-60)))
        return dict(coherent_velocity_relative_mass=rel,phase_aligned_correlation=corr,
                    phase_alignment=phase,delta_omega=other['omega']-base['omega'],
                    delta_A1=other['A1']-base['A1'],delta_Rout=other['Rout']-base['Rout'])


def collect(directory=PHASE_C):
    directory=Path(directory).resolve();state=read(directory/'campaign.json')
    committed={p['id'] for p in state['points']};valid=[];excluded=[];failed=[]
    for summarypath in sorted((directory/'attempts').glob('*/summary.json')):
        row=read(summarypath)
        if not row.get('accepted'):
            failed.append(dict(id=summarypath.parent.name,summary=str(summarypath),
                summary_sha256=sha(summarypath),Rout=row['Rout'],H=row['H'],
                failure=row.get('failure'),seconds=row['seconds'],budget_seconds=row.get('budget_seconds'),
                checkpoint=str(summarypath.parent/'checkpoint.npz')))
            continue
        try: point=validate_field(summarypath.parent)
        except (ValueError,KeyError,FileNotFoundError) as exc:
            excluded.append(dict(id=summarypath.parent.name,reason=str(exc)));continue
        point['committed_in_phaseC']=point['id'] in committed
        point['ledger_status']='committed' if point['id'] in committed else 'strict_screened_attempt_not_committed'
        valid.append(point)
    unique=[];duplicates=[]
    for point in sorted(valid,key=lambda p:(p['Rout'],p['H'],p['id'])):
        prior=next((p for p in unique if p['H']==point['H'] and abs(p['Rout']-point['Rout'])<1e-6),None)
        if prior is None:
            point['equivalent_solves']=[dict(id=point['id'],fields=point['fields'],sha256=point['fields_sha256'])]
            unique.append(point);continue
        discrepancy=coherent_discrepancy(prior,point)
        if discrepancy['coherent_velocity_relative_mass']>1e-6 or abs(discrepancy['delta_omega'])>1e-7:
            raise ValueError('Same-Rout/H roots differ; cannot silently deduplicate')
        row=dict(retained_id=prior['id'],other_id=point['id'],
            retained_fields_sha256=prior['fields_sha256'],other_fields_sha256=point['fields_sha256'],
            discrepancy=discrepancy,both_numerically_strict=True,both_committed=False)
        duplicates.append(row);prior['equivalent_solves'].append(dict(id=point['id'],fields=point['fields'],sha256=point['fields_sha256']))
    return dict(points=unique,duplicates=duplicates,failures=failed,excluded=excluded,
                numerical_ledger_status=state['status'],targets=state['targets'],
                source_ledger=str(directory/'campaign.json'),source_ledger_sha256=sha(directory/'campaign.json'),
                source_accepted_points_sha256=sha(directory/'accepted_points.json'))


def style():
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    plt.rcParams.update({'font.family':'serif','font.serif':['Times New Roman','Nimbus Roman','STIXGeneral','DejaVu Serif'],
        'mathtext.fontset':'stix','font.size':10,'axes.linewidth':.85,'axes.edgecolor':'black',
        'axes.spines.top':False,'axes.spines.right':False,'figure.facecolor':'white',
        'axes.facecolor':'white','pdf.fonttype':42,'savefig.facecolor':'white'})
    return plt


def render(directory,evidence):
    plt=style();directory=Path(directory);out=directory/'figures_C3';out.mkdir(exist_ok=True)
    points=evidence['points'];main=sorted([p for p in points if p['H']==5],key=lambda p:p['Rout'])
    higher=[p for p in points if p['H']==6];manifest=[]
    colors=['#0072B2','#E69F00','#009E73','#CC79A7']
    def finish(fig,name,size,axes,description):
        fig.set_size_inches(*size);fig.tight_layout()
        files=[]
        for suffix in ('pdf','png'):
            path=(out/f'{name}.{suffix}').resolve();fig.savefig(path,dpi=240);files.append(str(path))
        manifest.append(dict(name=name,files=files,size_inches=size,description=description,
            axes=[dict(xlim=list(ax.get_xlim()),ylim=list(ax.get_ylim()),xscale=ax.get_xscale(),yscale=ax.get_yscale()) for ax in axes]))
        plt.close(fig)
    fig,ax=plt.subplots(figsize=(7.2,5.15))
    for i,p in enumerate(main):
        with np.load(directory/'audit_C3/diagnostics'/f'{p["id"]}.npz') as d:
            r,vrms=d['radii'],d['vrms'];mask=r<=p['a'];label=f'H5, Rout={p["Rout"]:.2f}'
            if not p['committed_in_phaseC']:label+=' (uncommitted)'
            ax.semilogy(r[mask],vrms[mask],color=colors[i],lw=1.6,label=label)
            ax.semilogy(r[~mask],vrms[~mask],color=colors[i],ls='--',lw=1.1)
    dns=np.genfromtxt(NL03,delimiter=',',names=True)
    ax.semilogy(dns['R'],dns['vrms_normalized'],color='black',lw=1.5,label='NL03 T=6, Fig.15 digitization')
    ax.fill_between(dns['R'],dns['vrms_plot_lower'],dns['vrms_plot_upper'],color='.5',alpha=.14)
    if NL01.exists():
        dn=np.genfromtxt(NL01,delimiter=',',names=True)
        ax.errorbar(dn['R'],dn['vrms_normalized'],xerr=dn['R_plot_reading_tolerance'],
            yerr=np.stack([dn['vrms_normalized']-dn['vrms_plot_lower'],dn['vrms_plot_upper']-dn['vrms_normalized']]),
            fmt='s',ms=3,color='#D55E00',capsize=2,label='NL01 T=6, sparse Fig.6(b) contours')
    ax.set(xlabel=r'$R$',ylabel=r'$v_{\rm rms}/(R\Omega)$ at $z=1.3$',xlim=(515,730),ylim=(2e-8,.3))
    ax.legend(fontsize=7.3,loc='upper center',bbox_to_anchor=(.5,-.16),ncol=2,frameon=False);ax.text(.02,.98,'Dashed: sponge support',va='top',transform=ax.transAxes,fontsize=8)
    finish(fig,'C3_vrms_DNS',(7.2,5.15),[ax],'Comparable H5 profiles, locally normalized; dashed beyond a=Rout-100. NL01 is sparse, not a continuous curve.')
    fig,axes=plt.subplots(2,2,figsize=(7.4,5.6));flat=axes.ravel()
    panels=[('front_fig8','Fig.8 total-RMS front',583.),('omega',r'$\omega$',52.36),
            ('sponge_fraction_q','Coherent q sponge fraction',None),('g_q_fit_570_600',r'$g_q$ fit, $R=570$–$600$',.106)]
    for ax,(key,label,reference) in zip(flat,panels):
        ax.plot([p['Rout'] for p in main],[p[key] for p in main],'-o',color='#0072B2',ms=4,label='H5 comparable sequence')
        ax.plot([p['Rout'] for p in higher],[p[key] for p in higher],'s',mfc='white',mec='#D55E00',ms=5,label='H6 corrected checks')
        last=main[-1];ax.plot(last['Rout'],last[key],'o',mfc='white',mec='#0072B2',ms=6)
        ax.set(xlabel=r'$R_{\rm out}$',ylabel=label,xlim=(718,731))
        if reference is not None:
            text=f'DNS ≈{reference:g} (outside plotted range)'
            if key=='front_fig8':ax.text(.02,.05,text,transform=ax.transAxes,fontsize=7.6,va='bottom')
            elif key=='omega':ax.text(.98,.95,text,transform=ax.transAxes,fontsize=7.6,va='top',ha='right')
            else:ax.text(.03,.95,text,transform=ax.transAxes,fontsize=7.6,va='top')
    flat[0].legend(fontsize=7.2,loc='upper right');flat[2].axhline(.5,color='.4',ls=':',lw=1)
    flat[2].set_ylim(.48,.80);flat[0].set_ylim(587.3,588.8)
    flat[3].axhline(.132,color='.4',ls=':',lw=.9,label='Linear ≈0.132')
    flat[3].text(.97,.82,'Linear ≈0.132',ha='right',transform=flat[3].transAxes,fontsize=7.6,color='.3')
    fig.suptitle('Short H5 sequence: last increment nearly stationary; 730 H6 unfinished',fontsize=10)
    fig.tight_layout(rect=(0,0,1,.96))
    finish(fig,'C3_domain_trends',(7.4,5.6),list(flat),'H5 domain trend; H6 uncertainty shown with open squares. Open 730 H5 is strict but uncommitted; DNS references shown as text to preserve local trend scale.')
    fig,axes=plt.subplots(1,2,figsize=(7.4,3.35))
    for key,label,marker,color in [('sponge_fraction_q','Coherent q','o','#0072B2'),('sponge_fraction_cylindrical_R3',r'Coherent $R^3$','^','#009E73')]:
        axes[0].plot([p['Rout'] for p in main],[p[key] for p in main],'-'+marker,color=color,ms=4,label=label)
    axes[0].axhline(.5,color='.4',ls=':',lw=1);axes[0].set(xlabel=r'$R_{\rm out}$',ylabel='Sponge fraction',xlim=(718,731),ylim=(.48,1.));axes[0].legend(fontsize=8)
    pairs=evidence['harmonic_differences'];pairs=[p for p in pairs if p['H']==5 and p['Hnext']==6]
    if pairs:
        axes[1].plot([p['Rout'] for p in pairs],[p['delta_front'] for p in pairs],'-s',color='#D55E00',ms=4,label='H6 − H5 front')
    axes[1].axvline(730,color='.6',ls=':',lw=1);axes[1].text(.97,.94,'730 H6\nunavailable',ha='right',va='top',transform=axes[1].transAxes,fontsize=8)
    axes[1].set(xlabel=r'$R_{\rm out}$',ylabel='Same-Rout front difference',xlim=(718,731));axes[1].legend(fontsize=8)
    finish(fig,'C3_sponge_harmonic',(7.4,3.35),list(axes),'Both coherent measures remain substantial. H6-H5 front difference is absent at 730; no extrapolated H6 result.')
    write(out/'figure_manifest.json',dict(status='pending_parent_visual_review',figures=manifest,
        input_points=[dict(id=p['id'],fields=p['fields'],sha256=p['fields_sha256'],H=p['H'],Rout=p['Rout']) for p in points],
        diagnostics_directory=str((directory/'audit_C3/diagnostics').resolve()),
        transformations=['Chebyshev interpolation at z=1.3','RMS=sqrt(2 sum |q_v,j|^2), no extra division by R',
            'Logarithmic y-axis for RMS only','H5 main sequence; H6 independent checks','Linear connections only between available data; 730 H6 unavailable'],
        exclusion='No failed/pending root is plotted as a solved field; duplicate 730 solves represented once',
        references=dict(NL03=str(NL03),NL01=str(NL01))))
    return manifest


def finalize(directory=PHASE_C):
    directory=Path(directory).resolve();audit=directory/'audit_C3';audit.mkdir(exist_ok=True)
    evidence=collect(directory)
    if len(evidence['points'])!=8 or len(evidence['duplicates'])!=1:
        raise ValueError('Expected exactly eight unique trusted fields and one duplicate pair')
    evidence['points']=[diagnostic(p,audit) for p in evidence['points']]
    differences=[]
    for p in evidence['points']:
        higher=next((q for q in evidence['points'] if q['H']==p['H']+1 and abs(q['Rout']-p['Rout'])<1e-6),None)
        if higher is not None:
            differences.append(dict(Rout=p['Rout'],H=p['H'],Hnext=higher['H'],ids=[p['id'],higher['id']],
                delta_omega=higher['omega']-p['omega'],delta_front=None if p['front_fig8'] is None or higher['front_fig8'] is None else higher['front_fig8']-p['front_fig8'],
                delta_g_q_fit=higher['g_q_fit_570_600']-p['g_q_fit_570_600']))
    evidence['harmonic_differences']=differences
    main=sorted([p for p in evidence['points'] if p['H']==5],key=lambda p:p['Rout']);trend=[]
    for a,b in zip(main[:-1],main[1:]):
        dR=b['Rout']-a['Rout'];trend.append(dict(Rout_pair=[a['Rout'],b['Rout']],
            delta_omega=b['omega']-a['omega'],delta_front=b['front_fig8']-a['front_fig8'],
            domega_dRout=(b['omega']-a['omega'])/dR,dfront_dRout=(b['front_fig8']-a['front_fig8'])/dR,
            delta_g_q_fit=b['g_q_fit_570_600']-a['g_q_fit_570_600'],common_R_profile=profile_difference(audit,a,b)))
    evidence['H5_domain_trend']=trend
    evidence['interpretation']=dict(domain_convergence_established=False,
        short_trend='Front and omega decrease through 725, then change little and slightly reverse at 730; insufficient to establish an asymptotic limit.',
        support='All four H5 coherent q sponge fractions exceed 0.5, but front motion is not persistently outward. High boundary/sponge participation remains; causal support is unproved.',
        harmonic='H6 checks at source/720/725 resolve some harmonic sensitivity; missing 730 H6 prevents assigning the last H5 increment a verified H6 uncertainty.',
        unknowns=['730 H6 promotion unfinished','No data at 740–800','No independent N/dr refinement','No independent sponge-location/outflow-model variation','No temporal stability result'],
        source_relabeling=False)
    write(audit/'evidence.json',evidence);manifest=render(directory,evidence)
    lines=['# Phase C 已达到域长度的诊断草稿（C3）','',
        '采用一致 H5 序列比较，前沿与频率随出口半径增加先下降至 Rout=725，在 730 略有反转。这段短序列尚不能建立域收敛。',
        '原数值流程在 730 的同 Rout H5→H6 升阶校正处失败；原 ledger 中 730 及以后的目标仍为 pending，未被本报告改写。730/H5 已通过独立严格残差、分支筛选与截断可信度检查，但尚未提交为完成目标；本报告仅将其纳入科学诊断，两次等价求解只计一个状态。','',
        '| Rout | H | 原 ledger 状态 | omega | Fig.8 前沿 | g_q 端点 | g_q 拟合 | q 海绵比例 | R³ 海绵比例 | E_H/E1 |',
        '|---|---|---|---|---|---|---|---|---|---|']
    for p in evidence['points']:
        status='已提交' if p['committed_in_phaseC'] else '严格通过但未提交'
        values=[f'{p["Rout"]:.7f}',str(p['H']),status]+[f'{p[k]:.9g}' if p[k] is not None else '不可用' for k in ('omega','front_fig8','g_q_570_600','g_q_fit_570_600','sponge_fraction_q','sponge_fraction_cylindrical_R3','highest_to_fundamental')]
        lines.append('| '+' | '.join(values)+' |')
    lines+=['','## H5 的 2beta 次级前沿','', '| Rout | 复幅值 1e-6 首次向上穿越 R | 相对基频 5% 首次向上穿越 R |','|---|---|---|']
    for p in main:
        values=[f'{p[k]:.8g}' if p[k] is not None else '无覆盖内穿越' for k in ('front_2beta_absolute','front_2beta_5percent')]
        lines.append(f'| {p["Rout"]:.7f} | '+ ' | '.join(values)+' |')
    lines+=['','## 一致 H5 域长度趋势','', '| Rout 区间 | Δomega | Δfront | Δg_q 拟合 | 公共物理 R 区间 RMS 相对 L2 差 |','|---|---|---|---|---|']
    for t in trend:lines.append('| '+str(t['Rout_pair'])+' | '+' | '.join(f'{t[k]:.8g}' for k in ('delta_omega','delta_front','delta_g_q_fit'))+f' | {t["common_R_profile"].get("RMS_relative_L2",float("nan")):.8g} |')
    lines+=['',
        '725→730 的 H5 前沿及频率变化很小，但同时出现轻微反转，不能据此推断渐近极限。H6 检查只覆盖源点、720 与 725；由于 730/H6 尚未完成，最后一段 H5 变化没有对应的 H6 误差界。',
        '四个 H5 状态的相干 q 海绵比例均大于 0.5，说明海绵区参与仍显著；前沿并未持续向外平移，因此 C4 要求的“持续 ≥50% 海绵支持且前沿持续向外移动”联合判据没有满足。大比例本身不能证明边界导致该分支，仍需独立改变海绵位置和出口模型。','',
        '## 谐波检查与未完成升阶','', '| Rout | H 比较 | Δomega | Δfront | Δg_q 拟合 |','|---|---|---|---|---|']
    for t in differences:lines.append(f'| {t["Rout"]:.7f} | {t["H"]}→{t["Hnext"]} | {t["delta_omega"]:.8g} | {t["delta_front"]:.8g} | {t["delta_g_q_fit"]:.8g} |')
    for f in evidence['failures']:lines.append(f'- {f["id"]}，H{f["H"]}，Rout={f["Rout"]:g}：{f["failure"]}；耗时 {f["seconds"]:.1f} 秒，单点预算 {f["budget_seconds"]} 秒。证据：{f["summary"]}。')
    dup=evidence['duplicates'][0];lines+=['',
        '4800 秒预算下的重试实际因 Newton 线搜索失败而结束，不能将一次失败校正解释为 H6 解不存在。',
        f'730 重复状态来自 {dup["retained_id"]} 与 {dup["other_id"]}；相位对齐后的相干速度质量范数相对差为 {dup["discrepancy"]["coherent_velocity_relative_mass"]:.4g}。两个 ID、场文件哈希与标量差均保留在 audit_C3/evidence.json。','',
        '## 诊断定义、覆盖区间与 DNS 对比','',
        '在 z=1.3，q_v 已按局部圆盘速度归一化，RMS=sqrt(2 sum |q_v,j|²)，不再除以 R。均值与压力不计入相干能量。Fig.8 前沿使用总 RMS 对数增长率与固定 Phase A 局部绝对 pinch 的向下交点；N32 曲线只覆盖 R550–630。每个场的搜索区间和覆盖限制均单独保存。本次已达到的场存在覆盖内交点；对缺失或超出覆盖的前沿不外推。',
        '2beta 次级前沿分别采用复幅值 1e-6 和相对基频 5% 的阈值。R570–600 的端点增长率与最小二乘拟合增长率分开报告。海绵比例分别采用归一化 q 质量范数与物理 R³ 权重。',
        'NL03 使用 2016 年 Fig.15 已有数化曲线及图读包络；NL01 是独立的 Fig.6(b)、T=6 稀疏等值线样本，不能连接为虚构的连续 DNS 曲线。参考前沿约 583、频率约 52.36、DNS 增长率约 0.106、线性增长率约 0.132。当前前沿约 588、频率约 52.68 仍高于相应参考；增长率直接由实际场给出，不强行匹配 DNS。','',
        '## 后续检查与可追溯性','',
        '下一步从严格 725/H6 解继续，先解决 730/H6，再以不超过 2.5 的 Rout 增量扩展。还需独立检验 N、径向间距、海绵位置/宽度及有依据的出口模型。本报告没有时间稳定性结论。',
        f'原数值 ledger（未改）：{evidence["source_ledger"]}；SHA256：{evidence["source_ledger_sha256"]}。',
        f'派生场哈希、完整数值剖面、重复状态与失败证据：{str(audit.resolve())}。',
        f'图的数据映射、物理尺寸、准确坐标范围与变换：{str((directory/"figures_C3/figure_manifest.json").resolve())}。',
        '输出为可复现的 PDF 与 PNG；最终视觉审查由主管完成。']
    (directory/'REPORT.md').write_text('\n'.join(lines)+'\n')
    write(audit/'C3_DERIVED_DONE.json',dict(status='derived_report_and_figures_saved',unique_fields=8,
        duplicate_pairs=1,source_ledger_sha256=evidence['source_ledger_sha256'],
        report=str((directory/'REPORT.md').resolve()),figures=manifest,visual_review='pending_parent_review',
        production_campaign_success=False))
    return evidence

if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--out',type=Path,default=PHASE_C)
    args=parser.parse_args();e=finalize(args.out)
    print(f'Derived C3 evidence: {len(e["points"])} unique trusted fields; {len(e["duplicates"])} duplicate pair; numerical ledger untouched.')
