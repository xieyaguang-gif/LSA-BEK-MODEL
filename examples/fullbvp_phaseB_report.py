#!/usr/bin/env python3
"""Reproducible Phase B diagnostics, figures, and honest terminal report."""
import os
for key in ('OPENBLAS_NUM_THREADS','OMP_NUM_THREADS','MKL_NUM_THREADS','NUMEXPR_NUM_THREADS'):os.environ[key]='1'
os.environ.setdefault('MPLCONFIGDIR','/tmp/phaseB_matplotlib')
import hashlib,json,subprocess
from pathlib import Path
import numpy as np
from scipy.interpolate import BarycentricInterpolator, PchipInterpolator, interp1d
from scipy.optimize import brentq
from fullbvp_phaseB_solver import ROOT,write,common
from llsa.secondary.global_radial import radial_differences

def crossings(r,f,mask):
    out=[]
    for i in range(len(r)-1):
        if mask[i] and mask[i+1] and f[i]*f[i+1]<0:
            out.append(dict(R=float(r[i]-f[i]*(r[i+1]-r[i])/(f[i+1]-f[i])),direction='down' if f[i]>0 else 'up',bracket=[float(r[i]),float(r[i+1])]))
    return out

def integrate_from(r,f,a):
    x=np.r_[a,r[r>a]];y=np.r_[np.interp(a,r,f),f[r>a]];return float(np.trapezoid(y,x))

def diagnostic(point,out):
    source=ROOT/point['fields'];d=np.load(source);r=d['radii'];H=int(d['H']);m=len(d['xi']);q=d['field'].reshape(len(r),2*H+1,4,m)
    transform=lambda x:.6*x+.4*(x**3+.5*(1-x*x));xi=brentq(lambda x:transform(x)-(1.3-2)/(1.3+2),-1,1)
    probe=BarycentricInterpolator(d['xi'],q,axis=-1)(xi);v=probe[:,H+1:,1]
    rms=np.sqrt(2*np.sum(abs(v)**2,axis=1));fund=abs(v[:,0]);D,_=radial_differences(r)
    gq=D@np.log(np.maximum(fund,1e-250));grms=D@np.log(np.maximum(rms,1e-250))
    a=float(r[-1]-100);valid=(r>=550)&(r<=min(650,a))&(fund>fund.max()*1e-8)
    curvepath=ROOT/f'results/fullbvp_global_20261030/phaseA/A3/curve_N{int(d["n"])}_global.json';local=json.loads(curvepath.read_text());lr=np.array([x['r'] for x in local]);lg=np.array([x['growth'] for x in local]);valid&=(r>=lr.min())&(r<=lr.max())
    localg=PchipInterpolator(lr,lg,extrapolate=False)(r)
    xq=crossings(r,gq-localg,valid);xr=crossings(r,grms-localg,valid)
    choose=lambda xs:next((x['R'] for x in xs if x['direction']=='down'),None)
    threshold=1e-6;x2=crossings(r,abs(v[:,1])-threshold,(r>420)&(r<a)) if H>=2 else []
    xratio=crossings(r,abs(v[:,1])/np.maximum(fund,1e-250)-.05,(r>500)&(r<a)&(fund>fund.max()*1e-8)) if H>=2 else []
    density=np.einsum('rhvi,ij,rhvj->rh',q[:,H:,:3].conj(),d['mass'],q[:,H:,:3],optimize=True).real
    e=np.trapezoid(density,r,axis=0);physical=np.trapezoid(density*r[:,None]**3,r,axis=0)
    coherent=density[:,1:].sum(axis=1);weights=np.r_[1,np.full(H,2)];shares=e*weights/(e@weights)
    region=(r>=570)&(r<=600);gfit=float(np.polyfit(r[region],np.log(np.maximum(fund[region],1e-250)),1)[0])
    gend=float((np.interp(600,r,np.log(np.maximum(fund,1e-250)))-np.interp(570,r,np.log(np.maximum(fund,1e-250))))/30)
    grend=float((np.interp(600,r,np.log(np.maximum(rms,1e-250)))-np.interp(570,r,np.log(np.maximum(rms,1e-250))))/30)
    # Parseval check uses an independent physical sector reconstruction.
    theta=2*np.pi*np.arange(2*H+3)/(2*H+3);realv=2*np.real(np.einsum('rj,jt->rt',v,np.exp(1j*np.arange(1,H+1)[:,None]*theta)))
    parseval=float(np.max(abs(np.sqrt(np.mean(realv**2,axis=1))-rms))/max(rms.max(),1e-30))
    result=dict(**point,front_fig8=choose(xr),front_fundamental=choose(xq),front_status='downward_crossing_found' if choose(xr) is not None else 'expected_downward_crossing_absent',front_all_crossings=xr,front_fundamental_crossings=xq,front_definition='z=1.3 d ln(v_rms/(R Omega))/dR = local global-operator absolute-pinch -Im(alpha0); most upstream downward crossing before outer sponge; no upward-crossing fallback',front_2beta_absolute=next((x['R'] for x in x2 if x['direction']=='up'),None),threshold_2beta=threshold,threshold_2beta_units='complex Fourier q_v,2 normalized by R*Omega, not height-integrated energy; convention need not equal paper spectrum',front_2beta_5percent=next((x['R'] for x in xratio if x['direction']=='up'),None),g_q_570_600=gend,g_q_least_squares_570_600=gfit,g_rms_570_600=grend,harmonic_energy_shares_with_conjugates=shares.tolist(),coherent_energy_shares=(e[1:]/e[1:].sum()).tolist(),sponge_fraction_q=integrate_from(r,coherent,a)/float(np.trapezoid(coherent,r)),sponge_fraction_cylindrical_R3=integrate_from(r,coherent*r**3,a)/float(np.trapezoid(coherent*r**3,r)),omega_minus_52_36=point['omega']-52.36,vrms_max=float(rms.max()),parseval_relative_error=parseval,sha256=hashlib.sha256(source.read_bytes()).hexdigest(),local_pinch_source=str(curvepath.relative_to(ROOT)),local_pinch_sha256=hashlib.sha256(curvepath.read_bytes()).hexdigest())
    target=out/'diagnostics';target.mkdir(exist_ok=True)
    np.savez_compressed(target/(point['id']+'.npz'),radii=r,probe_v=v,vrms=rms,gq=gq,grms=grms,local_growth=localg,energy_density=density)
    write(target/(point['id']+'.json'),result);return result

def compare_fields(base,ref):
    b=np.load(ROOT/base['fields']);c=np.load(ROOT/ref['fields']);r=b['radii'];mask=r<=c['radii'][-1];r=r[mask];H1=int(b['H']);H2=int(c['H']);m1=len(b['xi']);m2=len(c['xi']);h=min(H1,H2)
    q=b['field'][mask].reshape(len(r),2*H1+1,4,m1)[:,H1:H1+h+1,:3]
    v=c['field'].reshape(len(c['radii']),2*H2+1,4,m2)[:,H2:H2+h+1,:3]
    if m1!=m2:v=BarycentricInterpolator(c['xi'],v,axis=-1)(b['xi'])
    v=interp1d(c['radii'],v,axis=0,kind='cubic')(r);rw=common.radial_weights(r)
    inner=lambda u,v:np.einsum('rvi,ij,rvj,r->',u.conj(),b['mass'],v,rw,optimize=True)
    phase=np.angle(inner(q[:,1],v[:,1]));v*=np.exp(-1j*np.arange(h+1)*phase)[None,:,None,None]
    return dict(common_physical_domain=[float(r[0]),float(r[-1])],q1_relative_mass=float(np.sqrt(inner(v[:,1]-q[:,1],v[:,1]-q[:,1]).real/inner(q[:,1],q[:,1]).real)),mean_relative_mass=float(np.sqrt(inner(v[:,0]-q[:,0],v[:,0]-q[:,0]).real/max(inner(q[:,0],q[:,0]).real,1e-30))),phase_alignment=float(phase))

def render(out,summary):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    plt.rcParams.update({'font.family':'serif','font.serif':['Times New Roman','Nimbus Roman','STIXGeneral','DejaVu Serif'],'mathtext.fontset':'stix','font.size':10,'axes.linewidth':.9,'lines.linewidth':1.4,'savefig.dpi':220})
    figs=out/'figures';figs.mkdir(exist_ok=True);points=summary['points'];primary=[p for p in points if p['role'] in ['weak_seed','primary_arclength']]
    def finish(fig,name):
        fig.savefig(figs/(name+'.png'));fig.savefig(figs/(name+'.pdf'));plt.close(fig)
        subprocess.run(['pdftoppm','-png','-singlefile','-scale-to','1800',str(figs/(name+'.pdf')),str(figs/(name+'_pdf_render'))],check=True,stdout=subprocess.DEVNULL,stderr=subprocess.DEVNULL)
    fig,ax=plt.subplots(1,2,figsize=(7.4,3.4),layout='constrained')
    for axis,key,ylabel in [(ax[0],'Rout',r'$R_{\rm out}$'),(ax[1],'omega',r'$\omega$')]:
        axis.plot([p['A1'] for p in primary],[p[key] for p in primary],'-o',ms=3,color='#155b98',label='H3 arclength')
        for H,color in [(3,'#232323'),(4,'#a52b2b')]:
            pp=[p for p in points if p['role']=='fold_refinement' and p['H']==H]
            if pp:axis.plot([p['A1'] for p in pp],[p[key] for p in pp],'s',ms=3,color=color,label=f'H{H} fold samples')
        axis.set(xlabel=r'$A_1=\sqrt{E_1}$',ylabel=ylabel);axis.tick_params(direction='in',top=True,right=True)
    ax[1].axhline(52.36,color='.5',ls=':',label='DNS 52.36');ax[0].legend(fontsize=8);ax[1].legend(fontsize=8)
    fig.suptitle('Moving-domain rotating waves: accepted roots',fontsize=11);finish(fig,'branch_Rout_A1_omega')
    upper=[p for p in primary if p.get('upper_branch')];selected=[upper[-1]] if upper else (primary[-1:] if primary else [])
    selected+=[p for p in points if p['role']=='upper_target']
    fig,ax=plt.subplots(figsize=(7.4,4.1),layout='constrained')
    for p in selected:
        d=np.load(out/'diagnostics'/(p['id']+'.npz'));r=d['radii'];rms=d['vrms'];mask=r<=p['a'];line,=ax.semilogy(r[mask],rms[mask],label=f'BVP H{p["H"]}, Rout={p["Rout"]:.2f}, A1={p["A1"]:.3f}')
        ax.semilogy(r[~mask],rms[~mask],ls='--',color=line.get_color(),alpha=.6)
    dns=np.genfromtxt(ROOT/'results/appelquist2016_branch_20261029/literature/fig15_nl03_T6_vrms.csv',delimiter=',',names=True)
    ax.semilogy(dns['R'],dns['vrms_normalized'],color='#a52b2b',label='NL03 DNS, T=6 (digitized)');ax.fill_between(dns['R'],dns['vrms_plot_lower'],dns['vrms_plot_upper'],color='#a52b2b',alpha=.14)
    nl01=out/'reference_data/nl01_T6_vrms.csv'
    if nl01.exists():
        dn=np.genfromtxt(nl01,delimiter=',',names=True)
        ax.errorbar(dn['R'],dn['vrms_normalized'],xerr=dn['R_plot_reading_tolerance'],yerr=np.stack([dn['vrms_normalized']-dn['vrms_plot_lower'],dn['vrms_plot_upper']-dn['vrms_normalized']]),fmt='s',ms=3,capsize=2,color='#387c44',elinewidth=.8,label='NL01 T=6, Fig.6(b) contour levels')
    ymax=max(.2,1.2*float(np.max(dns['vrms_plot_upper'])),1.2*max([p['vrms_max'] for p in selected],default=0))
    ax.set(xlabel=r'$R$',ylabel=r'$v_{\rm rms}/(R\Omega)$ at $z=1.3$',xlim=(515,max(730,max([p['Rout'] for p in selected],default=730))),ylim=(2e-8,ymax));ax.legend(fontsize=7.6,loc='lower right');ax.text(.02,.97,('NL01: graphical envelopes, sparse contours' if nl01.exists() else 'NL01 digitized curve unavailable')+'\nDashed: outer sponge',transform=ax.transAxes,va='top',fontsize=8)
    missing=[str(int(float(k))) for k,v in summary['targets'].items() if v['status']!='accepted']
    if missing:ax.set_title('Upper target roots unavailable: Rout = '+', '.join(missing),fontsize=10)
    finish(fig,'vrms_DNS')
    fig,ax=plt.subplots(1,2,figsize=(7.4,3.6),layout='constrained')
    pp=[p for p in upper if p['front_fig8'] is not None]
    if pp:ax[0].plot([p['Rout'] for p in pp],[p['front_fig8'] for p in pp],'-o',ms=3,label='Fig.8 growth crossing')
    ax[0].axhline(583,color='#a52b2b',ls=':',label='2016 extrapolation ~583');ax[0].set(xlabel=r'$R_{\rm out}$',ylabel=r'$r_{\rm end}$');ax[0].legend(fontsize=8)
    if upper:ax[1].plot([p['Rout'] for p in upper],[p['omega'] for p in upper],'-o',ms=3,label='Upper H3')
    ax[1].axhline(52.36,color='#a52b2b',ls=':',label='DNS 52.36');ax[1].set(xlabel=r'$R_{\rm out}$',ylabel=r'$\omega$');ax[1].legend(fontsize=8)
    for axis,available in [(ax[0],bool(pp)),(ax[1],bool(upper))]:
        if not available:axis.set_xlim(700,800);axis.text(.5,.4,'Upper-branch data unavailable',transform=axis.transAxes,ha='center',fontsize=9)
    fig.suptitle('Domain convergence unestablished' if not summary['domain_convergence']['established'] else 'Upper branch domain diagnostics',fontsize=11);finish(fig,'front_Rout')
    write(figs/'visual_audit.json',dict(status='pending human inspection',figures=['branch_Rout_A1_omega','vrms_DNS','front_Rout'],formats=['PNG','PDF'],pdf_rasters=True,physical_sizes_inches=[[7.4,3.4],[7.4,4.1],[7.4,3.6]],normalization='complex q_v coefficients already normalized by local disk speed; mean excluded',missing='unavailable target data explicitly labelled; no invented curves',font_fallback='Times New Roman, Nimbus Roman, STIXGeneral, DejaVu Serif'))

def finalize(out):
    out=Path(out);campaign=json.loads((out/'campaign.json').read_text());points=[diagnostic(p,out) for p in campaign['points']]
    upper=[p for p in points if p['role']=='primary_arclength' and p.get('upper_branch')];last=upper[-1] if upper else None
    refinements=[]
    for c in campaign.get('refinements',[]):
        row=dict(c);base=next((p for p in points if p['id']==c['source']),None);ref=next((p for p in points if p['id']==c['point']),None)
        if base and ref:
            row.update(compare_fields(base,ref));row['delta_front']=None if base['front_fig8'] is None or ref['front_fig8'] is None else ref['front_fig8']-base['front_fig8'];row['delta_gq']=ref['g_q_570_600']-base['g_q_570_600'];row['delta_sponge_R3']=ref['sponge_fraction_cylindrical_R3']-base['sponge_fraction_cylindrical_R3']
        refinements.append(row)
    targets=campaign.get('targets',{});good=[p for p in points if p['role']=='upper_target'];convergence=dict(established=False,status='insufficient_reachable_outlet_range' if len(good)<3 else 'finite_sequence_requires_assessment',target_count=len(good),omega_range=None,front_range=None)
    convergence['consecutive_differences']=[]
    if len(good)>=2:
        good.sort(key=lambda p:p['Rout'])
        convergence['omega_range']=float(np.ptp([p['omega'] for p in good]));fronts=[p['front_fig8'] for p in good if p['front_fig8'] is not None];convergence['front_range']=float(np.ptp(fronts)) if len(fronts)>1 else None
        for a,b in zip(good[:-1],good[1:]):convergence['consecutive_differences'].append(dict(Rout_pair=[a['Rout'],b['Rout']],delta_omega=b['omega']-a['omega'],delta_front=None if a['front_fig8'] is None or b['front_fig8'] is None else b['front_fig8']-a['front_fig8']))
    if len(good)>=3:
        dw=[abs(x['delta_omega']) for x in convergence['consecutive_differences']];df=[abs(x['delta_front']) for x in convergence['consecutive_differences'] if x['delta_front'] is not None]
        convergence['trend']=dict(omega_increments_decreasing=bool(np.all(np.diff(dw)<0)),front_increments_decreasing=bool(len(df)>=2 and np.all(np.diff(df)<0)),interpretation='Finite sequence only; decreasing increments do not alone establish a domain-independent limit.')
    else:convergence['trend']=dict(status='insufficient_points_for_increment_trend')
    summary=dict(status=campaign['status'],scientific_stop=campaign['stop'],folds=campaign.get('folds',{}),upper_last=last,domain_convergence=convergence,targets=targets,refinements=refinements,points=points,failed_attempts=[p for p in campaign['attempts'] if not p['accepted']],temporal_stability=None,temporal_stability_status='not_computed',mapping_checks=json.loads((out/'mapping_checks.json').read_text()),run_policy=json.loads((out/'run_policy.json').read_text()),DNS=dict(NL03='results/appelquist2016_branch_20261029/literature/fig15_nl03_T6_vrms.csv',NL01=None,NL01_status='digitized_curve_not_present'),front_uncertainty='radial crossing bracket plus N/H/dr differences reported separately; no fabricated asymptotic limit',harmonic_stop=campaign.get('harmonic_stop'),startup=dict(old_grid_neutral=719.9581594895074,new_intervals=800,weak_roots=[dict(A1=p['A1'],Rout=p['Rout'],omega=p['omega']) for p in points if p['role']=='weak_seed']),provenance=dict(solver='examples/fullbvp_phaseB_solver.py',driver='examples/fullbvp_phaseB_campaign.py',postprocessor='examples/fullbvp_phaseB_report.py',prior_results_read_only=True))
    nl01=out/'reference_data/nl01_T6_vrms.csv'
    if nl01.exists():summary['DNS'].update(NL01=str(nl01),NL01_status='six_contour_level_samples_from_Fig6b_T6',NL01_uncertainty='graphical comparison envelope: R±2.5; log10 RMS±0.5; not statistical confidence',NL01_provenance=str(out/'reference_data/nl01_T6_provenance.json'))
    write(out/'summary.json',summary);render(out,summary)
    fold=summary['folds'].get('H3',summary['folds'].get('H3_coarse',{}));f4=summary['folds'].get('H4',{})
    fnum=lambda x:'未计算' if x is None else f'{x:.8g}'
    lines=['# Full-BVP Phase B：移动出口非线性旋转波','',f'数值状态：{campaign["status"]}；科学停止原因：{campaign["stop"]}。',f'H3 折点估计：Rout={fnum(fold.get("Rout"))}，A1={fnum(fold.get("A1"))}；H4：Rout={fnum(f4.get("Rout"))}，A1={fnum(f4.get("A1"))}。',f'H4−H3 折点出口差：{fnum(summary["folds"].get("H4_minus_H3_Rout"))}。折点来自局部五点二次拟合，原始左右样点和切向量均保留。','']
    if last:
        lines += [f'最后可接受上支：Rout={last["Rout"]:.8f}，A1={last["A1"]:.7g}，omega={last["omega"]:.8f}（相对 52.36 差 {last["omega_minus_52_36"]:+.6g}）。',f'Fig.8 交点前沿={fnum(last["front_fig8"])}；2beta 复谱系数达到 1e-6 的位置={fnum(last["front_2beta_absolute"])}。',f'R570–600 的 g_q={last["g_q_570_600"]:.7g}；总 RMS 对数增长率={last["g_rms_570_600"]:.7g}。DNS 图读约 0.106。',f'外海绵能量比例：归一化 q 范数 {100*last["sponge_fraction_q"]:.3f}%；圆柱物理动能 R³ 权重 {100*last["sponge_fraction_cylindrical_R3"]:.3f}%。','']
    else:lines+=['尚未获得可接受上支点；上支频率、前沿及域收敛均未计算。','']
    lines += ['出口域收敛尚未建立。2016 年 R_cg≈583 是有限时间 DNS 趋势的外推，omega≈52.36 是相干谱峰；本任务的同频旋转波不包含 DNS 的湍流和瞬态。分支存在不能证明时间稳定性；未计算 Floquet。','','## 方程、移动网格与验收','Ro=−1，Rin=400，beta=68；未知量为均值修正、±1…±H 谐波、实 omega 与 Rout。入口海绵 0.8 S((420−R)/10)，出口海绵 28 S((R−(Rout−100))/40)，均值修正不受海绵阻尼。没有辅助体积力或中性缺陷抵消。','R=400+(Rout−400)xi，主支固定 800 个区间，Rout=800 时 dr=0.5。D_R=D_xi/L，D_RR=D_xixi/L²；Rout 导数同时包含局部系数、两端边界、两段海绵、非线性径向输运及种子能量约束的物理积分测度。','两点弱振幅能量校正后，以加权割线预测和伪弧长超平面校正延拓。固定度量取 Rref=720，速度 q 范数尺度 0.1，omega 尺度 1，Rout 尺度 10；压力不计入长度范数。每点保存完整场、预测切向量、Newton/Krylov 原始收敛证据，最新割线存 continuation_state.npz。','N32/dr≈0.5 参考中性 Rout=719.95815949；主支网格改为 800 区间后重新校正弱根，不能要求其复现细网格名义 720.46986。','接受条件：原始无强迫残差相对最终场无穷范数 <1e−7，残差相对基频非线性反馈 <1e−3，相位及分支约束 <1e−8；独立物理网格重建复核。最高谐波能量/基频>1e−2 的收敛根只归档而不收入可接受点清单。','','## 诊断与不确定度','A1=√E1，E1=∫q1* M_z q1 dR。该分支坐标是归一化速度的范数。全场谐波份额用 E0+2ΣEj；物理圆柱动能以 R³dRdz 为测度。','在 z=1.3 对复系数作映射 Chebyshev 插值；v_rms/(R Omega)=√(2Σ|q_v,j|²)，去除均值且不再除以 R。Parseval 重构误差见各点 JSON。','主前沿使用 d ln|q_v,1|/dR 与 Phase A 同一局部绝对 pinch −Im(alpha0) 的交点；总 RMS 定义另存供比较。保留海绵上游所有交点及方向，优先最上游向下交点。2beta 次诊断阈值为复系数 1e−6，另存相对基频 5% 位置；该谱归一化不冒称与论文完全相同。','NL03 使用已存图15数化曲线及图读包络。现有 results 中未找到 NL01 数化曲线，图内明确标为不可用。','网格比较在相同物理 E1 校正，分别改变 N、径向步长、H；场差在公共物理半径区间相位对齐后计算。频率、前沿和域出口的变化见下表及 summary.json。','', '| 检查 | 状态 | ΔRout | Δomega | Δfront |','|---|---|---:|---:|---:|']
    for r in refinements:lines.append(f'| {r["label"]} | {r["status"]} | {fnum(r.get("delta_Rout"))} | {fnum(r.get("delta_omega"))} | {fnum(r.get("delta_front"))} |')
    lines+=['','## 数据与复现','`accepted_points.json` 为机器可读验收清单，`attempts/*/fields.npz` 为接受根重启场，失败尝试保存 checkpoint.npz、summary.json 和 progress.json。所有原有结果保持只读。', '`commands.sh` 给出完整命令；中断后同一命令按 campaign.json 阶段恢复。终止标记为 DONE.json 或 FAILED.json；DONE 表示按预定科学停止条件完成本次流程，不表示达到了 Rout=800 或域收敛。','`mapping_checks.json` 给出原方程与完整移动导数校验；`figures/visual_audit.json` 记录 PNG/PDF 人工审阅状态，自动出图后仍需审图。','局部来源：docs/APPELQUIST_2016_NONLINEAR_COMPARISON.md；Phase A/A3/curve_N*_global.json；旧 continuation 弱振幅场。每个接受场及局部曲线的 SHA256 已存入 diagnostics/*.json。']
    lines+=['','## 上支目标与渐近比较','| Rout 目标 | 状态 | omega | Fig.8 前沿 | g_q(570–600) | R³ 海绵比例 |','|---|---|---:|---:|---:|---:|']
    for key,value in sorted(targets.items(),key=lambda kv:float(kv[0])):
        point=next((p for p in good if abs(p['Rout']-float(key))<1e-5),None)
        lines.append(f'| {key} | {value["status"]} | {fnum(None if point is None else point["omega"])} | {fnum(None if point is None else point["front_fig8"])} | {fnum(None if point is None else point["g_q_570_600"])} | {fnum(None if point is None else point["sponge_fraction_cylindrical_R3"])} |')
    if not targets:lines.append('| — | 尚无目标点 | 未计算 | 未计算 | 未计算 | 未计算 |')
    lines+=['','相邻目标差值：','| Rout 区间 | Δomega | Δfront |','|---|---:|---:|']
    for row in convergence['consecutive_differences']:lines.append(f'| {row["Rout_pair"][0]:g} → {row["Rout_pair"][1]:g} | {fnum(row["delta_omega"])} | {fnum(row["delta_front"])} |')
    if not convergence['consecutive_differences']:lines.append('| 无可比相邻目标 | 未计算 | 未计算 |')
    lines+=['','没有足够的出口序列时不作趋势判定；即使相邻差值减小，也不能仅凭这一点宣称存在已收敛的无穷域极限。']
    if last:lines+=[f'最后接受上支 g_q 相对 Phase A 线性参考 0.132 的差为 {last["g_q_570_600"]-.132:+.7g}，相对 DNS 图读约 0.106 的差为 {last["g_q_570_600"]-.106:+.7g}；这两个参考代表不同流动状态。']
    lines+=['','折点插值敏感性（3 点结果减 5 点结果，不等同总离散误差）：','| H | ΔRout | ΔA1 | 5 点拟合残差 RMS |','|---|---:|---:|---:|']
    for H in [3,4]:
        f=summary['folds'].get(f'H{H}',{});lines.append(f'| {H} | {fnum(f.get("three_vs_five_Rout"))} | {fnum(f.get("three_vs_five_A1"))} | {fnum(f.get("fit_rms"))} |')
    text='\n'.join(lines)+'\n'
    text=text.replace('主前沿使用 d ln|q_v,1|/dR 与 Phase A 同一局部绝对 pinch −Im(alpha0) 的交点；总 RMS 定义另存供比较。保留海绵上游所有交点及方向，优先最上游向下交点。','Fig.8 主前沿使用总归一化方位 RMS 的对数径向增长率与 Phase A 同一局部绝对 pinch −Im(alpha0) 的交点；基频 q_v,1 定义另存供比较。保留海绵上游所有交点及方向，仅选最上游向下交点，若不存在则标为未计算而不改选向上交点。')
    if nl01.exists():text=text.replace('现有 results 中未找到 NL01 数化曲线，图内明确标为不可用。','NL01 从原论文 Fig.6(b) 的 T=6 端点恢复六个等值线交点（log10 RMS 为 −7…−2）；仅画稀疏点和图读包络 R±2.5、±0.5 decade，不连接为虚构连续曲线。详见 reference_data/nl01_T6_provenance.json。')
    (out/'REPORT.md').write_text(text)

if __name__=='__main__':
    import argparse
    p=argparse.ArgumentParser();p.add_argument('--out',default='results/fullbvp_global_20261030/phaseB');a=p.parse_args();finalize(a.out)
