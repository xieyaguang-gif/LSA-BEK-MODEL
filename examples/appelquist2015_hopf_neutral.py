#!/usr/bin/env python3
"""Track a fixed-domain global eigenbranch to a sponge-start Hopf neutral point."""
from __future__ import annotations
import argparse
import json
from pathlib import Path
import signal
import time
import numpy as np
from scipy import sparse
from scipy.optimize import brentq
from scipy.interpolate import BarycentricInterpolator,interp1d
from scipy.sparse.linalg import LinearOperator,eigs,splu
from appelquist2015_hopf_common import (ROOT,HopfSponge,make_problem,complex_block,
    sponge_profile_and_derivative,sponge_start_matrix,mass_norm,mass_correlation,
    radial_weights,case_name)
from llsa.secondary.radial import pack_field,unpack_field

OUT=ROOT/'results/appelquist2015_hopf_20261029/neutral'
OLD=ROOT/'results/appelquist2015_tests_20261029/linear'
REF=50.5


def dump(path,data):
    path.write_text(json.dumps(data,indent=2,allow_nan=False)+'\n')


def checks():
    p=make_problem(8,10.,620.,harmonics=2,nonlinear=True)
    L=p._linear_matrix(False)
    records={}
    for j in (0,1,2):
        A,B=complex_block(p,j)
        offset=0 if j==0 else (2*j-1)*p.s
        ix=np.concatenate([i*p.width+offset+np.arange(p.s) for i in range(len(p.radii))])
        extracted=L[ix][:,ix] if j==0 else L[ix][:,ix]+1j*L[ix+p.s][:,ix]
        records[f'j{j}_real_extraction_max']=float(abs(A-extracted).max())
        pplus=make_problem(8,10.,620.,harmonics=2,omega=REF+.125)
        Af,_=complex_block(pplus,j)
        records[f'j{j}_frequency_derivative_max']=float(abs((Af-A)/.125+1j*j*B).max())
        h=1e-3
        Ap,_=complex_block(make_problem(8,10.,620.+h,harmonics=2),j)
        Am,_=complex_block(make_problem(8,10.,620.-h,harmonics=2),j)
        Da=sponge_start_matrix(p,j)
        records[f'j{j}_sponge_derivative_max']=float(abs((Ap-Am)/(2*h)-Da).max())
    for damp in (False,True):
        q=make_problem(8,10.,620.,harmonics=2,damp_mean=damp)
        h=1e-3
        Ap,_=complex_block(make_problem(8,10.,620.+h,harmonics=2,damp_mean=damp),0)
        Am,_=complex_block(make_problem(8,10.,620.-h,harmonics=2,damp_mean=damp),0)
        records[f'mean_damping_{damp}_derivative_max']=float(abs((Ap-Am)/(2*h)-sponge_start_matrix(q,0)).max())
    records['nonlinear_parameter_passed']=p.config.nonlinear is True
    assert records['nonlinear_parameter_passed']
    assert max(v for k,v in records.items() if not isinstance(v,bool))<1e-9
    dump(OUT/'common_api_checks.json',records)
    print('CHECKS '+json.dumps(records),flush=True)


class Branch:
    def __init__(self,n,dr,path,nearby_seed=False):
        self.n,self.dr,self.path=n,dr,path
        self.p=make_problem(n,dr,630.)
        self.A630,self.B=complex_block(self.p,1)
        self.original_profile=self.p.sponge_strength.copy()
        old_name=f'paper_R730_N{n}_dr{dr:g}'
        records=json.loads((OLD/old_name/'modes.json').read_text())
        record=max([r for r in records if r['accepted'] and 52<r['omega_lab']<54],key=lambda r:r['sigma'])
        self.seed=np.load(OLD/old_name/'eigenpairs.npz')['modes'][record['mode']].ravel()
        self.seed_lambda=complex(-record['sigma'],record['omega_lab']-REF)
        self.seed_a=630.;self.seed_source='previous accepted a630 eigenmode on this same grid'
        if nearby_seed:
            source=OUT/case_name(n,2*dr)/'neutral_state.npz'
            previous=np.load(source)
            if int(previous['n'])!=n:raise ValueError('Nearby radial seed must have the same N')
            self.seed=interp1d(previous['radii'],previous['q1'],axis=0,kind='cubic')(self.p.radii).ravel()
            self.seed_lambda=complex(-float(previous['sigma_residual']),float(previous['omega_c'])-REF)
            self.seed_a=float(previous['a_c'])
            self.seed_source=f'{source}: interpolation is an initial guess only; all fine-grid eigenvalues/root conditions solved independently'
        self.cache={};self.history=[]

    def matrix(self,a):
        self.p.outlet_sponge=HopfSponge(float(a),40.,28.)
        self.p.sponge_strength=self.p.outlet_sponge.profile(self.p.radii)
        diag=np.zeros((len(self.p.radii),self.p.s))
        diag[1:-1,self.p.velocity_rows]=(self.p.sponge_strength[1:-1]-self.original_profile[1:-1])[:,None]/self.p.radii[1:-1,None]
        return (self.A630+sparse.diags(diag.ravel(),format='csc')).tocsc()

    def solve(self,a,repeat=False):
        key=round(float(a),11)
        if not repeat and key in self.cache:return self.cache[key]
        started=time.perf_counter();A=self.matrix(a)
        if self.cache:
            near=min(self.cache,key=lambda x:abs(x-a));target=self.cache[near]['lambda'];reference=self.cache[near]['q']
        else:target=self.seed_lambda;reference=self.seed
        shift=target+(.23-.19j if repeat else .07+.09j)
        lu=splu(A-shift*self.B)
        K=LinearOperator(A.shape,matvec=lambda v:lu.solve(self.B@v),dtype=complex)
        rng=np.random.default_rng(2026 if repeat else 2015)
        mu,vectors=eigs(K,k=3,which='LM',tol=5e-12,maxiter=1000,ncv=20,
            v0=rng.normal(size=A.shape[0])+1j*rng.normal(size=A.shape[0]))
        values=shift+1/mu
        candidates=[]
        for i,value in enumerate(values):
            corr=mass_correlation(self.p,reference,vectors[:,i])
            candidates.append((abs(value-target)+2*(1-corr),i,corr))
        _,ix,corr=min(candidates)
        lam=values[ix];q=vectors[:,ix]
        q/=mass_norm(self.p,q)
        v=q.reshape(len(self.p.radii),4,self.p.ops[0].m)[:,:3]
        phase=np.angle(v.ravel()[np.argmax(abs(v))]);q*=np.exp(-1j*phase)
        residual=float(np.linalg.norm(A@q-lam*(self.B@q))/(np.linalg.norm(A@q)+abs(lam)*np.linalg.norm(self.B@q)))
        if corr<.8 or abs(lam-target)>1.5 or residual>1e-7:
            raise RuntimeError(f'Branch tracking failed: a={a},corr={corr},lambda={lam},residual={residual}')
        result=dict(a=float(a),**{'lambda':lam},q=q,correlation=corr,eigen_residual=residual,
            omega=float(REF+lam.imag),sigma=float(-lam.real),seconds=time.perf_counter()-started)
        record={k:v for k,v in result.items() if k not in ('q','lambda')}
        record.update(repeat=repeat,shift_real=float(shift.real),shift_imag=float(shift.imag))
        self.history.append(record)
        dump(self.path/'tracking_history.json',self.history)
        print('EVAL '+json.dumps(record),flush=True)
        if not repeat:self.cache[key]=result
        return result

    def neutral(self):
        started=time.perf_counter()
        if self.seed_a!=630.:
            upper=self.solve(self.seed_a+.1);lower=self.solve(self.seed_a-.1)
        else:
            upper=self.solve(630.)
            for value in (625.,620.,615.):
                lower=self.solve(value)
                if lower['sigma']*upper['sigma']<0:break
                upper=lower
        if lower['sigma']*upper['sigma']>=0:raise RuntimeError('Actual sigma(a) path [615,630] did not straddle zero')
        ac=brentq(lambda a:self.solve(a)['sigma'],lower['a'],upper['a'],xtol=1e-9,rtol=1e-12)
        neutral=self.solve(ac);repeat=self.solve(ac,repeat=True)
        q,lam=neutral['q'],neutral['lambda'];omega=neutral['omega']
        A=self.matrix(ac);p=self.p;B=self.B
        # Adjoint generalized pencil, independent starting vector and shift.
        shift=lam+.127+.173j;lu=splu(A-shift*B)
        K=LinearOperator(A.shape,matvec=lambda v:lu.solve(B@v,trans='H'),dtype=complex)
        rng=np.random.default_rng(2026)
        mu,left=eigs(K,k=3,which='LM',tol=1e-12,maxiter=1000,ncv=20,
            v0=rng.normal(size=A.shape[0])+1j*rng.normal(size=A.shape[0]))
        values=(shift.conjugate()+1/mu).conj();ix=np.argmin(abs(values-lam));w=left[:,ix]
        w/=np.vdot(w,B@q).conjugate()
        projected=-1j*np.vdot(w,sponge_start_matrix(p,1)@q)
        h=.05;plus=self.solve(ac+h);minus=self.solve(ac-h)
        finite=(complex(plus['omega'],plus['sigma'])-complex(minus['omega'],minus['sigma']))/(2*h)
        p=self.p;self.matrix(ac)
        full=np.zeros((len(p.radii),3,p.s),complex);full[:,2]=q.reshape(len(p.radii),p.s);full[:,0]=full[:,2].conj()
        independent=make_problem(self.n,self.dr,ac,omega=omega)
        ref=unpack_field(independent.evaluate_reference(pack_field(full),np.zeros((3,p.s),complex)),len(p.radii),1,p.s)[:,2].ravel()
        scale=max(abs(q).max(),1e-30)
        m=p.ops[0].m;qq=q.reshape(len(p.radii),p.s)
        bc=max(abs(qq[:,np.r_[0,m-1,m,2*m-1,2*m,3*m-1]]).max(),abs(qq[0,p.velocity_rows]).max(),abs((p.D@qq)[-1,p.velocity_rows]).max())/max(abs(qq[:,:3*m]).max(),1e-30)
        component=(abs(w)@(abs(A)@abs(q))+abs(lam)*abs(w)@(abs(B)@abs(q)))/abs(np.vdot(w,B@q))
        overlap=np.vdot(w,B@q)
        report=dict(n=self.n,dr=self.dr,Rin=400.,Rout=730.,harmonics=1,damp_mean=False,
            outer_start=float(ac),a_c=float(ac),omega_c=float(omega),omega_disk=float(omega-68),
            sigma_residual=neutral['sigma'],eigen_residual=neutral['eigen_residual'],
            original_equation_residual_inf=float(abs(ref).max()),original_equation_residual_scaled=float(abs(ref).max()/scale),
            velocity_bc_relative=float(bc),q1_mass_norm=mass_norm(p,q),
            left_mass_overlap=[float(overlap.real),float(overlap.imag)],
            left_eigenvalue_difference=float(abs(values[ix]-lam)),
            left_residual=float(np.linalg.norm(A.getH()@w-lam.conjugate()*(B@w))/(np.linalg.norm(A.getH()@w)+abs(lam)*np.linalg.norm(B@w))),
            repeat_eigenvalue_difference=float(abs(repeat['lambda']-lam)),repeat_mass_correlation=mass_correlation(p,q,repeat['q']),
            dOmega_da_projection=[float(projected.real),float(projected.imag)],
            dOmega_da_formula='-i * p^H (A_a q1) / (p^H B1 q1); B1 independent of a on the fixed domain',
            transversality_dsigma_da=float(projected.imag),nonlinear=False,
            dOmega_da_central_difference=[float(finite.real),float(finite.imag)],derivative_fd_step=h,
            derivative_relative_error=float(abs(projected-finite)/abs(finite)),
            eigenvalue_condition=float(np.linalg.norm(w)*np.linalg.norm(q)/abs(overlap)),
            componentwise_condition=float(component),epsilon_times_componentwise_condition=float(np.finfo(float).eps*component),
            normalization='integral q1_velocity^H M_z q1_velocity dR=1; left_mode^H B1 q1=1; q1 largest velocity real positive',
            seed_source=self.seed_source,
            seconds=time.perf_counter()-started,tracking_history=self.history,
            note='omega_c is real; sigma_residual is the remaining neutral root defect. Fixed Rout730, varying outer_start only.')
        accepted=(abs(report['sigma_residual'])<1e-9 and report['eigen_residual']<1e-7
            and report['original_equation_residual_scaled']<1e-7 and bc<1e-7
            and abs(overlap-1)<1e-7 and report['repeat_eigenvalue_difference']<1e-6
            and report['left_residual']<1e-7 and report['left_eigenvalue_difference']<1e-6
            and report['derivative_relative_error']<1e-3)
        report['accepted']=bool(accepted)
        profile,da=sponge_profile_and_derivative(p)
        np.savez_compressed(self.path/'neutral_state.npz',q1=qq,left_mode=w.reshape(qq.shape),
            radii=p.radii,z=p.ops[0].grid.z,xi=p.ops[0].grid.xi,mass=p.ops[0].grid.mass,
            radial_weights=radial_weights(p.radii),omega_c=omega,a_c=ac,sigma_residual=neutral['sigma'],
            n=self.n,dr=self.dr,profile=profile,dprofile_da=da,B_diagonal=B.diagonal().reshape(qq.shape),
            dOmega_da_projection=projected,dOmega_da_central_difference=finite)
        dump(self.path/'summary.json',report)
        if not accepted:raise RuntimeError('Neutral quality checks failed; see summary.json')
        (self.path/'READY').write_text('Accepted matching-grid neutral state; q1 mass norm=1, p^H B q1=1.\n')
        print('NEUTRAL '+json.dumps({k:report[k] for k in ('n','dr','a_c','omega_c','sigma_residual','derivative_relative_error','seconds','accepted')}),flush=True)
        return report


def refine_derivative(n,dr):
    path=OUT/case_name(n,dr)
    state=np.load(path/'neutral_state.npz')
    ac=float(state['a_c']);omega=float(state['omega_c']);sigma=float(state['sigma_residual'])
    check_path=path/'derivative_check';check_path.mkdir(exist_ok=True)
    b=Branch(n,dr,check_path)
    q=state['q1'].ravel();left=state['left_mode'].ravel()
    b.cache[round(ac,11)]=dict(a=ac,**{'lambda':complex(-sigma,omega-REF)},q=q)
    b.matrix(ac)
    projected=-1j*np.vdot(left,sponge_start_matrix(b.p,1)@q)/np.vdot(left,b.B@q)
    h=.01;plus=b.solve(ac+h);minus=b.solve(ac-h)
    fd=(complex(plus['omega'],plus['sigma'])-complex(minus['omega'],minus['sigma']))/(2*h)
    result=dict(n=n,dr=dr,a_c=ac,step=h,projection=[float(projected.real),float(projected.imag)],
        central_difference=[float(fd.real),float(fd.imag)],relative_error=float(abs(fd-projected)/abs(fd)),
        seconds=sum(r['seconds'] for r in b.history),note='Frozen neutral q1/left_mode/a_c unchanged; only independent derivative verification refined.')
    dump(path/'derivative_refinement.json',result)
    print('DERIVATIVE '+json.dumps(result),flush=True)


def audit():
    cases=[];archives={};validations=[];total=0.
    configurations=[(32,1.),(32,.5),(40,.5)]
    if (OUT/'N32_dr0.25/READY').exists():configurations.insert(2,(32,.25))
    for n,dr in configurations:
        name=case_name(n,dr);path=OUT/name
        if not (path/'READY').exists():raise RuntimeError(f'{name} neutral state not READY')
        report=json.loads((path/'summary.json').read_text())
        with np.load(path/'neutral_state.npz') as raw:state={k:raw[k] for k in raw.files}
        archives[name]=state
        p=make_problem(n,dr,float(state['a_c']),omega=float(state['omega_c']))
        A,B=complex_block(p,1);q=state['q1'].ravel();left=state['left_mode'].ravel()
        mass=mass_norm(p,q);overlap=np.vdot(left,B@q)
        residual=float(abs(A@q).max()/max(abs(q).max(),1e-30))
        validation=dict(case=name,q1_mass_norm=mass,left_mass_overlap=[float(overlap.real),float(overlap.imag)],
            rebuilt_real_frequency_equation_scaled_inf=residual,stored_B_max_difference=float(abs(B.diagonal()-state['B_diagonal'].ravel()).max()),
            left_residual=report['left_residual'],left_eigenvalue_difference=report['left_eigenvalue_difference'])
        validation['passed']=bool(abs(mass-1)<1e-10 and abs(overlap-1)<1e-7 and residual<1e-7
            and validation['stored_B_max_difference']==0 and report['left_residual']<1e-7 and report['left_eigenvalue_difference']<1e-6)
        validations.append(validation)
        if not validation['passed']:raise RuntimeError(f'Archive verification failed: {validation}')
        refined=path/'derivative_refinement.json'
        if refined.exists():report['derivative_refinement']=json.loads(refined.read_text());total+=report['derivative_refinement']['seconds']
        report.setdefault('dOmega_da_formula','-i * p^H (A_a q1) / (p^H B1 q1); B1 independent of a on the fixed domain')
        report.setdefault('transversality_dsigma_da',report['dOmega_da_projection'][1]);report.setdefault('nonlinear',False)
        cases.append(dict(case=name,**report))
        total+=json.loads((path/'completion.json').read_text())['seconds']
        failed=path/'prior_attempt_failures.json'
        if failed.exists():total+=sum(r['seconds'] for r in json.loads(failed.read_text()))
    correlations=[]
    pairs=[('N32_dr1','N32_dr0.5'),('N32_dr0.5','N40_dr0.5')]
    if 'N32_dr0.25' in archives:pairs.insert(1,('N32_dr0.5','N32_dr0.25'))
    for aa,bb in pairs:
        a,b=archives[aa],archives[bb];r=np.intersect1d(a['radii'],b['radii']);ia=np.searchsorted(a['radii'],r);ib=np.searchsorted(b['radii'],r)
        va=a['q1'].reshape(len(a['radii']),4,len(a['xi']))[ia,:3]
        vb=b['q1'].reshape(len(b['radii']),4,len(b['xi']))[ib,:3]
        va=BarycentricInterpolator(a['xi'],va,axis=-1)(b['xi'])
        def inner(x,y):return np.einsum('rvi,ij,rvj,r->',x.conj(),b['mass'],y,radial_weights(r),optimize=True)
        corr=float(abs(inner(va,vb))/np.sqrt(inner(va,va).real*inner(vb,vb).real))
        correlations.append(dict(left=aa,right=bb,mass_correlation=corr,
            delta_a=float(b['a_c']-a['a_c']),delta_omega=float(b['omega_c']-a['omega_c']),
            note='Each state is at its own neutral parameter on its own grid; correlation alone does not establish parameter/derivative convergence.'))
    summary=dict(cases=cases,common_api_checks=json.loads((OUT/'common_api_checks.json').read_text()),
        endpoint_step_check=json.loads((OUT/'endpoint_step_check.json').read_text()),archive_validations=validations,
        cross_grid_correlations=correlations,numerical_seconds_including_failed_attempts=total,
        status=f'{len(cases)} matching-grid neutral states accepted and frozen',
        limitations=['Fixed Rout730 with varying outer sponge start; not the earlier varying-Rout comparison.',
            'Current dq/dR=0 outlet and mapped-infinite-z closure differ from the paper.',
            'No complete-spectrum exclusion of other unstable modes or time/Floquet attraction proof.',
            'N32/dr1 derivative differs substantially from dr0.5 and has stronger nonnormal conditioning.',
            'No claim of nonlinear saturation follows from a linear Hopf neutral point alone.'])
    dump(OUT/'summary.json',summary);dump(OUT/'artifact_validation.json',dict(all_passed=True,cases=validations))
    lines=['# 固定Rout730的全局Hopf中性点','',
        '日期2026-10-29。仅移动外海绵起点a，固定Rin400、Rout730、rise40、strength28，内海绵0.8*S((420-R)/10)。旧轮归档与共享核心保持只读。','',
        '| N | dr | a_c | omega_c（惯性系） | sigma残余 | d omega/da | d sigma/da |',
        '|---:|---:|---:|---:|---:|---:|---:|']
    for c in cases:
        derivative=c['dOmega_da_projection']
        lines.append(f"| {c['n']} | {c['dr']:g} | {c['a_c']:.10f} | {c['omega_c']:.10f} | {c['sigma_residual']:+.3e} | {derivative[0]:.9f} | {derivative[1]:.9f} |")
    lines.extend(['','每个配置分别寻找自己的中性点，禁止把一个网格的a_c或omega_c直接用于另一网格的弱非线性响应。omega_c为实数；sigma残余显式保存。','',
        '右模态满足 integral(q1_velocity^H M_z q1_velocity)dR=1，并将最大速度节点相位固定为实正；左模态满足p^H B1 q1=1。neutral_state.npz保存q1、left_mode、radii、z、xi、mass、radial_weights、B_diagonal、a_c、omega_c、sigma_residual和海绵/导数采样。','',
        '## 独立验证','',
        '公共构造器对j=0、1、2直接装配单谐波复矩阵，小网格校验逐项与原全实矩阵一致；频率导数误差最大2.83e-16；海绵起点导数中心差分误差最大4.21e-12。nonlinear=True显式生效，均值阻尼真/假均测试。平滑阶跃及导数在极近端点保持有限。','',
        '基频导数为dAj/domega=-i*j*Bj，Bj不乘j；固定域下Bj不依赖a。中性频率方向导数使用 dOmega/da=-i*p^H(A_a q1)/(p^H B1 q1)，并与实际参数中心差分核对。','',
        '| 配置 | 原方程绝对残差 | 换shift/v0特征值差 | 左模态残差 | 导数核验相对差 |','|---|---:|---:|---:|---:|'])
    for c in cases:
        der=c.get('derivative_refinement',{}).get('relative_error',c['derivative_relative_error'])
        lines.append(f"| {c['case']} | {c['original_equation_residual_inf']:.3e} | {c['repeat_eigenvalue_difference']:.3e} | {c['left_residual']:.3e} | {der:.3e} |")
    lines.extend(['',
        'N32/dr1的h=.05导数差分相对差7.28e-4，减小到h=.01后降至2.97e-5；冻结的中性q1、左模态及a_c未修改。该粗径向网格的中性导数和非正规敏感性与dr=.5仍有明显差异，因此其弱非线性系数必须与细网格结果分别报告。','',
        '## 分支追踪、失败与边界范围','',
        '从a630已核验增长模态出发，初次直接跳到615的质量相关仅0.454，按追踪标准拒绝；随后以5单位步长建立实际sigma异号括区，每步质量相关约0.908，并用括区根求解。原失败状态和耗时保留在N32_dr1/prior_attempt_failures.json。临界附近再换复shift与随机初始向量，未按谱排序索引跳分支。',''])
    if 'N32_dr0.25' in archives:
        lines.extend(['N32/dr0.25将dr0.5中性场插值仅作为初猜，在粗网格a_c±0.1处使用独立细网格算子建立实际异号括区，再求出自己的a_c与omega_c。未把插值场或粗网格临界参数直接当作细网格中性解。',''])
    for r in correlations:
        lines.append(f"- {r['left']}→{r['right']}：质量相关{r['mass_correlation']:.9f}，delta a={r['delta_a']:+.6f}，delta omega={r['delta_omega']:+.6f}。")
    lines.extend(['',
        '当前出口为dR(q)=0、q=u_prime/R，物理速度满足dR(u_prime)=u_prime/R；法向映射至无穷远并令三速度归零。它们与论文有限z=20.5及相应边界闭合不同。本轮a是海绵起点，不是论文的有效r_end，也不是上轮变Rout得到的阈值。','',
        f'累计数值工作（含失败起步及补充导数核验，不含小规模API检查）约{total:.1f}s；针对已发现的径向导数敏感性，本轮获准由8分钟延至12分钟。每进程BLAS/OMP/MKL=1。所有档案重新构造原实频率算子核验，见artifact_validation.json。', '',
        f'{len(cases)}个中性态分别作为匹配网格弱非线性响应的输入；线性中性点本身不证明非线性饱和、其他谱的稳定性或最终时间吸引性。',''])
    (OUT/'REPORT.md').write_text('\n'.join(lines))
    print('AUDIT '+json.dumps(dict(cases=len(cases),all_passed=True,numerical_seconds=total)),flush=True)


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--n',type=int);parser.add_argument('--dr',type=float)
    parser.add_argument('--checks',action='store_true');parser.add_argument('--seconds',type=int,default=180)
    parser.add_argument('--derivative-check',action='store_true')
    parser.add_argument('--audit',action='store_true')
    parser.add_argument('--nearby-seed',action='store_true')
    args=parser.parse_args();OUT.mkdir(parents=True,exist_ok=True)
    def alarm(signum,frame):raise TimeoutError('Neutral solve numerical budget reached')
    signal.signal(signal.SIGALRM,alarm);signal.alarm(args.seconds)
    if args.checks:checks();return
    if args.audit:audit();return
    if args.derivative_check:refine_derivative(args.n,args.dr);return
    path=OUT/case_name(args.n,args.dr);path.mkdir(parents=True,exist_ok=True)
    if (path/'completion.json').exists():
        previous=json.loads((path/'completion.json').read_text())
        if previous['status']=='failed':
            failure_path=path/'prior_attempt_failures.json'
            failures=json.loads(failure_path.read_text()) if failure_path.exists() else []
            failures.append(previous);dump(failure_path,failures)
    started=time.perf_counter()
    try:
        Branch(args.n,args.dr,path,nearby_seed=args.nearby_seed).neutral()
        dump(path/'completion.json',dict(status='complete',seconds=time.perf_counter()-started))
    except Exception as exc:
        dump(path/'completion.json',dict(status='failed',error=repr(exc),seconds=time.perf_counter()-started));raise


if __name__=='__main__':main()
