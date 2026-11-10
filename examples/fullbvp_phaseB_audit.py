#!/usr/bin/env python3
"""Read-only numerical audit; writes new derived audit/report artifacts only."""
import os
for key in ('OPENBLAS_NUM_THREADS','OMP_NUM_THREADS','MKL_NUM_THREADS','NUMEXPR_NUM_THREADS'):
    os.environ[key]='1'
os.environ.setdefault('MPLCONFIGDIR','/tmp/phaseB_audit_matplotlib')
import argparse,csv,hashlib,json,subprocess
from pathlib import Path
import numpy as np
from scipy.interpolate import BarycentricInterpolator,interp1d
from numpy.polynomial.legendre import leggauss
from fullbvp_phaseB_solver import ROOT,Mapping,Metric,pack,write,common,linear,convection

def read(path):return json.loads(Path(path).read_text())
def number(x,fmt='.7g'):return '未计算' if x is None else format(x,fmt)
def digest(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()
def qmap(x):
    t=.6*x+.4*(x**3+.5*(1-x*x));tp=.6+.4*(3*x*x-x)
    return t,tp
def mass_at(xi,nquad):
    x,w=leggauss(nquad);t,tp=qmap(x);z=2*(1+t)/(1-t)
    basis=BarycentricInterpolator(xi,np.eye(len(xi)),axis=0)(x)
    weights=w*4*tp/(1-t)**2
    return basis.T@(weights[:,None]*basis),basis,weights,z

def analyze(out):
    campaign=read(out/'campaign.json')
    # Frozen raw manifest and per-point original diagnostic files are the sources.
    raw=campaign['points']
    allpoints=[]
    for p in raw:
        d=read(out/'diagnostics'/(p['id']+'.json'))
        d['original_upper_branch']=d.get('upper_branch')
        d['legacy_harmonic_energy_shares_with_mean_invalid_physical_interpretation']=d.pop('harmonic_energy_shares_with_conjugates',None)
        d['physical_mean_energy_all_height']=None
        d['energy_labels']=['legacy_discrete_mean_norm','positive_j1','positive_j2','positive_j3','positive_j4'][:len(d['energies'])]
        allpoints.append(d)
    primary=[p for p in allpoints if p['role'] in ['weak_seed','primary_arclength']]
    A=np.array([p['A1'] for p in primary]);imax=int(np.argmax(A))
    turning=int(np.flatnonzero(np.diff([p['Rout'] for p in primary])>0)[0]+1)
    M=Mapping(32,3,800);metric=Metric(M)
    rref=metric.p.radii;rw=common.radial_weights(rref);mass=M.op.grid.mass
    inner=lambda a,b:np.einsum('rvi,ij,rvj,r->',a.conj(),mass,b,rw,optimize=True)
    trajectories=[];cache={};previous=None;delta_previous=None
    for i,p in enumerate(primary):
        d=np.load(ROOT/p['fields']);q=d['field'];v=q.reshape(M.nr,7,4,M.m)[:,3:,:3]
        y=np.r_[pack(q),float(d['omega']),float(d['Rout'])]
        ref=d['qref'].reshape(M.nr,4,M.m)[:,:3]
        overlap=inner(ref,v[:,1])/np.sqrt(inner(ref,ref).real*inner(v[:,1],v[:,1]).real)
        rec=read(ROOT/p['summary'])
        row=dict(index=i,id=p['id'],A1=p['A1'],Rout=p['Rout'],omega=p['omega'],
            discrete_mean_norm=p['energies'][0],mean_far_w_max=float(abs(v[:,0,2,-1]).max()),
            phase_sensitivity_normalized=float(overlap.real),phase_residual_normalized=float(overlap.imag),
            newton_updates=len(rec['iterations'])-1,predictor_accepted_without_update=len(rec['iterations'])==1,
            independent_relative_field_inf=rec['independent']['relative_field_inf'],
            residual_over_feedback=p['residual_over_fundamental_feedback'],
            highest_to_fundamental=p['highest_to_fundamental'])
        if previous is not None:
            delta=y-previous;ds2=metric.dot(delta,delta);row['actual_metric_step']=float(np.sqrt(ds2))
            dv=(v-cache[i-1])
            parts=[float(inner(dv[:,j],dv[:,j]).real*(1 if j==0 else 2)/metric.fs**2) for j in range(4)]
            parts += [float(delta[-2]**2/metric.ws**2),float(delta[-1]**2/metric.rs**2)]
            row['metric_step_fractions']=dict(zip(['mean','j1','j2','j3','omega','Rout'],[x/ds2 for x in parts]))
            if i>1:
                t=d['tangent'];projection=metric.dot(t,delta)
                row['predictor_projection']=float(projection)
                row['predictor_corrector_distance']=float(np.sqrt(max(metric.dot(delta-projection*t,delta-projection*t),0)))
                if delta_previous is not None:
                    row['cos_previous_actual_step']=metric.dot(delta,delta_previous)/np.sqrt(ds2*metric.dot(delta_previous,delta_previous))
            delta_previous=delta
        previous=y;cache[i]=v
        p['primary_index']=i;p['newton_updates']=row['newton_updates']
        trajectories.append(row)
    strict=[i for i in range(turning,imax+1) if trajectories[i]['newton_updates']>0 and trajectories[i]['independent_relative_field_inf']<1e-9]
    representative=strict[-1]
    comparisons=[]
    for i in range(imax+1,len(primary)):
        target=A[i];ix=np.sort(np.argsort(abs(A[:representative+1]-target))[:4])
        x=A[ix]
        predicted=BarycentricInterpolator(x,np.stack([cache[k] for k in ix]),axis=0)(target)
        predicted_R=float(BarycentricInterpolator(x,[primary[k]['Rout'] for k in ix])(target))
        actual_r=np.linspace(400,primary[i]['Rout'],M.nr)
        common_r=actual_r[actual_r<=min(predicted_R,actual_r[-1])]
        actual=cache[i][:len(common_r)]
        predicted=interp1d(np.linspace(400,predicted_R,M.nr),predicted,axis=0,kind='cubic')(common_r)
        crw=common.radial_weights(common_r)
        cin=lambda a,b:np.einsum('rvi,ij,rvj,r->',a.conj(),mass,b,crw,optimize=True)
        phase=float(np.angle(cin(predicted[:,1],actual[:,1])))
        aligned=actual*np.exp(-1j*np.arange(4)*phase)[None,:,None,None]
        rel=[float(np.sqrt(cin(aligned[:,j]-predicted[:,j],aligned[:,j]-predicted[:,j]).real/max(cin(predicted[:,j],predicted[:,j]).real,1e-300))) for j in range(4)]
        dr=float(primary[i]['Rout']-predicted_R)
        dw=float(primary[i]['omega']-BarycentricInterpolator(x,[primary[k]['omega'] for k in ix])(target))
        comparisons.append(dict(index=i,id=primary[i]['id'],forward_indices=ix.tolist(),A1=float(target),
            extrapolated=bool(target<x.min() or target>x.max()),phase_alignment=phase,common_physical_R_range=[float(common_r[0]),float(common_r[-1])],mean_norm_interpretation='legacy_discrete_allheight_not_finite_physical_mean_error',
            delta_Rout=dr,delta_omega=dw,relative_mean=rel[0],relative_j1=rel[1],
            relative_j2=rel[2],relative_j3=rel[3],
            return_matches_forward=bool(rel[0]<1e-3 and rel[1]<1e-4 and abs(dr)<.002 and abs(dw)<1e-4)))
    verified_return=[x['index'] for x in comparisons if x['return_matches_forward'] and not x['extrapolated']]
    return_start=min(i for i in verified_return if all(j in verified_return for j in range(i,len(primary)-1)))
    foldA=campaign['folds']['H3']['A1']
    for i,p in enumerate(primary):
        if i<turning:classification='first_pass_lower_or_fold'
        elif i<=representative:classification='first_pass_upper'
        elif i<=imax:classification='first_pass_upper_predictor_only_extension'
        elif i<return_start:classification='high_amplitude_mean_dominated_excursion_unresolved'
        elif p['A1']>foldA:classification='return_retracing_upper'
        else:classification='return_retracing_lower'
        p['branch_classification']=classification
        p['upper_branch']=classification in ['first_pass_upper','first_pass_upper_predictor_only_extension']
        p['audited_unique_upper_sample']=classification=='first_pass_upper'
        trajectories[i]['classification']=classification
    for p in allpoints:
        if p['role']=='mesh_refinement':
            p['branch_classification']='lower_branch_resolution_check'
            p['upper_branch']=False
    # Mean-difference localization at a tightly corrected high-excursion root.
    locindex=42;item=next(x for x in comparisons if x['index']==locindex);ix=item['forward_indices']
    pred=BarycentricInterpolator(A[ix],np.stack([cache[k] for k in ix]),axis=0)(A[locindex])
    rloc=np.linspace(400,primary[locindex]['Rout'],M.nr);wloc=common.radial_weights(rloc)
    pred_R=float(BarycentricInterpolator(A[ix],[primary[k]['Rout'] for k in ix])(A[locindex]))
    pred=interp1d(np.linspace(400,pred_R,M.nr),pred,axis=0,kind='cubic')(rloc)
    difference=cache[locindex][:,0]-pred[:,0]
    component=np.einsum('rvi,ij,rvj,r->v',difference.conj(),mass,difference,wloc,optimize=True).real
    density=np.einsum('rvi,ij,rvj->r',difference.conj(),mass,difference,optimize=True).real
    m0,basis,weights,zq=mass_at(M.op.grid.xi,2*M.n+80)
    samples=difference@basis.T
    rz=np.sum(abs(samples)**2,axis=1)*weights[None,:]*wloc[:,None]
    bins=[(0,5),(5,20),(20,100),(100,500),(500,np.inf)]
    layers=[dict(z_lower=lo,z_upper=None if np.isinf(hi) else hi,fraction=float(rz[:,(zq>=lo)&(zq<hi)].sum()/rz.sum())) for lo,hi in bins]
    sensitivity=[];fixed_sensitivity=[]
    for nq in [144,288,576]:
        mq,*_=mass_at(M.op.grid.xi,nq)
        energy=float(np.einsum('rvi,ij,rvj,r->',difference.conj(),mq,difference,wloc,optimize=True).real)
        sensitivity.append(dict(quadrature_nodes=nq,discrete_difference_norm_squared=energy))
        for idx in [representative,locindex]:
            values=[float(np.einsum('rvi,ij,rvj,r->',cache[idx][:,j].conj(),mq,cache[idx][:,j],common.radial_weights(np.linspace(400,primary[idx]['Rout'],M.nr)),optimize=True).real) for j in [0,1]]
            fixed_sensitivity.append(dict(index=idx,quadrature_nodes=nq,legacy_E0=values[0],E1=values[1]))
    location=dict(index=locindex,matched_forward=item,component_fraction=dict(zip(['u0','v0','w0'],(component/component.sum()).tolist())),
        original_quadrature_difference_norm_squared=float(component.sum()),height_fractions=layers,
        radial_fraction_R_below_600=float(np.sum(density[rloc<600]*wloc[rloc<600])/np.sum(density*wloc)),
        radial_outer_sponge_fraction=float(np.sum(density[rloc>=primary[locindex]['a']]*wloc[rloc>=primary[locindex]['a']])/np.sum(density*wloc)),
        mean_w_infinity_max=trajectories[locindex]['mean_far_w_max'],
        difference_w_infinity_max=float(abs(difference[:,2,-1]).max()),
        quadrature_sensitivity=sensitivity,fixed_field_quadrature_sensitivity=fixed_sensitivity,
        interpretation='Mean closure admits nonzero w0 at infinity; the all-height L2 integral then diverges. Existing finite quadrature mean norm is a discretization-dependent continuation metric, not finite physical mean energy. This evidence does not establish a second physical fold.')
    failures=[]
    for a in campaign['attempts']:
        if a['accepted']:continue
        d=read(ROOT/a['summary']);last=d['iterations'][-1];ind=d.get('independent')
        scaled_defect=None
        if ind:
            ck=np.load((ROOT/a['summary']).parent/'checkpoint.npz');cq=ck['field']
            cp,blocks,_,_=M.at(float(ck['Rout']),float(ck['omega']))
            denominator=max(np.linalg.norm(pack(linear(blocks,cq,M.H))),np.linalg.norm(pack(convection(cp,cq))),1e-300)
            absolute_l2=ind['template_difference_relative']*denominator
            bound=np.sqrt(sum(np.linalg.norm(abs(blocks[j])@abs(cq[:,M.H+j]).ravel())**2 for j in range(M.H+1)))
            scaled_defect=dict(reconstructed_absolute_difference_l2=float(absolute_l2),difference_over_field_l2=float(absolute_l2/np.linalg.norm(pack(cq))),difference_over_absolute_operator_bound=float(absolute_l2/bound),scale='bound is || |A| |q| ||, no cancellation')
        failures.append(dict(id=a['name'],failure=a['failure'],seconds=a['seconds'],template_difference_reconstruction=scaled_defect,
            newton_updates=len(d['iterations'])-1,last_relative_field_inf=last['relative_field_inf'],
            last_feedback_ratio=last['residual_over_fundamental_feedback'],
            independent=ind,maximum_linear_relative=max([x['linear']['relative_original'] for x in d['iterations'] if 'linear' in x],default=None),
            interpretation='cancellation_normalization_guard_rejection_not_Newton_nonconvergence' if ind else 'Newton_iteration_limit_at_high_amplitude'))
    first=[primary[k] for k in [21,26,31,representative]]
    fd=np.load(out/'diagnostics'/(primary[representative]['id']+'.npz'))
    dns=np.genfromtxt(ROOT/'results/appelquist2016_branch_20261029/literature/fig15_nl03_T6_vrms.csv',delimiter=',',names=True)
    rms_comparison=[]
    for R in [570.,585.,600.]:
        bv=float(np.exp(np.interp(R,fd['radii'],np.log(np.maximum(fd['vrms'],1e-300)))))
        dv=float(np.exp(np.interp(R,dns['R'],np.log(dns['vrms_normalized']))))
        rms_comparison.append(dict(R=R,BVP=bv,NL03=dv,BVP_over_NL03=bv/dv))
    differences=[dict(ids=[a['id'],b['id']],Rout_pair=[a['Rout'],b['Rout']],delta_Rout=b['Rout']-a['Rout'],delta_omega=b['omega']-a['omega'],delta_front=b['front_fig8']-a['front_fig8'],domega_dRout=(b['omega']-a['omega'])/(b['Rout']-a['Rout']),dfront_dRout=(b['front_fig8']-a['front_fig8'])/(b['Rout']-a['Rout'])) for a,b in zip(first[:-1],first[1:])]
    neutralfit=np.polyfit([p['A1']**2 for p in primary[:2]],[p['Rout'] for p in primary[:2]],1)
    b4=read(out/'upper_refinement_audit/B4_SUMMARY.json') if (out/'upper_refinement_audit/B4_SUMMARY.json').exists() else dict(status='prepared_not_launched',source='branch_0026_try0',points=[],comparisons=[])
    from fullbvp_phaseB_report import compare_fields
    lower_checks=[]
    for x in campaign['refinements']:
        b=next(p for p in allpoints if p['id']==x['source']);v=next(p for p in allpoints if p['id']==x['point'])
        cmp=compare_fields(b,v);legacy=cmp.pop('mean_relative_mass')
        lower_checks.append(dict(x,scientific_role='lower_branch_only_not_B4',legacy_mean_relative_mass_invalid_physical_interpretation=legacy,**cmp))
    findings=dict(status='audited_results_B4_upper_pending' if b4['status']=='prepared_not_launched' else 'audited_results_with_upper_refinement',
        raw_accepted_record_count=len(raw),primary_record_count=len(primary),first_pass_record_count=imax+1,
        later_primary_record_count=len(primary)-imax-1,distinct_solution_count=None,
        count_interpretation='95 accepted records, not 95 distinct upper-branch states; nearby states and retracing are not independent domain samples',
        first_positive_dRout_index=turning,maximum_amplitude_index=imax,strict_high_amplitude_index=representative,
        verified_return_start_index=return_start,strict_high_amplitude_point=primary[representative],
        maximum_record=primary[imax],last_record=primary[-1],first_upper_selected=first,strict_high_RMS_NL03_comparison=rms_comparison,
        lower_branch_resolution_checks=lower_checks,
        upper_branch_resolution_checks=b4,
        phase_gauge=dict(minimum_real_normalized_sensitivity=float(min(t['phase_sensitivity_normalized'] for t in trajectories[2:])),
            maximum_imaginary_normalized_projection=float(max(abs(t['phase_residual_normalized']) for t in trajectories[2:])),
            diagnosis='No evidence for vanishing fixed-reference phase sensitivity in this path.'),
        stop_interpretation='Four final independent template-comparison guard rejections after Newton had met residual criteria; not four terminal Newton failures.',
        mean_farfield_localization=location,folds=campaign['folds'],
        neutral_same_grid_Rout_E1_extrapolation=float(neutralfit[-1]),
        subcritical_width_same_grid_approx=float(neutralfit[-1]-campaign['folds']['H3']['Rout']),
        front_conventions=dict(phaseA_594_19277='height-integrated-energy logarithmic-growth intersection, not z=1.3 Fig8 RMS',
            phaseA_N32_dr05_v13=597.9766953924515,phaseA_N40_dr025_v13=597.547850,
            phaseB_weak_same_grid_total_rms=primary[0]['front_fig8'],phaseB_definition='total normalized azimuthal RMS at z=1.3'),
        domain_convergence=dict(established=False,status='Rout730_750_780_800_unreached',
            reliable_first_upper_Rout_range=[primary[turning]['Rout'],primary[representative]['Rout']],
            maximum_observed_Rout_on_first_upper=primary[imax]['Rout'],selected_consecutive_differences=differences,
            explanation='Available upper outlet interval is only about2.12; shrinking raw increments with unequal Rout spacing and retraced lower roots do not establish a domain limit.'),
        targets={str(x):dict(status='unavailable',Rout=x,front=None,omega=None) for x in [730,750,780,800]},
        physical_mean_energy_all_height=None,physical_mean_energy_status='undefined_for_nonzero_w0_infinity',
        coherent_harmonic_energy_definition='2*sum Ej over positive harmonics; E1 and EH/E1 unchanged; cylindrical coherent fractions use R^3 dR dz',
        temporal_stability=None,temporal_stability_status='not_computed',
        failures=failures,trajectory=trajectories,return_comparisons=comparisons,points=allpoints,
        source_hashes={'campaign.json':digest(out/'campaign.json'),'accepted_points.json':digest(out/'accepted_points.json')})
    return findings

def render(out,s,only=None):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    plt.rcParams.update({'font.family':'serif','font.serif':['Times New Roman','Nimbus Roman','STIXGeneral','DejaVu Serif'],'mathtext.fontset':'stix','font.size':9.5,'axes.linewidth':.9,'lines.linewidth':1.4,'savefig.dpi':220})
    figdir=out/'figures';figdir.mkdir(exist_ok=True)
    primary=[p for p in s['points'] if 'primary_index' in p]
    rep=s['strict_high_amplitude_index'];maximum=s['maximum_amplitude_index'];ret=s['verified_return_start_index']
    segments=[(primary[:rep+1],'#195d96','-','First pass, corrected roots'),
        (primary[rep:maximum+1],'#c07316',':','Predictor-only extension'),
        (primary[maximum:ret+1],'#a32828','--','Mean-dominated excursion'),
        (primary[ret-1:],'#777777','--','Return / retracing')]
    def finish(fig,name):
        if only is not None and name!=only:
            plt.close(fig);return
        fig.savefig(figdir/(name+'.png'));fig.savefig(figdir/(name+'.pdf'));plt.close(fig)
        subprocess.run(['pdftoppm','-png','-singlefile','-scale-to','1700',str(figdir/(name+'.pdf')),str(figdir/(name+'_pdf_render'))],check=True,stdout=subprocess.DEVNULL,stderr=subprocess.DEVNULL)
    fig,axes=plt.subplots(1,2,figsize=(7.7,3.6),layout='constrained')
    for ax,key,label in [(axes[0],'Rout',r'$R_{\rm out}$'),(axes[1],'omega',r'$\omega$')]:
        for pp,color,ls,text in segments:ax.plot([p['A1'] for p in pp],[p[key] for p in pp],ls,color=color,label=text)
        for H,color in [(3,'#161616'),(4,'#943f75')]:
            pp=[p for p in s['points'] if p['role']=='fold_refinement' and p['H']==H]
            ax.plot([p['A1'] for p in pp],[p[key] for p in pp],'s',ms=3,color=color,label=f'H{H} local fold samples')
        ax.set(xlabel=r'$A_1=\sqrt{E_1}$',ylabel=label);ax.tick_params(direction='in',top=True,right=True)
    axes[1].axhline(52.36,color='.5',ls=':',label='DNS 52.36')
    axes[0].legend(fontsize=6.9,loc='lower left');axes[1].legend(fontsize=6.9,loc='lower left')
    fig.suptitle('The discrete-BVP fold is verified; accepted records include retracing',fontsize=10)
    finish(fig,'branch_Rout_A1_omega')
    fig,ax=plt.subplots(figsize=(7.7,4.6),layout='constrained')
    for p in s['first_upper_selected']:
        d=np.load(out/'diagnostics'/(p['id']+'.npz'));r=d['radii'];mask=r<=p['a']
        line,=ax.semilogy(r[mask],d['vrms'][mask],label=f'Upper H3: Rout={p["Rout"]:.2f}, A1={p["A1"]:.3f}')
        ax.semilogy(r[~mask],d['vrms'][~mask],color=line.get_color(),ls='--',alpha=.65)
    dns=np.genfromtxt(ROOT/'results/appelquist2016_branch_20261029/literature/fig15_nl03_T6_vrms.csv',delimiter=',',names=True)
    ax.semilogy(dns['R'],dns['vrms_normalized'],color='#a52626',lw=1.8,label='NL03 DNS, T=6: digitized Fig.15')
    ax.fill_between(dns['R'],dns['vrms_plot_lower'],dns['vrms_plot_upper'],color='#a52626',alpha=.12)
    dn=np.genfromtxt(out/'reference_data/nl01_T6_vrms.csv',delimiter=',',names=True)
    ax.errorbar(dn['R'],dn['vrms_normalized'],xerr=dn['R_plot_reading_tolerance'],yerr=np.stack([dn['vrms_normalized']-dn['vrms_plot_lower'],dn['vrms_plot_upper']-dn['vrms_normalized']]),fmt='s',ms=3,color='#367b40',capsize=2,elinewidth=.7,label='NL01 T=6: six Fig.6(b) contour samples')
    ax.set(xlabel=r'$R$',ylabel=r'$v_{\rm rms}/(R\Omega)$ at $z=1.3$',xlim=(515,725),ylim=(2e-8,max(.2,1.2*max(dns['vrms_plot_upper']))))
    ax.legend(fontsize=7,loc='lower right')
    ax.text(.02,.97,'Dashed BVP: outer sponge\nNL01 bars: graphical reading envelopes',transform=ax.transAxes,va='top',fontsize=7.8)
    ax.set_title('First-pass upper roots; Rout = 730, 750, 780, 800 unavailable',fontsize=10)
    finish(fig,'vrms_DNS')
    fig,axes=plt.subplots(1,2,figsize=(7.7,3.6),layout='constrained')
    pp=primary[s['first_positive_dRout_index']:rep+1];ext=primary[rep+1:maximum+1]
    for ax,key,label,ref in [(axes[0],'front_fig8',r'$r_{\rm end}$ from total RMS',583),(axes[1],'omega',r'$\omega$',52.36)]:
        ax.plot([p['Rout'] for p in pp],[p[key] for p in pp],'-o',ms=2.8,color='#195d96',label='First-pass upper')
        ax.plot([p['Rout'] for p in ext],[p[key] for p in ext],'o',mfc='none',ms=4,color='#c07316',label='Predictor-only records')
        ax.axhline(ref,color='#a52626',ls=':',label=f'2016 reference {ref:g}')
        ax.set(xlabel=r'$R_{\rm out}$',ylabel=label)
        ax.legend(fontsize=7.1,loc='best')
    fig.suptitle('Only a short upper-branch outlet interval: no domain-convergence limit',fontsize=10)
    finish(fig,'front_Rout')
    fig,axes=plt.subplots(2,2,figsize=(7.7,5.4),layout='constrained')
    tr=s['trajectory']
    for ax,key,label in [(axes[0,0],'A1',r'$A_1$'),(axes[0,1],'Rout',r'$R_{\rm out}$'),(axes[1,0],'discrete_mean_norm',r'Discrete mean norm $E_0$'),(axes[1,1],'phase_sensitivity_normalized','Normalized phase sensitivity')]:
        ax.plot([t['index'] for t in tr],[t[key] for t in tr],color='#195d96')
        ax.axvspan(rep,ret,color='#c07316',alpha=.13)
        for idx,text in [(rep,'36'),(maximum,'39'),(81,'81')]:
            ax.plot(idx,tr[idx][key],'o',ms=3,color='#a52626')
        ax.set(xlabel='Accepted primary-record index',ylabel=label)
    axes[1,0].set_title('Finite quadrature norm; not all-height physical energy',fontsize=7.5,pad=7)
    axes[1,1].set_ylim(.88,1.01)
    fig.suptitle('High-amplitude excursion is mean dominated; phase projection stays nonzero',fontsize=10)
    finish(fig,'continuation_audit')
    if only is None:write(figdir/'visual_audit.json',dict(status='pending_final_visual_review',figures=['branch_Rout_A1_omega','vrms_DNS','front_Rout','continuation_audit'],PNG_and_PDF_rasters=True,source='fullbvp_phaseB_audit.py',interpretation='Audited first-pass / excursion / return classifications; no invented upper-target samples.'))

def report(out,s):
    hi=s['strict_high_amplitude_point'];mx=s['maximum_record'];end=s['last_record'];weak=s['points'][0];f3=s['folds']['H3'];f4=s['folds']['H4'];loc=s['mean_farfield_localization']
    lines=['# Phase B 最终审计：离散BVP折点已核实，上支域收敛未建立','',
        f'原流程保存95个接受记录（其中82个主支记录），不能解释为95个不同上支解。首次去程跨过一个经过H3/H4局部校正核实的出口折点；后段出现均值主导的小范围折返，并重走上、下支，最终回到弱幅解。Rout=730、750、780、800均未达到，域收敛尚未建立。','',
        f'**严格高幅代表为#36**：A1={hi["A1"]:.7f}，Rout={hi["Rout"]:.6f}，omega={hi["omega"]:.6f}，Fig.8前沿={hi["front_fig8"]:.4f}。该点做了2次Newton更新，独立原方程相对场残差7.84e−12。',
        f'**最大幅值记录为#39**：A1≈{mx["A1"]:.5f}，Rout≈{mx["Rout"]:.4f}，omega≈{mx["omega"]:.5f}。它没有Newton更新，预测场直接通过旧阈值，残差约9.12e−8；可报告为最大记录，不能据此确认新的物理折点。',
        f'**末次记录#81是回程弱幅下支**：A1={end["A1"]:.7f}，Rout={end["Rout"]:.6f}。原sticky upper_branch标签及基于它生成的“最后上支”判断已在派生报告中撤回，原campaign和尝试场保持不变。','',
        f'在N32、800径向区间上，H3折点Rout={f3["Rout"]:.6f}，A1={f3["A1"]:.6f}；H4为{f4["Rout"]:.6f}、{f4["A1"]:.6f}。H4−H3出口差为{s["folds"]["H4_minus_H3_Rout"]:+.6f}，仅代表谐波截断敏感性，不是总误差。',
        f'同一网格两弱幅点按E1外推中性出口约{s["neutral_same_grid_Rout_E1_extrapolation"]:.6f}，对应亚临界范围约{s["subcritical_width_same_grid_approx"]:.3f}；没有混用细网格线性中性720.46986。该二点外推只用于尺度比较。','',
        f'严格高幅点的omega比DNS52.36高{hi["omega"]-52.36:.6f}；前沿比2016渐近外推约583靠外{hi["front_fig8"]-583:.4f}。g_q(570–600)={hi["g_q_570_600"]:.7f}，相对线性近似0.132差{hi["g_q_570_600"]-.132:+.7f}，相对DNS图读约0.106差{hi["g_q_570_600"]-.106:+.7f}。',
        f'该点coherent海绵能量比例仍为q范数{100*hi["sponge_fraction_q"]:.3f}%、圆柱R³物理测度{100*hi["sponge_fraction_cylindrical_R3"]:.3f}%；不能据此宣称自然饱和或出口独立。尚未计算Floquet，分支存在不是时间稳定性分类。','',
        '## B2：折点、完整路径与终止原因',
        '| 折点 | Rout | A1 | 3点−5点ΔRout | 3点−5点ΔA1 | 拟合RMS |','|---|---:|---:|---:|---:|---:|']
    for name,f in [('H3',f3),('H4',f4)]:lines.append(f'| {name} | {f["Rout"]:.7f} | {f["A1"]:.7f} | {f["three_vs_five_Rout"]:+.3g} | {f["three_vs_five_A1"]:+.3g} | {f["fit_rms"]:.3g} |')
    lines+=['','局部拟合使用各自五个固定物理E1校正点，顶点位于拟合区间内、左右斜率异号；已有折点结论保留。折点位置尚无N40和dr≈0.25的完整误差预算。','',
        '首次去程#0–36含弱支、折点和高幅上支；#37–39是零次Newton预测接受扩展；#40至回程重合段之前的高幅变化不能直接归为相位等价重复。较后回程在相同A1对首次去程作四点场插值并相位对齐后，基频和均值高度重合，支持“重走”而不是继续向大出口发展。比较保留插值节点和外推标志，不把插值差冒称严格同参数根距离。','',
        f'相位约束对旋转相位的归一化实灵敏度最小为{s["phase_gauge"]["minimum_real_normalized_sensitivity"]:.6f}，高幅附近约0.912–0.915；没有相位钉扎退化的证据。实际步对预测切向量的投影保持正值，但均值方向贡献可占几乎全部弧长，参数Rout或A1仍可反向。','',
        '高幅附近有4次真正的Newton迭代上限失败。最终#82的4次停止尝试则不同：Newton已满足旧残差准则，被独立模板比较拒收。其template_difference_relative略高于1e−10，而该量以近乎相消的线性/非线性项为分母；弱幅反馈很小时，相对比值放大舍入差。最后一次原方程残差约2.06e−14（相对场），不能写为物理方程不收敛。此次审计不把被拒场追加入接受清单。','',
        '| 尝试 | 原因 | Newton更新 | 最终场相对残差 | 模板相对差 | 无抵消算子尺度差 |','|---|---|---:|---:|---:|---:|']
    for f in s['failures']:
        cause='独立模板比较门限' if f['independent'] else '高幅Newton迭代上限'
        bound=None if f['template_difference_reconstruction'] is None else f['template_difference_reconstruction']['difference_over_absolute_operator_bound']
        lines.append(f'| {f["id"]} | {cause} | {f["newton_updates"]} | {f["last_relative_field_inf"]:.3g} | {number(None if f["independent"] is None else f["independent"]["template_difference_relative"],".3g")} | {number(bound,".3g")} |')
    lines+=['','## 均值远场定位：续接度量敏感性证据',
        f'在#36–39的实际步中，固定参考弧长度量的均值项占{100*s["trajectory"][36]["metric_step_fractions"]["mean"]:.4f}%，基频项仅{100*s["trajectory"][36]["metric_step_fractions"]["j1"]:.4f}%。这解释了为什么弧长继续前进并不等于可观测幅值或Rout有效前进；因果性仍待修改度量后复算。',
        f'#42与相同A1的首次去程插值相比，均值离散范数差约20%，基频差约1.1e−4；均值差的{100*loc["component_fraction"]["w0"]:.4f}%来自w0。径向R<600占{100*loc["radial_fraction_R_below_600"]:.2f}%，外海绵占{100*loc["radial_outer_sponge_fraction"]:.2f}%。',
        f'#42的w0(z=∞)最大幅值{loc["mean_w_infinity_max"]:.4g}，差场端点{loc["difference_w_infinity_max"]:.4g}。均值闭合允许非零远场法向速度；质量矩阵构造的注释则要求速度在无穷远衰减。非零极限使全高度L2均值积分发散，因此原E0和含E0的弧长度量是依赖有限求积的离散量，不能称为有限物理总均值能量。',
        '这里将E0改称“离散均值范数”，不再把E0+2ΣEj的份额解释为物理总能量份额。E1、非零谐波份额、EH/E1阈值及coherent海绵份额仍按衰减扰动计算。','',
        '| 法向范围 | 原离散均值差范数份额 |','|---|---:|']
    for z in loc['height_fractions']:lines.append(f'| {z["z_lower"]}–{number(z["z_upper"]) if z["z_upper"] is not None else "∞"} | {100*z["fraction"]:.6f}% |')
    lines+=['','固定同一差场，仅增加全高度Gauss求积点：','| 求积点数 | 离散均值差范数平方 |','|---|---:|']
    for x in loc['quadrature_sensitivity']:lines.append(f'| {x["quadrature_nodes"]} | {x["discrete_difference_norm_squared"]:.7g} |')
    lines+=['','同一固定场的均值与基频求积对照：','| 记录 | 求积点 | 旧E0 | E1 |','|---|---:|---:|---:|']
    for x in loc['fixed_field_quadrature_sensitivity']:lines.append(f'| #{x["index"]} | {x["quadrature_nodes"]} | {x["legacy_E0"]:.7g} | {x["E1"]:.7g} |')
    lines+=['','非零远场w0来自当前均值闭合，本身不自动否定离散BVP根；问题是将它放入要求远场衰减的全高度能量度量。这提供了远场均值主导的续接数值敏感性证据。尚未通过改变度量并严格重复校正、切向量零空间检查或扩展折点求解来定位高幅障碍，所以不宣称第二物理折点，也不宣称唯一根因已经证明。','',
        '## B3：真正首次上支的流动诊断',
        '| 记录 | A1 | Rout | omega | Fig.8前沿 | g_q | g_RMS | q海绵比例 | R³海绵比例 |','|---|---:|---:|---:|---:|---:|---:|---:|---:|']
    for p in s['first_upper_selected']:lines.append(f'| {p["id"]} | {p["A1"]:.6f} | {p["Rout"]:.5f} | {p["omega"]:.6f} | {p["front_fig8"]:.4f} | {p["g_q_570_600"]:.7f} | {p["g_rms_570_600"]:.7f} | {p["sponge_fraction_q"]:.5f} | {p["sponge_fraction_cylindrical_R3"]:.5f} |')
    lines+=['','| 相邻Rout | Δomega | Δfront | Δomega/ΔRout | Δfront/ΔRout |','|---|---:|---:|---:|---:|']
    for x in s['domain_convergence']['selected_consecutive_differences']:lines.append(f'| {x["Rout_pair"][0]:.4f}→{x["Rout_pair"][1]:.4f} | {x["delta_omega"]:+.6f} | {x["delta_front"]:+.4f} | {x["domega_dRout"]:+.6f} | {x["dfront_dRout"]:+.4f} |')
    lines+=['','严格高幅#36与NL03的幅值量级对照（双方均在log-RMS中径向线性插值）：','| R | BVP RMS | NL03 RMS | BVP/NL03 |','|---|---:|---:|---:|']
    for x in s['strict_high_RMS_NL03_comparison']:lines.append(f'| {x["R"]:g} | {x["BVP"]:.6g} | {x["NL03"]:.6g} | {x["BVP_over_NL03"]:.5f} |')
    lines+=['','前沿接近并不意味着DNS幅值吻合；该BVP相干RMS在这三处仅约为NL03的17%–30%。','', '该序列随振幅增大表现为omega下降、前沿向内移动、coherent海绵占比下降。但可靠首次上支出口只覆盖约716.807–718.932，Rout=730/750/780/800全不可用；未建立出口域收敛。回程的Rout约719.95属于弱幅下支，不得补入这组域收敛数据。','',
        f'Fig.8统一使用z=1.3总归一化方位RMS的对数增长率与局部绝对pinch −Im(alpha0)的向下交点；基频交点另存。Phase A的594.19277使用高度积分能量增长率，是另一诊断。Phase A同口径N32/dr≈0.5的v13交点597.97670，细网格为597.54785。本次弱幅RMS交点{weak["front_fig8"]:.5f}，与N32同口径相符；严格高幅前沿相对本次弱幅向内移{weak["front_fig8"]-hi["front_fig8"]:.4f}，不能把不同定义的3.35差值当非线性迁移。',
        'RMS=√(2Σ|q_v,j|²)，q_v已用当地盘速归一化，不再次除以R。NL03使用既存Fig.15数化曲线；NL01仅画Fig.6(b)、T=6的六个等值线点，误差棒为图读包络而非统计置信区间。2beta辅助阈值为复系数1e−6，源数据保存其位置和相对基频5%位置，不将阈值位置混称Fig.8前沿。','',
        '## B4：原检查属于下支；真实上支独立补算',
        '旧N40、dr≈0.25、H4三个检查都以末次A1≈0.005713回程弱根为源，只能保留为下支分辨率检查。其N40出口变化约+0.51566提示法向离散敏感性，不能直接平移推算高幅折点。','',
        '| 旧下支检查 | ΔRout | Δomega |','|---|---:|---:|']
    for x in s['lower_branch_resolution_checks']:lines.append(f'| {x["label"]} | {number(x["delta_Rout"])} | {number(x["delta_omega"])} |')
    lines += ['',f'真实上支补算状态：{s["upper_branch_resolution_checks"]["status"]}。自治脚本为examples/fullbvp_phaseB_upper_refinement.py，源点首次上支#26，A1≈0.199506，E3/E1≈0.005706；依次严格校正基准、N40、dr≈0.25、H4，匹配物理E1。每点1200秒，CPU单线程，无GPU，独立B4_DONE.json/B4_FAILED.json。当前回合未启动该任务。' if s['upper_branch_resolution_checks']['status']=='prepared_not_launched' else f'真实上支独立补算状态：{s["upper_branch_resolution_checks"]["status"]}。结果见下表及upper_refinement_audit/B4_SUMMARY.json。']
    lines+=['','| 真正上支检查 | 状态 | ΔRout | Δomega | Δfront | 均值z≤20相对差 | 均值z≤40相对差 |','|---|---|---:|---:|---:|---:|---:|']
    for x in s['upper_branch_resolution_checks'].get('comparisons',[]):
        finite=x.get('finite_height_mean_comparisons',[])
        values={v['zmax']:v['relative_velocity_L2'] for v in finite}
        lines.append(f'| {x["label"]} | {x["status"]} | {number(x.get("delta_Rout"))} | {number(x.get("delta_omega"))} | {number(x.get("delta_front"))} | {number(values.get(20))} | {number(values.get(40))} |')
    if not s['upper_branch_resolution_checks'].get('comparisons'):lines.append('| 待独立补算 | 未启动 | 未计算 | 未计算 | 未计算 | 未计算 | 未计算 |')
    lines+=['','## 数据和复现','原campaign.json、accepted_points.json及attempts/原始数据未改。新audit/保存逐步方向/度量份额、相位投影、回程场比较、失败证据和远场定位。audited_points.json提供修正分类；原95条验收数保留但不冒称独立解数。','',
        '运行审计：python3 examples/fullbvp_phaseB_audit.py --out '+str(out),
        '唯一待启动任务（须在持久的沙箱外执行）：setsid bash '+str(out/'commands_B4_upper.sh')+' > '+str(out/'upper_refinement_workflow.log')+' 2>&1 < /dev/null &',
        '图件由同一审计脚本生成PNG/PDF及PDF栅格审阅副本，visual_audit.json必须在目视审查后更新。旧DONE.json代表原流程按停止条件收尾，不代表本次B4补算完成。']
    (out/'REPORT.md').write_text('\n'.join(lines)+'\n')

def main(out):
    out=Path(out).resolve();auditdir=out/'audit';auditdir.mkdir(exist_ok=True)
    s=analyze(out)
    write(auditdir/'trajectory.json',s['trajectory']);write(auditdir/'return_comparisons.json',s['return_comparisons'])
    write(auditdir/'mean_farfield_localization.json',s['mean_farfield_localization']);write(auditdir/'failure_evidence.json',s['failures'])
    with (auditdir/'trajectory.csv').open('w') as f:
        cols=['index','id','A1','Rout','omega','classification','newton_updates','independent_relative_field_inf','phase_sensitivity_normalized','discrete_mean_norm','mean_far_w_max']
        writer=csv.DictWriter(f,fieldnames=cols,extrasaction='ignore');writer.writeheader();writer.writerows(s['trajectory'])
    write(out/'audited_points.json',dict(record_count=95,distinct_solution_count=None,points=s['points']))
    write(out/'summary.json',s);report(out,s);render(out,s)
    write(auditdir/'AUDIT_DONE.json',dict(status=s['status'],upper_B4_status=s['upper_branch_resolution_checks']['status'],raw_data_preserved=True,source_hashes=s['source_hashes'],visual_review='pending'))
    print(json.dumps(dict(raw_records=s['raw_accepted_record_count'],strict_high=s['strict_high_amplitude_point']['id'],maximum=s['maximum_record']['id'],return_start=s['verified_return_start_index'],phase_min=s['phase_gauge']['minimum_real_normalized_sensitivity'],B4=s['upper_branch_resolution_checks']['status']),ensure_ascii=False))
if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--out',default=str(ROOT/'results/fullbvp_global_20261030/phaseB'));a=p.parse_args();main(a.out)
