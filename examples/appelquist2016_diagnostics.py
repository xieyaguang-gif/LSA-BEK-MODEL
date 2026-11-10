#!/usr/bin/env python3
"""Accepted-branch diagnostics at z=1.3; no flow solve or time simulation.

The input is an explicit frozen accepted-root manifest.  RMS is evaluated from
complex Fourier coefficients interpolated in mapped Chebyshev coordinate.
"""
from __future__ import annotations
import argparse
import csv
import hashlib
import importlib.util
import json
from pathlib import Path
import shutil
import sys
import time
import numpy as np
from scipy.interpolate import BarycentricInterpolator
from scipy.optimize import brentq

ROOT=Path(__file__).resolve().parents[1]
BASE=ROOT/'results/appelquist2016_branch_20261029'
DIAG=BASE/'diagnostics'
STYLE_SOURCE=Path('/home/lthpc/.codex/skills/lthpc-python-paper-plots/scripts/lthpc_plot_style.py')
SEED=ROOT/'results/appelquist2015_hopf_20261029/corrector/N32_dr0.5/fields.npz'
SEED_GRID=ROOT/'results/appelquist2015_hopf_20261029/weakly_nonlinear/N32_dr0.5/grid_archive.npz'
REFERENCE=BASE/'literature/reference.json'
PAPER_CURVE=BASE/'literature/fig15_nl03_T6_vrms.csv'
PROBE_Z=1.3


def write(path,obj):
    path.write_text(json.dumps(obj,indent=2,allow_nan=False)+'\n')


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def absolute(path):
    p=Path(path)
    return p if p.is_absolute() else ROOT/p


def mapping(x):
    return .6*x+.4*(x**3+.5*(1-x*x))


def xi_for_height(z):
    transformed=(z-2.)/(z+2.)
    return float(brentq(lambda x:mapping(x)-transformed,-1.,1.,xtol=5e-15))


def derivative(values,dr,order=5):
    out=np.full_like(np.asarray(values,float),np.nan)
    if order==5:
        out[2:-2]=(values[:-4]-8*values[1:-3]+8*values[3:-1]-values[4:])/(12*dr)
    elif order==7:
        out[3:-3]=(-values[:-6]+9*values[1:-5]-45*values[2:-4]+45*values[4:-2]-9*values[5:-1]+values[6:])/(60*dr)
    else:raise ValueError('Only centered five/seven point derivatives are used')
    return out


def integrate_from(r,values,start):
    """Split the existing trapezoidal integral at the exact parameter a."""
    trap=getattr(np,'trapezoid',np.trapz)
    keep=r>start
    x=np.r_[start,r[keep]]
    y=np.concatenate([np.interp(start,r,values)[None],values[keep]])
    return float(trap(y,x))


def load_root(entry):
    if entry.get('accepted') is not True:
        raise ValueError('Manifest entry is not explicitly accepted')
    path=absolute(entry['fields']);status_path=absolute(entry['summary'])
    status=json.loads(status_path.read_text())
    if status.get('accepted') is not True:
        raise ValueError(f'Unaccepted status file: {status_path}')
    with np.load(path) as data:
        arrays={key:data[key].copy() for key in data.files}
    if not bool(arrays.get('accepted',False)):
        raise ValueError(f'Unaccepted NPZ: {path}')
    q=arrays.get('field',arrays.get('modes'))
    if q is None:raise ValueError('No stored Fourier field')
    h=(q.shape[1]-1)//2;n=q.shape[-1]//4-1;r=arrays['radii']
    grid_source=None
    if all(key in arrays for key in ['xi','mass','radial_weights']):
        grid={key:arrays[key] for key in ['z','xi','mass','radial_weights']}
    else:
        grid_source=absolute(entry.get('grid',SEED_GRID))
        with np.load(grid_source) as data:
            grid={key:data[key].copy() for key in ['z','xi','mass','radial_weights']}
        if not np.allclose(arrays['z'][:-1],grid['z'][:-1],rtol=0,atol=1e-12):
            raise ValueError('Explicit fallback grid does not match archived z')
    if q.shape!=(len(r),2*h+1,4*(n+1)):raise ValueError('Invalid Fourier field axes')
    if len(grid['xi'])!=n+1 or len(grid['radial_weights'])!=len(r):raise ValueError('Grid dimensions differ')
    omega=float(arrays['omega']);a=float(arrays['a'])
    return q,r,h,n,omega,a,grid,dict(fields=str(path),summary=str(status_path),
        fields_sha256=digest(path),summary_sha256=digest(status_path),
        grid_fallback=None if grid_source is None else str(grid_source),
        grid_sha256=None if grid_source is None else digest(grid_source))


def all_crossings(r,ratio,valid,threshold,rms,fund,a):
    records=[]
    for i in range(len(r)-1):
        if not(valid[i] and valid[i+1]):continue
        x,y=ratio[i]-threshold,ratio[i+1]-threshold
        if not np.isfinite(x+y):continue
        if x*y>0 or x==y:continue
        if y==0 and i+1<len(r)-1:continue
        t=float(-x/(y-x));R=float(r[i]+t*(r[i+1]-r[i]))
        records.append(dict(threshold=threshold,R=R,direction='increasing' if y>x else 'decreasing',
            normalized_vrms=float((1-t)*rms[i]+t*rms[i+1]),
            fundamental_normalized_rms=float((1-t)*fund[i]+t*fund[i+1]),
            R_minus_sponge_start=R-a,in_outer_sponge=bool(R>=a),
            bracket=[float(r[i]),float(r[i+1])]))
    return records


def intervals(r,mask):
    indices=np.flatnonzero(mask)
    if not len(indices):return []
    chunks=np.split(indices,np.flatnonzero(np.diff(indices)>1)+1)
    return [dict(start_R=float(r[c[0]]),end_R=float(r[c[-1]]),width_R=float(r[c[-1]]-r[c[0]])) for c in chunks]


def diagnose(entry,out):
    q,r,h,n,omega,a,grid,source=load_root(entry)
    m=n+1;dr=float(r[1]-r[0])
    if not np.allclose(np.diff(r),dr,rtol=0,atol=1e-10):raise ValueError('Uniform radial grid required for documented stencils')
    xi=xi_for_height(PROBE_Z)
    values=q.reshape(len(r),2*h+1,4,m)
    probe=BarycentricInterpolator(grid['xi'],values,axis=-1)(xi)
    v=probe[:,:,1]
    vp=v[:,h+1:]
    rms_j=np.sqrt(2)*abs(vp)
    total=np.sqrt(np.sum(rms_j**2,axis=1));fund=rms_j[:,0]
    higher=np.sqrt(np.sum(rms_j[:,1:]**2,axis=1)) if h>1 else np.zeros_like(total)
    mean=v[:,h].real
    floor=max(1e-12,1e-6*float(total.max()))
    active=total>floor
    stencil_valid=np.convolve(active.astype(int),np.ones(7,dtype=int),mode='same')==7
    physical=(r>420+3*dr)&(r<a-3*dr)
    growth_mask=stencil_valid&physical
    logs=np.log(np.maximum(total,1e-300))
    g5=derivative(logs,dr,5);g7=derivative(logs,dr,7)
    gp_direct=derivative(np.log(np.maximum(r*total,1e-300)),dr,5)
    gphysical=g5+1/r
    g5[~growth_mask]=np.nan;g7[~growth_mask]=np.nan;gphysical[~growth_mask]=np.nan
    denominator_floor=max(1e-12,1e-6*float(fund.max()))
    ratio_valid=(fund>denominator_floor)&(r>420+3*dr)&(r<r[-1]-3*dr)
    ratio=(rms_j[:,1]/np.maximum(fund,1e-300)) if h>1 else np.zeros_like(total)
    cross=[];regions=[]
    for threshold in [.01,.05,.10]:
        cross.extend(all_crossings(r,ratio,ratio_valid,threshold,total,fund,a))
        regions.append(dict(threshold=threshold,intervals=intervals(r,ratio_valid&physical&(ratio>=threshold))))
    velocities=values[:,h+1:,:3]
    density=np.einsum('rhvi,ij,rhvj->rh',velocities.conj(),grid['mass'],velocities).real
    energies=grid['radial_weights']@density
    outer=np.array([integrate_from(r,density[:,j],a) for j in range(h)])
    pdensity=r[:,None]**2*density
    penergies=grid['radial_weights']@pdensity
    pouter=np.array([integrate_from(r,pdensity[:,j],a) for j in range(h)])
    vdensity=r[:,None]**3*density
    venergies=grid['radial_weights']@vdensity
    vouter=np.array([integrate_from(r,vdensity[:,j],a) for j in range(h)])
    before=(r>=420)&(r<a)
    peak=int(np.argmax(total));indices=np.flatnonzero(before);peak_before=indices[np.argmax(total[before])]
    # Independent sector reconstruction, mean removal, and Parseval; no time solve.
    phase=np.linspace(0,2*np.pi,max(64,8*(2*h+1)),endpoint=False)
    numbers=np.arange(-h,h+1)
    reconstructed=np.einsum('rh,ph->rp',v,np.exp(1j*phase[:,None]*numbers)).real
    empirical=np.sqrt(np.mean((reconstructed-reconstructed.mean(axis=1)[:,None])**2,axis=1))
    parseval_abs=float(abs(empirical-total).max())
    parseval_rel=float(np.max(abs(empirical[active]-total[active])/total[active]))
    times=[0.,.17,.31]
    time_error=0.
    for T in times:
        shifted=np.einsum('rh,ph->rp',v,np.exp(1j*(phase[:,None]-omega*2*np.pi*T)*numbers)).real
        trms=np.sqrt(np.mean((shifted-shifted.mean(axis=1)[:,None])**2,axis=1))
        time_error=max(time_error,float(abs(trms-total).max()/max(total.max(),1e-300)))
    # Analytic physical-height test: (z-2)/(z+2) is the known cubic map in xi.
    analytic=np.empty(m);analytic[:-1]=(grid['z'][:-1]-2)/(grid['z'][:-1]+2);analytic[-1]=1.
    test=BarycentricInterpolator(grid['xi'],(1+2j)*analytic)(xi)
    expected=(1+2j)*(PROBE_Z-2)/(PROBE_Z+2)
    interpolation_error=float(abs(test-expected))
    exponential=BarycentricInterpolator(grid['xi'],np.exp(-grid['z']))(xi)
    derivative_test_values=.03*(r-500)+.0001*(r-500)**2
    derivative_expected=.03+.0002*(r-500)
    derivative_error=float(abs(derivative(derivative_test_values,dr,5)[3:-3]-derivative_expected[3:-3]).max())
    physical_growth_identity=float(abs((gp_direct-(derivative(logs,dr,5)+1/r))[growth_mask]).max()) if growth_mask.any() else None
    checks=dict(complex_interpolation_known_function_abs=interpolation_error,
        exponential_interpolation_abs=float(abs(exponential-np.exp(-PROBE_Z))),
        mapped_target_height=float(2*(1+mapping(xi))/(1-mapping(xi))),
        parseval_absolute=parseval_abs,parseval_relative_active=parseval_rel,
        coherent_rms_time_invariance_relative=time_error,
        mean_reconstruction_error=float(abs(reconstructed.mean(axis=1)-mean).max()),
        conjugacy_max=float(abs(q-q[:,::-1].conj()).max()),
        analytic_log_derivative_max=derivative_error,
        physical_growth_plus_one_over_R_error=physical_growth_identity,
        g5_vs_g7_abs_p95=float(np.percentile(abs(g5[growth_mask]-g7[growth_mask]),95)) if growth_mask.any() else None,
        g5_vs_g7_abs_max=float(abs(g5[growth_mask]-g7[growth_mask]).max()) if growth_mask.any() else None)
    if interpolation_error>1e-11 or parseval_rel>1e-9 or time_error>1e-9 or derivative_error>1e-10:
        raise RuntimeError(f'Diagnostic validation failed: {checks}')
    point=dict(id=entry['id'],role=entry.get('role','continuation'),paired_H2_id=entry.get('paired_H2_id'),
        paired_H3_id=entry.get('paired_H3_id'),
        H=h,n=n,dr=dr,a=a,omega=omega,omega_disk_signed=omega-68,pattern_speed_over_disk=omega/68,
        E1=float(energies[0]),amplitude=float(np.sqrt(energies[0])),harmonic_energy=energies.tolist(),
        harmonic_mass_energy_fractions=(energies/energies.sum()).tolist(),
        coherent_mass_energy=2*float(energies.sum()),
        fundamental_mass_fraction_outer=float(outer[0]/energies[0]),
        total_coherent_mass_fraction_outer=float(outer.sum()/energies.sum()),
        fundamental_velocity_square_R2_fraction_outer=float(pouter[0]/penergies[0]),
        total_coherent_velocity_square_R2_fraction_outer=float(pouter.sum()/penergies.sum()),
        fundamental_cylindrical_energy_R3_fraction_outer=float(vouter[0]/venergies[0]),
        total_coherent_cylindrical_energy_R3_fraction_outer=float(vouter.sum()/venergies.sum()),
        vrms_max=float(total[peak]),vrms_peak_R=float(r[peak]),
        unsponged_vrms_max=float(total[peak_before]),unsponged_vrms_peak_R=float(r[peak_before]),
        unsponged_peak_distance_to_a=float(a-r[peak_before]),
        unsponged_peak_near_mask_end=bool(a-r[peak_before]<3*dr),
        physical_vrms_max=float((r*total).max()),
        normalized_mean_max_abs=float(abs(mean).max()),
        z=PROBE_Z,xi_at_probe=xi,nearest_grid_z=float(grid['z'][np.argmin(abs(grid['z']-PROBE_Z))]),
        growth_amplitude_floor=floor,ratio_fundamental_floor=denominator_floor,
        growth_valid_R=[float(r[growth_mask].min()),float(r[growth_mask].max())] if growth_mask.any() else None,
        crossings=cross,ratio_regions_before_sponge=regions,checks=checks,source=source,
        accepted=True,temporal_vrms_rate_of_base_solution=0.,temporal_rate_reason='Steady rotating-wave RMS; not a perturbation stability result.')
    pointdir=out/'points';pointdir.mkdir(parents=True,exist_ok=True)
    write(pointdir/f'{entry["id"]}.json',point)
    np.savez_compressed(pointdir/f'{entry["id"]}.npz',radii=r,probe_coefficients_v=v,
        rms_normalized=total,rms_physical=r*total,fundamental_rms=fund,higher_rms=higher,
        rms_harmonics=rms_j,mean_normalized=mean,mean_physical=r*mean,
        growth_normalized=g5,growth_physical=gphysical,growth_seven_point=g7,
        growth_mask=growth_mask,ratio_2_to_1=ratio,ratio_mask=ratio_valid,
        energy_density=density,velocity_square_R2_density=pdensity,cylindrical_energy_R3_density=vdensity,
        parameters_json=json.dumps({k:point[k] for k in ['id','H','a','omega','E1','amplitude']}))
    columns=np.column_stack([r,total,r*total,fund,higher,mean,r*mean,g5,gphysical,g7,growth_mask,ratio,ratio_valid,*[rms_j[:,j] for j in range(h)]])
    header=['R','vrms_normalized','vrms_physical','fundamental_rms','higher_rms','mean_normalized','mean_physical','g_normalized','g_physical','g_seven_point','growth_mask','ratio_2_to_1','ratio_mask']+[f'rms_j{j}' for j in range(1,h+1)]
    np.savetxt(pointdir/f'{entry["id"]}.csv',columns,delimiter=',',header=','.join(header),comments='')
    return point


def run(manifest_path,out,plots=True):
    start=time.monotonic()
    manifest=json.loads(Path(manifest_path).read_text())
    out.mkdir(parents=True,exist_ok=False)
    points=[diagnose(p,out) for p in manifest['points']]
    points.sort(key=lambda p:(p['E1'],p['H']))
    reference=json.loads(REFERENCE.read_text())
    summary=dict(manifest=str(manifest_path),manifest_sha256=digest(manifest_path),points=points,
        normalization=dict(probe_z=1.3,quantity='sqrt(2 sum_j>=1 abs(q_v,j)^2), q_v=vprime/R',
            physical_quantity='R * normalized RMS in velocity units sqrt(nu*Omega_disk)',
            mean='q0 excluded from RMS; stored independently',
            mapping='a_map=2,b=.6,c=.5; interpolate complex coefficients in xi, not modulus or nearest z'),
        growth=dict(method='centered 5-point derivative of log normalized RMS; compare centered 7-point',
            amplitude_mask='vrms > max(1e-12,1e-6 * case maximum); full 7-point stencil must be valid',
            geometry_mask='R > 420+3*dr and R < a-3*dr',physical_relation='g_physical = g_normalized+1/R'),
        fronts=dict(definition='all valid crossings of abs(q_v,2)/abs(q_v,1) at 1%,5%,10%',
            interpretation='operational relative-harmonic indicators, not paper r_end',
            crossing_interpolation='linear in ratio between adjacent radial nodes',
            absolute_1e_6_paper_threshold_applied=False),
        energy_fraction=dict(definition='sum positive-harmonic velocity mass energy; both signs cancel in fraction',
            numerator='piecewise-linear radial integral from exact a to outlet',
            denominator='archived trapezoidal radial weights',
            q_mass='integral sum_j q_j^H M_z q_j dR, continuation norm',
            velocity_square_R2='integral R^2 sum_j q_j^H M_z q_j dR, physical velocity square with dR dz measure',
            cylindrical_kinetic_R3='integral R^3 sum_j q_j^H M_z q_j dR, physical velocity square times cylindrical volume Jacobian R',
            constants='the azimuthal integral, 1/2, and conjugate harmonic factors cancel in each fraction'),
        reference=str(REFERENCE),reference_sha256=digest(REFERENCE),
        no_time_evolution=True,no_global_stability_spectrum=True,
        stability_note='Frozen-radius pier-* Floquet is not full-radial global temporal stability.',
        seconds=time.monotonic()-start)
    write(out/'summary.json',summary)
    scalar_keys=['id','role','H','n','dr','E1','amplitude','a','omega','omega_disk_signed','vrms_max','vrms_peak_R','unsponged_vrms_max','unsponged_vrms_peak_R','unsponged_peak_distance_to_a','fundamental_mass_fraction_outer','total_coherent_mass_fraction_outer','fundamental_velocity_square_R2_fraction_outer','total_coherent_velocity_square_R2_fraction_outer','fundamental_cylindrical_energy_R3_fraction_outer','total_coherent_cylindrical_energy_R3_fraction_outer']
    with (out/'branch_summary.csv').open('w',newline='') as f:
        writer=csv.DictWriter(f,fieldnames=scalar_keys);writer.writeheader();writer.writerows({k:p[k] for k in scalar_keys} for p in points)
    write(out/'all_crossings.json',[dict(id=p['id'],H=p['H'],E1=p['E1'],**c) for p in points for c in p['crossings']])
    if plots:render(out)
    print(json.dumps(dict(points=len(points),seconds=time.monotonic()-start,
        max_parseval_relative=max(p['checks']['parseval_relative_active'] for p in points),
        max_interpolation_error=max(p['checks']['complex_interpolation_known_function_abs'] for p in points))))


def prepare_style():
    target=DIAG/'plot_style_reference.py'
    if not target.exists():
        DIAG.mkdir(parents=True,exist_ok=True);shutil.copyfile(STYLE_SOURCE,target)
    spec=importlib.util.spec_from_file_location('paper_style',target)
    module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)
    return module


def render(out):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from matplotlib.lines import Line2D
    out=Path(out);summary=json.loads((out/'summary.json').read_text());points=summary['points']
    ref=json.loads(REFERENCE.read_text());style=prepare_style()
    figdir=out/'figures';figdir.mkdir(exist_ok=True)
    h_colors={2:'#0072B2',3:'#009E73',4:'#D55E00',5:'#7B3294'}
    h_markers={2:'o',3:'s',4:'D',5:'^'}
    def data(point):
        with np.load(out/'points'/f'{point["id"]}.npz') as d:
            return {k:d[k].copy() for k in d.files if k!='parameters_json'}
    def note(name,brief,selected,extra=None):
        meta=dict(brief=brief,points=[p['id'] for p in selected],
            sources=[p['source'] for p in selected],normalization=summary['normalization'],
            growth=summary['growth'],fronts=summary['fronts'],
            font=style.resolve_serif_font(),canvas_cm=[18.,9.2],formats=['PDF','PNG'],
            paper_reference=str(REFERENCE),paper_reference_sha256=digest(REFERENCE),
            extra=extra or {})
        write(figdir/f'{name}.json',meta)
    def finish(fig,name,brief,selected,extra=None):
        fig.savefig(figdir/f'{name}.pdf')
        fig.savefig(figdir/f'{name}.png',dpi=350)
        plt.close(fig);note(name,brief,selected,extra)
    def axes_base(fig,axes):
        for i,ax in enumerate(axes):
            style.style_axes(ax,linewidth=.9);style.add_panel_label(ax,f'({chr(97+i)})',x_offset_pt=0,y_offset_pt=3)
            ax.tick_params(which='both',direction='in',top=True,right=True)
        fig.subplots_adjust(left=.105,right=.975,bottom=.18,top=.88,wspace=.31)
    # Representative profiles: seed, largest H2, and all H at largest amplitude.
    selected=[min(points,key=lambda p:p['E1'])]
    h2=[p for p in points if p['H']==2]
    if h2:selected.append(max(h2,key=lambda p:p['E1']))
    emax=max(p['E1'] for p in points)
    selected.extend(p for p in points if abs(p['E1']/emax-1)<1e-6)
    if len({p['id'] for p in selected})<4 and h2:
        parent=max(h2,key=lambda p:p['E1'])['id']
        selected.extend(p for p in points if p.get('paired_H2_id')==parent)
    selected=list({p['id']:p for p in selected}.values());selected.sort(key=lambda p:(p['E1'],p['H']))
    amplitudes=[]
    for p in selected:
        if not any(abs(p['E1']/e-1)<1e-6 for e in amplitudes):amplitudes.append(p['E1'])
    palette=['#0072B2','#009E73','#7B3294','#D55E00']
    def profile_style(p):
        k=next(i for i,e in enumerate(amplitudes) if abs(p['E1']/e-1)<1e-6)
        same=[x for x in selected if abs(x['E1']/p['E1']-1)<1e-6]
        ls='--' if p['H']>min(x['H'] for x in same) else '-'
        return palette[k%len(palette)],ls
    profile_handles=[]
    for p in selected:
        color,ls=profile_style(p)
        profile_handles.append(Line2D([],[],color=color,ls=ls,lw=1.5,
            label=rf'$A_1={p["amplitude"]:.3g},\ H={p["H"]}$'))
    with style.plot_style(font_size=10,axes_linewidth=.9,data_linewidth=1.5):
        # Figure 1: continuation coordinates, not time histories.
        fig,axes=plt.subplots(1,2,figsize=style.figure_size_cm(18,9.2));axes_base(fig,axes)
        for H in sorted({p['H'] for p in points}):
            subset=sorted([p for p in points if p['H']==H],key=lambda p:p['E1'])
            x=[p['amplitude'] for p in subset]
            for ax,key in zip(axes,['a','omega']):
                ax.plot(x,[p[key] for p in subset],color=h_colors.get(H,'k'),marker=h_markers.get(H,'o'),
                    ms=4.5,lw=1.2,mfc='white' if H>2 else h_colors[H],label=f'H={H}')
                ax.set_xlabel(r'Branch amplitude $A_1=\sqrt{E_1}$')
        ac=619.9581617759198
        axes[0].axhline(ac,color='.45',ls=':',lw=1)
        axes[0].text(.51,.95,r'$a_c$ (linear model)',transform=axes[0].transAxes,va='top',fontsize=8.4,color='.35')
        axes[0].set_ylabel(r'Outer sponge start $a$')
        paper_omega=ref['targets']['omega_lab']['value']
        axes[1].axhline(paper_omega,color='#9F2222',ls='-.',lw=1)
        axes[1].text(.03,.06,'Appelquist 2016: 52.36\n(different model)',transform=axes[1].transAxes,fontsize=8.2,color='#9F2222')
        axes[1].set_ylabel(r'Inertial frequency $\omega$')
        axes[0].legend(loc='lower left',fontsize=8.5,ncol=1)
        fig.text(.5,.025,'Accepted periodic branch points; the horizontal axis is not time.',ha='center',fontsize=8.4)
        finish(fig,'01_branch','Accepted branch coordinates and harmonic comparisons',points,
            dict(paper_frequency_is_different_model_reference=True,turning_is_not_automatically_a_located_fold=True))
        # Figure 2: directly comparable RMS; no derivative of a digitized curve.
        fig,axes=plt.subplots(1,2,figsize=style.figure_size_cm(18,9.2));axes_base(fig,axes)
        fig.subplots_adjust(top=.76,bottom=.18)
        amin=min(p['a'] for p in selected)
        for ax in axes:ax.axvspan(amin,730,color='.93',zorder=0)
        for p in selected:
            d=data(p);r=d['radii'];color,ls=profile_style(p);inside=r<p['a']
            axes[0].semilogy(r[inside],d['rms_normalized'][inside],color=color,ls=ls)
            axes[0].semilogy(r[~inside],d['rms_normalized'][~inside],color=color,ls=ls,alpha=.4)
            axes[1].plot(r,d['growth_normalized'],color=color,ls=ls)
            axes[0].plot(p['a'],np.interp(p['a'],r,d['rms_normalized']),marker='v',color=color,ms=4,ls='none')
        paper_extra={}
        handles=profile_handles.copy()
        if PAPER_CURVE.exists():
            paper=np.genfromtxt(PAPER_CURVE,delimiter=',',names=True)
            axes[0].fill_between(paper['R'],paper['vrms_plot_lower'],paper['vrms_plot_upper'],color='#9F2222',alpha=.13,lw=0)
            axes[0].semilogy(paper['R'],paper['vrms_normalized'],color='#9F2222',ls='-.',lw=1.25)
            handles.append(Line2D([],[],color='#9F2222',ls='-.',label='DNS NL03, T=6'))
            paper_extra=dict(source=str(PAPER_CURVE),sha256=digest(PAPER_CURVE),
                no_shift_no_smoothing_no_extrapolation=True,band='graphical reading tolerance, not DNS variability',
                no_paper_growth_derivative=True,
                scope='DNS total finite-time RMS including turbulence; BVP coherent harmonics only; different boundaries.')
        axes[0].set_xlim(480,710);axes[0].set_ylim(1e-7,.2)
        axes[1].set_xlim(480,650);axes[1].axhline(0,color='.55',lw=.6)
        axes[0].set_ylabel(r'$v_{\mathrm{rms},N}(R,z=1.3)$')
        axes[1].set_ylabel(r'$g_v=\partial_R\ln v_{\mathrm{rms},N}$')
        for ax in axes:ax.set_xlabel(r'$R$')
        fig.legend(handles=handles,loc='upper center',bbox_to_anchor=(.5,1.0),ncol=3,fontsize=8.2,handlelength=2.5,columnspacing=1.)
        fig.text(.5,.025,'DNS: total azimuthal RMS at T=6; BVP: finite-H coherent RMS. Faded: after each sponge onset.',ha='center',fontsize=7.7)
        finish(fig,'02_vrms_growth','RMS at the same height and normalization, with masked spatial growth',selected,paper_extra)
        # Figure 3: component-wise harmonic content and separate mean correction.
        fig,axes=plt.subplots(1,2,figsize=style.figure_size_cm(18,9.2));axes_base(fig,axes)
        fig.subplots_adjust(top=.79)
        for ax in axes:ax.axvspan(amin,730,color='.93',zorder=0)
        for p in selected:
            d=data(p);r=d['radii'];color,ls=profile_style(p)
            mask=d['ratio_mask']&(d['ratio_2_to_1']>0)
            axes[0].semilogy(r[mask],d['ratio_2_to_1'][mask],color=color,ls=ls)
            axes[1].plot(r,d['mean_normalized'],color=color,ls=ls)
        for threshold in [.01,.05,.1]:
            axes[0].axhline(threshold,color='.55',ls=':',lw=.7)
            axes[0].text(523,threshold*1.08,f'{threshold:.0%}',fontsize=8,color='.4')
        axes[0].set_xlim(520,700);axes[0].set_ylim(1e-4,1)
        axes[1].set_xlim(480,730);axes[1].axhline(0,color='.5',lw=.6)
        axes[0].set_ylabel(r'$|q_{v,2}|/|q_{v,1}|$')
        axes[1].set_ylabel(r'Mean change $\Delta V_N=q_{v,0}$')
        for ax in axes:ax.set_xlabel(r'$R$')
        fig.legend(handles=profile_handles,loc='upper center',bbox_to_anchor=(.5,1.0),ncol=2,fontsize=8.4)
        fig.text(.5,.025,'The mean is excluded from RMS and is not damped by the sponge.',ha='center',fontsize=8.4)
        finish(fig,'03_harmonics_mean','Relative harmonic content and the separately stored mean change',selected,
            dict(thresholds_are_operational=True,mean_is_not_damped=True))
        # Figure 4: every valid crossing, and energy remaining in the sponge.
        fig,axes=plt.subplots(1,2,figsize=style.figure_size_cm(18,9.2));axes_base(fig,axes)
        fig.subplots_adjust(top=.84,bottom=.23)
        threshold_colors={.01:'#0072B2',.05:'#009E73',.1:'#7B3294'}
        for p in points:
            for cross in p['crossings']:
                color=threshold_colors[cross['threshold']]
                axes[0].scatter(p['E1'],cross['R'],s=17,
                    marker='^' if cross['direction']=='increasing' else 'v',
                    facecolors='none' if cross['in_outer_sponge'] else color,edgecolors=color,linewidths=.65)
        main=sorted([p for p in points if p['role']!='harmonic_check'],key=lambda p:p['E1'])
        axes[0].semilogx([p['E1'] for p in main],[p['a'] for p in main],color='.35',ls='--',lw=.8)
        axes[0].text(5e-5,615.,r'$a(E_1)$',fontsize=8.3,color='.35')
        axes[0].set_ylabel(r'All ratio-crossing radii $R$')
        axes[0].set_xlabel(r'Branch energy $E_1$')
        th_handles=[Line2D([],[],ls='none',marker='^',color=threshold_colors[t],label=f'{t:.0%}') for t in threshold_colors]
        axes[0].legend(handles=th_handles,loc='lower left',bbox_to_anchor=(0,1.05),fontsize=8.2,ncol=3,handletextpad=.3,columnspacing=.8)
        for H in sorted({p['H'] for p in points}):
            group=sorted([p for p in points if p['H']==H],key=lambda p:p['E1'])
            x=[p['E1'] for p in group];color=h_colors.get(H,'k')
            axes[1].semilogx(x,[100*p['total_coherent_mass_fraction_outer'] for p in group],color=color,marker=h_markers.get(H,'o'),ms=4,lw=1.2,label=f'H={H}')
            axes[1].semilogx(x,[100*p['total_coherent_cylindrical_energy_R3_fraction_outer'] for p in group],color=color,ls='--',lw=1)
        axes[1].set_ylim(70,100);axes[1].set_ylabel(r'Coherent energy at $R\geq a$ (%)')
        axes[1].set_xlabel(r'Branch energy $E_1$');axes[1].legend(loc='lower left',fontsize=8.5)
        fig.text(.5,.08,'Crossings: filled R < a, open R ≥ a; upward / downward markers give crossing direction.',ha='center',fontsize=7.7)
        fig.text(.5,.025,'Energy: solid q mass norm; dashed cylindrical kinetic energy (R³ weight). All coherent harmonics.',ha='center',fontsize=7.7)
        finish(fig,'04_crossings_sponge_energy','All operational crossings and the large energy fraction inside the sponge',points,
            dict(no_single_crossing_is_named_a_front=True,solid='q mass norm',dashed='cylindrical kinetic energy R^3 weighting'))
    pointwise_comparison(out,points)
    interval_growth_comparison(out)
    comparison_checks(out,points)
    write_report(out,summary,ref)
    write(figdir/'render_summary.json',dict(figures=4,formats=['png','pdf'],font=style.resolve_serif_font(),
        canvas_cm=[18,9.2],dpi_png=350,selected_profiles=[p['id'] for p in selected],
        visual_QA='pending manual inspection',time_evolution_plot_generated=False))
    print(json.dumps(dict(rendered_figures=4,selected=[p['id'] for p in selected],font=style.resolve_serif_font())))


def pointwise_comparison(out,points):
    if not PAPER_CURVE.exists():return
    paper=np.genfromtxt(PAPER_CURVE,delimiter=',',names=True)
    rows=[]
    for H in sorted({p['H'] for p in points if p['H']>=3}):
        p=max((x for x in points if x['H']==H),key=lambda x:x['E1'])
        with np.load(out/'points'/f'{p["id"]}.npz') as data:
            r=data['radii'];positive=data['probe_coefficients_v'][:,H+1:]
        for R in [570.,582.8,600.,620.]:
            if not paper['R'].min()<=R<=paper['R'].max():continue
            start=min(max(np.searchsorted(r,R)-2,0),len(r)-4)
            amplitude=BarycentricInterpolator(r[start:start+4],positive[start:start+4],axis=0)(R)
            vrms=float(np.sqrt(2*np.sum(abs(amplitude)**2)))
            source=float(10**np.interp(R,paper['R'],paper['log10_vrms']))
            lower=float(10**np.interp(R,paper['R'],np.log10(paper['vrms_plot_lower'])))
            upper=float(10**np.interp(R,paper['R'],np.log10(paper['vrms_plot_upper'])))
            dx=float(np.interp(R,paper['R'],paper['R_plot_reading_tolerance']))
            lo=max(R-dx,float(paper['R'].min()));hi=min(R+dx,float(paper['R'].max()))
            sample=np.r_[lo,paper['R'][(paper['R']>lo)&(paper['R']<hi)],hi]
            coord_lower=float(np.min(10**np.interp(sample,paper['R'],np.log10(paper['vrms_plot_lower']))))
            coord_upper=float(np.max(10**np.interp(sample,paper['R'],np.log10(paper['vrms_plot_upper']))))
            rows.append(dict(id=p['id'],H=H,E1=p['E1'],R=R,a=p['a'],BVP_in_outer_sponge=bool(R>=p['a']),
                paper_in_outer_sponge=False,BVP_vrms=vrms,paper_NL03_T6_vrms=source,ratio=vrms/source,
                ratio_vertical_lower=vrms/upper,ratio_vertical_upper=vrms/lower,
                ratio_coordinate_and_vertical_lower=vrms/coord_upper,ratio_coordinate_and_vertical_upper=vrms/coord_lower,
                paper_R_reading_tolerance=dx,paper_vertical_factor=upper/source))
    write(out/'pointwise_paper_comparison.json',dict(rows=rows,paper_source=str(PAPER_CURVE),
        interpolation='BVP: local four-node polynomial of complex coefficients then Parseval; paper: linear log10 RMS along extracted vector segments',
        uncertainty='graphical reading tolerance only, not statistical or model uncertainty',
        scope='NL03 T=6 total DNS RMS vs periodic coherent BVP RMS, different disk-edge/boundary models'))
    if rows:
        with (out/'pointwise_paper_comparison.csv').open('w',newline='') as f:
            w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)


def comparison_checks(out,points):
    results=[]
    byid={p['id']:p for p in points}
    for p in points:
        parent=p.get('paired_H3_id') or p.get('paired_H2_id')
        if parent is None or parent not in byid:continue
        before=byid[parent]
        with np.load(out/'points'/f'{parent}.npz') as x, np.load(out/'points'/f'{p["id"]}.npz') as y:
            r=x['radii'];mask=(r>420+3*p['dr'])&(r<min(p['a'],before['a'])-3*p['dr'])
            diff=y['rms_normalized'][mask]-x['rms_normalized'][mask]
            rmsrel=float(np.linalg.norm(diff)/np.linalg.norm(x['rms_normalized'][mask]))
            mean_diff=float(np.sqrt(np.sum((y['mean_normalized'][mask]-x['mean_normalized'][mask])**2)*p['dr']))
        results.append(dict(first=parent,second=p['id'],delta_a=p['a']-before['a'],delta_omega=p['omega']-before['omega'],
            E1_relative_change=p['E1']/before['E1']-1,common_undamped_probe_rms_relative_l2=rmsrel,
            common_undamped_mean_difference_l2=mean_diff))
    write(out/'harmonic_comparisons.json',results)


def interval_growth_comparison(out):
    pointwise=json.loads((out/'pointwise_paper_comparison.json').read_text())['rows']
    results=[]
    for H in sorted({x['H'] for x in pointwise}):
        left=next(x for x in pointwise if x['H']==H and x['R']==570.)
        right=next(x for x in pointwise if x['H']==H and x['R']==600.)
        width=right['R']-left['R']
        g=float(np.log(right['BVP_vrms']/left['BVP_vrms'])/width)
        paper=float(np.log(right['paper_NL03_T6_vrms']/left['paper_NL03_T6_vrms'])/width)
        def bound(row,lower,coordinate=False):
            key='ratio_coordinate_and_vertical_' if coordinate else 'ratio_vertical_'
            return row['BVP_vrms']/row[key+('upper' if lower else 'lower')]
        ranges={}
        for coordinate in [False,True]:
            lower=float(np.log(bound(right,True,coordinate)/bound(left,False,coordinate))/width)
            upper=float(np.log(bound(right,False,coordinate)/bound(left,True,coordinate))/width)
            ranges['paper_coordinate_and_vertical_range' if coordinate else 'paper_vertical_range']=[lower,upper]
        results.append(dict(H=H,id=left['id'],R_left=570.,R_right=600.,
            BVP_gbar_normalized=g,paper_gbar_normalized=paper,
            BVP_gbar_physical=g+np.log(600/570)/30,
            paper_gbar_physical=paper+np.log(600/570)/30,
            both_BVP_endpoints_before_sponge=True,**ranges))
    write(out/'interval_growth_comparison.json',dict(rows=results,
        formula='[ln vrms_N(600)-ln vrms_N(570)]/30; unit 1/R',
        uncertainty='Graphical-reading envelope, not statistical confidence: each endpoint vertical factor 1.04257; additional coordinate envelope samples the digitized curve within +/-0.59198 in R at each fixed nominal station; denominator remains nominal interval30',
        no_pointwise_DNS_derivative=True))
    with (out/'interval_growth_comparison.csv').open('w',newline='') as f:
        w=csv.DictWriter(f,fieldnames=list(results[0]));w.writeheader();w.writerows(results)


def write_report(out,summary,reference):
    points=summary['points'];largest=max((p for p in points if p['H']==max(x['H'] for x in points)),key=lambda p:p['E1'])
    target=reference['targets'];local=target['project_local_absolute_onset_R']['value']
    quantities=[
        dict(quantity='Local laminar absolute threshold',paper=507.,current=local,coordinate='R',Re_r_current=local**2,comparison='Paper value rounded; project independently audited local pinch threshold, not a global nonlinear front'),
        dict(quantity='Estimated asymptotic nonlinear front',paper=582.8,current=None,coordinate='R',Re_r_current=None,comparison='Paper extrapolation; no matched asymptotic front established by this branch'),
        dict(quantity='Finite-time NL03 front',paper=570.,current=None,coordinate='R',Re_r_current=None,comparison='Paper T=6 estimate; custom relative-harmonic crossings are different observables'),
        dict(quantity='Linear Hopf damping-start threshold',paper=None,current=619.9581617759198,coordinate='a',Re_r_current=None,comparison='Threshold of numerical boundary parameter, not a critical flow Reynolds number'),
        dict(quantity='Inertial coherent frequency',paper=52.36,current=largest['omega'],coordinate='omega per disk time t',Re_r_current=None,comparison='Different models and boundaries'),
        dict(quantity='Signed disk-frame frequency',paper=-15.64,current=largest['omega_disk_signed'],coordinate='omega-68',Re_r_current=None,comparison='Same signed frequency conversion'),
        dict(quantity='Local absolute temporal rate at R582.8',paper=.568,current=None,coordinate='per t',Re_r_current=None,comparison='Local theoretical linear absolute rate; not nonlinear global stability or final saturated growth'),
        dict(quantity='Base rotating-wave RMS temporal rate',paper=None,current=0.,coordinate='per t and per rotation',Re_r_current=None,comparison='Kinematic identity of a steady rotating wave; no perturbation stability computed')]
    write(out/'quantity_comparison.json',dict(rows=quantities,time='t=Omega_disk*t_dim; T=t/(2*pi); a rate per rotation is 2*pi times its per-t value',
        reynolds='R=r_dim*sqrt(Omega_disk/nu); Re_r=R^2 when using local disk speed and radius'))
    with (out/'quantity_comparison.csv').open('w',newline='') as f:
        w=csv.DictWriter(f,fieldnames=list(quantities[0]));w.writeheader();w.writerows(quantities)
    growth=[]
    for H in sorted({p['H'] for p in points if p['H']>=3}):
        p=max((x for x in points if x['H']==H),key=lambda x:x['E1'])
        with np.load(out/'points'/f'{p["id"]}.npz') as d:
            r=d['radii'];mask=d['growth_mask']
            for R in [570.,582.8,600.,620.]:
                inside=r[mask].min()<=R<=r[mask].max()
                gn=float(np.interp(R,r[mask],d['growth_normalized'][mask])) if inside else None
                growth.append(dict(id=p['id'],H=H,R=R,valid=bool(inside),g_normalized=gn,
                    g_physical=gn+1/R if inside else None,BVP_in_outer_sponge=bool(R>=p['a'])))
    write(out/'pointwise_spatial_growth.json',dict(rows=growth,unit='per R',paper_curve_differentiated=False))
    comparison=json.loads((out/'pointwise_paper_comparison.json').read_text())['rows']
    turns=json.loads((BASE/'turning_check/summary.json').read_text()) if (BASE/'turning_check/summary.json').exists() else {}
    q=f"""# Accepted-branch diagnostics and Appelquist 2016 comparison

The largest accepted H4 wave still places **{100*largest['total_coherent_mass_fraction_outer']:.2f}% of its coherent velocity q norm and {100*largest['total_coherent_cylindrical_energy_R3_fraction_outer']:.2f}% of its coherent cylindrical kinetic energy inside the outer sponge**. Its normalized RMS peaks at R={largest['vrms_peak_R']:.1f}, beyond a={largest['a']:.6f}; the largest value before the outer sponge occurs at R={largest['unsponged_vrms_peak_R']:.1f}, only {largest['unsponged_peak_distance_to_a']:.3f} upstream of a. These accepted finite-domain roots establish a nonlinear branch, while natural saturation and independence from the boundary remain unverified.

The frozen manifests contain {len(points)} accepted original-BVP roots, all N32, dr=0.5, with H2–H4. Failed checkpoints are excluded. All fields and status files are read-only inputs with SHA256 provenance in each diagnostic point. A1=sqrt(E1) is a continuation amplitude, not time. H3 turns in the damping-start parameter and the coarse H4 samples retain the change of direction; no precise fold or temporal stability exchange was computed. The H3/H4 comparison at A1=0.204 changes omega by {turns.get('comparisons',[{}])[-1].get('delta_omega',float('nan')):.6g}; the source solver comparison reports q1 mass difference 0.347% and mean mass difference 5.32%. The frequency and coherent probe RMS are less sensitive to H than the full mean correction.

The field convention is v'/R=sum_j q_v,j exp(ij(68 theta-omega t)). Complex coefficients are interpolated in mapped Chebyshev coordinate to physical z=1.3 before taking moduli. The exact mapping is a_map=2, b=0.6, c=0.5; the nearest grid node z=1.4212 is not used. The azimuthal RMS is sqrt(2 sum_(j>=1)|q_v,j|^2), with q0 excluded. Physical dimensionless RMS in units sqrt(nu Omega_disk) is R times this normalized RMS. Fundamental, higher-harmonic, total coherent RMS, and the mean are saved separately.

There are three distinct energy measures in the archive: q mass norm integral sum|q_velocity,j|^2 dz dR; R²-weighted physical velocity square with the same dz dR measure; and R³-weighted cylindrical kinetic energy, including the volume Jacobian R. Mean and laminar flow are excluded from all coherent energy fractions. Multiplicative azimuthal and conjugate-pair factors cancel in each fraction. The numerator is integrated from exact a with the existing piecewise-linear radial representation; the denominator uses the archived radial mass weights.

The spatial rate g_N=partial_R log(vrms_N) uses a centered five-point stencil checked against seven points. A full seven-point amplitude mask requires vrms>max(1e-12,1e-6 times each profile maximum); the geometric mask is R>420+3dr and R<a-3dr. Physical-velocity growth is g_N+1/R. Rates in the sponge or tiny tails are masked. The extracted DNS RMS curve is not differentiated into an apparent exact DNS growth curve.

The following ratios use the digitized red NL03 T=6 curve from figure 15. Values are normalized by local disk speed and evaluated at z=1.3 in both data sets. The interval accounts only for the graph-reading factor 1.04257 (about 4.3%); the machine-readable table additionally includes the horizontal reading tolerance Delta R=0.592. No extrapolation, radial shift, smoothing, or amplitude rescaling was used.

| H | R | Coherent BVP RMS | DNS total RMS | BVP / DNS | Graph-reading interval | BVP outer sponge |
|---:|---:|---:|---:|---:|---:|:---|
"""
    for r in comparison:
        q+=f"| {r['H']} | {r['R']:.1f} | {r['BVP_vrms']:.6g} | {r['paper_NL03_T6_vrms']:.6g} | {r['ratio']:.5f} | {r['ratio_vertical_lower']:.5f}–{r['ratio_vertical_upper']:.5f} | {'yes' if r['BVP_in_outer_sponge'] else 'no'} |\n"
    q+='\nA robust mean spatial growth comparison over R=570–600 uses only endpoint logarithms: g_bar=[ln vrms_N(600)−ln vrms_N(570)]/30. Both endpoints precede the BVP outer sponge. The graph-reading envelopes treat endpoint errors independently and are not statistical confidence intervals. The coordinate envelope samples the paper curve within ±0.592 of each nominal station, keeping the nominal interval30.\n\n| H | BVP mean growth | DNS mean growth | DNS vertical-reading range | DNS coordinate + vertical envelope |\n|---:|---:|---:|:---|:---|\n'
    for g in json.loads((out/'interval_growth_comparison.json').read_text())['rows']:
        v=g['paper_vertical_range'];c=g['paper_coordinate_and_vertical_range']
        q+=f"| {g['H']} | {g['BVP_gbar_normalized']:.7f} | {g['paper_gbar_normalized']:.7f} | {v[0]:.7f}–{v[1]:.7f} | {c[0]:.7f}–{c[1]:.7f} |\n"
    q+="""
The DNS observable is the total azimuthal RMS at finite T=6; this BVP retains only coherent harmonics of one rotating wave. Direct comparison is most informative upstream where the fundamental dominates. Downstream differences also include DNS broadband fluctuations and higher azimuthal wavenumbers. NL03 has a finite-thickness disk edge at R=700, outlet800, and a V/W sponge from765 with maximum8; this BVP has no physical disk edge, outlet730, harmonic sponge from a with maximum28, and undamped mean. Every tabulated paper station lies before its outer sponge. The smaller RMS ratios upstream therefore exceed the graph-reading discrepancy, but are not an error estimate between equivalent models.

| Quantity | Paper or audited reference | Current interpretation |
|:---|:---|:---|
| Local absolute onset | Paper R≈507; project R=507.39 | A local laminar pinch threshold; Re_r=R²≈257444 |
| Asymptotic nonlinear front | Estimated R=582.8 | Extrapolated paper front, not directly reached here |
| NL03 finite-time front | R≈570 at T=6 | A finite-time estimate, distinct from 582.8 |
| Linear global neutral parameter | Current a_c=619.95816 | Sponge-start threshold; not critical Reynolds |
"""
    q+=f"| Inertial frequency | Paper52.36 | H4 {largest['omega']:.8f}; different models |\n"
    q+=f"| Signed disk frequency | Paper−15.64 | H4 {largest['omega_disk_signed']:.8f}, omega−68 |\n"
    q+="""| Local temporal absolute rate at R582.8 | Paper0.568 per t | Local theory, not final nonlinear growth |
| Base-wave RMS temporal rate | Zero identically | An ansatz property, not a stability result |

Here t=Omega_disk*t_dim and T=t/(2pi). A temporal growth rate per revolution equals 2pi times the rate per t, even when a paper plots the per-t rate against T. No time evolution or global temporal stability spectrum is inferred from this branch. Frozen-radius pier-* Floquet modes are not the global temporal stability of the full radial wave.

Every valid 2beta/beta amplitude-ratio crossing at 1%, 5%, and 10% is saved with direction, RMS, distance to a, and whether it lies in the sponge. These are custom relative-harmonic indicators, not the paper's front criterion. Multiple crossings are retained. The author's absolute spectral threshold around1e-6 is not applied because its FFT/spectral normalization cannot be mapped uniquely to these complex coefficients.

"""
    q+='At the largest H4 point the crossings are:\n\n| Ratio | R | Direction | R−a | RMS | In outer sponge |\n|---:|---:|:---|---:|---:|:---|\n'
    for c in largest['crossings']:
        q+=f"| {c['threshold']:.0%} | {c['R']:.4f} | {c['direction']} | {c['R_minus_sponge_start']:.4f} | {c['normalized_vrms']:.5g} | {'yes' if c['in_outer_sponge'] else 'no'} |\n"
    maxcheck=lambda k:max(p['checks'][k] for p in points)
    q+=f"\nValidation over {len(points)} points: maximum active-tail Parseval relative error {maxcheck('parseval_relative_active'):.3g}; known complex interpolation error {maxcheck('complex_interpolation_known_function_abs'):.3g}; coherent RMS invariance under time/phase shifts {maxcheck('coherent_rms_time_invariance_relative'):.3g}; analytic log-derivative error {maxcheck('analytic_log_derivative_max'):.3g}. These verify postprocessing definitions, not resolution of the underlying nonlinear PDE.\n"
    q+="""
Four reproducible figure pairs (PNG/PDF) show the accepted branch, RMS and masked spatial growth, harmonic ratios and the separate mean, and all crossings together with energy inside the sponge. Each figure has source/mask/font metadata in its JSON sidecar. Times New Roman with STIX mathematics, 18×9.2 cm, PNG350dpi, vector PDF. `visual_QA.json` records the final image inspection. `commands.sh` reproduces diagnostics into a new output directory without overwriting the frozen archive.

Primary tables: `branch_summary.csv`, `pointwise_paper_comparison.csv`, `interval_growth_comparison.csv`, `quantity_comparison.csv`, `all_crossings.json`, `pointwise_spatial_growth.json`, and `harmonic_comparisons.json`. Each `points/` file includes complete diagnostic profiles and source hashes. The earlier `../accepted_branch/` folder is retained as an intermediate 14-point result; the final 19-point release is this `combined_branch/` directory. Its R² measure is explicitly named velocity-square; only R³ includes cylindrical volume.
"""
    (out/'REPORT.md').write_text(q)


def main():
    ap=argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--manifest',type=Path)
    ap.add_argument('--out',type=Path)
    ap.add_argument('--seed-checks',action='store_true')
    ap.add_argument('--no-plots',action='store_true')
    ap.add_argument('--render-only',type=Path)
    args=ap.parse_args()
    if args.render_only:render(args.render_only);return
    if args.seed_checks:
        DIAG.mkdir(parents=True,exist_ok=True)
        status=SEED.parent/'summary.json';s=json.loads(status.read_text())
        manifest=DIAG/'development_seed_manifest.json'
        write(manifest,dict(points=[dict(id='accepted_seed',accepted=True,fields=str(SEED),summary=str(status),
            grid=str(SEED_GRID),role='seed',E1=s['target_energy'])]))
        run(manifest,args.out or DIAG/'development_seed',plots=False);return
    if args.manifest is None or args.out is None:ap.error('--manifest and --out required')
    run(args.manifest,args.out,plots=not args.no_plots)


if __name__=='__main__':main()
