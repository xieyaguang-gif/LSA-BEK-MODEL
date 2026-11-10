"""Plot and quantify the archived beta42 spatial study; never fit model coefficients."""
import argparse
import json
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
from scipy.ndimage import gaussian_filter1d
from scipy.optimize import minimize_scalar


def curve(re, amplitude):
    """Same log-amplitude filter for DNS and PSE: sigma_Re=2, spacing=.25."""
    x = np.arange(re[0], re[-1]+1e-7, .25)
    loga = gaussian_filter1d(np.interp(x, re, np.log(np.maximum(amplitude, 1e-30))),
                             8, mode='reflect', truncate=4)
    return x, np.exp(loga), .751*np.gradient(loga, x, edge_order=2)


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--root', type=Path, default=Path('results/beta42_nonlinear_spatial'))
    a = p.parse_args()
    root = a.root
    def load(name):
        return np.genfromtxt(root/name/'march.csv', delimiter=',', names=True)
    def smooth(d):
        return curve(d['Re'], d['A1_vtheta_over_disk_speed'])
    dns = np.loadtxt(root/'inputs/af01_semtex_turn4_vtheta.csv', delimiter=',')
    mask = (dns[:,1] >= 280)&(dns[:,1] <= 520)
    dc = curve(dns[mask,1], dns[mask,4])
    main_data = load('primary')
    mc = smooth(main_data)
    lst = np.genfromtxt(root/'inputs/llsa_beta42_stationary.csv', delimiter=',', names=True)
    linear = load('checks/linear')
    lc = smooth(linear)
    steps = [smooth(load(k)) for k in ('checks/n48h5_dr5', 'primary', 'checks/n64h7_dr8')]
    evaluation = np.arange(330, 430.01, .25)
    sample = lambda c, col, x=evaluation: np.interp(x, c[0], c[col])
    rmse = lambda x: float(np.sqrt(np.mean(np.asarray(x)**2)))
    metrics = {'growth_definition':'d ln[sqrt(2)|v_beta42/(Omega_disk*r)|]/d r_delta',
               'filter':'linear interpolation of ln(A), spacing_Re=.25, Gaussian sigma_Re=2; reflect endpoints excluded from metrics',
               'fit':'one DNS inlet amplitude at Re=320; no downstream fitting in the PDE',
               'intervals':{}}
    for lo, hi in ((330,360),(365,385),(385,425)):
        x = evaluation[(evaluation >= lo)&(evaluation <= hi)]
        metrics['intervals'][f'{lo}_{hi}'] = {
            'growth_rmse':rmse(sample(mc,2,x)-sample(dc,2,x)),
            'mean_amplitude_relative_error':float(np.mean(abs(sample(mc,1,x)/sample(dc,1,x)-1)))}
    for key,c in [('DNS',dc),('PSE',mc)]:
        active = (c[0] >= 350)&(c[0] <= 425)
        i = np.flatnonzero(active)[np.argmax(c[1][active])]
        metrics[key+'_smoothed_peak'] = {'Re':float(c[0][i]),'amplitude':float(c[1][i])}
    metrics['point_comparison'] = [dict(Re=re, DNS_A=float(sample(dc,1,[re])[0]),
        PSE_A=float(sample(mc,1,[re])[0]), DNS_growth=float(sample(dc,2,[re])[0]),
        PSE_growth=float(sample(mc,2,[re])[0])) for re in (340,360,370,380,390,400,410,420,430)]

    plt.rcParams.update({'font.family':'serif','font.serif':['Times New Roman','DejaVu Serif'],
        'mathtext.fontset':'stix','font.size':12,'axes.labelsize':15,'axes.linewidth':1.1,
        'xtick.direction':'in','ytick.direction':'in','xtick.top':True,'ytick.right':True,
        'legend.frameon':False,'legend.fontsize':10,'savefig.dpi':230,'pdf.fonttype':42})
    colors = {'main':'#1764a0','linear':'#777777','half':'#bf6b19','double':'#16856f'}
    def axes_pair():
        fig,ax=plt.subplots(1,2,figsize=(11,4.4),layout='constrained')
        for i,axis in enumerate(ax):
            axis.set_xlim(330,430);axis.set_xlabel(r'$Re=|Ro|r_\delta$')
            axis.text(.02,.97,f'({chr(97+i)})',transform=axis.transAxes,va='top')
        return fig,ax
    def save(fig,name):
        for ext in ('png','pdf'):
            fig.savefig(root/f'{name}.{ext}',facecolor='white')
        plt.close(fig)
    fig,ax=axes_pair()
    for panel,col in ((0,1),(1,2)):
        spread = np.asarray([sample(c,col) for c in steps])
        ax[panel].fill_between(evaluation,spread.min(axis=0),spread.max(axis=0),
                              color=colors['main'],alpha=.18,label=r'PSE step spread ($\Delta r_\delta=5$--$8$)')
        ax[panel].plot(dc[0],dc[col],color='black',lw=1.8,label=r'AF0.1 DNS, stationary $\beta=42$')
        ax[panel].plot(mc[0],mc[col],color=colors['main'],lw=2,label=r'Nonlinear PSE, $N=64$, $H=7$')
    ax[0].set_ylabel(r'$A_{42}=\sqrt{2}|v_{42}/(\Omega_d r)|$')
    ax[0].set_ylim(0,.135);ax[0].legend(loc='upper left',bbox_to_anchor=(.02,.88))
    ax[1].plot(lst['Re'],lst['growth_minus_alpha_i_paper'],'--',color=colors['linear'],lw=1.5,label='Local linear stability')
    ax[1].plot(lc[0],lc[2],':',color='#aa5252',lw=1.5,label='Linear PSE')
    ax[1].axhline(0,color='.7',lw=.8)
    ax[1].set_ylabel(r'$g_{42}=\mathrm{d}\ln A_{42}/\mathrm{d}r_\delta$')
    ax[1].set_ylim(-.095,.11);ax[1].legend(loc='lower left',fontsize=9)
    save(fig,'af01_comparison')

    validation={}
    fig,ax=axes_pair()
    ax[0].plot(lst['Re'],lst['growth_minus_alpha_i_paper'],'--',color='.5',label='Local linear stability')
    ax[0].plot(lc[0],lc[2],color=colors['main'],label='Linear PSE')
    small=load('checks/small_nonlinear');sc=smooth(small)
    ax[0].plot(sc[0],sc[2],':',color=colors['double'],lw=2,label=r'Nonlinear PSE, $10^{-6} A_{in}$')
    ax[0].set_ylabel(r'$g_{42}$');ax[0].set_ylim(.05,.10);ax[0].legend(loc='lower center')
    validation['small_amplitude_limit_max_relative_A_difference']=float(np.max(abs(small['A1_vtheta_over_disk_speed']/linear['A1_vtheta_over_disk_speed']-1)))
    validation['linear_PSE_vs_local_growth_rmse_330_430']=rmse(sample(lc,2)-np.interp(evaluation,lst['Re'],lst['growth_minus_alpha_i_paper']))
    ref=main_data['A1_vtheta_over_disk_speed']
    for name,label,style in [('checks/n48h5_dr6','$N=48,H=5$','--'),('checks/n48h7_dr6','$N=48,H=7$',':')]:
        d=load(name)
        relative=(d['A1_vtheta_over_disk_speed']/ref-1)*100
        ax[1].plot(d['Re'],relative,style,lw=1.8,label=label)
        validation[name]={'max_relative_A_difference':float(np.max(abs(relative))/100)}
    ax[1].axhline(0,color='.7',lw=.8);ax[1].set_ylabel(r'$100(A/A_{N64,H7}-1)$ [%]')
    ax[1].legend(loc='upper left',bbox_to_anchor=(.03,.86))
    save(fig,'numerical_checks')
    for key in ('checks/n48h5_dr5','checks/n64h7_dr8'):
        c=smooth(load(key));v=sample(c,1)/sample(mc,1)-1
        validation[key]={'max_relative_A_difference_330_430':float(np.max(abs(v))),
                         'growth_rmse_330_430':rmse(sample(c,2)-sample(mc,2))}
    metrics['validation']=validation

    fig,ax=axes_pair(); inlet={}; base=smooth(load('inlet/reference'))
    for name,label,color in [('half',r'$0.5 A_{in}$',colors['half']),('reference',r'$A_{in}$',colors['main']),('double',r'$2 A_{in}$',colors['double'])]:
        c=smooth(load('inlet/'+name));x=np.arange(365,410.01,.25)
        objective=lambda shift: np.mean((sample(c,2,x+shift)-sample(base,2,x))**2)
        fit=minimize_scalar(objective,bounds=(-12,12),method='bounded') if name!='reference' else None
        shift=float(fit.x) if fit is not None else 0.
        inlet[name]={'shift_Re_left':shift,'growth_rmse_before':float(np.sqrt(objective(0))),
                     'growth_rmse_after':float(np.sqrt(objective(shift)))}
        # Discard a 3-sigma margin before translating: reflection at a truncated
        # endpoint must not masquerade as a physical recovery of the growth rate.
        interior=(c[0]>=c[0][0]+6)&(c[0]<=c[0][-1]-6)
        ax[0].plot(c[0][interior],c[2][interior],color=color,lw=1.8,label=label)
        ax[1].plot(c[0][interior]-shift,c[2][interior],color=color,lw=1.8,label=label+f', shift {shift:+.2f}')
    for axis in ax:
        axis.axhline(0,color='.7',lw=.8);axis.set_ylabel(r'$g_{42}$');axis.set_ylim(-.075,.10)
        axis.legend(loc='lower left',fontsize=10)
    ax[1].set_xlabel(r'$Re-\Delta Re$ (fitted translation)')
    metrics['inlet_translation']={'fit_interval_Re':[365,410], 'curves':inlet,
                                 'resolution':'all N=48, H=5, dr_delta=8'}
    save(fig,'inlet_translation')
    x=np.arange(365,415.01,.25)
    objective=lambda shift: np.mean((sample(mc,2,x+shift)-sample(dc,2,x))**2)
    fit=minimize_scalar(objective,bounds=(-15,15),method='bounded')
    metrics['PSE_to_DNS_translation']={'fit_interval_Re':[365,415], 'shift_Re_left':float(fit.x),
        'growth_rmse_before':float(np.sqrt(objective(0))), 'growth_rmse_after':float(np.sqrt(fit.fun)),
        'note':'postprocessing comparison only; primary PSE results are unshifted'}
    output=np.column_stack((evaluation,sample(dc,1),sample(mc,1),sample(dc,2),sample(mc,2),sample(lc,2),np.interp(evaluation,lst['Re'],lst['growth_minus_alpha_i_paper'])))
    np.savetxt(root/'comparison.csv',output,delimiter=',',header='Re,DNS_A,PSE_A,DNS_growth,PSE_growth,linear_PSE_growth,local_linear_growth',comments='')
    (root/'metrics.json').write_text(json.dumps(metrics,indent=2)+'\n')
    print(json.dumps(metrics,indent=2))


if __name__=='__main__':
    main()
