"""Plot beta42 PSE attempts for total inlet energies 1e4--1e8 E0."""
import json
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np


ROOT=Path(__file__).resolve().parents[1]/'results/beta42_nonlinear_spatial'
FACTORS=(10_000,100_000,1_000_000,10_000_000,100_000_000)


def path(factor):
    return (ROOT/'energy_sweep_re700_dr12/energy_x10000' if factor==10_000 else
            ROOT/f'energy_sweep_ultrastrong/energy_x{factor}')


def main():
    output=ROOT/'energy_sweep_ultrastrong'
    records=[]
    for factor in FACTORS:
        p=path(factor)
        d=np.atleast_1d(np.genfromtxt(p/'march.csv',delimiter=',',names=True))
        meta=json.loads((p/'metadata.json').read_text())
        signs=np.sign(np.diff(np.log(d['E1']))) if len(d)>1 else np.array([])
        ids=np.flatnonzero(signs[1:]!=signs[:-1])+1 if len(signs)>1 else []
        turns=[{'kind':'maximum' if signs[i-1]>0 else 'minimum',
                'Re':float(d['Re'][i]),'E1':float(d['E1'][i])} for i in ids]
        records.append((factor,d,meta,turns))
    plt.rcParams.update({'font.family':'serif','font.serif':['Times New Roman','DejaVu Serif'],
        'mathtext.fontset':'stix','font.size':11.5,'axes.labelsize':14,'axes.linewidth':1.1,
        'xtick.direction':'in','ytick.direction':'in','xtick.top':True,'ytick.right':True,
        'legend.frameon':False,'legend.fontsize':9.3,'pdf.fonttype':42,'savefig.dpi':230})
    colors=plt.get_cmap('plasma')(np.linspace(.08,.9,len(records)))
    fig,axes=plt.subplots(1,2,figsize=(11.2,4.4),layout='constrained')
    for color,(factor,d,_,_) in zip(colors,records):
        label=rf'$E_{{1,\mathrm{{in}}}}/E_0=10^{{{int(np.log10(factor))}}}$'
        for ax,y in ((axes[0],d['E1']),(axes[1],d['E1']/d['E1'][0])):
            if len(d)>1:
                ax.semilogy(d['Re'],y,color=color,lw=2,label=label)
                ax.plot(d['Re'][-1],y[-1],'x',color=color,ms=7,mew=1.6,clip_on=False)
            else:
                ax.semilogy(d['Re'],y,marker='X',ls='none',color=color,ms=7,label=label,clip_on=False)
    for i,ax in enumerate(axes):
        ax.set_xlim(315,700);ax.set_xlabel(r'$Re=|Ro|r_\delta$')
        ax.grid(axis='y',which='both',color='.9',lw=.55)
        ax.text(.02,.965,f'({chr(97+i)})',transform=ax.transAxes,va='top')
    axes[0].set_ylabel(r'fundamental energy $E_1$')
    axes[1].set_ylabel(r'$E_1/E_{1,\mathrm{in}}$')
    axes[0].legend(loc='upper right');axes[1].legend(loc='upper right')
    axes[1].text(.98,.035,r'$\times$: last converged point; $\mathbf{\times}$ at $Re=320$: first step failed',
                 transform=axes[1].transAxes,ha='right',va='bottom',color='.25',fontsize=9)
    for ext in ('png','pdf'):
        fig.savefig(output/f'beta42_ultrastrong_energy_sweep.{ext}',facecolor='white')
    plt.close(fig)
    report={'definition':'Existing 1e4 E0 case retained, then inlet energy multiplied by 10, 100, 1000 and 10000.',
        'parameters':{'Ro':-.751,'beta':42,'N':64,'H':7,'dr_delta':12,'target_Re':700},
        'transformations':'raw samples; log y axes; no smoothing, interpolation, fitting, translation or extrapolation',
        'cases':[]}
    for factor,d,meta,turns in records:
        report['cases'].append({'inlet_energy_factor':factor,
            'inlet_amplitude_factor':float(np.sqrt(factor)),
            'inlet_probe_amplitude':float(meta['inlet_amplitude']),
            'inlet_E1':float(d['E1'][0]),'points':len(d),'last_valid_Re':float(d['Re'][-1]),
            'turning_points':turns,'status':meta['status'],'failure':meta['failure']})
    (output/'ultrastrong_summary.json').write_text(json.dumps(report,indent=2)+'\n')
    print(json.dumps(report,indent=2))


if __name__=='__main__':
    main()
