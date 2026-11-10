"""Plot the longest internally consistent beta42 energy sweep and its stop points."""
import json
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np


ROOT=Path(__file__).resolve().parents[1]/'results/beta42_nonlinear_spatial'
FACTORS=(1,10,100,1000,10000)


def turning_points(d):
    sign=np.sign(np.diff(np.log(d['E1'])))
    ids=np.flatnonzero(sign[1:] != sign[:-1])+1
    return [{'kind':'maximum' if sign[i-1] > 0 else 'minimum',
             'Re':float(d['Re'][i]),'E1':float(d['E1'][i])} for i in ids]


def main():
    source=ROOT/'energy_sweep_re700_dr12'
    output=ROOT/'energy_sweep_re700'
    output.mkdir(parents=True,exist_ok=True)
    data=[]
    audit=[]
    for factor in FACTORS:
        path=source/f'energy_x{factor}'
        d=np.genfromtxt(path/'march.csv',delimiter=',',names=True)
        meta=json.loads((path/'metadata.json').read_text())
        energies=np.column_stack([d[f'E{i}'] for i in range(1,8)])
        data.append((factor,d))
        audit.append({'inlet_energy_factor':factor,'status':meta['status'],
            'last_valid_Re':float(d['Re'][-1]),'failure':meta['failure'],
            'turning_points':turning_points(d),
            'max_E7_over_sum_E1_E7':float(np.max(energies[:,-1]/energies.sum(axis=1)))})
    plt.rcParams.update({'font.family':'serif','font.serif':['Times New Roman','DejaVu Serif'],
        'mathtext.fontset':'stix','font.size':11.5,'axes.labelsize':14,'axes.linewidth':1.1,
        'xtick.direction':'in','ytick.direction':'in','xtick.top':True,'ytick.right':True,
        'legend.frameon':False,'legend.fontsize':9.5,'pdf.fonttype':42,'savefig.dpi':230})
    colors=plt.get_cmap('viridis')(np.linspace(.08,.9,len(data)))
    fig,axes=plt.subplots(1,2,figsize=(11.2,4.4),layout='constrained')
    for color,(factor,d) in zip(colors,data):
        label=rf'$E_{{1,\mathrm{{in}}}}/E_0={factor:g}$'
        for ax,y in ((axes[0],d['E1']),(axes[1],d['E1']/d['E1'][0])):
            ax.semilogy(d['Re'],y,color=color,lw=2,label=label)
            ax.plot(d['Re'][-1],y[-1],marker='x',ms=7,mew=1.6,color=color,clip_on=False)
    for i,ax in enumerate(axes):
        ax.set_xlim(320,700);ax.set_xlabel(r'$Re=|Ro|r_\delta$')
        ax.grid(axis='y',which='both',color='.9',lw=.55)
        ax.text(.02,.965,f'({chr(97+i)})',transform=ax.transAxes,va='top')
    axes[0].set_ylabel(r'fundamental energy $E_1$')
    axes[1].set_ylabel(r'$E_1/E_{1,\mathrm{in}}$')
    axes[0].legend(loc='upper right')
    axes[1].legend(loc='upper right')
    axes[1].text(.98,.035,r'$\times$: PSE stopping point',transform=axes[1].transAxes,
                 ha='right',va='bottom',color='.25')
    for ext in ('png','pdf'):
        fig.savefig(output/f'beta42_energy_to_re700_attempt.{ext}',facecolor='white')
    plt.close(fig)
    report={'requested_Re_end':700,'displayed_radial_step':12,
        'status':'No case reached Re=700; curves contain converged points only.',
        'transformations':'raw samples, logarithmic y axes; no interpolation, smoothing, fitting, translation, or extrapolation',
        'cases':audit,
        'step_audit':{}}
    for dr in (6,8,12):
        report['step_audit'][str(dr)]={}
        folder=ROOT/f'energy_sweep_re700{("" if dr==6 else f"_dr{dr}")}'
        for factor in FACTORS:
            meta=json.loads((folder/f'energy_x{factor}/metadata.json').read_text())
            report['step_audit'][str(dr)][str(factor)]={'status':meta['status'],'last_Re':meta['last_Re']}
    (output/'extension_audit.json').write_text(json.dumps(report,indent=2)+'\n')
    print(json.dumps(report,indent=2))


if __name__=='__main__':
    main()
