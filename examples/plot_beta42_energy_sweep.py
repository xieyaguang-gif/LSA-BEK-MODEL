"""Plot raw beta42 integral-energy evolution for five inlet energies."""
import json
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np


ROOT = Path(__file__).resolve().parents[1]/'results/beta42_nonlinear_spatial'
CASES = [
    (1, ROOT/'primary/march.csv'),
    (10, ROOT/'energy_sweep/energy_x10/march.csv'),
    (100, ROOT/'energy_sweep/energy_x100/march.csv'),
    (1000, ROOT/'energy_sweep/energy_x1000/march.csv'),
    (10000, ROOT/'energy_sweep/energy_x10000/march.csv'),
]


def main():
    output = ROOT/'energy_sweep'
    output.mkdir(parents=True, exist_ok=True)
    data = [(factor, np.genfromtxt(path, delimiter=',', names=True))
            for factor, path in CASES]
    plt.rcParams.update({
        'font.family':'serif', 'font.serif':['Times New Roman','DejaVu Serif'],
        'mathtext.fontset':'stix', 'font.size':11.5, 'axes.labelsize':14,
        'axes.linewidth':1.1, 'xtick.direction':'in', 'ytick.direction':'in',
        'xtick.top':True, 'ytick.right':True, 'legend.frameon':False,
        'legend.fontsize':9.5, 'pdf.fonttype':42, 'savefig.dpi':230})
    colors = plt.get_cmap('viridis')(np.linspace(.08,.9,len(data)))
    fig, axes = plt.subplots(1, 2, figsize=(10.8,4.35), layout='constrained')
    summary=[]
    for color,(factor,d) in zip(colors,data):
        label=rf'$E_{{1,\mathrm{{in}}}}/E_0={factor:g}$'
        axes[0].semilogy(d['Re'],d['E1'],color=color,lw=2,label=label)
        axes[1].semilogy(d['Re'],d['E1']/d['E1'][0],color=color,lw=2,label=label)
        peak=int(np.argmax(d['E1']))
        summary.append({
            'inlet_energy_factor':factor,
            'inlet_amplitude_factor':float(np.sqrt(factor)),
            'inlet_E1':float(d['E1'][0]),
            'peak_Re':float(d['Re'][peak]),
            'peak_E1':float(d['E1'][peak]),
            'peak_over_inlet':float(d['E1'][peak]/d['E1'][0]),
            'final_Re':float(d['Re'][-1]),
            'final_E1':float(d['E1'][-1]),
            'final_over_inlet':float(d['E1'][-1]/d['E1'][0]),
        })
    for i,ax in enumerate(axes):
        ax.set_xlim(320,438)
        ax.set_xlabel(r'$Re=|Ro|r_\delta$')
        ax.text(.025,.965,f'({chr(97+i)})',transform=ax.transAxes,va='top',ha='left')
        ax.grid(axis='y',which='both',color='.9',lw=.55)
    axes[0].set_ylabel(r'fundamental energy $E_1$')
    axes[1].set_ylabel(r'$E_1/E_{1,\mathrm{in}}$')
    axes[0].legend(loc='lower right')
    axes[1].legend(loc='upper right')
    for ext in ('png','pdf'):
        fig.savefig(output/f'beta42_inlet_energy_sweep.{ext}',facecolor='white')
    plt.close(fig)
    metadata={
        'quantity':'Pier integral energy E1 of the positive beta42 harmonic',
        'energy_definition':'r_delta^2 integral (|u1|^2+|v1|^2+|w1|^2) dz; positive harmonic only',
        'transformations':'raw march samples; logarithmic y axes; no smoothing, interpolation, translation, or fitting',
        'parameters':{'Ro':-.751,'beta':42,'N':64,'H':7,'dr_delta':6,
                      'Re_start':320,'last_Re':float(data[0][1]['Re'][-1])},
        'cases':summary,
    }
    (output/'energy_sweep_summary.json').write_text(json.dumps(metadata,indent=2)+'\n')
    np.savetxt(output/'energy_sweep_summary.csv',
               [[v[k] for k in ('inlet_energy_factor','inlet_amplitude_factor','inlet_E1',
                                 'peak_Re','peak_E1','peak_over_inlet','final_Re','final_E1','final_over_inlet')]
                for v in summary], delimiter=',',
               header='inlet_energy_factor,inlet_amplitude_factor,inlet_E1,peak_Re,peak_E1,peak_over_inlet,final_Re,final_E1,final_over_inlet',
               comments='')
    print(json.dumps(metadata,indent=2))


if __name__=='__main__':
    main()
