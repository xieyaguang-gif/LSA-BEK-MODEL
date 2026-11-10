"""Report which long-domain attempts actually solved their discrete equations.

Inputs: per-case metadata.json/energy.csv, plus the earlier PSE primary CSV.
E1 is the unnormalized positive-harmonic energy. gE=0.5*d(log E1)/dR uses
unsmoothed finite differences. Failed iterates are never plotted as solutions.
Internal verification figure, 7.4 x 6.2 inches; PDF and PNG only.
"""
import json
from pathlib import Path
import re

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt


def main():
    root = Path(__file__).resolve().parents[1]
    out = root/'results/radial_re700_validation'
    records, solutions = [], []
    for path in sorted(out.glob('*/metadata.json')):
        data = json.loads(path.read_text())
        record = dict(case=path.parent.name, **data)
        match = re.search(r'at R=([0-9.eE+-]+)', data.get('error', ''))
        if match:
            record['failed_inner_Re'] = abs(data['config']['ro'])*float(match.group(1).rstrip(':'))
        records.append(record)
        if data['status'] != 'converged_discrete_equations':
            continue
        curve = np.loadtxt(path.parent/'energy.csv', delimiter=',')
        if not np.all(np.isfinite(curve)) or np.any(curve[:,1] <= 0):
            raise ValueError(f'Invalid energy in {path}')
        record['maximum_last_harmonic_fraction'] = float(max(curve[:,-1]/curve[:,1:].sum(axis=1)))
        index = np.argmax(curve[:,1])
        record['peak_harmonic_energies'] = curve[index,1:].tolist()
        if data['arguments']['end'] >= 430:
            solutions.append((record, curve))
    archive = out/'nonconverged_attempts.json'
    if archive.exists():
        existing = {r['case'] for r in records}
        records.extend(r for r in json.loads(archive.read_text()) if r['case'] not in existing)
    records.sort(key=lambda r:r['case'])
    for record in records:
        match = re.search(r'at R=([0-9.eE+-]+)', record.get('error', ''))
        if match:
            record['failed_inner_Re'] = abs(record['config']['ro'])*float(match.group(1).rstrip(':'))
    curves = {record['case']: curve for record,curve in solutions}
    names = ('global_440_n24_h3_dr1', 'global_440_n24_h3_dr05', 'global_440_n32_h3_dr1')
    if all(name in curves for name in names):
        a,b,c = [curves[name] for name in names]
        checks = dict(radial_peak_relative_change=float(max(b[:,1])/max(a[:,1])-1),
            radial_E1_Re420_relative_change=float(np.interp(420,b[:,0],b[:,1])/np.interp(420,a[:,0],a[:,1])-1),
            wallnormal_peak_relative_change=float(max(c[:,1])/max(a[:,1])-1),
            interpolation='linear interpolation in Re, only for comparing E1 at Re=420')
        high = curves.get('global_440_n24_h5_dr05_gmres')
        if high is not None:
            checks['harmonic_peak_relative_change'] = float(max(high[:,1])/max(b[:,1])-1)
            checks['harmonic_E1_Re420_relative_change'] = float(np.interp(420,high[:,0],high[:,1])/np.interp(420,b[:,0],b[:,1])-1)
        (out/'grid_checks.json').write_text(json.dumps(checks,indent=2)+'\n')
    (out/'summary.json').write_text(json.dumps(dict(
        validation_to_Re700='not_established',
        figure_excludes='all failed/nonconverged iterates', cases=records), indent=2)+'\n')
    if not solutions:
        return
    plt.rcParams.update({'font.family':'serif', 'font.serif':['Times New Roman','STIXGeneral','DejaVu Serif'],
        'mathtext.fontset':'stix', 'font.size':11, 'axes.linewidth':1.,
        'xtick.direction':'in', 'ytick.direction':'in', 'xtick.top':True, 'ytick.right':True,
        'pdf.fonttype':42, 'savefig.dpi':240})
    fig, axes = plt.subplots(2,1,figsize=(7.4,6.2))
    colors = ['#0072B2','#D55E00','#009E73','#CC79A7','#111111','#56B4E9']
    styles = ['-','--','-.',':']
    largest_end = 0
    for i, (record, a) in enumerate(solutions):
        c = record['config']; end = record['arguments']['end']
        largest_end = max(largest_end,end)
        label = f"Global N{c['n']}, H{c['harmonics']}, " + rf"$\Delta R={record['step_r']:.2f}$"
        if end != 440:
            label += f" (out {end:g})"
        color, style = colors[i%len(colors)], styles[i%len(styles)]
        axes[0].semilogy(a[:,0],a[:,1],color=color,ls=style,lw=1.5,label=label)
        g = .5*abs(c['ro'])*np.gradient(np.log(a[:,1]),a[:,0],edge_order=2)
        axes[1].plot(a[:,0],g,color=color,ls=style,lw=1.5)
    pse = np.genfromtxt(root/'results/beta42_nonlinear_spatial/primary/march.csv',delimiter=',',names=True)
    axes[0].semilogy(pse['Re'],pse['E1'],color='.5',ls='--',lw=1.4,label='PSE N64, H7, '+r'$\Delta R=6$')
    g = .5*np.gradient(np.log(pse['E1']),pse['r_delta'],edge_order=2)
    axes[1].plot(pse['Re'],g,color='.5',ls='--',lw=1.4)
    axes[0].set(xlim=(320,700),xlabel=r'$Re$',ylabel=r'$E_1$')
    axes[1].set(xlim=(370,min(max(largest_end,440),480)),xlabel=r'$Re$',ylabel=r'$g_E=\frac{1}{2}\,d\ln E_1/dR$')
    axes[1].axhline(0,color='.6',lw=.8)
    for ax in axes:
        ax.axvline(440,color='.65',ls=':',lw=.8)
    if largest_end < 700:
        axes[0].axvspan(largest_end,700,facecolor='.95',edgecolor='.85',hatch='///',zorder=0)
        axes[0].text((largest_end+700)/2,.5,'No converged solution\nto the target Re = 700',
                     transform=axes[0].get_xaxis_transform(),ha='center',va='center',color='.35',fontsize=10)
    axes[0].set_title('(a) Discrete solutions; validation incomplete',loc='left',fontsize=10,pad=8)
    axes[1].set_title('(b) Saturation region; outlet effects retained',loc='left',fontsize=10,pad=8)
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles,labels,loc='upper center',bbox_to_anchor=(.53,.985),frameon=False,
               fontsize=8.5,ncol=2,columnspacing=1.,handlelength=2.2,labelspacing=.35)
    fig.subplots_adjust(left=.13,right=.97,bottom=.085,top=.77,hspace=.34)
    for ext in ('pdf','png'):
        fig.savefig(out/f'energy_validation.{ext}')
    plt.close(fig)
    print(f'Saved {out}/energy_validation.pdf and .png')


if __name__=='__main__':
    main()
