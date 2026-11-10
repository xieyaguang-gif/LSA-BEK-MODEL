"""Read accepted continuation stations; never plot Newton checkpoints."""
import argparse
import json
from pathlib import Path

import numpy as np


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--root', type=Path, default=Path('results/radial_extension'))
    p.add_argument('--plot', action='store_true')
    p.add_argument('--include-pilot', action='store_true',
                   help='Also plot exploratory resolutions below H5/N24')
    args = p.parse_args()
    accepted, attempts = [], []
    for path in sorted(args.root.glob('*/*/metadata.json')):
        record = json.loads(path.read_text())
        summary_path = path.parent.parent/'summary.json'
        settings = json.loads(summary_path.read_text()).get('arguments',{}) if summary_path.exists() else {}
        # The earliest driver predates the mean-damping option and used False.
        record['damp_mean'] = settings.get('damp_mean',record.get('damp_mean',False)) if settings else record.get('damp_mean')
        record['mean_damping_setting_source'] = ('arguments' if 'damp_mean' in settings
            else 'legacy default' if settings else 'metadata')
        record['outlet_alpha'] = settings.get('outlet_alpha',record.get('outlet_alpha'))
        record['case'] = str(path.parent.relative_to(args.root))
        record['resolution_role'] = ('pilot' if record['config']['harmonics'] < 5
                                     or record['config']['n'] < 24 else 'comparison')
        attempts.append(record)
        if record['status'] != 'converged_discrete_equations':
            continue
        if record['damp_mean'] is None or record['outlet_alpha'] is None:
            raise ValueError(f'Missing sponge/outlet settings for accepted case {path.parent}')
        a = np.loadtxt(path.parent/'energy.csv', delimiter=',', comments='#')
        stop = record['sponge_start']
        mask = a[:,0] <= stop
        region = mask & (a[:,0] >= 430)
        pos = np.flatnonzero(region)[np.argmin(a[region,1])]
        record['minimum_after_Re430'] = {'Re':float(a[pos,0]),'E1':float(a[pos,1])}
        record['E1_samples'] = {str(x):float(np.interp(x,a[:,0],a[:,1]))
                               for x in [460,480,500,520,525,530,540,550,600,650,700] if x <= stop}
        accepted.append(record)
    summary = dict(status='validation_in_progress', accepted=accepted, attempts=attempts,
                   max_converged_physical_Re=max((r['sponge_start'] for r in accepted), default=None),
                   note='Algebraic convergence is not grid convergence; only data before sponge onset are physical.')
    summary['max_Re_by_resolution'] = {}
    for r in accepted:
        key = f'H{r["config"]["harmonics"]}_N{r["config"]["n"]}'
        summary['max_Re_by_resolution'][key] = max(r['sponge_start'],
            summary['max_Re_by_resolution'].get(key,0))
    # Compare each station to the furthest H5/N24 field on a common physical
    # interval; numerical interpolation is reported explicitly.
    reference_candidates = [r for r in accepted if r['config']['harmonics'] == 5 and r['config']['n'] == 24
                            and not r['damp_mean'] and complex(r['outlet_alpha']) == 0]
    if reference_candidates:
        ref = max(reference_candidates, key=lambda r:(r['sponge_start'],
                  r.get('computational_end',r['sponge_start']+40)))
        a = np.loadtxt(args.root/ref['case']/'energy.csv', delimiter=',', comments='#')
        comparisons = []
        for r in accepted:
            b = np.loadtxt(args.root/r['case']/'energy.csv', delimiter=',', comments='#')
            stop = min(ref['sponge_start'], r['sponge_start'])
            errors = {}
            for upper in [500, 520, 530, 540, 550, 600, 650, 700]:
                if upper > stop:
                    continue
                mask = (b[:,0] >= 350) & (b[:,0] <= upper)
                values = np.interp(b[mask,0], a[:,0], a[:,1])
                errors[str(upper)] = float(np.max(abs(b[mask,1]/values-1)))
            comparisons.append(dict(case=r['case'], max_relative_E1_difference=errors))
        summary['comparisons'] = dict(reference=ref['case'], lower_Re=350,
            interpolation='linear in Re on the comparison grid', cases=comparisons)
    (args.root/'validation_summary.json').write_text(json.dumps(summary,indent=2)+'\n')
    if not args.plot or not accepted:
        return
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    plt.rcParams.update({'font.family':'serif','font.serif':['Times New Roman','STIXGeneral','DejaVu Serif'],
                         'mathtext.fontset':'stix','font.size':10,'axes.linewidth':.9,
                         'xtick.direction':'in','ytick.direction':'in','xtick.top':True,'ytick.right':True})
    # Numerical routes to the same equations should not crowd the physical
    # comparison plot. Keep their diagnostics in the JSON table instead.
    by_family = {}
    for r in accepted:
        if r['resolution_role'] == 'pilot' and not args.include_pilot:
            continue
        family = f'H={r["config"]["harmonics"]}, N={r["config"]["n"]}'
        if r['damp_mean']:
            family += '; mean sponge'
        if complex(r['outlet_alpha']) != 0:
            family += '; matched outlet'
        if r.get('dr',.5) > .51:
            family += f'; $\\Delta R={r["dr"]:.2g}$'
        if family not in by_family or (r['sponge_start'],r.get('computational_end',r['sponge_start']+40)) > (
                by_family[family]['sponge_start'],by_family[family].get('computational_end',by_family[family]['sponge_start']+40)):
            by_family[family] = r
    fig, (ax, detail) = plt.subplots(1,2,figsize=(9,3.6),constrained_layout=True)
    palette = ['#0072B2','#D55E00','#009E73','#CC79A7','#E69F00','#56B4E9']
    for i,(family,r) in enumerate(by_family.items()):
        a = np.loadtxt(args.root/r['case']/'energy.csv',delimiter=',',comments='#')
        mask = a[:,0] <= r['sponge_start']
        label = family + f'; $Re_s={r["sponge_start"]:g}$'
        color = palette[i%len(palette)]
        ls = ['-', '--', '-.', ':'][i % 4]
        ax.semilogy(a[mask,0],a[mask,1],color=color,lw=1.4,ls=ls,label=label)
        zoom = mask & (a[:,0]>=440)
        detail.semilogy(a[zoom,0],a[zoom,1],color=color,lw=1.4,ls=ls)
    ax.set(xlabel='$Re$',ylabel='$E_1$',title='(a) Steady primary-wave energy')
    ax.legend(frameon=False,fontsize=8,loc='lower right')
    detail.set(xlabel='$Re$',ylabel='$E_1$',title='(b) Decay and renewed growth')
    detail.axvline(520, color='.65', lw=.8, ls=':')
    for a in (ax,detail):
        a.tick_params(length=3.5)
    fig.savefig(args.root/'radial_extension.png',dpi=300)
    fig.savefig(args.root/'radial_extension.pdf')
    plt.close(fig)


if __name__ == '__main__':
    main()
