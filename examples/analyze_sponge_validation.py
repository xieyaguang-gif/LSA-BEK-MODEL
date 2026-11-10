"""Summarize and plot the converged outlet-sponge validation matrix."""
from pathlib import Path
import json

import matplotlib.pyplot as plt
import numpy as np


ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT/'results/radial_sponge_validation'
CASES = {
    'H3': 'h3_dr05_s460_w40_a10_end500',
    'H5': 'h5_dr05_s460_w40_a10_end500',
    'H5 fine': 'h5_dr035_s460_w40_a10_end500',
    'H7': 'h7_dr035_s460_w40_a10_end500',
    'strength 5': 'h5_dr05_s460_w40_a5_end500',
    'strength 20': 'h5_dr05_s460_w40_a20_end500',
    'Robin 0': 'h5_dr05_s460_w40_a10_alpha0_end500',
    'sponge 480': 'h5_dr05_s480_w40_a10_end520',
    'sponge 500': 'h5_dr05_s500_w40_a10_end540',
    'sponge 510': 'h5_dr05_s510_w40_a10_end550_physicalseed',
    'sponge 520': 'h5_dr05_s520_w40_a10_end560_physicalseed',
    'N32': 'h5_n32_dr05_s460_w40_a10_end500',
    'H5 fine s500': 'h5_dr035_s500_w40_a10_end540',
    'H7 s500': 'h7_dr035_s500_w40_a10_end540',
    'N32 s500': 'h5_n32_dr05_s500_w40_a10_end540',
}


def load(name):
    folder = DATA/CASES[name]
    metadata = json.loads((folder/'metadata.json').read_text())
    if metadata['status'] != 'converged_discrete_equations':
        raise RuntimeError(f'{name} has status {metadata["status"]}')
    values = np.loadtxt(folder/'energy.csv', delimiter=',', comments='#')
    return values[:, 0], values[:, 1:], metadata


def relative_to(name, reference, stop):
    x, e, _ = load(name)
    xr, er, _ = load(reference)
    keep = xr <= stop+1e-10
    comparison = np.interp(xr[keep], x, e[:, 0])
    return xr[keep], (comparison-er[keep, 0])/er[keep, 0]


def main():
    data = {name: load(name) for name in CASES}
    summary = {'comparison_region': 'Only Re at or before the relevant sponge start is physical.'}
    for name, (re, energy, metadata) in data.items():
        peak = int(np.argmax(energy[:, 0]))
        summary[name] = {
            'status': metadata['status'],
            'relative_residual': metadata['relative_residual'],
            'E1_peak': float(energy[peak, 0]),
            'E1_peak_Re': float(re[peak]),
            'peak_harmonic_energies': [float(np.max(energy[:, j])) for j in range(energy.shape[1])],
            'E1_at_sponge_start': float(np.interp(metadata['outlet_sponge']['start_Re'], re, energy[:, 0])),
            'E1_at_outlet': float(energy[-1, 0]),
        }
    checks = {}
    for name, ref, limits in [
        ('H3', 'H5', (420, 440, 460)),
        ('H7', 'H5 fine', (420, 440, 460)),
        ('H5 fine', 'H5', (420, 440, 460)),
        ('N32', 'H5', (420, 440, 460)),
        ('strength 5', 'H5', (420, 440, 460)),
        ('strength 20', 'H5', (420, 440, 460)),
        ('Robin 0', 'H5', (420, 440, 460)),
        ('sponge 480', 'H5', (420, 440, 460)),
        ('sponge 500', 'sponge 480', (440, 460, 480, 500)),
        ('sponge 510', 'sponge 500', (460, 480, 500, 510)),
        ('sponge 520', 'sponge 510', (480, 500, 510, 520)),
        ('H5 fine s500', 'sponge 500', (460, 480, 500)),
        ('H7 s500', 'H5 fine s500', (460, 480, 500)),
        ('N32 s500', 'sponge 500', (460, 480, 500)),
    ]:
        checks[f'{name} relative to {ref}'] = {}
        for stop in limits:
            _, rel = relative_to(name, ref, stop)
            checks[f'{name} relative to {ref}'][f'max_abs_relative_E1_to_Re{stop}'] = float(np.max(abs(rel)))
    summary['checks'] = checks
    (DATA/'summary.json').write_text(json.dumps(summary, indent=2)+'\n')

    plt.rcParams.update({
        'font.family': 'serif', 'font.serif': ['Times New Roman', 'Times', 'Nimbus Roman', 'DejaVu Serif'],
        'mathtext.fontset': 'stix', 'font.size': 9, 'axes.linewidth': .9,
        'xtick.direction': 'in', 'ytick.direction': 'in', 'xtick.top': True, 'ytick.right': True,
    })
    fig, axes = plt.subplots(2, 2, figsize=(7.2, 6.0), constrained_layout=True)

    ax = axes[0, 0]
    styles = {'H3': ('#4c78a8', '--'), 'H5': ('#e45756', '-'), 'H7': ('#2a9d8f', '-.')}
    for name in ('H3', 'H5', 'H7'):
        re, energy, _ = data[name]
        ax.semilogy(re, energy[:, 0], styles[name][1], color=styles[name][0], lw=1.5, label=name)
    ax.axvspan(460, 500, color='.92', zorder=-5)
    ax.set(xlim=(320, 500), ylim=(5e-2, 8e3), ylabel=r'$E_1$', title='(a) Fourier truncation')
    ax.legend(frameon=False, loc='lower left')

    ax = axes[0, 1]
    for name, color, ls in [('H5', '#e45756', '-'), ('sponge 480', '#7a5195', '--'),
                            ('sponge 500', '#2a9d8f', '-.'), ('sponge 520', '#4c78a8', ':')]:
        re, energy, _ = data[name]
        ax.semilogy(re, energy[:, 0], ls, color=color, lw=1.5, label=name.replace('sponge ', r'$Re_s=$'))
    ax.axvline(460, color='#e45756', lw=.8, alpha=.7)
    ax.axvline(480, color='#7a5195', lw=.8, alpha=.7)
    ax.axvline(500, color='#2a9d8f', lw=.8, alpha=.7)
    ax.axvline(520, color='#4c78a8', lw=.8, alpha=.7)
    ax.set(xlim=(390, 540), ylim=(10, 8e3), title='(b) Sponge-position sensitivity')
    ax.legend(frameon=False)

    ax = axes[1, 0]
    for name, color, ls in [('strength 5', '#4c78a8', '--'), ('strength 20', '#2a9d8f', '-.'),
                            ('Robin 0', '#7a5195', ':')]:
        x, rel = relative_to(name, 'H5', 460)
        ax.plot(x, 100*rel, ls, color=color, lw=1.35, label=name)
    ax.axhline(0, color='black', lw=.7)
    ax.set(xlim=(320, 460), xlabel=r'$Re$', ylabel=r'$100(E_1/E_{1,H5}-1)$ (\%)',
           title='(c) Absorber and terminal-boundary sensitivity')
    ax.legend(frameon=False, fontsize=8)

    ax = axes[1, 1]
    for name, ref, color, ls, label in [
        ('H5 fine s500', 'sponge 500', '#f58518', '--', r'$\Delta R:0.50\to0.35$'),
        ('H7 s500', 'H5 fine s500', '#2a9d8f', '-.', r'$H:5\to7$ at $\Delta R=0.35$'),
        ('N32 s500', 'sponge 500', '#4c78a8', ':', r'$N_z:24\to32$')]:
        x, rel = relative_to(name, ref, 500)
        ax.plot(x, 100*rel, ls, color=color, lw=1.35, label=label)
    ax.axhline(0, color='black', lw=.7)
    ax.set(xlim=(380, 500), xlabel=r'$Re$', ylabel='relative change in $E_1$ (\%)',
           title='(d) Discretization sensitivity')
    ax.legend(frameon=False, fontsize=8)

    for ax in axes.flat:
        ax.tick_params(length=3.5, width=.8)
    fig.savefig(DATA/'sponge_validation.png', dpi=300)
    fig.savefig(DATA/'sponge_validation.pdf')
    plt.close(fig)


if __name__ == '__main__':
    main()
