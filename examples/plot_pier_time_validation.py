"""Plot stored time histories and convergence, without smoothing or solver reruns.

Research diagnostics: 18 x 13 cm and 18 x 7 cm; PDF and 240 dpi PNG.
Amplitude is sqrt(E). The full transient is shown. Failed dt=.002 runs remain
in raw data and are excluded from the stable-step Richardson estimates.
"""
import argparse
import json
import os
import tempfile
from pathlib import Path

os.environ.setdefault('MPLCONFIGDIR', str(Path(tempfile.gettempdir())/'llsa_matplotlib'))
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np


def read(directory):
    return json.loads((directory/'comparison.json').read_text())


def save(fig, path):
    fig.savefig(path.with_suffix('.pdf'))
    fig.savefig(path.with_suffix('.png'), dpi=240)
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--input', type=Path, default=Path('results/pier_time'))
    parser.add_argument('--output', type=Path, default=Path('results/pier_time/figures'))
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    plt.rcParams.update({'font.family': 'serif', 'font.serif': ['STIXGeneral'],
                         'mathtext.fontset': 'stix', 'font.size': 9,
                         'axes.linewidth': .8, 'lines.linewidth': 1.2,
                         'xtick.direction': 'in', 'ytick.direction': 'in',
                         'pdf.fonttype': 42, 'savefig.facecolor': 'white'})
    cases = [read(args.input/'temporal_converged'), read(args.input/'weighted_converged')]
    colors = ['#0072B2', '#D55E00']
    titles = [r'Real $\hat\alpha=0$', r'Complex $\hat\alpha$ (weighted envelope)']
    fig, axes = plt.subplots(2, 2, figsize=(18/2.54, 13/2.54), constrained_layout=True)
    for row, (name, data, color, title) in enumerate(zip(
            ['temporal_converged', 'weighted_converged'], cases, colors, titles)):
        run = min(data['runs'], key=lambda r: r['dt'])
        trace = np.load(args.input/name/run['trace'])
        t, energy, probe = trace['time'], trace['energy'], trace['probe']
        amplitude = .5*np.log(energy/energy[0])
        phase = np.unwrap(np.angle(probe))
        phase -= phase[0]
        fit = run['fits'][-1]
        use = t >= fit['start']
        for col, y in enumerate([amplitude, phase]):
            ax = axes[row, col]
            ax.plot(t, y, color=color, label='Time advancement')
            poly = np.polyfit(t[use], y[use], 1)
            ax.plot(t[use], np.polyval(poly, t[use]), '--', color='black', label='Late-time fit')
            ax.axvspan(fit['start'], t[-1], color='black', alpha=.05, linewidth=0)
            ax.set(xlabel=r'$t$', ylabel=r'$\log[A(t)/A(0)]$' if col == 0 else r'$\arg c(t)-\arg c(0)$ (rad)',
                   title=title)
            ax.text(.025, .96 if col == 0 else .85, f'({"abcd"[2*row+col]})', transform=ax.transAxes, va='top')
            value = fit['growth'] if col == 0 else fit['omega_real']
            label = r'$\sigma$' if col == 0 else r'$\hat\omega_r$'
            ax.text(.97 if col == 0 else .04, .07 if col == 0 else .13,
                    f'{label} = {value:.5f}', transform=ax.transAxes, ha='right' if col == 0 else 'left')
    axes[0, 0].legend(loc='upper left', bbox_to_anchor=(.08, .95), frameon=False, fontsize=8)
    save(fig, args.output/'time_histories')

    fig, axes = plt.subplots(1, 2, figsize=(18/2.54, 7/2.54), constrained_layout=True)
    for data, color, title, marker in zip(cases, colors, ['Real wavenumber', 'Weighted envelope'], ['o', 's']):
        dts = np.array([r['dt'] for r in data['runs']])
        values = np.array([r['fits'][-1]['growth'] for r in data['runs']])
        limit = data['time_convergence']['growth']['extrapolated']
        axes[0].loglog(dts, abs(values-limit), marker=marker, color=color, label=title)
    axes[0].set(xlabel=r'$\Delta t$', ylabel=r'$|\sigma(\Delta t)-\sigma_{\mathrm{extrap}}|$', title='Time-step convergence')
    axes[0].legend(frameon=False, fontsize=8)
    axes[0].text(.05, .82, 'Measured order ≈ 2', transform=axes[0].transAxes)
    for n, folders, color, marker in [
        (80, ['weighted_n80_z320', 'weighted_converged'], '#0072B2', 'o'),
        (96, ['weighted_n96_z640', 'weighted_n96_z1280'], '#D55E00', 's')]:
        points = []
        for name in folders:
            d = read(args.input/name)
            r = min(d['runs'], key=lambda r: abs(r['dt']-.0005))
            points.append((d['parameters']['zmax'], r['fits'][-1]['growth']))
        z, sigma = np.array(points).T
        axes[1].plot(z, sigma, marker=marker, color=color, label=f'N = {n}')
    reference = cases[1]['spectral_comparison']['omega_hat'][1]
    axes[1].axhline(reference, color='black', linestyle='--', label='Floquet, semi-infinite')
    axes[1].set(xlabel=r'$z_{\max}$', ylabel=r'$\sigma$', title=r'Weighted envelope, $\Delta t=0.0005$')
    axes[1].ticklabel_format(axis='y', style='plain', useOffset=False)
    axes[1].legend(frameon=False, fontsize=8, loc='lower right')
    for index, ax in enumerate(axes):
        ax.text(.025, .96 if index == 0 else .85, f'({"ab"[index]})', transform=ax.transAxes, va='top')
    save(fig, args.output/'convergence')


if __name__ == '__main__':
    main()
