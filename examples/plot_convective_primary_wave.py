"""Construct and visualize a local finite-amplitude primary wave at R=300.

The selected E1=20 wave is a representative member of the nonlinear travelling-
wave family in the convectively unstable region.  It is not claimed to be the
amplitude selected by downstream nonlinear spatial evolution from a point source.
"""
from pathlib import Path
import csv
import json

import numpy as np

from llsa.secondary import PierOperators, WaveSolver


ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "results/convective_primary_r300_beta32"


def build_wave():
    op = PierOperators(radius=300, beta=32, n=60, mean_mode="parallel", ro=-1)
    solver = WaveSolver(op, 1)
    wave = solver.linear_seed(energy=.01, bracket=(.25, .32), target=25)
    continuation = []
    for energy in (.01, .03, .1, .3, 1, 3, 10, 20):
        wave = solver.solve(wave, energy=energy)
        continuation.append(dict(E1=energy, alpha=wave.alpha, omega=wave.omega,
                                 residual=wave.residual))
    for harmonics in range(2, 6):
        wave = WaveSolver(op, harmonics).solve(wave, energy=20)
    return wave, continuation


def main():
    import matplotlib.pyplot as plt
    from matplotlib.colors import TwoSlopeNorm

    OUT.mkdir(parents=True, exist_ok=True)
    wave, continuation = build_wave()
    wave.save(OUT / "primary_wave.npz")
    with (OUT / "energy_continuation.csv").open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=continuation[0])
        writer.writeheader(); writer.writerows(continuation)

    phase = np.linspace(0, 2*np.pi, 257)
    fields = wave.fields(phase)
    order = np.argsort(wave.operators.grid.z[np.isfinite(wave.operators.grid.z)])
    order = np.flatnonzero(np.isfinite(wave.operators.grid.z))[order]
    z = wave.operators.grid.z[order]
    u, v, w = (fields[:, j, order] for j in range(3))
    amp_w = np.ptp(w, axis=0)
    iz = int(np.argmax(amp_w))

    # Schematic annulus: locally repeat the computed wave phase over a narrow patch.
    radius = np.linspace(285, 315, 220)
    theta = np.linspace(0, 2*np.pi, 512, endpoint=False)
    local_phase = wave.alpha*(radius[:, None]-300) + wave.operators.beta*theta[None, :]
    profile = wave.fields(np.mod(local_phase.ravel(), 2*np.pi))[:, 2, order[iz]]
    annulus_w = profile.reshape(local_phase.shape)
    plt.rcParams.update({"font.family":"serif", "mathtext.fontset":"stix", "font.size":9,
                         "axes.linewidth":.8, "savefig.dpi":300})
    fig = plt.figure(figsize=(9.2, 6.6), constrained_layout=True)
    grid = fig.add_gridspec(2, 2)
    ax0 = fig.add_subplot(grid[0,0])
    ax1 = fig.add_subplot(grid[0,1])
    ax2 = fig.add_subplot(grid[1,0], projection="polar")
    ax3 = fig.add_subplot(grid[1,1])

    lim = np.max(np.abs(w))
    im = ax0.pcolormesh(phase/(2*np.pi), z, w.T, shading="auto", cmap="RdBu_r",
                        norm=TwoSlopeNorm(0, -lim, lim))
    fig.colorbar(im, ax=ax0, label="$w'$")
    ax0.set(xlabel="wave phase $\\phi/(2\\pi)$", ylabel="$z$", ylim=(0,8),
            title="Wall-normal velocity")

    samples = [0, .25, .5, .75]
    for frac in samples:
        j = int(frac*(len(phase)-1))
        ax1.plot(u[j], z, label=fr"$\phi/2\pi={frac:g}$")
    ax1.axvline(0, color="0.7", lw=.7)
    ax1.set(xlabel="$u'$", ylabel="$z$", ylim=(0,8), title="Radial-velocity profiles")
    ax1.legend(frameon=False, fontsize=7)

    m = ax2.pcolormesh(theta, radius, annulus_w, shading="auto", cmap="RdBu_r",
                       norm=TwoSlopeNorm(0, -np.max(abs(annulus_w)), np.max(abs(annulus_w))))
    ax2.set(xticks=[], yticks=[])
    ax2.set_title(fr"Local annular view at $z={z[iz]:.2f}$")
    fig.colorbar(m, ax=ax2, label="$w'$", shrink=.82)

    energies = wave.energies()
    ax3.semilogy(np.arange(1, len(energies)+1), energies, "o-", color="#0072B2")
    ax3.set(xlabel="harmonic $n$", ylabel="$E_n$", xticks=range(1,6),
            title="Nonlinear harmonic content")
    ax3.grid(axis="y", which="both", color="0.9", lw=.6)
    for i, ax in enumerate((ax0,ax1,ax2,ax3)):
        ax.text(.01, .98, f"({chr(97+i)})", transform=ax.transAxes, va="top", ha="left",
                fontweight="bold", color="white" if ax in (ax0,ax2) else "black")
        ax.tick_params(direction="in", top=True, right=True)
    fig.suptitle(fr"Convective-region primary wave: $Ro=-1$, $R=300$, $\beta=32$, "
                 fr"$E_1=20$, $\alpha={wave.alpha:.5f}$, $\omega={wave.omega:.4f}$", fontsize=11)
    # The polar pcolormesh makes SVG unnecessarily large; PDF remains editable.
    for suffix in ("png", "pdf"):
        fig.savefig(OUT / f"primary_wave.{suffix}")
    plt.close(fig)

    summary = dict(ro=-1, radius=300, beta=32, E1_target=20, harmonics=5, n=60,
        alpha=wave.alpha, omega=wave.omega, phase_speed=wave.omega/wave.alpha,
        energies=energies.tolist(), nonlinear_residual=wave.residual,
        peak_wall_normal_height=float(z[iz]),
        interpretation="Local finite-amplitude primary travelling wave in the convectively unstable region.",
        limitation="E1 is selected for visualization; a localized-forcing spatial march is required to predict the naturally selected amplitude.")
    (OUT / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
