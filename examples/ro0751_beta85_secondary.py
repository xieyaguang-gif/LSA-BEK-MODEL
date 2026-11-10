"""Secondary Floquet analysis based on the Ro=-0.751, beta=85 primary saddle.

The primary absolute saddle has complex alpha, whereas the present nonlinear
travelling-wave solver uses real alpha.  We therefore solve the local finite-
amplitude branch at the same Re and beta, and record that approximation in the
output metadata instead of silently identifying the two problems.
"""
from pathlib import Path
import csv
import json

import numpy as np

from llsa.secondary import (PierOperators, WaveSolver, FloquetProblem,
                            secondary_spectrum, secondary_saddle)

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "results/ro0751_beta85_secondary"
RE = 409.585625
RO = -.751
BETA = 85.0
RADIUS = RE / abs(RO)
E1 = 100.0


def make_wave(n=40):
    op = PierOperators(radius=RADIUS, beta=BETA, n=n, mean_mode="parallel", ro=RO)
    solver = WaveSolver(op, 1)
    wave = solver.linear_seed(.01, bracket=(.01, .2), target=75.0)
    for energy in (.01, .1, 1., 10., E1):
        wave = solver.solve(wave, energy=energy)
    for harmonics in range(2, 6):
        wave = WaveSolver(op, harmonics).solve(wave, energy=E1)
    return wave


def temporal_scan(wave, beta_values):
    rows = []
    for beta_hat in beta_values:
        problem = FloquetProblem(wave, beta_hat=beta_hat, harmonics=3,
                                 wall_condition="continuity")
        spectrum = secondary_spectrum(problem, alpha_hat=0, target=14+.65j, count=10)
        index = int(np.argmax(spectrum.omega.imag))
        value = spectrum.omega[index]
        rows.append(dict(beta_hat=beta_hat, omega_real=float(value.real),
                         growth=float(value.imag), residual=float(spectrum.residuals[index]),
                         harmonics=3, alpha_hat_real=0., alpha_hat_imag=0.))
    return rows


def energy_scan(energies):
    rows = []
    for energy in energies:
        op = PierOperators(radius=RADIUS, beta=BETA, n=40, mean_mode="parallel", ro=RO)
        solver = WaveSolver(op, 1)
        wave = solver.linear_seed(.01, bracket=(.01, .2), target=75.0)
        for value in (.01, .1, 1., 10., energy):
            wave = solver.solve(wave, energy=value)
        for harmonics in range(2, 6):
            wave = WaveSolver(op, harmonics).solve(wave, energy=energy)
        problem = FloquetProblem(wave, beta_hat=20, harmonics=3,
                                 wall_condition="continuity")
        spectrum = secondary_spectrum(problem, alpha_hat=0, target=14+.65j, count=10)
        index = int(np.argmax(spectrum.omega.imag))
        value = spectrum.omega[index]
        rows.append(dict(E1=energy, alpha=wave.alpha, omega=wave.omega,
                         secondary_omega_real=float(value.real),
                         secondary_growth=float(value.imag)))
    return rows


def plot(wave, problem, saddle, out):
    import matplotlib.pyplot as plt
    from matplotlib.colors import TwoSlopeNorm

    phase = np.linspace(0, 2*np.pi, 257)
    finite = np.isfinite(problem.wave.operators.grid.z)
    order = np.argsort(problem.wave.operators.grid.z[finite])
    nodes = np.flatnonzero(finite)[order]
    z = problem.wave.operators.grid.z[nodes]
    primary = wave.fields(phase)[:, :3, nodes]
    secondary = problem.fields(saddle.vector, phase)[:, :3, nodes]
    secondary_real = secondary.real
    scale = np.max(np.abs(secondary_real[:, 2]))
    secondary_real /= scale
    primary_w = primary[:, 2]
    theta = phase / BETA
    envelope = np.exp(1j * problem.beta_hat * theta)
    attached = (secondary[:, 2] * envelope[:, None]).real / scale

    plt.rcParams.update({"font.family":"serif", "mathtext.fontset":"stix", "font.size":9,
                         "axes.linewidth":.8, "savefig.dpi":300})
    fig, ax = plt.subplots(2, 2, figsize=(9.4, 6.4), constrained_layout=True)
    lim = np.max(abs(primary_w))
    im = ax[0,0].pcolormesh(phase/(2*np.pi), z, primary_w.T, shading="auto",
                            cmap="RdBu_r", norm=TwoSlopeNorm(0, -lim, lim))
    fig.colorbar(im, ax=ax[0,0], label="$w'_1$")
    ax[0,0].set(xlabel="$\\phi/(2\\pi)$", ylabel="$z$", ylim=(0,8),
                title="Finite-amplitude primary wave")
    im = ax[0,1].pcolormesh(phase/(2*np.pi), z, attached.T, shading="auto",
                            cmap="RdBu_r", norm=TwoSlopeNorm(0, -1, 1))
    fig.colorbar(im, ax=ax[0,1], label="normalized $w'_2$")
    ax[0,1].set(xlabel="$\\phi/(2\\pi)$", ylabel="$z$", ylim=(0,8),
                title="Secondary mode attached to primary phase")
    total = primary_w/lim + .25*attached
    im = ax[1,0].pcolormesh(phase/(2*np.pi), z, total.T, shading="auto",
                            cmap="RdBu_r", norm=TwoSlopeNorm(0, -1, 1))
    fig.colorbar(im, ax=ax[1,0], label="normalized $w'_1+0.25w'_2$")
    ax[1,0].set(xlabel="$\\phi/(2\\pi)$", ylabel="$z$", ylim=(0,8),
                title="Primary plus secondary reconstruction")
    ax[1,1].axis("off")
    text = ("Local Floquet result\n\n"
            f"Ro={RO}, Re={RE:.4f}, beta={BETA:.0f}\n"
            f"E1={E1:g}, H=5, K=5\n"
            f"beta_hat={problem.beta_hat:g}\n\n"
            f"alpha_hat={saddle.alpha_hat.real:.5f}"
            f"{saddle.alpha_hat.imag:+.5f}i\n"
            f"omega_hat={saddle.omega_hat.real:.5f}"
            f"{saddle.omega_hat.imag:+.5f}i\n\n"
            "positive Im(omega_hat):\nlocal secondary absolute growth")
    ax[1,1].text(.05, .95, text, va="top", fontsize=11)
    for i, axis in enumerate(ax.flat[:3]):
        axis.text(.01, .98, f"({chr(97+i)})", transform=axis.transAxes,
                  va="top", ha="left", fontweight="bold",
                  color="white" if i < 3 else "black")
        axis.tick_params(direction="in", top=True, right=True)
    fig.suptitle("Ro=-0.751, beta=85: local secondary Floquet analysis", fontsize=12)
    fig.savefig(out / "secondary_local_reconstruction.png")
    fig.savefig(out / "secondary_local_reconstruction.pdf")
    plt.close(fig)


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    wave = make_wave()
    wave.save(OUT / "primary_wave.npz")
    problem = FloquetProblem(wave, beta_hat=20, harmonics=5,
                             wall_condition="continuity")
    spectrum = secondary_spectrum(problem, alpha_hat=0, target=14+.65j, count=10)
    index = int(np.argmax(spectrum.omega.imag))
    initial = complex(spectrum.omega[index])
    saddle = secondary_saddle(problem, alpha_hat=0j, target=initial,
                               max_iter=20, max_step=.01, count=10)
    np.savez_compressed(OUT / "secondary_saddle.npz", vector=saddle.vector,
                        alpha_hat=saddle.alpha_hat, omega_hat=saddle.omega_hat,
                        beta_hat=problem.beta_hat, K=problem.K,
                        residual=saddle.residual)
    # The targeted branch is reliably isolated in this interval; outside it
    # the same shift can return a different remote eigenvalue.
    beta_rows = temporal_scan(wave, range(10, 28, 2))
    energy_rows = energy_scan((20., 50., 100., 200.))
    for name, rows in (("beta_hat_scan.csv", beta_rows), ("energy_scan.csv", energy_rows)):
        with (OUT / name).open("w", newline="") as stream:
            writer = csv.DictWriter(stream, fieldnames=rows[0].keys())
            writer.writeheader(); writer.writerows(rows)
    plot(wave, problem, saddle, OUT)
    summary = dict(
        ro=RO, reynolds=RE, radius=RADIUS, primary_beta=BETA,
        # beta=85 values linearly interpolated from the retained Re=400/410
        # saddle scan at Re=409.585625 (the separate beta=86 record is not used).
        source_primary_absolute_saddle=dict(re=RE, beta=85,
                                            alpha=[.2613424, .1489129],
                                            omega=[.0386228, 0.]),
        local_primary=dict(alpha=wave.alpha, omega=wave.omega,
                           energies=wave.energies().tolist(), harmonics=wave.harmonics,
                           n=wave.operators.m-1, mean_mode=wave.operators.mean_mode,
                           residual=wave.residual),
        secondary=dict(beta_hat=problem.beta_hat, harmonics=problem.K,
                       wall_condition=problem.wall_condition,
                       alpha_hat=[saddle.alpha_hat.real, saddle.alpha_hat.imag],
                       omega_hat=[saddle.omega_hat.real, saddle.omega_hat.imag],
                       residual=saddle.residual, converged=saddle.converged,
                       group_velocity=[saddle.group_velocity.real, saddle.group_velocity.imag]),
        interpretation="Local Floquet secondary analysis on a finite-amplitude primary wave at the primary absolute-neutral Re and beta.",
        limitation="The primary absolute saddle has complex alpha; the nonlinear travelling-wave continuation uses real alpha and E1=100 as an explicit local proxy. This is not yet the nonparallel global secondary eigenproblem.")
    (OUT / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
