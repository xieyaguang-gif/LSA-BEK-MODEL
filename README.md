# LSA-BEK-MODEL

Python research code for stability and nonlinear rotating-wave calculations in the Bödewadt–Ekman–von Kármán (BEK) family of rotating boundary layers. The Python package is named `llsa`.

This repository contains the numerical implementation, tests, and example/campaign drivers from the `llsa_python` research project, together with a compact paper dataset. Large raw flow fields, solver histories, original external reference datasets, manuscript files, environments, and caches are excluded.

## Contents

- `src/llsa/`: BEK base flow; local temporal/spatial spectra; complex-wavenumber saddles; nonlinear primary waves; secondary/Floquet operators; radial/global formulations; full boundary value problems and continuation.
- `examples/`: calculation, continuation, diagnostic, and plotting drivers. The `roscan_*` scripts include the cross-Rossby workflow; `roscan_nonlinear_solver.py` contains the associated nonlinear correction implementation.
- `tests/`: numerical and equation-level checks.
- `src/llsa/secondary/references.json`: small packaged literature-reference constants required by the validation module.

The global nonlinear formulation computes rotating-wave branches for specified harmonic truncations, domain boundaries, and damping conditions. A computed branch does not by itself establish temporal stability or turbulent transition.

## Paper data

[`datasets/bek_negative_ro/`](datasets/bek_negative_ro/) contains compact numerical data underlying Figures 1-4 and Tables I-II of the negative-Rossby-number BEK study: 11 cases, 224 branch points, field projections and damping-response envelopes, plus standalone plotting scripts. See its README for definitions and reproduction commands. No new flow solve is required to redraw the figures.

## Installation

Python 3.10 or newer is required. The core dependencies are NumPy and SciPy.

```bash
python3 -m venv .venv
.venv/bin/python -m pip install -e '.[test,plot]'
./run.sh --help
```

`requirements-lock.txt` records the original environment versions. The optional sparse-direct backend is listed separately in `requirements-pardiso.txt`.

## Quick start

```bash
./run.sh baseflow --ro=-0.751 --output results/bek
./run.sh temporal --re 500 --ro=-0.751 --beta 85 --alpha=0.26 --n 64 --output results/temporal
./run.sh spatial --re 500 --beta 85 --omega=0 --output results/spatial
```

`run.sh` uses the repository's `.venv` and defaults numerical libraries to one thread. Output arguments are filename prefixes. Use `--help` on individual commands for their parameters.

The local Reynolds number is positive, `Re = abs(Ro) * R`; `beta` is the azimuthal mode number. The local eigenproblem uses `A(alpha) q = omega B q`, with temporal growth `Im(omega)` and spatial growth `-Im(alpha)`. Check the individual global or nonlinear driver for its frequency and field scaling.

## Checks without external datasets

```bash
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 \
  .venv/bin/python -m pytest -q tests/test_numerics.py \
  -k 'not legacy_table_layouts_and_domain and not archived_matlab_values'
```

Some tests and historical campaign/postprocessing drivers need external base-flow tables, saved fields, reference PDFs, or local plotting/QA utilities. Those historical inputs are not part of the compact paper dataset. In particular, the two excluded tests above need the original `data/` reference tables. Archived absolute paths in a few historical example drivers should be configured for the user's environment before running them.

## Reproducibility and citation

The repository provides the source implementation and the compact negative-Rossby-number paper dataset. The data README maps the released inputs to each figure and table. Cite the repository URL and the exact commit used in a calculation:

https://github.com/xieyaguang-gif/LSA-BEK-MODEL
