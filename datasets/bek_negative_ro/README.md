# Negative-Rossby-number BEK data

Compact numerical data for **Local and global instability of BEK boundary layers at negative Rossby numbers: linear thresholds and nonlinear branches**.
The 11 cases comprise nine regularly spaced values from Ro = -0.9 to -0.1 and the independent reference flows Ro = -1 and -0.751.

## Contents

| File | Content |
|---|---|
| `scalar_results.csv` | 11 cases: local absolute thresholds, paired linear/nonlinear fronts and disk-scaled frequencies, N32 branch neutral/fold positions, and all Table I state parameters |
| `branches.csv` | All 224 branch points plotted in Figure 1, retaining harmonic order and radial/wall-normal resolution |
| `resolution_checks.csv` | The two values and full-precision differences behind each Table II comparison (eight quantities in seven table rows) |
| `data/wave_structure/` | Figure 2 envelopes and signed field projections at Ro = -0.5 |
| `data/damping_response/` | Figure 4 scalars and envelopes for damping strengths 14, 28, 56 |
| Other small `data/*.json` files | The plotting inputs for Figures 1 and 3, in the original plotting schema |
| `provenance.json`, `manifest.json` | Figure/table mapping, source fingerprints, and release-file checksums |

CSV tables are convenient for analysis. The small JSON inputs retain the plotting schema, so the figure code reads the original values without a new interpolation or fit. NPZ files contain numeric arrays only and can be loaded with `numpy.load(..., allow_pickle=False)`.

## Reproduce the four figures

From this directory, with Python 3.10+, NumPy and Matplotlib installed:

```bash
python3 verify.py
python3 plot_figures.py
python3 plot_wave_structure.py
python3 plot_damping_response.py
```

The commands write four PDF/PNG pairs under `figures/`, and do not solve the governing equations. They use Times New Roman when available, then Nimbus Roman or STIXGeneral. Numerical arrays are unchanged by the font choice. The source repository's `[plot]` installation extra supplies the plotting dependencies.

## Variables and scaling

- `R`, `z`: radius and height divided by sqrt(nu/Omega*); `Re_ca = abs(Ro) * R_ca`.
- `beta` is the global-wave azimuthal mode number; `beta_local` is the local absolute-threshold mode number. They differ in the cavity reference case.
- `Rout_L` and `Rout_NL` are the outer boundaries of the paired linear neutral and nonlinear states. They are separate from the internal front positions `Rf_L`, `Rf_NL`, and the auxiliary second-harmonic detection location `Rf_2`.
- Frequencies `omega_L_disk`, `omega_NL_disk`, and `omega_disk` use the inertial-frame disk timescale, omega_d = omega/d, where d = (2 - Ro - Ro^2)/2. Subtract beta for the frequency in the disk frame.
- `A1` is the scaled fundamental velocity coefficient norm. `N` is wall-normal order, `Nr` is the number of radial intervals (Nr + 1 nodes), and `H` is positive harmonic truncation.
- `Rout_neutral_branch_N32` and `Rout_fold_N32` are the branch-grid references used in Figure 1. These must not be substituted for the refined field-pair boundaries in Table I.
- `resolution_checks.csv` uses absolute changes for front/boundary locations and relative percent changes for the two A1 checks. Its N32/N40 field comparisons can change both radial and wall-normal resolution. `r_end` is the integrated linear-envelope location, distinct from the common azimuthal-velocity front `Rf`.

The common front is the first downward crossing of d(log(Vq))/dR with the matched local spatial growth rate, using Vq = sqrt(2 sum(j=1..H) |v_j(R,z=1.3)|^2). The factor R that reconstructs physical velocity is excluded from these scaled coefficients. The second harmonic uses the first upward crossing of |v2| = 1e-6 in the specified undamped window.

For Figure 2, `field_v1_real_twice` is 2 Re(v1) at phase beta theta - omega t = 0, and `field_v0_real` is the mean azimuthal correction. `field_r` contains native radial nodes; `field_z` contains 161 display heights obtained by the original barycentric interpolation. The arrays have shape (R, z), and no new smoothing is applied. The linear envelope is scaled by the recorded constant `linear_display_scale` solely for display. The metadata records the precise state and normalization.

This release contains the results underlying the paper figures and tables, rather than the large raw three-dimensional fields or solver histories. The numerical implementation remains in the parent repository. Cite this repository with the exact commit used; no external dataset identifier is assigned.
