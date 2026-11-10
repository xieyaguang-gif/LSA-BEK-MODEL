#!/usr/bin/env bash
set -euo pipefail
cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.."
root="results/beta42_nonlinear_spatial/energy_sweep"
run_energy() {
    label="$1"
    amplitude_factor="$2"
    ./run.sh beta42-spatial --n 64 --harmonics 7 --dr 6 --no-fields \
        --amplitude-factor "$amplitude_factor" --output "$root/$label"
}
# Energy scales as amplitude squared. The x1 curve is the existing primary run.
run_energy energy_x10 3.1622776601683795
run_energy energy_x100 10
run_energy energy_x1000 31.622776601683793
run_energy energy_x10000 100
echo 'Computed four added inlet energies. Run examples/plot_beta42_energy_sweep.py to plot all five.'
