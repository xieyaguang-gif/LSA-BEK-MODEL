#!/usr/bin/env bash
set -euo pipefail
cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.."
root="results/beta42_nonlinear_spatial/energy_sweep_ultrastrong"
# The existing 1e4 E0, dr=12 result is retained as the first curve.
for pair in '100000 316.22776601683796' '1000000 1000' '10000000 3162.2776601683795' '100000000 10000'; do
    read -r energy amplitude <<< "$pair"
    ./run.sh beta42-spatial --n 64 --harmonics 7 --dr 12 --re-end 700 \
        --amplitude-factor "$amplitude" --no-fields --output "$root/energy_x$energy" || true
done
echo 'Plot converged points with examples/plot_beta42_ultrastrong_energy.py.'
