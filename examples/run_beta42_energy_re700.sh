#!/usr/bin/env bash
set -euo pipefail
cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.."
for dr in 6 8 12; do
    suffix=""; [[ "$dr" == 6 ]] || suffix="_dr$dr"
    root="results/beta42_nonlinear_spatial/energy_sweep_re700$suffix"
    for pair in '1 1' '10 3.1622776601683795' '100 10' '1000 31.622776601683793' '10000 100'; do
        read -r energy amplitude <<< "$pair"
        ./run.sh beta42-spatial --n 64 --harmonics 7 --dr "$dr" --re-end 700 \
            --amplitude-factor "$amplitude" --no-fields --output "$root/energy_x$energy" || true
    done
done
echo 'The solver may stop before Re=700. Plot only converged points with examples/plot_beta42_energy_re700.py.'
