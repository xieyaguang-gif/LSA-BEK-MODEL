#!/usr/bin/env bash
set -euo pipefail
cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.."
study_root="results/beta42_nonlinear_spatial"
dns_input="$study_root/inputs/af01_semtex_turn4_vtheta.csv"
run_case() {
    case_name="$1"
    shift
    ./run.sh beta42-spatial --dns "$dns_input" --output "$study_root/$case_name" "$@"
    # Preserve the full field only for the principal result; checks need curves.
    if [[ "$case_name" != primary ]]; then
        rm -- "$study_root/$case_name/fields.npz"
    fi
}
run_case primary --n 64 --harmonics 7 --dr 6
run_case checks/n48h5_dr6 --n 48 --harmonics 5 --dr 6
run_case checks/n48h7_dr6 --n 48 --harmonics 7 --dr 6
run_case checks/n64h7_dr8 --n 64 --harmonics 7 --dr 8
run_case checks/n48h5_dr5 --n 48 --harmonics 5 --dr 5
run_case checks/linear --n 48 --harmonics 1 --dr 6 --linear --amplitude-factor 0.000001
run_case checks/small_nonlinear --n 48 --harmonics 3 --dr 6 --amplitude-factor 0.000001
run_case inlet/reference --n 48 --harmonics 5 --dr 8
run_case inlet/half --n 48 --harmonics 5 --dr 8 --amplitude-factor 0.5
run_case inlet/double --n 48 --harmonics 5 --dr 8 --amplitude-factor 2
echo 'Finished. Plot using examples/plot_beta42_nonlinear_spatial.py (NumPy, SciPy, Matplotlib required).'
