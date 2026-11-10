#!/usr/bin/env bash
set -euo pipefail
LLSA_ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
export PYTHONPATH="$LLSA_ROOT/src${PYTHONPATH:+:$PYTHONPATH}"
# Dense matrices here are small; many BLAS threads can cost far more than they save.
export OPENBLAS_NUM_THREADS="${LLSA_BLAS_THREADS:-1}"
export OMP_NUM_THREADS="${LLSA_BLAS_THREADS:-1}"
export MKL_NUM_THREADS="${LLSA_BLAS_THREADS:-1}"
exec "$LLSA_ROOT/.venv/bin/python" -m llsa "$@"
