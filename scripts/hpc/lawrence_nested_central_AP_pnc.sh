#!/bin/bash
# Athul Prem | _AP_pnc. Run in an existing interactive Lawrence compute shell.
set -euo pipefail
umask 077
if [[ $# -ne 6 ]]; then
    printf 'Usage: bash %s PROJECT_ROOT STUDY_RELATIVE PYTHON SOURCE_REVISION THICK_TRANSFER_SHA THIN_TRANSFER_SHA\n' "$0" >&2
    exit 2
fi
cd -P -- "$1"
: "${SLURM_JOB_ID:?Run inside the interactive Lawrence compute allocation}"
unset PYTHONHOME PYTHONPATH
export PYTHONNOUSERSITE=1
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 BLIS_NUM_THREADS=1 NUMEXPR_NUM_THREADS=1
exec "$3" -B -u tools/run_lawrence_nested_central_AP_pnc.py "$2" "$4" "$5" "$6"
