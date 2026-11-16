#!/bin/bash
#SBATCH --job-name=ap-pnc-bootstrap
#SBATCH --account=liu_lab
#SBATCH --partition=short
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=2
#SBATCH --mem=4G
#SBATCH --time=00:10:00
#SBATCH --output=/home/usd.local/kunming.dong/Phononic_Nanobeam_Design_Lab_AP_pnc/lawrence-bootstrap-%j.out
#SBATCH --open-mode=append
#SBATCH --no-requeue

# Authored for Athul Prem | _AP_pnc
set -euo pipefail
umask 077

if (( $# > 1 )); then
    printf 'Usage: sbatch %s [PROJECT_ROOT]\n' "$0" >&2
    exit 2
fi
cd -P -- "${1:-${SLURM_SUBMIT_DIR:?Submit from the project root or pass PROJECT_ROOT}}"
project_root=$PWD
job_id=${SLURM_JOB_ID:?Launch through Slurm with sbatch}
base_python="$HOME/miniforge3/bin/python3"
venv_dir="$project_root/.venv-lawrence"
runner="$project_root/tools/lawrence_solver_AP_pnc.py"
report_dir="$project_root/runs/bootstrap-$job_id"

if [[ ! -x "$base_python" || ! -f "$runner" ]]; then
    printf 'Expected the existing Miniforge Python and the Lawrence solver runner.\n' >&2
    exit 2
fi
mkdir -p -- "$project_root/runs"
mkdir -- "$report_dir"
{
    date -u +'%Y-%m-%dT%H:%M:%SZ'
    hostname
    scontrol show job "$job_id"
    cat /proc/self/cgroup
    awk '/ - cgroup2? /' /proc/self/mountinfo
} > "$report_dir/allocation.txt" 2>&1

# Reserve a new environment directory so a failed bootstrap remains inspectable.
mkdir -- "$venv_dir"
unset PYTHONHOME PYTHONPATH
export PYTHONNOUSERSITE=1
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1
export BLIS_NUM_THREADS=1 NUMEXPR_NUM_THREADS=1
"$base_python" -m venv "$venv_dir"
python="$venv_dir/bin/python"
"$python" -m pip --isolated --disable-pip-version-check install \
    --no-input --no-cache-dir --only-binary=:all: --retries 0 \
    'numpy>=2.2,<3' 'scipy>=1.15,<2' 'psutil>=7,<8' 'threadpoolctl>=3.5,<4' \
    'ipykernel>=6.29,<8'
"$python" -m pip check
"$python" -m pip freeze --all > "$report_dir/pip-freeze.txt"
"$python" - <<'PY' > "$report_dir/platform-and-blas.txt"
import json
import platform
import sys

import numpy
import scipy
from scipy.linalg import blas
from threadpoolctl import threadpool_info

print(sys.version)
print(platform.platform())
print(sys.executable)
numpy.show_config()
scipy.show_config()
blas.dgemm(alpha=1.0, a=numpy.eye(2), b=numpy.eye(2))
print(json.dumps(threadpool_info(), indent=2))
PY
"$python" -u "$runner" --smoke-only --attempt-dir "$report_dir/smoke"
kernel_name=phononic-ap-pnc
kernel_data_dir=$("$python" -c 'from jupyter_core.paths import jupyter_data_dir; print(jupyter_data_dir())')
kernel_dir="$kernel_data_dir/kernels/$kernel_name"
if [[ -e "$kernel_dir" || -L "$kernel_dir" ]]; then
    printf 'An existing kernel already uses the name %s. Preserved: %s\n' "$kernel_name" "$kernel_dir" >&2
    exit 2
fi
"$python" -m ipykernel install --user --name "$kernel_name" --display-name 'Phononic Nanobeam AP_pnc'
touch -- "$venv_dir/.lawrence-bootstrap-passed"
printf 'Bootstrap and solver smoke passed. Evidence: %s\n' "$report_dir"
