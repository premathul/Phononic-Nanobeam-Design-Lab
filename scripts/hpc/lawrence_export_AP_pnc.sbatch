#!/bin/bash
#SBATCH --job-name=ap-pnc-h80-export
#SBATCH --account=liu_lab
#SBATCH --partition=short
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=2
#SBATCH --mem=8G
#SBATCH --time=00:05:00
#SBATCH --output=/home/usd.local/kunming.dong/Phononic_Nanobeam_Design_Lab_AP_pnc/lawrence-export-%j.out
#SBATCH --open-mode=append
#SBATCH --no-requeue

# Athul Prem | _AP_pnc
set -euo pipefail
umask 077

if [[ $# -lt 2 || $# -gt 3 || ! "${2:-}" =~ ^[0-9]+$ ]]; then
    printf 'Usage: sbatch %s PROJECT_ROOT SOLVE_JOB_ID [STAGE_DIR]\n' "$0" >&2
    exit 2
fi
cd -P -- "$1"
project_root=$PWD
solve_job_id=$2
export_job_id=${SLURM_JOB_ID:?Launch through Slurm with sbatch}
stage_dir=${3:-"$project_root/results/runs/full_device_resource_profile/full_h80_operator_1r5fdlft"}
base_python="$HOME/miniforge3/bin/python3"
venv_dir="$project_root/.venv-lawrence-export"
exporter="$project_root/tools/export_lawrence_scattering_AP_pnc.py"
attempt="$project_root/runs/h80-pilot-$solve_job_id"
output="$project_root/runs/h80-export-$export_job_id"
if [[ ! -x "$base_python" || ! -f "$exporter" || ! -d "$attempt" || ! -d "$stage_dir" ]]; then
    printf 'Miniforge Python, the exporter, the saved stage and the completed solver attempt are required.\n' >&2
    exit 2
fi

unset PYTHONHOME PYTHONPATH
export PYTHONNOUSERSITE=1
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1
export BLIS_NUM_THREADS=1 NUMEXPR_NUM_THREADS=1
mkdir -- "$output"
{
    date -u +'%Y-%m-%dT%H:%M:%SZ'
    hostname
    scontrol show job "$export_job_id"
    cat /proc/self/cgroup
    awk '/ - cgroup2? /' /proc/self/mountinfo
} > "$output/allocation_AP_pnc.txt" 2>&1
printf 'Solve job: %s\nExport job: %s\nOutput: %s\n' "$solve_job_id" "$export_job_id" "$output"

"$base_python" -I - "$stage_dir/stages_AP_pnc.json" > "$output/requirements-pinned_AP_pnc.txt" <<'PY'
import hashlib
import json
from pathlib import Path
import sys

raw = Path(sys.argv[1]).read_bytes()
if hashlib.sha256(raw).hexdigest() != "1c50971e27bfc52e63000c9d25df100e21bbe28b7fa47d7549f82cf465531a78":
    raise SystemExit("Saved h80 stage manifest identity differs.")
packages = json.loads(raw)["source"]["packages"]
for name, version in packages.items():
    print(f"{name}=={version}")
print("threadpoolctl==3.8.0")
PY

# A new directory keeps either the successful environment or the failed setup intact.
mkdir -- "$venv_dir"
"$base_python" -m venv "$venv_dir"
python="$venv_dir/bin/python"
"$python" -m pip --isolated --disable-pip-version-check install \
    --no-input --no-cache-dir --only-binary=:all: --retries 0 -r "$output/requirements-pinned_AP_pnc.txt"
"$python" -m pip check
"$python" -m pip freeze --all > "$output/pip-freeze_AP_pnc.txt"
"$python" -B - <<'PY' > "$output/platform-and-blas_AP_pnc.json"
import json

from scipy.linalg import blas

from tools import lawrence_solver_AP_pnc as solver

identity = solver.solve_environment()
identity.pop("runner_sha256")
print(json.dumps(identity, indent=2))
PY

exec /usr/bin/time -v timeout --kill-after=5s 120s "$python" -B -u "$exporter" \
    --linux-replay --attempt-dir "$attempt" --slurm-job-id "$solve_job_id" --stage-dir "$stage_dir" --output-root "$output"
