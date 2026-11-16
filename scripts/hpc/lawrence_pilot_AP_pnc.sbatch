#!/bin/bash
#SBATCH --job-name=ap-pnc-h80-pilot
#SBATCH --account=liu_lab
#SBATCH --partition=himem
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=2
#SBATCH --mem=128G
#SBATCH --time=00:25:00
#SBATCH --output=/home/usd.local/kunming.dong/Phononic_Nanobeam_Design_Lab_AP_pnc/lawrence-pilot-%j.out
#SBATCH --open-mode=append
#SBATCH --no-requeue

# Authored for Athul Prem | _AP_pnc
set -euo pipefail
umask 077

if (( $# > 2 )); then
    printf 'Usage: sbatch %s [PROJECT_ROOT [STAGE_DIR]]\n' "$0" >&2
    exit 2
fi
cd -P -- "${1:-${SLURM_SUBMIT_DIR:?Submit from the project root or pass PROJECT_ROOT}}"
project_root=$PWD
job_id=${SLURM_JOB_ID:?Launch through Slurm with sbatch}
stage_dir=${2:-"$project_root/inputs/full_h80_operator_1r5fdlft"}
venv_dir="$project_root/.venv-lawrence"
python="$venv_dir/bin/python"
runner="$project_root/tools/lawrence_solver_AP_pnc.py"
attempt_dir="$project_root/runs/h80-pilot-$job_id"

if [[ ! -x "$python" || ! -f "$venv_dir/.lawrence-bootstrap-passed" || ! -f "$runner" ]]; then
    printf 'A successful Lawrence bootstrap and the solver runner are required.\n' >&2
    exit 2
fi
if [[ ! -d "$stage_dir" ]]; then
    printf 'Staged operator directory is unavailable: %s\n' "$stage_dir" >&2
    exit 2
fi

unset PYTHONHOME PYTHONPATH
export PYTHONNOUSERSITE=1
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1
export BLIS_NUM_THREADS=1 NUMEXPR_NUM_THREADS=1
mkdir -p -- "$project_root/runs"
date -u +'%Y-%m-%dT%H:%M:%SZ'
hostname
scontrol show job "$job_id"
cat /proc/self/cgroup
printf 'Stage: %s\nAttempt: %s\n' "$stage_dir" "$attempt_dir"
exec "$python" -u "$runner" --stage-dir "$stage_dir" \
    --attempt-dir "$attempt_dir" --wall-seconds 1200
