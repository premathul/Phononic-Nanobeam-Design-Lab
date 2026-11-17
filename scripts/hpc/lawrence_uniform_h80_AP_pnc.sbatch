#!/bin/bash
#SBATCH --job-name=ap-pnc-uniform-h80
#SBATCH --account=liu_lab
#SBATCH --partition=nodes
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=2
#SBATCH --mem=16G
#SBATCH --time=00:10:00
#SBATCH --output=/home/usd.local/kunming.dong/Phononic_Nanobeam_Design_Lab_AP_pnc/lawrence-uniform-h80-%j.out
#SBATCH --open-mode=append
#SBATCH --no-requeue

# Athul Prem | _AP_pnc
set -euo pipefail
umask 077

mesh_size_nm=80
if [[ ${1:-} == --h60 ]]; then
    mesh_size_nm=60
    shift
fi

if [[ $# -eq 4 && $2 == preflight ]]; then
    time_limit=150
    solve_options=()
elif [[ $# -eq 6 && ( $2 == solve-direct || ( $mesh_size_nm == 80 && $2 == solve ) ) && $6 =~ ^[1-9][0-9]*$ ]]; then
    time_limit=570
    solve_options=(--preflight-dir "$5" --max-dofs "$6")
else
    printf 'Usage: sbatch %s [--h60] PROJECT_ROOT {preflight|solve|solve-direct} H100_DIR PORTS_NPZ [PREFLIGHT_DIR MEASURED_DOFS]\n' "$0" >&2
    exit 2
fi
cd -P -- "$1"
project_root=$PWD
job_id=${SLURM_JOB_ID:?Launch through Slurm with sbatch}
output="$project_root/runs/uniform-h${mesh_size_nm}-$2-$job_id"
python="$project_root/.venv-lawrence-export/bin/python"
driver="$project_root/tools/run_lawrence_uniform_h80_AP_pnc.py"
if [[ ! -x "$python" || ! -f "$driver" || ! -d "$3" || ! -f "$4" ]]; then
    printf 'The pinned Linux environment, driver and both verified reference inputs are required.\n' >&2
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
    scontrol show job "$job_id"
    cat /proc/self/cgroup
    awk '/ - cgroup2? /' /proc/self/mountinfo
    sha256sum -- "$0"
} > "$output/allocation_AP_pnc.txt" 2>&1
printf 'Job: %s\nMode: %s\nOutput: %s\n' "$job_id" "$2" "$output"
exec /usr/bin/time -v -o "$output/time_AP_pnc.txt" timeout --signal=INT --kill-after=5s "${time_limit}s" \
    "$python" -B -u "$driver" "$2" --h100 "$3" --ports "$4" --output "$output" \
    --mesh-size-nm "$mesh_size_nm" "${solve_options[@]}"
