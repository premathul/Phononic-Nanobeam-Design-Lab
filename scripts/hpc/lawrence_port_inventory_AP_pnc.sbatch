#!/bin/bash
#SBATCH --job-name=ap-pnc-port-inventory
#SBATCH --account=liu_lab
#SBATCH --partition=short
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=2
#SBATCH --mem=4G
#SBATCH --time=00:05:00
#SBATCH --output=/home/usd.local/kunming.dong/Phononic_Nanobeam_Design_Lab_AP_pnc/lawrence-port-inventory-%j.out
#SBATCH --open-mode=append
#SBATCH --no-requeue

# Athul Prem | _AP_pnc
set -euo pipefail
umask 077
if [[ $# -ne 2 && $# -ne 3 ]]; then
    printf 'Usage: sbatch %s PROJECT_ROOT SAFE_DIR [EXISTING_INVENTORY_DIR]\n' "$0" >&2
    exit 2
fi
cd -P -- "$1"
job_id=${SLURM_JOB_ID:?Launch through Slurm with sbatch}
output="$PWD/runs/port-inventory-$job_id"
log_dir="$PWD/runs/port-inventory-job-$job_id"
options=()
if [[ $# -eq 3 ]]; then
    output=$3
    options+=(--resume)
fi
python="$PWD/.venv-lawrence-export/bin/python"
driver="$PWD/tools/run_lawrence_port_inventory_AP_pnc.py"
unset PYTHONHOME PYTHONPATH
export PYTHONNOUSERSITE=1
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1
export BLIS_NUM_THREADS=1 NUMEXPR_NUM_THREADS=1
mkdir -- "$log_dir"
{
    date -u +'%Y-%m-%dT%H:%M:%SZ'
    hostname
    scontrol show job "$job_id"
    cat /proc/self/cgroup
    awk '/ - cgroup2? /' /proc/self/mountinfo
    sha256sum -- "$0" "$driver"
} > "$log_dir/allocation_AP_pnc.txt" 2>&1
printf 'Job: %s\nOutput: %s\nAllocation logs: %s\n' "$job_id" "$output" "$log_dir"
exec /usr/bin/time -v -o "$log_dir/time_AP_pnc.txt" timeout --signal=INT --kill-after=5s 270s \
    "$python" -B -u "$driver" --safe-dir "$2" --output "$output" "${options[@]}"
