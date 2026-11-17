#!/bin/bash
#SBATCH --job-name=ap-pnc-thickness
#SBATCH --account=liu_lab
#SBATCH --partition=short
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=2
#SBATCH --mem=4G
#SBATCH --time=00:05:00
#SBATCH --output=/home/usd.local/kunming.dong/Phononic_Nanobeam_Design_Lab_AP_pnc/lawrence-thickness-%j.out
#SBATCH --open-mode=append
#SBATCH --no-requeue

# Athul Prem | _AP_pnc
set -euo pipefail
umask 077
if [[ $# -ne 4 ]]; then
    printf 'Usage: sbatch %s PROJECT_ROOT preflight H80_CONFIG H80_MESH_DIR\n' "$0" >&2
    printf '       sbatch %s PROJECT_ROOT safe PREFLIGHT_DIR PREFLIGHT_SHA\n' "$0" >&2
    exit 2
fi
case "$2" in
    preflight) options=(--h80-config "$3" --h80-mesh-dir "$4");;
    safe) options=(--preflight-dir "$3" --preflight-sha256 "$4");;
    *) exit 2;;
esac
cd -P -- "$1"
job_id=${SLURM_JOB_ID:?Launch through Slurm with sbatch}
output="$PWD/runs/thickness-h80-$2-$job_id"
python="$PWD/.venv-lawrence-export/bin/python"
driver="$PWD/tools/run_lawrence_thickness_AP_pnc.py"
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
    sha256sum -- "$0" "$driver"
} > "$output/allocation_AP_pnc.txt" 2>&1
printf 'Job: %s\nPhase: %s\nOutput: %s\n' "$job_id" "$2" "$output"
exec /usr/bin/time -v -o "$output/time_AP_pnc.txt" timeout --signal=INT --kill-after=5s 270s \
    "$python" -B -u "$driver" "$2" "${options[@]}" --output "$output"
