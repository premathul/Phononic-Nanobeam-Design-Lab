#!/bin/bash
#SBATCH --job-name=ap-pnc-thickness
#SBATCH --account=liu_lab
#SBATCH --partition=nodes
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=2
#SBATCH --mem=16G
#SBATCH --time=00:10:00
#SBATCH --output=/home/usd.local/kunming.dong/Phononic_Nanobeam_Design_Lab_AP_pnc/lawrence-thickness-%j.out
#SBATCH --open-mode=append
#SBATCH --no-requeue

# Athul Prem | _AP_pnc
set -euo pipefail
umask 077
if [[ $# -lt 5 || ! $3 =~ ^[1-9][0-9]*$ ]]; then
    printf 'Usage: sbatch %s PROJECT_ROOT {stage|solve|export} WALL_SECONDS INPUT_DIR INPUT_SHA [NEXT_DIR NEXT_SHA]\n' "$0" >&2
    exit 2
fi
case "$2" in
    stage) [[ $# -eq 7 ]] || exit 2; options=(--preflight-dir "$4" --preflight-sha256 "$5" --safe-dir "$6" --safe-sha256 "$7");;
    solve) [[ $# -eq 5 || $# -eq 7 ]] || exit 2; options=(--stage-dir "$4" --stage-sha256 "$5");
        if [[ $# -eq 7 ]]; then options+=(--prior-solve-dir "$6" --prior-solve-sha256 "$7"); fi;;
    export) [[ $# -eq 7 ]] || exit 2; options=(--stage-dir "$4" --stage-sha256 "$5" --solve-dir "$6" --solve-sha256 "$7");;
    *) exit 2;;
esac
cd -P -- "$1"
job_id=${SLURM_JOB_ID:?Launch through Slurm with sbatch}
output="$PWD/runs/thickness-h80-$2-$job_id"
python="$PWD/.venv-lawrence-export/bin/python"
driver="$PWD/tools/solve_lawrence_thickness_AP_pnc.py"
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
exec /usr/bin/time -v -o "$output/time_AP_pnc.txt" timeout --signal=INT --kill-after=5s "${3}s" \
    "$python" -B -u "$driver" "$2" "${options[@]}" --output "$output"
