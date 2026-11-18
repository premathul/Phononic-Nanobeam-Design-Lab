#!/bin/bash
#SBATCH --job-name=ap-pnc-f8875-h100
#SBATCH --account=liu_lab
#SBATCH --partition=nodes
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=2
#SBATCH --mem=16G
#SBATCH --time=00:10:00
#SBATCH --output=/home/usd.local/kunming.dong/Phononic_Nanobeam_Design_Lab_AP_pnc/lawrence-frequency-h100-%j.out
#SBATCH --open-mode=append
#SBATCH --no-requeue

# Athul Prem | _AP_pnc
# Stage/export: 16 GiB / 600 s allocation, at most 550 s worker.
# Solve: submit --mem=48G --time=00:15:00, at most 850 s worker.
# One selected point, two allocated CPUs, one numerical thread; no follow-up.
set -euo pipefail
umask 077
if [[ $# -lt 5 || ! $3 =~ ^[1-9][0-9]*$ ]]; then
    printf 'Usage: sbatch %s PROJECT_ROOT PHASE WALL_SECONDS PHASE_INPUTS\n' "$0" >&2
    printf '  stage  BODY_STAGE_DIR MESH_DIR SAFE_DIR ORIGINAL_1GHZ_A SAFE_VERIFICATION REFERENCE_POINT\n  solve  STAGE_DIR STAGE_SHA [PRIOR_SOLVE_DIR PRIOR_SOLVE_SHA]\n  export STAGE_DIR STAGE_SHA SOLVE_DIR SOLVE_SHA\n' >&2
    exit 2
fi
case "$2" in
    stage) [[ $# -eq 9 && $3 -le 550 ]] || exit 2
        options=(--body-stage-dir "$4" --mesh-dir "$5" --safe-dir "$6" --reference-operator "$7"
                 --safe-verification "$8" --reference-point "$9");;
    solve) [[ ( $# -eq 5 || $# -eq 7 ) && $3 -le 850 ]] || exit 2
        options=(--stage-dir "$4" --stage-sha256 "$5")
        if [[ $# -eq 7 ]]; then options+=(--prior-solve-dir "$6" --prior-solve-sha256 "$7"); fi;;
    export) [[ $# -eq 7 && $3 -le 550 ]] || exit 2
        options=(--stage-dir "$4" --stage-sha256 "$5" --solve-dir "$6" --solve-sha256 "$7");;
    *) exit 2;;
esac
cd -P -- "$1"
job_id=${SLURM_JOB_ID:?Launch through Slurm with sbatch}
output="$PWD/runs/h100-z4-frequency-887500000-$2-$job_id"
python="$PWD/.venv-lawrence-export/bin/python"
driver="$PWD/tools/run_lawrence_frequency_h100_AP_pnc.py"
unset PYTHONHOME PYTHONPATH
export PYTHONNOUSERSITE=1
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1
export BLIS_NUM_THREADS=1 NUMEXPR_NUM_THREADS=1 AP_PNC_WORKER_SECONDS="$3"
mkdir -- "$output"
{
    date -u +'%Y-%m-%dT%H:%M:%SZ'
    hostname
    scontrol show job "$job_id"
    cat /proc/self/cgroup
    awk '/ - cgroup2? /' /proc/self/mountinfo
    sha256sum -- "$0" "$driver" "$PWD/tools/run_lawrence_low_band_h100_AP_pnc.py"
} > "$output/allocation_AP_pnc.txt" 2>&1
printf 'Job: %s\nPhase: %s\nOutput: %s\n' "$job_id" "$2" "$output"
exec /usr/bin/time -v -o "$output/time_AP_pnc.txt" timeout --signal=INT --kill-after=5s "${3}s" \
    "$python" -B -u "$driver" "$2" "${options[@]}" --output "$output"
