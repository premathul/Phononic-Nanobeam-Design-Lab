#!/bin/bash
#SBATCH --job-name=ap-pnc-f8875
#SBATCH --account=liu_lab
#SBATCH --partition=nodes
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=2
#SBATCH --mem=16G
#SBATCH --time=00:10:00
#SBATCH --output=/home/usd.local/kunming.dong/Phononic_Nanobeam_Design_Lab_AP_pnc/lawrence-frequency-point-%j.out
#SBATCH --open-mode=append
#SBATCH --no-requeue

# Athul Prem | _AP_pnc
# Stage/export defaults: 16 GiB, two CPUs, 10 minutes; worker ceiling 550 s.
# Explicit solve admission may use --mem=128G --time=01:00:00; worker ceiling 3500 s.
# No retries, requeue, follow-on frequencies or resource escalation.
set -euo pipefail
umask 077
if [[ $# -lt 5 || ! $3 =~ ^[1-9][0-9]*$ ]]; then
    printf 'Usage: sbatch %s PROJECT_ROOT PHASE WALL_SECONDS PHASE_INPUTS\n' "$0" >&2
    printf '  stage  BODY_STAGE_DIR SAFE_DIR INTERVAL_DIR INTERVAL_LEDGER\n  solve  STAGE_DIR STAGE_SHA [PRIOR_SOLVE_DIR PRIOR_SOLVE_SHA]\n  export STAGE_DIR STAGE_SHA SOLVE_DIR SOLVE_SHA\n' >&2
    exit 2
fi
case "$2" in
    stage) [[ $# -eq 7 && $3 -le 550 ]] || exit 2
        options=(--body-stage-dir "$4" --safe-dir "$5" --interval-dir "$6" --interval-ledger "$7");;
    solve) [[ ( $# -eq 5 || $# -eq 7 ) && $3 -le 3500 ]] || exit 2
        options=(--stage-dir "$4" --stage-sha256 "$5")
        if [[ $# -eq 7 ]]; then options+=(--prior-solve-dir "$6" --prior-solve-sha256 "$7"); fi;;
    export) [[ $# -eq 7 && $3 -le 550 ]] || exit 2
        options=(--stage-dir "$4" --stage-sha256 "$5" --solve-dir "$6" --solve-sha256 "$7");;
    *) exit 2;;
esac
cd -P -- "$1"
job_id=${SLURM_JOB_ID:?Launch through Slurm with sbatch}
output="$PWD/runs/h80-z6-frequency-887500000-$2-$job_id"
python="$PWD/.venv-lawrence-export/bin/python"
driver="$PWD/tools/run_lawrence_frequency_point_AP_pnc.py"
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
