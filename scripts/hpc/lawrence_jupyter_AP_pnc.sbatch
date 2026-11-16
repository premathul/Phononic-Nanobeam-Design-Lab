#!/bin/bash
#SBATCH --job-name=ap-pnc-notebook
#SBATCH --account=liu_lab
#SBATCH --partition=short
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --mem=4G
#SBATCH --time=02:00:00
#SBATCH --output=/home/usd.local/kunming.dong/Phononic_Nanobeam_Design_Lab_AP_pnc/lawrence-notebook-%j.out
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
jupyter="$HOME/miniforge3/bin/jupyter-lab"
notebook_dir="$project_root/runs/notebook-$job_id"
node=$(hostname -s)

if [[ ! -x "$jupyter" || ! -f "$project_root/.venv-lawrence/.lawrence-bootstrap-passed" ]]; then
    printf 'The installed JupyterLab and a successful Lawrence bootstrap are required.\n' >&2
    exit 2
fi
mkdir -p -- "$project_root/runs"
mkdir -- "$notebook_dir"
mkdir -- "$notebook_dir/runtime"

# Use the server's generated token and keep it out of Slurm's output file.
unset PYTHONHOME PYTHONPATH JUPYTER_TOKEN JUPYTER_TOKEN_FILE
export PYTHONNOUSERSITE=1 JUPYTER_NO_CONFIG=1
export JUPYTER_RUNTIME_DIR="$notebook_dir/runtime"
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1
export BLIS_NUM_THREADS=1 NUMEXPR_NUM_THREADS=1
printf 'Jupyter node address: %s\nPrivate Jupyter log: %s\n' "$node" "$notebook_dir/jupyter.log"
exec "$jupyter" --no-browser --ServerApp.ip="$node" --ServerApp.port=8888 \
    --ServerApp.port_retries=0 --ServerApp.root_dir="$project_root" \
    > "$notebook_dir/jupyter.log" 2>&1
