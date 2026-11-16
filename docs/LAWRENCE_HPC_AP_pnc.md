# Lawrence HPC

Athul Prem | `_AP_pnc`

## Run the CPU pilot

From Windows PowerShell, connect with `ssh kunming.dong@lawrence.usd.edu`. Complete the USD login and Duo prompts in the terminal if prompted. Use the login node for transfers, hashes, and Slurm commands. Run Python setup, solver work, and notebooks in Slurm allocations. Keep the project directory private because it holds job records and data. [USD Lawrence login](https://usdrcg.gitbook.io/docs/lawrence-hpc/login)

From the Lawrence shell, use the dedicated project directory. Reuse `.venv-lawrence` when its `.lawrence-bootstrap-passed` marker exists. For a fresh deployment, the bootstrap creates the environment, records the resolved dependencies and BLAS configuration, runs a small solver check, and registers the `Phononic Nanobeam AP_pnc` kernel. Existing Miniforge and JupyterLab installations are used directly.

```bash
cd /home/usd.local/kunming.dong/Phononic_Nanobeam_Design_Lab_AP_pnc
if [[ -f .venv-lawrence/.lawrence-bootstrap-passed ]]; then
    printf 'Reusing the verified Lawrence environment.\n'
elif [[ -e .venv-lawrence ]]; then
    printf 'Review the previous bootstrap logs and environment before retrying.\n' >&2
else
    sbatch --test-only scripts/hpc/lawrence_bootstrap_AP_pnc.sbatch &&
        sbatch --parsable scripts/hpc/lawrence_bootstrap_AP_pnc.sbatch
fi
```

For a newly submitted bootstrap, save the job ID and check `lawrence-bootstrap-JOB_ID.out` and `runs/bootstrap-JOB_ID/` for the result. Before pilot submission, use the original saved operator, right-hand sides, stage and configuration records, and protected starting vector and journal in `inputs/`. Upload any missing originals and compare the remote SHA-256 values against the local copies. Keep the staged inputs intact.

```bash
stage="$PWD/inputs/full_h80_operator_1r5fdlft"
(
    cd "$stage"
    sha256sum A_AP_pnc.npz rhs_AP_pnc.npy stages_AP_pnc.json configuration_AP_pnc.json \
        h80_first_drive_correction_double_fill4_r4m1rklb/{terminal-000.npy,channels_AP_pnc.json}
)
sbatch --test-only scripts/hpc/lawrence_pilot_AP_pnc.sbatch "$PWD" "$stage"
sbatch --parsable scripts/hpc/lawrence_pilot_AP_pnc.sbatch "$PWD" "$stage"
```

Submit the pilot once after the bootstrap smoke, input hash, and scheduling checks pass. It requests two CPU cores and 128 GiB on `himem` for 25 minutes. The solver runs one numerical thread with a 1,200-second budget. The protected first channel must pass the original-equation residual checks before later channels run. The four remaining right-hand sides use the same factorization in order within the same job. The pilot stops at a failed channel or resource limit and retains the evidence. The current SciPy direct solver runs on CPUs.

Replace `JOB_ID` with the submitted job number. The queue shows active jobs and pending reasons. Accounting reports the final state and measured memory after exit. For the pilot, inspect `lawrence-pilot-JOB_ID.out` and `runs/h80-pilot-JOB_ID/`. A Slurm job result and the solver's per-channel acceptance records answer different questions. Use the solver records for scientific acceptance. [USD job submission](https://usdrcg.gitbook.io/docs/lawrence-hpc/submitting-jobs) · [Slurm accounting](https://slurm.schedmd.com/sacct.html)

```bash
squeue -j JOB_ID
sacct -j JOB_ID --format=JobID,State,ExitCode,Elapsed,ReqMem,AllocCPUS,MaxRSS
```

## Export and compare accepted vectors

`tools/export_lawrence_scattering_AP_pnc.py` restores the saved mesh, K/M and SAFE modes, reconstructs the original equation arrays exactly, replays the accepted vectors, and exports power and independent traction flux. It records the assembly, solve and export environments separately. Original end-plane phases are preserved.

The completed Linux export uses its own `.venv-lawrence-export` with the exact staged numerical package versions. `scripts/hpc/lawrence_export_AP_pnc.sbatch PROJECT_ROOT SOLVE_JOB_ID [STAGE_DIR]` sets up a fresh isolated export environment and runs one export in an 8-GiB, two-CPU, five-minute allocation, with a 120-second export limit. Preserve the existing environment and results; consult the handoff rather than resubmitting this setup job. `--linux-replay` permits an explicitly recorded platform/Python difference; physical source hashes, package versions, operator/RHS hashes and original residual gates remain checked.

Retrieve the public output directory and verify its files before opening it in the workbench or comparing it with `tools/compare_scattering_AP_pnc.py`. The current h100/h80 comparison is an offline result; it needs no new solve. Exact paths and the scientific acceptance outcomes are in the handoff.

## Open JupyterLab

After the bootstrap succeeds, submit the optional notebook job from the same Lawrence project directory. It requests one CPU and 4 GiB for two hours. Use the node assigned to the running job. The server log is private and contains the token generated for this session.

```bash
sbatch --parsable scripts/hpc/lawrence_jupyter_AP_pnc.sbatch
squeue -h -j JOB_ID -o '%T %N'
cat lawrence-notebook-JOB_ID.out
less runs/notebook-JOB_ID/jupyter.log
```

In a separate Windows PowerShell window, set `$node` to the address printed in the Slurm output and open the tunnel. Keep this window open while using the notebook.

```powershell
$node = 'NODE_ADDRESS_FROM_JOB_LOG'
ssh -N -o ExitOnForwardFailure=yes -L "127.0.0.1:32136:${node}:8888" kunming.dong@lawrence.usd.edu
```

Open `http://127.0.0.1:32136/lab` and enter the current token from the private Jupyter log. Select **Phononic Nanobeam AP_pnc** for NumPy and SciPy linear algebra using `ipykernel`. Keep the token in the local terminal and browser. Keep token-bearing logs and URLs out of Git and shared messages. End the notebook allocation with `scancel JOB_ID` and close the tunnel. `tmux` is optional for preserving a remote shell. Slurm batch jobs continue after the shell or tunnel closes. [USD Jupyter guide](https://usdrcg.gitbook.io/docs/lawrence-hpc/jupyter-on-lawrence) · [Jupyter token authentication](https://jupyter-server.readthedocs.io/en/stable/operators/security.html)
