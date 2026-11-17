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

## Run the h80 uniform-guide control

Check the [current handoff](AI_HANDOFF_AP_pnc.md) for job ownership and saved results before submitting anything. Reuse any completed preflight or solve. The control is a hole-free 3 µm guide at 1 GHz with a 1000 nm × 200 nm section, four thickness layers and the saved 12-by-4 semi-analytical finite element (SAFE) port modes. It uses `.venv-lawrence-export` and verifies the original h100 summary, fields and tagged mesh in `inputs/uniform_h100_AP_pnc/`, including the two checkpoint files in `mesh/`.

From the Lawrence project directory, the mesh-only job uses `short` with 4 GiB and a three-minute allocation. Its `uniform_h80_AP_pnc.json` receipt reports `mesh_ready` and the measured `case_max_dofs`. The original iterative attempt ended at its 570-second limit. For a fresh solve, use `solve-direct` with the checked preflight job ID and DOF count. It reuses the exact mesh and trace map, with the measured DOF count as the case cap. The allocation is two CPUs and 16 GiB for ten minutes on `nodes`, with one numerical thread and a 570-second worker limit.

```bash
batch=scripts/hpc/lawrence_uniform_h80_AP_pnc.sbatch
h100="$PWD/inputs/uniform_h100_AP_pnc"
ports="$PWD/results/runs/full_device_resource_profile/full_h80_operator_1r5fdlft/ports_AP_pnc.npz"
sbatch --parsable --partition=short --mem=4G --time=00:03:00 "$batch" "$PWD" preflight "$h100" "$ports"
```

```bash
preflight="$PWD/runs/uniform-h80-preflight-PREFLIGHT_JOB_ID"
cat "$preflight/uniform_h80_AP_pnc.json"
sbatch --parsable "$batch" "$PWD" solve-direct "$h100" "$ports" "$preflight" MEASURED_DOFS
```

The solve checks available node and job memory before factoring the complex128 equations. One factor serves all five drives. `direct_channels_AP_pnc.json` records both original-equation residuals and the checksum of each accepted vector as it is saved. Both residual limits are `1e-9`. The solve receipt in `runs/uniform-h80-solve-direct-SOLVE_JOB_ID/` identifies the public result. Keep the allocation, timing and channel records and check the independent flux balance.

Retrieve and verify the public result. From the local project directory, set `$h80` to that downloaded run directory and compare it with the saved h100 run and the qualified 20-by-6 section. The results retain the guide's original end-plane phases and apply to this uniform guide at 1 GHz. Leave the separate patterned-device results at their own end planes.

```powershell
$h100 = 'results/runs/full_device_uniform_diagnostic/h100_followup/uniform_3um_layered_h100_ny12_z4-e035d9a65c07-attempt2'
$h80 = 'DOWNLOADED_H80_PUBLIC_RUN_DIRECTORY'
.venv/Scripts/python.exe -B tools/compare_scattering_AP_pnc.py --reference $h100 --candidate $h80
.venv/Scripts/python.exe -B tools/compare_uniform_reference_AP_pnc.py --run $h80 --reference verification/full_device_port_resolution_AP_pnc.json --reference-case 20x6
```

## Run the h60 full-device control

Check the handoff for job ownership, approved receipt hashes, and current resource admission. Reuse completed phases and accepted drives. Both drivers use `.venv-lawrence-export` with one numerical thread. [Current handoff](AI_HANDOFF_AP_pnc.md)

```bash
sbatch --parsable scripts/hpc/lawrence_device_refinement_AP_pnc.sbatch PROJECT_ROOT H80_CONFIG H80_MESH_DIR PORTS_NPZ
sbatch --parsable scripts/hpc/lawrence_device_compute_AP_pnc.sbatch PROJECT_ROOT PHASE WALL_SECONDS INPUT_DIR INPUT_SHA [PRIOR_OR_SOLVE_DIR RECEIPT_SHA]
```

1. The mesh-only preflight uses `tools/run_lawrence_device_refinement_AP_pnc.py` with the original h80 configuration, tagged mesh checkpoint, and saved port modes. The `mesh_ready` receipt `device_h60_preflight_AP_pnc.json` has SHA-256 `ceb60bb34585bf583776ffa010dd2e040da542f036d392b3c79229af8438a8a0`. It verifies the original 675-DOF traces and end planes, four thickness layers, and all 26 hole walls with 32 segments around each rim. The lead interiors are included in the global refinement. The measured 680,265-DOF case cap applies to this assembly. The standard reporting cap remains 60,000 DOFs.
2. For `stage`, pass the preflight directory and the exact preflight receipt hash as `INPUT_DIR` and `INPUT_SHA`. The compute batch calls `tools/solve_lawrence_device_refinement_AP_pnc.py`. The stage uses 16 GiB, two CPUs, and a 550-second worker budget within a ten-minute Slurm allocation. It saves stiffness, mass, and the exact original complex equations in the stage directory with `stage_AP_pnc.json` before factorization.
3. For `solve`, pass the stage directory and exact `stage_AP_pnc.json` hash. Choose the factor allocation from measured sparse operator storage and fresh resource evidence. The solver checks node and Slurm memory headroom before factorization and at factor and drive boundaries. The Slurm memory limit and the worker timeout bound the process. One complex128 factor serves all pending drives. To recover after interruption, append the stopped prior solve directory and exact `solve_AP_pnc.json` hash. The driver replays accepted drives against the original equations and solves the pending drives only.
4. For `export`, pass the same stage directory/hash and append the completed solve directory and exact `solve_AP_pnc.json` hash. The export reconstructs the original equations, rechecks all five accepted vectors, and calculates channel power and independent traction flux at the original end planes. It identifies the public output in `export_AP_pnc.json`. Verify the returned files before comparing saved h80 and h60 results with `tools/compare_scattering_AP_pnc.py`.

Athul Prem's two-port coherent-drive test verifies power consistency for a small uniform guide. The h60 comparison measures in-plane sensitivity at 1 GHz with fixed ports and hole faceting. Separate thickness, faceting, port, and frequency checks are required for broader full-device claims.

## Run the h80 thickness control

The h80 control keeps the physical 200 nm thickness, 26 hole polygons, width subdivisions, end planes, and exact in-plane footprint. Six thickness layers change the body discretization and its matching port traces together. Reuse verified artifacts and check ownership, accepted-drive records, and current resource admission before submitting work. [Current handoff](AI_HANDOFF_AP_pnc.md)

```bash
sbatch --parsable scripts/hpc/lawrence_thickness_AP_pnc.sbatch PROJECT_ROOT preflight H80_CONFIG H80_MESH_DIR
sbatch --parsable scripts/hpc/lawrence_thickness_AP_pnc.sbatch PROJECT_ROOT safe PREFLIGHT_DIR PREFLIGHT_SHA
sbatch --parsable scripts/hpc/lawrence_thickness_compute_AP_pnc.sbatch PROJECT_ROOT stage WALL_SECONDS PREFLIGHT_DIR PREFLIGHT_SHA SAFE_DIR SAFE_SHA
sbatch --parsable scripts/hpc/lawrence_thickness_compute_AP_pnc.sbatch PROJECT_ROOT solve WALL_SECONDS STAGE_DIR STAGE_SHA [PRIOR_SOLVE_DIR PRIOR_SOLVE_SHA]
sbatch --parsable scripts/hpc/lawrence_thickness_compute_AP_pnc.sbatch PROJECT_ROOT export WALL_SECONDS STAGE_DIR STAGE_SHA SOLVE_DIR SOLVE_SHA
```

The mesh preflight checks exact x-y equality against the archived h80 mesh and records the measured displacement count for the case cap in `thickness_h80_preflight_AP_pnc.json`. The `safe` phase uses that same mesh's P2 trace layout to calculate the complete 12-by-6 semi-analytical finite element section at 1 GHz. It saves the local trace order, section operators and modes in `ports_AP_pnc.npz`, with their identity in `thickness_h80_safe_AP_pnc.json`.

Pass the exact SHA-256 of each prior receipt. Assembly reuses the accepted mesh and SAFE section to save the original stiffness, mass, harmonic operator and drives under `stage_AP_pnc.json`. Choose solve memory and time from the measured operator and live allocation checks. A solve can take a stopped prior `solve_AP_pnc.json` receipt to replay accepted drives and compute the remaining channels. Export uses the completed solve receipt to reconstruct the exact original equations, replay the vectors and check independent flux. The comparison uses both saved runs at the original end planes and measures the combined change in thickness discretization and its matching SAFE section at 1 GHz.

The completed control used jobs 1335307/1335308/1335309/1335321/1335341. Reuse their saved artifacts under `results/runs/full_device_resource_profile/lawrence_thickness_AP_pnc/`; no job remains active. The five-drive solve took 2005.88 s with 33.712 GiB peak RSS and 34.776 GiB cgroup peak inside 128 GiB. Exact export on stage host node038 took 25.56 s within 16 GiB. The z4/z6 comparison passes power (0.0000647273) but fails complex response (0.0167063); independent residual and flux checks do not establish spatial convergence. [Verified outcomes and identities](../verification/device_thickness_AP_pnc.json)

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
