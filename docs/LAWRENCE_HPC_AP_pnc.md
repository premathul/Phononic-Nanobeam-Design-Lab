# Lawrence HPC

Athul Prem | `_AP_pnc`

[Research documentation](README.md) · [Current ownership and execution state](AI_HANDOFF_AP_pnc.md) · [Verified preparation handoff](#prepared-26-hole-nested-central-response)

## Interactive Lawrence workflow

Root owns authentication, transfers and compute allocations. Run calculations projected above 10 GB (10,000,000,000 bytes) directly on Lawrence. Request an interactive compute allocation and invoke the runner directly from its compute shell. Batch job submission is prohibited. Keep authentication, transfers and allocation commands on the login node. Run Python setup, solves and any notebook kernels on a compute node. Use the existing JupyterLab for visibility.

On the project Windows host, run `& 'C:/Program Files/Git/bin/bash.exe' '/c/Users/kunmi/AppData/Local/PhononicNanobeam/Lawrence/connect-lawrence.sh'` in PowerShell. The launcher reuses an active Lawrence connection. For a new login it uses the current user's Windows Credential Manager entry `lawrence.usd.edu` for `Kunming.Dong`, and selects Duo Push when offered. A remote hostname and account from `hostname -s && id -un` establish a working command session. The host-specific launcher is installed outside the research repository. With its master active, use Git OpenSSH with `-S ~/.ssh/lawrence-pnc-active.ctl -O proxy` for further commands, and `-O exit` after releasing the allocation. Inspect existing allocations and remote checkpoints before starting a calculation. [USD Lawrence login](https://usdrcg.gitbook.io/docs/lawrence-hpc/login)

## Prepared 26-hole nested central response

Allocation 1336454 on node079 filled three gaps in the thin 995/997.5/1000 MHz dataset. Two one-point invocations of `tools/seed_frequency_map_AP_pnc.py` reused the accepted 126.5625 nm Linux 1 GHz preparation at 995 and 997.5 MHz. The 225 nm midpoint used an exact Windows preparation exported by `tools/transfer_frequency_preparation_AP_pnc.py` and solved on Linux. Independent flux checks bind the final checkpoints. The three new process peaks were 4,857,436, 4,859,752 and 1,938,236 KiB respectively. The two full native maps remain under `phononic_nested_3965855_AP_pnc/results/runs/physical_thickness_device_AP_pnc`. The returned point-data archive has SHA-256 `2ebd3f07ec44a0750a28d36eb864469443f734a15f7ba6ee300c787ed96cf89d`. The local import restored the five byte-identical shared artifacts from each local donor and verified all native fields and flux receipts. [Accounting](../results/runs/physical_thickness_device_AP_pnc/local_checks/lawrence_interactive_1336454_AP_pnc.txt) · [Windows preparation](../results/runs/physical_thickness_device_AP_pnc/local_checks/thin100_edge225_997p5_windows_preparation_AP_pnc.json)

Interactive allocation 1336453 ran the two 100 nm hole-pair controls on node079. The private stage contained the solver sources, original hole table and eight donor files from the repository. Each control assembled its own volume and section matrices and ran the native independent flux check. The 97,785-DOF cavity and 97,605-DOF outer-taper solves used 2,508,388 and 2,467,200 KiB peak resident memory, respectively. JupyterLab ran on the allocated node. The allocation returned after 3:28 with exit 0:0. Both full native maps, including the raw fields and operators, are retained locally and in the private `phononic_nested_3965855_AP_pnc/goal_pairs_0dde8b1_AP_pnc` directory. The retrieved archive SHA-256 is `95c20ff85b60c2662811d78831fe2c8317c6e4976157b4228b60ab7b0017d9c4`. [Accounting](../results/runs/physical_thickness_device_AP_pnc/local_checks/lawrence_interactive_1336453_AP_pnc.txt) · [Compared responses](../verification/goal_patch_response_1GHz_AP_pnc.json)

Allocation 1336448 ran the 1 GHz full-device regional controls and the two 997.5 MHz midpoints on node031 with one CPU and 12 GiB. It returned after 21:01. The peak resident set reported for the compute step was 4,689,516 KiB. Local source revisions `84eedb7`, `7f7b50c`, `2df994b` and `da74b88` retain the exact regional implementations used for the first thin central patch, its complement, the finer thin central patch and the thick central patch, respectively. The split benchmark uses the first revision. These runs used private source directories without Git metadata on Lawrence; their records include the executed file hashes. The midpoints reused exact Windows preparations in the existing Linux project environment. All six new single-point maps were retrieved with their field and flux records. [Accounting](../results/runs/physical_thickness_device_AP_pnc/local_checks/lawrence_interactive_1336448_AP_pnc.txt) | [Regional controls](../verification/taper_patch_response_1GHz_AP_pnc.json) | [Matched-frequency result](../verification/paired_nested_midpoint_AP_pnc.json)

The 168.75 nm central ceiling produced 137,745 displacement DOFs per device and a 12.2 GB projected Windows peak. Interactive allocation 1336415 on node031 ran both thicknesses at 995 and 1000 MHz and the additional 126.5625 nm thin 1 GHz control. The commands below describe the reusable procedure. Its [accounting record](../results/runs/physical_thickness_device_AP_pnc/local_checks/lawrence_interactive_1336415_AP_pnc.txt) shows a successful exit and returned resources. Reuse the [saved results](AI_HANDOFF_AP_pnc.md).

The Windows exporter `tools/transfer_frequency_preparation_AP_pnc.py export --from-map DONOR --receipt RECEIPT` verifies native preparation and prints the receipt SHA-256. Transfer the exact numerical sources, original configurations, donor publications, shared preparation files and export receipts privately. Use the receipts to bind the packaged source revision and verify the transferred data.

```bash
srun --account=liu_lab --partition=nodes --nodes=1 --ntasks=1 \
    --cpus-per-task=1 --mem=24G --time=00:30:00 --pty bash -l
```

From the compute shell, set `PROJECT_ROOT` to the private staged directory and `PYTHON_312` to its verified Linux Python 3.12 interpreter. Set `DONOR_REVISION`, `THICK_RECEIPT_SHA256` and `THIN_RECEIPT_SHA256` from the verified transfers. Keep the allocation shell open while the direct runner works.

```bash
bash "$PROJECT_ROOT/scripts/hpc/lawrence_nested_central_AP_pnc.sh" "$PROJECT_ROOT" \
    results/runs/physical_thickness_device_AP_pnc "$PYTHON_312" \
    "$DONOR_REVISION" "$THICK_RECEIPT_SHA256" "$THIN_RECEIPT_SHA256"
```

The launcher verifies the donor hashes and pinned packages, creates separate Linux maps and runs the four points sequentially through the native one-point entry. Retrieve both `*_linux` directories, their four independent flux receipts beside the maps and the four execution receipts in `local_checks`. Replay against the original Windows archive with `tools/compare_axial_nested_mesh_AP_pnc.py --linux-168p75`. Preserve Slurm accounting alongside the per-point sampled process-tree RSS and kernel peak child RSS. Keep Windows assembly and Linux solve provenance separate.

## Open existing JupyterLab

After authentication, verify the existing JupyterLab server and its assigned compute node from its private session record. Reconnect the local browser to that server for visibility into the run files and receipts. The solve runs in the interactive compute shell. If the local tunnel is absent, open one in a separate Windows PowerShell window using the verified node address and server port.

```powershell
$node = 'VERIFIED_JUPYTER_NODE'
$port = 'VERIFIED_JUPYTER_SERVER_PORT'
ssh -N -o ExitOnForwardFailure=yes -L "127.0.0.1:32136:${node}:${port}" kunming.dong@lawrence.usd.edu
```

Open `http://127.0.0.1:32136/lab` in the browser. Enter the current token from the private server log if prompted and keep token-bearing logs and URLs out of Git and shared messages. Use the **Phononic Nanobeam AP_pnc** kernel for project inspection when available. [USD Jupyter guide](https://usdrcg.gitbook.io/docs/lawrence-hpc/jupyter-on-lawrence) · [Jupyter token authentication](https://jupyter-server.readthedocs.io/en/stable/operators/security.html)

## Archived workflows and replay

The records below preserve the scientific procedures, resources and replay commands for earlier work. Historical scheduler wrappers remain in Git and the repository for provenance. Any newly justified computation follows the interactive rule above.

### CPU pilot

The pilot used a verified `.venv-lawrence` environment, identified by its `.lawrence-bootstrap-passed` marker. Bootstrap logs and environment records are retained as `lawrence-bootstrap-JOB_ID.out` and `runs/bootstrap-JOB_ID/`. The original saved operator, right-hand sides, stage and configuration records, and protected starting vector and journal are under `inputs/full_h80_operator_1r5fdlft/`.

The historical pilot used two CPU cores and 128 GiB on `himem` for 25 minutes, with one numerical thread and a 1,200-second worker budget. The protected first channel had to pass the original-equation residual checks before later channels ran. The remaining four right-hand sides shared the factorization. The pilot retained failed channels and resource stops. The current SciPy direct solver runs on CPUs.

For the pilot, inspect `lawrence-pilot-JOB_ID.out` and `runs/h80-pilot-JOB_ID/`. Use scheduler accounting for resource measurements and solver records for scientific acceptance. [Slurm accounting](https://slurm.schedmd.com/sacct.html)

```bash
squeue -j JOB_ID
sacct -j JOB_ID --format=JobID,State,ExitCode,Elapsed,ReqMem,AllocCPUS,MaxRSS
```

### Export and compare accepted vectors

`tools/export_lawrence_scattering_AP_pnc.py` restores the saved mesh, K/M and SAFE modes, reconstructs the original equation arrays exactly, replays the accepted vectors, and exports power and independent traction flux. It records the assembly, solve and export environments separately. Original end-plane phases are preserved.

The completed Linux export used `.venv-lawrence-export` with the exact staged numerical package versions. The historical wrapper `scripts/hpc/lawrence_export_AP_pnc.sbatch` configured the isolated export in an 8-GiB, two-CPU, five-minute allocation with a 120-second export limit. Preserve the existing environment and results. `--linux-replay` permits an explicitly recorded platform/Python difference; physical source hashes, package versions, operator/RHS hashes and original residual gates remain checked.

Retrieve the public output directory and verify its files before opening it in the workbench or comparing it with `tools/compare_scattering_AP_pnc.py`. The current h100/h80 comparison is an offline result; it needs no new solve. Exact paths and the scientific acceptance outcomes are in the handoff.

### h80 uniform-guide control

Reuse completed preflight and solve evidence. The control is a hole-free 3 µm guide at 1 GHz with a 1000 nm × 200 nm section, four thickness layers and the saved 12-by-4 semi-analytical finite element (SAFE) port modes. It used `.venv-lawrence-export` and verified the original h100 summary, fields and tagged mesh in `inputs/uniform_h100_AP_pnc/`, including the two checkpoint files in `mesh/`.

The mesh-only phase used `short` with 4 GiB and a three-minute allocation. Its `uniform_h80_AP_pnc.json` receipt reports `mesh_ready` and the measured `case_max_dofs`. The original iterative attempt ended at its 570-second limit. The `solve-direct` phase reused the exact mesh and trace map, with the measured DOF count as the case cap. It used two CPUs and 16 GiB for ten minutes on `nodes`, with one numerical thread and a 570-second worker limit. The original phase inputs are retained with `scripts/hpc/lawrence_uniform_h80_AP_pnc.sbatch` and the run receipts.

The solve checks available node and job memory before factoring the complex128 equations. One factor serves all five drives. `direct_channels_AP_pnc.json` records both original-equation residuals and the checksum of each accepted vector as it is saved. Both residual limits are `1e-9`. The solve receipt in `runs/uniform-h80-solve-direct-SOLVE_JOB_ID/` identifies the public result. Keep the allocation, timing and channel records and check the independent flux balance.

Retrieve and verify the public result. From the local project directory, set `$h80` to that downloaded run directory and compare it with the saved h100 run and the qualified 20-by-6 section. The results retain the guide's original end-plane phases and apply to this uniform guide at 1 GHz. Leave the separate patterned-device results at their own end planes.

```powershell
$h100 = 'results/runs/full_device_uniform_diagnostic/h100_followup/uniform_3um_layered_h100_ny12_z4-e035d9a65c07-attempt2'
$h80 = 'DOWNLOADED_H80_PUBLIC_RUN_DIRECTORY'
.venv/Scripts/python.exe -B tools/compare_scattering_AP_pnc.py --reference $h100 --candidate $h80
.venv/Scripts/python.exe -B tools/compare_uniform_reference_AP_pnc.py --run $h80 --reference verification/full_device_port_resolution_AP_pnc.json --reference-case 20x6
```

### Matched h60 uniform-guide control

The bounded h60 extension preserves the exact archived 675-DOF SAFE basis, four thickness layers and the 3 micrometre end planes. The preflight allocated 4 GiB for three minutes on `short` and wrote `uniform_h60_AP_pnc.json` under `runs/uniform-h60-preflight-JOB_ID/`. The direct-only solve used the measured `case_max_dofs` and a 32 GiB, ten-minute allocation on `nodes`. Its original `--h60` phase inputs are retained with the uniform wrapper and receipts.

The one-thread solve stages exact K/M/A/RHS with hashes before applying the existing direct-factor memory admission, then saves each accepted vector and the public output. Original relative and backward residual limits remain 1e-9; the target remains 1e-10. The 570-second worker limit is inside the ten-minute allocation. A failure retains the staged equations and diagnostics; it does not request a larger allocation or retry automatically. Retrieve and verify the outputs for independent equation and nodal-force replay, then compare against the saved h80 uniform result and the finite 20-by-6 SAFE reference. This finite uniform control does not establish full-device convergence or supply a phase correction for the patterned device.

### h60 full-device control

The historical drivers used `.venv-lawrence-export` with one numerical thread. Reuse completed phases and accepted drives. The original wrappers are `scripts/hpc/lawrence_device_refinement_AP_pnc.sbatch` and `scripts/hpc/lawrence_device_compute_AP_pnc.sbatch`. [Current handoff](AI_HANDOFF_AP_pnc.md)

1. The mesh-only preflight uses `tools/run_lawrence_device_refinement_AP_pnc.py` with the original h80 configuration, tagged mesh checkpoint, and saved port modes. The `mesh_ready` receipt `device_h60_preflight_AP_pnc.json` has SHA-256 `ceb60bb34585bf583776ffa010dd2e040da542f036d392b3c79229af8438a8a0`. It verifies the original 675-DOF traces and end planes, four thickness layers, and all 26 hole walls with 32 segments around each rim. The lead interiors are included in the global refinement. The measured 680,265-DOF case cap applies to this assembly. The standard reporting cap remains 60,000 DOFs.
2. For `stage`, pass the preflight directory and the exact preflight receipt hash as `INPUT_DIR` and `INPUT_SHA`. The compute batch calls `tools/solve_lawrence_device_refinement_AP_pnc.py`. The stage uses 16 GiB, two CPUs, and a 550-second worker budget within a ten-minute Slurm allocation. It saves stiffness, mass, and the exact original complex equations in the stage directory with `stage_AP_pnc.json` before factorization.
3. For `solve`, pass the stage directory and exact `stage_AP_pnc.json` hash. Choose the factor allocation from measured sparse operator storage and fresh resource evidence. The solver checks node and Slurm memory headroom before factorization and at factor and drive boundaries. The Slurm memory limit and the worker timeout bound the process. One complex128 factor serves all pending drives. To recover after interruption, append the stopped prior solve directory and exact `solve_AP_pnc.json` hash. The driver replays accepted drives against the original equations and solves the pending drives only.
4. For `export`, pass the same stage directory/hash and append the completed solve directory and exact `solve_AP_pnc.json` hash. The export reconstructs the original equations, rechecks all five accepted vectors, and calculates channel power and independent traction flux at the original end planes. It identifies the public output in `export_AP_pnc.json`. Verify the returned files before comparing saved h80 and h60 results with `tools/compare_scattering_AP_pnc.py`.

Athul Prem's two-port coherent-drive test verifies power consistency for a small uniform guide. The h60 comparison measures in-plane sensitivity at 1 GHz with fixed ports and hole faceting. Separate thickness, faceting, port, and frequency checks are required for broader full-device claims.

### h80 thickness control

The h80 control keeps the physical 200 nm thickness, 26 hole polygons, width subdivisions, end planes, and exact in-plane footprint. Six thickness layers change the body discretization and its matching port traces together. The original five phases were `preflight`, `safe`, `stage`, `solve` and `export`. Reuse their verified artifacts. [Current handoff](AI_HANDOFF_AP_pnc.md)

The mesh preflight checks exact x-y equality against the archived h80 mesh and records the measured displacement count for the case cap in `thickness_h80_preflight_AP_pnc.json`. The `safe` phase uses that same mesh's P2 trace layout to calculate the complete 12-by-6 semi-analytical finite element section at 1 GHz. It saves the local trace order, section operators and modes in `ports_AP_pnc.npz`, with their identity in `thickness_h80_safe_AP_pnc.json`.

Pass the exact SHA-256 of each prior receipt. Assembly reuses the accepted mesh and SAFE section to save the original stiffness, mass, harmonic operator and drives under `stage_AP_pnc.json`. Choose solve memory and time from the measured operator and live allocation checks. A solve can take a stopped prior `solve_AP_pnc.json` receipt to replay accepted drives and compute the remaining channels. Export uses the completed solve receipt to reconstruct the exact original equations, replay the vectors and check independent flux. The comparison uses both saved runs at the original end planes and measures the combined change in thickness discretization and its matching SAFE section at 1 GHz.

The completed control used jobs 1335307/1335308/1335309/1335321/1335341. Reuse their saved artifacts under `results/runs/full_device_resource_profile/lawrence_thickness_AP_pnc/`; no job remains active. The five-drive solve took 2005.88 s with 33.712 GiB peak RSS and 34.776 GiB cgroup peak inside 128 GiB. Exact export on stage host node038 took 25.56 s within 16 GiB. The z4/z6 comparison passes power (0.0000647273) but fails complex response (0.0167063); independent residual and flux checks do not establish spatial convergence. [Verified outcomes and identities](../verification/device_thickness_AP_pnc.json)

### Bounded port frequency inventory

`tools/run_lawrence_port_inventory_AP_pnc.py` reuses the exact saved h80 z6 SAFE section and archived 1 GHz modes. The fixed sample list is 0.1, 1, 2.8, 5.2 and 6 GHz. It performed four section eigensolves, with no volume mesh, body assembly or factorization. Reuse the completed inventory `runs/port-inventory-1335406/`.

`SAFE_DIR` is the immutable `runs/thickness-h80-safe-1335308/` archive. The runner pins its receipt and NPZ hashes, numerical source, Python and packages. The batch uses `.venv-lawrence-export`, 4 GiB, two allocated CPUs, one numerical thread and five minutes, with a 270-second worker limit. Admission requires current node/job headroom and output storage; there is no automatic escalation or retry of failed/interrupted points. Resume checks input identities, self-digested scientific receipts and raw arrays, and only attempts untouched pending points.

Each sample retains its original 1950 finite roots, signed energy directions, reconstructed traction, original polynomial backward residual and full directional flux-Gram checks. Limits remain 1e-8 for eigenpair backward/relative traction errors and 1e-7 W for flux errors. JSON/CSV/HTML retain explicit incomplete states and root IDs local to each frequency. The 1 GHz modes reference their original NPZ rather than a copied array. Initial/reuse allocation and time evidence is separated under `runs/port-inventory-job-JOB_ID/`.

Completed job 1335406 took 69.40 s with 486172 KiB peak process RSS. Independent Windows replay passed all five samples; each direction has 4/5/8/12/13 open channels. Reuse job 1335407 took 5.29 s and changed no inventory bytes or modification times. Both jobs exited 0:0. Transfer all retained files with their manifests and verify hashes before independent replay. Sparse fixed-section samples do not establish continuous branches, cutoff locations between points, full-device convergence or a gap. [Inventory evidence](../verification/port_inventory_AP_pnc.json)

### Bounded 0.1 GHz device point

Stage `1335473`, solve `1335475` and exact export `1335483` are accepted and retained. Reuse their saved evidence. [Current handoff](AI_HANDOFF_AP_pnc.md) · [Portable evidence](../verification/low_band_device_AP_pnc.json)

The selected low-band comparison uses the archived h80 six-layer device, physical 200 nm thickness and original 0/16.65 micrometre end planes. Its observable is the per-incident transmission, reflection and outgoing-channel conversion at 0.1 GHz, presented separately from the accepted five-channel 1 GHz anchor. `tools/run_lawrence_low_band_AP_pnc.py` reuses the exact mesh, K/M, trace maps and accepted four-channel 0.1 GHz SAFE modes. It prepares only new frequency-specific equations. Do not run mesh generation, body assembly or a SAFE eigensolve for this point.

The original phases are recorded with `scripts/hpc/lawrence_low_band_AP_pnc.sbatch` and the retained receipts. The stage/export ceiling was 16 GiB, two allocated CPUs, one numerical thread and ten minutes. The single solve ceiling was 128 GiB and sixty minutes, with a 3450-second worker budget. The stage used body stage `1335309`, SAFE section `1335308` and port inventory `1335406`.

Inspect `runs/h80-z6-low-band-stage-JOB_ID/stage_AP_pnc.json` before deciding on a solve. It must retain the original body/section identities, four signed incoming roots, the exact A/RHS hashes, the prepared CSC graph comparison, source/build identities and explicit factor-index exposure. SciPy 1.18.1 [casts SuperLU indices to C int](https://github.com/scipy/scipy/blob/v1.18.1/scipy/sparse/linalg/_dsolve/linsolve.py), and its [factor nonzero count uses a C-int sum](https://github.com/scipy/scipy/blob/v1.18.1/scipy/sparse/linalg/_dsolve/_superluobject.c). The historical 1 GHz factor count of 1,846,932,604 is a reference measurement, not a bound for changed-frequency pivoting or fill. Matching the input graph or using int64 input does not establish factor capacity. Preserve this uncertainty when admitting the one bounded trial.

The solve checked node and cgroup memory before factorization, retained the 1e-10 target and both original relative/backward 1e-9 residual gates, and saved each accepted drive separately. Export required all four accepted vectors and exact reconstructed equation hashes. It used the recorded stage node to preserve the numerical backend when reconstructing the equations.

Verify transfer hashes and independently replay the original equations and physical nodal-force flux. Keep the returned phase directories together under the local low-band archive so the replay can resolve their retained links. The independent tool loads A, K and M sequentially, validates all 32 outgoing complex coefficients and the full P2 field mapping. It also checks the complete signed propagating flux Gram, retains the original-basis left-incident defect R†R + T†T − I and its spectral norm, and independently checks one unequal complex superposition of displacement and recovered nodal-force columns. The 1e-7 W basis-entry gate is separate from the unchanged 0.01 physical-power/coherent gate. Results cover only the saved left-incident block, with measured basis accuracy stated; they do not establish full two-sided unitarity or reciprocity. The tool prints its evidence as JSON:

```powershell
.venv/Scripts/python.exe -B tools/replay_lawrence_device_point_AP_pnc.py --stage RETURNED_STAGE/stage_AP_pnc.json --solve RETURNED_SOLVE/solve_AP_pnc.json --export RETURNED_EXPORT/export_AP_pnc.json
```

Preserve every existing 0.01 power gate and link representative 3D fields to the accepted vectors. Neither algebraic success nor power balance resolves the existing failed spatial checks. The four-channel and five-channel points retain separate signed inventories; channel-by-channel frequency differences require independent signed correspondence. Select any next sample or refinement from the actual response and field evidence. No dense map, finer-mesh ladder, gap, resonance or Q claim follows from this single exploratory point.

### Matched 0.1 GHz four/six-layer control

`tools/run_lawrence_low_band_z4_AP_pnc.py` and `scripts/hpc/lawrence_low_band_z4_AP_pnc.sbatch` reuse the original Windows h80 four-layer mesh/K/M and the exact frequency-independent 12-by-4 section in `results/runs/full_device_port_resolution/12x4/attempt_lpnbd9kr/`. They preserve the physical geometry, nominal Ge properties and original end planes. No body mesh, body matrix or section matrix is regenerated; only the new 0.1 GHz section modes and frequency-dependent A/RHS are computed. The accepted six-layer point is reused.

The four archived phases are `safe BODY_STAGE_DIR MESH_DIR SECTION_DIR`, `stage SAFE_DIR SAFE_SHA ORIGINAL_1GHZ_A`, `solve STAGE_DIR STAGE_SHA`, and `export STAGE_DIR STAGE_SHA SOLVE_DIR SOLVE_SHA`. The original wrapper prefixed `PROJECT_ROOT PHASE WORKER_WALL_SECONDS` to the phase inputs. Exact receipt hashes bind the phases. The solver replayed accepted channels before reuse when given `PRIOR_SOLVE_DIR PRIOR_SOLVE_SHA`.

The bounded resource ceilings were 4 GiB/two CPUs/three minutes (150-second worker) for SAFE, 16 GiB/two CPUs/ten minutes (550 seconds) for preparation/export, and one 64-GiB/two-CPU/25-minute direct solve (1400 seconds). Each used one numerical thread. The archived solve checked the prepared CSC graph, initial factor storage/index exposure and actual Slurm/cgroup/node headroom before factorization. Exact cached export used the original stage node.

Retrieve each immutable phase and verify every transferred file. Independently replay the saved new modes, four original equations, nodal-force flux, coherent left-incident block and full P2 fields using:

```powershell
.venv/Scripts/python.exe -B tools/replay_lawrence_z4_control_AP_pnc.py --safe RETURNED_SAFE/safe_AP_pnc.json --stage RETURNED_STAGE/stage_AP_pnc.json --solve RETURNED_SOLVE/solve_AP_pnc.json --export RETURNED_EXPORT/export_AP_pnc.json --compare-z6
```

The comparison binds the separately accepted z6 replay, establishes signed incoming/reflected/transmitted mass-overlap and gauge correspondence, and retains original-plane complex amplitudes. Power and complex changes each use 0.01. A passing pair is a finite joint body/port thickness-discretization result, not full spatial or frequency convergence. The old failed 1 GHz comparisons remain independent evidence. See the current handoff and `verification/low_band_thickness_AP_pnc.json` for actual state and retained jobs; never rerun a phase merely to follow this recipe.

### Archived h100/h80 control at 0.1 GHz

`tools/run_lawrence_low_band_h100_AP_pnc.py` uses the original h100 four-layer mesh, K/M and global trace maps together with the accepted h80 z4 0.1 GHz modes from job 1335500. The original 1 GHz operator is a hash-verified graph reference only. There is no SAFE phase, eigensolve, mesh generation or matrix assembly. Exact historical source exceptions are recorded separately from current Linux preparation/solve/export; the original batched assembly control remains in the exported configuration.

The three archived phases were `stage BODY_STAGE_DIR MESH_DIR ACCEPTED_SAFE_DIR`, `solve STAGE_DIR STAGE_SHA`, and `export STAGE_DIR STAGE_SHA SOLVE_DIR SOLVE_SHA`. The original wrapper `scripts/hpc/lawrence_low_band_h100_AP_pnc.sbatch` prefixed `PROJECT_ROOT PHASE WORKER_WALL_SECONDS`. Preparation/export used 16 GiB and ten minutes (550-second worker). The bounded direct allocation used 64 GiB and 25 minutes (1400-second worker). Each used two allocated CPUs and one numerical thread. Admission checked the actual prepared operator, index exposure, storage and live memory. Exact cached export used the preparation node. Preserve each phase and failure.

After all transfer hashes agree, replay the new point once and compare against the exact independently accepted h80 result:

```powershell
.venv/Scripts/python.exe -B tools/replay_lawrence_h100_control_AP_pnc.py --stage RETURNED_STAGE/stage_AP_pnc.json --solve RETURNED_SOLVE/solve_AP_pnc.json --export RETURNED_EXPORT/export_AP_pnc.json --compare-h80
```

The replay checks original equations, physical nodal-force flux, coherent left-incident power and full P2 fields. The pair retains signed mode/gauge matching and original reference planes. Absolute power and complex increments use 0.01; relative reflection divides by max(abs(R_h100),1e-12) only for reporting. The global in-plane change affects lead interiors and the patterned region together. A finite pair pass is neither continuum convergence nor a weak-reflection error bound. Read the current handoff and `verification/low_band_inplane_AP_pnc.json` before acting; completed phases must be reused.

### Archived 0.8875 GHz device point

Accepted outcome: preparation 1335628, solve 1335630 and export 1335632 completed 0:0. The five-drive solve took 2009.77 seconds GNU elapsed with 35350312 KiB peak RSS and 37341077504 bytes job-cgroup peak; actual factor nonzeros 1846932631 pass the retained int32 count check. Cached export took 26.14 seconds with 3960536 KiB peak RSS. Independent Windows replay and signed/gauge-qualified comparison pass; reuse verification/device_frequency_8875_AP_pnc.json and results/runs/device_frequency_8875_AP_pnc/. These example commands document the accepted path, not permission to repeat it. No next job is queued.

The selected point tests the frequency dependence of the saved 1 GHz conversion pair on the exact h80 six-layer device. The independently verified signed section relation is already saved under `results/runs/device_frequency_8875_AP_pnc/relation/`; reuse it. All ten signed channels match with minimum mass overlap 0.999560627280313 under the unchanged 0.99 gate. Endpoint correspondence does not prove continuous branches. A frequency-response difference is not a mesh-convergence test, and all failed 1 GHz spatial comparisons remain failed.

`tools/run_lawrence_frequency_point_AP_pnc.py` pins the accepted 887500000 Hz full SAFE modes and interval receipts, exact archived mesh/K/M and physical end planes. It retains original Linux section, Windows frequency-mode and new Linux preparation/solve/export identities separately. There is no remeshing, body assembly or eigensolve. The prepared matrix must have the original CSC graph and reviewed int32 exposure. Initial factor allocation and historical aggregate factor counts are separate checks; future numeric-pivot fill is not certified before factorization.

The historical wrapper `scripts/hpc/lawrence_frequency_point_AP_pnc.sbatch` used the following phase arguments after `PROJECT_ROOT PHASE WORKER_WALL_SECONDS`.

- `stage BODY_STAGE_DIR SAFE_DIR INTERVAL_DIR INTERVAL_LEDGER`
- `solve STAGE_DIR STAGE_SHA [PRIOR_SOLVE_DIR PRIOR_SOLVE_SHA]`
- `export STAGE_DIR STAGE_SHA SOLVE_DIR SOLVE_SHA`

Preparation/export were bounded to 16 GiB, two CPUs and ten minutes with a 550-second worker limit. The direct solve used 128 GiB, two CPUs and one hour with a 3500-second worker after measured preparation, index/storage checks and fresh resource admission. All phases used one numerical thread. Keep target 1e-10, both original residual gates 1e-9 and all physical/coherent checks unchanged. Exact cached export used the preparation node.

Upload only the new runner/batch, `verification/port_count_interval_AP_pnc.json`, and the accepted study's `inputs_AP_pnc.json`, `point-887500000_AP_pnc.json` and `modes-887500000_AP_pnc.npz`. Retain original body/section inputs in place and verify uploaded bytes/hashes. The fresh interval input directory may be `inputs/device_frequency_8875_AP_pnc/`. Reuse every completed phase; never rerun the accepted section relation or preparation merely because a chat or transfer connection changed.

After returning complete receipts, original A/RHS, accepted vectors and exported files, independent Windows replay uses:

```powershell
.venv/Scripts/python.exe -B tools/replay_lawrence_frequency_point_AP_pnc.py --stage RETURNED_STAGE/stage_AP_pnc.json --solve RETURNED_SOLVE/solve_AP_pnc.json --export RETURNED_EXPORT/export_AP_pnc.json
```

Save its JSON stdout as a new receipt, retain its SHA-256, then use `tools/compare_device_frequency_AP_pnc.py compare --relation ACCEPTED_RELATION --candidate RETURNED_PUBLIC_DIRECTORY --verification INDEPENDENT_REPLAY --verification-sha256 REPLAY_SHA256 --output NEW_COMPARISON_DIRECTORY`. This validates the accepted relation bytes, reviewed replay source and receipt, shared physical inputs and all 50 coefficients before publishing raw/gauge-aligned response and directed conversion changes. The relative-power denominator floor 1e-12 is a reporting guard, not an uncertainty bound. Root supervises local replay with fresh available-memory admission and preserved failure output.

On the current Windows host, Git OpenSSH's normal multiplex session request failed even while the master reported running. Its documented `-O proxy` mode successfully reused the authenticated master for actual commands and SFTP without another authentication. The live control location and ownership belong in the handoff; test an actual read-only remote command before assuming a connection works. SFTP can use a temporary SSH wrapper that passes `-O proxy` and the owned control path. Never write passwords or Duo responses into project files. [OpenSSH multiplex control](https://man.openbsd.org/ssh.1)

### Archived 0.8875 GHz thickness control

Completed outcome: jobs 1335641/1335643/1335645/1335647 exited 0:0. The sole five-drive solve took 522.76 seconds with 15984996 KiB GNU peak RSS and 17133563904 bytes job-cgroup peak inside 64 GiB. Independent point replay passes; the declared weak-pair 10% increment and whole complex gate fail while both resolutions retain over 99% suppression versus 1 GHz. Reuse verification/device_frequency_thickness_8875_AP_pnc.json; these commands document the completed workflow, not a queued rerun.

`tools/run_lawrence_frequency_z4_AP_pnc.py` adds one four-layer point to the accepted six-layer frequency response. It reuses the exact 426627-DOF h80 body mesh/K/M, physical 200 nm thickness, original end planes and 675-DOF section operators. One new full SAFE eigensystem is required at 887500000 Hz; accepted z6 and 1 GHz computations are not repeated. Read the handoff for current ownership and completed phases before using this recipe.

The wrapper `scripts/hpc/lawrence_frequency_z4_AP_pnc.sbatch` used the following phase arguments after `PROJECT_ROOT PHASE WORKER_WALL_SECONDS`.

- `safe BODY_STAGE_DIR MESH_DIR SECTION_DIR`
- `stage SAFE_DIR SAFE_SHA ORIGINAL_1GHZ_A SAFE_VERIFICATION SAFE_VERIFICATION_SHA RELATION RELATION_SHA`
- `solve STAGE_DIR STAGE_SHA [PRIOR_SOLVE_DIR PRIOR_SOLVE_SHA]`
- `export STAGE_DIR STAGE_SHA SOLVE_DIR SOLVE_SHA`

The original graph-reference operator is retained under `inputs/full_h80_operator_1r5fdlft/A_AP_pnc.npz`; the body K/M and ports remain under the archived results stage. Do not assume those paths are interchangeable. SAFE uses 4 GiB / 180 seconds with a 150-second worker; preparation/export use 16 GiB / 600 seconds with a 550-second worker; the sole admitted solve uses 64 GiB / 1500 seconds with a 1400-second worker. Every phase allocates two CPUs and uses one numerical thread. The prepared CSC graph and int32 exposure, fresh node/job headroom and disk reserve must pass before factorization. Numeric pivot fill is checked after factorization; historical fill is not a guarantee. No automatic retry or allocation increase is authorized by this recipe.

First run `tools/compare_frequency_thickness_AP_pnc.py relation --safe SAFE_RECEIPT --safe-sha256 SAFE_SHA --runner-sha256 REVIEWED_RUNNER_SHA --output NEW_RELATION_DIRECTORY`. Then capture JSON stdout from `tools/replay_lawrence_frequency_z4_AP_pnc.py relation --safe SAFE_RECEIPT --safe-sha256 SAFE_SHA --relation RELATION_RECEIPT --relation-sha256 RELATION_SHA --runner-sha256 REVIEWED_RUNNER_SHA --comparison-sha256 REVIEWED_COMPARATOR_SHA`. This single independent pass checks all 1350 roots, both new signed relations and their root-permutation square through the accepted anchors. It does not require phase-gauge closure around the square. Stage admission binds both exact receipts.

After exact export, the same replay helper's `point` command requires `--safe`, `--stage`, `--solve`, `--export`, `--export-sha256`, `--runner-sha256` and an explicit local `--prebody-replay` path. Capture its JSON stdout once under the declared local memory/wall guards. The actual export stores phase_adjustment_applied: false inside its hash-bound public provenance; the verifier requires that explicit value plus the stage flag and exact execution links. Accepted pre-body and relation receipts retain their original producers. Exact source snapshots are preserved in local_checks/, and only the reviewed receipt/producer/snapshot combinations cross the narrow consumer-schema correction. The comparator's `compare` command binds `--relation`, `--prebody-replay` and `--point-replay` with each corresponding `-sha256`, plus `--replay-source-sha256`, `--candidate PUBLIC_RUN_DIRECTORY` and `--output NEW_COMPARISON_DIRECTORY`.

For each of the two distinct unit-power drives, target A is `abs(P_z4_8875-P_z6_8875)/P_z6_8875 <= 0.10`; target B is `P_z_8875/P_z_1GHz <= 0.01` separately at both resolutions. Use exact positive accepted per-direction denominators without floors. A is finite sensitivity, not 10% accuracy; B tests at least 99% suppression. Preserve both outcomes independently, all raw/aligned complex and phase changes, and the separate whole-response 0.01 gates. No passing finite increment establishes continuum convergence or a numerical floor.

### Archived 0.8875 GHz in-plane control

Completed outcome: stage 1335652, solve 1335654 and export 1335656 exited 0:0; the initial input-check failure 1335650 is retained. The sole solve took 322.31 seconds with 10710668 KiB GNU peak RSS. The independent replay passes; the pair-specific 10% sensitivity and whole complex gates fail, while both resolutions retain over 99% suppression versus 1 GHz. Reuse verification/device_frequency_inplane_8875_AP_pnc.json. The following recipe documents completed work, not a queued rerun.

The selected control adds only the archived h100 z4 point at 887500000 Hz and compares it with accepted h80 z4. Read the current handoff before executing: root owns all connections, transfers and jobs. The exact 307611-DOF mesh, K/M, 675-DOF section and accepted frequency-specific SAFE are reused; no mesh, matrix assembly or eigensolve is needed. The body traces differ, so shared section fields must be checked against each body's original coordinates and component maps.

The wrapper `scripts/hpc/lawrence_frequency_h100_AP_pnc.sbatch` used the following phase arguments after `PROJECT_ROOT PHASE WORKER_WALL_SECONDS`.

- `stage BODY_STAGE_DIR MESH_DIR SAFE_DIR ORIGINAL_1GHZ_A SAFE_VERIFICATION REFERENCE_POINT`
- `solve STAGE_DIR STAGE_SHA [PRIOR_SOLVE_DIR PRIOR_SOLVE_SHA]`
- `export STAGE_DIR STAGE_SHA SOLVE_DIR SOLVE_SHA`

The original h100 1 GHz A resides in the same `full_h100_operator_qr_jt4fp` directory as K/M and ports. Preparation/export are bounded to 16 GiB / 600 seconds (550-second worker); the sole solve is bounded to 48 GiB / 900 seconds (850-second worker), using --partition=himem when selecting himem001. All allocate two CPUs and use one numerical thread. The adapter binds the frozen h100 execution helper and accepted full-SAFE/reference-point receipts, while keeping original Windows assembly and new Linux execution provenance separate. The prepared CSC graph, int32 factor exposure, fresh node/job memory and disk admission must pass before factorization. Historical fill and runtime are not guarantees. No automatic retry, larger allocation or next point is queued.

After transfer hash verification, `tools/replay_lawrence_frequency_h100_AP_pnc.py` independently checks all five original equations, nodal-force and coherent power, 50 propagating / 6750 directional coefficients and full P2 fields. Its explicit local paths bind accepted SAFE/prebody/reference/relation receipts, retained producer sources, the reviewed new runner SHA and new phase/public artifacts. Accepted SAFE and signed matching are not numerically repeated. The original target is 1e-10 and both residual limits remain 1e-9.

For each separate unit-power directed pair, A is `abs(P100_8875-P80_8875)/P80_8875 <=0.10`, using exact positive h80 denominators without floors. B is `P_h_8875/P_h_1GHz <=0.01`, independently at both resolutions and directions. Keep generic whole-response power/complex 0.01 gates separate. These finite sensitivity and suppression observations do not establish accuracy, convergence or a numerical floor. This global increment changes lead interiors together with the patterned region; do not subtract uniform-guide phase or combine this increment with thickness sensitivity as an error estimate.
