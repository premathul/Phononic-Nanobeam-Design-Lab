# Phononic Nanobeam Design Lab

**Research owner: Athul Prem | `_AP_pnc`**

A configurable 3D anisotropic elastic-wave simulator for germanium nanobeams. It supports finite-body modes, periodic guided waves, power-normalized matched-port scattering, CSV-defined through-holes, saved results and an optional English desktop workbench. It uses Python, scikit-fem, SciPy, Gmsh and meshio on CPUs.

## Start the simulator

Use Python 3.12 in a dedicated environment:

```powershell
python -m venv .venv
.\.venv\Scripts\python.exe -m pip install -r requirements_3d_AP_pnc.txt
.\.venv\Scripts\python.exe simulate_AP_pnc.py --open
```

The default is a small homogeneous cantilever. `Launch_AP_pnc.cmd` runs the same entry point. To use the desktop interface:

```powershell
.\.venv\Scripts\python.exe -m pip install -r requirements_ui_AP_pnc.txt
.\.venv\Scripts\python.exe workbench_AP_pnc.py
```

You can also double-click `Launch_Workbench_AP_pnc.cmd`. Choose a preset, edit its parameters, check the mesh, then run. **Open saved run** reopens a completed result; **Compare saved run** compares compatible same-frequency runs without solving. A mesh-ready status is not a solved or converged result.

## Inspect the saved full-device dataset

The current delivery joins the accepted h80 six-layer device responses at **0.1, 0.8875 and 1 GHz** through **Open device results** in the workbench. The view selects a frequency and incident channel, shows T/R and native complex conversion coefficients, exports CSVs, and links the original configuration, fields and numerical controls. It uses retained small summaries/CSVs, without starting a solve.

The same read-only entry point is available from the command line:

```powershell
.\.venv\Scripts\python.exe -B tools/browse_device_results_AP_pnc.py --open
```

There are **14 incident-drive rows and 132 complex coefficients** across the three sparse samples. Root numbers are local to each frequency; the four/five-channel transition is explicit. The saved 0.8875/1 GHz endpoint correspondence is linked, not recomputed. Missing frequencies are not interpolated into a spectrum.

The archived arrays are not included in a fresh Git clone. The browser uses the retained workspace archive and reports missing inputs clearly. The portable scientific records below remain available in Git. Existing original fields stay in place; an exported table or generated report is not a self-contained copy of all arrays.

## Configure a study

```powershell
.\.venv\Scripts\python.exe simulate_AP_pnc.py --config configs/straight_scattering_AP_pnc.json --open
.\.venv\Scripts\python.exe simulate_AP_pnc.py --config configs/perforated_layered_reference_AP_pnc.json --mesh-only
```

The refined layered preset is a tested **two-hole** case, not full-device acceptance. `configs/historical_26hole_AP_pnc.json` is a coarse exploratory full-device preset. Inspect actual body/port DOFs and available resources before a large run; a DOF cap does not bound mesh-generation memory.

Configuration records geometry in nm and resolves it to SI; it includes the crystal-to-laboratory rotation, material provenance, physical thickness, hole placement, mesh and port resolution, solver and resource limits. The historical 26-hole table is placed with an 8325 nm offset in a 16650 nm beam, leaving 1500 nm leads. Nominal Ge C11/C12/C44 = 128.5/48.3/66.8 GPa and density 5323 kg/m³ are fixed-property controls, not measured cryogenic sample constants.

Layered meshes independently control in-plane size and thickness layers, while z layers also change the port discretization. Hole faceting is a separate control. Original scattering phases stay at the physical end planes; do not subtract a uniform-guide phase from the patterned response.

For long iterative runs, the opt-in checkpoint workflow preserves accepted channels and verifies original residuals on resume:

```powershell
.\.venv\Scripts\python.exe simulate_AP_pnc.py --config configs/straight_scattering_AP_pnc.json --checkpoint --wall-seconds 1800
.\.venv\Scripts\python.exe simulate_AP_pnc.py --resume PATH_TO_SAVED_ATTEMPT --wall-seconds 1800
```

Resume requires matching saved inputs/source identities. An interrupted run is not an accepted response. The [Lawrence runbook](docs/LAWRENCE_HPC_AP_pnc.md) covers memory-heavy execution and cached export; read the [current handoff](docs/AI_HANDOFF_AP_pnc.md) before any submission.

`tools/map_scattering_AP_pnc.py` provides bounded point-resumable frequency work; its `--max-new-points` option limits attempts. The small guide pilot is complete. The separate `tools/import_lawrence_frequency_map_AP_pnc.py` retains the older read-only 1 GHz anchor view. Neither starts a full-device campaign automatically.

## Compute a bounded frequency map

For a scattering preset, including **Historical 26-hole device · pilot**, use **Check mesh** first. In **Frequency map**, enter explicit increasing GHz samples separated by commas or spaces and choose **New point limit** (default **1**). **Run frequency map** reuses a matching checked mesh, keeps the existing body/port resource caps, and attempts at most that many unfinished points. Failed attempts count toward the limit.

**Stop** during a map requests a stop after the current point finishes. It does not interrupt a running factorization. The workbench stays busy until the worker exits and preserves accepted points. **Resume map…** reads the saved configuration and frequency list; it ignores unsaved study edits and reuses accepted points and shared preparation. A forcibly interrupted unfinished point may need another attempt. Resume requires the original native input/source/environment identities; imported read-only views cannot resume.

**Open saved map…** opens its existing table/report without loading scientific arrays or solving. Viewing an older publication does not require the current solver version. A complete map means all requested finite computations passed their point gates; it does not establish spatial or frequency convergence. Failed and pending points remain explicit.

The UI uses the existing `tools/map_scattering_AP_pnc.py` engine and its outputs. Its small end-to-end workflow is tested; that does not certify arbitrary 26-hole settings or authorize a large frequency campaign. Accepted device results already available through **Open device results** should be reused.

## What the results establish

| Evidence | Supported result | Limit |
| --- | --- | --- |
| [Mechanics](verification/straight_beam_validation_AP_pnc.json), [uniform guide](verification/straight_scattering_acceptance_AP_pnc.json), [two-hole phase check](verification/two_hole_phase_refinement_AP_pnc.json) | Small reference problems and stated finite refinement checks pass | Other geometry/frequency cases need their own evidence |
| [0.1 GHz device](verification/low_band_device_AP_pnc.json), [thickness](verification/low_band_thickness_AP_pnc.json), [in-plane](verification/low_band_inplane_AP_pnc.json) | Four accepted drives; both tested finite complex-response increments pass 0.01 | Full spatial/frequency convergence remains unestablished |
| [0.8875 GHz device](verification/device_frequency_8875_AP_pnc.json) | Five accepted drives; selected conversion is much smaller than at 1 GHz in the finite model | [Thickness](verification/device_frequency_thickness_8875_AP_pnc.json) and [in-plane](verification/device_frequency_inplane_8875_AP_pnc.json) sensitivity fail |
| [1 GHz full device](verification/device_thickness_AP_pnc.json) | Five accepted drives and saved 3D fields | Three full-device complex-response comparisons fail |
| [Regional energy](verification/field_energy_8875_AP_pnc.json), [phase-only bound](verification/lead_phase_8875_AP_pnc.json) | Small regional-energy change; phase-only model excluded at the existing tolerance | Error location and weak-conversion accuracy remain unresolved |

Passing original equations and physical flux qualifies a finite numerical response. It does not establish spatial convergence, a complete band gap, a resonance linewidth/Q or experimental accuracy. Sparse samples establish no continuous branch or converged spectrum. Driven fields use displacement per square root of incident watt; displacement ratios are not transmitted power.

## Tests and development

```powershell
.\.venv\Scripts\python.exe -B -m unittest discover -s tests -p "test*_AP_pnc.py" -v
```

The optional Qt environment enables offscreen desktop tests. Windows symbolic-link privilege may cause one explicit skip. CI also runs the existing mechanics, matched-port, two-hole and saved-evidence benchmarks. See [test scope](tests/README.md); current run counts and integration status belong in the handoff rather than a duplicated historical timeline here.

Use the [implementation plan](docs/IMPLEMENTATION_PLAN_AP_pnc.md) for priorities and stop conditions, the [physics specification](docs/PHYSICS_SPECIFICATION_AP_pnc.md) for equations and limits, and [references](docs/REFERENCES_AP_pnc.md) for source boundaries. Final research reporting follows [report requirements](reports/REPORT_REQUIREMENTS_AP_pnc.md). No email or public release is implied.

The active solver is `simulate_AP_pnc.py` with `src/phononic/`. Root `run_AP_pnc.py` is only a legacy archive-readiness checker. Original historical 2D evidence remains under `archive/`, `provenance/` and the saved result directories; [recovery status](docs/EVIDENCE_AND_RECOVERY_AP_pnc.md) explains its limitations. Archive recovery is not a prerequisite for current 3D development.
