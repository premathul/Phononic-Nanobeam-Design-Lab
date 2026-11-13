# Phononic Nanobeam Design Lab

**Research owner: Athul Prem | `_AP_pnc`**

A configurable three-dimensional anisotropic elastic-wave simulator for high-purity germanium nanobeams. It provides 3D straight-beam eigenmodes, periodic guided-wave dispersion, power-normalized matched-port scattering, optional circular through-hole geometry and offline interactive reports. The solver runs on a CPU using Python, scikit-fem, SciPy and Gmsh. All project documentation, software text and generated reports use English.

## Quick start

Use Python 3.12 in a dedicated virtual environment. Run these commands from the project directory.

```powershell
python -m venv .venv
.\.venv\Scripts\python.exe -m pip install -r requirements_3d_AP_pnc.txt
.\.venv\Scripts\python.exe simulate_AP_pnc.py --open
```

Once the environment is set up, you can double-click `Launch_AP_pnc.cmd`. The default case is a homogeneous 3D cantilever benchmark measuring 2000 nm long, 1000 nm wide and 200 nm thick. Its left end is clamped and the other surfaces are traction free.

Edit [straight_beam_AP_pnc.json](configs/straight_beam_AP_pnc.json) to configure a run, or use the following command-line options.

```powershell
# Change the physical thickness while keeping the in-plane geometry fixed.
.\.venv\Scripts\python.exe simulate_AP_pnc.py --thickness-nm 100 --open

# Model an infinite uniform guide with one Bloch-periodic cell and free sidewalls.
.\.venv\Scripts\python.exe simulate_AP_pnc.py --analysis bloch --cells 4 3 2 --k-points 9 --open

# Include the six rigid-body zero modes when requesting modes of a free beam.
.\.venv\Scripts\python.exe simulate_AP_pnc.py --boundary free --modes 10

# Drive each propagating channel of a uniform guide at 0.3 GHz.
.\.venv\Scripts\python.exe simulate_AP_pnc.py --config configs/straight_scattering_AP_pnc.json --open

# Select the opposite incident end and a different frequency.
.\.venv\Scripts\python.exe simulate_AP_pnc.py --config configs/straight_scattering_AP_pnc.json --incident-side right --frequency-ghz 0.5
```

Each case writes an interactive `report_AP_pnc.html`, a mode CSV, the resolved configuration, a summary with source and environment details, an NPZ with the full quadratic-element degree-of-freedom mapping, and a VTU for ParaView under `results/runs/`. The VTU samples normalized mode shapes at vertices. Physical displacement amplitudes would require a specified excitation. Bloch rows are sorted by frequency independently at each wavenumber. Branch tracking across wavenumbers remains pending.

Scattering runs instead write `scattering_AP_pnc.csv` with per-input-channel power ratios, `scattering_channels_AP_pnc.csv` with complex amplitudes and power in each propagating output channel, and `port_modes_AP_pnc.csv` with the complete discrete port root inventory. Their NPZ retains full driven responses, propagating and evanescent amplitudes, port fields and mappings. Displacements are responses per square root of incident watt; 1 W is a mathematical normalization. The report supports channel selection and phase adjustment. A successful matched-guide power check does not by itself establish phase accuracy or mesh convergence.

The runner checks completed outputs before reusing results with the same configuration, source and numerical environment. It preserves failed or interrupted attempts and retries in new directories. Large frequency sweeps still require per-frequency checkpoint and restart support.

## Verification and capabilities

The current Windows/Python 3.12 suite ran 41 tests: 40 passed and one was skipped because the process lacked permission to create a symbolic link. The checks cover tensor rotation, uniform strain, consistent mass, rigid-body subspaces, independent dense and sparse solutions, Bloch phase and the long-wavelength rod limit. Port checks cover trace mapping, directional energy flux, the dispersion-derivative identity, exact symmetry degeneracy, an independent 3D Bloch comparison, and explicit rejection of unresolved cutoffs. Scattering checks exercise both incidence directions, independent boundary flux, and reconstruction from exported fields.

The [three-mesh convergence record](verification/straight_beam_validation_AP_pnc.json) covers the first four cantilever modes of the default geometry with 1755, 5187 and 11475 degrees of freedom. The maximum frequency change between the two finest meshes was about 0.468%, and the minimum mass-weighted mode overlap was about 0.999992. Independent fixed-property assemblies and solves labeled 20 mK and 4 K yielded identical matrices and frequencies. Convergence for other geometries, thicknesses and frequency bands requires its own checks.

The [straight-guide scattering acceptance](verification/straight_scattering_acceptance_AP_pnc.json) covers nominal Ge at 1 GHz, a 600 × 200 nm cross-section, lengths of 1 and 1.5 µm, and four incident channels from each end. Complex transmission errors against a common finer SAFE reference were at most 0.00454 and 0.00683, below the 0.01 criterion. The final section refinement changed accumulated phase by at most 0.00534 rad over 1.5 µm. The 14553-DOF long-guide case had maximum reflection about 1.20 × 10⁻⁷ and direct power imbalance below 5.2 × 10⁻¹⁵. These checks establish this numerical benchmark, with a finite-element reference rather than an exact continuum error bound. Earlier coarse-mesh failures and refinement evidence remain saved and linked by checksums. The 0.3 GHz quick-start case has its own power check and does not inherit this mesh acceptance.

| Capability | Status |
| --- | --- |
| Three-dimensional homogeneous beam with P1/P2 vector tetrahedra | Available |
| Nominal cubic elasticity with crystal-to-lab rotation | Available with fixed properties. Low-temperature specimen calibration remains pending. |
| Finite-beam eigenmodes with free, one-end-clamped or two-end-clamped boundaries | Available |
| Bloch dispersion for guides with free sidewalls | Available for a finite branch sample. Channel completeness remains unverified. |
| 26-hole inputs and clearance checks | Verified with the original `position_nm` and example `x_nm` column formats |
| Gmsh three-dimensional through-holes | Verified on a two-hole case. The full 26-hole solve remains pending. |
| Port power transmission, reflection and mode conversion | Available for the uniform guide. Validation scope is recorded separately from each solve. |
| Temperature-dependent properties, physical loss, bent beams and rings | Planned |
| GPU solver | CPU measurements will inform whether GPU development is warranted. |

The unstressed linear-elastic model requires additional physics and validation to address intrinsic loss, radiative quality factor, experimental performance or single-phonon quantum states. A successful solve establishes execution. Mesh convergence and experimental validation require separate evidence. Offline report script interactions were checked. Browser file-access rules prevented screenshot verification.

## Run verification

```powershell
$env:OPENBLAS_NUM_THREADS='1'
$env:OMP_NUM_THREADS='1'
.\.venv\Scripts\python.exe -B -m unittest discover -s tests -p 'test*_AP_pnc.py' -v
.\.venv\Scripts\python.exe -B tools/validate_3d_AP_pnc.py
.\.venv\Scripts\python.exe -B tools/validate_scattering_AP_pnc.py --acceptance --output-dir results/runs/scattering-verification
```

The validation tool uses three independent mesh solves, mass-weighted mode matching and a fixed-property temperature control. It reports failed checks with a nonzero exit code and retains the computed values and original material inputs. The `--quick` option exercises the workflow on coarser meshes, which may fail the 0.5% frequency-change threshold.

The scattering acceptance command checks the saved reference/source hashes, reuses the unchanged short-guide evidence, and computes a new long-guide solution and finer SAFE reference. The largest local case took 19.64 s and reached approximately 605 MiB sampled process memory. Use a fresh output directory to retain results from each independent run. The baseline and `--refine` modes reproduce the earlier coarse stages; their expected phase failures are retained as evidence.

## Development plan and references

- [Implementation plan and reuse decisions](docs/IMPLEMENTATION_PLAN_AP_pnc.md)
- [Current development handoff](docs/AI_HANDOFF_AP_pnc.md)
- [Physics specification](docs/PHYSICS_SPECIFICATION_AP_pnc.md)
- [Long-term research work packages](docs/NEXT_STEPS_AP_pnc.md)
- [Material references and validity limits](docs/REFERENCES_AP_pnc.md)

The next milestone brings the original 26-hole geometry into the validated port workflow, with independent geometry and scattering convergence. Temperature-dependent properties, bends and larger sweeps follow only when the corresponding inputs and checks are established. QTCAD served as a workflow reference. The phonon solver uses the libraries listed above.

## Historical two-dimensional evidence

The historical files are retained in `archive/`, `results/archive_*`, `figures/archive_*` and `provenance/`. The Git download is about 32 MB, and the active solver runs independently of the archive.

All 199 files from the original upload are preserved byte for byte. The older inventory has 223 unresolved entries, including 222 missing entries and one dependency file whose content differs from the expected bytes. The missing files prevent exact reproduction of the latest 2D workflow. The new 3D solver has independent provenance. The historical gap near 3 GHz applies to a 2D in-plane model, and approximately −42.25 dB refers to a probe displacement ratio. Both require independent 3D power calculations before drawing conclusions about transmission. [Evidence and missing files](docs/EVIDENCE_AND_RECOVERY_AP_pnc.md)

Use these legacy entry points to inspect the saved evidence.

```powershell
.\.venv\Scripts\python.exe tools/audit_repository_AP_pnc.py
.\.venv\Scripts\python.exe verify_AP_pnc.py --fields --safety
.\.venv\Scripts\python.exe run_AP_pnc.py --name hpge_reference --dry-run
```

New figures and project files retain the attribution `Athul Prem | _AP_pnc`. Final research-report requirements are in `reports/`. External delivery requires current user authorization.
