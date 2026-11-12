# Germanium phononic cavity: reproducible FEM handoff

This folder contains the actual simulation scripts, original 26-hole workbook, meshes, numerical data and figures recovered from the Ge_PnC_Simulation project. It replaces the earlier incomplete preparation ZIP.

**Start with the latest cubic-Ge run, not the old 2.8 GHz prototype. The package is runnable; the device predictions are not publication-validated.** The current best tracked candidate is about **3.042098 GHz** in the sampled **2.953106–3.062947 GHz** in-plane gap. Finite-transmission mesh convergence and sponge sensitivity still fail selected checks. The 4 K bath changes thermal occupation, not the fixed elastic constants.

## 1. Inspect the saved results without installing anything

From this folder, with Python 3.11 or later:

```bash
python tools/verify_saved_results.py
```

This checks copied-source hashes, the archived artifact manifest, all 333 saved harmonic residuals and displacement-ratio calculations, and 60 saved cavity eigen-residuals. It does not run a simulation or repair files.

Open:

- `results/4K/run_20261011T002256Z/SUMMARY.md`
- `results/4K/run_20261011T002256Z/figures/band_structure.png`
- `results/4K/run_20261011T002256Z/figures/cavity_rank1_displacement.png`
- `results/4K/run_20261011T002256Z/figures/cavity_rank1_strain.png`
- `results/4K/run_20261011T002256Z/tables/validation_status.json`

The complete latest saved meshes, selected fields, mode arrays, tables, figures and logs are included. The archived run is preserved byte-for-byte except excluded Finder metadata/caches. Do not execute scripts inside the archived run: some original postprocessors overwrite their own outputs.

## 2. Install an isolated environment on the colleague's machine

Use Python 3.11–3.13. Do not copy Athul's Conda environment.

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
```

On Windows, activate with `.venv\Scripts\activate` instead. The requirements pin the numerical package versions used for the saved baseline, plus the workbook reader. They are top-level pins, not a complete cross-platform lockfile. Gmsh may also need operating-system shared libraries, such as libGLU on Linux. See `docs/ENVIRONMENT.md`.

## 3. Run a small reproduction first

```bash
python handoff.py --name colleague_001
```

The default smoke gate validates the original workbook, creates fresh nominal meshes, solves 4 GHz, runs 2/4/6/8/10 GHz for all three structures, and checks the constitutive/legacy assembly regression. New files go only into `results/4K/run_new_colleague_001/`. An existing run name is rejected. Nothing is uploaded.

Compare the fresh gate against the saved baseline:

```bash
python tools/verify_saved_results.py --compare-run results/4K/run_new_colleague_001
```

The comparison uses the stored numerical gates and a 0.02 dB tolerance for the five-frequency displacement ratios. A match confirms this small baseline reproduction, not full-spectrum convergence or physical calibration.

To inspect saved complex fields too:

```bash
python tools/verify_saved_results.py --fields
```

`--excel-python /path/to/python` is available when openpyxl lives in a separate existing interpreter. A normal fresh environment containing all requirements does not need it.

## 4. Continue only the stages you need

Use the same name with explicit `--resume`, or choose a new name:

```bash
python handoff.py --name colleague_001 --resume --stage bands
python handoff.py --name colleague_001 --resume --stage cavity
```

Stages and prerequisites:

| Stage | Work performed |
| --- | --- |
| `smoke` | Workbook, fresh nominal meshes, 4 GHz + five-frequency gate and operator checks |
| `sweep` | Smoke, then 111 frequencies for reference, periodic and defect beams |
| `bands` | Smoke, then sampled Bloch mesh/orientation study and long-wave benchmark |
| `cavity` | Smoke + bands, then 20 eigenmodes on each of three full-device meshes with identity tracking |
| `robustness` | Smoke + sweep + bands, then selected mesh and sponge comparisons |
| `all` | All five stages, including the existing failed physical-validation checks |

The runner never silently changes the archived baseline. It refuses a simultaneous run through an exclusive lock, verifies source/configuration/input-mesh hashes on resume, and checks saved output hashes before skipping a completed stage. If a crash leaves `.run.lock`, first verify no process is using that run before removing that lock. If sources, inputs or artifacts changed, use a new run name rather than force a resume.

For a full repeat, budget memory, storage and time first. The finest saved full-device mesh has 100,912 nodes and 197,934 triangles. The full run is deliberately opt-in:

```bash
python handoff.py --name full_001 --stage all
python results/4K/run_new_full_001/verify_eigen_scaling.py
python results/4K/run_new_full_001/check_legacy_bloch.py
python results/4K/run_new_full_001/finalize.py
```

Run the three final checks in that order. `finalize.py` writes reports/plots in the new run and requires the dense/sparse benchmark created by `verify_eigen_scaling.py`. Do not run it on the archived baseline. `--stage all` alone does not execute these three final scripts.

## What is included

- `handoff.py`, `tools/`: safe launcher and read-only verification.
- `data/`: original workbook and derived geometry.
- `scripts/`, `config.py`, `geometry/`, `mesh/`: actual historical project source and supporting files. Older entrypoints are retained for traceability; some overwrite their configured outputs. Use the documented launcher for new work.
- `results/4K/run_20261011T002256Z/`: preserved latest completed run.
- `results/4K/portable_template/`: portable staged copy of the original runner; numerical functions are unchanged.
- `reference/legacy/`: earlier isotropic summaries, tables and logs.
- `provenance/`: copied-original checksums, original project documentation and an exact portability patch.
- `verification/`: checks performed when preparing this handoff.
- `docs/RESULTS_AND_NEXT_GOALS.md`: reconciliation of older frequencies and the colleague's next tasks.

The local browser UI, Python environments, caches and unrelated project files are excluded. Historical result summaries can be superseded; use the timeline in `docs/RESULTS_AND_NEXT_GOALS.md` and the latest run's validation gates.

## GitHub upload

No repository has been created or changed. Unzip and review this folder, then upload it to the repository you choose. There are no intentionally included credentials or environments. Provenance logs retain original local project paths; review them and unpublished-research permissions before a public release. No new license is assigned.

One retained mode-array file is about 32 MiB, which exceeds GitHub's browser-upload per-file limit but is below the normal 100 MiB Git limit. Use Git/ GitHub Desktop for the complete folder, or put large result arrays in separate release/storage attachments and document the omissions. Do not claim the full saved-data checks passed after omitting required files. New test runs and caches are excluded by `.gitignore`.

## Verification scope

See `verification/REPORT.md` for the exact tests performed. Full Bloch/cavity/convergence campaigns were not rerun merely to package the project. No mechanical Q, calibrated optical output, single-phonon generation or experimentally validated 4 K HPGe device prediction is claimed.
