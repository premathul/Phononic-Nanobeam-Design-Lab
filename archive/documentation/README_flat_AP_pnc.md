# Germanium phononic cavity handoff: flat edition

All saved results, figures, meshes, workbooks and scripts are files in this one folder. `flat_manifest.json` maps each filename to its original path and SHA-256. All 496 files present before reorganization are retained byte-for-byte, including earlier smoke runs. No numerical model or saved result has been changed.

## Start here

1. Install Python 3.11–3.13 and `requirements.txt` in an environment outside this folder.
2. Run `python verify_AP_pnc.py --fields --safety` to check saved data and launcher protections.
3. Run `python run_AP_pnc.py --name colleague_001` for the small reproduction gate.
4. Run `python verify_AP_pnc.py --compare-name colleague_001` to compare it against the saved baseline.

Use `python run_AP_pnc.py --name colleague_001 --dry-run` to preview without a simulation.

The adapter restores original source paths in an operating-system temporary directory, runs the unchanged solver, and copies newly generated files back here as `new_colleague_001__...`. The temporary tree is removed afterwards. The original solver uses directories internally; the shared folder stays flat. Allow additional disk space for the temporary copy and requested simulation output. Keep the computer awake during a run. An interrupted or failed run retains outputs when the launcher can exit normally; a forced kill or power loss may leave uncollected temporary output, so do not force-quit it.

## Continue a run

```
python run_AP_pnc.py --name colleague_001 --resume --stage bands
python run_AP_pnc.py --name colleague_001 --resume --stage cavity
```

Supported stages: smoke, sweep, bands, cavity, robustness, all. Full campaigns are opt-in and may require substantial RAM, storage and time. The original source and stage hash checks still apply. Use a new label after changing source or parameters. A flat run lock prevents simultaneous writes under the same name.

After a completed full campaign, in order:

```
python run_AP_pnc.py --name colleague_001 --resume --postprocess verify_eigen_scaling
python run_AP_pnc.py --name colleague_001 --resume --postprocess check_legacy_bloch
python run_AP_pnc.py --name colleague_001 --resume --postprocess finalize
```

`--excel-python /path/to/python` supports an existing separate workbook-reader environment. A fresh environment with all requirements needs no special interpreter option.

## Find the saved evidence

- `latest__SUMMARY.md`: latest cubic-Ge summary.
- `latest__figures__band_structure.png`: sampled band structure.
- `latest__figures__cavity_rank1_displacement.png`: candidate displacement.
- `latest__figures__cavity_rank1_strain.png`: candidate strain.
- `latest__tables__validation_status.json`: actual validation gates.
- `docs__RESULTS_AND_NEXT_GOALS.md`: frequency lineage and next investigations.
- `docs__ENVIRONMENT.md`: dependencies and storage guidance.
- `verification__REPORT.md`: original preparation checks.

The candidate is approximately 3.042098 GHz within the sampled 2.953106–3.062947 GHz in-plane gap. Finite-transmission mesh and sponge checks remain failed. Flattening does not promote these predictions to experimental or publication validation.

## Naming and provenance

Every supplied Python file ends `_AP_pnc.py`. Only `run_AP_pnc.py` and `verify_AP_pnc.py` are direct user entrypoints; the other renamed scripts are provenance-preserved source consumed through the adapter. Do not run archival scripts individually. Their contents, internal imports and original documentation retain historical names intentionally. Use `flat_manifest.json` to resolve any historical path; `original_README.md` is superseded by this README for commands and layout.

For future figures, place a discreet visible title/footer beginning `athul_simulation_pnc_`. Historical plots are preserved unchanged rather than relabeled. New independent scripts should also end `_AP_pnc.py`.

No GitHub publication or upload has been performed. Check future large outputs against GitHub limits before sharing. The original nested folder and full pre-flatten ZIP are retained separately on the Desktop as recovery copies, not included in this upload folder.
