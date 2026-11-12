# Handoff verification, 11 November 2026

## Passed

- 300 copied-original file checksums and 180 archived artifact-manifest entries match. Only Finder metadata/caches are intentionally excluded.
- Saved 111-frequency grid, 333 harmonic residuals/power-balance checks and displacement-ratio arithmetic pass.
- All 60 saved cavity eigen-residuals pass; 14 NPZ field/mode files pass finite-value and shape checks.
- Fresh nominal meshes and smoke gate run from the portable handoff copy: 4 GHz test plus five frequencies for reference/periodic/defect beams, 16 harmonic solves total.
- Fresh five-frequency displacement ratios match the archived baseline exactly at the stored precision (maximum difference 0.0 dB). Geometry, patch and constitutive-regression gates pass.
- Small dense-versus-sparse eigenvalue benchmark passes at q=0, 0.37 and 1.
- Explicit resume rechecks hashes and skips the completed smoke stage without repeating its solves.
- Launcher safety tests cover dry-run, path rejection, protected archived output, missing output option, existing lock and unchanged numerical module.

The machine used the existing Python 3.13.11 mechanics interpreter and a separate existing openpyxl 3.1.2 interpreter. No dependency installation or global environment change was performed. Intel MKL emitted deprecation warnings; they are retained in logs.

## Not rerun

Full 111-frequency spectra, Bloch sweeps, full-device eigensolves across all meshes and robustness campaigns were not rerun just for packaging. Their original source, numerical evidence and checksums are included. Fresh-environment installation on the colleague's OS was not tested. This is not a claim of complete platform-independent numerical reproducibility.

## Physical limitations remain

Finite-transmission mesh convergence fails at 5.2 GHz and sponge sensitivity fails at 2.8 GHz. Some high-frequency points are under-resolved. Material/specimen calibration, orientation and 3D validation remain incomplete. Packaging and arithmetic checks do not turn these failures into passes.

## Files

`saved_and_fresh_checks.json` holds the check summary. `fresh_five_frequency_comparison.csv`, `fresh_single_4GHz.json`, `fresh_independent_operator_checks.json`, `fresh_dense_sparse_eigen_benchmark.json` and `fresh_environment.json` record the new small reproduction. Full smoke and resume logs are also supplied. Safety test output is in `safety_tests.log`.
