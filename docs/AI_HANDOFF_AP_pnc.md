# Current development handoff

Author: Athul Prem | _AP_pnc

## Scope and state

The original upload is cloned locally. Latest reliable historical inputs were audited; their missing launcher/numerics/config/latest tables do not block the independent 3D implementation. User authorized architecture changes and continuous development, and collaborator push permission is now confirmed. Push and merge through a reviewed PR after checks; no email authorized by the current user.

Active entrypoint: `simulate_AP_pnc.py` (or `Launch_AP_pnc.cmd` on this Windows machine). Old `run_AP_pnc.py` remains an archive readiness checker. Read README.md and docs/IMPLEMENTATION_PLAN_AP_pnc.md before expanding scope.

Implemented: real 3D homogeneous beam meshes; nominal cubic-Ge tensor and rotation; P1/P2 stiffness/consistent mass; free/clamped finite-body eigenmodes; Bloch guided-wave dispersion; source/environment-bound result reuse; failure-attempt preservation; CSV/NPZ/VTU and offline interactive HTML. Gmsh through-hole geometry is available at module level and tested on two holes. The actual 26-hole table is read/validated, not yet integrated into the public simulation runner.

## Verification actually completed

- Archive upload hashes preserved; available saved-smoke evidence passes. Original full latest workflow remains incomplete. `verification/development_baseline_AP_pnc.json` is the actual local audit record, including the initial Windows symlink-permission failure.
- 32 unit/integration tests: 31 passed, one Windows symlink-permission skip, no failures. Log: `verification/development_tests_AP_pnc.log`.
- Three-mesh 200nm clamped-free reference (1755/5187/11475 DOFs), first four modes: final frequency changes all <0.5%, mass-weighted overlaps >0.99999. `verification/straight_beam_validation_AP_pnc.json` and CSV bind results to exact working source hashes and environment.
- Fixed-property 0.02K/4K independent assemblies and solves: identical K/M hashes and frequencies. This is a null control, not a calibrated thermal-material prediction.
- Bloch tests: four Gamma zero modes, +/-k reciprocity of eigenfrequencies, phase matching, mass normalization and free-sidewall rod-speed benchmark. Finite requested branch counts do not establish channel completeness.
- Local CLI default model (6825 DOFs) and nine-wavevector Bloch pilot (945 DOFs) both completed, with interactive HTML output. Actual directories are printed by the runner; current default result is `results/runs/straight_beam_200nm-ea56bb1ac0e2/`, Bloch pilot `results/runs/straight_beam_200nm-a5e8900b7d62/`.
- HTML generation and embedded script interactions were checked. Browser screenshot inspection was blocked by the browser's file-URL policy; do not bypass it or describe visual browser verification as completed.

## Resource evidence

Machine: roughly 15.31 GiB RAM, 8 physical/16 logical CPU cores, RTX4070 Laptop with 8188 MiB VRAM. Available host memory initially about 3.5-3.8 GiB. The finest convergence level took about 6.32 seconds including its comparison, sampled process memory about 336 MiB; default 6825-DOF compute took about 2.19 seconds, about 181 MiB sampled. These are measured small-case results, not forecasts for full devices. CPU backend only. QTCAD official FAQ currently says GPU solve acceleration is not supported; GPU adaptation needs a measured bottleneck and a suitable generalized sparse eigensolver.

Original tracked data about 31.9 MB; virtual environment and generated runs are gitignored. No original science files deleted, no large campaign launched, no reset used.

## Next complete milestone

Implement and validate matched elastic lead ports and the power-normalized scattering problem. Start with a homogeneous, lossless straight guide: identify propagating/evanescent channels, directional power and normalization, solve incident/outgoing waves, verify T near 1 and R near 0 plus complete energy budget and independent boundary/modal/mesh convergence. Then bring the preserved 26-hole pattern into this path. Do not call Bloch eigenvectors a scattering solver or use sponge absorption/probe ratios as physical transmission.

Temperature-dependent material properties, curved/ring geometries, loss, full thermal transport and a complete user-facing parameter application remain later work. Keep crystal axes fixed globally, and use evidence rather than assuming a geometry wins.

## Known implementation/validation limits

- Finite-body mesh gate currently covers four modes of one 200nm homogeneous cantilever; neither all eight display modes nor other thicknesses inherit that gate.
- Bloch samples are frequency sorted, not branch tracked across k or proven complete. Degenerate-subspace tracking is still needed for broader campaigns.
- P2 displacement interpolation uses affine, straight-sided tetrahedral geometry. Circular hole surface accuracy must converge separately. Two-hole removed-volume error at 40nm mesh size remained about 2.18%; do not label the coarse perforated mesh geometry-converged.
- Full results are reused only after identity/output checks. Failed attempts are retained and rerun in a new attempt directory. Partial Bloch rows are recorded but not resumed; large sweep checkpoint reuse is not implemented.
- First CLI attempt revealed NumPy int32 JSON serialization failure. Fixed and end-to-end tested; its original fields remain local as failure evidence. Review also fixed boolean material inputs, initial lock cleanup and P2 field reconstruction metadata. Re-review those changed risks rather than repeating the entire physical review.

- The current runner is used serially. A simultaneous same-case startup or a truncated lock record needs explicit process-state recovery; automatic concurrent scheduling and general crash-lock recovery are deferred. Attempt directories protect already saved scientific files from overwrite.
