# Current development handoff

Author: Athul Prem | _AP_pnc

## Scope and state

The original upload is cloned locally. Latest reliable historical inputs were audited; their missing launcher/numerics/config/latest tables do not block the independent 3D implementation. User authorized architecture changes and continuous development, and collaborator push permission is now confirmed. Push and merge through a reviewed PR after checks; no email authorized by the current user.

Active entrypoint: `simulate_AP_pnc.py` (or `Launch_AP_pnc.cmd` on this Windows machine). Old `run_AP_pnc.py` remains an archive readiness checker. Read README.md and docs/IMPLEMENTATION_PLAN_AP_pnc.md before expanding scope.

Implemented: real 3D homogeneous/perforated beam meshes; nominal cubic-Ge tensor and rotation; P1/P2 stiffness/consistent mass; free/clamped finite-body eigenmodes; Bloch guided-wave dispersion; SAFE waveguide ports; power-normalized harmonic scattering from either end; source/environment-bound result reuse; failure-attempt preservation; CSV/NPZ/VTU and offline interactive HTML. The actual 26-hole table is integrated into the public runner with explicit origin placement and raw CSV provenance. New `perforated_AP_pnc.py` leaves the validated original numerical modules unchanged.

All project artifacts, software text and GitHub communications use English. Direct conversation and progress updates to the user may use Chinese. Six existing local reports were translated from saved fields without new solves; their numerical-file hashes and viewer payloads are unchanged. Presentation metadata records the old/new HTML hashes and generator identity. The original failed attempt retains its failed status.

## Verification actually completed

- Archive upload hashes preserved; available saved-smoke evidence passes. Original full latest workflow remains incomplete. `verification/development_baseline_AP_pnc.json` is the actual local audit record, including the initial Windows symlink-permission failure.
- Current integrated suite: 46 tests, 45 passed, one Windows symlink-permission skip, no failures. Log: `verification/development_tests_AP_pnc.log`. New tests check Gmsh end traces/geometry, CSV provenance and cache invalidation in addition to the established mechanics, SAFE and scattering checks.
- Three-mesh 200nm clamped-free reference (1755/5187/11475 DOFs), first four modes: final frequency changes all <0.5%, mass-weighted overlaps >0.99999. `verification/straight_beam_validation_AP_pnc.json` and CSV bind results to exact working source hashes and environment.
- Fixed-property 0.02K/4K independent assemblies and solves: identical K/M hashes and frequencies. This is a null control, not a calibrated thermal-material prediction.
- Bloch tests: four Gamma zero modes, +/-k reciprocity of eigenfrequencies, phase matching, mass normalization and free-sidewall rod-speed benchmark. Finite requested branch counts do not establish channel completeness.
- Local CLI default model (6825 DOFs) and nine-wavevector Bloch pilot (945 DOFs) both completed, with interactive HTML output. Actual directories are printed by the runner; current default result is `results/runs/straight_beam_200nm-ea56bb1ac0e2/`, Bloch pilot `results/runs/straight_beam_200nm-a5e8900b7d62/`.
- HTML generation and embedded script interactions were checked. Browser screenshot inspection was blocked by the browser's file-URL policy; do not bypass it or describe visual browser verification as completed.

- Matched-port CLI: `python simulate_AP_pnc.py --config configs/straight_scattering_AP_pnc.json`. The default 0.3 GHz, 2000 × 1000 × 200 nm control has 2625 DOFs and four incident channels, max R=1.2064e-6, max solve residual=2.02e-16. Actual local output: `results/runs/straight_scattering_200nm-c151b24df56f/`. Its power check does not establish spatial convergence.
- Bounded scattering acceptance at 1 GHz, 600 × 200 nm cross-section, 1 and 1.5 µm lengths, both incident ends: max complex transmission error against common finer SAFE reference 0.00453668/0.00683367; final section phase drift over 1.5 µm 0.00534454 rad; minimum mode overlap 0.999999820. Final long-guide max R about 1.20e-7, direct flux imbalance 5.11e-15, linear residual 5.66e-16. See `verification/straight_scattering_acceptance_AP_pnc.json` and CSV.
- Original `straight_scattering_validation_AP_pnc` failed coarse axial/absolute-phase gates; `straight_scattering_refinement_AP_pnc` failed only long-guide absolute phase against its common reference. Both records remain unchanged and their hashes are linked by final acceptance. A complete energy budget alone did not establish accurate propagation phase.
- Final acceptance reuses unchanged reference evidence after checking source/configuration/library hashes and performs one new long-guide solve plus a finer SAFE solve. Fresh CI output goes to `results/ci-scattering` with `python -B tools/validate_scattering_AP_pnc.py --acceptance --output-dir results/ci-scattering`. Do not rerun earlier cases merely because a conversation restarts.

- Two-hole 1 GHz study: 1800 × 600 × 200 nm beam, 250 nm holes at x=625/1175 nm. Body sizes 160/130/105 nm, 32 circle segments and 150 nm port target; extra controls use 48 segments, +300 nm leads at each end, 75 nm ports and 80 nm body size. Final main body T/R change 0.003661; curvature 0.0005393; lead 0.0015244; port 0.00134445; extra body 0.00399263. Maximum budget error 1.17e-14 and reciprocal amplitude error approximately 3.1e-13. Final complex-response change 0.03162461 fails the 0.01 phase gate. `verification/two_hole_scattering_validation_AP_pnc.json`/CSV retain 64 directional rows, actual complex matrices and the failed gate.
- Reusable raw cases/meshes are local under `results/perforated_validation/`. `tools/validate_perforated_AP_pnc.py` checks computational inputs/source hashes before reuse. The first two results survived a summary-serialization failure and have explicit equivalent-checkpoint provenance; no physics was rerun for that repair. A port-interpolation ordering error was fixed using saved basis data only. The complete requested study returns 2 solely for the retained body phase failure. Default four-case validation covers power/geometry; CI labels that scope explicitly.
- Targeted diagnosis excluded a modal phase-alignment artifact: the h105/h80 port coordinates, triangulations and k values are identical, and overlaps are unity to numerical precision. The largest change is same-channel mode-2 transmission, with a 0.03205515 rad phase shift. The body and wall thickness discretizations changed together, so the remaining error has not been isolated to one subregion. No additional solve was needed for this diagnosis.
- Full 26-hole public CLI pilot: `python simulate_AP_pnc.py --config configs/historical_26hole_AP_pnc.json`. Actual local output `results/runs/historical_26hole_pilot_200nm-265d8f375158/`; 56220 P2 DOFs, 8019 tetrahedra, five input channels at 1 GHz, T=0.96006–0.99974, max direct power imbalance=2.23e-13, max solve residual=2.29e-16. The 16650 nm beam is 1000 nm wide and 200 nm thick; 8325 nm origin shift leaves 1500 nm from each outer hole rim to its port. Portable pilot record and CSV are in `verification/historical_26hole_pilot_AP_pnc.*`. This is completed_unverified, not a band-gap or converged-device result.

## Resource evidence

Machine: roughly 15.31 GiB RAM, 8 physical/16 logical CPU cores, RTX4070 Laptop with 8188 MiB VRAM. Available host memory initially about 3.5-3.8 GiB. The finest convergence level took about 6.32 seconds including its comparison, sampled process memory about 336 MiB; default 6825-DOF compute took about 2.19 seconds, about 181 MiB sampled. These are measured small-case results, not forecasts for full devices. CPU backend only. QTCAD official FAQ currently says GPU solve acceleration is not supported; GPU adaptation needs a measured bottleneck and a suitable generalized sparse eigensolver.

Original tracked data about 31.9 MB; virtual environment and generated runs are gitignored. No original science files deleted, no large campaign launched, no reset used.

Scattering resource measurements: default CLI 0.34 s and 105.4 MiB sampled process peak; final 14553-DOF long benchmark 19.64 s and 605.23 MiB. The finer SAFE reference used 429 trace DOFs and 1.29 s. These remain small CPU cases, not evidence that full-device or GPU calculations will cost the same.

New measured costs: 17817-DOF targeted two-hole body case 18.51 s/614.6 MiB; 10908-DOF refined-port case 9.11 s/366.8 MiB; 56220-DOF original device 14.76 s/1088.8 MiB. The latter's longer, thinner sparsity pattern differs, so DOF alone does not predict cost. Only its config raises the FEM cap to 60000. Gmsh builds the mesh before the FEM cap is checked; extreme meshing settings are not protected by that cap. No GPU solve or large sweep has been launched.

## Next complete milestone

Resolve the observed perforated complex-response convergence and establish a bounded full-device mesh/port comparison before broad spectral or resonance claims. Use saved eight-case checkpoints to identify the actual phase error and target a smaller additional calculation if justified; do not repeat the completed campaign. Full-device frequencies, geometry refinement and channel counts need their own acceptance checks. Keep pilot execution, power convergence and complex-response convergence distinct.

Temperature-dependent material properties, curved/ring geometries, loss, full thermal transport and a complete user-facing parameter application remain later work. Keep crystal axes fixed globally, and use evidence rather than assuming a geometry wins.

## Known implementation/validation limits

- Finite-body mesh gate currently covers four modes of one 200nm homogeneous cantilever; neither all eight display modes nor other thicknesses inherit that gate.
- Bloch samples are frequency sorted, not branch tracked across k or proven complete. Degenerate-subspace tracking is still needed for broader campaigns.
- SAFE computes all roots of the finite cross-section pencil, with flux orthogonalization of degenerate propagating modes and explicit failure at unresolved cutoffs. This does not prove continuum channel completeness. Current leads have identical sections and uniform material at both ends. The final phase reference is numerical and has no rigorous continuum error bound.
- `matched_reference_passed` is a per-run power-only check. `spatial_convergence_established` and `calibrated_scattering_available` remain false in arbitrary run summaries; separate benchmark acceptance does not certify a different mesh, frequency or device. Full driven responses use displacement per square root of incident watt, not mass-normalized eigenvectors.
- P2 displacement interpolation uses affine, straight-sided tetrahedral geometry. Circular hole surface accuracy must converge separately. Two-hole removed-volume error at 40nm mesh size remained about 2.18%; do not label the coarse perforated mesh geometry-converged.
- The new port-compatible mesher fixes regular circle faceting independently of body size. Its 32/48 segments give 0.641315%/0.285334% removed-volume errors. This newer path is separate from the original coarse geometry test above. Full 26-hole mesh and phase accuracy remain unverified; the two-hole power acceptance does not transfer to that device.
- Full results are reused only after identity/output checks. Failed attempts are retained and rerun in a new attempt directory. Partial Bloch rows are recorded but not resumed; large sweep checkpoint reuse is not implemented.
- First CLI attempt revealed NumPy int32 JSON serialization failure. Fixed and end-to-end tested; its original fields remain local as failure evidence. Review also fixed boolean material inputs, initial lock cleanup and P2 field reconstruction metadata. Re-review those changed risks rather than repeating the entire physical review.

- The current runner is used serially. A simultaneous same-case startup or a truncated lock record needs explicit process-state recovery; automatic concurrent scheduling and general crash-lock recovery are deferred. Attempt directories protect already saved scientific files from overwrite.
