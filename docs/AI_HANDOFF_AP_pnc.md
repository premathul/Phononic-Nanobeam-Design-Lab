# Working prompt for the next research AI

Author and research owner: **Athul Prem** | Project/file tag: **`_AP_pnc`**

## Task

Continue the research in this repository toward a verified three-dimensional mode-resolved study of phonon propagation in nominal pure HPGe. Compare identical in-plane patterns at 100, 200 and 1000 nm thicknesses. Establish which lead modes propagate and how geometry and supported temperature-dependent material changes affect power transmission, reflection, conversion and localization. Compare straight devices to open circular-pattern bends, then closed and coupled rings where justified. Produce substantial, scientifically distinct results and a detailed final report for Athul Prem.

Begin by reading root AGENTS.md and README.md, the supplied continuation PDF, EVIDENCE_AND_RECOVERY_AP_pnc.md, PHYSICS_SPECIFICATION_AP_pnc.md and NEXT_STEPS_AP_pnc.md. Use the current folder map. Archived instructions reference a superseded flat layout.

## Current facts to preserve

All 199 uploaded files are preserved. The historical manifest has 496 entries; 273 are resolvable through original files or exact-hash aliases, and 223 are unresolved. The full original FEM runner is unavailable; the current run wrapper is a readiness checker and does not execute a solver. The active NumPy utility is a geometry/analytic-chain scaffold. The templates are specifications, not a completed simulation platform.

Available saved-smoke evidence can be checked: 26 output hashes, five frequencies for three structures, residual/arithmetic consistency and the saved 4 GHz field. Full latest-band/cavity evidence is incomplete. Preserve the distinction between those checks, historical claims and new results. A previous `PASS` line is not evidence that you ran the test.

## First working session

Run `python tools/audit_repository_AP_pnc.py`, `python verify_AP_pnc.py --fields --safety`, the active unit tests, and `python run_AP_pnc.py --name hpge_reference --dry-run`. Record actual exit codes and the exact revision. Inspect the missing catalogue and available sources. Recover original dependencies by hash if possible; otherwise document the block and make any replacement solver an independent implementation with new verification.

Produce a short session record containing: current state, commands run, verified evidence, unresolved inputs, changed files, next discriminating test, resource use and whether any simulation actually executed. Never imply that the planned 54-spectrum matrix is already complete.

## Implementation contract

Use the planned modules as a decomposition, not a claim that they already exist:

| Proposed module | Responsibility |
| --- | --- |
| `material_registry_AP_pnc.py` | Sourced properties, interpolation, frame rotation and temperature policy |
| `geometry_factory_AP_pnc.py` | Straight/bent/ring construction, dimensions and validity |
| `mesh_controls_AP_pnc.py` | Curvature, wavelength and through-thickness refinement |
| `lead_modes_AP_pnc.py` | Complete lead channels, cutoffs, tracking and power normalization |
| `solve_scattering_AP_pnc.py` | Driven elastic solution, full mode conversion and energy accounting |
| `sweep_campaign_AP_pnc.py` | Dependency-aware execution, resource limits and restart |
| `analyze_results_AP_pnc.py` | Metrics, uncertainty, lineage and claim tables |
| `plot_results_AP_pnc.py` | Scientific plots with Athul Prem and the required identifier |
| `validate_physics_AP_pnc.py` | Small independent checks, convergence and regression gates |

Use strict configuration validation and SI units. Save the resolved configuration, source commit, source/input hashes and solver/environment versions with every run. Compute a full case identity and a separate mechanical-operator identity for correct reuse. Save completed tasks atomically; protect shared outputs with locks; do not resume under changed scientific inputs.

Set resource budgets after a pilot. Avoid saving a large displacement vector at every sampled frequency unless needed. Keep compact numerical tables for all points, and fields for selected passbands/gaps/resonances/conversion/failure cases. Preserve failed or blocked cases as data with clear state. Do not overwrite original evidence or hide unsupported material ranges.

## Scientific reasoning requirements

Explain the governing equations, assumptions, parameter provenance, observable definitions and uncertainty behind every claimed result. Match geometry/material/ports across comparisons. A 2D thickness prefactor cannot supply 3D thickness physics. Freeze the laboratory crystal tensor through bends. Use the physical elastic flux and complete outgoing channels. Keep temperature occupancy separate from material/property changes, sponge absorption separate from physical loss, and closed-ring modes separate from open-device linewidths.

Validate before expanding the matrix. Inspect mode identity and field/subspace overlap, not only frequency indices. Establish numerical floors and independent convergence; tighten tolerances when an effect is small. Treat negative outcomes as valid observations. Do not promise publication; provide the evidence needed for independent review.

## Files, figures and attribution

Every new independent project file uses `_AP_pnc` before its extension, with Athul Prem in readable headers or metadata. Preserve required standard filenames as described in AUTHORSHIP_AND_NAMING_AP_pnc.md. Every new figure has a visible top line `Athul Prem | athul_simulation_pnc_<identifier>` and a corresponding `_AP_pnc` filename. Keep historical images unchanged and identify any new derivative explicitly.

## Final deliverable and sending

Prepare the report and supporting package specified in `reports/REPORT_REQUIREMENTS_AP_pnc.md`. Include the strongest supported findings, exact executed counts, methods, convergence, full uncertainties, failed cases, limitations and reproducible commands. Deliver a PDF and editable version plus compact data/configuration/figure indexes. State what remains unresolved.

**Send the completed report to athul.prem@coyotes.usd.edu.** Use an authenticated email connector available in the execution environment and the subject `HPGe phonon propagation research report | Athul Prem | _AP_pnc`. Validate attachment names and contents before sending. Record the actual returned message/receipt reference. If sending is unavailable, save a delivery-ready email draft and report, and clearly state that delivery remains pending. Never create a fictional sent status or place credentials in the repository.
