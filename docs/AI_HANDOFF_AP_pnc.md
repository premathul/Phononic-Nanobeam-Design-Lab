# Straight-device execution handoff

Athul Prem | _AP_pnc

[Documentation](README.md) · [Scientific status](RESEARCH_STATUS_AP_pnc.md) · [Only active implementation plan](IMPLEMENTATION_PLAN_AP_pnc.md)

## Active ownership and scope

The content-architecture coordinator owns this update and GitHub integration. Documentation, saved-reader/workbench and report-generator work have separate file ownership. The scope is the existing software and saved finite evidence. No full-device mesh, solve, sweep, allocation or external message is queued. The earlier research and figure-delivery chats are idle; the current workspace was clean apart from the user's `修改计划` inputs when this work began.

The three revision materials were read against their pinned public baseline `4f189fd` (PR #57). They describe an older public snapshot. The later local research state at `29bcae5` already contains the nine-point grid and finer regional controls. Do not repeat completed controls because the audit describes them as pending. The [plan](IMPLEMENTATION_PLAN_AP_pnc.md) reconciles the work packages with that state; [research status](RESEARCH_STATUS_AP_pnc.md) owns the concise scientific interpretation.

## Saved work to recover

| Saved item | Exact evidence / reader | Recovery boundary |
| --- | --- | --- |
| Archived 200 nm h80/z6 at 0.1, 0.8875 and 1 GHz | [Dataset manifest](../configs/saved_device_dataset_AP_pnc.json), 14 drive totals and 132 coefficients | Standard-library reading is portable; quadratic fields and original replay need their archive |
| Thin 100 nm, three meshes × three frequencies | [Nine-point receipt](../verification/thin_mesh_frequency_AP_pnc.json), six native CSV/map pairs and [notebook](../results/runs/physical_thickness_device_AP_pnc/thin100_1GHz_mesh_response_AP_pnc.ipynb) | 45 drive totals, 450 native complex coefficients, exact configurations; retain reference-drive versus native-root distinction |
| Finer 1 GHz central/exterior four-case control | [Four-case receipt](../verification/finer_taper_patch_response_1GHz_AP_pnc.json) | Complete finite intervention; no spatial convergence implied |
| Paired thickness and selected hole-pair controls | [Paired samples](../verification/paired_nested_midpoint_AP_pnc.json), [pair responses](../verification/goal_patch_response_1GHz_AP_pnc.json) | Retain own physical thickness, layer, wall and mode-match identities |
| Endpoint projection | [Three per-mesh receipts](../verification/snapshot_midpoint_AP_pnc) and [compact replay](../tools/compare_snapshot_midpoint_AP_pnc.py) | Output comparison passes; every projected solution fails the native equation limit; direct solutions remain authoritative |
| Geometry, 1 GHz sensitivity and 887.5 MHz reciprocity | [Figure gallery](../figures/README.md) | PR #57 delivered the three families; ten PDF/PNG/CSV files were reproducible without native fields |

Original planes remain x=0 and 16.65 µm; raw phases and frequency-local roots are preserved. The thin hierarchy uses fixed 32-edge holes and two quadratic thickness layers. Every direct point passes its original equations and independent flux checks. Spatial and frequency convergence remain unresolved. Reuse the separate layer, wall, full-length uniform-guide, port-refinement and finite-mirror records through the [verification index](../verification/README.md); do not repeat the completed regional-energy or phase-only audits.

## Runtime and provenance

Use `.venv/Scripts/python.exe`. On 20 November 2026 the environment launcher still pointed to a removed bundled Python. Its `pyvenv.cfg` home/executable were redirected to the installed 26.1119 runtime; the existing packages were retained. Python 3.12.14, NumPy 2.5.4, SciPy 1.18.1, scikit-fem 12.0.2 and PySide6 6.12.0 imported successfully. This local environment repair is not a scientific input change or a new provenance claim for old results.

New reports distinguish calculation completion, original equation/column flux checks, coherent propagating-block power and uniform-guide power/phase reference qualification. Historical summaries without the additive qualification dimensions remain historical, with their original hashes and statuses. A report-only source change still changes the existing source identity; do not bypass exact-input cache/resume checks. Recover an original calculation with its recorded implementation/environment when needed.

The last recorded Lawrence allocations 1336454 (node079) and 1336477 (node031) finished with exit 0:0 and were released. The associated JupyterLab stopped. The finer complementary one-point runner took 138.18 s with peak 5,008,196 KiB; its retrieved 90,746,408-byte archive matched SHA-256 `c04086f0679cf9c3002340b042de05c7d867e6bc49b940c05fbe10dc1b17fdfb`. These are historical receipts, not a current capacity observation. No new remote allocation was requested for this task. [Lawrence runbook](LAWRENCE_HPC_AP_pnc.md)

The finest new thin points used Linux assembly/solve; the coarse midpoint reused exact Windows preparation and solved on Linux. Complete native maps remain in the original local and Lawrence archives. The coordinator owns transfers, login and interactive allocations. Recheck the dedicated SSH connection before requesting login. Projected peaks above 10 GB run directly on a Lawrence compute node; no batch submission. Keep one new point per invocation and preserve unique failures.

## Publication boundary and verification

This publication uses clean public ancestry and exact selected evidence from the completed local research. The user's revision documents and untracked full fields/operators are outside the portable bundle. Historical paths in unchanged summaries describe provenance; availability is reported separately by the reader.

The earlier public [PR #57 CI](https://github.com/premathul/Phononic-Nanobeam-Design-Lab/actions/runs/44616692354) passed 466 tests with no skips and the prescribed benchmarks. The prior local nine-point integration ran 518 tests: 517 passed, one Windows symlink privilege skip. These receipts apply to those versions. For this milestone, the first integrated run executed 526 tests with one Windows symlink skip: two report-state omissions, six frozen-source compatibility errors and one transient memory-guard failure were identified. The report text was corrected, the frozen inventory producer restored byte-for-byte, and the affected 111-test suite then passed with no skips. No scientific threshold or memory reserve was relaxed. The new qualification tests and offscreen reader/workbench tests also pass. Headless Chrome checks at 1280 px and 390 px verify selection, URL reload/history, invalid-selection fallback, selected CSV export, no page errors and no horizontal page overflow. The current pull request's existing full Windows CI and physical benchmarks are the final public integration record; previous-version counts are not fresh validation.
