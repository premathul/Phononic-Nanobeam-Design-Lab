# Straight-device execution handoff

Athul Prem | _AP_pnc

[Documentation](README.md) · [Scientific status](RESEARCH_STATUS_AP_pnc.md) · [Only active implementation plan](IMPLEMENTATION_PLAN_AP_pnc.md)

## Active ownership and scope

The active coordinator is implementing [Athul's PR #58 comment](https://github.com/premathul/Phononic-Nanobeam-Design-Lab/pull/58#issuecomment-7046690925): display all three saved central-mesh frequency traces together for reference drives 3 and 5. The reader and standalone-figure implementations are complete and their workers have finished; the coordinator owns final GitHub integration. The source is the existing nine-point thin grid. No mesh, solve, frequency sweep or remote allocation is needed. Local branch: `codex/thin-frequency-traces`; the publication branch is based on clean public `a192342`.

The previous architecture milestone is complete in [PR #58](https://github.com/premathul/Phononic-Nanobeam-Design-Lab/pull/58), merge `99cae893968b4626737a9fe258e626f2fbf820c6`. It delivered the navigation, saved-reader/workbench, report generators and additive finite qualification. The [plan](IMPLEMENTATION_PLAN_AP_pnc.md) is the single current delivery sequence; [research status](RESEARCH_STATUS_AP_pnc.md) owns the scientific interpretation. No earlier research worker is active locally.

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

This update uses clean public ancestry and exact pinned evidence from the completed local research. Source data, native roots and original phases remain unchanged. The original revision documents and full fields/operators are outside the portable bundle.

[PR #58 Windows CI](https://github.com/premathul/Phononic-Nanobeam-Design-Lab/actions/runs/44780368245) passed on `d569e1e45a1f7930999ad650475f4447448cb9b7`: 527 tests with no skips and the required physical benchmark/saved-evidence steps. The full-device scientific failures remain failures; their integrity checks pass. The earlier failed CI is retained in that pull request. Those results are reused for unchanged solver behavior.

The new trace milestone has independently reproduced the six 997.5→1000 MHz increments from the pinned loader: drive 3 +0.7475703/+5.2401393/+9.7795674 and drive 5 +0.6941248/+5.1215967/+9.5084983 percentage points, in 225/168.75/126.5625 nm order. All 20 affected reader/data/figure tests pass. The 18 plotted samples and six changes match the pinned exports; PDF/PNG/CSV reproduction is byte-identical. Headless Chrome checks pass at 1280 and 390 px: both panels retain all three meshes when selection changes, URL reload and the ten-coefficient selected export still work, and no page errors or horizontal page overflow occur. Existing physical data and previous figures are unchanged. The new pull request CI will provide the integrated source-version check.
