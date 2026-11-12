# Evidence status and archive recovery

Author: **Athul Prem** | Project/file tag: **`_AP_pnc`**
Audit baseline: `068cb8c92d179bfb2049f886278e95a0e7d9c709` | 12 November 2026.

## What was preserved

Every one of the 199 tracked files in the uploaded repository has an exact-byte successor listed in [the relocation manifest](../provenance/relocation_manifest_AP_pnc.json). Historical scripts, arrays, meshes, plots and documents were moved/renamed, not scientifically edited. Active wrappers and documentation are new files; their original versions remain in the archive. The attached continuation PDF was copied unchanged into `docs/references/`.

The original flat manifest lists 496 expected archive entries. Its expectation is larger than the actual GitHub upload. The inventory is:

| Condition | Entries |
| --- | ---: |
| Manifest entries whose named upload was present and hash-correct | 189 |
| Named uploads present but with different bytes | 1 |
| Named uploads absent | 306 |
| Of the absent names, exact expected bytes already exist in another uploaded file | 84 |
| Expected entries now resolvable through the catalogue | 273 |
| Entries still unresolved after exact-hash aliasing | 223 |

The unresolved count comprises **222 missing entries and one changed `requirements.txt`**. The uploaded dependency file contains only NumPy; the historical manifest expected a different file. The active current requirements file is explicitly scoped to the lightweight checks, with the uploaded version preserved under `archive/environment/`.

The seven available Git commits do not contain a complete previous archive to restore. Exact-hash aliases are evidence-preserving mappings, not newly computed or inferred replacement data. An alias satisfies an expected byte identity, but does not create the missing solver dependencies.

## Critical unavailable components

The archive's original launcher (`handoff.py`), portable `numerics.py`, portable `config.json`, material/methods notes and postprocessors are unavailable. So are the original copied-source hash map and several full latest-run tables/fields, including the latest band/cavity validation data. The complete list, with expected SHA-256, is [missing_archive_files_AP_pnc.json](../provenance/missing_archive_files_AP_pnc.json).

The original adapter attempted to restore every manifest entry before running anything; it therefore could not execute from the upload. The new launcher reports this explicitly. It does not simulate, silently skip required inputs, replace the original numerical module or claim that dependency installation repairs missing source.

Even after exact recovery, the execution adapter must be reviewed and enabled before launching the old smoke workflow. The presently supplied `run_AP_pnc.py` is a read-only readiness interface. An actual launch returns a blocked status without creating output directories.

## Checks independently reproducible from the uploaded bytes

The available `run_new_verified_smoke` stage marker refers to 26 outputs. All 26 are present and match their recorded SHA-256. The five-frequency table contains 2, 4, 6, 8 and 10 GHz for reference, periodic and defect structures: 15 saved harmonic records.

| Quantity checked from available data | Result |
| --- | ---: |
| Difference between stored and recomputed relative-amplitude dB | 0 within the stated arithmetic check |
| Maximum stored harmonic residual | 5.29641e-13 |
| Maximum stored source-power versus sponge-dissipation balance error | 8.63074e-14 |
| Defect/reference displacement ratio at 4 GHz | -42.2487 dB |
| Available field nodes / triangles | 18598 / 35540 |

The saved 4 GHz NPZ contains finite point, connectivity, complex-displacement and strain-energy arrays with consistent dimensions. These checks establish **saved-artifact consistency**. They are not new FEM solves, not proof of mesh convergence and not calibrated port-transmission validation. Source power balanced by a numerical sponge is not a complete physical T/R/absorption measurement.

Primary repository evidence is the [saved stage marker](../results/archive_smoke/run_new_verified_smoke/logs/stage_smoke_AP_pnc.json), [five-frequency table](../results/archive_smoke/run_new_verified_smoke/tables/five_frequency_comparison_AP_pnc.csv), [material operator record](../results/archive_smoke/run_new_verified_smoke/logs/material_operator_AP_pnc.json) and [configuration](../results/archive_smoke/run_new_verified_smoke/config_AP_pnc.json). The active verifier resolves their historical names through the catalogue and checks the actual files.

## Historical results versus independently rechecked data

| Claim | Evidence status in this upload |
| --- | --- |
| Latest sampled in-plane gap 2.953106-3.062947 GHz | Reported in historical documentation and the supplied brief; complete latest underlying tables absent |
| Candidate 3.042098001 GHz, last change -0.2090%, overlap 0.999781 | Reported historical result; full latest field/convergence lineage absent |
| 333 harmonic and 60 cavity residual checks passed | An archived verification report is present; full underlying data unavailable for rechecking all records |
| 2.8 GHz sponge span 3.26848 dB, 5.2 GHz change -9.053996 dB | Reported failed checks; keep visible until corresponding source tables are recovered |
| 4 GHz relative displacement response about -42.2487 dB | Recomputable from the available cubic-Ge five-frequency table |
| Older 2.830034-2.935547 GHz band-gap table | Present older lineage; must not substitute for the reported latest cubic-Ge result |
| 3D HPGe temperature/thickness/circular comparison | Proposed work; no corresponding new campaign in this reorganization |

The preserved [historical summary](../archive/documentation/RESULTS_AND_NEXT_GOALS_AP_pnc.md) and [saved verification record](../verification/archive/saved_and_fresh_checks_AP_pnc.json) remain readable. Their statements retain their original dates/context; old `PASS` labels do not mean those tests were rerun now.

## Exact recovery procedure for the next AI

1. Read the missing-entry list, expected hashes and original paths; prioritize the execution-critical entries above.
2. Locate Athul's complete original folder/backup through available authorized access. This repository does not claim access to his laptop.
3. Compare every recovered file against its expected SHA-256 before copying it to a newly attributed recovery path.
4. Add its repository path and explicit recovery provenance to the active catalogue; retain the original inventory and missing-list snapshot.
5. Rerun the available audit, strict complete-archive check and saved-data checks. A changed historical hash remains unresolved unless deliberately handled as a new scientific version.
6. Review the recovered launcher/numerical implementation and update the execution adapter for the folder structure. Only then run the small fresh smoke comparison and record its source/environment/output identities.

Do not block independent literature review or a new separately versioned 3D implementation merely because historical recovery is incomplete. Keep their lineage separate, and record that the original full baseline has not been reproduced.
