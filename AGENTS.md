# Instructions for the continuing research AI

Author and research owner: **Athul Prem**. Project tag: **`_AP_pnc`**.

## Objective and current state

Continue the HPGe phonon-propagation research described in README.md and docs/NEXT_STEPS_AP_pnc.md. Develop a 3D mode-resolved system comparing 100, 200 and 1000 nm thicknesses, supported temperature scenarios and straight/bent/circular geometries. This repository currently contains preserved partial historical evidence and a lightweight runnable scaffold. The 3D solver and campaign are proposed work. Never describe a template, dry run, analytic chain or archival log as a fresh FEM result.

## Read first

Read README.md, docs/EVIDENCE_AND_RECOVERY_AP_pnc.md, docs/AI_HANDOFF_AP_pnc.md, docs/NEXT_STEPS_AP_pnc.md, docs/PHYSICS_SPECIFICATION_AP_pnc.md and reports/REPORT_REQUIREMENTS_AP_pnc.md. Inspect provenance/archive_catalog_AP_pnc.json before using historical paths. The current folder structure supersedes flat-layout instructions in archived documents and the conditional flat preference in the attached PDF.

## Preserve evidence and authorship

- Preserve every relocated historical file's bytes. Use provenance/relocation_manifest_AP_pnc.json to verify them. Changes to a scientific source create a new version, not a rewritten historical source.
- Name new project-specific files `<description>_AP_pnc.<extension>`. Add `Author: Athul Prem | _AP_pnc` in text headers and author/file_tag in supported metadata. Preserve conventional tool filenames with explicit attribution in their content.
- New figures have a visible top line `Athul Prem | athul_simulation_pnc_<identifier>` and filenames `athul_simulation_pnc_<identifier>_AP_pnc.<extension>`.
- Retain failed cases, parameters, uncertainty and source hashes. Do not fabricate missing files, repair expected hashes to match different bytes, suppress failed gates or fill failed solves with an interpolated value.

## Work order

1. Run the available audit and unit checks. Report archive completeness independently of hash preservation.
2. Recover exact missing sources/inputs from Athul's complete original project if available. Verify expected SHA-256. Missing critical files remain a documented block; write an independently versioned replacement only as new implementation with new validation.
3. Establish material provenance and valid regimes before temperature-dependent claims.
4. Validate 3D homogeneous modes and matched-port energy transport before patterned-device sweeps.
5. Establish spatial, modal, spectral and boundary convergence independently. Use subspace tracking near degeneracies.
6. Pilot memory, runtime and disk use; configure limits before large campaigns. Checkpoint compact results, keep selected fields, and avoid redundant fixed-property solves after a successful null test.
7. Complete the evidence ledger, final report and reproducibility package. Follow the delivery instructions below.

## Physical limits

The 2D in-plane archive does not establish a complete 3D gap. A probe displacement ratio is not power transmission. A sponge is not physical intrinsic loss. Bose occupation alone does not change classical normalized scattering. Chemical purity does not prove isotope enrichment. A real eigenfrequency does not establish radiative Q. Do not infer qubit T1/T2*, single-phonon generation or experimental device performance from classical elastic plots.

Hold the single-crystal tensor in laboratory coordinates through bends. Keep geometry, temperature policy, crystal orientation, port normalization and model version explicit. Treat the planned tolerances as starting gates and tighten them to resolve the claimed effect. Separate numerical verification, model adequacy and experimental validation.

## Final report and email

At the end of the research work, produce a detailed scientific PDF plus editable report, compact supporting data, figure index, resolved configurations and reproduction commands. The recipient explicitly specified by Athul is **athul.prem@coyotes.usd.edu**. Send the completed report to that exact address using the executing environment's authenticated email connector when available. Use the subject `HPGe phonon propagation research report | Athul Prem | _AP_pnc`.

Do not send an empty completion message, an unvalidated success claim or unfinished results as a final validated report. If research stops at a genuine blocker, prepare a clearly labeled status/limitations report. If email sending is unavailable, save the delivery-ready report and email draft, tell Athul that delivery is pending and retain the artifacts. Never record `sent` without a successful provider response. No email automation or credentials are installed by this repository reorganization.
