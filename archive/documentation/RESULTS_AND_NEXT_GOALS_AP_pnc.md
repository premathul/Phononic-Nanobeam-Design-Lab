# Results timeline and next goals

## Which result should be reproduced?

Use the latest completed cubic-Ge run `results/4K/run_20261011T002256Z/` as the main handoff baseline. Keep the earlier isotropic model separate.

1. Early isotropic report: sampled gap 2.739599–2.906810 GHz; coarse 30 nm full-cavity winner 2.781573 GHz. This was not the final cavity identity or corrected band sampling.
2. Later isotropic convergence: the 2.781573 GHz identity ends at 2.761484841 GHz. A different identity, initially 2.759873 GHz, becomes the central-localization winner at 2.742280392 GHz. It has strain fractions 0.3745/0.4984/0.6098 within |x|<1/2/2.5 µm and a strain maximum at x=-0.7951 µm.
3. Isotropic physics audit: more complete k sampling corrected the gap to 2.726287–2.843144 GHz. Preserve the earlier numbers as history, not final validation.
4. Latest nominal cubic-Ge model: sampled [100] in-plane gap 2.953106–3.062947 GHz; best tracked candidate 3.042098001 GHz. This is a constitutive-model change, not a measured cooling shift or a demonstrated same-mode track across material models.

Historical evidence is in `reference/legacy/cavity_convergence/tables/cavity_mesh_convergence.csv` and `reference/legacy/physics_validation_2026_10_07/SUMMARY.md`.

## Latest model and evidence

The exact 26-hole workbook is unchanged. Beam width 1000 nm and leads 1500 nm remain provisional. The model is in-plane cubic-anisotropic plane stress, linear triangles, consistent mass, unit thickness and artificial lead sponge. The assumed orientation is [100] beam on (001). Nominal C11/C12/C44 are 128.5/48.3/66.8 GPa and density is 5323 kg/m³. These are not verified 4 K specimen properties. Temperature enters only Bose occupation.

The best candidate has strain fractions 0.2426/0.4443/0.5148 and displacement fractions 0.2528/0.4260/0.5257 within |x|<1/2/2.5 µm. Maximum strain is at x=-0.8547 µm. Effective 2D mode area is 7.28085 µm². Final frequency change is -0.2090% and previous-mesh overlap is 0.999781. It passes the stated best-mode thresholds but is only moderately centered, not a strongly confined validated device mode.

All 333 stored harmonic solves and 60 cavity eigen-residuals pass numerical residual checks. This does not validate the physical model. The strongest sampled defect/reference displacement ratio is -42.249 dB at 4.0 GHz, not transmitted power.

## Failures that must remain visible

- At 2.8 GHz, sponge sensitivity spans 3.26848 dB, failing the 1 dB threshold.
- At 5.2 GHz, final mesh refinement changes transmission by -9.053996 dB, failing the 1 dB threshold.
- Some high-frequency sweep points are under-resolved.
- The coarse [110] sensitivity has no corresponding complete in-plane gap.
- Specimen orientation, verified low-temperature properties, thickness, loss, thermal contraction and support stress remain missing or unverified.
- No 3D validation or calibrated port/energy-flux transmission study is available.

See the latest run's `tables/transmission_robustness_summary.csv`, `tables/wavelength_resolution_diagnostic.csv`, `tables/validation_status.json` and `MATERIAL_PROVENANCE.md`. Rejected numerical attempts are documented in `NUMERICAL_METHODS.md`; their corrected accepted outputs should not be confused with those rejected attempts.

## Colleague's first deliverable

1. Run the saved-data verifier and the fresh smoke gate.
2. Return the source revision, commands, environment, `five_frequency_comparison.csv`, operator checks and verification log.
3. Reproduce the selected bands/cavity stages before changing geometry. Preserve the same mode identities using shape overlaps, not frequency alone.
4. Report failures and parameter differences explicitly. Keep each run in a unique folder.

## Next research priorities

1. Obtain actual orientation, thickness/metrology and verified low-temperature elastic/density/loss data. Read the cited Fine paper and erratum before adopting constants.
2. Resolve the 2.8 GHz boundary sensitivity and 5.2 GHz mesh sensitivity with targeted checks before repeating broad spectra. Develop a defensible nonreflecting boundary and calibrated energy-flux/port metric.
3. Compare against a cubic 3D free-surface model with actual geometry before device-level claims.
4. After stable mode identity and numerical robustness, test modest fabrication variations using justified tolerance ranges. Preserve the original workbook and mirrors initially; save each candidate separately.

Do not claim publication-ready HPGe predictions, radiation-limited Q, single-phonon generation or optical readout strength from this reduced mechanical model. Do not publish, push or contact others on Athul's behalf without his direction.
