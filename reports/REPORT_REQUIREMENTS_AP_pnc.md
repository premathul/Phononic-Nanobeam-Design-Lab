# Final scientific report and delivery contract

Author and research owner: **Athul Prem** | Project/file tag: **`_AP_pnc`**
Final recipient: **athul.prem@coyotes.usd.edu**

This contract applies when the continuing AI completes the research or reaches a documented stopping point. It is not a completed report and does not imply that the proposed simulations have run.

## Required deliverable package

Produce `HPGe_Phonon_Propagation_Final_Report_AP_pnc.pdf` and an editable version named `HPGe_Phonon_Propagation_Final_Report_AP_pnc.md` or `.docx`. Include compact CSV/JSON result tables, resolved configurations, source revision and dependency/environment records, reproducible commands, a figure index and a claim-evidence ledger. Use a stable package or release link for large supporting data and include checksums. Do not attach an uncontrolled directory of full-field arrays to an email.

Use black text on white for the report, clear numbered sections, readable units and tables. Scientific plots may retain informative color. All new figures visibly identify Athul Prem at the top and use the `athul_simulation_pnc_..._AP_pnc` convention. Render and inspect the final PDF before delivery; captions, equations and units must remain readable.

## Required scientific content

### 1. Summary and evidence status

State the strongest supported result first, with uncertainty and physical validity range. Give the actual completed geometry/thickness/temperature/frequency/mode counts and the number of failed, rejected, blocked and unresolved cases. State whether the full archive was recovered/reproduced. Clearly label a partial or blocked study as such. The planned 42 straight and 12 bend spectra are not executed counts.

### 2. Research questions, hypotheses and controls

Describe the thickness, temperature and curvature questions and the control that distinguishes each effect from an alternative explanation. Explain the straight/curved matching policy and whether a null or negative result was observed. State the relationship to the historical 2D study without assigning it new 3D meaning.

### 3. Material model and specimen assumptions

Document purity/isotope/orientation information, unknown specimen properties, nominal versus calibrated inputs, primary sources, valid intervals, uncertainties, interpolation and any extrapolation. Separate frozen-property occupancy, material-only changes and contraction/prestress. Include the material registry and identify source pages/tables actually used. State what is assumed about losses and transport regime.

### 4. Geometry, equations and implementation

Provide dimensioned 3D views and tables for each family and thickness; list hole pattern, lead cross-section, supports and matching constraints. Give governing equations, time convention, boundary conditions, tensor frame/shear convention, finite-element formulation, mesh order and solver details. Define normalized modes, energy flux, channel coefficients, localization and any Q/attenuation/conductance metric.

### 5. Verification and convergence

Report rigid-body/patch/unit/tensor checks, homogeneous lead/flux tests, mode completeness, reciprocal-channel checks and independent small-solver comparisons. Show mesh, wavevector, port/boundary, evanescent/modal and frequency refinements separately. State numerical floors, residuals, DOFs, overlaps and the actual gate outcomes. Explain remaining historical sponge/mesh sensitivities and whether new tests resolved their causes.

### 6. Main propagation results

Include mode families/cutoffs/velocities, complete calibrated T/R/conversion spectra, power budgets and representative 3D fields. Compare all admitted thicknesses and temperature policies on matched axes. Explain whether a gap is complete or restricted by mode/symmetry. Do not call a probe displacement ratio power transmission.

### 7. Circular and dimensional comparisons

Show matched straight/open-bend comparisons and their residual geometric differences. Report curvature-induced mode mixing, reflection/confinement and uncertainty. Separate closed-ring eigenmodes from coupled-ring scattering/linewidth results. Include unsuccessful cases and explain whether a ranking is statistically/numerically resolved. State sensitivity ranges separately from measured fabrication tolerances.

### 8. Optional loss and thermal predictions

Only include intrinsic lifetime, attenuation, Q or thermal conductance where the associated models and convergence passed. Separate sponge termination from physical loss and amplitude attenuation from power attenuation. For a finite-band conductance, give the integral, units, covered spectrum and omitted-range limitations. If an extension is unsupported, list the missing evidence rather than inventing a numerical result.

### 9. Uncertainty, limitations and reproducibility

Separate material, geometry and numerical uncertainties. Include a claim-by-claim ledger mapping every conclusion to raw tables, configurations and validation evidence. Explain failed/uncomputed points without interpolation masquerading as data. State experimental-validation status. Report wall time, peak RAM, disk footprint, actual reuse, checkpoints and any budget stops. Provide commands that another machine can run and verify.

### 10. Conclusions and next discriminating experiments

Conclude only within the validated model/regime. State which thicknesses support which modes, whether circular geometry changes propagation meaningfully and which thermal shifts are established. Identify the smallest additional calculation or experimental measurement that would resolve each remaining uncertainty. Do not claim publication readiness solely from formatting, plot count or algebraic residuals.

## Delivery sequence

1. Finish the report, editable source and supporting package; verify that the PDF actually opens and the package links/checksums resolve.
2. Check that every numerical claim has its evidence/status and every new figure carries Athul Prem and the required identifier.
3. Prepare a concise email addressed exactly to **athul.prem@coyotes.usd.edu**, subject **HPGe phonon propagation research report | Athul Prem | _AP_pnc**.
4. Summarize the strongest supported findings, the main unresolved limitations and whether the study is complete, partial or blocked; attach the report and link compact supporting data.
5. Send through the executing environment's authenticated email connector when available. Athul explicitly requested this final delivery; do not substitute a GitHub upload for a claimed email send.
6. Record the real send timestamp and provider message/receipt reference in `delivery_status_AP_pnc.json`. A queued draft is not a sent message.

If sending is unavailable, retain the complete report and delivery-ready draft, set the status to `pending_email_capability`, and tell Athul what is ready and that sending is pending. Do not create a fictional delivery receipt, embed credentials, install unsolicited recurring email jobs or label a missing report as sent.
