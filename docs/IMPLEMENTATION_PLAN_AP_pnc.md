# Straight-device implementation plan

Athul Prem | _AP_pnc

[Documentation](README.md) · [Current scientific status](RESEARCH_STATUS_AP_pnc.md) · [Execution handoff](AI_HANDOFF_AP_pnc.md)

## Outcome and current milestone

Deliver a usable configurable 3D elastic-wave simulator and reproducible finite straight 26-hole data. The solver, workbench, saved-result browser, bounded maps, exact-input recovery, regional meshing and nested controls are delivered. PRs [56](https://github.com/premathul/Phononic-Nanobeam-Design-Lab/pull/56) and [57](https://github.com/premathul/Phononic-Nanobeam-Design-Lab/pull/57) delivered preparation reuse, snapshot kernels and saved-data figures; they are not pending tasks.

The content-architecture revision is delivered in [PR #58](https://github.com/premathul/Phononic-Nanobeam-Design-Lab/pull/58), and the three central-mesh frequency traces for drives 3/5 in [PR #59](https://github.com/premathul/Phononic-Nanobeam-Design-Lab/pull/59). The completed saved-data milestone follows [Athul's 21 November comment](https://github.com/premathul/Phononic-Nanobeam-Design-Lab/pull/59#issuecomment-7059248061): preserve the distinction between the positive endpoint change for every fixed coherent mixture of drives 3/5 and its unconverged magnitude. Use the pinned nine-point dataset; no new full-device calculation is needed.

The immediate scientific observable remains per-incident total T/R near 1 GHz. Qualification of those powers, individual weak conversions, complex coefficients at original planes, spectral features and experiment are separate contracts in the [physics specification](PHYSICS_SPECIFICATION_AP_pnc.md#validation-and-uncertainty-contract). The [status record](RESEARCH_STATUS_AP_pnc.md) owns the concise interpretation and source links.

## Delivered coherent endpoint-caption milestone

Delivered in [PR #60](https://github.com/premathul/Phononic-Nanobeam-Design-Lab/pull/60). Its [Windows CI](https://github.com/premathul/Phononic-Nanobeam-Design-Lab/actions/runs/44917860219) passed 532 tests without skips and all required benchmark/integrity steps. The three transmission/reflection intervals reproduce Athul's values, with matrix closure below 1×10⁻¹⁰ percentage points. All 23 affected tests, standard-library browser generation, desktop/mobile checks and byte-identical PDF/PNG/CSV reproduction pass. No new calculation is queued.

Reconstruct Δ(T†T) and Δ(R†R) for 997.5→1000 MHz from incident columns 3/5 and all five outgoing rows, using the saved incident phase correspondence at original planes. Compute both Hermitian eigenvalue intervals, retain their complex matrices, native roots, phases and input hashes in one compact CSV, and show the bounds in Figure 4 and the standard-library offline reader. Preserve the original 18 plotted samples and all native data.

Check the analytic 2×2 bounds against NumPy, phase/permutation invariance, coherent closure and the requirement to retain all outgoing rows. Verify artifact reproduction and desktop/mobile readability; run affected tests and the existing integrated CI once. Stop after the caption, exact export, source/data integrity and authorized GitHub integration are verified. Positive bounds establish the direction only for fixed normalized mixtures on the saved meshes/endpoints; magnitude, continuum sign, intermediate monotonicity, slope and linewidth remain unresolved. No new research calculation is queued by this comment.

## Delivered frequency-comparison milestone

Delivered in [PR #59](https://github.com/premathul/Phononic-Nanobeam-Design-Lab/pull/59): [final CI](https://github.com/premathul/Phononic-Nanobeam-Design-Lab/actions/runs/44789667841) passed 529 tests with no skips and the existing physical benchmark/saved-evidence checks. The next scientific decision below remains unqueued.

The delivery contains two frequency-comparison panels in the offline reader and one reproducible PDF/PNG figure with the exact 18 plotted rows. Both drives show all three mesh levels, with identical frequency/transmission scales, sample markers, explicit reference-drive/native-root identities and the 997.5→1000 MHz changes in percentage points. The caption distinguishes agreement of the direction from the unresolved response magnitude, slope and linewidth.

Use the pinned comparison reader and unchanged native exports. Check the plotted values/identities against those inputs, the six interval changes against the saved rows, standard-library browser generation, desktop/mobile readability and unaffected selection/export behavior. Run the affected tests and existing CI once for the integrated implementation; retain genuine failures and skips separately. Stop after the reader, figure, captions and authorized GitHub integration are delivered. The previous milestone's [527-test CI](https://github.com/premathul/Phononic-Nanobeam-Design-Lab/actions/runs/44780368245) remains evidence for its source version.

## Completed numerical work to reuse

- [Nine thin-device points](../verification/thin_mesh_frequency_AP_pnc.json): 995/997.5/1000 MHz at 225/168.75/126.5625 nm central edge ceilings, fixed 32-edge holes and two quadratic layers. All nine original equation and independent flux checks pass. Native tables retain 45 drive totals and 450 complex coefficients. All six mesh comparisons fail the total-power target for drives 3/5; every drive fails at least one complex increment.
- [Finer regional four-case control](../verification/finer_taper_patch_response_1GHz_AP_pnc.json): both central and exterior changes matter to thin drives 3/5. Their mixed T increments are +1.9543/+1.9245 percentage points. The [preceding control](../verification/taper_patch_response_1GHz_AP_pnc.json), [pair interventions](../verification/goal_patch_response_1GHz_AP_pnc.json) and [signed regional work](../verification/regional_reflection_work_1GHz_AP_pnc.json) remain finite interventions, not error localization or convergence.
- [Paired physical thickness samples](../verification/paired_nested_midpoint_AP_pnc.json), [mode correspondence](../verification/paired_mode_frequency_AP_pnc.json), [layer controls](../verification/thin_device_layer_response_AP_pnc.json) and [wall-faceting controls](../verification/fixed_wall_response_AP_pnc.json) are complete within their declared comparisons. Do not conflate physical thickness with layer count or transfer one-layer lead evidence to a two-layer baseline.
- [Endpoint projection](../verification/snapshot_midpoint_AP_pnc): the ten-field spaces reproduce held-out midpoint outputs within the finite output limits but fail the original 1e-9 equation-residual requirement on every drive. Direct fields remain the accepted solutions; the projection is an output benchmark.
- [Published figures](../figures/README.md), [archived h80 bundle](../data/saved_device_AP_pnc), regional lead/layer controls, saved regional-energy and phase-only diagnostics are delivered. Strong reciprocal agreement at 887.5 MHz does not establish an accurate weak conversion magnitude.

## Next scientific decision; no calculation queued

The unresolved thin-device spatial response blocks a quantitative continuum spectrum for drives 3/5. A new calculation must discriminate a stated source of that uncertainty. More frequency samples on one existing mesh do not resolve it. Retain central/exterior interaction, thickness/compatible-port effects and fixed hole-wall geometry when choosing the next control.

A future isolated refinement must identify the exact accepted donor, source/environment, field/mesh hashes, physical thickness, parent mesh and changed field. `mesh.mesh_size_nm` remeshes a parent; `mesh.central_max_edge_nm` refines eligible central cells of a fixed parent. Check nonzero intended change, actual sizes/quality and protected cells, cut traces, leads, ports, hole walls and material before resource admission. Do not bypass exact-input/source checks. Stop before a solve if isolation, compatibility or resource admission fails. An admitted invocation attempts at most one new point and retains all failed gates.

Spatial qualification requires a systematic, interpretable sequence appropriate to the observable, with at least three resolutions and separate relevant axes. A small total-power increment does not qualify a weak relative power or original-plane phase. Once spatial anchors are adequate, test a narrow interval with signed mode/subspace matching and halved frequency spacing around claimed features. Three sparse points, the 56.25 MHz four/five-channel bracket, a 121-point survey and a 1001-point budget do not certify a cutoff, linewidth, complete gap, Q or attenuation floor.

Broader thickness, bends/rings, temperature, loss and thermal transport remain conditional scientific questions. Each requires its own model inputs and qualification. The future 42 straight/12 bend spectra in the [report contract](../reports/REPORT_REQUIREMENTS_AP_pnc.md) are not an active campaign.

## Bounded lead-only proposal

[Athul's PR #48 comment](https://github.com/premathul/Phononic-Nanobeam-Design-Lab/pull/48#issuecomment-6969077996) requests the saved-data limits and a bounded control proposal for discussion before another solve. The proposed comparison remains unqueued; this comment does not itself authorize another solve or refinement.

Perturb only strict lead-interior cells of accepted h100 z4 at 0.8875 GHz toward h80. Freeze every tetrahedron intersecting the patterned interval [1.5, 15.15] micrometres, including crossing cells, its P2 coordinates/connectivity and actual facet frontier. Preserve material, 200 nm thickness, holes, original end traces, frequency-specific SAFE and radiation conditions. Preflight must demonstrate a conforming frontier and unchanged protected local element operators; shared assembled interface rows may change through lead contributions. If exact isolation is impossible, redesign before solving.

For each directed weak conversion separately, predeclare abs(Pcontrol - Ph100)/Ph100 <= 0.10 using the actual positive h100 power, without a denominator floor. Retain original-equation and physical/coherent-power gates and separate whole-block power/complex 0.01 checks. Two drives answer only the selected pair; five are needed for the full left block. A failure establishes sensitivity to that finite lead perturbation; a pass does not prove pattern dominance or continuum convergence. Actual DOFs, sparsity, factor-index exposure and fresh memory/CPU/disk admission precede execution authorization. No additive regional error decomposition or uniform-guide phase subtraction is justified.

An internal-plane traveling-wave extractor is deferred. Total displacement alone contains forward/backward and evanescent interference. If a specific decision later needs it, validate P2 trace/traction, both directional bases, conditioning, reconstruction and flux on small known fields first.

## Execution and stop policy

Use the existing scikit-fem/SciPy/Gmsh/meshio solver, Qt/headless worker and saved-data generators. The coordinator owns authentication, transfers and interactive Lawrence allocations. Projected peaks above 10 GB run directly on an allocated Lawrence compute node; batch submission is prohibited. Reuse matching checkpoints; preserve separate Windows assembly and Linux solve provenance. Existing saved results do not acquire new source identities merely because report styling changes.

Run meaningful affected tests, one integrated milestone suite and existing CI/physical benchmarks for changed behavior. Preserve raw data and unique failed evidence. No review loop, reset, unsolicited allocation, third-party message or expanded campaign is part of this saved-data task. Stop when the comparison view, reproducible figure, source/data integrity and authorized GitHub integration are verified.
