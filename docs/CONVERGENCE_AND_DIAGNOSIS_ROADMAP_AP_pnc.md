# Two-path research roadmap: convergence, diagnosis and the role of 2D sweeps

Author: Athul Prem | _AP_pnc
Prepared: 17 November 2026.
Evidence baseline: PR #37, merged as `98aaca33c328f39b6fdb6df5619549a20f1cb663`.

## Purpose and execution boundary

This document records two research paths requested by Athul: how successful 3D verification can lead to a publishable frequency-response study, and how to diagnose failure without spending resources on an uncontrolled refinement ladder. It also defines what the existing 2D multi-domain sweeps can and cannot contribute.

This is a planning deliverable, not a submission script or authorization for a new campaign. It starts no jobs, changes no solver, mesh, temperature, material or acceptance threshold, and does not replace current run ownership in [AI_HANDOFF_AP_pnc.md](AI_HANDOFF_AP_pnc.md). Before any future proposed change, discuss its scientific purpose with Athul and follow the current execution/resource permissions. Preserve accepted checkpoints and all failures. Review the latest handoff before acting; this baseline will age.

## 1. What is actually established

| Evidence at this baseline | Supported conclusion | Not established |
| --- | --- | --- |
| Four left-incident h80/z6 drives at 0.1 GHz; transmission 0.980155–0.999849 | High transmission in the saved finite 3D discretization at this one frequency | A passband over an interval, a gap, or spatial convergence |
| Replayed original equations, independent nodal-force flux and full field identities | Strong internal verification of the retained solution and export | Experimental validity or accuracy of the continuum/material model |
| Left-block defect norm of R†R + T†T − I = 1.00825e-10, with direct complex-superposition replay | Coherent conservation within the saved four-column left-incident block | Full two-sided unitarity or reciprocity |
| 1 GHz complex changes: h100/h80 0.0399272; h80/h60 0.0268670; h80 z4/z6 0.0167063 | Separate in-plane and joint body/port thickness sensitivities remain significant | Passing the unchanged 0.01 complex criterion |
| Matched 3 µm uniform h80/h60 control passes at complex change 0.00281635 | Improved finite-guide propagation control | A correction to subtract from the 16.65 µm patterned-device phase |
| Section inventory: 4/5/8/12/13 open channels per direction at 0.1/1/2.8/5.2/6 GHz | Frequency-local propagating-channel inventories | Intervening cutoffs, continuous branch correspondence or body transmission at all five frequencies |

The modal conservation defect is comparable to measured flux-basis departures. Its small value is not evidence of ten-decimal physical accuracy. Numerical verification, discretization convergence, model adequacy and experimental validation are four distinct claims.

Sources: [low-band evidence](../verification/low_band_device_AP_pnc.json), [thickness evidence](../verification/device_thickness_AP_pnc.json), [h60 evidence](../verification/device_h60_AP_pnc.json), [uniform control](../verification/uniform_h60_AP_pnc.json), [port inventory](../verification/port_inventory_AP_pnc.json).

## 2. Path A: successful 3D convergence toward a publishable study

### A1. Freeze the scientific question and comparison contract

Choose one observable before choosing a large sweep: for example, stability of low-band reflection, a frequency-local transmission minimum, or a defect-localized mode. Record geometry, hole polygons, physical thickness, crystal-to-lab rotation, nominal elastic tensor/density, boundary conditions, port planes, incident side and frequency. Keep those physical inputs fixed during numerical refinement.

Retain the raw complex scattering coefficients at the original planes, not only power plots. Compare modes after signed physical correspondence and permitted gauge alignment; root numbers alone are not identities. Report unmatched or ambiguous modes rather than forcing an assignment. Do not apply a uniform-guide phase subtraction.

**Output:** a one-page case contract with source/input hashes, observable, comparison metric, tolerance, resource ceiling and stop condition.

### A2. Complete a discriminating spatial control before expanding frequency coverage

The existing plan proposes a matched 0.1 GHz h80 z4/z6 comparison. It can test whether two stronger and two weaker reflectances persist when thickness discretization changes jointly in the body and ports. Reuse the accepted z6 solution and archived compatible z4 inputs. A new compatible z4 frequency-specific port basis must be checked before any proposed solve.

A passing pair is one finite increment, not continuum convergence. Use the existing three-level in-plane evidence, and plan additional independent directional evidence only where needed for the intended claim. Thickness, port width, faceting, axial propagation and patterned-region resolution must not be conflated. For any new selected refinement, document which quantities truly stay fixed and how its cost is admitted.

Keep the existing 0.01 power and complex-response gates. For the weak 0.1 GHz reflectances, approximately 1.91213e-4 and 1.50783e-4, an absolute power tolerance of 0.01 is too loose by itself to resolve their ordering or relative size. Also report absolute changes, relative changes with a stated small-signal floor, and matched complex-reflection changes. Do not silently replace the existing pass criterion with a new one; distinguish a general gate from claim-specific uncertainty.

**Exit evidence:** stable claim-relevant observables across appropriate successive refinements, signed mode matching, raw failure history and a defensible uncertainty statement. Richardson extrapolation or an observed convergence order is conditional on a credible asymptotic regime and compatible refinement family, not automatic.

### A3. Verify the scattering formulation over the selected regime

At each representative frequency, require the original-equation relative/backward checks, independent physical flux and complete incident/output channel inventory. Preserve the current solver target and gates. A preconditioned or transformed residual is not a substitute for the original residual.

Retain the left-block coherent check where only left incidence is computed. If the scientific claim requires arbitrary two-sided excitation, full scattering symmetry or reciprocity, propose the corresponding right-incident calculation and basis-aware checks explicitly; do not infer those results from left incidence alone. Reciprocity tests require consistent reciprocal-channel pairing, signs and normalization. Near-cutoff modes need special care because power normalization and classification can become ill-conditioned.

Check lead truncation/radiation treatment, relevant evanescent content and port discretization with bounded controls appropriate to the actual formulation. Passing total power conservation can coexist with inaccurate phase or an incorrectly represented boundary.

**Exit evidence:** a clearly scoped verified scattering operator, not an unqualified “unitary solver” label.

### A4. Build a bounded frequency map around a physical question

Reuse verified mesh/K/M and frequency-independent section operators where physically applicable; construct the correct frequency-specific modes and equations. Use the existing resumable map and archive importer rather than recomputing accepted anchors. The importer itself is computation-disabled.

Begin with a justified bounded set, not a default dense survey. Identify where amplitudes, phases, mode counts or candidate resonances change. Propose local sampling refinement there and demonstrate that the feature location, depth or width is stable to finer sampling. Separate four-channel and five-channel inventories unless correspondence is established. Break tracking at cutoffs, failures and ambiguous/degenerate matches; a subspace treatment may be needed for degeneracies.

A smooth interpolated plot cannot certify missing frequencies. Label each point as converged for the stated observable, unconverged, failed or pending. Spatial checks at 0.1 or 1 GHz cannot automatically certify 2.8–6 GHz, where wavelengths and open-channel counts differ.

**Output:** a reproducible table of T/R/conversion and raw complex coefficients, selected fields, frequency-local channels, uncertainty and status at every retained point.

### A5. Test the proposed interpretation, not only the plot

- **Stopband claim:** distinguish finite-device suppression from a periodic Bloch gap. Check relevant polarizations/branches and finite-length effects; a suppressed input channel is not a complete elastic gap.
- **Cavity claim:** require spatial localization, mode tracking and separation from boundary/lead artifacts. An isolated closed-system eigenfrequency does not give radiation loss or Q.
- **Q claim:** specify the open-system/decay/linewidth method, frequency resolution and loss channels. Numerical sponges are not intrinsic material loss.
- **Thermal claim:** distinguish temperature-dependent material properties from Bose occupation weighting. Fixed linear-elastic coefficients do not gain a temperature-dependent normalized classical scattering response merely by changing occupation.
- **Device claim:** quantify the effect of realistic geometry/material uncertainty separately from numerical error before making fabrication predictions.

Failure to find a gap or useful localized mode is a legitimate bounded result. Do not move the goalposts to guarantee a positive outcome.

### A6. Assemble the publication package and independent cross-check

Package resolved SI inputs, model assumptions, source/environment versions, exact commands, accepted checks, failed comparisons, compact response tables and field-generation scripts. Link large archived arrays by verified provenance without claiming that Git alone contains a full runnable dataset.

Have an independent implementation first match the uniform and two-hole controls, then selected full-device points with identical conventions. Independence means more than running the same code on another operating system. A commercial solver is a cross-check, not an automatic truth source.

A publication-ready methods claim needs a clear physical question, demonstrated novelty or useful negative result, reproducibility, claim-specific numerical uncertainty and appropriately limited material assumptions. Passing CI or a mesh-pair threshold alone does not guarantee publication. Athul reviews the final scientific interpretation before submission or public release.

## 3. Path B: stepwise diagnosis when a gate fails

Work from the earliest failed layer. Do not diagnose spatial convergence using an unaccepted linear solve, and do not diagnose a material effect by changing the numerical model at the same time.

| Step | Evidence to inspect first | Bounded response | Stop / return condition |
| --- | --- | --- | --- |
| B1. Provenance or startup | Input/source hashes, phase receipts, compatible helper versions, units and run ownership | Fix the identified workflow mismatch in a separately reviewed version; preserve the rejected attempt | No numerical launch until exact intended inputs and ownership are known |
| B2. Resource admission | Matrix size/nnz, factor-index exposure, per-job and aggregate RAM/disk, wall time and checkpoint integrity | Use measured pilot data; select one supported resource/method change, not an identical blind retry | If safe admission fails, retain a resource-limited result; do not lower scientific gates |
| B3. Linear solve | Original relative/backward residuals, history, finite values, scaling and preconditioner effects | Reuse the exact operator and best verified iterate; compare one discriminating solver intervention | Plateau without meaningful improvement ends that attempt; no accepted scattering export |
| B4. Flux or coherent power | Traction signs, outward normals, exp(-iωt) convention, power normalization, missing channels, signed Gram, independent nodal force | Replay saved columns and small counterexamples before another body solve | Repair demonstrated implementation error; rerun affected checks, not unrelated campaigns |
| B5. Mode comparison | Signed wavenumbers/energy direction, overlap, phase gauge, near-degeneracy and cutoff proximity | Establish correspondence on a common physical representation; mark ambiguous comparisons | No per-channel convergence claim while matching is unresolved |
| B6. Spatial phase or amplitude | Separate in-plane, thickness, port-width, faceting and lead sensitivity; original-plane complex coefficients | Choose one control that distinguishes competing explanations | A pass in one direction does not close other failures |
| B7. Frequency feature | Local sample spacing, mode-count changes, nearby poles/minima and failed samples | Propose bounded local frequency refinement after valid solves | No linewidth/gap interval inferred across unresolved samples |
| B8. Model adequacy | 2D reduction, crystal orientation, geometry, boundary conditions, fixed-property assumptions and losses | State the missing physical input and compare justified model alternatives separately | Do not tune parameters to fit a desired curve without independent support |

### Diagnosing the current phase-convergence failure

The shrinking same-sign 1 GHz phase increments are useful finite evidence, but do not prove an asymptotic error law. The uniform controls suggest propagation sensitivity, while their shorter length prevents transferring an error bound to the patterned device. The z4/z6 failure separately shows joint body/port thickness sensitivity.

Use saved fields and comparisons first. A future frozen-lead/patterned-region control would help isolate where a change matters only if the lead elements and conforming interface are demonstrably unchanged. A finer-port comparison is not a pure body test. A changed hole polygon is a geometry change as well as a discretization intervention unless a consistent curved-boundary convergence study is defined.

If a diagnostic identifies a code defect, correct it with a targeted regression and assess which accepted results it invalidates. Preserve old outputs as superseded evidence; do not relabel them as verified. If no defect is found and the affordable controls remain unstable, report the observable as unconverged. A smaller verified problem, a more efficient method, or a carefully qualified negative numerical result is preferable to unlimited refinement.

### Minimum failure record

For every stopped branch retain: intended question; exact inputs; completed and uncomputed stages; first failing metric with value and threshold; resource record; accepted checkpoint identities; one justified next diagnostic; and an explicit stop condition. Distinguish “failed numerically,” “not converged,” “not modeled” and “not computed.”

## 4. Existing 2D multi-domain sweeps: useful screening, not a substitute for 3D

The historical reference/periodic/defect calculations and wider 2D sweeps remain useful. However, their lineage is mixed and recovery is incomplete. The repository audit rechecks 15 harmonic records at 2/4/6/8/10 GHz and available 4 GHz fields; some later gap/cavity tables and execution-critical source files are missing. A table or historical PASS label is not proof that every domain/frequency has been independently reproduced.

Use the [evidence/recovery ledger](EVIDENCE_AND_RECOVERY_AP_pnc.md) to classify each dataset before comparison. The saved 4 GHz defect/reference value of approximately −42.2487 dB is a **relative displacement response**, not calibrated transmitted power. The historical latest reported in-plane gap of 2.953106–3.062947 GHz and candidate 3.042098001 GHz lack complete latest lineage in this upload; do not mix them with older numbers or present them as verified 3D predictions.

| Appropriate use of 2D results | Required safeguards | What must not be inferred |
| --- | --- | --- |
| Rank candidate geometries and frequency windows cheaply | Same declared reduced equations, forcing, boundaries, material assumptions and observable; inspect their own mesh/sponge sensitivity | Accurate 3D ranking without representative cross-checks |
| Compare reference, periodic and defect responses | Keep drive/measurement definitions identical and distinguish displacement, energy and flux | A displacement notch equals a power stopband |
| Form mechanism hypotheses from fields | Track polarization and localization; identify whether plane stress, plane strain or another reduction was used | Missing out-of-plane/thickness modes are absent physically |
| Debug simpler cases and test trends | Match a justified limiting case of the 3D model and compare equivalent observables | Agreement of unrelated 2D and 3D curves validates either model |
| Select a small set for future 3D testing | Choose contrasting controls and uncertainty-sensitive cases, not only visually favorable examples | Dense 2D sampling repairs sparse or unconverged 3D data |

A 2D in-plane gap can be bypassed by 3D flexural, torsional or thickness-related branches. A multi-domain sweep broadens coverage within its reduced model; it does not restore omitted degrees of freedom. Conversely, 3D difficulty does not make the 2D work worthless: it can support an explicitly reduced-model study if its own methods, convergence and claims are defensible.

**Practical bridge:** catalogue existing 2D cases → separate verifiable and historical-only evidence → identify one physical trend → choose matched reference/periodic/defect controls → compare equivalent 3D observables at a few representative frequencies → retain agreement and disagreement → expand only where the scientific question and numerical evidence justify it. This sequence is a proposal, not a new run queue.

## 5. Deliverables and decision rule

1. Preserve the accepted 0.1 and 1 GHz anchors and all failed convergence evidence.
2. Record one bounded next comparison with explicit ownership and resource admission.
3. If claim-relevant spatial controls pass, follow Path A toward a frequency-resolved and independently checked dataset.
4. If any layer fails, use Path B to isolate it; do not skip to a broad sweep.
5. Use 2D data for qualified screening and mechanism tests, never to relabel unresolved 3D results.
6. Deliver a final report separating what is verified, what remains uncertain, what failed, and what was never attempted.

This roadmap adds no physical result. It consolidates the current repository evidence and a prospective research decision process; the existing implementation plan and live handoff retain operational ownership.
