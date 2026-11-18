# Straight-device development plan

Author: Athul Prem | _AP_pnc

## Goal and current decision

Deliver a usable configurable 3D elastic-wave simulator and a reproducible straight 26-hole device response dataset for independent methods verification. Saved-result access is complete; the immediate deliverable is the bounded compute/resume workflow through the existing workbench. A completed finite solve, a passing spatial increment, a converged spectrum and experimental validation are different achievements.

The user's one-time assessment on 2026-11-18 found a delivery imbalance. The solver, Qt workbench, recovery and accepted HPC responses are real progress. Recent PRs #40–49 concentrated on separate diagnostics and their surrounding publication/verification machinery, while the 0.1, 0.8875 and 1 GHz results remained disconnected from the user workflow. The 193246-byte handoff, 61258-byte plan and repeated historical narratives increased recovery cost. Necessary original-equation, flux and claim-specific convergence checks remain; another diagnostic does not automatically become the next milestone.

This assessment is complete and is not a recurring review gate. The user also requested removal of useless history and junk. Keep current instructions concise, use existing evidence records for history, and retain original data, unique failures, required checkpoints and reproducible code. Remove obsolete empty templates, duplicate plans and rebuildable caches. Do not create a replacement pile of cleanup reports.

## Delivery sequence and stop conditions

1. **Completed: saved-device browser ([PR #50](https://github.com/premathul/Phononic-Nanobeam-Design-Lab/pull/50)).** Reuse the unified accepted h80 z6 0.1/0.8875/1 GHz view, 14 drive rows, 132 native coefficients and original field/configuration links. Its acceptance and cleanup are complete; do not repeat them.
2. **Current: core workbench frequency workflow.** Connect the existing native frequency-map implementation to the desktop/headless worker: configured device → existing mesh preflight → explicit increasing frequency list → bounded point calculation → stop at a point boundary → resume exact saved inputs → open existing map. Default to one new point per invocation. The 26-hole preset must retain its hole table/placement, dimensions and resource caps. Reuse native checkpoints, map CSV/report and numerical gates; do not add a parallel orchestration or reporting framework.
3. **Acceptance and stop condition.** A portable tiny two-point public-worker case runs one point, resumes only the unfinished point, and preserves the accepted point/shared work. Check 26-hole configuration transfer without a large solve, positive finite frequency/budget validation, busy/stop behavior, failed/pending point status, and noncomputing saved-map opening. Point-boundary stop finishes the current calculation; it is not immediate cancellation or in-point solver recovery. Opening an old map must not restore arrays or require the current solver identity; resuming does require exact native checkpoint/source/environment identity. Imported read-only maps cannot resume. Run the focused checks and one required integrated/CI milestone, then deliver and stop expanding it.
4. **Next scientific decision.** Use the unified dataset and Athul's review to choose one observable and a control that could change its interpretation. The current unresolved question is regional sensitivity of the small reflected conversion. Keep the bounded lead-only proposal below for research discussion. Internal-plane extraction and more parity/energy/phase audits remain deferred unless they answer a concrete decision. Core workflow development does not authorize a new full-device calculation.
5. **Later, conditional.** Broader frequency coverage and independent spatial controls follow selected claims and measured resources. A full 121-point survey, bends/rings, temperature/material campaigns, intrinsic loss/Q, thermal transport and GPU development are not prerequisites for this usable workflow.

Routine authorized development continues automatically in this chat without asking whether to continue. Each milestone must produce a usable capability, decision-relevant evidence, or fix a demonstrated blocker. Reuse accepted results; do not create a new assessment/report cycle at every activation.

## Evidence to reuse

| Result | What it establishes | Remaining limit |
| --- | --- | --- |
| Configurable 3D modes, Bloch and matched-port scattering; Qt workbench and checkpoint recovery | Usable numerical workflow and small reference checks | Arbitrary configurations do not inherit benchmark acceptance |
| h80 z6 at 0.1 GHz: four left drives | Original equations, physical/coherent flux and field mapping pass | Finite h100/h80 and z4/z6 checks pass; full spatial/frequency convergence remains unestablished |
| h80 z6 at 0.8875 GHz: five left drives | Accepted equations/flux/fields; conversion power about 7.9691e-7 versus 0.00108076 at 1 GHz | Weak conversion fails the independent z4/z6 and h100/h80 sensitivity targets |
| h80 z6 at 1 GHz: five left drives | Accepted finite response and saved fields | All three recorded full-device complex-response comparisons fail 0.01 |
| Saved regional energy and phase-only controls | Regional fraction increment 0.00192099 passes 0.02; magnitude residual bound 0.01727118 excludes diagonal reflectionless/unmixed phase-only changes at 0.01 | Neither result locates the error or proves pattern dominance |

Use [low-band evidence](../verification/low_band_device_AP_pnc.json), [low-band thickness](../verification/low_band_thickness_AP_pnc.json), [low-band in-plane](../verification/low_band_inplane_AP_pnc.json), [887.5 MHz point](../verification/device_frequency_8875_AP_pnc.json), [887.5 MHz thickness](../verification/device_frequency_thickness_8875_AP_pnc.json), [887.5 MHz in-plane](../verification/device_frequency_inplane_8875_AP_pnc.json), [1 GHz thickness](../verification/device_thickness_AP_pnc.json), [regional energy](../verification/field_energy_8875_AP_pnc.json) and [phase-only bound](../verification/lead_phase_8875_AP_pnc.json). The remaining ledgers and raw outputs stay in place.

Root numbers are frequency-local. The four/five-channel change prevents inventing a common low-band branch index. The accepted 0.8875/1 GHz endpoint correspondence is a finite signed field match, not continuous branch identity. Sparse points must not be drawn as a converged spectrum or used to infer a gap, resonance linewidth, Q or numerical floor. Keep native reference planes and phases.

## Bounded lead-only proposal

[Athul's PR #48 comment](https://github.com/premathul/Phononic-Nanobeam-Design-Lab/pull/48#issuecomment-6969077996) requests the saved-data limits and a bounded control proposal for discussion before another solve. The proposed comparison remains unqueued; this comment does not itself authorize another solve or refinement.

Perturb only strict lead-interior cells of accepted h100 z4 at 0.8875 GHz toward h80. Freeze every tetrahedron intersecting the patterned interval [1.5, 15.15] micrometres, including crossing cells, its P2 coordinates/connectivity and actual facet frontier. Preserve material, 200 nm thickness, holes, original end traces, frequency-specific SAFE and radiation conditions. Preflight must demonstrate a conforming frontier and unchanged protected local element operators; shared assembled interface rows may change through lead contributions. If exact isolation is impossible, redesign before solving.

For each directed weak conversion separately, predeclare abs(Pcontrol - Ph100)/Ph100 <= 0.10 using the actual positive h100 power, without a denominator floor. Retain original-equation and physical/coherent-power gates and separate whole-block power/complex 0.01 checks. Two drives answer only the selected pair; five are needed for the full left block. A failure establishes sensitivity to that finite lead perturbation; a pass does not prove pattern dominance or continuum convergence. Actual DOFs, sparsity, factor-index exposure and fresh memory/CPU/disk admission precede execution authorization. No additive regional error decomposition or uniform-guide phase subtraction is justified.

An internal-plane traveling-wave extractor is deferred. Total displacement alone contains forward/backward and evanescent interference. If a specific decision later needs it, validate P2 trace/traction, both directional bases, conditioning, reconstruction and flux on small known fields first.

## Implementation and execution

The active public solver is `simulate_AP_pnc.py` with `src/phononic/`; `run_AP_pnc.py` is a legacy archive-readiness checker. The existing scikit-fem/SciPy/Gmsh/meshio components, optional layered mesh, matched SAFE ports and Qt/headless worker remain the implementation base. Use [physics specification](PHYSICS_SPECIFICATION_AP_pnc.md), [reference notes](REFERENCES_AP_pnc.md), [Lawrence runbook](LAWRENCE_HPC_AP_pnc.md) and the [current handoff](AI_HANDOFF_AP_pnc.md) for their respective details.

Root owns connections, archive execution, scheduler and integration. Independent implementation and verification have distinct file ownership. Before expensive work, inspect actual workers/checkpoints and measure resources; resume matching accepted inputs rather than restarting. Preserve failed numerical evidence and exact original arrays. Run meaningful affected tests and one required integrated milestone check; CI retains the existing physical benchmarks. Never loosen scientific gates.

The previous detailed plan is retained in Git at [fea418a](https://github.com/premathul/Phononic-Nanobeam-Design-Lab/blob/fea418a3ded1d622f80125262864ea15b17948b2/docs/IMPLEMENTATION_PLAN_AP_pnc.md). It is historical context, not another active queue. No reset, unsolicited cloud job, collaborator message or public release is authorized by this plan.
