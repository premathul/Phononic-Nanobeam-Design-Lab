# Straight-device research status

Author: **Athul Prem** | Project/file tag: **`_AP_pnc`**

[Research documentation](README.md) · [Project home](../README.md) · [Figures and plot data](../figures/README.md)

The configurable 3D solver, workbench and saved-data reader are delivered. The current research result is a set of finite, power-normalized device responses and controlled mesh comparisons. Their quantitative use is limited by the observable and numerical family below.

## What the saved comparisons support

Here `h` is the target in-plane size in nm on the archived global meshes, and `z` counts quadratic finite-element layers through the physical thickness. Regional meshes use separate lead and central controls. The archived and regional models use the same hole geometry and original physical port planes. Read comparisons within their own mesh families.

| Observable and conditions | Recorded outcome | Supported use |
| --- | --- | --- |
| Original equations and flux for the archived 200 nm h80/z6 device at 0.1, 0.8875 and 1 GHz | The saved left-incident points pass the recorded equation, physical power and coherent checks. [Pinned configurations, summaries and tables](../data/README.md#portable-saved-device-data) | Inspect and reproduce the recorded finite responses. The left-incident check covers the saved propagating channels. |
| Total per-drive transmission and reflection on the 200 nm device | Both regional 1 GHz panels pass the 0.01 absolute power-increment limit, equal to one percentage point of incident power. Low-band h100/h80 and z4/z6 controls also pass their recorded finite checks. [Regional panels](../figures/README.md#figure-2--1-ghz-regional-transmission-sensitivity) · [Low-band comparisons](../verification/README.md#archived-global-meshes) | Report finite increments for the stated pairs. Small absolute changes can still be large relative changes when the baseline power is small. |
| Original-plane complex response on the 200 nm device | Archived 0.1 GHz in-plane and thickness increments pass 0.01. Archived 0.8875 and 1 GHz comparisons and the regional 1 GHz lead and thickness-layer comparisons exceed 0.01. [Archived comparisons](../verification/README.md#archived-global-meshes) · [Regional comparisons](../verification/README.md#regional-meshes) | State the complex sensitivity, including phase. A quantitative phase claim requires stable matched complex coefficients and a qualified uniform-guide phase reference. |
| Selected reciprocal reflected conversion at 887.5 MHz on h100/z4, h80/z4 and h80/z6 | Both directions satisfy the suppression target against the same mesh at 1 GHz. The 10% finite-change target fails on both tested refinement axes, using each comparison's positive finer-power denominator. [Reciprocity figure and data](../figures/README.md#figure-3--8875-mhz-reflected-mode-conversion) | Report same-mesh reciprocity and finite suppression, with the directed powers and denominators. The absolute conversion and the origin of its mesh sensitivity remain unresolved. |
| Spectral claims | The saved points are sparse. The 56.25 MHz channel-count bracket and frequency-local mode identities constrain those samples. [Port inventory and bracket](../verification/README.md#channel-and-field-diagnostics) | Report sampled frequencies and counts. Continuous branches, unique cutoffs, linewidth, Q, a complete band gap and an attenuation floor require dedicated spatial and frequency evidence. |

## Thin-device research dataset

The 100 nm, 26-hole device has nine direct responses at 995, 997.5 and 1000 MHz on three fixed-facet meshes. The original equations and independent flux checks pass. Drives 3 and 5 exceed the one-percentage-point power limit on all six adjacent-mesh comparisons, and every drive exceeds the 0.01 complex limit on at least one comparison. At 1 GHz, the finer four-case comparison shows that refining the exterior and center both affect drives 3 and 5. [Nine-point record](../verification/thin_mesh_frequency_AP_pnc.json) · [Four-case regional record](../verification/finer_taper_patch_response_1GHz_AP_pnc.json)

The endpoint projection agrees with the held-out midpoint on the tested response increments, while every projected solution exceeds the native equation-residual limit. Use the direct points for reported device response. Paired 100/200 nm comparisons and tests that vary hole faceting or thickness layers remain finite controls. [Projection receipts](../verification/snapshot_midpoint_AP_pnc) · [Additional regional and thickness evidence](../verification/README.md#thin-device-and-paired-thickness-controls)

## Evidence required for the next claim

The immediate unresolved power question is whether the 100 nm device's drive 3 and 5 transmission increments stabilize as spatial resolution is increased. Use the completed regional comparisons to choose one discriminating change, with the remaining mesh axes, physical geometry, modes and original reference planes fixed as applicable. Keep the existing power and complex gates. A quantitative weak-conversion or phase claim needs its own comparison using the corresponding threshold above. The archived 887.5 MHz lead-interior proposal is a separate research question under discussion.

For each claim, record its observable, configuration and mesh family, matched mode identities and original planes, comparison conditions, stated threshold and denominator, measured outcome and uncertainty limit. Retain original equation and independent flux evidence, genuine failures and the exact source and data identities. Establish a stable trend on at least three systematic resolutions before extending a claim over a short frequency interval. Physical predictions for a cryogenic sample also require applicable measured dimensions and material properties and independent experiment. [Active sequence and stop conditions](IMPLEMENTATION_PLAN_AP_pnc.md) · [Validation and uncertainty contract](PHYSICS_SPECIFICATION_AP_pnc.md#validation-and-uncertainty-contract)
