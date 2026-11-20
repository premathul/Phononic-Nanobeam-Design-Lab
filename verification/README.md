# Verification records

Author: **Athul Prem** | Project/file tag: **`_AP_pnc`**

[Research documentation](../docs/README.md) · [Research status and claim limits](../docs/RESEARCH_STATUS_AP_pnc.md) · [Figures](../figures/README.md)

A saved equation and flux pass verifies one finite response. A passing refinement increment qualifies one observable for a stated pair. Full spatial convergence, frequency convergence and experimental validation each require additional evidence. The records below retain failed gates and original reference planes.

## Small benchmarks and workflow checks

[Homogeneous cantilever](straight_beam_validation_AP_pnc.json) · [Uniform matched guide](straight_scattering_acceptance_AP_pnc.json) · [Two-hole phase refinement](two_hole_phase_refinement_AP_pnc.json) · [Layered mesh](layered_mesh_acceptance_AP_pnc.json) · [Bounded frequency map](frequency_map_AP_pnc.json)

[Tests and CI scope](../tests/README.md) distinguishes these cases from full-device results.

## Archived global meshes

| Frequency | Original finite response and comparisons |
| --- | --- |
| 0.1 GHz | [h80/z6 response](low_band_device_AP_pnc.json), [h100/h80 in-plane](low_band_inplane_AP_pnc.json), [z4/z6 thickness layers](low_band_thickness_AP_pnc.json) |
| 0.8875 GHz | [h80/z6 response and endpoint comparison](device_frequency_8875_AP_pnc.json), [h100/h80 in-plane](device_frequency_inplane_8875_AP_pnc.json), [z4/z6 thickness layers](device_frequency_thickness_8875_AP_pnc.json) |
| 1 GHz | [h80/z6 response and z4/z6 thickness layers](device_thickness_AP_pnc.json), [h60 fixed-port comparison](device_h60_AP_pnc.json), [earlier global refinements](device_convergence_followup_AP_pnc.json) |

See the [portable data manifest](../configs/saved_device_dataset_AP_pnc.json) for the sparse clean-clone dataset and exact geometry and point inputs.

## Regional meshes

The 200 nm thick 1 GHz device has separate [lead-resolution](regional_lead_1GHz_AP_pnc.json) and [thickness-layer](regional_thickness_1GHz_AP_pnc.json) records. Both retain passing finite total-power checks and failed complex increments. These comparisons use distinct numerical controls and a different mesh family from the archived h80/z6 responses.

## Thin-device and paired thickness controls

- [Thin-device nine-point mesh/frequency grid](thin_mesh_frequency_AP_pnc.json) and [endpoint projection on the held-out midpoint](snapshot_midpoint_AP_pnc).
- [First closed four-case regional comparison](taper_patch_response_1GHz_AP_pnc.json), [finer complementary comparison](finer_taper_patch_response_1GHz_AP_pnc.json), [split equation check](split_device_1GHz_AP_pnc.json), [isolated hole-pair solves](goal_patch_response_1GHz_AP_pnc.json) and [signed reflection work](regional_reflection_work_1GHz_AP_pnc.json).
- [Paired thickness midpoint responses](paired_nested_midpoint_AP_pnc.json), [signed mode matches](paired_mode_frequency_AP_pnc.json), [thin-device layer control](thin_device_layer_response_AP_pnc.json), [fixed hole-wall control](fixed_wall_response_AP_pnc.json), [full-length uniform guide](uniform_thin_full_length_1GHz_AP_pnc.json) and [port phase reference](uniform_thin_port_refinement_1GHz_AP_pnc.json).
- [Periodic cell](mirror_cell_phase_AP_pnc.json) and [eight-period finite mirror](finite_mirror_1ghz_AP_pnc.json) cover models with different cross-sections and lengths from the full device.

## Channel and field diagnostics

[Finite port inventory](port_inventory_AP_pnc.json) and [sampled channel-count interval](port_count_interval_AP_pnc.json) identify the resolved sample counts and their scope. [Saved regional field energy](field_energy_8875_AP_pnc.json) and [phase-only compatibility](lead_phase_8875_AP_pnc.json) record the 887.5 MHz diagnoses. Those diagnoses leave the spatial source of the weak-conversion sensitivity unresolved. [Publication reciprocity plot and raw complex blocks](../figures/README.md#figure-3--8875-mhz-reflected-mode-conversion)

## Dated verification archive

The [11 November 2026 handoff verification](archive/REPORT_AP_pnc.md) records the earlier 2D package and its original check scope. The [12 November 2026 reorganization verification](REORGANIZATION_VERIFICATION_AP_pnc.md) records uploaded-byte preservation, twelve tests and the original unresolved archive entries. The [current handoff](../docs/AI_HANDOFF_AP_pnc.md) gives the relevant integrated tests and CI receipts.
