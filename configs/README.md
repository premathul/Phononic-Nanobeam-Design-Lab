# Study configuration

Author: **Athul Prem** | Project/file tag: **`_AP_pnc`**

[Project home](../README.md) · [Geometry and saved inputs](../data/README.md) · [Run and resume tools](../tools/README.md)

Pass a preset to [the simulator](../simulate_AP_pnc.py) with `--config`, or open it in the desktop workbench. Each study records geometry, material and crystal orientation, boundary and analysis, body and port resolution, and resource limits. Lengths in the input are in nm and the solver resolves them to SI.

| Study | Configuration | Intended use |
| --- | --- | --- |
| Straight cantilever | [Modes](straight_beam_AP_pnc.json) | Small homogeneous clamped-free eigenmode example |
| Straight uniform guide | [Matched-port scattering](straight_scattering_AP_pnc.json) | Small unperforated guide at 0.3 GHz |
| Two-hole beam, unstructured | [Basic scattering](perforated_beam_AP_pnc.json) and [fine body phase reference](perforated_phase_reference_AP_pnc.json) | Perforated example and finer body mesh for finite phase comparisons |
| Two-hole beam, layered | [Layered scattering](perforated_layered_AP_pnc.json) and [refined layered reference](perforated_layered_reference_AP_pnc.json) | Independent in-plane, thickness and faceting controls on a small beam |
| Full 26-hole device, 200 nm thick | [Global-mesh pilot](historical_26hole_AP_pnc.json) and [coarse regional preset](historical_26hole_regional_AP_pnc.json) | Exploratory 1 GHz inputs. Their meshes differ from the saved h80/z6 reference. |
| Full 26-hole device, 100 nm thick | [Regional thickness input](historical_26hole_regional_100nm_AP_pnc.json) | 1 GHz, two quadratic layers, 300/150 nm central/lead target |
| Thin-device thickness-layer controls | [168.75 nm ceiling](thin100_central168p75_z3_AP_pnc.json) and [126.5625 nm ceiling](thin100_central126p5625_z3_AP_pnc.json) | Three-layer regional inputs for the 100 nm device at 1 GHz |

The [saved-device dataset manifest](saved_device_dataset_AP_pnc.json) pins the archived h80/z6 point files and control hashes for the [portable browser](../data/README.md#portable-saved-device-data). Original solver configurations can be downloaded from the saved results. The dataset manifest describes their provenance and is separate from those configurations.

## Mesh controls

For a layered study, `mesh.mesh_size_nm` sets the parent in-plane target. With **Independent lead size**, `mesh.lead_mesh_size_nm` separately sizes the leads. `mesh.patterned_x_interval_nm: [start, end]` places the regional cuts in solid gaps outside the holes. CLI equivalents are `--lead-mesh-size-nm` and `--patterned-x-interval-nm START END`. The original port planes remain at the beam ends.

The optional `mesh.central_max_edge_nm` refines the central interior of a fixed regional parent. Use **Central edge target (nm)** in the workbench or `--central-max-edge-nm` in the simulator. A blank or zero target in the workbench disables nested refinement. Hold the parent size, lead and port settings, cuts, hole faceting and thickness layers fixed for a nested comparison. The refinement preserves lead cells, cut traces and polygonal hole rims. Near a cut, a larger local edge can be required to retain the fixed trace. The target must be at least the longest rim edge.

```powershell
.\.venv\Scripts\python.exe simulate_AP_pnc.py --config configs/perforated_layered_AP_pnc.json --mesh-size-nm 180 --lead-mesh-size-nm 150 --patterned-x-interval-nm 300 1500 --central-max-edge-nm 90 --mesh-only
```

On a full-device study, check the actual body and port degrees of freedom and measured resource requirements before solving. A DOF cap applies after mesh generation. Through-thickness layers change the compatible port section. Physical thickness, layer count and hole faceting are separate comparison inputs. See the [model and validation rules](../docs/PHYSICS_SPECIFICATION_AP_pnc.md).

## Research templates

The [campaign template](campaign_template_AP_pnc.json) and [material registry template](material_registry_template_AP_pnc.json) describe proposed study design, source needs and unresolved properties. Execution is disabled for both templates. Use an active preset for the solver.
