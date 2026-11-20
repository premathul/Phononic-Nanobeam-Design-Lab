# Geometry and saved data

Author: **Athul Prem** | Project/file tag: **`_AP_pnc`**

[Project home](../README.md) · [Figures](../figures/README.md) · [Numerical evidence](../verification/README.md)

## Geometry inputs

| File | Scope and coordinates |
| --- | --- |
| [Original 26-hole workbook](geometry/nanobeam_hole_array_AP_pnc.xlsx) | Source design for the full hole pattern |
| [Derived 26-hole CSV](geometry/hole_geometry_derived_AP_pnc.csv) | `hole_number`, `position_nm` from the pattern center, `diameter_nm` and `region`. The full-device configurations place the pattern with a +8325 nm offset in the 16650 nm beam. |
| [Eight-hole example](examples/half_array_AP_pnc.csv) | Example for the lightweight geometry tool with `x_nm`, `diameter_nm` and `region` |
| [Two-hole example](examples/two_holes_AP_pnc.csv) | Geometry input for the bounded two-hole scattering studies |

Use an explicit column conversion when moving between interfaces. Historical [CAD](geometry/cad) and [meshes](meshes/legacy) retain their original bytes and units.

## Portable saved-device data

The [compact bundle](saved_device_AP_pnc) and [checksum manifest](../configs/saved_device_dataset_AP_pnc.json) reproduce the 200 nm archived h80/z6 responses at 0.1, 0.8875 and 1 GHz from a clean clone. Its 26 files include exact configurations, summaries, modal tables, native coefficients and comparison records. The manifest also binds the already tracked 1 GHz acceptance record and geometry table. The dataset has 14 incident drives and 132 complex coefficients. Root numbers identify the modes at each frequency. Review the sampled counts, original planes and failed sensitivity gates alongside the amplitudes.

```powershell
python -B tools/browse_device_results_AP_pnc.py --open
```

Use **Browse device results** in the workbench to open the same read-only browser. Choose a frequency and incident mode, inspect its reflection and transmission, or download the corresponding coefficients, exact configuration and geometry. Copy the selection link to reopen the saved point and frequency-local root. Use **Load configuration…** in the workbench to open a downloaded JSON as an editable study. For scientific comparisons of complex amplitudes, use the signed mode matches in the [verification receipts](../verification/README.md).

Full quadratic fields, vertex visualization, original 3D reports and large native checkpoints are archive-only. The reader labels unavailable artifacts and makes existing copies accessible from the matching workspace. Historical paths inside original summaries record provenance. Full equation replay requires the corresponding native archive. [Result custody and lineage](../results/README.md)

## Thin-device research dataset

The 100 nm thin-device grid covers 995, 997.5 and 1000 MHz at central edge ceilings of 225, 168.75 and 126.5625 nm. The browser can export any drive's ten native coefficients or the full grid of 45 drive powers and 450 complex coefficients. It records reference-drive labels, the frequency-local incident roots and paths to six original CSV tables. The source configurations and the 26-hole geometry are available for download. [Nine-point receipt](../verification/thin_mesh_frequency_AP_pnc.json) · [Notebook and six native tables](../results/runs/physical_thickness_device_AP_pnc/thin100_1GHz_mesh_response_AP_pnc.ipynb)

The same view exposes the paired thickness, isolated hole-pair and finer four-case regional controls when the research data are available. [Research controls and limitations](../verification/README.md#thin-device-and-paired-thickness-controls)
