# Input data and geometry

Author: **Athul Prem** | Project/file tag: **`_AP_pnc`**

The original 26-hole workbook and derived CSV are in `geometry/`; historical CAD is in `geometry/cad/`, and original meshes in `meshes/legacy/`. Their bytes and units are unchanged. `examples/half_array_AP_pnc.csv` contains only eight holes for the lightweight CLI tests. It is not the full research geometry. The full CSV uses `position_nm`; the small CLI example uses `x_nm`. Use an explicit schema conversion before passing the full CSV to a different interface. Preserve source files and store generated geometry with configuration/hash provenance.
