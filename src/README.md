# Simulator source

Author: **Athul Prem** | Project/file tag: **`_AP_pnc`**

[Project home](../README.md) · [Configuration guide](../configs/README.md) · [Running and analysis tools](../tools/README.md)

The active 3D entry point is [simulate_AP_pnc.py](../simulate_AP_pnc.py), which uses [src/phononic](phononic). It supports finite-body modes, periodic guided waves and matched-port elastic scattering for straight beams with optional CSV-defined through-holes.

| Source | Responsibility |
| --- | --- |
| [Workflow](phononic/workflow_AP_pnc.py) | Resolve inputs to SI, prepare studies and record run provenance |
| [Materials](phononic/material_AP_pnc.py), [elasticity](phononic/elasticity_AP_pnc.py) and [layered mesh](phononic/layered_AP_pnc.py) | Rotate the cubic stiffness into the laboratory frame, build FEM operators and control in-plane and thickness resolution |
| [Lead modes](phononic/lead_modes_AP_pnc.py), [port modes](phononic/port_modes_AP_pnc.py) and [scattering](phononic/scattering_AP_pnc.py) | Solve the lead section, normalize physical flux and couple the finite body to matched ports |
| [Checkpoint solver](phononic/resumable_scattering_AP_pnc.py) and [reports](phononic/report_AP_pnc.py) | Preserve point state, replay original-equation checks and export scientific results |
| [Desktop workbench](../workbench_AP_pnc.py) and [worker](../tools/workbench_worker_AP_pnc.py) | Edit, preflight, run or resume bounded studies and open existing results |

The [lightweight geometry and chain tool](nanobeam_AP_pnc.py) checks hole spacing and evaluates an illustrative alternating mass-spring chain. Try `python src/nanobeam_AP_pnc.py geometry data/examples/half_array_AP_pnc.csv --beam-width-nm 600` or `python src/nanobeam_AP_pnc.py chain`. The archived 2D workflow is indexed separately in the [historical archive](../archive/README.md).

For numerical assumptions or verification scope, see the [physics specification](../docs/PHYSICS_SPECIFICATION_AP_pnc.md) and the [test map](../tests/README.md).
