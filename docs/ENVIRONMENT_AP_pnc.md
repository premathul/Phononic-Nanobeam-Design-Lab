# Execution environment

Author: **Athul Prem** | Project/file tag: **`_AP_pnc`**

[Research documentation](README.md) · [Simulator quickstart](../README.md#start-the-simulator) · [Configuration guide](../configs/README.md)

| Workflow | Runtime |
| --- | --- |
| [Offline saved-device browser](../data/README.md#portable-saved-device-data) | Python standard library and the pinned JSON/CSV bundle |
| Three-dimensional solver and plots | Python 3.12 with [numerical requirements](../requirements_3d_AP_pnc.txt), including NumPy, SciPy, scikit-fem, Gmsh, Matplotlib and meshio |
| Optional desktop workbench | Solver environment plus [PySide6 requirements](../requirements_ui_AP_pnc.txt) |
| Historical checks | [Lightweight requirements](../requirements.txt) with the [archive tools](../tools/README.md#historical-archive-tools) |

On the Windows workspace, use `.venv/Scripts/python.exe`. `Launch_AP_pnc.cmd` and `Launch_Workbench_AP_pnc.cmd` open the active simulator and workbench respectively. In the workbench, **Load saved study…** loads editable inputs and attaches the study's saved report. **Browse device results** opens the independent result browser. Use **Open report** to return to a report attached to your current study.

Each scientific result retains its original Python, platform, package and source identities. Linux solve/export provenance and original Windows assembly provenance are recorded separately. If an app update moves the base Python path, recover the environment before a new run and reuse accepted results. See the [current handoff](AI_HANDOFF_AP_pnc.md) for the project environment and ownership.

For memory-heavy calculations, follow the [Lawrence runbook](LAWRENCE_HPC_AP_pnc.md) and use measured resources, interactive allocations and bounded checkpoints. Keep large results and virtual environments outside Git.

The preserved [historical environment note](../archive/documentation/ENVIRONMENT_AP_pnc.md) describes the earlier macOS and 2D model. Consult [archive recovery](EVIDENCE_AND_RECOVERY_AP_pnc.md) for missing original sources.
