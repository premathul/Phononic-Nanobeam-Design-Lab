# Execution environment

Author: Athul Prem | _AP_pnc

The active 3D solver uses Python 3.12, NumPy, SciPy, scikit-fem, Gmsh, Matplotlib and meshio. Install `requirements_3d_AP_pnc.txt` in a dedicated virtual environment; `requirements_ui_AP_pnc.txt` adds the optional PySide6 Essentials workbench. Follow the current README for launch and test commands. On this Windows workspace use `.venv/Scripts/python.exe`.

Each scientific result retains its original Python/platform/package and source identities. A working current environment does not authorize rewriting an old result's provenance. Linux solve/export and original Windows assembly records stay separate. The app-update virtual-environment base-path repair is recorded in the current handoff; do not reinstall packages or repeat accepted simulations merely because an application was updated.

The saved-device browser uses only the Python standard library. It reads compact pinned JSON/CSV sources, links retained field arrays, and never launches a solver. A fresh Git clone contains portable evidence but not the large workspace archive.

Memory-heavy work follows the Lawrence runbook and current handoff. Measure actual resources, use bounded allocations/checkpoints, and keep large results and environments out of Git. No automatic GPU/cloud campaign or environment reset is part of setup.

`requirements.txt`, `run_AP_pnc.py` and `verify_AP_pnc.py` serve the historical lightweight archive audit, not the active 3D environment. The unchanged [historical environment note](../archive/documentation/ENVIRONMENT_AP_pnc.md) describes the old macOS/2D context; its missing sources are not repaired by installing packages.
