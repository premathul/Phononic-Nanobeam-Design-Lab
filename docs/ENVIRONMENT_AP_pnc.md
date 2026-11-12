# Execution environment and dependencies

Author: **Athul Prem** | Project/file tag: **`_AP_pnc`**

## Current lightweight checks

The active source and tests use Python's standard library and NumPy. The current `requirements.txt` deliberately describes that scope. Use a virtual environment appropriate to the host and run commands from the repository root. Python 3.11 or newer is the documented target.

```bash
python -m pip install -r requirements.txt
python tools/audit_repository_AP_pnc.py
python verify_AP_pnc.py --fields --safety
python -m unittest discover -s tests -p 'test*_AP_pnc.py' -v
```

The audit can run without a FEM installation. NPZ field checks and the analytic utility require NumPy. No archived script should be executed directly from its renamed location: its internal imports and historical paths are retained unchanged for evidence. The active `run_AP_pnc.py` performs a read-only readiness check and deliberately blocks real execution, including after recovery until the adapter has been reviewed/enabled.

## Historical FEM environment

The archived environment document reports Python 3.13.11 on x86_64 macOS, NumPy 2.2.6, SciPy 1.15.2, Gmsh 4.15.0, Matplotlib 3.10.8 and an openpyxl 3.1.2 workbook reader. These are historical reports, not a lock file tested on every platform. The full expected original dependency file is missing; installing these packages does not restore missing numerical code.

Once sources are recovered, make a separate environment, check native Gmsh/shared-library compatibility and record Python/OS/architecture/BLAS plus all installed versions. Do not mix incompatible architecture binaries. Follow the selected solver's official installation guidance. Preserve an exact source revision and package lock/export with each newly verified run. See the unchanged [historical environment note](../archive/documentation/ENVIRONMENT_AP_pnc.md) for its original context.

## The planned 3D implementation

No FEM backend has been selected or installed by the folder reorganization. Choose a backend based on 3D anisotropy, eigenproblems, port modes, sparse solves, energy flux and available hardware. Document the choice and validate small benchmarks before scaling. Solver scratch, temporary meshes and virtual environments should not be committed.

Large campaigns need a measured pilot and explicit RAM/disk/time limits. The template leaves those budgets unresolved and execution disabled. Report memory per factorization, per-mode solve time, field size and expected total storage. Checkpoint and stop cleanly if a limit is approached. A frequency-count cap alone is not a memory budget.
