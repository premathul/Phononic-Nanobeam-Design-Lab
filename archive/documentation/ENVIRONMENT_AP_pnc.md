# Environment

## Documented latest runner

Required imports: NumPy, SciPy, Gmsh, Matplotlib and openpyxl. QTCAD is not used, despite the name of one existing local interpreter. pandas, meshio, scikit-fem, pytest and Jupyter are not required by the documented latest-run workflow.

Saved mechanics environment and current handoff smoke test:

- Python 3.13.11, conda-forge, x86_64 on macOS.
- NumPy 2.2.6.
- SciPy 1.15.2.
- Gmsh 4.15.0.
- Matplotlib 3.10.8.
- Workbook reader: openpyxl 3.1.2 in a separate existing interpreter.

The new runner defaults to using its own interpreter for openpyxl; `--excel-python` supports the original split environment. No global environment or installed package was changed for the handoff.

`requirements.txt` pins these top-level package versions. It is not an exact lock of native libraries, BLAS, OS or transitive dependencies. Save your own Python, OS, architecture and package metadata with each new run. The runner records its core environment under `logs/environment.json`. No bitwise identity across systems is promised.

## Platform notes

Use a fresh virtual environment. Confirm Python/pip refer to that environment. Gmsh's Python package needs its platform shared library; on Linux missing libGLU is a common system dependency. On macOS, do not mix arm64 and x86_64 binaries in one environment. Follow Gmsh's official installation guidance if an import fails. Do not replace the solver or alter material constants to make a dependency error disappear.

The original project also contains a scikit-fem prototype and older tests. Only if investigating that older scaffold, install `requirements-legacy.txt` and use a separate copy. The original requirements incorrectly included `python==3.11` as a pip package; it is retained only under provenance, not in the installable requirements.

## Storage and computation

The archived latest data include meshes and selected field/eigenmode arrays. New output directories can grow substantially with repeated full runs. Do not copy environments into the repo. Start with the smoke gate, inspect memory/storage, then select the relevant stage. A full-device fine mesh contains about 202,000 displacement degrees of freedom before eigensolver work. This package does not impose a verified RAM minimum.
