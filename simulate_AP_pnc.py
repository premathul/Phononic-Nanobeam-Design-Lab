#!/usr/bin/env python3
"""Run a configured 3D beam study. Author: Athul Prem | _AP_pnc."""
from pathlib import Path
import os
import sys

# Small local cases do not benefit from competing BLAS thread pools.
for variable in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ[variable] = "1"
sys.path.insert(0, str(Path(__file__).resolve().parent / "src"))

if __name__ == "__main__":
    from phononic.workflow_AP_pnc import main
    raise SystemExit(main())
