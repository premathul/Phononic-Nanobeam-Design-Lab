"""Use pattern-symmetric ordering for an unchanged native sparse direct solve. Athul Prem | _AP_pnc."""
from scipy.sparse.linalg import splu
from tools import map_scattering_AP_pnc as native

def factor(operator):
    return splu(operator,permc_spec='MMD_AT_PLUS_A')

native.splu=factor
raise SystemExit(native.main())
