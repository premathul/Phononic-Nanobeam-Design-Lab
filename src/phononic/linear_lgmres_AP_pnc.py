"""Augmented sparse harmonic solver. Author: Athul Prem | _AP_pnc."""

from math import isfinite
from numbers import Real
from time import perf_counter

import numpy as np
from scipy.linalg import get_blas_funcs
from scipy.sparse import issparse
from scipy.sparse.linalg import LinearOperator, lgmres, spilu


def solve_ilu_lgmres(A, rhs: np.ndarray, report: dict | None = None, *, fill_factor: float = 10.0) -> np.ndarray:
    """Solve original complex CSC equations with one shared incomplete LU factor.

    Each right-hand side starts its own LGMRES augmentation. The report records
    the original-operator residual at every outer callback and at the returned
    solution, including the last attempted channel on failure.
    """
    if not issparse(A) or A.format != "csc" or A.shape[0] != A.shape[1] or not A.shape[0] or A.dtype.kind != "c":
        raise ValueError("ILU-LGMRES requires a nonempty square complex CSC matrix.")
    rhs = np.asarray(rhs)
    if rhs.ndim != 2 or rhs.shape[0] != A.shape[0] or not rhs.shape[1]:
        raise ValueError("ILU-LGMRES requires a nonempty right-hand-side matrix with one row per equation.")
    if not np.isfinite(A.data).all() or not np.isfinite(rhs).all():
        raise ValueError("ILU-LGMRES operator and right-hand sides must be finite.")
    if (not isinstance(fill_factor, Real) or isinstance(fill_factor, (bool, np.bool_))
            or fill_factor < 1 or not isfinite(fill_factor)):
        raise ValueError("ILU-LGMRES fill_factor must be a finite real number greater than or equal to one.")

    ilu_settings = {"drop_tol": 1e-4, "fill_factor": float(fill_factor), "permc_spec": "COLAMD"}
    lgmres_settings = {"rtol": 1e-10, "atol": 0, "inner_m": 50, "outer_k": 3, "maxiter": 30}
    limit = 1e-9
    diagnostics = report if report is not None else {}
    diagnostics.update(method="ilu_lgmres", preconditioner=ilu_settings.copy(), lgmres=lgmres_settings.copy(),
                       relative_residual_limit=limit, backward_residual_limit=limit,
                       operator_nonzeros=int(A.nnz),
                       operator_storage_bytes=sum(array.nbytes for array in (A.data, A.indices, A.indptr)),
                       channels=[])
    # SuperLU may canonicalize CSC indices in place. Preserve the caller's matrix.
    if not A.has_canonical_format:
        A = A.copy()
        A.sum_duplicates()
    operator_norm = float(np.asarray(abs(A).sum(axis=1)).max())
    started = perf_counter()
    try:
        factor = spilu(A, **ilu_settings)
    finally:
        diagnostics["preconditioner"]["elapsed_seconds"] = perf_counter() - started
    diagnostics["preconditioner"]["factor_nonzeros"] = int(factor.nnz)
    preconditioner = LinearOperator(A.shape, matvec=factor.solve, dtype=A.dtype)

    solution = np.empty(rhs.shape, dtype=A.dtype)
    for column, b in enumerate(rhs.T):
        started = perf_counter()
        channel = {"column": column, "raw_info": None, "info": None, "iterations": 0,
                   "relative_residual_history": [], "relative_residual": None,
                   "relative_backward_residual": None, "terminal_confirmation": None}
        diagnostics["channels"].append(channel)
        nrm2 = get_blas_funcs("nrm2", (b, A.data))
        b_norm = nrm2(b)
        b_max = float(np.max(abs(b)))

        def record_residual(iterate):
            residual_norm = nrm2(A @ iterate - b)
            relative = float(residual_norm / b_norm) if b_norm else float(residual_norm)
            channel["relative_residual_history"].append(relative if isfinite(relative) else None)
            channel["iterations"] += 1

        try:
            x, info = lgmres(A, b, M=preconditioner, callback=record_residual, **lgmres_settings)
            channel.update(raw_info=int(info), info=int(info))
            residual = A @ x - b
            residual_norm = nrm2(residual)
            relative = float(residual_norm / b_norm) if b_norm else float(residual_norm)
            scale = operator_norm * float(np.max(abs(x))) + b_max
            residual_max = float(np.max(abs(residual)))
            backward = residual_max / scale if scale else residual_max
            channel.update(relative_residual=relative if isfinite(relative) else None,
                           relative_backward_residual=backward if isfinite(backward) else None)
            residual_ok = (np.isfinite(x).all() and isfinite(scale)
                           and isfinite(relative) and relative <= limit and isfinite(backward) and backward <= limit)

            # SciPy checks before each outer update. The final update can meet
            # tolerance while the raw return still reports maxiter.
            if info == lgmres_settings["maxiter"] and residual_ok and relative <= lgmres_settings["rtol"]:
                confirmation = {"info": None, "callbacks": 0, "accepted": False}
                channel["terminal_confirmation"] = confirmation
                same_start = False

                def check_initial(iterate):
                    nonlocal same_start
                    confirmation["callbacks"] += 1
                    same_start = confirmation["callbacks"] == 1 and np.array_equal(iterate, x)

                confirmed, confirm_info = lgmres(A, b, x0=x.copy(), M=None, maxiter=1,
                                                rtol=lgmres_settings["rtol"], atol=0, callback=check_initial)
                accepted = confirm_info == 0 and same_start and np.array_equal(confirmed, x)
                confirmation.update(info=int(confirm_info), accepted=bool(accepted))
                if accepted:
                    channel["info"] = int(confirm_info)

            if channel["info"] != 0 or not residual_ok:
                raise RuntimeError(f"ILU-LGMRES failed for RHS {column + 1} "
                                   f"(info={info}, relative residual {relative:.3e}, backward residual {backward:.3e}).")
            solution[:, column] = x
        finally:
            channel["elapsed_seconds"] = perf_counter() - started
    return solution
