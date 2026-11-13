"""Bounded sparse harmonic solver. Author: Athul Prem | _AP_pnc."""

from math import isfinite
from numbers import Real
from time import perf_counter

import numpy as np
from scipy.sparse import issparse
from scipy.sparse.linalg import LinearOperator, gmres, spilu


def solve_ilu_gmres(A, rhs: np.ndarray, report: dict | None = None, *, fill_factor: float = 10.0) -> np.ndarray:
    """Solve complex CSC equations with one incomplete LU preconditioner.

    GMRES acts on the original operator. Every column must converge and meet
    both the relative two-norm and infinity-norm backward residual limits. When supplied,
    the report records settings and diagnostics through the last attempted column.
    """
    if not issparse(A) or A.format != "csc" or A.shape[0] != A.shape[1] or A.dtype.kind != "c":
        raise ValueError("ILU-GMRES requires a square complex CSC matrix.")
    rhs = np.asarray(rhs)
    if rhs.ndim != 2 or rhs.shape[0] != A.shape[0] or not rhs.shape[1]:
        raise ValueError("ILU-GMRES requires a nonempty right-hand-side matrix with one row per equation.")
    if not np.isfinite(A.data).all() or not np.isfinite(rhs).all():
        raise ValueError("ILU-GMRES operator and right-hand sides must be finite.")
    if (not isinstance(fill_factor, Real) or isinstance(fill_factor, (bool, np.bool_))
            or fill_factor < 1 or not isfinite(fill_factor)):
        raise ValueError("ILU-GMRES fill_factor must be a finite real number greater than or equal to one.")

    preconditioner_settings = {"drop_tol": 1e-4, "fill_factor": float(fill_factor), "permc_spec": "COLAMD"}
    gmres_settings = {"rtol": 1e-10, "atol": 0, "restart": 50, "maxiter": 10, "callback_type": "pr_norm"}
    limit = 1e-9
    diagnostics = report if report is not None else {}
    diagnostics.update(method="ilu_gmres", preconditioner=preconditioner_settings.copy(),
                       gmres=gmres_settings.copy(), relative_residual_limit=limit, backward_residual_limit=limit,
                       operator_nonzeros=int(A.nnz),
                       operator_storage_bytes=sum(array.nbytes for array in (A.data, A.indices, A.indptr)),
                       channels=[])
    # Use the physical operator norm and preserve the original matrix. spilu
    # canonicalizes CSC indices in place when given a noncanonical input.
    if not A.has_canonical_format:
        A = A.copy()
        A.sum_duplicates()
    operator_norm = float(np.asarray(abs(A).sum(axis=1)).max())
    started = perf_counter()
    factor = spilu(A, **preconditioner_settings)
    diagnostics["preconditioner"].update(factor_nonzeros=int(factor.nnz), elapsed_seconds=perf_counter() - started)
    preconditioner = LinearOperator(A.shape, matvec=factor.solve, dtype=A.dtype)
    solution = np.empty(rhs.shape, dtype=A.dtype)
    for column, b in enumerate(rhs.T):
        history = []
        started = perf_counter()
        x, info = gmres(A, b, M=preconditioner, callback=history.append, **gmres_settings)
        residual = A @ x - b
        b_norm = np.linalg.norm(b)
        relative = float(np.linalg.norm(residual) / b_norm) if b_norm else float(np.linalg.norm(residual))
        scale = operator_norm * np.linalg.norm(x, ord=np.inf) + np.linalg.norm(b, ord=np.inf)
        residual_max = np.linalg.norm(residual, ord=np.inf)
        backward = float(residual_max / scale) if scale else float(residual_max)
        diagnostics["channels"].append({
            "column": column, "info": int(info), "iterations": len(history),
            "preconditioned_residual_history": [float(value) if np.isfinite(value) else None for value in history],
            "relative_residual": relative if np.isfinite(relative) else None,
            "relative_backward_residual": backward if np.isfinite(backward) else None,
            "elapsed_seconds": perf_counter() - started,
        })
        if (info != 0 or not np.isfinite(x).all() or not np.isfinite(scale)
                or not np.isfinite(relative) or relative > limit or not np.isfinite(backward) or backward > limit):
            raise RuntimeError(f"ILU-GMRES failed for RHS {column + 1} "
                               f"(info={info}, relative residual {relative:.3e}, backward residual {backward:.3e}).")
        solution[:, column] = x
    return solution
