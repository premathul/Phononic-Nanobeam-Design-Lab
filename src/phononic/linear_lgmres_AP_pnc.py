"""Augmented sparse harmonic solver. Author: Athul Prem | _AP_pnc."""

from collections.abc import Callable
from copy import deepcopy
from math import isfinite
from numbers import Integral, Real
from time import perf_counter

import numpy as np
from scipy.linalg import get_blas_funcs
from scipy.sparse import issparse
from scipy.sparse.linalg import LinearOperator, lgmres, spilu


def _channel_diagnostics(column: int) -> dict:
    return {"column": column, "raw_info": None, "info": None, "iterations": 0,
            "relative_residual_history": [], "relative_residual": None,
            "relative_backward_residual": None, "terminal_confirmation": None}


def _original_residual(A, x, b, nrm2, b_norm, b_max, operator_norm, limit, channel):
    residual = A @ x - b
    residual_norm = nrm2(residual)
    relative = float(residual_norm / b_norm) if b_norm else float(residual_norm)
    scale = operator_norm * float(np.max(abs(x))) + b_max
    residual_max = float(np.max(abs(residual)))
    backward = residual_max / scale if scale else residual_max
    channel.update(relative_residual=relative if isfinite(relative) else None,
                   relative_backward_residual=backward if isfinite(backward) else None)
    residual_ok = (np.isfinite(x).all() and isfinite(scale) and isfinite(b_norm)
                   and isfinite(relative) and relative <= limit and isfinite(backward) and backward <= limit)
    return relative, backward, residual_ok


def _validate_checkpoint_vectors(vectors, name, shape, dtype):
    if vectors is None:
        return {}
    if not isinstance(vectors, dict):
        raise ValueError(f"ILU-LGMRES {name} must map zero-based RHS columns to vectors.")
    validated = {}
    for column, vector in vectors.items():
        if (not isinstance(column, Integral) or isinstance(column, (bool, np.bool_))
                or not 0 <= column < shape[1]):
            raise ValueError(f"ILU-LGMRES {name} contains an invalid RHS column: {column}.")
        if (not isinstance(vector, np.ndarray) or vector.shape != (shape[0],)
                or not np.can_cast(vector.dtype, dtype, casting="safe") or not np.isfinite(vector).all()):
            raise ValueError(f"ILU-LGMRES {name} for RHS {column + 1} must be a finite vector of shape {(shape[0],)}.")
        validated[int(column)] = vector.astype(dtype, copy=False)
    return validated


def solve_ilu_lgmres(A, rhs: np.ndarray, report: dict | None = None, *, fill_factor: float = 10.0,
                    single_precision: bool = False, inner_m: int = 50, outer_k: int = 3, maxiter: int = 30,
                    accepted_solutions: dict[int, np.ndarray] | None = None,
                    initial_guesses: dict[int, np.ndarray] | None = None,
                    checkpoint_callback: Callable[[str, int, np.ndarray, dict], None] | None = None) -> np.ndarray:
    """Solve original complex CSC equations with one shared incomplete LU factor.

    Each right-hand side starts its own LGMRES augmentation. The report records
    the original-operator residual at every outer callback and at the returned
    solution, including the last attempted channel on failure. Single precision
    applies only to the ILU factorization and its inverse applications.

    Accepted vectors are rechecked against the original equations and bypass
    iteration. Initial guesses start only their specified columns. The optional
    callback synchronously receives a snapshot as (event, zero-based column,
    vector, channel diagnostics). Events are finite outer iterates, new accepted
    solutions, and the latest finite vector on numerical failure, respectively
    named "iterate", "accepted", and "terminal".
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
    if not isinstance(single_precision, (bool, np.bool_)):
        raise ValueError("ILU-LGMRES single_precision must be a boolean.")
    checkpointed = accepted_solutions is not None or initial_guesses is not None or checkpoint_callback is not None
    if checkpointed and A.dtype != np.dtype(np.complex128):
        raise ValueError("Resumable ILU-LGMRES requires a complex128 working operator.")
    if single_precision and A.dtype != np.dtype(np.complex128):
        raise ValueError("Single-precision ILU requires a complex128 working operator.")
    if single_precision and not np.can_cast(rhs.dtype, A.dtype, casting="safe"):
        raise ValueError("Single-precision ILU requires right-hand sides representable as complex128.")
    if checkpointed and not np.can_cast(rhs.dtype, A.dtype, casting="safe"):
        raise ValueError("Resumable ILU-LGMRES requires right-hand sides representable as complex128.")
    if single_precision or checkpointed:
        rhs = rhs.astype(A.dtype, copy=False)
    for name, value, minimum in (("inner_m", inner_m, 1), ("outer_k", outer_k, 0), ("maxiter", maxiter, 1)):
        if not isinstance(value, Integral) or isinstance(value, (bool, np.bool_)) or value < minimum:
            raise ValueError(f"ILU-LGMRES {name} must be an integer greater than or equal to {minimum}.")
    if checkpoint_callback is not None and not callable(checkpoint_callback):
        raise ValueError("ILU-LGMRES checkpoint_callback must be callable.")

    accepted_solutions = _validate_checkpoint_vectors(accepted_solutions, "accepted_solutions", rhs.shape, A.dtype)
    initial_guesses = _validate_checkpoint_vectors(initial_guesses, "initial_guesses", rhs.shape, A.dtype)
    if accepted_solutions.keys() & initial_guesses.keys():
        raise ValueError("ILU-LGMRES initial_guesses cannot include already accepted RHS columns.")

    ilu_settings = {"drop_tol": 1e-4, "fill_factor": float(fill_factor), "permc_spec": "COLAMD"}
    lgmres_settings = {"rtol": 1e-10, "atol": 0, "inner_m": int(inner_m), "outer_k": int(outer_k),
                       "maxiter": int(maxiter)}
    limit = 1e-9
    diagnostics = report if report is not None else {}
    factor_dtype = np.dtype(np.complex64) if single_precision else A.dtype
    diagnostics.update(method="ilu_lgmres", working_dtype=str(A.dtype), factor_dtype=str(factor_dtype),
                       preconditioner=ilu_settings.copy(), lgmres=lgmres_settings.copy(),
                       relative_residual_limit=limit, backward_residual_limit=limit,
                       operator_nonzeros=int(A.nnz),
                       operator_storage_bytes=sum(array.nbytes for array in (A.data, A.indices, A.indptr)),
                       channels=[])
    # SuperLU may canonicalize CSC indices in place. Preserve the caller's matrix.
    if not A.has_canonical_format:
        A = A.copy()
        A.sum_duplicates()
    operator_norm = float(np.asarray(abs(A).sum(axis=1)).max())
    solution = np.empty(rhs.shape, dtype=A.dtype)
    reused_channels = {}
    for column, vector in sorted(accepted_solutions.items()):
        started = perf_counter()
        b = rhs[:, column]
        nrm2 = get_blas_funcs("nrm2", (b, A.data))
        channel = _channel_diagnostics(column)
        relative, backward, residual_ok = _original_residual(
            A, vector, b, nrm2, nrm2(b), float(np.max(abs(b))), operator_norm, limit, channel,
        )
        if not residual_ok:
            raise ValueError(f"Saved ILU-LGMRES solution for RHS {column + 1} failed the original residual gates "
                             f"(relative residual {relative:.3e}, backward residual {backward:.3e}).")
        channel.update(info=0, reused=True, elapsed_seconds=perf_counter() - started)
        reused_channels[column] = channel
        solution[:, column] = vector

    if len(reused_channels) == rhs.shape[1]:
        diagnostics["channels"].extend(reused_channels.values())
        diagnostics["preconditioner"].update(skipped=True, elapsed_seconds=0.0, factor_nonzeros=0)
        return solution

    started = perf_counter()
    try:
        if single_precision:
            with np.errstate(over="ignore"):
                factor_input = A.astype(factor_dtype)
            if not np.isfinite(factor_input.data).all():
                raise ValueError("The operator cannot be represented as a finite complex64 preconditioner input.")
            factor = spilu(factor_input, **ilu_settings)
            del factor_input
        else:
            factor = spilu(A, **ilu_settings)
    finally:
        diagnostics["preconditioner"]["elapsed_seconds"] = perf_counter() - started
    diagnostics["preconditioner"]["factor_nonzeros"] = int(factor.nnz)
    if single_precision:
        def apply_factor(vector):
            return factor.solve(vector.astype(factor_dtype)).astype(A.dtype)

        preconditioner = LinearOperator(A.shape, matvec=apply_factor, dtype=A.dtype)
    else:
        preconditioner = LinearOperator(A.shape, matvec=factor.solve, dtype=A.dtype)

    for column, b in enumerate(rhs.T):
        if column in reused_channels:
            diagnostics["channels"].append(reused_channels[column])
            continue
        started = perf_counter()
        channel = _channel_diagnostics(column)
        x0 = initial_guesses.get(column)
        if x0 is not None:
            channel["warm_started"] = True
            x0 = x0.copy()
        diagnostics["channels"].append(channel)
        nrm2 = get_blas_funcs("nrm2", (b, A.data))
        b_norm = nrm2(b)
        b_max = float(np.max(abs(b)))
        latest_finite = x0.copy() if checkpoint_callback is not None and x0 is not None else None
        publishing = False
        latest_checked = False

        def checkpoint(event, vector):
            nonlocal publishing
            publishing = True
            checkpoint_callback(event, column, vector.copy(), deepcopy(channel))
            publishing = False

        def record_residual(iterate):
            nonlocal latest_finite
            residual_norm = nrm2(A @ iterate - b)
            relative = float(residual_norm / b_norm) if b_norm else float(residual_norm)
            channel["relative_residual_history"].append(relative if isfinite(relative) else None)
            channel["iterations"] += 1
            if checkpoint_callback is not None and np.isfinite(iterate).all():
                latest_finite = iterate.copy()
                checkpoint("iterate", latest_finite)

        try:
            x, info = lgmres(A, b, x0=x0, M=preconditioner, callback=record_residual, **lgmres_settings)
            channel.update(raw_info=int(info), info=int(info))
            terminal_is_finite = checkpoint_callback is not None and np.isfinite(x).all()
            if terminal_is_finite:
                latest_finite = x
            relative, backward, residual_ok = _original_residual(A, x, b, nrm2, b_norm, b_max, operator_norm,
                                                                limit, channel)
            latest_checked = terminal_is_finite

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
            if checkpoint_callback is not None:
                channel["elapsed_seconds"] = perf_counter() - started
                checkpoint("accepted", x)
            solution[:, column] = x
        except BaseException as error:
            if publishing:
                # A caller's checkpoint failure is separate from a numerical failure.
                channel["info"] = None
            elif isinstance(error, Exception) and latest_finite is not None:
                if not latest_checked:
                    _original_residual(A, latest_finite, b, nrm2, b_norm, b_max, operator_norm, limit, channel)
                channel["elapsed_seconds"] = perf_counter() - started
                checkpoint("terminal", latest_finite)
            raise
        finally:
            channel["elapsed_seconds"] = perf_counter() - started
    return solution
