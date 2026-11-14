"""Small scaled-coordinate correction experiment. Author: Athul Prem | _AP_pnc."""

import argparse
import json
from math import isfinite
import platform

import numpy as np
import scipy
from scipy.linalg.blas import dznrm2
from scipy.sparse import csc_matrix, issparse
from scipy.sparse.linalg import LinearOperator, gcrotmk, spilu


DTYPE = np.dtype(np.complex128)
BATCH_ENTRIES = 2**18
FACTOR_OPTIONS = {"fill_factor": 4.0, "drop_tol": 1e-4, "permc_spec": "COLAMD"}
RELATIVE_TARGET = 1e-10
ORIGINAL_LIMIT = 1e-9
BACKWARD_LIMIT = 1e-9


def _csc_batches(A, batch_entries):
    for start in range(0, A.nnz, batch_entries):
        stop = min(start + batch_entries, A.nnz)
        # Right insertion handles repeated pointers and batches that split a column.
        columns = np.searchsorted(A.indptr, np.arange(start, stop, dtype=A.indptr.dtype), side="right")
        columns -= 1
        yield A.data[start:stop], A.indices[start:stop], columns


def scaled_factor_input(A, *, batch_entries=BATCH_ENTRIES):
    """Build one temporary scaled CSC plus scales and the original infinity norm."""
    if (not issparse(A) or A.format != "csc" or A.dtype != DTYPE or A.shape[0] != A.shape[1]
            or A.shape[0] == 0 or not A.has_canonical_format):
        raise ValueError("Expected a nonempty square canonical complex128 CSC matrix.")
    if batch_entries < 1:
        raise ValueError("Batch size must be positive.")
    row_scale = np.zeros(A.shape[0], dtype=np.float64)
    column_scale = np.zeros_like(row_scale)
    row_sums = np.zeros_like(row_scale)
    for data, rows, columns in _csc_batches(A, batch_entries):
        magnitudes = np.abs(data)
        np.maximum.at(row_scale, rows, magnitudes)
        np.maximum.at(column_scale, columns, magnitudes)
        np.add.at(row_sums, rows, magnitudes)
        del magnitudes
    for scale in (row_scale, column_scale):
        if not 0 < scale.min() <= scale.max() < np.inf:
            raise ValueError("Scaling requires finite, strictly positive row and column maxima.")
        np.sqrt(scale, out=scale)
        np.reciprocal(scale, out=scale)
    operator_infinity_norm = float(row_sums.max())
    if not isfinite(operator_infinity_norm):
        raise ValueError("The original operator infinity norm must be finite.")
    del row_sums

    factor_input = A.copy()
    for data, rows, columns in _csc_batches(factor_input, batch_entries):
        data *= row_scale[rows]
        data *= column_scale[columns]
    return factor_input, row_scale, column_scale, operator_infinity_norm


def scaled_operator(operator, row_scale, column_scale):
    """Apply D_r A D_c using an original matvec with a writable result.

    The two scales are retained. A call allocates the mapped input and uses
    the original matvec's output in place for the row scaling.
    """
    def matvec(vector):
        product = operator @ (column_scale * vector.reshape(-1))
        product *= row_scale
        return product

    return LinearOperator(operator.shape, matvec=matvec, dtype=DTYPE)


def _number_or_none(value):
    return float(value) if isfinite(value) else None


def _confirm_terminal(A, b, candidate):
    callbacks = 0
    unchanged_start = False

    def check_initial(iterate):
        nonlocal callbacks, unchanged_start
        callbacks += 1
        unchanged_start = callbacks == 1 and np.array_equal(iterate, candidate)

    confirmed, info = gcrotmk(A, b, x0=candidate.copy(), rtol=RELATIVE_TARGET, atol=0.0,
                              m=1, k=0, maxiter=1, callback=check_initial)
    accepted = info == 0 and unchanged_start and np.array_equal(confirmed, candidate)
    return {"info": int(info), "callbacks": callbacks, "unchanged_start": unchanged_start,
            "accepted": accepted}


def solve_correction(A, b, x0, *, batch_entries=BATCH_ENTRIES, maxiter=8):
    """Return diagnostics and finite original-coordinate vectors for one small system."""
    if not 1 <= maxiter <= 8:
        raise ValueError("The small check supports one to eight GCROT outer iterations.")
    for vector in (b, x0):
        if vector.shape != (A.shape[0],) or vector.dtype != DTYPE or not np.isfinite(vector).all():
            raise ValueError("The RHS and seed must be finite complex128 vectors of the operator dimension.")
    b_norm = dznrm2(b)
    if not 0 < b_norm < np.inf:
        raise ValueError("The correction requires a finite, nonzero RHS norm.")

    factor_input, row_scale, column_scale, a_inf = scaled_factor_input(A, batch_entries=batch_entries)
    b_inf = float(np.max(np.abs(b)))
    q = A @ x0
    q *= -1
    q += b
    seed_relative_residual = _number_or_none(dznrm2(q) / b_norm)
    q *= row_scale
    rho = dznrm2(q)
    if not 0 < rho < np.inf:
        raise ValueError("The correction requires a finite, nonzero scaled seed residual.")
    q /= rho
    q_norm = dznrm2(q)
    # In exact arithmetic ||q||_2 = 1 and b - A x = rho D_r^-1 (q - B y).
    # Therefore ||q - B y||_2 <= rtol suffices for the original relative target.
    rtol = RELATIVE_TARGET * b_norm * float(row_scale.min()) / rho
    if not 0 < rtol < np.inf:
        raise ValueError("The transformed relative tolerance must be finite and positive.")

    factor = spilu(factor_input, **FACTOR_OPTIONS)
    del factor_input
    B = scaled_operator(A, row_scale, column_scale)
    # Both diagonals already enter through B, q and the final reconstruction.
    preconditioner = LinearOperator(A.shape, dtype=DTYPE, matvec=lambda vector: factor.solve(vector.reshape(-1)))
    settings = {"m": 25, "k": 5, "maxiter": maxiter, "atol": 0.0, "rtol": rtol}
    history = []
    candidates = {"callbacks": [], "terminal": None}
    last_callback = None

    def evaluate(y):
        candidate = column_scale * y
        candidate *= rho
        candidate += x0
        residual = b - A @ candidate
        scale = a_inf * float(np.max(np.abs(candidate))) + b_inf
        relative = dznrm2(residual) / b_norm
        backward = float(np.max(np.abs(residual))) / scale if 0 < scale < np.inf else np.nan
        return candidate, {
            "finite": bool(np.isfinite(candidate).all()),
            "relative_residual": _number_or_none(relative),
            "relative_backward_residual": _number_or_none(backward),
            "relative_target_passed": 0 <= relative <= RELATIVE_TARGET,
            "relative_gate_passed": 0 <= relative <= ORIGINAL_LIMIT,
            "backward_gate_passed": 0 <= backward <= BACKWARD_LIMIT,
        }

    def callback(y):
        nonlocal last_callback
        last_callback = y.copy()
        candidate, diagnostics = evaluate(y)
        history.append(diagnostics)
        if diagnostics["finite"]:
            candidates["callbacks"].append(candidate)

    y, raw_info = gcrotmk(B, q, M=preconditioner, callback=callback, **settings)
    factor_nonzeros = int(factor.nnz)
    del preconditioner, factor
    candidate, terminal = evaluate(y)
    terminal["normalized_transformed_relative_residual"] = _number_or_none(dznrm2(q - B @ y) / q_norm)
    if terminal["finite"]:
        candidates["terminal"] = candidate
    residual_ok = all(terminal[key] for key in ("finite", "relative_target_passed", "relative_gate_passed",
                                               "backward_gate_passed"))
    final_update = last_callback is not None and not np.array_equal(y, last_callback)
    confirmation = None
    info = int(raw_info)
    if raw_info == maxiter and final_update and residual_ok:
        confirmation = _confirm_terminal(A, b, candidate)
        if confirmation["accepted"]:
            info = 0
    vector_count = len(candidates["callbacks"]) + int(candidates["terminal"] is not None)
    report = {
        "accepted": info == 0 and residual_ok,
        "relative_target": RELATIVE_TARGET,
        "relative_residual_limit": ORIGINAL_LIMIT,
        "backward_residual_limit": BACKWARD_LIMIT,
        "operator_infinity_norm": a_inf,
        "rhs_norm": b_norm,
        "seed": {"relative_residual": seed_relative_residual},
        "scaled_seed_residual_norm": rho,
        "transformed_rhs_norm": q_norm,
        "factorization": {**FACTOR_OPTIONS, "dtype": DTYPE.name, "factor_nonzeros": factor_nonzeros},
        "solver": {**settings, "raw_info": int(raw_info), "info": info,
                   "returned_update_after_last_callback": final_update, "terminal_confirmation": confirmation},
        "callbacks": history,
        "terminal": terminal,
        "candidate_storage_bytes": vector_count * x0.nbytes,
    }
    return report, candidates


def small_dense_check():
    """Run a standalone check returning JSON-compatible results."""
    dense = np.array([[9 + 2j, 17 - 3j, 0, 2], [-1 + 2j, 6, 3j, 0],
                      [0.5j, -1, 4 + 1j, 8], [0, 2 + 1j, -1j, 7]], dtype=DTYPE)
    A = csc_matrix(dense)
    b = np.array([1.2 + 0.7j, 1.1 - 0.4j, -2.3j, 2.7 + 0.8j], dtype=DTYPE)
    x0 = np.array([0.125 - 0.25j, 0.5j, -0.5, -0.25j], dtype=DTYPE)
    report, candidates = solve_correction(A, b, x0, batch_entries=3)
    reference = np.linalg.solve(dense, b)
    candidate = candidates["terminal"]
    if candidate is None:
        raise AssertionError("The small check produced no finite terminal vector.")
    residual = b - dense @ candidate
    relative = float(np.linalg.norm(residual) / np.linalg.norm(b))
    backward = float(np.linalg.norm(residual, np.inf) / (
        np.linalg.norm(dense, np.inf) * np.linalg.norm(candidate, np.inf) + np.linalg.norm(b, np.inf)
    ))
    solution_error = float(np.linalg.norm(candidate - reference) / np.linalg.norm(reference))
    passed = (report["accepted"] and relative <= RELATIVE_TARGET and backward <= BACKWARD_LIMIT
              and solution_error <= RELATIVE_TARGET)
    return {"research_owner": "Athul Prem", "project_tag": "_AP_pnc", "passed": passed,
            "environment": {"python": platform.python_version(), "platform": platform.platform(),
                            "numpy": np.__version__, "scipy": scipy.__version__},
            "independent_dense": {"relative_residual": relative, "relative_backward_residual": backward,
                                  "relative_solution_error": solution_error},
            "correction": report}


if __name__ == "__main__":
    argparse.ArgumentParser(description="Run the small synthetic scaled-coordinate correction check.").parse_args()
    result = small_dense_check()
    print(json.dumps(result, indent=2, allow_nan=False))
    raise SystemExit(0 if result["passed"] else 1)
