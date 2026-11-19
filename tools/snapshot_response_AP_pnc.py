"""Trial response from endpoint volume snapshots. Author: Athul Prem | _AP_pnc."""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class SnapshotResponse:
    """Galerkin field and full-equation diagnostics for each target drive."""

    solution: np.ndarray
    relative_residuals: np.ndarray
    minimum_relative_residuals: np.ndarray
    rank: int
    action_rank: int
    condition: float


def solve_snapshot_response(
    snapshots: np.ndarray,
    apply_operator: Callable[[np.ndarray], np.ndarray],
    rhs: np.ndarray,
    *,
    rank_rtol: float = 1e-12,
) -> SnapshotResponse:
    """Project the exact target-frequency equations onto the snapshot span.

    Snapshots and forcing columns must share the target operator's volume DOF
    ordering. The action takes a matrix of fields and returns one force column
    per field. A single RHS vector gives a solution vector.

    The basis Q retains left singular vectors with sigma > rank_rtol * sigma_max.
    The field solves (Q.H A Q)c = Q.H b and the condition is cond_2(Q.H A Q).
    An empty span or a singular projected solve raises an exception.

    Residuals are ||A u - b||_2 / ||b||_2 on the reconstructed fields, with a
    zero residual reported for an exact zero drive. The minimum-residual fit
    solves min_c ||(A Q)c - b||_2 via least squares. Its action rank uses NumPy's
    cutoff eps * max((A Q).shape) relative to the largest action singular value.
    If that rank is below the snapshot rank, the fit is for the retained action
    directions and cannot establish the minimum over the entire snapshot span.
    """
    snapshots = np.asarray(snapshots, dtype=np.complex128 if np.iscomplexobj(snapshots) else np.float64)
    rhs = np.asarray(rhs, dtype=np.complex128 if np.iscomplexobj(rhs) else np.float64)
    if snapshots.ndim != 2 or min(snapshots.shape) == 0 or not np.isfinite(snapshots).all():
        raise ValueError("snapshots must be a finite, nonempty volume-by-snapshot matrix")
    single_drive = rhs.ndim == 1
    if single_drive:
        rhs = rhs[:, None]
    if rhs.ndim != 2 or rhs.shape[0] != snapshots.shape[0] or not rhs.shape[1] or not np.isfinite(rhs).all():
        raise ValueError("rhs must contain finite drive columns in the snapshot volume DOF order")
    if not 0 <= rank_rtol < 1:
        raise ValueError("rank_rtol must be in [0, 1)")

    basis, singular_values, _ = np.linalg.svd(snapshots, full_matrices=False)
    rank = int(np.count_nonzero(singular_values > rank_rtol * singular_values[0]))
    if rank == 0:
        raise ValueError("snapshots have zero numerical rank")
    basis = basis[:, :rank]

    def apply(fields):
        forces = np.asarray(apply_operator(fields))
        if forces.shape != fields.shape or not np.isfinite(forces).all():
            raise ValueError("operator action must return one finite volume force column per field")
        return forces

    applied_basis = apply(basis)
    adjoint = basis.conj().T
    reduced_operator = adjoint @ applied_basis
    coefficients = np.linalg.solve(reduced_operator, adjoint @ rhs)
    minimum_coefficients, _, action_rank, _ = np.linalg.lstsq(applied_basis, rhs, rcond=None)
    del adjoint, applied_basis

    rhs_norms = np.linalg.norm(rhs, axis=0)

    def residuals(fields):
        norms = np.linalg.norm(apply(fields) - rhs, axis=0)
        return np.divide(norms, rhs_norms, out=np.where(norms == 0, 0.0, np.inf), where=rhs_norms > 0)

    solution = basis @ coefficients
    return SnapshotResponse(
        solution=solution[:, 0] if single_drive else solution,
        relative_residuals=residuals(solution),
        minimum_relative_residuals=residuals(basis @ minimum_coefficients),
        rank=rank,
        action_rank=int(action_rank),
        condition=float(np.linalg.cond(reduced_operator)),
    )
