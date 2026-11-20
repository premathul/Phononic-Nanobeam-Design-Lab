"""Exact axial splitting of assembled finite-element equations. Athul Prem | _AP_pnc."""

from __future__ import annotations

from dataclasses import dataclass
from numbers import Integral
from time import perf_counter

import numpy as np
from scipy.sparse import csr_matrix, issparse
from scipy.sparse.linalg import splu


@dataclass(frozen=True)
class AxialPartition:
    """Slab labels and separating DOFs in the original mesh and basis order.

    The cut coordinates select element groups. The actual separating surface
    follows retained mesh facets and may be nonplanar. DOFs in multiple groups
    are retained, including any shared only along an edge or at a vertex.
    """

    cut_planes_m: tuple[float, ...]
    element_regions: np.ndarray
    interior_dofs: tuple[np.ndarray, ...]
    separator_dofs: np.ndarray
    frontier_facets: np.ndarray


def build_axial_partition(system, cuts_m, *, protected_interval_m=None) -> AxialPartition:
    """Group existing tetrahedra by axial centroid without changing the mesh.

    If given, both endpoints of ``protected_interval_m`` must be in ``cuts_m``.
    Keep every tetrahedron touching that closed interval inside its regions,
    including tetrahedra with centroids just outside the interval.
    """
    mesh = system.mesh
    cuts = np.asarray(cuts_m, dtype=float)
    vertex_x = mesh.p[0]
    if (cuts.ndim != 1 or cuts.size == 0 or not np.isfinite(cuts).all()
            or np.any(np.diff(cuts) <= 0) or cuts[0] <= vertex_x.min() or cuts[-1] >= vertex_x.max()):
        raise ValueError("Axial cuts must be finite, strictly increasing and inside the beam.")

    element_x = vertex_x[mesh.t]
    regions = np.searchsorted(cuts, element_x.mean(axis=0), side="right")
    if protected_interval_m is not None:
        interval = np.asarray(protected_interval_m, dtype=float)
        if interval.shape != (2,) or not np.isfinite(interval).all() or interval[0] >= interval[1]:
            raise ValueError("The protected interval must have two increasing finite endpoints.")
        lower, upper = interval
        tolerance = 16 * np.finfo(float).eps * np.max(abs(vertex_x))
        endpoints = np.searchsorted(cuts, interval)
        if (not np.isin(interval, cuts).all() or endpoints[0] >= endpoints[1]):
            raise ValueError("Both protected interval endpoints must be axial cuts.")
        intersects = ((element_x.max(axis=0) >= lower - tolerance)
                      & (element_x.min(axis=0) <= upper + tolerance))
        regions[intersects] = np.clip(regions[intersects], endpoints[0] + 1, endpoints[1])

    region_count = cuts.size + 1
    if np.any(np.bincount(regions, minlength=region_count) == 0):
        raise ValueError("Each axial region must contain at least one tetrahedron.")

    element_dofs = system.basis.element_dofs
    first_region = np.full(system.basis.N, region_count, dtype=regions.dtype)
    last_region = np.full(system.basis.N, -1, dtype=regions.dtype)
    for local_dofs in element_dofs:
        np.minimum.at(first_region, local_dofs, regions)
        np.maximum.at(last_region, local_dofs, regions)
    if np.any(last_region == -1):
        raise ValueError("Every displacement DOF must belong to a tetrahedron.")
    shared = first_region != last_region
    interior = tuple(np.flatnonzero((first_region == region) & ~shared) for region in range(region_count))
    neighbors = mesh.f2t
    internal_facets = np.flatnonzero(neighbors[1] >= 0)
    adjacent = regions[neighbors[:, internal_facets]]
    frontier = internal_facets[adjacent[0] != adjacent[1]]
    return AxialPartition(tuple(cuts.tolist()), regions, interior, np.flatnonzero(shared), frontier)


def _real_if_exact(matrix):
    if np.iscomplexobj(matrix.data) and not np.any(matrix.data.imag):
        return matrix.real.copy()
    return matrix


def _factor(matrix, description):
    try:
        return splu(matrix)
    except RuntimeError as error:
        raise RuntimeError(f"Split factorization failed for {description}: {error}") from error


def _solve_factor(factor, matrix_dtype, rhs):
    # Internal regions of the lossless body can use a real factor even when
    # the native matched lead conditions make the full equation complex.
    if matrix_dtype.kind != "c" and np.iscomplexobj(rhs):
        solution = factor.solve(np.asfortranarray(rhs.real, dtype=matrix_dtype))
        if np.any(rhs.imag):
            solution = solution.astype(rhs.dtype)
            solution.imag = factor.solve(np.asfortranarray(rhs.imag, dtype=matrix_dtype))
        return solution
    return factor.solve(np.asfortranarray(rhs, dtype=matrix_dtype))


def _scatter_correction(correction, support, size):
    """Place a dense local interface update in the sparse global Schur system."""
    row_pointers = np.zeros(size + 1, dtype=np.int64)
    row_pointers[support + 1] = support.size
    np.cumsum(row_pointers, out=row_pointers)
    return csr_matrix((correction.ravel(), np.tile(support, support.size), row_pointers), shape=(size, size))


def _condense_region(operator, interior, labels, region, separator, separator_columns, rhs, block_size):
    interior_columns = operator[:, interior]
    row_labels = labels[interior_columns.indices]
    outside = (row_labels >= 0) & (row_labels != region)
    if np.any(interior_columns.data[outside]):
        raise ValueError(f"Original operator directly couples distinct region interiors at region {region}.")

    diagonal = _real_if_exact(interior_columns[interior, :])
    reaction = _real_if_exact(interior_columns[separator, :])
    del interior_columns
    coupling = _real_if_exact(separator_columns[interior, :])
    support = np.union1d(np.flatnonzero(np.diff(coupling.indptr)), reaction.indices)
    coupling = coupling[:, support]
    reaction = reaction[support, :].tocsr()

    factor_started = perf_counter()
    factor = _factor(diagonal, f"region {region} interior")
    diagnostics = {
        "interior_dofs": int(interior.size),
        "interface_dofs": int(support.size),
        "interior_operator_nnz": int(diagonal.nnz),
        "interior_dtype": str(diagonal.dtype),
        "factor_nnz": int(factor.nnz),
        "factor_seconds": perf_counter() - factor_started,
    }
    started = perf_counter()
    correction = np.empty((support.size, support.size), dtype=np.result_type(diagonal.dtype, coupling.dtype, reaction.dtype))
    for start in range(0, support.size, block_size):
        block = slice(start, min(start + block_size, support.size))
        loads = coupling[:, block].toarray(order="F")
        correction[:, block] = reaction @ _solve_factor(factor, diagonal.dtype, loads)
    interior_rhs = rhs[interior]
    rhs_delta = reaction @ _solve_factor(factor, diagonal.dtype, interior_rhs) if np.any(interior_rhs) else 0
    diagnostics.update(
        condensation_seconds=perf_counter() - started,
        local_dense_update_bytes=int(correction.nbytes),
        maximum_interface_rhs_columns=min(block_size, int(support.size)),
    )
    del factor
    return support, coupling, _scatter_correction(correction, support, separator.size), rhs_delta, diagnostics


def solve_partitioned_operator(operator, rhs, partition: AxialPartition, *, rhs_block_size: int = 8, report=None):
    """Solve the native matched system by local Schur elimination and reconstruction.

    Use with the ``linear_solver`` argument of ``solve_scattering`` to retain the
    original lead modes, end planes, fields, nodal forces and flux calculations.
    All work uses the supplied operator, including its native radiation terms.
    The separator system is sparse across regions. Dense interface load blocks
    are bounded by ``rhs_block_size`` during reduction. Each region is factored
    again for reconstruction so that only one region factor is resident.
    A singular fixed-interface region requires a different partition.
    """
    if (not isinstance(rhs_block_size, Integral) or isinstance(rhs_block_size, (bool, np.bool_))
            or rhs_block_size < 1):
        raise ValueError("Interface RHS block size must be a positive integer.")
    rhs = np.asarray(rhs)
    if (not issparse(operator) or operator.shape[0] != operator.shape[1] or not operator.shape[0]
            or rhs.ndim != 2 or rhs.shape[0] != operator.shape[0] or not rhs.shape[1]):
        raise ValueError("Split solve requires a square sparse operator and a nonempty RHS matrix.")
    dtype = np.result_type(operator.dtype, rhs.dtype, np.float64)
    if dtype not in (np.dtype(np.float64), np.dtype(np.complex128)):
        raise ValueError("Split solve requires numeric data representable in real or complex double precision.")
    operator = operator.tocsc().astype(dtype, copy=False)
    rhs = rhs.astype(dtype, copy=False)
    if not np.isfinite(operator.data).all() or not np.isfinite(rhs).all():
        raise ValueError("Split solve requires a finite operator and finite right-hand sides.")

    separator = partition.separator_dofs
    labels = np.full(operator.shape[0], -2, dtype=int)
    labels[separator] = -1
    for region, interior in enumerate(partition.interior_dofs):
        if np.any(labels[interior] != -2):
            raise ValueError("Separator and region interiors must be disjoint.")
        labels[interior] = region
    if not separator.size or np.any(labels == -2):
        raise ValueError("Split solve requires a separator and a complete DOF partition.")

    if report is None:
        report = {}
    started = perf_counter()
    report.update(separator_dofs=int(separator.size), rhs_block_size=int(rhs_block_size), regions=[])
    separator_columns = operator[:, separator]
    reduced = separator_columns[separator, :].tocsr()
    reduced_rhs = rhs[separator]
    reconstruction = []
    for region, interior in enumerate(partition.interior_dofs):
        details = {"region": region, "elements": int(np.count_nonzero(partition.element_regions == region))}
        report["regions"].append(details)
        if interior.size:
            support, coupling, correction, rhs_delta, diagnostics = _condense_region(
                operator, interior, labels, region, separator, separator_columns, rhs, rhs_block_size,
            )
            details.update(diagnostics)
            reduced -= correction
            reduced_rhs[support] -= rhs_delta
            reconstruction.append((interior, support, coupling, details))
            del correction
        else:
            details.update(interior_dofs=0, interface_dofs=0, interior_operator_nnz=0, factor_nnz=0)
    del separator_columns

    if not np.isfinite(reduced.data).all() or not np.isfinite(reduced_rhs).all():
        raise RuntimeError("Split condensation produced nonfinite interface equations.")
    report["separator_operator_nnz"] = int(reduced.nnz)
    separator_started = perf_counter()
    separator_factor = _factor(reduced.tocsc(), "separator")
    report["separator_factor_nnz"] = int(separator_factor.nnz)
    field = np.empty_like(rhs)
    field[separator] = _solve_factor(separator_factor, reduced.dtype, reduced_rhs)
    del separator_factor, reduced
    report["separator_solve_seconds"] = perf_counter() - separator_started

    for interior, support, coupling, details in reconstruction:
        rebuild_started = perf_counter()
        diagonal = _real_if_exact(operator[:, interior][interior, :])
        factor = _factor(diagonal, f"region {details['region']} reconstruction")
        field[interior] = _solve_factor(factor, diagonal.dtype, rhs[interior] - coupling @ field[separator[support]])
        del factor, diagonal
        details["reconstruction_seconds"] = perf_counter() - rebuild_started
    if not np.isfinite(field).all():
        raise RuntimeError("Split reconstruction produced a nonfinite field.")
    report["total_seconds"] = perf_counter() - started
    return field
