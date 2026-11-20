"""Transfer native P2 displacement on a fixed solid. Author: Athul Prem | _AP_pnc."""

from dataclasses import dataclass
from math import isfinite

import numpy as np
from scipy.sparse import csr_matrix
from scipy.spatial import cKDTree
from skfem import Dofs, ElementTetP2, MeshTet


BARYCENTRIC_ATOL = 5e-10
STENCIL_ATOL = 2e-9
_P2 = ElementTetP2()
_EDGE_START, _EDGE_END = np.asarray(_P2.refdom.edges).T


@dataclass(frozen=True)
class P2Transfer:
    """A scalar nodal map applied to each native xyz component independently."""

    scalar_matrix: csr_matrix
    shared_target: np.ndarray
    shared_source: np.ndarray
    diagnostics: dict

    def apply(self, source_fields: np.ndarray) -> np.ndarray:
        """Transfer a native flat field or a matrix with one field per column."""
        fields = np.asarray(source_fields)
        source_nodes = self.scalar_matrix.shape[1]
        if (fields.ndim not in (1, 2) or fields.shape[0] != 3 * source_nodes
                or fields.dtype.kind not in "fc" or not np.isfinite(fields).all()):
            raise ValueError("Expected finite floating-point fields in native xyz-interleaved DOF order.")
        blocks = fields.reshape(source_nodes, -1)
        target = self.scalar_matrix @ blocks
        target[self.shared_target] = blocks[self.shared_source]
        if not np.isfinite(target).all():
            raise ValueError("The transferred field is nonfinite.")
        return target.reshape((3 * self.scalar_matrix.shape[0], *fields.shape[1:]))


def _layout(mesh):
    if not isinstance(mesh, MeshTet) or not mesh.affine or not mesh.nelements:
        raise ValueError("P2 transfer requires a nonempty affine tetrahedral mesh.")
    if not np.isfinite(mesh.p).all():
        raise ValueError("Mesh coordinates must be finite.")
    dofs = Dofs(mesh, _P2)
    vertices = mesh.p.T
    edge_midpoints = 0.5 * (vertices[mesh.edges[0]] + vertices[mesh.edges[1]])
    return dofs, np.concatenate((vertices, edge_midpoints))


def _port_checks(source, target, source_dofs, target_dofs, target_to_source):
    diagnostics = {}
    for name in ("xleft", "xright"):
        source_facets = (source.boundaries or {}).get(name)
        target_facets = (target.boundaries or {}).get(name)
        if source_facets is None and target_facets is None:
            continue
        if (source_facets is None or target_facets is None or not len(source_facets)
                or not len(target_facets)):
            raise ValueError(f"Both meshes must declare the physical {name} port.")
        source_triangles = source.facets[:, source_facets].T
        target_triangles = target.facets[:, target_facets].T
        source_nodes = source_dofs.get_facet_dofs(source_facets).all()
        target_nodes = target_dofs.get_facet_dofs(target_facets).all()
        mapped_triangles = target_to_source[target_triangles]
        if (
            not np.array_equal(np.sort(source_nodes), np.sort(target_to_source[target_nodes]))
            or not np.array_equal(
                np.unique(np.sort(source_triangles, axis=1), axis=0),
                np.unique(np.sort(mapped_triangles, axis=1), axis=0),
            )
        ):
            raise ValueError(f"Physical {name} port triangulation or P2 nodes differ.")
        x = source.p[0, source_triangles]
        if not (x == x.flat[0]).all():
            raise ValueError(f"Physical {name} port is not on a single x plane.")
        diagnostics[name] = {"x_m": float(x.flat[0]), "facets": len(source_facets),
                             "exact_vector_dofs": 3 * len(target_nodes)}
    return diagnostics


def _candidate_batches(tree, points, radius, batch_size):
    # Limit the number of point/element pairs even in dense areas of the tree.
    pair_limit = max(64_000, tree.n)
    for start in range(0, len(points), batch_size):
        stop = min(start + batch_size, len(points))
        counts = tree.query_ball_point(points[start:stop], radius, p=np.inf, return_length=True)
        offsets = np.concatenate(([0], np.cumsum(counts)))
        left = 0
        while left < len(counts):
            right = np.searchsorted(offsets, offsets[left] + pair_limit, side="right") - 1
            batch = slice(start + left, start + right)
            candidates = tree.query_ball_point(points[batch], radius, p=np.inf, return_sorted=True)
            rows = np.repeat(np.arange(right - left), counts[left:right])
            elements = np.concatenate(candidates, dtype=np.int32)
            yield batch, rows, elements
            left = right


def _locate(points, rows, elements, origins, jacobians, inverses, lower, upper, coord_atol):
    xyz = points[rows]
    in_bounds = np.all((xyz >= lower[elements] - coord_atol)
                       & (xyz <= upper[elements] + coord_atol), axis=1)
    rows, elements, xyz = rows[in_bounds], elements[in_bounds], xyz[in_bounds]
    offset = xyz - origins[elements]
    reference = np.einsum("nij,nj->ni", inverses[elements], offset)
    bary = np.column_stack((1.0 - reference.sum(axis=1), reference))
    smallest = bary.min(axis=1)
    largest = bary.max(axis=1)
    plausible = (smallest >= -BARYCENTRIC_ATOL) & (largest <= 1.0 + BARYCENTRIC_ATOL)
    bounded = np.clip(bary[plausible], 0.0, 1.0)
    bounded /= bounded.sum(axis=1, keepdims=True)
    reconstruction = np.einsum("nij,nj->ni", jacobians[elements[plausible]], bounded[:, 1:])
    error = np.max(np.abs(reconstruction - offset[plausible]), axis=1)
    contains = error <= coord_atol
    return (rows[plausible][contains], elements[plausible][contains], bounded[contains],
            smallest[plausible][contains], error[contains])


def build_p2_transfer(source_mesh: MeshTet, target_mesh: MeshTet, *, batch_size: int = 1024) -> P2Transfer:
    """Construct a pointwise map for affine P2 meshes in the same physical coordinates.

    Shared nodes and edge midpoints are copied exactly. Other P2 nodes require
    containment in an actual source tetrahedron. Named end ports must coincide
    in geometry and triangulation, preserving the entire quadratic trace.
    """
    if not isinstance(batch_size, int) or batch_size < 1:
        raise ValueError("Transfer batch size must be a positive integer.")
    source_dofs, source_points = _layout(source_mesh)
    target_dofs, target_points = _layout(target_mesh)
    nearest_dist, nearest_index = cKDTree(source_points).query(target_points, k=2, p=np.inf)
    exact = nearest_dist[:, 0] == 0.0
    if np.any(exact & (nearest_dist[:, 1] == 0.0)):
        raise ValueError("An exact P2 target node coincides with multiple source DOFs.")
    shared_target = np.flatnonzero(exact)
    shared_source = nearest_index[exact, 0]
    target_to_source = np.where(exact, nearest_index[:, 0], -1)
    ports = _port_checks(source_mesh, target_mesh, source_dofs, target_dofs, target_to_source)
    unshared = np.flatnonzero(~exact)

    tets = source_mesh.p.T[source_mesh.t.T]
    coord_atol = 32 * np.finfo(float).eps * max(np.abs(tets).max(), np.abs(target_points).max())
    if not isfinite(coord_atol) or coord_atol <= 0:
        raise ValueError("Invalid physical coordinate scale for P2 transfer.")
    origins = tets[:, 0]
    jacobians = (tets[:, 1:] - origins[:, None]).transpose(0, 2, 1)
    try:
        inverses = np.linalg.inv(jacobians)
    except np.linalg.LinAlgError as exc:
        raise ValueError("Source mesh contains a degenerate tetrahedron.") from exc
    if not np.isfinite(inverses).all():
        raise ValueError("Source mesh contains a degenerate tetrahedron.")
    lower, upper = tets.min(axis=1), tets.max(axis=1)
    centers = tets.mean(axis=1)
    radius = float(np.abs(tets - centers[:, None]).max() + coord_atol)
    centroid_tree = cKDTree(centers)

    indices = np.zeros((target_dofs.N, 10), dtype=np.int32)
    weights = np.zeros((target_dofs.N, 10), dtype=float)
    indices[shared_target, 0] = shared_source
    weights[shared_target, 0] = 1.0
    multiple_parents = 0
    minimum_barycentric = 0.0
    max_reconstruction = 0.0
    max_trace_difference = 0.0
    query_points = target_points[unshared]
    for batch, rows, elements in _candidate_batches(centroid_tree, query_points, radius, batch_size):
        points = query_points[batch]
        rows, elements, bary, raw_min, error = _locate(
            points, rows, elements, origins, jacobians, inverses, lower, upper, coord_atol)
        counts = np.bincount(rows, minlength=len(points))
        target_indices = unshared[batch]
        if (counts == 0).any():
            first = int(np.flatnonzero(counts == 0)[0])
            raise ValueError(f"Target P2 node {target_indices[first]} lies outside the source solid "
                             f"at {points[first].tolist()} m.")
        minimum_barycentric = min(minimum_barycentric, float(raw_min.min()))
        max_reconstruction = max(max_reconstruction, float(error.max()))
        coefficients = np.column_stack((bary * (2.0 * bary - 1.0),
                                        4.0 * bary[:, _EDGE_START] * bary[:, _EDGE_END]))
        element_indices = source_dofs.element_dofs[:, elements].T
        primary = np.cumsum(counts) - counts
        # Compare in global DOFs, where a shared facet/edge has one P2 trace.
        alternatives = np.flatnonzero(np.arange(len(rows)) != primary[rows])
        if len(alternatives):
            other = primary[rows[alternatives]]
            matches = element_indices[alternatives, :, None] == element_indices[other, None, :]
            delta_a = coefficients[alternatives] - np.einsum("nij,nj->ni", matches, coefficients[other])
            delta_b = coefficients[other] - np.einsum("nij,ni->nj", matches, coefficients[alternatives])
            delta = np.maximum(np.abs(delta_a).max(axis=1), np.abs(delta_b).max(axis=1))
            if (delta > STENCIL_ATOL).any():
                first = alternatives[int(np.flatnonzero(delta > STENCIL_ATOL)[0])]
                raise ValueError(f"Target P2 node {target_indices[rows[first]]} has ambiguous "
                                 "source tetrahedra with different P2 traces.")
            max_trace_difference = max(max_trace_difference, float(delta.max()))
            multiple_parents += int(np.count_nonzero(counts > 1))
        indices[target_indices] = element_indices[primary]
        weights[target_indices] = coefficients[primary]

    matrix = csr_matrix((weights.ravel(), indices.ravel(), np.arange(0, 10 * (target_dofs.N + 1), 10)),
                        shape=(target_dofs.N, source_dofs.N))
    matrix.eliminate_zeros()
    matrix.sort_indices()
    return P2Transfer(matrix, shared_target, shared_source, {
        "source_scalar_dofs": int(source_dofs.N),
        "target_scalar_dofs": int(target_dofs.N),
        "exact_shared_scalar_dofs": len(shared_target),
        "interpolated_scalar_dofs": len(unshared),
        "multiple_parent_scalar_dofs": multiple_parents,
        "coordinate_tolerance_m": coord_atol,
        "maximum_reconstruction_error_m": max_reconstruction,
        "barycentric_tolerance": BARYCENTRIC_ATOL,
        "minimum_accepted_barycentric": minimum_barycentric,
        "trace_stencil_tolerance": STENCIL_ATOL,
        "maximum_trace_stencil_difference": max_trace_difference,
        "ports": ports,
    })
