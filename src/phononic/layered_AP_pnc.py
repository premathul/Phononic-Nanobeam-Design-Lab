"""Layered affine tetrahedra for straight beams. Author: Athul Prem | _AP_pnc."""

from __future__ import annotations

from math import ceil
from numbers import Integral
from typing import Sequence

import numpy as np
from skfem import MeshTet

from .geometry_AP_pnc import (
    _finite_real, _GMSH_LOCK, _METERS_PER_CAD_UNIT, BeamGeometry, Hole, HolePattern,
    validate_hole_clearances,
)


def _footprint(model, geometry: BeamGeometry, holes: Sequence[Hole], mesh_size_m: float,
               port_cells: int, circle_points: int) -> tuple[np.ndarray, np.ndarray]:
    """Mesh the x-y section once, with identical translated port edges."""
    unit = _METERS_PER_CAD_UNIT
    length, width = geometry.length_m / unit, geometry.width_m / unit
    occ = model.occ
    corners = [occ.addPoint(x, y, 0) for x, y in
               ((0, -width / 2), (length, -width / 2), (length, width / 2), (0, width / 2))]
    outer = [occ.addLine(a, b) for a, b in zip(corners, corners[1:] + corners[:1])]
    loops = [occ.addCurveLoop(outer)]
    circle_arcs = []
    for hole in holes:
        x, radius = hole.x_m / unit, hole.diameter_m / (2 * unit)
        center = occ.addPoint(x, 0, 0)
        points = [occ.addPoint(x + dx, dy, 0) for dx, dy in
                  ((radius, 0), (0, radius), (-radius, 0), (0, -radius))]
        arcs = [occ.addCircleArc(a, center, b) for a, b in zip(points, points[1:] + points[:1])]
        circle_arcs.extend(arcs)
        loops.append(occ.addCurveLoop(arcs))
    face = occ.addPlaneSurface(loops)
    occ.synchronize()

    longitudinal_cells = ceil(geometry.length_m / mesh_size_m / (1 + 1e-12))
    for edge in (outer[0], outer[2]):
        model.mesh.setTransfiniteCurve(edge, longitudinal_cells + 1)
    left, right = outer[3], outer[1]
    for edge in (left, right):
        model.mesh.setTransfiniteCurve(edge, port_cells + 1)
    for arc in circle_arcs:
        model.mesh.setTransfiniteCurve(arc, circle_points // 4 + 1)
    translation = np.eye(4)
    translation[0, 3] = length
    model.mesh.setPeriodic(1, [right], [left], translation.ravel())
    model.mesh.generate(2)

    master, slave_nodes, master_nodes, _ = model.mesh.getPeriodicNodes(1, right)
    if master != left or len(slave_nodes) != port_cells + 1 or len(master_nodes) != port_cells + 1:
        raise RuntimeError("Gmsh did not generate the requested matched port edges")

    node_tags, xyz, _ = model.mesh.getNodes(returnParametricCoord=False)
    tags = model.mesh.getElementsByType(2, face)[1]
    if not len(tags):
        raise RuntimeError("Gmsh did not generate footprint triangles")
    used, triangles = np.unique(tags, return_inverse=True)
    tag_order = np.argsort(node_tags)
    xy = xyz.reshape(-1, 3)[tag_order[np.searchsorted(node_tags[tag_order], used)], :2] * unit
    # Use the prescribed end coordinates after checking that the periodic mesh
    # produced the uniform subdivision to within roundoff.
    left_indices = np.searchsorted(used, master_nodes)
    right_indices = np.searchsorted(used, slave_nodes)
    if not np.array_equal(used[left_indices], master_nodes) or not np.array_equal(used[right_indices], slave_nodes):
        raise RuntimeError("A periodic edge node is missing from the footprint")
    order_y = np.argsort(xy[left_indices, 1])
    left_indices, right_indices = left_indices[order_y], right_indices[order_y]
    y = np.linspace(-geometry.width_m / 2, geometry.width_m / 2, port_cells + 1)
    if not np.allclose(xy[left_indices, 1], y, rtol=0, atol=geometry.width_m * 1e-10):
        raise RuntimeError("The footprint does not have the requested uniform port-edge spacing")
    xy[left_indices, 0] = 0
    xy[right_indices, 0] = geometry.length_m
    xy[left_indices, 1] = xy[right_indices, 1] = y

    # Both end edges now order from low y to high y. This gives both port faces
    # the same quad diagonal when the sorted footprint is split into tetrahedra.
    order = np.lexsort((xy[:, 0], xy[:, 1]))
    return xy[order], np.sort(np.argsort(order)[triangles].reshape(-1, 3), axis=1)


def make_layered_mesh(
    geometry: BeamGeometry,
    holes: HolePattern | Sequence[Hole],
    mesh_size_m: float,
    *,
    port_mesh_size_m: float,
    thickness_layers: int,
    circle_points: int = 24,
    max_port_dofs: int = 200,
    element_order: int = 2,
) -> MeshTet:
    """Return a straight affine mesh extruded from an independently meshed footprint.

    Lengths are in metres. ``mesh_size_m`` sets the target in the x-y plane,
    ``port_mesh_size_m`` sets only the transverse width subdivision, and
    ``thickness_layers`` sets the number of equal z intervals throughout the
    body and at both ports. The same footprint is used on every z plane.
    Exact circular arcs in CAD are sampled at ``circle_points`` vertices per
    rim. Use an empty hole sequence for a uniform control. The port budget
    includes all three displacement components at the requested element order.
    """
    for name, size in (("mesh_size_m", mesh_size_m), ("port_mesh_size_m", port_mesh_size_m)):
        if not _finite_real(size) or size <= 0:
            raise ValueError(f"{name} must be positive and finite")
    for name, value in (("thickness_layers", thickness_layers), ("max_port_dofs", max_port_dofs),
                        ("circle_points", circle_points), ("element_order", element_order)):
        if not isinstance(value, Integral) or isinstance(value, bool) or value < 1:
            raise ValueError(f"{name} must be a positive integer")
    if circle_points < 8 or circle_points % 4:
        raise ValueError("circle_points must be a multiple of 4 and at least 8")
    if element_order not in (1, 2):
        raise ValueError("element_order must be 1 or 2")
    placed = holes.holes if isinstance(holes, HolePattern) else tuple(holes)
    if placed:
        validate_hole_clearances(geometry, placed)
    port_cells = ceil(geometry.width_m / port_mesh_size_m / (1 + 1e-12))
    port_dofs = 3 * (element_order * port_cells + 1) * (element_order * thickness_layers + 1)
    if port_dofs > max_port_dofs:
        raise ValueError(f"Port has {port_dofs} DOFs, exceeding dense solver cap {max_port_dofs}")

    import gmsh

    with _GMSH_LOCK:
        if gmsh.isInitialized():
            raise RuntimeError("layered meshing requires its own Gmsh session")
        gmsh.initialize(readConfigFiles=False, interruptible=False)
        try:
            gmsh.option.setNumber("General.Terminal", 0)
            gmsh.option.setNumber("General.NumThreads", 1)
            gmsh.option.setNumber("Mesh.MeshSizeFromPoints", 0)
            gmsh.option.setNumber("Mesh.MeshSizeFromCurvature", 0)
            gmsh.option.setNumber("Mesh.MeshSizeExtendFromBoundary", 0)
            gmsh.option.setNumber("Mesh.MeshSizeMax", mesh_size_m / _METERS_PER_CAD_UNIT)
            gmsh.model.add("layered_beam_AP_pnc")
            xy, triangles = _footprint(gmsh.model, geometry, placed, mesh_size_m, port_cells, circle_points)
        finally:
            gmsh.finalize()

    n = len(xy)
    z = np.linspace(-geometry.thickness_m / 2, geometry.thickness_m / 2, thickness_layers + 1)
    points = np.column_stack((np.tile(xy, (len(z), 1)), np.repeat(z, n)))
    a, b, c = triangles.T
    # For a<b<c, every lateral quad on edge i<j uses diagonal i→j' in
    # both adjacent prisms. Primes denote vertices on the upper z plane.
    slab = np.column_stack((np.stack((a, b, c, c + n)),
                            np.stack((a, b, b + n, c + n)),
                            np.stack((a, a + n, b + n, c + n))))
    tets = (slab[None, :, :] + n * np.arange(thickness_layers)[:, None, None]).transpose(1, 0, 2).reshape(4, -1)
    mesh = MeshTet(points.T, tets).oriented()
    if not np.all(mesh.mapping().detA > 0):
        raise RuntimeError("The layered mesh contains a nonpositive tetrahedron")
    boundary = mesh.boundary_facets()
    x = mesh.p[0, mesh.facets[:, boundary]]
    left = np.all(x == 0, axis=0)
    right = np.all(x == geometry.length_m, axis=0)
    expected = 2 * port_cells * thickness_layers
    if left.sum() != expected or right.sum() != expected:
        raise RuntimeError("The layered mesh has an incomplete port trace")
    return mesh.with_boundaries({"xleft": boundary[left], "xright": boundary[right],
                                 "free": boundary[~(left | right)]}).with_subdomains({"solid": np.arange(mesh.nelements)})
