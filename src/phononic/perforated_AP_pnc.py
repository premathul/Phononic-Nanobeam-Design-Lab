"""Perforated beam meshes with matching waveguide traces.

Author: Athul Prem | _AP_pnc
"""

from __future__ import annotations

from math import ceil, cos, pi
from numbers import Integral
from typing import Sequence

import numpy as np
from skfem import MeshTet

from .geometry_AP_pnc import (
    _finite_real, _GMSH_LOCK, _METERS_PER_CAD_UNIT, BeamGeometry, Hole, HolePattern,
    validate_hole_clearances,
)


def _make_solid(occ, length: float, width: float, thickness: float, holes: Sequence[Hole]):
    """Extrude exact circular quadrant arcs so each wall is a four-sided patch."""
    z = -thickness / 2
    corners = [occ.addPoint(x, y, z) for x, y in
               ((0, -width / 2), (length, -width / 2),
                (length, width / 2), (0, width / 2))]
    outer = [occ.addLine(a, b) for a, b in zip(corners, corners[1:] + corners[:1])]
    loops = [occ.addCurveLoop(outer)]
    for hole in holes:
        x = hole.x_m / _METERS_PER_CAD_UNIT
        radius = hole.diameter_m / (2 * _METERS_PER_CAD_UNIT)
        center = occ.addPoint(x, 0, z)
        points = [occ.addPoint(x + dx, dy, z) for dx, dy in
                  ((radius, 0), (0, radius), (-radius, 0), (0, -radius))]
        arcs = [occ.addCircleArc(a, center, b) for a, b in zip(points, points[1:] + points[:1])]
        loops.append(occ.addCurveLoop(arcs))
    face = occ.addPlaneSurface(loops)
    volumes = [tag for dim, tag in occ.extrude([(2, face)], 0, 0, thickness) if dim == 3]
    occ.synchronize()
    if len(volumes) != 1:
        raise RuntimeError("beam extrusion must produce a single solid")
    return volumes[0]


def _verify_port_mesh(model, ports, ny: int, nz: int) -> None:
    left, right = ports
    master, right_nodes, left_nodes, _ = model.mesh.getPeriodicNodes(2, right)
    vertex_count = (ny + 1) * (nz + 1)
    if master != left or len(left_nodes) != vertex_count or len(right_nodes) != vertex_count:
        raise RuntimeError("Gmsh did not produce the requested periodic end mesh")
    pairs = dict(zip(right_nodes, left_nodes))
    left_triangles = model.mesh.getElementsByType(2, left)[1].reshape(-1, 3)
    right_triangles = model.mesh.getElementsByType(2, right)[1].reshape(-1, 3)
    expected = {tuple(sorted(face)) for face in left_triangles}
    mapped = {tuple(sorted(pairs[node] for node in face)) for face in right_triangles}
    if len(left_triangles) != 2 * ny * nz or len(right_triangles) != len(expected) or mapped != expected:
        raise RuntimeError("Gmsh end-face triangles do not match under translation")


def _read_mesh(model, volume: int, groups: dict[str, list[int]]) -> MeshTet:
    import meshio
    from skfem.io import from_meshio

    _, tet_nodes = model.mesh.getElementsByType(4, volume)
    if not len(tet_nodes):
        raise RuntimeError("Gmsh produced no linear tetrahedra")
    node_tags, xyz, _ = model.mesh.getNodes(returnParametricCoord=False)
    # Construction centers for the exact CAD arcs are not vertices of the solid.
    # Retain only nodes attached to a tetrahedron, even if Gmsh meshed a center.
    used, tets = np.unique(tet_nodes, return_inverse=True)
    order = np.argsort(node_tags)
    coordinates = xyz.reshape(-1, 3)[order[np.searchsorted(node_tags[order], used)]]
    triangles = []
    sets = {}
    start = 0
    for name, tags in groups.items():
        nodes = np.concatenate([model.mesh.getElementsByType(2, tag)[1] for tag in tags])
        faces = np.searchsorted(used, nodes).reshape(-1, 3)
        triangles.append(faces)
        stop = start + len(faces)
        sets[name] = [np.empty(0, dtype=int), np.arange(start, stop)]
        start = stop
    tetrahedra = tets.reshape(-1, 4)
    sets["solid"] = [np.arange(len(tetrahedra)), np.empty(0, dtype=int)]
    external = meshio.Mesh(coordinates * _METERS_PER_CAD_UNIT,
                           [("tetra", tetrahedra), ("triangle", np.vstack(triangles))], cell_sets=sets)
    mesh = from_meshio(external, ignore_orientation=True).oriented()
    if mesh.boundaries is None or set(mesh.boundaries) != set(groups):
        raise RuntimeError("failed to import all Gmsh surface groups")
    tagged = np.sort(np.concatenate(list(mesh.boundaries.values())))
    if not np.array_equal(tagged, mesh.boundary_facets()):
        raise RuntimeError("Gmsh surface groups must partition the tetrahedral exterior")
    return mesh


def make_perforated_mesh(
    geometry: BeamGeometry,
    holes: HolePattern | Sequence[Hole],
    mesh_size_m: float,
    *,
    port_mesh_size_m: float,
    circle_points: int = 24,
    max_port_dofs: int = 200,
    element_order: int = 2,
) -> MeshTet:
    """Return affine tetrahedra with matching triangular end faces in SI units.

    The end faces use structured triangles with subdivisions set only by
    ``port_mesh_size_m``. The free surfaces and volume use ``mesh_size_m`` as
    the target size, subject to the prescribed end and wall subdivisions.
    Each exact cylindrical CAD wall is discretized as a regular inscribed
    prism with ``circle_points`` sides, independent of ``mesh_size_m``. Axial
    wall subdivisions through the thickness follow the body mesh size.
    Use an empty hole sequence for a Gmsh control with the same end meshes.
    ``element_order`` applies to the port budget, since the geometry is affine.
    """
    for name, size in (("mesh_size_m", mesh_size_m), ("port_mesh_size_m", port_mesh_size_m)):
        if not _finite_real(size) or size <= 0:
            raise ValueError(f"{name} must be positive and finite")
    if (not isinstance(circle_points, Integral) or isinstance(circle_points, bool)
            or circle_points < 8 or circle_points % 4):
        raise ValueError("circle_points must be a multiple of 4 and at least 8")
    if (not isinstance(max_port_dofs, Integral) or isinstance(max_port_dofs, bool)
            or max_port_dofs < 1):
        raise ValueError("max_port_dofs must be a positive integer")
    if not isinstance(element_order, Integral) or isinstance(element_order, bool) or element_order not in (1, 2):
        raise ValueError("element_order must be 1 or 2")
    placed = holes.holes if isinstance(holes, HolePattern) else tuple(holes)
    if placed:
        validate_hole_clearances(geometry, placed)

    # Avoid adding a whole subdivision due to roundoff in an exact size ratio.
    ny = ceil(geometry.width_m / port_mesh_size_m / (1 + 1e-12))
    nz = ceil(geometry.thickness_m / port_mesh_size_m / (1 + 1e-12))
    port_dofs = 3 * (element_order * ny + 1) * (element_order * nz + 1)
    if port_dofs > max_port_dofs:
        raise ValueError(f"Port has {port_dofs} DOFs, exceeding dense solver cap {max_port_dofs}")
    length = geometry.length_m / _METERS_PER_CAD_UNIT
    width = geometry.width_m / _METERS_PER_CAD_UNIT
    thickness = geometry.thickness_m / _METERS_PER_CAD_UNIT
    wall_layers = ceil(geometry.thickness_m / mesh_size_m / (1 + 1e-12))

    import gmsh

    with _GMSH_LOCK:
        if gmsh.isInitialized():
            raise RuntimeError("perforated meshing requires its own Gmsh session")
        gmsh.initialize(readConfigFiles=False, interruptible=False)
        try:
            gmsh.option.setNumber("General.Terminal", 0)
            gmsh.option.setNumber("General.NumThreads", 1)
            gmsh.option.setNumber("Mesh.MeshSizeFromPoints", 0)
            gmsh.option.setNumber("Mesh.MeshSizeFromCurvature", 0)
            gmsh.option.setNumber("Mesh.MeshSizeExtendFromBoundary", 0)
            gmsh.option.setNumber("Mesh.MeshSizeMax", mesh_size_m / _METERS_PER_CAD_UNIT)
            model = gmsh.model
            model.add("matched_perforated_beam_AP_pnc")
            volume = _make_solid(model.occ, length, width, thickness, placed)
            groups = {"xleft": [], "xright": [], "free": []}
            walls = []
            for _, tag in model.getBoundary([(3, volume)]):
                surface_type = model.getType(2, tag)
                center_x = model.occ.getCenterOfMass(2, tag)[0]
                name = "free"
                if surface_type == "Plane" and abs(center_x) <= length * 1e-10:
                    name = "xleft"
                elif surface_type == "Plane" and abs(center_x - length) <= length * 1e-10:
                    name = "xright"
                elif surface_type == "Cylinder":
                    walls.append(tag)
                groups[name].append(tag)
            if len(groups["xleft"]) != 1 or len(groups["xright"]) != 1 or len(walls) != 4 * len(placed):
                raise RuntimeError("beam must have two end faces and four cylindrical patches per hole")

            for wall in walls:
                for _, curve in model.getBoundary([(2, wall)]):
                    segments = circle_points // 4 if model.getType(1, curve) == "Circle" else wall_layers
                    model.mesh.setTransfiniteCurve(curve, segments + 1)
                model.mesh.setTransfiniteSurface(wall)

            left, right = groups["xleft"][0], groups["xright"][0]
            for end in (left, right):
                for _, curve in model.getBoundary([(2, end)]):
                    vertices = model.getBoundary([(1, curve)])
                    coordinates = [model.getValue(0, vertex, []) for _, vertex in vertices]
                    direction = abs(coordinates[1] - coordinates[0])
                    model.mesh.setTransfiniteCurve(curve, (ny if direction[1] > direction[2] else nz) + 1)
            # Only the master is transfinite. Its translated copy retains the
            # same triangle diagonals, including at symmetric cross-sections.
            model.mesh.setTransfiniteSurface(left)
            translation = np.eye(4)
            translation[0, 3] = length
            model.mesh.setPeriodic(2, [right], [left], translation.ravel())
            for name, tags in groups.items():
                model.addPhysicalGroup(2, tags, name=name)
            model.addPhysicalGroup(3, [volume], name="solid")
            model.mesh.generate(3)
            _verify_port_mesh(model, (left, right), ny, nz)
            mesh = _read_mesh(model, volume, groups)
        finally:
            gmsh.finalize()
    return mesh


def volume_diagnostics(
    mesh: MeshTet, geometry: BeamGeometry, holes: HolePattern | Sequence[Hole]
) -> dict[str, float | int | None]:
    """Compare affine mesh volumes to the exact box and through cylinders.

    The relative errors are absolute. Rim segment counts and radial sagitta
    describe the affine boundary relative to the exact circular geometry.
    These quantities and removed-volume error are undefined without holes.
    """
    placed = holes.holes if isinstance(holes, HolePattern) else tuple(holes)
    box = geometry.length_m * geometry.width_m * geometry.thickness_m
    removed = pi * geometry.thickness_m * sum((hole.diameter_m / 2)**2 for hole in placed)
    solid = box - removed
    meshed = float(np.sum(abs(mesh.mapping().detA)) / 6)
    meshed_removed = box - meshed
    counts = []
    sagittas = []
    if placed:
        z = mesh.p[2]
        for plane in (z.min(), z.max()):
            on_face = abs(z - plane) <= geometry.thickness_m * 1e-10
            x, y = mesh.p[:2, on_face]
            for hole in placed:
                radius = hole.diameter_m / 2
                dx = x - hole.x_m
                on_circle = abs(np.hypot(dx, y) - radius) <= radius * 1e-8
                theta = np.sort(np.arctan2(y[on_circle], dx[on_circle]))
                if len(theta) < 3:
                    raise ValueError("mesh is missing a circular rim for a specified hole")
                counts.append(len(theta))
                max_angle = np.diff(theta, append=theta[0] + 2 * pi).max()
                sagittas.append(radius * (1 - cos(max_angle / 2)))
    return {
        "analytic_solid_volume_m3": solid,
        "meshed_solid_volume_m3": meshed,
        "analytic_removed_volume_m3": removed,
        "meshed_removed_volume_m3": meshed_removed,
        "relative_solid_volume_error": abs(meshed - solid) / solid,
        "relative_removed_volume_error": abs(meshed_removed - removed) / removed if removed else None,
        "rim_segments_min": min(counts, default=None),
        "rim_segments_max": max(counts, default=None),
        "max_radial_sagitta_m": max(sagittas, default=None),
    }
