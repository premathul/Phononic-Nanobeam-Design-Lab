"""Three-dimensional straight beam geometry in SI units.

Author: Athul Prem | _AP_pnc
"""

from __future__ import annotations

import csv
from dataclasses import dataclass
from hashlib import sha256
from io import StringIO
from math import isfinite
from numbers import Integral, Real
from pathlib import Path
from threading import Lock
from typing import Sequence

import numpy as np
from skfem import MeshTet

_METERS_PER_NANOMETER = 1e-9
_METERS_PER_CAD_UNIT = 1e-6
_GMSH_LOCK = Lock()


def _finite_real(value: float) -> bool:
    return isinstance(value, Real) and not isinstance(value, bool) and isfinite(value)


@dataclass(frozen=True)
class BeamGeometry:
    """Beam dimensions along laboratory x (length), y (width), and z (thickness)."""

    length_m: float
    width_m: float
    thickness_m: float

    def __post_init__(self) -> None:
        for name, value in vars(self).items():
            if not _finite_real(value) or value <= 0:
                raise ValueError(f"{name} must be positive and finite")


@dataclass(frozen=True)
class Hole:
    """A through-hole on y = 0 with its axis parallel to z."""

    x_m: float
    diameter_m: float
    region: str = ""

    def __post_init__(self) -> None:
        if not _finite_real(self.x_m):
            raise ValueError("x_m must be finite")
        if not _finite_real(self.diameter_m) or self.diameter_m <= 0:
            raise ValueError("diameter_m must be positive and finite")


@dataclass(frozen=True)
class HolePattern:
    """Holes in source order, with positions relative to the source origin."""

    holes: tuple[Hole, ...]
    source_path: Path
    source_sha256: str


def read_hole_pattern(path: str | Path) -> HolePattern:
    """Read a nanometer CSV (position_nm or x_nm, diameter_nm) into meters.

    The source origin is retained, including negative positions in the historical
    array. Use an explicit offset when placing the pattern in a beam with x >= 0.
    """
    path = Path(path)
    source = path.read_bytes()
    reader = csv.DictReader(StringIO(source.decode("utf-8-sig"), newline=""))
    headers = reader.fieldnames or []
    position_columns = [column for column in headers if column in {"position_nm", "x_nm"}]
    if len(position_columns) != 1 or headers.count("diameter_nm") != 1:
        raise ValueError(f"{path} requires diameter_nm and one of position_nm or x_nm")
    x_column = position_columns[0]
    holes = []
    for row in reader:
        try:
            holes.append(Hole(
                float(row[x_column]) * _METERS_PER_NANOMETER,
                float(row["diameter_nm"]) * _METERS_PER_NANOMETER,
                row.get("region") or "",
            ))
        except (TypeError, ValueError) as error:
            raise ValueError(f"{path}, line {reader.line_num}: {error}") from error
    if not holes:
        raise ValueError(f"{path} must contain at least one hole")
    return HolePattern(tuple(holes), path, sha256(source).hexdigest())


def validate_hole_clearances(
    geometry: BeamGeometry, holes: HolePattern | Sequence[Hole]
) -> dict[str, float | None]:
    """Return minimum positive ligaments in meters for holes placed on [0, L]."""
    placed_holes = holes.holes if isinstance(holes, HolePattern) else holes
    ordered = sorted(placed_holes, key=lambda hole: hole.x_m)
    if not ordered:
        raise ValueError("at least one hole is required")
    side = min((geometry.width_m - hole.diameter_m) / 2 for hole in ordered)
    end = min(min(hole.x_m, geometry.length_m - hole.x_m) - hole.diameter_m / 2
              for hole in ordered)
    between = min((right.x_m - left.x_m - (left.diameter_m + right.diameter_m) / 2
                   for left, right in zip(ordered, ordered[1:])), default=None)
    if side <= 0 or end <= 0:
        raise ValueError("holes must have positive clearance from the beam sides and ends")
    if between is not None and between <= 0:
        raise ValueError("adjacent holes must have positive clearance")
    return {
        "min_side_ligament_m": side,
        "min_end_ligament_m": end,
        "min_interhole_gap_m": between,
    }


def make_straight_mesh(
    geometry: BeamGeometry, cells: Sequence[int] = (12, 6, 3)
) -> MeshTet:
    """Mesh the solid box with six linear tetrahedra per Cartesian cell.

    The beam extends from x=0 to x=L, with y and z centered on zero. The
    boundary labels xleft, xright, and free partition the exterior faces.
    """
    if len(cells) != 3 or any(
        not isinstance(n, Integral) or isinstance(n, bool) or n <= 0 for n in cells
    ):
        raise ValueError("cells must contain three positive integers")
    nx, ny, nz = cells
    length, width, thickness = geometry.length_m, geometry.width_m, geometry.thickness_m
    mesh = MeshTet.init_tensor(
        np.linspace(0, length, nx + 1),
        np.linspace(-width / 2, width / 2, ny + 1),
        np.linspace(-thickness / 2, thickness / 2, nz + 1),
    ).oriented()
    # Classify whole facets at the exact Cartesian endpoints. An absolute
    # default tolerance could merge distinct surfaces on nanoscale meshes.
    boundary = mesh.boundary_facets()
    x = mesh.p[0, mesh.facets[:, boundary]]
    left = np.all(x == 0, axis=0)
    right = np.all(x == length, axis=0)
    return mesh.with_boundaries({
        "xleft": boundary[left],
        "xright": boundary[right],
        "free": boundary[~(left | right)],
    })


def make_perforated_mesh(
    geometry: BeamGeometry, holes: HolePattern | Sequence[Hole], mesh_size_m: float
) -> MeshTet:
    """Cut exact circular cylinders in Gmsh and return a tagged linear mesh in meters.

    Supply hole centers in the beam's x=0 to x=L frame. Gmsh and meshio are
    optional dependencies for this function. The mesh uses planar facets to
    approximate the exact cylinder walls, which need geometric refinement.
    """
    if not _finite_real(mesh_size_m) or mesh_size_m <= 0:
        raise ValueError("mesh_size_m must be positive and finite")
    placed_holes = holes.holes if isinstance(holes, HolePattern) else holes
    validate_hole_clearances(geometry, placed_holes)

    import gmsh
    import meshio
    from skfem.io import from_meshio

    length = geometry.length_m / _METERS_PER_CAD_UNIT
    width = geometry.width_m / _METERS_PER_CAD_UNIT
    thickness = geometry.thickness_m / _METERS_PER_CAD_UNIT
    with _GMSH_LOCK:
        if gmsh.isInitialized():
            raise RuntimeError("perforated meshing requires its own Gmsh session")
        gmsh.initialize(readConfigFiles=False, interruptible=False)
        try:
            gmsh.option.setNumber("General.Terminal", 0)
            gmsh.option.setNumber("General.NumThreads", 1)
            gmsh.model.add("perforated_beam_AP_pnc")
            occ = gmsh.model.occ
            beam = occ.addBox(0, -width / 2, -thickness / 2, length, width, thickness)
            # Extend each cutter beyond both broad faces for a complete cut.
            cylinders = [(3, occ.addCylinder(hole.x_m / _METERS_PER_CAD_UNIT, 0,
                                            -thickness, 0, 0, 2 * thickness,
                                            hole.diameter_m / (2 * _METERS_PER_CAD_UNIT)))
                         for hole in placed_holes]
            volumes, _ = occ.cut([(3, beam)], cylinders)
            occ.synchronize()
            if len(volumes) != 1 or volumes[0][0] != 3:
                raise RuntimeError("perforation must produce one solid volume")
            volume = volumes[0][1]
            groups = {"xleft": [], "xright": [], "free": []}
            port_tolerance = length * 1e-10
            for _, tag in gmsh.model.getBoundary(volumes):
                center_x, _, _ = occ.getCenterOfMass(2, tag)
                name = "free"
                if gmsh.model.getType(2, tag) == "Plane":
                    if abs(center_x) <= port_tolerance:
                        name = "xleft"
                    elif abs(center_x - length) <= port_tolerance:
                        name = "xright"
                groups[name].append(tag)
            if not all(groups.values()):
                raise RuntimeError("perforated solid is missing an exterior surface group")
            for name, tags in groups.items():
                gmsh.model.addPhysicalGroup(2, tags, name=name)
            gmsh.model.addPhysicalGroup(3, [volume], name="solid")
            gmsh.option.setNumber("Mesh.MeshSizeMax", mesh_size_m / _METERS_PER_CAD_UNIT)
            gmsh.model.mesh.generate(3)

            node_tags, xyz, _ = gmsh.model.mesh.getNodes(returnParametricCoord=False)
            order = np.argsort(node_tags)
            sorted_tags = node_tags[order]
            points = xyz.reshape(-1, 3)[order] * _METERS_PER_CAD_UNIT
            _, tet_nodes = gmsh.model.mesh.getElementsByType(4, volume)
            if len(tet_nodes) == 0:
                raise RuntimeError("Gmsh produced no linear tetrahedra")
            tets = np.searchsorted(sorted_tags, tet_nodes).reshape(-1, 4)
            triangles = []
            cell_sets = {}
            start = 0
            for name, tags in groups.items():
                nodes = np.concatenate([gmsh.model.mesh.getElementsByType(2, tag)[1]
                                        for tag in tags])
                faces = np.searchsorted(sorted_tags, nodes).reshape(-1, 3)
                triangles.append(faces)
                stop = start + len(faces)
                cell_sets[name] = [np.empty(0, dtype=int), np.arange(start, stop)]
                start = stop
            cell_sets["solid"] = [np.arange(len(tets)), np.empty(0, dtype=int)]
        finally:
            gmsh.finalize()

    imported = meshio.Mesh(points, [("tetra", tets), ("triangle", np.vstack(triangles))],
                           cell_sets=cell_sets)
    # Import the element and facet tags from the Gmsh surface groups. Each
    # tagged facet is exterior, so the tetrahedron fixes its outward normal.
    mesh = from_meshio(imported, ignore_orientation=True).oriented()
    boundaries = mesh.boundaries
    if boundaries is None or set(boundaries) != set(groups):
        raise RuntimeError("failed to import all three Gmsh surface groups")
    tagged = np.sort(np.concatenate(list(boundaries.values())))
    if not np.array_equal(tagged, mesh.boundary_facets()):
        raise RuntimeError("Gmsh surface groups must partition the tetrahedral exterior")
    return mesh
