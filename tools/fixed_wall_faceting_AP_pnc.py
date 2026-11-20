#!/usr/bin/env python3
"""Build and verify a fixed-connectivity affine wall control. Athul Prem | _AP_pnc."""

from __future__ import annotations

import argparse
from collections.abc import Sequence
from dataclasses import replace
import json
from numbers import Integral
from pathlib import Path
import sys

import numpy as np
from skfem import MeshTet

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT / "src"), str(ROOT)]

from phononic.geometry_AP_pnc import BeamGeometry, Hole, validate_hole_clearances
from phononic.mesh_checkpoint_AP_pnc import load_checkpoint
from tools import map_scattering_AP_pnc as native


RIM_EDGES = 48
CHORDS = RIM_EDGES // 2
CAD_TOLERANCE = 1e-8   # Relative radial tolerance and angular tolerance in radians.
MEASURE_RTOL = 1e-10


def load_wall_faceting_source(donor: Path) -> tuple[MeshTet, BeamGeometry, tuple[Hole, ...], dict]:
    """Verify the native map and its mesh artifacts using the saved source identity."""
    donor = donor.resolve()
    checkpoint_path = donor / native.CHECKPOINT
    if not checkpoint_path.is_file() or checkpoint_path.resolve() != checkpoint_path:
        raise ValueError("Choose a native donor checkpoint in its own directory.")
    checkpoint_sha = native._sha(checkpoint_path)
    state = native.read_map_checkpoint(donor)
    inputs = state["inputs"]
    config = inputs["configuration"]
    settings = config["mesh"]
    pattern = inputs["hole_pattern"]
    if settings["kind"] != "layered" or settings["circle_resolution"] != RIM_EDGES or not pattern:
        raise ValueError("The source must be a native layered map with circular 48-segment holes.")
    geometry = BeamGeometry(*(config["geometry"][key] * 1e-9 for key in native.workflow.DIMENSIONS_NM))
    holes = tuple(Hole(**hole) for hole in pattern["holes"])
    interval = tuple(value * 1e-9 for value in settings.get("patterned_x_interval_nm", ())) or None
    shared = state["shared"]
    if not shared:
        raise ValueError("The native map has no saved mesh preparation.")
    if shared["reference_planes_x_m"] != [0.0, geometry.length_m]:
        raise ValueError("The saved map reference planes differ from the configured geometry.")
    mesh_dir, mesh_hash, mesh_inputs = native.workflow._mesh_location(
        Path(), geometry, pattern, settings, inputs["numerical_identity"]
    )
    artifacts = shared["artifacts"]
    manifest_path = native._verified(artifacts["mesh_summary"], donor)
    mesh_path = native._verified(artifacts["mesh"], donor)
    if (shared["mesh_hash"] != mesh_hash or mesh_path != manifest_path.parent / "mesh_AP_pnc.npz"
            or manifest_path != manifest_path.parent / native.RUN_SUMMARY
            or manifest_path.parent.relative_to(donor).parts[1:] != mesh_dir.parts
            or not manifest_path.parent.relative_to(donor).parts[0].startswith("shared-")):
        raise ValueError("The native donor mesh layout or input identity differs from its checkpoint.")
    mesh, manifest = load_checkpoint(manifest_path.parent, mesh_hash)
    if (manifest["mesh_inputs"] != mesh_inputs or manifest["hole_pattern_provenance"] != pattern
            or manifest["mesh_sha256"] != artifacts["mesh"]["sha256"]
            or native._sha(checkpoint_path) != checkpoint_sha):
        raise ValueError("The donor mesh manifest or native checkpoint changed.")
    source = {
        "donor_map": donor.relative_to(ROOT).as_posix() if donor.is_relative_to(ROOT) else str(donor),
        "donor_checkpoint_sha256": checkpoint_sha,
        "donor_source_git_commit": state["source_git_commit"],
        "donor_mesh_hash": mesh_hash,
        "donor_mesh_sha256": artifacts["mesh"]["sha256"],
        "donor_mesh_manifest_sha256": artifacts["mesh_summary"]["sha256"],
        "donor_hole_csv_sha256": pattern["source_sha256"],
        "thickness_layers": settings["thickness_layers"],
        "patterned_x_interval_m": interval,
    }
    return mesh, geometry, holes, source


def _layer_columns(points: np.ndarray, geometry: BeamGeometry, thickness_layers: int):
    """Identify corresponding vertices by exact xy equality across the z levels."""
    z, levels = np.unique(points[2], return_inverse=True)
    xy, columns = np.unique(points[:2].T, axis=0, return_inverse=True)
    expected_z = np.linspace(-geometry.thickness_m / 2, geometry.thickness_m / 2, thickness_layers + 1)
    if z.shape != expected_z.shape or not np.allclose(z, expected_z, atol=0, rtol=MEASURE_RTOL):
        raise ValueError("Expected the original uniform z levels throughout the beam.")
    if len(levels) != len(z) * len(xy) or len(np.unique(levels * len(xy) + columns)) != len(levels):
        raise ValueError("Each xy column must contain one vertex at every z level.")
    node = np.empty((len(z), len(xy)), dtype=int)
    node[levels, columns] = np.arange(len(levels))
    return z, levels, xy, columns, node


def _surface_facets(mesh: MeshTet, geometry: BeamGeometry, levels: np.ndarray, thickness_layers: int):
    """Extract actual exterior hole-wall triangles and the two cap triangulations."""
    tags = mesh.boundaries or {}
    exterior = mesh.boundary_facets()
    if (not {"xleft", "xright", "free"} <= tags.keys()
            or not np.array_equal(np.sort(np.concatenate(list(tags.values()))), exterior)):
        raise ValueError("Boundary tags must partition the actual tetrahedral exterior.")
    triangles = mesh.facets[:, exterior]
    left = np.all(mesh.p[0, triangles] == 0, axis=0)
    right = np.all(mesh.p[0, triangles] == geometry.length_m, axis=0)
    if (not left.any() or not right.any() or not np.array_equal(exterior[left], np.sort(tags["xleft"]))
            or not np.array_equal(exterior[right], np.sort(tags["xright"]))):
        raise ValueError("Expected tagged end traces on the original x planes.")
    bottom = np.all(levels[triangles] == 0, axis=0)
    top = np.all(levels[triangles] == thickness_layers, axis=0)
    side = (np.all(mesh.p[1, triangles] == -geometry.width_m / 2, axis=0)
            | np.all(mesh.p[1, triangles] == geometry.width_m / 2, axis=0))
    wall_facets = exterior[~(left | right | bottom | top | side)]
    if not np.all(np.isin(wall_facets, tags["free"])):
        raise ValueError("The hole walls must belong to the original free exterior.")
    return mesh.facets[:, wall_facets], (triangles[:, bottom], triangles[:, top])


def _cap_boundary_edges(triangles: np.ndarray) -> np.ndarray:
    edges = np.hstack((triangles[[0, 1]], triangles[[0, 2]], triangles[[1, 2]])).T
    edges, counts = np.unique(np.sort(edges, axis=1), axis=0, return_counts=True)
    if np.any(counts > 2):
        raise ValueError("An exterior cap edge has more than two incident cap triangles.")
    return edges[counts == 1]


def _rim_columns(xy: np.ndarray, wall_columns: np.ndarray, hole: Hole) -> np.ndarray:
    relative = (xy[wall_columns] - [hole.x_m, 0.0]) / (hole.diameter_m / 2)
    is_rim = np.abs(np.linalg.norm(relative, axis=1) - 1.0) <= CAD_TOLERANCE
    angles = (np.arctan2(relative[is_rim, 1], relative[is_rim, 0]) + np.pi / RIM_EDGES) % (2 * np.pi)
    order = np.argsort(angles)
    # Start at the +x CAD anchor and alternate counterclockwise on every z level.
    expected = (np.arange(RIM_EDGES) + 0.5) * 2 * np.pi / RIM_EDGES
    if len(order) != RIM_EDGES or not np.allclose(angles[order], expected, rtol=0, atol=CAD_TOLERANCE):
        raise ValueError(f"Hole at x={hole.x_m:g} m lacks one unambiguous regular 48-edge CAD rim.")
    return wall_columns[is_rim][order]


def _verify_rim_connectivity(rings: np.ndarray, walls: np.ndarray, cap_edges: tuple[np.ndarray, np.ndarray]) -> None:
    """Check each wall rectangle's two triangles and its cap junctions by vertex index."""
    wall_set = {tuple(face) for face in np.sort(walls.T, axis=1)}
    if len(wall_set) != walls.shape[1] or len(wall_set) != 2 * RIM_EDGES * (len(rings) - 1):
        raise ValueError("Each through-hole wall requires 96 exterior triangles per slab.")
    for bottom, top in zip(rings[:-1], rings[1:], strict=True):
        for a, b, c, d in zip(bottom, np.roll(bottom, -1), top, np.roll(top, -1), strict=True):
            diagonal_ad = {tuple(sorted(face)) for face in ((a, b, d), (a, c, d))}
            diagonal_bc = {tuple(sorted(face)) for face in ((a, b, c), (b, c, d))}
            if diagonal_ad <= wall_set:
                wall_set -= diagonal_ad
            elif diagonal_bc <= wall_set:
                wall_set -= diagonal_bc
            else:
                raise ValueError("A wall rectangle lacks its two matching exterior triangles.")
    if wall_set:
        raise ValueError("Unexpected wall triangle beyond the expected rim rectangles.")
    for ring, edges in zip((rings[0], rings[-1]), cap_edges, strict=True):
        expected = {tuple(sorted(edge)) for edge in zip(ring, np.roll(ring, -1), strict=True)}
        actual = edges[np.isin(edges, ring).any(axis=1)]
        if {tuple(edge) for edge in actual} != expected:
            raise ValueError("Actual cap mesh and rim do not share exactly the 48 cycle edges.")


def _area(rim: np.ndarray, hole: Hole) -> float:
    """Shoelace area after shifting the origin to the CAD center to avoid cancellation."""
    x, y = rim - np.array([[hole.x_m], [0.0]])
    return float(0.5 * np.sum(x * np.roll(y, -1) - y * np.roll(x, -1)))


def _signed_determinants(points: np.ndarray, tets: np.ndarray) -> np.ndarray:
    a, b, c, d = points[:, tets].transpose(1, 2, 0)
    return np.einsum("ij,ij->i", b - a, np.cross(c - a, d - a))


def _cap_area(points: np.ndarray, triangles: np.ndarray) -> float:
    a, b, c = points[:2, triangles].transpose(1, 2, 0)
    u, v = b - a, c - a
    return float(0.5 * np.sum(np.abs(u[:, 0] * v[:, 1] - u[:, 1] * v[:, 0])))


def _protected_traces(mesh: MeshTet, moved: np.ndarray, touched: np.ndarray,
                      geometry: BeamGeometry, interval: tuple[float, float] | None) -> dict:
    end_counts = {}
    for tag in ("xleft", "xright"):
        faces = mesh.boundaries[tag]
        if np.any(moved[mesh.facets[:, faces]]):
            raise ValueError("The control would move a reference-plane vertex.")
        end_counts[tag] = len(faces)
    record = {"reference_planes_x_m": [0.0, geometry.length_m], "unchanged_end_facets": end_counts}
    if interval is None:
        return record
    start, stop = interval
    if not 0 < start < stop < geometry.length_m:
        raise ValueError("The patterned interval must lie strictly inside the beam.")
    x = mesh.p[0, mesh.t]
    low, high = x.min(axis=0), x.max(axis=0)
    lead_cells = []
    cut_facets = []
    for cut, lead in ((start, high <= start), (stop, low >= stop)):
        if not lead.any() or np.any((low < cut) & (high > cut)) or np.any(touched[lead]):
            raise ValueError("Original lead cells must remain isolated at the regional cut.")
        cut_faces = np.flatnonzero(np.all(mesh.p[0, mesh.facets] == cut, axis=0))
        if not len(cut_faces) or np.any(mesh.f2t[:, cut_faces] < 0) or np.any(moved[mesh.facets[:, cut_faces]]):
            raise ValueError("The original regional cut must be a fixed two-sided trace.")
        lead_cells.append(int(np.count_nonzero(lead)))
        cut_facets.append(len(cut_faces))
    return {**record, "regional_cuts_x_m": list(interval), "unchanged_cut_facets": cut_facets,
            "unchanged_lead_tetrahedra": lead_cells}


def make_fixed_wall_faceting_control(
    mesh: MeshTet, geometry: BeamGeometry, holes: Sequence[Hole], *, thickness_layers: int,
    patterned_x_interval_m: tuple[float, float] | None = None,
) -> tuple[MeshTet, dict]:
    """Project alternate original rim nodes to adjacent chords with fixed mesh indices.

    The affine control has 24 physical wall sides and 48 finite-element edges
    on every ring. Original corner-node and tetrahedron order, z coordinates,
    and physical tags are preserved. This function performs no assembly or solve.
    """
    if not isinstance(mesh, MeshTet) or not mesh.affine or not np.all(np.isfinite(mesh.p)):
        raise ValueError("An affine finite-coordinate tetrahedral mesh is required.")
    if not isinstance(thickness_layers, Integral) or isinstance(thickness_layers, bool) or thickness_layers < 1:
        raise ValueError("The source must have at least one whole thickness layer.")
    validate_hole_clearances(geometry, holes)
    if patterned_x_interval_m is not None and any(
        hole.x_m - hole.diameter_m / 2 <= patterned_x_interval_m[0]
        or hole.x_m + hole.diameter_m / 2 >= patterned_x_interval_m[1] for hole in holes
    ):
        raise ValueError("Every circular hole must lie strictly inside the regional cuts.")
    z, levels, xy, columns, node = _layer_columns(mesh.p, geometry, thickness_layers)
    tet_levels = levels[mesh.t]
    slabs = tet_levels.min(axis=0)
    if not np.all(tet_levels.max(axis=0) == slabs + 1):
        raise ValueError("Each tetrahedron must span a single original z slab.")
    walls, caps = _surface_facets(mesh, geometry, levels, thickness_layers)
    cap_edges = tuple(_cap_boundary_edges(triangles) for triangles in caps)
    wall_columns = np.unique(columns[walls])
    hole_owner = np.full(mesh.nvertices, -1, dtype=int)
    rings_by_hole = []
    for index, hole in enumerate(holes):
        rings = node[:, _rim_columns(xy, wall_columns, hole)]
        if np.any(hole_owner[rings] != -1):
            raise ValueError("A mesh rim vertex matches more than one CAD hole.")
        hole_owner[rings] = index
        rings_by_hole.append(rings)
    owners = hole_owner[walls]
    if np.any(owners < 0) or np.any(owners != owners[:1]):
        raise ValueError("Some exterior hole-wall triangles do not belong to a single CAD rim.")

    points = mesh.p.copy()
    moved = np.zeros(mesh.nvertices, dtype=bool)
    hole_records = []
    for index, (hole, rings) in enumerate(zip(holes, rings_by_hole, strict=True)):
        _verify_rim_connectivity(rings, walls[:, owners[0] == index], cap_edges)
        moved[rings[:, 1::2]] = True
        original = mesh.p[:2, rings[0]]
        retained = original[:, ::2]
        midpoints = (retained + np.roll(retained, -1, axis=1)) / 2
        points[:2, rings[:, 1::2]] = midpoints[:, None, :]
        before = _area(original, hole)
        chord_area = _area(retained, hole)
        projected = np.array([_area(points[:2, ring], hole) for ring in rings])
        radius = hole.diameter_m / 2
        analytic = [count / 2 * radius**2 * np.sin(2 * np.pi / count) for count in (RIM_EDGES, CHORDS)]
        if (not 0 < chord_area < before or not np.allclose([before, chord_area], analytic, rtol=CAD_TOLERANCE, atol=0)
                or not np.allclose(projected, chord_area, rtol=MEASURE_RTOL, atol=0)):
            raise ValueError("The rim areas disagree with the original circle and projected chord polygons.")
        hole_records.append({
            "hole_index": index + 1, "x_m": hole.x_m, "diameter_m": hole.diameter_m,
            "original_area_m2": before, "chord_polygon_area_m2": chord_area,
            "projected_area_m2": float(projected[0]),
            "max_projected_ring_area_error_m2": float(np.max(np.abs(projected - chord_area))),
            "void_volume_decrease_m3": (before - chord_area) * geometry.thickness_m,
        })

    if (not np.array_equal(np.any(points != mesh.p, axis=0), moved)
            or not np.array_equal(points[2], mesh.p[2])):
        raise ValueError("Only the selected alternate xy rim vertices may move.")
    displacement = np.linalg.norm(points[:2, moved] - mesh.p[:2, moved], axis=0)
    touched = moved[mesh.t].any(axis=0)
    traces = _protected_traces(mesh, moved, touched, geometry, patterned_x_interval_m)
    before_det = _signed_determinants(mesh.p, mesh.t)
    after_det = _signed_determinants(points, mesh.t)
    if (np.any(~np.isfinite(before_det)) or np.any(~np.isfinite(after_det))
            or np.any(before_det == 0) or np.any(np.sign(before_det) != np.sign(after_det))):
        raise ValueError("The control produces a nonpositive relative signed tetrahedral determinant.")
    if not np.array_equal(before_det[~touched], after_det[~touched]):
        raise ValueError("A tetrahedron with all vertices fixed changed its determinant.")

    before_area = geometry.length_m * geometry.width_m - sum(h["original_area_m2"] for h in hole_records)
    after_area = geometry.length_m * geometry.width_m - sum(h["chord_polygon_area_m2"] for h in hole_records)
    before_volume = np.bincount(slabs, weights=np.abs(before_det) / 6, minlength=thickness_layers)
    after_volume = np.bincount(slabs, weights=np.abs(after_det) / 6, minlength=thickness_layers)
    for coords, area, volume in ((mesh.p, before_area, before_volume), (points, after_area, after_volume)):
        cap_areas = [_cap_area(coords, triangles) for triangles in caps]
        if (not np.allclose(cap_areas, area, rtol=MEASURE_RTOL, atol=0)
                or not np.allclose(volume, area * np.diff(z), rtol=MEASURE_RTOL, atol=0)):
            raise ValueError("Cap areas or tetrahedral slab volumes disagree with the rim polygons.")
    volume_change = float(np.sum(np.abs(after_det[touched]) - np.abs(before_det[touched])) / 6)
    expected_change = float(sum(h["void_volume_decrease_m3"] for h in hole_records))
    if not np.isclose(volume_change, expected_change, rtol=MEASURE_RTOL, atol=0):
        raise ValueError("Touched tetrahedra volume change disagrees with the removed void volume.")

    control = replace(mesh, doflocs=points)
    if (not np.array_equal(control.p, points) or not np.array_equal(control.t, mesh.t)
            or not np.array_equal(control.facets, mesh.facets)):
        raise ValueError("Reconstructing the affine control changed geometry or connectivity.")
    report = {
        "author": "Athul Prem", "file_tag": "_AP_pnc", "schema_version": 1,
        "scope": "Geometry-only fixed-connectivity control on the original affine tetrahedra.",
        "spatial_convergence_established": False, "frequency_convergence_established": False,
        "control_source_sha256": native._sha(Path(__file__).resolve()),
        "tetrahedra_sha256": native._array_digest(mesh.t),
        "original_points_sha256": native._array_digest(mesh.p), "control_points_sha256": native._array_digest(points),
        "rim_anchor": "Positive x radius, proceeding counterclockwise with odd zero-based indices moved.",
        "original_physical_rim_sides": RIM_EDGES, "control_physical_rim_sides": CHORDS,
        "finite_element_rim_edges_per_ring": RIM_EDGES, "wall_triangles_per_hole_per_slab": 2 * RIM_EDGES,
        "z_planes_m": z.tolist(), "mesh_vertices": int(mesh.nvertices), "mesh_tetrahedra": int(mesh.nelements),
        "boundary_facet_counts": {name: len(ids) for name, ids in mesh.boundaries.items()},
        "subdomain_cell_counts": {name: len(ids) for name, ids in (mesh.subdomains or {}).items()},
        "moved_vertices": int(moved.sum()), "moved_vertices_per_z_plane": np.bincount(levels[moved]).tolist(),
        "min_vertex_displacement_m": float(displacement.min()), "max_vertex_displacement_m": float(displacement.max()),
        "touched_tetrahedra": int(touched.sum()), "touched_tetrahedra_per_slab": np.bincount(slabs[touched]).tolist(),
        "minimum_absolute_original_determinant_m3": float(np.min(np.abs(before_det))),
        "minimum_absolute_control_determinant_m3": float(np.min(np.abs(after_det))),
        "minimum_relative_signed_tet_determinant": float(np.min(after_det[touched] / before_det[touched])),
        "maximum_relative_signed_tet_determinant": float(np.max(after_det[touched] / before_det[touched])),
        "original_solid_area_m2": before_area, "projected_solid_area_m2": after_area,
        "original_tetrahedral_volume_per_slab_m3": before_volume.tolist(),
        "projected_tetrahedral_volume_per_slab_m3": after_volume.tolist(),
        "expected_solid_volume_increase_m3": expected_change, "measured_solid_volume_increase_m3": volume_change,
        "holes": hole_records, **traces,
    }
    return control, report


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--from-map", required=True, type=Path, help="Existing native layered map with 48-edge rims.")
    parser.add_argument("--receipt", type=Path, help="Optional new JSON file outside the source map.")
    args = parser.parse_args(argv)
    try:
        donor = args.from_map.resolve()
        mesh, geometry, holes, source = load_wall_faceting_source(donor)
        _, report = make_fixed_wall_faceting_control(
            mesh, geometry, holes, thickness_layers=source["thickness_layers"],
            patterned_x_interval_m=source["patterned_x_interval_m"],
        )
        payload = json.dumps({**report, "source": source}, indent=2, allow_nan=False) + "\n"
        if args.receipt is None:
            print(payload, end="")
        else:
            if args.receipt.resolve().is_relative_to(donor):
                raise ValueError("The receipt must be outside the source map.")
            with args.receipt.open("x", encoding="utf-8") as stream:
                stream.write(payload)
    except (OSError, ValueError, TypeError, KeyError) as exc:
        parser.exit(1, f"Fixed wall faceting control failed. {exc}\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
