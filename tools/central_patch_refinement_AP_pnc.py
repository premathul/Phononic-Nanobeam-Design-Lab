#!/usr/bin/env python3
"""Isolate the nested 26-hole taper and cavity refinement or its complement. Athul Prem | _AP_pnc."""

from __future__ import annotations

import argparse
from collections import defaultdict, deque
import hashlib
import json
from pathlib import Path
import sys

import numpy as np
from skfem import MeshTet

ROOT = Path(__file__).resolve().parents[1]
if __name__ == "__main__":
    sys.path.insert(0, str(ROOT))

from tools import map_scattering_AP_pnc as native


def _load_source(directory):
    directory = directory.resolve()
    checkpoint = native._link(directory / native.CHECKPOINT, directory)
    source = native.read_map_checkpoint(directory)
    shared = source["shared"]
    artifacts = shared["artifacts"]
    manifest_path = native._verified(artifacts["mesh_summary"], directory)
    mesh_path = native._verified(artifacts["mesh"], directory)
    inputs = source["inputs"]
    config = inputs["configuration"]
    pattern = inputs["hole_pattern"]
    settings = config["mesh"]
    geometry = native.workflow.BeamGeometry(*(config["geometry"][key] * 1e-9
                                             for key in native.workflow.DIMENSIONS_NM))
    identity = inputs["numerical_identity"]
    assembly_revision = source["source_git_commit"]
    reuse = source.get("preparation_reuse")
    if reuse is not None:
        receipt = reuse["receipt"]
        receipt_bytes = (json.dumps(receipt, indent=2, allow_nan=False) + "\n").encode("utf-8")
        if (reuse["method"] != "windows_to_linux_native_preparation"
                or reuse["receipt_sha256"] != hashlib.sha256(receipt_bytes).hexdigest()
                or receipt["donor_shared_sha256"] != native.workflow.digest(shared)):
            raise ValueError("The original assembly environment does not match the transfer receipt.")
        identity = receipt["donor_numerical_identity"]
        assembly_revision = receipt["donor_source_git_commit"]
    _, mesh_hash, mesh_inputs = native.workflow._mesh_location(
        Path(), geometry, pattern, settings, identity)
    if shared["mesh_hash"] != mesh_hash or mesh_path != manifest_path.with_name("mesh_AP_pnc.npz"):
        raise ValueError("The native mesh paths or mesh input identity disagree.")
    mesh, manifest = native.workflow.load_checkpoint(manifest_path.parent, mesh_hash)
    if (manifest["mesh_inputs"] != mesh_inputs or manifest["hole_pattern_provenance"] != pattern
            or manifest["mesh_sha256"] != artifacts["mesh"]["sha256"]
            or native._sha(directory / native.CHECKPOINT) != checkpoint["sha256"]):
        raise ValueError("The original mesh manifest or source checkpoint disagrees.")
    pins = {
        "directory": directory.relative_to(ROOT).as_posix() if directory.is_relative_to(ROOT) else str(directory),
        "source_git_commit": source["source_git_commit"], "checkpoint": checkpoint,
        "assembly_source_git_commit": assembly_revision, "assembly_platform": identity["platform"],
        "mesh_hash": mesh_hash, "mesh": artifacts["mesh"], "mesh_summary": artifacts["mesh_summary"],
    }
    return mesh, config, pattern, shared["reference_planes_x_m"], pins


def load_nested_patch_sources(coarse_directory: Path, fine_directory: Path):
    """Load the exact native mesh checkpoints with their provenance pins."""
    coarse, coarse_config, coarse_pattern, planes, coarse_pins = _load_source(coarse_directory)
    fine, fine_config, fine_pattern, fine_planes, fine_pins = _load_source(fine_directory)
    coarse_settings, fine_settings = (config["mesh"].copy() for config in (coarse_config, fine_config))
    coarse_target = coarse_settings.pop("central_max_edge_nm")
    fine_target = fine_settings.pop("central_max_edge_nm")
    if (coarse_settings != fine_settings or coarse_target <= fine_target or coarse_pattern != fine_pattern
            or len(coarse_pattern["holes"]) != 26 or coarse_settings["kind"] != "layered"
            or coarse_settings["element_order"] != 2 or not coarse_settings.get("patterned_x_interval_nm")
            or planes != fine_planes or planes != [0.0, coarse_config["geometry"]["length_nm"] * 1e-9]
            or any(coarse_config[key] != fine_config[key] for key in ("geometry", "material", "analysis"))):
        raise ValueError("Choose both nested P2 meshes for the same original 26-hole geometry and analysis.")
    holes = coarse_pattern["holes"]
    cuts = [(holes[a]["x_m"] + holes[b]["x_m"]) / 2 for a, b in ((7, 8), (17, 18))]
    return coarse, fine, {
        "coarse": {**coarse_pins, "central_max_edge_nm": coarse_target},
        "fine": {**fine_pins, "central_max_edge_nm": fine_target},
        "hole_pattern_sha256": coarse_pattern["source_sha256"],
        "reference_planes_x_m": planes, "nominal_mirror_cuts_x_m": cuts,
        "regional_cuts_x_m": [x * 1e-9 for x in coarse_settings["patterned_x_interval_nm"]],
    }


def _footprint(mesh):
    """Extract native ordered columns and the planar triangle of every tetrahedron."""
    z = np.unique(mesh.p[2])
    if not mesh.affine or len(z) < 2 or mesh.nvertices % len(z):
        raise ValueError("Expected native affine tetrahedra in at least one complete layer.")
    n = mesh.nvertices // len(z)
    xy = mesh.p[:2, :n].T
    if (not np.isfinite(mesh.p).all() or len(np.unique(xy, axis=0)) != n
            or not np.array_equal(mesh.p[2], np.repeat(z, n))
            or not np.array_equal(mesh.p[:2], np.tile(xy.T, (1, len(z))))):
        raise ValueError("Every native xy vertex must repeat in the same order across all z levels.")
    t = mesh.t.T
    planar = np.sort(t[np.count_nonzero(t < n, axis=1) == 3], axis=1)[:, :3]
    by_column = {tuple(face): i for i, face in enumerate(planar)}
    owners = np.array([by_column.get(tuple(sorted(set(face))), -1) for face in t % n])
    low, high = (t // n).min(axis=1), (t // n).max(axis=1)
    if (not len(planar) or len(by_column) != len(planar) or np.any(owners < 0)
            or np.any(high != low + 1)
            or not np.all(np.bincount(low * len(planar) + owners, minlength=(len(z) - 1) * len(planar)) == 3)
            or not np.all(mesh.mapping().detA > 0)):
        raise ValueError("The donor does not contain three positive native tetrahedra per triangle per layer.")
    return z, xy, planar, owners


def _descendant_parents(coarse_triangles, fine_triangles):
    # Centroids identify the parent. All three child vertices and the total
    # child area must then agree with the candidate parent independently.
    centroids = fine_triangles.mean(axis=1)
    parents = np.full(len(centroids), -1)
    fine_areas = np.abs(np.linalg.det(fine_triangles[:, 1:] - fine_triangles[:, :1]))
    tolerance = 1e-10
    for index, vertices in enumerate(coarse_triangles):
        possible = np.flatnonzero(np.all((centroids >= vertices.min(axis=0))
                                         & (centroids <= vertices.max(axis=0)), axis=1))
        transform = vertices[1:] - vertices[0]
        uv = np.linalg.solve(transform.T, (centroids[possible] - vertices[0]).T).T
        selected = possible[(uv > 0).all(axis=1) & (uv.sum(axis=1) < 1)]
        coordinates = (fine_triangles[selected] - vertices[0]).reshape(-1, 2)
        local = np.linalg.solve(transform.T, coordinates.T).T
        if (not len(selected) or np.any(parents[selected] >= 0) or np.any(local < -tolerance)
                or np.any(local.sum(axis=1) > 1 + tolerance)
                or not np.isclose(fine_areas[selected].sum(), abs(np.linalg.det(transform)),
                                  atol=0, rtol=tolerance)):
            raise ValueError("The fine footprint is not an area-preserving subdivision of the coarse footprint.")
        parents[selected] = index
    if np.any(parents < 0):
        raise ValueError("A fine triangle is outside every coarse footprint triangle.")
    return parents


def _edges(face):
    a, b, c = face  # Footprint and facet vertex IDs are sorted.
    return (a, b), (a, c), (b, c)


def _compatible_patch(coarse, fine, inside):
    # The two sides of every split coarse edge must use the same level.
    # Remove whole split-connected components which touch the frozen region.
    fine_edges = {edge for triangle in fine for edge in _edges(triangle)}
    split_edges = defaultdict(list)
    for i, triangle in enumerate(coarse):
        for edge in _edges(triangle):
            if edge not in fine_edges:
                split_edges[edge].append(i)
    neighbors = defaultdict(list)
    for incident in split_edges.values():
        if len(incident) == 2:
            a, b = incident
            neighbors[a].append(b)
            neighbors[b].append(a)
        elif len(incident) > 2:
            raise ValueError("More than two coarse triangles share an edge.")
    active = inside.copy()
    queue = deque(np.flatnonzero(~inside))
    while queue:
        for index in neighbors[queue.popleft()]:
            if active[index]:
                active[index] = False
                queue.append(index)
    return active


def _facet_keys(faces):
    return {tuple(face) for face in np.sort(faces.T, axis=1)}


def _frontier(mesh, active):
    first, second = mesh.f2t
    boundary = (second >= 0) & (active[first] != active[second])
    return mesh.facets[:, boundary]


def _surface(mesh):
    ids = mesh.boundary_facets()
    points = mesh.p[:, mesh.facets[:, ids]]
    x, y, z = points
    start, end = mesh.p[0].min(), mesh.p[0].max()
    left, right = np.all(x == start, axis=0), np.all(x == end, axis=0)
    side = np.all(y == mesh.p[1].min(), axis=0) | np.all(y == mesh.p[1].max(), axis=0)
    rim = (np.ptp(z, axis=0) > 0) & ~(left | right | side)
    return {"xleft": ids[left], "xright": ids[right], "free": ids[~(left | right)]}, ids[rim]


def _p2_dofs(faces):
    vertices = np.unique(faces)
    edges = {edge for triangle in np.sort(faces.T, axis=1) for edge in _edges(triangle)}
    return 3 * (len(vertices) + len(edges))


def _tetrahedron_rows(tetrahedra):
    rows = np.sort(tetrahedra.T.astype("<i8", copy=False), axis=1)
    return rows[np.lexsort(rows.T[::-1])]


def make_central_patch_control(coarse: MeshTet, fine: MeshTet, cuts_m: tuple[float, float],
                               *, region: str = "central"):
    """Use exact fine donor cells on one side of a conforming coarse trace.

    The central control freezes whole coarse triangles intersected by either
    cut. The complementary control assigns those and every other frozen
    parent to its refined region. Both use the same partition of the native
    bottom footprint and copy tetrahedra in their original local order.
    """
    if region not in ("central", "complement"):
        raise ValueError("Choose the central or complement region.")
    z, xy, planar, coarse_owner = _footprint(coarse)
    z_fine, xy_fine, planar_fine, fine_owner = _footprint(fine)
    lower, upper = cuts_m
    if not xy[:, 0].min() < lower < upper < xy[:, 0].max():
        raise ValueError("Place both ordered patch cuts strictly inside the beam.")
    if not np.array_equal(z, z_fine) or not np.array_equal(xy, xy_fine[:len(xy)]):
        raise ValueError("The two footprints must share coarse vertex coordinates, order and z planes.")
    parents = _descendant_parents(xy[planar], xy_fine[planar_fine])
    child_counts = np.bincount(parents, minlength=len(planar))
    x = xy[planar, 0]
    low, high = x.min(axis=1), x.max(axis=1)
    inside = (low >= lower) & (high <= upper)
    central = _compatible_patch(planar, planar_fine, inside)
    active = central if region == "central" else ~central
    split_parents = child_counts > 1
    changed = active & split_parents
    if not changed.any():
        raise ValueError(f"No nested refinement lies inside the conforming {region} patch.")
    keep = ~changed[coarse_owner]
    fine_parents = parents[fine_owner]
    add = changed[fine_parents]

    # Keep all coarse IDs. Append only the used fine-only columns so the
    # geometry on each unchanged element and on the shared trace stays exact.
    n, n_fine = len(xy), len(xy_fine)
    fine_vertices = np.unique(fine.t[:, add])
    new = fine_vertices[fine_vertices % n_fine >= n]
    to_control = np.full(fine.nvertices, -1)
    inherited = (np.arange(len(z))[:, None] * n_fine + np.arange(n)).ravel()
    # Recombine the two disjoint sets of replacement cells with all untouched
    # cells, in the native fine vertex numbering. Compare exact connectivity.
    central_add = (central & split_parents)[fine_parents]
    complement_add = (~central & split_parents)[fine_parents]
    recombined = _tetrahedron_rows(np.column_stack((inherited[coarse.t[:, ~split_parents[coarse_owner]]],
                                                    fine.t[:, central_add], fine.t[:, complement_add])))
    if not np.array_equal(recombined, _tetrahedron_rows(fine.t)):
        raise ValueError("The two parent refinements do not reproduce the complete fine donor topology.")
    to_control[inherited] = np.arange(coarse.nvertices)
    to_control[new] = coarse.nvertices + np.arange(len(new))
    points = np.column_stack((coarse.p, fine.p[:, new]))
    tets = np.column_stack((coarse.t[:, keep], to_control[fine.t[:, add]]))
    control = MeshTet(np.ascontiguousarray(points), np.ascontiguousarray(tets))
    if (not np.array_equal(control.p[:, control.t[:, :keep.sum()]], coarse.p[:, coarse.t[:, keep]])
            or not np.array_equal(control.p[:, control.t[:, keep.sum():]], fine.p[:, fine.t[:, add]])
            or not np.array_equal(np.unique(control.t), np.arange(points.shape[1]))
            or not np.all(control.mapping().detA > 0)
            or np.any(np.bincount(control.t2f.ravel()) > 2)):
        raise ValueError("The assembled patch lost donor cell geometry, positivity or facet incidence.")

    coarse_surface, coarse_rim = _surface(coarse)
    surface, rim = _surface(control)
    before = coarse.boundary_facets()
    donor = fine.boundary_facets()
    expected_boundary = np.column_stack((coarse.facets[:, before[keep[coarse.f2t[0, before]]]],
                                         to_control[fine.facets[:, donor[add[fine.f2t[0, donor]]]]]))
    if (_facet_keys(expected_boundary) != _facet_keys(control.facets[:, control.boundary_facets()])
            or _facet_keys(coarse.facets[:, coarse_rim]) != _facet_keys(control.facets[:, rim])
            or any(_facet_keys(coarse.facets[:, coarse_surface[tag]]) != _facet_keys(control.facets[:, surface[tag]])
                   for tag in ("xleft", "xright"))):
        raise ValueError("The patch changed an end plane or rim, or created an interior crack.")

    coarse_frontier = _frontier(coarse, active[coarse_owner])
    frontier = _frontier(control, np.concatenate((active[coarse_owner[keep]], active[fine_parents[add]])))
    if not frontier.size or _facet_keys(frontier) != _facet_keys(coarse_frontier):
        raise ValueError("The patch does not share the exact coarse trace vertices, edges and faces.")
    coarse_volume = float(np.sum(coarse.mapping().detA) / 6)
    volume = float(np.sum(control.mapping().detA) / 6)
    if not np.isclose(volume, coarse_volume, atol=0, rtol=1e-12):
        raise ValueError("The patch changed the total solid volume.")

    control = control.with_boundaries(surface).with_subdomains({"solid": np.arange(control.nelements)})
    crossing = [(low < cut) & (high > cut) for cut in cuts_m]
    report = {
        **native.AUTHOR, **native.CONVERGENCE, "schema_version": 1,
        "method": ("nested_coarse_mirror_fine_taper_and_cavity" if region == "central"
                   else "nested_coarse_taper_and_cavity_fine_complement"),
        "region": region,
        "control_source_sha256": native._sha(Path(__file__)),
        "control_coordinates_and_connectivity_sha256": native._array_digest(control.p, control.t),
        "coarse_coordinates_and_connectivity_sha256": native._array_digest(coarse.p, coarse.t),
        "fine_coordinates_and_connectivity_sha256": native._array_digest(fine.p, fine.t),
        "element_order": 2, "z_planes_m": z.tolist(),
        "reference_planes_x_m": [float(xy[:, 0].min()), float(xy[:, 0].max())],
        "nominal_patch_cuts_x_m": list(cuts_m),
        "changed_axial_extent_x_m": [float(low[changed].min()), float(high[changed].max())],
        "coarse_planar_triangles": len(planar),
        "fine_donor_planar_triangles": len(planar_fine),
        "control_planar_triangles": int((~changed).sum() + child_counts[changed].sum()),
        "wholly_inside_coarse_triangles": int(inside.sum()),
        "frozen_cross_cut_planar_triangles": [int((mask & ~changed).sum()) for mask in crossing],
        "wholly_inside_coarse_triangles_frozen_by_conformity": int((inside & ~central).sum()),
        "replaced_coarse_planar_triangles": int(changed.sum()),
        "coarse_parent_partition": {
            "index_order": "Zero-based native bottom-footprint triangle order.",
            "central_parent_indices": np.flatnonzero(central).tolist(),
            "complement_parent_indices": np.flatnonzero(~central).tolist(),
            "central_active_mask_sha256": native._array_digest(central),
            "complement_active_mask_sha256": native._array_digest(~central),
            "nominal_cut_crossing_parent_indices": [np.flatnonzero(mask).tolist() for mask in crossing],
            "replaced_parent_indices": np.flatnonzero(changed).tolist(),
            "replaced_cut_crossing_planar_triangles": [int((mask & changed).sum()) for mask in crossing],
            "recombined_tetrahedra": len(recombined),
            "fine_donor_topology_sha256": native._array_digest(recombined),
        },
        "retained_coarse_tetrahedra": int(keep.sum()),
        "inserted_fine_donor_tetrahedra": int(add.sum()),
        "unchanged_rim_facets": len(rim), "unchanged_patch_frontier_facets": frontier.shape[1],
        "unchanged_patch_frontier_p2_dofs": _p2_dofs(frontier),
        "mesh_vertices": int(control.nvertices), "mesh_tetrahedra": int(control.nelements),
        "ndof": 3 * (int(control.nvertices) + control.edges.shape[1]),
        "boundary_facet_counts": {name: len(facets) for name, facets in control.boundaries.items()},
        "end_face_dofs": {name: _p2_dofs(control.facets[:, surface[name]]) for name in ("xleft", "xright")},
        "minimum_tetrahedron_determinant_m3": float(control.mapping().detA.min()),
        "coarse_solid_volume_m3": coarse_volume, "control_solid_volume_m3": volume,
    }
    return control, report


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--coarse-map", required=True, type=Path)
    parser.add_argument("--fine-map", required=True, type=Path)
    parser.add_argument("--region", choices=("central", "complement"), default="central")
    parser.add_argument("--output-mesh", type=Path, help="Write a new mesh NPZ outside both source maps.")
    args = parser.parse_args(argv)
    try:
        coarse, fine, source = load_nested_patch_sources(args.coarse_map, args.fine_map)
        mesh, report = make_central_patch_control(coarse, fine, tuple(source["nominal_mirror_cuts_x_m"]),
                                                  region=args.region)
        if args.output_mesh is not None:
            output = args.output_mesh.resolve()
            if any(output.is_relative_to(donor.resolve()) for donor in (args.coarse_map, args.fine_map)):
                raise ValueError("Write the hybrid mesh outside both source maps.")
            with output.open("xb") as stream:
                np.savez_compressed(stream, points_m=mesh.p, tetrahedra=mesh.t, **mesh.boundaries, **mesh.subdomains)
            report["saved_mesh_npz_sha256"] = native._sha(output)
        print(json.dumps({**report, "source": source}, indent=2, allow_nan=False))
    except (OSError, ValueError, TypeError, KeyError) as exc:
        parser.exit(1, f"Central patch refinement failed. {exc}\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
