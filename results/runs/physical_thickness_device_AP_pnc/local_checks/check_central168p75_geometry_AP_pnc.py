"""Inspect the paired 225 to 168.75 nm preflight meshes. Athul Prem | _AP_pnc."""

import argparse
import json
from pathlib import Path
import sys

import numpy as np
from skfem import Dofs, ElementTetP2, ElementVector

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from check_central225_geometry_AP_pnc import footprint_edges
from check_geometry_AP_pnc import PARTITION_CHECK, ROOT, STUDY, artifact, equal_simplex_counts, load, mesh_record
from check_geometry_AP_pnc import partition, read, read_map_checkpoint
from check_thick225_geometry_AP_pnc import map_mesh
from check_thick400_z3_geometry_AP_pnc import check_scaled_meshes
from check_thin400_rim48_geometry_AP_pnc import check_rims
from check_thin400_z3_geometry_AP_pnc import check_boundary_tags
from phononic.assembly_batched_AP_pnc import _DisplacementDofLayout


TRACE_NAMES = ("left_port", "left_cut", "right_cut", "right_port")
LEVELS_NM = (225, 168.75)


def end_layout(mesh, summary, planes):
    element = ElementVector(ElementTetP2())
    body = _DisplacementDofLayout(mesh, element, Dofs(mesh, element), 4)
    assert body.N == summary["ndof"]
    components = np.empty(body.N, dtype=int)
    for index, dofs in enumerate(body.split_indices()):
        components[dofs] = index
    traces = {}
    for boundary, plane in zip(("xleft", "xright"), planes, strict=True):
        dofs = body.get_dofs(facets=mesh.boundaries[boundary]).all()
        assert len(dofs) == summary["end_face_dofs"][boundary] == 135
        np.testing.assert_allclose(body.doflocs[0, dofs], plane, rtol=0, atol=2e-20)
        layout = np.unique(np.column_stack((body.doflocs[1:, dofs].T, components[dofs])), axis=0)
        assert layout.shape == (len(dofs), 3)
        traces[boundary] = layout
    return traces


def inspect(name, thickness_nm, preflight, prior_receipt):
    reference = STUDY / f"{name}_central225_1GHz"
    accepted = read_map_checkpoint(reference)
    assert [(point["frequency_GHz"], point["status"]) for point in accepted["points"]] == [(1.0, "accepted")]
    old_mesh, old, old_dir = map_mesh(accepted, reference)
    new_mesh, new = load(preflight)
    prior = read(prior_receipt)
    assert prior["status"] == "passed" and all(old[key] == prior["candidate"][key] for key in ("mesh_hash", "mesh_sha256"))
    assert preflight.name == new["mesh_hash"]

    original, refined = old["mesh_inputs"], new["mesh_inputs"]
    settings = original["settings"]
    assert (settings["mesh_size_nm"], settings["lead_mesh_size_nm"], settings["element_order"],
            settings["thickness_layers"], settings["circle_resolution"]) == (LEVELS_NM[0], 150, 2, 2, 32)
    assert settings["patterned_x_interval_nm"] == [1000, 15650]
    assert refined == {**original, "settings": {**settings, "mesh_size_nm": LEVELS_NM[1]}}
    pattern = new["hole_pattern_provenance"]
    assert pattern == old["hole_pattern_provenance"] == accepted["inputs"]["hole_pattern"]
    assert new["hole_count"] == old["hole_count"] == len(pattern["holes"]) == 26
    source = artifact(ROOT / pattern["source_csv"])
    assert source["sha256"] == pattern["source_sha256"]

    length, width, thickness = refined["dimensions_m"]
    assert [length, width, thickness] == [16650 * 1e-9, 1000 * 1e-9, thickness_nm * 1e-9]
    planes = [0., length]
    cuts = [value * 1e-9 for value in settings["patterned_x_interval_nm"]]
    assert accepted["shared"]["reference_planes_x_m"] == planes
    for mesh in (old_mesh, new_mesh):
        check_boundary_tags(mesh)
    old_cells, old_faces, old_edges = partition(old_mesh, original)
    new_cells, new_faces, new_edges = partition(new_mesh, refined)
    for side in ("left", "right"):
        equal_simplex_counts(old_cells[side], new_cells[side], f"{name} fixed {side} lead")
    fixed_faces = (*TRACE_NAMES, "hole_walls")
    for label in fixed_faces:
        equal_simplex_counts(old_faces[label], new_faces[label], f"{name} fixed {label}")
    assert old_edges == new_edges

    central_counts = [cells["central"].total() for cells in (old_cells, new_cells)]
    assert central_counts[1] > central_counts[0]
    rims, max_error = check_rims(new_mesh, pattern["holes"], np.unique(new_mesh.p[2]),
                                 settings["circle_resolution"], new_faces["hole_walls"].total())
    old_layout, new_layout = (end_layout(mesh, summary, planes) for mesh, summary in ((old_mesh, old), (new_mesh, new)))
    for boundary, layout in old_layout.items():
        np.testing.assert_array_equal(layout, new_layout[boundary], err_msg=f"{name} fixed {boundary} P2 vector layout")

    return {
        "physical_thickness_nm": thickness_nm,
        "reference_checkpoint": artifact(reference / "checkpoint_AP_pnc.json"),
        "reference_geometry_evidence": artifact(prior_receipt),
        "reference": mesh_record(old_dir, old), "candidate": mesh_record(preflight, new),
        "same_P2_end_trace_coordinates_and_components": True, "verified_rims_at_three_planes": rims,
        "maximum_rim_vertex_error_m": max_error, "hole_pattern_csv": source,
        "lead_tetrahedra": {side: new_cells[side].total() for side in ("left", "right")},
        "exact_shared_faces": {label: new_faces[label].total() for label in fixed_faces},
        "central_tetrahedra": central_counts,
        "exact_shared_central_tetrahedra": (old_cells["central"] & new_cells["central"]).total(),
        "central_bottom_edges": [footprint_edges(mesh, cuts) for mesh in (old_mesh, new_mesh)],
    }


def compare(thin_preflight, thick_preflight):
    thin = inspect("thin100", 100, thin_preflight.resolve(), STUDY / "central225_geometry_AP_pnc.json")
    thick = inspect("thick200", 200, thick_preflight.resolve(), STUDY / "thick200_central225_geometry_AP_pnc.json")
    for level in ("reference", "candidate"):
        check_scaled_meshes({"mesh": thin[level]}, {"mesh": thick[level]})
    assert all(thin[key] == thick[key] for key in (
        "hole_pattern_csv", "lead_tetrahedra", "exact_shared_faces", "central_tetrahedra", "central_bottom_edges",
    ))
    return {
        "author": "Athul Prem", "file_tag": "_AP_pnc", "status": "passed",
        "central_targets_nm": LEVELS_NM, "lead_target_nm": 150, "quadratic_thickness_layers": 2,
        "hole_count": 26, "rim_segments_per_hole_per_plane": 32,
        "reference_planes_x_m": [0., 16650 * 1e-9], "regional_cut_planes_nm": [1000, 15650],
        "thin_to_thick_coordinate_scale": [1, 1, 2], "exact_paired_ordered_connectivity_and_tags": True,
        "cases": {"thin100": thin, "thick200": thick},
        "provenance": {"geometry_replay": artifact(Path(__file__).resolve()), "regional_partition": artifact(PARTITION_CHECK)},
    }


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("thin_preflight", type=Path)
    parser.add_argument("thick_preflight", type=Path)
    args = parser.parse_args()
    print(json.dumps(compare(args.thin_preflight, args.thick_preflight), indent=2, allow_nan=False))
