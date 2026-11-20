"""Verify the archived nested 26-hole meshes. Author: Athul Prem | _AP_pnc."""

from hashlib import file_digest
import json
from pathlib import Path
import sys

import numpy as np

STUDY = Path(__file__).resolve().parent
ROOT = STUDY.parents[2]
sys.path.insert(0, str(ROOT / "src"))

from phononic import workflow_AP_pnc as workflow
from phononic.refinement.central_AP_pnc import refine_footprint


PARENTS = {
    "thin100": STUDY / "thin100_central400_1GHz",
    "thick200": STUDY.parent / "regional_mesh_AP_pnc" / "lead150_z2_1GHz",
}
TARGETS = {"300": "300", "225": "225", "168.75": "168p75"}


def read(path):
    return json.loads(path.read_bytes())


def sha(path):
    with path.open("rb") as stream:
        return file_digest(stream, "sha256").hexdigest()


def archived_parent(directory):
    checkpoint = read(directory / "checkpoint_AP_pnc.json")
    assert checkpoint.pop("checkpoint_data_sha256") == workflow.digest(checkpoint)
    shared = checkpoint["shared"]
    paths = {name: directory / shared["artifacts"][name]["path"]
             for name in ("mesh", "mesh_summary")}
    for name, path in paths.items():
        assert sha(path) == shared["artifacts"][name]["sha256"]
    assert paths["mesh"].parent == paths["mesh_summary"].parent
    return workflow.load_checkpoint(paths["mesh"].parent, shared["mesh_hash"])


def cells(coordinates):
    return {tuple(sorted(map(tuple, vertices))) for vertices in coordinates.transpose(2, 1, 0)}


def fixed_geometry(mesh, cuts):
    left, right = cuts
    tx = mesh.p[0, mesh.t]
    lead = (tx.max(axis=0) <= left) | (tx.min(axis=0) >= right)
    facets = mesh.p[:, mesh.facets]
    x, y, z = facets
    cut = np.all(x == left, axis=0) | np.all(x == right, axis=0)
    assert np.all(mesh.f2t[:, cut] >= 0)
    cut_tetrahedra = mesh.f2t[:, cut]
    assert np.all(lead[cut_tetrahedra].sum(axis=0) == 1)
    exterior = mesh.boundary_facets()
    outer_sides = np.all(y == mesh.p[1].min(), axis=0) | np.all(y == mesh.p[1].max(), axis=0)
    central = np.all((left <= x) & (x <= right), axis=0)
    rim = exterior[((np.ptp(z, axis=0) > 0) & central & ~outer_sides)[exterior]]
    return {
        "lead_tetrahedra": cells(mesh.p[:, mesh.t[:, lead]]),
        "cut_facets": cells(facets[:, :, cut]),
        "rim_facets": cells(facets[:, :, rim]),
        "port_facets": {name: cells(facets[:, :, mesh.boundaries[name]]) for name in ("xleft", "xright")},
    }


def footprint(mesh):
    bottom = np.flatnonzero(mesh.p[2] == mesh.p[2].min())
    index = np.full(mesh.nvertices, -1)
    index[bottom] = np.arange(len(bottom))
    t = index[mesh.t.T]
    t = np.sort(t[np.count_nonzero(t >= 0, axis=1) == 3], axis=1)[:, 1:]
    assert len(t) * 6 == mesh.nelements
    return mesh.p[:2, bottom].T, t


def measure(mesh, record, cuts, xy, triangles, target):
    left, right = cuts
    x = xy[triangles, 0]
    central = triangles[np.all((left <= x) & (x <= right), axis=1)]
    central_xy = xy[central]
    areas = np.abs(np.linalg.det(central_xy[:, 1:] - central_xy[:, :1])) / 2
    assert np.all(areas > 0)
    assert np.all(mesh.mapping().detA > 0)
    cell_lengths = (np.linalg.norm(central_xy[:, 1] - central_xy[:, 0], axis=1),
                    np.linalg.norm(central_xy[:, 2] - central_xy[:, 1], axis=1),
                    np.linalg.norm(central_xy[:, 2] - central_xy[:, 0], axis=1))

    unique, incidence = np.unique(np.concatenate([central[:, [a, b]] for a, b in ((0, 1), (0, 2), (1, 2))]),
                                  axis=0, return_counts=True)
    assert np.all(incidence <= 2)
    edge_xy = xy[unique]
    length = np.linalg.norm(edge_xy[:, 0] - edge_xy[:, 1], axis=1)
    edge_x, edge_y = edge_xy.transpose(2, 0, 1)
    cut_left = np.all(edge_x == left, axis=1)
    cut_right = np.all(edge_x == right, axis=1)
    cut = cut_left | cut_right
    assert np.all(incidence[cut] == 1)
    outer = np.all(edge_y == xy[:, 1].max(), axis=1) | np.all(edge_y == xy[:, 1].min(), axis=1)
    rim = (incidence == 1) & ~cut & ~outer
    eligible = ~(cut | rim)
    grading = np.maximum(length[cut_left].max() - (edge_x.min(axis=1) - left) / 2,
                         length[cut_right].max() - (right - edge_x.max(axis=1)) / 2)
    transition = eligible & (grading > target)
    if np.isfinite(target):
        np.testing.assert_array_less(length[eligible], np.maximum(grading[eligible], target) + 1e-21)
    assert record["ndof"] == 3 * (mesh.nvertices + mesh.edges.shape[1])
    return {
        "planar_triangles": len(triangles),
        "central_planar_triangles": len(central),
        "tetrahedra": mesh.nelements,
        "body_dofs": record["ndof"],
        "bulk_max_edge_nm": float(length[eligible & ~transition].max() * 1e9),
        "transition_max_edge_nm": float(length[transition].max() * 1e9) if transition.any() else None,
        "cut_max_edge_nm": float(length[cut].max() * 1e9),
        "rim_max_edge_nm": float(length[rim].max() * 1e9),
        "area_weighted_mean_triangle_max_edge_nm": float(np.average(np.max(cell_lengths, axis=0), weights=areas) * 1e9),
        "central_polygon_area_nm2": float(areas.sum() * 1e18),
        "archive_mesh_sha256": record["mesh_sha256"],
        "mesh_hash": record["mesh_hash"],
    }


def main():
    cases = {}
    identity = workflow.source_identity()
    for key, parent_path in PARENTS.items():
        mesh, record = archived_parent(parent_path)
        settings = record["mesh_inputs"]["settings"]
        cuts = tuple(value * 1e-9 for value in settings["patterned_x_interval_nm"])
        fixed = fixed_geometry(mesh, cuts)
        xy, triangles = footprint(mesh)
        parent = measure(mesh, record, cuts, xy, triangles, np.inf)
        parent.update({f"fixed_{name}_match": True for name in fixed})
        parent["planar_staged_and_direct_match"] = None
        case = {"parent_archive_map": parent_path.relative_to(ROOT).as_posix(), "parent": parent, "targets": {}}
        previous = xy, triangles
        for label, token in TARGETS.items():
            config_path = STUDY / f"{key}_central400_edge{token}_rim32_configuration_AP_pnc.json"
            config = workflow.resolve_config(read(config_path), check_body_budget=False)
            geometry, holes, pattern = workflow._resolve_geometry(config)
            directory, mesh_hash, _ = workflow._mesh_location(STUDY, geometry, pattern, config["mesh"], identity)
            mesh, record = workflow.load_checkpoint(directory, mesh_hash)
            assert len(holes) == 26
            target = float(label) * 1e-9
            candidate = footprint(mesh)
            row = measure(mesh, record, cuts, *candidate, target)
            np.testing.assert_allclose(row["central_polygon_area_nm2"], parent["central_polygon_area_nm2"], rtol=1e-14, atol=0.)
            for name, value in fixed_geometry(mesh, cuts).items():
                assert value == fixed[name], name
                row[f"fixed_{name}_match"] = True
            for donor in ((xy, triangles), previous):
                refined = refine_footprint(*donor, cuts, target)
                for actual, expected in zip(candidate, refined, strict=True):
                    np.testing.assert_array_equal(actual, expected)
            previous = candidate
            row["planar_staged_and_direct_match"] = True
            case["targets"][label] = row
        cases[key] = case
    result = {
        "author": "Athul Prem", "file_tag": "_AP_pnc", "schema_version": 1,
        "planar_scope": "Triangle nesting and edge statistics refer to the bottom x-y footprint.",
        "geometry_scope": "Archived three-dimensional lead tetrahedra, ports, shared cut facets and hole-wall facets are compared by exact coordinate sets.",
        "verifier": {"path": Path(__file__).resolve().relative_to(ROOT).as_posix(), "sha256": sha(Path(__file__))},
        "refiner_sha256": identity["source_sha256"]["src/phononic/refinement/central_AP_pnc.py"],
        "cases": cases,
    }
    output = STUDY / "nested_central_geometry_AP_pnc.json"
    workflow.write_json(output, result)
    print(output)


if __name__ == "__main__":
    main()
