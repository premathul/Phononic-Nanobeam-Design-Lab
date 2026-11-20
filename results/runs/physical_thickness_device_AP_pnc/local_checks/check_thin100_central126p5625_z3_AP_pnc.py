"""Check the selected fixed-footprint three-layer control. Athul Prem | _AP_pnc."""

import argparse
import json
import math
from pathlib import Path
import sys
from types import SimpleNamespace

import numpy as np
from skfem import Dofs, ElementTetP2, ElementVector

STUDY = Path(__file__).resolve().parents[1]
ROOT = STUDY.parents[2]
sys.path[:0] = [str(STUDY), str(ROOT)]

from check_thin400_z1_geometry_AP_pnc import footprint_and_slabs, hole_rim_edges, partition
from check_thin400_z3_geometry_AP_pnc import check_boundary_tags
from phononic.assembly_batched_AP_pnc import _DisplacementDofLayout
from phononic.lead_modes_AP_pnc import _end_dof_pairs
from verify_nested_central_geometry_AP_pnc import archived_parent
from tools import map_scattering_AP_pnc as native

CONFIG = ROOT / "configs/thin100_central126p5625_z3_AP_pnc.json"
LEDGER = ROOT / "verification/thin_device_nested_response_AP_pnc.json"
REFERENCE = STUDY / "thin100_central400_edge126p5625_rim32_linux"


def inspect(mesh, record):
    inputs = record["mesh_inputs"]
    settings = inputs["settings"]
    layers = settings["thickness_layers"]
    length, width, thickness = inputs["dimensions_m"]
    check_boundary_tags(mesh)
    cells, faces, lead_edges = partition(mesh, inputs)
    xy, triangles, slabs = footprint_and_slabs(mesh, thickness, layers)
    rims = hole_rim_edges(faces["hole_walls"], inputs["pattern"]["holes"])
    assert np.isfinite(mesh.mapping().detA).all() and np.all(mesh.mapping().detA > 0)

    element = ElementVector(ElementTetP2())
    layout = _DisplacementDofLayout(mesh, element, Dofs(mesh, element), intorder=4)
    period, left, right = _end_dof_pairs(SimpleNamespace(mesh=mesh, basis=layout))
    assert period == length and layout.N == record["ndof"]
    assert record["end_face_dofs"] == {"xleft": len(left), "xright": len(right)}
    assert len(left) == 3 * (2 * 4 + 1) * (2 * layers + 1)
    for end, indices, plane in (("xleft", left, 0.), ("xright", right, length)):
        np.testing.assert_array_equal(np.sort(indices), np.sort(layout.get_dofs(facets=mesh.boundaries[end]).all()))
        np.testing.assert_array_equal(layout.doflocs[0, indices], np.full(len(indices), plane))

    cuts = [value * 1e-9 for value in settings["patterned_x_interval_nm"]]
    x = xy[0, triangles]
    footprints = {
        "left_lead": int(np.count_nonzero(x.max(axis=0) <= cuts[0])),
        "central": int(np.count_nonzero((x.min(axis=0) >= cuts[0]) & (x.max(axis=0) <= cuts[1]))),
        "right_lead": int(np.count_nonzero(x.min(axis=0) >= cuts[1])),
    }
    assert sum(footprints.values()) == triangles.shape[1]
    return xy, triangles, slabs, rims, {
        "thickness_layers": layers, "z_planes_m": np.unique(mesh.p[2]).tolist(),
        "dimensions_m": [length, width, thickness], "mesh_hash": record["mesh_hash"],
        "mesh_vertices": int(mesh.nvertices), "mesh_tetrahedra": int(mesh.nelements),
        "body_dofs": int(layout.N), "end_face_dofs": record["end_face_dofs"],
        "regional_planar_triangles": footprints,
        "regional_tetrahedra": {name: partition.total() for name, partition in cells.items()},
        "lead_outer_edge_segments": lead_edges,
        "face_triangles": {name: elements.total() for name, elements in faces.items()},
        "mesh_build_resources": record["mesh_resources"],
        "compatible_tagged_p2_end_traces": True,
    }


def check(candidate_dir):
    link = lambda path: native._link(path, ROOT)
    ledger = native._read(LEDGER)
    anchor = ledger["levels"]["126.5625"]
    state = native.read_map_checkpoint(REFERENCE)
    assert [(point["frequency_GHz"], point["status"]) for point in state["points"]] == [(1, "accepted")]
    assert anchor["archive"]["checkpoint"] == link(REFERENCE / native.CHECKPOINT)
    assert anchor["archive"]["map"] == link(REFERENCE / native.MAP)

    config = native.workflow.resolve_config(native._read(CONFIG))
    reference_config = state["inputs"]["configuration"]
    assert config == {
        **reference_config,
        "mesh": {**reference_config["mesh"], "thickness_layers": 3},
        "resources": {**reference_config["resources"], "max_dofs": 280035},
    }
    assert reference_config["mesh"]["thickness_layers"] == 2
    assert config["resources"] == {"max_dofs": 280035, "max_port_dofs": 400}

    identity = native.workflow.source_identity()
    for path in ("tools/map_scattering_AP_pnc.py", "tools/compare_scattering_AP_pnc.py"):
        identity["source_sha256"][path] = native._sha(ROOT / path)
    reference_identity = state["inputs"]["numerical_identity"]
    assert identity["source_sha256"] == reference_identity["source_sha256"]
    assert identity["packages"] == reference_identity["packages"]
    geometry, _, pattern = native.workflow._resolve_geometry(config)
    assert pattern == state["inputs"]["hole_pattern"]
    assert state["shared"]["reference_planes_x_m"] == [0., geometry.length_m]
    saved_mesh_inputs = native._read(candidate_dir / native.RUN_SUMMARY)["mesh_inputs"]
    mesh_host = {name: saved_mesh_inputs[name] for name in ("python", "platform")}
    expected_dir, key, _ = native.workflow._mesh_location(
        candidate_dir.parent.parent, geometry, pattern, config["mesh"], {**identity, **mesh_host}
    )
    assert candidate_dir == expected_dir
    candidate, new = native.workflow.load_checkpoint(candidate_dir, key)
    reference, old = archived_parent(REFERENCE)
    assert old["mesh_hash"] == anchor["mesh_hash"]
    assert old["mesh_sha256"] == anchor["archive"]["mesh"]["sha256"]
    assert new["mesh_inputs"]["settings"] == config["mesh"]
    assert new["mesh_inputs"]["dimensions_m"] == old["mesh_inputs"]["dimensions_m"]
    assert new["mesh_inputs"]["pattern"] == old["mesh_inputs"]["pattern"]

    xy, triangles, slabs, rims, stats = inspect(reference, old)
    xy3, triangles3, slabs3, rims3, stats3 = inspect(candidate, new)
    footprint_sha = native._array_digest(xy, triangles)
    assert native._array_digest(xy3, triangles3) == footprint_sha
    for slab in slabs + slabs3:
        np.testing.assert_array_equal(slab, slabs[0], err_msg="Layer-local tetrahedron indices differ")
    assert rims == rims3 and len(rims) == 26 * 32
    assert stats["regional_planar_triangles"] == stats3["regional_planar_triangles"]
    assert stats["lead_outer_edge_segments"] == stats3["lead_outer_edge_segments"]
    assert stats["end_face_dofs"] == {"xleft": 135, "xright": 135}
    assert stats3["end_face_dofs"] == {"xleft": 189, "xright": 189}
    assert stats3["body_dofs"] == config["resources"]["max_dofs"]

    lower_receipt = STUDY / "local_checks/thin100_central400_edge168p75_rim32_linux_response_1GHz_execution_AP_pnc.json"
    lower = native._read(lower_receipt)
    upper = anchor["measured_linux_map_execution"]
    assert upper["receipt"] == link(ROOT / upper["receipt"]["path"]) and upper["exit_status"] == 0
    assert lower["exit_code"] == 0 and lower["host_name"] == "node031.lawrence"
    dofs_low, dofs_high = ledger["levels"]["168.75"]["body_dofs"], stats["body_dofs"]
    rss_low, rss_high = lower["peak_child_rss_bytes"], int(upper["peak_rss_kib"]) * 1024
    exponent = math.log(rss_high / rss_low) / math.log(dofs_high / dofs_low)
    ratio = stats3["body_dofs"] / dofs_high

    return {
        **native.AUTHOR, "schema_version": 1, "status": "passed_mesh_only",
        "frequency_hz": 1000000000, "physical_thickness_nm": 100,
        "reference_planes_x_m": state["shared"]["reference_planes_x_m"],
        "configuration": link(CONFIG), "verification_source": link(Path(__file__).resolve()),
        "accepted_reference_checkpoint": anchor["archive"]["checkpoint"],
        "accepted_reference_mesh": anchor["archive"]["mesh"],
        "candidate_preflight": link(candidate_dir / native.RUN_SUMMARY),
        "candidate_mesh": link(candidate_dir / "mesh_AP_pnc.npz"),
        "source_git_commit_local": identity["git_commit"],
        "source_git_commit_reference": state["source_git_commit"],
        "exact_native_source_match_count": len(identity["source_sha256"]),
        "native_source_sha256": native.workflow.digest(identity["source_sha256"]),
        "shared_package_versions": identity["packages"],
        "verification_python_platform": {name: identity[name] for name in ("python", "platform")},
        "candidate_mesh_python_platform": mesh_host,
        "reference_python_platform": {name: reference_identity[name] for name in ("python", "platform")},
        "footprint": {
            "ordered_xy_and_triangles_sha256": footprint_sha,
            "max_abs_xy_difference_m": float(np.max(np.abs(xy - xy3))),
            "vertices": xy.shape[1], "triangles": triangles.shape[1],
            "exact_layer_local_tetrahedron_connectivity": True,
            "hole_count": 26, "exact_projected_hole_rim_edges": len(rims), "rim_segments_per_hole": 32,
        },
        "accepted_linux_z2": stats, "candidate_mesh_z3": stats3,
        "resource_projection": {
            "linux_measured_anchors": [
                {"body_dofs": dofs_low, "peak_rss_bytes": rss_low, "receipt": link(lower_receipt)},
                {"body_dofs": dofs_high, "peak_rss_bytes": rss_high, "receipt": upper["receipt"]},
            ],
            "fitted_rss_dof_exponent": exponent,
            "fitted_projection_bytes": round(rss_high * ratio**exponent),
            "quadratic_sensitivity_bytes": round(rss_high * ratio**2),
            "scope": "The anchors ran on node031 at 1 GHz. The smaller anchor reused Windows body assembly. "
                     "DOF scaling estimates the native Linux peak for allocation planning. "
                     "The candidate factor fill and solve memory peak remain unmeasured.",
        },
    }


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--candidate-mesh", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    evidence = json.dumps(check(args.candidate_mesh.resolve()), indent=2, allow_nan=False)
    with args.output.open("x", encoding="utf-8") as output:
        output.write(evidence + "\n")
    print(args.output)
