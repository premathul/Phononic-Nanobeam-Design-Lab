"""Check the thin 126.5625 nm native mesh before assembly. Author: Athul Prem | _AP_pnc."""

import argparse
from copy import deepcopy
from hashlib import sha256
import json
from pathlib import Path
import sys

import numpy as np

from verify_nested_central_geometry_AP_pnc import (
    ROOT, STUDY, archived_parent, fixed_geometry, footprint, measure, read, refine_footprint, sha, workflow,
)

sys.path.insert(0, str(ROOT))
from tools import map_scattering_AP_pnc as native

PARENT = STUDY / "thin100_central400_edge168p75_rim32_linux"
CONFIG = STUDY / "thin100_central400_edge126p5625_rim32_configuration_AP_pnc.json"
PARENT_TARGET, TARGET = 168.75, 126.5625


def check(cache):
    state = native.read_map_checkpoint(PARENT)
    publication = native.read_saved_map(PARENT)
    reuse = state["preparation_reuse"]
    receipt = reuse["receipt"]
    shared = state["shared"]
    parent_checkpoint = PARENT / native.CHECKPOINT
    receipt_bytes = (json.dumps(receipt, indent=2, allow_nan=False) + "\n").encode("utf-8")
    if (publication["checkpoint_sha256"] != sha(parent_checkpoint)
            or not any(point["frequency_GHz"] == 1.0 and point["status"] == "accepted" for point in state["points"])
            or reuse["method"] != "windows_to_linux_native_preparation"
            or sha256(receipt_bytes).hexdigest() != reuse["receipt_sha256"]
            or receipt["donor_shared_sha256"] != workflow.digest(shared)
            or not receipt["donor_numerical_identity"]["platform"].startswith("Windows")):
        raise ValueError("The accepted parent does not bind the transferred Windows preparation.")

    config = workflow.resolve_config(read(CONFIG))
    archived_config = deepcopy(state["inputs"]["configuration"])
    if archived_config["mesh"]["central_max_edge_nm"] != PARENT_TARGET:
        raise ValueError("Expected the accepted 168.75 nm parent configuration.")
    archived_config["mesh"]["central_max_edge_nm"] = TARGET
    archived_config["resources"]["max_dofs"] = config["resources"]["max_dofs"]
    if workflow.digest(config) != workflow.digest(archived_config):
        raise ValueError("Only the central edge target and the body DOF cap may change from the parent.")

    identity = workflow.source_identity()
    geometry, holes, pattern = workflow._resolve_geometry(config)
    if (len(holes) != 26 or pattern != state["inputs"]["hole_pattern"]
            or shared["reference_planes_x_m"] != [0., geometry.length_m]):
        raise ValueError("The native configuration changes the original holes or reference planes.")
    mesh_dir, mesh_hash, _ = workflow._mesh_location(Path(cache), geometry, pattern, config["mesh"], identity)
    candidate_mesh, candidate_record = workflow.load_checkpoint(mesh_dir, mesh_hash)
    parent_mesh, parent_record = archived_parent(PARENT)
    cuts = tuple(value * 1e-9 for value in config["mesh"]["patterned_x_interval_nm"])
    parent_footprint = footprint(parent_mesh)
    candidate_footprint = footprint(candidate_mesh)
    expected = refine_footprint(*parent_footprint, cuts, TARGET * 1e-9)
    for actual, refined in zip(candidate_footprint, expected, strict=True):
        np.testing.assert_array_equal(actual, refined, err_msg="Native mesh differs from exact archived-parent bisection.")

    fixed_parent = fixed_geometry(parent_mesh, cuts)
    fixed_candidate = fixed_geometry(candidate_mesh, cuts)
    for name, coordinates in fixed_parent.items():
        if coordinates != fixed_candidate[name]:
            raise ValueError(f"The native mesh changes fixed {name} coordinates or connectivity.")
    if (candidate_record["end_face_dofs"] != parent_record["end_face_dofs"]
            or candidate_record["ndof"] > config["resources"]["max_dofs"]):
        raise ValueError("The native mesh changes the port resolution or exceeds the body DOF cap.")

    parent = measure(parent_mesh, parent_record, cuts, *parent_footprint, PARENT_TARGET * 1e-9)
    candidate = measure(candidate_mesh, candidate_record, cuts, *candidate_footprint, TARGET * 1e-9)
    np.testing.assert_allclose(candidate["central_polygon_area_nm2"], parent["central_polygon_area_nm2"],
                               rtol=1e-14, atol=0.)
    if candidate["bulk_max_edge_nm"] >= parent["bulk_max_edge_nm"]:
        raise ValueError("The native mesh fails to reduce the bulk maximum edge length.")
    return {**native.AUTHOR, "schema_version": 1, "status": "passed_native_mesh_geometry_preflight",
            "target_nm": TARGET, "parent_target_nm": PARENT_TARGET,
            "configuration": native._link(CONFIG, ROOT), "parent_map_checkpoint": native._link(parent_checkpoint, ROOT),
            "windows_transfer_receipt_sha256": reuse["receipt_sha256"],
            "parent_windows_shared_mesh": shared["artifacts"]["mesh"],
            "candidate_mesh_preflight": native._link((mesh_dir / native.RUN_SUMMARY).resolve(), ROOT),
            "geometry_verifier": native._link(Path(__file__).resolve(), ROOT),
            "refiner_sha256": identity["source_sha256"]["src/phononic/refinement/central_AP_pnc.py"],
            "fixed_geometry": dict.fromkeys(fixed_parent, True), "planar_archived_parent_bisection_match": True,
            "reference_planes_x_m": shared["reference_planes_x_m"], "end_face_dofs": candidate_record["end_face_dofs"],
            "parent": parent, "candidate": candidate}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mesh-cache-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    try:
        result = check(args.mesh_cache_root)
        with args.output.open("x", encoding="utf-8") as stream:
            json.dump(result, stream, indent=2, allow_nan=False)
            stream.write("\n")
    except (AssertionError, KeyError, OSError, TypeError, ValueError) as exc:
        parser.exit(2, f"Geometry preflight failed: {exc}\n")
    print(f"Geometry preflight passed: {args.output}")


if __name__ == "__main__":
    main()
