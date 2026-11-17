"""Preflight the fixed-port h60 full-device mesh. Author: Athul Prem | _AP_pnc."""

from __future__ import annotations

import argparse
from dataclasses import dataclass
import hashlib
import json
import os
from pathlib import Path
import socket
import sys

import numpy as np
from skfem import Dofs, ElementTetP2, ElementVector, MeshTet

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT / "src"), str(ROOT)]

from phononic import workflow_AP_pnc as workflow
from phononic.assembly_batched_AP_pnc import _DisplacementDofLayout
from tools import lawrence_solver_AP_pnc as lawrence
from tools.run_lawrence_uniform_h80_AP_pnc import artifact, end_triangles, trace_keys


AUTHOR = {"author": "Athul Prem", "file_tag": "_AP_pnc"}
RECEIPT = "device_h60_preflight_AP_pnc.json"
H80_MESH_HASH = "89cfa12d6b14109eb95dc873b3682aa5650045ccf5fe3a0d723e8d45b176ba33"
H80_INPUT_SHA256 = {
    "configuration": "f0dd7ad1cebb573d25e3e665335b894bdba440ee19befc62236630110f7839e0",
    "mesh_summary": "4e0eda32c22840717c79456614bdc07130810b703d77e2ca0a7dcf8efaa6e023",
    "mesh": "1bf8ad7aea98d1800b8bb8892954d5810b0907b7a7d4fd42979ae22dc75456de",
    "ports": "18c0ad4c8b28e597503a28bd5157c867bb09744c94e1eb1aff23f024cc879c8b",
}
SOURCE_MAP_SHA256 = "b4003df3dda37617a26658100dd562278a9e301ca514928ab815a0ec0438039f"
PACKAGES = {"numpy": "2.5.4", "scipy": "1.18.1", "scikit-fem": "12.0.2", "meshio": "5.3.5",
            "gmsh": "4.15.2", "matplotlib": "3.11.2", "psutil": "7.2.2"}
PORT_DOF_COUNTS = {"xleft": 675, "xright": 675}
WALL_VERTEX_TOLERANCE_M = 1e-15


@dataclass
class Reference:
    config: dict
    mesh: MeshTet
    mesh_record: dict
    trace_order: tuple
    artifacts: dict


def displacement_layout(mesh):
    element = ElementVector(ElementTetP2())
    return _DisplacementDofLayout(mesh, element, Dofs(mesh, element), 4)


def port_trace_ids(mesh, layout):
    return tuple(layout.get_dofs(facets=mesh.boundaries[side]).all() for side in PORT_DOF_COUNTS)


def components(layout):
    result = np.empty(layout.N, dtype=np.int8)
    for axis, ids in enumerate(layout.split_indices()):
        result[ids] = axis
    return result


def read_reference(config_path, mesh_dir, ports_path):
    inputs = {"configuration": config_path, "mesh_summary": mesh_dir / "summary_AP_pnc.json",
              "mesh": mesh_dir / "mesh_AP_pnc.npz", "ports": ports_path}
    artifacts = {name: artifact(path) for name, path in inputs.items()}
    for name, expected in H80_INPUT_SHA256.items():
        if artifacts[name]["sha256"] != expected:
            raise ValueError(f"The saved h80 {name} differs from the verified artifact.")
    mesh, record = workflow.load_checkpoint(mesh_dir, H80_MESH_HASH)
    layout = displacement_layout(mesh)
    coordinate, component = layout.doflocs.T, components(layout)
    with np.load(ports_path, allow_pickle=False) as ports:
        forward = ports["positive_indices"]
        forward = forward[ports["propagating"][forward]]
        if (ports["frequency_hz"].item() != 1e9 or forward.tolist() != list(lawrence.INCIDENT_ROOTS)
                or layout.N != record["ndof"] or record["end_face_dofs"] != PORT_DOF_COUNTS):
            raise ValueError("The saved h80 SAFE frequency, incident root order or layout differs.")
        orders = []
        for side, tagged_ids in zip(("left", "right"), port_trace_ids(mesh, layout), strict=True):
            ids = ports[f"{side}_dofs"]
            if ids.shape != (675,) or not np.array_equal(np.sort(ids), np.sort(tagged_ids)):
                raise ValueError(f"The saved h80 {side} SAFE IDs do not cover its tagged end face.")
            orders.append(trace_keys(coordinate, component, ids))
    if orders[0] != orders[1] or len(set(orders[0])) != 675:
        raise ValueError("The h80 SAFE rows must share one unique transverse-coordinate and component order.")
    config = json.loads(config_path.read_bytes())
    config["name"] = "historical_26hole_layered_h60_ny12_z4"
    config["mesh"]["mesh_size_nm"] = 60
    return Reference(workflow.resolve_config(config, check_body_budget=False), mesh, record, orders[0], artifacts)


def map_trace(mesh, layout, order):
    coordinate, component = layout.doflocs.T, components(layout)
    mapped = []
    for ids in port_trace_ids(mesh, layout):
        lookup = dict(zip(trace_keys(coordinate, component, ids), ids, strict=True))
        if len(ids) != len(order) or lookup.keys() != set(order):
            raise ValueError("The h60 port lacks an exact transverse-coordinate and component match.")
        mapped.append(np.asarray([lookup[key] for key in order]))
    return tuple(mapped)


def hole_wall(mesh, free_faces, z_planes, hole, segments):
    # Use rim angle and layer number to label the same physical vertices in either mesh.
    x, y, z = mesh.p
    radius = hole["diameter_m"] / 2
    dx = x - hole["x_m"]
    on_rim = np.abs(np.hypot(dx, y) - radius) <= 1e-8 * radius
    faces = free_faces[:, np.all(on_rim[free_faces], axis=0)]
    nodes = np.unique(faces)
    angle = np.arctan2(y[nodes], dx[nodes])
    rim = np.rint(angle * segments / (2 * np.pi)).astype(int) % segments
    labels = np.searchsorted(z_planes, z[nodes]) * segments + rim
    triangles = np.sort(labels[np.searchsorted(nodes, faces)].T, axis=1)
    triangles = np.unique(triangles, axis=0)
    count = 2 * segments * (len(z_planes) - 1)
    if (not np.array_equal(np.sort(labels), np.arange(len(z_planes) * segments))
            or len(triangles) != count or faces.shape[1] != count):
        raise ValueError(f"Unexpected rim nodes or wall facet coverage at hole x={hole['x_m']} m.")
    return triangles, mesh.p[:, nodes[np.argsort(labels)]]


def verify_boundaries(reference, mesh, mesh_inputs):
    length, _, thickness = mesh_inputs["dimensions_m"]
    ports = {}
    for side, x in zip(PORT_DOF_COUNTS, (0, length), strict=True):
        triangles = end_triangles(mesh, side)
        for item in (mesh, reference):
            facets = item.facets[:, item.boundaries[side]]
            if not np.all(item.p[0, facets] == x):
                raise ValueError(f"The tagged {side} port has a different physical end plane.")
        if len(triangles) != 96 or triangles != end_triangles(reference, side):
            raise ValueError(f"The tagged {side} end-face triangulation differs from h80.")
        ports[side] = {"x_m": x, "triangles": len(triangles), "dofs": PORT_DOF_COUNTS[side],
                       "yz_triangles_sha256": hashlib.sha256(np.asarray(triangles, dtype="<f8").tobytes()).hexdigest()}
    levels = np.unique(mesh.p[2])
    if (not np.array_equal(levels, np.unique(reference.p[2]))
            or not np.array_equal(levels, np.linspace(-thickness / 2, thickness / 2, 5))):
        raise ValueError("The z planes differ from h80 or the requested four layers.")

    current_free = mesh.facets[:, mesh.boundaries["free"]]
    reference_free = reference.facets[:, reference.boundaries["free"]]
    segments = mesh_inputs["settings"]["circle_resolution"]
    holes = mesh_inputs["pattern"]["holes"]
    maximum = 0.0
    for hole in holes:
        old_faces, old_nodes = hole_wall(reference, reference_free, levels, hole, segments)
        new_faces, new_nodes = hole_wall(mesh, current_free, levels, hole, segments)
        if not np.array_equal(old_faces, new_faces):
            raise ValueError(f"The triangulation of the hole wall at x={hole['x_m']} m differs from h80.")
        maximum = max(maximum, float(np.max(np.linalg.norm(new_nodes - old_nodes, axis=0))))
    if maximum > WALL_VERTEX_TOLERANCE_M:
        raise ValueError("The h60 hole-wall coordinates differ from h80 beyond the mesh comparison tolerance.")
    return {"ports": ports, "exact_h80_port_triangles_and_end_planes": True, "z_planes_m": levels.tolist(),
            "exact_h80_z_planes": True, "hole_count": len(holes), "rim_segments_per_hole_per_plane": segments,
            "wall_facets_per_hole": 2 * segments * (len(levels) - 1), "exact_h80_wall_connectivity": True,
            "maximum_h80_hole_wall_vertex_change_m": maximum, "hole_wall_vertex_tolerance_m": WALL_VERTEX_TOLERANCE_M}


def preflight(reference, output, identity):
    geometry, _, pattern = workflow._resolve_geometry(reference.config)
    mesh_dir, mesh_hash, inputs = workflow._mesh_location(output, geometry, pattern, reference.config["mesh"], identity)
    original = reference.mesh_record["mesh_inputs"]
    expected = {**original, "settings": {**original["settings"], "mesh_size_nm": 60}}
    # Record the native Linux mesh identity while holding the physical meshing inputs fixed.
    if inputs != {**expected, "python": identity["python"], "platform": identity["platform"]}:
        raise ValueError("The physical inputs, mesher sources or package versions differ from the fixed h80 reference.")
    result = workflow.preflight_mesh(reference.config, output)
    if (result["mesh_hash"] != mesh_hash or result["reused"] or result["end_face_dofs"] != PORT_DOF_COUNTS
            or result["ndof"] <= reference.mesh_record["ndof"] or result["hole_count"] != 26
            or result["max_dofs"] != 60_000 or result["within_body_budget"]):
        raise ValueError("Unexpected h60 mesh identity, geometry, DOF counts or standard reporting cap.")
    mesh, record = workflow.load_checkpoint(mesh_dir, mesh_hash)
    boundaries = verify_boundaries(reference.mesh, mesh, inputs)
    layout = displacement_layout(mesh)
    if layout.N != record["ndof"]:
        raise ValueError("The h60 displacement layout differs from the reported mesh DOF count.")
    left, right = map_trace(mesh, layout, reference.trace_order)
    return {"status": "mesh_ready", "preflight": result, "boundary_verification": boundaries,
            "mesh_summary": artifact(mesh_dir / "summary_AP_pnc.json"), "mesh": artifact(mesh_dir / "mesh_AP_pnc.npz"),
            "trace_sha256": workflow.digest([left.tolist(), right.tolist()]), "case_max_dofs": layout.N}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("mode", choices=("preflight",))
    parser.add_argument("--h80-config", type=Path, required=True)
    parser.add_argument("--h80-mesh-dir", type=Path, required=True)
    parser.add_argument("--ports", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    identity = workflow.source_identity()
    record = {**AUTHOR, "mode": args.mode, "status": "running", "implementation": identity,
              "driver": artifact(Path(__file__)), "trace_helpers": artifact(ROOT / "tools/run_lawrence_uniform_h80_AP_pnc.py"),
              "hostname": socket.gethostname(), "slurm_job_id": os.environ.get("SLURM_JOB_ID"),
              "numerical_thread_environment": {name: os.environ.get(name) for name in lawrence.THREADS},
              "safe_eigensolve_performed": False, "assembly_performed": False, "solve_performed": False,
              "scope": "Full-device global in-plane refinement includes the lead interiors."}
    args.output.mkdir(parents=True, exist_ok=True)
    receipt = args.output / RECEIPT
    with receipt.open("x", encoding="utf-8") as stream:
        json.dump(record, stream, indent=2)
    try:
        if workflow.digest(identity["source_sha256"]) != SOURCE_MAP_SHA256 or identity["packages"] != PACKAGES:
            raise ValueError("The current numerical sources or versions differ from the reviewed Lawrence environment.")
        reference = read_reference(args.h80_config, args.h80_mesh_dir, args.ports)
        record.update(configuration=reference.config, h80_inputs=reference.artifacts, h80_mesh_hash=H80_MESH_HASH,
                      incident_port_roots=list(lawrence.INCIDENT_ROOTS))
        record.update(preflight(reference, args.output, identity))
    except BaseException as error:
        record.update(status="interrupted" if isinstance(error, KeyboardInterrupt) else "failed",
                      error=f"{type(error).__name__}: {error}")
        raise
    finally:
        workflow.write_json(receipt, record)
    print(f"{record['status']}: {receipt.resolve()} ({record['case_max_dofs']} displacement DOFs)", flush=True)


if __name__ == "__main__":
    main()
