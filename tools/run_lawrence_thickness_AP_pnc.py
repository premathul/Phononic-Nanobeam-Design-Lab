"""Preflight and extract SAFE modes for the h80 z6 control. Athul Prem | _AP_pnc."""

from __future__ import annotations

import argparse
from dataclasses import fields
import hashlib
import json
import os
from pathlib import Path
import socket
import sys
from types import SimpleNamespace

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT / "src"), str(ROOT)]

from phononic import workflow_AP_pnc as workflow
from phononic.material_AP_pnc import CubicMaterial
from phononic.port_modes_AP_pnc import PortModes, assemble_port, solve_port_modes
from tools import lawrence_solver_AP_pnc as lawrence
from tools import run_lawrence_device_refinement_AP_pnc as fixed_port
from tools import run_lawrence_uniform_h80_AP_pnc as uniform
from tools.run_lawrence_uniform_h80_AP_pnc import artifact, end_triangles


PREFLIGHT = "thickness_h80_preflight_AP_pnc.json"
SAFE = "thickness_h80_safe_AP_pnc.json"
PORTS = "ports_AP_pnc.npz"
FIXED_PORT_HELPER_SHA256 = "50bb2bb36319b9a1d4af67ef1105a702381488d41df71a3ebfbc5ed1d5376535"
UNIFORM_HELPER_SHA256 = "1944ddfeb5c5ace7d2842086996aab658ce999803ab225ce61e4a44078db3013"
SOLVE_HELPER_SHA256 = "dd9e07d4955de6cf4cac97eec0485bfc34bd1d7f8173d4707718c2ea4ba89034"
LAYERS = 6
PORT_DOFS = 975
BODY_DOFS = 616239
TETRAHEDRA = 130968
VERTICES = 29750
NUMERICAL_KEYS = ("source_sha256", "packages", "python")


def execution(driver, mode):
    identity = workflow.source_identity()
    threads = {name: os.environ.get(name) for name in lawrence.THREADS}
    job = os.environ.get("SLURM_JOB_ID")
    if (sys.platform != "linux" or not job or threads != lawrence.THREADS
            or workflow.digest(identity["source_sha256"]) != fixed_port.SOURCE_MAP_SHA256
            or identity["packages"] != fixed_port.PACKAGES):
        raise ValueError("Use the reviewed single-threaded Linux Slurm environment.")
    helpers = {"fixed_port": (fixed_port, FIXED_PORT_HELPER_SHA256), "uniform": (uniform, UNIFORM_HELPER_SHA256),
               "solver": (lawrence, SOLVE_HELPER_SHA256)}
    checked = {name: artifact(Path(module.__file__)) for name, (module, _) in helpers.items()}
    if any(checked[name]["sha256"] != expected for name, (_, expected) in helpers.items()):
        raise ValueError("The reviewed footprint, admission or original-equation helper source differs.")
    return {**fixed_port.AUTHOR, "mode": mode, "status": "running", "implementation": identity,
            "driver": artifact(driver), "hostname": socket.gethostname(), "slurm_job_id": job,
            "numerical_thread_environment": threads, "numerical_backend": lawrence.solve_environment(), "helpers": checked}


def read_receipt(directory, name, checksum, status, expected_driver, current):
    path = directory / name
    link = artifact(path)
    if link["sha256"] != checksum:
        raise ValueError(f"Receipt checksum differs: {path}")
    record = json.loads(path.read_bytes())
    if (record["status"] != status or record["driver"]["sha256"] != expected_driver
            or any(record["implementation"][key] != current["implementation"][key] for key in NUMERICAL_KEYS)):
        raise ValueError(f"The prior phase or numerical environment differs: {path}")
    return record, link


def footprint(mesh):
    facets = mesh.facets[:, mesh.boundary_facets()]
    lower = mesh.p[2].min()
    triangles = facets[:, np.all(mesh.p[2, facets] == lower, axis=0)]
    return sorted(tuple(sorted(map(tuple, face))) for face in mesh.p[:2, triangles].transpose(2, 1, 0))


def section_triangles(y, z):
    result = []
    for y0, y1 in zip(y[:-1], y[1:], strict=True):
        for z0, z1 in zip(z[:-1], z[1:], strict=True):
            a, b, c, d = (y0, z0), (y1, z0), (y0, z1), (y1, z1)
            result.extend((tuple(sorted((a, b, d))), tuple(sorted((a, c, d)))))
    return sorted(result)


def verify_geometry(reference, change, inputs):
    # Connectivity is expressed in physical coordinates so numbering changes are harmless.
    old_footprint, new_footprint = footprint(reference), footprint(change)
    if old_footprint != new_footprint or len(new_footprint) != 7276:
        raise ValueError("The h80 z6 in-plane footprint differs from the archived h80 z4 mesh.")
    footprint_sha = hashlib.sha256(np.asarray(new_footprint, dtype="<f8").tobytes()).hexdigest()
    length, width, thickness = inputs["dimensions_m"]
    y = np.linspace(-width / 2, width / 2, 13)
    z = np.linspace(-thickness / 2, thickness / 2, LAYERS + 1)
    if not np.array_equal(np.unique(change.p[2]), z):
        raise ValueError("The z6 mesh lacks the seven prescribed thickness planes.")
    expected = section_triangles(y, z)
    ends = {}
    for side, x in (("xleft", 0), ("xright", length)):
        target = change.facets[:, change.boundaries[side]]
        original = reference.facets[:, reference.boundaries[side]]
        if (not np.all(change.p[0, target] == x) or not np.all(reference.p[0, original] == x)
                or not np.array_equal(np.unique(reference.p[1, original]), y)
                or end_triangles(change, side) != expected):
            raise ValueError(f"The {side} end plane, width grid, or z6 triangulation differs.")
        ends[side] = {"x_m": x, "triangles": len(expected), "trace_dofs": PORT_DOFS}
    old_z = np.unique(reference.p[2])
    old_free, new_free = (mesh.facets[:, mesh.boundaries["free"]] for mesh in (reference, change))
    holes = inputs["pattern"]["holes"]
    for hole in holes:
        old_faces, old_nodes = fixed_port.hole_wall(reference, old_free, old_z, hole, 32)
        new_faces, new_nodes = fixed_port.hole_wall(change, new_free, z, hole, 32)
        old_rim = old_nodes[:2, :32]
        if (not np.array_equal(new_nodes[:2], np.tile(old_rim, (1, len(z))))
                or len(old_faces) != 256 or len(new_faces) != 384):
            raise ValueError(f"A physical hole wall changed at x={hole['x_m']} m.")
        rim_edges = np.sort(new_faces % 32, axis=1)
        if not np.all(np.isin(rim_edges[:, 2] - rim_edges[:, 0], (1, 31))):
            raise ValueError(f"A z6 wall face omits the original adjacent rim chord at x={hole['x_m']} m.")
    return {"exact_h80_xy_footprint": True, "xy_triangles": len(new_footprint), "xy_footprint_sha256": footprint_sha,
            "port_end_planes": ends, "z_planes_m": z.tolist(), "wall_facets_per_hole": 384,
            "hole_count": len(holes), "rim_segments_per_hole_per_plane": 32, "exact_h80_wall_rim_coordinates": True}


def preflight(args, record):
    paths = {"configuration": args.h80_config, "mesh_summary": args.h80_mesh_dir / "summary_AP_pnc.json",
             "mesh": args.h80_mesh_dir / "mesh_AP_pnc.npz"}
    saved = {name: artifact(path) for name, path in paths.items()}
    if any(item["sha256"] != fixed_port.H80_INPUT_SHA256[name] for name, item in saved.items()):
        raise ValueError("An original h80 configuration or mesh artifact differs.")
    reference, reference_record = workflow.load_checkpoint(args.h80_mesh_dir, fixed_port.H80_MESH_HASH)
    config = json.loads(args.h80_config.read_bytes())
    config["name"] = "historical_26hole_layered_h80_ny12_z6"
    config["mesh"]["thickness_layers"] = LAYERS
    config["resources"]["max_port_dofs"] = PORT_DOFS
    config = workflow.resolve_config(config, check_body_budget=False)
    geometry, _, pattern = workflow._resolve_geometry(config)
    identity = record["implementation"]
    mesh_dir, mesh_hash, inputs = workflow._mesh_location(args.output, geometry, pattern, config["mesh"], identity)
    unchanged = reference_record["mesh_inputs"]
    expected = {**unchanged, "python": identity["python"], "platform": identity["platform"],
                "settings": {**unchanged["settings"], "thickness_layers": LAYERS}}
    if inputs != expected:
        raise ValueError("The resolved in-plane or source inputs differ from the original h80 case.")
    result = workflow.preflight_mesh(config, args.output)
    if (result["mesh_hash"] != mesh_hash or result["reused"] or result["within_body_budget"]
            or result["max_dofs"] != 60_000 or result["end_face_dofs"] != {"xleft": PORT_DOFS, "xright": PORT_DOFS}
            or (result["ndof"], result["mesh_tetrahedra"], result["mesh_vertices"], result["hole_count"])
            != (BODY_DOFS, TETRAHEDRA, VERTICES, 26)):
        raise ValueError("Unexpected h80 z6 mesh counts or case settings.")
    mesh, _ = workflow.load_checkpoint(mesh_dir, mesh_hash)
    record.update(h80_inputs=saved, configuration=config, mesh_preflight=result)
    proof = verify_geometry(reference, mesh, inputs)
    record.update(status="mesh_ready", geometry_verification=proof, mesh_hash=mesh_hash,
                  mesh_summary=artifact(mesh_dir / "summary_AP_pnc.json"), mesh=artifact(mesh_dir / "mesh_AP_pnc.npz"),
                  displacement_dofs=BODY_DOFS, assembly_performed=False, safe_eigensolve_performed=False)


def read_preflight_mesh(directory, ready):
    mesh_dir = directory / "_meshes" / ready["mesh_hash"]
    if (artifact(mesh_dir / "summary_AP_pnc.json") != ready["mesh_summary"]
            or artifact(mesh_dir / "mesh_AP_pnc.npz") != ready["mesh"]):
        raise ValueError("The saved h80 z6 mesh differs from its preflight.")
    mesh, mesh_record = workflow.load_checkpoint(mesh_dir, ready["mesh_hash"])
    return mesh_dir, mesh, mesh_record


def material_from(config):
    params = config["material"]
    return CubicMaterial(C11=params["C11_Pa"], C12=params["C12_Pa"], C44=params["C44_Pa"],
                         rho=params["density_kg_m3"], rotation=params["crystal_rotation"])


def safe(args, record):
    ready, link = read_receipt(args.preflight_dir, PREFLIGHT, args.preflight_sha256, "mesh_ready",
                               record["driver"]["sha256"], record)
    _, mesh, mesh_record = read_preflight_mesh(args.preflight_dir, ready)
    config = ready["configuration"]
    layout = fixed_port.displacement_layout(mesh)
    if layout.N != mesh_record["ndof"] or layout.N != ready["displacement_dofs"]:
        raise ValueError("The SAFE layout differs from the approved volume mesh.")
    freq = config["analysis"]["frequency_GHz"] * 1e9
    with workflow.measured_resources() as resources:
        port = assemble_port(SimpleNamespace(mesh=mesh, basis=layout), material_from(config), max_dofs=PORT_DOFS)
        modes = solve_port_modes(port, freq)
    roots = [ids[modes.propagating[ids]].tolist() for ids in (modes.positive_indices, modes.negative_indices)]
    maximum_residual = float(np.max(modes.relative_residuals))
    if (port.basis.N != PORT_DOFS or modes.displacements.shape != (PORT_DOFS, 2 * PORT_DOFS)
            or [len(side) for side in roots] != [5, 5] or not 0 <= maximum_residual < 1e-8):
        raise ValueError("The h80 z6 SAFE section lacks the expected roots, channels or residual accuracy.")
    path = args.output / PORTS
    with path.open("xb") as stream:
        np.savez_compressed(stream, left_dofs=port.left_dofs, right_dofs=port.right_dofs,
                            section_vertices_yz_m=port.basis.mesh.p.T, section_triangles=port.basis.mesh.t.T,
                            coordinates_yz_m=port.basis.doflocs.T, component_dofs_xyz=np.asarray(port.basis.split_indices()),
                            K0=port.K0, Q=port.Q, G=port.G, M=port.M, reference_speed_m_s=port.reference_speed_m_s,
                            **{field.name: getattr(modes, field.name) for field in fields(PortModes) if field.name != "port"})
    record.update(status="safe_ready", preflight_receipt=link, mesh=ready["mesh"], configuration=config, ports=artifact(path),
                  frequency_hz=freq, trace_dofs=PORT_DOFS, discrete_roots=2 * PORT_DOFS, incident_port_roots=roots[0],
                  backward_port_roots=roots[1], maximum_port_relative_residual=maximum_residual,
                  trace_sha256=workflow.digest([port.left_dofs.tolist(), port.right_dofs.tolist()]),
                  resources=resources, safe_eigensolve_performed=True, body_assembly_performed=False)


def run_phase(args, driver, receipt_name, operation):
    args.output.mkdir(parents=True, exist_ok=True)
    record = execution(driver, args.mode)
    receipt = args.output / receipt_name
    with receipt.open("x", encoding="utf-8") as stream:
        json.dump(record, stream, indent=2)
    try:
        operation(args, record)
    except BaseException as error:
        record.update(status="interrupted" if isinstance(error, KeyboardInterrupt) else "failed",
                      error=f"{type(error).__name__}: {error}")
        raise
    finally:
        workflow.write_json(receipt, record)
    print(f"{record['status']}: {receipt.resolve()}", flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    phases = parser.add_subparsers(dest="mode", required=True)
    mesh_parser = phases.add_parser("preflight")
    mesh_parser.add_argument("--h80-config", type=Path, required=True)
    mesh_parser.add_argument("--h80-mesh-dir", type=Path, required=True)
    port_parser = phases.add_parser("safe")
    port_parser.add_argument("--preflight-dir", type=Path, required=True)
    port_parser.add_argument("--preflight-sha256", required=True)
    for phase in (mesh_parser, port_parser):
        phase.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    name, operation = (PREFLIGHT, preflight) if args.mode == "preflight" else (SAFE, safe)
    run_phase(args, Path(__file__), name, operation)


if __name__ == "__main__":
    main()
