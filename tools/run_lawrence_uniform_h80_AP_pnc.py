"""Preflight and solve the fixed-port h80 uniform control. Author: Athul Prem | _AP_pnc."""

from __future__ import annotations

import argparse
from dataclasses import dataclass, fields
import hashlib
import json
import os
from pathlib import Path
import socket
import sys
from time import monotonic
from types import SimpleNamespace

import numpy as np
from scipy.sparse.linalg import splu
from skfem import Dofs, ElementTetP2, ElementVector, MeshTet

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT / "src"), str(ROOT)]

from tools import lawrence_solver_AP_pnc as lawrence
from phononic import workflow_AP_pnc as workflow
from phononic.assembly_batched_AP_pnc import _DisplacementDofLayout
from phononic.lead_modes_AP_pnc import _end_dof_pairs
from phononic.linear_lgmres_AP_pnc import solve_ilu_lgmres
from phononic.material_AP_pnc import CubicMaterial
from phononic.port_modes_AP_pnc import PortModes

AUTHOR = {"author": "Athul Prem", "file_tag": "_AP_pnc"}
H100 = {
    "summary_AP_pnc.json": "78f35d10fb3e6c87a845ee6959e97fede408328acb9a268aeadaa6231d1fd9e9",
    "selected_fields_AP_pnc.npz": "9d7116746086187def6e5f05df5c8aa4595a91af49bce0f1c97688f0937a634b",
    "mesh/summary_AP_pnc.json": "2aee71878b146f8a0e1913e7f1def3e31ee7b4059a01be48126e45d62cbea4a4",
    "mesh/mesh_AP_pnc.npz": "c5374505a9ae3e6f3b05f2ee601bd4f0c33f2ec05de289a66453a1514b392cbb",
}
PORTS_SHA = "18c0ad4c8b28e597503a28bd5157c867bb09744c94e1eb1aff23f024cc879c8b"
SOURCE_MAP_SHA = "b4003df3dda37617a26658100dd562278a9e301ca514928ab815a0ec0438039f"
LEGACY_PREFLIGHT_DRIVER_SHA = "148a0c285a845a27db7bf09378b326d7db19db42c664d8ad049899a6e5137425"
LEGACY_PREFLIGHT_RECEIPT_SHA = "0c5ca02942c87820a0e18fd0765967941cd57a13269448bf745d032191f7ebe1"
DIRECT_METHOD = "lawrence_uniform_superlu_direct"
TRACE_DOFS = 675
TARGET = 1e-10
RESIDUAL_LIMIT = 1e-9
RECEIPT = "uniform_h80_AP_pnc.json"
THREAD_VARIABLES = ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "BLIS_NUM_THREADS",
                    "NUMEXPR_NUM_THREADS")


@dataclass
class Reference:
    config: dict
    mesh: MeshTet
    trace_order: tuple
    incident_roots: list[int]
    modes: dict
    provenance: dict


def artifact(path):
    with path.open("rb") as stream:
        return {"path": str(path.resolve()), "sha256": hashlib.file_digest(stream, "sha256").hexdigest(),
                "bytes": path.stat().st_size}


def read_json(path):
    return json.loads(path.read_bytes())


def trace_keys(coords, component, ids):
    return tuple((y, z, int(c)) for (y, z), c in zip(coords[ids, 1:], component[ids]))


def check_channels(report, roots):
    channels = report.get("channels", [])
    if (report["lgmres"]["rtol"] != TARGET or report["relative_residual_limit"] != RESIDUAL_LIMIT
            or report["backward_residual_limit"] != RESIDUAL_LIMIT
            or [row["column"] for row in channels] != list(range(len(roots)))
            or any(row["info"] != 0 or not 0 <= row["relative_residual"] <= RESIDUAL_LIMIT
                   or not 0 <= row["relative_backward_residual"] <= RESIDUAL_LIMIT for row in channels)):
        raise ValueError("At least one drive failed the original-equation acceptance checks.")


def read_reference(directory, ports, identity):
    inputs = {name: artifact(directory / name) for name in H100}
    saved_ports = artifact(ports)
    if any(inputs[name]["sha256"] != expected for name, expected in H100.items()) or saved_ports["sha256"] != PORTS_SHA:
        raise ValueError("A saved h100 control or SAFE input differs from the verified original.")
    summary = read_json(directory / "summary_AP_pnc.json")
    if (summary["status"] != "completed_unverified" or summary["power_balance_passed"] is not True
            or summary["matched_reference_passed"] is not True or summary["hole_count"] != 0
            or identity["packages"] != summary["implementation"]["packages"]):
        raise ValueError("The h100 reference or numerical package versions differ from the verified control.")
    check_channels(summary["linear_solver"], summary["incident_mode_indices"])
    checkpoint = summary["mesh_checkpoint"]
    mesh, mesh_record = workflow.load_checkpoint(directory / "mesh", checkpoint["mesh_hash"])
    if mesh_record["mesh_sha256"] != checkpoint["mesh_sha256"] or mesh_record["ndof"] != summary["ndof"]:
        raise ValueError("The h100 checkpoint differs from the completed h100 control.")
    with (np.load(directory / "selected_fields_AP_pnc.npz", allow_pickle=False) as public,
          np.load(ports, allow_pickle=False) as stored):
        modes = {field.name: stored[field.name] for field in fields(PortModes) if field.name != "port"}
        for name in ("wavenumbers_rad_m", "displacements", "axial_forces"):
            if not np.array_equal(public[f"port_{name}"], modes[name]):
                raise ValueError(f"The staged SAFE array {name} differs from the h100 control.")
        coords, component = public["dof_coordinates_m"], public["dof_component"]
        planes = (float(mesh.p[0].min()), float(mesh.p[0].max()))
        orders = []
        for side, plane in zip(("left", "right"), planes):
            ids = public[f"port_{side}_dofs"]
            if ids.shape != (TRACE_DOFS,) or not np.all(coords[ids, 0] == plane):
                raise ValueError(f"The h100 {side} port no longer identifies its original end plane.")
            orders.append(trace_keys(coords, component, ids))
        if orders[0] != orders[1] or len(set(orders[0])) != TRACE_DOFS:
            raise ValueError("The saved h100 port rows must share one unique transverse component order.")
        forward, backward = (modes[name][modes["propagating"][modes[name]]]
                             for name in ("positive_indices", "negative_indices"))
        if (float(modes["frequency_hz"]) != summary["frequency_hz"] or len(forward) != 5 or len(backward) != 5
                or not np.array_equal(forward, public["incident_mode_indices"])
                or not np.array_equal(forward, public["transmission_mode_indices"])
                or not np.array_equal(backward, public["reflection_mode_indices"])):
            raise ValueError("The complete SAFE basis does not reproduce the saved five-channel order.")
        modes["frequency_hz"] = modes["frequency_hz"].item()
    config = summary["configuration"]
    config["name"] = "uniform_3um_layered_h80_ny12_z4"
    config["mesh"]["mesh_size_nm"] = 80
    config["resources"]["max_dofs"] = 60_000
    provenance = {"h100_inputs": inputs, "h100_implementation": summary["implementation"],
                  "original_h100_safe_source": summary["cached_port_provenance"]["source"]["modes"],
                  "reused_safe_artifact": saved_ports, "safe_eigensolve_performed": False}
    return Reference(workflow.resolve_config(config), mesh, orders[0], forward.tolist(), modes, provenance)


def end_triangles(mesh, side):
    faces = mesh.facets[:, mesh.boundaries[side]]
    return sorted(tuple(sorted(map(tuple, triangle))) for triangle in mesh.p[1:, faces].transpose(2, 1, 0))


def map_trace(mesh, basis, reference):
    old_mesh = reference.mesh
    if (not np.array_equal(np.unique(mesh.p[2]), np.unique(old_mesh.p[2]))
            or mesh.p[0].min() != old_mesh.p[0].min() or mesh.p[0].max() != old_mesh.p[0].max()
            or any(end_triangles(mesh, side) != end_triangles(old_mesh, side) for side in ("xleft", "xright"))):
        raise ValueError("The h80 end planes, tagged transverse triangulations or thickness planes differ from h100.")
    _, left, right = _end_dof_pairs(SimpleNamespace(mesh=mesh, basis=basis))
    component = np.empty(basis.N, dtype=np.int8)
    for axis, indices in enumerate(basis.split_indices()):
        component[indices] = axis
    mapped = []
    for ids in (left, right):
        lookup = dict(zip(trace_keys(basis.doflocs.T, component, ids), ids))
        if len(ids) != TRACE_DOFS or lookup.keys() != set(reference.trace_order):
            raise ValueError("The h80 port is missing an exact transverse-coordinate/component match.")
        mapped.append(np.asarray([lookup[key] for key in reference.trace_order]))
    return tuple(mapped)


def preflight(reference, output):
    result = workflow.preflight_mesh(reference.config, output)
    mesh_dir = Path(result["output_directory"])
    mesh, record = workflow.load_checkpoint(mesh_dir, result["mesh_hash"])
    element = ElementVector(ElementTetP2())
    layout = _DisplacementDofLayout(mesh, element, Dofs(mesh, element), 4)
    if (result["reused"] or layout.N != record["ndof"] or record["hole_count"] != 0
            or record["end_face_dofs"] != {"xleft": TRACE_DOFS, "xright": TRACE_DOFS}):
        raise ValueError("Unexpected h80 preflight geometry, body DOF count or port DOF count.")
    left, right = map_trace(mesh, layout, reference)
    return {"status": "mesh_ready", "preflight": result, "mesh_summary": artifact(mesh_dir / "summary_AP_pnc.json"),
            "trace_sha256": workflow.digest([left.tolist(), right.tolist()]), "case_max_dofs": layout.N}


def direct_factor_admission(operator):
    if sys.platform != "linux" or not os.environ.get("SLURM_JOB_ID"):
        raise ValueError("Direct factorization requires a Linux Slurm allocation.")
    quota = int(os.environ.get("SLURM_MEM_PER_NODE", "0")) * lawrence.MIB
    controls = lawrence.cgroup_controls()
    if quota <= 0 or not any(cap is not None and lawrence.is_job_group(directory)
                             for directory, cap, *_ in controls):
        raise ValueError("The Slurm node quota or bounded job memory cgroup is unavailable.")
    snapshot = lawrence.memory_snapshot(quota, controls)
    csc_bytes = sum(array.nbytes for array in (operator.data, operator.indices, operator.indptr))
    # Match SciPy 1.18.1 SuperLU's initial allocation estimate and keep explicit reserves.
    incremental_bytes = 30 * 40 * operator.nnz + 2 * csc_bytes + 512 * lawrence.MIB
    job_required = incremental_bytes + lawrence.JOB_RESERVE
    node_required = incremental_bytes + lawrence.NODE_RESERVE
    return {"slurm_node_quota_bytes": quota, "estimated_incremental_bytes": incremental_bytes,
            "job_headroom_required_bytes": job_required, "node_available_required_bytes": node_required,
            "before_factor": snapshot, "admitted": (snapshot["job_headroom_bytes"] >= job_required
                                                       and snapshot["node_available_bytes"] >= node_required)}


def solve_direct_original(operator, rhs, roots, output, report):
    if (operator.format != "csc" or operator.dtype != np.complex128 or rhs.dtype != np.complex128
            or rhs.shape != (operator.shape[0], len(roots)) or operator.shape[0] != operator.shape[1]
            or not np.isfinite(operator.data).all() or not np.isfinite(rhs).all()):
        raise ValueError("Direct factorization requires finite complex128 CSC equations and ordered drives.")
    norm = lawrence.infinity_norm(operator)
    report.update(method=DIRECT_METHOD, working_dtype=str(operator.dtype),
                  operator_csc_sha256=lawrence.array_sha(operator.data, operator.indices, operator.indptr),
                  rhs_array_sha256=lawrence.array_sha(rhs), operator_nonzeros=int(operator.nnz),
                  relative_residual_limit=RESIDUAL_LIMIT, backward_residual_limit=RESIDUAL_LIMIT, channels=[])
    receipt_path = output / "direct_channels_AP_pnc.json"
    receipt = {**AUTHOR, "status": "admission", "solver_report": report}
    try:
        admission = direct_factor_admission(operator)
        report["factor_admission"] = admission
        if not admission["admitted"]:
            raise RuntimeError("The original operator does not fit the current direct factorization budget.")
        receipt["status"] = "factorizing"
        workflow.write_json(receipt_path, receipt)
        started = monotonic()
        factor = splu(operator.copy(), permc_spec="COLAMD")
        report["factor"] = {"permc_spec": "COLAMD", "elapsed_seconds": monotonic() - started,
                            "nonzeros": int(factor.nnz)}
        receipt["status"] = "solving"
        workflow.write_json(receipt_path, receipt)
        result = np.empty_like(rhs)
        for column, root in enumerate(roots):
            b = rhs[:, column]
            vector = factor.solve(b)
            metrics = lawrence.residual_metrics(operator @ vector - b, b, vector, norm)
            passed = all(0 <= value <= RESIDUAL_LIMIT for value in metrics.values())
            row = {"column": column, "incident_port_root": root, "info": 0 if passed else 1, **metrics}
            report["channels"].append(row)
            if not passed:
                raise RuntimeError(f"Direct drive {column + 1} failed the original-equation residual checks.")
            row["accepted_vector"] = lawrence.write_vector(output / f"accepted-{column:03d}_AP_pnc.npy", vector)
            result[:, column] = vector
            workflow.write_json(receipt_path, receipt)
        report["original_relative_target_met"] = all(row["relative_residual"] <= TARGET for row in report["channels"])
        receipt["status"] = "drives_accepted"
        return result
    except BaseException as error:
        receipt.update(status="interrupted" if isinstance(error, KeyboardInterrupt) else "failed",
                       error=f"{type(error).__name__}: {error}")
        raise
    finally:
        workflow.write_json(receipt_path, receipt)


def solve(reference, output, preflight_dir, max_dofs, execution, *, direct=False):
    preflight_path = preflight_dir / RECEIPT
    preflight_artifact = artifact(preflight_path)
    ready = read_json(preflight_path)
    same_driver = ready["driver"] == execution["driver"]
    legacy_driver = (direct and ready["driver"]["sha256"] == LEGACY_PREFLIGHT_DRIVER_SHA
                     and preflight_artifact["sha256"] == LEGACY_PREFLIGHT_RECEIPT_SHA)
    if (ready["status"] != "mesh_ready" or ready["configuration"] != reference.config
            or ready["cached_port_provenance"] != reference.provenance or not (same_driver or legacy_driver)
            or any(ready["implementation"][key] != execution["implementation"][key]
                   for key in ("source_sha256", "packages", "python"))):
        raise ValueError("A matching independent h80 preflight is required before the solve.")
    geometry, _, _ = workflow._resolve_geometry(reference.config)
    mesh_dir, mesh_hash, _ = workflow._mesh_location(preflight_dir, geometry, None, reference.config["mesh"],
                                                     ready["implementation"])
    if artifact(mesh_dir / "summary_AP_pnc.json") != ready["mesh_summary"]:
        raise ValueError("The h80 mesh manifest changed after preflight.")
    mesh, record = workflow.load_checkpoint(mesh_dir, mesh_hash)
    if (record["mesh_sha256"] != ready["preflight"]["mesh_sha256"]
            or not record["ndof"] == ready["case_max_dofs"] == max_dofs):
        raise ValueError("The requested solve cap or saved h80 mesh differs from measured preflight.")
    config = reference.config
    if direct:
        config["resources"] = {key: value for key, value in config["resources"].items()
                               if not key.startswith(("ilu_", "lgmres_"))}
        config["resources"]["linear_solver"] = "direct"
    resources = config["resources"]
    resources["max_dofs"] = max_dofs
    params = config["material"]
    material = CubicMaterial(C11=params["C11_Pa"], C12=params["C12_Pa"], C44=params["C44_Pa"],
                             rho=params["density_kg_m3"], rotation=params["crystal_rotation"])
    with workflow.measured_resources() as assembly_resources:
        system = workflow._assemble_system(mesh, config, material)
    left, right = map_trace(mesh, system.basis, reference)
    if workflow.digest([left.tolist(), right.tolist()]) != ready["trace_sha256"]:
        raise ValueError("The assembled SAFE trace order differs from the verified preflight.")
    port = SimpleNamespace(elastic_system=system, basis=SimpleNamespace(N=TRACE_DOFS), left_dofs=left, right_dofs=right)
    modes = PortModes(port=port, **reference.modes)
    report = {}

    def solve_original(operator, rhs):
        if direct:
            return solve_direct_original(operator, rhs, reference.incident_roots, output, report)
        result = solve_ilu_lgmres(operator, rhs, report=report, fill_factor=resources["ilu_fill_factor"],
                                  single_precision=resources["ilu_precision"] == "single", **workflow.lgmres_options(resources))
        check_channels(report, reference.incident_roots)
        report["original_relative_target_met"] = all(row["relative_residual"] <= TARGET for row in report["channels"])
        return result

    provenance = {**reference.provenance, "preflight_receipt": preflight_artifact, "preflight_driver": ready["driver"],
                  "preflight_case_max_dofs": ready["case_max_dofs"], "reused_preflight_mesh": True,
                  "end_planes_m": [float(mesh.p[0].min()), float(mesh.p[0].max())],
                  "trace_sha256": ready["trace_sha256"], "linux_assembly_and_solve": execution}
    if direct:
        provenance.update(method=DIRECT_METHOD, direct_solver_helpers=artifact(Path(lawrence.__file__)),
                          legacy_preflight_driver_reused=legacy_driver)
    prepared = {"mesh_directory": mesh_dir, "mesh_checkpoint": (mesh, record), "system": system,
                "pattern": None, "assembly_resources": assembly_resources, "port_modes": modes,
                "solver_report": report, "linear_solver": solve_original,
                "attempt": workflow.digest(provenance), "provenance": provenance}
    result = workflow.run_study(config, output, _prepared_scattering=prepared)
    if (result["status"] != "completed_unverified" or result["power_balance_passed"] is not True
            or result["matched_reference_passed"] is not True or result["incident_mode_indices"] != reference.incident_roots):
        raise ValueError(f"The independently recovered flux or matched uniform checks failed in {result['output_directory']}.")
    receipt = {"status": "completed_unverified", "case_max_dofs": max_dofs,
               "public_summary": artifact(Path(result["output_directory"]) / "summary_AP_pnc.json")}
    if direct:
        receipt["direct_channels"] = artifact(output / "direct_channels_AP_pnc.json")
    return receipt


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("mode", choices=("preflight", "solve", "solve-direct"))
    parser.add_argument("--h100", type=Path, required=True)
    parser.add_argument("--ports", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--preflight-dir", type=Path)
    parser.add_argument("--max-dofs", type=int)
    args = parser.parse_args()
    if args.mode != "preflight" and (args.preflight_dir is None or args.max_dofs is None or args.max_dofs <= 0):
        parser.error("The solve requires --preflight-dir and its measured --max-dofs.")
    if args.mode == "preflight" and (args.preflight_dir is not None or args.max_dofs is not None):
        parser.error("The preflight uses the standard reporting cap.")
    identity = workflow.source_identity()
    execution = {"implementation": identity, "driver": artifact(Path(__file__)), "hostname": socket.gethostname(),
                 "slurm_job_id": os.environ.get("SLURM_JOB_ID"),
                 "numerical_thread_environment": {name: os.environ.get(name) for name in THREAD_VARIABLES}}
    receipt = args.output / RECEIPT
    record = {**AUTHOR, **execution, "status": "running", "mode": args.mode}
    with receipt.open("x", encoding="utf-8") as stream:
        json.dump(record, stream, indent=2)
    try:
        if workflow.digest(identity["source_sha256"]) != SOURCE_MAP_SHA:
            raise ValueError("The numerical source differs from the reviewed h80 implementation.")
        reference = read_reference(args.h100, args.ports, identity)
        record.update(configuration=reference.config, cached_port_provenance=reference.provenance)
        if args.mode == "preflight":
            record.update(preflight(reference, args.output))
        else:
            record.update(solve(reference, args.output, args.preflight_dir, args.max_dofs, execution,
                                direct=args.mode == "solve-direct"))
    except BaseException as error:
        record.update(status="interrupted" if isinstance(error, KeyboardInterrupt) else "failed",
                      error=f"{type(error).__name__}: {error}")
        raise
    finally:
        workflow.write_json(receipt, record)
    print(f"{record['status']}: {receipt.resolve()} ({record['case_max_dofs']} displacement DOFs)", flush=True)


if __name__ == "__main__":
    main()
