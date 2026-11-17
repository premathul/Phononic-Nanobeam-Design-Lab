"""Stage, solve and replay the fixed-port h60 device. Author: Athul Prem | _AP_pnc."""

from __future__ import annotations

import argparse
from dataclasses import fields
import json
import os
from pathlib import Path
from shutil import disk_usage
import socket
import sys
from types import SimpleNamespace

import numpy as np
from scipy.sparse import load_npz, save_npz

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT / "src"), str(ROOT)]

from phononic import workflow_AP_pnc as workflow
from phononic.elasticity_AP_pnc import ElasticSystem
from phononic.material_AP_pnc import CubicMaterial
from phononic.port_modes_AP_pnc import PortModes
from phononic.scattering_AP_pnc import solve_scattering
from tools import lawrence_solver_AP_pnc as lawrence
from tools import run_lawrence_device_refinement_AP_pnc as preflight
from tools import run_lawrence_uniform_h80_AP_pnc as uniform
from tools.run_lawrence_uniform_h80_AP_pnc import artifact, direct_factor_admission


METHOD = "lawrence_device_superlu_direct"
PREFLIGHT_DRIVER_SHA256 = "50bb2bb36319b9a1d4af67ef1105a702381488d41df71a3ebfbc5ed1d5376535"
SOLVER_HELPER_SHA256 = "dd9e07d4955de6cf4cac97eec0485bfc34bd1d7f8173d4707718c2ea4ba89034"
FACTOR_ADMISSION_HELPER_SHA256 = "1944ddfeb5c5ace7d2842086996aab658ce999803ab225ce61e4a44078db3013"
BATCH_ELEMENTS = 2048
ROOTS = list(lawrence.INCIDENT_ROOTS)
NUMERICAL_KEYS = ("source_sha256", "packages", "python")
ARRAY_FILES = {"K": "K_AP_pnc.npz", "M": "M_AP_pnc.npz", "A": "A_AP_pnc.npz", "rhs": "rhs_AP_pnc.npy"}


class OperatorCaptured(Exception):
    """Stop the scattering calculation after staging its original equations."""


def read_receipt(path, checksum):
    link = artifact(path)
    if link["sha256"] != checksum:
        raise ValueError(f"Receipt SHA-256 differs: {path}")
    return json.loads(path.read_bytes()), link


def check_execution(previous, current):
    if (previous["driver"]["sha256"] != current["driver"]["sha256"]
            or previous["direct_solver_helpers"]["sha256"] != current["direct_solver_helpers"]["sha256"]
            or any(previous["implementation"][key] != current["implementation"][key] for key in NUMERICAL_KEYS)):
        raise ValueError("The saved driver or numerical environment differs from the current phase.")


def read_mesh(directory, link, mesh_hash):
    if artifact(directory / "summary_AP_pnc.json") != link:
        raise ValueError("The saved mesh manifest changed.")
    return workflow.load_checkpoint(directory, mesh_hash)


def matrix_bytes(matrix):
    return sum(array.nbytes for array in (matrix.data, matrix.indices, matrix.indptr))


def equation_hashes(operator, rhs):
    return {"operator_csc_sha256": lawrence.array_sha(operator.data, operator.indices, operator.indptr),
            "rhs_array_sha256": lawrence.array_sha(rhs)}


def make_ports(system, ids, link):
    path = Path(link["path"])
    if artifact(path) != link or link["sha256"] != preflight.H80_INPUT_SHA256["ports"]:
        raise ValueError("The saved h80 SAFE section changed.")
    left, right = (np.asarray(values) for values in ids)
    port = SimpleNamespace(elastic_system=system, basis=SimpleNamespace(N=len(left)), left_dofs=left, right_dofs=right)
    with np.load(path, allow_pickle=False) as data:
        modes = {field.name: data[field.name] for field in fields(PortModes)
                 if field.name not in ("port", "frequency_hz")}
        return PortModes(port=port, frequency_hz=data["frequency_hz"].item(), **modes)


def stage(args, record):
    ready, link = read_receipt(args.preflight_dir / preflight.RECEIPT, args.preflight_sha256)
    if (ready["status"] != "mesh_ready" or ready["driver"]["sha256"] != PREFLIGHT_DRIVER_SHA256
            or artifact(Path(preflight.__file__))["sha256"] != PREFLIGHT_DRIVER_SHA256
            or any(ready["implementation"][key] != record["implementation"][key] for key in NUMERICAL_KEYS)
            or ready["incident_port_roots"] != ROOTS):
        raise ValueError("An independent, reviewed h60 preflight is required before assembly.")
    inputs = ready["h80_inputs"]
    reference = preflight.read_reference(Path(inputs["configuration"]["path"]), Path(inputs["mesh"]["path"]).parent,
                                         Path(inputs["ports"]["path"]))
    if reference.artifacts != inputs or reference.config != ready["configuration"]:
        raise ValueError("The preflight reference inputs changed.")
    mesh_hash = ready["preflight"]["mesh_hash"]
    mesh_dir = args.preflight_dir / "_meshes" / mesh_hash
    mesh, mesh_record = read_mesh(mesh_dir, ready["mesh_summary"], mesh_hash)
    measured_dofs = ready["case_max_dofs"]
    if (mesh_record["ndof"] != measured_dofs or mesh_record["mesh_sha256"] != ready["mesh"]["sha256"]
            or mesh_record["end_face_dofs"] != preflight.PORT_DOF_COUNTS or mesh_record["hole_count"] != 26):
        raise ValueError("The h60 mesh differs from its independent preflight.")

    config = workflow.resolve_config(reference.config)
    config["resources"].update(max_dofs=measured_dofs, linear_solver="direct", assembly_batch_elements=BATCH_ELEMENTS)
    params = config["material"]
    material = CubicMaterial(C11=params["C11_Pa"], C12=params["C12_Pa"], C44=params["C44_Pa"],
                             rho=params["density_kg_m3"], rotation=params["crystal_rotation"])
    record.update(status="assembling", configuration=config, preflight_receipt=link, h80_inputs=inputs,
                  mesh_directory=str(mesh_dir.resolve()), mesh_summary=ready["mesh_summary"], mesh_hash=mesh_hash,
                  mesh=ready["mesh"], trace_sha256=ready["trace_sha256"], safe_eigensolve_performed=False,
                  incident_port_roots=ROOTS, factorization_performed=False)
    workflow.write_json(args.output / "stage_AP_pnc.json", record)
    with workflow.measured_resources() as assembly_resources:
        system = workflow._assemble_system(mesh, config, material)
    ids = [item.tolist() for item in preflight.map_trace(mesh, system.basis, reference.trace_order)]
    if workflow.digest(ids) != ready["trace_sha256"]:
        raise ValueError("The assembled SAFE order differs from the verified h60 preflight.")
    modes = make_ports(system, ids, inputs["ports"])

    def capture(operator, rhs):
        if (operator.format != "csc" or operator.dtype != np.complex128 or not operator.has_canonical_format
                or rhs.dtype != np.complex128 or rhs.shape != (measured_dofs, len(ROOTS))
                or operator.shape != (measured_dofs, measured_dofs)
                or not np.isfinite(rhs).all() or not np.isfinite(operator.data).all()):
            raise ValueError("Unexpected format or values in the original h60 equations.")
        storage = matrix_bytes(system.K) + matrix_bytes(system.M) + matrix_bytes(operator) + rhs.nbytes
        if disk_usage(args.output).free < storage + lawrence.GIB:
            raise RuntimeError("The measured stage arrays and 1 GiB disk reserve do not fit.")
        arrays = {}
        for name, matrix in (("K", system.K), ("M", system.M), ("A", operator)):
            path = args.output / ARRAY_FILES[name]
            save_npz(path, matrix, compressed=False)
            arrays[name] = artifact(path)
        path = args.output / ARRAY_FILES["rhs"]
        np.save(path, rhs, allow_pickle=False)
        arrays["rhs"] = artifact(path)
        record.update(status="operator_staged", array_files=arrays, **equation_hashes(operator, rhs),
                      displacement_dofs=measured_dofs, operator_nonzeros=int(operator.nnz),
                      operator_csc_bytes=matrix_bytes(operator), stage_array_bytes=storage,
                      assembly_resources=assembly_resources, port_dof_ids=ids)
        raise OperatorCaptured

    try:
        solve_scattering(system, modes, modes.frequency_hz, config["analysis"]["incident_side"], linear_solver=capture)
    except OperatorCaptured:
        return
    raise RuntimeError("The stage did not capture the original harmonic equations.")


def read_stage(args, current):
    saved, link = read_receipt(args.stage_dir / "stage_AP_pnc.json", args.stage_sha256)
    check_execution(saved, current)
    if (saved["status"] != "operator_staged" or saved["incident_port_roots"] != ROOTS
            or saved["safe_eigensolve_performed"] is not False or saved["factorization_performed"] is not False):
        raise ValueError("The h60 operator stage is incomplete or differs from the approved case.")
    return saved, link


def check_array_file(directory, name, saved):
    path = directory / ARRAY_FILES[name]
    if artifact(path) != saved["array_files"][name]:
        raise ValueError(f"Saved h60 array changed: {name}")
    return path


def check_equations(operator, rhs, saved):
    size = saved["displacement_dofs"]
    if (operator.format != "csc" or operator.dtype != np.complex128 or not operator.has_canonical_format
            or operator.shape != (size, size) or operator.nnz != saved["operator_nonzeros"]
            or rhs.dtype != np.complex128 or rhs.shape != (size, len(ROOTS))
            or any(saved[key] != digest for key, digest in equation_hashes(operator, rhs).items())):
        raise ValueError("Operator or right-hand side differs from the exact staged h60 equations.")


def replay_channels(source, expected_stage, operator, rhs, norm, *, complete):
    report, link = source
    channels = report["linear_solver"]["channels"]
    if (report["stage"] != expected_stage or complete and report["status"] != "drives_accepted"
            or len(channels) > len(ROOTS) or complete and len(channels) != len(ROOTS)):
        raise ValueError("The saved solve does not cover the required staged drives.")
    for column, row in enumerate(channels):
        vector = row["accepted_vector"]
        expected_file = f"accepted-{column:03d}_AP_pnc.npy"
        if (row["column"] != column or row["incident_port_root"] != ROOTS[column] or row["info"] != 0
                or not lawrence.accepted(row) or vector["file"] != expected_file):
            raise ValueError("The saved solve has an invalid or reordered accepted channel.")
        path = Path(link["path"]).parent / expected_file
        file = artifact(path)
        if file["bytes"] != vector["bytes"] or file["sha256"] != vector["sha256"]:
            raise ValueError("The accepted vector checksum or size differs.")
        displacement = np.load(path, allow_pickle=False)
        b = rhs[:, column]
        if displacement.dtype != np.complex128 or displacement.shape != b.shape or not np.isfinite(displacement).all():
            raise ValueError("The accepted displacement layout differs.")
        replay = lawrence.residual_metrics(operator @ displacement - b, b, displacement, norm)
        if not lawrence.accepted(replay):
            raise ValueError("An accepted vector failed original-equation replay.")
        yield column, {**row, **replay}, displacement


def solve(args, record):
    staged, link = read_stage(args, record)
    operator = load_npz(check_array_file(args.stage_dir, "A", staged))
    rhs = np.load(check_array_file(args.stage_dir, "rhs", staged), allow_pickle=False)
    check_equations(operator, rhs, staged)
    norm = lawrence.infinity_norm(operator)
    if disk_usage(args.output).free < 4 * rhs.nbytes + lawrence.GIB:
        raise RuntimeError("The bounded drive checkpoints and 1 GiB disk reserve do not fit.")
    receipt = args.output / "solve_AP_pnc.json"
    report = {"method": METHOD, "working_dtype": str(operator.dtype),
              **{key: staged[key] for key in ("operator_csc_sha256", "rhs_array_sha256", "operator_nonzeros")},
              "relative_target": lawrence.RELATIVE_TARGET, "relative_residual_limit": lawrence.RESIDUAL_LIMIT,
              "backward_residual_limit": lawrence.RESIDUAL_LIMIT, "channels": []}
    record.update(status="checking_original_equations", stage=link, linear_solver=report, candidate_history=[])
    if args.prior_solve_dir is not None:
        prior = read_receipt(args.prior_solve_dir / "solve_AP_pnc.json", args.prior_solve_sha256)
        check_execution(prior[0], record)
        record["prior_solve"] = prior[1]
        for column, row, vector in replay_channels(prior, link, operator, rhs, norm, complete=False):
            accepted = lawrence.write_vector(args.output / f"accepted-{column:03d}_AP_pnc.npy", vector)
            report["channels"].append({**row, "accepted_vector": accepted, "reused": True})
            workflow.write_json(receipt, record)
    first = len(report["channels"])
    if first == len(ROOTS):
        record["status"] = "drives_accepted"
        return
    controls = lawrence.cgroup_controls()

    def check_memory_at_boundary():
        quota = report["factor_admission"]["slurm_node_quota_bytes"]
        snapshot = lawrence.memory_snapshot(quota, controls)
        report["last_memory_boundary"] = snapshot
        if reason := lawrence.memory_block(snapshot):
            raise RuntimeError(reason)

    def admit():
        admission = direct_factor_admission(operator)
        report["factor_admission"] = admission
        record["status"] = "factorizing" if admission["admitted"] else "factor_refused"
        workflow.write_json(receipt, record)
        if not admission["admitted"]:
            raise RuntimeError("Measured h60 CSC does not fit the available factorization headroom.")

    def publish(event, data, vector=None):
        if event == "factor":
            report["factor"] = {"permc_spec": "COLAMD", **data}
            record["status"] = "solving"
        else:
            column = first + data["column"]
            data = {**data, "column": column, "incident_port_root": ROOTS[column]}
            if event == "candidate":
                label = f"accepted-{column:03d}_AP_pnc.npy" if data["target_met"] else f"candidate-{column:03d}-{data['update']:03d}_AP_pnc.npy"
                checkpoint = lawrence.write_vector(args.output / label, vector)
                data["vector"] = checkpoint
                if data["target_met"]:
                    replay = lawrence.replay_candidate(args.output / label, checkpoint, operator, rhs[:, column], norm)
                    report["channels"].append({**data, **replay, "info": 0, "accepted_vector": checkpoint})
            record["candidate_history"].append({"event": event, **data})
        try:
            if event in ("factor", "candidate"):
                check_memory_at_boundary()
        finally:
            workflow.write_json(receipt, record)

    count = lawrence.solve_drives(operator, rhs[:, first:], np.zeros_like(rhs[:, first]), norm, publish, admit)
    if first + count != len(ROOTS) or len(report["channels"]) != len(ROOTS):
        raise RuntimeError("A drive failed the original-equation checks within the bounded corrections.")
    record["status"] = "drives_accepted"


def export(args, record):
    staged, stage_link = read_stage(args, record)
    solved = read_receipt(args.solve_dir / "solve_AP_pnc.json", args.solve_sha256)
    check_execution(solved[0], record)
    mesh_dir = Path(staged["mesh_directory"])
    mesh, mesh_record = read_mesh(mesh_dir, staged["mesh_summary"], staged["mesh_hash"])
    layout = preflight.displacement_layout(mesh)
    if (layout.N != staged["displacement_dofs"] or workflow.digest(staged["port_dof_ids"]) != staged["trace_sha256"]
            or mesh_record["mesh_sha256"] != staged["mesh"]["sha256"]):
        raise ValueError("The export layout differs from the staged h60 mesh and ports.")
    K, M = (load_npz(check_array_file(args.stage_dir, name, staged)) for name in ("K", "M"))
    system = ElasticSystem(mesh, layout, K, M)
    modes = make_ports(system, staged["port_dof_ids"], staged["h80_inputs"]["ports"])
    solver_report = {**solved[0]["linear_solver"], "channels": []}

    def restore_solution(operator, rhs):
        check_equations(operator, rhs, staged)
        solution = np.empty_like(rhs)
        norm = lawrence.infinity_norm(operator)
        for column, row, vector in replay_channels(solved, stage_link, operator, rhs, norm, complete=True):
            solver_report["channels"].append(row)
            solution[:, column] = vector
        return solution

    execution_keys = ("implementation", "driver", "hostname", "slurm_job_id", "numerical_thread_environment",
                      "numerical_backend")
    assembly_execution, solve_execution = ({key: phase[key] for key in execution_keys} for phase in (staged, solved[0]))
    provenance = {"method": METHOD, "linux_assembly_and_solve": {**solve_execution, "assembly": assembly_execution},
                  "linux_export": {key: record[key] for key in execution_keys}, "assembly_stage": stage_link,
                  "solve_receipt": solved[1], "preflight_receipt": staged["preflight_receipt"],
                  "direct_solver_helpers": record["direct_solver_helpers"], "h80_inputs": staged["h80_inputs"],
                  "reused_preflight_mesh": True, "safe_eigensolve_performed": False, "phase_adjustment_applied": False}
    prepared = {"mesh_directory": mesh_dir, "mesh_checkpoint": (mesh, mesh_record), "system": system,
                "pattern": mesh_record["hole_pattern_provenance"], "assembly_resources": staged["assembly_resources"],
                "port_modes": modes, "solver_report": solver_report, "linear_solver": restore_solution,
                "attempt": workflow.digest(provenance), "provenance": provenance}
    result = workflow.run_study(staged["configuration"], args.output, _prepared_scattering=prepared)
    if (result["status"] != "completed_unverified" or result["power_balance_passed"] is not True
            or result["hole_count"] != 26 or result["incident_mode_indices"] != ROOTS):
        raise ValueError(f"The recovered h60 patterned scattering failed validation: {result['output_directory']}")
    record.update(status="completed_unverified", stage=stage_link, solve=solved[1],
                  public_summary=artifact(Path(result["output_directory"]) / "summary_AP_pnc.json"))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    phases = parser.add_subparsers(dest="mode", required=True)
    for mode in ("stage", "solve", "export"):
        phase = phases.add_parser(mode)
        phase.add_argument("--output", type=Path, required=True)
        if mode == "stage":
            phase.add_argument("--preflight-dir", type=Path, required=True)
            phase.add_argument("--preflight-sha256", required=True)
        else:
            phase.add_argument("--stage-dir", type=Path, required=True)
            phase.add_argument("--stage-sha256", required=True)
        if mode == "solve":
            phase.add_argument("--prior-solve-dir", type=Path)
            phase.add_argument("--prior-solve-sha256")
        if mode == "export":
            phase.add_argument("--solve-dir", type=Path, required=True)
            phase.add_argument("--solve-sha256", required=True)
    args = parser.parse_args()
    if args.mode == "solve" and (args.prior_solve_dir is None) != (args.prior_solve_sha256 is None):
        parser.error("Resume requires both the prior solve directory and its receipt SHA-256.")
    identity = workflow.source_identity()
    record = {**preflight.AUTHOR, "mode": args.mode, "status": "running", "implementation": identity,
              "driver": artifact(Path(__file__)), "direct_solver_helpers": artifact(Path(lawrence.__file__)),
              "factor_admission_helper": artifact(Path(uniform.__file__)),
              "hostname": socket.gethostname(), "slurm_job_id": os.environ.get("SLURM_JOB_ID"),
              "numerical_thread_environment": {name: os.environ.get(name) for name in lawrence.THREADS}}
    args.output.mkdir(parents=True, exist_ok=True)
    receipt = args.output / f"{args.mode}_AP_pnc.json"
    with receipt.open("x", encoding="utf-8") as stream:
        json.dump(record, stream, indent=2)
    try:
        if (sys.platform != "linux" or not record["slurm_job_id"]
                or record["numerical_thread_environment"] != lawrence.THREADS
                or record["direct_solver_helpers"]["sha256"] != SOLVER_HELPER_SHA256
                or record["factor_admission_helper"]["sha256"] != FACTOR_ADMISSION_HELPER_SHA256
                or workflow.digest(identity["source_sha256"]) != preflight.SOURCE_MAP_SHA256
                or identity["packages"] != preflight.PACKAGES):
            raise ValueError("Run the h60 phases in the reviewed single-threaded Linux Slurm environment.")
        record["numerical_backend"] = lawrence.solve_environment()
        {"stage": stage, "solve": solve, "export": export}[args.mode](args, record)
    except BaseException as error:
        record.update(status="interrupted" if isinstance(error, KeyboardInterrupt) else "failed",
                      error=f"{type(error).__name__}: {error}")
        raise
    finally:
        workflow.write_json(receipt, record)
    print(f"{record['status']}: {receipt.resolve()}", flush=True)


if __name__ == "__main__":
    main()
