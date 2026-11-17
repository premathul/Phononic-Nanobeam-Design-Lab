"""Stage, solve and export the h80 z6 thickness control. Athul Prem | _AP_pnc."""

from __future__ import annotations

import argparse
from dataclasses import fields
import json
from pathlib import Path
from shutil import disk_usage
import sys
from types import SimpleNamespace

import numpy as np
from scipy.sparse import load_npz, save_npz

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT / "src"), str(ROOT)]

from phononic import workflow_AP_pnc as workflow
from phononic.elasticity_AP_pnc import ElasticSystem
from phononic.port_modes_AP_pnc import PortModes
from phononic.scattering_AP_pnc import solve_scattering
from tools import lawrence_solver_AP_pnc as lawrence
from tools import run_lawrence_thickness_AP_pnc as prep
from tools.run_lawrence_uniform_h80_AP_pnc import artifact, direct_factor_admission
from tools.solve_lawrence_device_refinement_AP_pnc import OperatorCaptured, equation_hashes, matrix_bytes


METHOD = "lawrence_device_thickness_superlu_direct"
PREP_DRIVER_SHA = "2149939222a8702fdc6c1a61441908dff5dede27859154e033f9142aaa403f18"
STAGED_HELPER_SHA = "32efc149510d22744a0478ff9c171e770ba4cf9bf535b2ca5f2ceeb15c529730"
MATRIX_FILES = {"K": "K_AP_pnc.npz", "M": "M_AP_pnc.npz", "A": "A_AP_pnc.npz", "rhs": "rhs_AP_pnc.npy"}
EXECUTION_KEYS = ("implementation", "driver", "hostname", "slurm_job_id", "numerical_thread_environment", "numerical_backend")


def read_link(link):
    path = Path(link["path"])
    if artifact(path) != link:
        raise ValueError(f"The saved file differs: {path}")
    return path


def read_phase(directory, phase, sha, status, current):
    return prep.read_receipt(directory, f"{phase}_AP_pnc.json", sha, status, current["driver"]["sha256"], current)


def make_modes(system, admitted_ports, trace_digest, roots, freq):
    with np.load(read_link(admitted_ports), allow_pickle=False) as saved:
        left, right = (saved[key] for key in ("left_dofs", "right_dofs"))
        if (left.shape != (prep.PORT_DOFS,) or right.shape != left.shape
                or workflow.digest([left.tolist(), right.tolist()]) != trace_digest
                or saved["frequency_hz"].item() != freq):
            raise ValueError("The saved SAFE trace order or frequency differs from the admitted case.")
        slab_components = prep.fixed_port.components(system.basis)
        section_components = np.empty(len(left), dtype=np.int8)
        for component, indices in enumerate(saved["component_dofs_xyz"]):
            section_components[indices] = component
        for saved_ids, trace_ids in zip((left, right), prep.fixed_port.port_trace_ids(system.mesh, system.basis), strict=True):
            if (not np.array_equal(np.sort(saved_ids), np.sort(trace_ids))
                    or not np.array_equal(slab_components[saved_ids], section_components)
                    or not np.allclose(system.basis.doflocs[1:, saved_ids].T, saved["coordinates_yz_m"], rtol=0, atol=1e-15)):
                raise ValueError("The saved SAFE coordinates do not map to the complete tagged trace.")
        port = SimpleNamespace(elastic_system=system, basis=SimpleNamespace(N=len(left)), left_dofs=left, right_dofs=right)
        modes = PortModes(port=port, frequency_hz=freq, **{item.name: saved[item.name] for item in fields(PortModes)
                          if item.name not in ("port", "frequency_hz")})
    incident = modes.positive_indices[modes.propagating[modes.positive_indices]]
    if incident.tolist() != roots:
        raise ValueError("The saved modes do not reproduce the admitted incident order.")
    return modes


def stage(args, record):
    ready, preflight_link = prep.read_receipt(args.preflight_dir, prep.PREFLIGHT, args.preflight_sha256,
                                            "mesh_ready", PREP_DRIVER_SHA, record)
    safe, safe_link = prep.read_receipt(args.safe_dir, prep.SAFE, args.safe_sha256, "safe_ready", PREP_DRIVER_SHA, record)
    if (safe["preflight_receipt"] != preflight_link or safe["mesh"] != ready["mesh"]
            or safe["configuration"] != ready["configuration"] or safe["safe_eigensolve_performed"] is not True):
        raise ValueError("The SAFE receipt differs from the admitted h80 z6 preflight.")
    mesh_dir, mesh, mesh_record = prep.read_preflight_mesh(args.preflight_dir, ready)
    config = workflow.resolve_config(ready["configuration"])
    resources = config["resources"]
    resources.update(max_dofs=ready["displacement_dofs"], linear_solver="direct", assembly_batch_elements=2048)
    roots = safe["incident_port_roots"]
    record.update(status="assembling", configuration=config, mesh_directory=str(mesh_dir.resolve()),
                  mesh_hash=ready["mesh_hash"], mesh=ready["mesh"], mesh_summary=ready["mesh_summary"],
                  preflight_receipt=preflight_link, safe_receipt=safe_link, safe_modes=safe["ports"],
                  linux_safe={key: safe[key] for key in EXECUTION_KEYS}, trace_sha256=safe["trace_sha256"],
                  incident_port_roots=roots, frequency_hz=safe["frequency_hz"], displacement_dofs=mesh_record["ndof"],
                  factorization_performed=False)
    workflow.write_json(args.output / "stage_AP_pnc.json", record)
    with workflow.measured_resources() as measured:
        system = workflow._assemble_system(mesh, config, prep.material_from(config))
    modes = make_modes(system, safe["ports"], safe["trace_sha256"], roots, safe["frequency_hz"])

    def capture(operator, rhs):
        validate_equations(operator, rhs, record)
        if not np.isfinite(operator.data).all() or not np.isfinite(rhs).all():
            raise ValueError("The assembled equations contain nonfinite values.")
        storage = sum(matrix_bytes(matrix) for matrix in (system.K, system.M, operator)) + rhs.nbytes
        if disk_usage(args.output).free < storage + lawrence.GIB:
            raise RuntimeError("The stage equations and 1 GiB disk reserve do not fit.")
        arrays = {}
        for name, matrix in (("K", system.K), ("M", system.M), ("A", operator)):
            path = args.output / MATRIX_FILES[name]
            save_npz(path, matrix, compressed=False)
            arrays[name] = artifact(path)
        path = args.output / MATRIX_FILES["rhs"]
        np.save(path, rhs, allow_pickle=False)
        arrays["rhs"] = artifact(path)
        record.update(status="operator_staged", array_files=arrays, **equation_hashes(operator, rhs),
                      operator_nonzeros=int(operator.nnz), operator_csc_bytes=matrix_bytes(operator),
                      stage_array_bytes=storage, assembly_resources=measured)
        raise OperatorCaptured

    try:
        solve_scattering(system, modes, modes.frequency_hz, config["analysis"]["incident_side"], linear_solver=capture)
    except OperatorCaptured:
        return
    raise RuntimeError("The stage did not capture the original harmonic equations.")


def validate_equations(operator, rhs, staged, *, replay=False):
    ndof = staged["displacement_dofs"]
    if (operator.format != "csc" or operator.dtype != np.complex128 or not operator.has_canonical_format
            or operator.shape != (ndof, ndof) or rhs.dtype != np.complex128
            or rhs.shape != (ndof, len(staged["incident_port_roots"]))):
        raise ValueError("The h80 z6 original-equation layout differs.")
    if replay and (operator.nnz != staged["operator_nonzeros"]
                   or any(staged[key] != value for key, value in equation_hashes(operator, rhs).items())):
        raise ValueError("The reconstructed operator or drives differ from the admitted exact equations.")


def load_stage(args, record):
    staged, link = read_phase(args.stage_dir, "stage", args.stage_sha256, "operator_staged", record)
    if staged["factorization_performed"] is not False:
        raise ValueError("The saved stage must contain only the admitted prepared equations.")
    return staged, link


def replay_channels(receipt, receipt_path, roots, stage_link, operator, rhs, norm, *, complete):
    report = receipt["linear_solver"]
    channels = report["channels"]
    if (receipt["stage"] != stage_link or report["method"] != METHOD or len(channels) > len(roots)
            or complete and (receipt["status"] != "drives_accepted" or len(channels) != len(roots))):
        raise ValueError("The saved solve does not cover the required staged drives.")
    for column, row in enumerate(channels):
        saved = row["accepted_vector"]
        expected_name = f"accepted-{column:03d}_AP_pnc.npy"
        path = receipt_path.parent / expected_name
        if (row["column"] != column or row["incident_port_root"] != roots[column] or row["info"] != 0
                or not lawrence.accepted(row) or saved["file"] != expected_name or path.stat().st_size != saved["bytes"]):
            raise ValueError("A saved accepted channel has changed identity or order.")
        metrics = lawrence.replay_candidate(path, saved, operator, rhs[:, column], norm)
        yield {**row, **metrics}, np.load(path, allow_pickle=False)


def solve(args, record):
    staged, stage_link = load_stage(args, record)
    roots = staged["incident_port_roots"]
    operator = load_npz(read_link(staged["array_files"]["A"]))
    rhs = np.load(read_link(staged["array_files"]["rhs"]), allow_pickle=False)
    validate_equations(operator, rhs, staged, replay=True)
    if disk_usage(args.output).free < 4 * rhs.nbytes + lawrence.GIB:
        raise RuntimeError("The saved drive vectors and 1 GiB disk reserve do not fit.")
    norm = lawrence.infinity_norm(operator)
    receipt = args.output / "solve_AP_pnc.json"
    report = {"method": METHOD, "working_dtype": str(operator.dtype),
              **{key: staged[key] for key in ("operator_csc_sha256", "rhs_array_sha256", "operator_nonzeros")},
              "relative_target": lawrence.RELATIVE_TARGET, "relative_residual_limit": lawrence.RESIDUAL_LIMIT,
              "backward_residual_limit": lawrence.RESIDUAL_LIMIT, "channels": []}
    record.update(status="checking_original_equations", stage=stage_link, linear_solver=report, candidate_history=[])
    if args.prior_solve_dir is not None:
        path = args.prior_solve_dir / "solve_AP_pnc.json"
        status = json.loads(path.read_bytes())["status"]
        prior, link = read_phase(args.prior_solve_dir, "solve", args.prior_solve_sha256, status, record)
        record["prior_solve"] = link
        for row, vector in replay_channels(prior, path, roots, stage_link, operator, rhs, norm, complete=False):
            accepted = lawrence.write_vector(args.output / row["accepted_vector"]["file"], vector)
            report["channels"].append({**row, "accepted_vector": accepted, "reused": True})
            workflow.write_json(receipt, record)
    first = len(report["channels"])
    if first == len(roots):
        record["status"] = "drives_accepted"
        return
    controls = lawrence.cgroup_controls()

    def admit():
        admission = direct_factor_admission(operator)
        report["factor_admission"] = admission
        record["status"] = "factorizing" if admission["admitted"] else "factor_refused"
        workflow.write_json(receipt, record)
        if not admission["admitted"]:
            raise RuntimeError("The measured equations do not fit the current factor admission budget.")

    def publish(event, data, vector=None):
        if event == "factor":
            report["factor"] = {"permc_spec": "COLAMD", **data}
            record["status"] = "solving"
        else:
            column = first + data["column"]
            data = {**data, "column": column, "incident_port_root": roots[column]}
            if event == "candidate":
                label = (f"accepted-{column:03d}_AP_pnc.npy" if data["target_met"] else
                         f"candidate-{column:03d}-{data['update']:03d}_AP_pnc.npy")
                saved = lawrence.write_vector(args.output / label, vector)
                data["vector"] = saved
                if data["target_met"]:
                    metrics = lawrence.replay_candidate(args.output / label, saved, operator, rhs[:, column], norm)
                    report["channels"].append({**data, **metrics, "info": 0, "accepted_vector": saved})
            record["candidate_history"].append({"event": event, **data})
        try:
            if event in ("factor", "candidate"):
                snapshot = lawrence.memory_snapshot(report["factor_admission"]["slurm_node_quota_bytes"], controls)
                report["last_memory_boundary"] = snapshot
                if reason := lawrence.memory_block(snapshot):
                    raise RuntimeError(reason)
        finally:
            workflow.write_json(receipt, record)

    solved = lawrence.solve_drives(operator, rhs[:, first:], np.zeros_like(rhs[:, first]), norm, publish, admit)
    if first + solved != len(roots) or len(report["channels"]) != len(roots):
        raise RuntimeError("A drive failed the original-equation checks within the bounded corrections.")
    record["status"] = "drives_accepted"


def export(args, record):
    staged, stage_link = load_stage(args, record)
    solved, solve_link = read_phase(args.solve_dir, "solve", args.solve_sha256, "drives_accepted", record)
    mesh_dir = Path(staged["mesh_directory"])
    read_link(staged["mesh_summary"])
    read_link(staged["mesh"])
    mesh, mesh_record = workflow.load_checkpoint(mesh_dir, staged["mesh_hash"])
    layout = prep.fixed_port.displacement_layout(mesh)
    matrices = [load_npz(read_link(staged["array_files"][name])) for name in ("K", "M")]
    system = ElasticSystem(mesh, layout, *matrices)
    roots = staged["incident_port_roots"]
    modes = make_modes(system, staged["safe_modes"], staged["trace_sha256"], roots, staged["frequency_hz"])
    report = {**solved["linear_solver"], "channels": []}

    def restore(operator, rhs):
        validate_equations(operator, rhs, staged, replay=True)
        vectors, norm = np.empty_like(rhs), lawrence.infinity_norm(operator)
        for row, vector in replay_channels(solved, Path(solve_link["path"]), roots, stage_link, operator, rhs, norm, complete=True):
            report["channels"].append(row)
            vectors[:, row["column"]] = vector
        return vectors

    assembly, solving, exporting = ({key: phase[key] for key in EXECUTION_KEYS} for phase in (staged, solved, record))
    provenance = {"method": METHOD, "linux_assembly_and_solve": {**solving, "assembly": assembly},
                  "linux_export": exporting, "assembly_stage": stage_link, "solve_receipt": solve_link,
                  "direct_solver_helpers": record["helpers"]["solver"], "reused_preflight_mesh": True,
                  "safe_eigensolve_performed": True, "phase_adjustment_applied": False,
                  **{name: staged[name] for name in ("preflight_receipt", "safe_receipt", "safe_modes", "linux_safe")}}
    prepared = {"mesh_directory": mesh_dir, "mesh_checkpoint": (mesh, mesh_record), "system": system,
                "pattern": mesh_record["hole_pattern_provenance"], "assembly_resources": staged["assembly_resources"],
                "port_modes": modes, "solver_report": report, "linear_solver": restore,
                "attempt": workflow.digest(provenance), "provenance": provenance}
    result = workflow.run_study(staged["configuration"], args.output, _prepared_scattering=prepared)
    if (result["status"] != "completed_unverified" or result["power_balance_passed"] is not True
            or result["hole_count"] != 26 or result["incident_mode_indices"] != roots):
        raise ValueError(f"The h80 z6 scattering failed validation: {result['output_directory']}")
    record.update(status="completed_unverified", stage=stage_link, solve=solve_link,
                  public_summary=artifact(Path(result["output_directory"]) / "summary_AP_pnc.json"))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    modes = parser.add_subparsers(dest="mode", required=True)
    inputs = {"stage": ("preflight", "safe"), "solve": ("stage",), "export": ("stage", "solve")}
    for mode, prefixes in inputs.items():
        phase = modes.add_parser(mode)
        phase.add_argument("--output", type=Path, required=True)
        for prefix in prefixes:
            phase.add_argument(f"--{prefix}-dir", type=Path, required=True)
            phase.add_argument(f"--{prefix}-sha256", required=True)
        if mode == "solve":
            phase.add_argument("--prior-solve-dir", type=Path)
            phase.add_argument("--prior-solve-sha256")
    args = parser.parse_args()
    if args.mode == "solve" and (args.prior_solve_dir is None) != (args.prior_solve_sha256 is None):
        parser.error("Resume requires the prior solve directory and receipt checksum.")
    if (artifact(Path(prep.__file__))["sha256"] != PREP_DRIVER_SHA
            or artifact(ROOT / "tools/solve_lawrence_device_refinement_AP_pnc.py")["sha256"] != STAGED_HELPER_SHA):
        raise ValueError("The reviewed preparation or staged-equation helper source differs.")
    prep.run_phase(args, Path(__file__), f"{args.mode}_AP_pnc.json", {"stage": stage, "solve": solve, "export": export}[args.mode])


if __name__ == "__main__":
    main()
