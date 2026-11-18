"""Reuse the archived h80 z4 body at 887.5 MHz. Athul Prem | _AP_pnc."""

from __future__ import annotations

import argparse
from dataclasses import replace
import json
import os
from pathlib import Path
from shutil import disk_usage
import socket
import sys

import numpy as np
from scipy.sparse import load_npz, save_npz
from scipy.sparse.linalg._dsolve import _superlu

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT / "src"), str(ROOT)]

from phononic import workflow_AP_pnc as workflow
from phononic.elasticity_AP_pnc import ElasticSystem
from phononic.port_modes_AP_pnc import solve_port_modes
from phononic.resumable_scattering_AP_pnc import _array_digest
from phononic.scattering_AP_pnc import solve_scattering
from tools import run_lawrence_low_band_z4_AP_pnc as archived_z4
from tools import run_lawrence_low_band_AP_pnc as low_band
from tools import run_lawrence_port_inventory_AP_pnc as inventory
from tools import solve_lawrence_device_refinement_AP_pnc as equations
from tools import solve_lawrence_thickness_AP_pnc as thickness
from tools.run_lawrence_uniform_h80_AP_pnc import artifact, direct_factor_admission


FREQUENCY_HZ = 887_500_000
BODY_DOFS, PORT_DOFS, CHANNELS = 426627, 675, 5
Z4_RUNNER_SHA = "59539b922aa2e37a059bb0f159442b0bac5347b7996bb097a343cf4a4a2216a8"
SECTION_RECEIPT_SHA, SECTION_SHA, TRACE_SHA = (archived_z4.SECTION_RECEIPT_SHA,
                                            archived_z4.SECTION_SHA, archived_z4.TRACE_SHA)
lawrence = low_band.lawrence
fixed_port = low_band.prep.fixed_port
EXPLORATORY = {**low_band.EXPLORATORY, "joint_body_port_thickness_sensitivity": True}
RESOURCE_CEILINGS = {"safe": {"memory_gib": 4, "allocation_seconds": 180, "worker_seconds": 150},
                     "stage": {"memory_gib": 16, "allocation_seconds": 600, "worker_seconds": 550},
                     "solve": {"memory_gib": 64, "allocation_seconds": 1500, "worker_seconds": 1400},
                     "export": {"memory_gib": 16, "allocation_seconds": 600, "worker_seconds": 550}}


def checked_sources():
    sources = archived_z4.checked_sources()
    link = artifact(Path(archived_z4.__file__))
    if link["sha256"] != Z4_RUNNER_SHA:
        raise ValueError("The reviewed four-layer archive helper changed.")
    return {**sources, archived_z4.__name__: link}


def run_phase(args, operation):
    implementation = workflow.source_identity()
    threads = {name: os.environ.get(name) for name in lawrence.THREADS}
    job = os.environ.get("SLURM_JOB_ID")
    if (sys.platform != "linux" or not job or threads != lawrence.THREADS
            or workflow.digest(implementation["source_sha256"]) != fixed_port.SOURCE_MAP_SHA256
            or implementation["packages"] != fixed_port.PACKAGES
            or int(os.environ.get("SLURM_CPUS_PER_TASK", "0")) != 2
            or not 0 < int(os.environ.get("SLURM_MEM_PER_NODE", "0")) <= RESOURCE_CEILINGS[args.mode]["memory_gib"] * 1024):
        raise ValueError("Use the reviewed single-threaded Linux Slurm environment.")
    record = {**fixed_port.AUTHOR, **EXPLORATORY, "mode": args.mode, "status": "running", "implementation": implementation,
              "driver": artifact(Path(__file__)), "hostname": socket.gethostname(), "slurm_job_id": job,
              "numerical_thread_environment": threads, "numerical_backend": lawrence.solve_environment(),
              "reviewed_sources": checked_sources(), "superlu_library": artifact(Path(_superlu.__file__)),
              "resource_ceiling": RESOURCE_CEILINGS[args.mode], "automatic_retry": False,
              "automatic_allocation_escalation": False}
    args.output.mkdir(parents=True, exist_ok=True)
    receipt = args.output / f"{args.mode}_AP_pnc.json"
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


# Pure archive readers retain their accepted historical source and identity checks.
archive_link = archived_z4.archive_link
restore_mesh = archived_z4.restore_mesh
read_section = archived_z4.read_section
verify_trace = archived_z4.verify_trace
read_phase = archived_z4.read_phase
factor_index_exposure = archived_z4.factor_index_exposure


def read_body(directory, mesh_dir, current):
    body = archived_z4.read_body(directory, mesh_dir, current)
    body["configuration"]["analysis"]["frequency_GHz"] = FREQUENCY_HZ / 1e9
    return body


def validate_point(point):
    roots, returning = point["incident_port_roots"], point["backward_port_roots"]
    if (point["frequency_hz"] != FREQUENCY_HZ or point["displacement_dofs"] != BODY_DOFS
            or point["discrete_roots"] != 2 * PORT_DOFS
            or len(roots) != CHANNELS or len(returning) != CHANNELS
            or any(type(root) is not int or not 0 <= root < 2 * PORT_DOFS for root in roots + returning)
            or len(set(roots + returning)) != 2 * CHANNELS
            or point["positive_open_channels"] != CHANNELS or point["negative_open_channels"] != CHANNELS
            or point["cutoff_status"] != "solver_resolved_at_sample"
            or point["degeneracy_status"] != "none_detected" or point["coincident_propagating_root_groups"]
            or point["trace_sha256"] != TRACE_SHA
            or tuple(point["reference_planes_m"]) != low_band.END_PLANES_M):
        raise ValueError("The 887.5 MHz z4 point requires exactly five resolved signed channels and the archived body.")
    rows = point["open_channels"]
    if (len(rows) != 2 * CHANNELS
            or any([row["root_id"] for row in rows if row["direction"] == sign] != expected
                   for sign, expected in (("positive", roots), ("negative", returning)))):
        raise ValueError("The SAFE signed roots differ from the complete open-channel inventory.")
    return roots, returning


def same_artifact(first, second):
    # Original Windows replay paths and Linux execution paths remain distinct provenance.
    return all(first.get(key) == second.get(key) for key in ("bytes", "sha256"))


def prebody_checks(ready, safe_link, verification_path, verification_sha, relation_path, relation_sha):
    """Bind externally reviewed SAFE/section matching before preparing any body equations."""
    roots, returning = validate_point(ready)
    verification, checked = equations.read_receipt(verification_path, verification_sha)
    relation, matched = equations.read_receipt(relation_path, relation_sha)
    if (verification["status"] != "independently_verified_safe"
            or verification["frequency_hz"] != FREQUENCY_HZ or verification["problem_dofs"] != BODY_DOFS
            or verification["port_dofs"] != PORT_DOFS
            or verification["all_discrete_roots_replayed"] != 2 * PORT_DOFS
            or verification["positive_open_channels"] != CHANNELS or verification["negative_open_channels"] != CHANNELS
            or verification["incident_port_roots"] != roots or verification["backward_port_roots"] != returning
            or not same_artifact(verification["safe_receipt"], safe_link)
            or not same_artifact(verification["safe_modes"], ready["safe_modes"])
            or verification["relation_status"] != "matched"
            or verification["assigned_root_square_consistent"] is not True
            or not same_artifact(verification["relation"], matched)):
        raise ValueError("The independent full SAFE and signed-relation verification differs from the selected point.")
    if (relation["status"] != "signed_section_relation_accepted" or relation["frequency_hz"] != FREQUENCY_HZ
            or relation["positive_open_channels"] != CHANNELS or relation["negative_open_channels"] != CHANNELS
            or relation["overlap_gate"] != 0.99
            or not same_artifact(relation["candidate_safe_receipt"], safe_link)
            or not same_artifact(relation["candidate_safe_modes"], ready["safe_modes"])
            or relation["root_permutation_square"]["status"] != "consistent"):
        raise ValueError("The accepted signed section relation differs from the selected SAFE sample.")
    return {"safe_verification": checked, "signed_relation": matched}


def safe(args, record):
    record["workspace_admission"] = low_band.admit_workspace(args.output, 128 * lawrence.MIB, memory_bytes=1536 * lawrence.MIB)
    body = read_body(args.body_stage_dir, args.mesh_dir, record)
    receipt, receipt_link = equations.read_receipt(args.section_dir / "checkpoint_AP_pnc.json", SECTION_RECEIPT_SHA)
    section_link = archive_link(args.section_dir / "arrays_AP_pnc.npz", {"sha256": SECTION_SHA, "bytes": 12973330})
    if (receipt["status"] != "solved" or receipt["case"] != "12x4"
            or receipt["arrays"]["sha256"] != SECTION_SHA):
        raise ValueError("The archived section receipt differs from the accepted 12x4 control.")
    port, anchor = read_section(section_link, body["archived_safe_modes"])
    mesh, layout, _ = restore_mesh(body)
    verify_trace(port, mesh, layout)
    anchor_check = inventory.validate_modes(anchor)
    if (anchor.positive_indices[anchor.propagating[anchor.positive_indices]].tolist() != list(lawrence.INCIDENT_ROOTS)
            or anchor_check["positive_open_channels"] != 5):
        raise ValueError("The saved 1 GHz modes do not reproduce the accepted anchor order.")
    record.update(**body, status="section_verified", safe_section=section_link, section_receipt=receipt_link,
                  archived_1ghz_replay=anchor_check, trace_sha256=TRACE_SHA,
                  safe_section_array_sha256=_array_digest(port.basis.mesh.p, port.basis.mesh.t, port.K0, port.M, port.Q, port.G),
                  body_assembly_performed=False, section_assembly_performed=False, mesh_generated=False,
                  anchor_eigensolve_performed=False, factorization_performed=False, frequency_hz=FREQUENCY_HZ)
    workflow.write_json(args.output / "safe_AP_pnc.json", record)
    with workflow.measured_resources() as measured:
        modes = solve_port_modes(port, FREQUENCY_HZ)
    path = args.output / f"modes-{FREQUENCY_HZ}_AP_pnc.npz"
    with path.open("xb") as stream:
        np.savez_compressed(stream, **{name: getattr(modes, name) for name in inventory.MODE_FIELDS})
    # Preserve the unique raw eigensolve even when its subsequent acceptance fails.
    record.update(status="checking_safe_modes", safe_modes=artifact(path), safe_eigensolve_performed=True,
                  resources=measured)
    workflow.write_json(args.output / "safe_AP_pnc.json", record)
    accepted = inventory.validate_modes(modes)
    roots = [ids[modes.propagating[ids]].tolist() for ids in (modes.positive_indices, modes.negative_indices)]
    record.update(incident_port_roots=roots[0], backward_port_roots=roots[1], **accepted)
    validate_point(record)
    record["status"] = "safe_ready"


def restore_system(staged):
    mesh, layout, mesh_record = restore_mesh(staged)
    port, _ = read_section(staged["safe_section"], staged["archived_safe_modes"])
    verify_trace(port, mesh, layout)
    matrices = [load_npz(thickness.read_link(staged["array_files"][key])) for key in ("K", "M")]
    if any(matrix.shape != (layout.N, layout.N) or matrix.dtype != np.float64 for matrix in matrices):
        raise ValueError("The archived body matrix does not match the real four-layer displacement layout.")
    system = ElasticSystem(mesh, layout, *matrices)
    with np.load(thickness.read_link(staged["safe_modes"]), allow_pickle=False) as arrays:
        modes = inventory.read_modes(arrays, replace(port, elastic_system=system))
    roots = [ids[modes.propagating[ids]].tolist() for ids in (modes.positive_indices, modes.negative_indices)]
    if (modes.frequency_hz != FREQUENCY_HZ or roots != [staged["incident_port_roots"], staged["backward_port_roots"]]
            or [len(side) for side in roots] != [CHANNELS, CHANNELS]):
        raise ValueError("The prepared modes differ from the accepted 887.5 MHz section.")
    return system, modes, mesh_record


def stage(args, record):
    ready, safe_link = read_phase(args.safe_dir, "safe", args.safe_sha256, "safe_ready", record)
    admitted = prebody_checks(ready, safe_link, args.safe_verification, args.safe_verification_sha256,
                              args.relation, args.relation_sha256)
    record.update(**admitted, **{key: ready[key] for key in ("configuration", "body_stage", "body_assembly_source", "mesh_directory",
                  "mesh_hash", "mesh", "mesh_summary", "array_files", "archived_safe_modes", "assembly_resources",
                  "displacement_dofs", "safe_section", "safe_modes", "section_receipt", "trace_sha256",
                  "safe_section_array_sha256", "frequency_hz", "incident_port_roots", "backward_port_roots",
                  "open_channels", "cutoff_status", "degeneracy_status", "reference_planes_m", "discrete_roots",
                  "positive_open_channels", "negative_open_channels", "coincident_propagating_root_groups")},
                  safe_receipt=safe_link, status="constructing_frequency_equations", factorization_performed=False,
                  mesh_generated=False, body_assembly_performed=False, section_assembly_performed=False,
                  safe_eigensolve_performed=False, phase_adjustment_applied=False, **EXPLORATORY)
    record["workspace_admission"] = low_band.admit_workspace(args.output, 2 * lawrence.GIB)
    archive_link(args.reference_operator, {"bytes": lawrence.ARTIFACTS["A_AP_pnc.npz"][0],
                                          "sha256": lawrence.ARTIFACTS["A_AP_pnc.npz"][1]})
    system, modes, _ = restore_system(record)

    def capture(operator, rhs):
        thickness.validate_equations(operator, rhs, record)
        if not np.isfinite(operator.data).all() or not np.isfinite(rhs).all():
            raise ValueError("The frequency-specific equations contain nonfinite entries.")
        exposure = factor_index_exposure(operator, args.reference_operator)
        if not exposure["matches_archived_1_ghz_csc_graph"]:
            raise ValueError("The prepared frequency operator graph differs from the archived z4 device.")
        storage = equations.matrix_bytes(operator) + rhs.nbytes
        if disk_usage(args.output).free < storage + lawrence.GIB:
            raise RuntimeError("The prepared equations and 1 GiB disk reserve do not fit.")
        operator_path = args.output / thickness.MATRIX_FILES["A"]
        rhs_path = args.output / thickness.MATRIX_FILES["rhs"]
        with operator_path.open("xb") as stream:
            save_npz(stream, operator, compressed=False)
        with rhs_path.open("xb") as stream:
            np.save(stream, rhs, allow_pickle=False)
        record["array_files"].update(A=artifact(operator_path), rhs=artifact(rhs_path))
        record.update(status="operator_staged", **equations.equation_hashes(operator, rhs),
                      operator_nonzeros=int(operator.nnz), operator_csc_bytes=equations.matrix_bytes(operator),
                      stage_array_bytes=storage, factor_index_exposure=exposure)
        raise equations.OperatorCaptured

    try:
        with workflow.measured_resources() as measured:
            solve_scattering(system, modes, FREQUENCY_HZ, "left", linear_solver=capture)
    except equations.OperatorCaptured:
        record["equation_preparation_resources"] = measured
        return
    raise RuntimeError("The scattering core did not expose the original equations.")


def load_stage(args, record):
    staged, link = read_phase(args.stage_dir, "stage", args.stage_sha256, "operator_staged", record)
    validate_point(staged)
    verified = prebody_checks(staged, staged["safe_receipt"], Path(staged["safe_verification"]["path"]),
                              staged["safe_verification"]["sha256"], Path(staged["signed_relation"]["path"]),
                              staged["signed_relation"]["sha256"])
    if verified != {key: staged[key] for key in ("safe_verification", "signed_relation")}:
        raise ValueError("The prepared point's independent admission artifacts changed.")
    exposure = staged["factor_index_exposure"]
    if (staged["displacement_dofs"] != BODY_DOFS or staged["frequency_hz"] != FREQUENCY_HZ
            or staged["factorization_performed"] is not False
            or len(staged["incident_port_roots"]) != CHANNELS or len(staged["backward_port_roots"]) != CHANNELS
            or exposure["matches_archived_1_ghz_csc_graph"] is not True
            or staged["trace_sha256"] != TRACE_SHA or exposure["signed_index_maximum"] != low_band.INDEX_MAX
            or exposure["operator_nonzeros"] != staged["operator_nonzeros"]
            or exposure["initial_factor_slots_per_store"] != 30 * staged["operator_nonzeros"]
            or not 0 < exposure["initial_factor_slots_per_store"] <= low_band.INDEX_MAX):
        raise ValueError("The prepared four-layer point or reviewed index exposure differs.")
    return staged, link


def solve(args, record):
    staged, stage_link = load_stage(args, record)
    roots = staged["incident_port_roots"]
    record["workspace_admission"] = low_band.admit_workspace(args.output, 4 * staged["array_files"]["rhs"]["bytes"] + lawrence.GIB)
    operator = load_npz(thickness.read_link(staged["array_files"]["A"]))
    rhs = np.load(thickness.read_link(staged["array_files"]["rhs"]), allow_pickle=False)
    thickness.validate_equations(operator, rhs, staged, replay=True)
    exposure = staged["factor_index_exposure"]
    if (operator.indices.dtype != np.int32 or operator.indptr.dtype != np.int32
            or lawrence.array_sha(operator.indices, operator.indptr) != exposure["operator_graph_sha256"]):
        raise ValueError("The admitted CSC graph or index width changed.")
    norm = lawrence.infinity_norm(operator)
    receipt = args.output / "solve_AP_pnc.json"
    report = {"method": thickness.METHOD, "working_dtype": str(operator.dtype),
              **{key: staged[key] for key in ("operator_csc_sha256", "rhs_array_sha256", "operator_nonzeros")},
              "relative_target": lawrence.RELATIVE_TARGET, "relative_residual_limit": lawrence.RESIDUAL_LIMIT,
              "backward_residual_limit": lawrence.RESIDUAL_LIMIT, "channels": []}
    record.update(status="checking_original_equations", stage=stage_link, linear_solver=report,
                  factor_index_exposure=exposure, candidate_history=[])
    if args.prior_solve_dir is not None:
        path = args.prior_solve_dir / receipt.name
        status = json.loads(path.read_bytes())["status"]
        prior, link = read_phase(args.prior_solve_dir, "solve", args.prior_solve_sha256, status, record)
        record["prior_solve"] = link
        for row, vector in thickness.replay_channels(prior, path, roots, stage_link, operator, rhs, norm, complete=False):
            saved = lawrence.write_vector(args.output / row["accepted_vector"]["file"], vector)
            report["channels"].append({**row, "accepted_vector": saved, "reused": True})
            workflow.write_json(receipt, record)
    first = len(report["channels"])
    if first == len(roots):
        previous_report = prior["linear_solver"]
        factor = previous_report["factor"]
        low_band.validate_factor_count(factor["factor_nonzeros"], operator.shape[0])
        report.update(factor=factor, factor_reused_from=previous_report.get("factor_reused_from", link))
        record["status"] = "drives_accepted"
        return
    controls = lawrence.cgroup_controls()

    def admit():
        admission = direct_factor_admission(operator)
        report["factor_admission"] = admission
        record["status"] = "factorizing" if admission["admitted"] else "factor_refused"
        workflow.write_json(receipt, record)
        if not admission["admitted"]:
            raise RuntimeError("The measured equations exceed live node or job factor admission.")

    def publish(event, data, vector=None):
        record["candidate_history"].append({"event": event, **data})
        try:
            if event == "factor":
                report["factor"] = {"permc_spec": "COLAMD", **data}
                low_band.validate_factor_count(data["factor_nonzeros"], operator.shape[0])
                record["status"] = "solving"
            else:
                column = first + data["column"]
                data.update(column=column, incident_port_root=roots[column])
                record["candidate_history"][-1].update(data)
                if event == "candidate":
                    name = (f"accepted-{column:03d}_AP_pnc.npy" if data["target_met"] else
                            f"candidate-{column:03d}-{data['update']:03d}_AP_pnc.npy")
                    saved = lawrence.write_vector(args.output / name, vector)
                    record["candidate_history"][-1]["vector"] = saved
                    if data["target_met"]:
                        metrics = lawrence.replay_candidate(args.output / name, saved, operator, rhs[:, column], norm)
                        report["channels"].append({**data, **metrics, "info": 0, "accepted_vector": saved})
            if event in ("factor", "candidate"):
                snapshot = lawrence.memory_snapshot(report["factor_admission"]["slurm_node_quota_bytes"], controls)
                report["last_memory_boundary"] = snapshot
                if reason := lawrence.memory_block(snapshot):
                    raise RuntimeError(reason)
        finally:
            workflow.write_json(receipt, record)

    solved = lawrence.solve_drives(operator, rhs[:, first:], np.zeros_like(rhs[:, first]), norm, publish, admit)
    if first + solved != len(roots) or len(report["channels"]) != len(roots):
        raise RuntimeError("A drive failed the bounded original-equation corrections.")
    record["status"] = "drives_accepted"


def export(args, record):
    staged, stage_link = load_stage(args, record)
    solved, solve_link = read_phase(args.solve_dir, "solve", args.solve_sha256, "drives_accepted", record)
    record["workspace_admission"] = low_band.admit_workspace(args.output, 2 * lawrence.GIB)
    system, modes, mesh_record = restore_system(staged)
    roots = staged["incident_port_roots"]
    report = {**solved["linear_solver"], "channels": []}

    def restore(operator, rhs):
        thickness.validate_equations(operator, rhs, staged, replay=True)
        vectors, norm = np.empty_like(rhs), lawrence.infinity_norm(operator)
        for row, vector in thickness.replay_channels(solved, Path(solve_link["path"]), roots,
                                                     stage_link, operator, rhs, norm, complete=True):
            report["channels"].append(row)
            vectors[:, row["column"]] = vector
        return vectors

    preparation, solving, exporting = ({key: phase[key] for key in thickness.EXECUTION_KEYS}
                                        for phase in (staged, solved, record))
    provenance = {**EXPLORATORY, "method": thickness.METHOD, "frequency_equation_preparation": preparation,
                  "linux_solve": solving, "linux_export": exporting, "prepared_frequency_stage": stage_link,
                  "solve_receipt": solve_link, "mesh_generated": False, "body_assembly_performed": False,
                  "section_assembly_performed": False, "safe_eigensolve_performed": False, "phase_adjustment_applied": False,
                  **{key: staged[key] for key in ("body_stage", "body_assembly_source", "safe_receipt", "section_receipt",
                      "open_channels", "safe_section", "safe_modes", "trace_sha256", "reference_planes_m",
                      "cutoff_status", "degeneracy_status", "safe_verification", "signed_relation")}}
    prepared = {"mesh_directory": Path(staged["mesh_directory"]), "mesh_checkpoint": (system.mesh, mesh_record), "system": system,
                "pattern": mesh_record["hole_pattern_provenance"], "assembly_resources": staged["assembly_resources"],
                "port_modes": modes, "solver_report": report, "linear_solver": restore,
                "attempt": workflow.digest(provenance), "provenance": provenance}
    result = workflow.run_study(staged["configuration"], args.output, _prepared_scattering=prepared)
    if (result["status"] != "completed_unverified" or result["power_balance_passed"] is not True
            or result["frequency_hz"] != FREQUENCY_HZ or result["hole_count"] != 26
            or result["incident_mode_indices"] != roots or len(report["channels"]) != len(roots)):
        raise ValueError(f"The 887.5 MHz z4 point failed original-equation or physical validation: {result['output_directory']}")
    public = Path(result["output_directory"])
    record.update(status="completed_unverified", stage=stage_link, solve=solve_link, **EXPLORATORY,
                  public_summary=artifact(public / "summary_AP_pnc.json"),
                  public_fields={name: artifact(public / name) for name in ("selected_fields_AP_pnc.npz",
                                                                           "selected_vertex_shapes_AP_pnc.vtu")})


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    phases = parser.add_subparsers(dest="mode", required=True)
    for mode, directories, receipts in (("safe", ("body-stage", "mesh", "section"), ()),
                                       ("stage", ("safe",), ("safe",)), ("solve", ("stage",), ("stage",)),
                                       ("export", ("stage", "solve"), ("stage", "solve"))):
        phase = phases.add_parser(mode)
        phase.add_argument("--output", type=Path, required=True)
        for name in directories:
            phase.add_argument(f"--{name}-dir", type=Path, required=True)
        for name in receipts:
            phase.add_argument(f"--{name}-sha256", required=True)
        if mode == "stage":
            phase.add_argument("--reference-operator", type=Path, required=True)
            for name in ("safe-verification", "relation"):
                phase.add_argument(f"--{name}", type=Path, required=True)
                phase.add_argument(f"--{name}-sha256", required=True)
        if mode == "solve":
            phase.add_argument("--prior-solve-dir", type=Path)
            phase.add_argument("--prior-solve-sha256")
    args = parser.parse_args()
    if args.mode == "solve" and (args.prior_solve_dir is None) != (args.prior_solve_sha256 is None):
        parser.error("Resume requires the prior directory and receipt checksum.")
    args.output = args.output.resolve()
    for name, value in vars(args).items():
        if name.endswith("_dir") and value is not None and args.output.is_relative_to(value.resolve()):
            parser.error("The output must be outside every archived input directory.")
    if args.output.exists() and any(path.name not in ("allocation_AP_pnc.txt", "time_AP_pnc.txt")
                                    for path in args.output.iterdir()):
        parser.error("Use a fresh phase directory, preserving every prior attempt.")
    run_phase(args, {"safe": safe, "stage": stage, "solve": solve, "export": export}[args.mode])


if __name__ == "__main__":
    main()
