"""Prepare, solve or export the archived h80 z6 device at 887.5 MHz. Athul Prem | _AP_pnc."""

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
from phononic.resumable_scattering_AP_pnc import _array_digest
from phononic.scattering_AP_pnc import solve_scattering
from tools import lawrence_solver_AP_pnc as lawrence
from tools import run_lawrence_low_band_AP_pnc as low_band
from tools import run_lawrence_port_inventory_AP_pnc as inventory
from tools import run_lawrence_thickness_AP_pnc as prep
from tools import solve_lawrence_device_refinement_AP_pnc as equations
from tools import solve_lawrence_thickness_AP_pnc as thickness
from tools.run_lawrence_uniform_h80_AP_pnc import artifact, direct_factor_admission


FREQUENCY_HZ = 887_500_000
ROOTS = (1877, 1881, 1870, 1858, 1851)
BACKWARD_ROOTS = (1878, 1882, 1871, 1859, 1852)
END_PLANES_M = low_band.END_PLANES_M
INDEX_MAX = low_band.INDEX_MAX
EXPLORATORY = {**low_band.EXPLORATORY, "frequency_difference_is_spatial_convergence_check": False}
LOW_BAND_DRIVER_SHA = "b4272e2c107ccee6c463f7a84b77785dbce1e9f0d541fa4d44039af40476f932"
INTERVAL_LEDGER_SHA = "ada05174f12e937bc709b124de6ff97956381ceceb9f2f8595c863b91bdd517f"
INTERVAL_INPUTS_SHA = "e14c59270fcf53d0b26ec41fe46db25ec0d208936101e1f456c50ecf9b05ed26"
INTERVAL_POINT_SHA = "a53557ae5365570a39046b68f12eb0fd0e9e620d7df7d080561ec998e552e6e8"
INTERVAL_MODES_SHA = "93e4e089e6b329244b204ca10e3612653824b1e569476a51ea1924654a83b81c"
INTERVAL_PRODUCER_SHA = "89905221fa6571e8b188905a60b6ecaca03c24539e76090a260d4aea46a754c4"

# These immutable helpers keep their historical source identities and constants.
admit_workspace = low_band.admit_workspace
read_body_stage = low_band.read_body_stage
factor_index_exposure = low_band.factor_index_exposure
validate_factor_count = low_band.validate_factor_count


def checked_sources():
    link = artifact(Path(low_band.__file__))
    if link["sha256"] != LOW_BAND_DRIVER_SHA:
        raise ValueError("The reviewed low-band helper changed.")
    return {**low_band.checked_sources(), low_band.__name__: link}


def run_phase(args, operation):
    identity = workflow.source_identity()
    threads = {name: os.environ.get(name) for name in lawrence.THREADS}
    job = os.environ.get("SLURM_JOB_ID")
    if (sys.platform != "linux" or not job or threads != lawrence.THREADS
            or workflow.digest(identity["source_sha256"]) != prep.fixed_port.SOURCE_MAP_SHA256
            or identity["packages"] != prep.fixed_port.PACKAGES):
        raise ValueError("Use the reviewed single-threaded Linux Slurm environment.")
    sources = checked_sources()
    record = {**prep.fixed_port.AUTHOR, "mode": args.mode, "status": "running", "implementation": identity,
              "driver": artifact(Path(__file__)), "hostname": socket.gethostname(), "slurm_job_id": job,
              "numerical_thread_environment": threads, "numerical_backend": lawrence.solve_environment(),
              "reviewed_sources": sources, "superlu_library": artifact(Path(_superlu.__file__))}
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


def read_interval(directory, ledger_path, section_dir, body, current):
    """Read the accepted sample; keep raw producer and original archive identities separate."""
    ledger, ledger_link = equations.read_receipt(ledger_path, INTERVAL_LEDGER_SHA)
    inputs, inputs_link = equations.read_receipt(directory / "inputs_AP_pnc.json", INTERVAL_INPUTS_SHA)
    point, point_link = equations.read_receipt(directory / f"point-{FREQUENCY_HZ}_AP_pnc.json", INTERVAL_POINT_SHA)
    identity = inputs["identity"]
    if (workflow.digest(identity) != inputs["identity_sha256"]
            or point["identity_sha256"] != inputs["identity_sha256"]
            or point["point_sha256"] != workflow.digest({key: value for key, value in point.items() if key != "point_sha256"})
            or point["status"] != "accepted" or point["frequency_hz"] != FREQUENCY_HZ
            or ledger["status"] != "independently_verified_finite_count_interval"
            or ledger["inputs"] != inputs or ledger["state"]["status"] != "complete"
            or (ledger["state"]["lower_hz"], ledger["state"]["upper_hz"]) != (831_250_000, FREQUENCY_HZ)
            or [row for row in ledger["points"] if row["frequency_hz"] == FREQUENCY_HZ] != [point]
            or identity["producer"]["sha256"] != INTERVAL_PRODUCER_SHA):
        raise ValueError("The selected accepted sample or independently verified interval ledger differs.")
    archive = identity["archive"]
    original, producer = archive["original_implementation"], archive["current_implementation"]
    if (any(original[key] != body["implementation"][key] for key in prep.NUMERICAL_KEYS)
            or any(producer[key] != current["implementation"][key] for key in ("source_sha256", "packages"))
            or producer["python"] != "3.12.14" or not producer["platform"].startswith("Windows-")
            or original["python"] != "3.12.12" or not original["platform"].startswith("Linux-")):
        raise ValueError("The raw mode producer or original section numerical identity differs.")
    port, _, section_archive = inventory.read_archive(section_dir, original)
    section_hash = _array_digest(port.basis.mesh.p, port.basis.mesh.t, port.basis.doflocs,
                                 port.left_dofs, port.right_dofs, port.K0, port.M, port.Q, port.G)
    section_link = artifact(section_dir / inventory.PORTS)
    safe_link = artifact(section_dir / inventory.SAFE)
    if (archive["section_array_sha256"] != section_hash or archive["trace_sha256"] != section_archive["trace_sha256"]
            or any(archive["section_modes"][key] != section_link[key] for key in ("bytes", "sha256"))
            or any(archive["safe_receipt"][key] != safe_link[key] for key in ("bytes", "sha256"))
            or body["safe_modes"] != section_link or body["trace_sha256"] != section_archive["trace_sha256"]
            or body["safe_receipt"] != safe_link
            or any(body["configuration"][key] != section_archive["configuration"][key]
                   for key in ("geometry", "mesh", "material", "analysis"))):
        raise ValueError("The accepted sample, archived SAFE section and saved body differ.")
    metrics = point["metrics"]
    if (metrics["discrete_roots"] != 2 * inventory.PORT_DOFS
            or metrics["positive_open_channels"] != metrics["negative_open_channels"] or metrics["positive_open_channels"] != 5
            or metrics["cutoff_status"] != "solver_resolved_at_sample"
            or metrics["degeneracy_status"] != "none_detected" or metrics["coincident_propagating_root_groups"]
            or point["actual_environment"] != "inputs_AP_pnc.json:identity.archive.current_implementation"
            or point["modes"]["path"] != f"modes-{FREQUENCY_HZ}_AP_pnc.npz"):
        raise ValueError("The selected full SAFE sample has unresolved channels or provenance.")
    for direction, expected in (("positive", ROOTS), ("negative", BACKWARD_ROOTS)):
        if tuple(row["root_id"] for row in metrics["open_channels"] if row["direction"] == direction) != expected:
            raise ValueError("The accepted sample signed channel order differs.")
    modes_link = artifact(directory / point["modes"]["path"])
    if (modes_link["sha256"] != INTERVAL_MODES_SHA
            or any(point["modes"][key] != modes_link[key] for key in ("bytes", "sha256"))):
        raise ValueError("The accepted full SAFE modes differ from their saved artifact.")
    return port, {"interval_inputs": inputs_link, "interval_point": point_link, "interval_ledger": ledger_link,
            "safe_section": section_link, "safe_modes": modes_link, "safe_receipt": safe_link,
            "trace_sha256": body["trace_sha256"], "safe_section_array_sha256": section_hash,
            "safe_archive_directory": str(section_dir.resolve()), "original_section_implementation": original,
            "section_mode_implementation": producer, "section_mode_execution_provenance": point["execution_provenance"],
            "open_channels": metrics["open_channels"], "cutoff_status": metrics["cutoff_status"],
            "degeneracy_status": metrics["degeneracy_status"], "cross_frequency_branch_identity": "unresolved"}


def restore_system(staged, implementation, *, port=None):
    if any(implementation[key] != staged["original_section_implementation"][key] for key in prep.NUMERICAL_KEYS):
        raise ValueError("The live Linux numerical identity differs from the original archived section.")
    mesh_dir = Path(staged["mesh_directory"])
    for key, name in (("mesh", "mesh_AP_pnc.npz"), ("mesh_summary", "summary_AP_pnc.json")):
        if thickness.read_link(staged[key]) != mesh_dir / name:
            raise ValueError("A mesh checkpoint artifact refers to a different directory.")
    mesh, mesh_record = workflow.load_checkpoint(mesh_dir, staged["mesh_hash"])
    basis = prep.fixed_port.displacement_layout(mesh)
    if basis.N != prep.BODY_DOFS or mesh_record["hole_count"] != 26:
        raise ValueError("The restored body does not have the accepted P2 displacement layout.")
    if port is None:
        port, _, _ = inventory.read_archive(Path(staged["safe_archive_directory"]), staged["original_section_implementation"])
    if workflow.digest([port.left_dofs.tolist(), port.right_dofs.tolist()]) != staged["trace_sha256"]:
        raise ValueError("The restored SAFE maps differ from the admitted trace order.")
    body_components = prep.fixed_port.components(basis)
    port_components = prep.fixed_port.components(port.basis)
    tagged = prep.fixed_port.port_trace_ids(mesh, basis)
    for saved, actual, plane in zip((port.left_dofs, port.right_dofs), tagged, END_PLANES_M, strict=True):
        if (not np.array_equal(np.sort(saved), np.sort(actual))
                or not np.array_equal(body_components[saved], port_components)
                or not np.allclose(basis.doflocs[0, saved], plane, rtol=0, atol=1e-15)
                or not np.allclose(basis.doflocs[1:, saved], port.basis.doflocs, rtol=0, atol=1e-15)):
            raise ValueError("The saved SAFE coordinates/components differ from the complete tagged end trace.")
    matrices = [load_npz(thickness.read_link(staged["array_files"][key])) for key in ("K", "M")]
    if any(matrix.shape != (basis.N, basis.N) or matrix.dtype != np.float64 for matrix in matrices):
        raise ValueError("A saved body matrix differs from the admitted real P2 layout.")
    system = ElasticSystem(mesh, basis, *matrices)
    with np.load(thickness.read_link(staged["safe_modes"]), allow_pickle=False) as arrays:
        modes = inventory.read_modes(arrays, replace(port, elastic_system=system))
    for side, directions, expected in (("positive", modes.positive_indices, ROOTS),
                                       ("negative", modes.negative_indices, BACKWARD_ROOTS)):
        receipt_roots = tuple(row["root_id"] for row in staged["open_channels"] if row["direction"] == side)
        if tuple(directions[modes.propagating[directions]]) != expected or receipt_roots != expected:
            raise ValueError("The saved SAFE open-channel order differs from the signed inventory.")
    if modes.frequency_hz != FREQUENCY_HZ:
        raise ValueError("The raw SAFE modes are for a different frequency.")
    return system, modes, mesh_record


def stage(args, record):
    body, body_link = read_body_stage(args.body_stage_dir, record)
    disk_budget = body["operator_csc_bytes"] + prep.BODY_DOFS * len(ROOTS) * 16 + lawrence.GIB
    record["workspace_admission"] = admit_workspace(args.output, disk_budget)
    port, section = read_interval(args.interval_dir, args.interval_ledger, args.safe_dir, body, record)
    config = workflow.resolve_config(body["configuration"])
    config["analysis"]["frequency_GHz"] = FREQUENCY_HZ / 1e9
    record.update(status="constructing_frequency_equations", configuration=config, body_stage=body_link,
                  **{key: body[key] for key in ("mesh_directory", "mesh_hash", "mesh", "mesh_summary",
                                               "preflight_receipt", "assembly_resources", "displacement_dofs")},
                  **section, **EXPLORATORY, frequency_hz=FREQUENCY_HZ, incident_port_roots=list(ROOTS),
                  backward_port_roots=list(BACKWARD_ROOTS), reference_planes_m=list(END_PLANES_M),
                  phase_adjustment_applied=False,
                  array_files={key: body["array_files"][key] for key in ("K", "M")},
                  mesh_generated=False, body_assembly_performed=False, safe_eigensolve_performed=False,
                  factorization_performed=False)
    workflow.write_json(args.output / "stage_AP_pnc.json", record)
    system, modes, _ = restore_system(record, record["implementation"], port=port)
    reference = thickness.read_link(body["array_files"]["A"])

    def capture(operator, rhs):
        thickness.validate_equations(operator, rhs, record)
        if not np.isfinite(operator.data).all() or not np.isfinite(rhs).all():
            raise ValueError("The frequency-specific equations contain nonfinite entries.")
        exposure = factor_index_exposure(operator, reference, body["operator_nonzeros"])
        if not exposure["matches_archived_1_ghz_csc_graph"]:
            raise ValueError("The prepared frequency operator graph differs from the archived device.")
        storage = equations.matrix_bytes(operator) + rhs.nbytes
        if disk_usage(args.output).free < storage + lawrence.GIB:
            raise RuntimeError("The prepared equations and 1 GiB disk reserve do not fit.")
        paths = {key: args.output / thickness.MATRIX_FILES[key] for key in ("A", "rhs")}
        with paths["A"].open("xb") as stream:
            save_npz(stream, operator, compressed=False)
        with paths["rhs"].open("xb") as stream:
            np.save(stream, rhs, allow_pickle=False)
        record["array_files"].update({key: artifact(path) for key, path in paths.items()})
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
    raise RuntimeError("The scattering core did not expose the original harmonic equations.")


def load_stage(args, record):
    staged, link = thickness.load_stage(args, record)
    exposure = staged["factor_index_exposure"]
    if (staged["frequency_hz"] != FREQUENCY_HZ or tuple(staged["incident_port_roots"]) != ROOTS
            or tuple(staged["backward_port_roots"]) != BACKWARD_ROOTS
            or staged["superlu_library"] != record["superlu_library"]
            or staged["reviewed_sources"] != record["reviewed_sources"]
            or tuple(staged["reference_planes_m"]) != END_PLANES_M
            or staged["interval_inputs"]["sha256"] != INTERVAL_INPUTS_SHA
            or staged["interval_point"]["sha256"] != INTERVAL_POINT_SHA
            or staged["interval_ledger"]["sha256"] != INTERVAL_LEDGER_SHA
            or staged["safe_modes"]["sha256"] != INTERVAL_MODES_SHA
            or exposure["matches_archived_1_ghz_csc_graph"] is not True
            or exposure["signed_index_maximum"] != INDEX_MAX
            or exposure["operator_nonzeros"] != staged["operator_nonzeros"]
            or exposure["initial_factor_slots_per_store"] != 30 * staged["operator_nonzeros"]
            or not 0 < exposure["initial_factor_slots_per_store"] <= INDEX_MAX):
        raise ValueError("The prepared frequency, graph exposure or numerical sources differ.")
    return staged, link


def solve(args, record):
    staged, stage_link = load_stage(args, record)
    record["workspace_admission"] = admit_workspace(args.output, 4 * staged["array_files"]["rhs"]["bytes"] + lawrence.GIB)
    operator = load_npz(thickness.read_link(staged["array_files"]["A"]))
    rhs = np.load(thickness.read_link(staged["array_files"]["rhs"]), allow_pickle=False)
    thickness.validate_equations(operator, rhs, staged, replay=True)
    exposure = staged["factor_index_exposure"]
    if (operator.indices.dtype != np.int32 or operator.indptr.dtype != np.int32
            or lawrence.array_sha(operator.indices, operator.indptr) != exposure["operator_graph_sha256"]):
        raise ValueError("The admitted complex128 CSC graph or index width changed.")

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
        prior, link = thickness.read_phase(args.prior_solve_dir, "solve", args.prior_solve_sha256, status, record)
        if prior["superlu_library"] != record["superlu_library"] or prior["reviewed_sources"] != record["reviewed_sources"]:
            raise ValueError("The previous accepted vectors came from a different numerical build.")
        record["prior_solve"] = link
        for row, vector in thickness.replay_channels(prior, path, ROOTS, stage_link, operator, rhs, norm, complete=False):
            saved = lawrence.write_vector(args.output / row["accepted_vector"]["file"], vector)
            report["channels"].append({**row, "accepted_vector": saved, "reused": True})
            workflow.write_json(receipt, record)
    first = len(report["channels"])
    if first == len(ROOTS):
        previous_report = prior["linear_solver"]
        factor = previous_report["factor"]
        validate_factor_count(factor["factor_nonzeros"], operator.shape[0])
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
                validate_factor_count(data["factor_nonzeros"], operator.shape[0])
                record["status"] = "solving"
            else:
                column = first + data["column"]
                data.update(column=column, incident_port_root=ROOTS[column])
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
    if first + solved != len(ROOTS) or len(report["channels"]) != len(ROOTS):
        raise RuntimeError("A drive failed the original-equation checks within the bounded corrections.")
    record["status"] = "drives_accepted"


def export(args, record):
    staged, stage_link = load_stage(args, record)
    solved, solve_link = thickness.read_phase(args.solve_dir, "solve", args.solve_sha256, "drives_accepted", record)
    if solved["superlu_library"] != record["superlu_library"] or solved["reviewed_sources"] != record["reviewed_sources"]:
        raise ValueError("The solved vectors came from a different numerical build.")
    output_reserve = 3 * (staged["safe_modes"]["bytes"] + staged["array_files"]["rhs"]["bytes"]) + lawrence.GIB
    record["workspace_admission"] = admit_workspace(args.output, output_reserve)
    system, modes, mesh_record = restore_system(staged, record["implementation"])
    mesh_dir = Path(staged["mesh_directory"])
    report = {**solved["linear_solver"], "channels": []}

    def restore(operator, rhs):
        thickness.validate_equations(operator, rhs, staged, replay=True)
        vectors, norm = np.empty_like(rhs), lawrence.infinity_norm(operator)
        for row, vector in thickness.replay_channels(solved, Path(solve_link["path"]), ROOTS,
                                                     stage_link, operator, rhs, norm, complete=True):
            report["channels"].append(row)
            vectors[:, row["column"]] = vector
        return vectors

    preparation, solving, exporting = ({key: phase[key] for key in thickness.EXECUTION_KEYS}
                                        for phase in (staged, solved, record))
    provenance = {**EXPLORATORY, "method": thickness.METHOD, "frequency_equation_preparation": preparation,
                  "linux_solve": solving, "linux_export": exporting, "prepared_frequency_stage": stage_link,
                  "solve_receipt": solve_link, "mesh_generated": False, "body_assembly_performed": False,
                  "safe_eigensolve_performed": False, "phase_adjustment_applied": False,
                  **{key: staged[key] for key in ("body_stage", "interval_inputs", "interval_point", "interval_ledger", "open_channels",
                                                 "original_section_implementation", "section_mode_implementation",
                                                 "section_mode_execution_provenance", "cross_frequency_branch_identity",
                                                 "safe_section", "safe_modes", "safe_receipt", "trace_sha256",
                                                 "reference_planes_m", "cutoff_status", "degeneracy_status")}}
    prepared = {"mesh_directory": mesh_dir, "mesh_checkpoint": (system.mesh, mesh_record), "system": system,
                "pattern": mesh_record["hole_pattern_provenance"], "assembly_resources": staged["assembly_resources"],
                "port_modes": modes, "solver_report": report, "linear_solver": restore,
                "attempt": workflow.digest(provenance), "provenance": provenance}
    result = workflow.run_study(staged["configuration"], args.output, _prepared_scattering=prepared)
    if (result["status"] != "completed_unverified" or result["power_balance_passed"] is not True
            or result["frequency_hz"] != FREQUENCY_HZ or result["hole_count"] != 26
            or tuple(result["incident_mode_indices"]) != ROOTS or len(report["channels"]) != len(ROOTS)):
        raise ValueError(f"The 0.8875 GHz point failed original-equation or physical validation: {result['output_directory']}")
    public = Path(result["output_directory"])
    fields = {name: artifact(public / name) for name in ("selected_fields_AP_pnc.npz", "selected_vertex_shapes_AP_pnc.vtu")}
    record.update(status="completed_unverified", stage=stage_link, solve=solve_link, **EXPLORATORY,
                  public_summary=artifact(public / "summary_AP_pnc.json"), public_fields=fields)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    phases = parser.add_subparsers(dest="mode", required=True)
    for mode, directories, receipts in (("stage", ("body-stage", "safe", "interval"), ()),
                                       ("solve", ("stage",), ("stage",)),
                                       ("export", ("stage", "solve"), ("stage", "solve"))):
        phase = phases.add_parser(mode)
        phase.add_argument("--output", type=Path, required=True)
        for name in directories:
            phase.add_argument(f"--{name}-dir", type=Path, required=True)
        for name in receipts:
            phase.add_argument(f"--{name}-sha256", required=True)
        if mode == "stage":
            phase.add_argument("--interval-ledger", type=Path, required=True)
        if mode == "solve":
            phase.add_argument("--prior-solve-dir", type=Path)
            phase.add_argument("--prior-solve-sha256")
    args = parser.parse_args()
    if args.mode == "solve" and (args.prior_solve_dir is None) != (args.prior_solve_sha256 is None):
        parser.error("Resume requires the prior solve directory and receipt checksum.")
    args.output = args.output.resolve()
    for name, value in vars(args).items():
        if name.endswith("_dir") and value is not None and args.output.is_relative_to(value.resolve()):
            parser.error("The output must be outside every saved input directory.")
    if args.output.exists() and any(path.name not in ("allocation_AP_pnc.txt", "time_AP_pnc.txt")
                                    for path in args.output.iterdir()):
        parser.error("Use a fresh output directory for each phase or explicit resume.")

    run_phase(args, {"stage": stage, "solve": solve, "export": export}[args.mode])


if __name__ == "__main__":
    main()
