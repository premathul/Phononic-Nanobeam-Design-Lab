"""Prepare, solve or export the archived h80 z6 device at 100 MHz. Athul Prem | _AP_pnc."""

from __future__ import annotations

import argparse
from dataclasses import replace
import json
import os
from pathlib import Path
from shutil import disk_usage
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
from tools import run_lawrence_port_inventory_AP_pnc as inventory
from tools import run_lawrence_thickness_AP_pnc as prep
from tools import solve_lawrence_device_refinement_AP_pnc as equations
from tools import solve_lawrence_thickness_AP_pnc as thickness
from tools.run_lawrence_uniform_h80_AP_pnc import artifact, direct_factor_admission


FREQUENCY_HZ = 100_000_000
ROOTS = (1948, 1916, 1940, 1873)
BACKWARD_ROOTS = (1949, 1917, 1941, 1874)
END_PLANES_M = (0.0, 1.665e-5)
BODY_STAGE_SHA = "95c1cfde8d51d77ab74fe9d2c424c1eda0822556a30664cbe581999f97e0ea5f"
INVENTORY_INPUTS_SHA = "4fb564051f508ecbba2970dd98733b7e65a7fdb0c1f3b2f7793c32560600f0e6"
INVENTORY_POINT_SHA = "c7d81677fbe2d67649e943c609562a6407e24a97e6ebb4b878583e39d8806344"
INVENTORY_DRIVER_SHA = "5bfb37a4c4320df4d8e9fd65421c3bf3595775c2dbe297916e06d196255de44c"
THICKNESS_DRIVER_SHA = "741f6fb09127de7eb590686e9bca40951864c04b261d87da667e13861fc75490"
INDEX_MAX = int(np.iinfo(np.int32).max)
ARCHIVED_FACTOR_NNZ = 1_846_932_604
EXPLORATORY = {"spatial_convergence_established": False, "frequency_convergence_established": False}


def checked_sources():
    reviewed = {prep: thickness.PREP_DRIVER_SHA, equations: thickness.STAGED_HELPER_SHA,
                thickness: THICKNESS_DRIVER_SHA, inventory: INVENTORY_DRIVER_SHA}
    links = {module.__name__: artifact(Path(module.__file__)) for module in reviewed}
    if any(links[module.__name__]["sha256"] != digest for module, digest in reviewed.items()):
        raise ValueError("A reviewed archive or prepared-equation helper changed.")
    return links


def admit_workspace(output, disk_bytes, *, memory_bytes=6 * lawrence.GIB):
    """Cover the measured 1 GHz preparation/export peaks with additional headroom."""
    quota = int(os.environ.get("SLURM_MEM_PER_NODE", "0")) * lawrence.MIB
    cpus = int(os.environ.get("SLURM_CPUS_PER_TASK", "0"))
    controls = lawrence.cgroup_controls()
    if quota <= 0 or cpus < 2 or not any(cap is not None and lawrence.is_job_group(path)
                                         for path, cap, *_ in controls):
        raise ValueError("A bounded Slurm memory allocation with at least two CPUs is required.")
    snapshot = lawrence.memory_snapshot(quota, controls)
    snapshot.update(slurm_memory_bytes=quota, slurm_cpus=cpus, required_memory_bytes=memory_bytes,
                    disk_available_bytes=disk_usage(output).free, required_disk_bytes=disk_bytes)
    if (min(snapshot["job_headroom_bytes"], snapshot["node_available_bytes"]) < memory_bytes
            or snapshot["disk_available_bytes"] < disk_bytes):
        raise RuntimeError(f"Insufficient workspace resources: {json.dumps(snapshot)}")
    return snapshot


def read_body_stage(directory, current):
    body, link = prep.read_receipt(directory, "stage_AP_pnc.json", BODY_STAGE_SHA, "operator_staged",
                                   THICKNESS_DRIVER_SHA, current)
    if (body["slurm_job_id"] != "1335309" or body["frequency_hz"] != inventory.ANCHOR_HZ
            or body["displacement_dofs"] != prep.BODY_DOFS or body["factorization_performed"] is not False):
        raise ValueError("The archived stage differs from the accepted h80 z6 body.")
    return body, link


def read_inventory(directory, section_dir, body, current):
    inputs, inputs_link = equations.read_receipt(directory / "inventory_inputs_AP_pnc.json", INVENTORY_INPUTS_SHA)
    point, point_link = equations.read_receipt(directory / f"point-{FREQUENCY_HZ}_AP_pnc.json", INVENTORY_POINT_SHA)
    identity = inputs["identity"]
    if (workflow.digest(identity) != inputs["identity_sha256"]
            or point["identity_sha256"] != inputs["identity_sha256"]
            or point["point_sha256"] != workflow.digest({key: value for key, value in point.items() if key != "point_sha256"})
            or point["status"] != "accepted" or point["frequency_hz"] != FREQUENCY_HZ
            or point["execution"]["slurm_job_id"] != "1335406"):
        raise ValueError("The selected inventory sample differs from its acceptance receipt.")
    port, _, archive = inventory.read_archive(section_dir, current["implementation"])
    section_hash = _array_digest(port.basis.mesh.p, port.basis.mesh.t, port.basis.doflocs,
                                 port.left_dofs, port.right_dofs, port.K0, port.M, port.Q, port.G)
    section_link = artifact(section_dir / inventory.PORTS)
    if (identity["provenance"]["archive"] != archive or identity["section_array_sha256"] != section_hash
            or body["safe_modes"] != section_link or body["trace_sha256"] != archive["trace_sha256"]):
        raise ValueError("The accepted sample, archived SAFE section and saved body trace maps differ.")
    if any(body["configuration"][key] != archive["configuration"][key]
           for key in ("geometry", "mesh", "material", "analysis")):
        raise ValueError("The saved body and accepted section differ in geometry or physics inputs.")
    if point["modes"]["path"] != f"modes-{FREQUENCY_HZ}_AP_pnc.npz":
        raise ValueError("The accepted inventory mode filename differs.")
    modes_link = artifact(directory / point["modes"]["path"])
    if any(point["modes"][key] != modes_link[key] for key in ("bytes", "sha256")):
        raise ValueError("The accepted inventory modes differ from the saved artifact.")
    return port, {"inventory_inputs": inputs_link, "inventory_point": point_link, "safe_section": section_link,
            "safe_modes": modes_link, "safe_receipt": body["safe_receipt"], "trace_sha256": body["trace_sha256"],
            "safe_section_array_sha256": section_hash, "safe_archive_directory": str(section_dir.resolve()),
            "section_inventory_execution": point["execution"], "open_channels": point["open_channels"],
            "cutoff_status": point["cutoff_status"], "degeneracy_status": point["degeneracy_status"]}


def restore_system(staged, implementation, *, port=None):
    mesh_dir = Path(staged["mesh_directory"])
    for key, name in (("mesh", "mesh_AP_pnc.npz"), ("mesh_summary", "summary_AP_pnc.json")):
        if thickness.read_link(staged[key]) != mesh_dir / name:
            raise ValueError("A mesh checkpoint artifact refers to a different directory.")
    mesh, mesh_record = workflow.load_checkpoint(mesh_dir, staged["mesh_hash"])
    basis = prep.fixed_port.displacement_layout(mesh)
    if basis.N != prep.BODY_DOFS or mesh_record["hole_count"] != 26:
        raise ValueError("The restored body does not have the accepted P2 displacement layout.")
    if port is None:
        port, _, _ = inventory.read_archive(Path(staged["safe_archive_directory"]), implementation)
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


def factor_index_exposure(operator, reference_path, reference_nnz):
    """Inspect CSC indices and compare the prepared graph with the prior direct solve."""
    nnz = int(operator.nnz)
    initial_slots = 30 * nnz
    if (operator.indices.dtype != np.int32 or operator.indptr.dtype != np.int32 or np.dtype(np.intc).itemsize != 4
            or operator.shape[0] >= INDEX_MAX or not 0 < nnz <= INDEX_MAX or initial_slots > INDEX_MAX):
        raise ValueError("The operator or initial allocation exceeds the reviewed SuperLU int32 interface.")
    with np.load(reference_path, allow_pickle=False) as old:
        same_pattern = (np.array_equal(old["shape"], operator.shape)
                        and np.array_equal(old["indptr"], operator.indptr)
                        and np.array_equal(old["indices"], operator.indices))
    estimated_initial_bytes = 40 * initial_slots + 2 * equations.matrix_bytes(operator) + 512 * lawrence.MIB
    return {"csc_index_dtype": str(operator.indices.dtype), "csc_pointer_dtype": str(operator.indptr.dtype),
            "signed_index_maximum": INDEX_MAX, "operator_nonzeros": nnz,
            "operator_graph_sha256": lawrence.array_sha(operator.indices, operator.indptr),
            "initial_factor_slots_per_store": initial_slots,
            "initial_factor_slots_headroom": INDEX_MAX - initial_slots,
            "estimated_initial_factor_incremental_bytes": estimated_initial_bytes,
            "required_job_headroom_at_factor_bytes": estimated_initial_bytes + lawrence.JOB_RESERVE,
            "required_node_available_at_factor_bytes": estimated_initial_bytes + lawrence.NODE_RESERVE,
            "matches_archived_1_ghz_csc_graph": same_pattern, "archived_operator_nonzeros": reference_nnz,
            "archived_factor_job_id": "1335321", "archived_aggregate_factor_nonzeros": ARCHIVED_FACTOR_NNZ,
            "archived_aggregate_report_headroom": INDEX_MAX - ARCHIVED_FACTOR_NNZ,
            "archived_aggregate_report_headroom_fraction": (INDEX_MAX - ARCHIVED_FACTOR_NNZ) / INDEX_MAX,
            "numeric_pivoting_may_change_fill": True, "new_factor_index_capacity_established": False,
            "interface_source": ("https://github.com/scipy/scipy/blob/v1.18.1/scipy/sparse/linalg/_dsolve/linsolve.py"
                                 "#L394")}


def stage(args, record):
    body, body_link = read_body_stage(args.body_stage_dir, record)
    disk_budget = body["operator_csc_bytes"] + prep.BODY_DOFS * len(ROOTS) * 16 + lawrence.GIB
    record["workspace_admission"] = admit_workspace(args.output, disk_budget)
    port, section = read_inventory(args.inventory_dir, args.safe_dir, body, record)
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
            or exposure["signed_index_maximum"] != INDEX_MAX
            or exposure["operator_nonzeros"] != staged["operator_nonzeros"]
            or exposure["initial_factor_slots_per_store"] != 30 * staged["operator_nonzeros"]
            or not 0 < exposure["initial_factor_slots_per_store"] <= INDEX_MAX):
        raise ValueError("The prepared frequency, graph exposure or numerical sources differ.")
    return staged, link


def validate_factor_count(count, ndof):
    if not 2 * ndof <= count <= INDEX_MAX:
        raise RuntimeError("SuperLU returned an invalid signed aggregate factor count.")


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
                  **{key: staged[key] for key in ("body_stage", "inventory_inputs", "inventory_point", "open_channels",
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
        raise ValueError(f"The 0.1 GHz point failed original-equation or physical validation: {result['output_directory']}")
    public = Path(result["output_directory"])
    fields = {name: artifact(public / name) for name in ("selected_fields_AP_pnc.npz", "selected_vertex_shapes_AP_pnc.vtu")}
    record.update(status="completed_unverified", stage=stage_link, solve=solve_link, **EXPLORATORY,
                  public_summary=artifact(public / "summary_AP_pnc.json"), public_fields=fields)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    phases = parser.add_subparsers(dest="mode", required=True)
    for mode, directories, receipts in (("stage", ("body-stage", "safe", "inventory"), ()),
                                       ("solve", ("stage",), ("stage",)),
                                       ("export", ("stage", "solve"), ("stage", "solve"))):
        phase = phases.add_parser(mode)
        phase.add_argument("--output", type=Path, required=True)
        for name in directories:
            phase.add_argument(f"--{name}-dir", type=Path, required=True)
        for name in receipts:
            phase.add_argument(f"--{name}-sha256", required=True)
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

    def execute(options, record):
        record["reviewed_sources"] = checked_sources()
        record["superlu_library"] = artifact(Path(_superlu.__file__))
        {"stage": stage, "solve": solve, "export": export}[options.mode](options, record)

    prep.run_phase(args, Path(__file__), f"{args.mode}_AP_pnc.json", execute)


if __name__ == "__main__":
    main()
