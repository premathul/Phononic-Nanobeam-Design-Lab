"""Reuse the archived h100 z4 body and accepted 100 MHz ports. Athul Prem | _AP_pnc."""

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
from phononic.scattering_AP_pnc import solve_scattering
from tools import run_lawrence_low_band_z4_AP_pnc as z4
from tools.run_lawrence_uniform_h80_AP_pnc import artifact, direct_factor_admission

low_band, thickness, equations, lawrence = z4.low_band, z4.thickness, z4.equations, z4.lawrence
BODY_DOFS, PORT_DOFS, FREQUENCY_HZ = 307611, 675, 100_000_000
BODY_STAGE_SHA = "48fab07dff1f8f79fa76b76ace8aac4246019c68666e03f4dd2fe755de8cbd9f"
MESH_SUMMARY = {"sha256": "b295d82921ffd21c58ee8824b4ccd7c24831e967f8881e87fd0c8dd41e79c9c6", "bytes": 9558}
SAFE_RECEIPT_SHA = "59907315426ecf8b240ca8c37a40fe6cb8b019f9b79a941ce7fc65065c771cbe"
SAFE_MODES_SHA = "f290088d3ca25210979444ed011e113e165bb4fd41ec90a04d51693f60261fee"
TRACE_SHA = "51c54331436318f92a0a2b329acabe40606bb74e6b0e8ee0db7a88ce776c3905"
ROOTS, BACKWARD_ROOTS = (1348, 1346, 1344, 1294), (1349, 1347, 1345, 1295)
Z4_RUNNER_SHA = "59539b922aa2e37a059bb0f159442b0bac5347b7996bb097a343cf4a4a2216a8"
SOURCE_PAIRS = {
    "src/phononic/linear_lgmres_AP_pnc.py": (
        "459589f4e5de39c888966305e28dbd2b16f2947ec427ae9c131c69ae47c609df",
        "4e56e8a07df53ccc78927f860b69ad46dc5f495b7791acb008d9aa0ef71b5120"),
    "src/phononic/report_AP_pnc.py": (
        "871fc6a358958c7908ad17f57521bdb51772ea82739d83cdbe5b4b536a5145c5",
        "552a34e90558ba29440cb7d6951e8e857a744930e6f49c3bf38cb0357cbe3a71"),
    "src/phononic/workflow_AP_pnc.py": (
        "2ecccace6feee9456ed9a86744f246db7df34ade75a21dd3f0512d5ec553cbfa",
        "0bdfc18674d68ae3f62947001f0a8aa0591f466e4d1ba895f0f962fb0bb3915d"),
    "src/phononic/resumable_scattering_AP_pnc.py": (
        None, "dba6002d982647810f3fc714fdce73a0f34e306e06e8519df2d54778002205ec"),
}
def checked_sources():
    sources = z4.checked_sources()
    link = artifact(Path(z4.__file__))
    if link["sha256"] != Z4_RUNNER_SHA:
        raise ValueError("The reviewed four-layer cached-execution helper changed.")
    return {**sources, z4.__name__: link}


def run_phase(args, operation):
    identity = workflow.source_identity()
    threads = {name: os.environ.get(name) for name in lawrence.THREADS}
    job = os.environ.get("SLURM_JOB_ID")
    if (sys.platform != "linux" or not job or threads != lawrence.THREADS
            or workflow.digest(identity["source_sha256"]) != z4.fixed_port.SOURCE_MAP_SHA256
            or identity["packages"] != z4.fixed_port.PACKAGES):
        raise ValueError("Use the reviewed single-threaded Linux Slurm environment.")
    record = {**z4.fixed_port.AUTHOR, "mode": args.mode, "status": "running", "implementation": identity,
              "driver": artifact(Path(__file__)), "hostname": socket.gethostname(), "slurm_job_id": job,
              "numerical_thread_environment": threads, "numerical_backend": lawrence.solve_environment(),
              "reviewed_sources": checked_sources(), "superlu_library": artifact(Path(_superlu.__file__))}
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


def verify_body_sources(original, current):
    old, new = (dict(source["source_sha256"]) for source in (original, current))
    for path, expected in SOURCE_PAIRS.items():
        if (old.get(path), new.get(path)) != expected:
            raise ValueError("A declared historical h100 execution-source pair differs.")
        old.pop(path, None)
        new.pop(path, None)
    if (old != new or original["packages"] != current["packages"]
            or not original["platform"].startswith("Windows") or not current["platform"].startswith("Linux")):
        raise ValueError("The archived body and current physical numerical kernels or packages differ.")
    return {"basis": "Exact saved K/M artifacts and identical body/SAFE assembly kernels",
            "historical_execution_source_pairs": {path: list(pair) for path, pair in SOURCE_PAIRS.items()}}


def read_inputs(args, record):
    body, body_link = equations.read_receipt(args.body_stage_dir / "stages_AP_pnc.json", BODY_STAGE_SHA)
    compatibility = verify_body_sources(body["source"], record["implementation"])
    if (body["status"] != "operator_ready" or body["factorization_attempted"] is not False
            or body["mesh_comparison"]["ndof"] != BODY_DOFS):
        raise ValueError("The archived h100 body differs from the prepared original.")
    inputs = {key: z4.archive_link(args.body_stage_dir / Path(body[key]["path"]).name, body[key])
              for key in ("configuration", "K", "M", "port_modes")}
    mesh_link = z4.archive_link(args.mesh_dir / "mesh_AP_pnc.npz", body["mesh"])
    summary_link = z4.archive_link(args.mesh_dir / "summary_AP_pnc.json", MESH_SUMMARY)
    safe, safe_link = equations.read_receipt(args.safe_dir / "safe_AP_pnc.json", SAFE_RECEIPT_SHA)
    if (safe["status"] != "safe_ready" or safe["driver"]["sha256"] != Z4_RUNNER_SHA
            or safe["slurm_job_id"] != "1335500" or safe["frequency_hz"] != FREQUENCY_HZ
            or tuple(safe["incident_port_roots"]) != ROOTS or tuple(safe["backward_port_roots"]) != BACKWARD_ROOTS
            or safe["safe_modes"]["sha256"] != SAFE_MODES_SHA or safe["safe_section"]["sha256"] != z4.SECTION_SHA
            or safe["trace_sha256"] != z4.TRACE_SHA
            or any(safe["implementation"][key] != record["implementation"][key] for key in z4.low_band.prep.NUMERICAL_KEYS)):
        raise ValueError("The shared 0.1 GHz port acceptance differs from the admitted four-layer result.")
    config = json.loads(Path(inputs["configuration"]["path"]).read_bytes())
    shared = safe["configuration"]
    if (any(config[key] != shared[key] for key in ("geometry", "material"))
            or {**config["mesh"], "mesh_size_nm": 80} != shared["mesh"]
            or config["mesh"]["mesh_size_nm"] != 100
            or config["analysis"] != {**shared["analysis"], "frequency_GHz": 1}):
        raise ValueError("The h100 control and accepted h80 point differ in physical inputs or section settings.")
    config["analysis"]["frequency_GHz"] = FREQUENCY_HZ / 1e9
    config["resources"] = {"max_dofs": BODY_DOFS, "max_port_dofs": PORT_DOFS, "linear_solver": "direct",
                           "assembly_batch_elements": config["resources"]["assembly_batch_elements"]}
    mode_link = z4.archive_link(args.safe_dir / "modes-100000000_AP_pnc.npz", safe["safe_modes"])
    for key in ("safe_section", "section_receipt", "archived_safe_modes"):
        thickness.read_link(safe[key])
    return {"configuration": config, "body_stage": body_link, "body_assembly_source": body["source"],
            "body_source_compatibility": compatibility, "mesh_directory": str(args.mesh_dir.resolve()),
            "mesh_hash": body["mesh_hash"], "mesh": mesh_link, "mesh_summary": summary_link,
            "array_files": {key: inputs[key] for key in ("K", "M")}, "archived_safe_modes": inputs["port_modes"],
            "assembly_resources": body["stages"]["production_batched_assembly"], "displacement_dofs": BODY_DOFS,
            "safe_receipt": safe_link, "safe_modes": mode_link, "safe_section": safe["safe_section"],
            "section_receipt": safe["section_receipt"], "safe_reference_ports": safe["archived_safe_modes"],
            "trace_sha256": TRACE_SHA, "frequency_hz": FREQUENCY_HZ, "incident_port_roots": list(ROOTS),
            "backward_port_roots": list(BACKWARD_ROOTS), "open_channels": safe["open_channels"],
            "cutoff_status": safe["cutoff_status"], "degeneracy_status": safe["degeneracy_status"],
            "reference_planes_m": list(low_band.END_PLANES_M)}, body["A"]


def restore_system(staged):
    mesh_dir = Path(staged["mesh_directory"])
    if (thickness.read_link(staged["mesh"]) != mesh_dir / "mesh_AP_pnc.npz"
            or thickness.read_link(staged["mesh_summary"]) != mesh_dir / "summary_AP_pnc.json"):
        raise ValueError("The h100 mesh and checkpoint refer to different directories.")
    mesh, mesh_record = workflow.load_checkpoint(mesh_dir, staged["mesh_hash"])
    layout = z4.fixed_port.displacement_layout(mesh)
    if (layout.N != BODY_DOFS or mesh_record["hole_count"] != 26 or mesh_record["mesh_tetrahedra"] != 61368
            or mesh_record["end_face_dofs"] != {"xleft": PORT_DOFS, "xright": PORT_DOFS}):
        raise ValueError("The restored h100 displacement layout differs from its saved mesh.")
    port, anchor = z4.read_section(staged["safe_section"], staged["safe_reference_ports"])
    with np.load(thickness.read_link(staged["archived_safe_modes"]), allow_pickle=False) as saved:
        for key in z4.inventory.MODE_FIELDS:
            if not np.array_equal(np.asarray(getattr(anchor, key)), saved[key]):
                raise ValueError(f"The h100 and shared archived 1 GHz mode arrays differ: {key}")
        port = replace(port, left_dofs=saved["left_dofs"], right_dofs=saved["right_dofs"])
    if workflow.digest([port.left_dofs.tolist(), port.right_dofs.tolist()]) != TRACE_SHA:
        raise ValueError("The original h100 SAFE trace order differs.")
    z4.verify_trace(port, mesh, layout)
    matrices = [load_npz(thickness.read_link(staged["array_files"][key])) for key in ("K", "M")]
    if any(matrix.shape != (layout.N, layout.N) or matrix.dtype != np.float64 for matrix in matrices):
        raise ValueError("The saved h100 K/M differ from the real P2 displacement space.")
    system = ElasticSystem(mesh, layout, *matrices)
    with np.load(thickness.read_link(staged["safe_modes"]), allow_pickle=False) as saved:
        modes = z4.inventory.read_modes(saved, replace(port, elastic_system=system))
    roots = [ids[modes.propagating[ids]].tolist() for ids in (modes.positive_indices, modes.negative_indices)]
    if modes.frequency_hz != FREQUENCY_HZ or roots != [list(ROOTS), list(BACKWARD_ROOTS)]:
        raise ValueError("The reused modes differ from the accepted 0.1 GHz port basis.")
    return system, modes, mesh_record


def factor_index_exposure(operator, reference, reference_nnz):
    exposure = low_band.factor_index_exposure(operator, reference, reference_nnz)
    for key in ("archived_factor_job_id", "archived_aggregate_factor_nonzeros", "archived_aggregate_report_headroom",
                "archived_aggregate_report_headroom_fraction"):
        exposure.pop(key)
    return {**exposure, "archived_h100_direct_factor_available": False}


def stage(args, record):
    record["workspace_admission"] = low_band.admit_workspace(args.output, 2 * lawrence.GIB)
    inputs, original_operator = read_inputs(args, record)
    record.update(**inputs, status="constructing_frequency_equations", factorization_performed=False,
                  mesh_generated=False, body_assembly_performed=False, section_assembly_performed=False,
                  safe_eigensolve_performed=False, phase_adjustment_applied=False, **low_band.EXPLORATORY)
    z4.archive_link(args.body_stage_dir / "A_AP_pnc.npz", original_operator)
    system, modes, _ = restore_system(record)

    def capture(operator, rhs):
        thickness.validate_equations(operator, rhs, record)
        if not np.isfinite(operator.data).all() or not np.isfinite(rhs).all():
            raise ValueError("The new frequency equations contain nonfinite entries.")
        exposure = factor_index_exposure(operator, args.body_stage_dir / "A_AP_pnc.npz", original_operator["nnz"])
        storage = equations.matrix_bytes(operator) + rhs.nbytes
        if disk_usage(args.output).free < storage + lawrence.GIB:
            raise RuntimeError("The new equations and 1 GiB disk reserve do not fit.")
        operator_path, rhs_path = (args.output / thickness.MATRIX_FILES[key] for key in ("A", "rhs"))
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
    raise RuntimeError("The scattering core did not expose the original harmonic equations.")


def load_stage(args, record):
    staged, link = z4.read_phase(args.stage_dir, "stage", args.stage_sha256, "operator_staged", record)
    exposure = staged["factor_index_exposure"]
    if (staged["displacement_dofs"] != BODY_DOFS or staged["frequency_hz"] != FREQUENCY_HZ
            or staged["factorization_performed"] is not False or staged["trace_sha256"] != TRACE_SHA
            or tuple(staged["incident_port_roots"]) != ROOTS or tuple(staged["backward_port_roots"]) != BACKWARD_ROOTS
            or exposure["signed_index_maximum"] != low_band.INDEX_MAX
            or exposure["operator_nonzeros"] != staged["operator_nonzeros"]
            or exposure["initial_factor_slots_per_store"] != 30 * staged["operator_nonzeros"]
            or not 0 < exposure["initial_factor_slots_per_store"] <= low_band.INDEX_MAX):
        raise ValueError("The prepared h100 point or index exposure differs.")
    return staged, link


def solve(args, record):
    staged, stage_link = load_stage(args, record)
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
        prior, link = z4.read_phase(args.prior_solve_dir, "solve", args.prior_solve_sha256, status, record)
        record["prior_solve"] = link
        for row, vector in thickness.replay_channels(prior, path, ROOTS, stage_link, operator, rhs, norm, complete=False):
            saved = lawrence.write_vector(args.output / row["accepted_vector"]["file"], vector)
            report["channels"].append({**row, "accepted_vector": saved, "reused": True})
            workflow.write_json(receipt, record)
    first = len(report["channels"])
    if first == len(ROOTS):
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
            raise RuntimeError("The prepared equations exceed live node or job factor admission.")

    def publish(event, data, vector=None):
        record["candidate_history"].append({"event": event, **data})
        try:
            if event == "factor":
                report["factor"] = {"permc_spec": "COLAMD", **data}
                low_band.validate_factor_count(data["factor_nonzeros"], operator.shape[0])
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
        raise RuntimeError("A drive failed the bounded original-equation corrections.")
    record["status"] = "drives_accepted"


def export(args, record):
    staged, stage_link = load_stage(args, record)
    solved, solve_link = z4.read_phase(args.solve_dir, "solve", args.solve_sha256, "drives_accepted", record)
    record["workspace_admission"] = low_band.admit_workspace(args.output, 2 * lawrence.GIB)
    system, modes, mesh_record = restore_system(staged)
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
    provenance = {**low_band.EXPLORATORY, "method": thickness.METHOD, "frequency_equation_preparation": preparation,
                  "linux_solve": solving, "linux_export": exporting, "prepared_frequency_stage": stage_link,
                  "solve_receipt": solve_link, "mesh_generated": False, "body_assembly_performed": False,
                  "section_assembly_performed": False, "safe_eigensolve_performed": False, "phase_adjustment_applied": False,
                  **{key: staged[key] for key in ("body_stage", "body_assembly_source", "body_source_compatibility",
                     "safe_receipt", "section_receipt", "open_channels", "safe_section", "safe_modes", "safe_reference_ports",
                     "trace_sha256", "reference_planes_m", "cutoff_status", "degeneracy_status")}}
    prepared = {"mesh_directory": Path(staged["mesh_directory"]), "mesh_checkpoint": (system.mesh, mesh_record), "system": system,
                "pattern": mesh_record["hole_pattern_provenance"], "assembly_resources": staged["assembly_resources"],
                "port_modes": modes, "solver_report": report, "linear_solver": restore,
                "attempt": workflow.digest(provenance), "provenance": provenance}
    result = workflow.run_study(staged["configuration"], args.output, _prepared_scattering=prepared)
    if (result["status"] != "completed_unverified" or result["power_balance_passed"] is not True
            or result["frequency_hz"] != FREQUENCY_HZ or result["hole_count"] != 26
            or tuple(result["incident_mode_indices"]) != ROOTS or len(report["channels"]) != len(ROOTS)):
        raise ValueError(f"The h100 point failed physical or original-equation checks: {result['output_directory']}")
    public = Path(result["output_directory"])
    record.update(status="completed_unverified", stage=stage_link, solve=solve_link, **low_band.EXPLORATORY,
                  public_summary=artifact(public / "summary_AP_pnc.json"),
                  public_fields={name: artifact(public / name) for name in ("selected_fields_AP_pnc.npz",
                                                                           "selected_vertex_shapes_AP_pnc.vtu")})


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    phases = parser.add_subparsers(dest="mode", required=True)
    for mode, directories, receipts in (("stage", ("body-stage", "mesh", "safe"), ()),
                                       ("solve", ("stage",), ("stage",)), ("export", ("stage", "solve"), ("stage", "solve"))):
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
        parser.error("Resume requires the prior directory and receipt checksum.")
    args.output = args.output.resolve()
    for name, value in vars(args).items():
        if name.endswith("_dir") and value is not None and args.output.is_relative_to(value.resolve()):
            parser.error("The output must be outside every archived input directory.")
    if args.output.exists() and any(path.name not in ("allocation_AP_pnc.txt", "time_AP_pnc.txt")
                                    for path in args.output.iterdir()):
        parser.error("Use a fresh phase directory, preserving prior attempts.")
    run_phase(args, {"stage": stage, "solve": solve, "export": export}[args.mode])


if __name__ == "__main__":
    main()
