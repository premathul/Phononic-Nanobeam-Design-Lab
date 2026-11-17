"""Reuse the archived h80 z4 body at 100 MHz. Athul Prem | _AP_pnc."""

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
from skfem import Basis, ElementTriP2, ElementVector, MeshTri

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT / "src"), str(ROOT)]

from phononic import workflow_AP_pnc as workflow
from phononic.elasticity_AP_pnc import ElasticSystem
from phononic.port_modes_AP_pnc import PortSystem, solve_port_modes
from phononic.resumable_scattering_AP_pnc import _array_digest
from phononic.scattering_AP_pnc import solve_scattering
from tools import export_lawrence_scattering_AP_pnc as archived_export
from tools import run_lawrence_low_band_AP_pnc as low_band
from tools import run_lawrence_port_inventory_AP_pnc as inventory
from tools import solve_lawrence_device_refinement_AP_pnc as equations
from tools import solve_lawrence_thickness_AP_pnc as thickness
from tools.run_lawrence_uniform_h80_AP_pnc import artifact, direct_factor_admission


FREQUENCY_HZ = 100_000_000
BODY_DOFS, PORT_DOFS = 426627, 675
BODY_STAGE_SHA = "1c50971e27bfc52e63000c9d25df100e21bbe28b7fa47d7549f82cf465531a78"
SECTION_RECEIPT_SHA = "677c64a816ea42a8b37e1358d5549d63ecaeec7623a9366139ba64d301bbea4e"
SECTION_SHA = "2a2aa9347921ac6940f851f2eb53bc06480a28a6cf67c27a000faa357b1404e8"
TRACE_SHA = "2752f71b459282bb33610b39ce99fd25eee6784a510e4c6013415a9357e6fd7b"
LOW_BAND_SHA = "b4272e2c107ccee6c463f7a84b77785dbce1e9f0d541fa4d44039af40476f932"
EXPORT_HELPER_SHA = "4c2e76c81817154afeef4a0a9a9ff5b691d297544b9ae7659dd3fcc087aa4ad1"
PRIOR_FACTOR_NNZ = 803_307_165
lawrence = low_band.lawrence
fixed_port = low_band.prep.fixed_port


def checked_sources():
    sources = low_band.checked_sources()
    for module, digest in ((low_band, LOW_BAND_SHA), (archived_export, EXPORT_HELPER_SHA)):
        link = artifact(Path(module.__file__))
        if link["sha256"] != digest:
            raise ValueError("A reviewed z4 archive or execution helper changed.")
        sources[module.__name__] = link
    return sources


def run_phase(args, operation):
    implementation = workflow.source_identity()
    threads = {name: os.environ.get(name) for name in lawrence.THREADS}
    job = os.environ.get("SLURM_JOB_ID")
    if (sys.platform != "linux" or not job or threads != lawrence.THREADS
            or workflow.digest(implementation["source_sha256"]) != fixed_port.SOURCE_MAP_SHA256
            or implementation["packages"] != fixed_port.PACKAGES):
        raise ValueError("Use the reviewed single-threaded Linux Slurm environment.")
    record = {**fixed_port.AUTHOR, "mode": args.mode, "status": "running", "implementation": implementation,
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


def archive_link(path, original):
    link = artifact(path)
    if any(link[key] != original[key] for key in ("bytes", "sha256")):
        raise ValueError(f"An archived artifact differs: {path}")
    return link


def read_body(directory, mesh_dir, current):
    body, receipt = equations.read_receipt(directory / "stages_AP_pnc.json", BODY_STAGE_SHA)
    archived_export.check_replay_environment(body["source"], current["implementation"], linux_replay=True)
    if (body["status"] != "operator_ready" or body["mesh_comparison"]["ndof"] != BODY_DOFS
            or body["factorization_attempted"] is not False or body["solve_attempted"] is not False):
        raise ValueError("The saved stage differs from the admitted h80 z4 body.")
    paths = {key: archive_link(directory / Path(body[key]["path"]).name, body[key])
             for key in ("configuration", "K", "M", "port_modes")}
    paths.update({key: archive_link(mesh_dir / name, body[old])
                  for key, old, name in (("mesh", "mesh", "mesh_AP_pnc.npz"),
                                         ("mesh_summary", "mesh_checkpoint_summary", "summary_AP_pnc.json"))})
    config = json.loads(Path(paths["configuration"]["path"]).read_bytes())
    if (config["mesh"]["thickness_layers"] != 4 or config["mesh"]["mesh_size_nm"] != 80
            or config["geometry"] != {"length_nm": 16650, "width_nm": 1000, "thickness_nm": 200,
                                      "pattern": {"csv": "data/geometry/hole_geometry_derived_AP_pnc.csv",
                                                  "x_offset_nm": 8325}}
            or config["analysis"] != {"kind": "scattering", "boundary": "matched_ports", "frequency_GHz": 1,
                                      "incident_side": "left"}):
        raise ValueError("The archived four-layer geometry or analysis differs.")
    config["analysis"]["frequency_GHz"] = FREQUENCY_HZ / 1e9
    config["resources"] = {"max_dofs": BODY_DOFS, "max_port_dofs": PORT_DOFS, "linear_solver": "direct"}
    return {"body_stage": receipt, "body_assembly_source": body["source"], "configuration": config,
            "mesh_directory": str(mesh_dir.resolve()), "mesh_hash": body["mesh_hash"],
            "mesh": paths["mesh"], "mesh_summary": paths["mesh_summary"],
            "array_files": {key: paths[key] for key in ("K", "M")}, "archived_safe_modes": paths["port_modes"],
            "assembly_resources": body["stages"]["production_batched_assembly"],
            "displacement_dofs": BODY_DOFS, "reference_planes_m": list(low_band.END_PLANES_M)}


def restore_mesh(staged):
    mesh_dir = Path(staged["mesh_directory"])
    if (thickness.read_link(staged["mesh"]) != mesh_dir / "mesh_AP_pnc.npz"
            or thickness.read_link(staged["mesh_summary"]) != mesh_dir / "summary_AP_pnc.json"):
        raise ValueError("The archived mesh and checkpoint are in different directories.")
    mesh, mesh_record = workflow.load_checkpoint(mesh_dir, staged["mesh_hash"])
    layout = fixed_port.displacement_layout(mesh)
    if (layout.N != BODY_DOFS or mesh_record["hole_count"] != 26 or mesh_record["mesh_tetrahedra"] != 87312
            or mesh_record["end_face_dofs"] != {"xleft": PORT_DOFS, "xright": PORT_DOFS}):
        raise ValueError("The archived four-layer displacement layout differs.")
    return mesh, layout, mesh_record


def read_section(section_link, modes_link):
    with np.load(thickness.read_link(section_link), allow_pickle=False) as arrays:
        mesh = MeshTri(np.ascontiguousarray(arrays["section_vertices_yz_m"].T),
                       np.ascontiguousarray(arrays["section_triangles"].T))
        basis = Basis(mesh, ElementVector(ElementTriP2(), dim=3), intorder=4)
        if (basis.N != PORT_DOFS or not np.array_equal(basis.doflocs.T, arrays["coordinates_yz_m"])
                or not np.array_equal(basis.split_indices(), arrays["component_dofs_xyz"])
                or not np.array_equal(mesh.p.T, arrays["section_vertices_yz_m"])
                or not np.array_equal(mesh.t.T, arrays["section_triangles"])):
            raise ValueError("The saved 12x4 section basis or coordinate/component order differs.")
        matrices = {key: arrays[name] for key, name in (("K0", "K0"), ("M", "mass_kg_per_m"), ("Q", "Q"), ("G", "G"))}
        if any(value.shape != (PORT_DOFS, PORT_DOFS) or value.dtype != np.float64 or not np.isfinite(value).all()
               for value in matrices.values()):
            raise ValueError("A saved section operator differs from the real 675-DOF space.")
        speed = float(arrays["reference_speed_m_s"])
        with np.load(thickness.read_link(modes_link), allow_pickle=False) as archived:
            port = PortSystem(None, basis, archived["left_dofs"], archived["right_dofs"], **matrices,
                              reference_speed_m_s=speed)
            anchor = inventory.read_modes(archived, port)
        aliases = {"wavenumbers_rad_m": "k_rad_m", "powers_w": "signed_power_w"}
        for field in inventory.MODE_FIELDS:
            if field == "frequency_hz":
                continue
            if not np.array_equal(getattr(anchor, field), arrays[aliases.get(field, field)]):
                raise ValueError(f"The section's archived 1 GHz SAFE array differs: {field}")
    if (anchor.frequency_hz != 1e9 or speed != np.sqrt(128500000000.0 / 5323)
            or workflow.digest([port.left_dofs.tolist(), port.right_dofs.tolist()]) != TRACE_SHA):
        raise ValueError("The archived section speed, trace order or anchor frequency differs.")
    return port, anchor


def verify_trace(port, mesh, layout):
    body_components = fixed_port.components(layout)
    port_components = fixed_port.components(port.basis)
    for saved, tagged, plane in zip((port.left_dofs, port.right_dofs), fixed_port.port_trace_ids(mesh, layout),
                                    low_band.END_PLANES_M, strict=True):
        if (saved.shape != (PORT_DOFS,) or not np.array_equal(np.sort(saved), np.sort(tagged))
                or not np.allclose(layout.doflocs[0, saved], plane, rtol=0, atol=1e-15)
                or not np.array_equal(layout.doflocs[1:, saved], port.basis.doflocs)
                or not np.array_equal(body_components[saved], port_components)):
            raise ValueError("The archived SAFE section does not match the complete tagged body traces.")


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
        accepted = inventory.validate_modes(modes)
    roots = [ids[modes.propagating[ids]].tolist() for ids in (modes.positive_indices, modes.negative_indices)]
    if [len(side) for side in roots] != [4, 4]:
        raise ValueError("The 0.1 GHz z4 section has an unexpected propagating channel count.")
    path = args.output / "modes-100000000_AP_pnc.npz"
    with path.open("xb") as stream:
        np.savez_compressed(stream, **{name: getattr(modes, name) for name in inventory.MODE_FIELDS})
    record.update(status="safe_ready", safe_modes=artifact(path), incident_port_roots=roots[0], backward_port_roots=roots[1],
                  safe_eigensolve_performed=True, resources=measured, **accepted)


def read_phase(directory, name, digest, status, record):
    previous, link = thickness.read_phase(directory, name, digest, status, record)
    if (previous["superlu_library"] != record["superlu_library"]
            or previous["reviewed_sources"] != record["reviewed_sources"]):
        raise ValueError("A prior phase uses a different reviewed source or numerical build.")
    return previous, link


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
            or [len(side) for side in roots] != [4, 4]):
        raise ValueError("The prepared modes differ from the accepted 0.1 GHz section.")
    return system, modes, mesh_record


def factor_index_exposure(operator, reference):
    result = low_band.factor_index_exposure(operator, reference, lawrence.NNZ)
    result.update(archived_factor_job_id="1335030", archived_aggregate_factor_nonzeros=PRIOR_FACTOR_NNZ,
                  archived_aggregate_report_headroom=low_band.INDEX_MAX - PRIOR_FACTOR_NNZ,
                  archived_aggregate_report_headroom_fraction=(low_band.INDEX_MAX - PRIOR_FACTOR_NNZ) / low_band.INDEX_MAX)
    return result


def stage(args, record):
    ready, safe_link = read_phase(args.safe_dir, "safe", args.safe_sha256, "safe_ready", record)
    record.update(**{key: ready[key] for key in ("configuration", "body_stage", "body_assembly_source", "mesh_directory",
                  "mesh_hash", "mesh", "mesh_summary", "array_files", "archived_safe_modes", "assembly_resources",
                  "displacement_dofs", "safe_section", "safe_modes", "section_receipt", "trace_sha256",
                  "safe_section_array_sha256", "frequency_hz", "incident_port_roots", "backward_port_roots",
                  "open_channels", "cutoff_status", "degeneracy_status", "reference_planes_m")},
                  safe_receipt=safe_link, status="constructing_frequency_equations", factorization_performed=False,
                  mesh_generated=False, body_assembly_performed=False, section_assembly_performed=False,
                  safe_eigensolve_performed=False, phase_adjustment_applied=False, **low_band.EXPLORATORY)
    record["workspace_admission"] = low_band.admit_workspace(args.output, 2 * lawrence.GIB)
    archive_link(args.reference_operator, {"bytes": lawrence.ARTIFACTS["A_AP_pnc.npz"][0],
                                          "sha256": lawrence.ARTIFACTS["A_AP_pnc.npz"][1]})
    system, modes, _ = restore_system(record)

    def capture(operator, rhs):
        thickness.validate_equations(operator, rhs, record)
        if not np.isfinite(operator.data).all() or not np.isfinite(rhs).all():
            raise ValueError("The frequency-specific equations contain nonfinite entries.")
        exposure = factor_index_exposure(operator, args.reference_operator)
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
    exposure = staged["factor_index_exposure"]
    if (staged["displacement_dofs"] != BODY_DOFS or staged["frequency_hz"] != FREQUENCY_HZ
            or staged["factorization_performed"] is not False
            or len(staged["incident_port_roots"]) != 4 or len(staged["backward_port_roots"]) != 4
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
    provenance = {**low_band.EXPLORATORY, "method": thickness.METHOD, "frequency_equation_preparation": preparation,
                  "linux_solve": solving, "linux_export": exporting, "prepared_frequency_stage": stage_link,
                  "solve_receipt": solve_link, "mesh_generated": False, "body_assembly_performed": False,
                  "section_assembly_performed": False, "safe_eigensolve_performed": False, "phase_adjustment_applied": False,
                  **{key: staged[key] for key in ("body_stage", "body_assembly_source", "safe_receipt", "section_receipt",
                      "open_channels", "safe_section", "safe_modes", "trace_sha256", "reference_planes_m",
                      "cutoff_status", "degeneracy_status")}}
    prepared = {"mesh_directory": Path(staged["mesh_directory"]), "mesh_checkpoint": (system.mesh, mesh_record), "system": system,
                "pattern": mesh_record["hole_pattern_provenance"], "assembly_resources": staged["assembly_resources"],
                "port_modes": modes, "solver_report": report, "linear_solver": restore,
                "attempt": workflow.digest(provenance), "provenance": provenance}
    result = workflow.run_study(staged["configuration"], args.output, _prepared_scattering=prepared)
    if (result["status"] != "completed_unverified" or result["power_balance_passed"] is not True
            or result["frequency_hz"] != FREQUENCY_HZ or result["hole_count"] != 26
            or result["incident_mode_indices"] != roots or len(report["channels"]) != len(roots)):
        raise ValueError(f"The z4 point failed original-equation or physical validation: {result['output_directory']}")
    public = Path(result["output_directory"])
    record.update(status="completed_unverified", stage=stage_link, solve=solve_link, **low_band.EXPLORATORY,
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
