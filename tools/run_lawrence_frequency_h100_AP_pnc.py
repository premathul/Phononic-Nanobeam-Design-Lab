"""Reuse the archived h100 body and accepted 887.5 MHz ports. Athul Prem | _AP_pnc.

Only the selected inputs and execution envelope change. An isolated copy of the
frozen h100 adapter retains its equation capture, direct solve and cached export.
Normally imported modules and every original archive receipt remain unchanged.
"""
from __future__ import annotations

import argparse
import importlib.util
import json
import os
from pathlib import Path
import re
import socket
import subprocess
import sys
from types import SimpleNamespace

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT / "src"), str(ROOT)]

from phononic import workflow_AP_pnc as workflow
from tools import run_lawrence_low_band_z4_AP_pnc as z4
from tools.run_lawrence_uniform_h80_AP_pnc import artifact

lawrence, thickness, equations, low_band = z4.lawrence, z4.thickness, z4.equations, z4.low_band
FREQUENCY_HZ, BODY_DOFS, PORT_DOFS = 887_500_000, 307611, 675
ROOTS, BACKWARD_ROOTS = (1302, 1346, 1292, 1286, 1282), (1303, 1347, 1293, 1287, 1283)
HELPER_PATH = ROOT / "tools/run_lawrence_low_band_h100_AP_pnc.py"
HELPER_SHA = "cf8086c56176ca0719e9b38f04e89188021c09419ae2b770963a7742af7fc4c0"
BODY_STAGE_SHA = "48fab07dff1f8f79fa76b76ace8aac4246019c68666e03f4dd2fe755de8cbd9f"
MESH_SUMMARY = {"sha256": "b295d82921ffd21c58ee8824b4ccd7c24831e967f8881e87fd0c8dd41e79c9c6", "bytes": 9558}
TRACE_SHA = "51c54331436318f92a0a2b329acabe40606bb74e6b0e8ee0db7a88ce776c3905"
PROVIDER_TRACE_SHA = "2752f71b459282bb33610b39ce99fd25eee6784a510e4c6013415a9357e6fd7b"
SAFE_SHA = "7d3dda2761e3f25d2f6470259cd18306688e2e9ce7afc5f2d9c9ee92ea5eae20"
MODES_SHA = "a2cf1204490983c766d84292bfc85a0182da9997e56fbf2e7281251e11864b32"
PROVIDER_RUNNER_SHA = "f849c02a536dd3d615d26a26ac1eb189bb0bd607c9f0ccda87cf79e8b86524de"
PREBODY_SHA = "f45245144be6035c72ffee5ced2d96aa349b939bde89f9bb36f6ca82550bdca4"
REFERENCE_POINT_SHA = "17bcad69b7d567331489d32a798f6755df292272473df78e9338ba259683fd34"
RELATION_SHA = "5ee5c8006803ee34ae7ef554588fdfa1f887c065c032c924cb79a6cc0558b9b4"
REFERENCE_OPERATOR = {"sha256": "ff752c271bcfc9031cf22e63c05ce35569c24777b7c34d313bce5b3736a021c7", "bytes": 470825885}
OPERATOR_NNZ = 23479709
OPERATOR_GRAPH_SHA = "0a6f3c06ed58f6e5c630d14d7f19730da4bdf80afe5db4a3866a13d190f34197"
RESOURCE_CEILINGS = {"stage": {"memory_gib": 16, "allocation_seconds": 600, "worker_seconds": 550},
                     "solve": {"memory_gib": 48, "allocation_seconds": 900, "worker_seconds": 850},
                     "export": {"memory_gib": 16, "allocation_seconds": 600, "worker_seconds": 550}}
EXTRA_KEYS = ("reused_execution_helper", "accepted_safe_prebody", "accepted_reference_point", "signed_relation",
              "safe_verification", "safe_provider_trace_sha256", "safe_provider_body_stage", "reference_operator")


def same_artifact(first, second):
    return all(first.get(key) == second.get(key) for key in ("bytes", "sha256"))


def frozen_helper():
    link = artifact(HELPER_PATH)
    if link["sha256"] != HELPER_SHA:
        raise ValueError("The frozen h100 execution helper changed.")
    return link


def validate_provider(safe, safe_link, proof, proof_link, accepted):
    if (safe["status"] != "safe_ready" or safe["driver"]["sha256"] != PROVIDER_RUNNER_SHA
            or safe["slurm_job_id"] != "1335641" or safe["frequency_hz"] != FREQUENCY_HZ
            or safe["displacement_dofs"] != 426627 or safe["discrete_roots"] != 2 * PORT_DOFS
            or safe["positive_open_channels"] != 5 or safe["negative_open_channels"] != 5
            or tuple(safe["incident_port_roots"]) != ROOTS or tuple(safe["backward_port_roots"]) != BACKWARD_ROOTS
            or safe["trace_sha256"] != PROVIDER_TRACE_SHA
            or safe["safe_modes"]["sha256"] != MODES_SHA
            or safe["safe_section"]["sha256"] != z4.SECTION_SHA
            or safe["section_receipt"]["sha256"] != z4.SECTION_RECEIPT_SHA
            or safe["body_stage"]["sha256"] != z4.BODY_STAGE_SHA
            or safe["cutoff_status"] != "solver_resolved_at_sample" or safe["degeneracy_status"] != "none_detected"
            or safe["coincident_propagating_root_groups"] or safe["reference_planes_m"] != list(low_band.END_PLANES_M)):
        raise ValueError("The accepted 887.5 MHz provider frequency, signed basis or section differs.")
    if any(safe[key] is not False for key in ("body_assembly_performed", "section_assembly_performed", "mesh_generated",
                                              "anchor_eigensolve_performed", "factorization_performed")):
        raise ValueError("The accepted SAFE provider changed the archived body or section.")
    if (proof["status"] != "independently_verified_safe" or proof["frequency_hz"] != FREQUENCY_HZ
            or proof["problem_dofs"] != 426627 or proof["port_dofs"] != PORT_DOFS
            or proof["all_discrete_roots_replayed"] != 2 * PORT_DOFS
            or proof["positive_open_channels"] != 5 or proof["negative_open_channels"] != 5
            or tuple(proof["incident_port_roots"]) != ROOTS or tuple(proof["backward_port_roots"]) != BACKWARD_ROOTS
            or not same_artifact(proof["safe_receipt"], safe_link)
            or not same_artifact(proof["safe_modes"], safe["safe_modes"])
            or proof["relation_status"] != "matched" or proof["assigned_root_square_consistent"] is not True
            or proof["relation"]["sha256"] != RELATION_SHA):
        raise ValueError("The independent full SAFE and signed-relation acceptance differs.")
    if (accepted["status"] != "point_verified_within_saved_discretization"
            or accepted["frequency_hz"] != FREQUENCY_HZ or accepted["problem_dofs"] != 426627
            or accepted["port_dofs"] != PORT_DOFS or accepted["reference_planes_m"] != list(low_band.END_PLANES_M)
            or not same_artifact(accepted["phase_receipts"]["safe"], safe_link)
            or not same_artifact(accepted["prebody_verification"], proof_link)
            or accepted["coherent_left_incidence"]["incident_roots"] != list(ROOTS)
            or accepted["coherent_left_incidence"]["reflected_roots"] != list(BACKWARD_ROOTS)):
        raise ValueError("The independently accepted h80 reference point differs from the shared SAFE basis.")


def read_acceptance(safe_path, proof_path, accepted_path):
    safe, safe_link = equations.read_receipt(safe_path, SAFE_SHA)
    proof, proof_link = equations.read_receipt(proof_path, PREBODY_SHA)
    accepted, accepted_link = equations.read_receipt(accepted_path, REFERENCE_POINT_SHA)
    validate_provider(safe, safe_link, proof, proof_link, accepted)
    return safe, safe_link, proof, proof_link, accepted_link


def read_inputs(args, record):
    body, body_link = equations.read_receipt(args.body_stage_dir / "stages_AP_pnc.json", BODY_STAGE_SHA)
    core = frozen_core()
    compatibility = core.verify_body_sources(body["source"], record["implementation"])
    if (body["status"] != "operator_ready" or body["factorization_attempted"] is not False
            or body["mesh_comparison"]["ndof"] != BODY_DOFS):
        raise ValueError("The archived h100 body differs from its exact original stage.")
    inputs = {key: z4.archive_link(args.body_stage_dir / Path(body[key]["path"]).name, body[key])
              for key in ("configuration", "K", "M", "port_modes")}
    mesh = z4.archive_link(args.mesh_dir / "mesh_AP_pnc.npz", body["mesh"])
    mesh_summary = z4.archive_link(args.mesh_dir / "summary_AP_pnc.json", MESH_SUMMARY)
    reference_operator = z4.archive_link(args.reference_operator, REFERENCE_OPERATOR)
    if not same_artifact(reference_operator, body["A"]) or body["A"]["nnz"] != OPERATOR_NNZ:
        raise ValueError("The original h100 1 GHz operator identity differs.")
    safe, safe_link, proof, proof_link, accepted_link = read_acceptance(
        args.safe_dir / "safe_AP_pnc.json", args.safe_verification, args.reference_point)
    if any(safe["implementation"][key] != record["implementation"][key] for key in low_band.prep.NUMERICAL_KEYS):
        raise ValueError("The accepted SAFE provider and current numerical identities differ.")
    config = json.loads(Path(inputs["configuration"]["path"]).read_bytes())
    shared = safe["configuration"]
    if (any(config[key] != shared[key] for key in ("geometry", "material"))
            or {**config["mesh"], "mesh_size_nm": 80} != shared["mesh"]
            or config["mesh"]["mesh_size_nm"] != 100
            or config["analysis"] != {**shared["analysis"], "frequency_GHz": 1}):
        raise ValueError("The h100 body and accepted h80 point differ in physical inputs or section settings.")
    config["analysis"]["frequency_GHz"] = FREQUENCY_HZ / 1e9
    config["resources"] = {"max_dofs": BODY_DOFS, "max_port_dofs": PORT_DOFS, "linear_solver": "direct",
                           "assembly_batch_elements": config["resources"]["assembly_batch_elements"]}
    modes = z4.archive_link(args.safe_dir / f"modes-{FREQUENCY_HZ}_AP_pnc.npz", safe["safe_modes"])
    for key in ("safe_section", "section_receipt", "archived_safe_modes"):
        thickness.read_link(safe[key])
    return {"configuration": config, "body_stage": body_link, "body_assembly_source": body["source"],
            "body_source_compatibility": compatibility, "mesh_directory": str(args.mesh_dir.resolve()),
            "mesh_hash": body["mesh_hash"], "mesh": mesh, "mesh_summary": mesh_summary,
            "array_files": {key: inputs[key] for key in ("K", "M")}, "archived_safe_modes": inputs["port_modes"],
            "assembly_resources": body["stages"]["production_batched_assembly"], "displacement_dofs": BODY_DOFS,
            "safe_receipt": safe_link, "safe_modes": modes, "safe_section": safe["safe_section"],
            "section_receipt": safe["section_receipt"], "safe_reference_ports": safe["archived_safe_modes"],
            "trace_sha256": TRACE_SHA, "safe_provider_trace_sha256": PROVIDER_TRACE_SHA,
            "frequency_hz": FREQUENCY_HZ, "incident_port_roots": list(ROOTS), "backward_port_roots": list(BACKWARD_ROOTS),
            "open_channels": safe["open_channels"], "cutoff_status": safe["cutoff_status"],
            "degeneracy_status": safe["degeneracy_status"], "reference_planes_m": list(low_band.END_PLANES_M),
            "safe_provider_body_stage": safe["body_stage"], "reference_operator": reference_operator,
            "reused_execution_helper": frozen_helper(), "accepted_safe_prebody": proof_link,
            "safe_verification": proof_link, "accepted_reference_point": accepted_link,
            "signed_relation": proof["relation"]}, body["A"]


def validate_stage(staged):
    if (staged["body_stage"]["sha256"] != BODY_STAGE_SHA or staged["reused_execution_helper"] != frozen_helper()
            or staged["safe_provider_trace_sha256"] != PROVIDER_TRACE_SHA
            or staged["phase_adjustment_applied"] is not False
            or staged["reference_planes_m"] != list(low_band.END_PLANES_M)
            or not same_artifact(staged["reference_operator"], REFERENCE_OPERATOR)
            or staged["operator_nonzeros"] != OPERATOR_NNZ
            or staged["factor_index_exposure"]["operator_graph_sha256"] != OPERATOR_GRAPH_SHA
            or staged["factor_index_exposure"]["matches_archived_1_ghz_csc_graph"] is not True):
        raise ValueError("The prepared h100 point lost its exact archive, phase or graph acceptance.")
    safe, safe_link, proof, proof_link, point_link = read_acceptance(
        Path(staged["safe_receipt"]["path"]), Path(staged["accepted_safe_prebody"]["path"]),
        Path(staged["accepted_reference_point"]["path"]))
    if (not same_artifact(staged["safe_receipt"], safe_link)
            or not same_artifact(staged["safe_modes"], safe["safe_modes"])
            or not same_artifact(staged["safe_section"], safe["safe_section"])
            or not same_artifact(staged["section_receipt"], safe["section_receipt"])
            or not same_artifact(staged["safe_reference_ports"], safe["archived_safe_modes"])
            or staged["signed_relation"] != proof["relation"]
            or staged["safe_verification"] != proof_link or staged["accepted_safe_prebody"] != proof_link
            or staged["accepted_reference_point"] != point_link):
        raise ValueError("The prepared point differs from its independently accepted shared-port evidence.")


def frozen_core():
    frozen_helper()
    spec = importlib.util.spec_from_file_location("_frequency_h100_frozen_AP_pnc", HELPER_PATH)
    core = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(core)
    return core


def execution_core():
    core = frozen_core()
    core.FREQUENCY_HZ, core.ROOTS, core.BACKWARD_ROOTS = FREQUENCY_HZ, ROOTS, BACKWARD_ROOTS
    core.read_inputs = read_inputs
    original_index, original_load = core.factor_index_exposure, core.load_stage

    def exact_index(operator, reference, reference_nnz):
        exposure = original_index(operator, reference, reference_nnz)
        if (operator.nnz != OPERATOR_NNZ or exposure["operator_graph_sha256"] != OPERATOR_GRAPH_SHA
                or exposure["matches_archived_1_ghz_csc_graph"] is not True):
            raise ValueError("The prepared frequency operator graph differs from the exact h100 archive.")
        return exposure

    def accepted_stage(args, record):
        staged, link = original_load(args, record)
        validate_stage(staged)
        record.update({key: staged[key] for key in EXTRA_KEYS})
        record.update(phase_adjustment_applied=False, reference_planes_m=list(low_band.END_PLANES_M))
        return staged, link

    def cached_study(configuration, output, *, _prepared_scattering):
        prepared = dict(_prepared_scattering)
        provenance = dict(prepared["provenance"])
        stage = equations.read_receipt(Path(provenance["prepared_frequency_stage"]["path"]),
                                       provenance["prepared_frequency_stage"]["sha256"])[0]
        provenance.update({key: stage[key] for key in EXTRA_KEYS})
        provenance.update(phase_adjustment_applied=False, reference_planes_m=list(low_band.END_PLANES_M))
        prepared["provenance"] = provenance
        prepared["attempt"] = workflow.digest(provenance)
        return workflow.run_study(configuration, output, _prepared_scattering=prepared)

    core.factor_index_exposure, core.load_stage = exact_index, accepted_stage
    # The proxy belongs only to this isolated execution namespace.
    core.workflow = SimpleNamespace(**{**vars(workflow), "run_study": cached_study})
    return core


def allocation_seconds(job):
    result = subprocess.run(["scontrol", "show", "job", job, "-o"], capture_output=True, text=True,
                            check=True, timeout=10)
    match = re.search(r"\bTimeLimit=(?:(\d+)-)?(\d+):(\d+):(\d+)\b", result.stdout)
    if match is None:
        raise ValueError("The Slurm allocation has no finite, readable time limit.")
    days, hours, minutes, seconds = (int(value or 0) for value in match.groups())
    return days * 86400 + hours * 3600 + minutes * 60 + seconds


def run_phase(args, operation):
    identity = workflow.source_identity()
    job = os.environ.get("SLURM_JOB_ID")
    threads = {name: os.environ.get(name) for name in lawrence.THREADS}
    ceiling = RESOURCE_CEILINGS[args.mode]
    quota = int(os.environ.get("SLURM_MEM_PER_NODE", "0")) * lawrence.MIB
    worker = int(os.environ.get("AP_PNC_WORKER_SECONDS", "0"))
    if (sys.platform != "linux" or not job or threads != lawrence.THREADS
            or workflow.digest(identity["source_sha256"]) != z4.fixed_port.SOURCE_MAP_SHA256
            or identity["packages"] != z4.fixed_port.PACKAGES
            or int(os.environ.get("SLURM_CPUS_PER_TASK", "0")) != 2
            or not 0 < quota <= ceiling["memory_gib"] * lawrence.GIB
            or not 0 < worker <= ceiling["worker_seconds"]):
        raise ValueError("Use the reviewed single-threaded Linux Slurm resource ceilings.")
    seconds = allocation_seconds(job)
    controls = lawrence.cgroup_controls()
    if (not 0 < seconds <= ceiling["allocation_seconds"] or worker >= seconds
            or not any(cap is not None and 0 < cap <= quota and lawrence.is_job_group(directory)
                       for directory, cap, *_ in controls)):
        raise ValueError("The bounded Slurm allocation or job memory cgroup is unavailable.")
    core = frozen_core()
    helper = frozen_helper()
    sources = {**core.checked_sources(), "frozen_h100_execution": helper}
    snapshot = lawrence.memory_snapshot(quota, controls)
    if reason := lawrence.memory_block(snapshot):
        raise RuntimeError(reason)
    record = {**z4.fixed_port.AUTHOR, "mode": args.mode, "status": "running", "implementation": identity,
              "driver": artifact(Path(__file__)), "reused_execution_helper": helper, "hostname": socket.gethostname(),
              "slurm_job_id": job, "numerical_thread_environment": threads,
              "numerical_backend": lawrence.solve_environment(), "reviewed_sources": sources,
              "superlu_library": artifact(Path(core._superlu.__file__)), "resource_ceiling": ceiling,
              "allocation_limit_seconds": seconds, "worker_limit_seconds": worker,
              "initial_memory_boundary": snapshot, "automatic_retry": False, "automatic_allocation_escalation": False}
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


def stage(args, record):
    execution_core().stage(args, record)


def solve(args, record):
    execution_core().solve(args, record)


def export(args, record):
    execution_core().export(args, record)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    phases = parser.add_subparsers(dest="mode", required=True)
    for mode, directories, receipts in (("stage", ("body-stage", "mesh", "safe"), ()),
                                       ("solve", ("stage",), ("stage",)),
                                       ("export", ("stage", "solve"), ("stage", "solve"))):
        phase = phases.add_parser(mode)
        phase.add_argument("--output", type=Path, required=True)
        for name in directories:
            phase.add_argument(f"--{name}-dir", type=Path, required=True)
        for name in receipts:
            phase.add_argument(f"--{name}-sha256", required=True)
        if mode == "stage":
            for name in ("reference-operator", "safe-verification", "reference-point"):
                phase.add_argument(f"--{name}", type=Path, required=True)
        if mode == "solve":
            phase.add_argument("--prior-solve-dir", type=Path)
            phase.add_argument("--prior-solve-sha256")
    args = parser.parse_args()
    if args.mode == "solve" and (args.prior_solve_dir is None) != (args.prior_solve_sha256 is None):
        parser.error("Accepted-prefix recovery requires both the prior directory and receipt checksum.")
    args.output = args.output.resolve()
    for name, value in vars(args).items():
        if isinstance(value, Path) and name != "output":
            target = value.resolve()
            archive = target if name.endswith("_dir") else target.parent
            if args.output.is_relative_to(archive):
                parser.error("The output must be outside every archived input directory.")
    if args.output.exists() and any(path.name not in ("allocation_AP_pnc.txt", "time_AP_pnc.txt")
                                    for path in args.output.iterdir()):
        parser.error("Use a fresh phase directory, preserving every prior attempt.")
    run_phase(args, {"stage": stage, "solve": solve, "export": export}[args.mode])


if __name__ == "__main__":
    main()
