"""Independent five-drive h100 replay at 887.5 MHz. Athul Prem | _AP_pnc.

Consumes the exact accepted h80 SAFE/relation evidence without repeating it.
Checks the distinct h100 body trace against the shared section's coordinates.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import sys
from time import monotonic

THREADS = dict.fromkeys(("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS",
                         "BLIS_NUM_THREADS", "NUMEXPR_NUM_THREADS"), "1")
if __name__ == "__main__":
    os.environ.update(THREADS)

import numpy as np
import psutil
from scipy.sparse import load_npz

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from tools import replay_lawrence_device_point_AP_pnc as shared
from tools import replay_lawrence_h100_control_AP_pnc as h100
from tools import replay_lawrence_z4_control_AP_pnc as z4

FREQUENCY, DOFS, PORT_DOFS, CHANNELS = 887_500_000, 307611, 675, 5
INCIDENT, RETURNING = (1302, 1346, 1292, 1286, 1282), (1303, 1347, 1293, 1287, 1283)
BODY_SHA, MESH_SUMMARY_SHA, TRACE_SHA = h100.STAGE_SHA, h100.MESH_SUMMARY_SHA, h100.TRACE_SHA
SAFE_SHA = "7d3dda2761e3f25d2f6470259cd18306688e2e9ce7afc5f2d9c9ee92ea5eae20"
MODES_SHA = "a2cf1204490983c766d84292bfc85a0182da9997e56fbf2e7281251e11864b32"
PREBODY_SHA = "f45245144be6035c72ffee5ced2d96aa349b939bde89f9bb36f6ca82550bdca4"
POINT_SHA = "17bcad69b7d567331489d32a798f6755df292272473df78e9338ba259683fd34"
RELATION_SHA = "5ee5c8006803ee34ae7ef554588fdfa1f887c065c032c924cb79a6cc0558b9b4"
SAFE_RUNNER_SHA = "f849c02a536dd3d615d26a26ac1eb189bb0bd607c9f0ccda87cf79e8b86524de"
EXECUTION_HELPER_SHA = "cf8086c56176ca0719e9b38f04e89188021c09419ae2b770963a7742af7fc4c0"
PREBODY_PRODUCER_SHA = "34e2307663a9ea2c3cb864562a58559a6b94c8710219ea17fd305013e0110572"
POINT_PRODUCER_SHA = "6eb0599a919eb3d1de7f99bed9a7e9cc452e6d2c6becf8b45379ae173e6afdbf"
HELPERS = ((shared, "83c98faae6f07071394fc472a1f44d96897258e09b6266db60402da63e06a6e0"),
           (h100, "3d658fa7a17771f836eae02df9b61e246f74292295c980d549dd12fd8f650b5e"),
           (z4, "cf2c4f441d23f9b772c4595c118f3bc3d88de14ea930422441f89c7515fc8ce7"))
EXECUTION_KEYS = ("implementation", "driver", "hostname", "slurm_job_id", "numerical_thread_environment", "numerical_backend")
CONTEXT_KEYS = ("accepted_safe_prebody", "accepted_reference_point", "safe_verification", "signed_relation",
                "safe_provider_trace_sha256", "safe_provider_body_stage", "reused_execution_helper", "reference_operator")
require = shared.require


def read(path):
    return json.loads(Path(path).read_bytes())


def artifact(path):
    path = Path(path)
    return {"path": str(path.resolve()), "bytes": path.stat().st_size, "sha256": shared.file_sha(path)}


def valid_sha(value):
    return isinstance(value, str) and len(value) == 64 and all(c in "0123456789abcdef" for c in value)


def same_artifact(first, second):
    def name(link):
        return link.get("path", link.get("file", "")).replace("\\", "/").rsplit("/", 1)[-1]
    return name(first) == name(second) and all(first[key] == second[key] for key in ("bytes", "sha256"))


def checked_link(path, link):
    require(same_artifact(artifact(path), link), f"Artifact name, size or SHA-256 differs: {path}")
    return Path(path)


def checked_helpers():
    for module, expected in HELPERS:
        require(shared.file_sha(Path(module.__file__)) == expected, "A frozen independent replay helper changed.")


def frequency_core():
    """Configure a private helper module; leave historical global constants intact."""
    checked_helpers()
    spec = importlib.util.spec_from_file_location("_frequency_h100_replay_AP_pnc", Path(shared.__file__))
    core = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(core)
    core.FREQUENCY, core.OMEGA = FREQUENCY, 2 * np.pi * FREQUENCY
    core.DOFS, core.PORT_DOFS, core.TRACE_SHA = DOFS, PORT_DOFS, TRACE_SHA
    core.INCIDENT, core.RETURNING = INCIDENT, RETURNING
    return core


def check_body(stage, original, config, runner_sha):
    require(valid_sha(runner_sha) and stage["driver"]["sha256"] == runner_sha
            and stage["status"] == "operator_staged" and stage["frequency_hz"] == FREQUENCY
            and stage["displacement_dofs"] == DOFS and stage["trace_sha256"] == TRACE_SHA
            and stage["safe_provider_trace_sha256"] == z4.TRACE_SHA and TRACE_SHA != z4.TRACE_SHA
            and stage["body_stage"]["sha256"] == BODY_SHA
            and stage["body_assembly_source"] == original["source"]
            and stage["body_source_compatibility"]["historical_execution_source_pairs"]
                == {key: list(pair) for key, pair in h100.SOURCE_CHANGES.items()}
            and same_artifact(stage["mesh"], original["mesh"])
            and stage["mesh_summary"]["sha256"] == MESH_SUMMARY_SHA and stage["mesh_summary"]["bytes"] == 9558
            and same_artifact(stage["archived_safe_modes"], original["port_modes"])
            and all(same_artifact(stage["array_files"][key], original[key]) for key in ("K", "M"))
            and stage["reference_planes_m"] == list(shared.PLANES)
            and all(stage[key] is False for key in ("mesh_generated", "body_assembly_performed", "section_assembly_performed",
                    "safe_eigensolve_performed", "factorization_performed", "phase_adjustment_applied")),
            "The h100 point differs from the exact archived body, distinct trace or no-new-SAFE scope.")
    expected = {**config, "analysis": {**config["analysis"], "frequency_GHz": FREQUENCY / 1e9},
                "resources": {"max_dofs": DOFS, "max_port_dofs": PORT_DOFS, "linear_solver": "direct",
                              "assembly_batch_elements": config["resources"]["assembly_batch_elements"]}}
    require(stage["configuration"] == expected, "The h100 frequency point changed archived physical inputs or resources.")
    return h100.verify_sources(original["source"], stage["implementation"])


def check_accepted_chain(stage, safe, prebody, point, relation, prebody_source, point_source):
    """Inspect exact accepted receipts; perform no SAFE or relation numerical replay."""
    require(stage["safe_receipt"]["sha256"] == SAFE_SHA and safe["status"] == "safe_ready"
            and safe["frequency_hz"] == FREQUENCY and safe["trace_sha256"] == z4.TRACE_SHA
            and stage["safe_provider_body_stage"] == safe["body_stage"]
            and safe["driver"]["sha256"] == SAFE_RUNNER_SHA
            and safe["safe_modes"]["sha256"] == MODES_SHA
            and all(same_artifact(stage[key], safe[key]) for key in ("safe_modes", "safe_section", "section_receipt"))
            and stage["safe_section"]["sha256"] == z4.SECTION_SHA
            and stage["section_receipt"]["sha256"] == z4.SECTION_RECEIPT_SHA
            and same_artifact(stage["safe_reference_ports"], safe["archived_safe_modes"])
            and all(stage["implementation"][key] == safe["implementation"][key] for key in ("source_sha256", "packages", "python")),
            "The shared frequency-specific SAFE artifacts, provider trace or numerical sources differ.")
    require(prebody["status"] == "independently_verified_safe" and prebody["frequency_hz"] == FREQUENCY
            and prebody["problem_dofs"] == 426627 and prebody["port_dofs"] == PORT_DOFS
            and prebody["all_discrete_roots_replayed"] == 2 * PORT_DOFS
            and prebody["positive_open_channels"] == prebody["negative_open_channels"] == CHANNELS
            and same_artifact(prebody["safe_receipt"], stage["safe_receipt"])
            and same_artifact(prebody["safe_modes"], stage["safe_modes"])
            and same_artifact(stage["safe_verification"], stage["accepted_safe_prebody"])
            and stage["accepted_safe_prebody"]["sha256"] == PREBODY_SHA
            and same_artifact(prebody["relation"], stage["signed_relation"])
            and stage["signed_relation"]["sha256"] == RELATION_SHA
            and prebody["relation_status"] == "matched"
            and prebody["assigned_root_square_consistent"] is True
            and relation["status"] == "signed_section_relation_accepted" and relation["frequency_hz"] == FREQUENCY
            and prebody["implementation"]["verifier"]["sha256"] == PREBODY_PRODUCER_SHA
            and shared.file_sha(Path(prebody_source)) == PREBODY_PRODUCER_SHA,
            "The exact accepted SAFE/relation receipt or retained verifier source differs.")
    require(stage["accepted_reference_point"]["sha256"] == POINT_SHA
            and point["status"] == "point_verified_within_saved_discretization"
            and point["frequency_hz"] == FREQUENCY and point["problem_dofs"] == 426627
            and same_artifact(point["phase_receipts"]["safe"], stage["safe_receipt"])
            and same_artifact(point["prebody_verification"], stage["accepted_safe_prebody"])
            and same_artifact(point["relation"], stage["signed_relation"])
            and point["implementation"]["verifier"]["sha256"] == POINT_PRODUCER_SHA
            and shared.file_sha(Path(point_source)) == POINT_PRODUCER_SHA,
            "The accepted h80 reference point or exact verifier source differs.")
    require(all(phase["incident_port_roots"] == list(INCIDENT) and phase["backward_port_roots"] == list(RETURNING)
                for phase in (stage, safe, prebody))
            and all(prebody[key] is False for key in ("body_solve_performed", "body_assembly_performed",
                    "spatial_convergence_established", "frequency_convergence_established", "continuous_branch_identity_established")),
            "The accepted signed channel order or finite-scope evidence differs.")
    return {"status": "reused_exact_accepted_safe_and_relation", "safe_eigensolve_performed": False,
            "safe_numerical_replay_performed": False, "relation_numerical_replay_performed": False,
            "prebody_producer": artifact(prebody_source), "point_producer": artifact(point_source)}


def check_phases(stage, solved, exported, runner_sha):
    require(solved["status"] == "drives_accepted" and exported["status"] == "completed_unverified"
            and solved["stage"] == exported["stage"]
            and all(phase["driver"]["sha256"] == runner_sha
                    and phase["implementation"] == stage["implementation"]
                    and phase["reviewed_sources"] == stage["reviewed_sources"]
                    and phase["superlu_library"] == stage["superlu_library"]
                    and phase["reused_execution_helper"]["sha256"] == EXECUTION_HELPER_SHA
                    and phase["numerical_thread_environment"] == THREADS
                    and phase["implementation"]["platform"].startswith("Linux-")
                    and isinstance(phase["slurm_job_id"], str) and phase["slurm_job_id"].isascii()
                    and phase["slurm_job_id"].isdecimal() for phase in (stage, solved, exported))
            and all(phase[key] is False for phase in (stage, exported)
                    for key in ("phase_adjustment_applied", "spatial_convergence_established", "frequency_convergence_established"))
            and exported["reference_planes_m"] == stage["reference_planes_m"] == list(shared.PLANES)
            and all(exported[key] == stage[key] for key in CONTEXT_KEYS),
            "The new h100 phase chain, reviewed wrapper or unchanged original-plane provenance differs.")
    report = solved["linear_solver"]
    count = report["factor"]["factor_nonzeros"]
    require(report["relative_target"] == shared.TARGET
            and report["relative_residual_limit"] == report["backward_residual_limit"] == shared.RESIDUAL
            and report["factor"]["permc_spec"] == "COLAMD" and type(count) is int
            and 2 * DOFS <= count <= np.iinfo(np.int32).max
            and all(report[key] == stage[key] for key in ("operator_csc_sha256", "rhs_array_sha256", "operator_nonzeros"))
            and [row["incident_port_root"] for row in report["channels"]] == list(INCIDENT),
            "The original residual gates, factor-index capacity or five accepted drive identities differ.")
    return report


def check_publication(summary, stage, solved, exported, report):
    """Check the actual cached-export schema, including its nested provenance."""
    provenance = summary["checkpointed_scattering"]
    require(provenance["phase_adjustment_applied"] is False
            and provenance["reference_planes_m"] == list(shared.PLANES)
            and provenance["prepared_frequency_stage"] == exported["stage"]
            and provenance["solve_receipt"] == exported["solve"]
            and all(provenance[key] == stage[key] for key in CONTEXT_KEYS)
            and all(provenance[name] == {key: phase[key] for key in EXECUTION_KEYS}
                    for name, phase in (("frequency_equation_preparation", stage), ("linux_solve", solved), ("linux_export", exported)))
            and summary["status"] == "completed_unverified" and summary["frequency_hz"] == FREQUENCY
            and summary["incident_side"] == "left" and summary["hole_count"] == 26
            and summary["ndof"] == DOFS and summary["port_dofs"] == PORT_DOFS
            and summary["num_incident_channels"] == CHANNELS and summary["incident_mode_indices"] == list(INCIDENT)
            and summary["power_balance_passed"] is True and summary["spatial_convergence_established"] is False
            and summary["configuration"] == stage["configuration"]
            and all(summary["linear_solver"][key] == report[key] for key in ("relative_target", "relative_residual_limit",
                    "backward_residual_limit", "operator_csc_sha256", "rhs_array_sha256", "operator_nonzeros")),
            "The public h100 point differs from its five-drive equations, phase chain or finite scope.")
    return {"stage_phase_adjustment_applied": False, "export_phase_adjustment_applied": False,
            "public_provenance_phase_adjustment_applied": False, "reference_planes_m": list(shared.PLANES),
            "basis": "Explicit unchanged phase and exact hash-linked stage/solve/export execution provenance."}


def load_shared_ports(section_path, mapping_path, reference_ports_path, modes_path):
    with np.load(section_path, allow_pickle=False) as section, np.load(mapping_path, allow_pickle=False) as mapping, \
            np.load(reference_ports_path, allow_pickle=False) as reference:
        values = {key: section[key] for key in ("coordinates_yz_m", "component_dofs_xyz")}
        values.update(left_dofs=mapping["left_dofs"], right_dofs=mapping["right_dofs"])
        require(all(np.array_equal(mapping[key], reference[key]) for key in z4.FIELDS),
                "The h100 and provider archived section fields differ despite their distinct body traces.")
        trace = json.dumps([values["left_dofs"].tolist(), values["right_dofs"].tolist()], separators=(",", ":"))
        require(hashlib.sha256(trace.encode()).hexdigest() == TRACE_SHA,
                "The distinct archived h100 trace order changed.")
    with np.load(modes_path, allow_pickle=False) as saved:
        modes = {key: saved[key] for key in (*z4.FIELDS, "frequency_hz")}
    require(modes["frequency_hz"].item() == FREQUENCY
            and all(np.array_equal(modes[key][modes["propagating"][modes[key]]], roots)
                    for key, roots in (("positive_indices", INCIDENT), ("negative_indices", RETURNING)))
            and modes["displacements"].shape == modes["axial_forces"].shape == (PORT_DOFS, 2 * PORT_DOFS)
            and all(modes[key].dtype == np.complex128 for key in ("displacements", "axial_forces", "wavenumbers_rad_m")),
            "The reused modes differ from the complete accepted 887.5 MHz signed basis.")
    return values, modes


def compare_public_tables(public, blocks, totals):
    with (public / "scattering_channels_AP_pnc.csv").open(newline="", encoding="utf-8") as stream:
        rows = list(csv.DictReader(stream))
    expected = {(kind, incoming, outgoing): (column, row)
                for kind, roots in (("reflection", RETURNING), ("transmission", INCIDENT))
                for column, incoming in enumerate(INCIDENT) for row, outgoing in enumerate(roots)}
    require(len(rows) == len(expected) == 50, "The public table must retain all 50 propagating coefficients.")
    for row in rows:
        key = row["kind"], int(row["incident_port_index"]), int(row["outgoing_port_index"])
        require(key in expected, "A public coefficient is duplicated or has another signed root.")
        column, index = expected.pop(key)
        coefficient, power = blocks[key[0]]
        require(int(row["incident_mode_order"]) == column + 1
                and np.isclose(complex(float(row["amplitude_real"]), float(row["amplitude_imag"])), coefficient[index, column], rtol=1e-8, atol=1e-10)
                and np.isclose(float(row["power_fraction"]), power[index, column], rtol=1e-8, atol=1e-10),
                "A public complex coefficient or power differs from independent replay.")
    with (public / "scattering_AP_pnc.csv").open(newline="", encoding="utf-8") as stream:
        rows = list(csv.DictReader(stream))
    require(len(rows) == CHANNELS, "The public table must retain all five incident drives.")
    r, t, direct_r, direct_t, _ = totals
    metrics = {"reflection": r, "transmission": t, "reflection_flux": direct_r, "transmission_flux": direct_t,
               "power_balance": 1 - r - t, "flux_power_balance": 1 - direct_r - direct_t}
    for column, row in enumerate(rows):
        require(int(row["incident_mode_order"]) == column + 1 and float(row["frequency_GHz"]) == FREQUENCY / 1e9
                and all(np.isclose(float(row[key]), values[column], rtol=1e-8, atol=1e-10) for key, values in metrics.items()),
                "A public frequency or incident-drive power row differs.")


def check_coefficient_layout(fields):
    """Reject broadcasting: every published directional coefficient is checked."""
    for kind in ("reflection", "transmission"):
        require(fields[f"{kind}_amplitudes_full"].shape == (PORT_DOFS, CHANNELS)
                and fields[f"{kind}_amplitudes"].shape == (CHANNELS, CHANNELS)
                and fields[f"{kind}_all_mode_indices"].shape == (PORT_DOFS,),
                "The exported modal arrays do not retain all 50 propagating and 6750 directional coefficients.")


def replay_point(args):
    started = monotonic()
    checked_helpers()
    require(valid_sha(args.runner_sha256) and shared.file_sha(args.runner) == args.runner_sha256,
            "The externally reviewed new wrapper SHA-256 differs.")
    exported = shared.read_receipt(args.export, args.export_sha256)
    stage = read(checked_link(args.stage, exported["stage"]))
    solved = read(checked_link(args.solve, exported["solve"]))
    original = shared.read_receipt(args.body_stage_dir / "stages_AP_pnc.json", BODY_SHA)
    config = read(checked_link(args.body_stage_dir / "configuration_AP_pnc.json", original["configuration"]))
    changes = check_body(stage, original, config, args.runner_sha256)
    safe = read(checked_link(args.safe, stage["safe_receipt"]))
    prebody = shared.read_receipt(checked_link(args.prebody_replay, stage["accepted_safe_prebody"]), PREBODY_SHA)
    point = shared.read_receipt(checked_link(args.accepted_reference_point, stage["accepted_reference_point"]), POINT_SHA)
    relation = shared.read_receipt(checked_link(args.relation, stage["signed_relation"]), RELATION_SHA)
    reuse = check_accepted_chain(stage, safe, prebody, point, relation, args.prebody_source, args.point_source)
    require(shared.file_sha(args.safe) == SAFE_SHA, "The accepted SAFE receipt changed.")
    report = check_phases(stage, solved, exported, args.runner_sha256)
    summary_path = checked_link(args.public_dir / "summary_AP_pnc.json", exported["public_summary"])
    summary = read(summary_path)
    phase_evidence = check_publication(summary, stage, solved, exported, report)
    for name in ("selected_fields_AP_pnc.npz", "selected_vertex_shapes_AP_pnc.vtu", "scattering_AP_pnc.csv",
                 "scattering_channels_AP_pnc.csv", "port_modes_AP_pnc.csv", "configuration_AP_pnc.json"):
        require(shared.file_sha(args.public_dir / name) == summary["output_sha256"][name], f"Public output changed: {name}.")
    require(read(args.public_dir / "configuration_AP_pnc.json") == stage["configuration"], "Public physical inputs differ.")
    body = {key: checked_link(args.body_stage_dir / f"{key}_AP_pnc.npz", original[key]) for key in ("K", "M")}
    arrays = {key: checked_link(args.stage.parent / name, stage["array_files"][key])
              for key, name in (("A", "A_AP_pnc.npz"), ("rhs", "rhs_AP_pnc.npy"))}
    mesh = checked_link(args.mesh_dir / "mesh_AP_pnc.npz", original["mesh"])
    checked_link(args.mesh_dir / "summary_AP_pnc.json", stage["mesh_summary"])
    mapping = checked_link(args.body_stage_dir / "ports_AP_pnc.npz", original["port_modes"])
    checked_link(args.section, stage["safe_section"])
    checked_link(args.section_receipt, stage["section_receipt"])
    checked_link(args.reference_ports, stage["safe_reference_ports"])
    checked_link(args.modes, stage["safe_modes"])
    available = psutil.virtual_memory().available
    required = max(stage["array_files"][key]["bytes"] for key in ("A", "K", "M")) + 3 * DOFS * CHANNELS * 16 + 512 * 2**20
    require(available >= required, f"Independent replay requires {required} B available; measured {available} B.")
    core = frequency_core()
    vectors = np.empty((DOFS, CHANNELS), dtype=np.complex128, order="F")
    for column, row in enumerate(report["channels"]):
        require(row["column"] == column and row["target_met"] is True and row["info"] == 0,
                "A drive lacks its accepted original-equation record.")
        vector = np.load(checked_link(args.solve.parent / f"accepted-{column:03d}_AP_pnc.npy", row["accepted_vector"]), allow_pickle=False)
        require(vector.shape == (DOFS,) and vector.dtype == np.complex128 and np.isfinite(vector).all(), "Invalid accepted displacement vector.")
        vectors[:, column] = vector
    operator, rhs = load_npz(arrays["A"]), np.load(arrays["rhs"], allow_pickle=False, mmap_mode="r")
    require(operator.shape == (DOFS, DOFS) and operator.format == "csc" and operator.has_canonical_format
            and operator.dtype == np.complex128 and operator.indices.dtype == operator.indptr.dtype == np.int32
            and operator.nnz == stage["operator_nonzeros"] and np.isfinite(operator.data).all()
            and rhs.shape == vectors.shape and rhs.dtype == np.complex128 and np.isfinite(rhs).all()
            and shared.array_sha(operator.data, operator.indices, operator.indptr) == stage["operator_csc_sha256"]
            and shared.array_sha(rhs) == stage["rhs_array_sha256"], "The original complex128 CSC equation arrays differ.")
    relative, backward, rhs_norm, matrix_norm = core.check_original_equations(operator, rhs, vectors)
    del operator
    force = core.body_action(body["K"], vectors) - core.OMEGA**2 * core.body_action(body["M"], vectors)
    section, modes = load_shared_ports(args.section, mapping, args.reference_ports, args.modes)
    with np.load(args.public_dir / "selected_fields_AP_pnc.npz", allow_pickle=False) as fields:
        check_coefficient_layout(fields)
        core.replay_fields(fields, mesh, section, modes, vectors)
        blocks, totals, errors, rhs_error, boundary_error, coherent = core.replay_ports(
            modes, fields, section["left_dofs"], section["right_dofs"], vectors, force, rhs, rhs_norm)
    compare_public_tables(args.public_dir, blocks, totals)
    r, t, direct_r, direct_t, net = totals
    return {"author": "Athul Prem", "file_tag": "_AP_pnc", "status": "point_verified_within_saved_discretization",
            "frequency_hz": FREQUENCY, "problem_dofs": DOFS, "port_dofs": PORT_DOFS, "reference_planes_m": list(shared.PLANES),
            "public_summary": artifact(summary_path), "public_summary_sha256": exported["public_summary"]["sha256"],
            "phase_receipts": {key: artifact(getattr(args, key)) for key in ("stage", "solve", "export")},
            "accepted_shared_safe": artifact(args.safe), "accepted_reference_point": artifact(args.accepted_reference_point),
            "prebody_verification": artifact(args.prebody_replay), "prebody_replay": artifact(args.prebody_replay),
            "relation": artifact(args.relation), "accepted_safe_reuse": reuse, "reference_plane_evidence": phase_evidence,
            "source_compatibility": changes, "implementation": {"verifier": artifact(Path(__file__)),
                "runner": artifact(args.runner), "shared_replay": artifact(Path(shared.__file__)),
                "h100_provenance_helper": artifact(Path(h100.__file__))},
            "original_equations_passed": True, "physical_nodal_force_power_passed": True, "all_modal_coefficients_passed": True,
            "full_p2_field_identity_passed": True, "coherent_left_incidence_passed": True,
            "propagating_coefficients_checked": 50, "full_directional_coefficients_checked": 6750,
            "maximum_relative_residual": float(max(relative)), "maximum_backward_residual": float(max(backward)),
            "matrix_infinity_norm": matrix_norm, "relative_target": core.TARGET, "relative_residual_limit": core.RESIDUAL,
            "backward_residual_limit": core.RESIDUAL, "power_limit": core.POWER,
            "maximum_power_errors": dict(zip(core.POWER_CHECKS, errors.tolist(), strict=True)), "coherent_left_incidence": coherent,
            "channels": [{"column": i, "incident_root": root, "reverse_root": RETURNING[i], "relative_residual": relative[i],
                "backward_residual": backward[i], "independent_rhs_relative_error": rhs_error[i],
                "independent_body_boundary_relative_residual": boundary_error[i], "reflection": r[i], "transmission": t[i],
                "reflection_flux": direct_r[i], "transmission_flux": direct_t[i], "net_outward_power_ratio": net[i]}
                for i, root in enumerate(INCIDENT)], "elapsed_seconds": monotonic() - started,
            "available_memory_bytes_before_replay": available, "required_memory_bytes_before_replay": required,
            "safe_eigensolve_performed": False, "safe_numerical_replay_performed": False, "relation_numerical_replay_performed": False,
            "body_solve_performed": False, "body_assembly_performed": False, "spatial_convergence_established": False,
            "frequency_convergence_established": False, "full_two_port_unitarity_established": False, "reciprocity_established": False}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("stage", "solve", "export", "safe", "prebody-replay", "accepted-reference-point", "relation", "prebody-source",
                 "point-source", "runner", "body-stage-dir", "mesh-dir", "public-dir", "section", "section-receipt", "reference-ports", "modes"):
        parser.add_argument(f"--{name}", type=Path, required=True)
    parser.add_argument("--runner-sha256", required=True)
    parser.add_argument("--export-sha256", required=True)
    args = parser.parse_args()
    for name, value in vars(args).items():
        if isinstance(value, Path):
            setattr(args, name, value.resolve())
    result = replay_point(args)
    print(json.dumps(result, indent=2, allow_nan=False, default=lambda value: value.item() if isinstance(value, np.generic) else value))


if __name__ == "__main__":
    main()
