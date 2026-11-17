"""Independently replay the cached 100 MHz h100/h80 control. Athul Prem | _AP_pnc."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path, PurePosixPath
import sys
from time import monotonic

if __name__ == "__main__":
    for key in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "BLIS_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
        os.environ[key] = "1"

import numpy as np
import psutil
from scipy.sparse import load_npz

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from tools import compare_scattering_AP_pnc as comparison
from tools import replay_lawrence_device_point_AP_pnc as replay
from tools import replay_lawrence_z4_control_AP_pnc as z4

ARCHIVE = ROOT / "results/runs/full_device_resource_profile/full_h100_operator_qr_jt4fp"
STAGE_SHA = "48fab07dff1f8f79fa76b76ace8aac4246019c68666e03f4dd2fe755de8cbd9f"
MESH_SUMMARY_SHA = "b295d82921ffd21c58ee8824b4ccd7c24831e967f8881e87fd0c8dd41e79c9c6"
TRACE_SHA = "51c54331436318f92a0a2b329acabe40606bb74e6b0e8ee0db7a88ce776c3905"
H80_SAFE = ROOT / "results/runs/low_band_thickness_AP_pnc/h80-z4-low-band-safe-1335500/safe_AP_pnc.json"
H80_SAFE_SHA = "59907315426ecf8b240ca8c37a40fe6cb8b019f9b79a941ce7fc65065c771cbe"
MODES_SHA = "f290088d3ca25210979444ed011e113e165bb4fd41ec90a04d51693f60261fee"
H80_REPLAY = ROOT / "results/runs/low_band_thickness_AP_pnc/local_checks/independent_replay_comparison_AP_pnc.json"
H80_REPLAY_SHA = "05931896767037407fece43873906c6d469a00fe212cac8dd2dbb24d67c3ef77"
H80_PUBLIC = ROOT / "results/runs/low_band_thickness_AP_pnc/h80-z4-low-band-export-1335512/historical_26hole_layered_h80_ny12_z4-a80dd9c4d431"
RUNNER = ROOT / "tools/run_lawrence_low_band_h100_AP_pnc.py"
RUNNER_SHA = "cf8086c56176ca0719e9b38f04e89188021c09419ae2b770963a7742af7fc4c0"
DOFS, PORT_DOFS, FREQUENCY = 307611, 675, 100_000_000
SOURCE_CHANGES = {
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
        None, "dba6002d982647810f3fc714fdce73a0f34e306e06e8519df2d54778002205ec")}
require = replay.require


def verify_sources(original, current):
    old, new = (source["source_sha256"] for source in (original, current))
    changes = {key: (old.get(key), new.get(key)) for key in old.keys() | new.keys() if old.get(key) != new.get(key)}
    require(changes == SOURCE_CHANGES and original["packages"] == current["packages"]
            and original["platform"].startswith("Windows") and current["platform"].startswith("Linux"),
            "The h100 archive has an unreviewed source change or different numerical packages.")
    return {key: {"archived": first, "current": second} for key, (first, second) in changes.items()}


def check_body(stage, original, configuration):
    require(stage["status"] == "operator_staged" and stage["frequency_hz"] == FREQUENCY
            and stage["displacement_dofs"] == DOFS and stage["trace_sha256"] == TRACE_SHA
            and stage["body_stage"]["sha256"] == STAGE_SHA
            and stage["body_assembly_source"] == original["source"]
            and stage["body_source_compatibility"]["historical_execution_source_pairs"]
                == {key: list(pair) for key, pair in SOURCE_CHANGES.items()}
            and z4.same_artifact(stage["mesh"], original["mesh"])
            and stage["mesh_summary"]["sha256"] == MESH_SUMMARY_SHA
            and all(z4.same_artifact(stage["array_files"][key], original[key]) for key in ("K", "M"))
            and all(stage[key] is False for key in ("mesh_generated", "body_assembly_performed", "section_assembly_performed",
                    "safe_eigensolve_performed", "factorization_performed", "phase_adjustment_applied")),
            "The h100 point does not reuse the exact archived body and accepted frequency ports.")
    require(all(stage["configuration"][key] == configuration[key] for key in ("geometry", "mesh", "material"))
            and stage["configuration"]["analysis"] == {**configuration["analysis"], "frequency_GHz": 0.1},
            "The h100 control changed physical inputs or incidence.")
    return verify_sources(original["source"], stage["implementation"])


def shared_ports(stage, original, safe, accepted):
    require(stage["safe_receipt"]["sha256"] == H80_SAFE_SHA and safe["status"] == "safe_ready"
            and safe["frequency_hz"] == FREQUENCY and safe["safe_modes"]["sha256"] == MODES_SHA
            and z4.same_artifact(stage["safe_modes"], safe["safe_modes"])
            and stage["safe_section"]["sha256"] == z4.SECTION_SHA
            and stage["section_receipt"]["sha256"] == z4.SECTION_RECEIPT_SHA
            and z4.same_artifact(stage["safe_reference_ports"], safe["archived_safe_modes"])
            and z4.same_artifact(stage["archived_safe_modes"], original["port_modes"])
            and accepted["phase_receipts"]["safe"]["sha256"] == H80_SAFE_SHA
            and accepted["status"] == "point_verified_within_saved_discretization",
            "The h100 point does not reference the independently accepted h80 SAFE data.")
    incident, returning = (accepted["coherent_left_incidence"][key] for key in ("incident_roots", "reflected_roots"))
    require(len(incident) == len(returning) == 4 and stage["incident_port_roots"] == safe["incident_port_roots"] == incident
            and stage["backward_port_roots"] == safe["backward_port_roots"] == returning,
            "The shared frequency-specific signed channel identities differ.")
    with np.load(z4.SECTION / "arrays_AP_pnc.npz", allow_pickle=False) as section, \
            np.load(ARCHIVE / "ports_AP_pnc.npz", allow_pickle=False) as mapping:
        values = {key: section[key] for key in ("coordinates_yz_m", "component_dofs_xyz")}
        values.update(left_dofs=mapping["left_dofs"], right_dofs=mapping["right_dofs"])
    modes_path = replay.check_link(H80_SAFE.parent / PurePosixPath(safe["safe_modes"]["path"]).name, safe["safe_modes"])
    with np.load(modes_path, allow_pickle=False) as arrays:
        modes = {key: arrays[key] for key in (*z4.FIELDS, "frequency_hz")}
    require(modes["frequency_hz"].item() == FREQUENCY
            and np.array_equal(modes["positive_indices"][modes["propagating"][modes["positive_indices"]]], incident)
            and np.array_equal(modes["negative_indices"][modes["propagating"][modes["negative_indices"]]], returning),
            "The saved SAFE arrays do not cover the accepted signed channel order.")
    return values, modes, incident, returning


def replay_point(stage_path, solve_path, export_path):
    started = monotonic()
    original = replay.read_receipt(ARCHIVE / "stages_AP_pnc.json", STAGE_SHA)
    configuration = z4.read(replay.check_link(ARCHIVE / "configuration_AP_pnc.json", original["configuration"]))
    safe = replay.read_receipt(H80_SAFE, H80_SAFE_SHA)
    accepted = replay.read_receipt(H80_REPLAY, H80_REPLAY_SHA)
    export = z4.read(export_path)
    stage = replay.read_receipt(replay.check_link(stage_path, export["stage"]), export["stage"]["sha256"])
    solved = replay.read_receipt(replay.check_link(solve_path, export["solve"]), export["solve"]["sha256"])
    changes = check_body(stage, original, configuration)
    require(solved["status"] == "drives_accepted" and export["status"] == "completed_unverified"
            and solved["stage"] == export["stage"]
            and replay.file_sha(RUNNER) == RUNNER_SHA
            and all(phase["driver"]["sha256"] == RUNNER_SHA for phase in (stage, solved, export))
            and all(phase["reviewed_sources"] == stage["reviewed_sources"]
                    and phase["superlu_library"] == stage["superlu_library"] for phase in (solved, export)),
            "The completed h100 phases use different source, build or staged-equation identities.")
    require(all(phase[flag] is False for phase in (stage, export)
                for flag in ("spatial_convergence_established", "frequency_convergence_established")),
            "A finite control claims spatial or frequency convergence.")
    body = {key: replay.check_link(ARCHIVE / f"{key}_AP_pnc.npz", original[key]) for key in ("K", "M")}
    arrays = {key: replay.check_link(stage_path.parent / name, stage["array_files"][key])
              for key, name in (("A", "A_AP_pnc.npz"), ("rhs", "rhs_AP_pnc.npy"))}
    mesh_path = replay.check_link(ROOT / original["mesh"]["path"], original["mesh"])
    replay.read_receipt(mesh_path.parent / "summary_AP_pnc.json", MESH_SUMMARY_SHA)
    replay.check_link(ARCHIVE / "ports_AP_pnc.npz", original["port_modes"])
    require(replay.file_sha(z4.SECTION / "arrays_AP_pnc.npz") == z4.SECTION_SHA, "The saved 12x4 section changed.")
    section, modes, incident, returning = shared_ports(stage, original, safe, accepted)
    public = export_path.parent / PurePosixPath(export["public_summary"]["path"]).parent.name
    summary_path = replay.check_link(public / "summary_AP_pnc.json", export["public_summary"])
    summary = z4.read(summary_path)
    require(summary["status"] == "completed_unverified" and summary["frequency_hz"] == FREQUENCY
            and summary["incident_side"] == "left" and summary["hole_count"] == 26
            and summary["ndof"] == DOFS and summary["port_dofs"] == PORT_DOFS
            and summary["power_balance_passed"] is True and summary["spatial_convergence_established"] is False,
            "The public h100 point scope or power acceptance differs.")
    for name in ("selected_fields_AP_pnc.npz", "selected_vertex_shapes_AP_pnc.vtu", "scattering_AP_pnc.csv",
                 "scattering_channels_AP_pnc.csv", "port_modes_AP_pnc.csv", "configuration_AP_pnc.json"):
        require(replay.file_sha(public / name) == summary["output_sha256"][name], f"Public h100 output changed: {name}.")
    report = solved["linear_solver"]
    require(report["relative_target"] == replay.TARGET
            and report["relative_residual_limit"] == report["backward_residual_limit"] == replay.RESIDUAL
            and report["factor"]["permc_spec"] == "COLAMD"
            and 2 * DOFS <= report["factor"]["factor_nonzeros"] <= np.iinfo(np.int32).max
            and all(report[key] == stage[key] == summary["linear_solver"][key] for key in
                    ("operator_csc_sha256", "rhs_array_sha256", "operator_nonzeros"))
            and summary["configuration"] == z4.read(public / "configuration_AP_pnc.json") == stage["configuration"],
            "The h100 residual gates, equations or public configuration differ.")
    channels = report["channels"]
    require([row["incident_port_root"] for row in channels] == incident, "Four ordered accepted h100 drives are required.")
    vectors = np.empty((DOFS, 4), dtype=np.complex128, order="F")
    for column, row in enumerate(channels):
        require(row["column"] == column and row["target_met"] is True and row["info"] == 0,
                "An h100 drive lacks its original-equation acceptance record.")
        path = replay.check_link(solve_path.parent / f"accepted-{column:03d}_AP_pnc.npy", row["accepted_vector"])
        vector = np.load(path, allow_pickle=False)
        require(vector.shape == (DOFS,) and vector.dtype == np.complex128 and np.isfinite(vector).all(),
                "An accepted h100 displacement vector has invalid dimensions or values.")
        vectors[:, column] = vector
    available = psutil.virtual_memory().available
    required = stage["array_files"]["A"]["bytes"] + 512 * 2**20
    require(available >= required, f"Independent h100 replay requires {required} B available; measured {available} B.")
    operator = load_npz(arrays["A"])
    rhs = np.load(arrays["rhs"], allow_pickle=False, mmap_mode="r")
    require(operator.shape == (DOFS, DOFS) and operator.format == "csc" and operator.dtype == np.complex128
            and operator.indices.dtype == operator.indptr.dtype == np.int32 and operator.has_canonical_format
            and rhs.shape == vectors.shape and rhs.dtype == np.complex128
            and replay.array_sha(operator.data, operator.indices, operator.indptr) == stage["operator_csc_sha256"]
            and replay.array_sha(rhs) == stage["rhs_array_sha256"], "The h100 original equation arrays differ.")
    relative, backward, rhs_norm, matrix_norm = replay.check_original_equations(operator, rhs, vectors)
    del operator
    force = replay.body_action(body["K"], vectors, expected_dofs=DOFS)
    force -= (2 * np.pi * FREQUENCY)**2 * replay.body_action(body["M"], vectors, expected_dofs=DOFS)
    with np.load(public / "selected_fields_AP_pnc.npz", allow_pickle=False) as fields:
        replay.replay_fields(fields, mesh_path, section, modes, vectors, expected_dofs=DOFS,
                             incident_roots=incident, returning_roots=returning, trace_sha256=TRACE_SHA)
        blocks, totals, errors, rhs_error, boundary_error, coherent = replay.replay_ports(
            modes, fields, section["left_dofs"], section["right_dofs"], vectors, force, rhs, rhs_norm,
            incident_roots=incident, returning_roots=returning)
    replay.compare_public_tables(public, blocks, totals, incident_roots=incident, returning_roots=returning)
    reflection, transmission, direct_r, direct_t, net = totals
    representative = int(np.argmax(reflection))
    return {"author": "Athul Prem", "file_tag": "_AP_pnc", "status": "point_verified_within_saved_discretization",
            "frequency_hz": FREQUENCY, "problem_dofs": DOFS, "port_dofs": PORT_DOFS,
            "reference_planes_m": replay.PLANES, "public_summary": z4.artifact(summary_path),
            "phase_receipts": {name: z4.artifact(path) for name, path in (("stage", stage_path), ("solve", solve_path), ("export", export_path))},
            "accepted_shared_safe": z4.artifact(H80_SAFE), "accepted_h80_replay": z4.artifact(H80_REPLAY),
            "source_compatibility": changes,
            "implementation": {"verifier": z4.artifact(Path(__file__)), "shared_replay": z4.artifact(Path(replay.__file__)),
                               "z4_replay": z4.artifact(Path(z4.__file__)), "comparison": z4.artifact(Path(comparison.__file__))},
            "relative_target": replay.TARGET, "relative_residual_limit": replay.RESIDUAL,
            "backward_residual_limit": replay.RESIDUAL, "power_limit": replay.POWER, "matrix_infinity_norm": matrix_norm,
            "maximum_relative_residual": float(max(relative)), "maximum_backward_residual": float(max(backward)),
            "maximum_power_errors": dict(zip(replay.POWER_CHECKS, errors.tolist(), strict=True)),
            "coherent_left_incidence": coherent,
            "channels": [{"incident_root": root, "reverse_root": returning[col], "relative_residual": relative[col],
                "backward_residual": backward[col], "rhs_relative_error": rhs_error[col],
                "body_boundary_relative_residual": boundary_error[col], "reflection": reflection[col], "transmission": transmission[col],
                "reflection_flux": direct_r[col], "transmission_flux": direct_t[col], "net_outward_power_ratio": net[col],
                "converted_reflection": reflection[col] - blocks["reflection"][1][col, col],
                "converted_transmission": transmission[col] - blocks["transmission"][1][col, col]}
                for col, root in enumerate(incident)],
            "representative_field": {"incident_root": incident[representative], "column": representative,
                "field": z4.artifact(public / "selected_fields_AP_pnc.npz"),
                "accepted_vector": channels[representative]["accepted_vector"], "mesh": original["mesh"],
                "units": "m per square root watt", "basis_order": 2},
            "available_memory_bytes_before_equations": available, "required_memory_bytes_before_equations": required,
            "elapsed_seconds": monotonic() - started,
            "spatial_convergence_established": False, "frequency_convergence_established": False}


def compare_blocks(reference, candidate):
    matches, blocks, rows = comparison.compare_blocks(reference, candidate)
    power = max(block["maximum_power_change"] for block in blocks.values())
    complex_change = max(block["maximum_complex_change"] for block in blocks.values())
    r = blocks["reflection"]
    reference_r, candidate_r = (np.asarray(r[f"{name}_power"]) for name in ("reference", "candidate"))
    ranking_a, ranking_b = np.argsort(-reference_r), np.argsort(-candidate_r)
    stronger, weaker = ranking_a[:2], ranking_a[2:]
    relative_rows = [{"reference_incident_root": row["reference_incident_root"],
                     "candidate_incident_root": row["candidate_incident_root"],
                     "reflection_h100": row["reference_power"], "reflection_h80": row["candidate_power"],
                     "signed_absolute_change": row["power_change"],
                     "signed_relative_change": row["power_change"] / max(abs(row["reference_power"]), 1e-12),
                     "maximum_complex_reflection_change": row["maximum_complex_change"]}
                    for row in rows if row["quantity"] == "reflection"]
    return {"matched_channels": matches, "blocks": blocks, "rows": rows, "maximum_power_change": power,
            "maximum_complex_change": complex_change, "power_change_limit": replay.POWER, "complex_change_limit": replay.POWER,
            "power_passed": power < replay.POWER, "complex_passed": complex_change < replay.POWER,
            "mesh_pair_passed": max(power, complex_change) < replay.POWER,
            "reference_reflection_order_descending": ranking_a.tolist(),
            "candidate_reflection_order_in_reference_basis_descending": ranking_b.tolist(),
            "two_stronger_channels_preserved": bool(set(stronger) == set(ranking_b[:2])),
            "reference_stronger_weaker_reflection_separation": float(min(reference_r[stronger]) - max(reference_r[weaker])),
            "candidate_separation_of_reference_groups": float(min(candidate_r[stronger]) - max(candidate_r[weaker])),
            "weak_reflection_diagnostic": {"reference": "h100", "denominator_floor": 1e-12,
                "formula": "(R_h80-R_h100)/max(abs(R_h100),1e-12)", "relative_acceptance_gate": None,
                "floor_interpretation": "Reporting guard only; no numerical noise bound or relative-accuracy gate.",
                "channels": relative_rows}}


def compare_verified_point(h100_report):
    accepted = replay.read_receipt(H80_REPLAY, H80_REPLAY_SHA)
    public = Path(h100_report["public_summary"]["path"]).parent
    first, fixed, first_record = comparison.read_run(public)
    second, other, second_record = comparison.read_run(H80_PUBLIC)
    require(first_record["summary_sha256"] == h100_report["public_summary"]["sha256"]
            and second_record["summary_sha256"] == accepted["public_summary"]["sha256"]
            and h100_report["status"] == accepted["status"] == "point_verified_within_saved_discretization",
            "A compared point lacks its exact independently accepted summary.")
    fixed = z4.physical_pair(first, second, fixed, other)
    for summary, dofs, size in ((first, DOFS, 100), (second, 426627, 80)):
        mesh = summary["configuration"]["mesh"]
        require(summary["ndof"] == dofs and summary["port_dofs"] == PORT_DOFS and summary["power_balance_passed"] is True
                and mesh["mesh_size_nm"] == size and mesh["thickness_layers"] == 4
                and mesh["element_order"] == 2 and mesh["circle_resolution"] == 32,
                "The comparison does not contain the accepted h100/h80 z4 profiles.")
    first_data = comparison.read_scattering(public, first, fixed)
    second_data = comparison.read_scattering(H80_PUBLIC, second, fixed)
    require(first_data[0].discretization_sha256 != second_data[0].discretization_sha256,
            "An in-plane control needs different body meshes.")
    return {"author": "Athul Prem", "file_tag": "_AP_pnc", "status": "compared",
            "kind": "global_in_plane_mesh_sensitivity", "reference": first_record, "candidate": second_record,
            "physical_inputs": fixed, "accepted_h80_replay": z4.artifact(H80_REPLAY),
            **compare_blocks(first_data, second_data), "phase_adjustment_applied": False,
            "spatial_convergence_established": False, "frequency_convergence_established": False,
            "scope": "One h100/h80 global in-plane increment at 0.1 GHz changes both lead interiors and the patterned region. "
                     "Original end-plane phases are retained. Finite pair checks do not establish continuum or weak-reflection error bounds."}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("stage", "solve", "export"):
        parser.add_argument(f"--{name}", type=Path, required=True)
    parser.add_argument("--compare-h80", action="store_true")
    args = parser.parse_args()
    result = replay_point(*(getattr(args, name).resolve() for name in ("stage", "solve", "export")))
    if args.compare_h80:
        result["in_plane_comparison"] = compare_verified_point(result)
    print(json.dumps(result, indent=2, allow_nan=False, default=lambda value: value.item() if isinstance(value, np.generic) else value))


if __name__ == "__main__":
    main()
