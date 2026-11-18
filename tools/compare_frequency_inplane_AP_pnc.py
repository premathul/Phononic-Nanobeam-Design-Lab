"""Compare one saved h100/h80 887.5 MHz control. Athul Prem | _AP_pnc.

Only bounded JSON, CSV and producer source files are read. Exact shared SAFE
artifact identities establish the coefficient basis; no fields are rematched.
"""
from __future__ import annotations

import argparse
import cmath
import csv
from html import escape
from io import StringIO
import json
import math
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT), str(ROOT / "src")]
from phononic.workflow_AP_pnc import digest
from tools import compare_device_frequency_AP_pnc as response_reader

AUTHOR = {"author": "Athul Prem", "file_tag": "_AP_pnc"}
FREQUENCY, ANCHOR_FREQUENCY = 887_500_000, 1_000_000_000
ROOTS, REVERSE = [1302, 1346, 1292, 1286, 1282], [1303, 1347, 1293, 1287, 1283]
ANCHOR_ROOTS, ANCHOR_REVERSE = [1348, 1311, 1292, 1284, 1274], [1349, 1312, 1293, 1285, 1275]
PAIR, ANCHOR_PAIR = ((1302, 1283), (1282, 1303)), ((1311, 1275), (1274, 1312))
PLANES = [0., 1.665e-5]
POWER_LIMIT, COMPLEX_LIMIT, PAIR_LIMIT, SUPPRESSION_LIMIT = .01, .01, .10, .01
SAFE_MODES_SHA = "a2cf1204490983c766d84292bfc85a0182da9997e56fbf2e7281251e11864b32"
SAFE_SECTION_SHA = "2a2aa9347921ac6940f851f2eb53bc06480a28a6cf67c27a000faa357b1404e8"
SECTION_RECEIPT_SHA = "677c64a816ea42a8b37e1358d5549d63ecaeec7623a9366139ba64d301bbea4e"
SHARED_PORT_TABLE_SHA = "05772e7246425c3dabf48b2b230ab1a71e966db6e58bbb31112c3cb22f396480"
H80_POWERS = (9.916496528595719e-8, 9.91649649635775e-8)
ANCHOR_POWERS = {"h100": (.001232799344468317, .001232799342515478),
                 "h80": (.0011241681176131276, .0011241681176200138)}
STUDY = "results/runs/device_frequency_thickness_8875_AP_pnc/"
H100 = "results/runs/full_device_layered_preflight/historical_26hole_layered_h100_ny12_z4-2d4bb91449a7/"
H80 = ("results/runs/full_device_resource_profile/lawrence_h80_AP_pnc/"
       "h80-export-1335045/historical_26hole_layered_h80_ny12_z4-3385249d64d1/")
PINS = {
    "accepted_safe": (STUDY + "h80-z4-frequency-887500000-safe-1335641/safe_AP_pnc.json", "7d3dda2761e3f25d2f6470259cd18306688e2e9ce7afc5f2d9c9ee92ea5eae20"),
    "accepted_prebody": (STUDY + "independent_prebody_AP_pnc.json", "f45245144be6035c72ffee5ced2d96aa349b939bde89f9bb36f6ca82550bdca4"),
    "accepted_point": (STUDY + "independent_point_AP_pnc.json", "17bcad69b7d567331489d32a798f6755df292272473df78e9338ba259683fd34"),
    "accepted_relation": (STUDY + "relation/mode_relation_AP_pnc.json", "5ee5c8006803ee34ae7ef554588fdfa1f887c065c032c924cb79a6cc0558b9b4"),
    "accepted_thickness_comparison": (STUDY + "comparison/frequency_thickness_comparison_AP_pnc.json", "3549b1fe80b7fb283b66ef8e93701c88dcc9bd6b80f9a20d992b9b695724860e"),
    "accepted_anchor_relation": ("results/runs/comparisons/scattering-f6e75a9694c42a2c/summary_AP_pnc.json", "e7f66845ccdbf4aca57b6976fbc1320b07ead0d9e0007a768766a93b01f5056f"),
    "h100_anchor_summary": (H100 + "summary_AP_pnc.json", "1a873882e9f16fd1604081a48319c80c92af34cac5f15a1a849fb1a3faabe0e8"),
    "h80_anchor_summary": (H80 + "summary_AP_pnc.json", "f65f61eaf62d69fa0574fe8566ba088fd7e090a019685bdac3db95261b61619f"),
    "response_reader_source": ("tools/compare_device_frequency_AP_pnc.py", "36bb5effcd52ca7fde7da7f876acf8fb34bdaaf897bb4a6e8e7e28bb364ef623"),
    "accepted_point_source": ("tools/replay_lawrence_frequency_z4_AP_pnc.py", "6eb0599a919eb3d1de7f99bed9a7e9cc452e6d2c6becf8b45379ae173e6afdbf"),
    "accepted_prebody_source": (STUDY + "local_checks/replay_lawrence_frequency_z4_prebody_AP_pnc.py", "34e2307663a9ea2c3cb864562a58559a6b94c8710219ea17fd305013e0110572"),
    "accepted_relation_source": (STUDY + "local_checks/compare_frequency_thickness_relation_AP_pnc.py", "1fb709dfacb1c9e2a45d2ac4d43fce7ab5a5c471021f6026bb85406936ecb08b"),
}
SCOPE = ("One global h100/h80 in-plane discretization increment at 887.5 MHz, including lead interiors, "
         "with exact shared 675-DOF SAFE columns and original 0/16.65 micrometre planes. The directed "
         "10% increment and 99% suppression-retention targets are separate finite observations, not "
         "accuracy estimates, a numerical floor, an uncertainty bound or convergence. All previous "
         "failed spatial comparisons remain unresolved. Endpoint relations do not establish continuous branches.")
require, sha = response_reader.require, response_reader.sha


def checked_file(path, expected_sha):
    path = Path(path).resolve(strict=True)
    require(path.is_file() and path.suffix in (".json", ".csv", ".py")
            and path.stat().st_size <= 4 * 2**20, "Only bounded JSON/CSV/source inputs are permitted.")
    require(response_reader.matching.is_sha256(expected_sha) and sha(path) == expected_sha,
            "An externally accepted artifact SHA-256 differs.")
    return path


def artifact(path, workspace=ROOT):
    return response_reader.link(path, workspace)


def basename(value):
    return str(value.get("path", value.get("file", ""))).replace("\\", "/").rsplit("/", 1)[-1]


def same_artifact(first, second):
    return (basename(first) == basename(second)
            and all(first.get(key) == second.get(key) for key in ("sha256", "bytes")))


def read_json(path):
    require(Path(path).stat().st_size <= 4 * 2**20, "A bounded JSON input exceeds 4 MiB.")
    return json.loads(Path(path).read_bytes())


def load_pins(workspace=ROOT):
    root, paths, inputs = Path(workspace).resolve(), {}, {}
    for role, (relative, expected) in PINS.items():
        path = checked_file(root / relative, expected)
        require(path.is_relative_to(root), "A pinned scientific role escapes the workspace.")
        paths[role], inputs[role] = path, artifact(path, root)
    return paths, inputs


def response_inputs(summary_path, expected_sha, workspace, inputs, prefix, roots, reverse, frequency_hz):
    summary_path = checked_file(summary_path, expected_sha)
    summary, tables = read_json(summary_path), []
    inputs[prefix + "_summary"] = artifact(summary_path, workspace)
    for role, name in (("totals", "scattering_AP_pnc.csv"), ("coefficients", "scattering_channels_AP_pnc.csv")):
        path = checked_file(summary_path.parent / name, summary["output_sha256"][name])
        inputs[prefix + "_" + role] = artifact(path, workspace)
        with path.open(newline="", encoding="utf-8") as stream:
            tables.append(list(csv.DictReader(stream)))
    return summary, response_reader.parse_response(summary, *tables, roots, reverse, frequency_hz)


def validate_replay(value, summary_path, expected_dofs):
    require(value["status"] == "point_verified_within_saved_discretization"
            and value["frequency_hz"] == FREQUENCY and value["problem_dofs"] == expected_dofs
            and value["port_dofs"] == 675 and value["reference_planes_m"] == PLANES
            and same_artifact(value["public_summary"], artifact(summary_path))
            and value["public_summary_sha256"] == sha(summary_path)
            and value["relative_target"] == 1e-10
            and value["relative_residual_limit"] == value["backward_residual_limit"] == 1e-9
            and 0 <= value["maximum_relative_residual"] <= 1e-9
            and 0 <= value["maximum_backward_residual"] <= 1e-9
            and value["propagating_coefficients_checked"] == 50
            and value["full_directional_coefficients_checked"] == 6750
            and all(value[key] is True for key in ("original_equations_passed", "physical_nodal_force_power_passed",
                "all_modal_coefficients_passed", "full_p2_field_identity_passed", "coherent_left_incidence_passed")),
            "The response lacks exact accepted original-equation/flux/field/coherent replay.")
    rows = value["channels"]
    require(len(rows) == 5 and [row["column"] for row in rows] == list(range(5))
            and [row["incident_root"] for row in rows] == ROOTS
            and [row["reverse_root"] for row in rows] == REVERSE,
            "The independent replay signed drive columns differ.")
    coherent = value["coherent_left_incidence"]
    require(coherent["incident_roots"] == coherent["transmitted_roots"] == ROOTS
            and coherent["reflected_roots"] == REVERSE,
            "The coherent replay does not bind the complete signed basis.")


def validate_reference(paths, inputs):
    safe, prebody, point, relation, prior = (read_json(paths[name]) for name in
        ("accepted_safe", "accepted_prebody", "accepted_point", "accepted_relation", "accepted_thickness_comparison"))
    require(safe["status"] == "safe_ready" and safe["frequency_hz"] == FREQUENCY
            and safe["discrete_roots"] == 1350 and safe["positive_open_channels"] == safe["negative_open_channels"] == 5
            and safe["safe_modes"]["sha256"] == SAFE_MODES_SHA
            and safe["safe_section"]["sha256"] == SAFE_SECTION_SHA
            and safe["section_receipt"]["sha256"] == SECTION_RECEIPT_SHA
            and safe["reference_planes_m"] == PLANES and safe["degeneracy_status"] == "none_detected",
            "The accepted shared SAFE basis identity differs.")
    for direction, roots, sign in (("positive", ROOTS, 1), ("negative", REVERSE, -1)):
        rows = [row for row in safe["open_channels"] if row["direction"] == direction]
        require([row["root_id"] for row in rows] == roots
                and all(row["k_imag_rad_m"] == 0 and row["k_real_rad_m"] * sign > 0
                    and abs(row["signed_power_w"] - sign) <= 1e-7 for row in rows),
                "The accepted SAFE signed root columns differ.")
    require(same_artifact(point["phase_receipts"]["safe"], inputs["accepted_safe"])
            and same_artifact(point["prebody_replay"], inputs["accepted_prebody"])
            and same_artifact(point["relation"], inputs["accepted_relation"])
            and same_artifact(prebody["safe_receipt"], inputs["accepted_safe"])
            and same_artifact(prebody["safe_modes"], safe["safe_modes"])
            and same_artifact(prebody["relation"], inputs["accepted_relation"])
            and prebody["status"] == "independently_verified_safe"
            and prebody["relation_status"] == "matched"
            and relation["status"] == "signed_section_relation_accepted"
            and relation["root_permutation_square"]["status"] == "consistent",
            "The exact accepted reference proof chain differs.")
    require(point["implementation"]["verifier"]["sha256"] == inputs["accepted_point_source"]["sha256"]
            and prebody["implementation"]["verifier"]["sha256"] == inputs["accepted_prebody_source"]["sha256"]
            and relation["producer"]["sha256"] == inputs["accepted_relation_source"]["sha256"],
            "An accepted historical producer identity differs.")
    require(prior["status"] == "compared_finite_thickness_increment" and prior["complex_passed"] is False
            and prior["pair_specific_targets"]["directed_increment_passed"] is False
            and prior["other_prior_failed_spatial_comparisons_remain_unresolved"] is True,
            "The previous failed spatial controls were not retained.")
    return safe, point, relation, prior


def validate_shared_basis(candidate, reference, safe, candidate_replay, inputs):
    for summary, dofs, mesh_size in ((candidate, 307611, 100), (reference, 426627, 80)):
        require(summary["status"] == "completed_unverified" and summary["ndof"] == dofs
                and summary["port_dofs"] == 675 and summary["hole_count"] == 26
                and summary["configuration"]["mesh"]["mesh_size_nm"] == mesh_size,
                "The bounded archived in-plane dimensions differ.")
        provenance = summary["checkpointed_scattering"]
        require(provenance["reference_planes_m"] == PLANES and provenance["phase_adjustment_applied"] is False
                and same_artifact(provenance["safe_receipt"], inputs["accepted_safe"])
                and same_artifact(provenance["safe_modes"], safe["safe_modes"])
                and same_artifact(provenance["safe_section"], safe["safe_section"])
                and same_artifact(provenance["section_receipt"], safe["section_receipt"])
                and provenance["open_channels"] == safe["open_channels"],
                "The two responses do not share exact signed SAFE columns at the original planes.")
    for name in ("geometry", "material"):
        require(candidate["configuration"][name] == reference["configuration"][name], "The common physical device changed.")
    meshes = [summary["configuration"]["mesh"] for summary in (candidate, reference)]
    require({k: v for k, v in meshes[0].items() if k != "mesh_size_nm"}
            == {k: v for k, v in meshes[1].items() if k != "mesh_size_nm"},
            "A mesh parameter other than global in-plane size changed.")
    require(candidate["hole_pattern_provenance"] == reference["hole_pattern_provenance"], "The placed hole pattern changed.")
    for stem in (*response_reader.matching.CORE, "layered", "assembly_batched"):
        key = f"src/phononic/{stem}_AP_pnc.py"
        require(candidate["implementation"]["source_sha256"][key] == reference["implementation"]["source_sha256"][key],
                "A physical numerical source changed.")
    require(candidate["implementation"]["packages"] == reference["implementation"]["packages"],
            "Numerical package versions changed between responses.")
    require(same_artifact(candidate_replay["accepted_shared_safe"], inputs["accepted_safe"])
            and same_artifact(candidate_replay["prebody_replay"], inputs["accepted_prebody"])
            and same_artifact(candidate_replay["relation"], inputs["accepted_relation"])
            and same_artifact(candidate_replay["accepted_reference_point"], inputs["accepted_point"]),
            "The new replay does not bind the exact accepted reference proof.")
    provenance = candidate["checkpointed_scattering"]
    for phase, field in (("stage", "prepared_frequency_stage"), ("solve", "solve_receipt")):
        require(same_artifact(candidate_replay["phase_receipts"][phase], provenance[field]),
                "The new replay is linked to different execution phases.")
    return {"status": "identical_saved_basis", "safe_receipt": inputs["accepted_safe"],
            "safe_modes": safe["safe_modes"], "safe_section": safe["safe_section"],
            "incoming_roots": ROOTS, "reflected_roots": REVERSE, "transmitted_roots": ROOTS,
            "candidate_gauge_complex": [[1., 0.]] * 5, "candidate_gauge_phase_rad": [0.] * 5,
            "basis_evidence": "Exact hash-identical saved SAFE displacement/traction columns, independently replayed for both points; root labels alone are not used.",
            "field_matching_recomputed": False}


def validate_anchor_relation(relation, anchors, frequency_relation, inputs):
    require(relation["status"] == "compared" and relation["complex_passed"] is False
            and relation["mesh_pair_passed"] is False, "The retained 1 GHz spatial failure differs.")
    for name, role in (("h100", "reference"), ("h80", "candidate")):
        accepted = relation[role]
        require(accepted["summary_sha256"] == inputs[name + "_anchor_summary"]["sha256"]
                and accepted["run_hash"] == anchors[name][0]["run_hash"]
                and accepted["output_sha256"] == anchors[name][0]["output_sha256"]
                and accepted["output_sha256"]["port_modes_AP_pnc.csv"] == SHARED_PORT_TABLE_SHA,
                "A 1 GHz anchor differs from its accepted signed-basis proof.")
    for name, roots in (("incoming", ANCHOR_ROOTS), ("transmission", ANCHOR_ROOTS), ("reflection", ANCHOR_REVERSE)):
        group = relation["matched_channels"][name]
        require(group["reference_roots"] == group["candidate_roots"] == roots
                and min(group["mass_overlaps"]) >= .99
                and max(abs(value) for value in group["candidate_gauge_phase_rad"]) <= 1e-12
                and group["reference_wavenumber_rad_m"] == group["candidate_wavenumber_rad_m"],
                "The accepted 1 GHz root/gauge correspondence differs.")
    edge = frequency_relation["edges"]["z4_frequency"]
    require(edge["status"] == "matched", "The accepted signed frequency endpoint relation is unresolved.")
    maps = {name: dict(zip(edge["matched_channels"][name]["reference_roots"],
                               edge["matched_channels"][name]["candidate_roots"], strict=True))
            for name in ("incoming", "reflection")}
    require(tuple((maps["incoming"][a], maps["reflection"][b]) for a, b in ANCHOR_PAIR) == PAIR,
            "The accepted signed frequency relation selects another conversion pair.")


def power_change(reference, candidate):
    require(all(math.isfinite(value) and value >= 0 for value in (reference, candidate)), "Invalid finite powers.")
    delta = candidate - reference
    return {"reference_power": reference, "candidate_power": candidate, "power_change": delta,
            "absolute_power_change": abs(delta), "relative_power_change": delta / reference if reference > 0 else None,
            "relative_denominator": reference, "denominator_floor_applied": False}


def complex_change(first, second):
    delta = second - first
    return {"complex_delta": [delta.real, delta.imag], "absolute_complex_change": abs(delta),
            "relative_denominator_amplitude": abs(first),
            "relative_absolute_complex_change": abs(delta) / abs(first) if first != 0 else None,
            "phase_change_rad": cmath.phase(second / first) if first != 0 and second != 0 else None}


def compare_responses(reference, candidate):
    lookups = [{(row["kind"], row["incident_root"], row["outgoing_root"]): row
                for row in response["coefficients"]} for response in (reference, candidate)]
    require(set(lookups[0]) == set(lookups[1]) and len(lookups[0]) == 50,
            "The complete signed coefficient identities differ.")
    coefficients = []
    for kind in ("reflection", "transmission"):
        for incoming in ROOTS:
            for outgoing in REVERSE if kind == "reflection" else ROOTS:
                first, second = (lookup[kind, incoming, outgoing] for lookup in lookups)
                a, b = complex(*first["amplitude"]), complex(*second["amplitude"])
                coefficients.append({"kind": kind, "incident_root": incoming, "outgoing_root": outgoing,
                    "reference_raw_amplitude": first["amplitude"], "candidate_raw_amplitude": second["amplitude"],
                    "candidate_gauge_aligned_amplitude": second["amplitude"],
                    "raw_complex_change": complex_change(a, b), "gauge_aligned_complex_change": complex_change(a, b),
                    **power_change(first["power"], second["power"])})
    by_root = [{row["root"]: row for row in response["rows"]} for response in (reference, candidate)]
    require(all(set(value) == set(ROOTS) for value in by_root), "The complete per-drive rows differ.")
    rows = [{"incident_root": root, "quantity": quantity,
             **power_change(by_root[0][root][quantity], by_root[1][root][quantity])}
            for root in ROOTS for quantity in ("R", "T", "converted_R", "converted_T", "total_converted_power")]
    selected = [next(row for row in coefficients if row["kind"] == "reflection"
                     and (row["incident_root"], row["outgoing_root"]) == pair) for pair in PAIR]
    phases = [{"incident_root": row["incident_root"],
               "phase_change_rad": row["gauge_aligned_complex_change"]["phase_change_rad"]}
              for row in coefficients if row["kind"] == "transmission" and row["incident_root"] == row["outgoing_root"]]
    return {"per_drive": rows, "coefficients": coefficients, "selected_pair": selected,
            "raw_responses": {"reference_h100": reference, "candidate_h80": candidate},
            "same_channel_transmission_phase_changes": phases,
            "maximum_power_change": max(row["absolute_power_change"] for row in rows if row["quantity"] in ("R", "T")),
            "maximum_complex_change": max(row["gauge_aligned_complex_change"]["absolute_complex_change"] for row in coefficients)}


def pair_targets(h100, h80, anchor_h100, anchor_h80):
    require(all(len(value) == 2 for value in (h100, h80, anchor_h100, anchor_h80)),
            "Exactly two separate directed conversions are required.")
    increments, suppression = [], []
    for direction, (first, second) in enumerate(zip(h100, h80, strict=True), 1):
        require(all(math.isfinite(value) and value >= 0 for value in (first, second)) and second > 0,
                "The exact h80 directed-power denominator must be positive and finite.")
        ratio = abs(first - second) / second
        increments.append({"direction": direction, "h100_power": first, "accepted_h80_power": second,
                           "signed_h100_minus_h80_power": first - second, "absolute_power_change": abs(first - second),
                           "relative_absolute_power_change": ratio, "exact_denominator": second,
                           "limit": PAIR_LIMIT, "passed": ratio <= PAIR_LIMIT})
        for name, power, anchor in (("h100", first, anchor_h100[direction - 1]), ("h80", second, anchor_h80[direction - 1])):
            require(math.isfinite(anchor) and anchor > 0, "A suppression anchor must be positive and finite.")
            fraction = power / anchor
            suppression.append({"direction": direction, "resolution": name, "power_8875": power,
                                "power_one_ghz": anchor, "ratio": fraction, "decrease_fraction": 1 - fraction,
                                "limit": SUPPRESSION_LIMIT, "passed": fraction <= SUPPRESSION_LIMIT})
    return {"directed_increment_rows": increments, "directed_increment_passed": all(row["passed"] for row in increments),
            "suppression_retention_rows": suppression, "suppression_retention_passed": all(row["passed"] for row in suppression),
            "denominator_floor_applied": False, "two_separate_unit_power_drives": True,
            "interpretation": "Finite observational targets only; no accuracy, numerical-floor or uncertainty claim."}


def selected_powers(response, pairs):
    values = {(row["kind"], row["incident_root"], row["outgoing_root"]): row["power"] for row in response["coefficients"]}
    return [values["reflection", a, b] for a, b in pairs]


def build_comparison(candidate_directory, point_path, point_sha, replay_source_sha, workspace=ROOT):
    paths, inputs = load_pins(workspace)
    safe, reference_point, relation, prior = validate_reference(paths, inputs)
    point_path = checked_file(point_path, point_sha)
    verifier_path = checked_file(Path(workspace) / "tools/replay_lawrence_frequency_h100_AP_pnc.py", replay_source_sha)
    point = read_json(point_path)
    require(same_artifact(point["implementation"]["verifier"], artifact(verifier_path, workspace)),
            "The new independent replay producer differs from its externally pinned source.")
    inputs.update(point_replay=artifact(point_path, workspace), point_replay_source=artifact(verifier_path, workspace))
    candidate_path = Path(candidate_directory) / "summary_AP_pnc.json"
    h100_summary, h100 = response_inputs(candidate_path, point["public_summary_sha256"], workspace, inputs,
                                        "h100_8875", ROOTS, REVERSE, FREQUENCY)
    reference_path = Path(workspace) / prior["inputs"]["z4_8875_summary"]["path"]
    h80_summary, h80 = response_inputs(reference_path, prior["inputs"]["z4_8875_summary"]["sha256"], workspace, inputs,
                                      "h80_8875", ROOTS, REVERSE, FREQUENCY)
    validate_replay(point, candidate_path, 307611)
    validate_replay(reference_point, reference_path, 426627)
    basis = validate_shared_basis(h100_summary, h80_summary, safe, point, inputs)
    anchors = {name: response_inputs(paths[name + "_anchor_summary"], PINS[name + "_anchor_summary"][1],
        workspace, inputs, name + "_one_ghz", ANCHOR_ROOTS, ANCHOR_REVERSE, ANCHOR_FREQUENCY) for name in ("h100", "h80")}
    validate_anchor_relation(read_json(paths["accepted_anchor_relation"]), anchors, relation, inputs)
    powers = {name: selected_powers(anchors[name][1], ANCHOR_PAIR) for name in anchors}
    require(all(tuple(powers[name]) == ANCHOR_POWERS[name] for name in powers)
            and tuple(selected_powers(h80, PAIR)) == H80_POWERS,
            "An exact accepted directed-power denominator differs from raw CSV evidence.")
    for name, (anchor_summary, _) in anchors.items():
        summary = h100_summary if name == "h100" else h80_summary
        for key in ("geometry", "material", "mesh"):
            require(anchor_summary["configuration"][key] == summary["configuration"][key],
                    "A frequency comparison changes physical or spatial inputs.")
        for stem in (*response_reader.matching.CORE, "layered", "assembly_batched"):
            key = f"src/phononic/{stem}_AP_pnc.py"
            require(anchor_summary["implementation"]["source_sha256"][key]
                    == summary["implementation"]["source_sha256"][key],
                    "A physical numerical source changed between frequency anchors.")
    compared = compare_responses(h100, h80)
    targets = pair_targets(selected_powers(h100, PAIR), selected_powers(h80, PAIR), powers["h100"], powers["h80"])
    power_passed, complex_passed = compared["maximum_power_change"] < POWER_LIMIT, compared["maximum_complex_change"] < COMPLEX_LIMIT
    return {**AUTHOR, "schema_version": 1, "phase": "device_inplane_comparison",
            "status": "compared_finite_inplane_increment", "frequency_hz": FREQUENCY,
            "reference_resolution": "h100_z4", "candidate_resolution": "h80_z4", "inputs": inputs,
            "shared_basis": basis, **compared, "pair_specific_targets": targets,
            "input_power_checks_passed": True, "power_change_limit": POWER_LIMIT, "complex_change_limit": COMPLEX_LIMIT,
            "power_passed": power_passed, "complex_passed": complex_passed,
            "mesh_pair_passed": power_passed and complex_passed,
            "power_units": "Dimensionless power fraction for each separate 1 W incident drive.",
            "signed_increment": "h80 minus h100 in response rows; target A explicitly uses abs(h100-h80)/h80.",
            "pair_complex_phase_gate_applied": False, "denominator_floor_applied": False,
            "preserved_8875_thickness_control": {"proof": inputs["accepted_thickness_comparison"],
                "pair_specific_targets": prior["pair_specific_targets"], "maximum_complex_change": prior["maximum_complex_change"],
                "complex_passed": prior["complex_passed"]},
            "preserved_one_ghz_inplane_failure": inputs["accepted_anchor_relation"],
            "old_and_new_execution_provenance_distinct": True,
            "spatial_convergence_established": False, "frequency_convergence_established": False,
            "continuous_branch_identity_established": False, "numerical_floor_established": False,
            "phase_adjustment_applied": False, "field_matching_recomputed": False,
            "other_prior_failed_spatial_comparisons_remain_unresolved": True, "scope": SCOPE}


def write_outputs(output, result, workspace=ROOT):
    output = Path(output)
    payload = {**result, "producer": artifact(__file__, workspace)}
    payload["content_sha256"] = digest(payload)
    name = "frequency_inplane_comparison_AP_pnc"
    files = {name + ".json": json.dumps(payload, indent=2, allow_nan=False) + "\n"}
    rows = []
    for row in result["coefficients"]:
        flat = {key: value for key, value in row.items() if not isinstance(value, (dict, list))}
        for key, value in row.items():
            if isinstance(value, list):
                flat.update({key + "_real": value[0], key + "_imag": value[1]})
            elif isinstance(value, dict):
                flat.update({key + "_" + subkey: subvalue for subkey, subvalue in value.items() if not isinstance(subvalue, list)})
                flat.update({key + "_delta_real": value["complex_delta"][0], key + "_delta_imag": value["complex_delta"][1]})
        rows.append(flat)
    stream = StringIO(newline="")
    writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
    writer.writeheader()
    writer.writerows(rows)
    files[name + ".csv"] = stream.getvalue()
    body = f"<h1>887.5 MHz in-plane control</h1><p>Athul Prem | athul_simulation_pnc_frequency_inplane_8875</p><p>{escape(SCOPE)}</p>"
    body += (f"<p>Whole-response maximum power change: {result['maximum_power_change']:.12g} (pass: {result['power_passed']}); "
             f"maximum complex change: {result['maximum_complex_change']:.12g} (pass: {result['complex_passed']}). Each gate is 0.01.</p>")
    body += "<h2>A: Separate directed-power increments</h2><table><tr><th>Direction</th><th>h100 power</th><th>h80 power</th><th>|ΔP|/Ph80</th><th>≤0.10</th></tr>"
    targets = result["pair_specific_targets"]
    for row in targets["directed_increment_rows"]:
        body += f"<tr><td>{row['direction']}</td><td>{row['h100_power']:.12g}</td><td>{row['accepted_h80_power']:.12g}</td><td>{row['relative_absolute_power_change']:.12g}</td><td>{row['passed']}</td></tr>"
    body += "</table><h2>B: Separate suppression-retention ratios</h2><table><tr><th>Resolution</th><th>Direction</th><th>P8875/P1GHz</th><th>≤0.01</th></tr>"
    for row in targets["suppression_retention_rows"]:
        body += f"<tr><td>{row['resolution']}</td><td>{row['direction']}</td><td>{row['ratio']:.12g}</td><td>{row['passed']}</td></tr>"
    body += "</table><p>Both directions must pass individually. Exact positive accepted denominators; no floor substitution. Each row is a separate unit-power drive.</p>"
    body += "<h2>Selected complex coefficients</h2><table><tr><th>Signed pair</th><th>Raw and aligned |ΔS|</th><th>|ΔS|/|Sh100|</th><th>Δphase (rad)</th></tr>"
    number = lambda value: "undefined" if value is None else f"{value:.12g}"
    for row in result["selected_pair"]:
        change = row["gauge_aligned_complex_change"]
        body += f"<tr><td>{row['incident_root']}→{row['outgoing_root']}</td><td>{number(change['absolute_complex_change'])}</td><td>{number(change['relative_absolute_complex_change'])}</td><td>{number(change['phase_change_rad'])}</td></tr>"
    body += ("</table><p>Exact shared SAFE columns make the raw and gauge-aligned bases identical. Phase is diagnostic only, undefined for a zero coefficient. "
             "Previous 887.5 MHz z4/z6 target A failed (about 87.5563%); its complex change 0.01753014 failed. All earlier 1 GHz spatial failures remain unresolved.</p>"
             f'<p><a href="{name}.json">Complete source-bound JSON</a> · <a href="{name}.csv">All 50 complex coefficients</a></p>')
    files["report_AP_pnc.html"] = ('<!doctype html><meta charset="utf-8"><title>Frequency in-plane control | Athul Prem</title>'
        '<style>body{font:16px system-ui;max-width:1100px;margin:40px auto;padding:0 20px}table{border-collapse:collapse;margin:20px 0}td,th{border:1px solid #aaa;padding:8px;text-align:right}</style>' + body)
    # Check every existing publication before writing anything; reuse preserves
    # bytes and modification times, and conflicting scientific output is refused.
    for filename, text in files.items():
        path = output / filename
        require(not path.exists() or path.read_bytes() == text.encode("utf-8"),
                "Existing scientific output differs; choose a new output directory.")
    output.mkdir(parents=True, exist_ok=True)
    for filename, text in files.items():
        path = output / filename
        if not path.exists():
            path.write_bytes(text.encode("utf-8"))
    return output / (name + ".json")


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--candidate", type=Path, required=True)
    parser.add_argument("--point-replay", type=Path, required=True)
    parser.add_argument("--point-replay-sha256", required=True)
    parser.add_argument("--replay-source-sha256", required=True)
    parser.add_argument("--workspace", type=Path, default=ROOT)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args(argv)
    result = build_comparison(args.candidate, args.point_replay, args.point_replay_sha256, args.replay_source_sha256, args.workspace)
    print(write_outputs(args.output, result, args.workspace))
    return 0  # Failed observational/scientific targets are completed comparisons.


if __name__ == "__main__":
    raise SystemExit(main())
