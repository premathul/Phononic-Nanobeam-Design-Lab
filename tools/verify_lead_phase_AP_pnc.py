"""Independent finite phase-only audit. Athul Prem | _AP_pnc.

Read only bounded hash-bound JSON/CSV/source files. Reconstruct the accepted
left-incident coefficients directly, without importing the diagnostic producer
or the historical comparator. No numerical array, field, fit or solve is used.
"""
from __future__ import annotations

import argparse
import cmath
import csv
import hashlib
import json
import math
from pathlib import Path
import platform
from time import monotonic

import psutil

ROOT = Path(__file__).resolve().parents[1]
COMPARISON = "results/runs/device_frequency_inplane_8875_AP_pnc/comparison/frequency_inplane_comparison_AP_pnc.json"
COMPARISON_SHA = "107d1c62817349d0a6900650d10dd3826ee492ca4bc36abc8195ee3c3e7911c7"
FREQUENCY = 887_500_000
INCOMING = (1302, 1346, 1292, 1286, 1282)
REFLECTED = (1303, 1347, 1293, 1287, 1283)
PAIRS = (("reflection", 1302, 1283), ("reflection", 1282, 1303))
LIMIT = .01
SAFE_MODES_SHA = "a2cf1204490983c766d84292bfc85a0182da9997e56fbf2e7281251e11864b32"
SAFE_SECTION_SHA = "2a2aa9347921ac6940f851f2eb53bc06480a28a6cf67c27a000faa357b1404e8"


def require(condition, message):
    if not condition:
        raise ValueError(message)


def sha256(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def is_sha(value):
    return isinstance(value, str) and len(value) == 64 and all(c in "0123456789abcdef" for c in value)


class Budget:
    def __init__(self):
        self.started = monotonic()
        self.peak_rss = 0

    def check(self):
        process = psutil.Process()
        rss = sum(p.memory_info().rss for p in [process, *process.children(recursive=True)])
        self.peak_rss = max(self.peak_rss, rss)
        require(monotonic() - self.started <= 30., "Verifier wall ceiling exceeded.")
        require(rss <= 256 * 2**20, "Verifier process-tree RSS ceiling exceeded.")
        require(psutil.virtual_memory().available >= 512 * 2**20, "Physical-memory reserve breached.")


def bounded(path, workspace, expected_sha=None, expected_bytes=None):
    root, path = Path(workspace).resolve(), Path(path).resolve(strict=True)
    require(path.is_relative_to(root) and path.is_file(), "Input escapes the workspace.")
    require(path.suffix in (".json", ".csv", ".py") and path.stat().st_size <= 2 * 2**20,
            "Only bounded JSON/CSV/source inputs are permitted.")
    if expected_sha is not None:
        require(is_sha(expected_sha) and sha256(path) == expected_sha, "Input SHA-256 differs.")
    if expected_bytes is not None:
        require(path.stat().st_size == expected_bytes, "Input byte count differs.")
    return path


def read_json(path):
    def invalid(value):
        raise ValueError(f"Nonfinite JSON constant: {value}")
    return json.loads(Path(path).read_text(encoding="utf-8"), parse_constant=invalid)


def artifact(path, workspace):
    path = Path(path).resolve()
    return {"path": path.relative_to(Path(workspace).resolve()).as_posix(),
            "sha256": sha256(path), "bytes": path.stat().st_size}


def linked(item, workspace):
    return bounded(Path(workspace) / item["path"], workspace, item["sha256"], item["bytes"])


def close(actual, expected, name):
    require(isinstance(actual, (int, float)) and not isinstance(actual, bool)
            and math.isfinite(actual) and math.isfinite(expected), f"Invalid {name}.")
    require(math.isclose(actual, expected, rel_tol=2e-12, abs_tol=2e-15), f"Independent {name} differs.")


def validate_basis(basis):
    require(basis["status"] == "identical_saved_basis"
            and basis["incoming_roots"] == list(INCOMING)
            and basis["reflected_roots"] == list(REFLECTED)
            and basis["transmitted_roots"] == list(INCOMING), "Shared signed basis differs.")
    require(basis["safe_modes"]["sha256"] == SAFE_MODES_SHA
            and basis["safe_section"]["sha256"] == SAFE_SECTION_SHA,
            "Accepted SAFE artifact identity differs.")
    require(basis["candidate_gauge_complex"] == [[1., 0.]] * 5
            and basis["candidate_gauge_phase_rad"] == [0.] * 5
            and basis["field_matching_recomputed"] is False, "Native common gauge differs.")


def parse_response(summary, rows, dofs):
    require(summary["frequency_hz"] == FREQUENCY and summary["incident_side"] == "left"
            and summary["incident_mode_indices"] == list(INCOMING)
            and summary["num_incident_channels"] == 5 and summary["port_dofs"] == 675
            and summary["ndof"] == dofs and summary["end_face_dofs"] == {"xleft": 675, "xright": 675}
            and summary["scattering_power_normalization_available"] is True
            and summary["power_balance_passed"] is True, "Accepted response scope differs.")
    require(len(rows) == 50, "Expected exactly 50 raw coefficient rows.")
    result = {}
    for row in rows:
        key = (row["kind"], int(row["incident_port_index"]), int(row["outgoing_port_index"]))
        require(key not in result and key[1] in INCOMING, "Duplicate or unknown raw coefficient row.")
        require(int(row["incident_mode_order"]) == INCOMING.index(key[1]) + 1,
                "Incident channel order differs.")
        value = complex(float(row["amplitude_real"]), float(row["amplitude_imag"]))
        power = float(row["power_fraction"])
        require(math.isfinite(value.real) and math.isfinite(value.imag) and math.isfinite(power)
                and power >= 0., "Nonfinite or negative raw coefficient data.")
        close(power, value.real**2 + value.imag**2, "raw |amplitude| squared")
        result[key] = (value, power)
    expected = {(kind, incident, outgoing) for kind, outgoing_roots in
                (("reflection", REFLECTED), ("transmission", INCOMING))
                for incident in INCOMING for outgoing in outgoing_roots}
    require(set(result) == expected, "Raw signed coefficient coverage differs.")
    for incident in INCOMING:
        require(abs(math.fsum(power for key, (_, power) in result.items() if key[1] == incident) - 1.) < LIMIT,
                "Independent complete unit-drive power sum fails.")
    return result


def reconstruct(comparison_path, workspace, expected_sha=COMPARISON_SHA, budget=None):
    comparison_path = bounded(comparison_path, workspace, expected_sha)
    comparison = read_json(comparison_path)
    require(comparison["frequency_hz"] == FREQUENCY and comparison["reference_resolution"] == "h100_z4"
            and comparison["candidate_resolution"] == "h80_z4"
            and comparison["complex_change_limit"] == LIMIT, "Accepted comparison scope differs.")
    validate_basis(comparison["shared_basis"])
    inputs, responses = {}, {}
    for role in ("accepted_safe", "accepted_prebody", "accepted_point", "point_replay"):
        path = linked(comparison["inputs"][role], workspace)
        inputs[role] = artifact(path, workspace)
        receipt = read_json(path)
        require(receipt["frequency_hz"] == FREQUENCY, "Reused receipt frequency differs.")
        expected_status = {"accepted_safe": "safe_ready", "accepted_prebody": "independently_verified_safe"}.get(
            role, "point_verified_within_saved_discretization")
        require(receipt["status"] == expected_status, "Reused accepted receipt status differs.")
        if role in ("accepted_point", "point_replay"):
            label = "h80" if role == "accepted_point" else "h100"
            require(receipt["public_summary_sha256"] == comparison["inputs"][f"{label}_8875_summary"]["sha256"]
                    and receipt["relative_residual_limit"] == receipt["backward_residual_limit"] == 1e-9,
                    "Reused original-equation provenance differs.")
        if budget:
            budget.check()
    require(comparison["shared_basis"]["safe_receipt"] == comparison["inputs"]["accepted_safe"],
            "Shared SAFE receipt differs.")
    for label, dofs in (("h100", 307611), ("h80", 426627)):
        summary_role, table_role = f"{label}_8875_summary", f"{label}_8875_coefficients"
        summary_path = linked(comparison["inputs"][summary_role], workspace)
        table_path = linked(comparison["inputs"][table_role], workspace)
        summary = read_json(summary_path)
        require(summary["output_sha256"]["scattering_channels_AP_pnc.csv"] == sha256(table_path)
                and summary_path.parent == table_path.parent, "Original summary/table binding differs.")
        with table_path.open(newline="", encoding="utf-8") as stream:
            rows = list(csv.DictReader(stream))
        responses[label] = parse_response(summary, rows, dofs)
        inputs[summary_role], inputs[table_role] = artifact(summary_path, workspace), artifact(table_path, workspace)
        if budget:
            budget.check()
    result = []
    for key in sorted(responses["h100"]):
        old, old_power = responses["h100"][key]
        new, new_power = responses["h80"][key]
        old_mag, new_mag = math.hypot(old.real, old.imag), math.hypot(new.real, new.imag)
        delta = new - old
        result.append({"kind": key[0], "incident_root": key[1], "outgoing_root": key[2],
                       "reference_raw_amplitude": [old.real, old.imag],
                       "candidate_raw_amplitude": [new.real, new.imag],
                       "reference_power": old_power, "candidate_power": new_power,
                       "reference_magnitude": old_mag, "candidate_magnitude": new_mag,
                       "magnitude_lower_bound": abs(new_mag - old_mag),
                       "raw_complex_delta": [delta.real, delta.imag],
                       "absolute_raw_complex_change": math.hypot(delta.real, delta.imag),
                       "diagnostic_phase_change_rad": None if old == 0 or new == 0 else cmath.phase(new / old),
                       "selected_directed_pair": key in PAIRS})
    return comparison, inputs, result


def verify_entries(published, independent):
    require(len(published) == len(independent), "Published coefficient count differs.")
    actual = {}
    for row in published:
        key = (row["kind"], row["incident_root"], row["outgoing_root"])
        require(key not in actual, "Duplicate published coefficient row.")
        actual[key] = row
    for expected in independent:
        key = (expected["kind"], expected["incident_root"], expected["outgoing_root"])
        require(key in actual, "Published coefficient coverage differs.")
        row = actual[key]
        for name, value in expected.items():
            if name in ("kind", "incident_root", "outgoing_root", "selected_directed_pair"):
                require(type(row[name]) is type(value) and row[name] == value, f"Published {name} differs.")
            elif value is None:
                require(row[name] is None, "A zero coefficient has an undefined phase.")
            elif isinstance(value, list):
                require(isinstance(row[name], list) and len(row[name]) == 2, f"Invalid published {name}.")
                for component, target in zip(row[name], value):
                    close(component, target, name)
            else:
                close(row[name], value, name)


def verify_csv(path, rows):
    """Parse the published flat table independently of the producer writer."""
    columns = ("kind", "incident_root", "outgoing_root", "reference_real", "reference_imag", "candidate_real",
               "candidate_imag", "reference_power", "candidate_power", "reference_magnitude", "candidate_magnitude",
               "magnitude_lower_bound", "raw_complex_delta_real", "raw_complex_delta_imag", "absolute_raw_complex_change",
               "diagnostic_phase_change_rad", "selected_directed_pair")
    parsed = []
    with path.open(newline="", encoding="utf-8") as stream:
        reader = csv.DictReader(stream)
        require(tuple(reader.fieldnames or ()) == columns, "Published CSV columns differ.")
        for raw in reader:
            require(None not in raw and all(value is not None for value in raw.values()), "Malformed published CSV row.")
            require(raw["selected_directed_pair"] in ("True", "False"), "Invalid CSV selected-pair flag.")
            entry = {"kind": raw["kind"], "incident_root": int(raw["incident_root"]),
                     "outgoing_root": int(raw["outgoing_root"]),
                     "selected_directed_pair": raw["selected_directed_pair"] == "True"}
            for name in ("reference_power", "candidate_power", "reference_magnitude", "candidate_magnitude",
                         "magnitude_lower_bound", "absolute_raw_complex_change"):
                entry[name] = float(raw[name])
            for name, prefix in (("reference_raw_amplitude", "reference"), ("candidate_raw_amplitude", "candidate"),
                                 ("raw_complex_delta", "raw_complex_delta")):
                entry[name] = [float(raw[prefix + "_real"]), float(raw[prefix + "_imag"])]
            entry["diagnostic_phase_change_rad"] = (None if raw["diagnostic_phase_change_rad"] == ""
                                                    else float(raw["diagnostic_phase_change_rad"]))
            parsed.append(entry)
    verify_entries(parsed, rows)


def verify_publication(workspace, diagnostic_path, producer_sha256, expected_comparison_sha=COMPARISON_SHA):
    """The external producer hash must identify the source actually executed."""
    budget = Budget()
    budget.check()
    root = Path(workspace).resolve()
    path = bounded(diagnostic_path, root)
    publication = read_json(path)
    require(is_sha(producer_sha256), "An external executed-producer SHA-256 is required.")
    require(publication["producer"]["sha256"] == producer_sha256, "Executed producer identity differs.")
    linked(publication["producer"], root)
    comparison_path = linked(publication["accepted_comparison"], root)
    comparison, inputs, rows = reconstruct(comparison_path, root, expected_comparison_sha, budget)
    require(publication["status"] == "finite_phase_only_audit_completed"
            and publication["schema_version"] == 1, "Publication status/schema differs.")
    require(publication["shared_basis"] == comparison["shared_basis"], "Published shared basis differs.")
    require(publication["original_response_sources"] == {role: comparison["inputs"][role] for role in
            ("h100_8875_summary", "h100_8875_coefficients", "h80_8875_summary", "h80_8875_coefficients")},
            "Published original response provenance differs.")
    scope = publication["scope"]
    require(scope["frequency_hz"] == FREQUENCY and scope["reference_resolution"] == "h100_z4"
            and scope["candidate_resolution"] == "h80_z4" and scope["incident_side"] == "left"
            and scope["block_shape"] == [10, 5] and scope["coefficient_count"] == 50
            and scope["reference_planes_m"] == [0., 1.665e-5] and scope["native_incident_gauge_preserved"] is True,
            "Published fixed comparison scope differs.")
    for name in ("new_solve_performed", "new_assembly_performed", "arrays_read", "phase_fitting_performed",
                 "phase_adjustment_applied", "deembedding_performed", "uniform_phase_subtraction_performed",
                 "spatial_convergence_established", "numerical_floor_established"):
        require(scope[name] is False, f"Published {name} flag differs.")
    require(publication["method"]["tested_model"] == "Diagonal, reflectionless, unmixed, unit-modulus lead propagation.",
            "Published phase-only model differs.")
    acceptance_keys = ("maximum_power_change", "maximum_complex_change", "power_change_limit", "complex_change_limit",
                       "power_passed", "complex_passed", "mesh_pair_passed", "pair_specific_targets",
                       "other_prior_failed_spatial_comparisons_remain_unresolved")
    require(publication["preserved_original_acceptance"] == {name: comparison[name] for name in acceptance_keys},
            "Published original acceptance evidence differs.")
    verify_entries(publication["entries"], rows)
    maximum = max(row["magnitude_lower_bound"] for row in rows)
    frobenius = math.sqrt(math.fsum(row["magnitude_lower_bound"]**2 for row in rows))
    close(publication["aggregate"]["maximum_magnitude_lower_bound"], maximum, "maximum bound")
    close(publication["aggregate"]["frobenius_magnitude_lower_bound"], frobenius, "Frobenius bound")
    gate = publication["aggregate"]
    require(gate["existing_complex_change_limit"] == LIMIT
            and gate["frobenius_acceptance_gate_applied"] is False, "The original complex gate changed.")
    verdict = "phase-only model excluded at existing tolerance" if maximum > LIMIT else "not excluded"
    require(gate["verdict"] == verdict and gate["bound_strictly_exceeds_existing_limit"] is (maximum > LIMIT),
            "Published phase-only verdict differs.")
    expected_maximum = {(row["kind"], row["incident_root"], row["outgoing_root"]) for row in rows
                        if row["magnitude_lower_bound"] == maximum}
    published_maximum = [(row["kind"], row["incident_root"], row["outgoing_root"])
                         for row in gate["maximum_bound_entries"]]
    require(len(published_maximum) == len(expected_maximum) and set(published_maximum) == expected_maximum,
            "Published maximum-bound identities differ.")
    verify_entries(publication["selected_pair"], [row for row in rows if row["selected_directed_pair"]])
    publishing = publication["publication"]
    require(publishing["json"] == path.name and publishing["csv"] == "lead_phase_AP_pnc.csv",
            "Published table filename differs.")
    table = bounded(path.parent / publishing["csv"], root, publishing["csv_sha256"])
    verify_csv(table, rows)
    budget.check()
    return {"author": "Athul Prem", "file_tag": "_AP_pnc", "schema_version": 1,
            "status": "independently_verified_finite_phase_only_audit", "diagnostic": artifact(path, root),
            "producer": publication["producer"], "external_executed_producer_sha256": producer_sha256,
            "publication_csv": artifact(table, root),
            "verifier": artifact(__file__, ROOT), "accepted_comparison": artifact(comparison_path, root),
            "reused_inputs": inputs, "frequency_hz": FREQUENCY, "verified_coefficient_rows": 50,
            "selected_directed_pairs": [row for row in rows if row["selected_directed_pair"]],
            "maximum_magnitude_lower_bound": maximum, "frobenius_magnitude_lower_bound": frobenius,
            "complex_change_limit": LIMIT, "verdict": verdict,
            "interpretation": "Rejects only a diagonal, reflectionless, unmixed phase-only explanation. "
                              "Does not locate the discrepancy, exclude lead reflections or mixing, "
                              "establish pattern dominance, or certify convergence.",
            "python_version": platform.python_version(), "elapsed_seconds": monotonic() - budget.started,
            "sampled_peak_process_tree_rss_bytes": budget.peak_rss}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--workspace", type=Path, default=ROOT)
    parser.add_argument("--diagnostic", type=Path, required=True)
    parser.add_argument("--producer-sha256", required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    result = verify_publication(args.workspace, args.diagnostic, args.producer_sha256)
    output = args.output.resolve()
    require(output.is_relative_to(args.workspace.resolve()) and output.suffix == ".json",
            "Verifier output escapes the workspace or is not JSON.")
    payload = json.dumps(result, indent=2, allow_nan=False) + "\n"
    require(len(payload.encode()) <= 2 * 2**20, "Verifier output ceiling exceeded.")
    require(not output.exists(), "Refusing to overwrite existing verification evidence.")
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(payload, encoding="utf-8")
    print(json.dumps({"status": result["status"], "verified_coefficient_rows": 50, "verdict": result["verdict"]}))


if __name__ == "__main__":
    main()
