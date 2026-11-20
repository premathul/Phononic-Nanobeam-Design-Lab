#!/usr/bin/env python3
"""Compare paired saved axial powers and mesh controls at 995–1000 MHz. Author: Athul Prem | _AP_pnc."""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
import runpy
import sys

ROOT = Path(__file__).resolve().parents[1]
if __name__ == "__main__":
    sys.path.insert(0, str(ROOT))

from tools import compare_axial_device_power_AP_pnc as axial
from tools import map_scattering_AP_pnc as native
from tools.compare_scattering_AP_pnc import require


ANCHOR = ROOT / "verification/axial_device_rim_pairs_1GHz_AP_pnc.json"
CENTRAL_ANCHOR = ROOT / "verification/axial_device_third_pair_1GHz_AP_pnc.json"
LOWER_HZ, MIDPOINT_HZ, UPPER_HZ = 995_000_000, 997_500_000, 1_000_000_000
ROLES = {"thick": (200, "reference"), "thin": (100, "candidate")}
POWERS = ("T", "R")
POINT_PINS = ("map", "checkpoint", "point_summary", "independent_flux_replay")
MODE_FIELDS = ("drive", "root", "k_rad_m", "axial_kinetic_energy_fraction", *POWERS)


def _same_mode(first, second):
    return (all(first[key] == second[key] for key in ("drive", "root", "k_rad_m"))
            and all(math.isclose(first[key], second[key], abs_tol=1e-12, rel_tol=0)
                    for key in ("axial_kinetic_energy_fraction", *POWERS)))


def _native_pointer(endpoint):
    path = native._verified(endpoint["archive"]["map"], ROOT)
    require(path == ROOT / endpoint["map_directory"] / native.MAP, "Endpoint map paths disagree.")
    return path.parent


def _read_replay(path, frequency_hz, thickness, anchor, anchor_pin):
    evidence = native._read(path)
    reference, candidate = evidence["reference"], evidence["candidate"]
    baseline = anchor["maps"][f"{'thin' if thickness == 100 else 'thick'}_rim48"]
    match = evidence["mode_match"]
    require(evidence["status"] == "matched_finite_axial_endpoints"
            and reference["frequency_hz"] == UPPER_HZ and candidate["frequency_hz"] == frequency_hz
            and evidence["physical_thickness_nm"] == thickness
            and evidence["provenance"]["published_1GHz_anchor"] == anchor_pin
            and evidence["original_numerical_limits"] == anchor["original_numerical_limits"]
            and all(reference["archive"][key] == baseline["archive"][key] for key in POINT_PINS)
            and _same_mode(reference["native_axial_drive"], baseline["native_axial_drive"])
            and axial.MIN_OVERLAP <= match["mass_overlap"] <= 1
            and match["mass_overlap"] > match["best_competing_overlap"] + axial.K_GAP_TOL,
            f"Expected verified {thickness} nm axial evidence at {frequency_hz:g} Hz against the published anchor.")
    return evidence, _native_pointer(candidate)


def _compact_case(record, replay, directory, limits):
    candidate = replay["candidate"]
    archive = candidate["archive"]
    for key in POINT_PINS[:-1]:
        link = record["inputs"][key]
        require(archive[key] == {**link, "path": (directory / link["path"]).relative_to(ROOT).as_posix()},
                f"Endpoint {key} differs from the native response in {directory}.")
    require(_same_mode(candidate["native_axial_drive"], record)
            and replay["mesh"] == record["mesh"], f"Replayed axial drive or discretization differs in {directory}.")
    flux = native._read(native._verified(archive["independent_flux_replay"], ROOT))
    checks = candidate["independent_flux_checks"]
    require(flux["status"] == "passed_saved_discretization_flux_replay"
            and flux["frequency_hz"] == candidate["frequency_hz"]
            and flux["reference_planes_x_m"] == replay["reference_planes_x_m"]
            and flux["fields_sha256"] == record["inputs"]["fields"]["sha256"]
            and flux["point_summary"] == record["inputs"]["point_summary"]
            and flux["map_checkpoint_sha256"] == archive["checkpoint"]["sha256"]
            and flux["original_rhs_matches_saved_sha256"] is True
            and checks["original_rhs_sha256_matches"] is True
            and 0 <= checks["maximum_nodal_force_relative_residual"]
            < limits["original_equation_and_nodal_force_relative_residuals"]
            and 0 <= checks["coherent_defect_spectral_norm"] < limits["single_drive_and_coherent_power_defect"],
            f"Independent equations and flux evidence differ in {directory}.")
    return {**{key: record[key] for key in MODE_FIELDS},
            "within_thickness_overlap_to_1GHz": replay["mode_match"]["mass_overlap"],
            "native_map": archive["map"], "native_point_summary": archive["point_summary"],
            "independent_flux_replay": archive["independent_flux_replay"]}


def _sample(evidence_paths, frequency_hz, anchor, anchor_pin):
    replays = {role: _read_replay(evidence_paths[role], frequency_hz, thickness, anchor, anchor_pin)
               for role, (thickness, _) in ROLES.items()}
    compared = axial.compare_axial_power(replays["thick"][1], replays["thin"][1], frequency_hz / 1e9)
    require(compared["status"] == "matched", compared.get("reason", "Unresolved cross-thickness axial match."))
    cases = {}
    for role, (thickness, compared_role) in ROLES.items():
        evidence, directory = replays[role]
        record, mesh = compared[compared_role], evidence["mesh"]
        require(record["thickness_nm"] == thickness and mesh["element_order"] == 2
                and mesh["thickness_layers"] == anchor["quadratic_layers"]
                and mesh["mesh_size_nm"] == anchor["central_target_nm"]
                and mesh["lead_mesh_size_nm"] == anchor["lead_target_nm"]
                and mesh["circle_resolution"] == 48 and compared["hole_count"] == evidence["hole_count"] == 26
                and compared["reference_planes_m"] == evidence["reference_planes_x_m"] == anchor["reference_planes_x_m"]
                and compared["incident_side"] == evidence["incident_side"] == anchor["incident_side"],
                f"Unexpected geometry, mesh or reference planes in {directory}.")
        cases[role] = _compact_case(record, evidence, directory, anchor["original_numerical_limits"])
    return {"frequency_hz": frequency_hz, **cases, "cross_thickness_match": compared["match"],
            "thin_minus_thick": {key: cases["thin"][key] - cases["thick"][key] for key in POWERS}}


def _anchor_sample(anchor):
    responses = {}
    for role in ROLES:
        saved = anchor["maps"][f"{role}_rim48"]
        for link in saved["archive"].values():
            native._verified(link, ROOT)
        responses[role] = {**saved["native_axial_drive"], "native_point_summary": saved["archive"]["point_summary"],
                           "independent_flux_replay": saved["archive"]["independent_flux_replay"]}
    match = anchor["axial_power_comparisons"]["thickness_200_to_100_nm_rim48"]
    return {"frequency_hz": UPPER_HZ, **responses,
            "cross_thickness_match": {key: match[key] for key in ("minimum_reciprocal_overlap", "maximum_competing_overlap")},
            "thin_minus_thick": {key: responses["thin"][key] - responses["thick"][key] for key in POWERS}}


def _overlap_bounds(match):
    return {"minimum_reciprocal_overlap": min(match["reference_to_candidate_overlap"],
                                              match["candidate_to_reference_overlap"]),
            "maximum_competing_overlap": max(match["reference_to_candidate_best_competing_overlap"],
                                             match["candidate_to_reference_best_competing_overlap"])}


def _rim32_case(directory, compared, donor_pin, read_point):
    donor_directory = native._verified(donor_pin, ROOT).parent
    donor = native.read_map_checkpoint(donor_directory)
    checkpoint = native.read_map_checkpoint(directory)
    require(checkpoint["preparation_reuse"]["donor_checkpoint_sha256"] == donor_pin["sha256"]
            and checkpoint["shared"] == donor["shared"]
            and checkpoint["inputs"] == {**donor["inputs"], "frequencies_GHz": [LOWER_HZ / 1e9]},
            f"Rim32 preparation or inputs differ from the 1 GHz anchor: {directory}.")
    flux_path = directory.with_name(directory.name + "_flux_AP_pnc.json")
    checked, config, _ = read_point(directory, flux_path, LOWER_HZ / 1e9)
    drives = checked["native_drives"]
    drive = drives[compared["drive"] - 1]
    require(checked["physical_thickness_nm"] == compared["thickness_nm"]
            and config["mesh"] == compared["mesh"] and config["mesh"]["circle_resolution"] == 32
            and drive["native_root_id"] == compared["root"]
            and math.isclose(drive["reflection"], compared["R"], abs_tol=1e-12, rel_tol=0),
            f"The accepted and independently replayed rim32 drives differ in {directory}.")
    return {**{key: compared[key] for key in ("drive", "root", "R")},
            "native_map": checked["archive"]["map"], "native_point_summary": checked["archive"]["point_summary"],
            "independent_flux_replay": checked["archive"]["independent_flux_replay"]}


def _rim_changes(sample48, cases32, cross_thickness_match, rim_matches):
    rim32 = {role: cases32[role]["R"] for role in ROLES}
    rim48 = {role: sample48[role]["R"] for role in ROLES}
    rim32["thin_minus_thick"] = rim32["thin"] - rim32["thick"]
    rim48["thin_minus_thick"] = sample48["thin_minus_thick"]["R"]
    return {"frequency_hz": sample48["frequency_hz"], "rim32_axial": cases32, "rim32_R": rim32, "rim48_R": rim48,
            "rim48_minus_rim32_R_percentage_points": {group: 100 * (rim48[group] - rim32[group]) for group in rim32},
            "rim32_cross_thickness_match": cross_thickness_match, "within_thickness_rim32_to_rim48_matches": rim_matches}


def _rim_control(directories, lower, upper, anchor, read_point):
    frequency = LOWER_HZ / 1e9
    across = axial.compare_axial_power(directories["thick"], directories["thin"], frequency)
    require(across["status"] == "matched" and across["incident_side"] == anchor["incident_side"]
            and across["reference_planes_m"] == anchor["reference_planes_x_m"],
            across.get("reason", "The rim32 pair lacks a matched axial drive at the original planes."))
    cases, matches, upper_cases, upper_matches = {}, {}, {}, {}
    published = anchor["axial_power_comparisons"]
    for role, (thickness, compared_role) in ROLES.items():
        directory = directories[role].resolve()
        map48 = native._verified(lower[role]["native_map"], ROOT).parent
        comparison = axial.compare_axial_power(directory, map48, frequency)
        require(comparison["status"] == "matched", comparison.get("reason", "Unresolved rim32 to rim48 match."))
        require(_same_mode(comparison["reference"], across[compared_role])
                and _same_mode(comparison["candidate"], lower[role])
                and {**comparison["reference"]["mesh"], "circle_resolution": 48} == comparison["candidate"]["mesh"],
                f"The mode or nominal mesh settings differ at {thickness} nm.")
        saved = anchor["maps"][f"{role}_rim32"]
        cases[role] = _rim32_case(directory, across[compared_role], saved["archive"]["checkpoint"], read_point)
        matches[role] = _overlap_bounds(comparison["match"])
        upper_cases[role] = {key: saved["native_axial_drive"][key] for key in ("drive", "root", "R")}
        upper_matches[role] = {key: published[f"rim_32_to_48_at_{thickness}_nm"][key] for key in matches[role]}
    lower_control = _rim_changes(lower, cases, _overlap_bounds(across["match"]), matches)
    upper_cross = {key: published["thickness_200_to_100_nm_rim32"][key] for key in matches["thick"]}
    return {"lower": lower_control, "upper": _rim_changes(upper, upper_cases, upper_cross, upper_matches),
            "acceptance_replay_reader": anchor["provenance"]["acceptance_replay_reader"],
            "scope": "Finite hole-wall and accompanying central remeshing sensitivity at both endpoints. "
                     "The original end planes are retained. These increments give no continuum error bound."}


def _central_donor(central, prior, label):
    if label in prior["maps"]:
        saved = prior["maps"][label]
        require(_same_mode(saved["native_axial_drive"], central["maps"][label]),
                f"The two 1 GHz publications differ for {label}.")
        directory = (ROOT / saved["map_directory"]).resolve()
        pins = saved["map_relative_archive"]
    else:
        require(label == "200_nm_central_225_nm", f"No accepted 1 GHz map for {label}.")
        archive = central["provenance"]["new_thick_central_225_archive"]
        directory = native._verified(archive["map"], ROOT).parent
        pins = {role: {**archive[role], "path": (ROOT / archive[role]["path"]).relative_to(directory).as_posix()}
                for role in ("map", "checkpoint", "point_summary")}
    require(directory.is_relative_to(ROOT), f"Archived 1 GHz map is outside the workspace: {directory}.")
    record, _ = axial._native_point(directory, UPPER_HZ / 1e9)
    require(record["point_status"] == "accepted", f"The 1 GHz donor point is not accepted: {directory}.")
    for pin in pins.values():
        path = (directory / pin["path"]).resolve()
        require(path.is_relative_to(directory) and native._sha(path) == pin["sha256"],
                f"The archived 1 GHz input differs from its publication: {path}.")
    return native._link(directory / native.CHECKPOINT, ROOT)


def _central_row(target, lower, upper, cross_thickness_match, previous_mesh_matches=None):
    row = {"central_target_nm": target, "axial_R_995MHz": lower, "axial_R_1000MHz": upper,
           "995MHz_cross_thickness_match": cross_thickness_match}
    if previous_mesh_matches is not None:
        row["995MHz_matches_to_previous_mesh"] = previous_mesh_matches
    lower_contrast = 100 * (lower["thin"]["R"] - lower["thick"]["R"])
    upper_contrast = 100 * (upper["thin"]["R"] - upper["thick"]["R"])
    row["thin_minus_thick_R_percentage_points"] = {"995MHz": lower_contrast, "1000MHz": upper_contrast,
                                                    "995_minus_1000MHz": lower_contrast - upper_contrast}
    return row


def _central_control(directory, rim, anchor, read_point):
    central = native._read(CENTRAL_ANCHOR)
    prior_pin = central["provenance"]["prior_five_map_result"]
    prior = native._read(native._verified(prior_pin, ROOT))
    targets = [400, 300, 225]
    require(central["status"] == "completed_finite_paired_comparison" and central["frequency_hz"] == UPPER_HZ
            and central["central_targets_nm"] == targets and anchor["central_target_nm"] == targets[0]
            and all(central[key] == anchor[key] for key in ("reference_planes_x_m", "incident_side", "hole_count")),
            "The published 1 GHz central-refinement and rim anchors disagree.")
    rows = [_central_row(targets[0], rim["lower"]["rim32_axial"], rim["upper"]["rim32_axial"],
                         rim["lower"]["rim32_cross_thickness_match"])]
    frequency = LOWER_HZ / 1e9
    for target in targets[1:]:
        paths = {role: (directory / f"{role}{thickness}_central{target}_rim32_995MHz").resolve()
                 for role, (thickness, _) in ROLES.items()}
        paired = axial.compare_axial_power(paths["thick"], paths["thin"], frequency)
        require(paired["status"] == "matched", paired.get("reason", f"Unresolved axial thickness match at {target} nm."))
        cases, upper, matches = {}, {}, {}
        for role, (thickness, compared_role) in ROLES.items():
            label = f"{thickness}_nm_central_{target}_nm"
            selected = paired[compared_role]
            expected_mesh = {**central["common_mesh_except_central_target"], "mesh_size_nm": target}
            old = rows[-1]["axial_R_995MHz"][role]
            old_path = native._verified(old["native_map"], ROOT).parent
            refined = axial.compare_axial_power(old_path, paths[role], frequency)
            require(refined["status"] == "matched" and _same_mode(refined["candidate"], selected)
                    and all(refined["reference"][key] == old[key] for key in ("root", "drive", "R"))
                    and selected["mesh"] == expected_mesh and selected["thickness_nm"] == thickness,
                    f"Unexpected mesh settings or axial mode correspondence at {label}.")
            cases[role] = _rim32_case(paths[role], selected, _central_donor(central, prior, label), read_point)
            upper[role] = {key: central["maps"][label][key] for key in ("drive", "root", "R")}
            matches[role] = _overlap_bounds(refined["match"])
        rows.append(_central_row(target, cases, upper, _overlap_bounds(paired["match"]), matches))
    contrasts = [row["thin_minus_thick_R_percentage_points"] for row in rows]
    return {"hole_wall_segments": 32, "samples": rows,
            "300_to_225_increment_percentage_points": {key: contrasts[-1][key] - contrasts[-2][key]
                                                        for key in contrasts[-1]},
            "thin_reflects_less_at_both_endpoints_across_targets": all(row["995MHz"] < 0 and row["1000MHz"] < 0
                                                                       for row in contrasts),
            "contrast_lower_at_995MHz_across_targets": all(row["995_minus_1000MHz"] < 0 for row in contrasts),
            "published_1GHz_central_anchor": native._link(CENTRAL_ANCHOR, ROOT),
            "prior_1GHz_anchor_inputs": prior_pin,
            "scope": "Finite paired axial reflection on three central meshes and two endpoint frequencies. "
                     "These increments give no continuum error bound or statement about unsampled frequencies."}


def compare_axial_frequency_pair(lower_paths, midpoint_paths=None, rim32_directories=None, central_root=None):
    """Return finite endpoint results with optional midpoint and mesh controls."""
    anchor = native._read(ANCHOR)
    require(anchor["status"] == "completed_finite_paired_comparison" and anchor["frequency_hz"] == UPPER_HZ,
            "Expected the published 1 GHz rim48 anchor.")
    anchor_pin = native._link(ANCHOR, ROOT)
    upper = _anchor_sample(anchor)
    lower = _sample(lower_paths, LOWER_HZ, anchor, anchor_pin)
    groups = (*ROLES, "thin_minus_thick")
    result = {"schema_version": 1, **native.AUTHOR, "status": "matched_finite_axial_power_samples",
              "physical_thickness_nm": {role: thickness for role, (thickness, _) in ROLES.items()},
              "hole_count": 26, "hole_wall_segments": 48, "quadratic_layers": anchor["quadratic_layers"],
              "central_target_nm": anchor["central_target_nm"], "lead_target_nm": anchor["lead_target_nm"],
              "reference_planes_x_m": anchor["reference_planes_x_m"], "incident_side": anchor["incident_side"],
              "original_numerical_limits": anchor["original_numerical_limits"], "lower": lower, "upper": upper,
              "endpoint_change_995_minus_1000_percentage_points": {
                  group: {key: 100 * (lower[group][key] - upper[group][key]) for key in POWERS} for group in groups},
              "scope": "T and R are native incident-power fractions for the matched finite axial drive. Drive and root "
                       "identities are frequency-local. A midpoint residual quantifies discrete departure from the line "
                       "through the two endpoints. Spatial convergence, frequency convergence and continuous branch "
                       "identity remain unresolved. Prior complex-response, all-channel and weak-conversion failures stand.",
              "provenance": {"published_1GHz_anchor": anchor_pin,
                             "lower_within_thickness_replays": {role: native._link(path.resolve(), ROOT)
                                                                for role, path in lower_paths.items()},
                             "paired_frequency_tool": native._link(Path(__file__).resolve(), ROOT),
                             "native_axial_comparator": native._link(Path(axial.__file__).resolve(), ROOT)}}
    if midpoint_paths is not None:
        midpoint = _sample(midpoint_paths, MIDPOINT_HZ, anchor, anchor_pin)
        prediction = {group: {key: (lower[group][key] + upper[group][key]) / 2 for key in POWERS} for group in groups}
        midpoint.update(linear_endpoint_prediction=prediction, observed_minus_linear_percentage_points={
            group: {key: 100 * (midpoint[group][key] - prediction[group][key]) for key in POWERS} for group in groups})
        result["midpoint"] = midpoint
        result["provenance"]["midpoint_within_thickness_replays"] = {
            role: native._link(path.resolve(), ROOT) for role, path in midpoint_paths.items()}
    if rim32_directories is not None:
        reader = native._verified(anchor["provenance"]["acceptance_replay_reader"], ROOT)
        read_point = runpy.run_path(str(reader))["read_point"]
        rim = _rim_control(rim32_directories, lower, upper, anchor, read_point)
        result["rim32_to_rim48_control"] = rim
        if central_root is not None:
            result["rim32_central_refinement"] = _central_control(central_root, rim, anchor, read_point)
    else:
        require(central_root is None, "Central refinement requires the 400 nm rim32 pair.")
    return result


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--study-root", type=Path, help="Local device archive for all three samples and mesh controls.")
    for role in ROLES:
        parser.add_argument(f"--{role}-lower", type=Path, help="Verified 995 MHz endpoint JSON.")
        parser.add_argument(f"--{role}-midpoint", type=Path, help="Verified 997.5 MHz endpoint JSON.")
        parser.add_argument(f"--{role}-rim32", type=Path, help="Accepted 995 MHz rim32 native map directory.")
    parser.add_argument("--central-refinement-root", type=Path, help="Archive directory of the 995 MHz rim32 central controls.")
    parser.add_argument("--output", type=Path, help="Write the compact paired JSON. Defaults to stdout.")
    args = parser.parse_args(argv)
    lower = {role: getattr(args, f"{role}_lower") for role in ROLES}
    midpoint = {role: getattr(args, f"{role}_midpoint") for role in ROLES}
    rim32 = {role: getattr(args, f"{role}_rim32") for role in ROLES}
    if args.study_root is not None:
        if not args.study_root.is_dir():
            parser.error(f"Local pilot archive is unavailable: {args.study_root}")
        checks = args.study_root / "local_checks"
        lower["thick"] = lower["thick"] or checks / "axial_thick_endpoint_995MHz_AP_pnc.json"
        lower["thin"] = lower["thin"] or ROOT / "verification/axial_thin_endpoint_995MHz_AP_pnc.json"
        for role, (thickness, _) in ROLES.items():
            midpoint[role] = midpoint[role] or checks / f"axial_{role}_endpoint_997p5MHz_AP_pnc.json"
            rim32[role] = rim32[role] or args.study_root / f"{role}{thickness}_central400_rim32_995MHz"
    if not all(lower.values()):
        parser.error("Supply the local pilot archive with --study-root or both 995 MHz endpoint records.")
    if (midpoint["thick"] is None) != (midpoint["thin"] is None):
        parser.error("Supply both midpoint endpoint records.")
    if (rim32["thick"] is None) != (rim32["thin"] is None):
        parser.error("Supply both native rim32 maps.")
    try:
        evidence = compare_axial_frequency_pair(lower, midpoint if midpoint["thick"] else None,
                                               rim32 if rim32["thick"] else None,
                                               args.central_refinement_root or args.study_root)
        payload = json.dumps(evidence, indent=2, allow_nan=False)
        if args.output:
            input_paths = {ANCHOR, CENTRAL_ANCHOR,
                           *(path.resolve() for path in (*lower.values(), *midpoint.values()) if path is not None)}
            require(args.output.resolve() not in input_paths, "Choose an output different from the input evidence.")
            temporary = args.output.with_suffix(args.output.suffix + ".tmp")
            temporary.write_text(payload + "\n", encoding="utf-8", newline="\n")
            temporary.replace(args.output)
        else:
            print(payload)
    except FileNotFoundError as exc:
        parser.exit(1, f"{parser.prog}: Local pilot archive file is missing: {exc.filename or exc}\n")
    except (KeyError, OSError, TypeError, ValueError) as exc:
        parser.exit(1, f"{parser.prog}: {exc}\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
