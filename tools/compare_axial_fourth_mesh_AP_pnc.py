#!/usr/bin/env python3
"""Compare the fourth central mesh against the finite axial pilot. Author: Athul Prem | _AP_pnc."""

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
from tools import compare_axial_frequency_pair_AP_pnc as pair
from tools import map_scattering_AP_pnc as native
from tools.compare_axial_layer_frequency_AP_pnc import compare_layer_frequency
from tools.compare_scattering_AP_pnc import require


PILOT = ROOT / "verification/axial_device_frequency_pair_AP_pnc.json"
TARGET_NM = 168.75
SAMPLES = {"995MHz": pair.LOWER_HZ, "1GHz": pair.UPPER_HZ}
ENDPOINT_CHANGE = "995_minus_1000MHz"
CONTRAST = "thin_minus_thick_R_percentage_points"
TARGET_FRACTION = 0.10


def _native_mode_agrees(first, second):
    return (all(first[key] == second[key] for key in ("drive", "root"))
            and math.isclose(first["R"], second["R"], abs_tol=1e-12, rel_tol=0))


def _prior_map(central, prior, level_225, role, sample):
    if sample == "995MHz":
        saved = level_225["axial_R_995MHz"][role]
        return native._verified(saved["native_map"], ROOT).parent, saved
    thickness, _ = pair.ROLES[role]
    link = pair._central_donor(central, prior, f"{thickness}_nm_central_225_nm")
    return (ROOT / link["path"]).parent, level_225["axial_R_1000MHz"][role]


def _execution(directory, sample, checkpoint):
    phase = f"response_{sample}"
    path = directory.parent / "local_checks" / f"{directory.name}_{phase}_execution_AP_pnc.json"
    receipt = native._read(path)
    argv = receipt["argv"]
    config = (ROOT / argv[argv.index("--config") + 1]).resolve()
    require(receipt["phase"] == phase and receipt["exit_code"] == 0 and receipt["guard"] is None
            and (ROOT / argv[argv.index("--output") + 1]).resolve() == directory
            and [float(value) for value in argv[argv.index("--frequencies-ghz") + 1:argv.index("--frequencies-ghz") + 3]]
            == checkpoint["inputs"]["frequencies_GHz"]
            and argv[argv.index("--max-new-points") + 1] == "1"
            and config == directory.with_name(f"{directory.name}_configuration_AP_pnc.json")
            and native._sha(config) == receipt["configuration_sha256"]
            and receipt["source_revision"] == checkpoint["source_git_commit"]
            and receipt["host_platform"] == checkpoint["inputs"]["numerical_identity"]["platform"]
            and all(receipt["thread_environment"][key] == value
                    for key, value in checkpoint["inputs"]["thread_environment"].items())
            and receipt["elapsed_seconds"] <= receipt["wall_seconds"]
            and receipt["peak_sampled_process_tree_private_sum_bytes"] <= receipt["private_limit_bytes"],
            f"The successful guarded response receipt differs from the saved map: {path}.")
    return {"receipt": native._link(path, ROOT), "elapsed_seconds": receipt["elapsed_seconds"],
            "peak_sampled_process_tree_private_sum_bytes": receipt["peak_sampled_process_tree_private_sum_bytes"]}


def _verify_sample(directory, selected, sample, frequency, mesh, reader, replay_sources):
    flux_path = directory.with_name(f"{directory.name}_{sample}_flux_AP_pnc.json")
    accepted, config, _ = reader(directory, flux_path, frequency)
    drive = accepted["native_drives"][selected["drive"] - 1]
    require(config["mesh"] == mesh and drive["native_root_id"] == selected["root"]
            and drive["k_rad_m"] == selected["k_rad_m"]
            and all(math.isclose(drive[key], selected[short], abs_tol=1e-12, rel_tol=0)
                    for key, short in (("reflection", "R"), ("transmission", "T"))),
            f"Native axial response differs from the independently replayed drive in {directory} at {sample}.")
    replay = native._read(flux_path)
    require([replay[key] for key in ("replay_source_sha256", "regional_replay_source_sha256")] == replay_sources,
            f"Independent flux producers differ from those used in the accepted pilot: {flux_path}.")
    _, record = axial._native_point(directory, frequency)
    return {**{key: selected[key] for key in pair.MODE_FIELDS},
            "original_equation_identity": {key: replay["original_equations_saved"][key]
                                           for key in ("operator_csc_sha256", "rhs_sha256")},
            "original_numerical_checks": accepted["maximum_numerical_checks"],
            "independent_flux_checks": accepted["independent_flux_checks"],
            "guarded_response": _execution(directory, sample, record[2]),
            "archive": {key: accepted["archive"][key] for key in pair.POINT_PINS}}, record


def _endpoint_mode_match(directory, endpoints):
    points, indices = [], []
    for selected, point in endpoints:
        saved = axial._read_case(directory, {"inputs": {}}, point)
        matches = [i for i, channel in enumerate(saved.channels) if _native_mode_agrees(channel, selected)]
        require(len(matches) == 1, f"The saved axial drive differs from its accepted response in {directory}.")
        points.append(saved)
        indices.append(matches[0])
    first, second = points
    i, j = indices
    overlap, competing = axial._match_evidence(axial._overlap(first, second), i, j)
    reverse, reverse_competing = axial._match_evidence(axial._overlap(second, first), j, i)
    require(first.section.discretization_sha256 == second.section.discretization_sha256
            and min(overlap, reverse) >= axial.MIN_OVERLAP
            and overlap > competing + axial.K_GAP_TOL and reverse > reverse_competing + axial.K_GAP_TOL,
            f"The two frequencies lack a unique axial overlap on the same port section in {directory}.")
    require(all(endpoints[0][0]["original_equation_identity"][key] != endpoints[1][0]["original_equation_identity"][key]
                for key in endpoints[0][0]["original_equation_identity"]),
            f"The endpoints did not retain distinct original scattering equations in {directory}.")
    return {"minimum_reciprocal_overlap": min(overlap, reverse),
            "maximum_competing_overlap": max(competing, reverse_competing)}


def compare_fourth_mesh(study_root, *, include_layer_control=False):
    study_root = Path(study_root).resolve()
    require(study_root.is_dir(), f"Local fourth-mesh archive is unavailable: {study_root}.")
    pilot = native._read(PILOT)
    central_control = pilot["rim32_central_refinement"]
    rows = central_control["samples"]
    require([row["central_target_nm"] for row in rows] == [400, 300, 225]
            and central_control["hole_wall_segments"] == 32 and pilot["lead_target_nm"] == 150
            and pilot["quadratic_layers"] == 2 and pilot["hole_count"] == 26,
            "Expected the frozen three-mesh axial pilot on the two-layer rim32 device.")
    central = native._read(native._verified(central_control["published_1GHz_central_anchor"], ROOT))
    prior = native._read(native._verified(central_control["prior_1GHz_anchor_inputs"], ROOT))
    rim_anchor = native._read(native._verified(pilot["provenance"]["published_1GHz_anchor"], ROOT))
    source_pins = rim_anchor["provenance"]["independent_flux_sources"]
    for link in source_pins:
        native._verified(link, ROOT)
    replay_sources = [link["sha256"] for link in source_pins]
    replay_reader = pilot["rim32_to_rim48_control"]["acceptance_replay_reader"]
    reader = runpy.run_path(str(native._verified(replay_reader, ROOT)))["read_point"]
    mesh = {**central["common_mesh_except_central_target"], "mesh_size_nm": TARGET_NM}
    geometry_path = study_root / "local_checks/central168p75_geometry_AP_pnc.json"
    geometry = native._read(geometry_path)
    require(geometry["status"] == "passed" and geometry["central_targets_nm"] == [225, TARGET_NM]
            and geometry["lead_target_nm"] == pilot["lead_target_nm"]
            and geometry["quadratic_thickness_layers"] == pilot["quadratic_layers"]
            and geometry["rim_segments_per_hole_per_plane"] == central_control["hole_wall_segments"]
            and geometry["reference_planes_x_m"] == pilot["reference_planes_x_m"]
            and geometry["exact_paired_ordered_connectivity_and_tags"] is True,
            "The preflight geometry check differs from the required fourth-mesh pair.")
    directories = {role: (study_root / f"{role}{thickness}_central168p75_rim32").resolve()
                   for role, (thickness, _) in pair.ROLES.items()}
    for role, (thickness, _) in pair.ROLES.items():
        state = native.read_map_checkpoint(directories[role])
        shared = state["shared"]
        measured = geometry["cases"][f"{role}{thickness}"]["candidate"]
        require(state["inputs"]["frequencies_GHz"] == [value / 1e9 for value in SAMPLES.values()]
                and shared["mesh_hash"] == measured["mesh_hash"]
                and shared["artifacts"]["mesh"]["sha256"] == measured["mesh_sha256"],
                f"The map frequencies or checked body mesh differ in {directories[role]}.")

    samples, endpoints = {}, {role: [] for role in pair.ROLES}
    for sample, frequency_hz in SAMPLES.items():
        frequency = frequency_hz / 1e9
        cross = axial.compare_axial_power(directories["thick"], directories["thin"], frequency)
        require(cross["status"] == "matched" and cross["reference_planes_m"] == pilot["reference_planes_x_m"]
                and cross["incident_side"] == pilot["incident_side"] and cross["hole_count"] == pilot["hole_count"],
                cross.get("reason", f"Axial pairing or geometry at {sample} differs from the pilot."))
        cases, refined = {}, {}
        for role, (thickness, compared_role) in pair.ROLES.items():
            selected = cross[compared_role]
            original, saved = _prior_map(central, prior, rows[-1], role, sample)
            same_thickness = axial.compare_axial_power(original, directories[role], frequency)
            require(same_thickness["status"] == "matched" and selected["thickness_nm"] == thickness
                    and _native_mode_agrees(same_thickness["reference"], saved)
                    and pair._same_mode(same_thickness["candidate"], selected),
                    f"Unresolved axial correspondence to the 225 nm mesh for {role} at {sample}.")
            refined[role] = pair._overlap_bounds(same_thickness["match"])
            cases[role], point = _verify_sample(directories[role], selected, sample, frequency, mesh, reader, replay_sources)
            endpoints[role].append((cases[role], point))
        samples[sample] = {"frequency_hz": frequency_hz, **cases,
                           "cross_thickness_match": pair._overlap_bounds(cross["match"]),
                           "matches_to_225_nm": refined}

    current = {"995MHz": 100 * (samples["995MHz"]["thin"]["R"] - samples["995MHz"]["thick"]["R"]),
               "1000MHz": 100 * (samples["1GHz"]["thin"]["R"] - samples["1GHz"]["thick"]["R"])}
    current[ENDPOINT_CHANGE] = current["995MHz"] - current["1000MHz"]
    last = {key: current[key] - rows[-1][CONTRAST][key] for key in current}
    previous = {key: rows[-1][CONTRAST][key] - rows[-2][CONTRAST][key] for key in current}
    target = TARGET_FRACTION * abs(rows[-1][CONTRAST][ENDPOINT_CHANGE])
    contrasts = [row[CONTRAST] for row in rows] + [current]
    result = {"schema_version": 1, **native.AUTHOR, "status": "completed_finite_fourth_mesh_comparison",
            "central_target_nm": TARGET_NM, "mesh": mesh, "samples": samples,
            "within_thickness_endpoint_matches": {role: _endpoint_mode_match(directories[role], endpoints[role])
                                                   for role in pair.ROLES},
            "reference_planes_x_m": pilot["reference_planes_x_m"], "incident_side": pilot["incident_side"],
            "original_numerical_limits": pilot["original_numerical_limits"],
            "contrast_series": [{"central_target_nm": size, CONTRAST: values}
                                for size, values in zip((400, 300, 225, TARGET_NM), contrasts, strict=True)],
            "negative_contrast_at_both_endpoints_on_all_meshes": all(
                values["995MHz"] < 0 and values["1000MHz"] < 0 for values in contrasts),
            "contrast_more_negative_at_995MHz_on_all_meshes": all(values[ENDPOINT_CHANGE] < 0 for values in contrasts),
            "signed_300_to_225_increments_percentage_points": previous,
            "signed_225_to_168p75_increments_percentage_points": last,
            "absolute_endpoint_contrast_change_increments_percentage_points": {
                "300_to_225": abs(previous[ENDPOINT_CHANGE]), "225_to_168p75": abs(last[ENDPOINT_CHANGE])},
            "latest_absolute_increment_is_smaller": abs(last[ENDPOINT_CHANGE]) < abs(previous[ENDPOINT_CHANGE]),
            "exploratory_endpoint_contrast_change_target": {"fraction_of_225_nm_change": TARGET_FRACTION,
                                                          "percentage_points": target,
                                                          "met": abs(last[ENDPOINT_CHANGE]) <= target},
            "scope": "Finite matched axial reflection at two sampled frequencies. The exploratory target concerns the "
                     "last mesh increment. Continuum error, frequency convergence and the broader complex response "
                     "remain unresolved.",
            "provenance": {"frozen_three_mesh_pilot": native._link(PILOT, ROOT),
                           "checked_geometry": native._link(geometry_path, ROOT),
                           "acceptance_reader": replay_reader, "independent_flux_sources": source_pins,
                           "comparison_source": native._link(Path(__file__).resolve(), ROOT)}}
    if include_layer_control:
        layers = compare_layer_frequency(study_root, pilot, reader, replay_sources)
        layer_step = layers["signed_endpoint_change_increments_percentage_points"]["2_to_3"]
        central_step = last[ENDPOINT_CHANGE]
        layers["last_layer_vs_last_central_increment"] = {
            "two_to_three_layers_at_central_400_nm_percentage_points": layer_step,
            "central_225_to_168p75_nm_at_two_layers_percentage_points": central_step,
            "opposite_nonzero_signs": layer_step * central_step < 0,
            "absolute_layer_to_central_increment_ratio": abs(layer_step / central_step) if central_step else None,
            "scope": "The two measured increments use different fixed mesh settings. "
                     "Their coupled mesh sensitivity remains unresolved.",
        }
        result["through_thickness_control"] = layers
    return result


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--study-root", type=Path, required=True, help="Local device run archive.")
    parser.add_argument("--include-layer-control", action="store_true", help="Include the fixed-400 nm layer control.")
    parser.add_argument("--output", type=Path, help="Write the fourth-pair verification JSON. Defaults to stdout.")
    args = parser.parse_args(argv)
    try:
        result = compare_fourth_mesh(args.study_root, include_layer_control=args.include_layer_control)
        payload = json.dumps(result, indent=2, allow_nan=False)
        if args.output:
            require(args.output.resolve() not in (PILOT, pair.CENTRAL_ANCHOR, pair.ANCHOR),
                    "Choose a separate result path for the fourth-mesh comparison.")
            temporary = args.output.with_suffix(args.output.suffix + ".tmp")
            temporary.write_text(payload + "\n", encoding="utf-8", newline="\n")
            temporary.replace(args.output)
        else:
            print(payload)
    except (KeyError, OSError, TypeError, ValueError) as exc:
        parser.exit(1, f"{parser.prog}: {exc}\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
