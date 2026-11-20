"""Compare equal incoherent power using accepted native 26-hole maps. Athul Prem | _AP_pnc."""

import argparse
import csv
import json
import math
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT))

from tools import map_scattering_AP_pnc as native
from tools.compare_scattering_AP_pnc import require


STUDY = Path(__file__).resolve().parent
REGIONAL = ROOT / "results/runs/regional_mesh_AP_pnc"
VERIFY = ROOT / "verification"
FREQUENCY_GHZ = 1.0
CHANNELS = 5
POWER_LIMIT = 0.01
PORT_RESIDUAL_LIMIT = 1e-8
FLUX_GRAM_LIMIT_W = 1e-7
TOTALS_ROUNDOFF = 1e-12
POWER_KEYS = ("transmission", "reflection")


def read_json(path):
    return json.loads(path.read_bytes())


def read_csv(path):
    with path.open(newline="", encoding="utf-8") as stream:
        return list(csv.DictReader(stream))


def pin(path, sha256=None):
    record = native._link(path.resolve(), ROOT)
    require(sha256 is None or record["sha256"] == sha256, f"Archived input changed: {path}.")
    return record


def maximum_below(values, limit):
    maximum = max(abs(value) for value in values)
    require(all(math.isfinite(value) and abs(value) < limit for value in values),
            f"A saved numerical check exceeds its original limit of {limit}.")
    return maximum


def read_flux(path, checkpoint, checkpoint_sha256, point, summary, per_drive):
    replay = read_json(path)
    coherent = replay["coherent_left_incidence"]
    shared = checkpoint["shared"]
    equations = summary["checkpointed_scattering"]["original_equations"]
    require(replay["status"] == "passed_saved_discretization_flux_replay"
            and replay["point_summary"] == point["accepted"]["summary"]
            and replay["map_checkpoint_sha256"] == checkpoint_sha256
            and replay["map_source_git_commit"] == checkpoint["source_git_commit"]
            and replay["frequency_hz"] == summary["frequency_hz"]
            and replay["reference_planes_x_m"] == shared["reference_planes_x_m"]
            and replay["fields_sha256"] == summary["output_sha256"][native.FIELDS]
            and replay["original_equations_saved"] == equations
            and replay["original_rhs_matches_saved_sha256"] is True
            and replay["body_dofs"] == summary["ndof"] and replay["port_dofs"] == summary["port_dofs"]
            and replay["stiffness_mass_and_port_section"] == {
                key: shared["artifacts"][key] for key in ("stiffness", "mass", "port_section")}
            and coherent["incident_roots"] == summary["incident_mode_indices"],
            f"Independent flux receipt is from a different saved point: {path}.")
    require(replay["single_drive_power_limit"] == coherent["coherent_power_limit"] == POWER_LIMIT
            and replay["physical_boundary_force_relative_residual_limit"] == native.RESIDUAL_LIMIT
            and coherent["flux_gram_entry_limit_w"] == FLUX_GRAM_LIMIT_W,
            f"Original replay limits differ: {path}.")
    replay_powers = replay["single_drive_powers_in_incident_root_order"]
    powers_by_drive = zip(*(replay_powers[key] for key in
                           ("modal_transmission", "modal_reflection", "direct_transmission", "direct_reflection")),
                         strict=True)
    for row, (modal_t, modal_r, direct_t, direct_r) in zip(per_drive, powers_by_drive, strict=True):
        maximum_below([modal_t - row["transmission"], modal_r - row["reflection"]], TOTALS_ROUNDOFF)
        maximum_below([direct_t - modal_t, direct_r - modal_r, direct_t + direct_r - 1], POWER_LIMIT)
        row.update(independent_direct_transmission=direct_t, independent_direct_reflection=direct_r)
    gram_errors = coherent["flux_gram_errors"]
    return {
        "maximum_nodal_force_relative_residual": maximum_below(
            replay["physical_boundary_force_relative_residuals"], native.RESIDUAL_LIMIT),
        "maximum_single_drive_power_error": maximum_below(
            list(replay["single_drive_power_error_maxima"].values()), POWER_LIMIT),
        "coherent_defect_spectral_norm": maximum_below(
            [coherent["coherent_coefficient_power_defect_spectral_norm"]], POWER_LIMIT),
        "maximum_flux_gram_entry_error_W": maximum_below(
            [error for group in gram_errors.values() for error in group.values()]
            + [coherent["flux_gram_cross_direction_maximum_w"]], FLUX_GRAM_LIMIT_W),
    }


def read_point(directory, flux_path, frequency_ghz):
    directory = directory.resolve()
    published = native.read_saved_map(directory)
    checkpoint = native.read_map_checkpoint(directory)
    checkpoint_pin = pin(directory / native.CHECKPOINT, published["checkpoint_sha256"])
    require(published["status"] == "complete" and published["inputs_sha256"] == checkpoint["inputs_sha256"],
            f"Expected a complete native frequency map: {directory}.")
    point, = (row for row in checkpoint["points"] if row["frequency_GHz"] == frequency_ghz)
    published_point = published["points"][point["index"] - 1]
    require(point["status"] == "accepted" and published_point["accepted"] == point["accepted"],
            f"Native point acceptance differs: {directory}.")
    inputs = checkpoint["inputs"]
    shared = checkpoint["shared"]
    point_dir, summary = native._read_accepted(point, inputs["configuration"], shared, directory)
    config = summary["configuration"]
    require(summary["hole_count"] == len(inputs["hole_pattern"]["holes"]) == 26
            and summary["hole_pattern_provenance"] == inputs["hole_pattern"]
            and read_json(point_dir / "configuration_AP_pnc.json") == config
            and summary["incident_side"] == "left" and summary["boundary"] == "matched_ports"
            and summary["scattering_power_normalization_available"] is True,
            f"Expected matched-port power-normalized left incidence on the 26-hole device: {point_dir}.")
    roots = summary["incident_mode_indices"]
    propagating = {int(row["port_index"]): row for row in read_csv(point_dir / "port_modes_AP_pnc.csv")
                   if row["direction"] == "+x" and row["propagating"] == "True"}
    require(len(roots) == len(propagating) == CHANNELS and set(roots) == propagating.keys()
            and [row["port_index"] for row in published_point["incoming_channels"]] == roots,
            f"Expected all five native +x incident channels: {point_dir}.")

    per_drive = []
    totals = read_csv(point_dir / "scattering_AP_pnc.csv")
    require(len(totals) == CHANNELS, f"Expected power totals for exactly five drives: {point_dir}.")
    for drive, (root, total, channel) in enumerate(zip(roots, totals, published_point["incoming_channels"],
                                                      strict=True), 1):
        k_rad_m = float(propagating[root]["k_real_rad_m"])
        require(int(total["incident_mode_order"]) == drive and float(total["frequency_GHz"]) == frequency_ghz
                and k_rad_m == channel["k_rad_m"] and float(propagating[root]["signed_power_W"]) > 0,
                f"Native drive identity or sign differs: {point_dir}.")
        row = {"drive": drive, "native_root_id": root, "k_rad_m": k_rad_m,
               **{key: float(total[key]) for key in POWER_KEYS}}
        require(all(row[key] >= 0 for key in POWER_KEYS), f"Invalid modal power: {point_dir} drive {drive}.")
        maximum_below([row[key] - channel[key] for key in POWER_KEYS], TOTALS_ROUNDOFF)
        maximum_below([row["transmission"] + row["reflection"] - 1], POWER_LIMIT)
        per_drive.append(row)

    equations = summary["checkpointed_scattering"]["original_equations"]
    checks = {"maximum_" + key: max(row[key] for row in equations["channels"])
              for key in ("relative_residual", "relative_backward_residual")}
    checks["maximum_port_relative_residual"] = maximum_below(
        [summary["maximum_relative_port_residual"]], PORT_RESIDUAL_LIMIT)
    checks["maximum_native_power_error"] = maximum_below(
        [summary[key] for key in ("max_power_imbalance", "maximum_flux_power_imbalance",
                                 "maximum_net_outward_power_ratio", "maximum_modal_flux_difference")], POWER_LIMIT)
    archive = {"map": pin(directory / native.MAP), "checkpoint": checkpoint_pin,
               "point_summary": pin(point_dir / native.RUN_SUMMARY, point["accepted"]["summary"]["sha256"]),
               "map_coefficients": pin(directory / native.TABLE, published["output_sha256"][native.TABLE])}
    for name in ("configuration_AP_pnc.json", "scattering_AP_pnc.csv", "scattering_channels_AP_pnc.csv",
                 "port_modes_AP_pnc.csv", native.FIELDS):
        archive[name] = pin(point_dir / name, summary["output_sha256"][name])
    for name in ("mesh", "port_section"):
        link = shared["artifacts"][name]
        path = native._verified(link, directory)
        archive[name] = {**link, "path": path.relative_to(ROOT).as_posix()}
    record = {"physical_thickness_nm": config["geometry"]["thickness_nm"],
              "mesh_hash": shared["mesh_hash"],
              "source_git_commit": checkpoint["source_git_commit"],
              "map_driver_sha256": inputs["numerical_identity"]["source_sha256"]["tools/map_scattering_AP_pnc.py"],
              "body_dofs": summary["ndof"], "port_dofs": summary["port_dofs"],
              "reference_planes_x_m": shared["reference_planes_x_m"], "native_drives": per_drive,
              "maximum_numerical_checks": checks, "archive": archive}
    record["independent_flux_checks"] = read_flux(
        flux_path, checkpoint, checkpoint_pin["sha256"], point, summary, per_drive)
    archive["independent_flux_replay"] = pin(flux_path)
    return record, config, inputs["hole_pattern"]


def mean_changes(reference_rows, candidate_rows, prefix=""):
    result = {}
    for key in POWER_KEYS:
        reference = math.fsum(row[prefix + key] for row in reference_rows) / len(reference_rows)
        candidate = math.fsum(row[prefix + key] for row in candidate_rows) / len(candidate_rows)
        delta = candidate - reference
        result[key] = {"reference_fraction": reference, "candidate_fraction": candidate,
                       "difference_fraction": delta, "signed_change_percentage_points": 100 * delta,
                       "absolute_change_percentage_points": 100 * abs(delta),
                       "relative_change_percent": 100 * delta / reference if reference else None}
    return result


def unchanged_settings(config):
    return {**{key: value for key, value in config.items() if key not in ("geometry", "name")},
            "geometry": {key: value for key, value in config["geometry"].items() if key != "thickness_nm"}}


def compare(reference_dir, reference_flux, candidate_dir, candidate_flux, geometry_receipt):
    reference, reference_config, reference_holes = read_point(reference_dir, reference_flux, FREQUENCY_GHZ)
    candidate, candidate_config, candidate_holes = read_point(candidate_dir, candidate_flux, FREQUENCY_GHZ)
    require([reference["physical_thickness_nm"], candidate["physical_thickness_nm"]] == [200, 100]
            and unchanged_settings(reference_config) == unchanged_settings(candidate_config)
            and reference_holes == candidate_holes
            and reference["reference_planes_x_m"] == candidate["reference_planes_x_m"],
            "The comparison requires unchanged footprint, material, settings and end planes at 200/100 nm.")
    geometry = read_json(geometry_receipt)
    signed_port = geometry["signed_port_wavenumbers"]
    require(geometry["status"] == signed_port["status"] == "passed"
            and signed_port["frequency_hz"] == FREQUENCY_GHZ * 1e9
            and geometry["exact_ordered_connectivity_and_boundary_tags"] is True
            and all(geometry[key]["mesh_hash"] == record["mesh_hash"]
                    and geometry[key]["mesh_sha256"] == record["archive"]["mesh"]["sha256"]
                    for key, record in (("baseline", reference), ("candidate", candidate)))
            and geometry["reference_checkpoint"]["sha256"] == reference["archive"]["checkpoint"]["sha256"]
            and all(signed_port[field]["sha256"] == candidate["archive"][key]["sha256"]
                    for field, key in (("map_checkpoint", "checkpoint"), ("shared_device_section", "port_section"),
                                       ("device_modal_csv", "port_modes_AP_pnc.csv")))
            and all(signed_port["directions"][direction]["count"] == CHANNELS for direction in ("+x", "-x")),
            "Geometry and signed-port checks must pass for these exact meshes.")
    reusable_config = ROOT / "configs/historical_26hole_regional_100nm_AP_pnc.json"
    require(read_json(reusable_config) == candidate_config, "Stored 100 nm configuration differs from the native point.")
    config_pin = pin(reusable_config, geometry["candidate_configuration"]["sha256"])
    lead_pin = pin(VERIFY / "physical_lead_thickness_1GHz_AP_pnc.json",
                   geometry["standalone_section"]["ledger"]["sha256"])
    controls = {}
    for name in ("regional_thickness_1GHz_AP_pnc.json", "regional_central_1GHz_AP_pnc.json"):
        path = VERIFY / name
        control = read_json(path)
        controls[name] = {"archive": pin(path), **control["comparison"]}
        if name == "regional_central_1GHz_AP_pnc.json":
            require(all(reference["archive"][key]["sha256"] == control["archive"][label]["sha256"]
                        for key, label in (("map", "map"), ("point_summary", "point"),
                                           ("independent_flux_replay", "flux"))),
                    "The 200 nm point differs from the existing central resolution control.")
    reference_rows = reference["native_drives"]
    candidate_rows = candidate["native_drives"]
    return {
        "schema_version": 1, "author": "Athul Prem", "file_tag": "_AP_pnc",
        "frequency_hz": FREQUENCY_GHZ * 1e9, "incident_side": "left", "hole_count": 26,
        "excitation": {"total_incident_power_normalized": 1.0, "power_fraction_per_drive": 1 / CHANNELS,
                       "description": "Equal mutually incoherent power in each of five propagating +x channels. "
                                      "Average the five native total powers independently at each thickness."},
        "reference": reference, "candidate": candidate,
        "common_configuration_except_name_and_thickness": unchanged_settings(reference_config),
        "geometry_and_port_control": {"exact_coordinate_mapping": geometry["exact_coordinate_mapping"],
                                      "exact_ordered_connectivity_and_boundary_tags": True,
                                      "matching_triangles": geometry["matching_triangles"],
                                      "100_nm_signed_k_match_to_same_thickness_lead": {
                                          direction: {key: signed_port["directions"][direction][key] for key in
                                                      ("count", "maximum_abs_difference_rad_m")}
                                          for direction in ("+x", "-x")}},
        "equal_incoherent_modal_power": mean_changes(reference_rows, candidate_rows),
        "equal_incoherent_independent_direct_power": mean_changes(reference_rows, candidate_rows, "independent_direct_"),
        "original_numerical_limits": {"original_equation_relative_and_backward_residual": native.RESIDUAL_LIMIT,
                                      "nodal_force_relative_residual": native.RESIDUAL_LIMIT,
                                      "native_port_relative_residual": PORT_RESIDUAL_LIMIT,
                                      "single_drive_and_coherent_power_defect": POWER_LIMIT,
                                      "flux_gram_entry_W": FLUX_GRAM_LIMIT_W},
        "existing_200_nm_mesh_controls": controls,
        "provenance": {"comparison_driver": pin(Path(__file__)), "geometry_control": pin(geometry_receipt),
                       "independent_lead_thickness": lead_pin, "reusable_100_nm_configuration": config_pin},
        "scope": "Finite, linear, lossless, coarse two-layer P2 responses at 1 GHz with nominal fixed-property "
                 "germanium and the original physical end planes. Native root IDs and complex coefficients remain "
                 "specific to each thickness. The equal-power means require no cross-thickness channel pairing.",
        "limits": "Passing equation and flux checks establishes finite numerical acceptance. Existing 200 nm mesh "
                  "controls still fail their original complex-increment criteria. Spatial and frequency convergence, "
                  "continuum mode completeness and experimental validation remain unestablished. Relative changes "
                  "use the 200 nm mean as denominator and carry no relative accuracy guarantee for weak reflection.",
    }


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--reference", type=Path, default=REGIONAL / "central300_z2_1GHz")
    parser.add_argument("--reference-flux", type=Path, default=REGIONAL / "local_checks/central300_flux_AP_pnc.json")
    parser.add_argument("--candidate", type=Path, default=STUDY / "thin100_1GHz")
    parser.add_argument("--candidate-flux", type=Path, default=STUDY / "thin100_flux_AP_pnc.json")
    parser.add_argument("--geometry", type=Path, default=STUDY / "geometry_AP_pnc.json")
    parser.add_argument("--output", type=Path, default=VERIFY / "physical_device_thickness_1GHz_AP_pnc.json")
    args = parser.parse_args()
    result = compare(args.reference, args.reference_flux, args.candidate, args.candidate_flux, args.geometry)
    with args.output.open("x", encoding="utf-8", newline="\n") as stream:
        json.dump(result, stream, indent=2, allow_nan=False)
        stream.write("\n")
