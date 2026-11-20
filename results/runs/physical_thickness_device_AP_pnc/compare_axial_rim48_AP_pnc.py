"""Replay the 1 GHz axial-drive hole-wall refinement. Athul Prem | _AP_pnc."""

import argparse
import json
import math
from pathlib import Path

from compare_thickness_AP_pnc import FREQUENCY_GHZ, ROOT, STUDY, VERIFY, native, pin, read_json, read_point, require
from tools.compare_axial_device_power_AP_pnc import LIMITS, compare_axial_power


MAPS = {"rim32": "thin100_central400", "rim48": "thin100_central400_rim48"}
GEOMETRY = STUDY / f"{MAPS['rim48']}_geometry_AP_pnc.json"
PRIOR = VERIFY / "axial_device_three_layer_pairs_1GHz_AP_pnc.json"
OUTPUT = VERIFY / "axial_thin_rim_refinement_1GHz_AP_pnc.json"
AXIAL_FIELDS = ("drive", "root", "k_rad_m", "axial_kinetic_energy_fraction", "T", "R")
SECTION_ARRAYS = {"points_m", "triangles", "K0", "M", "Q", "G", "reference_speed_m_s"}
FLUX_SOURCES = {
    "replay_source_sha256": ROOT / "tools/replay_lawrence_device_point_AP_pnc.py",
    "regional_replay_source_sha256": STUDY.parent / "regional_mesh_AP_pnc/local_checks/replay_regional_flux_AP_pnc.py",
}


def require_pin(record):
    return pin(ROOT / record["path"], record["sha256"])


def check_geometry(geometry, label, case, config, config_pin, checkpoint):
    row = geometry["cases"][label]
    archive = case["archive"]
    require(row["map_checkpoint_path"] == archive["checkpoint"]["path"]
            and row["map_inputs_sha256"] == checkpoint["inputs_sha256"]
            and row["shared_sha256"] == native.workflow.digest(checkpoint["shared"])
            and row["mesh"]["mesh_hash"] == case["mesh_hash"]
            and row["mesh"]["mesh_sha256"] == archive["mesh"]["sha256"]
            and require_pin(row["configuration"]) == config_pin
            and require_pin(row["port_section"]) == archive["port_section"]
            and row["port"]["saved_section_matches_both_tagged_P2_body_traces"] is True
            and row["rim_segments_per_hole_per_plane"] == config["mesh"]["circle_resolution"]
            and all(row["regional_tetrahedra"][key] == count
                    for key, count in geometry["exact_shared_lead_tetrahedra"].items())
            and geometry["exact_shared_cut_port_triangles"] == row["trace_triangles"],
            f"Geometry receipt differs from the accepted {label} map, port traces or rim resolution.")


def check_execution(case, reference_config_pin, candidate_config_pin):
    receipts = {}
    for phase in ("mesh", "preparation", "response", "flux"):
        path = STUDY / "local_checks" / f"{MAPS['rim48']}_{phase}_execution_AP_pnc.json"
        row = read_json(path)
        config_pin = reference_config_pin if phase == "mesh" else candidate_config_pin
        require(row["phase"] == phase and row["exit_code"] == 0 and row["guard"] is None
                and row["source_revision"] == case["source_git_commit"]
                and row["configuration_sha256"] == config_pin["sha256"]
                and row["elapsed_seconds"] <= row["wall_seconds"]
                and row["min_sampled_commit_available_bytes"] >= row["commit_reserve_bytes"]
                and row["min_sampled_physical_available_bytes"] >= row["available_reserve_bytes"]
                and row["peak_sampled_process_tree_rss_sum_bytes"] <= row["resident_limit_bytes"]
                and row["peak_sampled_process_tree_private_sum_bytes"] <= row["private_limit_bytes"],
                f"The {phase} execution receipt differs from the archived run or its guards.")
        if phase == "mesh":
            argv = row["argv"]
            require("--mesh-only" in argv and argv[argv.index("--circle-resolution") + 1] == "48"
                    and Path(argv[argv.index("--config") + 1]).as_posix() == config_pin["path"],
                    "The preflight must override the original configuration to 48 hole-wall segments.")
        receipts[phase] = pin(path)
        if phase == "response":
            require(row["admission"]["ndof"] == case["body_dofs"], "The admitted system differs from the solved system.")
            resources = {key: row[key] for key in (
                "host_platform", "thread_environment", "elapsed_seconds", "sample_interval_seconds",
                "peak_sampled_process_tree_rss_sum_bytes", "peak_sampled_process_tree_private_sum_bytes",
                "min_sampled_commit_available_bytes", "min_sampled_physical_available_bytes",
            )}
            resources.update({key: row["admission"][key] for key in ("actual_K_nonzeros", "actual_M_nonzeros")})
    return {"guarded_execution_receipts": receipts, "response_resources": resources}


def compare():
    prior, geometry = read_json(PRIOR), read_json(GEOMETRY)
    require(prior["status"] == "completed_finite_paired_comparison" and geometry["status"] == "passed"
            and prior["frequency_hz"] == geometry["frequency_hz"] == FREQUENCY_GHZ * 1e9
            and geometry["reference_planes_x_m"] == prior["reference_planes_x_m"]
            and set(geometry["identical_ordered_section_arrays"]) == SECTION_ARRAYS,
            "The prior thickness contrast and geometry control must reproduce the accepted 1 GHz setup.")
    comparator = require_pin(prior["provenance"]["public_comparator"])
    reader = require_pin(prior["provenance"]["acceptance_replay_reader"])
    geometry_sources = [require_pin(geometry[key]) for key in (
        "verification_script", "regional_partition_script", "hole_wall_extraction_script",
    )]
    hole_pin = require_pin(geometry["hole_pattern_csv"])
    require(hole_pin["sha256"] == prior["hole_source_sha256"], "The original 26-hole table changed.")

    maps = {}
    cases = {}
    directories = {}
    configurations = {}
    numerical_identities = {}
    hole_patterns = {}
    flux_sources = {key: pin(path) for key, path in FLUX_SOURCES.items()}
    for label, name in MAPS.items():
        directory = STUDY / f"{name}_1GHz"
        config_path = STUDY / f"{name}_configuration_AP_pnc.json"
        flux_path = STUDY / f"{name}_flux_AP_pnc.json"
        case, config, holes = read_point(directory, flux_path, FREQUENCY_GHZ)
        checkpoint = native.read_map_checkpoint(directory)
        config_pin = pin(config_path)
        require(config == read_json(config_path)
                and holes["source_sha256"] == hole_pin["sha256"] and holes["source_csv"] == hole_pin["path"]
                and case["reference_planes_x_m"] == prior["reference_planes_x_m"]
                and case["physical_thickness_nm"] == 100 and config["mesh"]["element_order"] == 2
                and config["mesh"]["thickness_layers"] == 2 and config["mesh"]["circle_resolution"] == int(label[3:])
                and config["mesh"]["mesh_size_nm"] == prior["central_target_nm"] == 400
                and config["mesh"]["lead_mesh_size_nm"] == prior["lead_target_nm"] == 150
                and case["body_dofs"] <= config["resources"]["max_dofs"]
                and case["port_dofs"] <= config["resources"]["max_port_dofs"],
                f"Unexpected physical settings or resource limits for {label}.")
        flux = read_json(flux_path)
        require(all(flux[key] == source["sha256"] for key, source in flux_sources.items()),
                f"The archived {label} independent flux replay sources changed.")
        check_geometry(geometry, label, case, config, config_pin, checkpoint)
        maps[label] = {
            "map_directory": directory.relative_to(ROOT).as_posix(), "configuration": config_pin,
            "rim_segments_per_hole": config["mesh"]["circle_resolution"],
            "incident_channels_checked": len(case["native_drives"]),
            **{key: case[key] for key in ("body_dofs", "port_dofs", "source_git_commit", "map_driver_sha256",
                                         "maximum_numerical_checks", "independent_flux_checks")},
            "archive": {key: case["archive"][key] for key in (
                "map", "checkpoint", "point_summary", "mesh", "port_section", "independent_flux_replay",
            )},
        }
        cases[label], directories[label], configurations[label] = case, directory, config
        hole_patterns[label] = holes
        numerical_identities[label] = checkpoint["inputs"]["numerical_identity"]
    reference, candidate = configurations.values()
    require(candidate == {**reference, "name": candidate["name"],
                          "mesh": {**reference["mesh"], "circle_resolution": 48},
                          "resources": {**reference["resources"], "max_dofs": candidate["resources"]["max_dofs"]}}
            and hole_patterns["rim32"] == hole_patterns["rim48"],
            "The rim comparison requires the same nominal hole pattern and physical settings.")
    require(numerical_identities["rim32"] == numerical_identities["rim48"],
            "The two native maps must use the same numerical source and runtime identities.")

    old_thin = prior["maps"]["thin_z2"]
    baseline = maps["rim32"]
    require(baseline["map_directory"] == old_thin["map_directory"]
            and all(baseline["archive"][key]["sha256"] == old_thin[f"{key}_sha256"] for key in ("map", "checkpoint")),
            "The accepted 32-segment baseline must be the thin map from the earlier thickness contrast.")
    result = compare_axial_power(directories["rim32"], directories["rim48"], FREQUENCY_GHZ)
    require(result["status"] == "matched", result.get("reason", "The native axial modes failed to match."))
    for label, role in zip(MAPS, ("reference", "candidate"), strict=True):
        axial = {key: result[role][key] for key in AXIAL_FIELDS}
        direct = cases[label]["native_drives"][axial["drive"] - 1]
        require(axial["root"] == direct["native_root_id"] and axial["k_rad_m"] == direct["k_rad_m"],
                f"The independent axial flux drive differs from the matched mode for {label}.")
        maps[label]["native_axial_drive"] = axial
        maps[label]["independent_axial_flux"] = {key: direct[f"independent_direct_{name}"]
                                                  for key, name in (("T", "transmission"), ("R", "reflection"))}
    require(all(math.isclose(baseline["native_axial_drive"][key], old_thin["native_axial_drive"][key],
                             rel_tol=0, abs_tol=1e-12) for key in AXIAL_FIELDS),
            "The matched axial drive must reproduce the accepted 32-segment response.")
    old_comparison = prior["axial_power_comparisons"]["thickness_200_to_100_nm_z2"]
    contrast = old_comparison["power_changes"]["R"]["signed_change_percentage_points"]
    require((old_comparison["reference"], old_comparison["candidate"]) == ("thick_z2", "thin_z2")
            and math.isfinite(contrast) and contrast != 0,
            "Expected the nonzero 32-segment two-layer thin-minus-thick reflection contrast.")
    changes = {key: {"signed_change_percentage_points": row["percentage_points"],
                     "relative_change_percent": row["relative_change_percent"]} for key, row in result["delta"].items()}
    absolute_rim_change = abs(changes["R"]["signed_change_percentage_points"])
    return {
        "schema_version": 1, "author": "Athul Prem", "file_tag": "_AP_pnc", "status": "completed_finite_rim_refinement",
        "frequency_hz": result["frequency_hz"], "incident_side": result["incident_side"], "hole_count": result["hole_count"],
        "physical_thickness_nm": 100, "quadratic_layers": 2, "central_target_nm": 400, "lead_target_nm": 150,
        "reference_planes_x_m": result["reference_planes_m"], "maps": maps, "mode_match": result["match"],
        "geometry_checks": {key: geometry[key] for key in (
            "exact_shared_lead_tetrahedra", "exact_shared_cut_port_triangles", "identical_ordered_section_arrays",
            "central_tetrahedra", "exact_shared_central_tetrahedra",
        )},
        "power_changes_32_to_48_segments": changes,
        "reflection_comparison_for_scale": {
            "rim_change_absolute_percentage_points": absolute_rim_change,
            "earlier_thin_minus_thick_signed_percentage_points": contrast,
            "absolute_rim_change_over_absolute_thickness_contrast": absolute_rim_change / abs(contrast),
        },
        "original_numerical_limits": prior["original_numerical_limits"],
        "execution": check_execution(cases["rim48"], maps["rim32"]["configuration"], maps["rim48"]["configuration"]),
        "provenance": {
            "replay": pin(Path(__file__)), "public_comparator": comparator, "acceptance_replay_reader": reader,
            "prior_thickness_comparison": pin(PRIOR), "geometry": pin(GEOMETRY), "hole_pattern_csv": hole_pin,
            "geometry_sources": geometry_sources, "independent_flux_sources": list(flux_sources.values()),
        },
        "scope": "T and R are native power fractions for the matched axial drive. Changes subtract the 32-segment "
                 "response from the 48-segment response and use the 32-segment power as the relative baseline. "
                 "The changed segmentation remeshes the central region. The earlier thickness contrast "
                 "provides scale only. The 200 nm device has no 48-segment result, so its paired contrast is unresolved.",
        "limits": LIMITS,
    }


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=OUTPUT)
    args = parser.parse_args()
    result = compare()
    with args.output.open("x", encoding="utf-8", newline="\n") as stream:
        json.dump(result, stream, indent=2, allow_nan=False)
        stream.write("\n")
