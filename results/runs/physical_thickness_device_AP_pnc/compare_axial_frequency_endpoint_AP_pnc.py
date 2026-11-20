"""Replay a same-discretization axial endpoint comparison. Athul Prem | _AP_pnc."""

import argparse
import json
import math
import os
from pathlib import Path

os.environ.update(dict.fromkeys(
    ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "BLIS_NUM_THREADS", "NUMEXPR_NUM_THREADS"), "1"
))

import numpy as np

import compare_thickness_AP_pnc as accepted
from compare_axial_rim48_AP_pnc import FLUX_SOURCES
from compare_thickness_AP_pnc import ROOT, VERIFY, native, pin, read_csv, read_json, require
from tools import compare_axial_device_power_AP_pnc as axial
from tools import replay_lawrence_device_point_AP_pnc as flux


PRIOR = VERIFY / "axial_device_rim_pairs_1GHz_AP_pnc.json"
SEEDER = ROOT / "tools/seed_frequency_map_AP_pnc.py"
ARTIFACTS = ("mesh_summary", "mesh", "stiffness", "mass", "port_section")
LIMITS = {
    "original_equation_and_nodal_force_relative_residuals": native.RESIDUAL_LIMIT,
    "native_port_relative_residual": accepted.PORT_RESIDUAL_LIMIT,
    "single_drive_and_coherent_power_defect": accepted.POWER_LIMIT,
    "flux_gram_entry_W": accepted.FLUX_GRAM_LIMIT_W,
}


def load_endpoint(directory, frequency, flux_path, flux_sources):
    case, config, holes = accepted.read_point(directory, flux_path, frequency)
    record, selected = axial._native_point(directory, frequency)
    saved = axial._read_case(directory, record, selected)
    _, _, checkpoint = selected
    replay = read_json(flux_path)
    checks = case["independent_flux_checks"]
    require(all(replay[key] == source["sha256"] for key, source in flux_sources.items())
            and set(replay["single_drive_power_error_maxima"]) == set(flux.POWER_CHECKS)
            and len(replay["physical_boundary_force_relative_residuals"]) == len(saved.channels)
            and len(replay["rhs_trace_reconstruction_relative_differences"]) == len(saved.channels),
            f"The independent replay source or all-drive checks differ for {directory}.")
    checks["original_rhs_sha256_matches"] = replay["original_rhs_matches_saved_sha256"]
    checks["maximum_rhs_trace_reconstruction_relative_error"] = accepted.maximum_below(
        replay["rhs_trace_reconstruction_relative_differences"], native.RESIDUAL_LIMIT)
    superposition = replay["coherent_left_incidence"]["direct_superposition"]
    require(set(superposition["power_errors"]) == set(flux.POWER_CHECKS)
            and math.isclose(superposition["incident_power_w"], 1, rel_tol=0, abs_tol=accepted.FLUX_GRAM_LIMIT_W),
            f"The direct coherent flux checks are incomplete for {directory}.")
    checks["maximum_direct_superposition_power_error"] = accepted.maximum_below(
        list(superposition["power_errors"].values()), accepted.POWER_LIMIT)

    port_csv = ROOT / case["archive"]["port_modes_AP_pnc.csv"]["path"]
    mode_rows = {int(row["port_index"]): row for row in read_csv(port_csv)}
    for mode, row in zip(saved.channels, case["native_drives"], strict=True):
        signed = mode_rows[mode["root"]]
        power = float(signed["signed_power_W"])
        require((mode["drive"], mode["root"], mode["k_rad_m"]) ==
                (row["drive"], row["native_root_id"], row["k_rad_m"])
                and signed["direction"] == "+x" and signed["propagating"] == "True"
                and math.isclose(power, 1., rel_tol=0, abs_tol=accepted.FLUX_GRAM_LIMIT_W)
                and all(math.isclose(mode[key], row[name], rel_tol=0, abs_tol=accepted.TOTALS_ROUNDOFF)
                        for key, name in (("T", "transmission"), ("R", "reflection"))),
                f"The saved signed modal drive differs for {directory}.")
        mode.update(signed_power_W=power, direction=signed["direction"])
    endpoint = {
        "frequency_hz": frequency * 1e9,
        "map_directory": directory.relative_to(ROOT).as_posix(),
        "inputs_sha256": checkpoint["inputs_sha256"],
        "source_git_commit": case["source_git_commit"],
        "map_driver_sha256": case["map_driver_sha256"],
        "frequency_specific_equations": {
            key: replay["original_equations_saved"][key] for key in ("operator_csc_sha256", "rhs_sha256")
        },
        "all_incident_drives_checked": len(saved.channels),
        "maximum_native_checks": case["maximum_numerical_checks"],
        "independent_flux_checks": checks,
        "incident_modes": [{key: value for key, value in mode.items() if key not in ("T", "R")}
                           for mode in saved.channels],
        "archive": {key: case["archive"][key] for key in (
            "map", "checkpoint", "point_summary", "configuration_AP_pnc.json", "port_modes_AP_pnc.csv",
            native.FIELDS, "independent_flux_replay",
        )},
    }
    return endpoint, case, saved, config, holes, checkpoint


def shared_artifacts(reference, reference_state, candidate, candidate_state):
    source, target = (state["shared"] for state in (reference_state, candidate_state))
    require(all(source[key] == target[key] for key in ("mesh_hash", "reference_planes_x_m")),
            "Mesh identity or original reference planes changed.")
    artifacts = {}
    for key in ARTIFACTS:
        links = []
        for directory, shared in ((reference, source), (candidate, target)):
            link = shared["artifacts"][key]
            path = native._verified(link, directory)
            links.append({**link, "path": path.relative_to(ROOT).as_posix()})
        first, second = links
        require(all(first[field] == second[field] for field in ("sha256", "bytes")),
                f"Prepared {key} differs from the donor.")
        artifacts[key] = {"reference_path": first["path"], "candidate_path": second["path"],
                          "bytes": first["bytes"], "sha256": first["sha256"]}
    return artifacts


def guarded_execution(reference, candidate, frequency, flux_path, config_pin, checkpoint, body_dofs):
    receipts, response = {}, {}
    for phase in ("preparation", "response", "flux"):
        path = candidate.parent / "local_checks" / f"{candidate.name}_{phase}_execution_AP_pnc.json"
        row = read_json(path)
        require(row["phase"] == phase and row["exit_code"] == 0 and row["guard"] is None
                and row["source_revision"] == checkpoint["source_git_commit"]
                and row["configuration_sha256"] == config_pin["sha256"]
                and row["host_platform"] == checkpoint["inputs"]["numerical_identity"]["platform"]
                and all(row["thread_environment"][key] == value
                        for key, value in checkpoint["inputs"]["thread_environment"].items())
                and row["elapsed_seconds"] <= row["wall_seconds"]
                and row["min_sampled_commit_available_bytes"] >= row["commit_reserve_bytes"]
                and row["min_sampled_physical_available_bytes"] >= row["available_reserve_bytes"]
                and row["peak_sampled_process_tree_rss_sum_bytes"] <= row["resident_limit_bytes"]
                and row["peak_sampled_process_tree_private_sum_bytes"] <= row["private_limit_bytes"],
                f"The guarded {phase} receipt differs from the archived run or its limits.")
        argv = row["argv"]
        output = (ROOT / argv[argv.index("--output") + 1]).resolve()
        if phase == "flux":
            script = next((i for i, part in enumerate(argv)
                           if (ROOT / part).resolve() == FLUX_SOURCES["regional_replay_source_sha256"]), None)
            require(script is not None and (ROOT / argv[script + 1]).resolve() == candidate
                    and float(argv[script + 2]) == frequency
                    and output == flux_path.resolve(), "The flux stage did not replay the candidate endpoint.")
        else:
            require(any((ROOT / part).resolve() == SEEDER for part in argv)
                    and row["seeder_source_sha256"] == checkpoint["preparation_reuse"]["seeder_source_sha256"]
                    and (ROOT / argv[argv.index("--from-map") + 1]).resolve() == reference
                    and float(argv[argv.index("--frequencies-ghz") + 1]) == frequency
                    and argv[argv.index("--max-new-points") + 1] == ("0" if phase == "preparation" else "1")
                    and output == candidate, f"The {phase} command differs from the seeded calculation.")
        receipts[phase] = pin(path)
        if phase == "response":
            admission = row["admission"]
            require(admission["ndof"] == body_dofs, "The admitted body size differs from the solved system.")
            response = {key: row[key] for key in (
                "elapsed_seconds", "peak_sampled_process_tree_rss_sum_bytes",
                "peak_sampled_process_tree_private_sum_bytes", "min_sampled_commit_available_bytes",
                "min_sampled_physical_available_bytes",
            )}
            response.update({key: admission[key] for key in ("actual_K_nonzeros", "actual_M_nonzeros")})
    return {"guarded_execution_receipts": receipts, "response_resources": response}


def compare_modes(first, second, endpoints, cases):
    choices = [axial._axial_choice(point.channels) for point in (first, second)]
    for endpoint, case, point, (index, _) in zip(endpoints, cases, (first, second), choices, strict=True):
        if index is not None:
            modes = point.channels
            wave = np.array([row["k_rad_m"] for row in modes])
            gap = float(np.min(abs(np.delete(wave, index) - wave[index])))
            drive = case["native_drives"][index]
            endpoint["native_axial_drive"] = {
                **modes[index], "nearest_incident_wavenumber_gap_rad_m": gap,
                "relative_incident_wavenumber_gap": gap / float(np.max(abs(wave))),
                "independent_direct_T": drive["independent_direct_transmission"],
                "independent_direct_R": drive["independent_direct_reflection"],
            }
    for endpoint, (_, reason) in zip(endpoints, choices, strict=True):
        if reason:
            return {"status": "unresolved", "reason": f"{endpoint['frequency_hz']:g} Hz: {reason}"}

    # The verified identical section gives a common ordered P2 basis. Use its mass directly.
    a, b, mass = first.displacement, second.displacement, first.section.mass
    a_norm, b_norm = [np.einsum("cni,cni->i", u.conj(), mass @ u).real for u in (a, b)]
    cross = abs(np.einsum("cni,cnj->ij", a.conj(), mass @ b))
    overlap = cross / np.sqrt(np.outer(a_norm, b_norm))
    require(np.isfinite(overlap).all() and np.max(overlap) <= 1 + 1e-10, "Invalid native section mass overlap.")
    (i, _), (j, _) = choices
    matched, competing = axial._match_evidence(np.minimum(1, overlap), i, j)
    evidence = {
        "method": "Absolute displacement overlap integrated with the unchanged physical section mass matrix.",
        "axial_kinetic_energy_fraction_threshold": axial.AXIAL_MAJORITY,
        "minimum_mass_overlap": axial.MIN_OVERLAP,
        "minimum_relative_wavenumber_gap_and_overlap_margin": axial.K_GAP_TOL,
        "mass_overlap": matched, "best_competing_overlap": competing, "overlap_margin": matched - competing,
    }
    if matched < axial.MIN_OVERLAP or matched <= competing + axial.K_GAP_TOL:
        return {"status": "unresolved", "mode_match": evidence,
                "reason": "The axial modes lack a unique endpoint mass-overlap match."}
    return {"status": "matched_finite_axial_endpoints", "mode_match": evidence,
            "power_changes_candidate_minus_reference": {
                key: axial._delta(first.channels[i][key], second.channels[j][key]) for key in ("T", "R")
            }}


def compare(reference, candidate, reference_frequency, candidate_frequency, reference_flux, candidate_flux):
    require(all(math.isfinite(x) and x > 0 for x in (reference_frequency, candidate_frequency))
            and reference_frequency != candidate_frequency, "Expected distinct finite positive endpoint frequencies.")
    flux_sources = {key: pin(path) for key, path in FLUX_SOURCES.items()}
    first, first_case, first_saved, first_config, holes, first_state = load_endpoint(
        reference, reference_frequency, reference_flux, flux_sources)
    second, second_case, second_saved, second_config, second_holes, second_state = load_endpoint(
        candidate, candidate_frequency, candidate_flux, flux_sources)
    require(native._point_config(first_config, candidate_frequency) == second_config and holes == second_holes
            and first_saved.settings == second_saved.settings
            and all(first_case[key] == second_case[key] for key in ("body_dofs", "port_dofs")),
            "Expected the same 26-hole geometry, material, discretization and signed port settings.")
    require(all(first["frequency_specific_equations"][key] != second["frequency_specific_equations"][key]
                for key in first["frequency_specific_equations"]),
            "Expected distinct frequency-specific original equations with the shared preparation.")
    require(first_state["inputs"]["frequencies_GHz"] == [reference_frequency]
            and second_state["inputs"] == {**first_state["inputs"], "frequencies_GHz": [candidate_frequency]},
            "The frequency list must be the only change to the native map inputs and numerical environment.")
    seeder_pin = pin(SEEDER)
    reuse = second_state["preparation_reuse"]
    require(reuse == {"donor_checkpoint_sha256": first["archive"]["checkpoint"]["sha256"],
                      "donor_shared_sha256": native.workflow.digest(first_state["shared"]),
                      "donor_source_git_commit": first["source_git_commit"],
                      "seeder_source_sha256": seeder_pin["sha256"]}
            and first_state["shared"] == second_state["shared"],
            "Seed provenance or preparation metadata differs from the pinned donor checkpoint.")
    artifacts = shared_artifacts(reference, first_state, candidate, second_state)

    prior = read_json(PRIOR)
    thickness = first_case["physical_thickness_nm"]
    prior_anchor = prior["maps"][{100: "thin_rim48", 200: "thick_rim48"}[thickness]]
    require(prior["status"] == "completed_finite_paired_comparison" and prior["frequency_hz"] == first["frequency_hz"]
            and prior["original_numerical_limits"] == LIMITS
            and all(first_case["archive"][key] == linked for key, linked in prior_anchor["archive"].items()),
            "The archived reference differs from the accepted 1 GHz 48-segment point.")
    configuration_pin = prior_anchor["configuration"]
    require(pin(ROOT / configuration_pin["path"], configuration_pin["sha256"]) == configuration_pin
            and read_json(ROOT / configuration_pin["path"]) == first_config,
            "The archived donor configuration differs from the published 1 GHz input.")
    hole_pin = pin(ROOT / holes["source_csv"], holes["source_sha256"])
    result = compare_modes(first_saved, second_saved, (first, second), (first_case, second_case))
    if "native_axial_drive" in first:
        require(first["source_git_commit"] == prior_anchor["source_git_commit"]
                and first["map_driver_sha256"] == prior_anchor["map_driver_sha256"]
                and all(math.isclose(first["native_axial_drive"][key], value, rel_tol=0, abs_tol=1e-12)
                    for key, value in prior_anchor["native_axial_drive"].items()),
                "The selected reference mode differs from the published axial response.")
    return {
        "schema_version": 1, **native.AUTHOR, **result,
        "reference": first, "candidate": second,
        "physical_thickness_nm": thickness, "hole_count": len(holes["holes"]),
        "mesh": first_config["mesh"],
        "body_dofs": first_case["body_dofs"], "port_dofs": first_case["port_dofs"],
        "reference_planes_x_m": first_case["reference_planes_x_m"], "incident_side": "left",
        "identical_preparation": {"mesh_hash": first_case["mesh_hash"], "artifacts": artifacts, "reuse": reuse},
        "original_numerical_limits": LIMITS,
        "execution": guarded_execution(reference, candidate, candidate_frequency, candidate_flux, configuration_pin,
                                       second_state, second_case["body_dofs"]),
        "provenance": {
            "endpoint_replay": pin(Path(__file__)), "published_1GHz_anchor": pin(PRIOR),
            "hole_pattern_csv": hole_pin, "donor_configuration": configuration_pin,
            "seeder": seeder_pin, "acceptance_replay_reader": pin(Path(accepted.__file__)),
            "axial_comparator": pin(Path(axial.__file__)), "independent_flux_sources": flux_sources,
        },
        "scope": "T and R are fractions of native incident power for the two selected axial drives. "
                 "Signed changes use 1 GHz as the reference. Drive orders and root numbers are frequency-local. "
                 "The unchanged mesh and physical end planes give a finite two-endpoint result for one drive. "
                 "Spatial and frequency convergence, continuous branch identity and interior response remain unresolved.",
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("reference", type=Path, help="Accepted 1 GHz native map.")
    parser.add_argument("candidate", type=Path, help="Accepted seeded native map.")
    parser.add_argument("--candidate-frequency-ghz", type=float, default=0.995)
    parser.add_argument("--reference-flux", type=Path)
    parser.add_argument("--candidate-flux", type=Path)
    parser.add_argument("--output", type=Path, help="Optional new JSON evidence file. Defaults to stdout.")
    args = parser.parse_args()
    reference, candidate = args.reference.resolve(), args.candidate.resolve()
    reference_flux = args.reference_flux or reference.with_name(reference.name.removesuffix("_1GHz") + "_flux_AP_pnc.json")
    candidate_flux = args.candidate_flux or candidate.with_name(candidate.name + "_flux_AP_pnc.json")
    try:
        result = compare(reference, candidate, 1.0, args.candidate_frequency_ghz, reference_flux, candidate_flux)
        payload = json.dumps(result, indent=2, allow_nan=False) + "\n"
        if args.output:
            with args.output.open("x", encoding="utf-8", newline="\n") as stream:
                stream.write(payload)
        else:
            print(payload, end="")
    except (KeyError, OSError, TypeError, ValueError) as exc:
        parser.exit(1, f"{parser.prog}: {exc}\n")


if __name__ == "__main__":
    main()
