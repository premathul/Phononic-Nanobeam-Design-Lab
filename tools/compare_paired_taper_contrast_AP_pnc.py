#!/usr/bin/env python3
"""Compare paired 1 GHz taper refinements at 100 and 200 nm. Author: Athul Prem | _AP_pnc."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import runpy
import sys

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if __name__ == "__main__":
    sys.path.insert(0, str(ROOT))

from tools import central_patch_refinement_AP_pnc as patch
from tools import compare_paired_mode_frequency_AP_pnc as power
from tools import map_scattering_AP_pnc as native
from tools.compare_scattering_AP_pnc import require


STUDY = ROOT / "results/runs/physical_thickness_device_AP_pnc"
PRIOR = ROOT / "verification/axial_device_nested_central_AP_pnc.json"
THICKNESS = {"thin": 100, "thick": 200}
STAGES = {"baseline": (225.0, "central400_edge225_rim32"),
          "central_only": (225.0, "taper_patch225to168p75_1GHz"),
          "full_refinement": (168.75, "central400_edge168p75_rim32_linux")}
TRANSITIONS = {"central_only_minus_baseline": ("baseline", "central_only"),
               "full_minus_central_only": ("central_only", "full_refinement"),
               "full_minus_baseline": ("baseline", "full_refinement")}
PATCH_HASHES = {100: "78f598f326d88101fcbed5104d01042dfa9408f106119b2cc3e22c35fd6c2085",
                200: "041d61c3010e537f855136cf9030a4054ec22d888a3c8a66e8a892dc2fef1ae5"}
POWERS = ("transmission", "reflection")


def _read_case(study, role, stage, prior, reader):
    thickness = THICKNESS[role]
    target, suffix = STAGES[stage]
    directory = study / f"{role}{thickness}_{suffix}"
    flux_path = study / f"{directory.name}_1GHz_flux_AP_pnc.json"
    record, config, pattern = reader(directory, flux_path, 1.0)
    checkpoint = native.read_map_checkpoint(directory)
    require(record["physical_thickness_nm"] == thickness and record["reference_planes_x_m"] == prior["reference_planes_x_m"]
            and config["mesh"] == {**prior["parent_mesh"], "central_max_edge_nm": target}
            and config["mesh"]["element_order"] == config["mesh"]["thickness_layers"] == 2
            and config["analysis"]["incident_side"] == prior["incident_side"] == "left",
            f"The native {role} {stage} point differs in thickness, mesh, or original reference planes.")
    if stage != "central_only":
        level = next(row for row in prior["levels"] if row["central_max_edge_nm"] == target)
        expected = level["samples"]["1GHz"][role]["archive"]
        require(all(record["archive"][key] == pin for key, pin in expected.items()),
                f"The {role} {stage} archive differs from the published nested-device data.")

    flux = native._read(flux_path)
    require([flux[name] for name in ("replay_source_sha256", "regional_replay_source_sha256")]
            == [pin["sha256"] for pin in prior["provenance"]["independent_flux_sources"]],
            f"The {role} {stage} point uses a different independent flux replay.")
    summary_path = native._verified(record["archive"]["point_summary"], ROOT)
    summary = native._read(summary_path)
    channels = [{"T": row["transmission"], "R": row["reflection"]} for row in record["native_drives"]]
    require(len(channels) == power.CHANNELS, f"The {role} {stage} point requires five open inputs.")
    trace = power.all_mode_power(summary_path.parent, summary, channels, flux, record["archive"],
                                 prior["original_numerical_limits"]["single_drive_and_coherent_power_defect"])

    shared = checkpoint["shared"]
    manifest_path = native._verified(shared["artifacts"]["mesh_summary"], directory)
    mesh, manifest = native.workflow.load_checkpoint(manifest_path.parent, shared["mesh_hash"])
    require(manifest["mesh_sha256"] == record["archive"]["mesh"]["sha256"]
            and manifest["ndof"] == record["body_dofs"] and manifest["hole_pattern_provenance"] == pattern,
            f"The {role} {stage} response differs from its original mesh or hole provenance.")
    solve = checkpoint["inputs"]["numerical_identity"]
    transfer = checkpoint.get("preparation_reuse", {}).get("receipt", {})
    facts = {key: record[key] for key in ("mesh_hash", "body_dofs", "port_dofs", "source_git_commit",
                                        "maximum_numerical_checks", "independent_flux_checks")}
    facts.update({"equal_incoherent_power_fractions": {key: trace[f"mean_{key}_fraction"] for key in POWERS},
                  "mean_power_closure_defect": trace["mean_power_closure_defect"],
                  "mean_modal_minus_direct_power_fractions": trace["modal_minus_direct_flux_mean_fractions"],
                  "assembly_platform": transfer.get("donor_numerical_identity", solve)["platform"],
                  "solve_platform": solve["platform"],
                  "original_equation_identity": {key: flux["original_equations_saved"][key]
                                                  for key in ("operator_csc_sha256", "rhs_sha256")},
                  "archive": {key: record["archive"][key] for key in (
                      "map", "map_coefficients", "checkpoint", "point_summary", "mesh", "independent_flux_replay")}})
    facts["archive"]["mesh_summary"] = native._link(manifest_path, ROOT)
    return mesh, config, pattern, checkpoint, manifest, facts


def _verify_patch(cases, cuts, thickness):
    coarse, _, _, coarse_state, _, _ = cases["baseline"]
    full, _, _, full_state, _, _ = cases["full_refinement"]
    actual, _, pattern, state, manifest, _ = cases["central_only"]
    donors = state["taper_patch_donors"]
    control = pattern["taper_patch_refinement"]
    diagnostics = manifest["geometry_diagnostics"]["taper_patch_refinement"]
    require(state.get("refined_region", "central") == control.get("refined_region", "central") == "central"
            and state["inputs"]["frequencies_GHz"] == [1.0] and "preparation_reuse" not in state
            and donors["nominal_mirror_cuts_x_m"] == diagnostics["nominal_patch_cuts_x_m"] == list(cuts)
            and donors["hole_pattern_sha256"] == pattern["source_sha256"]
            and control == manifest["mesh_inputs"]["taper_patch_refinement"] and control["donors"] == donors,
            f"The {thickness} nm patch does not bind the declared taper cuts and donor selection.")
    for name, source, stage in (("coarse", coarse_state, "baseline"), ("fine", full_state, "full_refinement")):
        shared = source["shared"]
        require(donors[name]["mesh_hash"] == shared["mesh_hash"]
                and donors[name]["checkpoint"]["sha256"] == cases[stage][-1]["archive"]["checkpoint"]["sha256"]
                and all(donors[name][key] == shared["artifacts"][key] for key in ("mesh", "mesh_summary")),
                f"The {thickness} nm taper control refers to a different {name} donor.")
    rebuilt, checks = patch.make_central_patch_control(coarse, full, cuts)
    require(np.array_equal(actual.p, rebuilt.p) and np.array_equal(actual.t, rebuilt.t)
            and diagnostics["control_coordinates_and_connectivity_sha256"]
            == control["mesh_geometry_connectivity_sha256"] == checks["control_coordinates_and_connectivity_sha256"]
            == PATCH_HASHES[thickness],
            f"The {thickness} nm taper control does not reproduce the exact central refinement.")
    return checks


def _delta(candidate, reference):
    return {key: 100 * (candidate[key] - reference[key]) for key in POWERS}


def replay(study=STUDY):
    study = Path(study).resolve()
    require(study.is_relative_to(ROOT), "Keep all native inputs under this checkout for portable archive links.")
    prior = native._read(PRIOR)
    reader_pin = prior["provenance"]["acceptance_reader"]
    reader = runpy.run_path(str(native._verified(reader_pin, ROOT)))["read_point"]
    cases = {role: {stage: _read_case(study, role, stage, prior, reader) for stage in STAGES} for role in THICKNESS}
    _, baseline, pattern, *_ = cases["thin"]["baseline"]
    cuts = tuple((pattern["holes"][a]["x_m"] + pattern["holes"][b]["x_m"]) / 2 for a, b in ((7, 8), (17, 18)))
    fixed_geometry = {key: value for key, value in baseline["geometry"].items() if key != "thickness_nm"}
    for role, stages in cases.items():
        for stage, (_, config, holes, *_) in stages.items():
            require({key: value for key, value in config["geometry"].items() if key != "thickness_nm"} == fixed_geometry
                    and config["material"] == baseline["material"] and config["analysis"] == baseline["analysis"]
                    and {key: value for key, value in holes.items() if key != "taper_patch_refinement"} == pattern,
                    f"The {role} {stage} response changes the fixed geometry, material or analysis.")
    hole_pin = native._link(ROOT / pattern["source_csv"], ROOT)
    require(hole_pin["sha256"] == pattern["source_sha256"], "The source 26-hole table differs from the saved pattern.")
    patch_checks = {role: _verify_patch(cases[role], cuts, thickness) for role, thickness in THICKNESS.items()}
    topology = {}
    for stage in STAGES:
        (thin, *_, thin_facts), (thick, *_, thick_facts) = (cases[role][stage] for role in THICKNESS)
        thin_points, thick_points = thin.p.copy(), thick.p.copy()
        thin_points[2] /= THICKNESS["thin"] * 1e-9
        thick_points[2] /= THICKNESS["thick"] * 1e-9
        require(np.array_equal(np.unique(thin_points[2]), [-0.5, 0.0, 0.5])
                and np.array_equal(thin_points, thick_points) and np.array_equal(thin.t, thick.t)
                and thin.boundaries.keys() == thick.boundaries.keys()
                and all(np.array_equal(thin.boundaries[tag], thick.boundaries[tag]) for tag in thin.boundaries)
                and thin_facts["body_dofs"] == thick_facts["body_dofs"] and thin_facts["port_dofs"] == thick_facts["port_dofs"],
                f"The {stage} pair does not share the exact xy coordinates, normalized z and tetrahedral connectivity.")
        topology[stage] = {"mesh_vertices": int(thin.nvertices), "mesh_tetrahedra": int(thin.nelements),
                           "normalized_z_coordinates_and_connectivity_sha256": native._array_digest(thin_points, thin.t)}

    facts = {role: {stage: case[-1] for stage, case in mesh_cases.items()} for role, mesh_cases in cases.items()}
    operators = {case["original_equation_identity"]["operator_csc_sha256"]
                 for mesh_facts in facts.values() for case in mesh_facts.values()}
    require(len(operators) == len(STAGES) * len(THICKNESS), "Expected six distinct original body equations.")
    means = {role: {stage: case["equal_incoherent_power_fractions"] for stage, case in stages.items()}
             for role, stages in facts.items()}
    contrasts = {stage: _delta(means["thin"][stage], means["thick"][stage]) for stage in STAGES}
    increments = {
        edge: {**{role: _delta(means[role][second], means[role][first]) for role in THICKNESS},
               "change_in_thin_minus_thick": {key: contrasts[second][key] - contrasts[first][key] for key in POWERS}}
        for edge, (first, second) in TRANSITIONS.items()
    }
    return {
        **native.AUTHOR, **native.CONVERGENCE, "schema_version": 1, "frequency_hz": 1e9,
        "incident_side": "left", "physical_thickness_nm": THICKNESS, "equal_power_input_count": power.CHANNELS,
        "hole_count": len(pattern["holes"]), "hole_pattern": hole_pin,
        "quadratic_thickness_layers": baseline["mesh"]["thickness_layers"],
        "coarse_to_fine_edge_targets_nm": [STAGES["baseline"][0], STAGES["full_refinement"][0]],
        "reference_planes_x_m": prior["reference_planes_x_m"], "nominal_patch_cuts_x_m": list(cuts),
        "central_only_replaced_planar_triangles": patch_checks["thin"]["replaced_coarse_planar_triangles"],
        "matched_meshes_after_z_normalization": topology,
        "paired_thickness_contrasts_percentage_points_100nm_minus_200nm": contrasts,
        "signed_mesh_increments_percentage_points": increments, "cases": facts,
        "original_numerical_limits": prior["original_numerical_limits"],
        "scope": "Each mean gives five native propagating left inputs equal mutually incoherent power. "
                 "The averages are computed within each thickness without pairing individual drives. "
                 "The full-minus-central increment changes the rest of the mesh with the center already refined. "
                 "The independent 200 nm complementary response and its interaction with central refinement are unresolved. "
                 "These finite 1 GHz powers leave continuum and experimental accuracy unresolved.",
        "provenance": {"published_full_device_comparison": native._link(PRIOR, ROOT), "acceptance_reader": reader_pin,
                       "trace_reader": native._link(Path(power.__file__).resolve(), ROOT),
                       "mesh_refiner": native._link(Path(patch.__file__).resolve(), ROOT),
                       "comparison_source": native._link(Path(__file__).resolve(), ROOT)},
    }


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--study", type=Path, default=STUDY)
    parser.add_argument("--output", type=Path, help="Write a new comparison JSON file.")
    args = parser.parse_args(argv)
    try:
        result = replay(args.study)
        payload = json.dumps(result, indent=2, allow_nan=False) + "\n"
        if args.output is None:
            print(payload, end="")
        else:
            with args.output.open("x", encoding="utf-8") as stream:
                stream.write(payload)
    except (OSError, ValueError, TypeError, KeyError) as exc:
        parser.exit(1, f"Paired taper contrast failed. {exc}\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
