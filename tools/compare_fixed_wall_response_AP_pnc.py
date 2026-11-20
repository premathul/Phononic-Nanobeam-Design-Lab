#!/usr/bin/env python3
"""Replay the saved fixed-connectivity wall response. Author: Athul Prem | _AP_pnc."""

from __future__ import annotations

import argparse
from dataclasses import dataclass
import json
import math
from pathlib import Path
import runpy
import sys

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if __name__ == "__main__":
    sys.path.insert(0, str(ROOT))

from tools import compare_axial_nested_mesh_AP_pnc as nested
from tools import compare_scattering_AP_pnc as scattering
from tools import map_scattering_AP_pnc as native
from tools import run_fixed_wall_response_AP_pnc as wall
from tools.compare_scattering_AP_pnc import require


STUDY = ROOT / "results/runs/physical_thickness_device_AP_pnc"
REFERENCE = ROOT / "verification/axial_device_rim_pairs_1GHz_AP_pnc.json"
CORRESPONDENCE = ROOT / "verification/paired_mode_frequency_AP_pnc.json"
THICKNESSES_NM = {"thin": 100, "thick": 200}
CANDIDATES = {role: STUDY / f"{role}{nm}_central400_rim48_chord24_1GHz_windows_prepared_linux"
              for role, nm in THICKNESSES_NM.items()}
KINDS = ("transmission", "reflection")
ROUND_OFF = 1e-12


@dataclass(frozen=True)
class Response:
    record: dict
    checkpoint: dict
    pattern: dict
    summary: dict
    fixed: dict
    shared_paths: dict[str, Path]


def read_response(directory, flux, read_point, limits, thickness_nm):
    record, config, pattern = read_point(directory, flux, wall.FREQUENCY_GHZ)
    checkpoint = native.read_map_checkpoint(directory)
    shared = checkpoint["shared"]
    summary_path = native._verified(record["archive"]["point_summary"], ROOT)
    summary, fixed, _ = scattering.read_run(summary_path.parent)
    require(record["physical_thickness_nm"] == thickness_nm and summary["num_incident_channels"] == wall.CHANNELS
            and config["mesh"] == {"kind": "layered", "element_order": 2, "mesh_size_nm": 400,
                                   "lead_mesh_size_nm": 150, "circle_resolution": 48, "thickness_layers": 2,
                                   "port_size_nm": 250, "patterned_x_interval_nm": [1000, 15650]},
            f"Expected the five-drive {thickness_nm} nm device at the original mesh settings in {directory}.")
    residual_limit = limits["original_equation_and_nodal_force_relative_residuals"]
    require(all(0 <= record["maximum_numerical_checks"]["maximum_" + key] < residual_limit
                for key in ("relative_residual", "relative_backward_residual")),
            f"Original equation residual exceeds the saved limit in {directory}.")
    paths = {name: native._verified(pin, directory) for name, pin in shared["artifacts"].items()}
    record["archive"].update({name: {**shared["artifacts"][name], "path": path.relative_to(ROOT).as_posix()}
                              for name, path in paths.items()})
    return Response(record, checkpoint, pattern, summary, fixed, paths)


def check_preparation(a, b, candidate_dir):
    reference, a_state = a.record, a.checkpoint
    candidate, b_state = b.record, b.checkpoint
    receipt_sha = b_state["fixed_wall_preparation_sha256"]
    preparation = wall._read_preparation(candidate_dir, receipt_sha)
    prepared = preparation["inputs"]
    executed = b_state["inputs"]
    reuse = b_state["preparation_reuse"]
    require(preparation["shared"] == b_state["shared"]
            and all(prepared[key] == executed[key] for key in
                    ("configuration", "hole_pattern", "frequencies_GHz", "thread_environment"))
            and reuse["receipt_sha256"] == receipt_sha and reuse["method"] == wall.METHOD
            and reuse["assembly_numerical_identity"] == prepared["numerical_identity"]
            and reuse["assembly_source_git_commit"] == preparation["assembly_source_git_commit"]
            and reuse["original_end_sections_match_exactly"] is True,
            "The accepted candidate does not bind the pinned fixed-wall preparation.")
    reference_sha = reference["archive"]["checkpoint"]["sha256"]
    donor = preparation["donor"]
    require(donor["checkpoint_sha256"] == reference_sha and donor["map_sha256"] == reference["archive"]["map"]["sha256"]
            and donor["accepted_summary"] == a_state["points"][0]["accepted"]["summary"]
            and donor["source_git_commit"] == a_state["source_git_commit"]
            and donor["mesh_hash"] == a_state["shared"]["mesh_hash"]
            and all(donor[key] == a_state["shared"]["artifacts"][key] for key in ("mesh", "port_section")),
            "The fixed-wall preparation names a different 48-sided donor.")
    require(a.pattern == {key: value for key, value in b.pattern.items() if key != "fixed_wall_geometry"}
            and reference["reference_planes_x_m"] == candidate["reference_planes_x_m"]
            and (reference["body_dofs"], reference["port_dofs"]) == (candidate["body_dofs"], candidate["port_dofs"]),
            "The native pair differs in nominal holes, original planes or degrees of freedom.")
    assembly, solve = prepared["numerical_identity"], executed["numerical_identity"]
    reference_identity = a_state["inputs"]["numerical_identity"]
    baseline_hashes, assembly_hashes = reference_identity["source_sha256"], assembly["source_sha256"]
    differing_sources = {key: {"reference_sha256": value, "candidate_sha256": assembly_hashes.get(key)}
                         for key, value in baseline_hashes.items() if value != assembly_hashes.get(key)}
    require(all(assembly[key] == solve[key] for key in ("source_sha256", "packages"))
            and assembly["packages"] == reference_identity["packages"]
            and differing_sources.keys() <= wall.DONOR_LEGACY_SOURCES
            and all(scattering.is_sha256(change["candidate_sha256"]) for change in differing_sources.values()),
            "The pair uses different response kernels or numerical package versions.")
    return {
        "receipt": native._link(candidate_dir / wall.RECEIPT, ROOT),
        "reference_assembly_and_solve_platform": reference_identity["platform"],
        "candidate_assembly_platform": assembly["platform"], "candidate_solve_platform": solve["platform"],
        "candidate_assembly_source_git_commit": preparation["assembly_source_git_commit"],
        "candidate_solve_source_git_commit": b_state["source_git_commit"],
        "candidate_source_sha256": {path.relative_to(ROOT).as_posix(): assembly_hashes[path.relative_to(ROOT).as_posix()]
                                    for path in (Path(wall.__file__).resolve(), wall.WALL_SOURCE,
                                                 Path(native.__file__).resolve())},
        "baseline_to_candidate_source_differences": differing_sources,
    }


def check_geometry(a_paths, b_paths, fixed_wall, donor_mesh_sha):
    manifest = native._read(b_paths["mesh_summary"])
    geometry = manifest["geometry_diagnostics"]["fixed_wall_faceting"]
    require(fixed_wall == manifest["mesh_inputs"]["fixed_wall_geometry"]
            and fixed_wall["method"] == wall.METHOD
            and fixed_wall["donor_mesh_sha256"] == donor_mesh_sha
            and fixed_wall["geometry_diagnostics_sha256"] == native.workflow.digest(geometry)
            and fixed_wall["source_sha256"] == geometry["control_source_sha256"]
            and (fixed_wall["polygon_sides_per_hole"], fixed_wall["boundary_edges_per_hole"]) == (24, 48)
            and len(geometry["holes"]) == 26 and geometry["original_physical_rim_sides"] == 48
            and geometry["control_physical_rim_sides"] == 24 and geometry["finite_element_rim_edges_per_ring"] == 48
            and 0 < geometry["minimum_relative_signed_tet_determinant"],
            "The mesh manifest describes a different shape change or has invalid geometry evidence.")
    with np.load(a_paths["mesh"], allow_pickle=False) as first, np.load(b_paths["mesh"], allow_pickle=False) as second:
        points, deformed, cells = first["points_m"], second["points_m"], first["tetrahedra"]
        moved = (points != deformed).any(axis=0)
        require(set(first.files) == set(second.files)
                and all(np.array_equal(first[key], second[key]) for key in first.files if key != "points_m")
                and np.array_equal(points[2], deformed[2])
                and native._array_digest(points) == geometry["original_points_sha256"]
                and native._array_digest(deformed) == geometry["control_points_sha256"]
                and native._array_digest(cells) == geometry["tetrahedra_sha256"]
                and native._array_digest(deformed, cells) == fixed_wall["deformed_coordinates_and_connectivity_sha256"]
                and moved.sum() == geometry["moved_vertices"] > 0
                and moved[cells].any(axis=0).sum() == geometry["touched_tetrahedra"] > 0,
                "The accepted meshes do not reproduce the declared coordinate changes and fixed connectivity.")
        planar = {"original_xy_points_sha256": native._array_digest(points[:2]),
                  "control_xy_points_sha256": native._array_digest(deformed[:2])}
    with np.load(a_paths["port_section"], allow_pickle=False) as first, np.load(b_paths["port_section"], allow_pickle=False) as second:
        require(all(np.array_equal(first[key], second[key]) for key in wall.SECTION_ARRAYS),
                "The wall control changes an end trace, section matrix or degree-of-freedom map.")

    metrics = ("original_physical_rim_sides", "control_physical_rim_sides", "finite_element_rim_edges_per_ring", "z_planes_m",
               "tetrahedra_sha256", "original_points_sha256", "control_points_sha256", "moved_vertices",
               "touched_tetrahedra", "max_vertex_displacement_m", "minimum_relative_signed_tet_determinant",
               "measured_solid_volume_increase_m3", "unchanged_end_facets", "unchanged_cut_facets",
               "unchanged_lead_tetrahedra")
    return {**{key: geometry[key] for key in metrics}, **planar, "identical_end_section_arrays_and_dof_maps": True}


def drive_powers(a, b, blocks, incoming):
    by_root = [{drive["native_root_id"]: drive for drive in record["native_drives"]} for record in (a, b)]
    rows = []
    for i, (root_a, root_b) in enumerate(zip(incoming["reference_roots"], incoming["candidate_roots"], strict=True)):
        first, second = by_root[0][root_a], by_root[1][root_b]
        row = {"reference": {key: first[key] for key in ("drive", "native_root_id", "k_rad_m")},
               "candidate": {key: second[key] for key in ("drive", "native_root_id", "k_rad_m")},
               "candidate_minus_reference_percentage_points": {}}
        for kind in KINDS:
            require(all(math.isclose(blocks[kind][f"{side}_power"][i], drive[kind], rel_tol=0, abs_tol=ROUND_OFF)
                        for side, drive in (("reference", first), ("candidate", second))),
                    f"Aligned block and native {kind} power differ for drive {i + 1}.")
            row["reference"][kind], row["candidate"][kind] = first[kind], second[kind]
            row["candidate_minus_reference_percentage_points"][kind] = 100 * (second[kind] - first[kind])
        rows.append(row)
    means = {side: {kind: math.fsum(row[side][kind] for row in rows) / len(rows) for kind in KINDS}
             for side in ("reference", "candidate")}
    means["candidate_minus_reference_percentage_points"] = {
        kind: 100 * (means["candidate"][kind] - means["reference"][kind]) for kind in KINDS}
    return rows, means


def compare_wall(thickness_nm, anchor, candidate_dir, candidate_flux, read_point, limits):
    reference_dir = native._verified(anchor["map"], ROOT).parent
    reference_flux = native._verified(anchor["independent_flux_replay"], ROOT)
    a = read_response(reference_dir, reference_flux, read_point, limits, thickness_nm)
    require(all(a.record["archive"][key] == pin for key, pin in anchor.items()), "The saved rim48 donor differs from its pin.")
    b = read_response(candidate_dir, candidate_flux, read_point, limits, thickness_nm)
    preparation = check_preparation(a, b, candidate_dir)
    geometry = check_geometry(a.shared_paths, b.shared_paths, b.pattern["fixed_wall_geometry"],
                              a.record["archive"]["mesh"]["sha256"])
    scattering.compatible_inputs(a.summary, b.summary, a.fixed, b.fixed)
    first, second = (scattering.read_scattering(native._verified(case.record["archive"]["point_summary"], ROOT).parent,
                                                case.summary, case.fixed) for case in (a, b))
    matches, blocks, _ = scattering.compare_blocks(first, second)
    drives, means = drive_powers(a.record, b.record, blocks, matches["incoming"])
    maximum_power = max(block["maximum_power_change"] for block in blocks.values())
    maximum_complex = max(block["maximum_complex_change"] for block in blocks.values())
    saved_cases = {side: {key: case.record[key] for key in ("body_dofs", "port_dofs", "source_git_commit",
                                                         "maximum_numerical_checks", "independent_flux_checks", "archive")}
                   for side, case in (("reference", a), ("candidate", b))}
    return {
        "physical_thickness_nm": thickness_nm, "geometry": geometry, "preparation": preparation, "native_cases": saved_cases,
        "signed_mode_matches": matches, "drive_power_fractions": drives, "equal_incoherent_mean_power_fractions": means,
        "per_block_maxima": {name: nested._block_increment(block, matches["incoming"], matches[name])
                             for name, block in blocks.items()},
        "maximum_absolute_total_drive_power_change": maximum_power, "total_drive_power_change_limit": scattering.LIMIT,
        "total_drive_power_increment_passed": maximum_power < scattering.LIMIT,
        "maximum_absolute_complex_amplitude_change": maximum_complex, "complex_amplitude_change_limit": scattering.LIMIT,
        "complex_amplitude_increment_passed": maximum_complex < scattering.LIMIT,
    }


def thin_minus_thick(a, b):
    """Return within-shape thickness contrasts and their change in percentage points."""
    contrasts = {side: {kind: 100 * (a[side][kind] - b[side][kind]) for kind in KINDS}
                 for side in ("reference", "candidate")}
    contrasts["candidate_minus_reference"] = {
        kind: contrasts["candidate"][kind] - contrasts["reference"][kind] for kind in KINDS}
    return contrasts


def compare_thickness(thin, thick):
    a, b = thin["geometry"], thick["geometry"]
    require(all(a[key] == b[key] for key in ("tetrahedra_sha256", "original_xy_points_sha256", "control_xy_points_sha256"))
            and np.array_equal(2 * np.array(a["z_planes_m"]), b["z_planes_m"]),
            "The thin and thick perturbations do not share the same planar deformation and scaled z levels.")
    paired = native._read(CORRESPONDENCE)
    frequency_hz = wall.FREQUENCY_GHZ * 1e9
    sample, = (row for row in paired["frequency_samples"] if row["frequency_hz"] == frequency_hz)
    bridge, = (row for row in paired["sampled_thickness_correspondence"]["samples"] if row["frequency_hz"] == frequency_hz)
    for role, case in (("thin", thin), ("thick", thick)):
        require(all(sample["acceptance"][role]["native_archive"][key] == case["native_cases"]["reference"]["archive"][key]
                    for key in ("map", "checkpoint", "point_summary", "port_section")),
                f"The published cross-thickness correspondence uses a different {role} baseline.")
    rows = bridge["sampled_correspondences"]
    require(bridge["100_to_150_nm"]["all_five_one_to_one"] and bridge["150_to_200_nm"]["all_five_one_to_one"]
            and len(rows) == wall.CHANNELS,
            "The archived 1 GHz thickness correspondence has missing or unresolved links.")
    by_root = [{row["reference"]["native_root_id"]: row for row in case["drive_power_fractions"]}
               for case in (thin, thick)]
    contrasts = []
    for row in rows:
        roots = row["section_roots_100_150_200_nm"]
        thin_drive, thick_drive = by_root[0][roots[0]], by_root[1][roots[-1]]
        contrasts.append({
            "reference_section_roots_100_150_200_nm": roots,
            "candidate_incident_roots_100_200_nm": [drive["candidate"]["native_root_id"]
                                                    for drive in (thin_drive, thick_drive)],
            "thin_minus_thick_percentage_points": thin_minus_thick(thin_drive, thick_drive),
        })
    return {
        "equal_incoherent_mean_thin_minus_thick_percentage_points": thin_minus_thick(
            thin["equal_incoherent_mean_power_fractions"], thick["equal_incoherent_mean_power_fractions"]),
        "sampled_drive_contrasts": contrasts,
        "sampled_native_thickness_correspondence": native._link(CORRESPONDENCE, ROOT),
        "same_planar_geometry_and_element_connectivity_scaled_in_z": True,
    }


def replay(candidates):
    published = native._read(REFERENCE)
    limits = published["original_numerical_limits"]
    reader_pin = published["provenance"]["acceptance_replay_reader"]
    read_point = runpy.run_path(str(native._verified(reader_pin, ROOT)))["read_point"]
    cases = {role: compare_wall(THICKNESSES_NM[role], published["maps"][f"{role}_rim48"]["archive"],
                                directory, flux, read_point, limits) for role, (directory, flux) in candidates.items()}
    result = {
        **native.AUTHOR, "schema_version": 1, "status": "compared_saved_fixed_wall_response", **native.CONVERGENCE,
        "frequency_hz": wall.FREQUENCY_GHZ * 1e9, "hole_count": 26, "incident_side": published["incident_side"],
        "reference_planes_x_m": published["reference_planes_x_m"], "original_numerical_limits": limits,
        "by_thickness": cases,
        "scope": "The 24-sided chord polygon retains 48 finite-element edges per rim and the original element "
                 "connectivity. Moving alternating rim vertices changes the wall shape, solid volume and geometry "
                 "of wall-adjacent tetrahedra. The comparison measures their finite combined effect at 1 GHz. "
                 "Phases use the original physical end planes. Power changes are candidate minus reference. "
                 "The equal-incoherent mean uses equal incident power in the five native propagating inputs. "
                 "Continuum error and spatial or frequency convergence remain unresolved.",
        "provenance": {"published_rim48_anchor": native._link(REFERENCE, ROOT), "native_acceptance_reader": reader_pin,
                       "alignment_source": native._link(Path(scattering.__file__).resolve(), ROOT),
                       "block_summary_source": native._link(Path(nested.__file__).resolve(), ROOT),
                       "comparison_source": native._link(Path(__file__).resolve(), ROOT)},
    }
    if "thick" in cases:
        result["paired_thickness_effect"] = compare_thickness(cases["thin"], cases["thick"])
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--thin-only", action="store_true", help="Compare only the 100 nm device.")
    for role, default in CANDIDATES.items():
        parser.add_argument(f"--{role}-candidate", type=Path, default=default, help="Accepted fixed-wall native map.")
        parser.add_argument(f"--{role}-flux", type=Path, help="Independent candidate flux receipt.")
    parser.add_argument("--output", type=Path, help="Write a JSON report instead of printing it.")
    args = parser.parse_args()
    try:
        candidates = {}
        for role in ("thin",) if args.thin_only else THICKNESSES_NM:
            candidate = getattr(args, f"{role}_candidate").resolve()
            flux = (getattr(args, f"{role}_flux") or candidate.with_name(f"{candidate.name}_flux_AP_pnc.json")).resolve()
            candidates[role] = candidate, flux
        payload = json.dumps(replay(candidates), indent=2, allow_nan=False) + "\n"
        if args.output is None:
            print(payload, end="")
        else:
            destination = args.output.resolve()
            require(destination not in (REFERENCE, CORRESPONDENCE) and not destination.is_relative_to(ROOT / "results/runs"),
                    "Choose an output outside the original study archive and donor response record.")
            staging = destination.with_suffix(destination.suffix + ".tmp")
            staging.write_text(payload, encoding="utf-8", newline="\n")
            staging.replace(destination)
    except (OSError, ValueError, KeyError, TypeError) as exc:
        parser.exit(1, f"{parser.prog}: {exc}\n")


if __name__ == "__main__":
    main()
