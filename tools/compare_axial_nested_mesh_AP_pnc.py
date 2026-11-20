#!/usr/bin/env python3
"""Replay the nested central-mesh axial comparison at 995 and 1000 MHz. Author: Athul Prem | _AP_pnc."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path
import runpy
import sys

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if __name__ == "__main__":
    sys.path.insert(0, str(ROOT))

from tools import compare_axial_device_power_AP_pnc as axial
from tools import compare_axial_fourth_mesh_AP_pnc as fourth
from tools import compare_axial_frequency_pair_AP_pnc as pair
from tools import compare_scattering_AP_pnc as scattering
from tools import map_scattering_AP_pnc as native
from tools.compare_scattering_AP_pnc import require


FOURTH = ROOT / "verification/axial_device_fourth_central_AP_pnc.json"
STUDY_ROOT = ROOT / "results/runs/physical_thickness_device_AP_pnc"
TARGET_NAMES = {300.: "300", 225.: "225", 168.75: "168p75"}
REFINER = "src/phononic/refinement/central_AP_pnc.py"


def _parent_archive(row, anchor, role, sample):
    if sample == "1GHz":
        pins = anchor["maps"][f"{role}_rim32"]["archive"]
    else:
        published = row["axial_R_995MHz"][role]
        pins = {"map": published["native_map"], "point_summary": published["native_point_summary"],
                "independent_flux_replay": published["independent_flux_replay"]}
    paths = {key: native._verified(pin, ROOT) for key, pin in pins.items()}
    return paths["map"].parent, paths["independent_flux_replay"], pins


def _read_sample(directory, flux_path, selected, frequency, mesh, reader, replay_sources):
    accepted, config, _ = reader(directory, flux_path, frequency)
    drive = accepted["native_drives"][selected["drive"] - 1]
    require(config["mesh"] == mesh and drive["native_root_id"] == selected["root"]
            and drive["k_rad_m"] == selected["k_rad_m"]
            and all(math.isclose(drive[key], selected[short], abs_tol=1e-12, rel_tol=0)
                    for key, short in (("reflection", "R"), ("transmission", "T"))),
            f"Native axial response differs from the independently replayed drive in {directory}.")
    replay = native._read(flux_path)
    require([replay[key] for key in ("replay_source_sha256", "regional_replay_source_sha256")] == replay_sources,
            f"Independent flux producers differ from the accepted pilot: {flux_path}.")
    return {**{key: selected[key] for key in pair.MODE_FIELDS},
            "original_equation_identity": {key: replay["original_equations_saved"][key]
                                           for key in ("operator_csc_sha256", "rhs_sha256")},
            "original_numerical_checks": accepted["maximum_numerical_checks"],
            "independent_flux_checks": accepted["independent_flux_checks"],
            "archive": {key: accepted["archive"][key] for key in pair.POINT_PINS}}


def _mode_match(first, second, *, same_section=False):
    a, i = first
    b, j = second
    require(a.settings == b.settings, "The selected fields use different beam settings or reference planes.")
    if same_section:
        require(a.section.discretization_sha256 == b.section.discretization_sha256,
                "The two endpoints for a thickness use different native port sections.")
    forward, competing = axial._match_evidence(axial._overlap(a, b), i, j)
    reverse, reverse_competing = axial._match_evidence(axial._overlap(b, a), j, i)
    require(min(forward, reverse) >= axial.MIN_OVERLAP and forward > competing + axial.K_GAP_TOL
            and reverse > reverse_competing + axial.K_GAP_TOL,
            "A selected axial field lacks a unique reciprocal mass-overlap match.")
    return {"minimum_reciprocal_overlap": min(forward, reverse),
            "maximum_competing_overlap": max(competing, reverse_competing)}


def _admission(directory, mesh):
    path = directory.parent / "local_checks" / f"{directory.name}_admission_AP_pnc.json"
    saved = native._read(path)
    require(saved["mesh_ndof"] == mesh["body_dofs"],
            f"The memory projection uses a different native mesh in {directory}.")
    return {"actual_K_nonzeros": saved["actual_K_nonzeros"], "actual_M_nonzeros": saved["actual_M_nonzeros"],
            "projected_windows_private_bytes": saved["projected_private_bytes"],
            "receipt": native._link(path, ROOT)}


def _linux_preparation(directory, checkpoint):
    donor = directory.with_name(directory.name.removesuffix("_linux"))
    windows = native.read_map_checkpoint(donor)
    windows_identity = windows["inputs"]["numerical_identity"]
    linux_identity = checkpoint["inputs"]["numerical_identity"]
    assembly, solve = windows_identity["platform"], linux_identity["platform"]
    require(assembly.startswith("Windows-") and solve.startswith("Linux-"),
            f"The archived assembly and solve platforms differ from the expected Windows-to-Linux pair in {directory}.")
    require(linux_identity["source_sha256"] == windows_identity["source_sha256"],
            f"The Linux map uses different numerical source files from the Windows donor in {directory}.")
    for key in ("geometry", "mesh", "material", "analysis"):
        require(checkpoint["inputs"]["configuration"][key] == windows["inputs"]["configuration"][key],
                f"The Linux map changed the original {key} settings in {directory}.")
    require(checkpoint["shared"] == windows["shared"],
            f"The Linux map changed the original Windows shared preparation in {directory}.")
    reuse = checkpoint["preparation_reuse"]
    receipt = reuse["receipt"]
    donor_pin = native._link(donor / native.CHECKPOINT, ROOT)
    transfer_path = donor.with_name(donor.name + "_transfer_AP_pnc.json")
    transfer_pin = native._link(transfer_path, ROOT)
    receipt_bytes = (json.dumps(receipt, indent=2, allow_nan=False) + "\n").encode("utf-8")
    require(reuse["method"] == "windows_to_linux_native_preparation"
            and reuse["receipt_sha256"] == transfer_pin["sha256"] == hashlib.sha256(receipt_bytes).hexdigest()
            and receipt["donor_checkpoint_sha256"] == donor_pin["sha256"]
            and receipt["donor_publication_sha256"] == native._sha(donor / native.MAP)
            and receipt["donor_shared_sha256"] == native.workflow.digest(windows["shared"])
            and receipt["donor_source_git_commit"] == windows["source_git_commit"]
            and receipt["donor_numerical_identity"] == windows_identity
            and receipt["donor_thread_environment"] == windows["inputs"]["thread_environment"],
            f"The cross-platform preparation receipt differs from the original Windows donor in {directory}.")
    pins = {}
    for key in native.SHARED_FILES:
        original = windows["shared"]["artifacts"][key]
        native._verified(original, donor)
        native._verified(original, directory)
        pins[key] = {**original, "path": (donor / original["path"]).relative_to(ROOT).as_posix()}
    return {"assembly_platform": assembly, "solve_platform": solve, "solve_host": "Lawrence (Slurm)",
            "assembly_source_git_commit": windows["source_git_commit"],
            "solve_source_git_commit": checkpoint["source_git_commit"],
            "windows_preparation_checkpoint": donor_pin,
            "windows_transfer_receipt": transfer_pin,
            "preparation_reuse": {"method": reuse["method"], "receipt_sha256": reuse["receipt_sha256"],
                                  "exporter_source_sha256": receipt["exporter_source_sha256"]},
            "copied_shared_artifacts": pins}


def _linux_response(directory, sample, checkpoint, preparation):
    phase = f"response_{sample}"
    path = directory.parent / "local_checks" / f"{directory.name}_{phase}_execution_AP_pnc.json"
    receipt = native._read(path)
    argv = receipt["argv"]
    config = (ROOT / argv[argv.index("--config") + 1]).resolve()
    native_revision = checkpoint["source_git_commit"]
    if native_revision is None:
        expected_revision = preparation["assembly_source_git_commit"]
        expected_basis = "packaged_revision_verified_against_windows_donor"
    else:
        expected_revision = native_revision
        expected_basis = "git_head"
    job_id = receipt["slurm_job_id"]
    allocated_bytes = receipt["slurm_allocated_memory_bytes"]
    require(receipt["phase"] == phase and receipt["exit_code"] == 0 and receipt["guard"] is None
            and (ROOT / argv[argv.index("--output") + 1]).resolve() == directory
            and [float(value) for value in argv[argv.index("--frequencies-ghz") + 1:argv.index("--frequencies-ghz") + 3]]
            == checkpoint["inputs"]["frequencies_GHz"] and argv[argv.index("--max-new-points") + 1] == "1"
            and config == directory.with_name(directory.name.removesuffix("_linux") + "_configuration_AP_pnc.json")
            and native._sha(config) == receipt["configuration_sha256"]
            and expected_revision is not None and receipt["source_revision"] == expected_revision
            and receipt["source_revision_basis"] == expected_basis
            and receipt["host_platform"] == preparation["solve_platform"]
            and receipt["windows_donor_checkpoint_sha256"] == preparation["windows_preparation_checkpoint"]["sha256"]
            and all(receipt["thread_environment"][key] == value
                    for key, value in checkpoint["inputs"]["thread_environment"].items())
            and isinstance(job_id, str) and job_id.isascii() and job_id.isdigit()
            and all(isinstance(receipt[key], str) and receipt[key].strip() for key in ("slurm_partition", "slurm_node"))
            and type(allocated_bytes) is int and 0 < receipt["resident_limit_bytes"] <= allocated_bytes
            and 0 < receipt["elapsed_seconds"] <= receipt["wall_seconds"]
            and 0 < receipt["peak_sampled_process_tree_rss_sum_bytes"] <= receipt["resident_limit_bytes"]
            and receipt["peak_child_rss_bytes"] > 0,
            f"The successful guarded Linux response receipt differs from the saved map: {path}.")
    return {"receipt": native._link(path, ROOT), **{key: receipt[key] for key in (
        "elapsed_seconds", "source_revision", "source_revision_basis", "slurm_job_id", "slurm_partition",
        "slurm_node", "slurm_allocated_memory_bytes", "peak_sampled_process_tree_rss_sum_bytes", "peak_child_rss_bytes")}}


def _channel_input(record):
    summary_path = native._verified(record["archive"]["point_summary"], ROOT)
    checkpoint_path = native._verified(record["archive"]["checkpoint"], ROOT)
    checkpoint = native.read_map_checkpoint(checkpoint_path.parent)
    section = native._verified(checkpoint["shared"]["artifacts"]["port_section"], checkpoint_path.parent)
    summary, fixed, _ = scattering.read_run(summary_path.parent)
    return summary_path.parent, summary, fixed, checkpoint["inputs"]["numerical_identity"], section


def _maximum_entry(reference, candidate, incident, outgoing):
    change = candidate - reference
    row, column = np.unravel_index(np.argmax(abs(change)), change.shape)
    return row, column, {
        "maximum_absolute_change": float(abs(change[row, column])),
        "reference_drive_and_output_slot": [int(column + 1), int(row + 1)],
        "reference_incident_and_outgoing_root": [incident["reference_roots"][column], outgoing["reference_roots"][row]],
        "candidate_incident_and_outgoing_root": [incident["candidate_roots"][column], outgoing["candidate_roots"][row]],
    }


def _block_increment(block, incident, outgoing):
    reference, candidate = (np.asarray(block[name]) for name in ("reference", "candidate_aligned"))
    a, b = (value[..., 0] + 1j * value[..., 1] for value in (reference, candidate))
    pa, pb = abs(a)**2, abs(b)**2
    row, column, power = _maximum_entry(pa, pb, incident, outgoing)
    power.update(reference_power_fraction=float(pa[row, column]), candidate_power_fraction=float(pb[row, column]))
    row, column, complex_change = _maximum_entry(a, b, incident, outgoing)
    complex_change.update(reference_coefficient_re_im=reference[row, column].tolist(),
                          candidate_aligned_coefficient_re_im=candidate[row, column].tolist())
    if min(abs(a[row, column]), abs(b[row, column])) > scattering.PHASE_AMPLITUDE_FLOOR:
        complex_change["wrapped_phase_change_degrees"] = float(np.angle(b[row, column] / a[row, column], deg=True))
    drive = int(np.argmax(abs(np.asarray(block["power_change"]))))
    return {"entry_power": power, "complex_amplitude": complex_change,
            "total_drive_power": {"maximum_absolute_change": block["maximum_power_change"],
                                  "reference_drive": drive + 1,
                                  "reference_incident_root": incident["reference_roots"][drive],
                                  "candidate_incident_root": incident["candidate_roots"][drive],
                                  "reference_power_fraction": block["reference_power"][drive],
                                  "candidate_power_fraction": block["candidate_power"][drive]}}


def _channel_increment(reference, candidate):
    a_dir, a, fixed, windows, a_section = _channel_input(reference)
    b_dir, b, other, linux, b_section = _channel_input(candidate)
    require(windows["platform"].startswith("Windows-") and linux["platform"].startswith("Linux-")
            and all(windows[key] == linux[key] for key in ("source_sha256", "packages"))
            and a["num_incident_channels"] == b["num_incident_channels"] == 5
            and a["power_balance_passed"] is True and b["power_balance_passed"] is True,
            "The full-channel comparison requires accepted same-input Windows and Linux responses with five drives.")
    fixed, _ = scattering.compatible_inputs(a, b, fixed, other)
    with np.load(a_section, allow_pickle=False) as first, np.load(b_section, allow_pickle=False) as second:
        require(all(np.array_equal(first[name], second[name])
                    for name in ("points_m", "triangles", "K0", "M", "Q", "G", "reference_speed_m_s")),
                "The full-channel comparison requires identical saved section geometry and operators.")
    first = scattering.read_scattering(a_dir, a, fixed)
    second = scattering.read_scattering(b_dir, b, fixed)
    matches, blocks, _ = scattering.compare_blocks(first, second)
    power = max(block["maximum_power_change"] for block in blocks.values())
    complex_change = max(block["maximum_complex_change"] for block in blocks.values())
    return {"minimum_signed_mass_overlap": min(min(group["mass_overlaps"]) for group in matches.values()),
            "maximum_relative_wavenumber_increment": max(max(group["relative_wavenumber_change"])
                                                          for group in matches.values()),
            "blocks": {name: _block_increment(block, matches["incoming"], matches[name])
                       for name, block in blocks.items()},
            "maximum_total_drive_power_change": power, "total_drive_power_passed": power < scattering.LIMIT,
            "maximum_complex_amplitude_change": complex_change, "complex_amplitude_passed": complex_change < scattering.LIMIT}


def _full_channel_increment(earlier, later):
    return {"reference_central_max_edge_nm": earlier["central_max_edge_nm"],
            "candidate_central_max_edge_nm": later["central_max_edge_nm"],
            "execution": {"reference_assembly": "Windows", "reference_solve": "Windows",
                          "candidate_assembly": "Windows", "candidate_solve": "Linux on Lawrence"},
            "total_drive_power_limit": scattering.LIMIT, "complex_amplitude_limit": scattering.LIMIT,
            "minimum_signed_mass_overlap_required": scattering.MIN_OVERLAP,
            "method": "The five signed, unit-power incident, reflected and transmitted modes are aligned by section mass overlap. "
                      "Gates apply to absolute changes in summed T or R per drive and to absolute complex coefficient changes. "
                      "Per-entry power fractions use squared coefficient magnitudes. Phases refer to the original end planes.",
            "scope": "All four comparisons change both the central footprint resolution and the solve platform. "
                     "They quantify finite changes across the saved configurations at the two endpoint frequencies.",
            "comparison_source": native._link(Path(scattering.__file__).resolve(), ROOT),
            "samples": {sample: {role: _channel_increment(earlier["samples"][sample][role], later["samples"][sample][role])
                                 for role in pair.ROLES} for sample in fourth.SAMPLES}}


def _comparison_level(study_root, target, pilot, anchor, parent, geometry, common_mesh, reader, replay_sources,
                      previous, sensitivity, *, linux):
    samples, fields, geometry_rows = {}, {}, {}
    admission, host_provenance = {}, {}
    mesh = common_mesh if target is None else {**common_mesh, "central_max_edge_nm": target}
    for sample, hz in fourth.SAMPLES.items():
        frequency = hz / 1e9
        if target is None:
            inputs = {role: _parent_archive(parent, anchor, role, sample) for role in pair.ROLES}
            directories = {role: paths[0] for role, paths in inputs.items()}
        else:
            suffix = "_linux" if linux else ""
            directories = {role: study_root / f"{role}{thickness}_central400_edge{TARGET_NAMES[target]}_rim32{suffix}"
                           for role, (thickness, _) in pair.ROLES.items()}
        across = axial.compare_axial_power(directories["thick"], directories["thin"], frequency)
        require(across["status"] == "matched" and across["incident_side"] == pilot["incident_side"]
                and across["hole_count"] == pilot["hole_count"]
                and across["reference_planes_m"] == pilot["reference_planes_x_m"],
                across.get("reason", f"Expected matching physical sections and axial drives at {sample}."))
        samples[sample] = {"frequency_hz": hz, "cross_thickness_match": pair._overlap_bounds(across["match"])}
        for role, (thickness, comparison_role) in pair.ROLES.items():
            selected = across[comparison_role]
            directory = directories[role]
            require(selected["thickness_nm"] == thickness and selected["mesh"] == mesh,
                    f"Unexpected thickness or nested mesh parameters for {directory}.")
            case = geometry["cases"][f"{role}{thickness}"]
            measured = case["parent"] if target is None else case["targets"][f"{target:g}"]
            require(selected["inputs"]["mesh_sha256"] == measured["archive_mesh_sha256"]
                    and all(measured[f"fixed_{name}_match"] is True
                            for name in ("lead_tetrahedra", "cut_facets", "rim_facets", "port_facets"))
                    and (target is None or (measured["planar_staged_and_direct_match"] is True
                                            and 0 < measured["bulk_max_edge_nm"] <= target + 1e-9)),
                    f"The saved response differs from the checked nested mesh in {directory}.")
            geometry_rows[role] = measured
            if target is None:
                original = parent[f"axial_R_{hz // 1_000_000}MHz"][role]
                _, flux_path, pins = inputs[role]
                record = _read_sample(directory, flux_path, selected, frequency, mesh, reader, replay_sources)
                require(fourth._native_mode_agrees(selected, original)
                        and all(record["archive"][key] == pin for key, pin in pins.items()),
                        f"The saved response does not match the published 400 nm parent in {directory}.")
                _, point = axial._native_point(directory, frequency)
            else:
                if linux:
                    flux_path = directory.with_name(f"{directory.name}_{sample}_flux_AP_pnc.json")
                    record = _read_sample(directory, flux_path, selected, frequency, mesh, reader, replay_sources)
                    _, point = axial._native_point(directory, frequency)
                    if role not in host_provenance:
                        host_provenance[role] = _linux_preparation(directory, point[2])
                    record["guarded_response"] = _linux_response(directory, sample, point[2], host_provenance[role])
                else:
                    record, point = fourth._verify_sample(directory, selected, sample, frequency, mesh,
                                                          reader, replay_sources)
                require(point[2]["inputs"]["frequencies_GHz"] == [value / 1e9 for value in fourth.SAMPLES.values()],
                        f"The nested map uses different frequency samples in {directory}.")
                require(point[2]["inputs"]["numerical_identity"]["source_sha256"][REFINER] == geometry["refiner_sha256"],
                        f"The native map and checked geometry use different refinement source in {directory}.")
                if role not in admission:
                    admission[role] = _admission(directory.with_name(directory.name.removesuffix("_linux")), measured)
            require(point[2]["shared"]["mesh_hash"] == measured["mesh_hash"],
                    f"The inputs differ from the checked native mesh in {directory}.")
            samples[sample][role] = record
            saved = axial._read_case(directory, {"inputs": {}}, point)
            index, = (i for i, channel in enumerate(saved.channels) if pair._same_mode(channel, selected))
            record["incident_channels_checked"] = len(saved.channels)
            fields[(role, sample)] = (saved, index)

    lower, upper = (samples[sample] for sample in fourth.SAMPLES)
    contrast = {"995MHz": 100 * (lower["thin"]["R"] - lower["thick"]["R"]),
                "1000MHz": 100 * (upper["thin"]["R"] - upper["thick"]["R"])}
    contrast[fourth.ENDPOINT_CHANGE] = contrast["995MHz"] - contrast["1000MHz"]
    if target is None:
        require(all(math.isclose(value, parent[fourth.CONTRAST][key], rel_tol=0, abs_tol=1e-12)
                    for key, value in contrast.items()), "The replayed 400 nm parent contrast differs from its publication.")
    for role in pair.ROLES:
        require(all(lower[role]["original_equation_identity"][key] != upper[role]["original_equation_identity"][key]
                    for key in lower[role]["original_equation_identity"]),
                f"The two {role} endpoints reuse an original equation identity.")
    result = {"central_max_edge_nm": target, "geometry": geometry_rows, "samples": samples,
              fourth.CONTRAST: contrast,
              "within_thickness_endpoint_matches": {
                  role: _mode_match(fields[(role, "995MHz")], fields[(role, "1GHz")], same_section=True)
                  for role in pair.ROLES}}
    if admission:
        result["resource_admission"] = admission
    if host_provenance:
        result["host_provenance"] = host_provenance
    if previous is not None:
        earlier, earlier_fields = previous
        increments = {key: value - earlier[fourth.CONTRAST][key] for key, value in contrast.items()}
        increment = abs(increments[fourth.ENDPOINT_CHANGE])
        result["change_from_previous_reported_level"] = {
            "previous_central_max_edge_nm": earlier["central_max_edge_nm"],
            "signed_increments_percentage_points": increments,
            "absolute_endpoint_change_increment_over_target": increment / sensitivity,
            "endpoint_change_increment_meets_target": increment <= sensitivity,
            "within_thickness_matches": {
                sample: {role: _mode_match(earlier_fields[(role, sample)], fields[(role, sample)])
                         for role in pair.ROLES} for sample in fourth.SAMPLES},
        }
        if linux and earlier["central_max_edge_nm"] == 225:
            result["change_from_previous_reported_level"]["full_channel_comparison"] = _full_channel_increment(earlier, result)
    return result, fields


def compare_nested_mesh(study_root, targets, *, linux_168p75=False):
    study_root = Path(study_root).resolve()
    require(list(targets) == [target for target in TARGET_NAMES if target in targets],
            "Choose distinct nested levels in descending order: 300, 225, 168.75 nm.")
    require(not linux_168p75 or 168.75 in targets, "Include 168.75 nm in --targets to select the Linux maps.")
    old = native._read(FOURTH)
    pilot_pin = old["provenance"]["frozen_three_mesh_pilot"]
    pilot = native._read(native._verified(pilot_pin, ROOT))
    anchor_pin = pilot["provenance"]["published_1GHz_anchor"]
    anchor = native._read(native._verified(anchor_pin, ROOT))
    replay_reader = pilot["rim32_to_rim48_control"]["acceptance_replay_reader"]
    reader = runpy.run_path(str(native._verified(replay_reader, ROOT)))["read_point"]
    source_pins = anchor["provenance"]["independent_flux_sources"]
    for pin in source_pins:
        native._verified(pin, ROOT)
    replay_sources = [pin["sha256"] for pin in source_pins]
    parent = pilot["rim32_central_refinement"]["samples"][0]
    common_mesh = {**old["mesh"], "mesh_size_nm": parent["central_target_nm"]}
    require(common_mesh["mesh_size_nm"] == 400 and common_mesh["circle_resolution"] == 32
            and common_mesh["thickness_layers"] == 2 and common_mesh["lead_mesh_size_nm"] == 150,
            "The prior records do not describe the 400 nm, two-layer, rim32 parent with 150 nm leads.")
    sensitivity = old["exploratory_endpoint_contrast_change_target"]["percentage_points"]
    previous_225 = pilot["rim32_central_refinement"]["samples"][-1]
    require(previous_225["central_target_nm"] == 225 and sensitivity > 0
            and math.isclose(sensitivity, 0.10 * abs(previous_225[fourth.CONTRAST][fourth.ENDPOINT_CHANGE]),
                             rel_tol=0, abs_tol=1e-15), "The inherited endpoint sensitivity target differs from the pilot.")
    geometry_path = study_root / "nested_central_geometry_AP_pnc.json"
    geometry = native._read(geometry_path)
    verifier = geometry["verifier"]
    verifier_path = (ROOT / verifier["path"]).resolve()
    require(verifier_path.is_relative_to(ROOT) and native._sha(verifier_path) == verifier["sha256"],
            "The archived geometry verifier differs from the checked geometry record.")
    levels = []
    previous = None
    for target in (None, *targets):
        previous = _comparison_level(study_root, target, pilot, anchor, parent, geometry, common_mesh, reader,
                                     replay_sources, previous, sensitivity, linux=linux_168p75 and target == 168.75)
        levels.append(previous[0])
    unreported_levels = []
    for target, label in TARGET_NAMES.items():
        if target in targets:
            continue
        rows = {}
        for role, (thickness, _) in pair.ROLES.items():
            mesh = geometry["cases"][f"{role}{thickness}"]["targets"].get(f"{target:g}")
            if mesh is None:
                continue
            directory = study_root / f"{role}{thickness}_central400_edge{label}_rim32"
            rows[role] = {"geometry": mesh, "resource_admission": _admission(directory, mesh)}
        unreported_levels.append({"central_max_edge_nm": target, **rows})
    return {"schema_version": 1, **native.AUTHOR, "status": "completed_selected_nested_comparison",
            "levels": levels, "levels_without_paired_comparison": unreported_levels,
            "parent_mesh": common_mesh, "physical_thickness_nm": pilot["physical_thickness_nm"],
            "hole_count": pilot["hole_count"], "incident_side": pilot["incident_side"],
            "reference_planes_x_m": pilot["reference_planes_x_m"],
            "original_numerical_limits": pilot["original_numerical_limits"],
            "observable": "D = C(995 MHz) - C(1000 MHz), with C(f) = 100 [R_100nm(f) - R_200nm(f)] in percentage points.",
            "exploratory_endpoint_change_increment_target_percentage_points": sensitivity,
            "exploratory_target_basis": "Ten percent of the absolute D on the prior independently remeshed 225 nm control.",
            "resource_projection_scope": "The Windows private-memory projection scales previous Windows direct-solve "
                                         "peaks by stiffness nonzero count. Windows response receipts report private "
                                         "commit. Lawrence receipts give their Slurm memory allocation, sampled process-tree "
                                         "resident set size and kernel-reported peak child resident set size. Factor fill and "
                                         "available system resources can vary on another run.",
            "scope": "T and R are fractions of each native incident drive's power at the original end planes. Matches use "
                     "the +x incident fields and reciprocal mass-weighted overlaps at fixed y and normalized thickness. "
                     "They support correspondence at the two sampled frequencies. The observed D increment measures finite "
                     "response sensitivity. Continuum error, continuous branch identity, all-channel complex convergence, "
                     "through-thickness and hole-wall uncertainty, and frequency convergence remain unresolved.",
            "provenance": {"parent_paired_frequency_record": pilot_pin, "parent_1GHz_record": anchor_pin,
                           "inherited_sensitivity_target": native._link(FOURTH, ROOT),
                           "checked_geometry": native._link(geometry_path, ROOT), "acceptance_reader": replay_reader,
                           "independent_flux_sources": source_pins,
                           "comparison_source": native._link(Path(__file__).resolve(), ROOT)}}


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--study-root", type=Path, default=STUDY_ROOT, help="Local native run archive.")
    parser.add_argument("--targets", type=float, nargs="*", choices=TARGET_NAMES, default=list(TARGET_NAMES),
                        help="Replay these saved refinement targets in descending order. An empty list replays the parent only.")
    parser.add_argument("--linux-168p75", action="store_true",
                        help="Use the Lawrence solves with copied Windows preparation at 168.75 nm.")
    parser.add_argument("--output", type=Path, help="Write the comparison JSON outside the saved run archive.")
    args = parser.parse_args(argv)
    try:
        payload = json.dumps(compare_nested_mesh(args.study_root, args.targets, linux_168p75=args.linux_168p75),
                             indent=2, allow_nan=False) + "\n"
        if args.output:
            output = args.output.resolve()
            require(output.suffix == ".json" and not output.is_relative_to((ROOT / "results/runs").resolve())
                    and output not in (FOURTH, fourth.PILOT, pair.ANCHOR),
                    "Choose a distinct JSON output path outside the saved run archive.")
            temporary = output.with_suffix(".json.tmp")
            temporary.write_text(payload, encoding="utf-8", newline="\n")
            temporary.replace(output)
        else:
            print(payload, end="")
    except (KeyError, OSError, TypeError, ValueError) as exc:
        parser.exit(1, f"{parser.prog}: {exc}\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
