#!/usr/bin/env python3
"""Replay the saved 1 GHz layer control for the thin device. Author: Athul Prem | _AP_pnc."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import runpy
import shlex
import sys

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if __name__ == "__main__":
    sys.path.insert(0, str(ROOT))

from tools import compare_axial_nested_mesh_AP_pnc as nested
from tools import compare_scattering_AP_pnc as scattering
from tools import map_scattering_AP_pnc as native
from tools.compare_scattering_AP_pnc import require
from phononic.port_modes_AP_pnc import _flux_gram


CENTRAL = ROOT / "verification/thin_device_nested_response_AP_pnc.json"
LAYERS = ROOT / "verification/axial_device_three_layer_pairs_1GHz_AP_pnc.json"
STUDY = ROOT / "results/runs/physical_thickness_device_AP_pnc"
OUTPUT = ROOT / "verification/thin_device_layer_response_AP_pnc.json"
FREQUENCY_GHZ = 1.0
CHANNEL_COUNT = 5
NATIVE_POWER_ROUNDOFF = 1e-12
QUADRATIC_LAYERS = (1, 2, 3)
PORTS = {"incoming": "left", "reflection": "left", "transmission": "right"}
POWERS = {"transmission": "T", "reflection": "R"}
FINE = STUDY / "thin100_central400_edge126p5625_z3_rim32_linux"
FINE_ADMISSION = STUDY / "local_checks/thin100_central126p5625_z3_mesh_admission_AP_pnc.json"
FINE_TARGET_NM = 126.5625
COARSE = STUDY / "thin100_central400_edge168p75_z3_rim32_linux"
COARSE_ADMISSION = STUDY / "local_checks/thin100_central168p75_z3_mesh_admission_AP_pnc.json"
COARSE_TARGET_NM = 168.75


def _read_modes(record, anchor=None):
    directory, summary, fixed, identity, _ = nested._channel_input(record)
    if anchor is not None:
        prior_summary, prior_fixed, prior_identity = anchor
        scattering.compatible_inputs(prior_summary, summary, prior_fixed, fixed)
        require(prior_identity["packages"] == identity["packages"],
                "The saved layer case and central anchor use different numerical packages.")
    require(summary["num_incident_channels"] == CHANNEL_COUNT and summary["power_balance_passed"] is True,
            f"Expected five accepted drives in {directory}.")
    return scattering.read_scattering(directory, summary, fixed), (summary, fixed, identity)


def _read_case(layer, published, geometry, fixed_mesh, references, reader, flux_sources):
    directory = (ROOT / published["map_directory"]).resolve()
    flux = directory.with_name(f"{directory.name.removesuffix('_1GHz')}_flux_AP_pnc.json")
    record, config, holes = reader(directory, flux, FREQUENCY_GHZ)
    replay = native._read(flux)
    require(all(record["archive"][key]["sha256"] == published[f"{key}_sha256"] for key in ("map", "checkpoint"))
            and record["reference_planes_x_m"] == geometry["reference_planes_x_m"]
            and config["mesh"] == {**fixed_mesh, "thickness_layers": layer}
            and config["material"] == geometry["material"]
            and config["geometry"]["thickness_nm"] == record["physical_thickness_nm"] == 100
            and len(holes["holes"]) == geometry["hole_count"] == 26
            and holes["source_sha256"] == geometry["hole_pattern_csv"]["sha256"]
            and [replay[key] for key in ("replay_source_sha256", "regional_replay_source_sha256")] == flux_sources,
            f"The saved inputs or independent flux producers differ in {directory}.")
    case = geometry["cases"][f"thin_z{layer}"]
    checkpoint = native.read_map_checkpoint(directory)
    require(checkpoint["inputs_sha256"] == case["map_inputs_sha256"]
            and native.workflow.digest(checkpoint["shared"]) == case["shared_sha256"]
            and record["archive"]["mesh"]["sha256"] == case["mesh"]["mesh_sha256"]
            and record["archive"]["port_section"]["sha256"] == case["port_section"]["sha256"]
            and record["body_dofs"] == case["mesh"]["ndof"]
            and record["port_dofs"] == case["port"]["vector_dofs_at_each_end"]
            and case["port"]["saved_section_matches_both_tagged_P2_body_traces"] is True,
            f"The geometry and both end traces differ from the saved verification in {directory}.")
    data, (_, _, identity) = _read_modes(record, references)
    facts = _case_evidence(record)
    facts["platform"] = identity["platform"]
    return facts, data, record["native_drives"]


def _case_evidence(record):
    facts = {key: record[key] for key in ("body_dofs", "port_dofs", "source_git_commit", "map_driver_sha256",
                                         "maximum_numerical_checks", "independent_flux_checks")}
    facts["archive"] = {key: record["archive"][key] for key in (
        "map", "checkpoint", "point_summary", "port_section", native.FIELDS, "scattering_AP_pnc.csv", "independent_flux_replay")}
    return facts


def _anchor_match(matches, anchor_pair):
    selected = {}
    for group, port in PORTS.items():
        match = matches[group]
        index = match["reference_roots"].index(anchor_pair[group])
        selected[group] = {
            "port": port,
            "native_root_id": match["candidate_roots"][index],
            "signed_wavenumber_rad_m": match["candidate_wavenumber_rad_m"][index],
            "mass_overlap": match["mass_overlaps"][index],
            "gauge_phase_rad": match["candidate_gauge_phase_rad"][index],
        }
    require(selected["incoming"]["native_root_id"] == selected["transmission"]["native_root_id"],
            "The two end planes give different matches for the same forward mode.")
    return selected


def _sample(facts, data, drives, anchor, anchor_pair):
    matches, _, _ = scattering.compare_blocks(anchor, data)
    selected = _anchor_match(matches, anchor_pair)
    drive = next(row for row in drives if row["native_root_id"] == selected["incoming"]["native_root_id"])
    return {**facts, "signed_anchor_match": selected, "native_drive": drive}


def _checked_flux_basis(modes, coherent, fields_path, limit):
    groups = {"incident": "incoming", "reflected": "reflection", "transmitted": "transmission"}
    roots = {key: modes[group].roots for key, group in groups.items()}
    require(all(roots[key].tolist() == coherent[f"{key}_roots"] for key in groups)
            and np.array_equal(roots["incident"], roots["transmitted"]),
            f"The propagating root order differs from the independent flux replay: {fields_path}.")
    size = CHANNEL_COUNT
    indices = np.concatenate((roots["incident"], roots["reflected"]))
    with np.load(fields_path, allow_pickle=False) as fields:
        require(np.all(fields["frequencies_hz"] == FREQUENCY_GHZ * 1e9),
                f"The saved flux basis is for a different frequency: {fields_path}.")
        displacement = fields["port_displacements"][:, indices]
        force = fields["port_axial_forces"][:, indices]
    signed = _flux_gram(displacement, force, 2 * np.pi * FREQUENCY_GHZ * 1e9)
    identity = np.eye(size)
    errors = {"incident": signed[:size, :size] - identity,
              "reflected": -signed[size:, size:] - identity,
              "transmitted": signed[:size, :size] - identity}
    gram_maxima = {key: float(np.max(abs(error))) for key, error in errors.items()}
    cross = float(np.max(abs(signed[:size, size:])))
    require(coherent["flux_gram_entry_limit_w"] == limit
            and all(error < limit for error in (*gram_maxima.values(), cross)),
            f"The signed propagating flux Gram fails the original 1 W basis limit: {fields_path}.")
    saved_maxima = [max(coherent["flux_gram_errors"][key].values()) for key in groups]
    require(np.allclose([*gram_maxima.values(), cross],
                        [*saved_maxima, coherent["flux_gram_cross_direction_maximum_w"]],
                        rtol=0, atol=NATIVE_POWER_ROUNDOFF),
            f"The recomputed flux Gram differs from the independent replay: {fields_path}.")
    return {**gram_maxima, "cross_direction": cross}


def _transmission_eigenchannels(record, data, flux, limits):
    _, modes, blocks = data
    drives = record["native_drives"]
    fields_path = native._verified(record["archive"][native.FIELDS], ROOT)
    require(len(drives) == CHANNEL_COUNT
            and [row["drive"] for row in drives] == list(range(1, CHANNEL_COUNT + 1))
            and [row["native_root_id"] for row in drives] == modes["incoming"].roots.tolist(),
            f"Native drive powers are out of order: {fields_path}.")
    require(all(block.shape == (CHANNEL_COUNT, CHANNEL_COUNT) and np.isfinite(block).all()
                for block in blocks.values()), f"Expected finite native five-channel blocks: {fields_path}.")
    coherent = flux["coherent_left_incidence"]
    gram_maxima = _checked_flux_basis(modes, coherent, fields_path, limits["flux_gram_entry_W"])

    operators = {key: block.conj().T @ block for key, block in blocks.items()}
    defect = operators["reflection"] + operators["transmission"] - np.eye(CHANNEL_COUNT)
    defect_norm = float(np.max(abs(np.linalg.eigvalsh(defect))))
    limit = limits["single_drive_and_coherent_power_defect"]
    require(coherent["coherent_power_limit"] == limit and defect_norm < limit,
            f"Native reflections and transmissions fail the original coherent power limit: {fields_path}.")
    saved_defect = np.asarray(coherent["defect_matrix_real"]) + 1j * np.asarray(coherent["defect_matrix_imag"])
    require(saved_defect.shape == defect.shape, f"Invalid independent coherent power matrix: {fields_path}.")
    maximum_defect_difference = float(np.max(abs(defect - saved_defect)))
    require(maximum_defect_difference < NATIVE_POWER_ROUNDOFF
            and abs(defect_norm - coherent["coherent_coefficient_power_defect_spectral_norm"]) < NATIVE_POWER_ROUNDOFF,
            f"The native coherent power matrix differs from the independent replay: {fields_path}.")

    native_powers = {key: np.asarray([row[key] for row in drives]) for key in POWERS}
    maximum_column_differences = {
        short: float(np.max(abs(np.diag(operators[key]) - native_powers[key])))
        for key, short in POWERS.items()
    }
    require(all(error < NATIVE_POWER_ROUNDOFF for error in maximum_column_differences.values()),
            f"Column powers differ from the original native drive totals: {fields_path}.")
    # NumPy returns singular values in descending order. Keep every result as computed.
    eigenvalues = np.linalg.svd(blocks["transmission"], compute_uv=False)**2
    trace = float(np.trace(operators["transmission"]).real)
    native_trace = float(np.sum(native_powers["transmission"]))
    require(eigenvalues[-1] >= 0 and eigenvalues[0] <= 1 + defect_norm + NATIVE_POWER_ROUNDOFF
            and abs(trace - native_trace) < NATIVE_POWER_ROUNDOFF
            and abs(float(np.sum(eigenvalues)) - trace) < NATIVE_POWER_ROUNDOFF,
            f"The transmission eigenvalues or trace fail the native power checks: {fields_path}.")
    return {
        "eigenvalues_descending": eigenvalues.tolist(),
        "trace": trace,
        "equal_incoherent_mean_power_fraction": trace / CHANNEL_COUNT,
        "checks": {
            "maximum_flux_gram_entry_errors_W": gram_maxima,
            "coherent_coefficient_power_defect_spectral_norm": defect_norm,
            "maximum_defect_entry_difference_from_independent_replay": maximum_defect_difference,
            "maximum_column_power_differences_from_native_drives": maximum_column_differences,
            "trace_minus_sum_of_native_transmission_powers": trace - native_trace,
            "sum_of_eigenvalues_minus_trace": float(np.sum(eigenvalues)) - trace,
        },
    }


def _eigenchannel_changes(reference, candidate):
    before, after = (np.asarray(case["eigenvalues_descending"]) for case in (reference, candidate))
    changes = after - before
    return {
        "signed_eigenvalue_changes_percentage_points_descending_order": (100 * changes).tolist(),
        "signed_trace_change": candidate["trace"] - reference["trace"],
        "signed_equal_incoherent_mean_change_percentage_points": 100 * (
            candidate["equal_incoherent_mean_power_fraction"] - reference["equal_incoherent_mean_power_fraction"]),
        "power_operator_norm_lower_bound_after_unitary_basis_changes": float(np.max(abs(changes))),
        "amplitude_operator_norm_lower_bound_after_unitary_basis_changes": float(np.max(abs(np.sqrt(after) - np.sqrt(before)))),
    }


def _increment(reference, candidate, first, second, central_delta=None, *, native_drives=None):
    matches, blocks, rows = scattering.compare_blocks(first, second)
    for group, match in matches.items():
        index = match["reference_roots"].index(reference["signed_anchor_match"][group]["native_root_id"])
        require(match["candidate_roots"][index] == candidate["signed_anchor_match"][group]["native_root_id"],
                f"The successive {group} match disagrees with the central anchor.")
    index = reference["native_drive"]["drive"] - 1
    power = {}
    for quantity, short in POWERS.items():
        a, b = (case["native_drive"][quantity] for case in (reference, candidate))
        require(np.allclose([a, b], [blocks[quantity][key][index] for key in ("reference_power", "candidate_power")],
                            rtol=0, atol=NATIVE_POWER_ROUNDOFF), f"Native {short} differs from the matched complex block.")
        power[short] = 100.0 * (b - a)
    max_power = max(block["maximum_power_change"] for block in blocks.values())
    max_complex = max(block["maximum_complex_change"] for block in blocks.values())
    result = {"signed_change_percentage_points": power}
    if central_delta is not None:
        result["absolute_T_increment_over_finest_central_increment"] = abs(power["T"] / central_delta)
    result.update(minimum_signed_mass_overlap=min(min(match["mass_overlaps"]) for match in matches.values()),
                  maximum_total_drive_power_change=max_power, total_drive_power_passed=max_power < scattering.LIMIT,
                  maximum_complex_amplitude_change=max_complex, complex_amplitude_passed=max_complex < scattering.LIMIT)
    if native_drives is not None:
        before, after = ({row["native_root_id"]: row for row in drives} for drives in native_drives)
        a, b = (matches["incoming"][key] for key in ("reference_roots", "candidate_roots"))
        per_drive = []
        for i, (root_a, root_b) in enumerate(zip(a, b, strict=True)):
            a_row, b_row = before[root_a], after[root_b]
            changes = {}
            for quantity, short in POWERS.items():
                native_power = [a_row[quantity], b_row[quantity]]
                require(np.allclose(native_power, [blocks[quantity][key][i] for key in ("reference_power", "candidate_power")],
                                    rtol=0, atol=NATIVE_POWER_ROUNDOFF),
                        f"Native {short} differs from the matched complex block on drive {i + 1}.")
                changes[short] = 100.0 * (native_power[1] - native_power[0])
            complex_change = max(row["maximum_complex_change"] for row in rows if row["incident_channel"] == i + 1)
            per_drive.append({"reference_drive": a_row["drive"], "candidate_drive": b_row["drive"],
                              "reference_root_id": root_a, "candidate_root_id": root_b,
                              "signed_change_percentage_points": changes,
                              "total_drive_power_passed": max(map(abs, changes.values())) < 100 * scattering.LIMIT,
                              "maximum_complex_amplitude_change": complex_change,
                              "complex_amplitude_passed": complex_change < scattering.LIMIT})
        result["per_drive"] = per_drive
    return result


def _linux_time(directory, configuration):
    path = STUDY / "local_checks" / f"{directory.name}_1GHz_time_AP_pnc.txt"
    fields = dict(line.strip().split(": ", maxsplit=1)
                  for line in path.read_text(encoding="utf-8").splitlines() if ": " in line)
    argv = shlex.split(fields["Command being timed"].strip('"'))
    peak_rss_kib = int(fields["Maximum resident set size (kbytes)"])
    exit_status = int(fields["Exit status"])
    require("tools/map_scattering_AP_pnc.py" in argv
            and (ROOT / argv[argv.index("--config") + 1]).resolve() == ROOT / configuration["path"]
            and (ROOT / argv[argv.index("--output") + 1]).resolve() == directory
            and float(argv[argv.index("--frequencies-ghz") + 1]) == FREQUENCY_GHZ
            and argv[argv.index("--max-new-points") + 1] == "1"
            and exit_status == 0 and peak_rss_kib > 0,
            "The GNU time record differs from the accepted native layer-control invocation.")
    return {"receipt": native._link(path, ROOT), "exit_status": exit_status, "peak_rss_kib": peak_rss_kib,
            "elapsed_wall_time": fields["Elapsed (wall clock) time (h:mm:ss or m:ss)"]}


def _coarser_footprint(audit, admission, checkpoint, mesh_manifest, second_layer):
    reference = native._verified(admission["accepted_reference_checkpoint"], ROOT)
    identity = checkpoint["inputs"]["numerical_identity"]
    require(admission["status"] == "passed_mesh_only" and admission["frequency_hz"] == FREQUENCY_GHZ * 1e9
            and admission["physical_thickness_nm"] == second_layer["physical_thickness_nm"] == 100
            and admission["reference_planes_x_m"] == second_layer["reference_planes_x_m"]
            and admission["exact_native_source_match_count"] == len(identity["source_sha256"])
            and admission["native_source_sha256"] == native.workflow.digest(identity["source_sha256"])
            and admission["shared_package_versions"] == identity["packages"]
            and native.workflow.resolve_config(native._read(native._verified(admission["configuration"], ROOT)))
            == checkpoint["inputs"]["configuration"],
            "The coarser Windows admission and accepted Linux configuration or numerical source differ.")
    old_mesh, old = audit["archived_parent"](reference.parent)
    new_mesh, new = native.workflow.load_checkpoint(mesh_manifest.parent, checkpoint["shared"]["mesh_hash"])
    old_inputs, new_inputs = (mesh["mesh_inputs"] for mesh in (old, new))
    require(new_inputs["settings"] == checkpoint["inputs"]["configuration"]["mesh"]
            and all(old_inputs[key] == new_inputs[key] for key in ("dimensions_m", "pattern"))
            and all(new_inputs["pattern"][key] == checkpoint["inputs"]["hole_pattern"][key]
                    for key in ("source_sha256", "holes")),
            "The actual coarser mesh changes the hole pattern, physical dimensions or layer settings.")
    xy, triangles, slabs, rims, stats = audit["inspect"](old_mesh, old)
    xy3, triangles3, slabs3, rims3, stats3 = audit["inspect"](new_mesh, new)
    expected = admission["footprint"]
    require(np.array_equal(xy, xy3) and np.array_equal(triangles, triangles3)
            and native._array_digest(xy, triangles) == expected["ordered_xy_and_triangles_sha256"]
            and native._array_digest(slabs[0]) == expected["one_slab_ordered_connectivity_sha256"]
            and all(np.array_equal(slab, slabs[0]) for slab in slabs + slabs3)
            and rims == rims3 and len(rims) == expected["exact_projected_hole_rim_edges"] == 26 * 32
            and stats == admission["accepted_linux_z2"]
            and all(stats3[key] == value for key, value in admission["candidate_mesh_z3"].items()
                    if key not in ("mesh_hash", "mesh_build_resources")),
            "The native coarser three-layer mesh differs from the admitted footprint, layer topology or tagged traces.")
    return {"footprint": expected, "configuration": admission["configuration"],
            "accepted_reference_checkpoint": admission["accepted_reference_checkpoint"],
            "accepted_reference_mesh": admission["accepted_reference_mesh"],
            "candidate_preflight": native._link(mesh_manifest, ROOT),
            "candidate_mesh": native._link(mesh_manifest.with_name("mesh_AP_pnc.npz"), ROOT),
            "candidate_mesh_z3": stats3,
            "candidate_mesh_python_platform": {key: new_inputs[key] for key in ("python", "platform")}}


def _footprint_geometry(directory, checkpoint, record, summary, second_layer, admission_path):
    admission = native._read(admission_path)
    require(__debug__, "Run the geometry verifier with Python assertions enabled.")
    verifier = native._verified(admission["verification_source"], ROOT)
    mesh_manifest = native._verified(checkpoint["shared"]["artifacts"]["mesh_summary"], directory)
    audit = runpy.run_path(str(verifier))
    try:
        verified = (audit["check"](mesh_manifest.parent) if admission_path == FINE_ADMISSION else
                    _coarser_footprint(audit, admission, checkpoint, mesh_manifest, second_layer))
    except AssertionError as exc:
        raise ValueError(f"Fixed-footprint verification failed for the Linux mesh at {mesh_manifest.parent}. {exc}") from exc
    candidate = verified["candidate_mesh_z3"]
    require(verified["footprint"] == admission["footprint"]
            and verified["accepted_reference_checkpoint"] == second_layer["archive"]["checkpoint"]
            and verified["accepted_reference_mesh"] == second_layer["archive"]["mesh"]
            and verified["configuration"] == admission["configuration"]
            and verified["candidate_preflight"] == native._link(mesh_manifest, ROOT)
            and verified["candidate_mesh"] == record["archive"]["mesh"]
            and candidate["mesh_hash"] == checkpoint["shared"]["mesh_hash"] == record["mesh_hash"]
            and candidate["body_dofs"] == record["body_dofs"]
            and candidate["end_face_dofs"] == summary["end_face_dofs"]
            and all(value == record["port_dofs"] for value in candidate["end_face_dofs"].values())
            and candidate["compatible_tagged_p2_end_traces"] is True,
            "The fixed footprint or tagged port traces differ on the actual Linux preparation.")
    with np.load(native._verified(record["archive"]["port_section"], ROOT), allow_pickle=False) as section, \
            np.load(native._verified(record["archive"][native.FIELDS], ROOT), allow_pickle=False) as fields:
        require(all(np.array_equal(section[f"{side}_dofs"], fields[f"port_{side}_dofs"]) for side in ("left", "right")),
                "The response fields and native port preparation differ at an end trace.")
    return {"footprint": verified["footprint"], "windows_mesh_admission": native._link(admission_path, ROOT),
            "geometry_verifier": admission["verification_source"], "configuration": admission["configuration"],
            "accepted_two_layer_mesh": verified["accepted_reference_mesh"],
            "linux_three_layer_mesh": verified["candidate_mesh"], "linux_three_layer_mesh_summary": verified["candidate_preflight"]}, \
        verified["candidate_mesh_python_platform"]


def _footprint_pair(central, reader, anchor, anchor_pair, references, flux_sources, *, target, directory, admission):
    saved = central["levels"][str(target)]
    two_checkpoint = native._verified(saved["archive"]["checkpoint"], ROOT)
    two_flux = native._verified(saved["archive"]["independent_flux_replay"], ROOT)
    two_record, two_config, two_holes = reader(two_checkpoint.parent, two_flux, FREQUENCY_GHZ)
    three_flux = directory.with_name(f"{directory.name}_1GHz_flux_AP_pnc.json")
    three_record, three_config, three_holes = reader(directory, three_flux, FREQUENCY_GHZ)
    checkpoint = native.read_map_checkpoint(directory)
    two_replay, three_replay = native._read(two_flux), native._read(three_flux)
    require(two_record["archive"] == saved["archive"]
            and two_record["native_drives"] == saved["native_drives"]
            and two_config["mesh"]["central_max_edge_nm"] == target
            and two_config["mesh"]["thickness_layers"] == 2
            and three_config == {**two_config, "mesh": {**two_config["mesh"], "thickness_layers": 3},
                                 "resources": {**two_config["resources"], "max_dofs": three_record["body_dofs"]}}
            and three_holes == two_holes and three_record["physical_thickness_nm"] == saved["physical_thickness_nm"] == 100
            and three_record["reference_planes_x_m"] == saved["reference_planes_x_m"] == central["reference_planes_x_m"]
            and all([flux[key] for key in ("replay_source_sha256", "regional_replay_source_sha256")] == flux_sources
                    for flux in (two_replay, three_replay))
            and checkpoint["inputs"]["frequencies_GHz"] == [FREQUENCY_GHZ]
            and "preparation_reuse" not in checkpoint,
            "The fixed-footprint pair differs in settings, original planes or independent flux provenance.")
    two_data, (_, _, two_identity) = _read_modes(two_record, references)
    three_data, (summary, _, three_identity) = _read_modes(three_record, references)
    geometry, mesh_host = _footprint_geometry(directory, checkpoint, three_record, summary, saved, admission)
    two_state = native.read_map_checkpoint(two_checkpoint.parent)
    two_reuse = two_state.get("preparation_reuse", {}).get("receipt", {})
    two_assembly = two_reuse.get("donor_numerical_identity", two_identity)
    require(all(identity["platform"].startswith("Linux-") and "-x86_64-" in identity["platform"]
                for identity in (two_identity, three_identity))
            and two_assembly["platform"] == saved["platform"]["assembly"]
            and two_identity["platform"] == saved["platform"]["solve"]
            and all(two_assembly[key] == two_identity[key] for key in ("source_sha256", "packages"))
            and all(mesh_host[key] == summary["implementation"][key] == three_identity[key] for key in ("python", "platform"))
            and three_identity["source_sha256"] == two_identity["source_sha256"]
            and three_identity["packages"] == two_identity["packages"]
            and all(three_identity["source_sha256"][path] == sha
                    for path, sha in summary["implementation"]["source_sha256"].items())
            and summary["assembly_resources"] == checkpoint["shared"]["assembly_resources"],
            "The three-layer response must use native Linux preparation and the common numerical source.")
    measured = _linux_time(directory, geometry["configuration"])
    equation_keys = ("operator_csc_sha256", "rhs_sha256")
    two_equations = two_replay["original_equations_saved"]
    equations = {key: three_replay["original_equations_saved"][key] for key in equation_keys}
    require(all(two_equations[key] == saved["original_equation_identity"][key] != equations[key] for key in equation_keys),
            "The saved equation identities differ from the two-layer publication or are shared by the three-layer calculation.")
    two = {**_case_evidence(two_record), "platform": saved["platform"],
           "original_equation_identity": saved["original_equation_identity"]}
    if "measured_linux_map_execution" in saved:
        two["measured_linux_map_execution"] = saved["measured_linux_map_execution"]
    two = _sample(two, two_data, two_record["native_drives"], anchor, anchor_pair)
    three = _sample({**_case_evidence(three_record),
                     "platform": {"assembly": mesh_host["platform"], "solve": three_identity["platform"]},
                     "original_equation_identity": equations, "measured_linux_map_execution": measured},
                    three_data, three_record["native_drives"], anchor, anchor_pair)
    limits = central["original_numerical_limits"]
    two["transmission_eigenchannels"] = _transmission_eigenchannels(two_record, two_data, two_replay, limits)
    three["transmission_eigenchannels"] = _transmission_eigenchannels(three_record, three_data, three_replay, limits)
    return (geometry, (two, two_data, two_record["native_drives"]), (three, three_data, three_record["native_drives"]))


def _fine_comparison(geometry, two, three, central_delta, parent_increment):
    increment = _increment(two[0], three[0], two[1], three[1], central_delta)
    interaction = {key: value - parent_increment["signed_change_percentage_points"][key]
                   for key, value in increment["signed_change_percentage_points"].items()}
    return {"central_max_edge_nm": FINE_TARGET_NM, "fixed_geometry": geometry, "levels": {2: two[0], 3: three[0]},
            "increment_2_to_3": increment, "fine_minus_parent_increment_percentage_points": interaction}


def _fixed_within_layer(first, second, audit):
    archives = [case["archive"] for case in (first, second)]
    meshes = [audit["archived_parent"](native._verified(archive["checkpoint"], ROOT).parent) for archive in archives]
    first_mesh, first_manifest = meshes[0]
    second_mesh, second_manifest = meshes[1]
    a, b = first_manifest["mesh_inputs"], second_manifest["mesh_inputs"]
    require(a["settings"] == {**b["settings"], "central_max_edge_nm": COARSE_TARGET_NM}
            and a["settings"]["central_max_edge_nm"] == COARSE_TARGET_NM
            and b["settings"]["central_max_edge_nm"] == FINE_TARGET_NM
            and all(a[key] == b[key] for key in ("pattern", "dimensions_m"))
            and len(a["pattern"]["holes"]) == 26,
            "Central refinement changed geometry settings beyond the edge ceiling.")
    cuts = [nm * 1e-9 for nm in a["settings"]["patterned_x_interval_nm"]]
    reference, candidate = (audit["fixed_geometry"](mesh, cuts) for mesh in (first_mesh, second_mesh))
    require(reference == candidate, "Central refinement altered fixed lead cells, cuts, hole rims or port facets.")
    sections = [native._verified(archive["port_section"], ROOT) for archive in archives]
    with np.load(sections[0], allow_pickle=False) as before, np.load(sections[1], allow_pickle=False) as after:
        require(all(np.array_equal(before[key], after[key]) for key in ("points_m", "triangles")),
                "Central refinement changed the native matched-port section coordinates or topology.")
    return {**dict.fromkeys(reference, True), "section_coordinates_and_topology": True}


def _factorial_comparison(coarse, fine, central):
    coarse_geometry, coarse_two, coarse_three = coarse
    _, fine_two, fine_three = fine
    geometry_pin = central["fixed_geometry"]["geometry_verifier"]
    audit = runpy.run_path(str(native._verified(geometry_pin, ROOT)))

    def edge(a, b):
        require(all(a[0]["original_equation_identity"][key] != b[0]["original_equation_identity"][key]
                    for key in a[0]["original_equation_identity"]), "The compared solves reuse an original equation identity.")
        result = _increment(a[0], b[0], a[1], b[1], native_drives=(a[2], b[2]))
        result["transmission_eigenchannel_changes"] = _eigenchannel_changes(
            a[0]["transmission_eigenchannels"], b[0]["transmission_eigenchannels"])
        return result

    by_layer = {2: (coarse_two, fine_two), 3: (coarse_three, fine_three)}
    fixed = {layer: _fixed_within_layer(a[0], b[0], audit) for layer, (a, b) in by_layer.items()}
    central_increments = {layer: edge(a, b) for layer, (a, b) in by_layer.items()}
    layer_increments = {COARSE_TARGET_NM: edge(coarse_two, coarse_three), FINE_TARGET_NM: edge(fine_two, fine_three)}
    interaction = {key: layer_increments[FINE_TARGET_NM]["signed_change_percentage_points"][key]
                        - layer_increments[COARSE_TARGET_NM]["signed_change_percentage_points"][key] for key in POWERS.values()}
    require(all(np.isclose(value, central_increments[3]["signed_change_percentage_points"][key]
                           - central_increments[2]["signed_change_percentage_points"][key], rtol=0, atol=1e-12)
                for key, value in interaction.items()), "Central and thickness increments do not close the same interaction.")
    return {"coarser_footprint": {"central_max_edge_nm": COARSE_TARGET_NM, "fixed_geometry": coarse_geometry,
                                  "levels": {2: coarse_two[0], 3: coarse_three[0]}},
            "finer_footprint_central_max_edge_nm": FINE_TARGET_NM,
            "transmission_eigenchannel_method": {
                "native_propagating_channels_per_direction": CHANNEL_COUNT,
                "native_power_reconciliation_absolute_tolerance": NATIVE_POWER_ROUNDOFF,
                "description": "Squared singular values of the native transmission matrix give the eigenvalues of "
                               "the transmitted-power operator T^H T at the original planes. Each eigenvalue is the "
                               "transmitted fraction for a corresponding unit-power coherent input. They are ranked "
                               "from largest to smallest without native mode labels. The trace is the sum for five "
                               "separate 1 W drives. Trace divided by five gives the transmission fraction for equal "
                               "mutually incoherent input power.",
                "unitary_bounds": "The maximum change in ranked eigenvalues lower-bounds the operator norm of the "
                                  "transmitted-power difference under any unitary input basis change. The maximum "
                                  "change in ranked singular values gives the corresponding bound for the amplitude "
                                  "matrix under arbitrary unitary input and output basis changes. A nonzero spectral "
                                  "change rules out unitary port mixing alone for that finite comparison.",
            },
            "within_layer_fixed_geometry": fixed, "within_layer_geometry_verifier": geometry_pin,
            "finer_minus_coarser_central_increments": central_increments,
            "three_minus_two_layer_increments": layer_increments,
            "finer_minus_coarser_layer_increment_percentage_points": interaction,
            "scope": "The four finite corners retain the original physical planes and a common signed mode anchor. "
                     "The 168.75 nm two-layer case reuses Windows preparation with a Linux solve. The other three "
                     "cases use native Linux preparation and solving. The interaction measures central and layer "
                     "sensitivity on these footprints, including compatible port resolution and the disclosed "
                     "preparation differences. Continuum and frequency convergence remain unresolved."}


def replay(*, parent_only=False):
    central = native._read(CENTRAL)
    published = native._read(LAYERS)
    earlier = native._read(native._verified(published["provenance"]["earlier_thin_three_layer_trend"], ROOT))
    geometry_pin = earlier["provenance"]["geometry"]
    geometry = native._read(native._verified(geometry_pin, ROOT))
    prior = native._read(native._verified(central["provenance"]["prior_nested_comparison"], ROOT))
    reader_pin = prior["provenance"]["acceptance_reader"]
    reader = runpy.run_path(str(native._verified(reader_pin, ROOT)))["read_point"]
    flux_sources = prior["provenance"]["independent_flux_sources"]
    for pin in flux_sources:
        native._verified(pin, ROOT)
    flux_hashes = [link["sha256"] for link in flux_sources]
    require(geometry["status"] == "passed"
            and geometry["frequency_hz"] == central["frequency_hz"] == FREQUENCY_GHZ * 1e9
            and geometry["reference_planes_x_m"] == central["reference_planes_x_m"]
            and geometry["planar_footprint"]["exact_ordered_xy_coordinates_and_triangles_on_every_z_plane"] is True
            and geometry["through_thickness_slabs"]["exact_ordered_tetrahedra_with_slab_local_indices"] is True,
            "Expected the verified fixed footprint, slab topology and original end planes.")

    anchor_record = central["levels"]["225.0"]
    anchor, references = _read_modes(anchor_record)
    first_step = central["increments"]["225_to_168.75"]["drive_3_to_slot_3"]
    finest_step = central["increments"]["168.75_to_126.5625"]["drive_3_to_slot_3"]
    input_root, reflection_root = first_step["reflection"]["reference_root_pair"]
    tx_input_root, transmission_root = first_step["transmission"]["reference_root_pair"]
    require(anchor_record["native_drives"][2]["native_root_id"] == input_root == tx_input_root,
            "The published central anchor differs from the incident drive in the saved 225 nm response.")
    anchor_pair = {"incoming": input_root, "reflection": reflection_root, "transmission": transmission_root}
    central_delta = 100 * finest_step["transmission"]["signed_drive_total_power_change"]
    samples, data = {}, {}
    for layer in QUADRATIC_LAYERS:
        item, data[layer], drives = _read_case(layer, published["maps"][f"thin_z{layer}"], geometry, prior["parent_mesh"],
                                               references, reader, flux_hashes)
        samples[layer] = _sample(item, data[layer], drives, anchor, anchor_pair)
    increments = {f"{a}_to_{b}": _increment(samples[a], samples[b], data[a], data[b], central_delta)
                  for a, b in zip(QUADRATIC_LAYERS, QUADRATIC_LAYERS[1:])}
    fine = {}
    if not parent_only:
        fine_pair = _footprint_pair(central, reader, anchor, anchor_pair, references, flux_hashes,
                                    target=FINE_TARGET_NM, directory=FINE, admission=FINE_ADMISSION)
        coarse_pair = _footprint_pair(central, reader, anchor, anchor_pair, references, flux_hashes,
                                      target=COARSE_TARGET_NM, directory=COARSE, admission=COARSE_ADMISSION)
        fine["fine_footprint"] = _fine_comparison(*fine_pair, central_delta, increments["2_to_3"])
        fine["central_layer_control"] = _factorial_comparison(coarse_pair, fine_pair, central)
    return {**native.AUTHOR, "schema_version": 1, "status": "compared_saved_thin_device_layers",
            "frequency_hz": geometry["frequency_hz"], "physical_thickness_nm": 100,
            "incident_side": "left", "reference_planes_x_m": geometry["reference_planes_x_m"],
            "geometry_targets_nm": geometry["geometry_targets_nm"], "fixed_planar_footprint": geometry["planar_footprint"],
            "original_numerical_limits": central["original_numerical_limits"],
            "minimum_signed_mass_overlap_required": scattering.MIN_OVERLAP,
            "minimum_overlap_competitor_margin": scattering.K_GAP_TOL,
            "total_drive_power_change_limit": scattering.LIMIT, "complex_amplitude_change_limit": scattering.LIMIT,
            "central_anchor": {"central_max_edge_nm": 225, "native_drive": 3, "signed_roots": anchor_pair,
                               "finest_central_step_nm": [168.75, 126.5625],
                               "finest_central_T_increment_percentage_points": central_delta},
            "levels": samples, "increments": increments, **fine,
            "provenance": {"comparison_source": native._link(Path(__file__).resolve(), ROOT),
                           "matching_source": native._link(Path(scattering.__file__).resolve(), ROOT),
                           "central_anchor_comparison": native._link(CENTRAL, ROOT),
                           "published_layer_maps": native._link(LAYERS, ROOT),
                           "geometry_receipt": geometry_pin, "acceptance_reader": reader_pin,
                           "independent_flux_sources": flux_sources},
            "scope": "Signed lab displacement modes are matched by mass-weighted overlap on the richer section. "
                     "Each end uses the same native section with its verified body trace. T and R sum the original "
                     "propagating modal powers for the matched left input. The saved independent flux receipts check "
                     "all five inputs and their coherent superpositions. Complex changes align mode gauges at the "
                     "original end planes. Body layers and compatible port discretization change together on each "
                     "fixed footprint. Finite comparisons at 1 GHz leave continuum and frequency convergence unresolved."}


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, help=f"Write JSON to a file, for example {OUTPUT.relative_to(ROOT)}.")
    parser.add_argument("--parent-only", action="store_true", help="Replay only the three cases on the 400 nm footprint.")
    args = parser.parse_args(argv)
    try:
        if args.output:
            output = args.output.resolve()
            require(output == OUTPUT or not output.exists(), "Choose a new JSON file or the layer response record.")
            require(not output.is_relative_to(STUDY), "Place the comparison outside the original saved study.")
            require(output != OUTPUT or not args.parent_only, "Use a separate result file for the parent-only replay.")
        result = replay(parent_only=args.parent_only)
        if args.output:
            native.workflow.write_json(output, result)
        else:
            print(json.dumps(result, indent=2, allow_nan=False))
    except (KeyError, OSError, TypeError, ValueError) as exc:
        parser.exit(1, f"{parser.prog}: {exc}\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
