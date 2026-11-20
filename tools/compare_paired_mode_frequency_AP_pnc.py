#!/usr/bin/env python3
"""Replay finite per-mode and all-mode thickness responses. Author: Athul Prem | _AP_pnc."""

from __future__ import annotations

import argparse
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
from tools import map_scattering_AP_pnc as native
from tools import verify_port_count_interval_AP_pnc as section_replay
from tools.compare_scattering_AP_pnc import LIMIT, Section, require

VERIFY = ROOT / "verification"
FREQUENCY = VERIFY / "axial_device_frequency_pair_AP_pnc.json"
NESTED = VERIFY / "axial_device_nested_central_AP_pnc.json"
THIN_NESTED = VERIFY / "thin_device_nested_response_AP_pnc.json"
LAYER_PAIR = VERIFY / "axial_device_fourth_central_AP_pnc.json"
THIN_LAYER = VERIFY / "thin_device_layer_response_AP_pnc.json"
THICKNESS = {"thin": 100, "thick": 200}
FREQUENCIES_HZ = (995_000_000, 997_500_000, 1_000_000_000)
CHANNELS = 5
POWER_ROUNDOFF = 1e-12
TARGETS_NM = (225, 168.75)
TARGET_DRIVE = 3
MIDPOINT_NM = 150
SECTION_ROOT = ROOT / "results/runs/physical_thickness_section_AP_pnc"
SECTION_RECEIPT = SECTION_ROOT / "mid150_section_AP_pnc.json"
SECTION_TAGS = ("995MHz", "997p5MHz", "1000MHz")


def all_mode_power(directory, summary, channels, receipt, archive, power_limit):
    modes, blocks, _ = native._point_modes(directory, summary)
    coherent = receipt["coherent_left_incidence"]
    for group, roots in (("incoming", "incident_roots"), ("transmission", "transmitted_roots"),
                         ("reflection", "reflected_roots")):
        native_roots = modes[group].roots.tolist()
        require(len(native_roots) == len(set(native_roots)) == CHANNELS and native_roots == coherent[roots],
                f"The {group} block needs all five native propagating modes in {directory}.")
    grams = {name: block.conj().T @ block for name, block in blocks.items()}
    diagonals = {name: gram.diagonal().real for name, gram in grams.items()}
    for name, column_key in (("transmission", "T"), ("reflection", "R")):
        require(np.allclose(diagonals[name], [row[column_key] for row in channels], rtol=0, atol=POWER_ROUNDOFF),
                f"The {name} block and reported drive powers differ in {directory}.")
    traces = {name: math.fsum(values) for name, values in diagonals.items()}
    mean_t, mean_r = (traces[name] / CHANNELS for name in ("transmission", "reflection"))
    defect = grams["reflection"] + grams["transmission"] - np.eye(CHANNELS)
    defect_norm = float(np.max(abs(np.linalg.eigvalsh(defect))))
    saved_defect = np.array(coherent["defect_matrix_real"]) + 1j * np.array(coherent["defect_matrix_imag"])
    require(saved_defect.shape == defect.shape, f"Invalid saved coherent closure matrix in {directory}.")
    defect_replay_error = float(np.max(abs(defect - saved_defect)))
    require(defect_norm < power_limit and defect_replay_error < POWER_ROUNDOFF
            and math.isclose(defect_norm, coherent["coherent_coefficient_power_defect_spectral_norm"],
                             rel_tol=0, abs_tol=POWER_ROUNDOFF),
            f"The complete block fails coherent closure or disagrees with the archived replay in {directory}.")

    # The archived direct values come from displacement and nodal-force work, with no new solve.
    direct = receipt["single_drive_powers_in_incident_root_order"]
    direct_mean_differences = {name: traces[name] / CHANNELS - math.fsum(direct[f"direct_{name}"]) / CHANNELS
                               for name in blocks}
    superposition = coherent["direct_superposition"]
    weights = (np.array(superposition["incident_coefficients_real"])
               + 1j * np.array(superposition["incident_coefficients_imag"]))
    require(weights.shape == (CHANNELS,) and abs(np.vdot(weights, weights).real - 1) < POWER_ROUNDOFF,
            f"Invalid archived coherent incident combination in {directory}.")
    coherent_differences = {}
    for name, gram in grams.items():
        power = float(np.vdot(weights, gram @ weights).real)
        require(math.isclose(power, superposition[f"modal_{name}"], rel_tol=0, abs_tol=POWER_ROUNDOFF),
                f"The {name} block disagrees with the archived coherent modal power in {directory}.")
        coherent_differences[name] = power - superposition[f"direct_{name}"]
    require(all(abs(value) < power_limit for value in (*direct_mean_differences.values(),
                                                        *coherent_differences.values())),
            f"The all-mode and coherent powers disagree with the archived direct flux in {directory}.")
    return {
        "transmission_trace": traces["transmission"], "reflection_trace": traces["reflection"],
        "mean_transmission_fraction": mean_t, "mean_reflection_fraction": mean_r,
        "mean_power_closure_defect": mean_t + mean_r - 1,
        "coherent_power_defect_spectral_norm": defect_norm,
        "maximum_coherent_defect_replay_difference": defect_replay_error,
        "modal_minus_direct_flux_mean_fractions": direct_mean_differences,
        "modal_minus_direct_flux_coherent_sample_fractions": coherent_differences,
        "native_archive": {key: archive[key] for key in ("map", native.FIELDS, "independent_flux_replay")},
    }


def read_case(pins, frequency_hz, thickness, mesh, limits, provenance, read_point):
    directory = native._verified(pins["map"], ROOT).parent
    flux_path = native._verified(pins["independent_flux_replay"], ROOT)
    accepted, config, holes = read_point(directory, flux_path, frequency_hz / 1e9)
    require(all(accepted["archive"][key] == pin for key, pin in pins.items())
            and accepted["physical_thickness_nm"] == thickness and config["mesh"] == mesh
            and accepted["reference_planes_x_m"] == provenance["reference_planes_x_m"]
            and holes["source_sha256"] == provenance["hole_source_sha256"],
            f"Native archive or geometry differs from the accepted input: {directory}.")
    receipt = native._read(flux_path)
    require([receipt[key] for key in ("replay_source_sha256", "regional_replay_source_sha256")]
            == provenance["flux_producer_sha256"], f"Unexpected independent flux producers in {flux_path}.")
    numerical = accepted["maximum_numerical_checks"]
    residual_limit = limits["original_equation_and_nodal_force_relative_residuals"]
    require(all(0 <= numerical["maximum_" + key] < residual_limit
                for key in ("relative_residual", "relative_backward_residual")),
            f"Original residual exceeds the saved limit in {directory}.")
    record, point = axial._native_point(directory, frequency_hz / 1e9)
    case = axial._read_case(directory, record, point)
    for drive, native_drive in zip(accepted["native_drives"], case.channels, strict=True):
        require(drive["native_root_id"] == native_drive["root"]
                and drive["k_rad_m"] == native_drive["k_rad_m"]
                and all(math.isclose(drive[key], native_drive[power], rel_tol=0, abs_tol=POWER_ROUNDOFF)
                        for key, power in (("transmission", "T"), ("reflection", "R"))),
                f"Native port powers differ from the independent flux evidence: {directory}.")
    _, _, checkpoint = point
    solve = checkpoint["inputs"]["numerical_identity"]
    transfer = checkpoint.get("preparation_reuse", {}).get("receipt", {})
    sources = ("map", "checkpoint", "point_summary", "independent_flux_replay", "port_section", native.FIELDS)
    archive = {
        "native_archive": {key: accepted["archive"][key] for key in sources},
        "assembly_platform": transfer.get("donor_numerical_identity", solve)["platform"],
        "solve_platform": solve["platform"],
        "source_git_commit": checkpoint["source_git_commit"],
        "maximum_original_equation_residual": numerical["maximum_relative_residual"],
        "independent_flux_checks": accepted["independent_flux_checks"],
    }
    summary_path = native._verified(accepted["archive"]["point_summary"], ROOT)
    trace = all_mode_power(summary_path.parent, native._read(summary_path), case.channels, receipt,
                           accepted["archive"], limits["single_drive_and_coherent_power_defect"])
    return case, archive, trace


def matches(reference, candidate):
    require(reference.settings == candidate.settings, "The lead material or physical reference settings differ.")
    forward = axial._overlap(reference, candidate)
    reverse = axial._overlap(candidate, reference)
    rows = []
    for i, j in enumerate(forward.argmax(axis=1)):
        overlap_a, other_a = axial._match_evidence(forward, i, j)
        overlap_b, other_b = axial._match_evidence(reverse, j, i)
        minimum = min(overlap_a, overlap_b)
        if minimum < axial.MIN_OVERLAP:
            status = "unresolved_below_overlap_threshold"
        elif overlap_a <= other_a + axial.K_GAP_TOL or overlap_b <= other_b + axial.K_GAP_TOL:
            status = "unresolved_competing_modes"
        else:
            status = "matched"
        rows.append({"reference_drive": reference.channels[i]["drive"],
                     "best_candidate_drive": candidate.channels[j]["drive"], "status": status,
                     "minimum_reciprocal_mass_overlap": minimum, "maximum_competing_mass_overlap": max(other_a, other_b)})
    return rows


def power_change(reference, candidate):
    return {key: 100 * (candidate[key] - reference[key]) for key in ("T", "R")}


def cross_thickness_powers(thin, thick):
    cross = matches(thin, thick)
    for match in cross:
        if match["status"] == "matched":
            a, b = match["reference_drive"] - 1, match["best_candidate_drive"] - 1
            match["thin_minus_thick_percentage_points"] = power_change(thick.channels[b], thin.channels[a])
    return cross


def frequency_sample(frequency, cases, anchors, archives):
    sample = {"frequency_hz": frequency, "native_drives": {role: case.channels for role, case in cases.items()},
              "cross_thickness_matches_from_thin": cross_thickness_powers(cases["thin"], cases["thick"]),
              "acceptance": archives}
    if frequency != FREQUENCIES_HZ[-1]:
        sample["within_thickness_matches_from_1000MHz"] = {role: matches(anchors[role], point)
                                                          for role, point in cases.items()}
    return sample


def spatial_sample(frequency, coarse, fine, rim48):
    levels, adjacent, contrast = [], {}, []
    for target, (points, archives, _) in zip(TARGETS_NM, (coarse, fine), strict=True):
        levels.append({"central_max_edge_nm": target, "native_drives": {role: p.channels for role, p in points.items()},
                       "matches_from_rim48": {role: matches(rim48[role], point) for role, point in points.items()},
                       "cross_thickness_matches_from_thin": cross_thickness_powers(points["thin"], points["thick"]),
                       "acceptance": archives})
    for role in THICKNESS:
        a, b = coarse[0][role], fine[0][role]
        rows = matches(a, b)
        for match in rows:
            if match["status"] == "matched":
                before = a.channels[match["reference_drive"] - 1]
                after = b.channels[match["best_candidate_drive"] - 1]
                match.update(finer_minus_coarser_percentage_points=power_change(before, after),
                             total_power_increment_passed=max(abs(after[key] - before[key]) for key in ("T", "R")) < LIMIT)
        adjacent[role] = rows
    coarse_cross, fine_cross = (level["cross_thickness_matches_from_thin"] for level in levels)
    for before in coarse_cross:
        thin_step = adjacent["thin"][before["reference_drive"] - 1]
        thick_step = adjacent["thick"][before["best_candidate_drive"] - 1]
        after = fine_cross[thin_step["best_candidate_drive"] - 1]
        if all(row["status"] == "matched" for row in (before, after, thin_step, thick_step)):
            require(after["best_candidate_drive"] == thick_step["best_candidate_drive"],
                    f"The nested thickness matches disagree with the within-thickness matches at {frequency} Hz.")
            a, b = (row["thin_minus_thick_percentage_points"] for row in (before, after))
            contrast.append({"thin_drive_on_coarser_mesh": before["reference_drive"],
                             "finer_minus_coarser_contrast_percentage_points": {key: b[key] - a[key] for key in ("T", "R")}})
    return {"frequency_hz": frequency, "levels": levels, "nested_increment": adjacent,
            "matched_thickness_contrast_increment": contrast}


def read_midpoint_section(baseline):
    receipt = native._read(SECTION_RECEIPT)
    cases, archives, _ = baseline[FREQUENCIES_HZ[-1]]
    thin, thick = cases["thin"], cases["thick"]
    width = thick.settings["width_nm"]
    require(thin.settings == thick.settings and thick.settings["incident_side"] == "left"
            and receipt["material"] == thick.settings["material"]
            and receipt["lead_axis_lab"] == "+x" and receipt["lateral_boundary"] == "traction_free"
            and receipt["wave_convention"] == "exp(i*k*x - i*omega*t)"
            and receipt["frequencies_hz"] == list(FREQUENCIES_HZ)
            and tuple(receipt[key] for key in ("physical_width_nm", "physical_thickness_nm", "element_order",
                                               "width_intervals", "thickness_layers", "port_dofs", "section_triangles"))
            == (width, MIDPOINT_NM, 2, 4, 2, 135, 16),
            "The uniform midpoint section has different physical or discretization settings.")
    for role, thickness in THICKNESS.items():
        saved = receipt["native_endpoint_sections"][f"{thickness}_nm"]
        endpoint = archives[role]["native_archive"]["port_section"]
        require(saved["section"] == endpoint and all(
            saved["frequency_checkpoints"][tag] == baseline[f][1][role]["native_archive"]["checkpoint"]
            for f, tag in zip(FREQUENCIES_HZ, SECTION_TAGS, strict=True)),
            "The midpoint receipt uses different native endpoint sections or checkpoints.")
    native._verified(receipt["driver"], ROOT)
    require(native._verified(receipt["independent_replay"], ROOT) == Path(section_replay.__file__).resolve(),
            "The midpoint receipt uses a different independent replay source.")

    section_path = native._verified(receipt["section_arrays"], SECTION_ROOT)
    require(section_path == SECTION_ROOT / "mid150_section_AP_pnc.npz", "Unexpected midpoint section array path.")
    with np.load(section_path, allow_pickle=False) as stored:
        require(str(stored["inputs_sha256"]) == receipt["inputs_sha256"], "Unbound midpoint section arrays.")
        points, triangles = stored["points_m"], stored["triangles"]
        operators = {name: stored[name] for name in ("K0", "M", "Q", "G")}
        width_m, thickness_m, speed = (float(stored[key]) for key in ("width_m", "thickness_m", "reference_speed_m_s"))
    require(math.isclose(width_m, width * 1e-9, rel_tol=1e-14)
            and math.isclose(thickness_m, MIDPOINT_NM * 1e-9, rel_tol=1e-14)
            and points.shape == (2, 15) and np.isrealobj(points) and np.isfinite(points).all()
            and triangles.shape == (3, 16) and triangles.dtype.kind in "iu"
            and np.allclose(np.ptp(points, axis=1), [width_m, thickness_m], rtol=1e-14, atol=0),
            "Invalid physical width, thickness or midpoint mesh dimensions.")
    # The P2 scalar and vector DOF layouts are fixed by the identical ordered mesh.
    normalized = points / np.array([[1], [thickness_m]])
    for case in cases.values():
        mesh = case.section.basis.mesh
        require(np.array_equal(triangles, mesh.t) and np.array_equal(normalized[0], mesh.p[0])
                and np.allclose(normalized[1], mesh.p[1], rtol=0, atol=1e-14)
                and np.array_equal(case.section.trace_to_scalar, thick.section.trace_to_scalar),
                "The midpoint and endpoint sections differ in y, normalized z, topology or DOF layout.")
    component = thick.section.trace_to_scalar[0]
    mass = operators["M"]
    require(mass.shape == (135, 135) and np.isrealobj(mass) and np.isfinite(mass).all(),
            "Invalid midpoint mass matrix.")
    section = Section(thick.section.basis, thick.section.trace_to_scalar,
                      mass[np.ix_(component, component)], receipt["section_arrays"]["sha256"])
    return section, operators, width_m, speed, receipt["section_arrays"]


def read_midpoint_modes(frequency, tag, shared, settings):
    section, operators, width_m, speed, section_arrays = shared
    receipt_path = SECTION_ROOT / f"mid150_{tag}_AP_pnc.json"
    receipt = native._read(receipt_path)
    require(receipt["status"] == "accepted" and receipt["frequency_hz"] == frequency
            and receipt["cutoff_status"] == "solver_resolved_at_sample"
            and receipt["section_arrays"] == section_arrays
            and native._verified(receipt["section_metadata"], SECTION_ROOT) == SECTION_RECEIPT,
            "The accepted midpoint mode receipt does not match the section or frequency.")
    modes_path = native._verified(receipt["mode_arrays"], SECTION_ROOT)
    require(modes_path == SECTION_ROOT / f"mid150_{tag}_modes_AP_pnc.npz", "Unexpected midpoint mode array path.")
    with np.load(modes_path, allow_pickle=False) as stored:
        require(str(stored["section_sha256"]) == section_arrays["sha256"], "Unbound midpoint mode arrays.")
        modes = {key: stored[key] for key in section_replay.MODE_FIELDS}
    metrics = section_replay.replay_modes(operators, modes, frequency, length_m=width_m, reference_speed_m_s=speed)
    section_replay.compare_metrics(receipt["metrics"], metrics)
    positive = [row for row in metrics["open_channels"] if row["direction"] == "positive"]
    roots = [row["root_id"] for row in positive]
    channels = [{"root": row["root_id"], "k_rad_m": row["k_real_rad_m"], "signed_power_w": row["signed_power_w"]}
                for row in positive]
    require(all(row["signed_power_w"] > 0 for row in channels), "A midpoint incident competitor has invalid signed power.")
    displacement = modes["displacements"][:, roots][section.trace_to_scalar]
    record = {"receipt": native._link(receipt_path, ROOT), "modes": native._link(modes_path, ROOT),
              "positive_open_channels": metrics["positive_open_channels"],
              "negative_open_channels": metrics["negative_open_channels"],
              "maximum_eigen_backward_residual": metrics["maximum_eigen_backward_residual"],
              "maximum_relative_traction_error": metrics["maximum_relative_traction_error"],
              "maximum_directional_flux_gram_error_w": max(metrics["positive_flux_gram_error_w"],
                                                            metrics["negative_flux_gram_error_w"])}
    return axial.SavedPoint(settings, record, channels, section, displacement)


def relative_wavenumber_gaps(channels):
    k = np.array([row["k_rad_m"] for row in channels])
    require(len(k) >= 2 and np.isfinite(k).all() and max(abs(k)) > 0,
            "The section needs at least two finite incident wavenumbers for a separation check.")
    gaps = abs(k[:, None] - k[None, :]) / max(abs(k))
    np.fill_diagonal(gaps, np.inf)
    return gaps.min(axis=1)


def adjacent_midpoint_matches(reference, candidate):
    forward, reverse = axial._overlap(reference, candidate), axial._overlap(candidate, reference)
    reference_gaps = relative_wavenumber_gaps(reference.channels)
    candidate_gaps = relative_wavenumber_gaps(candidate.channels)
    rows = []
    for i, j in enumerate(forward.argmax(axis=1)):
        a, other_a = axial._match_evidence(forward, i, j)
        b, other_b = axial._match_evidence(reverse, j, i)
        gap_a, gap_b = float(reference_gaps[i]), float(candidate_gaps[j])
        if min(gap_a, gap_b) <= axial.K_GAP_TOL:
            status = "unresolved_wavenumber_separation"
        elif min(a, b) < axial.MIN_OVERLAP:
            status = "unresolved_below_overlap_threshold"
        elif a <= other_a + axial.K_GAP_TOL or b <= other_b + axial.K_GAP_TOL:
            status = "unresolved_competing_modes"
        else:
            status = "matched"
        rows.append({"reference_channel": i + 1, "best_candidate_channel": int(j) + 1, "status": status,
                     "reference_to_candidate_mass_overlap": a, "candidate_to_reference_mass_overlap": b,
                     "reference_to_candidate_maximum_competing_overlap": other_a,
                     "candidate_to_reference_maximum_competing_overlap": other_b,
                     "reference_relative_wavenumber_gap": gap_a, "candidate_relative_wavenumber_gap": gap_b})
    all_five = (len(rows) == len(candidate.channels) == 5 and all(row["status"] == "matched" for row in rows)
                and len({row["best_candidate_channel"] for row in rows}) == 5)
    return {"all_five_one_to_one": all_five, "matches": rows}


def component_energy_fractions(point, channel):
    displacement = point.displacement[:, :, channel - 1]
    energies = np.einsum("ci,ci->c", displacement.conj(), displacement @ point.section.mass.T).real
    require(np.isfinite(energies).all() and np.all(energies >= 0) and energies.sum() > 0,
            "A focus mode has invalid section kinetic energy.")
    return dict(zip(("x", "y", "z"), (energies / energies.sum()).tolist(), strict=True))


def midpoint_sample(frequency, cases, midpoint):
    thin, thick = cases["thin"], cases["thick"]
    a, b = adjacent_midpoint_matches(thin, midpoint), adjacent_midpoint_matches(midpoint, thick)
    result = {"frequency_hz": frequency, "midpoint_acceptance": midpoint.record, "midpoint_channels": midpoint.channels,
              "100_to_150_nm": a, "150_to_200_nm": b, "sampled_correspondences": []}
    for first in a["matches"]:
        thin_drive, middle_channel = first["reference_channel"], first["best_candidate_channel"]
        second = b["matches"][middle_channel - 1]
        if first["status"] != "matched" or second["status"] != "matched":
            continue
        thick_drive = second["best_candidate_channel"]
        channels = (thin_drive, middle_channel, thick_drive)
        points = (thin, midpoint, thick)
        selected = [point.channels[channel - 1] for point, channel in zip(points, channels, strict=True)]
        track = {"section_channels_100_150_200_nm": list(channels),
                 "section_roots_100_150_200_nm": [row["root"] for row in selected],
                 "k_rad_m_100_150_200_nm": [row["k_rad_m"] for row in selected]}
        result["sampled_correspondences"].append(track)
        if thin_drive == TARGET_DRIVE and a["all_five_one_to_one"] and b["all_five_one_to_one"]:
            result["focus_thin_drive_endpoint_power_via_150nm"] = {
                **track,
                "section_kinetic_energy_fractions_xyz_100_150_200_nm": [
                    component_energy_fractions(point, channel) for point, channel in zip(points, channels, strict=True)],
                "thin_native_power_fractions": {key: selected[0][key] for key in ("T", "R")},
                "thick_native_power_fractions": {key: selected[2][key] for key in ("T", "R")},
                "thin_minus_thick_percentage_points": power_change(selected[2], selected[0]),
            }
    return result


def nested_focus_via_midpoint(midpoint, spatial):
    focus = midpoint["focus_thin_drive_endpoint_power_via_150nm"]
    thin, _, thick = focus["section_channels_100_150_200_nm"]
    rows = []
    for level in spatial["levels"]:
        links = {role: level["matches_from_rim48"][role][drive - 1]
                 for role, drive in (("thin", thin), ("thick", thick))}
        row = {"central_max_edge_nm": level["central_max_edge_nm"], "within_thickness_matches_from_rim48": links}
        indices = {role: link["best_candidate_drive"] - 1 for role, link in links.items()}
        gaps = {role: float(relative_wavenumber_gaps(level["native_drives"][role])[i]) for role, i in indices.items()}
        row["nested_relative_wavenumber_gaps"] = gaps
        if all(link["status"] == "matched" for link in links.values()) and min(gaps.values()) > axial.K_GAP_TOL:
            selected = {role: level["native_drives"][role][i] for role, i in indices.items()}
            row.update(native_drives=selected,
                       thin_minus_thick_percentage_points=power_change(selected["thick"], selected["thin"]))
        rows.append(row)
    return rows


def trace_change(reference, candidate):
    return {
        "transmission_trace": candidate["transmission_trace"] - reference["transmission_trace"],
        "mean_transmission_percentage_points": 100 * (candidate["mean_transmission_fraction"]
                                                      - reference["mean_transmission_fraction"]),
    }


def paired_trace_sample(traces):
    return {"by_thickness": traces, "thin_minus_thick": trace_change(traces["thick"], traces["thin"])}


def paired_trace_change(reference, candidate):
    return {
        "by_thickness": {role: trace_change(reference["by_thickness"][role], candidate["by_thickness"][role])
                         for role in THICKNESS},
        "thin_minus_thick": {
            key: value - reference["thin_minus_thick"][key] for key, value in candidate["thin_minus_thick"].items()},
    }


def trace_spatial_controls(baseline, controls, parent_mesh, layer_pair, thin_layer, load):
    paired_results = []
    for frequency, layer_label in zip(FREQUENCIES_HZ[::2], ("995MHz", "1000MHz"), strict=True):
        central_levels = [{"central_max_edge_nm": target, **paired_trace_sample(controls[target, frequency][2])}
                          for target in (None, 300, *TARGETS_NM)]
        layer_levels = []
        for saved in layer_pair["through_thickness_control"]["samples"]:
            layer = saved["quadratic_layers"]
            mesh = {**parent_mesh, "thickness_layers": layer}
            archives = {role: saved[role][layer_label]["archive"] for role in THICKNESS}
            results = paired_trace_sample(load(archives, frequency, mesh)[2])
            layer_levels.append({"quadratic_layers": layer, **results})
        require([level["quadratic_layers"] for level in layer_levels] == [1, 2, 3]
                and layer_levels[1]["by_thickness"] == central_levels[0]["by_thickness"],
                f"The {frequency} Hz spatial controls differ from the declared paired mesh series.")
        paired_results.append({
            "frequency_hz": frequency,
            "rim48_minus_rim32_parent": paired_trace_change(central_levels[0], paired_trace_sample(baseline[frequency][2])),
            "fixed_rim32_nested_central_levels": central_levels,
            "successive_nested_central_increments": [
                {"from_central_max_edge_nm": a["central_max_edge_nm"], "to_central_max_edge_nm": b["central_max_edge_nm"],
                 **paired_trace_change(a, b)} for a, b in zip(central_levels, central_levels[1:])],
            "rim32_parent_thickness_layers": layer_levels,
            "successive_thickness_layer_increments": [
                {"from_quadratic_layers": a["quadratic_layers"], "to_quadratic_layers": b["quadratic_layers"],
                 **paired_trace_change(a, b)} for a, b in zip(layer_levels, layer_levels[1:])],
        })

    thin_levels = []
    for saved in (thin_layer["central_layer_control"]["coarser_footprint"], thin_layer["fine_footprint"]):
        target = saved["central_max_edge_nm"]
        levels = {}
        for layer in (2, 3):
            mesh = {**parent_mesh, "central_max_edge_nm": target, "thickness_layers": layer}
            archive = saved["levels"][str(layer)]["archive"]
            levels[str(layer)] = load({"thin": archive}, FREQUENCIES_HZ[-1], mesh)[2]["thin"]
        thin_levels.append({"central_max_edge_nm": target, "by_quadratic_layers": levels,
                            "three_minus_two_layer_increment": trace_change(levels["2"], levels["3"])})
    coarse, fine = (item["by_quadratic_layers"] for item in thin_levels)
    require([item["central_max_edge_nm"] for item in thin_levels] == [168.75, 126.5625]
            and coarse["2"] == paired_results[-1]["fixed_rim32_nested_central_levels"][-1]["by_thickness"]["thin"],
            "The thin-only extension uses a different nested anchor.")
    return {
        "paired_endpoints": paired_results,
        "thin_only_1000MHz": {
            "frequency_hz": FREQUENCIES_HZ[-1],
            "levels": thin_levels,
            "finer_minus_coarser_by_quadratic_layers": {layer: trace_change(coarse[layer], fine[layer]) for layer in coarse},
        },
        "scope": "The parent meshes use a 400 nm central and 150 nm lead target. A null central edge ceiling denotes "
                 "the parent. Changing rim resolution also regenerates the central mesh. The nested 32-segment "
                 "levels preserve the hole walls, leads and port traces. Changing the layer count includes the "
                 "compatible port discretization. Spatial comparisons use only sampled shared frequencies. "
                 "The 997.5 MHz pair is available on the 48-segment mesh. The thin-only finer levels have no thick pair.",
    }


def replay():
    paired = native._read(FREQUENCY)
    rim_pin = paired["provenance"]["published_1GHz_anchor"]
    rim = native._read(native._verified(rim_pin, ROOT))
    nested = native._read(NESTED)
    layer_pair = native._read(LAYER_PAIR)
    thin_layer = native._read(THIN_LAYER)
    reader_pin = rim["provenance"]["acceptance_replay_reader"]
    read_point = runpy.run_path(str(native._verified(reader_pin, ROOT)))["read_point"]
    source_pins = rim["provenance"]["independent_flux_sources"]
    limits = paired["original_numerical_limits"]
    require(nested["original_numerical_limits"] == rim["original_numerical_limits"] == limits
            and nested["provenance"]["acceptance_reader"] == reader_pin
            and nested["provenance"]["independent_flux_sources"] == source_pins
            and all(nested[key] == rim[key] == paired[key] for key in
                    ("reference_planes_x_m", "hole_count", "incident_side")),
            "The accepted response families use different physical settings or numerical limits.")
    require(all(source["original_numerical_limits"] == limits
                and all(source[key] == paired[key] for key in ("reference_planes_x_m", "incident_side"))
                for source in (layer_pair, thin_layer)),
            "The saved layer controls use different reference planes, incidence or numerical limits.")
    provenance = {"reference_planes_x_m": paired["reference_planes_x_m"],
                  "hole_source_sha256": rim["provenance"]["hole_pattern_csv"]["sha256"],
                  "flux_producer_sha256": [pin["sha256"] for pin in source_pins]}

    baseline, controls = {}, {}

    def load(pins, frequency, mesh):
        loaded = {role: read_case(pin, frequency, THICKNESS[role], mesh, limits, provenance, read_point)
                  for role, pin in pins.items()}
        reference = next(iter(baseline.values()))[0]["thin"] if baseline else loaded["thin"][0]
        require(all(case.settings == reference.settings for case, _, _ in loaded.values()),
                "A saved response differs in material, placed holes, incidence or reference geometry.")
        return tuple({role: row[index] for role, row in loaded.items()} for index in range(3))

    parent_mesh = nested["parent_mesh"]
    rim48 = {**parent_mesh, "circle_resolution": 48}
    for label, frequency in zip(("lower", "midpoint", "upper"), FREQUENCIES_HZ, strict=True):
        require(paired[label]["frequency_hz"] == frequency, "Unexpected finite sample frequency.")
        if label == "upper":
            pins = {role: rim["maps"][role + "_rim48"]["archive"] for role in THICKNESS}
        else:
            endpoints = paired["provenance"][label + "_within_thickness_replays"]
            pins = {role: native._read(native._verified(endpoints[role], ROOT))["candidate"]["archive"]
                    for role in THICKNESS}
        baseline[frequency] = load(pins, frequency, rim48)
    require([level["central_max_edge_nm"] for level in nested["levels"]] == [None, 300, *TARGETS_NM],
            "The saved nested comparison uses a different central edge series.")
    for level in nested["levels"]:
        target = level["central_max_edge_nm"]
        require(all(level["geometry"][role]["fixed_" + key + "_match"] is True
                    for role in THICKNESS for key in ("lead_tetrahedra", "cut_facets", "rim_facets", "port_facets")),
                f"The saved central edge level {target} changes the fixed regional geometry.")
        mesh = parent_mesh if target is None else {**parent_mesh, "central_max_edge_nm": target}
        for label, frequency in zip(("995MHz", "1GHz"), FREQUENCIES_HZ[::2], strict=True):
            pins = {role: level["samples"][label][role]["archive"] for role in THICKNESS}
            controls[target, frequency] = load(pins, frequency, mesh)

    anchor = baseline[FREQUENCIES_HZ[-1]][0]
    for points, _, _ in baseline.values():
        for role, point in points.items():
            require(point.record["inputs"]["mesh_sha256"] == anchor[role].record["inputs"]["mesh_sha256"]
                    and point.section.discretization_sha256 == anchor[role].section.discretization_sha256,
                    f"The rim48 {role} frequencies use different saved mesh or section preparations.")
    samples = [frequency_sample(f, cases, anchor, archives) for f, (cases, archives, _) in baseline.items()]
    midpoint_shared = read_midpoint_section(baseline)
    midpoint_samples = [midpoint_sample(f, baseline[f][0], read_midpoint_modes(f, tag, midpoint_shared,
                                                                             anchor["thin"].settings))
                        for f, tag in zip(FREQUENCIES_HZ, SECTION_TAGS, strict=True)]
    nested_samples = [spatial_sample(frequency, controls[225, frequency], controls[168.75, frequency],
                                     baseline[frequency][0]) for frequency in FREQUENCIES_HZ[::2]]
    for midpoint, spatial in zip(midpoint_samples[::2], nested_samples, strict=True):
        if "focus_thin_drive_endpoint_power_via_150nm" in midpoint:
            midpoint["nested_focus_endpoint_powers_via_150nm_and_rim48"] = nested_focus_via_midpoint(midpoint, spatial)
    trace_samples = [{"frequency_hz": f, **paired_trace_sample(baseline[f][2])} for f in FREQUENCIES_HZ]
    return {**native.AUTHOR, "schema_version": 1, "status": "compared_saved_finite_mode_powers",
            "physical_thickness_nm": THICKNESS, "frequency_anchor_hz": FREQUENCIES_HZ[-1],
            **{key: paired[key] for key in ("reference_planes_x_m", "incident_side", "hole_count")},
            "frequency_sample_mesh": rim48, "nested_control_parent_mesh": nested["parent_mesh"],
            "matching": {"minimum_reciprocal_mass_overlap": axial.MIN_OVERLAP,
                         "minimum_competitor_margin": axial.K_GAP_TOL,
                         "method": "Reciprocal interpolation of Cartesian displacement at fixed y and z/thickness "
                                   "with each target section's native mass matrix."},
            "original_numerical_limits": limits, "total_power_increment_fraction_limit": LIMIT,
            "frequency_samples": samples,
            "all_mode_power": {
                "incident_channel_count": CHANNELS,
                "transmitted_channel_count": CHANNELS,
                "reflected_channel_count": CHANNELS,
                "archive_modal_power_roundoff_tolerance": POWER_ROUNDOFF,
                "convention": "The native complex blocks t and r map all left-incident propagating modes to "
                              "outgoing propagating modes at the original physical planes. The signed flux for "
                              "a unit-amplitude mode has magnitude 1 W. Both traces are dimensionless sums "
                              "of the power fractions for five separate unit-power incident drives. Tr(t†t) and Tr(r†r) "
                              "are invariant under unitary changes of the corresponding power-normalized "
                              "input and output bases, so cross-thickness mode matching is unnecessary. Dividing "
                              "each trace by five gives the power fraction for equal incoherent incident powers. "
                              "The lossless closure is Tr(t†t) + Tr(r†r) = 5, or 1 for the incoherent means. "
                              "Differences of the mean fractions multiplied by 100 are percentage points.",
                "rim48_frequency_samples": trace_samples,
                "rim48_995_minus_1000MHz": paired_trace_change(trace_samples[-1], trace_samples[0]),
                "spatial_controls": trace_spatial_controls(baseline, controls, parent_mesh, layer_pair, thin_layer, load),
                "scope": "Frequencies are in Hz. The three sampled frequencies and finite spatial increments "
                         "leave continuum and frequency convergence unresolved. The saved direct flux and "
                         "coherent closure checks verify conservation for each finite discretization. "
                         "Prior complex and weak-conversion sensitivity failures stand.",
            },
            "sampled_thickness_correspondence": {
                "physical_thickness_nm": [100, MIDPOINT_NM, 200],
                "focus_thin_drive": TARGET_DRIVE,
                "minimum_relative_wavenumber_gap": axial.K_GAP_TOL,
                "midpoint_section_receipt": native._link(SECTION_RECEIPT, ROOT),
                "midpoint_section_arrays": native._link(native._verified(midpoint_shared[-1], SECTION_ROOT), ROOT),
                "samples": midpoint_samples,
                "scope": "Each row follows sampled positive-flux displacement matches at one fixed frequency. "
                         "Channels and roots are local to the section and frequency. The qualified focus comparison "
                         "uses only the saved 100 and 200 nm device powers at their original planes. "
                         "Nested contrasts also require within-thickness links from the 48-segment rim pair. "
                         "The 150 nm calculation supplies uniform-section modes only. Continuous branch identity, "
                         "intermediate device powers and spatial or frequency convergence remain unresolved.",
            },
            "nested_control": {
                "focus_thin_drive_at_1000MHz": TARGET_DRIVE,
                "samples": nested_samples},
            "scope": "T and R are sums over propagating output modes, in fractions of native incident power. "
                     "Mode matches connect only the sampled frequencies and keep the original physical planes. "
                     "An unresolved cross-thickness entry identifies an overlap candidate without a thickness contrast. "
                     "The two nested ceilings keep the 32-segment rim, lead and port traces fixed and use distinct "
                     "solve platforms. The frequency samples use 48 segments per hole. The nested increments leave "
                     "continuum and frequency convergence unresolved. Prior complex and weak-conversion failures stand.",
            "provenance": {"paired_axial_frequency_samples": native._link(FREQUENCY, ROOT),
                           "rim48_anchor": rim_pin, "nested_controls_and_execution": native._link(NESTED, ROOT),
                           "thin_1GHz_finer_mesh_and_complex_comparison": native._link(THIN_NESTED, ROOT),
                           "paired_400nm_layer_control": native._link(LAYER_PAIR, ROOT),
                           "thin_1GHz_central_and_layer_control": native._link(THIN_LAYER, ROOT),
                           "acceptance_reader": reader_pin, "independent_flux_sources": source_pins,
                           "overlap_source": native._link(Path(axial.__file__).resolve(), ROOT),
                           "replay_source": native._link(Path(__file__).resolve(), ROOT)}}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, help="Write a verification JSON instead of printing it.")
    args = parser.parse_args()
    try:
        data = replay()
        payload = json.dumps(data, indent=2, allow_nan=False) + "\n"
        if args.output is None:
            print(payload, end="")
        else:
            path = args.output.resolve()
            rim = ROOT / data["provenance"]["rim48_anchor"]["path"]
            require(path not in (FREQUENCY, NESTED, THIN_NESTED, LAYER_PAIR, THIN_LAYER, rim)
                    and not path.is_relative_to(ROOT / "results/runs"),
                    "Choose an output outside the original response archive and prior comparisons.")
            staging = path.with_suffix(path.suffix + ".tmp")
            staging.write_text(payload, encoding="utf-8", newline="\n")
            staging.replace(path)
    except (KeyError, OSError, TypeError, ValueError) as exc:
        parser.exit(1, f"{parser.prog}: {exc}\n")


if __name__ == "__main__":
    main()
