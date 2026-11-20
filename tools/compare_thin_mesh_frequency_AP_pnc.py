"""Replay nine fixed-facet thin-device powers. Author: Athul Prem | _AP_pnc."""

from __future__ import annotations

import argparse
from dataclasses import dataclass
from math import fsum
from pathlib import Path
import runpy
import sys

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if __name__ == "__main__":
    sys.path.insert(0, str(ROOT))

from tools import compare_paired_mode_frequency_AP_pnc as paired
from tools import compare_paired_nested_midpoint_AP_pnc as midpoint
from tools import compare_taper_patch_response_AP_pnc as taper
from tools import thin_frequency_channel_change_AP_pnc as temporal
from tools.compare_scattering_AP_pnc import LIMIT, MIN_OVERLAP, require


native = paired.native
STUDY = midpoint.STUDY
OUTPUT = ROOT / "verification/thin_mesh_frequency_AP_pnc.json"
FREQUENCIES = paired.FREQUENCIES_HZ
TARGETS = (225.0, 168.75, 126.5625)
AXIS = range(3)
ANCHOR = (0, 2)
QUANTITIES = midpoint.QUANTITIES
FREQUENCY_TAGS = ("995MHz", "997p5MHz", "1GHz")
MAPS = (
    ("thin100_central400_edge225_rim32", "thin100_edge225_997p5_linux", "thin100_central400_edge225_rim32"),
    ("thin100_central400_edge168p75_rim32_linux", "thin100_central400_edge168p75_rim32_997p5MHz_linux",
     "thin100_central400_edge168p75_rim32_linux"),
    ("thin100_edge126p5625_995_997p5_linux", "thin100_edge126p5625_995_997p5_linux",
     "thin100_central400_edge126p5625_rim32_linux"),
)
ARCHIVE_KEYS = ("map", "checkpoint", "point_summary", native.FIELDS, "independent_flux_replay")
SECTION_KEYS = ("points_m", "triangles", *taper.SECTION_MATRICES)


@dataclass(frozen=True)
class Point:
    record: dict
    modes: dict
    blocks: dict
    modal_powers: dict
    evidence: dict


def _fixed_maps(published, old_midpoint, old_fine):
    levels = {row["central_max_edge_nm"]: row for row in published["levels"]}
    return {
        (m, f): levels[TARGETS[m]]["samples"][FREQUENCY_TAGS[f]]["thin"]["archive"]
        for m in (0, 1) for f in (0, 2)
    } | {
        (1, 1): old_midpoint["nested_rim32"]["by_thickness"]["thin"]["acceptance_at_samples"][1]["native_archive"],
        (2, 2): old_fine["levels"][str(TARGETS[2])]["archive"],
    }


def _same_provenance(first, second):
    return all(taper._same_pin(first[k], second[k]) for k in midpoint.SHARED)


def _read_preparations(study, prior):
    state_files = {}
    for names in MAPS:
        for name in dict.fromkeys(names):
            directory = (study / name).resolve()
            require(directory.parent == study, f"Unexpected archive location for {name}.")
            state, files = midpoint._shared_inputs(directory)
            require(state["inputs"]["frequencies_GHz"] == [f / 1e9 for f, n in zip(FREQUENCIES, names, strict=True)
                                                           if n == name]
                    and all(point["status"] == "accepted" for point in state["points"])
                    and state["shared"]["reference_planes_x_m"] == prior["reference_planes_x_m"],
                    f"The final accepted frequency plan or original end planes differ at {directory}.")
            state_files[name] = state, files
        reference_state, reference_files = state_files[names[2]]
        for name in dict.fromkeys(names):
            state, files = state_files[name]
            require(state["shared"]["mesh_hash"] == reference_state["shared"]["mesh_hash"]
                    and _same_provenance(reference_files, files),
                    f"The exact mesh, K, M or port-section preparation changed within a mesh level at {name}.")
    return state_files


def _read_point(directory, flux_path, frequency, target, pin, state, prior, read_point):
    record, config, pattern = read_point(directory, flux_path, frequency / 1e9)
    require(all(record["archive"][key] == value for key, value in pin.items())
            and config["mesh"] == {**prior["parent_mesh"], "central_max_edge_nm": target}
            and config["geometry"]["thickness_nm"] == record["physical_thickness_nm"] == 100
            and config["analysis"]["frequency_GHz"] == frequency / 1e9
            and config["analysis"]["incident_side"] == "left"
            and config["resources"].get("linear_solver", "direct") == "direct"
            and record["reference_planes_x_m"] == prior["reference_planes_x_m"],
            f"The saved point or fixed thin-device inputs differ at {directory}, {frequency} Hz.")
    receipt = native._read(flux_path)
    require([receipt[key] for key in ("replay_source_sha256", "regional_replay_source_sha256")]
            == [link["sha256"] for link in prior["provenance"]["independent_flux_sources"]],
            f"The independent flux producers differ at {flux_path}.")
    summary_path = native._verified(record["archive"]["point_summary"], ROOT)
    summary = native._read(summary_path)
    channels = [{"T": row["transmission"], "R": row["reflection"]} for row in record["native_drives"]]
    closure = paired.all_mode_power(summary_path.parent, summary, channels, receipt, record["archive"],
                                   prior["original_numerical_limits"]["single_drive_and_coherent_power_defect"])
    modes, blocks, modal_powers = native._point_modes(summary_path.parent, summary)
    solve = state["inputs"]["numerical_identity"]
    donor = state.get("preparation_reuse", {}).get("receipt", {})
    evidence = {"frequency_hz": frequency, "source_git_commit": state["source_git_commit"],
                "assembly_platform": donor.get("donor_numerical_identity", solve)["platform"],
                "solve_platform": solve["platform"], "preparation": state.get("preparation_reuse", {}).get("method"),
                "original_equations": {key: receipt["original_equations_saved"][key]
                                       for key in ("operator_csc_sha256", "rhs_sha256")},
                "maximum_numerical_checks": record["maximum_numerical_checks"],
                "independent_flux_checks": record["independent_flux_checks"],
                "coherent_power_defect_spectral_norm": closure["coherent_power_defect_spectral_norm"],
                "archive": {key: record["archive"][key] for key in ARCHIVE_KEYS}}
    return Point(record, modes, blocks, modal_powers, evidence), config, pattern


def _read_grid(study, prior, pins, reader):
    states = _read_preparations(study, prior)
    points, shared_levels = {}, []
    section, trace, fixed = None, None, None
    numerical = states[MAPS[0][2]][0]["inputs"]["numerical_identity"]
    for m, names in enumerate(MAPS):
        endpoint = study / names[2]
        state, _ = states[names[2]]
        mesh_hash = state["shared"]["mesh_hash"]
        summary_path = native._verified(state["shared"]["artifacts"]["mesh_summary"], endpoint)
        mesh, manifest = native.workflow.load_checkpoint(summary_path.parent, mesh_hash)
        shared_levels.append({"central_max_edge_nm": TARGETS[m], "body_dofs": manifest["ndof"], "mesh_hash": mesh_hash,
                              "shared_inputs_by_map": {name: states[name][1] for name in dict.fromkeys(names)}})
        for f, name in enumerate(names):
            directory = study / name
            flux_path = (study / f"{name}_{FREQUENCY_TAGS[f]}_flux_AP_pnc.json").resolve()
            require(flux_path.parent == study, f"Unexpected independent receipt location for {name}.")
            state, files = states[name]
            identity = state["inputs"]["numerical_identity"]
            require(all(numerical[key] == identity[key] for key in ("source_sha256", "packages")),
                    f"The numerical sources or packages differ at {name}.")
            point, config, pattern = _read_point(directory, flux_path, FREQUENCIES[f], TARGETS[m],
                                                pins.get((m, f), {}), state, prior, reader)
            physical = (config["geometry"], config["material"], pattern)
            if fixed is None:
                fixed = physical
            require(physical == fixed and point.record["body_dofs"] == manifest["ndof"]
                    and point.record["archive"]["mesh"] == files["mesh"],
                    f"The physical geometry, material or checked mesh differs at {name}.")
            field_path = native._verified(point.record["archive"][native.FIELDS], ROOT)
            current, port_trace = taper._read_section_and_trace(
                ROOT / files["port_section"]["path"], field_path, mesh, prior["reference_planes_x_m"],
                point.record["port_dofs"])
            if section is None:
                section, trace = current, port_trace
            require(all(np.array_equal(section[key], current[key]) for key in SECTION_KEYS)
                    and all(np.array_equal(a, b) for side in trace
                            for a, b in zip(trace[side], port_trace[side], strict=True)),
                    f"Native section geometry, operators or ordered physical traces differ at {name}.")
            points[m, f] = point
    require(len({p.evidence["original_equations"]["operator_csc_sha256"] for p in points.values()}) == len(points),
            "The nine native volume equations must have distinct operator identities.")
    require(all(a["body_dofs"] < b["body_dofs"] for a, b in zip(shared_levels, shared_levels[1:])),
            "Every finer central refinement must add native body DOFs.")
    return points, shared_levels, section, numerical


def _edges():
    return (
        *(((m, f), (m, f + 1)) for m in AXIS for f in range(2)),
        *(((m, f), (m + 1, f)) for m in range(2) for f in AXIS),
    )


def _compose(order, alignment):
    return [alignment[i] for i in order]


def _channel_orders(matches):
    permutation = list(range(paired.CHANNELS))
    for edge, groups in matches.items():
        require(all(sorted(groups[group]["candidate_order"]) == permutation for group in native.GROUPS),
                f"A signed mode match on edge {edge} is incomplete.")
    for m in range(2):
        for f in range(2):
            a, b, c, d = (m, f), (m, f + 1), (m + 1, f), (m + 1, f + 1)
            for group in native.GROUPS:
                ab, bd, ac, cd = (matches[edge][group]["candidate_order"] for edge in ((a, b), (b, d), (a, c), (c, d)))
                require(_compose(ab, bd) == _compose(ac, cd),
                        f"The signed {group} correspondence around grid square {m},{f} depends on the path.")
    orders = {ANCHOR: {group: permutation for group in native.GROUPS}}
    for f in (1, 0):
        a, b = (0, f), (0, f + 1)
        orders[a] = {group: _compose(orders[b][group], np.argsort(matches[a, b][group]["candidate_order"]).tolist())
                     for group in native.GROUPS}
    for m in (1, 2):
        for f in AXIS:
            a, b = (m - 1, f), (m, f)
            orders[b] = {group: _compose(orders[a][group], matches[a, b][group]["candidate_order"])
                         for group in native.GROUPS}
    return orders


def _match_grid(points, mass):
    matches = {}
    for a, b in _edges():
        result = native.match_modes(points[a].modes, points[b].modes, mass)
        require(result["status"] == "matched", f"Unresolved signed modes on grid edge {a} to {b}: {result['groups']}")
        matches[a, b] = result["groups"]
    return matches, _channel_orders(matches)


def _spatial_amplitudes(points, matches, orders):
    maxima = {}
    for m in range(2):
        for f in AXIS:
            a, b = (m, f), (m + 1, f)
            match = matches[a, b]
            phase = {group: np.exp(1j * np.asarray(row["candidate_gauge_phase_rad"])) for group, row in match.items()}
            incoming = match["incoming"]["candidate_order"]
            columns = orders[a]["incoming"]
            maxima[m, f] = {}
            for quantity in QUANTITIES:
                rows = match[quantity]["candidate_order"]
                aligned = phase[quantity][:, None] * points[b].blocks[quantity][np.ix_(rows, incoming)] / phase["incoming"]
                # Retain the native complex-amplitude convention used by the archived spatial gates.
                maxima[m, f][quantity] = np.max(abs(aligned - points[a].blocks[quantity]), axis=0)[columns]
    return maxima


def _power_summary(samples):
    low, mid, high = samples
    return {"samples": samples,
            "endpoint_change_percentage_points": {q: 100 * (high[f"{q}_fraction"] - low[f"{q}_fraction"])
                                                  for q in QUANTITIES},
            "midpoint_minus_linear_percentage_points": {
                q: 100 * (mid[f"{q}_fraction"] - (low[f"{q}_fraction"] + high[f"{q}_fraction"]) / 2)
                for q in QUANTITIES}}


def _power_levels(points, orders, shared_levels):
    layers = []
    for m, share in enumerate(shared_levels):
        drives = []
        for i in range(paired.CHANNELS):
            samples = []
            for f, frequency in enumerate(FREQUENCIES):
                row = points[m, f].record["native_drives"][orders[m, f]["incoming"][i]]
                samples.append({"frequency_hz": frequency, "native_drive": row["drive"],
                                "native_root_id": row["native_root_id"], "wavenumber_rad_m": row["k_rad_m"],
                                **{f"{q}_fraction": row[q] for q in QUANTITIES}})
            drives.append({"reference_drive": i + 1, **_power_summary(samples)})
        mean = [{"frequency_hz": freq, **{
            f"{q}_fraction": fsum(drive["samples"][f][f"{q}_fraction"] for drive in drives) / paired.CHANNELS
            for q in QUANTITIES}} for f, freq in enumerate(FREQUENCIES)]
        layers.append({**share, "samples": [points[m, f].evidence for f in AXIS],
                       "matched_drives": drives, "equal_incoherent_mean": _power_summary(mean)})
    return layers


def _change(before, after, pointwise_gate=False):
    data = {"sample_changes_percentage_points": []}
    for a, b in zip(before["samples"], after["samples"], strict=True):
        require(a["frequency_hz"] == b["frequency_hz"], "The spatial comparison needs common frequencies.")
        delta = {q: 100 * (b[f"{q}_fraction"] - a[f"{q}_fraction"]) for q in QUANTITIES}
        data["sample_changes_percentage_points"].append({
            "frequency_hz": a["frequency_hz"], **delta,
            **({"transmission_gate_passed": abs(b["transmission_fraction"] - a["transmission_fraction"]) < LIMIT}
               if pointwise_gate else {}),
        })
    for source, dest in (("endpoint_change_percentage_points", "endpoint_change_difference_percentage_points"),
                         ("midpoint_minus_linear_percentage_points", "midpoint_departure_difference_percentage_points")):
        data[dest] = {q: after[source][q] - before[source][q] for q in QUANTITIES}
    changes = data["sample_changes_percentage_points"]
    lo = min(row["transmission"] for row in changes)
    hi = max(row["transmission"] for row in changes)
    residual_floor = (hi - lo) / 2
    data["constant_transmission_offset_test"] = {
        "best_constant_offset_percentage_points": (hi + lo) / 2,
        "minimum_possible_max_absolute_residual_percentage_points": residual_floor,
        "minimum_difference_frequencies_hz": [row["frequency_hz"] for row in changes if row["transmission"] == lo],
        "maximum_difference_frequencies_hz": [row["frequency_hz"] for row in changes if row["transmission"] == hi],
        "residual_floor_meets_spatial_transmission_limit": residual_floor < 100 * LIMIT,
    }
    return data


def _spatial_increments(layers, amplitudes):
    edges = []
    for m, (before, after) in enumerate(zip(layers, layers[1:])):
        drives = [{"reference_drive": a["reference_drive"], **_change(a, b, pointwise_gate=True)}
                  for a, b in zip(before["matched_drives"], after["matched_drives"], strict=True)]
        gates = []
        for f, frequency in enumerate(FREQUENCIES):
            rows = [drive["sample_changes_percentage_points"][f] for drive in drives]
            failed = [d["reference_drive"] for d, row in zip(drives, rows, strict=True)
                      if not row["transmission_gate_passed"]]
            for i, row in enumerate(rows):
                change = {q: float(amplitudes[m, f][q][i]) for q in QUANTITIES}
                row.update(maximum_absolute_complex_amplitude_change_by_block=change,
                           complex_amplitude_gate_passed=max(change.values()) < LIMIT,
                           total_power_gate_passed=max(abs(row[q]) for q in QUANTITIES) / 100 < LIMIT)
            maximum_complex = {q: float(max(amplitudes[m, f][q])) for q in QUANTITIES}
            maximum_power = {q: max(abs(row[q]) for row in rows) / 100 for q in QUANTITIES}
            gates.append({"frequency_hz": frequency, "failed_reference_drives": failed, "all_drives_passed": not failed,
                          "maximum_absolute_transmission_change_fraction": maximum_power["transmission"],
                          "maximum_absolute_power_change_fraction_by_block": maximum_power,
                          "maximum_absolute_total_drive_power_change_fraction": max(maximum_power.values()),
                          "all_drives_total_power_passed": all(row["total_power_gate_passed"] for row in rows),
                          "maximum_absolute_complex_amplitude_change_by_block": maximum_complex,
                          "maximum_absolute_complex_amplitude_change": max(maximum_complex.values()),
                          "all_drives_complex_amplitude_passed": all(row["complex_amplitude_gate_passed"] for row in rows)})
        edges.append({"coarser_central_max_edge_nm": before["central_max_edge_nm"],
                      "finer_central_max_edge_nm": after["central_max_edge_nm"], "per_drive": drives,
                      "equal_incoherent_mean": _change(before["equal_incoherent_mean"], after["equal_incoherent_mean"]),
                      "transmission_gates": gates})
    return edges


def _check_published_1ghz(spatial, prior):
    require(prior["frequency_hz"] == FREQUENCIES[2]
            and prior["total_drive_power_change_limit"] == prior["complex_amplitude_change_limit"] == LIMIT,
            "The archived thin-device comparison uses different frequency or spatial gates.")
    for edge in spatial:
        a, b = (edge[key] for key in ("coarser_central_max_edge_nm", "finer_central_max_edge_nm"))
        tag = f"{a:.8g}_to_{b:.8g}"
        saved = prior["increments"][tag]
        current = edge["transmission_gates"][2]
        for q in QUANTITIES:
            require(abs(current["maximum_absolute_power_change_fraction_by_block"][q]
                        - saved["blocks"][q]["total_drive_power"]["maximum_absolute_change"]) <= paired.POWER_ROUNDOFF
                    and abs(current["maximum_absolute_complex_amplitude_change_by_block"][q]
                            - saved["blocks"][q]["complex_amplitude"]["maximum_absolute_change"]) <= paired.POWER_ROUNDOFF,
                    f"The archived 1 GHz {tag} {q} power/amplitude maxima changed.")
        require(current["all_drives_total_power_passed"] == saved["total_drive_power_passed"]
                and current["all_drives_complex_amplitude_passed"] == saved["complex_amplitude_passed"],
                f"The archived 1 GHz {tag} spatial gate outcome changed.")


def replay(study=STUDY):
    study = Path(study).resolve()
    require(study.is_relative_to(ROOT), "Keep the native archives under this checkout for portable evidence links.")
    prior = native._read(paired.NESTED)
    old_midpoint, old_fine = (native._read(path) for path in (midpoint.OUTPUT, paired.THIN_NESTED))
    reader_pin = prior["provenance"]["acceptance_reader"]
    reader = runpy.run_path(str(native._verified(reader_pin, ROOT)))["read_point"]
    points, shared, section, numerical = _read_grid(study, prior, _fixed_maps(prior, old_midpoint, old_fine), reader)
    matches, orders = _match_grid(points, section["M"])
    layers = _power_levels(points, orders, shared)
    spatial = _spatial_increments(layers, _spatial_amplitudes(points, matches, orders))
    _check_published_1ghz(spatial, old_fine)
    return {**native.AUTHOR, **native.CONVERGENCE, "schema_version": 1, "status": "compared_saved_thin_mesh_frequency",
            "frequencies_hz": list(FREQUENCIES), "central_max_edges_nm": list(TARGETS), "physical_thickness_nm": 100,
            "thickness_layers": 2, "circle_resolution": 32, "incident_side": "left",
            "reference_planes_x_m": prior["reference_planes_x_m"], "mesh_parameters_common_to_levels": prior["parent_mesh"],
            "original_numerical_limits": prior["original_numerical_limits"], "spatial_transmission_limit_fraction": LIMIT,
            "spatial_total_power_limit_fraction": LIMIT, "spatial_complex_amplitude_limit": LIMIT,
            "minimum_required_signed_mass_overlap": MIN_OVERLAP,
            "mesh_levels": layers, "spatial_increments": spatial,
            "published_1ghz_spatial_gates_reproduced": True,
            "frequency_local_channel_changes": temporal.frequency_local_block_changes(
                points, matches, orders, section_mass=section["M"]),
            "channel_correspondence": {
                "reference_central_max_edge_nm": TARGETS[ANCHOR[0]], "reference_frequency_hz": FREQUENCIES[ANCHOR[1]],
                "reference_native_incident_roots": points[ANCHOR].modes["incoming"].roots.tolist(),
                "commuting_adjacent_grid_squares": 4,
                "signed_matches": [
                    {"reference_point": [TARGETS[a[0]], FREQUENCIES[a[1]]],
                     "candidate_point": [TARGETS[b[0]], FREQUENCIES[b[1]]], "groups": groups}
                    for (a, b), groups in matches.items()]},
            "scope": "The endpoint change subtracts 995 MHz from 1000 MHz. The midpoint departure subtracts the "
                     "average of those endpoints from 997.5 MHz. Spatial changes subtract the coarser mesh from the "
                     "finer mesh. The transmission offset residual is the exact three-sample lower bound on the "
                     "maximum absolute residual from any frequency-independent additive T offset. The five-input mean "
                     "uses equal incoherent powers. These are finite samples on one "
                     "fixed-facet geometry. Behavior between samples, continuum accuracy and linewidth or Q remain unresolved.",
            "provenance": {"prior_reports": [native._link(path, ROOT) for path in (paired.NESTED, midpoint.OUTPUT, paired.THIN_NESTED)],
                           "acceptance_reader": reader_pin, "independent_flux_sources": prior["provenance"]["independent_flux_sources"],
                           "packages": numerical["packages"], "numerical_source_sha256": numerical["source_sha256"],
                           "comparison_sources": [native._link(Path(path).resolve(), ROOT) for path in (
                               __file__, paired.__file__, midpoint.__file__, taper.__file__, temporal.__file__, native.__file__)]}}


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--study-root", type=Path, default=STUDY)
    parser.add_argument("--output", type=Path, default=OUTPUT)
    args = parser.parse_args(argv)
    output = args.output.resolve()
    try:
        require(not output.is_relative_to(args.study_root.resolve()) and (output == OUTPUT or not output.exists()),
                "Write to the canonical report or a new file outside the native archives.")
        native.workflow.write_json(output, replay(args.study_root))
    except (ArithmeticError, OSError, RuntimeError, TypeError, ValueError, KeyError) as exc:
        parser.exit(1, f"Thin mesh-frequency comparison failed. {exc}\n")
    print(output)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
