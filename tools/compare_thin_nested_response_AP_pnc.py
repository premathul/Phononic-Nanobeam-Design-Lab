#!/usr/bin/env python3
"""Replay the five-channel 1 GHz thin-device central refinement. Author: Athul Prem | _AP_pnc."""

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


STUDY = ROOT / "results/runs/physical_thickness_device_AP_pnc"
PRIOR = ROOT / "verification/axial_device_nested_central_AP_pnc.json"
TARGETS = {225.: "225_rim32", 168.75: "168p75_rim32_linux", 126.5625: "126p5625_rim32_linux"}
TARGET = 126.5625
DRIVE_INDEX = 2
FREQUENCY_GHZ = 1.


def _read_case(study, target, prior, reader):
    directory = study / f"thin100_central400_edge{TARGETS[target]}"
    flux_path = directory.with_name(f"{directory.name}_1GHz_flux_AP_pnc.json")
    record, config, _ = reader(directory, flux_path, FREQUENCY_GHZ)
    run_dir, summary, fixed, identity, section = nested._channel_input(record)
    checkpoint = native.read_map_checkpoint(directory)
    require(config["mesh"] == {**prior["parent_mesh"], "central_max_edge_nm": target}
            and config["geometry"]["thickness_nm"] == 100
            and record["reference_planes_x_m"] == prior["reference_planes_x_m"]
            and len(record["native_drives"]) == summary["num_incident_channels"] == 5
            and config["resources"].get("linear_solver", "direct") == "direct"
            and summary["power_balance_passed"] is True,
            f"Unexpected mesh, physical thickness, reference planes or solver in {directory}.")
    if target != TARGET:
        saved = next(row for row in prior["levels"] if row["central_max_edge_nm"] == target)["samples"]["1GHz"]["thin"]
        require(all(record["archive"][key] == pin for key, pin in saved["archive"].items()),
                f"The previously accepted 1 GHz point differs from its publication in {directory}.")
    else:
        require(checkpoint["inputs"]["frequencies_GHz"] == [FREQUENCY_GHZ]
                and "preparation_reuse" not in checkpoint,
                "The 126.5625 nm candidate must be a native Linux assembly and solve.")
    replay = native._read(flux_path)
    expected_sources = [link["sha256"] for link in prior["provenance"]["independent_flux_sources"]]
    require([replay[key] for key in ("replay_source_sha256", "regional_replay_source_sha256")] == expected_sources,
            f"The independent flux producers differ from the accepted pilot in {directory}.")
    record["original_equation_identity"] = {key: replay["original_equations_saved"][key]
                                            for key in ("operator_csc_sha256", "rhs_sha256")}
    reuse = checkpoint.get("preparation_reuse", {}).get("receipt", {})
    record["platform"] = {"assembly": reuse.get("donor_numerical_identity", identity)["platform"],
                          "solve": identity["platform"]}
    return record, (run_dir, summary, fixed, identity, section)


def _check_geometry(receipt, parent, candidate):
    parent_checkpoint = native._verified(parent["archive"]["checkpoint"], ROOT)
    parent_state = native.read_map_checkpoint(parent_checkpoint.parent)
    candidate_checkpoint = native._verified(candidate["archive"]["checkpoint"], ROOT)
    candidate_state = native.read_map_checkpoint(candidate_checkpoint.parent)
    candidate_manifest = candidate_state["shared"]["artifacts"]["mesh_summary"]
    native._verified(receipt["geometry_verifier"], ROOT)
    geometry_config = native._read(native._verified(receipt["configuration"], ROOT))
    require(receipt["status"] == "passed_native_mesh_geometry_preflight"
            and receipt["target_nm"] == TARGET and receipt["parent_target_nm"] == 168.75
            and receipt["reference_planes_x_m"] == parent["reference_planes_x_m"] == candidate["reference_planes_x_m"]
            and receipt["planar_archived_parent_bisection_match"] is True
            and all(receipt["fixed_geometry"][key] is True
                    for key in ("lead_tetrahedra", "cut_facets", "rim_facets", "port_facets"))
            and receipt["parent_map_checkpoint"]["sha256"] == parent["archive"]["checkpoint"]["sha256"]
            and receipt["windows_transfer_receipt_sha256"] == parent_state["preparation_reuse"]["receipt_sha256"]
            and receipt["parent_windows_shared_mesh"] == parent_state["shared"]["artifacts"]["mesh"]
            and all(receipt["candidate_mesh_preflight"][key] == candidate_manifest[key] for key in ("sha256", "bytes"))
            and native.workflow.resolve_config(geometry_config) == candidate_state["inputs"]["configuration"]
            and receipt["refiner_sha256"] == candidate_state["inputs"]["numerical_identity"]["source_sha256"][nested.REFINER],
            "The geometry receipt does not bind both native maps, shared preparation and fixed footprint.")
    manifest = native._read(native._verified(candidate_manifest, candidate_checkpoint.parent))
    require(receipt["end_face_dofs"] == manifest["end_face_dofs"]
            and all(value == parent["port_dofs"] == candidate["port_dofs"] for value in receipt["end_face_dofs"].values()),
            "The measured matched-port DOF counts differ from the saved responses.")
    for role, saved in (("parent", parent), ("candidate", candidate)):
        measured = receipt[role]
        require(measured["mesh_hash"] == saved["mesh_hash"]
                and measured["archive_mesh_sha256"] == saved["archive"]["mesh"]["sha256"]
                and measured["body_dofs"] == saved["body_dofs"],
                f"The measured {role} mesh differs from its accepted solve.")
    return receipt


def _linux_time(study, config_link):
    directory = study / f"thin100_central400_edge{TARGETS[TARGET]}"
    path = study / "local_checks" / f"{directory.name}_1GHz_time_AP_pnc.txt"
    fields = dict(line.strip().split(": ", maxsplit=1)
                  for line in path.read_text(encoding="utf-8").splitlines() if ": " in line)
    argv = shlex.split(fields["Command being timed"].strip('"'))
    peak_rss_kib = int(fields["Maximum resident set size (kbytes)"])
    exit_status = int(fields["Exit status"])
    require("tools/map_scattering_AP_pnc.py" in argv
            and (ROOT / argv[argv.index("--config") + 1]).resolve() == ROOT / config_link["path"]
            and (ROOT / argv[argv.index("--output") + 1]).resolve() == directory
            and float(argv[argv.index("--frequencies-ghz") + 1]) == FREQUENCY_GHZ
            and argv[argv.index("--max-new-points") + 1] == "1"
            and exit_status == 0 and peak_rss_kib > 0,
            "The GNU time receipt does not identify a successful single native Linux map run.")
    return {"receipt": native._link(path, ROOT), "exit_status": exit_status, "peak_rss_kib": peak_rss_kib,
            "elapsed_wall_time": fields["Elapsed (wall clock) time (h:mm:ss or m:ss)"],
            "scope": "GNU time measurement for the native Linux map invocation, including assembly and solve."}


def _slot(block, incident, outgoing):
    a, b = (complex(*block[key][DRIVE_INDEX][DRIVE_INDEX]) for key in ("reference", "candidate_aligned"))
    power_a, power_b = abs(a)**2, abs(b)**2
    phase = float(np.angle(b / a, deg=True)) if min(abs(a), abs(b)) > scattering.PHASE_AMPLITUDE_FLOOR else None
    return {"reference_root_pair": [incident["reference_roots"][DRIVE_INDEX], outgoing["reference_roots"][DRIVE_INDEX]],
            "candidate_root_pair": [incident["candidate_roots"][DRIVE_INDEX], outgoing["candidate_roots"][DRIVE_INDEX]],
            "reference_coeff_re_im": [a.real, a.imag], "candidate_aligned_coeff_re_im": [b.real, b.imag],
            "reference_power": power_a, "candidate_power": power_b, "signed_power_change": power_b - power_a,
            "wrapped_phase_change_degrees": phase, "absolute_complex_change": abs(b - a),
            "reference_drive_total_power": block["reference_power"][DRIVE_INDEX],
            "candidate_drive_total_power": block["candidate_power"][DRIVE_INDEX],
            "signed_drive_total_power_change": block["power_change"][DRIVE_INDEX]}


def _compare(before, after):
    a_dir, a, fixed, a_identity, a_section = before
    b_dir, b, other, b_identity, b_section = after
    require(all(a_identity[key] == b_identity[key] for key in ("source_sha256", "packages")),
            "The refinement pair uses different numerical source or packages.")
    fixed, _ = scattering.compatible_inputs(a, b, fixed, other)
    with np.load(a_section, allow_pickle=False) as first, np.load(b_section, allow_pickle=False) as second:
        require(all(np.array_equal(first[key], second[key]) for key in ("points_m", "triangles")),
                "The refinement changed the saved section coordinates or triangulation.")
        section_difference = {}
        for key in ("K0", "M", "Q", "G", "reference_speed_m_s"):
            scale = float(np.max(abs(first[key])))
            error = float(np.max(abs(first[key] - second[key])))
            section_difference[key] = error / scale if scale else (0. if error == 0 else None)
    first = scattering.read_scattering(a_dir, a, fixed)
    second = scattering.read_scattering(b_dir, b, fixed)
    matches, blocks, _ = scattering.compare_blocks(first, second)
    require(matches["incoming"]["reference_roots"][DRIVE_INDEX] == matches["transmission"]["reference_roots"][DRIVE_INDEX]
            and matches["incoming"]["candidate_roots"][DRIVE_INDEX] == matches["transmission"]["candidate_roots"][DRIVE_INDEX],
            "Transmitted slot 3 is not the matched diagonal of incident drive 3.")
    power = max(block["maximum_power_change"] for block in blocks.values())
    complex_change = max(block["maximum_complex_change"] for block in blocks.values())
    return {"minimum_signed_mass_overlap": min(min(value["mass_overlaps"]) for value in matches.values()),
            "matched_channels": matches,
            "reference_to_candidate_section_operator_relative_max_entry_change": section_difference,
            "maximum_total_drive_power_change": power, "total_drive_power_passed": power < scattering.LIMIT,
            "maximum_complex_amplitude_change": complex_change, "complex_amplitude_passed": complex_change < scattering.LIMIT,
            "drive_3_to_slot_3": {name: _slot(block, matches["incoming"], matches[name]) for name, block in blocks.items()},
            "blocks": {name: {**nested._block_increment(block, matches["incoming"], matches[name]),
                              "all_five_drive_totals": {key: block[key] for key in ("reference_power", "candidate_power", "power_change")},
                              "all_coefficients_re_im": {key: block[key] for key in ("reference", "candidate_aligned")}}
                       for name, block in blocks.items()}}


def replay(study):
    study = Path(study).resolve()
    prior = native._read(PRIOR)
    reader_pin = prior["provenance"]["acceptance_reader"]
    reader = runpy.run_path(str(native._verified(reader_pin, ROOT)))["read_point"]
    records, data = {}, {}
    for target in TARGETS:
        records[target], data[target] = _read_case(study, target, prior, reader)
    require(all(records[target]["platform"]["solve"].startswith("Linux-")
                and "-x86_64-" in records[target]["platform"]["solve"] for target in (168.75, TARGET))
            and records[168.75]["platform"]["assembly"].startswith("Windows-")
            and records[TARGET]["platform"]["assembly"] == records[TARGET]["platform"]["solve"],
            "Expected Linux x86-64 solves with Windows reference and native Linux candidate preparations.")
    geometry_path = study / "local_checks/thin100_central400_edge126p5625_rim32_geometry_AP_pnc.json"
    geometry = _check_geometry(native._read(geometry_path), records[168.75], records[TARGET])
    records[TARGET]["measured_linux_map_execution"] = _linux_time(study, geometry["configuration"])
    increments = {}
    for a, b in zip(TARGETS, list(TARGETS)[1:]):
        require(all(records[a]["original_equation_identity"][key] != records[b]["original_equation_identity"][key]
                    for key in records[a]["original_equation_identity"]),
                "The mesh pair reuses a saved original equation identity.")
        increments[f"{a:g}_to_{b:.4f}".rstrip("0")] = _compare(data[a], data[b])
    return {**native.AUTHOR, "schema_version": 1, "status": "compared_saved_thin_device_refinements",
            "frequency_hz": FREQUENCY_GHZ * 1e9, "physical_thickness_nm": 100,
            "reference_planes_x_m": prior["reference_planes_x_m"], "incident_side": prior["incident_side"],
            "original_numerical_limits": prior["original_numerical_limits"],
            "total_drive_power_change_limit": scattering.LIMIT, "complex_amplitude_change_limit": scattering.LIMIT,
            "minimum_signed_mass_overlap_required": scattering.MIN_OVERLAP,
            "wrapped_phase_amplitude_floor": scattering.PHASE_AMPLITUDE_FLOOR,
            "fixed_geometry": geometry, "levels": {str(target): record for target, record in records.items()},
            "increments": increments,
            "scope": "The matched blocks have five left-incident drives and 25 propagating coefficients per direction. "
                     "Power uses native incident-power fractions. Phase differences are wrapped at the original planes "
                     "after signed SAFE gauge alignment. The 225 to 168.75 nm pair also changed solve platform. "
                     "The 168.75 to 126.5625 nm pair changes preparation platform and central mesh. These finite "
                     "increments at 1 GHz leave continuum and frequency convergence unresolved.",
            "provenance": {"prior_nested_comparison": native._link(PRIOR, ROOT), "acceptance_reader": reader_pin,
                           "geometry_receipt": native._link(geometry_path, ROOT),
                           "comparison_source": native._link(Path(__file__).resolve(), ROOT),
                           "block_reporting_source": native._link(Path(nested.__file__).resolve(), ROOT),
                           "matching_source": native._link(Path(scattering.__file__).resolve(), ROOT)}}


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--study-root", type=Path, default=STUDY)
    parser.add_argument("--output", type=Path, help="Write the single comparison JSON. Defaults to stdout.")
    args = parser.parse_args(argv)
    try:
        payload = json.dumps(replay(args.study_root), indent=2, allow_nan=False) + "\n"
        if args.output:
            output = args.output.resolve()
            require(output != PRIOR and not output.is_relative_to(args.study_root.resolve()),
                    "Place the comparison JSON outside the original native maps and prior evidence.")
            staging = output.with_suffix(output.suffix + ".tmp")
            staging.write_text(payload, encoding="utf-8", newline="\n")
            staging.replace(output)
        else:
            print(payload, end="")
    except (KeyError, OSError, TypeError, ValueError) as exc:
        parser.exit(1, f"{parser.prog}: {exc}\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
