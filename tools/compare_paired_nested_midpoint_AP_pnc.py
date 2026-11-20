#!/usr/bin/env python3
"""Compare three finite frequency samples on the same nested device mesh. Athul Prem | _AP_pnc."""

from __future__ import annotations

import argparse
from math import isclose, isfinite
from pathlib import Path
import runpy
import sys

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if __name__ == "__main__":
    sys.path.insert(0, str(ROOT))

from tools import compare_paired_mode_frequency_AP_pnc as paired
from tools import map_scattering_AP_pnc as native
from tools.compare_scattering_AP_pnc import require

STUDY = ROOT / "results/runs/physical_thickness_device_AP_pnc"
PRIOR = ROOT / "verification/paired_mode_frequency_AP_pnc.json"
OUTPUT = ROOT / "verification/paired_nested_midpoint_AP_pnc.json"
TARGET = 168.75
FREQUENCIES = paired.FREQUENCIES_HZ
QUANTITIES = ("transmission", "reflection")
SHARED = ("mesh", "stiffness", "mass", "port_section")


def _power_stats(fractions):
    lower, middle, upper = fractions
    require(all(isfinite(x) for row in fractions for x in row.values()), "Power table contains a nonfinite value.")
    return {
        "sample_values_times_100": [{key: 100 * row[key] for key in QUANTITIES} for row in fractions],
        "linear_midpoint_times_100": {key: 50 * (lower[key] + upper[key]) for key in QUANTITIES},
        "midpoint_minus_linear_percentage_points": {
            key: 100 * (middle[key] - (lower[key] + upper[key]) / 2) for key in QUANTITIES
        },
    }


def _shared_inputs(directory):
    state = native.read_map_checkpoint(directory)
    files = {}
    for key in SHARED:
        pin = state["shared"]["artifacts"][key]
        path = native._verified(pin, directory)
        files[key] = {**pin, "path": path.relative_to(ROOT).as_posix()}
    return state, files


def _load_series(role, level, mesh, limits, provenance, reader):
    prefix = f"{role}{paired.THICKNESS[role]}_central400_edge168p75_rim32"
    endpoints = STUDY / f"{prefix}_linux"
    midpoint = STUDY / f"{prefix}_997p5MHz_linux"
    endpoint_state, endpoint_files = _shared_inputs(endpoints)
    midpoint_state, midpoint_files = _shared_inputs(midpoint)
    require(endpoint_state["inputs"]["frequencies_GHz"] == [f / 1e9 for f in FREQUENCIES[::2]]
            and midpoint_state["inputs"]["frequencies_GHz"] == [FREQUENCIES[1] / 1e9]
            and endpoint_state["shared"]["mesh_hash"] == midpoint_state["shared"]["mesh_hash"] == level["geometry"][role]["mesh_hash"]
            and endpoint_files["mesh"]["sha256"] == level["geometry"][role]["archive_mesh_sha256"]
            and all((endpoint_files[name]["sha256"], endpoint_files[name]["bytes"])
                    == (midpoint_files[name]["sha256"], midpoint_files[name]["bytes"]) for name in SHARED),
            f"The 997.5 MHz {role} preparation differs from the accepted endpoint mesh, K/M or section.")
    midpoint_flux = STUDY / f"{midpoint.name}_997p5MHz_flux_AP_pnc.json"
    pins = (level["samples"]["995MHz"][role]["archive"],
            {"map": native._link(midpoint / native.MAP, ROOT), "independent_flux_replay": native._link(midpoint_flux, ROOT)},
            level["samples"]["1GHz"][role]["archive"])
    directories = (endpoints, midpoint, endpoints)
    require(all(native._verified(pin["map"], ROOT).parent == directory for pin, directory in zip(pins, directories, strict=True)),
            f"Unexpected paired nested map path for {role}.")
    saved = [paired.read_case(pin, frequency, paired.THICKNESS[role], mesh, limits, provenance, reader)
             for pin, frequency in zip(pins, FREQUENCIES, strict=True)]
    modes = []
    for _, acceptance, _ in saved:
        summary_path = native._verified(acceptance["native_archive"]["point_summary"], ROOT)
        summary = native._read(summary_path)
        modes.append(native._point_modes(summary_path.parent, summary)[0])
    with np.load(ROOT / endpoint_files["port_section"]["path"], allow_pickle=False) as arrays:
        match = [_match(modes[a], modes[b], arrays["M"]) for a, b in ((0, 1), (1, 2), (0, 2))]
    for group in native.GROUPS:
        low_mid, mid_high, low_high = (row[group]["candidate_order"] for row in match)
        require([mid_high[i] for i in low_mid] == low_high,
                f"The two adjacent signed {group} matches disagree with the direct endpoint match for {role}.")
    drives = []
    for index in range(paired.CHANNELS):
        order = [index, match[0]["incoming"]["candidate_order"][index], match[2]["incoming"]["candidate_order"][index]]
        rows = [case.channels[i] for (case, _, _), i in zip(saved, order, strict=True)]
        fractions = [{"transmission": row["T"], "reflection": row["R"]} for row in rows]
        drives.append({"native_drive_order_at_samples": [row["drive"] for row in rows],
                       "native_root_ids_at_samples": [row["root"] for row in rows], **_power_stats(fractions)})
    record = {"shared_artifact_bytes_match": True, "endpoint_shared": endpoint_files, "midpoint_shared": midpoint_files,
              "signed_modes": dict(zip(("995_to_997p5MHz", "997p5_to_1000MHz", "995_to_1000MHz"), match, strict=True)),
              "acceptance_at_samples": [acceptance for _, acceptance, _ in saved], "matched_drive_power": drives}
    return saved, record


def _match(reference, candidate, section_mass):
    match = native.match_modes(reference, candidate, section_mass)
    require(match["status"] == "matched", f"A within-thickness signed mode match is unresolved. {match['groups']}")
    return match["groups"]


def _extract_means(trace):
    return {key: trace[f"mean_{key}_fraction"] for key in QUANTITIES}


def _mean_and_contrast(series):
    means = {role: _power_stats(rows) for role, rows in series.items()}
    contrast = _power_stats([{key: thin[key] - thick[key] for key in QUANTITIES}
                             for thin, thick in zip(series["thin"], series["thick"], strict=True)])
    return {"equal_incoherent_mean_power": means, "thin_minus_thick": contrast}


def _sign_relation(first, second):
    if first == 0 or second == 0:
        return "both_zero" if first == second else "one_zero"
    return "same" if (first > 0) == (second > 0) else "opposite"


def replay():
    prior = native._read(PRIOR)
    prior_pins = prior["provenance"]
    nested = native._read(native._verified(prior_pins["nested_controls_and_execution"], ROOT))
    rim = native._read(native._verified(prior_pins["rim48_anchor"], ROOT))
    hole_pin = rim["provenance"]["hole_pattern_csv"]
    native._verified(hole_pin, ROOT)
    reader_pin = prior_pins["acceptance_reader"]
    reader = runpy.run_path(str(native._verified(reader_pin, ROOT)))["read_point"]
    flux_sources = prior_pins["independent_flux_sources"]
    limits = prior["original_numerical_limits"]
    require(nested["provenance"]["acceptance_reader"] == reader_pin
            and nested["provenance"]["independent_flux_sources"] == flux_sources
            and nested["physical_thickness_nm"] == paired.THICKNESS
            and all(nested[key] == prior[key] for key in ("original_numerical_limits", "reference_planes_x_m", "hole_count", "incident_side")),
            "The paired response families use different reference geometry, thicknesses or numerical acceptance.")
    provenance = {"reference_planes_x_m": nested["reference_planes_x_m"], "hole_source_sha256": hole_pin["sha256"],
                  "flux_producer_sha256": [pin["sha256"] for pin in flux_sources]}
    level, = (row for row in nested["levels"] if row["central_max_edge_nm"] == TARGET)
    mesh = {**nested["parent_mesh"], "central_max_edge_nm": TARGET}
    loaded = {role: _load_series(role, level, mesh, limits, provenance, reader) for role in paired.THICKNESS}
    settings = loaded["thin"][0][0][0].settings
    require(all(case.settings == settings for saved, _ in loaded.values() for case, _, _ in saved),
            "The paired nested device samples differ in material or placed geometry.")
    nested_stats = _mean_and_contrast({role: [_extract_means(trace) for _, _, trace in saved]
                                       for role, (saved, _) in loaded.items()})

    coarse = prior["all_mode_power"]["rim48_frequency_samples"]
    require(prior["frequency_sample_mesh"] == {**nested["parent_mesh"], "circle_resolution": 48}
            and [row["frequency_hz"] for row in coarse] == list(FREQUENCIES), "Unexpected coarse rim48 comparison.")
    for row in coarse:
        for data in row["by_thickness"].values():
            for pin in data["native_archive"].values():
                native._verified(pin, ROOT)
            require(all(isclose(data[f"mean_{key}_fraction"], data[f"{key}_trace"] / paired.CHANNELS,
                                rel_tol=0, abs_tol=paired.POWER_ROUNDOFF) for key in QUANTITIES),
                    "The published coarse mean does not equal its mode-summed power divided by the drive count.")
    coarse_stats = _mean_and_contrast({role: [_extract_means(row["by_thickness"][role]) for row in coarse]
                                       for role in paired.THICKNESS})
    a, b = (row["thin_minus_thick"] for row in (nested_stats, coarse_stats))
    contrast_key = "midpoint_minus_linear_percentage_points"
    comparison = {
        "nested_minus_rim48_midpoint_contrast_deviation_percentage_points": {key: a[contrast_key][key] - b[contrast_key][key] for key in QUANTITIES},
        "midpoint_contrast_deviation_sign_relation": {key: _sign_relation(a[contrast_key][key], b[contrast_key][key]) for key in QUANTITIES},
        "sampled_contrast_sign_relation": [{key: _sign_relation(x[key], y[key]) for key in QUANTITIES}
                                            for x, y in zip(a["sample_values_times_100"], b["sample_values_times_100"], strict=True)],
    }
    return {**native.AUTHOR, **native.CONVERGENCE, "schema_version": 1, "status": "compared_saved_nested_midpoints",
            "frequencies_hz": list(FREQUENCIES), "physical_thickness_nm": paired.THICKNESS,
            "reference_planes_x_m": provenance["reference_planes_x_m"], "incident_side": nested["incident_side"],
            "incident_count_per_thickness": paired.CHANNELS, "nested_mesh": mesh, "rim48_mesh": prior["frequency_sample_mesh"],
            "original_numerical_limits": limits,
            "convention": ("Each power is a fraction of incident power. The means average five equal-power incoherent inputs. "
                           "Displayed powers are percentages, while contrasts and signed differences are percentage points. "
                           "The midpoint deviation subtracts the average of the two endpoint values. Native drive identities "
                           "are matched at the samples separately for each thickness. The mean contrast needs no cross-thickness match."),
            "nested_rim32": {**nested_stats, "by_thickness": {role: record for role, (_, record) in loaded.items()}},
            "coarser_rim48": coarse_stats, "contrast_comparison": comparison,
            "scope": ("Finite 995, 997.5 and 1000 MHz responses at the original end planes. The two mesh families also differ "
                      "in hole faceting. The sampled chord deviations establish no bound between samples or in the continuum."),
            "provenance": {"paired_mode_frequency": native._link(PRIOR, ROOT), "nested_endpoints": prior_pins["nested_controls_and_execution"],
                           "coarse_native_archive_at_samples": [{role: data["native_archive"] for role, data in row["by_thickness"].items()} for row in coarse],
                           "hole_pattern": hole_pin, "acceptance_reader": reader_pin, "independent_flux_sources": flux_sources,
                           "comparisons": [native._link(Path(module).resolve(), ROOT) for module in (__file__, paired.__file__, native.__file__)]}}


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=OUTPUT)
    args = parser.parse_args(argv)
    try:
        require(args.output.resolve() == OUTPUT or not args.output.resolve().exists(),
                "Choose the canonical report or a new output path.")
        native.workflow.write_json(args.output, replay())
    except (OSError, RuntimeError, ValueError, KeyError) as exc:
        parser.exit(1, f"Nested midpoint comparison failed. {exc}\n")
    print(args.output.resolve())
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
