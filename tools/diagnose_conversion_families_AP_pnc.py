"""Describe the accepted 1 GHz conversion pair using saved sections. Athul Prem | _AP_pnc."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
from pathlib import Path
import sys
from time import monotonic

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT / "src"), str(ROOT)]

from tools import diagnose_port_polarization_AP_pnc as metric
from tools import compare_scattering_AP_pnc as comparison_tool

AUTHOR = {"author": "Athul Prem", "file_tag": "_AP_pnc"}
FREQUENCY_HZ = 1_000_000_000
TOLERANCE = 1e-10
RELATIVE_FLOOR = 1e-12
JSON_NAME = "conversion_families_AP_pnc.json"
CSV_NAME = "conversion_families_AP_pnc.csv"
FIXED_ROOTS = {"z4": (1348, 1311, 1292, 1284, 1274), "z6": (1940, 1877, 1867, 1855, 1837)}
FIXED_REVERSE = {"z4": (1349, 1312, 1293, 1285, 1275), "z6": (1941, 1878, 1868, 1856, 1838)}
BASE = "results/runs/full_device_resource_profile/"
POINT_PATHS = {
    "h80_z4": BASE + "lawrence_h80_AP_pnc/h80-export-1335045/historical_26hole_layered_h80_ny12_z4-3385249d64d1",
    "h80_z6": BASE + "lawrence_thickness_AP_pnc/thickness-h80-export-1335341/historical_26hole_layered_h80_ny12_z6-585ccac1d4f2",
}
ROLE_PINS = {
    "metric_source": ("tools/diagnose_port_polarization_AP_pnc.py", "367eb1fa5ed0296df548716cb8eb02e422dd639a23f8ad434dc0e357af5d1e79"),
    "matching_source": ("tools/compare_scattering_AP_pnc.py", "148f7efa58ad668a7a57be9db68a3241ffb67ed94067499ce6dcd4a0f8c59313"),
    "device_thickness_verification": ("verification/device_thickness_AP_pnc.json", "81f1e195956c1986d067fc397ea1836c284d9e645073e032d323772918532f05"),
    "thickness_comparison": (BASE + "lawrence_thickness_AP_pnc/comparisons/scattering-b06d34228c560e43/summary_AP_pnc.json", "8a5aea4238d3f09bdd7e97f82cd73d2b74f6a64846ca946633aec08d0b6dede3"),
    "z4_safe_receipt": ("results/runs/full_device_port_resolution/12x4/attempt_lpnbd9kr/checkpoint_AP_pnc.json", "677c64a816ea42a8b37e1358d5549d63ecaeec7623a9366139ba64d301bbea4e"),
    "z6_safe_receipt": (BASE + "lawrence_thickness_AP_pnc/thickness-h80-safe-1335308/thickness_h80_safe_AP_pnc.json", "b5c0d73656e90aa5ba312939c0ef25633af5a775974f5fc778031584d55c78f3"),
    "z4_section": ("results/runs/full_device_port_resolution/12x4/attempt_lpnbd9kr/arrays_AP_pnc.npz", "2a2aa9347921ac6940f851f2eb53bc06480a28a6cf67c27a000faa357b1404e8"),
    "z6_section": (BASE + "lawrence_thickness_AP_pnc/thickness-h80-safe-1335308/ports_AP_pnc.npz", "a13a58e47dee91e4c7d75bb9ec77feb7656ae674392c2f1d7485129da76d0e6a"),
    "h80_z4_summary": (POINT_PATHS["h80_z4"] + "/summary_AP_pnc.json", "f65f61eaf62d69fa0574fe8566ba088fd7e090a019685bdac3db95261b61619f"),
    "h80_z6_summary": (POINT_PATHS["h80_z6"] + "/summary_AP_pnc.json", "040a219f54446219b6a610cf08382ff7e93a6dcb01f31f3be810977651301c4e"),
    "h80_z4_scattering": (POINT_PATHS["h80_z4"] + "/scattering_AP_pnc.csv", "5b718ef9bbdf526540f027af433342209bed3dd8f26b9258ee2bb42158074fc0"),
    "h80_z6_scattering": (POINT_PATHS["h80_z6"] + "/scattering_AP_pnc.csv", "54cb1ff9cf3d69cef63f3c9858ec7587efee7cec4fe9324751f7094ea323598a"),
    "h80_z4_coefficients": (POINT_PATHS["h80_z4"] + "/scattering_channels_AP_pnc.csv", "3f8094e029e3abd3c705eb72f0e9f5c360f6e12a00e77fe48f17ce5fb01c9978"),
    "h80_z6_coefficients": (POINT_PATHS["h80_z6"] + "/scattering_channels_AP_pnc.csv", "bca65a396d45a15c0f3b8ba58a5d85e9add2b1a7b24b3ec111bbebb0310d6014"),
}
METADATA_ROLES = ("device_thickness_verification", "thickness_comparison", "z4_safe_receipt", "z6_safe_receipt",
                  "h80_z4_summary", "h80_z6_summary")
METRIC_KEYS = ("component_fractions_xyz", "template_scores_tx_ty_tz_rx", "template_projection_fraction",
               "template_remainder_fraction")


def require(condition, message):
    if not condition:
        raise ValueError(message)


def file_sha(path):
    with Path(path).open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def bind_input(workspace, role, inputs):
    """Bind each scientific role to its own exact path and digest."""
    relative, expected = ROLE_PINS[role]
    root = Path(workspace).resolve()
    path = (root / relative).resolve(strict=True)
    require(path.is_relative_to(root) and path.is_file(), f"Input role escapes the workspace: {role}")
    require(file_sha(path) == expected, f"The accepted identity differs for role {role}.")
    inputs.append({"role": role, "path": relative, "sha256": expected, "bytes": path.stat().st_size})
    return path


def validate_metadata(documents):
    """Reject valid archive files used in the wrong accepted scientific role."""
    ledger, control = (documents[key] for key in ("device_thickness_verification", "thickness_comparison"))
    require(control["status"] == "compared" and control["input_power_checks_passed"] is True
            and control["power_change_limit"] == control["complex_change_limit"] == .01
            and control["power_passed"] is True and control["complex_passed"] is False
            and control["mesh_pair_passed"] is False, "The accepted failed 1 GHz control status differs.")
    previous = ledger["h80_z4_z6_mesh_pair"]
    require(previous["comparison_summary_sha256"] == ROLE_PINS["thickness_comparison"][1]
            and previous["mesh_pair_passed"] is False
            and previous["maximum_complex_change"] == control["maximum_complex_change"]
            and previous["maximum_power_change"] == control["maximum_power_change"],
            "The retained failed control does not bind to its accepted ledger.")
    for section, label in (("z4", "reference"), ("z6", "candidate")):
        roots, reverse = list(FIXED_ROOTS[section]), list(FIXED_REVERSE[section])
        point = "h80_" + section
        record, summary = control[label], documents[point + "_summary"]
        require(record["summary_sha256"] == ROLE_PINS[point + "_summary"][1]
                and record["run_hash"] == summary["run_hash"]
                and summary["frequency_hz"] == FREQUENCY_HZ and summary["incident_side"] == "left"
                and summary["incident_mode_indices"] == roots and summary["num_incident_channels"] == 5
                and summary["power_balance_passed"] is True,
                f"The accepted summary/run/roots differ for {point}.")
        for role, name in (("scattering", "scattering_AP_pnc.csv"), ("coefficients", "scattering_channels_AP_pnc.csv")):
            require(record["output_sha256"][name] == summary["output_sha256"][name] == ROLE_PINS[point + "_" + role][1],
                    f"The accepted CSV role differs for {point}/{role}.")
        for group, expected in (("incoming", roots), ("transmission", roots), ("reflection", reverse)):
            match = control["matched_channels"][group]
            require(match[label + "_roots"] == expected, f"The accepted signed match differs: {section}/{group}.")
            values = np.asarray(match[label + "_wavenumber_rad_m"])
            require(values.shape == (5,) and np.isfinite(values).all()
                    and np.all(values * (-1 if group == "reflection" else 1) > 0),
                    "The accepted signed wavenumbers differ.")
        config = summary["configuration"]
        require(config["analysis"]["frequency_GHz"] == 1 and config["geometry"]["thickness_nm"] == 200
                and config["material"]["density_kg_m3"] == 5323
                and config["material"]["temperature_policy"] == "fixed_property_control",
                "The accepted frequency, thickness or material assumptions differ.")
    first, second = documents["z4_safe_receipt"], documents["z6_safe_receipt"]
    require(first["status"] == "solved" and first["trace_dofs"] == 675
            and first["inputs"]["frequency_hz"] == FREQUENCY_HZ
            and first["arrays"]["sha256"] == ROLE_PINS["z4_section"][1], "The z4 section receipt differs.")
    require(second["status"] == "safe_ready" and second["trace_dofs"] == 975
            and second["frequency_hz"] == FREQUENCY_HZ
            and second["ports"]["sha256"] == ROLE_PINS["z6_section"][1]
            and second["incident_port_roots"] == list(FIXED_ROOTS["z6"])
            and second["backward_port_roots"] == list(FIXED_REVERSE["z6"])
            and ledger["safe_section"]["receipt_sha256"] == ROLE_PINS["z6_safe_receipt"][1]
            and ledger["safe_section"]["ports_file_sha256"] == ROLE_PINS["z6_section"][1]
            and ledger["scattering_export"]["public_summary_sha256"] == ROLE_PINS["h80_z6_summary"][1],
            "The z6 section or accepted response receipt differs.")


def unique_reverse_partners(roots, negative_roots, waves):
    """Use signed wavenumbers, never root-number adjacency, to identify reversal."""
    roots, negative_roots, waves = np.asarray(roots), np.asarray(negative_roots), np.asarray(waves)
    require(len(roots) == len(negative_roots) == 5 and len(set(roots)) == len(set(negative_roots)) == 5
            and not set(roots).intersection(negative_roots), "Five distinct signed roots are required.")
    result = []
    for root in roots:
        partners = negative_roots[np.isclose(waves[negative_roots], -waves[root], rtol=1e-12, atol=0)]
        require(len(partners) == 1, "A propagating root lacks a unique reverse-wavenumber partner.")
        result.append(int(partners[0]))
    require(len(set(result)) == 5, "Reverse-wavenumber matching is not one-to-one.")
    return np.asarray(result)


def read_modes(path, section, roots, expected_reverse, expected_dofs):
    """Read only section arrays and selected forward/backward fields, never body fields."""
    require(Path(path).stat().st_size <= 64 * 1024**2, "Section mode archive exceeds 64 MiB.")
    roots, expected_reverse = np.asarray(roots), np.asarray(expected_reverse)
    with np.load(path, allow_pickle=False) as arrays:
        if section == "z6":
            require(float(arrays["frequency_hz"]) == FREQUENCY_HZ, "The z6 mode frequency differs.")
        k = arrays["k_rad_m" if section == "z4" else "wavenumbers_rad_m"]
        powers = arrays["signed_power_w" if section == "z4" else "powers_w"]
        propagating, positive, negative = (arrays[key] for key in ("propagating", "positive_indices", "negative_indices"))
        residuals, displacement = arrays["relative_residuals"], arrays["displacements"]
        n = expected_dofs
        require(k.shape == powers.shape == propagating.shape == residuals.shape == (2 * n,)
                and propagating.dtype == np.bool_ and positive.shape == negative.shape == (n,)
                and positive.dtype.kind in "iu" and negative.dtype.kind in "iu"
                and np.array_equal(np.sort(np.r_[positive, negative]), np.arange(2 * n))
                and displacement.shape == (n, 2 * n) and displacement.dtype == np.complex128,
                "The saved complete mode partition, dimensions or type differs.")
        require(np.array_equal(positive[propagating[positive]], roots), "The five accepted incoming roots differ.")
        backward = negative[propagating[negative]]
        require(np.array_equal(backward, expected_reverse), "The five accepted reflected roots differ.")
        reverse = unique_reverse_partners(roots, backward, k)
        selected = np.r_[roots, reverse]
        require(np.isfinite(k[selected]).all() and np.all(k[selected].imag == 0)
                and np.all(k[roots].real > 0) and np.all(k[reverse].real < 0)
                and np.allclose(powers[roots], 1, rtol=0, atol=1e-7)
                and np.allclose(powers[reverse], -1, rtol=0, atol=1e-7)
                and np.isfinite(residuals[selected]).all() and np.all(residuals[selected] >= 0)
                and np.all(residuals[selected] <= 1e-8) and np.isfinite(displacement[:, selected]).all(),
                "The accepted signed modes, normalization or residuals differ.")
        return {"roots": roots, "reverse_roots": reverse, "k": k[roots].real, "reverse_k": k[reverse].real,
                "powers": powers[roots], "displacements": displacement[:, roots].copy(),
                "reverse_displacements": displacement[:, reverse].copy()}


def parse_response_tables(summary, total_rows, coefficient_rows, modes):
    """Join all five channels and 50 coefficients by explicit signed identities."""
    roots, reverse = modes["roots"].tolist(), modes["reverse_roots"].tolist()
    require(summary["frequency_hz"] == FREQUENCY_HZ and summary["incident_side"] == "left"
            and summary["incident_mode_indices"] == roots and summary["num_incident_channels"] == 5
            and summary["power_balance_passed"] is True, "Response summary frequency or signed roots differ.")
    require(len(total_rows) == 5 and len(coefficient_rows) == 50, "Response tables require five totals and 50 coefficients.")
    totals = {}
    for row in total_rows:
        order = int(row["incident_mode_order"])
        require(order in range(1, 6) and order not in totals and float(row["frequency_GHz"]) == 1,
                "Total rows duplicate a channel or differ in frequency/order.")
        values = [float(row[key]) for key in ("reflection", "transmission")]
        require(np.isfinite(values).all() and min(values) >= 0 and abs(sum(values) - 1) < .01,
                "A total row has invalid reflection/transmission powers.")
        totals[order] = values
    entries = {}
    for row in coefficient_rows:
        kind, incoming, outgoing = row["kind"], int(row["incident_port_index"]), int(row["outgoing_port_index"])
        order = int(row["incident_mode_order"])
        require(kind in ("reflection", "transmission") and order in range(1, 6) and roots[order - 1] == incoming
                and outgoing in (reverse if kind == "reflection" else roots), "A coefficient uses the wrong signed channel role.")
        key = (kind, incoming, outgoing)
        require(key not in entries, "A complex coefficient is duplicated.")
        real, imag, power = (float(row[name]) for name in ("amplitude_real", "amplitude_imag", "power_fraction"))
        require(np.isfinite([real, imag, power]).all() and power >= 0
                and math.isclose(power, real**2 + imag**2, rel_tol=0, abs_tol=TOLERANCE),
                "A coefficient power differs from its complex amplitude.")
        entries[key] = {"kind": kind, "incident_root": incoming, "outgoing_root": outgoing,
                        "amplitude_real": real, "amplitude_imag": imag, "power_fraction": power}
    rows, coefficients = [], []
    for order, (root, partner) in enumerate(zip(roots, reverse, strict=True), 1):
        powers, converted = {}, {}
        for kind, outgoing_roots, same in (("reflection", reverse, partner), ("transmission", roots, root)):
            group = [entries[kind, root, outgoing] for outgoing in outgoing_roots]
            powers[kind] = math.fsum(item["power_fraction"] for item in group)
            converted[kind] = math.fsum(item["power_fraction"] for item in group if item["outgoing_root"] != same)
            coefficients.extend(group)
        R, T = totals[order]
        require(abs(R - powers["reflection"]) <= TOLERANCE and abs(T - powers["transmission"]) <= TOLERANCE,
                "The complete coefficient power sums differ from accepted totals.")
        rows.append({"root": root, "reverse_root": partner, "R": R, "T": T,
                     "converted_R": converted["reflection"], "converted_T": converted["transmission"]})
    return {"run_hash": summary["run_hash"], "rows": rows, "coefficients": coefficients}


def check_signed_matches(raw, matching_sections, control):
    """Reproduce both directions, retaining the accepted modal gauge convention."""
    first, second = matching_sections["z4"], matching_sections["z6"]
    result = {}
    for group in ("incoming", "reflection"):
        reflected = group == "reflection"
        objects = []
        for name, section in (("z4", first), ("z6", second)):
            data = raw[name]
            objects.append(comparison_tool.Modes(data["reverse_roots" if reflected else "roots"],
                data["reverse_k" if reflected else "k"],
                data["reverse_displacements" if reflected else "displacements"][section.trace_to_scalar]))
        _, _, actual = comparison_tool.align(*objects, second.mass, comparison_tool.interpolation(first, second), True)
        expected = control["matched_channels"][group]
        for key in ("reference_roots", "candidate_roots"):
            require(actual[key] == expected[key], f"The independently replayed {group} roots differ.")
        for key in ("mass_overlaps", "reference_wavenumber_rad_m", "candidate_wavenumber_rad_m"):
            require(np.allclose(actual[key], expected[key], rtol=0, atol=TOLERANCE), f"The replayed {group}/{key} differs.")
        phase_error = np.angle(np.exp(1j * (np.asarray(actual["candidate_gauge_phase_rad"])
                                          - expected["candidate_gauge_phase_rad"])))
        require(np.max(abs(phase_error)) <= TOLERANCE, "The accepted signed modal gauges differ.")
        result[group] = actual
    require(control["matched_channels"]["incoming"] == control["matched_channels"]["transmission"],
            "The transmitted and incoming bases do not share the accepted signed match.")
    return result


def coefficient_changes(responses, control):
    """Apply outgoing phase / incoming phase, preserving every original coefficient."""
    tables = {name: {(row["kind"], row["incident_root"], row["outgoing_root"]): row
                     for row in response["coefficients"]} for name, response in responses.items()}
    incoming = control["matched_channels"]["incoming"]
    output = []
    for kind in ("reflection", "transmission"):
        outgoing = control["matched_channels"][kind]
        a_saved, b_saved = (np.asarray(control["blocks"][kind][key]) for key in ("reference", "candidate_aligned"))
        require(a_saved.shape == b_saved.shape == (5, 5, 2), "Accepted complex block dimensions differ.")
        for i, (a_in, b_in) in enumerate(zip(incoming["reference_roots"], incoming["candidate_roots"], strict=True)):
            for j, (a_out, b_out) in enumerate(zip(outgoing["reference_roots"], outgoing["candidate_roots"], strict=True)):
                a, b = tables["h80_z4"][kind, a_in, a_out], tables["h80_z6"][kind, b_in, b_out]
                aa, bb = complex(a["amplitude_real"], a["amplitude_imag"]), complex(b["amplitude_real"], b["amplitude_imag"])
                aligned = bb * np.exp(1j * (outgoing["candidate_gauge_phase_rad"][j]
                                           - incoming["candidate_gauge_phase_rad"][i]))
                require(abs(aa - complex(*a_saved[j, i])) <= TOLERANCE
                        and abs(aligned - complex(*b_saved[j, i])) <= TOLERANCE,
                        "A signed gauge-aligned coefficient differs from the accepted comparison.")
                delta = aligned - aa
                output.append({"kind": kind, "z4_incident_root": a_in, "z6_incident_root": b_in,
                    "z4_outgoing_root": a_out, "z6_outgoing_root": b_out,
                    "z4_amplitude": [aa.real, aa.imag], "z6_raw_amplitude": [bb.real, bb.imag],
                    "z6_aligned_amplitude": [aligned.real, aligned.imag],
                    "aligned_complex_delta": [delta.real, delta.imag], "absolute_complex_change": abs(delta),
                    "z4_power": a["power_fraction"], "z6_power": b["power_fraction"],
                    "z6_minus_z4_power": b["power_fraction"] - a["power_fraction"]})
    return output


def select_conversion_pair(response):
    """Rank unordered channel pairs by the sum of their two reflected conversion powers."""
    roots = [row["root"] for row in response["rows"]]
    reverse = {row["root"]: row["reverse_root"] for row in response["rows"]}
    entries = {(row["incident_root"], row["outgoing_root"]): row for row in response["coefficients"]
               if row["kind"] == "reflection"}
    pairs = []
    for i, first in enumerate(roots):
        for second in roots[i + 1:]:
            directions = [entries[first, reverse[second]], entries[second, reverse[first]]]
            pairs.append({"forward_roots": [first, second], "reverse_roots": [reverse[first], reverse[second]],
                          "two_direction_power_sum": math.fsum(row["power_fraction"] for row in directions),
                          "directions": directions})
    pairs.sort(key=lambda pair: pair["two_direction_power_sum"], reverse=True)
    return {"selection": "Largest sum of the two off-diagonal reflected powers; descriptive ranking only.",
            "all_offdiagonal_reflection_zero": pairs[0]["two_direction_power_sum"] == 0,
            "selected_pair": pairs[0] if pairs[0]["two_direction_power_sum"] > 0 else None,
            "ranked_pairs": pairs}


def make_matched_rows(sections, responses, control):
    rows = []
    match = control["matched_channels"]["incoming"]
    lookup = {name: {row["root"]: row for row in value["rows"]} for name, value in responses.items()}
    for index, (a_root, b_root) in enumerate(zip(match["reference_roots"], match["candidate_roots"], strict=True)):
        a, b = lookup["h80_z4"][a_root], lookup["h80_z6"][b_root]
        row = {"z4_root": a_root, "z6_root": b_root, "mass_overlap": match["mass_overlaps"][index]}
        for key in METRIC_KEYS:
            delta = np.asarray(b[key]) - a[key]
            row["z6_minus_z4_" + key] = delta.tolist()
        for key in ("R", "T", "converted_R", "converted_T"):
            delta = b[key] - a[key]
            row["z6_minus_z4_" + key] = delta
            row["z6_minus_z4_relative_" + key] = delta / max(abs(a[key]), RELATIVE_FLOOR)
        rows.append(row)
    return rows


def diagnose(workspace=ROOT):
    """Compute one immutable archive diagnosis in memory, without modifying sources."""
    started = monotonic()
    inputs = []
    for role in ("metric_source", "matching_source"):
        path = bind_input(workspace, role, inputs)
        module = metric if role == "metric_source" else comparison_tool
        require(path == Path(module.__file__).resolve(), "Frozen helper import differs from the pinned source path.")
    documents = {role: json.loads(bind_input(workspace, role, inputs).read_bytes()) for role in METADATA_ROLES}
    validate_metadata(documents)
    control, ledger = documents["thickness_comparison"], documents["device_thickness_verification"]
    sections, raw, matching_sections = {}, {}, {}
    for name in ("z4", "z6"):
        path = bind_input(workspace, name + "_section", inputs)
        data, matching_sections[name] = metric.read_section(path, name)
        raw[name] = read_modes(path, name, FIXED_ROOTS[name], FIXED_REVERSE[name], len(data["coordinates_yz_m"]))
        value = metric.diagnose_modes(coordinates_yz_m=data["coordinates_yz_m"],
            component_dofs_xyz=data["component_dofs_xyz"], mass_kg_per_m=data["mass_kg_per_m"],
            displacements=raw[name]["displacements"])
        value.update(area_m2=value["mass_per_length_kg_m"] / 5323, density_kg_m3=5323,
                     displacement_dofs=len(data["coordinates_yz_m"]), mode_order=list(FIXED_ROOTS[name]))
        for i, row in enumerate(value["modes"]):
            row.update(root=int(raw[name]["roots"][i]), reverse_root=int(raw[name]["reverse_roots"][i]),
                       wavenumber_rad_m=float(raw[name]["k"][i]), signed_power_w=float(raw[name]["powers"][i]))
        sections[name] = value
    signed_matches = check_signed_matches(raw, matching_sections, control)
    responses = {}
    for point in ("h80_z4", "h80_z6"):
        paths = {kind: bind_input(workspace, point + "_" + kind, inputs) for kind in ("scattering", "coefficients")}
        tables = {}
        for kind, path in paths.items():
            with path.open(newline="", encoding="utf-8") as stream:
                tables[kind] = list(csv.DictReader(stream))
        name = point[-2:]
        response = parse_response_tables(documents[point + "_summary"], tables["scattering"], tables["coefficients"], raw[name])
        response["section"] = name
        for row, mode in zip(response["rows"], sections[name]["modes"], strict=True):
            row.update(mode)
        responses[point] = response
    changes = coefficient_changes(responses, control)
    return {**AUTHOR, "schema_version": 1, "status": "diagnosed_within_saved_sections",
        "question": "Which rigid-template contents and remainders associate with the dominant 1 GHz reflected conversion pair, and how do they change between the saved z4/z6 sections?",
        "frequency_hz": FREQUENCY_HZ, "inputs": inputs,
        "sources": [{"path": str(Path(__file__).resolve()), "sha256": file_sha(__file__)}],
        "sections": sections, "responses": responses,
        "signed_matches": signed_matches, "matched_rows": make_matched_rows(sections, responses, control),
        "coefficient_changes": changes,
        "dominant_conversion_pair": {name: select_conversion_pair(value) for name, value in responses.items()},
        "numerical_tolerance": TOLERANCE, "relative_reporting_floor": RELATIVE_FLOOR,
        "definitions": {"templates": ["(1,0,0)", "(0,1,0)", "(0,0,1)", "(0,-(z-z_c),y-y_c)"],
            "template_order": ["Tx", "Ty", "Tz", "Rx"],
            "template_score": "abs(q^H M u)^2 / ((q^H M q)*(u^H M u)); mass-weighted shape overlap, not power",
            "template_remainder_fraction": "Content outside all four tested rigid templates; retain mixed or higher section shapes",
            "converted_R": "Sum of reflected power except the unique negative-wavenumber partner of the incident root",
            "converted_T": "Sum of transmitted power except the incident positive-wavenumber root",
            "complex_alignment": "S_z6_aligned[out,in] = exp(i*phase_out) * S_z6_raw[out,in] / exp(i*phase_in), using accepted signed section overlaps",
            "relative_delta": "(value_z6-value_z4)/max(abs(value_z4),1e-12); reporting guard only, not uncertainty",
            "signed_delta": "z6 minus z4, at the original physical end planes"},
        "preserved_scientific_status": {**{key: control[key] for key in ("input_power_checks_passed", "power_change_limit",
            "complex_change_limit", "maximum_power_change", "maximum_complex_change", "power_passed", "complex_passed", "mesh_pair_passed")},
            "maximum_same_channel_transmission_phase_change_rad": ledger["h80_z4_z6_mesh_pair"]["maximum_same_channel_transmission_phase_change_rad"],
            "reference_planes_m": [0., 1.665e-5]},
        "scope": {"body_arrays_loaded": False, "body_fields_loaded": False, "matrix_assembly_performed": False,
            "safe_eigensolve_performed": False, "body_solve_performed": False, "new_frequency_point_computed": False,
            "phase_adjustment_applied": False, "spatial_convergence_established": False, "frequency_convergence_established": False,
            "physical_branch_classification_established": False, "conversion_mechanism_established": False,
            "cross_frequency_branch_tracking_performed": False},
        "limits": ["Finite-section descriptors and finite thickness-discretization increments are not a conversion mechanism or continuum classification.",
            "All template scores and remainder are retained; the numerically largest template is only descriptive.",
            "The accepted 1 GHz complex-response thickness comparison failed, as did the retained in-plane comparisons.",
            "Nominal fixed Ge is not measured cryogenic sample data. Hole faceting and spatial/frequency convergence remain unresolved.",
            "No 0.1 GHz analysis was repeated and no cross-frequency branch identity is asserted."],
        "elapsed_seconds": monotonic() - started}


def publish(document, output):
    """Create a fresh compact ten-row CSV and JSON, never overwrite an archive."""
    output = Path(output).resolve()
    require(not output.exists(), "Select a fresh output directory; existing results must be preserved.")
    output.mkdir(parents=True)
    columns = ["point", "section", "root", "reverse_root", "wavenumber_rad_m", "signed_power_w", "R", "T", "converted_R", "converted_T",
               "kinetic_x", "kinetic_y", "kinetic_z", "score_tx", "score_ty", "score_tz", "score_rx",
               "template_projection_fraction", "template_remainder_fraction", "descriptive_largest_template"]
    with (output / CSV_NAME).open("x", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=columns)
        writer.writeheader()
        for point, response in document["responses"].items():
            for row in response["rows"]:
                values = {key: row[key] for key in columns if key in row}
                values.update(point=point, section=response["section"])
                values.update(zip(("kinetic_x", "kinetic_y", "kinetic_z"), row["component_fractions_xyz"]))
                values.update(zip(("score_tx", "score_ty", "score_tz", "score_rx"), row["template_scores_tx_ty_tz_rx"]))
                writer.writerow(values)
    result = {**document, "output_sha256": {CSV_NAME: file_sha(output / CSV_NAME)}}
    with (output / JSON_NAME).open("x", encoding="utf-8") as stream:
        json.dump(result, stream, indent=2, allow_nan=False)
        stream.write("\n")
    return output / JSON_NAME


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", required=True, type=Path, help="Fresh directory for JSON and compact CSV")
    args = parser.parse_args()
    require(not args.output.exists(), "Select a fresh output directory.")
    result = diagnose()
    print(json.dumps({"status": result["status"], "json": str(publish(result, args.output)), "rows": 10,
                      "elapsed_seconds": result["elapsed_seconds"]}))


if __name__ == "__main__":
    main()
