"""Independently verify saved 1 GHz conversion families. Athul Prem | _AP_pnc.

Only section topology/displacement columns and small accepted records are read.
The frozen explicit P2 quadrature is reused without saved mass or assemblers.
"""

from __future__ import annotations

import argparse
import csv
import itertools
import json
import os
from pathlib import Path
import sys

if __name__ == "__main__":
    os.environ.update(dict.fromkeys(("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS",
                                    "BLIS_NUM_THREADS", "NUMEXPR_NUM_THREADS"), "1"))

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from tools import verify_port_polarization_AP_pnc as quadrature

TOLERANCE = 1e-10
GEOMETRY_RELATIVE_TOLERANCE = 1e-12
RELATIVE_REPORTING_FLOOR = 1e-12
FREQUENCY_HZ = 1_000_000_000
FIXED_ROOTS = {"z4": [1348, 1311, 1292, 1284, 1274], "z6": [1940, 1877, 1867, 1855, 1837]}
QUADRATURE_SHA256 = "d5cdd47c7df2de902224de6f7c70299862187fa372a0eecaaa38e358e4693779"
FROZEN_SOURCE_HASHES = {
    "tools/diagnose_port_polarization_AP_pnc.py": "367eb1fa5ed0296df548716cb8eb02e422dd639a23f8ad434dc0e357af5d1e79",
    "tools/compare_scattering_AP_pnc.py": "148f7efa58ad668a7a57be9db68a3241ffb67ed94067499ce6dcd4a0f8c59313",
}
ARCHIVE = "results/runs/full_device_resource_profile/"
Z4_DIRECTORY = ARCHIVE + "lawrence_h80_AP_pnc/h80-export-1335045/historical_26hole_layered_h80_ny12_z4-3385249d64d1/"
Z6_DIRECTORY = ARCHIVE + "lawrence_thickness_AP_pnc/thickness-h80-export-1335341/historical_26hole_layered_h80_ny12_z6-585ccac1d4f2/"
ROLE_PINS = {
    "metric_source": ("tools/diagnose_port_polarization_AP_pnc.py", "367eb1fa5ed0296df548716cb8eb02e422dd639a23f8ad434dc0e357af5d1e79"),
    "matching_source": ("tools/compare_scattering_AP_pnc.py", "148f7efa58ad668a7a57be9db68a3241ffb67ed94067499ce6dcd4a0f8c59313"),
    "device_thickness_verification": ("verification/device_thickness_AP_pnc.json", "81f1e195956c1986d067fc397ea1836c284d9e645073e032d323772918532f05"),
    "z4_safe_receipt": ("results/runs/full_device_port_resolution/12x4/attempt_lpnbd9kr/checkpoint_AP_pnc.json", "677c64a816ea42a8b37e1358d5549d63ecaeec7623a9366139ba64d301bbea4e"),
    "z6_safe_receipt": (ARCHIVE + "lawrence_thickness_AP_pnc/thickness-h80-safe-1335308/thickness_h80_safe_AP_pnc.json", "b5c0d73656e90aa5ba312939c0ef25633af5a775974f5fc778031584d55c78f3"),
    "thickness_comparison": (ARCHIVE + "lawrence_thickness_AP_pnc/comparisons/scattering-b06d34228c560e43/summary_AP_pnc.json", "8a5aea4238d3f09bdd7e97f82cd73d2b74f6a64846ca946633aec08d0b6dede3"),
    "z4_section": ("results/runs/full_device_port_resolution/12x4/attempt_lpnbd9kr/arrays_AP_pnc.npz", "2a2aa9347921ac6940f851f2eb53bc06480a28a6cf67c27a000faa357b1404e8"),
    "z6_section": (ARCHIVE + "lawrence_thickness_AP_pnc/thickness-h80-safe-1335308/ports_AP_pnc.npz", "a13a58e47dee91e4c7d75bb9ec77feb7656ae674392c2f1d7485129da76d0e6a"),
    "h80_z4_summary": (Z4_DIRECTORY + "summary_AP_pnc.json", "f65f61eaf62d69fa0574fe8566ba088fd7e090a019685bdac3db95261b61619f"),
    "h80_z4_scattering": (Z4_DIRECTORY + "scattering_AP_pnc.csv", "5b718ef9bbdf526540f027af433342209bed3dd8f26b9258ee2bb42158074fc0"),
    "h80_z4_coefficients": (Z4_DIRECTORY + "scattering_channels_AP_pnc.csv", "3f8094e029e3abd3c705eb72f0e9f5c360f6e12a00e77fe48f17ce5fb01c9978"),
    "h80_z6_summary": (Z6_DIRECTORY + "summary_AP_pnc.json", "040a219f54446219b6a610cf08382ff7e93a6dcb01f31f3be810977651301c4e"),
    "h80_z6_scattering": (Z6_DIRECTORY + "scattering_AP_pnc.csv", "54cb1ff9cf3d69cef63f3c9858ec7587efee7cec4fe9324751f7094ea323598a"),
    "h80_z6_coefficients": (Z6_DIRECTORY + "scattering_channels_AP_pnc.csv", "bca65a396d45a15c0f3b8ba58a5d85e9add2b1a7b24b3ec111bbebb0310d6014"),
}
METRIC_KEYS = ("component_fractions_xyz", "template_scores_tx_ty_tz_rx", "template_projection_fraction", "template_remainder_fraction")


def require(condition, message):
    if not condition:
        raise ValueError(message)


def close(actual, expected, label, tolerance=TOLERANCE):
    a, b = np.asarray(actual), np.asarray(expected)
    require(a.shape == b.shape and a.dtype.kind in "fciu" and b.dtype.kind in "fciu"
            and np.isfinite(a).all() and np.isfinite(b).all()
            and np.all(np.abs(a - b) <= tolerance), f"Independent {label} differs.")


def verified_inputs(document, root=ROOT):
    """Pin every role to its exact archive path and bytes, not a hash whitelist."""
    root, roles = Path(root).resolve(), {}
    require(len(document["inputs"]) == len(ROLE_PINS), "Scientific input role count differs.")
    for item in document["inputs"]:
        role = item["role"]
        require(role in ROLE_PINS and role not in roles, "Unknown or duplicate scientific input role.")
        relative, digest = ROLE_PINS[role]
        path = (root / item["path"]).resolve()
        require(path == (root / relative).resolve() and path.is_relative_to(root)
                and item["sha256"] == digest and path.stat().st_size == item["bytes"]
                and quadrature.sha256(path) == digest, f"Accepted scientific role identity differs: {role}.")
        roles[role] = path
    require(set(roles) == set(ROLE_PINS), "Scientific roles are incomplete.")
    return roles


def bind_scientific_roles(roles):
    """Check accepted ledger -> receipt/section and comparison -> run/CSV links."""
    read = lambda role: json.loads(roles[role].read_bytes())
    ledger = read("device_thickness_verification")
    z4, z6, comparison = read("z4_safe_receipt"), read("z6_safe_receipt"), read("thickness_comparison")
    digest = lambda role: quadrature.sha256(roles[role])
    require(z4["arrays"]["sha256"] == digest("z4_section")
            and z4["inputs"]["frequency_hz"] == FREQUENCY_HZ
            and z4["trace_dofs"] == 675 and z4["status"] == "solved", "The accepted z4 section/receipt binding differs.")
    require(ledger["safe_section"]["receipt_sha256"] == digest("z6_safe_receipt")
            and ledger["safe_section"]["ports_file_sha256"] == digest("z6_section")
            and z6["ports"]["sha256"] == digest("z6_section")
            and z6["ports"]["bytes"] == roles["z6_section"].stat().st_size
            and z6["frequency_hz"] == FREQUENCY_HZ
            and z6["incident_port_roots"] == FIXED_ROOTS["z6"] and z6["status"] == "safe_ready"
            and ledger["scattering_export"]["public_summary_sha256"] == digest("h80_z6_summary"),
            "The accepted z6 section/receipt binding differs.")
    require(ledger["h80_z4_z6_mesh_pair"]["comparison_summary_sha256"] == digest("thickness_comparison"),
            "The accepted thickness comparison binding differs.")
    records = {}
    for point, key in (("h80_z4", "reference"), ("h80_z6", "candidate")):
        record, summary = comparison[key], read(f"{point}_summary")
        require(record["summary_sha256"] == digest(f"{point}_summary")
                and record["run_hash"] == summary["run_hash"], "The accepted response run binding differs.")
        for suffix, filename in (("scattering", "scattering_AP_pnc.csv"), ("coefficients", "scattering_channels_AP_pnc.csv")):
            require(record["output_sha256"][filename] == summary["output_sha256"][filename] == digest(f"{point}_{suffix}"),
                    "The accepted response CSV binding differs.")
        records[point] = record
    return comparison, records


def reverse_partners(waves, powers, positive, negative, propagating):
    """Find a unique opposite-wavenumber partner; do not assume root+1."""
    positive, negative = np.asarray(positive, dtype=int), np.asarray(negative, dtype=int)
    require(len(positive) == len(negative) == 5 and len(set(positive) | set(negative)) == 10,
            "The finite section must retain five distinct roots per direction.")
    require(np.isfinite(waves[np.r_[positive, negative]]).all()
            and np.isfinite(powers[np.r_[positive, negative]]).all()
            and np.all(np.asarray(propagating)[np.r_[positive, negative]])
            and np.all(waves[positive].imag == 0) and np.all(waves[negative].imag == 0)
            and np.all(waves[positive].real > 0) and np.all(waves[negative].real < 0)
            and np.allclose(powers[positive], 1, rtol=0, atol=1e-8)
            and np.allclose(powers[negative], -1, rtol=0, atol=1e-8), "Invalid signed propagating root metadata.")
    result = []
    for root in positive:
        candidates = negative[np.abs(waves[negative] + waves[root]) <= 1e-12 * abs(waves[root])]
        require(len(candidates) == 1, "Opposite-wavenumber partner is absent or ambiguous.")
        result.append(int(candidates[0]))
    require(len(set(result)) == 5, "Opposite-wavenumber matching is not one-to-one.")
    return result


def load_section(path, name):
    """Load only independent quadrature inputs and ten signed mode columns."""
    require(path.stat().st_size <= 64 * 1024**2, "Section verifier refuses an oversized archive.")
    roots = FIXED_ROOTS[name]
    with np.load(path, allow_pickle=False) as arrays:
        args = {key: arrays[field] for key, field in {
            "vertices_yz_m": "section_vertices_yz_m", "triangles": "section_triangles",
            "coordinates_yz_m": "coordinates_yz_m", "component_dofs_xyz": "component_dofs_xyz"}.items()}
        waves = arrays["k_rad_m" if name == "z4" else "wavenumbers_rad_m"]
        powers = arrays["signed_power_w" if name == "z4" else "powers_w"]
        propagating = arrays["propagating"]
        positive = arrays["positive_indices"][propagating[arrays["positive_indices"]]]
        negative = arrays["negative_indices"][propagating[arrays["negative_indices"]]]
        require(positive.tolist() == roots, "The archived ordered positive roots differ.")
        if name == "z6":
            require(float(arrays["frequency_hz"]) == FREQUENCY_HZ, "The section frequency differs.")
        reverse = reverse_partners(waves, powers, positive, negative, propagating)
        residuals = arrays["relative_residuals"][np.r_[positive, reverse]]
        require(np.isfinite(residuals).all() and np.all((residuals >= 0) & (residuals <= 1e-8)),
                "The saved signed eigenpair residuals differ.")
        args["displacements"] = arrays["displacements"][:, roots]
        backward = arrays["displacements"][:, reverse]
    require(len(args["coordinates_yz_m"]) == (675 if name == "z4" else 975), "The section DOF count differs.")
    args["density_kg_m3"] = 5323.0
    return args, dict(args, displacements=backward), waves[roots].real, powers[roots], reverse


def signed_match(source, target, accepted, source_roots, target_roots, source_waves, target_waves):
    """Independently recover permutation AND complex gauge from P2 integration."""
    first = quadrature.interpolate_fields(source, target)
    args = dict(target, displacements=np.column_stack((first, target["displacements"])))
    integral = quadrature.integrate_section(**args)
    gram = np.asarray(integral["displacement_mass_gram_real"]) + 1j * np.asarray(integral["displacement_mass_gram_imag"])
    count = len(source_roots)
    cross = gram[:count, count:] / np.sqrt(np.outer(np.diag(gram)[:count].real, np.diag(gram)[count:].real))
    permutations = itertools.permutations(range(count))
    order = np.asarray(max(permutations, key=lambda p: np.abs(cross[np.arange(count), p]).sum()))
    overlaps = np.abs(cross[np.arange(count), order])
    other = abs(cross).copy()
    other[np.arange(count), order] = 0
    require(np.all(overlaps >= .99) and np.all(overlaps > other.max(axis=1) + 1e-8)
            and np.all(overlaps > other[:, order].max(axis=0) + 1e-8), "Independent signed match is ambiguous.")
    phases = cross[np.arange(count), order] / overlaps
    require(accepted["reference_roots"] == list(source_roots)
            and accepted["candidate_roots"] == [target_roots[i] for i in order], "Accepted signed root matching differs.")
    close(accepted["mass_overlaps"], np.minimum(overlaps, 1), "signed mass overlaps")
    close(np.exp(1j * np.asarray(accepted["candidate_gauge_phase_rad"])), phases, "signed comparison gauge")
    close(accepted["reference_wavenumber_rad_m"], source_waves, "reference signed waves", 0)
    close(accepted["candidate_wavenumber_rad_m"], np.asarray(target_waves)[order], "candidate signed waves", 0)
    return order, np.minimum(overlaps, 1), phases


def parse_response_tables(summary, total_rows, coefficient_rows, modes):
    """Independently join all 50 coefficients and derive reverse-k conversion."""
    roots, reverse = [m["root"] for m in modes], [m["reverse_root"] for m in modes]
    require(len(roots) == len(reverse) == 5 and len(set(roots + reverse)) == 10
            and summary["frequency_hz"] == FREQUENCY_HZ and summary["incident_side"] == "left"
            and summary["incident_mode_indices"] == roots and summary["num_incident_channels"] == 5
            and summary["power_balance_passed"] is True, "Accepted five-channel summary/root binding differs.")
    require(len(total_rows) == 5 and len(coefficient_rows) == 50, "All five totals and 50 coefficients are required.")
    blocks = {kind: np.zeros((5, 5), dtype=complex) for kind in ("reflection", "transmission")}
    power_blocks = {kind: np.zeros((5, 5)) for kind in blocks}
    seen, coefficients = set(), []
    for row in coefficient_rows:
        kind, inc, out = row["kind"], int(row["incident_port_index"]), int(row["outgoing_port_index"])
        require(kind in blocks and inc in roots, "Invalid coefficient kind or incident root.")
        outgoing = reverse if kind == "reflection" else roots
        require(out in outgoing and int(row["incident_mode_order"]) == roots.index(inc) + 1,
                "Invalid signed outgoing root or incident order.")
        key = kind, inc, out
        require(key not in seen, "Repeated coefficient root pair.")
        seen.add(key)
        re, im, power = (float(row[k]) for k in ("amplitude_real", "amplitude_imag", "power_fraction"))
        require(np.isfinite([re, im, power]).all() and power >= 0 and abs(power - abs(complex(re, im))**2) <= TOLERANCE,
                "Coefficient amplitude/power is nonfinite or inconsistent.")
        blocks[kind][outgoing.index(out), roots.index(inc)] = complex(re, im)
        power_blocks[kind][outgoing.index(out), roots.index(inc)] = power
        coefficients.append({"kind": kind, "incident_root": inc, "outgoing_root": out,
                             "amplitude_real": re, "amplitude_imag": im, "power_fraction": power})
    rows = []
    total_by_order = {int(row["incident_mode_order"]): row for row in total_rows}
    require(set(total_by_order) == set(range(1, 6)), "Total-row orders are missing or repeated.")
    for index, mode in enumerate(modes):
        source = total_by_order[index + 1]
        reflection, transmission = power_blocks["reflection"][:, index], power_blocks["transmission"][:, index]
        R, T = float(source["reflection"]), float(source["transmission"])
        require(int(source["incident_mode_order"]) == index + 1 and float(source["frequency_GHz"]) == 1,
                "Accepted total-row order/frequency differs.")
        require(np.isfinite([R, T]).all() and min(R, T) >= 0 and abs(R + T - 1) < .01,
                "The accepted total powers are invalid.")
        close([R, T], [reflection.sum(), transmission.sum()], "total coefficient powers")
        offdiagonal = np.arange(5) != index
        rows.append(dict(mode, R=R, T=T, converted_R=float(reflection[offdiagonal].sum()),
                         converted_T=float(transmission[offdiagonal].sum())))
    return {"rows": rows, "coefficients": coefficients, "blocks": blocks, "power_blocks": power_blocks}


def compare_tree(actual, expected, label):
    """Verify all derived fields while preserving the numerical gate."""
    if isinstance(expected, dict):
        require(isinstance(actual, dict), f"Independent {label} structure differs.")
        for key, value in expected.items():
            require(key in actual, f"Independent {label}.{key} is missing.")
            compare_tree(actual[key], value, f"{label}.{key}")
    elif isinstance(expected, list):
        require(isinstance(actual, list) and len(actual) == len(expected), f"Independent {label} list differs.")
        for index, (a, b) in enumerate(zip(actual, expected)):
            compare_tree(a, b, f"{label}[{index}]")
    elif isinstance(expected, (float, complex, np.floating)):
        close(actual, expected, label)
    else:
        require(actual == expected, f"Independent {label} differs.")


def read_csv(path):
    with path.open(newline="", encoding="utf-8") as stream:
        return list(csv.DictReader(stream))


def verify_diagnosis(document, root=ROOT):
    """Independently verify quadrature, signed gauges and complete conversion joins."""
    require(document["frequency_hz"] == FREQUENCY_HZ and document["numerical_tolerance"] == TOLERANCE,
            "The fixed frequency or numerical agreement gate differs.")
    for key in ("body_arrays_loaded", "body_fields_loaded", "matrix_assembly_performed", "safe_eigensolve_performed",
                "body_solve_performed", "new_frequency_point_computed", "phase_adjustment_applied",
                "spatial_convergence_established", "frequency_convergence_established",
                "physical_branch_classification_established", "conversion_mechanism_established",
                "cross_frequency_branch_tracking_performed"):
        require(document["scope"][key] is False, "The diagnosis overstates its saved-section scope.")
    require(quadrature.sha256(Path(quadrature.__file__)) == QUADRATURE_SHA256, "Frozen quadrature source identity differs.")
    sources = {Path(row["path"]).resolve(): row["sha256"] for row in document["sources"]}
    diagnosis_path = ROOT / "tools/diagnose_conversion_families_AP_pnc.py"
    require(sources.get(diagnosis_path.resolve()) == quadrature.sha256(diagnosis_path), "Diagnosis source identity differs.")
    roles = verified_inputs(document, root)
    comparison, records = bind_scientific_roles(roles)
    sections, forward, backward, waves, reverse, maximum = {}, {}, {}, {}, {}, 0.0
    for name in ("z4", "z6"):
        forward[name], backward[name], waves[name], powers, reverse[name] = load_section(roles[f"{name}_section"], name)
        expected = quadrature.integrate_section(**forward[name])
        require(np.isclose(expected["area_m2"], 2e-13, rtol=GEOMETRY_RELATIVE_TOLERANCE, atol=0)
                and np.isclose(expected["mass_per_length_kg_m"], 1.0646e-9, rtol=GEOMETRY_RELATIVE_TOLERANCE, atol=0),
                "The nominal section area or mass per length differs.")
        for i, mode in enumerate(expected["modes"]):
            mode.update(root=FIXED_ROOTS[name][i], reverse_root=reverse[name][i],
                        wavenumber_rad_m=float(waves[name][i]), signed_power_w=float(powers[i]))
        actual = document["sections"][name]
        maximum = max(maximum, quadrature.compare_metrics(actual, expected))
        require(actual["mode_order"] == FIXED_ROOTS[name], "The section mode order differs.")
        for a, b in zip(actual["modes"], expected["modes"]):
            for key in ("root", "reverse_root", "wavenumber_rad_m", "signed_power_w"):
                require(a[key] == b[key], f"The section {key} differs.")
        sections[name] = expected
    matches = {}
    for kind, arguments, roots, signs in (("incoming", forward, FIXED_ROOTS, 1), ("reflection", backward, reverse, -1)):
        matches[kind] = signed_match(arguments["z4"], arguments["z6"], comparison["matched_channels"][kind],
                                    roots["z4"], roots["z6"], signs * waves["z4"], signs * waves["z6"])
        published = document["signed_matches"][kind]
        accepted = comparison["matched_channels"][kind]
        for key in ("reference_roots", "candidate_roots", "mass_overlaps", "reference_wavenumber_rad_m", "candidate_wavenumber_rad_m", "relative_wavenumber_change"):
            compare_tree(published[key], accepted[key], f"published signed {kind} {key}")
        close(np.exp(1j * np.asarray(published["candidate_gauge_phase_rad"])),
              np.exp(1j * np.asarray(accepted["candidate_gauge_phase_rad"])), f"published signed {kind} gauge")
    compare_tree(comparison["matched_channels"]["transmission"], comparison["matched_channels"]["incoming"], "transmitted mode correspondence")
    responses = {}
    for point, name in (("h80_z4", "z4"), ("h80_z6", "z6")):
        summary = json.loads(roles[f"{point}_summary"].read_bytes())
        expected = parse_response_tables(summary, read_csv(roles[f"{point}_scattering"]),
                                         read_csv(roles[f"{point}_coefficients"]), sections[name]["modes"])
        actual = document["responses"][point]
        require(actual["section"] == name and actual["run_hash"] == records[point]["run_hash"], "The response section/run join differs.")
        compare_tree(actual["rows"], expected["rows"], f"{point} response rows")
        # Compare by signed root identity; CSV storage order is not physical identity.
        key = lambda row: (row["kind"], row["incident_root"], row["outgoing_root"])
        require(len(actual["coefficients"]) == 50, "Published coefficients are incomplete.")
        compare_tree(sorted(actual["coefficients"], key=key), sorted(expected["coefficients"], key=key), f"{point} coefficients")
        responses[point] = expected
    preserved = document["preserved_scientific_status"]
    for key in ("input_power_checks_passed", "power_change_limit", "complex_change_limit", "maximum_power_change", "maximum_complex_change", "power_passed", "complex_passed", "mesh_pair_passed"):
        compare_tree(preserved[key], comparison[key], f"preserved {key}")
    ledger = json.loads(roles["device_thickness_verification"].read_bytes())
    compare_tree(preserved["maximum_same_channel_transmission_phase_change_rad"],
                 ledger["h80_z4_z6_mesh_pair"]["maximum_same_channel_transmission_phase_change_rad"], "preserved phase increment")
    close(preserved["reference_planes_m"], [0, 1.665e-5], "original reference planes", 0)
    require(document["relative_reporting_floor"] == RELATIVE_REPORTING_FLOOR, "The reporting denominator floor differs.")
    return finish_verification(document, sections, responses, comparison, matches, maximum)


def finish_verification(document, sections, responses, comparison, matches, maximum):
    """Verify finite increments and pair selection after independent assembly."""
    order, overlaps, incoming_phase = matches["incoming"]
    outgoing_order, _, reflected_phase = matches["reflection"]
    a, b = responses["h80_z4"], responses["h80_z6"]
    require(len(document["matched_rows"]) == 5, "All five matched metric/response increments are required.")
    for i, row in enumerate(document["matched_rows"]):
        first, second = a["rows"][i], b["rows"][order[i]]
        compare_tree(row, {"z4_root": first["root"], "z6_root": second["root"], "mass_overlap": float(overlaps[i])}, "matched row")
        for key in METRIC_KEYS + ("R", "T", "converted_R", "converted_T"):
            delta = np.asarray(second[key]) - first[key]
            close(row[f"z6_minus_z4_{key}"], delta, f"matched {key} increment")
        for key in ("R", "T", "converted_R", "converted_T"):
            close(row[f"z6_minus_z4_relative_{key}"], (second[key] - first[key]) / max(abs(first[key]), RELATIVE_REPORTING_FLOOR),
                  f"relative {key} increment")
    changes, max_power, max_complex = [], 0.0, 0.0
    for kind, out_order, out_phase in (("reflection", outgoing_order, reflected_phase), ("transmission", order, incoming_phase)):
        first, raw = a["blocks"][kind], b["blocks"][kind][np.ix_(out_order, order)]
        aligned = out_phase[:, None] * raw / incoming_phase[None, :]
        saved = comparison["blocks"][kind]
        for field, value in (("reference", first), ("candidate_aligned", aligned)):
            encoded = np.asarray(saved[field])
            close(encoded[..., 0] + 1j * encoded[..., 1], value, f"accepted {kind} {field}")
        max_power = max(max_power, float(np.max(abs(np.sum(abs(aligned)**2, axis=0) - np.sum(abs(first)**2, axis=0)))))
        max_complex = max(max_complex, float(np.max(abs(aligned - first))))
        for i in range(5):
            for j in range(5):
                a_out = (FIXED_ROOTS["z4"][i] if kind == "transmission" else sections["z4"]["modes"][i]["reverse_root"])
                b_out = (FIXED_ROOTS["z6"][out_order[i]] if kind == "transmission" else sections["z6"]["modes"][out_order[i]]["reverse_root"])
                pair = lambda value: [float(value.real), float(value.imag)]
                changes.append({"kind": kind, "z4_incident_root": FIXED_ROOTS["z4"][j],
                    "z6_incident_root": FIXED_ROOTS["z6"][order[j]], "z4_outgoing_root": a_out, "z6_outgoing_root": b_out,
                    "z4_amplitude": pair(first[i, j]), "z6_raw_amplitude": pair(raw[i, j]),
                    "z6_aligned_amplitude": pair(aligned[i, j]), "aligned_complex_delta": pair(aligned[i, j] - first[i, j]),
                    "absolute_complex_change": float(abs(aligned[i, j] - first[i, j])), "z4_power": float(a["power_blocks"][kind][i, j]),
                    "z6_power": float(b["power_blocks"][kind][out_order[i], order[j]]),
                    "z6_minus_z4_power": float(b["power_blocks"][kind][out_order[i], order[j]] - a["power_blocks"][kind][i, j])})
    key = lambda row: (row["kind"], row["z4_incident_root"], row["z4_outgoing_root"])
    require(len(document["coefficient_changes"]) == 50, "All 50 aligned complex increments are required.")
    compare_tree(sorted(document["coefficient_changes"], key=key), sorted(changes, key=key), "aligned coefficient changes")
    close(max_power, comparison["maximum_power_change"], "retained maximum power increment")
    close(max_complex, comparison["maximum_complex_change"], "retained maximum complex increment")
    require(comparison["complex_passed"] is False and comparison["mesh_pair_passed"] is False,
            "The accepted 1 GHz spatial comparison failure must remain failed.")
    verify_dominant_pair(document, responses)
    return {"author": "Athul Prem", "file_tag": "_AP_pnc", "status": "independently_verified_saved_conversion_families",
        "method": "Frozen explicit P2 3x3 Gauss-Legendre/Duffy integration without saved mass or assemblers; independent signed CSV/coefficient joins.",
        "implementation_agreement_tolerance": TOLERANCE, "area_and_mass_relative_tolerance": GEOMETRY_RELATIVE_TOLERANCE,
        "maximum_dimensionless_metric_difference": maximum, "section_rows_checked": 10, "response_rows_checked": 10,
        "complex_coefficients_checked_per_response": 50, "aligned_complex_increments_checked": 50,
        "signed_matches_checked_per_direction": 5, "signed_gauge_phases_checked_modulo_2pi": True,
        "maximum_power_change": max_power, "maximum_complex_change": max_complex,
        "spatial_convergence_established": False, "frequency_convergence_established": False,
        "body_arrays_loaded": False, "saved_mass_loaded": False, "assembly_or_eigensolve_performed": False,
        "scope": "Finite saved-mode conversion-family association only; no mechanism, continuous branch or uncertainty claim."}


def verify_dominant_pair(document, responses):
    """Replay all ten unordered reflection-pair ranks and both coefficients."""
    pairs = {}
    for point, response in responses.items():
        modes, power = response["rows"], response["power_blocks"]["reflection"]
        table = {(row["incident_root"], row["outgoing_root"]): row for row in response["coefficients"] if row["kind"] == "reflection"}
        ranked = []
        for i in range(5):
            for j in range(i + 1, 5):
                roots = [modes[i]["root"], modes[j]["root"]]
                reverse = [modes[i]["reverse_root"], modes[j]["reverse_root"]]
                ranked.append({"forward_roots": roots, "reverse_roots": reverse,
                    "two_direction_power_sum": float(power[j, i] + power[i, j]),
                    "directions": [table[roots[0], reverse[1]], table[roots[1], reverse[0]]]})
        ranked.sort(key=lambda pair: pair["two_direction_power_sum"], reverse=True)
        zero = ranked[0]["two_direction_power_sum"] == 0
        expected = {"selection": "Largest sum of the two off-diagonal reflected powers; descriptive ranking only.",
                    "all_offdiagonal_reflection_zero": zero, "selected_pair": None if zero else ranked[0], "ranked_pairs": ranked}
        compare_tree(document["dominant_conversion_pair"][point], expected, f"{point} dominant converted pair")
        pairs[point] = expected["selected_pair"]
    return pairs


def verify_publication_csv(document, directory):
    name = "conversion_families_AP_pnc.csv"
    path = directory / name
    require(quadrature.sha256(path) == document["output_sha256"][name], "The publication CSV checksum differs.")
    rows = read_csv(path)
    expected = [(point, response["section"], row) for point, response in document["responses"].items() for row in response["rows"]]
    require(len(rows) == len(expected) == 10, "The publication must retain all ten rows.")
    for actual, (point, section, mode) in zip(rows, expected):
        require(actual["point"] == point and actual["section"] == section
                and int(actual["root"]) == mode["root"] and int(actual["reverse_root"]) == mode["reverse_root"]
                and actual["descriptive_largest_template"] == mode["descriptive_largest_template"], "The publication CSV root join differs.")
        values = {name: mode[name] for name in ("wavenumber_rad_m", "signed_power_w", "R", "T", "converted_R", "converted_T", "template_projection_fraction", "template_remainder_fraction")}
        values.update(zip(("kinetic_x", "kinetic_y", "kinetic_z"), mode["component_fractions_xyz"]))
        values.update(zip(("score_tx", "score_ty", "score_tz", "score_rx"), mode["template_scores_tx_ty_tz_rx"]))
        for name, value in values.items():
            close(float(actual[name]), value, f"publication CSV {name}")
    return len(rows)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--diagnosis", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    require(not args.output.exists() and args.output.resolve() != args.diagnosis.resolve(), "Independent verification requires a fresh output file.")
    document = json.loads(args.diagnosis.read_bytes())
    result = verify_diagnosis(document)
    result["publication_csv_rows_checked"] = verify_publication_csv(document, args.diagnosis.parent)
    result["diagnosis"] = {"path": str(args.diagnosis.resolve()), "bytes": args.diagnosis.stat().st_size, "sha256": quadrature.sha256(args.diagnosis)}
    result["verifier_sha256"] = quadrature.sha256(Path(__file__))
    result["frozen_quadrature_sha256"] = QUADRATURE_SHA256
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("x", encoding="utf-8") as stream:
        json.dump(result, stream, indent=2, allow_nan=False)
        stream.write("\n")
    print(json.dumps({"status": result["status"], "output": str(args.output.resolve()), "maximum_dimensionless_metric_difference": result["maximum_dimensionless_metric_difference"]}))


if __name__ == "__main__":
    main()
