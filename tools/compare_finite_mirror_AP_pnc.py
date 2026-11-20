"""Compare the saved eight-period mirror responses. Athul Prem | _AP_pnc."""

from __future__ import annotations

import argparse
from math import isclose, isfinite
from pathlib import Path
from statistics import mean
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT), str(ROOT / "src")]

from phononic.resumable_scattering_AP_pnc import AUTHOR, _link, _read, _verified
from phononic.workflow_AP_pnc import digest, write_json
from tools import run_mirror_cell_phase_AP_pnc as cell

OUTPUT = ROOT / "verification/finite_mirror_1ghz_AP_pnc.json"
CELL_COUNT = 8
POWER_LIMIT = 0.01
COMPLEX_LIMIT = 0.01
RESPONSE = "finite_8_AP_pnc.json"
CONTROL = "finite_1_AP_pnc.json"
POWERS = ("transmission", "reflection")
ROOTS = ("native_incident_root_ids", "native_transmission_root_ids", "native_reflection_root_ids")


def _source_pin(path):
    if path.is_relative_to(ROOT):
        return _link(path, ROOT)
    return {**_link(path, path.parent), "path": str(path)}


def _require(condition, message):
    if not condition:
        raise ValueError(message)


def _checked_case(folder, name, periods):
    path = folder / name
    receipt = _read(path)
    inputs = receipt["inputs"]
    response = receipt["result"]
    _require(digest(inputs) == receipt["inputs_sha256"] and digest(response) == receipt["result_sha256"],
             f"Saved inputs or numerical response changed in {path}.")
    source_cell = _read(_verified(inputs["cell_receipt"], folder))
    cell_inputs = source_cell["inputs"]
    _require(source_cell["status"] == "accepted"
             and source_cell["inputs_sha256"] == inputs["cell_inputs_sha256"] == digest(cell_inputs)
             and source_cell["trace_arrays"] == inputs["cell_trace_arrays"]
             and receipt["cell_geometry"] == cell_inputs["geometry"]
             and receipt["cell_mesh_controls"] == cell_inputs["mesh"], f"Saved mirror cell differs in {path}.")
    _verified(source_cell["trace_arrays"], folder)
    period = cell_inputs["geometry"]["period_nm"] * cell.NM
    _require(inputs["periods"] == response["cell_count"] == periods
             and response["frequency_hz"] == cell_inputs["frequency_hz"] == cell.FREQUENCY_HZ
             and isclose(response["period_m"], period, rel_tol=1e-14, abs_tol=0)
             and response["reference_planes_m"] == [0.0, periods * response["period_m"]],
             f"Frequency, period count, or physical end planes differ in {path}.")

    trace = response["native_port"]
    ordered = [response[key] for key in ROOTS]
    count = len(ordered[0])
    _require(count > 0 and ordered[0] == ordered[1] and all(len(set(ids)) == len(ids) == count for ids in ordered)
             and trace["trace_dof_count"] == trace["outgoing_columns_per_end"]
             and trace["trace_dof_count"] == trace["evanescent_columns_per_end"] + count
             and len(response["incident_powers_w"]) == count
             and all(abs(value - 1.0) <= 1e-7 for value in response["incident_powers_w"]),
             f"Saved propagating root ordering or 1 W incident normalization differs in {path}.")
    _require(all(len(response[key]) == count and all(isfinite(value) and value >= 0 for value in response[key])
                 for key in POWERS), f"Saved channel powers are invalid in {path}.")
    return {"source": _source_pin(path), "receipt": receipt, "cell": source_cell}


def _accepted(case):
    response = case["receipt"]["result"]
    residual, power = (max(response[key].values()) for key in ("relative_residuals", "power_errors"))
    return (response["status"] == "accepted" and residual < response["relative_residual_limit"]
            and power < response["power_error_limit"])


def _case_summary(case):
    receipt = case["receipt"]
    response = receipt["result"]
    accepted = _accepted(case)
    return {
        "source": case["source"], "source_cell_inputs_sha256": receipt["inputs"]["cell_inputs_sha256"],
        "target_nm": receipt["cell_mesh_controls"]["central_target_nm"],
        "actual_central_max_edge_nm": case["cell"]["mesh"]["max_central_edge_nm"],
        "cell_dofs": case["cell"]["mesh"]["quadratic_vector_dofs"],
        "status": "accepted" if accepted else "unresolved",
        "maximum_relative_numerical_residual": max(response["relative_residuals"].values()),
        "maximum_power_flux_error": max(response["power_errors"].values()),
        "native_port_operator_sha256": response["native_port"]["operator_sha256"],
        "native_root_order": {key: response[key] for key in ROOTS},
        "mean_power": {key: mean(response[key]) for key in POWERS} if accepted else None,
        "drives": [{"incident_native_root_id": root, **{key: response[key][index] for key in POWERS}}
                   for index, root in enumerate(response[ROOTS[0]])] if accepted else [],
    }


def _same_native_basis(reference, candidate):
    a, b = (case["receipt"] for case in (reference, candidate))
    reason = {
        "identical_assembled_section_operators": a["result"]["native_port"]["operator_sha256"]
            == b["result"]["native_port"]["operator_sha256"],
        "identical_translated_section_triangles": reference["cell"]["mesh"]["translated_end_triangles_sha256"]
            == candidate["cell"]["mesh"]["translated_end_triangles_sha256"],
        "identical_signed_native_root_order": all(a["result"][key] == b["result"][key] for key in ROOTS),
        "identical_environment": all(a["inputs"][key] == b["inputs"][key] for key in ("numerical_identity", "threads")),
        "identical_incident_flux_normalizations": a["result"]["incident_powers_w"] == b["result"]["incident_powers_w"],
    }
    return all(reason.values()), reason


def _complex_block(response, key, count):
    raw = response[f"{key}_amplitudes"]
    re, im = raw["real"], raw["imag"]
    _require(len(re) == len(im) == count and all(len(row) == count for row in re + im),
             f"Saved native {key} complex matrix shape differs.")
    return [complex(x, y) for row_x, row_y in zip(re, im, strict=True)
            for x, y in zip(row_x, row_y, strict=True)]


def _adjacent(first, second):
    reference, candidate = (_case_summary(case) for case in (first, second))
    same_basis, evidence = _same_native_basis(first, second)
    compared = {"reference_target_nm": reference["target_nm"], "candidate_target_nm": candidate["target_nm"],
                "native_basis_status": "matched" if same_basis else "unresolved", "native_basis_evidence": evidence}
    if reference["status"] != "accepted" or candidate["status"] != "accepted":
        return {**compared, "status": "unresolved_numerical_point"}
    compared["mean_power_change_percentage_points"] = {
        key: 100 * (candidate["mean_power"][key] - reference["mean_power"][key]) for key in POWERS}
    if not same_basis:
        return {**compared, "status": "mean_only"}

    a, b = (case["receipt"]["result"] for case in (first, second))
    count = len(a[ROOTS[0]])
    drives = [{"incident_native_root_id": root,
               **{f"{key}_change_percentage_points": 100 * (b[key][index] - a[key][index]) for key in POWERS}}
              for index, root in enumerate(a[ROOTS[0]])]
    power_max = {key: max(abs(row[f"{key}_change_percentage_points"]) for row in drives) for key in POWERS}
    pairs = {key: list(zip(_complex_block(a, key, count), _complex_block(b, key, count), strict=True))
             for key in POWERS}
    amplitude_max = {key: max(abs(x - y) for x, y in entries) for key, entries in pairs.items()}
    magnitude_max = {key: max(abs(abs(x) - abs(y)) for x, y in entries) for key, entries in pairs.items()}
    diagonal_changes = [abs(x - y) for x, y in pairs["transmission"][::count + 1]]
    diagonal_index = max(range(count), key=diagonal_changes.__getitem__)
    diagonal_max = diagonal_changes[diagonal_index]
    return {**compared, "status": "matched", "drives": drives,
            "max_abs_power_change_percentage_points": power_max,
            "passes_per_drive_power_limit": max(power_max.values()) <= 100 * POWER_LIMIT,
            "phase_gauge_invariant_amplitude_bounds": {
                "max_abs_diagonal_transmission_change": diagonal_max,
                "diagonal_transmission_native_root_id": a[ROOTS[0]][diagonal_index],
                "max_abs_coefficient_magnitude_change_by_block": magnitude_max,
                "max_abs_coefficient_magnitude_change": max(magnitude_max.values()),
                "establishes_complex_amplitude_limit_exceeded": max(diagonal_max, *magnitude_max.values()) > COMPLEX_LIMIT,
            },
            "conditional_native_gauge_comparison": {
                "max_abs_elementwise_complex_amplitude_change": amplitude_max,
                "passes_mesh_complex_amplitude_limit": max(amplitude_max.values()) <= COMPLEX_LIMIT,
            }}


def compare(study):
    study = Path(study).resolve()
    cases = {thickness: [_checked_case(cell.case_directory(study, thickness, target), RESPONSE, CELL_COUNT)
                         for target in cell.TARGETS_NM] for thickness in cell.THICKNESSES_NM}
    common_planes = cases[100][0]["receipt"]["result"]["reference_planes_m"]
    common_geometry = cases[100][0]["receipt"]["cell_geometry"]
    common_mesh = cases[100][0]["receipt"]["cell_mesh_controls"]
    materials = cases[100][0]["cell"]["inputs"]["material"]
    for thickness, rows in cases.items():
        for target, case in zip(cell.TARGETS_NM, rows, strict=True):
            saved = case["receipt"]
            _require(saved["cell_geometry"] == {**common_geometry, "thickness_nm": thickness}
                     and saved["cell_mesh_controls"] == {**common_mesh, "central_target_nm": target}
                     and saved["result"]["reference_planes_m"] == common_planes
                     and case["cell"]["inputs"]["material"] == materials,
                     f"Geometry, mesh controls, material, or original planes differ in {case['source']['path']}.")

    by_thickness = {str(thickness): {"cases": [_case_summary(case) for case in rows],
                                    "adjacent_mesh_changes": [_adjacent(a, b) for a, b in zip(rows, rows[1:])]}
                    for thickness, rows in cases.items()}
    contrasts = []
    for thin, thick in zip(by_thickness["100"]["cases"], by_thickness["200"]["cases"], strict=True):
        if thin["mean_power"] is None or thick["mean_power"] is None:
            delta = None
        else:
            delta = {key: 100 * (thin["mean_power"][key] - thick["mean_power"][key]) for key in POWERS}
        contrasts.append({"target_nm": thin["target_nm"], "thin_minus_thick_mean_percentage_points": delta})
    increments = [{"reference_target_nm": a["target_nm"], "candidate_target_nm": b["target_nm"],
                   "contrast_change_percentage_points": {
                       key: b["thin_minus_thick_mean_percentage_points"][key]
                            - a["thin_minus_thick_mean_percentage_points"][key] for key in POWERS}
                   if all(case["thin_minus_thick_mean_percentage_points"] is not None for case in (a, b)) else None}
                  for a, b in zip(contrasts, contrasts[1:])]

    control = _checked_case(cell.case_directory(study, 100, 225), CONTROL, 1)
    checked = control["receipt"]["result"]
    native = checked["native_countercheck"]
    _require(_accepted(control) and native["status"] == "accepted", "The one-cell native countercheck is unresolved.")
    return {
        **AUTHOR, "schema_version": 1, "frequency_hz": cell.FREQUENCY_HZ, "cell_count": CELL_COUNT,
        "geometry": {key: value for key, value in common_geometry.items() if key not in ("thickness_nm", "reference_planes_nm")},
        "original_reference_planes_m": common_planes, "material": materials,
        "section": {"width_nm": common_geometry["width_nm"], "thicknesses_nm": list(cell.THICKNESSES_NM),
                    "element_order": common_mesh["element_order"], "transverse_cells": common_mesh["transverse_cells"],
                    "thickness_layers": common_mesh["thickness_layers"],
                    "native_outgoing_columns_per_end": checked["native_port"]["outgoing_columns_per_end"],
                    "evanescent_columns_per_end": checked["native_port"]["evanescent_columns_per_end"]},
        "one_cell_native_countercheck": {"source": control["source"], "physical_thickness_nm": 100, "target_nm": 225,
                                         "original_reference_planes_m": checked["reference_planes_m"], **native},
        "power_normalization": "Per drive, unit incident flux. Mean is the equal-power incoherent average over all propagating lead inputs and is invariant under unitary basis changes.",
        "mode_match": "A within-thickness match requires identical assembled lead operators, translated section triangles, source and numerical environment, ordered signed SAFE roots, and incident powers.",
        "phase_gauge_invariant_bounds": "For diagonal transmission the same forward native mode is used at both ends, so its arbitrary phase cancels at the original planes. Changes of coefficient magnitude are unchanged by per-mode phase choices. Either maximum lower-bounds the largest complex change for the matched mode basis.",
        "conditional_complex_comparison": "Raw elementwise complex maxima assume the deterministic native gauge was reproduced. The finite receipts retain no per-vector phase hashes.",
        "mesh_power_limit_percentage_points": 100 * POWER_LIMIT, "mesh_complex_amplitude_limit": COMPLEX_LIMIT,
        "point_relative_numerical_residual_limit": checked["relative_residual_limit"],
        "point_power_flux_error_limit": checked["power_error_limit"],
        "by_thickness_nm": by_thickness, "mean_thickness_contrast": contrasts, "adjacent_contrast_changes": increments,
        "scope": "Eight identical cells cut from the repeated outer-mirror region, joined to uniform leads at their physical end planes. Fixed end strips, transverse trace, hole faceting and thickness-layer count bound this comparison. It samples one frequency and three in-plane refinements of each thickness. Full-device response and spatial or frequency convergence remain unestablished.",
        "comparison_source": _source_pin(Path(__file__).resolve()),
    }


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--study-root", type=Path, default=cell.STUDY)
    parser.add_argument("--output", type=Path, default=OUTPUT)
    args = parser.parse_args(argv)
    write_json(args.output, compare(args.study_root))
    print(f"Saved eight-period mirror comparison to {args.output.resolve()}")


if __name__ == "__main__":
    main()
