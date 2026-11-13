#!/usr/bin/env python3
"""Check power and phase in a matched 3D straight guide. Author: Athul Prem | _AP_pnc."""

from __future__ import annotations

import argparse
import csv
from dataclasses import dataclass
import hashlib
import json
from pathlib import Path
import sys

import numpy as np
import psutil
from scipy.optimize import linear_sum_assignment
from skfem import Basis

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from phononic.elasticity_AP_pnc import assemble_elasticity
from phononic.geometry_AP_pnc import BeamGeometry, make_straight_mesh
from phononic.material_AP_pnc import CubicMaterial
from phononic.port_modes_AP_pnc import PortModes, assemble_port, solve_port_modes
from phononic.scattering_AP_pnc import ScatteringResult, solve_scattering
from phononic.workflow_AP_pnc import measured_resources, source_identity, write_json

FREQUENCY_HZ = 1e9
WIDTH_M = 600e-9
THICKNESS_M = 200e-9
MAX_DOFS = 6237
REFINEMENT_MAX_DOFS = 9801
REFINEMENT_MAX_PORT_DOFS = 297
ACCEPTANCE_MAX_DOFS = 14553
ACCEPTANCE_REFERENCE_PORT_DOFS = 429
POWER_LIMIT = 0.01
COMPLEX_AMPLITUDE_LIMIT = 0.01
WAVENUMBER_RELATIVE_LIMIT = 0.01
ACCUMULATED_PHASE_LIMIT_RAD = 0.01
RESIDUAL_LIMIT = 1e-8
MATCH_MIN_OVERLAP = 0.90
NEAR_DEGENERATE_GAP = 0.001
SIDES = ("left", "right")

# A16 is also the coarsest transverse mesh. Reuse its recorded result for both checks.
CASES = {
    "A4": (1e-6, (4, 2, 1)),
    "A8": (1e-6, (8, 2, 1)),
    "A16": (1e-6, (16, 2, 1)),
    "B_middle": (1e-6, (16, 3, 2)),
    "B_fine": (1e-6, (16, 4, 3)),
    "C_long": (1.5e-6, (24, 3, 2)),
}
COMPARISONS = (
    ("axial_4_to_8", "A4", "A8"),
    ("axial_8_to_16", "A8", "A16"),
    ("transverse_coarse_to_middle", "A16", "B_middle"),
    ("transverse_middle_to_fine", "B_middle", "B_fine"),
    ("length_middle_mesh", "B_middle", "C_long"),
)
REFINEMENT_CASES = {
    "axial_refined": (1e-6, (24, 4, 3)),
    "transverse_refined": (1e-6, (16, 5, 4)),
    "length_refined": (1.5e-6, (24, 4, 3)),
}
REFINEMENT_COMPARISONS = (
    ("axial_refinement", "B_fine", "axial_refined"),
    ("transverse_refinement", "B_fine", "transverse_refined"),
    ("length_refinement", "B_fine", "length_refined"),
)
BASELINE_STEM = "straight_scattering_validation_AP_pnc"
REFINEMENT_STEM = "straight_scattering_refinement_AP_pnc"
ACCEPTANCE_LONG = (1.5e-6, (24, 5, 4))
ACCEPTANCE_REFERENCE_CELLS = (1, 6, 5)
NUMERICAL_SOURCES = (
    "__init__.py", "elasticity_AP_pnc.py", "geometry_AP_pnc.py", "lead_modes_AP_pnc.py",
    "material_AP_pnc.py", "port_modes_AP_pnc.py", "scattering_AP_pnc.py",
)


@dataclass(frozen=True)
class SectionModes:
    basis: Basis
    mass: np.ndarray
    displacements: np.ndarray


def phasor(value: complex) -> dict:
    return {"re": float(value.real), "im": float(value.imag)}


def as_complex(value: dict) -> complex:
    return complex(value["re"], value["im"])


def port_overlap(coarse: SectionModes, coarse_ids: np.ndarray, fine: SectionModes, fine_ids: np.ndarray) -> np.ndarray:
    coarse_basis, fine_basis = coarse.basis, fine.basis
    components = np.asarray(fine_basis.split_indices())
    points = fine_basis.doflocs[:, components[0]]
    if not np.array_equal(fine_basis.doflocs[:, components], np.broadcast_to(points[:, None], (2, 3, points.shape[1]))):
        raise ValueError("Port displacement components must share interpolation nodes.")
    # Avoid point-location roundoff on exact rectangular boundaries.
    lower, upper = coarse_basis.mesh.p.min(axis=1), coarse_basis.mesh.p.max(axis=1)
    inset = 32 * np.finfo(float).eps * (upper - lower)
    points = np.clip(points, (lower + inset)[:, None], (upper - inset)[:, None])
    scalar_basis = Basis(coarse_basis.mesh, coarse_basis.elem.elem)
    coarse_components = np.asarray(coarse_basis.split_indices())
    if not np.array_equal(scalar_basis.doflocs, coarse_basis.doflocs[:, coarse_components[0]]):
        raise ValueError("Scalar and vector port interpolation nodes do not align.")
    # The port has three displacement components on a two-dimensional section.
    # Interpolate each component with the underlying scalar triangular basis.
    probes = scalar_basis.probes(points)
    transferred = np.empty((fine_basis.N, len(coarse_ids)), dtype=complex)
    for coarse_dofs, fine_dofs in zip(coarse_components, components):
        transferred[fine_dofs] = probes @ coarse.displacements[np.ix_(coarse_dofs, coarse_ids)]
    modes = fine.displacements[:, fine_ids]
    mass = fine.mass
    norms = np.sqrt(np.sum((transferred.conj() * (mass @ transferred)).real, axis=0))
    fine_norms = np.sqrt(np.sum((modes.conj() * (mass @ modes)).real, axis=0))
    overlap = np.abs(transferred.conj().T @ (mass @ modes)) / np.outer(norms, fine_norms)
    if not np.isfinite(overlap).all() or (overlap > 1 + 1e-12).any():
        raise ValueError("Nonphysical mass-normalized port mode overlap.")
    return np.minimum(overlap, 1)


def polarization(port_modes: PortModes, index: int) -> list[float]:
    u = port_modes.displacements[:, index]
    kinetic = (u.conj() * (port_modes.port.M @ u)).real
    return [float(kinetic[component].sum() / kinetic.sum()) for component in port_modes.port.basis.split_indices()]


def rows_for_side(case: str, length: float, modes: PortModes, result: ScatteringResult) -> list[dict]:
    distance = length if result.incident_side == "left" else -length
    outgoing_rows = {int(mode): index for index, mode in enumerate(result.transmission_mode_indices)}
    rows = []
    for column, index in enumerate(result.incident_mode_indices):
        # Both lead faces use the exact same SAFE basis, so an unchanged wave keeps its root ID.
        outgoing = outgoing_rows[int(index)]
        k = modes.wavenumbers_rad_m[index].real
        phase = np.exp(1j * k * distance)
        amplitude = result.transmission_amplitudes[outgoing, column]
        R, T = result.reflection[column], result.transmission[column]
        R_flux, T_flux = result.reflection_flux[column], result.transmission_flux[column]
        power_disagreement = max(abs(R - R_flux), abs(T - T_flux),
                                 abs(result.reflection_gram[column] - R_flux),
                                 abs(result.transmission_gram[column] - T_flux))
        rows.append({
            "case": case, "incident_side": result.incident_side, "mode_index": column + 1,
            "safe_root_index": int(index), "propagating_channels": len(result.incident_mode_indices),
            "wavenumber_rad_m": k, "kinetic_fractions_xyz": polarization(modes, index),
            "incident_power_W": result.incident_powers_w[column], "R": R, "T": T,
            "R_flux": R_flux, "T_flux": T_flux, "R_gram": result.reflection_gram[column],
            "T_gram": result.transmission_gram[column], "power_disagreement": power_disagreement,
            "power_balance": result.power_balance[column], "flux_power_balance": result.flux_power_balance[column],
            "net_outward_power_ratio": result.net_outward_power_ratio[column],
            "linear_residual": result.relative_residuals[column],
            "condition_left": result.port_basis_condition[0], "condition_right": result.port_basis_condition[1],
            "transmission_amplitude": phasor(amplitude), "expected_phase_factor": phasor(phase),
            "deembedded_amplitude": phasor(amplitude / phase),
            "phase_error_rad": np.angle(amplitude / phase),
            "complex_amplitude_error": abs(amplitude - phase),
            "transmitted_mode_conversion_power": T - result.transmission_modal_powers[outgoing, column],
            "matched_power_passed": (abs(R) < POWER_LIMIT and abs(T - 1) < POWER_LIMIT
                                     and abs(result.power_balance[column]) < POWER_LIMIT
                                     and power_disagreement < POWER_LIMIT),
        })
    return rows


def run_case(case: str, record: dict, material: CubicMaterial, max_dofs: int, max_port_dofs: int) -> SectionModes:
    length, cells = record["length_m"], record["cells"]
    with measured_resources() as resources:
        record["resources"] = resources
        geometry = BeamGeometry(length, WIDTH_M, THICKNESS_M)
        system = assemble_elasticity(make_straight_mesh(geometry, cells), material,
                                    element_order=2, max_dofs=max_dofs)
        record["displacement_dofs"] = system.basis.N
        modes = solve_port_modes(assemble_port(system, material, max_dofs=max_port_dofs), FREQUENCY_HZ)
        record.update({"port_dofs": modes.port.basis.N, "safe_max_relative_residual": modes.relative_residuals.max()})
        for side in SIDES:
            result = solve_scattering(system, modes, FREQUENCY_HZ, incident_side=side)
            record["channels"].extend(rows_for_side(case, length, modes, result))
            record["amplitude_matrices"][side] = {
                "reflection": [[phasor(value) for value in row] for row in result.reflection_amplitudes],
                "transmission": [[phasor(value) for value in row] for row in result.transmission_amplitudes],
                "reflection_root_indices": result.reflection_mode_indices.tolist(),
                "transmission_root_indices": result.transmission_mode_indices.tolist(),
                "incident_root_indices": result.incident_mode_indices.tolist(),
            }
            del result
    return SectionModes(modes.port.basis, modes.port.M, modes.displacements)


def near_degenerate(wavenumbers: np.ndarray) -> np.ndarray:
    distance = abs(wavenumbers[:, None] - wavenumbers)
    scale = np.maximum(abs(wavenumbers[:, None]), abs(wavenumbers))
    np.fill_diagonal(distance, np.inf)
    return np.any(distance < NEAR_DEGENERATE_GAP * scale, axis=1)


def compare(name: str, left: dict, right: dict, coarse: SectionModes, fine: SectionModes,
            compare_amplitudes: bool = True) -> dict:
    result = {"comparison": name, "reference_case": left["name"], "compared_case": right["name"], "sides": {}}
    for side in SIDES:
        first, second = [[row for row in case["channels"] if row["incident_side"] == side]
                         for case in (left, right)]
        i_first, i_second = [np.asarray([row["safe_root_index"] for row in rows]) for rows in (first, second)]
        overlaps = port_overlap(coarse, i_first, fine, i_second)
        k_first, k_second = [np.asarray([row["wavenumber_rad_m"] for row in rows]) for rows in (first, second)]
        degenerate_first, degenerate_second = near_degenerate(k_first), near_degenerate(k_second)
        pair_i, pair_j = linear_sum_assignment(-overlaps)
        matches = []
        for i, j in zip(pair_i, pair_j):
            tracked = not (degenerate_first[i] or degenerate_second[j]) and overlaps[i, j] >= MATCH_MIN_OVERLAP
            row_first, row_second = first[i], second[j]
            k_change = abs(k_first[i] / k_second[j] - 1)
            match = {
                "reference_mode_index": row_first["mode_index"], "compared_mode_index": row_second["mode_index"],
                "mass_weighted_overlap": overlaps[i, j], "relative_wavenumber_change": k_change,
                "reference_kinetic_fractions_xyz": row_first["kinetic_fractions_xyz"],
                "compared_kinetic_fractions_xyz": row_second["kinetic_fractions_xyz"],
                "individual_mode_tracking": "matched" if tracked else "unresolved",
                "passed": tracked,
            }
            if compare_amplitudes and tracked:
                complex_change = abs(as_complex(row_first["deembedded_amplitude"]) - as_complex(row_second["deembedded_amplitude"]))
                match.update({
                    "deembedded_complex_amplitude_change": complex_change,
                    "absolute_phase_error_reference_rad": abs(row_first["phase_error_rad"]),
                    "absolute_phase_error_compared_rad": abs(row_second["phase_error_rad"]),
                    "passed": complex_change < COMPLEX_AMPLITUDE_LIMIT,
                })
            matches.append(match)
        same_count = len(first) == len(second)
        result["sides"][side] = {"matching_channel_counts": same_count, "overlap_matrix": overlaps.tolist(),
                                 "matches": matches, "passed": same_count and all(match["passed"] for match in matches)}
    result["passed"] = all(item["passed"] for item in result["sides"].values())
    return result


def compare_with_common_phase(
    name: str, first: dict, second: dict, first_modes: SectionModes, second_modes: SectionModes,
) -> dict:
    """Compare transmission at one length without removing each mesh's own phase."""
    result = compare(name, first, second, first_modes, second_modes)
    same_length = first["length_m"] == second["length_m"]
    for side, record in result["sides"].items():
        first_rows, second_rows = [
            {row["mode_index"]: row for row in case["channels"] if row["incident_side"] == side}
            for case in (first, second)
        ]
        for match in record["matches"]:
            if match["individual_mode_tracking"] != "matched":
                continue
            a, b = first_rows[match["reference_mode_index"]], second_rows[match["compared_mode_index"]]
            actual_a = as_complex(a["transmission_amplitude"])
            actual_b = as_complex(b["transmission_amplitude"])
            # Use a signed direction and the same SAFE root for both lengths.
            distance_change = (second["length_m"] - first["length_m"]) * (1 if side == "left" else -1)
            predicted_change = np.exp(1j * b["wavenumber_rad_m"] * distance_change)
            amplitude_difference = abs(actual_b - actual_a * predicted_change)
            accumulated = abs(a["wavenumber_rad_m"] - b["wavenumber_rad_m"]) * max(first["length_m"], second["length_m"])
            match.update({
                "raw_equal_length_amplitude_change": amplitude_difference if same_length else None,
                "raw_length_transfer_error": amplitude_difference if not same_length else None,
                "accumulated_wavenumber_phase_change_rad": accumulated,
                "passed": (amplitude_difference < COMPLEX_AMPLITUDE_LIMIT
                           and match["relative_wavenumber_change"] < WAVENUMBER_RELATIVE_LIMIT
                           and accumulated < ACCUMULATED_PHASE_LIMIT_RAD),
            })
        record["passed"] = record["matching_channel_counts"] and all(match["passed"] for match in record["matches"])
    result["passed"] = all(record["passed"] for record in result["sides"].values())
    return result


def compare_to_phase_reference(
    phase_reference: dict, case: dict, reference_modes: SectionModes, case_modes: SectionModes,
) -> dict:
    """Compare raw transmission against the finest available transverse SAFE root."""
    result = compare("common_phase_reference", phase_reference, case, reference_modes, case_modes, compare_amplitudes=False)
    for side, record in result["sides"].items():
        reference_rows, case_rows = [
            {row["mode_index"]: row for row in entry["channels"] if row["incident_side"] == side}
            for entry in (phase_reference, case)
        ]
        distance = case["length_m"] * (1 if side == "left" else -1)
        for match in record["matches"]:
            if match["individual_mode_tracking"] != "matched":
                continue
            reference_k = reference_rows[match["reference_mode_index"]]["wavenumber_rad_m"]
            row = case_rows[match["compared_mode_index"]]
            phase = np.exp(1j * reference_k * distance)
            actual = as_complex(row["transmission_amplitude"])
            error = abs(actual - phase)
            accumulated = abs(row["wavenumber_rad_m"] - reference_k) * max(phase_reference["length_m"], case["length_m"])
            match.update({
                "reference_wavenumber_rad_m": reference_k,
                "reference_phase_factor": phasor(phase),
                "absolute_complex_amplitude_error": error,
                "absolute_phase_error_rad": abs(np.angle(actual / phase)),
                "accumulated_wavenumber_phase_change_rad": accumulated,
                "passed": error < COMPLEX_AMPLITUDE_LIMIT,
            })
        record["passed"] = record["matching_channel_counts"] and all(match["passed"] for match in record["matches"])
    result["passed"] = all(record["passed"] for record in result["sides"].values())
    return result


def read_saved_case(path: Path, identity: dict, report: dict, case: str, plan: tuple) -> tuple[dict, dict]:
    payload = path.read_bytes()
    baseline = json.loads(payload)
    csv_path = path.with_suffix(".csv")
    csv_payload = csv_path.read_bytes()
    csv_digest = hashlib.sha256(csv_payload).hexdigest()
    if csv_digest != baseline["channel_csv_sha256"]:
        raise ValueError("The saved CSV checksum does not match its evidence record.")
    previous = baseline["implementation"]
    paths = [f"src/phononic/{name}" for name in NUMERICAL_SOURCES]
    if any(previous["source_sha256"][name] != identity["source_sha256"][name] for name in paths):
        raise ValueError("Saved numerical source checksums differ from current sources.")
    if any(previous["packages"][name] != identity["packages"][name] for name in ("numpy", "scipy", "scikit-fem")):
        raise ValueError("Saved NumPy, SciPy, or scikit-fem version differs from the current environment.")
    for key in ("frequency_hz", "width_m", "thickness_m", "element_order", "material"):
        if baseline[key] != report[key]:
            raise ValueError(f"Baseline {key} differs from the current study.")
    if previous["python"].split(".")[:2] != identity["python"].split(".")[:2]:
        raise ValueError("Saved Python major/minor differs from the current environment.")
    record = baseline["cases"][case]
    length, cells = plan
    if record["length_m"] != length or tuple(record["cells"]) != cells:
        raise ValueError(f"The saved {case} geometry and mesh differ from the planned comparison.")
    for side in SIDES:
        if sum(row["incident_side"] == side for row in record["channels"]) != 4:
            raise ValueError(f"The saved {case} case must contain four channels in both directions.")
    evidence = {
        "json_file": path.name, "json_sha256": hashlib.sha256(payload).hexdigest(),
        "csv_file": csv_path.name, "csv_sha256": csv_digest,
        "original_status": baseline["status"], "original_failed_checks": baseline["failed_checks"],
        "reused_cases": [case],
        "matching_numerical_source_sha256": {name: previous["source_sha256"][name] for name in paths},
        "saved_python": previous["python"], "saved_platform": previous["platform"],
    }
    return record, evidence


def read_acceptance_inputs(refinement_path: Path, identity: dict, report: dict) -> tuple[dict, dict]:
    saved_case, refinement_evidence = read_saved_case(
        refinement_path, identity, report, "transverse_refined", REFINEMENT_CASES["transverse_refined"],
    )
    refinement = json.loads(refinement_path.read_bytes())
    original = refinement["baseline_evidence"]
    original_json = refinement_path.with_name(original["json_file"])
    original_csv = refinement_path.with_name(original["csv_file"])
    original_json_payload, original_csv_payload = original_json.read_bytes(), original_csv.read_bytes()
    if (hashlib.sha256(original_json_payload).hexdigest() != original["json_sha256"]
            or hashlib.sha256(original_csv_payload).hexdigest() != original["csv_sha256"]):
        raise ValueError("The original failed validation files differ from the saved refinement evidence.")
    return saved_case, {"original": original, "refinement": refinement_evidence}


def recreate_baseline_section(record: dict, material: CubicMaterial) -> SectionModes:
    """Rebuild only the small SAFE section basis required for cross-mesh tracking."""
    geometry = BeamGeometry(record["length_m"], WIDTH_M, THICKNESS_M)
    mesh = make_straight_mesh(geometry, (1, *record["cells"][1:]))
    system = assemble_elasticity(mesh, material, element_order=2, max_dofs=1000)
    modes = solve_port_modes(assemble_port(system, material, max_dofs=record["port_dofs"]), FREQUENCY_HZ)
    for row in record["channels"]:
        root = row["safe_root_index"]
        if not modes.propagating[root] or not np.isclose(modes.wavenumbers_rad_m[root].real, row["wavenumber_rad_m"], rtol=1e-9):
            raise ValueError("The rebuilt SAFE section does not reproduce the saved root indexing.")
        if not np.allclose(polarization(modes, root), row["kinetic_fractions_xyz"], rtol=0, atol=1e-8):
            raise ValueError("The rebuilt SAFE section does not reproduce the saved kinetic polarization.")
    return SectionModes(modes.port.basis, modes.port.M, modes.displacements)


def build_acceptance_reference(material: CubicMaterial) -> tuple[dict, SectionModes]:
    """Solve SAFE roots on the finer section without a whole-length scattering solve."""
    length = ACCEPTANCE_LONG[0]
    with measured_resources() as resources:
        geometry = BeamGeometry(length, WIDTH_M, THICKNESS_M)
        mesh = make_straight_mesh(geometry, ACCEPTANCE_REFERENCE_CELLS)
        system = assemble_elasticity(mesh, material, element_order=2, max_dofs=3 * ACCEPTANCE_REFERENCE_PORT_DOFS)
        modes = solve_port_modes(assemble_port(system, material, max_dofs=ACCEPTANCE_REFERENCE_PORT_DOFS), FREQUENCY_HZ)
        channels = []
        for side, indices in zip(SIDES, (modes.positive_indices, modes.negative_indices)):
            roots = indices[modes.propagating[indices]]
            channels.extend({
                "incident_side": side, "mode_index": position, "safe_root_index": int(root),
                "wavenumber_rad_m": modes.wavenumbers_rad_m[root].real,
                "kinetic_fractions_xyz": polarization(modes, root),
            } for position, root in enumerate(roots, 1))
        reference = {
            "name": "safe_reference", "length_m": length, "cells": ACCEPTANCE_REFERENCE_CELLS,
            "scope": "Finer SAFE phase reference. No volume scattering solved and no continuum error bound established.",
            "trace_dofs": modes.port.basis.N, "mesh_extraction_dofs": system.basis.N,
            "safe_max_relative_residual": modes.relative_residuals.max(),
            "channels": channels, "resources": resources,
        }
    return reference, SectionModes(modes.port.basis, modes.port.M, modes.displacements)


def write_csv(path: Path, cases: dict) -> None:
    fields = (
        "author", "file_tag", "case", "length_m", "cells_x", "cells_y", "cells_z", "displacement_dofs", "port_dofs",
        "elapsed_seconds", "peak_process_memory_mib", "incident_side", "mode_index", "safe_root_index",
        "propagating_channels", "wavenumber_rad_m", "kinetic_x", "kinetic_y", "kinetic_z", "incident_power_W",
        "R", "T", "R_flux", "T_flux", "R_gram", "T_gram", "power_disagreement", "power_balance", "flux_power_balance",
        "net_outward_power_ratio", "linear_residual", "condition_left", "condition_right", "phase_error_rad",
        "complex_amplitude_error", "transmitted_mode_conversion_power", "matched_power_passed",
    )
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        for case in cases.values():
            x, y, z = case["cells"]
            for row in case["channels"]:
                writer.writerow({
                    **row, "author": "Athul Prem", "file_tag": "_AP_pnc", "length_m": case["length_m"],
                    "cells_x": x, "cells_y": y, "cells_z": z, "displacement_dofs": case["displacement_dofs"],
                    "port_dofs": case["port_dofs"], **case.get("resources", {}),
                    **dict(zip(("kinetic_x", "kinetic_y", "kinetic_z"), row["kinetic_fractions_xyz"])),
                })
    temporary.replace(path)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    scope = parser.add_mutually_exclusive_group()
    scope.add_argument("--quick", action="store_true", help="Run only the two small axial meshes.")
    scope.add_argument("--refine", action="store_true", help="Reuse the saved fine baseline and run three targeted refinements.")
    scope.add_argument("--acceptance", action="store_true", help="Run the final long-guide and SAFE-only checks.")
    parser.add_argument("--output-dir", type=Path, default=ROOT / "verification")
    parser.add_argument("--reference-dir", type=Path, default=ROOT / "verification",
                        help="Directory holding the preserved baseline and refinement evidence.")
    parser.add_argument("--baseline", type=Path,
                        help="Saved baseline JSON for --refine, with its companion CSV alongside it.")
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    stem = BASELINE_STEM
    if args.quick:
        stem = "straight_scattering_validation_quick_AP_pnc"
    elif args.refine:
        stem = REFINEMENT_STEM
    elif args.acceptance:
        stem = "straight_scattering_acceptance_AP_pnc"
    json_path, csv_path = (args.output_dir / f"{stem}.{ext}" for ext in ("json", "csv"))
    baseline_path = args.baseline or args.reference_dir / f"{BASELINE_STEM}.json"
    refinement_path = args.reference_dir / f"{REFINEMENT_STEM}.json"
    if args.refine and baseline_path.resolve() == json_path.resolve():
        parser.error("Refinement output cannot overwrite its input baseline.")
    if not args.quick and (json_path.exists() or csv_path.exists()):
        parser.error("Preserved validation evidence exists. Use a different --output-dir.")
    source = Path(__file__).resolve()
    identity = source_identity()
    identity["source_sha256"][source.relative_to(ROOT).as_posix()] = hashlib.sha256(source.read_bytes()).hexdigest()
    material = CubicMaterial()
    max_dofs = MAX_DOFS
    max_port_dofs = 200
    if args.refine or args.acceptance:
        max_dofs = ACCEPTANCE_MAX_DOFS if args.acceptance else REFINEMENT_MAX_DOFS
        max_port_dofs = REFINEMENT_MAX_PORT_DOFS
    report = {
        "author": "Athul Prem", "file_tag": "_AP_pnc", "schema_version": 1, "quick": args.quick,
        "refinement": args.refine,
        "acceptance": args.acceptance,
        "scope": "One 1 GHz uniform straight guide control. Whole-band, patterned-device and experimental validation are outside this control.",
        "frequency_hz": FREQUENCY_HZ, "width_m": WIDTH_M, "thickness_m": THICKNESS_M,
        "element_order": 2, "max_displacement_dofs": max_dofs, "max_port_dofs": max_port_dofs,
        "material": {"name": "nominal cubic Ge", "policy": "fixed_property_control", "C11_Pa": material.C11,
                     "C12_Pa": material.C12, "C44_Pa": material.C44, "rho_kg_m3": material.rho,
                     "crystal_to_lab_rotation": material.rotation.tolist()},
        "limits": {"power_difference": POWER_LIMIT, "complex_amplitude_difference": COMPLEX_AMPLITUDE_LIMIT,
                   "relative_residual": RESIDUAL_LIMIT, "matching_minimum_mass_weighted_overlap": MATCH_MIN_OVERLAP,
                   "near_degenerate_relative_wavenumber_gap": NEAR_DEGENERATE_GAP,
                   "relative_wavenumber_change": WAVENUMBER_RELATIVE_LIMIT,
                   "accumulated_wavenumber_phase_change_rad": ACCUMULATED_PHASE_LIMIT_RAD},
        "phase_reference": "exp(i*k*delta_x), delta_x is measured from incident to transmission face.",
        "tracking_method": "Maximum total mass-weighted overlap of interpolated port displacement fields; wavenumber drift and kinetic polarization are recorded.",
        "resource_method": "Elapsed includes assembly, port solve, and both incidence directions. Peak RSS is sampled for the current process.",
        "implementation": identity, "cases": {}, "comparisons": {},
    }

    port_modes = {}
    cases = report["cases"]
    stage = "initialization"
    try:
        cases_to_run = CASES
        comparisons_to_run = COMPARISONS
        compare_cases = compare
        if args.quick:
            cases_to_run = {name: CASES[name] for name in ("A4", "A8")}
            comparisons_to_run = COMPARISONS[:1]
        elif args.refine:
            stage = "saved fine baseline"
            record, report["baseline_evidence"] = read_saved_case(baseline_path, identity, report, "B_fine", CASES["B_fine"])
            cases["B_fine"] = {**record, "reused": True}
            port_modes["B_fine"] = recreate_baseline_section(record, material)
            cases_to_run = REFINEMENT_CASES
            comparisons_to_run = REFINEMENT_COMPARISONS
            compare_cases = compare_with_common_phase
        elif args.acceptance:
            stage = "saved finest transverse refinement"
            record, report["preserved_evidence"] = read_acceptance_inputs(refinement_path, identity, report)
            cases["transverse_refined"] = {**record, "reused": True}
            port_modes["transverse_refined"] = recreate_baseline_section(record, material)
            cases_to_run = {"long_fine": ACCEPTANCE_LONG}
            comparisons_to_run = (("length_fine_section", "transverse_refined", "long_fine"),)
            compare_cases = compare_with_common_phase
        for name, (length, cells) in cases_to_run.items():
            stage = name
            if args.acceptance and psutil.virtual_memory().available < 1024 * 2**20:
                raise RuntimeError("The bounded final run requires at least 1 GiB of available system memory.")
            record = {"name": name, "length_m": length, "cells": cells, "channels": [], "amplitude_matrices": {}}
            cases[name] = record
            port_modes[name] = run_case(name, record, material, max_dofs, max_port_dofs)
            print(f"{name}: {record['displacement_dofs']} DOFs, {record['resources']['elapsed_seconds']:.2f} s", flush=True)
        for comparison, left, right in comparisons_to_run:
            stage = comparison
            report["comparisons"][comparison] = compare_cases(comparison, cases[left], cases[right], port_modes[left], port_modes[right])
        channels = [row for record in cases.values() for row in record["channels"]]
        gates = {
            "matched_power_all_cases": all(row["matched_power_passed"] for row in channels),
            "linear_and_safe_residuals": (all(row["linear_residual"] <= RESIDUAL_LIMIT for row in channels)
                                          and all(record["safe_max_relative_residual"] <= RESIDUAL_LIMIT for record in cases.values())),
        }
        if args.acceptance:
            gates["direct_flux_balance_all_cases"] = all(
                abs(row["flux_power_balance"]) < POWER_LIMIT and abs(row["net_outward_power_ratio"]) < POWER_LIMIT
                for row in channels
            )
            stage = "finer SAFE phase reference"
            reference, reference_modes = build_acceptance_reference(material)
            report["common_phase_reference"] = reference
            report["common_phase_reference"]["comparisons"] = {}
            gates["safe_reference_residual"] = reference["safe_max_relative_residual"] <= RESIDUAL_LIMIT
            wavenumber_checks = []
            for name in cases:
                stage = f"common phase reference for {name}"
                result = compare_to_phase_reference(reference, cases[name], reference_modes, port_modes[name])
                report["common_phase_reference"]["comparisons"][name] = result
                gates[f"absolute_complex_amplitude_{name}"] = result["passed"]
                for side in result["sides"].values():
                    wavenumber_checks.append(side["matching_channel_counts"] and all(
                        match["individual_mode_tracking"] == "matched"
                        and match["relative_wavenumber_change"] < WAVENUMBER_RELATIVE_LIMIT
                        and match["accumulated_wavenumber_phase_change_rad"] < ACCUMULATED_PHASE_LIMIT_RAD
                        for match in side["matches"]
                    ))
            gates["transverse_wavenumber_drift_at_1p5um"] = all(wavenumber_checks)
            gates["length_fine_section"] = report["comparisons"]["length_fine_section"]["passed"]
        elif args.refine:
            reference = "transverse_refined"
            report["common_phase_reference"] = {
                "case": reference,
                "scope": "Numerical SAFE reference from the finest transverse section tested; its remaining discretization error is unbounded.",
                "comparisons": {},
            }
            for name in cases:
                stage = f"common phase reference for {name}"
                result = compare_to_phase_reference(cases[reference], cases[name], port_modes[reference], port_modes[name])
                report["common_phase_reference"]["comparisons"][name] = result
                gates[f"absolute_complex_amplitude_{name}"] = result["passed"]
            for name, _, _ in REFINEMENT_COMPARISONS:
                gates[name] = report["comparisons"][name]["passed"]
        elif not args.quick:
            for name in ("axial_8_to_16", "transverse_middle_to_fine", "length_middle_mesh"):
                gates[name] = report["comparisons"][name]["passed"]
            gates["absolute_complex_amplitude_on_finest_axial_transverse_and_long_cases"] = all(
                row["complex_amplitude_error"] < COMPLEX_AMPLITUDE_LIMIT
                for case in ("A16", "B_fine", "C_long") for row in cases[case]["channels"]
            )
        report["gate_results"] = gates
        report["failed_checks"] = [name for name, passed in gates.items() if not passed]
        report["status"] = "quick_smoke_passed" if args.quick else "passed"
        if report["failed_checks"]:
            report["status"] = "failed_numerical_gate"
    except (ValueError, KeyError, RuntimeError, OSError, MemoryError, np.linalg.LinAlgError) as exc:
        report.update({"status": "failed_execution", "failed_checks": [stage], "error": f"{type(exc).__name__}: {exc}"})
    report["full_straight_guide_acceptance_passed"] = args.acceptance and report["status"] == "passed"
    write_csv(csv_path, cases)
    report["channel_csv_sha256"] = hashlib.sha256(csv_path.read_bytes()).hexdigest()
    write_json(json_path, report)
    if report["failed_checks"]:
        print(report.get("error") or ", ".join(report["failed_checks"]), file=sys.stderr)
    print(f"{report['status']}: {json_path}")
    return 0 if report["status"] in ("passed", "quick_smoke_passed") else 2


if __name__ == "__main__":
    raise SystemExit(main())
