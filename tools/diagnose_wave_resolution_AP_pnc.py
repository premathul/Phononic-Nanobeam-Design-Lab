"""Measure axial and thickness mesh sensitivity in a uniform guide.

Author: Athul Prem | _AP_pnc
"""

from __future__ import annotations

import argparse
from hashlib import sha256
from pathlib import Path
import os
import sys


ROOT = Path(__file__).resolve().parents[1]
for variable in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ[variable] = "1"
sys.path.insert(0, str(ROOT / "src"))

import numpy as np
from scipy.optimize import linear_sum_assignment

from phononic.elasticity_AP_pnc import assemble_elasticity
from phononic.geometry_AP_pnc import BeamGeometry, make_straight_mesh
from phononic.material_AP_pnc import CubicMaterial
from phononic.port_modes_AP_pnc import assemble_port, solve_port_modes
from phononic.scattering_AP_pnc import solve_scattering
from phononic.workflow_AP_pnc import measured_resources, source_identity, write_json
from validate_scattering_AP_pnc import SectionModes, port_overlap, rows_for_side


FREQUENCY_HZ = 1e9
LENGTH_M = 1200e-9
WIDTH_M = 1000e-9
THICKNESS_M = 200e-9
MAX_DOFS = 10000
MAX_PORT_DOFS = 200
RESIDUAL_LIMIT = 1e-8
MIN_OVERLAP = 0.95
CASES = {
    "axial_200_thickness_1": (6, 5, 1),
    "axial_100_thickness_1": (12, 5, 1),
    "axial_200_thickness_2": (6, 5, 2),
    "axial_100_thickness_2": (12, 5, 2),
}
COMPARISONS = (
    ("axial_200_thickness_1", "axial_100_thickness_1", "axial"),
    ("axial_200_thickness_2", "axial_100_thickness_2", "axial"),
    ("axial_200_thickness_1", "axial_200_thickness_2", "thickness"),
    ("axial_100_thickness_1", "axial_100_thickness_2", "thickness"),
)


def pack(matrix: np.ndarray) -> list:
    return np.stack((matrix.real, matrix.imag), axis=-1).tolist()


def run_case(name: str, cells: tuple[int, int, int], material: CubicMaterial, case: dict) -> SectionModes:
    geometry = BeamGeometry(LENGTH_M, WIDTH_M, THICKNESS_M)
    case.update(cells=list(cells), axial_spacing_m=LENGTH_M / cells[0],
                thickness_layers=cells[2], thickness_spacing_m=THICKNESS_M / cells[2], status="running")
    with measured_resources() as resources:
        case["resources"] = resources
        mesh = make_straight_mesh(geometry, cells)
        system = assemble_elasticity(mesh, material, element_order=2, max_dofs=MAX_DOFS)
        modes = solve_port_modes(assemble_port(system, material, max_dofs=MAX_PORT_DOFS), FREQUENCY_HZ)
        result = solve_scattering(system, modes, FREQUENCY_HZ, incident_side="left")
        incoming = result.incident_mode_indices
        channels = rows_for_side(name, LENGTH_M, modes, result)
        # The same signed-power root defines both incident and transmitted
        # amplitudes. Thus the phase of the same-channel coefficient is gauge invariant.
        transmit_rows = {root: row for row, root in enumerate(result.transmission_mode_indices)}
        order = [transmit_rows[root] for root in incoming]
        transmitted = result.transmission_amplitudes[order]
        expected = np.exp(1j * modes.wavenumbers_rad_m[incoming].real * LENGTH_M)
        max_residual = max(float(modes.relative_residuals.max()), float(result.relative_residuals.max()))
        budget_error = max(float(np.max(abs(getattr(result, field)))) for field in (
            "power_balance", "flux_power_balance", "net_outward_power_ratio",
        ))
        case.update({
            "displacement_dofs": int(system.basis.N), "port_dofs": int(modes.port.basis.N),
            "channels": channels, "incident_root_indices": incoming.tolist(),
            "reflection_root_indices": result.reflection_mode_indices.tolist(),
            "transmission_root_indices": incoming.tolist(),
            "complex_encoding": "[real, imaginary] pairs, outgoing rows and incoming columns",
            "reflection_matrix": pack(result.reflection_amplitudes),
            "transmission_matrix": pack(transmitted),
            "expected_transmission_diagonal": pack(expected),
            "maximum_transmission_complex_error": float(abs(transmitted - np.diag(expected)).max()),
            "maximum_reflection_amplitude": float(abs(result.reflection_amplitudes).max()),
            "maximum_absolute_phase_error_rad": max(abs(row["phase_error_rad"]) for row in channels),
            "maximum_power_budget_error": budget_error, "maximum_relative_residual": max_residual,
            "maximum_modal_flux_disagreement": max(row["power_disagreement"] for row in channels),
        })
        if max_residual > RESIDUAL_LIMIT or budget_error > RESIDUAL_LIMIT:
            raise ValueError(f"Numerical integrity checks failed for {name}.")
    case["status"] = "solved"
    return SectionModes(modes.port.basis, modes.port.M, modes.displacements)


def compare_sections(a: dict, b: dict, modes_a: SectionModes, modes_b: SectionModes, axis: str) -> dict:
    roots_a = np.asarray(a["incident_root_indices"])
    roots_b = np.asarray(b["incident_root_indices"])
    overlaps = port_overlap(modes_a, roots_a, modes_b, roots_b)
    i, j = linear_sum_assignment(-overlaps)
    if len(i) != len(roots_a) or len(j) != len(roots_b) or min(overlaps[i, j]) < MIN_OVERLAP:
        raise ValueError("The propagating channels could not be matched across these meshes.")
    rows = []
    for left, right in zip(i, j):
        coarse, fine = a["channels"][left], b["channels"][right]
        k_change = fine["wavenumber_rad_m"] - coarse["wavenumber_rad_m"]
        if axis == "axial" and abs(k_change) > 1e-8 * abs(coarse["wavenumber_rad_m"]):
            raise ValueError("The discrete port wavenumber changed during axial-only refinement.")
        phase_a, phase_b = (row["phase_error_rad"] for row in (coarse, fine))
        rows.append({
            "reference_mode": int(left + 1), "refined_mode": int(right + 1), "mass_weighted_overlap": overlaps[left, right],
            "reference_phase_error_rad": phase_a, "refined_phase_error_rad": phase_b,
            "phase_error_change_rad": float(np.angle(np.exp(1j * (phase_b - phase_a)))),
            "safe_wavenumber_change_rad_m": k_change, "safe_phase_change_over_guide_rad": k_change * LENGTH_M,
        })
    return {"changed_axis": axis, "channels": rows,
            "maximum_absolute_phase_error_change_rad": max(abs(row["phase_error_change_rad"]) for row in rows),
            "maximum_absolute_safe_phase_change_rad": max(abs(row["safe_phase_change_over_guide_rad"]) for row in rows)}


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, required=True, help="New directory for the diagnostic results")
    args = parser.parse_args(argv)
    try:
        args.output_dir.mkdir(parents=True, exist_ok=False)
    except OSError as exc:
        parser.error(str(exc))
    output = args.output_dir / "wave_resolution_diagnostic_AP_pnc.json"
    identity = source_identity()
    for source in (Path(__file__).resolve(), ROOT / "tools/validate_scattering_AP_pnc.py"):
        identity["source_sha256"][source.relative_to(ROOT).as_posix()] = sha256(source.read_bytes()).hexdigest()
    material = CubicMaterial()
    record = {
        "author": "Athul Prem", "file_tag": "_AP_pnc", "status": "running",
        "scope": ("Uniform guide at 1 GHz, left incidence. Axial spacing changes at fixed discrete SAFE ports. "
                  "Thickness subdivision changes both body and SAFE section, so each result uses its own discrete reference. "
                  "SAFE phase drift is recorded separately. Patterned-device error and continuum accuracy require their own checks."),
        "phase_reference": "exp(i*k*L) for the same signed-power propagating root at both ends",
        "frequency_hz": FREQUENCY_HZ, "length_m": LENGTH_M, "width_m": WIDTH_M, "thickness_m": THICKNESS_M,
        "element_order": 2, "material": {**vars(material), "rotation": material.rotation.tolist(),
                                         "temperature_policy": "fixed_property_control"},
        "limits": {"max_displacement_dofs": MAX_DOFS, "max_port_dofs": MAX_PORT_DOFS,
                   "residual_and_power_budget_error": RESIDUAL_LIMIT, "minimum_channel_overlap": MIN_OVERLAP},
        "implementation": identity, "cases": {}, "comparisons": {},
    }
    write_json(output, record)
    sections = {}
    stage = None
    try:
        for name, cells in CASES.items():
            stage = name
            case = record["cases"][name] = {}
            sections[name] = run_case(name, cells, material, case)
            write_json(output, record)
            print(f"{name}: {case['displacement_dofs']} DOFs, max |phase error| {case['maximum_absolute_phase_error_rad']:.6g} rad", flush=True)
        for first, second, axis in COMPARISONS:
            stage = f"{first}_to_{second}"
            record["comparisons"][stage] = compare_sections(record["cases"][first], record["cases"][second],
                                                            sections[first], sections[second], axis)
        record["status"] = "completed_diagnostic"
    except (ValueError, RuntimeError, OSError, np.linalg.LinAlgError, KeyboardInterrupt) as exc:
        record.update(status="failed", failed_stage=stage, error=f"{type(exc).__name__}: {exc}")
        if stage in record["cases"] and record["cases"][stage]["status"] == "running":
            record["cases"][stage]["status"] = "failed"
        write_json(output, record)
        print(record["error"], file=sys.stderr)
        return 1
    write_json(output, record)
    print(f"Saved diagnostic: {output.resolve()}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
