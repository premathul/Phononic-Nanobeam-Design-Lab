#!/usr/bin/env python3
"""Verify four 3D cantilever modes across meshes. Author: Athul Prem | _AP_pnc."""

from __future__ import annotations

import argparse
import csv
import hashlib
from pathlib import Path
import sys

import numpy as np
from scipy.optimize import linear_sum_assignment

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from phononic.elasticity_AP_pnc import ElasticSystem, ModeResult, assemble_elasticity, solve_modes
from phononic.geometry_AP_pnc import BeamGeometry, make_straight_mesh
from phononic.material_AP_pnc import CubicMaterial
from phononic.workflow_AP_pnc import measured_resources, source_identity, write_json

LEVELS = ((6, 4, 2), (9, 6, 3), (12, 8, 4))
QUICK_LEVELS = ((2, 2, 1), (3, 3, 2), (4, 4, 2))
GEOMETRY = BeamGeometry(2e-6, 1e-6, 200e-9)
BOUNDARY = "clamped_free"
MODE_COUNT = 4
ORDER = 2
MAX_DOFS = 11_475
FREQUENCY_CHANGE_LIMIT = 0.005
OVERLAP_LIMIT = 0.99
RESIDUAL_LIMIT = 1e-8
DEGENERACY_GAP = 0.001
NULL_FREQUENCY_TOL = 1e-9


def assemble(cells: tuple[int, int, int], material: CubicMaterial) -> ElasticSystem:
    mesh = make_straight_mesh(GEOMETRY, cells)
    return assemble_elasticity(mesh, material, element_order=ORDER, max_dofs=MAX_DOFS)


def transfer_modes(coarse: ElasticSystem, modes: np.ndarray, fine: ElasticSystem) -> np.ndarray:
    """Evaluate the coarse vector field at every fine displacement node."""
    components = np.asarray(fine.basis.split_indices())
    points = fine.basis.doflocs[:, components[0]]
    if not np.array_equal(fine.basis.doflocs[:, components], np.broadcast_to(points[:, None], (3, 3, points.shape[1]))):
        raise ValueError("Displacement components must share the same interpolation nodes.")

    # Move box-boundary probes a few ulps inward to avoid roundoff in point location.
    lower, upper = coarse.mesh.p.min(axis=1), coarse.mesh.p.max(axis=1)
    inset = 32 * np.finfo(float).eps * (upper - lower)
    points = np.clip(points, (lower + inset)[:, None], (upper - inset)[:, None])
    transferred = np.empty((fine.basis.N, modes.shape[1]), dtype=modes.dtype)
    # Point location may check every tetrahedron. Bound its working array by batching.
    for start in range(0, points.shape[1], 128):
        section = slice(start, start + 128)
        probes = coarse.basis.probes(points[:, section])
        values = (probes @ modes).reshape(3, points[:, section].shape[1], modes.shape[1])
        transferred[components[:, section], :] = values
    return transferred


def near_degenerate(frequencies: np.ndarray) -> np.ndarray:
    close = np.diff(frequencies) / frequencies[1:] < DEGENERACY_GAP
    return np.r_[False, close] | np.r_[close, False]


def compare_levels(
    coarse: ElasticSystem, coarse_modes: ModeResult, fine: ElasticSystem, fine_modes: ModeResult
) -> dict:
    transferred = transfer_modes(coarse, coarse_modes.modes_full, fine)
    fine_vectors = fine_modes.modes_full
    mass_transferred, mass_fine = fine.M @ transferred, fine.M @ fine_vectors
    norms_transferred = np.sqrt(np.sum(transferred.conj() * mass_transferred, axis=0).real)
    norms_fine = np.sqrt(np.sum(fine_vectors.conj() * mass_fine, axis=0).real)
    overlaps = np.abs(transferred.conj().T @ mass_fine) / np.outer(norms_transferred, norms_fine)
    if not np.isfinite(overlaps).all() or (overlaps > 1 + 1e-12).any():
        raise ValueError("Invalid mass-normalized mode overlap.")
    overlaps = np.minimum(overlaps, 1.0)
    coarse_order, fine_order = linear_sum_assignment(-overlaps)
    coarse_degenerate = near_degenerate(coarse_modes.frequencies_hz)
    fine_degenerate = near_degenerate(fine_modes.frequencies_hz)

    matches = []
    for i, j in zip(coarse_order, fine_order):
        f_coarse = coarse_modes.frequencies_hz[i]
        f_fine = fine_modes.frequencies_hz[j]
        tracked = not (coarse_degenerate[i] or fine_degenerate[j])
        matches.append({
            "coarse_mode_index": int(i) + 1,
            "fine_mode_index": int(j) + 1,
            "coarse_frequency_hz": f_coarse,
            "fine_frequency_hz": f_fine,
            "relative_frequency_change": abs(f_fine - f_coarse) / f_fine,
            "mass_weighted_overlap": overlaps[i, j],
            "individual_mode_tracking": "matched" if tracked else "unresolved_near_degenerate",
        })
    return {"overlap_matrix": overlaps.tolist(), "matches": matches}


def matrix_hash(matrix) -> str:
    digest = hashlib.sha256(repr(matrix.shape).encode("ascii"))
    for array in (matrix.indptr, matrix.indices, matrix.data):
        digest.update(array.dtype.str.encode("ascii"))
        digest.update(array.tobytes())
    return digest.hexdigest()


def check_temperature_null(material: CubicMaterial) -> dict:
    cases = []
    for temperature in (0.02, 4.0):
        # Temperature is a label under this policy. Assemble and solve each case afresh.
        system = assemble(QUICK_LEVELS[0], material)
        modes = solve_modes(system, boundary=BOUNDARY, num_modes=MODE_COUNT)
        cases.append({
            "temperature_label_K": temperature,
            "K_sha256": matrix_hash(system.K),
            "M_sha256": matrix_hash(system.M),
            "frequencies_hz": modes.frequencies_hz.tolist(),
            "relative_residuals": modes.relative_residuals.tolist(),
        })
    first, second = cases
    relative_change = np.max(np.abs(np.asarray(first["frequencies_hz"]) / second["frequencies_hz"] - 1))
    same_operators = all(first[key] == second[key] for key in ("K_sha256", "M_sha256"))
    residuals_pass = all(max(case["relative_residuals"]) <= RESIDUAL_LIMIT for case in cases)
    return {
        "policy": "fixed_property_control",
        "scope": "Two independent solves of the same elastic inputs. Temperature labels have no constitutive effect.",
        "mesh_cells": QUICK_LEVELS[0],
        "cases": cases,
        "identical_mechanical_operators": same_operators,
        "max_relative_frequency_change": relative_change,
        "passed": same_operators and residuals_pass and relative_change <= NULL_FREQUENCY_TOL,
    }


def write_csv(path: Path, levels: list[dict]) -> None:
    fields = (
        "author", "file_tag", "level", "cells_x", "cells_y", "cells_z", "displacement_dofs",
        "mode_index", "frequency_hz", "relative_residual", "elapsed_seconds", "peak_process_memory_mib",
        "prior_level_mode_index", "relative_frequency_change", "mass_weighted_overlap", "individual_mode_tracking",
    )
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        for level in levels:
            matches = {match["fine_mode_index"]: match for match in level.get("comparison", {}).get("matches", [])}
            for index, (frequency, residual) in enumerate(zip(level["frequencies_hz"], level["relative_residuals"]), 1):
                match = matches.get(index, {})
                x, y, z = level["mesh_cells"]
                writer.writerow({
                    "author": "Athul Prem", "file_tag": "_AP_pnc", "level": level["level"],
                    "cells_x": x, "cells_y": y, "cells_z": z, "displacement_dofs": level["displacement_dofs"],
                    "mode_index": index, "frequency_hz": frequency, "relative_residual": residual,
                    "elapsed_seconds": level["resources"]["elapsed_seconds"],
                    "peak_process_memory_mib": level["resources"]["peak_process_memory_mib"],
                    "prior_level_mode_index": match.get("coarse_mode_index"),
                    **{key: match.get(key) for key in fields[-3:]},
                })
    temporary.replace(path)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--quick", action="store_true", help="Use tiny meshes to exercise the validation workflow.")
    parser.add_argument("--output-dir", type=Path, default=ROOT / "verification")
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    stem = "straight_beam_validation_quick_AP_pnc" if args.quick else "straight_beam_validation_AP_pnc"
    json_path = args.output_dir / f"{stem}.json"
    csv_path = args.output_dir / f"{stem}.csv"

    material = CubicMaterial()
    source = Path(__file__).resolve()
    identity = source_identity()
    identity["source_sha256"][source.relative_to(ROOT).as_posix()] = hashlib.sha256(source.read_bytes()).hexdigest()
    report = {
        "author": "Athul Prem", "file_tag": "_AP_pnc", "schema_version": 1,
        "scope": "Quick interface smoke case" if args.quick else "Numerical verification of four finite-beam modes",
        "quick": args.quick,
        "geometry_nm": {"length": 2000, "width": 1000, "thickness": 200},
        "material": {"name": "nominal cubic Ge", "policy": "fixed_property_control", "C11_Pa": material.C11,
                     "C12_Pa": material.C12, "C44_Pa": material.C44, "density_kg_m3": material.rho,
                     "crystal_to_lab_rotation": material.rotation.tolist()},
        "boundary": BOUNDARY, "displacement_order": ORDER, "num_modes": MODE_COUNT, "max_dofs": MAX_DOFS,
        "tracking_method": "Coarse nodal interpolation onto finer mesh, fine-mass normalized overlaps, maximum total overlap assignment.",
        "resource_method": "Elapsed time includes mesh, assembly, eigensolve and any comparison. RSS samples the current process at 50 ms intervals.",
        "gates": {"relative_frequency_change_below": FREQUENCY_CHANGE_LIMIT, "mass_weighted_overlap_above": OVERLAP_LIMIT,
                  "max_relative_residual": RESIDUAL_LIMIT, "near_degeneracy_relative_gap": DEGENERACY_GAP,
                  "temperature_null_max_relative_change": NULL_FREQUENCY_TOL},
        "implementation": identity, "levels": [],
    }
    previous_system, previous_modes = None, None
    stage = "mesh levels"
    try:
        for index, cells in enumerate(QUICK_LEVELS if args.quick else LEVELS, 1):
            stage = f"mesh level {index} {cells}"
            with measured_resources() as resources:
                system = assemble(cells, material)
                modes = solve_modes(system, boundary=BOUNDARY, num_modes=MODE_COUNT)
                level = {
                    "level": index, "mesh_cells": cells, "displacement_dofs": system.basis.N,
                    "frequencies_hz": modes.frequencies_hz.tolist(), "relative_residuals": modes.relative_residuals.tolist(),
                    "resources": resources,
                }
                report["levels"].append(level)
                if previous_system is not None:
                    level["comparison"] = compare_levels(previous_system, previous_modes, system, modes)
            print(f"Level {index}: {system.basis.N} DOFs, {resources['elapsed_seconds']:.2f} s", flush=True)
            previous_system, previous_modes = system, modes
        del previous_system, previous_modes, system, modes

        stage = "fixed-property temperature null"
        report["temperature_null"] = check_temperature_null(material)
        final_matches = report["levels"][-1]["comparison"]["matches"]
        decisions = {
            "frequency_change": all(match["relative_frequency_change"] < FREQUENCY_CHANGE_LIMIT for match in final_matches),
            "mass_weighted_overlap": all(match["mass_weighted_overlap"] > OVERLAP_LIMIT for match in final_matches),
            "individual_mode_tracking": all(match["individual_mode_tracking"] == "matched" for match in final_matches),
            "relative_residual": all(max(level["relative_residuals"]) <= RESIDUAL_LIMIT for level in report["levels"]),
            "temperature_null": report["temperature_null"]["passed"],
        }
        report["gate_results"] = decisions
        report["status"] = "passed" if all(decisions.values()) else "failed_numerical_gate"
        report["spatial_acceptance_passed"] = all(decisions.values()) and not args.quick
        report["failed_checks"] = [name for name, passed in decisions.items() if not passed]
    except (ValueError, RuntimeError, OSError, MemoryError, np.linalg.LinAlgError) as exc:
        report.update({"status": "failed_execution", "spatial_acceptance_passed": False,
                       "failed_checks": [stage], "error": f"{type(exc).__name__}: {exc}"})

    write_csv(csv_path, report["levels"])
    report["mode_csv_sha256"] = hashlib.sha256(csv_path.read_bytes()).hexdigest()
    write_json(json_path, report)
    detail = report.get("error") or ", ".join(report["failed_checks"])
    print(f"{report['status']}{f' ({detail})' if detail else ''}: {json_path}")
    return 0 if report["status"] == "passed" else 2


if __name__ == "__main__":
    raise SystemExit(main())
