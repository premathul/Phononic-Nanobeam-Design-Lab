#!/usr/bin/env python3
"""Bounded Gmsh perforation and power-scattering checks. Author: Athul Prem | _AP_pnc."""

from __future__ import annotations

import argparse
import csv
from dataclasses import asdict, dataclass
import hashlib
import inspect
import json
from pathlib import Path
import sys

import numpy as np
import psutil
from scipy.optimize import linear_sum_assignment
from scipy.linalg import solve_triangular
from scipy.spatial import cKDTree
from skfem import Basis, ElementTriP2, MeshTet, MeshTri, asm
from skfem.models.poisson import mass as scalar_mass

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from phononic.elasticity_AP_pnc import assemble_elasticity
from phononic.geometry_AP_pnc import BeamGeometry, Hole, read_hole_pattern
from phononic.material_AP_pnc import CubicMaterial
from phononic.perforated_AP_pnc import make_perforated_mesh, volume_diagnostics
from phononic.port_modes_AP_pnc import assemble_port, solve_port_modes
from phononic.scattering_AP_pnc import solve_scattering
from phononic.workflow_AP_pnc import digest, measured_resources, source_identity, write_json


FREQUENCY_HZ = 1e9
POWER_LIMIT = 0.01
RESIDUAL_LIMIT = 1e-8
CURVATURE_LIMIT = 0.01
GEOMETRY_VOLUME_TOL = 1e-7
MAX_DOFS = 30000
MAX_PORT_DOFS = 400
HOLE_SOURCE = ROOT / "data/geometry/hole_geometry_derived_AP_pnc.csv"
AUTHOR = {"author": "Athul Prem", "file_tag": "_AP_pnc"}
MAIN = ("empty", "holes_coarse", "holes_middle", "holes_fine")
EXTRA = ("curvature_fine", "long_leads", "port_fine", "body_phase_fine", "original_26")
NUMERICAL_MODULES = ("elasticity", "geometry", "lead_modes", "material", "perforated", "port_modes", "scattering")


@dataclass(frozen=True)
class Case:
    length_nm: float
    width_nm: float
    thickness_nm: float
    holes_nm: tuple[tuple[float, float], ...]
    body_size_nm: float
    port_size_nm: float = 150
    circle_points: int = 32
    source_sha256: str | None = None


def case_plan(name: str) -> Case:
    holes = ((625.0, 250.0), (1175.0, 250.0))
    if name == "empty":
        return Case(1800, 600, 200, (), 130)
    if name in ("holes_coarse", "holes_middle", "holes_fine"):
        size = {"holes_coarse": 160, "holes_middle": 130, "holes_fine": 105}[name]
        return Case(1800, 600, 200, holes, size)
    if name == "curvature_fine":
        return Case(1800, 600, 200, holes, 105, circle_points=48)
    if name == "long_leads":
        return Case(2400, 600, 200, ((925.0, 250.0), (1475.0, 250.0)), 105)
    if name == "port_fine":
        return Case(1800, 600, 200, holes, 105, port_size_nm=75)
    if name == "body_phase_fine":
        return Case(1800, 600, 200, holes, 80, circle_points=48)
    original = read_hole_pattern(HOLE_SOURCE)
    if len(original.holes) != 26:
        raise ValueError("The preserved original input must contain 26 holes.")
    minimum = min(hole.x_m - hole.diameter_m / 2 for hole in original.holes)
    maximum = max(hole.x_m + hole.diameter_m / 2 for hole in original.holes)
    positions = tuple(((hole.x_m - minimum) * 1e9 + 1500, hole.diameter_m * 1e9) for hole in original.holes)
    return Case((maximum - minimum) * 1e9 + 3000, 1000, 200, positions, 200, port_size_nm=200,
                source_sha256=original.source_sha256)


def file_sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def pack(values) -> list:
    array = np.asarray(values, dtype=complex)
    return np.stack((array.real, array.imag), axis=-1).tolist()


def unpack(values) -> np.ndarray:
    array = np.asarray(values)
    return array[..., 0] + 1j * array[..., 1]


def make_mesh(case: Case):
    geometry = BeamGeometry(case.length_nm * 1e-9, case.width_nm * 1e-9, case.thickness_nm * 1e-9)
    holes = tuple(Hole(x * 1e-9, d * 1e-9) for x, d in case.holes_nm)
    mesh = make_perforated_mesh(geometry, holes, case.body_size_nm * 1e-9, port_mesh_size_m=case.port_size_nm * 1e-9,
                                circle_points=case.circle_points, max_port_dofs=MAX_PORT_DOFS, element_order=2)
    geometry_checks = volume_diagnostics(mesh, geometry, holes)
    # Displacement P2 tetrahedra carry three DOFs per vertex and per edge.
    dofs = 3 * (mesh.nvertices + mesh.edges.shape[1])
    geometry_checks.update({"tetrahedra": mesh.nelements, "displacement_dofs": dofs})
    if holes:
        angle = 2 * np.pi / case.circle_points
        polygon_bias = 1 - np.sin(angle) / angle
        geometry_checks["predicted_polygon_removed_volume_bias"] = polygon_bias
        geometry_checks["polygon_volume_difference"] = abs(geometry_checks["relative_removed_volume_error"] - polygon_bias)
        geometry_checks["polygon_sagitta_upper_m"] = max(d for _, d in case.holes_nm) * 0.5e-9 * (1 - np.cos(angle / 2))
    return mesh, geometry_checks


def restore_mesh(path: Path) -> MeshTet:
    with np.load(path, allow_pickle=False) as arrays:
        return MeshTet(arrays["points_m"], arrays["tetrahedra"]).with_boundaries(
            {name: arrays[name] for name in ("xleft", "xright", "free")})


def time_reversal(modes, positive: np.ndarray, negative: np.ndarray):
    # Align each -power basis vector to the conjugate of its +power partner.
    plus = modes.displacements[:, positive].conj()
    minus = modes.displacements[:, negative]
    mass = modes.port.M
    inner = plus.conj().T @ (mass @ minus)
    plus_norm = np.sqrt(np.sum((plus.conj() * (mass @ plus)).real, axis=0))
    minus_norm = np.sqrt(np.sum((minus.conj() * (mass @ minus)).real, axis=0))
    overlap = abs(inner) / np.outer(plus_norm, minus_norm)
    i, j = linear_sum_assignment(-overlap)
    k_plus = modes.wavenumbers_rad_m[positive].real
    k_minus = modes.wavenumbers_rad_m[negative[j]].real
    relative_k_error = abs(k_plus + k_minus) / abs(k_plus)
    selected = overlap[i, j]
    if np.min(selected) < 0.99 or np.max(relative_k_error) > 1e-8:
        raise RuntimeError("Time reversal did not identify stable individual channel pairs.")
    alignment = inner[i, j] / abs(inner[i, j])
    return j, alignment, selected, relative_k_error


def scattering(mesh, material: CubicMaterial, empty: bool) -> dict:
    system = assemble_elasticity(mesh, material, element_order=2, max_dofs=MAX_DOFS)
    modes = solve_port_modes(assemble_port(system, material, max_dofs=MAX_PORT_DOFS), FREQUENCY_HZ)
    left, right = (solve_scattering(system, modes, FREQUENCY_HZ, incident_side=side) for side in ("left", "right"))
    positive, negative = left.incident_mode_indices, right.incident_mode_indices
    j, phase, overlap, k_error = time_reversal(modes, positive, negative)
    # The native Uminus = phase * conj(Uplus). Use conjugate bases on the opposite directions.
    r_left = phase[:, None] * left.reflection_amplitudes[j, :]
    t_right_left = phase[:, None] * right.transmission_amplitudes[np.ix_(j, j)] / phase[None, :]
    t_left_right = left.transmission_amplitudes
    r_right = right.reflection_amplitudes[:, j] / phase[None, :]
    matrix = np.block([[r_left, t_right_left], [t_left_right, r_right]])
    reciprocity_error = float(np.max(abs(matrix - matrix.T)))
    channels = {}
    for result, order in ((left, np.arange(positive.size)), (right, j)):
        root_ids = result.incident_mode_indices[order]
        channel_data = {name: np.asarray(getattr(result, name))[order].tolist() for name in (
            "reflection", "transmission", "reflection_flux", "transmission_flux", "reflection_gram", "transmission_gram",
            "power_balance", "flux_power_balance", "net_outward_power_ratio", "relative_residuals", "incident_powers_w",
        )}
        channel_data.update({"safe_root_indices": root_ids.tolist(), "port_basis_condition": list(result.port_basis_condition)})
        channels[result.incident_side] = channel_data
    budget = max(abs(np.asarray(side[name])).max() for side in channels.values()
                 for name in ("power_balance", "flux_power_balance", "net_outward_power_ratio"))
    flux_error = max(abs(np.asarray(side[quantity + estimator]) - side[quantity]).max()
                     for side in channels.values() for quantity in ("reflection", "transmission") for estimator in ("_flux", "_gram"))
    residual = max(max(side["relative_residuals"]) for side in channels.values())
    safe_residual = float(modes.relative_residuals.max())
    gates = {"energy_budget": budget < POWER_LIMIT, "modal_direct_flux_agreement": flux_error < POWER_LIMIT,
             "reciprocity": reciprocity_error < POWER_LIMIT, "residuals": max(residual, safe_residual) < RESIDUAL_LIMIT}
    if empty:
        gates["matched_no_hole_control"] = all(max(side["reflection"]) < POWER_LIMIT and max(abs(np.asarray(side["transmission"]) - 1)) < POWER_LIMIT
                                                for side in channels.values())
    # A coordinate-sorted, unit-mass representation pairs modes across identical port meshes.
    basis = modes.port.basis
    dofs = np.asarray(basis.split_indices())
    order = np.lexsort(basis.doflocs[:, dofs[0]])
    sorted_dofs = dofs[:, order].ravel()
    plus = modes.displacements[np.ix_(sorted_dofs, positive)]
    mass = modes.port.M[np.ix_(sorted_dofs, sorted_dofs)]
    # Save the small trace basis in weighted coordinates; its Gram product is the physical displacement overlap.
    mass_coordinates = np.linalg.cholesky(mass).T @ plus
    mass_coordinates /= np.linalg.norm(mass_coordinates, axis=0)
    port_mesh = basis.mesh
    node_order = np.lexsort(port_mesh.p)
    inverse = np.argsort(node_order)
    facets = np.sort(inverse[port_mesh.t].T, axis=1)
    return {"channels": channels, "propagating_modes_per_direction": positive.size, "port_dofs": basis.N,
            "safe_max_relative_residual": safe_residual, "max_power_budget_error": budget,
            "max_modal_flux_difference": flux_error, "max_reciprocity_error": reciprocity_error,
            "time_reversal_mode_overlap": overlap.tolist(), "time_reversal_relative_wavenumber_error": k_error.tolist(),
            "wavenumbers_positive_power_rad_m": modes.wavenumbers_rad_m[positive].real.tolist(),
            "trace_coordinates_m": basis.doflocs[:, sorted_dofs].tolist(),
            "trace_triangles": sorted(facets.tolist()), "trace_mode_coordinates": pack(mass_coordinates),
            "scattering_matrix_time_reversal_basis": pack(matrix), "gates": gates}


def compare_cases(first: dict, second: dict, cross_port_overlap: np.ndarray | None = None) -> dict:
    a, b = first["scattering"], second["scattering"]
    n_a = a["propagating_modes_per_direction"]
    n_b = b["propagating_modes_per_direction"]
    cross = cross_port_overlap
    if cross is None:
        if (a["trace_triangles"] != b["trace_triangles"] or np.shape(a["trace_coordinates_m"]) != np.shape(b["trace_coordinates_m"])
                or not np.allclose(a["trace_coordinates_m"], b["trace_coordinates_m"], rtol=0, atol=1e-16)):
            raise ValueError("Cross-case comparison requires the same physical port triangulation.")
        modes_a, modes_b = unpack(a["trace_mode_coordinates"]), unpack(b["trace_mode_coordinates"])
        cross = modes_a.conj().T @ modes_b
    i, j = linear_sum_assignment(-abs(cross))
    if len(j) != n_a or len(j) != n_b or np.min(abs(cross[i, j])) < 0.99:
        raise ValueError("Cross-case channel overlap did not identify the same propagating modes.")
    k_a = np.asarray(a["wavenumbers_positive_power_rad_m"])
    k_b = np.asarray(b["wavenumbers_positive_power_rad_m"])[j]
    relative_k_change = abs(k_a - k_b) / abs(k_a)
    if cross_port_overlap is None and np.max(relative_k_change) > 1e-8:
        raise ValueError("Cross-case channel wavenumbers changed despite fixed ports.")
    change = {side: {q: abs(np.asarray(a["channels"][side][q]) - np.asarray(b["channels"][side][q])[j]).tolist()
                     for q in ("reflection", "transmission")} for side in ("left", "right")}
    max_change = max(max(q) for side in change.values() for q in side.values())
    alignment = cross[i, j] / abs(cross[i, j])
    n = len(j)
    order = np.r_[j, j + n]
    # Unit-power incident and outgoing bases are complex conjugates of one another.
    s_a = unpack(a["scattering_matrix_time_reversal_basis"])
    s_b = unpack(b["scattering_matrix_time_reversal_basis"])[np.ix_(order, order)]
    input_alignment = np.r_[alignment, alignment.conj()]
    s_b = s_b / input_alignment[:, None] / input_alignment[None, :]
    shift = (second["configuration"]["length_nm"] - first["configuration"]["length_nm"]) * 0.5e-9
    correction = np.tile(np.exp(-1j * k_a * shift), 2)
    phase_change = float(np.max(abs(s_a - s_b * correction[:, None] * correction[None, :])))
    return {"channel_matching_overlap": np.minimum(abs(cross[i, j]), 1).tolist(),
            "matched_relative_wavenumber_change": relative_k_change.tolist(), "power_change_by_channel": change,
            "max_absolute_power_change": max_change, "deembedded_complex_scattering_change": phase_change,
            "lead_extension_each_end_m": shift, "power_converged": max_change < POWER_LIMIT,
            "complex_scattering_change_below_0p01": phase_change < POWER_LIMIT}


def trace_scalar_basis(record: dict, mesh_file: Path):
    with np.load(mesh_file, allow_pickle=False) as arrays:
        xyz = arrays["points_m"]
        end = xyz[1:, abs(xyz[0] - xyz[0].min()) < 1e-16]
    end = end[:, np.lexsort(end)]
    triangles = np.asarray(record["scattering"]["trace_triangles"]).T
    basis = Basis(MeshTri(end, triangles), ElementTriP2(), intorder=4)
    n = basis.N
    sorted_coordinates = np.asarray(record["scattering"]["trace_coordinates_m"])[:, :n]
    distance, order = cKDTree(basis.doflocs.T).query(sorted_coordinates.T)
    if np.max(distance) > 1e-16 or np.unique(order).size != n:
        raise ValueError("Rebuilt trace interpolation nodes do not match their saved coordinates.")
    scalar = asm(scalar_mass, basis)[order, :][:, order].toarray() * CubicMaterial().rho
    return basis, order, np.linalg.cholesky(scalar)


def compare_port_fields(first: dict, second: dict, first_mesh: Path, second_mesh: Path) -> np.ndarray:
    coarse, coarse_order, coarse_factor = trace_scalar_basis(first, first_mesh)
    fine, fine_order, fine_factor = trace_scalar_basis(second, second_mesh)
    points = fine.doflocs[:, fine_order]
    lower, upper = coarse.mesh.p.min(axis=1), coarse.mesh.p.max(axis=1)
    inset = 32 * np.finfo(float).eps * (upper - lower)
    points = np.clip(points, (lower + inset)[:, None], (upper - inset)[:, None])
    probes = coarse.probes(points).tocsr()[:, coarse_order]
    values = unpack(first["scattering"]["trace_mode_coordinates"])
    transferred = []
    for component in np.split(values, 3, axis=0):
        physical = solve_triangular(coarse_factor.T, component)
        transferred.append(fine_factor.T @ (probes @ physical))
    weighted = np.vstack(transferred)
    weighted /= np.linalg.norm(weighted, axis=0)
    return weighted.conj().T @ unpack(second["scattering"]["trace_mode_coordinates"])


def case_identity(case: Case, identity: dict) -> str:
    configuration = asdict(case)
    input_identity = {"case": configuration, "packages": identity["packages"], "python": identity["python"],
                      "frequency_hz": FREQUENCY_HZ, "max_dofs": MAX_DOFS, "max_port_dofs": MAX_PORT_DOFS}
    numerical_paths = [f"src/phononic/{module}_AP_pnc.py" for module in NUMERICAL_MODULES]
    computational_code = [inspect.getsource(function) for function in (make_mesh, restore_mesh, time_reversal, scattering, pack)]
    return digest({**input_identity, "source": {name: identity["source_sha256"][name] for name in numerical_paths},
                   "computational_code": computational_code, "power_limit": POWER_LIMIT, "residual_limit": RESIDUAL_LIMIT})


def checkpoint(root: Path, name: str, case: Case, identity: dict, mesh_only: bool) -> tuple[dict, Path]:
    configuration = asdict(case)
    case_hash = case_identity(case, identity)
    path = root / f"{name}-{case_hash[:12]}"
    path.mkdir(parents=True, exist_ok=True)
    mesh_path, mesh_record_path = (path / f"{stem}_AP_pnc.{ext}" for stem, ext in (("mesh", "npz"), ("mesh", "json")))
    final_path = path / "scattering_AP_pnc.json"
    equivalent = path / "equivalent_checkpoint_AP_pnc.json"
    if equivalent.exists():
        migration = json.loads(equivalent.read_bytes())
        existing = (root / migration["path"]).resolve()
        if (migration["case_hash"] != case_hash or not existing.is_relative_to(root)
                or file_sha(existing) != migration["sha256"]):
            raise ValueError(f"Equivalent checkpoint provenance check failed: {equivalent}")
        completed = json.loads(existing.read_bytes())
        if (completed["case_hash"] != migration["original_case_hash"] or completed["status"] != "solved"
                or file_sha(existing.with_name("mesh_AP_pnc.npz")) != completed["mesh_sha256"]):
            raise ValueError(f"Equivalent checkpoint does not match its source inputs or mesh: {existing}")
        return completed, existing
    if final_path.exists():
        completed = json.loads(final_path.read_bytes())
        if completed["case_hash"] != case_hash or file_sha(mesh_path) != completed["mesh_sha256"]:
            raise ValueError(f"Case checkpoint does not match its source inputs or mesh: {final_path}")
        return completed, final_path
    lock = path / ".running"
    with lock.open("x", encoding="utf-8") as stream:
        stream.write(str(psutil.Process().pid))
    record = {**AUTHOR, "name": name, "case_hash": case_hash, "configuration": configuration, "implementation": identity}
    try:
        if mesh_record_path.exists():
            record = json.loads(mesh_record_path.read_bytes())
            if record["case_hash"] != case_hash or file_sha(mesh_path) != record["mesh_sha256"]:
                raise ValueError(f"Saved preflight mesh checksum failed: {mesh_path}")
            mesh = None if mesh_only else restore_mesh(mesh_path)
        else:
            with measured_resources() as measurement:
                mesh, record["geometry_checks"] = make_mesh(case)
            np.savez_compressed(mesh_path, points_m=mesh.p, tetrahedra=mesh.t, **mesh.boundaries)
            record.update({"status": "meshed", "mesh_sha256": file_sha(mesh_path), "mesh_resources": measurement})
            write_json(mesh_record_path, record)
        if mesh_only:
            return record, mesh_record_path
        dofs = record["geometry_checks"]["displacement_dofs"]
        if dofs > MAX_DOFS:
            raise RuntimeError(f"Mesh has {dofs} DOFs, exceeding the pre-solve cap {MAX_DOFS}.")
        if psutil.virtual_memory().available < 2 * 2**30:
            raise RuntimeError("At least 2 GiB of available memory is required for this bounded validation.")
        with measured_resources() as measurement:
            record["scattering"] = scattering(mesh, CubicMaterial(), not case.holes_nm)
        record.update({"status": "solved", "scattering_resources": measurement})
        write_json(final_path, record)
        return record, final_path
    except (OSError, ValueError, RuntimeError, MemoryError, np.linalg.LinAlgError) as exc:
        record.update({"status": "failed", "error": f"{type(exc).__name__}: {exc}"})
        attempt = 1
        failure = path / f"failure_{attempt}_AP_pnc.json"
        while failure.exists():
            attempt += 1
            failure = path / f"failure_{attempt}_AP_pnc.json"
        write_json(failure, record)
        return record, failure
    finally:
        lock.unlink()


def write_verification(output: Path, report: dict, records: dict, identity: dict) -> Path:
    output.mkdir(parents=True, exist_ok=True)
    fields = ("author", "file_tag", "case", "incident_side", "mode_index", "safe_root_index", "displacement_dofs",
              "port_dofs", "reflection", "transmission", "reflection_flux", "transmission_flux", "reflection_gram",
              "transmission_gram", "power_balance", "flux_power_balance", "net_outward_power_ratio", "relative_residuals")
    basename = "two_hole_scattering_validation_AP_pnc"
    csv_path, json_path = (output / f"{basename}.{suffix}" for suffix in ("csv", "json"))
    if csv_path.exists() or json_path.exists():
        raise FileExistsError(f"Verification evidence already exists in {output}.")
    compact = {**report, "implementation": identity, "case_metrics": {}}
    with csv_path.open("x", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields, lineterminator="\n")
        writer.writeheader()
        for name, record in records.items():
            compact["case_metrics"][name] = {key: record[key] for key in (
                "status", "case_hash", "configuration", "implementation", "geometry_checks", "mesh_sha256",
                "mesh_resources", "scattering_resources", "error",
            ) if key in record}
            if record["status"] != "solved":
                continue
            science = record["scattering"]
            compact["case_metrics"][name]["scattering"] = {key: value for key, value in science.items()
                                                           if not key.startswith("trace_")}
            for side, channels in science["channels"].items():
                for i, root in enumerate(channels["safe_root_indices"]):
                    writer.writerow({**AUTHOR, "case": name, "incident_side": side, "mode_index": i + 1,
                                     "safe_root_index": root, "displacement_dofs": record["geometry_checks"]["displacement_dofs"],
                                     "port_dofs": science["port_dofs"], **{key: channels[key][i] for key in fields[8:]}})
    compact["channel_csv_sha256"] = file_sha(csv_path)
    write_json(json_path, compact)
    return json_path


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--case", action="append", choices=MAIN + EXTRA, help="Select one or more cases. Defaults to the four main cases.")
    parser.add_argument("--mesh-only", action="store_true", help="Record actual geometry and DOF costs without solving scattering.")
    parser.add_argument("--output-dir", type=Path, default=ROOT / "results" / "perforated_validation")
    parser.add_argument("--verification-dir", type=Path, help="Write a compact standalone JSON/CSV evidence pair.")
    args = parser.parse_args()
    output = args.output_dir.resolve()
    identity = source_identity()
    script = Path(__file__).resolve()
    identity["source_sha256"][script.relative_to(ROOT).as_posix()] = file_sha(script)
    records, pointers = {}, {}
    for name in dict.fromkeys(args.case or MAIN):
        record, path = checkpoint(output, name, case_plan(name), identity, args.mesh_only)
        records[name] = record
        pointers[name] = {"path": path.relative_to(output).as_posix(), "sha256": file_sha(path), "status": record["status"]}
        print(f"{name}: {record['status']}, {record.get('geometry_checks', {}).get('displacement_dofs', 'unknown')} DOFs", flush=True)
        if record["status"] == "failed":
            print(record["error"], file=sys.stderr)

    comparisons, gates = {}, {}
    for name, record in records.items():
        if record["status"] in ("meshed", "solved"):
            geometry = record["geometry_checks"]
            gates[f"{name}_geometry"] = bool(
                abs(geometry["relative_solid_volume_error"]) < CURVATURE_LIMIT
                and (geometry["relative_removed_volume_error"] is None
                     or (abs(geometry["relative_removed_volume_error"]) < CURVATURE_LIMIT
                         and geometry["polygon_volume_difference"] < GEOMETRY_VOLUME_TOL))
            )
        if not args.mesh_only and record["status"] == "solved":
            gates[f"{name}_scattering"] = all(record["scattering"]["gates"].values())
    if not args.mesh_only:
        for left, right in (("holes_coarse", "holes_middle"), ("holes_middle", "holes_fine"),
                            ("holes_fine", "curvature_fine"), ("holes_fine", "long_leads"),
                            ("holes_fine", "port_fine"), ("curvature_fine", "body_phase_fine")):
            if all(records.get(name, {}).get("status") == "solved" for name in (left, right)):
                label = f"{left}_to_{right}"
                try:
                    cross = None
                    if right == "port_fine":
                        meshes = [(output / pointers[name]["path"]).with_name("mesh_AP_pnc.npz") for name in (left, right)]
                        cross = compare_port_fields(records[left], records[right], *meshes)
                    comparisons[label] = compare_cases(records[left], records[right], cross)
                    gates[label] = comparisons[label]["power_converged"]
                    if right == "body_phase_fine":
                        gates["body_mesh_complex_scattering_converged"] = comparisons[label]["complex_scattering_change_below_0p01"]
                except ValueError as exc:
                    comparisons[label] = {"error": str(exc)}
                    gates[label] = False
    succeeded = all(record["status"] in ("meshed", "solved") for record in records.values()) and all(gates.values())
    main_gates = [f"{name}_{gate}" for name in MAIN for gate in ("geometry", "scattering")]
    main_gates.extend(("holes_coarse_to_holes_middle", "holes_middle_to_holes_fine"))
    power_gates_passed = all(gates.get(name, False) for name in main_gates)
    report = {**AUTHOR, "frequency_hz": FREQUENCY_HZ, "scope": "One-frequency uniform and perforated matched-port controls.",
              "mesh_only": args.mesh_only, "cases": pointers, "comparisons": comparisons, "gates": gates,
              "three_mesh_two_hole_control_passed": (not args.mesh_only and set(MAIN) <= records.keys()
                                                      and all(records[name]["status"] == "solved" for name in MAIN)
                                                      and power_gates_passed),
              "body_mesh_phase_status": ("converged_to_0p01" if gates.get("body_mesh_complex_scattering_converged")
                                         else "not_converged_to_0p01" if "body_mesh_complex_scattering_converged" in gates
                                         else "unresolved" if "body_phase_fine" in records
                                         else "no_targeted_phase_refinement"),
              "passed_requested_checks": succeeded}
    summary = output / f"perforated_validation_{digest(report)[:12]}_AP_pnc.json"
    if not summary.exists():
        write_json(summary, report)
    print(summary)
    if args.verification_dir is not None:
        print(write_verification(args.verification_dir, report, records, identity))
    return 0 if succeeded else 2


if __name__ == "__main__":
    raise SystemExit(main())
