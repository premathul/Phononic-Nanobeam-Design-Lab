"""Compute one 1 GHz repeated-mirror cell. Author: Athul Prem | _AP_pnc."""

from __future__ import annotations

import argparse
from dataclasses import dataclass
import json
from math import isfinite, sin, tau
import os
from pathlib import Path
import sys
from tempfile import TemporaryDirectory

THREADS = dict.fromkeys(("OPENBLAS_NUM_THREADS", "OMP_NUM_THREADS", "MKL_NUM_THREADS", "BLIS_NUM_THREADS"), "1")
if __name__ == "__main__":
    os.environ.update(THREADS)

import numpy as np
from skfem import Dofs, ElementTetP2, ElementVector

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT), str(ROOT / "src")]

from phononic import workflow_AP_pnc as workflow
from phononic.elasticity_AP_pnc import assemble_elasticity
from phononic.geometry_AP_pnc import BeamGeometry, Hole, read_hole_pattern
from phononic.layered_AP_pnc import make_layered_mesh
from phononic.lead_modes_AP_pnc import _end_dof_pairs
from phononic.material_AP_pnc import CubicMaterial
from phononic.port_modes_AP_pnc import assemble_port
from phononic.resumable_scattering_AP_pnc import AUTHOR, _array_digest, _exclusive, _link, _numerical_identity, _read, _verified

STUDY = ROOT / "results/runs/mirror_cell_phase_AP_pnc"
DEVICE_CONFIG = ROOT / "configs/historical_26hole_regional_AP_pnc.json"
SOLVER = ROOT / "tools/bloch_factors_AP_pnc.py"
FREQUENCY_HZ = 1_000_000_000
NM = 1e-9
THICKNESSES_NM = (100, 200)
TARGETS_NM = (225.0, 168.75, 126.5625)
LAYERS = (2, 3)
END_STRIP_NM = 50
TRANSVERSE_CELLS = 8
MAX_BODY_DOFS = 20_000
MAX_FACE_DOFS = 400
RECEIPT = "summary_AP_pnc.json"
TRACE_FILE = "traces_AP_pnc.npz"


@dataclass(frozen=True)
class CellSpec:
    geometry: BeamGeometry
    hole: Hole
    material: CubicMaterial
    inputs: dict


def case_directory(output_root, thickness_nm, target_nm, thickness_layers=2):
    target = str(float(target_nm)).removesuffix(".0").replace(".", "p")
    return Path(output_root) / f"t{thickness_nm}_z{thickness_layers}_edge{target}nm_AP_pnc"


def _case_spec(thickness_nm, target_nm, thickness_layers):
    if thickness_nm not in THICKNESSES_NM or target_nm not in TARGETS_NM or thickness_layers not in LAYERS:
        raise ValueError("Unsupported mirror cell thickness, central target or layer count.")

    configuration = _read(DEVICE_CONFIG)
    csv = ROOT / configuration["geometry"]["pattern"]["csv"]
    pattern = read_hole_pattern(csv)
    holes = sorted(pattern.holes, key=lambda hole: hole.x_m)
    positions = np.array([hole.x_m / NM for hole in holes])
    diameters = np.array([hole.diameter_m / NM for hole in holes])
    if (len(holes) != 26 or not np.allclose(positions, -positions[::-1], rtol=0, atol=1e-9)
            or not np.array_equal(diameters, diameters[::-1])):
        raise ValueError("Expected the centered, mirror-symmetric 26-hole source geometry.")

    repeated = {side: [index for index, hole in enumerate(holes) if hole.region == "Mirror" and sign * hole.x_m > 0]
                for side, sign in (("left", -1), ("right", 1))}
    for side, indices in repeated.items():
        if (len(indices) < 3 or np.any(np.diff(indices) != 1)
                or not np.allclose(np.diff(positions[indices]), 550.0, rtol=0, atol=1e-9)
                or not np.allclose(diameters[indices], 250.0, rtol=0, atol=1e-9)):
            raise ValueError(f"The {side} repeated mirror no longer has a 550 nm pitch and 250 nm diameter.")
    # Remove the last-digit roundoff from converting the source's nm coordinates to SI and back.
    pitch_nm = float(round(np.diff(positions[repeated["left"]])[0], 9))
    diameter_nm = float(round(diameters[repeated["left"][0]], 9))
    width_nm = configuration["geometry"]["width_nm"]
    circumference_facets = configuration["mesh"]["circle_resolution"]
    if width_nm != 1000 or circumference_facets != 32:
        raise ValueError("The reference width or rim faceting differs from the 1 GHz mirror control.")
    hole_link = _link(csv, ROOT)
    if hole_link["sha256"] != pattern.source_sha256:
        raise RuntimeError("The hole table changed while reading its provenance.")
    geometry = BeamGeometry(pitch_nm * NM, width_nm * NM, thickness_nm * NM)
    hole = Hole(geometry.length_m / 2, diameter_nm * NM, "Mirror")
    raw = configuration["material"]
    material = CubicMaterial(C11=raw["C11_Pa"], C12=raw["C12_Pa"], C44=raw["C44_Pa"],
                             rho=raw["density_kg_m3"], rotation=raw["crystal_rotation"])
    inputs = {
        "frequency_hz": FREQUENCY_HZ,
        "geometry": {"period_nm": pitch_nm, "width_nm": width_nm, "thickness_nm": thickness_nm,
                     "hole_center_nm": [pitch_nm / 2, 0], "hole_diameter_nm": diameter_nm,
                     "circle_segments": circumference_facets, "reference_planes_nm": [0, pitch_nm],
                     "context": "one cell from the repeated outer mirror"},
        "mesh": {"parent_central_size_nm": configuration["mesh"]["mesh_size_nm"],
                 "central_target_nm": target_nm, "central_interval_nm": [END_STRIP_NM, pitch_nm - END_STRIP_NM],
                 "end_strip_target_nm": width_nm / TRANSVERSE_CELLS, "transverse_cells": TRANSVERSE_CELLS,
                 "thickness_layers": thickness_layers, "element_order": 2,
                 "max_body_dofs": MAX_BODY_DOFS, "max_face_dofs": MAX_FACE_DOFS,
                 "family": "fixed-strip regional Gmsh parent, nested planar bisection, affine tetrahedra"},
        "material": raw,
        "boundary": "translated Bloch end traces, traction-free outer faces and hole wall",
        "wave_convention": "exp(i*k*x - i*omega*t), right displacement = lambda * left displacement",
        "hole_table": hole_link,
        "source_device_config": _link(DEVICE_CONFIG, ROOT),
        "repeated_outer_source_positions_nm": {
            side: positions[indices].round(9).tolist() for side, indices in repeated.items()},
        "threads": THREADS,
    }
    return CellSpec(geometry, hole, material, inputs)


def _build_mesh(spec):
    layout = spec.inputs["mesh"]
    return make_layered_mesh(
        spec.geometry, [spec.hole], layout["parent_central_size_nm"] * NM,
        port_mesh_size_m=spec.geometry.width_m / TRANSVERSE_CELLS,
        thickness_layers=layout["thickness_layers"], circle_points=spec.inputs["geometry"]["circle_segments"],
        max_port_dofs=MAX_FACE_DOFS, element_order=layout["element_order"],
        lead_mesh_size_m=layout["end_strip_target_nm"] * NM,
        patterned_x_interval_m=(END_STRIP_NM * NM, spec.geometry.length_m - END_STRIP_NM * NM),
        central_max_edge_m=layout["central_target_nm"] * NM,
    )


def build_case_mesh(thickness_nm, target_nm, thickness_layers=2):
    """Regenerate one case mesh with the pinned geometry and mesh controls."""
    return _build_mesh(_case_spec(thickness_nm, target_nm, thickness_layers))


def assemble_case(thickness_nm, target_nm, thickness_layers=2):
    """Rebuild one cell's native P2 operators for an independent check."""
    spec = _case_spec(thickness_nm, target_nm, thickness_layers)
    return assemble_elasticity(_build_mesh(spec), spec.material, max_dofs=MAX_BODY_DOFS)


def _end_mesh(mesh, side):
    faces = mesh.facets[:, mesh.boundaries[side]]
    yz = mesh.p[1:3, faces].transpose(2, 1, 0)
    # Canonical physical ordering independent of global vertex numbering.
    return np.asarray(sorted(tuple(sorted(map(tuple, face))) for face in yz))


def _mesh_metrics(mesh, spec):
    geometry, hole = spec.geometry, spec.hole
    face = mesh.facets[:, mesh.boundary_facets()]
    footprint = face[:, np.all(mesh.p[2, face] == -geometry.thickness_m / 2, axis=0)].T
    xy = mesh.p[:2].T
    edges, incidence = np.unique(np.sort(np.concatenate((footprint[:, [0, 1]], footprint[:, [0, 2]],
                                                          footprint[:, [1, 2]])), axis=1),
                                 axis=0, return_counts=True)
    if np.any(incidence > 2):
        raise RuntimeError("The cell footprint is not conforming.")
    boundary = xy[edges[incidence == 1]]
    outer = (np.all(boundary[:, :, 0] == 0, axis=1)
             | np.all(boundary[:, :, 0] == geometry.length_m, axis=1)
             | np.all(boundary[:, :, 1] == -geometry.width_m / 2, axis=1)
             | np.all(boundary[:, :, 1] == geometry.width_m / 2, axis=1))
    rim = np.asarray(sorted(tuple(sorted(map(tuple, edge))) for edge in boundary[~outer]))
    segments = spec.inputs["geometry"]["circle_segments"]
    if rim.shape != (segments, 2, 2) or np.unique(rim.reshape(-1, 2), axis=0).shape[0] != segments:
        raise RuntimeError("The cell footprint does not retain the expected polygonal mirror hole.")
    radius = hole.diameter_m / 2
    np.testing.assert_allclose(np.linalg.norm(rim - [hole.x_m, 0], axis=2), radius, rtol=1e-10, atol=0)

    directions = xy[footprint[:, 1:]] - xy[footprint[:, :1]]
    areas = abs(np.linalg.det(directions)) / 2
    area = float(areas.sum())
    volume = float(mesh.mapping().detA.sum() / 6)
    expected_area = geometry.length_m * geometry.width_m - segments * radius**2 * sin(tau / segments) / 2
    np.testing.assert_allclose(area, expected_area, rtol=1e-10, atol=0)
    np.testing.assert_allclose(volume, area * geometry.thickness_m, rtol=1e-12, atol=0)

    left, right = (_end_mesh(mesh, side) for side in ("xleft", "xright"))
    np.testing.assert_array_equal(left, right)
    end_cells = spec.inputs["mesh"]["thickness_layers"] * TRANSVERSE_CELLS
    if left.shape != (2 * end_cells, 3, 2):
        raise RuntimeError("The matching end faces differ from the prescribed trace layout.")
    edge_x = xy[edges, 0]
    cut_left, cut_right = END_STRIP_NM * NM, geometry.length_m - END_STRIP_NM * NM
    in_central = np.all((edge_x >= cut_left) & (edge_x <= cut_right), axis=1)
    face_x = xy[footprint, 0]
    strips = xy[footprint[np.all(face_x <= cut_left, axis=1) | np.all(face_x >= cut_right, axis=1)]]
    strips = np.asarray(sorted(tuple(sorted(map(tuple, face))) for face in strips))
    lengths = np.linalg.norm(xy[edges[:, 1]] - xy[edges[:, 0]], axis=1)
    max_central_nm = float(lengths[in_central].max() / NM)
    target_nm = spec.inputs["mesh"]["central_target_nm"]
    if max_central_nm > target_nm * (1 + 1e-12):
        raise RuntimeError(f"Central edge {max_central_nm:g} nm exceeds the requested {target_nm:g} nm.")
    dofs = int(Dofs(mesh, ElementVector(ElementTetP2())).N)
    if dofs > MAX_BODY_DOFS:
        raise RuntimeError(f"Cell has {dofs} displacement DOFs, exceeding {MAX_BODY_DOFS}.")
    return {
        "planar_vertices": int(np.unique(footprint).size), "planar_triangles": len(footprint),
        "tetrahedra": mesh.nelements, "quadratic_vector_dofs": dofs, "end_face_triangles": len(left),
        "end_face_quadratic_vector_dofs": 3 * (2 * TRANSVERSE_CELLS + 1) * (2 * spec.inputs["mesh"]["thickness_layers"] + 1),
        "planar_area_m2": area, "expected_polygonal_area_m2": expected_area, "volume_m3": volume,
        "min_planar_triangle_area_m2": float(areas.min()), "max_central_edge_nm": max_central_nm,
        "max_end_strip_edge_nm": float(lengths[~in_central].max() / NM),
        "frozen_rim_sha256": _array_digest(rim), "translated_end_triangles_sha256": _array_digest(left),
        "fixed_end_strips_sha256": _array_digest(strips),
        "geometry_connectivity_sha256": _array_digest(mesh.p, mesh.t),
    }


def _trace_data(system, material, result, identity):
    period, native_left, native_right = _end_dof_pairs(system)
    np.testing.assert_array_equal(result.left_dofs, native_left)
    np.testing.assert_array_equal(result.right_dofs, native_right)
    port = assemble_port(system, material, max_dofs=MAX_FACE_DOFS)
    ordering = np.argsort(port.left_dofs)
    permutation = ordering[np.searchsorted(port.left_dofs[ordering], native_left)]
    np.testing.assert_array_equal(port.left_dofs[permutation], native_left)
    np.testing.assert_array_equal(port.right_dofs[permutation], native_right)
    yz = system.basis.doflocs[1:3, native_left].T
    components = np.empty(system.basis.N, dtype=int)
    for component, indices in enumerate(system.basis.split_indices()):
        components[indices] = component

    modes = result.modes
    dynamic = result.boundary_dynamic_stiffness
    face_count = result.face_dof_count
    return {
        "inputs_sha256": np.asarray(identity), "frequency_hz": FREQUENCY_HZ, "period_m": period,
        "trace_mass": port.M[np.ix_(permutation, permutation)], "trace_dof_yz_m": yz,
        "trace_dof_components": components[native_left], "left_dofs": native_left, "right_dofs": native_right,
        "traces": np.column_stack([mode.left_trace for mode in modes]) if modes else np.empty((len(native_left), 0), complex),
        "right_traces": np.column_stack([mode.right_trace for mode in modes]) if modes else np.empty((len(native_right), 0), complex),
        "root_indices": np.asarray([mode.root_index for mode in modes], dtype=int),
        "lambda_factors": np.asarray([mode.multiplier for mode in modes], dtype=complex),
        "phase_rad": np.asarray([mode.phase_rad for mode in modes], dtype=float),
        "k_rad_m": np.asarray([mode.wave_number_rad_m for mode in modes], dtype=float),
        "signed_flux_w": np.asarray([mode.power_w for mode in modes], dtype=float),
        "directions": np.asarray([mode.direction for mode in modes], dtype=str),
        "homogeneous_roots": result.homogeneous_roots,
        "LL": dynamic[:face_count, :face_count], "LR": dynamic[:face_count, face_count:],
        "RL": dynamic[face_count:, :face_count], "RR": dynamic[face_count:, face_count:],
    }


def run_case(thickness_nm, target_nm, thickness_layers=2, *, output_root=STUDY, mesh_only=False):
    """Verify or compute one cell point and publish its evidence as one directory."""
    spec = _case_spec(thickness_nm, target_nm, thickness_layers)
    if mesh_only:
        with workflow.measured_resources() as resources:
            mesh = _build_mesh(spec)
            metrics = _mesh_metrics(mesh, spec)
        return {**AUTHOR, "status": "mesh_ready", "inputs": spec.inputs, "mesh": metrics, "resources": resources}

    from tools.bloch_factors_AP_pnc import solve_bloch_factors

    source = workflow.source_identity()
    source["source_sha256"].update({p.relative_to(ROOT).as_posix(): _link(p, ROOT)["sha256"]
                                    for p in (Path(__file__).resolve(), SOLVER)})
    inputs = {**spec.inputs, "numerical_identity": _numerical_identity(source)}
    identity = workflow.digest(inputs)
    directory = case_directory(output_root, thickness_nm, target_nm, thickness_layers).resolve()
    directory.parent.mkdir(parents=True, exist_ok=True)
    with _exclusive(directory.parent / f".{directory.name}.lock"):
        if directory.exists():
            saved = _read(directory / RECEIPT)
            if saved["inputs_sha256"] != identity or workflow.digest(saved["inputs"]) != identity:
                raise ValueError(f"Existing mirror cell inputs differ in {directory}.")
            if saved["status"] not in ("accepted", "unresolved"):
                raise RuntimeError(f"Earlier failure retained in {directory / RECEIPT}.")
            if _verified(saved["trace_arrays"], directory) != directory / TRACE_FILE:
                raise ValueError(f"Existing mirror cell trace path differs in {directory}.")
            return {**saved, "output_directory": str(directory), "reused": True}

        with TemporaryDirectory(prefix=f".{directory.name}.", dir=directory.parent) as temporary:
            stage = Path(temporary)
            record = {**AUTHOR, "schema_version": 1, "inputs": inputs, "inputs_sha256": identity,
                      "source_git_commit": source["git_commit"]}
            with workflow.measured_resources() as resources:
                try:
                    mesh = _build_mesh(spec)
                    record["mesh"] = _mesh_metrics(mesh, spec)
                    system = assemble_elasticity(mesh, spec.material, max_dofs=MAX_BODY_DOFS)
                    result = solve_bloch_factors(system, FREQUENCY_HZ,
                                                 max_face_dofs=MAX_FACE_DOFS, max_body_dofs=MAX_BODY_DOFS)
                    record.update(_solver_summary(result))
                    if result.boundary_dynamic_stiffness is None:
                        raise RuntimeError("Bloch cell condensation failed. See solver diagnostics.")
                    arrays = _trace_data(system, spec.material, result, identity)
                    np.savez_compressed(stage / TRACE_FILE, **arrays)
                    record.update(status="accepted" if result.status == "resolved" else "unresolved",
                                  trace_arrays=_link(stage / TRACE_FILE, stage),
                                  trace_conventions={
                                      "mass": "consistent y-z section displacement mass in kg/m",
                                      "components": "0, 1, 2 denote lab x, y, z displacement",
                                      "scale": "unit Euclidean norm of the left trace in metres",
                                      "flux": "signed right-face power in watts at the saved amplitude; positive toward +x",
                                      "condensed_blocks": "dynamic stiffness K - omega^2 M in N/m, outward end reactions",
                                  })
                except (RuntimeError, ValueError, AssertionError, np.linalg.LinAlgError) as exc:
                    record.update(status="failed", error=str(exc))
            workflow.write_json(stage / RECEIPT, {**record, "resources": resources})
            stage.rename(directory)

    if record["status"] == "failed":
        raise RuntimeError(f"Mirror cell failed. See {directory / RECEIPT}: {record['error']}")
    return {**record, "resources": resources, "output_directory": str(directory), "reused": False}


def _solver_summary(result):
    def finite(value):
        return value if isfinite(value) else None

    return {
        "solver_status": result.status, "solver_diagnostics": result.diagnostics,
        "modes": [{"root_index": mode.root_index,
                   "multiplier": [mode.multiplier.real, mode.multiplier.imag],
                   "phase_rad": mode.phase_rad, "direction": mode.direction, "signed_flux_w": finite(mode.power_w),
                   "relative_power": finite(mode.relative_power), "interior_residual": finite(mode.interior_residual),
                   "traction_residual": finite(mode.traction_residual), "radial_error": finite(mode.radial_error),
                   "chordal_uncertainty": finite(mode.chordal_uncertainty)} for mode in result.modes],
    }


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--thickness-nm", type=int, required=True, choices=THICKNESSES_NM)
    parser.add_argument("--target-nm", type=float, required=True, choices=TARGETS_NM)
    parser.add_argument("--thickness-layers", type=int, default=2, choices=LAYERS)
    parser.add_argument("--output-root", type=Path, default=STUDY)
    parser.add_argument("--mesh-only", action="store_true")
    args = parser.parse_args(argv)
    result = run_case(args.thickness_nm, args.target_nm, args.thickness_layers,
                      output_root=args.output_root, mesh_only=args.mesh_only)
    if args.mesh_only:
        print(json.dumps(result, indent=2))
    else:
        reused = " (reused)" if result["reused"] else ""
        print(f"{result['status']}{reused}: {len(result['modes'])} near-unit modes in {result['output_directory']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
