"""Check one split of an accepted matched-port device mesh. Athul Prem | _AP_pnc."""

from __future__ import annotations

import argparse
import csv
import json
from math import tau
import os
from pathlib import Path
import sys

if __name__ == "__main__":
    os.environ.update(dict.fromkeys(("OPENBLAS_NUM_THREADS", "OMP_NUM_THREADS", "MKL_NUM_THREADS",
                                     "BLIS_NUM_THREADS", "NUMEXPR_NUM_THREADS"), "1"))

import numpy as np
from scipy.linalg import get_blas_funcs

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT / "src"), str(ROOT)]

from phononic import workflow_AP_pnc as workflow
from phononic.linear_lgmres_AP_pnc import _original_residual
from phononic.port_modes_AP_pnc import PortModes, _flux_gram
from phononic.resumable_scattering_AP_pnc import (
    AUTHOR, THREADS, _array_digest, _exclusive, _link, _numerical_identity, _read, _sha, _verified,
)
from phononic.scattering_AP_pnc import solve_scattering
from tools import compare_scattering_AP_pnc, map_scattering_AP_pnc as native_map, split_device_response_AP_pnc as split

DEFAULT_MAP = ROOT / "results/runs/physical_thickness_device_AP_pnc/thin100_central400_edge225_rim32"
CUTS_NM = (5745.0, 10905.0)
SAME_MESH_LIMIT = 1e-8
POWER_LIMIT = 0.01


def _read_rows(path):
    with path.open(newline="", encoding="utf-8") as stream:
        return list(csv.DictReader(stream))


def _reference(directory, summary, port):
    """Read the saved basis, including all outgoing evanescent roots in native order."""
    system = port.elastic_system
    basis, mesh = system.basis, system.mesh
    with np.load(directory / native_map.FIELDS, allow_pickle=False) as archive:
        for key, expected in {
            "points_m": mesh.p.T, "tetrahedra": mesh.t.T,
            "element_dofs": basis.element_dofs, "nodal_dofs": basis.nodal_dofs,
            "dof_coordinates_m": basis.doflocs.T,
            "port_left_dofs": port.left_dofs, "port_right_dofs": port.right_dofs,
        }.items():
            if not np.array_equal(archive[key], expected):
                raise ValueError(f"Saved field and cached operators use different mesh or DOF ordering ({key}).")
        components = archive["dof_component"]
        for component, rows in enumerate(basis.split_indices()):
            if np.any(components[rows] != component):
                raise ValueError("Saved field uses a different displacement component order.")
        keys = ("displacements_per_sqrt_w", "incident_powers_w", "incident_mode_indices",
                "port_wavenumbers_rad_m", "port_displacements", "port_axial_forces",
                "reflection_all_mode_indices", "transmission_all_mode_indices", "frequencies_hz",
                "reflection_mode_indices", "transmission_mode_indices", "reflection_amplitudes",
                "transmission_amplitudes", "reflection_amplitudes_full", "transmission_amplitudes_full")
        data = {key: archive[key] for key in keys}
    mode_rows = _read_rows(directory / "port_modes_AP_pnc.csv")
    wave = data["port_wavenumbers_rad_m"]
    left, right = (data[f"{kind}_all_mode_indices"] for kind in ("reflection", "transmission"))
    count, frequency = summary["num_incident_channels"], summary["frequency_hz"]
    if (not all(np.isfinite(value).all() for value in data.values())
            or data["displacements_per_sqrt_w"].shape != (basis.N, count)
            or data["frequencies_hz"].tolist() != [frequency] * count
            or wave.shape != (2 * port.basis.N,)
            or len(mode_rows) != wave.size
            or any(data[key].shape != (port.basis.N, wave.size) for key in ("port_displacements", "port_axial_forces"))
            or [int(row["port_index"]) for row in mode_rows] != list(range(wave.size))
            or not np.array_equal(np.sort(np.r_[left, right]), np.arange(wave.size))
            or left.shape != port.left_dofs.shape or right.shape != port.right_dofs.shape
            or any(row["propagating"] not in ("True", "False") or row["direction"] not in ("+x", "-x") for row in mode_rows)
            or [complex(float(row["k_real_rad_m"]), float(row["k_imag_rad_m"])) for row in mode_rows] != wave.tolist()
            or any(mode_rows[root]["direction"] != sign for indices, sign in ((left, "-x"), (right, "+x")) for root in indices)):
        raise ValueError("Saved fields and mode table do not describe the same complete directed basis.")
    opens = np.array([row["propagating"] == "True" for row in mode_rows])
    powers = np.array([float(row["signed_power_W"]) for row in mode_rows])
    residuals = np.array([float(row["relative_residual"]) for row in mode_rows])
    if (not np.isfinite(powers).all() or not np.isfinite(residuals).all() or np.max(residuals) >= SAME_MESH_LIMIT
            or not np.array_equal(data["reflection_mode_indices"], left[opens[left]])
            or not np.array_equal(data["transmission_mode_indices"], right[opens[right]])
            or not np.array_equal(data["incident_mode_indices"], right[opens[right]])
            or data["incident_mode_indices"].tolist() != summary["incident_mode_indices"]
            or not np.allclose(data["incident_powers_w"], 1.0, atol=SAME_MESH_LIMIT, rtol=0)):
        raise ValueError("Saved incident or outgoing channel identities, powers or modal residuals differ.")
    modes = PortModes(port, frequency, wave, data["port_displacements"], data["port_axial_forces"],
                      right, left, opens, powers, residuals)
    return modes, data


def _partition_info(system, partition, protected_m):
    mesh, dofs, labels = system.mesh, system.basis.doflocs, partition.element_regions
    vertices = mesh.p[0, mesh.t]
    protected_elements = (vertices.max(axis=0) >= protected_m[0]) & (vertices.min(axis=0) <= protected_m[1])
    if np.any((labels[protected_elements] == 0) | (labels[protected_elements] == len(partition.interior_dofs) - 1)):
        raise ValueError("The partition moved a tetrahedron touching the protected interval into an outer segment.")
    neighbor_regions = labels[mesh.f2t[:, partition.frontier_facets]]
    min_region, max_region = neighbor_regions.min(axis=0), neighbor_regions.max(axis=0)
    if np.any(max_region - min_region != 1):
        raise ValueError("An interface facet crosses more than one requested cut.")
    seams = []
    for index, cut in enumerate(partition.cut_planes_m):
        facets = partition.frontier_facets[(min_region == index) & (max_region == index + 1)]
        if not facets.size:
            raise ValueError(f"No shared facets found at the nominal cut x={cut:g} m.")
        x = mesh.p[0, mesh.facets[:, facets]]
        seams.append({"nominal_cut_x_m": cut, "frontier_facets": int(facets.size),
                      "actual_frontier_x_range_m": [float(x.min()), float(x.max())]})
    separator = partition.separator_dofs
    return {"tetrahedra": int(mesh.nelements), "body_dofs": int(system.basis.N),
            "stiffness_nonzeros": int(system.K.nnz), "mass_nonzeros": int(system.M.nnz),
            "protected_interval_x_m": protected_m, "protected_tetrahedra": int(np.sum(protected_elements)),
            "seams": seams, "separator_dofs": int(separator.size),
            "separator_x_range_m": [float(dofs[0, separator].min()), float(dofs[0, separator].max())],
            "regions": [{"tetrahedra": int(np.sum(labels == i)), "interior_dofs": int(interior.size),
                         "interior_stiffness_nonzeros": int(system.K[interior, :][:, interior].nnz)}
                        for i, interior in enumerate(partition.interior_dofs)]}


def _residual_check(operator, rhs, fields, norm):
    nrm2 = get_blas_funcs("nrm2", (rhs,))
    rows = []
    for column, (x, b) in enumerate(zip(fields.T, rhs.T, strict=True)):
        row = {"column": column}
        _, _, row["passed"] = _original_residual(operator, x, b, nrm2, nrm2(b), float(np.max(abs(b))),
                                                  norm, native_map.RESIDUAL_LIMIT, row)
        rows.append(row)
    return {"channels": rows, "relative_and_backward_limit": native_map.RESIDUAL_LIMIT,
            "passed": all(row["passed"] for row in rows)}


def _coherent_flux_error(system, modes, result):
    omega = tau * modes.frequency_hz
    left, right = modes.port.left_dofs, modes.port.right_dofs
    x = result.displacements
    left_force = system.K[left] @ x - omega**2 * (system.M[left] @ x)
    right_force = system.K[right] @ x - omega**2 * (system.M[right] @ x)
    incident_u = modes.displacements[:, result.incident_mode_indices]
    incident_force = -modes.axial_forces[:, result.incident_mode_indices]
    incident = -_flux_gram(incident_u, incident_force, omega)
    reflect = _flux_gram(x[left] - incident_u, left_force - incident_force, omega)
    transmit = _flux_gram(x[right], right_force, omega)
    scale = np.sqrt(result.incident_powers_w[:, None] * result.incident_powers_w)
    total = _flux_gram(x[left], left_force, omega) + transmit
    return {"power_closure": float(np.linalg.norm((reflect + transmit - incident) / scale, ord=2)),
            "net_outward_flux": float(np.linalg.norm(total / scale, ord=2))}


def _compare(modes, candidate, reference, rows):
    metrics = ("reflection", "transmission", "power_balance", "reflection_flux", "transmission_flux", "flux_power_balance")
    counts = len(candidate.incident_mode_indices)
    if [int(row["incident_mode_order"]) for row in rows] != list(range(1, counts + 1)):
        raise ValueError("Saved power table uses a different incident-drive order.")
    identity_keys = ("incident_mode_indices", "reflection_mode_indices", "transmission_mode_indices",
                     "reflection_all_mode_indices", "transmission_all_mode_indices")
    if any(not np.array_equal(getattr(candidate, key), reference[key]) for key in identity_keys):
        raise ValueError("Reconstructed response uses different native incident or outgoing roots.")
    drive_rows = []
    max_change = 0.0
    for column, root in enumerate(candidate.incident_mode_indices):
        row = {"drive": column + 1, "incident_root": int(root)}
        for kind in ("reflection", "transmission"):
            values = getattr(candidate, f"{kind}_amplitudes")[:, column]
            old = reference[f"{kind}_amplitudes"][:, column]
            delta = values - old
            max_change = max(max_change, float(np.max(abs(delta))))
            row[f"{kind}_channels"] = [
                {"outgoing_root": int(outgoing), "saved_real": float(base.real), "saved_imag": float(base.imag),
                 "delta_real": float(change.real), "delta_imag": float(change.imag), "delta_absolute": float(abs(change))}
                for outgoing, base, change in zip(getattr(candidate, f"{kind}_mode_indices"), old, delta, strict=True)
            ]
            trace_delta = (getattr(candidate, f"{kind}_amplitudes_full")[:, column]
                           - reference[f"{kind}_amplitudes_full"][:, column])
            row[f"{kind}_all_outgoing_maximum_complex_change"] = float(np.max(abs(trace_delta)))
        row["power"] = {key: {"reconstructed": float(getattr(candidate, key)[column]), "saved": float(rows[column][key])}
                        for key in metrics}
        for values in row["power"].values():
            values["delta"] = values["reconstructed"] - values["saved"]
        row["net_outward_power_ratio"] = float(candidate.net_outward_power_ratio[column])
        drive_rows.append(row)
    max_power_change = max(abs(values["delta"]) for row in drive_rows for values in row["power"].values())
    x = reference["displacements_per_sqrt_w"]
    field_change = np.linalg.norm(candidate.displacements - x, axis=0) / np.linalg.norm(x, axis=0)
    power_errors = {"modal_power_closure": float(np.max(abs(candidate.power_balance))),
                    "nodal_flux_closure": float(np.max(abs(candidate.flux_power_balance))),
                    "net_outward_flux": float(np.max(abs(candidate.net_outward_power_ratio))),
                    "modal_vs_nodal_flux": float(max(np.max(abs(candidate.reflection - candidate.reflection_flux)),
                                                     np.max(abs(candidate.transmission - candidate.transmission_flux)))),
                    "coherent_nodal_flux": _coherent_flux_error(modes.port.elastic_system, modes, candidate)}
    closure = power_errors["coherent_nodal_flux"]
    power_pass = max(*(value for key, value in power_errors.items() if key != "coherent_nodal_flux"),
                     *closure.values()) < POWER_LIMIT
    return {"phase_gauge": "Exact saved complex SAFE columns and ordered root IDs at both original physical ends.",
            "incident_powers_w": candidate.incident_powers_w.tolist(), "drives": drive_rows,
            "maximum_complex_coefficient_change": max_change, "maximum_power_or_flux_change": max_power_change,
            "relative_displacement_l2_changes": field_change.tolist(),
            "same_mesh_comparison_limit": SAME_MESH_LIMIT, "power_error_limit": POWER_LIMIT,
            "power_errors": power_errors,
            "passed": power_pass and max(max_change, max_power_change, float(np.max(field_change))) < SAME_MESH_LIMIT}


def _check_point(directory, summary, shared, cuts_m, rhs_block_size, preflight):
    port = shared.port
    system = port.elastic_system
    ends = [float(system.mesh.p[0, :].min()), float(system.mesh.p[0, :].max())]
    if not np.allclose(ends, shared.record["reference_planes_x_m"], atol=1e-18, rtol=0):
        raise ValueError("Cached mesh does not use the original physical reference planes.")
    protected_m = (cuts_m[0], cuts_m[-1])
    partition = split.build_axial_partition(system, cuts_m, protected_interval_m=protected_m)
    partition_info = _partition_info(system, partition, protected_m)
    modes, reference = _reference(directory, summary, port)
    rows = _read_rows(directory / "scattering_AP_pnc.csv")
    equation_reference = summary["checkpointed_scattering"]["original_equations"]
    equations, solver = {}, {}

    def solve(operator, rhs):
        norm = float(np.asarray(abs(operator).sum(axis=1)).max())
        hashes = {"operator_csc_sha256": _array_digest(operator.data, operator.indices, operator.indptr),
                  "rhs_sha256": _array_digest(rhs)}
        equations.update({"reconstructed_equation_hashes": hashes,
                          "saved_equation_hashes": {key: equation_reference[key] for key in hashes},
                          "byte_identical_to_saved": {key: value == equation_reference[key] for key, value in hashes.items()},
                          "operator_nonzeros": int(operator.nnz)})
        saved_field = reference["displacements_per_sqrt_w"]
        equations["saved_field_in_reconstructed_equations"] = _residual_check(operator, rhs, saved_field, norm)
        if not equations["saved_field_in_reconstructed_equations"]["passed"]:
            raise ValueError("The archived displacement does not satisfy the reconstructed original equations.")
        if preflight:
            return saved_field
        field = split.solve_partitioned_operator(operator, rhs, partition, rhs_block_size=rhs_block_size, report=solver)
        equations["split_field_in_reconstructed_equations"] = _residual_check(operator, rhs, field, norm)
        return field

    result = {"partition": partition_info, "original_equations": equations, "split_solver": solver,
              "native_port": {"trace_dofs_per_end": int(port.basis.N),
                              "outgoing_roots_per_end": int(modes.negative_indices.size),
                              "propagating_roots_per_end": int(modes.negative_indices[modes.propagating[modes.negative_indices]].size),
                              "reference_planes_x_m": shared.record["reference_planes_x_m"],
                              "basis_sha256": _array_digest(modes.displacements, modes.axial_forces, modes.wavenumbers_rad_m,
                                                            modes.negative_indices, modes.positive_indices)},
              "native_measured_scattering_resources": summary["scattering_resources"]}
    try:
        candidate = solve_scattering(system, modes, modes.frequency_hz, linear_solver=solve)
        comparison = _compare(modes, candidate, reference, rows)
        result["same_mesh_comparison"] = comparison
        passed = comparison["passed"] and (preflight or equations["split_field_in_reconstructed_equations"]["passed"])
        result["status"] = ("ready" if preflight else "accepted") if passed else "unresolved"
    except (ArithmeticError, RuntimeError, ValueError) as exc:
        result.update(status="failed", failure={"type": type(exc).__name__, "message": str(exc)})
    return result


def run_saved_point(map_directory, frequency_hz, cuts_nm, rhs_block_size, output=None):
    """Verify a native point and publish an immutable split receipt, or preflight it without solving."""
    directory = Path(map_directory).resolve()
    state = native_map.read_map_checkpoint(directory)
    points = [point for point in state["points"] if point["frequency_GHz"] * 1e9 == frequency_hz and point["status"] == "accepted"]
    if len(points) != 1:
        raise ValueError("Choose exactly one accepted native frequency in hertz.")
    if len(cuts_nm) < 2 or not np.isfinite(cuts_nm).all() or np.any(np.diff(cuts_nm) <= 0) or rhs_block_size < 1:
        raise ValueError("At least two increasing finite cut coordinates and a positive RHS block size are required.")
    config = state["inputs"]["configuration"]
    point = points[0]
    field_directory, summary = native_map._read_accepted(point, config, state["shared"], directory)
    if (summary["incident_side"] != "left" or summary["configuration"]["mesh"]["element_order"] != 2
            or summary["hole_count"] != 26):
        raise ValueError("A left-incident 26-hole quadratic device point is required.")
    cuts_m = tuple(float(cut) * 1e-9 for cut in cuts_nm)
    holes = summary["hole_pattern_provenance"]["holes"]
    if any(abs(cut - hole["x_m"]) <= hole["diameter_m"] / 2 for cut in cuts_m for hole in holes):
        raise ValueError("Each nominal cut must be in a solid gap between holes.")
    source = workflow.source_identity()
    for filename in (__file__, split.__file__, native_map.__file__, compare_scattering_AP_pnc.__file__):
        path = Path(filename).resolve()
        source["source_sha256"][path.relative_to(ROOT).as_posix()] = _sha(path)
    inputs = {"frequency_hz": frequency_hz, "nominal_cuts_nm": list(cuts_nm), "rhs_block_size": rhs_block_size,
              "reference_map_checkpoint": _link(directory / native_map.CHECKPOINT, directory),
              "reference_point_summary": point["accepted"]["summary"],
              "reference_configuration": summary["configuration"],
              "shared_artifacts": state["shared"]["artifacts"],
              "reference_outputs_sha256": summary["output_sha256"],
              "reference_map_input_sha256": state["inputs_sha256"],
              "reference_map_source_git_commit": state["source_git_commit"],
              "reference_map_numerical_identity": state["inputs"]["numerical_identity"],
              "reference_solve_implementation": summary["implementation"],
              "reference_shared_assembly_provenance": state.get("preparation_reuse"),
              "numerical_identity": _numerical_identity(source), "threads": THREADS}
    identity = workflow.digest(inputs)

    def evaluate(preflight):
        with workflow.measured_resources() as resources:
            shared = native_map._restore_shared(state["shared"], config, directory)
            result = _check_point(field_directory, summary, shared, cuts_m, rhs_block_size, preflight)
        return {**AUTHOR, "schema_version": 1, "map_directory": str(directory), "source_git_commit": source["git_commit"],
                "scope": "Same discrete mesh, operators, SAFE modes and physical end planes. Finite algebraic equivalence check.",
                "inputs": inputs, "inputs_sha256": identity, "resources": resources,
                "result": result, "result_sha256": workflow.digest(result)}

    if output is None:
        return {**evaluate(preflight=True), "reused": False}

    output = Path(output).resolve()
    if output.is_relative_to(directory):
        raise ValueError("The split receipt must be outside the original native map directory.")
    output.parent.mkdir(parents=True, exist_ok=True)
    with _exclusive(output.with_name(f".{output.name}.lock")):
        if output.exists():
            previous = _read(output)
            if (previous["inputs_sha256"] != identity or workflow.digest(previous["inputs"]) != identity
                    or previous["result_sha256"] != workflow.digest(previous["result"])):
                raise ValueError("The existing split receipt changed or uses different inputs.")
            for artifact in inputs["shared_artifacts"].values():
                _verified(artifact, directory)
            return {**previous, "reused": True}
        record = evaluate(preflight=False)
        workflow.write_json(output, record)
    return {**record, "reused": False}


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--map-dir", type=Path, default=DEFAULT_MAP, help="Accepted native map directory.")
    parser.add_argument("--frequency-hz", type=int, default=1_000_000_000, help="One accepted map frequency in hertz.")
    parser.add_argument("--cuts-nm", type=float, nargs="+", default=CUTS_NM,
                        help="Ascending nominal cuts in nm. The first and last bound the protected interval.")
    parser.add_argument("--rhs-block-size", type=int, default=8, help="Interface right-hand sides per factor solve.")
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument("--preflight", action="store_true", help="Verify pins, partition and the original field without solving.")
    mode.add_argument("--output", type=Path, help="Path for the immutable split-solve receipt.")
    args = parser.parse_args(argv)
    receipt = run_saved_point(args.map_dir, args.frequency_hz, args.cuts_nm, args.rhs_block_size, args.output)
    result = receipt["result"]
    view = {"status": result["status"], "reused": receipt["reused"], "resources": receipt["resources"]}
    if args.preflight:
        view.update(partition=result["partition"], original_equations=result["original_equations"],
                    native_port=result["native_port"], native_measured_scattering_resources=result["native_measured_scattering_resources"])
    else:
        view["output"] = str(args.output.resolve())
    if "failure" in result:
        view["failure"] = result["failure"]
    print(json.dumps(view, indent=2, allow_nan=False))
    return 0 if result["status"] in ("accepted", "ready") else 1


if __name__ == "__main__":
    raise SystemExit(main())
