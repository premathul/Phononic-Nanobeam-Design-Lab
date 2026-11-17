"""Independently replay the saved 0.1 GHz h80 z6 device point. Athul Prem | _AP_pnc."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
from pathlib import Path, PurePosixPath
import platform
from time import monotonic

THREADS = dict.fromkeys(("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS",
                         "BLIS_NUM_THREADS", "NUMEXPR_NUM_THREADS"), "1")
if __name__ == "__main__":
    os.environ.update(THREADS)

import numpy as np
import psutil
import scipy
from scipy.sparse import load_npz


ROOT = Path(__file__).resolve().parents[1]
RUNS = ROOT / "results/runs"
BODY = RUNS / "full_device_resource_profile/lawrence_thickness_AP_pnc"
INVENTORY = RUNS / "port_frequency_inventory_AP_pnc/port-inventory-1335406"
BODY_STAGE_SHA = "95c1cfde8d51d77ab74fe9d2c424c1eda0822556a30664cbe581999f97e0ea5f"
TRACE_SHA = "3329f7c51b36d61862ee44edecca35beddc73721e58a616b7685e58b383655be"
MODES_SHA = "8e5f8bb414b96b73cbaf82910d78e26a3cfc6d64b166fdc8164a55374638ea84"
FREQUENCY, DOFS, PORT_DOFS = 100_000_000, 616239, 975
OMEGA = 2 * np.pi * FREQUENCY
PLANES = (0.0, 16.65e-6)
INCIDENT = (1948, 1916, 1940, 1873)
RETURNING = (1949, 1917, 1941, 1874)
TARGET, RESIDUAL, POWER = 1e-10, 1e-9, 0.01
FLUX_GRAM_LIMIT_W = 1e-7
POWER_CHECKS = ("modal_balance", "direct_balance", "net_outward",
                "modal_vs_direct_reflection", "modal_vs_direct_transmission")


def require(condition, message):
    if not condition:
        raise ValueError(message)


def file_sha(path):
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def read_receipt(path, digest):
    require(file_sha(path) == digest, f"Receipt SHA-256 differs: {path}")
    return json.loads(path.read_bytes())


def check_link(path, link):
    name = PurePosixPath(link.get("path", link.get("file"))).name
    require(path.name == name and path.stat().st_size == link["bytes"] and file_sha(path) == link["sha256"],
            f"Artifact name, size or SHA-256 differs: {path}")
    return path


def array_sha(*arrays):
    digest = hashlib.sha256()
    for array in arrays:
        digest.update(str((array.shape, array.dtype.str)).encode("ascii"))
        digest.update(memoryview(np.ascontiguousarray(array)).cast("B"))
    return digest.hexdigest()


def row_norm_csc(matrix):
    row_sums = np.zeros(matrix.shape[0])
    for start in range(0, matrix.nnz, 2**20):
        part = slice(start, start + 2**20)
        row_sums += np.bincount(matrix.indices[part], weights=np.abs(matrix.data[part]), minlength=len(row_sums))
    return float(row_sums.max())


def check_original_equations(matrix, rhs, vectors):
    residual = matrix @ vectors - rhs
    rhs_norm = np.linalg.norm(rhs, axis=0)
    matrix_norm = row_norm_csc(matrix)
    scale = matrix_norm * np.max(abs(vectors), axis=0) + np.max(abs(rhs), axis=0)
    require(np.all(np.isfinite(rhs_norm) & (rhs_norm > 0) & np.isfinite(scale) & (scale > 0)),
            "Original residual has an invalid normalization.")
    relative = np.linalg.norm(residual, axis=0) / rhs_norm
    backward = np.max(abs(residual), axis=0) / scale
    require(np.all(relative <= TARGET) and np.all(relative <= RESIDUAL) and np.all(backward <= RESIDUAL),
            f"Original equation replay failed: relative={relative}, backward={backward}.")
    return relative, backward, rhs_norm, matrix_norm


def work_power(displacement, outward_force, omega):
    """Outward energy per second is minus the time average of traction work."""
    velocity = -1j * omega * displacement
    return -0.5 * np.sum(np.real(velocity.conj() * outward_force), axis=0)


def check_power(incident_power, modal_r, modal_t, direct_r, direct_t, net):
    require(np.allclose(incident_power, 1.0, rtol=0, atol=FLUX_GRAM_LIMIT_W), "Incident normalization is not +1 W inward.")
    discrepancies = np.stack((1 - modal_r - modal_t, 1 - direct_r - direct_t, net,
                              modal_r - direct_r, modal_t - direct_t))
    require(np.all(abs(discrepancies) < POWER), "At least one of the five physical-power checks failed.")
    return np.max(abs(discrepancies), axis=1)


def work_gram(displacement, force):
    """Hermitian flux form for forces with a common normal and exp(-i omega t)."""
    velocity = -1j * OMEGA * displacement
    work = velocity.conj().T @ force
    return -0.25 * (work + work.conj().T)


def check_coherent_power(reflection, transmission, incident, reflected_basis, left_total, right_total,
                         *, incident_roots=None, returning_roots=None):
    """Check left-incident columns in their archived propagating basis."""
    incident_roots = INCIDENT if incident_roots is None else tuple(incident_roots)
    returning_roots = RETURNING if returning_roots is None else tuple(returning_roots)
    u_in, f_in = incident
    size = reflection.shape[1]
    identity = np.eye(size)
    u_reflected, f_reflected = reflected_basis
    # Both supplied left-side force blocks use the -x normal. Use +x for the
    # signed Gram to retain the cross terms of both propagating directions.
    signed_gram = work_gram(np.column_stack((u_in, u_reflected)), -np.column_stack((f_in, f_reflected)))
    incoming_gram = signed_gram[:size, :size]
    grams = {"incident": incoming_gram, "reflected": -signed_gram[size:, size:], "transmitted": incoming_gram}
    gram_errors = {}
    for name, gram in grams.items():
        error = gram - identity
        diagonal = np.diag(error)
        diagonal_error = float(np.max(abs(diagonal)))
        off_diagonal_error = float(np.max(abs(error - np.diag(diagonal))))
        require(max(diagonal_error, off_diagonal_error) <= FLUX_GRAM_LIMIT_W,
                f"The {name} propagating flux Gram differs from the unit-power basis.")
        gram_errors[name] = {"maximum_diagonal_error_w": diagonal_error,
                             "maximum_off_diagonal_magnitude_w": off_diagonal_error}
    cross_error = float(np.max(abs(signed_gram[:size, size:])))
    require(cross_error <= FLUX_GRAM_LIMIT_W, "The signed propagating flux Gram has a nonzero cross-direction block.")

    defect = reflection.conj().T @ reflection + transmission.conj().T @ transmission - identity
    eigenvalues = np.linalg.eigvalsh(defect)
    spectral_norm = float(np.max(abs(eigenvalues)))
    require(spectral_norm < POWER, f"The left-incident coherent power defect exceeds the {POWER} gate: {spectral_norm}.")

    order = np.arange(1, size + 1)
    weights = order + 1j * np.where(order % 2, 1, -2)
    weights /= np.linalg.norm(weights)
    u_incident, f_incident = u_in @ weights, f_in @ weights
    u_left, f_left = (columns @ weights for columns in left_total)
    u_right, f_right = (columns @ weights for columns in right_total)
    incident_power = -work_power(u_incident, f_incident, OMEGA)
    require(abs(incident_power - 1.0) <= FLUX_GRAM_LIMIT_W, "Coherent incident normalization is not +1 W inward.")
    direct_r = work_power(u_left - u_incident, f_left - f_incident, OMEGA) / incident_power
    direct_t = work_power(u_right, f_right, OMEGA) / incident_power
    net = (work_power(u_left, f_left, OMEGA) + work_power(u_right, f_right, OMEGA)) / incident_power
    modal_r, modal_t = (np.linalg.norm(block @ weights) ** 2 for block in (reflection, transmission))
    values = (incident_power, modal_r, modal_t, direct_r, direct_t, net)
    errors = check_power(*(np.atleast_1d(value) for value in values))
    return {"scope": "Left-incident propagating reflection and transmission in the original SAFE basis",
            "incident_roots": incident_roots, "reflected_roots": returning_roots, "transmitted_roots": incident_roots,
            "flux_gram_entry_limit_w": FLUX_GRAM_LIMIT_W, "flux_gram_errors": gram_errors,
            "flux_gram_cross_direction_maximum_w": cross_error,
            "defect_matrix_real": defect.real.tolist(), "defect_matrix_imag": defect.imag.tolist(),
            "defect_eigenvalues": eigenvalues.tolist(), "coherent_coefficient_power_defect_spectral_norm": spectral_norm,
            "coherent_power_limit": POWER,
            "direct_superposition": {"incident_coefficients_real": weights.real.tolist(),
                "incident_coefficients_imag": weights.imag.tolist(), "incident_coefficient_norm_squared": float(np.vdot(weights, weights).real),
                "incident_power_w": float(incident_power), "modal_reflection": float(modal_r),
                "modal_transmission": float(modal_t), "direct_reflection": float(direct_r),
                "direct_transmission": float(direct_t), "net_outward_power_ratio": float(net),
                "power_errors": dict(zip(POWER_CHECKS, errors.tolist(), strict=True))}}


def body_action(path, vectors, *, expected_dofs=None):
    expected_dofs = DOFS if expected_dofs is None else expected_dofs
    matrix = load_npz(path)
    require(matrix.shape == (expected_dofs, expected_dofs) and matrix.dtype == np.float64,
            f"Saved body matrix differs from the real displacement space: {path}")
    return matrix @ vectors


def replay_fields(fields, mesh_path, section, modes, vectors, *, expected_dofs=None,
                  incident_roots=None, returning_roots=None, trace_sha256=None):
    expected_dofs = DOFS if expected_dofs is None else expected_dofs
    incident_roots = INCIDENT if incident_roots is None else tuple(incident_roots)
    returning_roots = RETURNING if returning_roots is None else tuple(returning_roots)
    trace_sha256 = TRACE_SHA if trace_sha256 is None else trace_sha256
    with np.load(mesh_path, allow_pickle=False) as mesh:
        points, tetrahedra = mesh["points_m"].T, mesh["tetrahedra"].T
    arrays = {"points_m": points, "tetrahedra": tetrahedra, "displacements_per_sqrt_w": vectors,
              "element_order": 2, "frequencies_hz": np.full(len(incident_roots), FREQUENCY),
              "incident_mode_indices": incident_roots, "transmission_mode_indices": incident_roots,
              "reflection_mode_indices": returning_roots, "port_wavenumbers_rad_m": modes["wavenumbers_rad_m"],
              "port_displacements": modes["displacements"], "port_axial_forces": modes["axial_forces"],
              "port_left_dofs": section["left_dofs"], "port_right_dofs": section["right_dofs"]}
    for key, value in arrays.items():
        require(np.array_equal(fields[key], value), f"Exported field identity differs: {key}.")

    coordinates, components, elements = (fields[key] for key in ("dof_coordinates_m", "dof_component", "element_dofs"))
    require(coordinates.shape == (expected_dofs, 3) and components.shape == (expected_dofs,)
            and elements.shape == (30, len(tetrahedra)) and 0 <= elements.min() <= elements.max() < expected_dofs,
            "Exported P2 element-to-displacement field layout differs.")
    # The six TetP2 edge midpoints follow the four vertex nodes in the saved layout.
    sites = ((0, 0), (1, 1), (2, 2), (3, 3), (0, 1), (1, 2), (0, 2), (0, 3), (1, 3), (2, 3))
    for node, (first, second) in enumerate(sites):
        expected = (points[tetrahedra[:, first]] + points[tetrahedra[:, second]]) / 2
        ids = elements[3 * node:3 * node + 3]
        require(np.all(components[ids] == np.arange(3)[:, None])
                and np.allclose(coordinates[ids], expected, rtol=0, atol=2e-20),
                "A TetP2 node does not have three correctly located Cartesian field components.")
    node_ids = fields["nodal_dofs"]
    require(node_ids.shape == (3, len(points)) and 0 <= node_ids.min() <= node_ids.max() < expected_dofs
            and np.all(components[node_ids] == np.arange(3)[:, None])
            and np.allclose(coordinates[node_ids], points, rtol=0, atol=2e-20),
            "Exported vertex DOFs do not sample the saved spatial field.")

    left, right = section["left_dofs"], section["right_dofs"]
    trace_json = json.dumps([left.tolist(), right.tolist()], separators=(",", ":"))
    require(hashlib.sha256(trace_json.encode()).hexdigest() == trace_sha256, "Ordered trace map SHA-256 differs.")
    component_ids = np.empty(len(left), dtype=int)
    for component, ids in enumerate(section["component_dofs_xyz"]):
        component_ids[ids] = component
    for trace, plane in zip((left, right), PLANES, strict=True):
        plane_dofs = np.flatnonzero(np.isclose(coordinates[:, 0], plane, rtol=0, atol=2e-20))
        require(np.array_equal(np.sort(trace), plane_dofs) and np.array_equal(components[trace], component_ids)
                and np.allclose(coordinates[trace, 1:], section["coordinates_yz_m"], rtol=0, atol=2e-20),
                "An original-plane SAFE trace is incomplete or has an incorrect component map.")


def replay_ports(modes, fields, left, right, vectors, nodal_force, rhs, rhs_norm,
                 *, incident_roots=None, returning_roots=None):
    incident_roots = INCIDENT if incident_roots is None else tuple(incident_roots)
    returning_roots = RETURNING if returning_roots is None else tuple(returning_roots)
    plus, minus = (modes[key] for key in ("positive_indices", "negative_indices"))
    propagating = modes["propagating"]
    require(np.array_equal(plus[propagating[plus]], incident_roots) and np.array_equal(minus[propagating[minus]], returning_roots),
            "Signed open-channel order differs from the accepted 0.1 GHz point.")
    wavenumbers = modes["wavenumbers_rad_m"]
    counterparts = np.isclose(wavenumbers[list(incident_roots), None], -wavenumbers[list(returning_roots)].conj(),
                              rtol=1e-8, atol=0)
    require(np.array_equal(counterparts, np.eye(len(incident_roots), dtype=bool)),
            "A within-frequency counterpropagating channel does not have a unique reciprocal wavenumber.")
    u, f = modes["displacements"], modes["axial_forces"]
    u_in, f_in = u[:, incident_roots], -f[:, incident_roots]
    u_ref, u_trn = vectors[left] - u_in, vectors[right]
    f_ref, f_trn = nodal_force[left] - f_in, nodal_force[right]
    ref = np.linalg.solve(u[:, minus], u_ref)
    trn = np.linalg.solve(u[:, plus], u_trn)
    expected_rhs = f_in + f[:, minus] @ np.linalg.solve(u[:, minus], u_in)
    supported = np.zeros(len(vectors), dtype=bool)
    supported[left] = True
    rhs_error = np.linalg.norm(rhs[left] - expected_rhs, axis=0) / rhs_norm
    require(not np.any(rhs[~supported]) and np.all(rhs_error <= RESIDUAL),
            "The staged RHS does not match the incident traction and exact left support.")

    boundary_error = nodal_force.copy()
    boundary_error[left] = f_ref + f[:, minus] @ ref
    boundary_error[right] = f_trn - f[:, plus] @ trn
    physical_residual = np.linalg.norm(boundary_error, axis=0) / rhs_norm
    require(np.all(physical_residual <= RESIDUAL), "Independent body and signed boundary forces fail to balance.")
    incident_p = -work_power(u_in, f_in, OMEGA)
    direct_r = work_power(u_ref, f_ref, OMEGA) / incident_p
    direct_t = work_power(u_trn, f_trn, OMEGA) / incident_p
    net = (work_power(vectors[left], nodal_force[left], OMEGA) + work_power(u_trn, f_trn, OMEGA)) / incident_p
    result = {}
    for label, coefficients, direction, normal in (("reflection", ref, minus, -1), ("transmission", trn, plus, 1)):
        open_rows = propagating[direction]
        signed_power = work_power(u[:, direction[open_rows]], normal * f[:, direction[open_rows]], OMEGA)
        open_coeff = coefficients[open_rows]
        require(np.allclose(signed_power, 1.0, rtol=0, atol=FLUX_GRAM_LIMIT_W)
                and np.array_equal(fields[f"{label}_all_mode_indices"], direction)
                and np.allclose(fields[f"{label}_amplitudes_full"], coefficients, rtol=1e-8, atol=1e-10)
                and np.allclose(fields[f"{label}_amplitudes"], open_coeff, rtol=1e-8, atol=1e-10),
                f"Exported original-plane {label} modes or coefficients differ from independent projection.")
        result[label] = (open_coeff, abs(open_coeff)**2 * signed_power[:, None] / incident_p)
    require(np.allclose(fields["incident_powers_w"], incident_p, rtol=0, atol=1e-12), "Exported incident power differs.")
    modal_r, modal_t = (result[label][1].sum(axis=0) for label in ("reflection", "transmission"))
    power_maxima = check_power(incident_p, modal_r, modal_t, direct_r, direct_t, net)
    coherent = check_coherent_power(result["reflection"][0], result["transmission"][0], (u_in, f_in),
                                    (u[:, returning_roots], -f[:, returning_roots]),
                                    (vectors[left], nodal_force[left]), (u_trn, f_trn),
                                    incident_roots=incident_roots, returning_roots=returning_roots)
    return result, (modal_r, modal_t, direct_r, direct_t, net), power_maxima, rhs_error, physical_residual, coherent


def compare_public_tables(public, blocks, totals, *, incident_roots=None, returning_roots=None):
    incident_roots = INCIDENT if incident_roots is None else tuple(incident_roots)
    returning_roots = RETURNING if returning_roots is None else tuple(returning_roots)
    counts = len(incident_roots)
    with (public / "scattering_channels_AP_pnc.csv").open(newline="", encoding="utf-8") as stream:
        rows = list(csv.DictReader(stream))
    expected = {(kind, incoming, outgoing): (column, row)
                for kind, roots in (("reflection", returning_roots), ("transmission", incident_roots))
                for column, incoming in enumerate(incident_roots) for row, outgoing in enumerate(roots)}
    for row in rows:
        kind = row["kind"]
        column, index = expected.pop((kind, int(row["incident_port_index"]), int(row["outgoing_port_index"])))
        coeff, power = blocks[kind]
        published = complex(float(row["amplitude_real"]), float(row["amplitude_imag"]))
        require(int(row["incident_mode_order"]) == column + 1
                and np.isclose(published, coeff[index, column], rtol=1e-8, atol=1e-10)
                and np.isclose(float(row["power_fraction"]), power[index, column], rtol=1e-8, atol=1e-10),
                "A public complex coefficient or channel power differs from independent replay.")
    require(not expected, "Public table does not contain all 32 propagating coefficients.")
    with (public / "scattering_AP_pnc.csv").open(newline="", encoding="utf-8") as stream:
        total_rows = list(csv.DictReader(stream))
    require(len(total_rows) == counts, "Public table does not contain all four incident drives.")
    modal_r, modal_t, direct_r, direct_t, _ = totals
    metrics = {"reflection": modal_r, "transmission": modal_t, "reflection_flux": direct_r,
               "transmission_flux": direct_t, "power_balance": 1 - modal_r - modal_t,
               "flux_power_balance": 1 - direct_r - direct_t}
    for column, row in enumerate(total_rows):
        require(int(row["incident_mode_order"]) == column + 1 and float(row["frequency_GHz"]) == 0.1
                and all(np.isclose(float(row[key]), values[column], rtol=1e-8, atol=1e-10)
                        for key, values in metrics.items()), "A public incident-drive power row differs.")


def replay(stage_path, solve_path, export_path):
    started = monotonic()
    export = json.loads(export_path.read_bytes())
    stage = read_receipt(check_link(stage_path, export["stage"]), export["stage"]["sha256"])
    solve = read_receipt(check_link(solve_path, export["solve"]), export["solve"]["sha256"])
    body_path = BODY / "thickness-h80-stage-1335309/stage_AP_pnc.json"
    body_stage = read_receipt(body_path, BODY_STAGE_SHA)
    require(stage["status"] == "operator_staged" and solve["status"] == "drives_accepted"
            and export["status"] == "completed_unverified" and solve["stage"] == export["stage"]
            and stage["frequency_hz"] == FREQUENCY and stage["trace_sha256"] == TRACE_SHA
            and stage["incident_port_roots"] == list(INCIDENT) and stage["mesh"] == body_stage["mesh"],
            "Returned receipts do not bind a completed 0.1 GHz run to the saved body and trace.")
    require(stage["body_stage"]["sha256"] == BODY_STAGE_SHA
            and stage["safe_section"] == body_stage["safe_modes"]
            and all(stage[key] is False for key in ("mesh_generated", "body_assembly_performed",
                    "safe_eigensolve_performed", "factorization_performed", "phase_adjustment_applied"))
            and all(export[key] is False for key in ("spatial_convergence_established", "frequency_convergence_established")),
            "The point's reused inputs or numerical scope differ.")
    arrays = stage["array_files"]
    require(all(arrays[key] == body_stage["array_files"][key] for key in ("K", "M")),
            "The point does not reuse the archived body matrices.")
    body_paths = {key: check_link(body_path.parent / f"{key}_AP_pnc.npz", arrays[key]) for key in ("K", "M")}
    original_paths = {key: check_link(stage_path.parent / name, arrays[key])
                      for key, name in (("A", "A_AP_pnc.npz"), ("rhs", "rhs_AP_pnc.npy"))}
    mesh_link = body_stage["mesh"]
    mesh_relative = PurePosixPath(mesh_link["path"]).parts[-4:]
    mesh_path = check_link(BODY.joinpath(*mesh_relative), mesh_link)
    require(stage["safe_modes"]["sha256"] == MODES_SHA, "The returned point references a different SAFE mode set.")
    modes_path = check_link(INVENTORY / "modes-100000000_AP_pnc.npz", stage["safe_modes"])
    section_path = check_link(BODY / "thickness-h80-safe-1335308/ports_AP_pnc.npz", body_stage["safe_modes"])
    summary_link = export["public_summary"]
    public = export_path.parent / PurePosixPath(summary_link["path"]).parent.name
    summary_path = check_link(public / "summary_AP_pnc.json", summary_link)
    summary = json.loads(summary_path.read_bytes())
    require(summary["incident_side"] == "left" and summary["frequency_hz"] == FREQUENCY
            and summary["hole_count"] == 26 and summary["power_balance_passed"] is True
            and summary["spatial_convergence_established"] is False, "Public point scope or acceptance differs.")
    field_name, vtu_name = "selected_fields_AP_pnc.npz", "selected_vertex_shapes_AP_pnc.vtu"
    for name in (field_name, vtu_name, "scattering_AP_pnc.csv", "scattering_channels_AP_pnc.csv", "port_modes_AP_pnc.csv"):
        require(file_sha(public / name) == summary["output_sha256"][name], f"Public output SHA-256 differs: {name}.")

    report = solve["linear_solver"]
    reported_nnz = report["factor"]["factor_nonzeros"]
    require(report["relative_target"] == TARGET and report["relative_residual_limit"] == RESIDUAL
            and report["backward_residual_limit"] == RESIDUAL and report["factor"]["permc_spec"] == "COLAMD"
            and type(reported_nnz) is int and 2 * DOFS <= reported_nnz <= np.iinfo(np.int32).max,
            "Solver gates or recorded aggregate factor count are invalid.")
    require(stage["superlu_library"] == solve["superlu_library"] == export["superlu_library"]
            and all(report[key] == stage[key] for key in ("operator_csc_sha256", "rhs_array_sha256", "operator_nonzeros")),
            "A phase's SuperLU build or original equation identity differs.")
    channels = report["channels"]
    require([row["incident_port_root"] for row in channels] == list(INCIDENT), "Accepted drive order differs.")
    vectors = np.empty((DOFS, len(INCIDENT)), dtype=np.complex128, order="F")
    for column, row in enumerate(channels):
        require(row["column"] == column and row["target_met"] is True and row["info"] == 0,
                "A retained incident drive is missing its acceptance record.")
        path = check_link(solve_path.parent / f"accepted-{column:03d}_AP_pnc.npy", row["accepted_vector"])
        vector = np.load(path, allow_pickle=False)
        require(vector.shape == (DOFS,) and vector.dtype == np.complex128 and np.isfinite(vector).all(),
                "An accepted complex displacement vector has invalid layout or values.")
        vectors[:, column] = vector

    memory = psutil.virtual_memory().available
    required_bytes = arrays["A"]["bytes"] + 512 * 2**20
    if memory < required_bytes:
        raise RuntimeError(f"Replay of one matrix at a time requires {required_bytes} B available; measured {memory} B.")
    rhs = np.load(original_paths["rhs"], allow_pickle=False, mmap_mode="r")
    operator = load_npz(original_paths["A"])
    require(rhs.shape == vectors.shape and rhs.dtype == np.complex128 and operator.shape == (DOFS, DOFS)
            and operator.format == "csc" and operator.has_canonical_format and operator.dtype == np.complex128
            and operator.nnz == stage["operator_nonzeros"] and operator.indices.dtype == np.int32
            and operator.indptr.dtype == np.int32, "The original complex CSC equations differ in layout.")
    require(array_sha(operator.data, operator.indices, operator.indptr) == stage["operator_csc_sha256"]
            and array_sha(rhs) == stage["rhs_array_sha256"], "The original equation array identities differ.")
    relative, backward, rhs_norm, matrix_norm = check_original_equations(operator, rhs, vectors)
    del operator
    nodal_force = body_action(body_paths["K"], vectors)
    nodal_force -= OMEGA**2 * body_action(body_paths["M"], vectors)
    with np.load(modes_path, allow_pickle=False) as archive, np.load(section_path, allow_pickle=False) as section, \
            np.load(public / field_name, allow_pickle=False) as fields:
        modes = dict(archive)
        require(modes["frequency_hz"].item() == FREQUENCY, "SAFE modes use a different frequency.")
        replay_fields(fields, mesh_path, section, modes, vectors)
        left, right = section["left_dofs"], section["right_dofs"]
        port_result = replay_ports(modes, fields, left, right, vectors, nodal_force, rhs, rhs_norm)
    blocks, totals, power_maxima, rhs_error, physical_residual, coherent = port_result
    compare_public_tables(public, blocks, totals)
    r, t, r_direct, t_direct, net = totals
    same_r, same_t = (np.diag(blocks[key][1]) for key in ("reflection", "transmission"))
    representative = int(np.argmax(r))
    return {"author": "Athul Prem", "file_tag": "_AP_pnc", "frequency_hz": FREQUENCY,
            "status": "point_verified_within_saved_discretization", "reference_planes_m": PLANES,
            "problem_dofs": DOFS, "incident_roots": INCIDENT, "left_outgoing_roots": RETURNING,
            "stage_receipt_sha256": export["stage"]["sha256"], "solve_receipt_sha256": export["solve"]["sha256"],
            "export_receipt_sha256": file_sha(export_path), "public_summary_sha256": summary_link["sha256"],
            "factor_nonzeros_reported": reported_nnz, "matrix_infinity_norm": matrix_norm,
            "linux_superlu_library_sha256": solve["superlu_library"]["sha256"],
            "relative_l2_target": TARGET, "relative_l2_limit": RESIDUAL, "backward_limit": RESIDUAL,
            "power_limit": POWER, "maximum_power_errors": dict(zip(POWER_CHECKS, power_maxima.tolist(), strict=True)),
            "coherent_left_incidence": coherent,
            "channels": [{"incident_root": root, "reverse_root": RETURNING[column],
                          "relative_residual": relative[column], "backward_residual": backward[column],
                          "independent_rhs_relative_error": rhs_error[column],
                          "independent_body_boundary_relative_residual": physical_residual[column],
                          "reflection": r[column], "transmission": t[column], "reflection_flux": r_direct[column],
                          "transmission_flux": t_direct[column], "net_outward_power_ratio": net[column],
                          "converted_reflection": r[column] - same_r[column],
                          "converted_transmission": t[column] - same_t[column]}
                         for column, root in enumerate(INCIDENT)],
            "representative_field": {"selection": "largest reflected power in this point", "column": representative,
                  "incident_root": INCIDENT[representative], "path": str(public / field_name),
                  "field_sha256": summary["output_sha256"][field_name], "mesh_sha256": mesh_link["sha256"],
                  "accepted_vector_sha256": channels[representative]["accepted_vector"]["sha256"],
                  "units": "m per square root watt", "basis_order": 2, "visualization_vtu": str(public / vtu_name)},
            "platform": platform.platform(), "python": platform.python_version(), "numpy": np.__version__,
            "scipy": scipy.__version__, "thread_environment": THREADS, "available_memory_bytes_before_replay": memory,
            "required_memory_bytes_before_replay": required_bytes,
            "elapsed_seconds": monotonic() - started,
            "spatial_convergence_established": False, "frequency_convergence_established": False}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for phase in ("stage", "solve", "export"):
        parser.add_argument(f"--{phase}", type=Path, required=True, help=f"Returned {phase} receipt JSON")
    args = parser.parse_args()
    print(json.dumps(replay(args.stage.resolve(), args.solve.resolve(), args.export.resolve()), indent=2, allow_nan=False))


if __name__ == "__main__":
    main()
