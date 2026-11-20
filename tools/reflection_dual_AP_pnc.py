"""Recover native reflected-amplitude transpose duals from saved incident fields.

Author: Athul Prem | _AP_pnc
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass, replace
import json
import os
from pathlib import Path
import sys

if __name__ == "__main__":
    os.environ.update(dict.fromkeys(("OPENBLAS_NUM_THREADS", "OMP_NUM_THREADS", "MKL_NUM_THREADS",
                                     "BLIS_NUM_THREADS", "NUMEXPR_NUM_THREADS"), "1"))

import numpy as np
from scipy.linalg import lstsq, solve

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT / "src"), str(ROOT)]

from phononic import workflow_AP_pnc as workflow
from phononic.resumable_scattering_AP_pnc import AUTHOR, THREADS, _array_digest, _link, _sha
from phononic.scattering_AP_pnc import solve_scattering
from tools import map_scattering_AP_pnc as native_map
from tools import run_split_device_AP_pnc as split_reader


DUAL_LIMIT = 1e-8


@dataclass(frozen=True)
class ReflectionDual:
    """Left-trace extraction rows, incident-field mixing, and full-field dual columns.

    For a body perturbation at fixed ports, the first-order change of the
    native reflection matrix is ``-duals.T @ (delta_operator @ fields)``.
    Reflection rows stay in the requested outgoing-root order. All transposes
    in the bilinear pairing are ordinary complex transposes.
    """

    extraction: np.ndarray
    mixing: np.ndarray
    duals: np.ndarray
    metrics: dict
    passed: bool
    provenance: dict | None = None


def _column_errors(error, reference):
    l2 = np.linalg.norm(error, axis=0)
    maximum = np.max(abs(error), axis=0)
    return {"l2": l2.tolist(), "maximum": maximum.tolist(),
            "relative_l2": (l2 / np.linalg.norm(reference, axis=0)).tolist(),
            "relative_maximum": (maximum / np.max(abs(reference), axis=0)).tolist()}


def _norm_errors_pass(metrics):
    return max(*metrics["relative_l2"], *metrics["relative_maximum"]) <= DUAL_LIMIT


def recover_reflection_dual(operator, rhs, fields, left_dofs, outgoing_basis, reflected_positions):
    """Fit C^T = B H at the left end and check A^T (U H) = C^T on the whole body.

    ``rhs`` is the original full-DOF B, ``fields`` is the matching forward U,
    and the square ``outgoing_basis`` contains all left-outgoing trace columns
    in their native order. ``reflected_positions`` selects propagating columns
    of that basis by zero-based position. A failed fit or full equation returns
    its diagnostics with ``passed=False``.
    """
    size = operator.shape[0]
    left_dofs = np.asarray(left_dofs)
    positions = np.asarray(reflected_positions)
    trace_size = left_dofs.size
    if (operator.shape != (size, size) or fields.ndim != 2 or fields.shape != rhs.shape
            or rhs.shape[0] != size or not rhs.shape[1]
            or outgoing_basis.shape != (trace_size, trace_size)):
        raise ValueError("The equation, incident fields, and complete outgoing trace have incompatible shapes.")
    for indices, stop in ((left_dofs, size), (positions, trace_size)):
        if (indices.ndim != 1 or not indices.size or indices.dtype.kind not in "iu"
                or np.any(indices < 0) or np.any(indices >= stop) or np.unique(indices).size != indices.size):
            raise ValueError("Trace DOFs and selected outgoing positions must be unique in-range indices.")
    if not all(np.isfinite(value).all() for value in (rhs, fields, outgoing_basis)):
        raise ValueError("Matched-port data contain nonfinite entries.")
    off_left = np.ones(size, dtype=bool)
    off_left[left_dofs] = False
    if np.any(rhs[off_left] != 0):
        raise ValueError("Incident forces must be supported only on the left trace.")

    # Solve P^T C^T = E after scaling each basis column of P to unit length.
    # E selects outgoing positions in the complete basis, including evanescents.
    basis_norm = np.linalg.norm(outgoing_basis, axis=0)
    drives = rhs[left_dofs]
    drive_norm = np.linalg.norm(drives, axis=0)
    if np.any(basis_norm <= 0) or np.any(drive_norm <= 0):
        raise ValueError("The outgoing traces and incident drives must have nonzero columns.")
    selector = np.zeros((trace_size, positions.size))
    selector[positions, np.arange(positions.size)] = 1.0
    target = solve((outgoing_basis / basis_norm).T, selector / basis_norm[:, None])
    extract = target.T
    target_norm = np.linalg.norm(target, axis=0)
    scaled, _, rank, singular = lstsq(drives / drive_norm, target / target_norm)
    mixing = scaled * target_norm / drive_norm[:, None]
    fit = _column_errors(drives @ mixing - target, target)

    duals = fields @ mixing
    forward_action = operator @ fields
    transpose_action = operator.T @ duals
    skew_action = transpose_action - operator @ duals
    residual = transpose_action.copy()
    residual[left_dofs] -= target
    transpose = _column_errors(residual, target)
    magnitudes = abs(operator)
    norm, transpose_norm = (float(np.asarray(magnitudes.sum(axis=axis)).max()) for axis in (1, 0))
    del magnitudes
    transpose["relative_backward"] = (np.max(abs(residual), axis=0) / (
        transpose_norm * np.max(abs(duals), axis=0) + np.max(abs(target), axis=0))).tolist()
    transpose["left_maximum"] = np.max(abs(residual[left_dofs]), axis=0).tolist()
    transpose["off_left_maximum"] = np.max(abs(residual[off_left]), axis=0, initial=0).tolist()
    forward_residual = forward_action - rhs
    forward = _column_errors(forward_residual, rhs)
    forward["relative_backward"] = (np.max(abs(forward_residual), axis=0) / (
        norm * np.max(abs(fields), axis=0) + np.max(abs(rhs), axis=0))).tolist()
    forward["limit"] = native_map.RESIDUAL_LIMIT
    # Compare the two ordinary-transpose pairings before the incident offset.
    pairing_error = float(np.max(abs(duals.T @ rhs - extract @ fields[left_dofs])))
    extract_error = float(np.max(abs(extract @ outgoing_basis - selector.T)))
    metrics = {"dual_limit": DUAL_LIMIT, "source_fit": fit, "original_forward_residual": forward,
               "transpose_residual": transpose,
               "transpose_minus_original_action": _column_errors(skew_action, target),
               "normalized_drive_rank": int(rank), "normalized_drive_singular_values": singular.tolist(),
               "extraction_identity_max_error": extract_error, "bilinear_identity_max_error": pairing_error}
    passed = (rank == rhs.shape[1] and _norm_errors_pass(fit) and _norm_errors_pass(transpose)
              and max(transpose["relative_backward"]) <= DUAL_LIMIT
              and max(*forward["relative_l2"], *forward["relative_backward"]) <= native_map.RESIDUAL_LIMIT
              and max(extract_error, pairing_error) <= DUAL_LIMIT)
    return ReflectionDual(extract, mixing, duals, metrics, bool(passed))


def recover_saved_reflection_dual(map_directory, frequency_hz):
    """Recover duals at one accepted left-incident map point without a new solve."""
    directory = Path(map_directory).resolve()
    state = native_map.read_map_checkpoint(directory)
    points = [point for point in state["points"] if point["status"] == "accepted"
              and point["frequency_GHz"] * 1e9 == frequency_hz]
    if len(points) != 1:
        raise ValueError("Choose exactly one accepted native frequency in hertz.")
    config, point = state["inputs"]["configuration"], points[0]
    field_directory, summary = native_map._read_accepted(point, config, state["shared"], directory)
    if summary["incident_side"] != "left":
        raise ValueError("Choose a native left-incident point.")

    with workflow.measured_resources() as resources:
        shared = native_map._restore_shared(state["shared"], config, directory)
        port = shared.port
        system = port.elastic_system
        ends = [float(system.mesh.p[0].min()), float(system.mesh.p[0].max())]
        if not np.allclose(ends, shared.record["reference_planes_x_m"], atol=1e-18, rtol=0):
            raise ValueError("The cached mesh does not use the original physical reference planes.")
        modes, saved = split_reader._reference(field_directory, summary, port)
        fields = saved["displacements_per_sqrt_w"]
        outgoing = modes.negative_indices
        positions = np.flatnonzero(modes.propagating[outgoing])
        incident = saved["incident_mode_indices"]
        incident_basis = modes.displacements[:, incident]
        result = None

        def reuse_fields(operator, rhs):
            nonlocal result
            result = recover_reflection_dual(operator, rhs, fields, port.left_dofs,
                                             modes.displacements[:, outgoing], positions)
            equation_hashes = {"operator_csc_sha256": _array_digest(operator.data, operator.indices, operator.indptr),
                               "rhs_sha256": _array_digest(rhs)}
            original = summary["checkpointed_scattering"]["original_equations"]
            result.metrics["equation_hashes_match_archive"] = {
                key: value == original[key] for key, value in equation_hashes.items()}
            result.metrics["reconstructed_equation_sha256"] = equation_hashes
            end_skew = {}
            for side, dofs in (("left", port.left_dofs), ("right", port.right_dofs)):
                block = operator[:, dofs][dofs].toarray()
                action = (block.T - block) @ result.duals[dofs]
                end_skew[side] = _column_errors(action, result.extraction.T)
            result.metrics["end_block_transpose_minus_original_action"] = end_skew
            reflected = result.duals.T @ rhs - result.extraction @ incident_basis
            result.metrics["archive_dual_reflection_max_complex_error"] = float(
                np.max(abs(reflected - saved["reflection_amplitudes"])))
            return fields

        checked = solve_scattering(system, modes, frequency_hz, linear_solver=reuse_fields)
        direct_error = float(np.max(abs(checked.reflection_amplitudes - saved["reflection_amplitudes"])))
        result.metrics["archive_trace_reflection_max_complex_error"] = direct_error
        result.metrics["body_dofs"] = int(fields.shape[0])
        result.metrics["trace_dofs_per_end"] = int(port.left_dofs.size)
    implementation = workflow.source_identity()
    for source in (__file__, native_map.__file__, split_reader.__file__):
        path = Path(source).resolve()
        implementation["source_sha256"][path.relative_to(ROOT).as_posix()] = _sha(path)
    implementation["threads"] = {name: os.environ.get(name) for name in THREADS}
    provenance = {**AUTHOR, "schema_version": 1, "map_directory": str(directory), "frequency_hz": frequency_hz,
                  "reference_planes_x_m": ends, "incident_mode_indices": incident.tolist(),
                  "reflection_mode_indices": outgoing[positions].tolist(),
                  "reference_map_checkpoint": _link(directory / native_map.CHECKPOINT, directory),
                  "reference_point_summary": point["accepted"]["summary"],
                  "reference_outputs_sha256": summary["output_sha256"],
                  "reference_map_inputs_sha256": state["inputs_sha256"],
                  "shared_artifacts": shared.record["artifacts"],
                  "reference_solve_implementation": summary["implementation"],
                  "reference_assembly_provenance": state.get("preparation_reuse"),
                  "analysis_implementation": implementation, "resources": resources}
    passed = result.passed and max(direct_error, result.metrics["archive_dual_reflection_max_complex_error"]) <= DUAL_LIMIT
    return replace(result, passed=passed, provenance=provenance)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("map_directory", type=Path, help="Accepted native frequency-map directory.")
    parser.add_argument("--frequency-hz", type=float, default=1e9, help="Accepted point in hertz.")
    args = parser.parse_args(argv)
    result = recover_saved_reflection_dual(args.map_directory, args.frequency_hz)
    mixing = {"real": result.mixing.real.tolist(), "imag": result.mixing.imag.tolist()}
    print(json.dumps({"passed": result.passed, "metrics": result.metrics, "source_mixing": mixing,
                      "provenance": result.provenance}, indent=2, allow_nan=False))
    return 0 if result.passed else 1


if __name__ == "__main__":
    raise SystemExit(main())
