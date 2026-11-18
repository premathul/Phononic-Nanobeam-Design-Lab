#!/usr/bin/env python3
"""Run a bounded, resumable matched-port frequency map. Author: Athul Prem | _AP_pnc."""

from __future__ import annotations

import argparse
from collections.abc import Sequence
from dataclasses import dataclass
from html import escape
import json
import math
import os
from pathlib import Path
from shutil import copy2
import sys
from tempfile import mkdtemp
from urllib.parse import quote

import numpy as np
from scipy.linalg import get_blas_funcs
from scipy.sparse import load_npz, save_npz
from scipy.sparse.linalg import splu
from skfem import Basis, Dofs, ElementTetP1, ElementTetP2, ElementTriP1, ElementTriP2, ElementVector, MeshTri

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT / "src"), str(ROOT)]

from phononic import workflow_AP_pnc as workflow
from phononic.assembly_batched_AP_pnc import _DisplacementDofLayout
from phononic.elasticity_AP_pnc import ElasticSystem
from phononic.linear_lgmres_AP_pnc import _original_residual, solve_ilu_lgmres
from phononic.linear_solvers_AP_pnc import solve_ilu_gmres
from phononic.port_modes_AP_pnc import PortSystem
from phononic.resumable_scattering_AP_pnc import (
    AUTHOR, THREADS, _array_digest, _exclusive, _link, _numerical_identity,
    _read, _sha, _valid_residual, _verified,
)
from tools import compare_scattering_AP_pnc as comparison


CHECKPOINT = "checkpoint_AP_pnc.json"
MAP = "map_AP_pnc.json"
TABLE = "scattering_map_AP_pnc.csv"
REPORT = "report_AP_pnc.html"
RUN_SUMMARY = "summary_AP_pnc.json"
FIELDS = "selected_fields_AP_pnc.npz"
SHARED_FILES = ("mesh_summary", "mesh", "stiffness", "mass", "port_section")
POINT_FILES = ("configuration_AP_pnc.json", FIELDS, "scattering_AP_pnc.csv",
               "scattering_channels_AP_pnc.csv", "port_modes_AP_pnc.csv", REPORT,
               "selected_vertex_shapes_AP_pnc.vtu")
GROUPS = comparison.GROUPS
RESIDUAL_LIMIT = 1e-9
CONVERGENCE = {"spatial_convergence_established": False, "frequency_convergence_established": False}
SCOPE = ("Exploratory lossless 3D scattering at the listed frequencies and original end planes. "
         "Equations and flux checks qualify each accepted calculation. Spatial and frequency convergence remain "
         "unestablished. Matched links compare sampled port fields and do not establish the absence of intervening cutoffs.")


@dataclass(frozen=True)
class SharedInputs:
    checkpoint: tuple
    mesh_directory: Path
    port: PortSystem
    record: dict


class PointFailure(RuntimeError):
    def __init__(self, kind: str, message: str):
        super().__init__(message)
        self.kind = kind


def _point_config(config: dict, frequency: float) -> dict:
    return {**config, "analysis": {**config["analysis"], "frequency_GHz": frequency}}


def _write_checkpoint(path: Path, state: dict) -> None:
    workflow.write_json(path, {**state, "checkpoint_data_sha256": workflow.digest(state)})


def _prepare_shared(config, geometry, holes, pattern, identity, output, *, mesh_cache_root=None) -> SharedInputs:
    stage = Path(mkdtemp(prefix="shared-", dir=output))
    workflow.write_json(stage / "configuration_AP_pnc.json", config)
    if mesh_cache_root is not None:
        cached, mesh_hash, _ = workflow._mesh_location(Path(mesh_cache_root), geometry, pattern, config["mesh"], identity)
        if cached.exists():
            # Validate and retain a contained exact copy; the map stays resumable if the UI cache moves.
            workflow.load_checkpoint(cached, mesh_hash)
            internal, _, _ = workflow._mesh_location(stage, geometry, pattern, config["mesh"], identity)
            internal.mkdir(parents=True)
            for filename in (RUN_SUMMARY, "mesh_AP_pnc.npz"):
                copy2(cached / filename, internal / filename)
    mesh_check = workflow._prepare_mesh_checkpoint(config, geometry, holes, pattern, identity, stage)
    if not mesh_check["within_body_budget"]:
        raise ValueError(f"Mesh has {mesh_check['ndof']} DOFs, exceeding the configured body cap.")
    mesh_directory = Path(mesh_check["output_directory"])
    checkpoint = workflow.load_checkpoint(mesh_directory, mesh_check["mesh_hash"])
    params = config["material"]
    material = workflow.CubicMaterial(C11=params["C11_Pa"], C12=params["C12_Pa"], C44=params["C44_Pa"],
                                      rho=params["density_kg_m3"], rotation=params["crystal_rotation"])
    with workflow.measured_resources() as assembly_resources:
        system = workflow._assemble_system(checkpoint[0], config, material)
    with workflow.measured_resources() as section_resources:
        port = workflow.assemble_port(system, material, max_dofs=config["resources"]["max_port_dofs"])
    stiffness, mass, section = (stage / name for name in ("K.npz", "M.npz", "port_section_AP_pnc.npz"))
    save_npz(stiffness, system.K, compressed=False)
    save_npz(mass, system.M, compressed=False)
    np.savez(section, points_m=port.basis.mesh.p, triangles=port.basis.mesh.t,
             left_dofs=port.left_dofs, right_dofs=port.right_dofs,
             K0=port.K0, M=port.M, Q=port.Q, G=port.G, reference_speed_m_s=port.reference_speed_m_s)
    paths = (mesh_directory / RUN_SUMMARY, mesh_directory / "mesh_AP_pnc.npz", stiffness, mass, section)
    record = {"mesh_hash": mesh_check["mesh_hash"], "assembly_resources": assembly_resources,
              "section_resources": section_resources, "reference_planes_x_m": [0., geometry.length_m],
              "artifacts": {name: _link(path, output) for name, path in zip(SHARED_FILES, paths, strict=True)}}
    return SharedInputs(checkpoint, mesh_directory, port, record)


def _restore_shared(record: dict, config: dict, output: Path) -> SharedInputs:
    paths = {name: _verified(record["artifacts"][name], output) for name in SHARED_FILES}
    mesh_directory = paths["mesh_summary"].parent
    if paths["mesh"] != mesh_directory / "mesh_AP_pnc.npz":
        raise ValueError("Shared mesh file and checkpoint manifest differ.")
    checkpoint = workflow.load_checkpoint(mesh_directory, record["mesh_hash"])
    order = config["mesh"]["element_order"]
    mesh = checkpoint[0]
    element = ElementVector(ElementTetP1() if order == 1 else ElementTetP2())
    layout = _DisplacementDofLayout(mesh, element, Dofs(mesh, element), 2 * order)
    K, M = (load_npz(paths[name]) for name in ("stiffness", "mass"))
    if K.shape != (layout.N, layout.N) or M.shape != K.shape:
        raise ValueError("Saved body operators do not match the shared displacement layout.")
    system = ElasticSystem(mesh, layout, K, M)
    with np.load(paths["port_section"], allow_pickle=False) as arrays:
        element = ElementVector(ElementTriP1() if order == 1 else ElementTriP2(), dim=3)
        basis = Basis(MeshTri(arrays["points_m"], arrays["triangles"]), element, intorder=2 * order)
        left, right = arrays["left_dofs"], arrays["right_dofs"]
        K0, section_mass, Q, G = (arrays[name] for name in ("K0", "M", "Q", "G"))
        if (left.shape != (basis.N,) or right.shape != left.shape
                or any(matrix.shape != (basis.N, basis.N) for matrix in (K0, section_mass, Q, G))):
            raise ValueError("Saved section matrices do not match the shared trace layout.")
        port = PortSystem(system, basis, left, right, K0, section_mass, Q, G,
                          float(arrays["reference_speed_m_s"].item()))
    return SharedInputs(checkpoint, mesh_directory, port, record)


def _solve_equations(A, rhs, config: dict, report: dict, equations: dict):
    equations.update(operator_csc_sha256=_array_digest(A.data, A.indices, A.indptr), rhs_sha256=_array_digest(rhs))
    options = config["resources"]
    method = options.get("linear_solver", "direct")
    if method == "ilu_gmres":
        solution = solve_ilu_gmres(A, rhs, report=report, fill_factor=options.get("ilu_fill_factor", 10.))
    elif method == "ilu_lgmres":
        solution = solve_ilu_lgmres(A, rhs, report=report, fill_factor=options.get("ilu_fill_factor", 10.),
                                    single_precision=options.get("ilu_precision") == "single",
                                    **workflow.lgmres_options(options))
    else:
        solution = splu(A).solve(rhs)
    equations.update(relative_residual_limit=RESIDUAL_LIMIT, backward_residual_limit=RESIDUAL_LIMIT, channels=[])
    operator_norm = float(np.asarray(abs(A).sum(axis=1)).max())
    nrm2 = get_blas_funcs("nrm2", (rhs,))
    for column, (x, b) in enumerate(zip(solution.T, rhs.T, strict=True)):
        metrics = {"column": column}
        relative, backward, ok = _original_residual(A, x, b, nrm2, nrm2(b), float(np.max(abs(b))),
                                                     operator_norm, RESIDUAL_LIMIT, metrics)
        equations["channels"].append(metrics)
        if not ok:
            raise PointFailure("original_residual", f"Drive {column + 1} failed original residual gates "
                               f"({relative:.3e} relative, {backward:.3e} backward).")
    return solution


def _run_point(config: dict, shared: SharedInputs, attempt: Path) -> dict:
    frequency_hz = config["analysis"]["frequency_GHz"] * 1e9
    try:
        with workflow.measured_resources() as mode_resources:
            modes = workflow.solve_port_modes(shared.port, frequency_hz)
    except (ArithmeticError, RuntimeError, ValueError) as exc:
        message = str(exc).lower()
        kind = "unresolved_cutoff" if "cutoff" in message or "no propagating" in message else "port_modes"
        raise PointFailure(kind, str(exc)) from exc

    report = {"method": config["resources"].get("linear_solver", "direct")}
    equations = {}
    prepared = {
        "mesh_directory": shared.mesh_directory, "mesh_checkpoint": shared.checkpoint,
        "system": shared.port.elastic_system, "pattern": shared.checkpoint[1]["hole_pattern_provenance"],
        "assembly_resources": shared.record["assembly_resources"], "port_modes": modes, "solver_report": report,
        "linear_solver": lambda A, rhs: _solve_equations(A, rhs, config, report, equations),
        "attempt": str(attempt),
        "provenance": {"method": "frequency_map", "shared_sha256": workflow.digest(shared.record),
                       "original_equations": equations, "port_modes_resources": mode_resources,
                       "reference_planes_x_m": shared.record["reference_planes_x_m"]},
    }
    summary = workflow.run_study(config, attempt, _prepared_scattering=prepared)
    if summary["power_balance_passed"] is not True:
        raise PointFailure("power_balance", "The original scattering power and independent flux gates failed.")
    return summary


def _read_accepted(point: dict, settings: dict, shared: dict, output: Path) -> tuple[Path, dict]:
    path = _verified(point["accepted"]["summary"], output)
    summary = _read(path)
    expected = _point_config(settings, point["frequency_GHz"])
    provenance = summary["checkpointed_scattering"]
    equations = provenance["original_equations"]
    channels = equations["channels"]
    if (summary["status"] != "completed_unverified" or summary["configuration"] != expected
            or summary["frequency_hz"] != point["frequency_GHz"] * 1e9
            or summary["power_balance_passed"] is not True
            or provenance["shared_sha256"] != workflow.digest(shared)
            or provenance["reference_planes_x_m"] != shared["reference_planes_x_m"]
            or summary["mesh_checkpoint"]["mesh_hash"] != shared["mesh_hash"]
            or summary["mesh_checkpoint"]["mesh_sha256"] != shared["artifacts"]["mesh"]["sha256"]
            or not all(comparison.is_sha256(equations[key]) for key in ("operator_csc_sha256", "rhs_sha256"))
            or equations["relative_residual_limit"] != RESIDUAL_LIMIT
            or equations["backward_residual_limit"] != RESIDUAL_LIMIT
            or len(channels) != summary["num_incident_channels"] or not channels
            or [row["column"] for row in channels] != list(range(len(channels)))
            or not all(_valid_residual(row[key]) for row in channels
                       for key in ("relative_residual", "relative_backward_residual"))):
        raise ValueError(f"Accepted point identity or numerical gates differ: {path}.")
    comparison.check_outputs(path.parent, summary["output_sha256"], POINT_FILES)
    return path.parent, summary


def match_modes(reference: dict, candidate: dict, section_mass: np.ndarray) -> dict:
    """Match signed propagating groups on the unchanged physical section."""
    groups = {}
    for group in GROUPS:
        first, second = reference[group], candidate[group]
        try:
            if len(first.roots) != len(second.roots) or not len(first.roots):
                raise ValueError("Propagating channel count changed or is zero in this signed group.")
            for modes in (first, second):
                if np.any(np.diff(np.sort(modes.k)) <= comparison.K_GAP_TOL * np.max(abs(modes.k))):
                    raise ValueError("Degenerate wavenumbers do not distinguish individual channels.")
            order, _, diagnostics = comparison.align(first, second, section_mass, None, False)
            groups[group] = {"status": "matched", "candidate_order": order.tolist(), **diagnostics}
        except ValueError as exc:
            groups[group] = {"status": "unresolved", "reason": str(exc)}
    return {"status": "matched" if all(group["status"] == "matched" for group in groups.values()) else "unresolved",
            "groups": groups}


def _point_modes(directory: Path, summary: dict) -> tuple[dict, dict, dict]:
    sign = 1 if summary["incident_side"] == "left" else -1
    with np.load(directory / FIELDS, allow_pickle=False) as arrays:
        displacements, forces = arrays["port_displacements"], arrays["port_axial_forces"]
        wave = arrays["port_wavenumbers_rad_m"]
        incident_powers = arrays["incident_powers_w"]
        if not np.allclose(incident_powers, 1., atol=1e-8, rtol=0):
            raise ValueError(f"Invalid unit incident powers in {directory}.")
        modes, blocks, modal_powers = {}, {}, {}
        for group, prefix in zip(GROUPS, ("incident", "reflection", "transmission"), strict=True):
            roots = arrays[f"{prefix}_mode_indices"]
            u = displacements[:, roots]
            w = wave[roots]
            power = math.pi * summary["frequency_hz"] * np.sum((u.conj() * forces[:, roots]).imag, axis=0)
            if (not len(roots) or not np.isfinite(w).all() or np.any(w.imag != 0)
                    or not np.allclose(power, -sign if group == "reflection" else sign, atol=1e-8, rtol=0)):
                raise ValueError(f"Invalid signed propagating {group} modes in {directory}.")
            modes[group] = comparison.Modes(roots, w.real, u[None, :, :])
            if group != "incoming":
                block = arrays[f"{prefix}_amplitudes"]
                if block.shape != (len(roots), len(incident_powers)) or not np.isfinite(block).all():
                    raise ValueError(f"Invalid raw {group} amplitudes in {directory}.")
                blocks[group] = block
                modal_powers[group] = abs(block)**2 * abs(power[:, None]) / incident_powers[None, :]
    return modes, blocks, modal_powers


def _channel_rows(point, modes, blocks, modal_powers, labels, gauge):
    incoming = modes["incoming"].roots
    for group, block in blocks.items():
        aligned = gauge[group][:, None] * block / gauge["incoming"][None, :]
        for row, outgoing_root in enumerate(modes[group].roots):
            for column, incident_root in enumerate(incoming):
                raw, mapped = block[row, column], aligned[row, column]
                yield {"frequency_GHz": point["frequency_GHz"], "segment": point["segment"], "kind": group,
                       "incident_port_index": int(incident_root), "outgoing_port_index": int(outgoing_root),
                       "incident_channel": int(labels["incoming"][column]), "outgoing_channel": int(labels[group][row]),
                       "amplitude_real": float(raw.real), "amplitude_imag": float(raw.imag),
                       "power_fraction": float(modal_powers[group][row, column]),
                       "segment_amplitude_real": float(mapped.real), "segment_amplitude_imag": float(mapped.imag)}


def _render_report(result: dict) -> str:
    rows = []
    for point in result["points"]:
        if "accepted" in point:
            link = quote(point["report"], safe="/")
            channels = point["incoming_channels"]
            stats = " · ".join(f"{row['channel']}: T {row['transmission']:.6g}, R {row['reflection']:.6g}"
                               for row in channels)
            details = f'<a href="{link}">3D field report</a> · {stats}'
        elif point["attempts"]:
            details = escape(point["attempts"][-1].get("error", ""))
        else:
            details = ""
        match = point.get("track_from_previous", {}).get("status", "—")
        rows.append(f"<tr><td>{point['frequency_GHz']:.8g}</td><td>{point['status'].title()}</td>"
                    f"<td>{match.title()}</td><td>{details}</td></tr>")
    return f"""<!doctype html><html lang="en"><meta charset="utf-8"><title>Scattering frequency map</title>
<style>body{{font:16px system-ui,sans-serif;max-width:1050px;margin:3rem auto;padding:0 1rem}}
table{{border-collapse:collapse;width:100%}}td,th{{text-align:left;padding:.65rem;border-bottom:1px solid #dbe1e4}}</style>
<h1>Scattering frequency map</h1><p>Athul Prem | _AP_pnc</p><p>{escape(SCOPE)}</p>
<p>Mode labels and aligned amplitudes apply within each matched segment. Raw amplitudes retain each point's
native modal basis at the original end planes.</p>
<p>The linked reports display fresh body solve and field-file generation under “Cached export time”.
The point provenance records SAFE section-mode time separately.</p>
<table><thead><tr><th>GHz</th><th>Point</th><th>Link from previous point</th><th>Power by incident channel</th></tr>
</thead><tbody>{''.join(rows)}</tbody></table>
<p><a href="{TABLE}">Complex amplitudes and channel powers</a> · <a href="{MAP}">Map and provenance</a></p></html>
"""


def _publish(state, shared, accepted, output, *, stop_requested=False):
    result = {**state, "points": [point.copy() for point in state["points"]], "scope": SCOPE,
              "status": "complete" if len(accepted) == len(state["points"]) else "partial",
              "stop_requested": stop_requested,
              "matching_method": {"minimum_mass_overlap": comparison.MIN_OVERLAP,
                                  "relative_wavenumber_gap": comparison.K_GAP_TOL,
                                  "gauge": "Cumulative adjacent mass-overlap phases within each segment."}}
    rows = []
    previous, segment = None, 0
    for point in result["points"]:
        index = point["index"]
        if index not in accepted:
            previous = None
            continue
        directory, summary = accepted[index]
        modes, blocks, modal_powers = _point_modes(directory, summary)
        if previous is None:
            match = {"status": "segment_start", "reason": "No adjacent accepted point to link."}
        else:
            old_modes, old_labels, old_gauge = previous
            match = match_modes(old_modes, modes, shared.port.M)
        point["track_from_previous"] = match
        if match["status"] == "matched":
            labels, gauge = {}, {}
            for group, link in match["groups"].items():
                order = link["candidate_order"]
                labels[group] = np.empty(len(order), dtype=int)
                labels[group][order] = old_labels[group]
                gauge[group] = np.empty(len(order), dtype=np.complex128)
                gauge[group][order] = old_gauge[group] * np.exp(1j * np.asarray(link["candidate_gauge_phase_rad"]))
        else:
            segment += 1
            labels = {group: np.arange(1, len(mode.roots) + 1) for group, mode in modes.items()}
            gauge = {group: np.ones(len(mode.roots), dtype=np.complex128) for group, mode in modes.items()}
        point["segment"] = segment
        point["report"] = (directory / REPORT).relative_to(output).as_posix()
        point["resources"] = {"port_modes": summary["checkpointed_scattering"]["port_modes_resources"],
                              "scattering": summary["scattering_resources"],
                              "solve_and_field_files_elapsed_seconds": summary["elapsed_seconds"]}
        point["incoming_channels"] = [
            {"channel": int(channel), "port_index": int(root), "k_rad_m": float(k),
             **{group: float(np.sum(power[:, column])) for group, power in modal_powers.items()}}
            for column, (channel, root, k) in enumerate(zip(labels["incoming"], modes["incoming"].roots,
                                                            modes["incoming"].k, strict=True))]
        rows.extend(_channel_rows(point, modes, blocks, modal_powers, labels, gauge))
        previous = modes, labels, gauge
    if rows:
        workflow._write_csv(output / TABLE, rows)
    else:
        (output / TABLE).write_text("frequency_GHz,segment,kind,incident_port_index,outgoing_port_index,"
                                   "incident_channel,outgoing_channel,amplitude_real,amplitude_imag,power_fraction,"
                                   "segment_amplitude_real,segment_amplitude_imag\n", encoding="utf-8")
    temporary = output / (REPORT + ".tmp")
    temporary.write_text(_render_report(result), encoding="utf-8")
    temporary.replace(output / REPORT)
    result["output_sha256"] = {name: _sha(output / name) for name in (TABLE, REPORT)}
    result["checkpoint_sha256"] = _sha(output / CHECKPOINT)
    result["map_data_sha256"] = workflow.digest(result)
    workflow.write_json(output / MAP, result)
    return result


def read_map_checkpoint(directory: Path) -> dict:
    """Read native resumable inputs without loading mesh, operator, or field arrays."""
    checkpoint = Path(directory) / CHECKPOINT
    if not checkpoint.is_file():
        raise ValueError("Choose a native frequency-map checkpoint directory. Imported maps are read-only.")
    state = _read(checkpoint)
    if not isinstance(state, dict):
        raise ValueError("Choose a native frequency-map checkpoint directory.")
    saved_hash = state.get("checkpoint_data_sha256")
    payload = {key: value for key, value in state.items() if key != "checkpoint_data_sha256"}
    inputs = state.get("inputs", {})
    frequencies = inputs.get("frequencies_GHz", []) if isinstance(inputs, dict) else []
    if (workflow.digest(payload) != saved_hash or state.get("schema_version") != 1
            or state.get("execution_mode") is not None or not isinstance(inputs, dict)
            or not isinstance(inputs.get("configuration"), dict)
            or inputs["configuration"].get("analysis", {}).get("kind") != "scattering"
            or not frequencies or workflow.digest(inputs) != state.get("inputs_sha256")
            or not isinstance(state.get("points"), list) or len(state["points"]) != len(frequencies)
            or any(not isinstance(point, dict) for point in state["points"])
            or [(p.get("index"), p.get("frequency_GHz")) for p in state["points"]]
            != list(enumerate(frequencies, 1))):
        raise ValueError("Invalid native frequency-map checkpoint. Imported maps are read-only.")
    return state


def read_saved_map(directory: Path) -> dict:
    """Verify only the published small map, table, and report for read-only viewing."""
    directory = Path(directory).resolve()
    result = _read(directory / MAP)
    if not isinstance(result, dict):
        raise ValueError("Choose a published frequency-map directory.")
    payload = {key: value for key, value in result.items() if key != "map_data_sha256"}
    if (workflow.digest(payload) != result.get("map_data_sha256")
            or result.get("schema_version") != 1 or result.get("status") not in ("complete", "partial")
            or not isinstance(result.get("points"), list) or not result["points"]
            or any(not isinstance(point, dict) or point.get("status") not in
                   ("accepted", "pending", "running", "failed", "interrupted") for point in result["points"])):
        raise ValueError("Published frequency-map data changed or is invalid.")
    if result["status"] == "complete" and any(point["status"] != "accepted" for point in result["points"]):
        raise ValueError("Published map claims completion with unaccepted points.")
    hashes = result.get("output_sha256", {})
    if not isinstance(hashes, dict):
        raise ValueError("Published frequency-map output checksums are invalid.")
    comparison.check_outputs(directory, {name: hashes.get(name) for name in (TABLE, REPORT)}, (TABLE, REPORT))
    return result


def map_scattering(config: dict, frequencies_GHz: Sequence[float], output: Path, *, max_new_points: int | None = None,
                   should_cancel=None, progress_callback=None, mesh_cache_root: Path | None = None) -> dict:
    """Calculate or resume one fixed-geometry list, retaining accepted points and failed attempts."""
    frequencies = list(frequencies_GHz)
    if (not frequencies or any(not workflow._finite_number(f) or f <= 0 or not math.isfinite(f * 1e9) for f in frequencies)
            or any(b * 1e9 <= a * 1e9 for a, b in zip(frequencies, frequencies[1:]))):
        raise ValueError("Supply a nonempty, strictly increasing list of positive frequencies in GHz.")
    if max_new_points is not None and (type(max_new_points) is not int or max_new_points < 1):
        raise ValueError("max_new_points must be a positive integer.")
    if config.get("analysis", {}).get("kind") != "scattering":
        raise ValueError("Frequency mapping requires a matched-port scattering configuration.")
    settings = workflow.resolve_config(_point_config(config, config["analysis"].get("frequency_GHz", frequencies[0])))
    settings = json.loads(json.dumps(settings, allow_nan=False))
    geometry, holes, pattern = workflow._resolve_geometry(settings)
    implementation = workflow.source_identity()
    identity = _numerical_identity(implementation)
    identity["source_sha256"] = {**identity["source_sha256"], **{
        path.relative_to(ROOT).as_posix(): _sha(path) for path in (Path(__file__).resolve(), Path(comparison.__file__).resolve())}}
    inputs = {"configuration": settings, "frequencies_GHz": frequencies, "hole_pattern": pattern,
              "numerical_identity": identity, "thread_environment": {key: os.getenv(key) for key in THREADS}}
    inputs_sha = workflow.digest(inputs)
    output = Path(output).resolve()
    output.mkdir(parents=True, exist_ok=True)
    checkpoint = output / CHECKPOINT
    with _exclusive(output / ".frequency_map.lock"):
        if checkpoint.exists():
            state = read_map_checkpoint(output)
            state.pop("checkpoint_data_sha256")
            if state["inputs_sha256"] != inputs_sha:
                raise ValueError("Frequency-map checkpoint differs from the requested inputs, source or environment.")
        else:
            if any(path.name != ".frequency_map.lock" for path in output.iterdir()):
                raise ValueError("Select an empty output directory or an existing frequency-map checkpoint.")
            state = {**AUTHOR, **CONVERGENCE, "schema_version": 1, "inputs_sha256": inputs_sha, "inputs": inputs,
                     "source_git_commit": implementation["git_commit"], "shared": None,
                     "points": [{"index": i, "frequency_GHz": f, "status": "pending", "attempts": [], **CONVERGENCE}
                                for i, f in enumerate(frequencies, 1)]}
            _write_checkpoint(checkpoint, state)
        def progress(phase, frequency=None):
            if progress_callback is not None:
                summary = {"status": "running", "phase": phase, "map_directory": str(output),
                           "accepted_points": sum(p["status"] == "accepted" for p in state["points"]),
                           "total_points": len(frequencies)}
                if frequency is not None:
                    summary["current_frequency_GHz"] = frequency
                progress_callback(summary)

        progress("preparing")
        if should_cancel is not None and should_cancel():
            # No assembly or accepted field replay is needed for an untouched map.
            if state["shared"] is None:
                return _publish(state, None, {}, output, stop_requested=True)
            if (output / MAP).is_file():
                result = read_saved_map(output)
                if result.get("checkpoint_sha256") == _sha(checkpoint):
                    return {**result, "stop_requested": True}
            status = "complete" if all(p["status"] == "accepted" for p in state["points"]) else "partial"
            return {**state, "status": status, "stop_requested": True}
        if state["shared"] is None:
            shared = _prepare_shared(settings, geometry, holes, pattern, identity, output, mesh_cache_root=mesh_cache_root)
            state["shared"] = shared.record
            _write_checkpoint(checkpoint, state)
        else:
            shared = _restore_shared(state["shared"], settings, output)
        accepted = {point["index"]: _read_accepted(point, settings, shared.record, output)
                    for point in state["points"] if point["status"] == "accepted"}
        published = output / MAP
        if published.exists():
            result = _read(published)
            published_hash = result.pop("map_data_sha256")
            if workflow.digest(result) != published_hash:
                raise ValueError("Published frequency-map data changed.")
            if result["checkpoint_sha256"] == _sha(checkpoint):
                comparison.check_outputs(output, result["output_sha256"], (TABLE, REPORT))
                if len(accepted) == len(frequencies):
                    return {**result, "map_data_sha256": published_hash}
        attempted = 0
        stopped = False
        for point in state["points"]:
            if point["index"] in accepted:
                continue
            if should_cancel is not None and should_cancel():
                stopped = True
                break
            if max_new_points is not None and attempted >= max_new_points:
                break
            if point["status"] == "running":
                point["attempts"][-1].update(status="interrupted", error="Process exited before committing the point.")
            point_dir = output / f"point-{point['index']:03d}"
            point_dir.mkdir(exist_ok=True)
            attempt = Path(mkdtemp(prefix="attempt-", dir=point_dir))
            record = {"path": attempt.relative_to(output).as_posix(), "status": "running"}
            point["attempts"].append(record)
            point["status"] = "running"
            _write_checkpoint(checkpoint, state)
            attempted += 1
            config_at_frequency = _point_config(settings, point["frequency_GHz"])
            progress("solving", point["frequency_GHz"])
            try:
                summary = _run_point(config_at_frequency, shared, attempt)
                summary_path = Path(summary["output_directory"]) / RUN_SUMMARY
                point["accepted"] = {"summary": _link(summary_path, output)}
                try:
                    accepted[point["index"]] = _read_accepted(point, settings, shared.record, output)
                except BaseException:
                    point.pop("accepted")
                    raise
            except KeyboardInterrupt:
                point["status"] = "interrupted"
                record.update(status="interrupted", error="Interrupted before committing the point.")
                _write_checkpoint(checkpoint, state)
                _publish(state, shared, accepted, output)
                raise
            except (ArithmeticError, RuntimeError, ValueError) as exc:
                point["status"] = "failed"
                record.update(status="failed", failure_kind=getattr(exc, "kind", "scattering"), error=str(exc))
            else:
                point["status"] = "accepted"
                record["status"] = "accepted"
            _write_checkpoint(checkpoint, state)
            progress("point_finished", point["frequency_GHz"])
        progress("publishing")
        return _publish(state, shared, accepted, output, stop_requested=stopped)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True, type=Path)
    parser.add_argument("--frequencies-ghz", required=True, type=float, nargs="+",
                        help="Strictly increasing frequency samples in GHz.")
    parser.add_argument("--output", required=True, type=Path, help="New or existing map directory.")
    parser.add_argument("--max-new-points", type=int, help="Stop after this many point attempts in this invocation.")
    args = parser.parse_args(argv)
    try:
        result = map_scattering(_read(args.config), args.frequencies_ghz, args.output, max_new_points=args.max_new_points)
    except KeyboardInterrupt:
        print("Frequency map interrupted. Reuse the same inputs and output directory to resume.", file=sys.stderr)
        return 130
    except (OSError, RuntimeError, ValueError, KeyError, TypeError) as exc:
        print(f"Frequency map could not complete: {exc}", file=sys.stderr)
        return 2
    accepted = sum(point["status"] == "accepted" for point in result["points"])
    print(f"Accepted {accepted}/{len(result['points'])} frequencies. Report: {(args.output / REPORT).resolve()}")
    return 2 if any(point["status"] == "failed" for point in result["points"]) else 0


if __name__ == "__main__":
    raise SystemExit(main())
