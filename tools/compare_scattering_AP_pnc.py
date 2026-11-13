#!/usr/bin/env python3
"""Compare two completed matched-port studies offline. Author: Athul Prem | _AP_pnc."""

from __future__ import annotations

import argparse
import csv
from dataclasses import dataclass
import hashlib
from html import escape
from importlib.metadata import version
import json
import math
from pathlib import Path
import platform
import sys
from tempfile import TemporaryDirectory

import numpy as np
from scipy.optimize import linear_sum_assignment
from scipy.spatial import cKDTree
from skfem import Basis, ElementTriP1, ElementTriP2, MeshTet, MeshTri
from skfem.models.poisson import mass

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from phononic.workflow_AP_pnc import digest, write_json

AUTHOR = {"author": "Athul Prem", "file_tag": "_AP_pnc"}
CORE = ("elasticity", "geometry", "lead_modes", "material", "perforated", "port_modes", "scattering")
FILES = ("summary_AP_pnc.json", "configuration_AP_pnc.json", "selected_fields_AP_pnc.npz", "scattering_AP_pnc.csv")
OUTPUTS = ("comparison_AP_pnc.csv", "report_AP_pnc.html")
GROUPS = ("incoming", "reflection", "transmission")
LIMIT = 0.01
MIN_OVERLAP = 0.99
K_GAP_TOL = 1e-8
PHASE_AMPLITUDE_FLOOR = 1e-10


@dataclass(frozen=True)
class Section:
    basis: Basis
    trace_to_scalar: np.ndarray  # Three rows, ordered by lab component and scalar DOF.
    mass: np.ndarray
    discretization_sha256: str


@dataclass(frozen=True)
class Modes:
    roots: np.ndarray
    k: np.ndarray
    displacement: np.ndarray


def file_sha(path: Path) -> str:
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def require(condition: bool, message: str) -> None:
    if not condition:
        raise ValueError(message)


def check_outputs(directory: Path, hashes: dict, required: tuple[str, ...]) -> None:
    require(isinstance(hashes, dict) and all(name in hashes for name in required), f"Missing output checksums in {directory}.")
    for name, expected in hashes.items():
        path = directory / name
        require(Path(name).name == name and path.is_file() and path.resolve().parent == directory
                and file_sha(path) == expected, f"Saved output is missing or changed: {path}.")


def read_run(directory: Path) -> tuple[dict, dict, dict]:
    summary_path = directory / FILES[0]
    summary_bytes = summary_path.read_bytes()
    summary = json.loads(summary_bytes)
    require(summary.get("status") == "completed_unverified" and summary.get("analysis_kind") == "scattering",
            f"Select a completed public scattering run: {directory}.")
    outputs = summary["output_sha256"]
    check_outputs(directory, outputs, FILES[1:])
    configuration = json.loads((directory / FILES[1]).read_bytes())
    require(configuration == summary["configuration"], f"Saved configuration differs from summary: {directory}.")
    analysis = configuration["analysis"]
    frequency = summary["frequency_hz"]
    side = summary["incident_side"]
    require(analysis["kind"] == "scattering" and analysis["boundary"] == "matched_ports"
            and math.isfinite(frequency) and frequency > 0 and frequency == analysis["frequency_GHz"] * 1e9
            and side in ("left", "right") and side == analysis.get("incident_side", "left"),
            f"Invalid frequency, boundary, or incidence side in {directory}.")
    provenance = summary["hole_pattern_provenance"]
    geometry = configuration["geometry"]
    if provenance is not None:
        require("hole_pattern_AP_pnc.json" in outputs, f"Missing checked hole provenance: {directory}.")
        require(provenance == json.loads((directory / "hole_pattern_AP_pnc.json").read_bytes()),
                f"Saved hole provenance differs from summary: {directory}.")
    require((provenance is None) == ("pattern" not in geometry), f"Hole pattern is not traceable: {directory}.")
    holes = [] if provenance is None else sorted((hole["x_m"], hole["diameter_m"]) for hole in provenance["holes"])
    require(len(holes) == summary["hole_count"] and all(math.isfinite(x) and math.isfinite(d) and d > 0 for x, d in holes),
            f"Invalid placed hole coordinates: {directory}.")
    material = configuration["material"]
    fixed_material = {key: material[key] for key in ("C11_Pa", "C12_Pa", "C44_Pa", "density_kg_m3", "crystal_rotation", "temperature_policy")}
    require(fixed_material["temperature_policy"] == "fixed_property_control" and material["density_kg_m3"] > 0,
            f"Expected the supported fixed-property material model in {directory}.")
    core_hashes = summary["implementation"]["source_sha256"]
    core = {f"src/phononic/{module}_AP_pnc.py": core_hashes.get(f"src/phononic/{module}_AP_pnc.py") for module in CORE}
    require(all(isinstance(value, str) and len(value) == 64 for value in core.values()),
            f"Missing numerical source hashes in {directory}.")
    fixed = {"dimensions_nm": {key: geometry[key] for key in ("length_nm", "width_nm", "thickness_nm")},
             "placed_holes_m": holes, "material": fixed_material, "frequency_hz": frequency, "incident_side": side,
             "numerical_source_sha256": core}
    record = {"directory": str(directory), "run_hash": summary["run_hash"], "summary_sha256": hashlib.sha256(summary_bytes).hexdigest(),
              "output_sha256": outputs, "configuration": configuration, "hole_pattern_provenance": provenance,
              "implementation": summary["implementation"], "ndof": summary["ndof"], "port_dofs": summary["port_dofs"],
              "power_balance_passed": summary["power_balance_passed"]}
    return summary, fixed, record


def build_section(fields, order: int, dimensions: dict, density: float) -> Section:
    points, tetrahedra = fields["points_m"], fields["tetrahedra"]
    mesh_digest = hashlib.sha256(str(order).encode("ascii"))
    for array, dtype in ((points, "<f8"), (tetrahedra, "<i8")):
        mesh_digest.update(np.asarray(array.shape, dtype="<i8").tobytes())
        mesh_digest.update(np.asarray(array, dtype=dtype).tobytes())
    mesh = MeshTet(points.T, tetrahedra.T)
    length, width, thickness = (dimensions[name] * 1e-9 for name in ("length_nm", "width_nm", "thickness_nm"))
    size = np.array([width, thickness])
    x_left, x_right = points[:, 0].min(), points[:, 0].max()
    atol = 64 * np.finfo(float).eps * length
    require(abs(x_left) <= atol and abs(x_right - length) <= atol, "Saved end planes differ from the configured geometry.")
    boundary = mesh.facets[:, mesh.boundary_facets()]
    end = boundary[:, np.all(abs(points[boundary, 0] - x_left) <= atol, axis=0)]
    vertices, connectivity = np.unique(end, return_inverse=True)
    basis = Basis(MeshTri(points[vertices, 1:].T, connectivity.reshape(end.shape)),
                  ElementTriP1() if order == 1 else ElementTriP2(), intorder=2 * order)
    require(np.isclose(basis.dx.sum(), width * thickness, rtol=1e-10, atol=0)
            and np.allclose(basis.mesh.p.min(axis=1) / size, -0.5, rtol=0, atol=1e-10)
            and np.allclose(basis.mesh.p.max(axis=1) / size, 0.5, rtol=0, atol=1e-10),
            "Saved port area or extent differs from the configured section.")
    coordinates, components = fields["dof_coordinates_m"], fields["dof_component"]
    left, right = fields["port_left_dofs"], fields["port_right_dofs"]
    trace_count = 3 * basis.N
    for mapping, x in ((left, x_left), (right, x_right)):
        require(mapping.shape == (trace_count,) and mapping.dtype.kind in "iu" and mapping.min() >= 0
                and mapping.max() < len(coordinates) and len(np.unique(mapping)) == trace_count
                and np.all(abs(coordinates[mapping, 0] - x) <= atol), "Invalid saved volume-to-port mapping.")
    require(np.array_equal(components[left], components[right])
            and np.allclose(coordinates[left, 1:] / size, coordinates[right, 1:] / size, rtol=0, atol=1e-10),
            "Saved port maps do not have matching transverse coordinates and components.")
    by_component = []
    for component in range(3):
        selected = np.flatnonzero(components[left] == component)
        distance, order_in_trace = cKDTree(coordinates[left[selected], 1:] / size).query(basis.doflocs.T / size)
        require(len(selected) == basis.N and len(np.unique(order_in_trace)) == basis.N and np.max(distance) <= 1e-10,
                "Saved port trace does not match its reconstructed P1/P2 section.")
        by_component.append(selected[order_in_trace])
    return Section(basis, np.asarray(by_component), mass.assemble(basis).toarray() * density, mesh_digest.hexdigest())


def read_scattering(directory: Path, summary: dict, fixed: dict) -> tuple[Section, dict[str, Modes], dict]:
    config = summary["configuration"]
    incident_sign = 1 if fixed["incident_side"] == "left" else -1
    with np.load(directory / FILES[2], allow_pickle=False) as fields:
        order = int(fields["element_order"])
        require(order in (1, 2) and order == config["mesh"]["element_order"], "Unexpected saved displacement element order.")
        section = build_section(fields, order, fixed["dimensions_nm"], fixed["material"]["density_kg_m3"])
        k = fields["port_wavenumbers_rad_m"]
        displacement, forces = fields["port_displacements"], fields["port_axial_forces"]
        n_port = 3 * section.basis.N
        require(n_port == summary["port_dofs"] and k.shape == (2 * n_port,)
                and displacement.shape == forces.shape == (n_port, len(k)), "Saved SAFE array dimensions disagree.")
        n = summary["num_incident_channels"]
        require(type(n) is int and n > 0, "A comparison requires propagating incident channels.")
        modes = {}
        for group, key in (("incoming", "incident_mode_indices"), ("reflection", "reflection_mode_indices"),
                           ("transmission", "transmission_mode_indices")):
            roots = fields[key]
            require(roots.shape == (n,) and roots.dtype.kind in "iu" and roots.min() >= 0 and roots.max() < len(k)
                    and len(np.unique(roots)) == n, f"Invalid {group} channel indices or channel count.")
            wave = k[roots]
            require(np.isfinite(wave).all() and np.all(wave.imag == 0) and np.all(wave.real != 0), f"Unresolved {group} propagating roots.")
            require(np.all(np.diff(np.sort(wave.real)) > K_GAP_TOL * np.max(abs(wave.real))),
                    f"Unresolved degenerate {group} channels. Individual phase comparison is unavailable.")
            u, f = displacement[:, roots], forces[:, roots]
            power = np.pi * fixed["frequency_hz"] * np.sum((u.conj() * f).imag, axis=0)
            sign = -incident_sign if group == "reflection" else incident_sign
            require(np.allclose(power, sign, atol=1e-8, rtol=0), f"Invalid signed unit-power normalization for {group} channels.")
            modes[group] = Modes(roots, wave.real, u[section.trace_to_scalar])
        require(set(modes["incoming"].roots) == set(modes["transmission"].roots)
                and not set(modes["incoming"].roots) & set(modes["reflection"].roots), "Saved directional channel groups are inconsistent.")
        require(np.array_equal(fields["incident_mode_indices"], summary["incident_mode_indices"])
                and np.allclose(fields["incident_powers_w"], np.ones(n), atol=1e-8, rtol=0), "Saved incident powers or root order disagree.")
        matrices = {name: fields[f"{name}_amplitudes"] for name in GROUPS[1:]}
    with (directory / FILES[3]).open(newline="", encoding="utf-8") as stream:
        rows = list(csv.DictReader(stream))
    require(len(rows) == n and [int(row["incident_mode_order"]) for row in rows] == list(range(1, n + 1)),
            "Saved scattering CSV channel order disagrees.")
    require(all(math.isclose(float(row["frequency_GHz"]) * 1e9, summary["frequency_hz"], rel_tol=1e-14) for row in rows),
            "Saved scattering CSV frequency differs.")
    for name, matrix in matrices.items():
        require(matrix.shape == (n, n) and np.isfinite(matrix).all(), f"Invalid saved complex {name} block.")
        powers = np.sum(abs(matrix)**2, axis=0)
        require(np.allclose(powers, [float(row[name]) for row in rows], rtol=0, atol=1e-8), f"Saved {name} powers differ from complex amplitudes.")
    return section, modes, matrices


def interpolation(source: Section, target: Section):
    points = target.basis.doflocs
    lower, upper = source.basis.mesh.p.min(axis=1), source.basis.mesh.p.max(axis=1)
    inset = 32 * np.finfo(float).eps * (upper - lower)
    points = np.clip(points, (lower + inset)[:, None], (upper - inset)[:, None])
    return source.basis.probes(points).tocsr()


def align(reference: Modes, candidate: Modes, mass_matrix: np.ndarray, probe, project_reference: bool) -> tuple[np.ndarray, np.ndarray, dict]:
    first, second = reference.displacement, candidate.displacement
    if probe is not None:
        if project_reference:
            first = np.stack([probe @ component for component in first])
        else:
            second = np.stack([probe @ component for component in second])
    mass_first, mass_second = mass_matrix @ first, mass_matrix @ second
    cross = np.einsum("cni,cnj->ij", first.conj(), mass_second)
    norms_a = np.sqrt(np.einsum("cni,cni->i", first.conj(), mass_first).real)
    norms_b = np.sqrt(np.einsum("cni,cni->i", second.conj(), mass_second).real)
    require(np.all(norms_a > 0) and np.all(norms_b > 0), "A saved port mode has zero trace mass norm.")
    cross /= np.outer(norms_a, norms_b)
    i, j = linear_sum_assignment(-abs(cross))
    overlap = abs(cross[i, j])
    require(np.min(overlap) >= MIN_OVERLAP, f"Unresolved channel match: minimum mass overlap {np.min(overlap):.6g} is below {MIN_OVERLAP}.")
    unmatched = abs(cross).copy()
    unmatched[i, j] = 0
    require(np.all(overlap > unmatched[i].max(axis=1) + K_GAP_TOL)
            and np.all(overlap > unmatched[:, j].max(axis=0) + K_GAP_TOL),
            "Unresolved individual channel match: mass overlap does not distinguish competing branches.")
    phase = cross[i, j] / overlap
    return j, phase, {"reference_roots": reference.roots.tolist(), "candidate_roots": candidate.roots[j].tolist(),
                      "mass_overlaps": np.minimum(overlap, 1).tolist(), "candidate_gauge_phase_rad": np.angle(phase).tolist(),
                      "reference_wavenumber_rad_m": reference.k.tolist(), "candidate_wavenumber_rad_m": candidate.k[j].tolist(),
                      "relative_wavenumber_change": (abs(candidate.k[j] - reference.k) / abs(reference.k)).tolist()}


def compare_blocks(reference, candidate) -> tuple[dict, dict, list[dict]]:
    a_section, a_modes, a_matrices = reference
    b_section, b_modes, b_matrices = candidate
    n = len(a_modes["incoming"].roots)
    require(n == len(b_modes["incoming"].roots), "Propagating channel counts changed; the mesh pair cannot be phase compared.")
    # Project onto the richer trace, irrespective of reference/candidate order.
    project_reference = a_section.basis.N <= b_section.basis.N
    source, target = (a_section, b_section) if project_reference else (b_section, a_section)
    same_trace = (np.array_equal(source.basis.doflocs, target.basis.doflocs)
                  and np.array_equal(source.basis.mesh.t, target.basis.mesh.t))
    probe = None if same_trace else interpolation(source, target)
    matches = {name: align(a_modes[name], b_modes[name], target.mass, probe, project_reference) for name in GROUPS}
    i_order, i_phase, _ = matches["incoming"]
    blocks, rows = {}, []
    transmission_rows = [a_modes["transmission"].roots.tolist().index(root) for root in a_modes["incoming"].roots]
    for name in GROUPS[1:]:
        o_order, o_phase, _ = matches[name]
        a = a_matrices[name]
        # Both gauges refer to the same reference planes. No propagation phase is removed.
        b = o_phase[:, None] * b_matrices[name][np.ix_(o_order, i_order)] / i_phase[None, :]
        a_power, b_power = np.sum(abs(a)**2, axis=0), np.sum(abs(b)**2, axis=0)
        difference = abs(b - a)
        blocks[name] = {"reference": np.stack((a.real, a.imag), axis=-1).tolist(),
                        "candidate_aligned": np.stack((b.real, b.imag), axis=-1).tolist(),
                        "reference_power": a_power.tolist(), "candidate_power": b_power.tolist(),
                        "power_change": (b_power - a_power).tolist(),
                        "maximum_power_change": float(np.max(abs(b_power - a_power))),
                        "maximum_complex_change": float(np.max(difference))}
        for i in range(n):
            phase_a, phase_b, phase = None, None, None
            if name == "transmission":
                row = transmission_rows[i]
                if min(abs(a[row, i]), abs(b[row, i])) > PHASE_AMPLITUDE_FLOOR:
                    phase_a, phase_b = float(np.angle(a[row, i])), float(np.angle(b[row, i]))
                    phase = float(np.angle(b[row, i] / a[row, i]))
            rows.append({**AUTHOR, "quantity": name, "incident_channel": i + 1,
                         "reference_incident_root": int(a_modes["incoming"].roots[i]),
                         "candidate_incident_root": int(b_modes["incoming"].roots[i_order[i]]),
                         "reference_power": float(a_power[i]), "candidate_power": float(b_power[i]),
                         "power_change": float(b_power[i] - a_power[i]), "maximum_complex_change": float(np.max(difference[:, i])),
                         "reference_same_channel_transmission_phase_rad": phase_a,
                         "candidate_same_channel_transmission_phase_rad": phase_b,
                         "same_channel_transmission_phase_change_rad": phase})
    for field in ("reference", "candidate"):
        blocks["transmission"][f"same_channel_phase_{field}_rad"] = [
            row[f"{field}_same_channel_transmission_phase_rad"] for row in rows if row["quantity"] == "transmission"]
    blocks["transmission"]["same_channel_phase_change_rad"] = [row["same_channel_transmission_phase_change_rad"]
                                                               for row in rows if row["quantity"] == "transmission"]
    return {name: match[2] for name, match in matches.items()}, blocks, rows


def write_report(path: Path, result: dict, rows: list[dict]) -> None:
    table = []
    for row in rows:
        phase = row["same_channel_transmission_phase_change_rad"]
        table.append(f"<tr><td>{row['quantity'].title()}</td><td>{row['incident_channel']}</td>"
                     f"<td>{row['reference_power']:.6g}</td><td>{row['candidate_power']:.6g}</td><td>{row['power_change']:+.4g}</td>"
                     f"<td>{row['maximum_complex_change']:.4g}</td><td>{f'{phase:.4g}' if phase is not None else '—'}</td></tr>")
    power_status = "Passed" if result["power_passed"] else "Failed"
    complex_status = "Passed" if result["complex_passed"] else "Failed"
    reference_balance = "passed" if result["reference"]["power_balance_passed"] else "failed"
    candidate_balance = "passed" if result["candidate"]["power_balance_passed"] else "failed"
    if result["comparison_kind"] == "repeatability":
        outcome = "Same discretization; mesh convergence not assessed."
    else:
        outcome = "Mesh-pair checks passed." if result["mesh_pair_passed"] else "Mesh-pair checks failed."
    paths = "".join(f"<li>{side.title()} ({result[side]['ndof']:,} body DOFs, {result[side]['port_dofs']:,} port DOFs) "
                    f"<code>{escape(result[side]['directory'])}</code></li>" for side in ("reference", "candidate"))
    matches = "".join(f"<tr><td>{name.title()}</td><td>{min(group['mass_overlaps']):.8g}</td>"
                      f"<td>{max(group['relative_wavenumber_change']):.6g}</td></tr>"
                      for name, group in result["matched_channels"].items())
    path.write_text(f"""<!doctype html><html lang="en"><meta charset="utf-8"><title>Scattering mesh comparison</title>
<style>body{{font:16px system-ui,sans-serif;max-width:1000px;margin:3rem auto;padding:0 1rem;color:#172630}}
table{{border-collapse:collapse;width:100%}}th,td{{text-align:left;padding:.6rem;border-bottom:1px solid #dbe1e4}}
code{{overflow-wrap:anywhere}}small{{color:#40515a}}</style>
<h1>Scattering mesh comparison</h1><p>Athul Prem | _AP_pnc</p>
<p>{result['physical_inputs']['frequency_hz'] / 1e9:g} GHz · {result['physical_inputs']['incident_side'].title()} incidence</p>
<p><strong>Power change {power_status}</strong> ({result['maximum_power_change']:.6g}) ·
<strong>Complex amplitude change {complex_status}</strong> ({result['maximum_complex_change']:.6g}).
Each maximum must be below {LIMIT:g}.</p>
<p>Input power-balance checks: reference {reference_balance}; candidate {candidate_balance}.</p>
<p><strong>{outcome}</strong></p>
<p>{escape(result['scope'])}</p><ul>{paths}</ul>
<table><thead><tr><th>Quantity</th><th>Incident channel</th><th>Reference power</th><th>Candidate power</th>
<th>Power change</th><th>Max complex change</th><th>Same-channel phase change (rad)</th></tr></thead><tbody>{''.join(table)}</tbody></table>
<h2>Matched port channels</h2><table><thead><tr><th>Group</th><th>Minimum mass overlap</th>
<th>Maximum relative wavenumber change</th></tr></thead><tbody>{matches}</tbody></table>
<p><a href="comparison_AP_pnc.csv">Channel CSV</a> · <a href="summary_AP_pnc.json">Summary and provenance</a></p></html>
""", encoding="utf-8")


def compare_runs(reference: Path, candidate: Path, output_root: Path | None = None) -> dict:
    reference, candidate = Path(reference).resolve(), Path(candidate).resolve()
    require(reference != candidate, "Select two different completed runs.")
    a, fixed, a_record = read_run(reference)
    b, other, b_record = read_run(candidate)
    for key, value in fixed.items():
        require(value == other[key], f"Runs differ in {key.replace('_', ' ')}; a mesh comparison requires identical physical inputs and numerical core.")
    implementation = {"tool_sha256": file_sha(Path(__file__)), "python": platform.python_version(),
                      "packages": {name: version(name) for name in ("numpy", "scipy", "scikit-fem")}}
    comparison_hash = digest({"reference": a_record, "candidate": b_record, "implementation": implementation})
    output = Path(output_root or ROOT / "results/runs/comparisons").resolve() / f"scattering-{comparison_hash[:16]}"
    if output.exists():
        saved = json.loads((output / FILES[0]).read_bytes())
        require(saved["comparison_hash"] == comparison_hash and saved["status"] == "compared"
                and saved["reference"] == a_record and saved["candidate"] == b_record
                and saved["implementation"] == implementation, "Saved comparison identity differs.")
        check_outputs(output, saved["output_sha256"], OUTPUTS)
        return {**saved, "output_directory": str(output), "report_path": str(output / OUTPUTS[1]), "reused": True}

    reference_data, candidate_data = read_scattering(reference, a, fixed), read_scattering(candidate, b, fixed)
    reference_mesh = reference_data[0].discretization_sha256
    candidate_mesh = candidate_data[0].discretization_sha256
    comparison_kind = "repeatability" if reference_mesh == candidate_mesh else "mesh_sensitivity"
    matches, blocks, rows = compare_blocks(reference_data, candidate_data)
    power = max(block["maximum_power_change"] for block in blocks.values())
    complex_change = max(block["maximum_complex_change"] for block in blocks.values())
    input_checks = a_record["power_balance_passed"] is True and b_record["power_balance_passed"] is True
    result = {**AUTHOR, "schema_version": 1, "status": "compared", "comparison_hash": comparison_hash,
              "reference": a_record, "candidate": b_record, "physical_inputs": fixed, "implementation": implementation,
              "comparison_kind": comparison_kind, "input_power_checks_passed": input_checks,
              "discretization_sha256": {"reference": reference_mesh, "candidate": candidate_mesh},
              "matched_channels": matches, "blocks": blocks, "power_change_limit": LIMIT, "complex_change_limit": LIMIT,
              "maximum_power_change": power, "maximum_complex_change": complex_change, "power_passed": power < LIMIT,
              "complex_passed": complex_change < LIMIT,
              "mesh_pair_passed": comparison_kind == "mesh_sensitivity" and input_checks and max(power, complex_change) < LIMIT,
              "scope": "Saved runs are compared at the same reference planes after matching unit-power channels. "
                       "The comparison covers one frequency and incidence side. Mesh-pair checks require distinct "
                       "discretizations and passing input power-balance checks. A pass does not establish continuum "
                       "accuracy or a frequency-band result."}
    output.parent.mkdir(parents=True, exist_ok=True)
    with TemporaryDirectory(prefix="comparison-", dir=output.parent) as staging:
        staging = Path(staging)
        with (staging / OUTPUTS[0]).open("w", newline="", encoding="utf-8") as stream:
            writer = csv.DictWriter(stream, fieldnames=rows[0].keys())
            writer.writeheader()
            writer.writerows(rows)
        write_report(staging / OUTPUTS[1], result, rows)
        result["output_sha256"] = {name: file_sha(staging / name) for name in OUTPUTS}
        write_json(staging / FILES[0], result)
        staging.rename(output)
    return {**result, "output_directory": str(output), "report_path": str(output / OUTPUTS[1]), "reused": False}


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--reference", type=Path, required=True)
    parser.add_argument("--candidate", type=Path, required=True)
    parser.add_argument("--output-root", type=Path)
    args = parser.parse_args(argv)
    try:
        result = compare_runs(args.reference, args.candidate, args.output_root)
    except (OSError, ValueError, KeyError, TypeError, IndexError) as exc:
        print(f"Scattering comparison could not complete: {exc}", file=sys.stderr)
        return 2
    print(f"Compared: {result['output_directory']}")
    print(f"Power: {'passed' if result['power_passed'] else 'failed'} (max change {result['maximum_power_change']:.6g})")
    print(f"Complex amplitude: {'passed' if result['complex_passed'] else 'failed'} (max change {result['maximum_complex_change']:.6g})")
    for side in ("reference", "candidate"):
        print(f"{side.title()} input power balance: {'passed' if result[side]['power_balance_passed'] else 'failed'}")
    if result["comparison_kind"] == "repeatability":
        print("Same discretization; mesh convergence not assessed.")
    else:
        print(f"Mesh pair: {'passed' if result['mesh_pair_passed'] else 'failed'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
