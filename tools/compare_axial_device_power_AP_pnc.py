#!/usr/bin/env python3
"""Compare saved matched-port axial-drive powers. Author: Athul Prem | _AP_pnc."""

from __future__ import annotations

import argparse
from dataclasses import dataclass
import json
import math
from pathlib import Path
import sys

import numpy as np
from skfem import Basis, ElementTriP1, ElementTriP2, ElementVector, MeshTri

ROOT = Path(__file__).resolve().parents[1]
if __name__ == "__main__":
    sys.path.insert(0, str(ROOT))

from tools import map_scattering_AP_pnc as native
from tools.compare_scattering_AP_pnc import K_GAP_TOL, MIN_OVERLAP, Section, interpolation, require

AXIAL_MAJORITY = 0.5
LIMITS = (
    "Finite saved calculations at one frequency with the original physical sections and end planes. "
    "Normalized coordinates are used only for displacement matching; T and R are fractions of native incident power. "
    "This single-drive comparison leaves all-channel and complex-response failures in force. "
    "Weak reflection and relative changes near zero remain sensitive to discretization. "
    "Spatial and frequency convergence, a reflection floor, and continuous branch identity remain unresolved."
)


@dataclass(frozen=True)
class SavedPoint:
    settings: dict
    record: dict
    channels: list[dict]
    section: Section
    displacement: np.ndarray


def _section(path, fields_path, order, width_nm, thickness_nm, ndof, sha256):
    require(order in (1, 2), "Expected a P1 or P2 native port section.")
    with np.load(path, allow_pickle=False) as arrays, np.load(fields_path, allow_pickle=False) as fields:
        points, triangles, mass = arrays["points_m"], arrays["triangles"], arrays["M"]
        require(all(arrays[f"{side}_dofs"].shape == (ndof,)
                    and np.array_equal(arrays[f"{side}_dofs"], fields[f"port_{side}_dofs"])
                    for side in ("left", "right")), "Saved section and fields have different trace DOF maps.")
    size_m = np.array([width_nm, thickness_nm]) * 1e-9
    require(points.ndim == 2 and points.shape[0] == 2 and np.isrealobj(points) and np.isfinite(points).all()
            and triangles.ndim == 2 and triangles.shape[0] == 3 and triangles.dtype.kind in "iu"
            and triangles.size and triangles.min() >= 0 and triangles.max() < points.shape[1]
            and np.allclose(points.min(axis=1) / size_m, -0.5, atol=1e-10, rtol=0)
            and np.allclose(points.max(axis=1) / size_m, 0.5, atol=1e-10, rtol=0),
            "Saved port topology or extent differs from the configured straight section.")
    # Match at fixed y and z/thickness while retaining all three lab displacement components.
    points = points.copy()
    points[1] /= size_m[1]
    mesh = MeshTri(points, triangles)
    element = ElementTriP1() if order == 1 else ElementTriP2()
    scalar = Basis(mesh, element, intorder=2 * order)
    vector = Basis(mesh, ElementVector(element, dim=3), intorder=2 * order)
    components = np.stack(vector.split_indices())
    require(vector.N == ndof and mass.shape == (ndof, ndof) and np.isrealobj(mass) and np.isfinite(mass).all()
            and np.isclose(scalar.dx.sum(), size_m[0], rtol=1e-10, atol=0)
            and all(np.array_equal(vector.doflocs[:, ids], scalar.doflocs) for ids in components),
            "Invalid native port area, mass or component layout.")
    # The homogeneous section integrates rho*u*v on each Cartesian component.
    scalar_mass = mass[np.ix_(components[0], components[0])]
    return Section(scalar, components, scalar_mass, sha256)


def _native_point(directory, frequency_ghz):
    published = native.read_saved_map(directory)
    checkpoint = native.read_map_checkpoint(directory)
    pin = native._link(directory / native.CHECKPOINT, directory)
    require(pin["sha256"] == published["checkpoint_sha256"]
            and all(published[key] == checkpoint[key] for key in ("inputs_sha256", "inputs", "shared"))
            and len(published["points"]) == len(checkpoint["points"]),
            f"Native publication and checkpoint differ in {directory}.")
    inputs = {"map": native._link(directory / native.MAP, directory), "checkpoint": pin,
              "inputs_sha256": checkpoint["inputs_sha256"], "source_git_commit": checkpoint["source_git_commit"]}
    matches = [point for point in checkpoint["points"] if point["frequency_GHz"] == frequency_ghz]
    if not matches:
        return {"directory": str(directory), "inputs": inputs, "point_status": "absent"}, None
    require(len(matches) == 1, f"The frequency occurs multiple times in {directory}.")
    point = matches[0]
    published_point = published["points"][point["index"] - 1]
    require(all(published_point.get(key) == point.get(key) for key in ("index", "frequency_GHz", "status", "accepted")),
            f"Native publication and checkpoint disagree at the requested frequency in {directory}.")
    return {"directory": str(directory), "inputs": inputs, "point_status": point["status"]}, (
        point, published_point, checkpoint,
    )


def _read_case(directory, record, selected) -> SavedPoint:
    point, published, checkpoint = selected
    config = checkpoint["inputs"]["configuration"]
    pattern, shared = checkpoint["inputs"]["hole_pattern"], checkpoint["shared"]
    case_dir, summary = native._read_accepted(point, config, shared, directory)
    planes = shared["reference_planes_x_m"]
    geometry, material = config["geometry"], config["material"]
    length, width, thickness = (geometry[key] for key in ("length_nm", "width_nm", "thickness_nm"))
    holes = sorted((hole["x_m"], hole["diameter_m"]) for hole in pattern["holes"]) if pattern is not None else []
    side = summary["incident_side"]
    require(summary["analysis_kind"] == "scattering" and summary["boundary"] == "matched_ports"
            and config["analysis"]["boundary"] == "matched_ports" and side in ("left", "right")
            and side == config["analysis"].get("incident_side", "left")
            and summary["scattering_power_normalization_available"] is True
            and native._read(case_dir / "configuration_AP_pnc.json") == summary["configuration"]
            and summary["hole_pattern_provenance"] == pattern and summary["hole_count"] == len(holes)
            and all(math.isfinite(x) and math.isfinite(d) and d > 0 for x, d in holes)
            and all(math.isfinite(v) and v > 0 for v in (length, width, thickness, material["density_kg_m3"]))
            and len(planes) == 2 and all(map(math.isfinite, planes))
            and math.isclose(planes[1] - planes[0], length * 1e-9, rel_tol=1e-12),
            f"Invalid matched-port configuration, reference planes or power provenance in {case_dir}.")
    section_pin = shared["artifacts"]["port_section"]
    section_path = native._verified(section_pin, directory)
    fields_path = case_dir / native.FIELDS
    section = _section(section_path, fields_path, config["mesh"]["element_order"], width, thickness,
                       summary["port_dofs"], section_pin["sha256"])
    groups, _, powers = native._point_modes(case_dir, summary)
    incident = groups["incoming"]
    roots = incident.roots.tolist()
    require(roots == summary["incident_mode_indices"] == [r["port_index"] for r in published["incoming_channels"]]
            and len(roots) == len(set(roots)) == summary["num_incident_channels"],
            f"Native incident roots differ in {case_dir}.")
    displacement = incident.displacement[0, section.trace_to_scalar]
    energies = np.einsum("cni,cni->ci", displacement.conj(), section.mass @ displacement).real
    require(np.isfinite(energies).all() and np.all(energies >= 0) and np.all(energies.sum(axis=0) > 0),
            f"An incident mode has invalid section kinetic energy in {case_dir}.")
    fraction = energies[0] / energies.sum(axis=0)
    totals = [powers[name].sum(axis=0) for name in ("transmission", "reflection")]
    require(all(np.isfinite(value).all() and np.all(value >= 0) and np.allclose(
        value, [row[name] for row in published["incoming_channels"]], atol=1e-12, rtol=0)
                for name, value in zip(("transmission", "reflection"), totals, strict=True)),
            f"Native power totals disagree with the accepted publication in {case_dir}.")
    channels = [{"drive": i + 1, "root": root, "k_rad_m": float(incident.k[i]),
                 "axial_kinetic_energy_fraction": float(fraction[i]),
                 "T": float(totals[0][i]), "R": float(totals[1][i])} for i, root in enumerate(roots)]
    record["inputs"].update(point_summary=point["accepted"]["summary"], port_section=section_pin,
                            fields=native._link(fields_path, directory), mesh_sha256=shared["artifacts"]["mesh"]["sha256"],
                            run_hash=summary["run_hash"])
    record.update(thickness_nm=thickness, central_target_nm=config["mesh"].get("mesh_size_nm"), mesh=config["mesh"])
    settings = {"length_nm": length, "width_nm": width, "placed_holes_m": holes, "material": material,
                "incident_side": side, "reference_planes_m": planes}
    return SavedPoint(settings, record, channels, section, displacement)


def _axial_choice(channels):
    selected = [i for i, channel in enumerate(channels) if channel["axial_kinetic_energy_fraction"] > AXIAL_MAJORITY]
    if len(selected) != 1:
        return None, "Expected exactly one incident mode with a majority of kinetic energy along x."
    index = selected[0]
    k = np.array([row["k_rad_m"] for row in channels])
    if np.any(abs(np.delete(k, index) - k[index]) <= K_GAP_TOL * max(abs(k))):
        return index, "The axial wavenumber is degenerate with another incident mode."
    return index, None


def _overlap(source, target):
    probe = interpolation(source.section, target.section)
    projected = np.stack([probe @ component for component in source.displacement])
    actual, mass = target.displacement, target.section.mass
    first_mass, second_mass = mass @ projected, mass @ actual
    norm_a = np.einsum("cni,cni->i", projected.conj(), first_mass).real
    norm_b = np.einsum("cni,cni->i", actual.conj(), second_mass).real
    require(all(np.isfinite(norm).all() and np.all(norm > 0) for norm in (norm_a, norm_b)),
            "A projected incident field has invalid mass norm.")
    cross = np.einsum("cni,cnj->ij", projected.conj(), second_mass)
    overlaps = abs(cross) / np.outer(np.sqrt(norm_a), np.sqrt(norm_b))
    require(np.isfinite(overlaps).all() and np.all(overlaps <= 1 + 1e-10), "Invalid normalized section overlap.")
    return np.minimum(1, overlaps)


def _match_evidence(matrix, i, j):
    competing = max(np.delete(matrix[i], j).max(initial=0), np.delete(matrix[:, j], i).max(initial=0))
    return float(matrix[i, j]), float(competing)


def _delta(reference, candidate):
    signed = candidate - reference
    relative = signed / reference if reference else None
    return {"percentage_points": 100 * signed, "relative_change": relative,
            "relative_change_percent": 100 * relative if relative is not None else None,
            "relative_change_status": "defined" if relative is not None else "zero_reference"}


def compare_axial_power(reference, candidate, frequency_ghz):
    """Match the predominantly axial incident mode and compare its saved physical T/R."""
    frequency_ghz = float(frequency_ghz)
    frequency_hz = frequency_ghz * 1e9
    require(math.isfinite(frequency_hz) and frequency_hz > 0, "The comparison frequency must be finite and positive.")
    first_dir, second_dir = Path(reference).resolve(), Path(candidate).resolve()
    first_record, first_point = _native_point(first_dir, frequency_ghz)
    second_record, second_point = _native_point(second_dir, frequency_ghz)
    result = {"schema_version": 1, **native.AUTHOR, "status": "unresolved", "frequency_hz": frequency_hz,
              "reference": first_record, "candidate": second_record, "limits": LIMITS}
    if any(row["point_status"] != "accepted" for row in (first_record, second_record)):
        return {**result, "reason": "Both native maps need an accepted point at the exact requested frequency."}
    first = _read_case(first_dir, first_record, first_point)
    second = _read_case(second_dir, second_record, second_point)
    for key, value in first.settings.items():
        require(second.settings[key] == value, f"The two cases differ in {key.removesuffix('_m').replace('_', ' ')}.")
    result.update({key: first.settings[key] for key in ("length_nm", "width_nm", "reference_planes_m", "incident_side")},
                  hole_count=len(first.settings["placed_holes_m"]), minimum_mass_overlap=MIN_OVERLAP,
                  axial_majority_threshold=AXIAL_MAJORITY)
    i, first_reason = _axial_choice(first.channels)
    j, second_reason = _axial_choice(second.channels)
    for point, index in ((first, i), (second, j)):
        polarization = [{key: value for key, value in row.items() if key not in ("T", "R")} for row in point.channels]
        point.record["incident_modes"] = polarization
        if index is not None:
            point.record.update(polarization[index])
    if first_reason or second_reason:
        reason = f"Reference: {first_reason}" if first_reason else f"Candidate: {second_reason}"
        return {**result, "reason": reason}

    forward, forward_competitor = _match_evidence(_overlap(first, second), i, j)
    reverse, reverse_competitor = _match_evidence(_overlap(second, first), j, i)
    result["match"] = {"reference_to_candidate_overlap": forward, "candidate_to_reference_overlap": reverse,
                       "reference_to_candidate_best_competing_overlap": forward_competitor,
                       "candidate_to_reference_best_competing_overlap": reverse_competitor,
                       "method": "Reciprocal interpolation at fixed y and z/thickness, with each target's mass matrix."}
    if (min(forward, reverse) < MIN_OVERLAP or forward <= forward_competitor + K_GAP_TOL
            or reverse <= reverse_competitor + K_GAP_TOL):
        return {**result, "reason": "The selected axial fields lack a unique reciprocal mass-overlap match."}
    first.record.update(first.channels[i])
    second.record.update(second.channels[j])
    return {**result, "status": "matched", "delta": {key: _delta(first.record[key], second.record[key]) for key in ("T", "R")}}


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("reference", type=Path, help="Reference native map directory.")
    parser.add_argument("candidate", type=Path, help="Candidate native map directory.")
    parser.add_argument("--frequency-ghz", type=float, required=True, help="Exact common accepted frequency in GHz.")
    args = parser.parse_args(argv)
    try:
        result = compare_axial_power(args.reference, args.candidate, args.frequency_ghz)
        print(json.dumps(result, indent=2, allow_nan=False))
    except (KeyError, OSError, TypeError, ValueError) as exc:
        parser.exit(1, f"{parser.prog}: {exc}\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
