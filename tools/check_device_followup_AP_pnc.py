#!/usr/bin/env python3
"""Check saved 26-hole body and lead refinement. Author: Athul Prem | _AP_pnc."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

import numpy as np
from scipy.optimize import linear_sum_assignment
from scipy.spatial import cKDTree
from skfem import Basis, ElementTriP2, MeshTri
from skfem.models.poisson import mass

from check_device_refinement_AP_pnc import (
    ATOL, FREQUENCY_HZ, GROUPS, NUMERICAL_MODULES, POWER_LIMIT, QUANTITIES, RESIDUAL_LIMIT, ROOT,
    checked_array, checked_file, equivalent, file_sha, level_data, match_modes, read_csv, require, section_mass, unpack,
)


DEFAULT_EVIDENCE = ROOT / "verification/device_convergence_followup_AP_pnc.json"
AUTHOR = {"author": "Athul Prem", "file_tag": "_AP_pnc"}
PORT_LEVELS = (("195", 175, 6, 2), ("315", 100, 10, 2))
PORT_PAIRS = (("99", "195"), ("195", "315"))
LENGTHS_M = {"uniform_leads": 3e-6, "device_length": 16.65e-6}
CSV_FIELDS = (*AUTHOR, "comparison", "direction", "channel", "quantity", "reference", "refined", "change",
              "mass_overlap", "abs_delta_k_times_3um_rad", "abs_delta_k_times_16p65um_rad")


def check_source(identity: dict) -> None:
    for name in NUMERICAL_MODULES:
        relative = f"src/phononic/{name}_AP_pnc.py"
        require(identity["source_sha256"][relative] == file_sha(ROOT / relative), f"Core source changed: {relative}.")


def compare_body(coarse, fine) -> tuple[dict, list[dict]]:
    a_modes, a_matrices, a_channels = coarse
    b_modes, b_matrices, b_channels = fine
    pairs = {group: match_modes(a_modes[group], b_modes[group]) for group in GROUPS}
    incident_order, incident_phase, _ = pairs["incoming"]
    blocks, rows = {}, []
    for quantity in QUANTITIES:
        order, phase, _ = pairs[quantity]
        a = a_matrices[quantity]
        b = phase[:, None] * b_matrices[quantity][np.ix_(order, incident_order)] / incident_phase[None, :]
        power_a, power_b = a_channels[quantity], b_channels[quantity][incident_order]
        delta = power_b - power_a
        difference = abs(b - a)
        blocks[quantity] = {"maximum_absolute_power_change": float(np.max(abs(delta))),
                            "maximum_complex_amplitude_change": float(difference.max()),
                            "largest_complex_change_outgoing_incoming_index_1_based": [
                                int(i) + 1 for i in np.unravel_index(difference.argmax(), difference.shape)]}
        rows.extend({**AUTHOR, "comparison": "body_h160_to_h130", "direction": "left_incidence", "channel": i + 1,
                     "quantity": quantity, "reference": float(power_a[i]), "refined": float(power_b[i]), "change": float(value)}
                    for i, value in enumerate(delta))
    diagonal_phase = np.angle(np.diag(b) / np.diag(a))
    power = max(item["maximum_absolute_power_change"] for item in blocks.values())
    complex_change = max(item["maximum_complex_amplitude_change"] for item in blocks.values())
    return {"max_absolute_power_change": power, "max_complex_scattering_change": complex_change,
            "blocks": blocks, "matched_ports": {group: pair[2] for group, pair in pairs.items()},
            "same_channel_transmission_phase_change_rad": diagonal_phase.tolist(),
            "acceptance": {"power_limit": POWER_LIMIT, "complex_limit": POWER_LIMIT,
                           "power_passed": power < POWER_LIMIT, "complex_passed": complex_change < POWER_LIMIT}}, rows


def port_section(topology: dict, ny: int, nz: int):
    count = 3 * (2 * ny + 1) * (2 * nz + 1)
    width, thickness = 1000e-9, 200e-9
    extent = np.array([width, thickness])
    points = checked_array(topology["section_vertices_yz_m"], ((ny + 1) * (nz + 1), 2), "section vertices")
    triangles = np.asarray(topology["section_triangles"])
    require(triangles.shape == (2 * ny * nz, 3) and triangles.dtype.kind in "iu"
            and triangles.min() >= 0 and triangles.max() < len(points), "Invalid section triangles.")
    require(np.allclose(points.min(axis=0), -extent / 2, rtol=ATOL, atol=0)
            and np.allclose(points.max(axis=0), extent / 2, rtol=ATOL, atol=0), "Unexpected section extent.")
    basis = Basis(MeshTri(points.T, triangles.T), ElementTriP2(), intorder=4)
    require(np.isclose(basis.dx.sum(), width * thickness, rtol=ATOL, atol=0), "Unexpected section area.")
    coords = checked_array(topology["coordinates_yz_m"], (count, 2), "section coordinates")
    tags = np.asarray(topology["components_xyz_0_based"])
    require(tags.shape == (count,) and tags.dtype.kind in "iu" and set(tags) == {0, 1, 2}, "Invalid section components.")
    scalar = mass.assemble(basis).toarray() * 5323
    indices = []
    rebuilt = np.zeros((count, count))
    for component in range(3):
        selected = np.flatnonzero(tags == component)
        distance, order = cKDTree(basis.doflocs.T / extent).query(coords[selected] / extent)
        require(len(selected) == len(np.unique(order)) == basis.N and np.max(distance) < ATOL,
                "Section coordinates do not match the P2 basis.")
        rebuilt[np.ix_(selected, selected)] = scalar[np.ix_(order, order)]
        indices.append((selected, order))
    return basis, coords, indices, rebuilt


def propagating(level: dict, count: int) -> dict:
    output = {}
    require(set(level) == {"positive", "negative"}, "Both signed-power directions are required.")
    for direction, sign in (("positive", 1), ("negative", -1)):
        data = level[direction]
        roots = data["root_indices"]
        n = len(roots)
        require(n == len(set(roots)) == 5 and all(type(root) is int and 0 <= root < 2 * count for root in roots),
                f"Invalid {direction} root inventory.")
        wave = unpack(checked_array(data["wavenumbers_rad_m"], (n, 2), f"{direction} wavenumbers"))
        require(np.all(wave.imag == 0) and np.all(wave.real != 0), "A propagating root is not real and nonzero.")
        alias = data.get("conjugate_basis_of")
        require(alias is None or (direction == "negative" and alias == "positive"), "Invalid conjugate mode reference.")
        basis_data = level[alias] if alias else data
        u = unpack(checked_array(basis_data["displacements"], (count, n, 2), f"{direction} displacement"))
        force = unpack(checked_array(basis_data["axial_forces"], (count, n, 2), f"{direction} force"))
        if alias:
            require(np.array_equal(wave.real, -output[alias][1]), "Conjugate root wavenumbers differ.")
            u, force = u.conj(), force.conj()
        recorded = checked_array(data["signed_power_w"], (n,), f"{direction} signed power")
        watts = np.pi * FREQUENCY_HZ * np.sum(u.conj() * force, axis=0).imag
        require(np.allclose(watts, recorded, atol=ATOL, rtol=0) and np.allclose(watts, sign, atol=ATOL, rtol=0),
                f"{direction} modes do not carry signed unit power.")
        output[direction] = roots, wave.real, u
    require(not set(output["positive"][0]).intersection(output["negative"][0]), "Directional root inventories overlap.")
    return output


def compare_ports(coarse, fine, old_count: str, new_count: str) -> tuple[dict, list[dict]]:
    (basis, _, source_indices, _), source_modes = coarse
    (_, coords, target_indices, target_mass), target_modes = fine
    lower, upper = basis.mesh.p.min(axis=1), basis.mesh.p.max(axis=1)
    inset = 32 * np.finfo(float).eps * (upper - lower)
    points = np.clip(coords.T, (lower + inset)[:, None], (upper - inset)[:, None])
    results, rows = {}, []
    for side in ("positive", "negative"):
        roots_a, k_a, u = source_modes[side]
        roots_b, k_b, v = target_modes[side]
        transferred = np.empty_like(v)
        for (source, order), (target, _) in zip(source_indices, target_indices):
            probes = basis.probes(points[:, target]).tocsr()[:, order]
            transferred[target] = probes @ u[source]
        a_mass = target_mass @ transferred
        b_mass = target_mass @ v
        a_norm = np.sqrt(np.sum((transferred.conj() * a_mass).real, axis=0))
        b_norm = np.sqrt(np.sum((v.conj() * b_mass).real, axis=0))
        overlaps = abs(transferred.conj().T @ b_mass) / np.outer(a_norm, b_norm)
        require(np.all(np.isfinite(overlaps) & (overlaps <= 1 + 1e-12)), "Nonphysical port overlap.")
        i, order = linear_sum_assignment(-overlaps)
        matched = np.minimum(overlaps[i, order], 1)
        require(matched.min() >= 0.99, f"{side} modes do not match individually.")
        delta = k_b[order] - k_a
        drift = float(np.max(abs(delta)))
        results[side] = {"old_root_indices": roots_a, "new_root_indices": [roots_b[j] for j in order],
                         "min_overlap": float(matched.min()), "max_abs_delta_k_rad_m": drift,
                         "max_abs_delta_k_times_3um_rad": drift * LENGTHS_M["uniform_leads"],
                         "max_abs_delta_k_times_16p65um_rad": drift * LENGTHS_M["device_length"]}
        rows.extend({**AUTHOR, "comparison": f"SAFE_{old_count}_to_{new_count}", "direction": side, "channel": j + 1,
                     "quantity": "wavenumber_rad_m", "reference": float(k_a[j]), "refined": float(k_b[order[j]]),
                     "change": float(value), "mass_overlap": float(matched[j]),
                     "abs_delta_k_times_3um_rad": float(abs(value) * LENGTHS_M["uniform_leads"]),
                     "abs_delta_k_times_16p65um_rad": float(abs(value) * LENGTHS_M["device_length"])}
                    for j, value in enumerate(delta))
    return results, rows


def check(path: Path) -> dict:
    evidence = json.loads(path.read_bytes())
    reference = json.loads(checked_file(evidence["reference_26hole_refinement"]).read_bytes())
    require(evidence["schema_version"] == 1 and evidence["frequency_hz"] == reference["frequency_hz"] == FREQUENCY_HZ
            and reference["status"] == "failed_mesh_sensitivity", "Unexpected evidence frequency or reference status.")
    require(evidence["diagnostic_lengths_m"] == LENGTHS_M, "Lead phase diagnostic lengths differ.")
    pattern = reference["hole_pattern_provenance"]
    checked_file({"path": pattern["source_csv"], "sha256": pattern["source_sha256"]})
    body = evidence["body_h130"]
    old = reference["levels"]["160"]
    config = old["configuration"]
    equivalent(body["configuration"], {**config, "mesh": {**config["mesh"], "mesh_size_nm": 130},
               "resources": {**config["resources"], "max_dofs": body["configuration"]["resources"]["max_dofs"]}}, "h130 configuration")
    require(body["status"] == "completed_unverified" and old["ndof"] < body["ndof"] <= body["configuration"]["resources"]["max_dofs"]
            and old["mesh_tetrahedra"] < body["mesh_tetrahedra"], "Invalid h130 completion or body resolution.")
    require(body["power_balance_passed"] is True and body["spatial_convergence_established"] is False,
            "The h130 solve must retain the power-check and convergence status.")
    require(body["hole_wall_z_levels_nm"] == reference["thickness_layer_diagnostics"]["160"]["hole_wall_axial_levels_nm"],
            "The body comparison changed hole-wall thickness layers.")
    for key, value in reference["levels"]["200"]["geometry_checks"].items():
        require(np.isclose(body["geometry_checks"][key], value, rtol=1e-7, atol=0), f"h130 geometry differs for {key}.")
    check_source(old["implementation"])
    check_source(body["implementation"])
    port = reference["port_geometry"]
    trace_mass = section_mass(port, config["geometry"], config["material"]["density_kg_m3"])
    coords = checked_array(body["port_coordinates_yz_m"], (99, 2), "h130 trace coordinates")
    common = np.asarray(port["coordinates_yz_m"])
    require(np.max(abs((coords - common) / np.ptp(common, axis=0))) < ATOL, "The body comparison changed port coordinates.")
    result, rows = compare_body(level_data(old, "160", trace_mass), level_data(body, "130", trace_mass))
    equivalent(evidence["body_comparison"], result, "body comparison")
    gates = result["acceptance"]
    require(evidence["status"] == ("passed_body_mesh_pair" if gates["power_passed"] and gates["complex_passed"]
                                  else "failed_body_mesh_pair"), "Body acceptance status differs.")

    saved_port = old["port_modes"]
    ports = {"99": (port_section(port, 5, 1), propagating({"positive": saved_port["incoming"], "negative": saved_port["reflection"]}, 99))}
    require(set(evidence["SAFE_ports"]) == {"195", "315"}, "Unexpected SAFE port levels.")
    for label, size, ny, nz in PORT_LEVELS:
        level = evidence["SAFE_ports"][label]
        settings = level["input"]
        expected_material = {key: config["material"][key] for key in ("C11_Pa", "C12_Pa", "C44_Pa", "density_kg_m3")}
        expected_material.update(crystal_to_lab_rotation=config["material"]["crystal_rotation"],
                                 policy=config["material"]["temperature_policy"])
        equivalent(settings["material"], expected_material, f"{label} SAFE material")
        require(level["status"] == "solved" and level["frequency_hz"] == FREQUENCY_HZ
                and level["trace_dofs"] == int(label) and level["pencil_roots"] == 2 * int(label)
                and 0 <= level["maximum_relative_residual"] < RESIDUAL_LIMIT
                and settings["port_size_nm"] == size and settings["body_size_nm"] == 200
                and settings["geometry_nm"] == {"length_nm": 400, "width_nm": 1000, "thickness_nm": 200}
                and settings["holes"] == [] and settings["element_order"] == 2, f"Unexpected {label}-DOF SAFE settings.")
        check_source(level["source"])
        ports[label] = port_section(level["port_geometry"], ny, nz), propagating(level["propagating"], int(label))
    comparisons = {}
    for first, second in PORT_PAIRS:
        key = f"{first}_to_{second}"
        comparisons[key], additional = compare_ports(ports[first], ports[second], first, second)
        rows.extend(additional)
    equivalent(evidence["SAFE_comparisons"], comparisons, "SAFE comparisons")
    unresolved = max(side["max_abs_delta_k_times_3um_rad"] for side in comparisons["195_to_315"].values()) > POWER_LIMIT
    require(evidence["SAFE_diagnostic_status"] == ("unresolved_lead_phase_sensitivity" if unresolved else "lead_phase_diagnostic_below_0p01_rad"),
            "SAFE diagnostic status differs.")

    csv = path.with_suffix(".csv")
    require(file_sha(csv) == evidence["channel_csv_sha256"], "Companion CSV checksum differs.")
    saved_rows = read_csv(csv)
    require(len(saved_rows) == len(rows), "Companion CSV row count differs.")
    for saved, actual in zip(saved_rows, rows):
        for key in CSV_FIELDS:
            value = actual.get(key, "")
            equivalent(float(saved[key]) if isinstance(value, (int, float)) else saved[key], value, f"CSV {key}")
    return {"body": result, "ports": comparisons, "unresolved": unresolved}


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--evidence", type=Path, default=DEFAULT_EVIDENCE)
    args = parser.parse_args()
    try:
        result = check(args.evidence)
    except (OSError, ValueError, KeyError, TypeError, IndexError, np.linalg.LinAlgError) as exc:
        print(f"Saved device follow-up integrity check FAILED: {exc}", file=sys.stderr)
        return 1
    body = result["body"]
    print("Saved device follow-up integrity check PASSED.")
    print(f"1 GHz left incidence, h160 to h130 nm: power {'PASS' if body['acceptance']['power_passed'] else 'FAIL'} "
          f"({body['max_absolute_power_change']:.8g}); complex response "
          f"{'PASS' if body['acceptance']['complex_passed'] else 'FAIL'} ({body['max_complex_scattering_change']:.8g}); limit {POWER_LIMIT} each.")
    for key, sides in result["ports"].items():
        length_3um = max(side["max_abs_delta_k_times_3um_rad"] for side in sides.values())
        length_device = max(side["max_abs_delta_k_times_16p65um_rad"] for side in sides.values())
        print(f"SAFE {key.replace('_to_', ' to ')} trace DOFs: 5 modes each way; phase indicators "
              f"{length_3um:.6g} rad over 3 micrometres and {length_device:.6g} rad over 16.65 micrometres.")
    if result["unresolved"]:
        print("Port phase sensitivity remains unresolved.")
    print("SAFE indicators do not bound full-device scattering error.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
