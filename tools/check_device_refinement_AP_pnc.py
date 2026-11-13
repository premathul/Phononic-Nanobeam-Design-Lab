#!/usr/bin/env python3
"""Check saved 26-hole mesh sensitivity without a solve. Author: Athul Prem | _AP_pnc."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
import sys

import numpy as np
from scipy.optimize import linear_sum_assignment
from scipy.spatial import cKDTree
from skfem import Basis, ElementTriP2, MeshTri
from skfem.models.poisson import mass

from check_phase_refinement_AP_pnc import checked_array, checked_file, equivalent, require
from validate_perforated_AP_pnc import NUMERICAL_MODULES, POWER_LIMIT, RESIDUAL_LIMIT, ROOT, file_sha, pack, unpack


DEFAULT_EVIDENCE = ROOT / "verification/historical_26hole_refinement_AP_pnc.json"
CHANNELS = 5
TRACE_DOFS = 99
FREQUENCY_HZ = 1e9
ATOL = 1e-10
QUANTITIES = ("reflection", "transmission")
GROUPS = ("incoming", *QUANTITIES)
CHANNEL_FIELDS = (*QUANTITIES, "reflection_flux", "transmission_flux", "power_balance", "flux_power_balance")


def read_csv(path: Path) -> list[dict]:
    with path.open(encoding="utf-8", newline="") as stream:
        return list(csv.DictReader(stream))


def section_mass(port: dict, geometry: dict, density: float) -> np.ndarray:
    """Recover the trace coordinate mapping and consistent section mass."""
    vertices = checked_array(port["section_vertices_yz_m"], (12, 2), "port vertices")
    triangles = np.asarray(port["section_triangles"])
    require(triangles.shape == (10, 3) and triangles.dtype.kind in "iu"
            and triangles.min() >= 0 and triangles.max() < len(vertices), "Invalid port triangulation.")
    size = 1e-9 * np.array([geometry["width_nm"], geometry["thickness_nm"]])
    require(np.allclose(vertices.min(axis=0), -size / 2, rtol=1e-10, atol=0)
            and np.allclose(vertices.max(axis=0), size / 2, rtol=1e-10, atol=0), "Unexpected port extent.")
    basis = Basis(MeshTri(vertices.T, triangles.T), ElementTriP2(), intorder=4)
    require(np.isclose(basis.dx.sum(), np.prod(size), rtol=1e-10, atol=0), "Unexpected port area.")
    scalar_mass = mass.assemble(basis).toarray() * density
    coordinates = checked_array(port["coordinates_yz_m"], (TRACE_DOFS, 2), "trace coordinates")
    components = np.asarray(port["components_xyz_0_based"])
    require(components.shape == (TRACE_DOFS,) and components.dtype.kind in "iu"
            and set(components) == {0, 1, 2}, "Invalid trace components.")
    rebuilt = np.zeros((TRACE_DOFS, TRACE_DOFS))
    for component in range(3):
        selected = np.flatnonzero(components == component)
        distance, order = cKDTree(basis.doflocs.T / size).query(coordinates[selected] / size)
        require(len(order) == basis.N and len(np.unique(order)) == basis.N and np.max(distance) < ATOL,
                "Saved trace coordinates do not match the section basis.")
        rebuilt[np.ix_(selected, selected)] = scalar_mass[np.ix_(order, order)]
    saved = checked_array(port["mass_matrix_kg_per_m"], rebuilt.shape, "section mass")
    require(np.max(abs(saved - rebuilt)) < ATOL * np.max(abs(rebuilt)), "Section mass differs from the reconstructed trace.")
    return rebuilt


def level_data(level: dict, size: str, trace_mass: np.ndarray) -> tuple[dict, dict, dict]:
    label = f"h={size} nm"
    factor = np.linalg.cholesky(trace_mass).T
    channels = level["channels"]
    arrays = {key: checked_array(channels[key], (CHANNELS,), f"{label} {key}") for key in CHANNEL_FIELDS}
    residuals = checked_array(channels["relative_residuals"], (CHANNELS,), f"{label} linear residuals")
    require(np.all((residuals >= 0) & (residuals < RESIDUAL_LIMIT))
            and 0 <= level["port_max_relative_residual"] < RESIDUAL_LIMIT, f"{label}: excessive solve residual.")
    incoming = checked_array(channels["incident_powers_w"], (CHANNELS,), f"{label} incident power")
    require(np.allclose(incoming, 1, rtol=0, atol=ATOL), f"{label}: incident power differs from 1 W.")

    modes = {}
    for group in GROUPS:
        data = level["port_modes"][group]
        roots = data["root_indices"]
        require(len(roots) == len(set(roots)) == CHANNELS and all(type(i) is int and 0 <= i < 2 * TRACE_DOFS for i in roots),
                f"{label} {group}: invalid propagating root IDs.")
        wave = unpack(checked_array(data["wavenumbers_rad_m"], (CHANNELS, 2), f"{label} {group} wavenumbers"))
        k = wave.real
        require(np.all(wave.imag == 0) and np.all(k != 0) and np.min(np.diff(np.sort(k))) > 1e-8 * np.max(abs(k)),
                f"{label} {group}: unresolved individual propagating roots.")
        if group == "transmission":
            require(data["same_basis_as"] == "incoming", f"{label}: transmitted modes must use the incoming basis.")
            basis_data = level["port_modes"]["incoming"]
            order = [basis_data["root_indices"].index(root) for root in roots]
            require(np.array_equal(k, modes["incoming"][1][order]), f"{label}: aliased transmitted wavenumbers differ.")
        else:
            basis_data, order = data, np.arange(CHANNELS)
        shape = (TRACE_DOFS, CHANNELS, 2)
        u = unpack(checked_array(basis_data["displacements"], shape, f"{label} {group} displacement"))[:, order]
        force = unpack(checked_array(basis_data["axial_forces"], shape, f"{label} {group} axial force"))[:, order]
        watts = np.pi * FREQUENCY_HZ * np.sum(u.conj() * force, axis=0).imag
        signed = -1 if group == "reflection" else 1
        recorded_watts = checked_array(data["signed_power_w"], (CHANNELS,), f"{label} {group} signed power")
        require(np.allclose(watts, recorded_watts, rtol=0, atol=ATOL)
                and np.allclose(watts, signed, rtol=0, atol=ATOL), f"{label} {group}: signed unit power differs.")
        weighted = factor @ u
        weighted /= np.linalg.norm(weighted, axis=0)
        modes[group] = (roots, k, weighted, force, watts)
    require(modes["incoming"][0] == channels["incident_mode_indices"] == modes["transmission"][0]
            and not set(modes["incoming"][0]).intersection(modes["reflection"][0]), f"{label}: inconsistent directional root IDs.")
    require(np.allclose(modes["incoming"][4], incoming, rtol=0, atol=ATOL), f"{label}: inconsistent incident power.")

    matrices = {}
    for quantity in QUANTITIES:
        matrix = unpack(checked_array(level[quantity + "_amplitudes"], (CHANNELS, CHANNELS, 2), f"{label} {quantity} amplitudes"))
        power = (abs(matrix)**2).T @ abs(modes[quantity][4]) / incoming
        require(np.allclose(power, arrays[quantity], rtol=0, atol=ATOL), f"{label}: {quantity} power differs from amplitudes.")
        estimators = [arrays[quantity + "_flux"]]
        if quantity + "_gram" in channels:
            estimators.append(checked_array(channels[quantity + "_gram"], (CHANNELS,), f"{label} {quantity} Gram power"))
        require(all(np.max(abs(power - value)) < POWER_LIMIT for value in estimators), f"{label}: {quantity} flux disagrees.")
        matrices[quantity] = matrix
    for suffix, key in (("", "power_balance"), ("_flux", "flux_power_balance")):
        balance = 1 - arrays["reflection" + suffix] - arrays["transmission" + suffix]
        require(np.allclose(balance, arrays[key], rtol=0, atol=ATOL)
                and np.max(abs(balance)) < POWER_LIMIT, f"{label}: inconsistent power balance.")
    if "net_outward_power_ratio" in channels:
        outward = checked_array(channels["net_outward_power_ratio"], (CHANNELS,), f"{label} net outward power")
        require(np.max(abs(outward)) < POWER_LIMIT, f"{label}: total outward power does not balance.")
    return modes, matrices, {**arrays, "relative_residuals": residuals}


def match_modes(coarse: tuple, fine: tuple) -> tuple[np.ndarray, np.ndarray, dict]:
    roots_a, k_a, weighted_a, force_a, _ = coarse
    roots_b, k_b, weighted_b, force_b, _ = fine
    cross = weighted_a.conj().T @ weighted_b
    i, order = linear_sum_assignment(-abs(cross))
    selected = cross[i, order]
    k_change = abs(k_a - k_b[order]) / abs(k_a)
    require(np.min(abs(selected)) >= 0.99 and np.max(k_change) < 1e-8, "The fixed-port channels do not match.")
    phase = selected / abs(selected)
    force_error = np.linalg.norm(force_a - force_b[:, order] / phase, axis=0) / np.linalg.norm(force_a, axis=0)
    require(np.max(force_error) < 1e-8, "Matched port forces disagree after phase alignment.")
    return order, phase, {
        "reference_roots": roots_a, "refined_roots": [roots_b[index] for index in order],
        "mass_overlaps": np.minimum(abs(selected), 1).tolist(), "phase_angles_rad": np.angle(phase).tolist(),
        "relative_k_change": k_change.tolist(), "maximum_relative_axial_force_shape_change": float(np.max(force_error)),
        "wavenumber_reference_rad_m": k_a.tolist(), "wavenumber_refined_rad_m": k_b[order].tolist(),
    }


def comparison_blocks(coarse, fine, comparison: dict) -> tuple[dict, np.ndarray]:
    a_modes, a_matrices, a_channels = coarse
    b_modes, b_matrices, b_channels = fine
    matches = {key: match_modes(a_modes[key], b_modes[key]) for key in GROUPS}
    for key, label in (("incoming", "incoming"), ("reflection", "reflected"), ("transmission", "transmitted")):
        equivalent(comparison["matched_ports"][label], matches[key][2], f"{label} channel alignment")
    incident_order, incident_phase, _ = matches["incoming"]
    blocks = {}
    for quantity in QUANTITIES:
        order, phase, _ = matches[quantity]
        a = a_matrices[quantity]
        # Express fine output and input in the coarse port gauges.
        b = phase[:, None] * b_matrices[quantity][np.ix_(order, incident_order)] / incident_phase[None, :]
        power_a, power_b = a_channels[quantity], b_channels[quantity][incident_order]
        delta = power_b - power_a
        block = {"reference": pack(a), "refined_aligned": pack(b), "power_reference": power_a.tolist(),
                 "power_refined": power_b.tolist(), "power_change": delta.tolist(),
                 "maximum_absolute_power_change": float(np.max(abs(delta))),
                 "maximum_complex_amplitude_change": float(np.max(abs(b - a)))}
        if quantity == "transmission":
            block.update({"same_channel_phase_reference_rad": np.angle(np.diag(a)).tolist(),
                          "same_channel_phase_refined_rad": np.angle(np.diag(b)).tolist(),
                          "same_channel_phase_change_rad": np.angle(np.diag(b) / np.diag(a)).tolist()})
        equivalent(comparison["matrices"][quantity], block, f"{quantity} comparison")
        blocks[quantity] = block
    return blocks, incident_order


def check_channels_csv(path: Path, coarse, fine, blocks: dict, order: np.ndarray) -> None:
    a_modes, _, a = coarse
    _, _, b = fine
    rows = read_csv(path)
    indexed = {(row["quantity"], int(row["incident_channel"])): row for row in rows}
    require(len(rows) == len(indexed) == len(QUANTITIES) * CHANNELS
            and indexed.keys() == {(quantity, index + 1) for quantity in QUANTITIES for index in range(CHANNELS)},
            "Comparison CSV must include each reflection/transmission channel exactly once.")
    for (quantity, number), row in indexed.items():
        index, aligned = number - 1, order[number - 1]
        require(row["author"] == "Athul Prem" and row["file_tag"] == "_AP_pnc" and row["incident_side"] == "left",
                "Comparison CSV attribution or incident side differs.")
        block = blocks[quantity]
        expected = {"wavenumber_rad_m": a_modes["incoming"][1][index], "h200_power": a[quantity][index],
                    "h160_power": b[quantity][aligned], "power_change_h160_minus_h200": block["power_change"][index],
                    "h200_modal_flux": a[quantity + "_flux"][index], "h160_modal_flux": b[quantity + "_flux"][aligned],
                    "h200_linear_residual": a["relative_residuals"][index], "h160_linear_residual": b["relative_residuals"][aligned]}
        phase_fields = {"h200_same_channel_phase_rad": "same_channel_phase_reference_rad",
                        "h160_aligned_same_channel_phase_rad": "same_channel_phase_refined_rad",
                        "phase_change_rad": "same_channel_phase_change_rad"}
        if quantity == "transmission":
            expected.update({key: block[value][index] for key, value in phase_fields.items()})
        else:
            require(not any(row[key] for key in phase_fields), "Unexpected reflection phase columns in comparison CSV.")
        for key, value in expected.items():
            equivalent(float(row[key]), value, f"CSV {quantity} channel {number} {key}")


def check(path: Path) -> dict:
    evidence = json.loads(path.read_bytes())
    require(evidence["schema_version"] == 1 and evidence["incident_side"] == "left"
            and evidence["frequency_hz"] == FREQUENCY_HZ and evidence["element_order"] == 2,
            "Evidence must describe the left-incidence 1 GHz P2 comparison.")
    pointers = evidence["evidence"]
    pilot = json.loads(checked_file(pointers["portable_pilot_json"]).read_bytes())
    pilot_csv = checked_file(pointers["portable_pilot_csv"])
    require(pilot["status"] == "completed_unverified" and pilot["hole_count"] == 26
            and not pilot["spatial_convergence_established"]
            and pilot["channel_csv_sha256"] == file_sha(pilot_csv), "Pilot status or CSV identity differs.")
    pattern = evidence["hole_pattern_provenance"]
    require(pattern == pilot["hole_pattern_provenance"] and len(pattern["holes"]) == 26, "Hole pattern differs from the pilot.")
    checked_file({"path": pattern["source_csv"], "sha256": pattern["source_sha256"]})

    levels = evidence["levels"]
    require(set(levels) == {"200", "160"}, "Evidence must include the 200 and 160 nm mesh levels.")
    baseline = pilot["configuration"]
    require(baseline["analysis"] == {"kind": "scattering", "boundary": "matched_ports", "incident_side": "left", "frequency_GHz": 1}
            and baseline["mesh"]["element_order"] == 2, "Pilot settings differ from the comparison scope.")
    numerical_sources = {f"src/phononic/{name}_AP_pnc.py" for name in NUMERICAL_MODULES}
    for source in numerical_sources:
        digest = file_sha(ROOT / source)
        require(pilot["implementation"]["source_sha256"][source] == digest
                and all(level["implementation"]["source_sha256"][source] == digest for level in levels.values()),
                f"Core numerical source checksum differs: {source}.")
    for size, level in levels.items():
        expected = {**baseline, "mesh": {**baseline["mesh"], "mesh_size_nm": int(size)}}
        equivalent(level["configuration"], expected, f"h={size} nm configuration")
        require(0 < level["ndof"] <= level["effective_max_body_dofs"] and level["mesh_tetrahedra"] > 0,
                f"h={size} nm invalid mesh size or solve cap.")
        geometry = level["geometry_checks"]
        for key, value in pilot["geometry_diagnostics"].items():
            require(np.isclose(geometry[key], value, rtol=1e-7, atol=0), f"h={size} nm {key} differs from the pilot.")
    require(levels["200"]["ndof"] == pilot["ndof"] < levels["160"]["ndof"]
            and levels["200"]["mesh_tetrahedra"] == pilot["mesh_tetrahedra"] < levels["160"]["mesh_tetrahedra"],
            "The fine mesh must have more elements and displacement degrees of freedom.")
    require(pointers["pilot_fields"]["sha256"] == pilot["output_sha256"]["selected_fields_AP_pnc.npz"]
            and pointers["pilot_summary"]["sha256"] == pilot["source_summary_sha256"]
            and levels["200"]["mesh_identity"]["container"] == pointers["pilot_fields"], "Pilot field provenance differs.")
    require(levels["160"]["mesh_identity"]["container"] == pointers["refined_mesh"], "Refined mesh provenance differs.")
    comparison = evidence["comparison"]
    for key, link in (("reference_field_sha256", "pilot_fields"), ("refined_scattering_json_sha256", "refined_scattering_record"),
                      ("refined_scattering_npz_sha256", "refined_scattering_npz")):
        require(comparison[key] == pointers[link]["sha256"], f"Recorded provenance differs for {link}.")

    trace_mass = section_mass(evidence["port_geometry"], baseline["geometry"], baseline["material"]["density_kg_m3"])
    coords = checked_array(evidence["port_geometry"]["coordinates_yz_m"], (TRACE_DOFS, 2), "common trace coordinates")
    span = np.ptp(coords, axis=0)
    old_coords, new_coords = (checked_array(levels[h]["port_coordinates_yz_m"], coords.shape, f"h={h} native coordinates")
                              for h in ("200", "160"))
    require(max(np.max(abs((old_coords - coords) / span)), np.max(abs((new_coords - coords) / span))) < ATOL,
            "Level-specific trace coordinates do not match the shared component order.")
    coordinate_difference = float(np.max(np.linalg.norm(old_coords / span - new_coords / span, axis=1)))
    equivalent(comparison["maximum_normalized_port_coordinate_difference"], coordinate_difference, "trace coordinate difference")
    coarse, fine = (level_data(levels[size], size, trace_mass) for size in ("200", "160"))
    pilot_rows = read_csv(pilot_csv)
    require(len(pilot_rows) == CHANNELS and pilot["incident_mode_indices"] == coarse[0]["incoming"][0], "Pilot incident channels differ.")
    for index, row in enumerate(pilot_rows):
        require(int(row["incident_mode_order"]) == index + 1 and float(row["frequency_GHz"]) == FREQUENCY_HZ / 1e9,
                "Pilot CSV ordering or frequency differs.")
        for key, values in coarse[2].items():
            equivalent(float(row["relative_residual" if key == "relative_residuals" else key]), values[index], f"pilot channel {index + 1} {key}")

    blocks, order = comparison_blocks(coarse, fine, comparison)
    power = max(block["maximum_absolute_power_change"] for block in blocks.values())
    complex_change = max(block["maximum_complex_amplitude_change"] for block in blocks.values())
    equivalent(comparison["max_absolute_power_change"], power, "maximum power change")
    equivalent(comparison["max_complex_scattering_change"], complex_change, "maximum complex change")
    gates = {"power_limit": POWER_LIMIT, "complex_limit": POWER_LIMIT,
             "power_passed": power < POWER_LIMIT, "complex_passed": complex_change < POWER_LIMIT}
    equivalent(comparison["acceptance"], gates, "mesh sensitivity gates")
    converged = gates["power_passed"] and gates["complex_passed"]
    require(evidence["status"] == ("passed_mesh_sensitivity" if converged else "failed_mesh_sensitivity"),
            "Recorded status differs from the recalculated gates.")
    csv_path = path.with_suffix(".csv")
    require(file_sha(csv_path) == evidence["channel_csv_sha256"], "Comparison CSV checksum differs.")
    check_channels_csv(csv_path, coarse, fine, blocks, order)
    return {"power_change": power, "complex_change": complex_change, "converged": converged}


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--evidence", type=Path, default=DEFAULT_EVIDENCE)
    parser.add_argument("--require-converged", action="store_true", help="Return exit code 2 if a scientific gate fails.")
    args = parser.parse_args()
    try:
        result = check(args.evidence)
    except (OSError, ValueError, KeyError, TypeError, IndexError, np.linalg.LinAlgError) as exc:
        print(f"Saved device-refinement integrity check FAILED: {exc}", file=sys.stderr)
        return 1
    print("Saved device-refinement integrity check PASSED.")
    print(f"1 GHz, left incidence, h200 to h160 nm: max power change {result['power_change']:.8g}; "
          f"max complex amplitude change {result['complex_change']:.8g} (limit {POWER_LIMIT:.2g} each).")
    print(f"Scientific acceptance {'PASSED' if result['converged'] else 'FAILED'} for this mesh pair.")
    return 2 if args.require_converged and not result["converged"] else 0


if __name__ == "__main__":
    raise SystemExit(main())
