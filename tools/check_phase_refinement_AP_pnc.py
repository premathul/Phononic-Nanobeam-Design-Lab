#!/usr/bin/env python3
"""Check saved two-hole mesh refinement without solving. Author: Athul Prem | _AP_pnc."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
import sys

import numpy as np

from validate_perforated_AP_pnc import NUMERICAL_MODULES, POWER_LIMIT, RESIDUAL_LIMIT, ROOT, compare_cases, file_sha, unpack
from phononic.geometry_AP_pnc import read_hole_pattern
from phononic.workflow_AP_pnc import resolve_config


DEFAULT_EVIDENCE = ROOT / "verification/two_hole_phase_refinement_AP_pnc.json"
PLOT_PATH = ROOT / "figures/verification/two_hole_phase_refinement_AP_pnc.png"
MESH_SIZES = (105, 80, 60, 50)
SIDES = ("left", "right")
CHANNEL_COUNT = 4
CHECK_TOL = 1e-10
CHANNEL_FIELDS = (
    "reflection", "transmission", "reflection_flux", "transmission_flux", "reflection_gram", "transmission_gram",
    "power_balance", "flux_power_balance", "net_outward_power_ratio", "relative_residuals", "incident_powers_w",
)


def require(condition: bool, message: str) -> None:
    if not condition:
        raise ValueError(message)


def checked_array(values, shape: tuple[int, ...], label: str) -> np.ndarray:
    array = np.asarray(values, dtype=float)
    require(array.shape == shape and np.isfinite(array).all(), f"Invalid {label}: expected finite values with shape {shape}.")
    return array


def checked_file(reference: dict) -> Path:
    path = (ROOT / reference["path"]).resolve()
    require(path.is_relative_to(ROOT), f"Evidence path is outside the repository: {path}.")
    require(file_sha(path) == reference["sha256"], f"Evidence checksum differs: {path}.")
    return path


def equivalent(saved, actual, label: str) -> None:
    if isinstance(actual, dict):
        require(isinstance(saved, dict) and saved.keys() == actual.keys(), f"Recorded {label} fields differ.")
        for key, value in actual.items():
            equivalent(saved[key], value, f"{label}.{key}")
    elif isinstance(actual, (list, tuple)):
        require(isinstance(saved, list) and len(saved) == len(actual), f"Recorded {label} length differs.")
        for index, value in enumerate(actual):
            equivalent(saved[index], value, f"{label}[{index}]")
    elif isinstance(actual, (bool, np.bool_)):
        require(isinstance(saved, bool) and saved == actual, f"Recorded {label} differs.")
    elif isinstance(actual, (int, float, np.number)):
        require(isinstance(saved, (int, float)) and np.isclose(saved, actual, rtol=1e-9, atol=1e-12),
                f"Recorded {label} differs: saved {saved!r}, recomputed {actual!r}.")
    else:
        require(saved == actual, f"Recorded {label} differs.")


def check_geometry(geometry: dict, configuration: dict, label: str) -> None:
    length, width, thickness = (configuration[key] * 1e-9 for key in ("length_nm", "width_nm", "thickness_nm"))
    radii = np.asarray([diameter * 0.5e-9 for _, diameter in configuration["holes_nm"]])
    segments = configuration["circle_points"]
    require(geometry["rim_segments_min"] == geometry["rim_segments_max"] == segments,
            f"{label}: circular faceting differs.")

    angle = 2 * np.pi / segments
    box = length * width * thickness
    removed = np.pi * thickness * np.sum(radii**2)
    solid = box - removed
    polygon_removed = removed * np.sin(angle) / angle
    meshed_solid = geometry["meshed_solid_volume_m3"]
    meshed_removed = geometry["meshed_removed_volume_m3"]
    require(0 < meshed_solid < box and 0 < meshed_removed < box, f"{label}: invalid meshed solid or removed volume.")
    # SI volumes and sagittas require relative comparisons. An absolute tolerance
    # suitable for dimensionless quantities would accept incorrect nanoscale values.
    physical = {"analytic_solid_volume_m3": solid, "analytic_removed_volume_m3": removed,
                "meshed_solid_volume_m3": box - polygon_removed, "meshed_removed_volume_m3": polygon_removed,
                "max_radial_sagitta_m": radii.max() * (1 - np.cos(angle / 2)),
                "polygon_sagitta_upper_m": radii.max() * (1 - np.cos(angle / 2))}
    for key, expected in physical.items():
        require(np.isclose(geometry[key], expected, rtol=1e-7, atol=0), f"{label}: {key} differs from fixed geometry.")

    removed_error = abs(meshed_removed - removed) / removed
    polygon_bias = 1 - np.sin(angle) / angle
    errors = {"relative_solid_volume_error": abs(meshed_solid - solid) / solid,
              "relative_removed_volume_error": removed_error, "predicted_polygon_removed_volume_bias": polygon_bias,
              "polygon_volume_difference": abs(removed_error - polygon_bias)}
    for key, expected in errors.items():
        equivalent(geometry[key], expected, f"{label} {key}")


def check_level(level: dict, size_nm: int, expected_configuration: dict, source_hashes: dict) -> tuple[np.ndarray, dict]:
    label = f"h={size_nm} nm"
    require(level["status"] == "solved", f"{label}: missing completed solve evidence.")
    configuration = level["configuration"]
    require(configuration["body_size_nm"] == size_nm, f"{label}: body mesh target differs.")
    for key, value in expected_configuration.items():
        equivalent(configuration[key], value, f"{label} physical input {key}")
    hashes = level["implementation"]["source_sha256"]
    require(all(hashes.get(path) == value for path, value in source_hashes.items()),
            f"{label}: numerical source differs from the checked repository.")

    geometry = level["geometry_checks"]
    dofs = geometry["displacement_dofs"]
    require(isinstance(dofs, int) and dofs > 0 and geometry["tetrahedra"] > 0, f"{label}: invalid volume mesh counts.")
    check_geometry(geometry, expected_configuration, label)

    scattering = level["scattering"]
    n = scattering["propagating_modes_per_direction"]
    trace_dofs = scattering["port_dofs"]
    require(n == CHANNEL_COUNT and isinstance(trace_dofs, int) and 3 * n <= trace_dofs < dofs,
            f"{label}: unexpected propagating-channel or trace dimension.")
    require(set(scattering["channels"]) == set(SIDES), f"{label}: both incidence directions are required.")
    coordinates = checked_array(scattering["trace_coordinates_m"], (2, trace_dofs), f"{label} trace coordinates")
    for component in np.split(coordinates, 3, axis=1)[1:]:
        require(np.array_equal(component, coordinates[:, :trace_dofs // 3]), f"{label}: component trace coordinates differ.")
    triangles = np.asarray(scattering["trace_triangles"])
    require(triangles.ndim == 2 and triangles.shape[1] == 3 and triangles.size > 0 and triangles.dtype.kind in "iu"
            and triangles.min() >= 0 and triangles.max() < trace_dofs // 3, f"{label}: invalid trace triangulation.")
    k = checked_array(scattering["wavenumbers_positive_power_rad_m"], (n,), f"{label} propagation wavenumbers")
    separation = np.diff(np.sort(k))
    require(np.all(k != 0) and np.all(separation > 1e-8 * np.max(abs(k))),
            f"{label}: the saved roots do not resolve individual propagating branches.")
    modes = unpack(checked_array(scattering["trace_mode_coordinates"], (trace_dofs, n, 2), f"{label} trace modes"))
    require(np.allclose(np.linalg.norm(modes, axis=0), 1, atol=CHECK_TOL, rtol=0),
            f"{label}: saved mass-weighted modes are not normalized.")
    overlap = checked_array(scattering["time_reversal_mode_overlap"], (n,), f"{label} time-reversal overlap")
    k_error = checked_array(scattering["time_reversal_relative_wavenumber_error"], (n,), f"{label} time-reversal wavenumber error")
    require(np.all((0.99 <= overlap) & (overlap <= 1 + CHECK_TOL)) and np.all((0 <= k_error) & (k_error <= 1e-8)),
            f"{label}: time-reversal pairing is unresolved.")

    matrix = unpack(checked_array(scattering["scattering_matrix_time_reversal_basis"], (2 * n, 2 * n, 2),
                                  f"{label} complex scattering matrix"))
    reciprocity = float(np.max(abs(matrix - matrix.T)))
    unitarity = float(np.max(abs(matrix.conj().T @ matrix - np.eye(2 * n))))
    max_budget, max_flux, max_residual = 0.0, 0.0, 0.0
    root_lists = []
    for index, side in enumerate(SIDES):
        channel = scattering["channels"][side]
        roots = channel["safe_root_indices"]
        require(len(roots) == n and all(type(root) is int and 0 <= root < 2 * trace_dofs for root in roots),
                f"{label} {side}: invalid channel roots.")
        root_lists.extend(roots)
        values = {key: checked_array(channel[key], (n,), f"{label} {side} {key}") for key in CHANNEL_FIELDS}
        incident = slice(index * n, (index + 1) * n)
        through = slice((1 - index) * n, (2 - index) * n)
        for key, rows in (("reflection", incident), ("transmission", through)):
            modal = np.sum(abs(matrix[rows, incident]) ** 2, axis=0)
            require(np.allclose(modal, values[key], atol=CHECK_TOL, rtol=0), f"{label} {side}: {key} differs from |S|².")
            for estimator in ("_flux", "_gram"):
                max_flux = max(max_flux, float(np.max(abs(values[key] - values[key + estimator]))))
        require(np.allclose(values["incident_powers_w"], 1, atol=CHECK_TOL, rtol=0), f"{label} {side}: incident power is not unit normalized.")
        for suffix, balance in (("", "power_balance"), ("_flux", "flux_power_balance")):
            calculated = 1 - values["reflection" + suffix] - values["transmission" + suffix]
            require(np.allclose(calculated, values[balance], atol=CHECK_TOL, rtol=0), f"{label} {side}: inconsistent {balance}.")
            max_budget = max(max_budget, float(np.max(abs(calculated))))
        max_budget = max(max_budget, float(np.max(abs(values["net_outward_power_ratio"]))))
        require(np.all(values["relative_residuals"] >= 0), f"{label} {side}: negative linear residual.")
        max_residual = max(max_residual, float(max(values["relative_residuals"])))
    require(len(set(root_lists)) == 2 * n, f"{label}: directional root indices overlap.")
    safe_residual = scattering["safe_max_relative_residual"]
    require(0 <= safe_residual < RESIDUAL_LIMIT and max_residual < RESIDUAL_LIMIT, f"{label}: excessive solve residual.")
    for key, value in (("max_power_budget_error", max_budget), ("max_modal_flux_difference", max_flux),
                       ("max_reciprocity_error", reciprocity)):
        equivalent(scattering[key], value, f"{label} {key}")
    require(max(max_budget, max_flux, reciprocity, unitarity) < POWER_LIMIT,
            f"{label}: power budget, flux agreement, reciprocity, or flux unitarity exceeds {POWER_LIMIT}.")
    return matrix, {"displacement_dofs": dofs, "max_budget_error": max_budget, "max_modal_flux_error": max_flux,
                    "max_reciprocity_error": reciprocity, "max_flux_unitarity_error": unitarity}


def check_csv(path: Path, levels: dict, matrices: dict) -> None:
    with path.open(newline="", encoding="utf-8") as stream:
        reader = csv.DictReader(stream)
        rows = {}
        for row in reader:
            key = (int(row["target_body_mesh_nm"]), row["incident_side"], int(row["mode_index"]))
            require(key not in rows, f"Duplicate CSV channel: {key}.")
            rows[key] = row
    expected_keys = {(h, side, i + 1) for h in MESH_SIZES for side in SIDES for i in range(CHANNEL_COUNT)}
    require(rows.keys() == expected_keys, "Channel CSV must contain all four modes for both directions and all mesh levels.")
    n = CHANNEL_COUNT
    for index, side in enumerate(SIDES):
        same_channel = np.asarray([np.diag(matrices[h][(1 - index) * n:(2 - index) * n, index * n:(index + 1) * n])
                                   for h in MESH_SIZES])
        phases = np.unwrap(np.angle(same_channel), axis=0)
        for step, h in enumerate(MESH_SIZES):
            level = levels[str(h)]
            scattering = level["scattering"]
            channel = scattering["channels"][side]
            for i in range(n):
                row = rows[h, side, i + 1]
                require(row["author"] == "Athul Prem" and row["file_tag"] == "_AP_pnc", "CSV attribution differs.")
                for key, value in {"safe_root_index": channel["safe_root_indices"][i],
                                   "displacement_dofs": level["geometry_checks"]["displacement_dofs"],
                                   "port_dofs": scattering["port_dofs"],
                                   "wavenumber_positive_power_rad_m": scattering["wavenumbers_positive_power_rad_m"][i],
                                   "transmission_amplitude_real": same_channel[step, i].real,
                                   "transmission_amplitude_imag": same_channel[step, i].imag,
                                   "same_channel_transmission_phase_rad": phases[step, i],
                                   **{name: channel[name][i] for name in CHANNEL_FIELDS}}.items():
                    equivalent(float(row[key]), value, f"CSV h={h} {side} channel {i + 1} {key}")


def check(path: Path) -> tuple[list[dict], list[dict]]:
    evidence = json.loads(path.read_bytes())
    reference = evidence["reference_configuration"]
    config = resolve_config(json.loads(checked_file(reference).read_bytes()))
    equivalent(reference["resolved"], config, "resolved reference configuration")
    previous = evidence["previous_evidence"]
    original = json.loads(checked_file(previous["validation_report"]).read_bytes())
    checked_file(previous["validation_csv"])
    geometry, mesh, analysis = (config[key] for key in ("geometry", "mesh", "analysis"))
    require(analysis["kind"] == "scattering" and analysis["boundary"] == "matched_ports" and analysis["frequency_GHz"] == 1
            and mesh["kind"] == "gmsh" and mesh["element_order"] == 2 and mesh["mesh_size_nm"] == MESH_SIZES[-1],
            "Unexpected reference solve settings.")
    require(evidence["frequency_hz"] == analysis["frequency_GHz"] * 1e9 and evidence["element_order"] == mesh["element_order"],
            "Top-level frequency or element order differs from the reference solve.")
    pattern = geometry["pattern"]
    pattern_reference = reference["hole_pattern_provenance"]
    require(pattern_reference["source_csv"] == pattern["csv"], "Reference hole table differs from saved provenance.")
    hole_file = checked_file({"path": pattern["csv"], "sha256": pattern_reference["source_sha256"]})
    holes = read_hole_pattern(hole_file).holes
    expected = {key: geometry[key] for key in ("length_nm", "width_nm", "thickness_nm")}
    expected.update({"holes_nm": [(hole.x_m * 1e9 + pattern["x_offset_nm"], hole.diameter_m * 1e9) for hole in holes],
                     "port_size_nm": mesh["port_size_nm"], "circle_points": mesh["circle_resolution"]})
    require(len(holes) == 2, "Reference geometry must contain two holes.")

    numerical_paths = [f"src/phononic/{module}_AP_pnc.py" for module in NUMERICAL_MODULES]
    source_hashes = {name: file_sha(ROOT / name) for name in numerical_paths}
    require(evidence["numerical_source_sha256"] == source_hashes, "Top-level numerical source checksums differ.")
    levels = evidence["levels"]
    require(levels.keys() == {str(size) for size in MESH_SIZES}, "Unexpected set of body mesh levels.")
    for h, name in ((105, "curvature_fine"), (80, "body_phase_fine")):
        level = levels[str(h)]
        original_case = original["case_metrics"][name]
        original_data = original_case["scattering"]
        require(all(level[key] == value for key, value in original_case.items() if key != "scattering")
                and all(level["scattering"][key] == value for key, value in original_data.items()),
                f"h={h} nm: previously published solve data changed.")
    first = levels[str(MESH_SIZES[0])]["configuration"]
    fixed_configuration = {key: value for key, value in first.items() if key != "body_size_nm"}
    require(evidence["fixed_geometry"] == fixed_configuration, "Fixed geometry metadata differ from the saved levels.")
    material = config["material"]
    expected_material = {key: material[key] for key in ("C11_Pa", "C12_Pa", "C44_Pa")}
    expected_material.update({"rho_kg_m3": material["density_kg_m3"], "crystal_to_lab_rotation": material["crystal_rotation"],
                              "policy": material["temperature_policy"]})
    matrices, metrics = {}, []
    for h in MESH_SIZES:
        level = levels[str(h)]
        require(fixed_configuration == {key: value for key, value in level["configuration"].items() if key != "body_size_nm"},
                f"h={h} nm: a fixed input besides body mesh size changed.")
        if h in MESH_SIZES[-2:]:
            equivalent(level["material"], expected_material, f"h={h} nm material")
            require(level["element_order"] == mesh["element_order"] and level["frequency_hz"] == analysis["frequency_GHz"] * 1e9,
                    f"h={h} nm: frequency or element order differs.")
        matrix, metric = check_level(level, h, expected, source_hashes)
        matrices[h] = matrix
        metrics.append(metric)
    require(np.all(np.diff([metric["displacement_dofs"] for metric in metrics]) > 0),
            "Refining the body mesh must increase its displacement DOFs.")
    comparisons = []
    for coarse, fine in zip(MESH_SIZES[:-1], MESH_SIZES[1:]):
        result = compare_cases(levels[str(coarse)], levels[str(fine)])
        equivalent(evidence["successive_comparisons"][f"{coarse}_to_{fine}"], result, f"h={coarse}→{fine} nm comparison")
        comparisons.append(result)
    final = comparisons[-1]
    acceptance = evidence["acceptance"]
    expected_acceptance = {"reference_mesh_nm": MESH_SIZES[-2], "refined_mesh_nm": MESH_SIZES[-1],
                           "power_change_limit": POWER_LIMIT, "complex_change_limit": POWER_LIMIT,
                           "power_passed": final["power_converged"], "complex_response_passed": final["complex_scattering_change_below_0p01"]}
    changes = [item["deembedded_complex_scattering_change"] for item in comparisons]
    expected_acceptance["successive_complex_changes_decreased"] = bool(np.all(np.diff(changes) < 0))
    expected_acceptance["passed"] = expected_acceptance["power_passed"] and expected_acceptance["complex_response_passed"]
    for key, value in expected_acceptance.items():
        equivalent(acceptance[key], value, f"final acceptance {key}")
    status = "passed_final_body_mesh_pair" if expected_acceptance["passed"] else "failed_final_body_mesh_pair"
    require(evidence["status"] == status, "Top-level status disagrees with the recomputed final mesh-pair gates.")
    csv_path = path.with_suffix(".csv")
    require(file_sha(csv_path) == evidence["channel_csv_sha256"], "Companion channel CSV checksum differs.")
    check_csv(csv_path, levels, matrices)
    require(expected_acceptance["passed"], "Final mesh pair exceeded the 0.01 power or complex scattering change threshold.")
    return comparisons, metrics


def plot(comparisons: list[dict], metrics: list[dict]) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, axis = plt.subplots(figsize=(7.8, 4.8), layout="constrained")
    x = [metric["displacement_dofs"] for metric in metrics[1:]]
    for key, label, marker in (("deembedded_complex_scattering_change", "Complex scattering amplitude", "o"),
                               ("max_absolute_power_change", "Reflection or transmission power", "s")):
        axis.semilogy(x, [item[key] for item in comparisons], marker=marker, label=label)
    axis.axhline(POWER_LIMIT, color="0.4", linestyle="--", linewidth=1, label="Acceptance threshold (0.01)")
    axis.set_xticks(x, [f"{dofs:,}\n{a} → {b} nm" for dofs, a, b in zip(x, MESH_SIZES[:-1], MESH_SIZES[1:])])
    axis.set(xlabel="Displacement degrees of freedom on the finer mesh\nTarget body mesh step",
             ylabel="Maximum absolute change between meshes", title="1 GHz two-hole guide · fixed geometry and ports")
    axis.grid(axis="y", which="major", alpha=0.25)
    axis.legend(loc="best", frameon=False)
    fig.suptitle("Athul Prem | athul_simulation_pnc_two_hole_phase_refinement", fontsize=11)
    PLOT_PATH.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(PLOT_PATH, dpi=180)
    plt.close(fig)
    print(f"Saved {PLOT_PATH.relative_to(ROOT)}")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--evidence", type=Path, default=DEFAULT_EVIDENCE)
    parser.add_argument("--plot", action="store_true", help="Render the saved mesh-comparison figure.")
    args = parser.parse_args()
    try:
        comparisons, metrics = check(args.evidence)
    except (OSError, ValueError, KeyError, TypeError) as exc:
        print(f"Saved phase refinement check failed: {exc}", file=sys.stderr)
        return 2
    for coarse, fine, result in zip(MESH_SIZES[:-1], MESH_SIZES[1:], comparisons):
        print(f"{coarse} to {fine} nm: max power change {result['max_absolute_power_change']:.8g}, "
              f"max complex change {result['deembedded_complex_scattering_change']:.8g}")
    observed = np.asarray([item["deembedded_complex_scattering_change"] for item in comparisons])
    if np.all(np.diff(observed) < 0):
        print("The observed complex changes decreased across these three comparisons.")
    print("Saved two-hole refinement passed the final-pair checks at 1 GHz.")
    if args.plot:
        plot(comparisons, metrics)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
