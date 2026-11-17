"""Replay the saved 100 MHz z4 control and compare its signed z6 response. Athul Prem | _AP_pnc."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path, PurePosixPath
import sys
from time import monotonic

if __name__ == "__main__":
    for variable in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "BLIS_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
        os.environ[variable] = "1"

import numpy as np
import psutil
from scipy.sparse import load_npz

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from tools import compare_scattering_AP_pnc as comparison
from tools import replay_lawrence_device_point_AP_pnc as replay

ARCHIVE = ROOT / "results/runs/full_device_resource_profile/full_h80_operator_1r5fdlft"
SECTION = ROOT / "results/runs/full_device_port_resolution/12x4/attempt_lpnbd9kr"
STAGE_SHA = "1c50971e27bfc52e63000c9d25df100e21bbe28b7fa47d7549f82cf465531a78"
SECTION_SHA = "2a2aa9347921ac6940f851f2eb53bc06480a28a6cf67c27a000faa357b1404e8"
SECTION_RECEIPT_SHA = "677c64a816ea42a8b37e1358d5549d63ecaeec7623a9366139ba64d301bbea4e"
TRACE_SHA = "2752f71b459282bb33610b39ce99fd25eee6784a510e4c6013415a9357e6fd7b"
Z6_RECEIPT = ROOT / "results/runs/low_band_device_AP_pnc/local_checks/independent_replay_coherent_1335483_AP_pnc.json"
Z6_RECEIPT_SHA = "cb030385298fdbacd8ee84be40527a4a443d6060a3bbee191da39ce395131a2c"
Z6_PUBLIC = ROOT / "results/runs/low_band_device_AP_pnc/h80-z6-low-band-export-1335483/historical_26hole_layered_h80_ny12_z6-bb55473510a2"
RUNNER = ROOT / "tools/run_lawrence_low_band_z4_AP_pnc.py"
RUNNER_SHA = "59539b922aa2e37a059bb0f159442b0bac5347b7996bb097a343cf4a4a2216a8"
COORDINATOR = "src/phononic/resumable_scattering_AP_pnc.py"
COORDINATOR_PAIR = ("68c12549b47a9ec98abf7b39d78ad597dd114da00e326adf8ebbec060947b53f",
                    "dba6002d982647810f3fc714fdce73a0f34e306e06e8519df2d54778002205ec")
DOFS, PORT_DOFS, FREQUENCY = 426627, 675, 100_000_000
FIELDS = ("wavenumbers_rad_m", "displacements", "axial_forces", "positive_indices", "negative_indices",
          "propagating", "powers_w", "relative_residuals")
require = replay.require


def read(path):
    return json.loads(path.read_bytes())


def artifact(path):
    return {"path": str(path.resolve()), "sha256": replay.file_sha(path), "bytes": path.stat().st_size}


def same_artifact(actual, expected):
    return all(actual[key] == expected[key] for key in ("bytes", "sha256"))


def verify_numerical_sources(original, current):
    first, second = (dict(source["source_sha256"]) for source in (original, current))
    if (first.get(COORDINATOR), second.get(COORDINATOR)) == COORDINATOR_PAIR:
        first.pop(COORDINATOR)
        second.pop(COORDINATOR)
    require(first == second and original["packages"] == current["packages"],
            "The z4 archive and current numerical kernels or package versions differ.")


def replay_modes(modes, section):
    """Recover each root's original equilibrium and +x force from the saved operators."""
    U, F, k = (modes[key] for key in ("displacements", "axial_forces", "wavenumbers_rad_m"))
    size = len(section["coordinates_yz_m"])
    plus, minus = (modes[key] for key in ("positive_indices", "negative_indices"))
    opened = modes["propagating"]
    require(U.shape == F.shape == (size, 2 * size) and k.shape == opened.shape == (2 * size,)
            and opened.dtype == np.bool_ and plus.shape == minus.shape == (size,)
            and np.array_equal(np.sort(np.r_[plus, minus]), np.arange(2 * size))
            and all(np.isfinite(modes[key]).all() for key in FIELDS)
            and np.all(k[opened].imag == 0) and np.all(k[~opened].imag != 0)
            and np.all((modes["relative_residuals"] >= 0) & (modes["relative_residuals"] <= 1e-8)),
            "The SAFE roots or signed directional inventories are incomplete or invalid.")
    K0, mass, Q, G = (section[key] for key in ("K0", "mass_kg_per_m", "Q", "G"))
    omega = 2 * np.pi * FREQUENCY
    dynamic = K0 - omega**2 * mass
    norms = [np.linalg.norm(matrix, ord=np.inf) for matrix in (dynamic, G.T - G, Q)]
    worst_root = worst_force = worst_power = 0.0
    for start in range(0, 2 * size, 64):
        part = slice(start, start + 64)
        field, wave = U[:, part], k[part]
        axial, coupling = Q @ field, G @ field
        predicted_force = coupling + 1j * wave * axial
        # This form uses the independently reconstructed force in the original
        # transverse balance, rather than a transformed pencil residual.
        imbalance = dynamic @ field + 1j * wave * (G.T @ field - predicted_force)
        denominator = (norms[0] + abs(wave) * norms[1] + abs(wave)**2 * norms[2]) * np.linalg.norm(field, axis=0)
        force_scale = np.linalg.norm(coupling, axis=0) + abs(wave) * np.linalg.norm(axial, axis=0)
        require(np.all(np.isfinite(denominator) & (denominator > 0))
                and np.all(np.isfinite(force_scale) & (force_scale > 0)), "A SAFE replay normalization is invalid.")
        worst_root = max(worst_root, float(np.max(np.linalg.norm(imbalance, axis=0) / denominator)))
        worst_force = max(worst_force, float(np.max(np.linalg.norm(F[:, part] - predicted_force, axis=0) / force_scale)))
        powers = replay.work_power(field, predicted_force, omega)
        worst_power = max(worst_power, float(np.max(abs(powers - modes["powers_w"][part]))))
    require(worst_root <= 1e-8 and worst_force <= 1e-8 and worst_power <= 1e-7,
            "A SAFE root, reconstructed traction or signed power fails its original gate.")
    gram_errors = {}
    for label, sign, ids in (("positive", 1, plus), ("negative", -1, minus)):
        expected = sign * opened[ids].astype(float)
        gram = replay.work_gram(U[:, ids], F[:, ids])
        error = float(np.max(abs(gram - np.diag(expected))))
        require(error <= 1e-7 and np.all(sign * modes["powers_w"][ids[opened[ids]]] > 0)
                and np.all(sign * k[ids[~opened[ids]]].imag > 0),
                "A full SAFE directional flux Gram or evanescent decay direction fails.")
        gram_errors[label] = error
    incident, returning = plus[opened[plus]], minus[opened[minus]]
    require(len(incident) == len(returning) == 4, "The z4 point does not have four open channels per direction.")
    return incident.tolist(), returning.tolist(), {"roots_checked": 2 * size,
            "maximum_original_backward_residual": worst_root, "maximum_relative_traction_error": worst_force,
            "maximum_signed_power_difference_w": worst_power, "full_directional_flux_gram_errors_w": gram_errors}


def check_stage_identity(stage, original):
    require(stage["status"] == "operator_staged" and stage["frequency_hz"] == FREQUENCY
            and stage["displacement_dofs"] == DOFS and stage["trace_sha256"] == TRACE_SHA
            and stage["body_stage"]["sha256"] == STAGE_SHA
            and same_artifact(stage["mesh"], original["mesh"])
            and same_artifact(stage["mesh_summary"], original["mesh_checkpoint_summary"])
            and all(same_artifact(stage["array_files"][name], original[name]) for name in ("K", "M"))
            and all(stage[key] is False for key in ("mesh_generated", "body_assembly_performed", "factorization_performed",
                                                   "phase_adjustment_applied")),
            "The z4 stage differs from the fixed archived body, trace or no-factor preparation scope.")
    config = read(replay.check_link(ARCHIVE / "configuration_AP_pnc.json", original["configuration"]))
    require(all(stage["configuration"][key] == config[key] for key in ("geometry", "mesh", "material"))
            and stage["configuration"]["analysis"] == {**config["analysis"], "frequency_GHz": 0.1},
            "The z4 control changed physical inputs or incidence.")
    verify_numerical_sources(original["source"], stage["implementation"])


def replay_point(stage_path, safe_path, solve_path, export_path):
    started = monotonic()
    original = replay.read_receipt(ARCHIVE / "stages_AP_pnc.json", STAGE_SHA)
    section_receipt = replay.read_receipt(SECTION / "checkpoint_AP_pnc.json", SECTION_RECEIPT_SHA)
    require(section_receipt["arrays"]["sha256"] == SECTION_SHA, "Archived z4 section identity differs.")
    export = read(export_path)
    stage = replay.read_receipt(replay.check_link(stage_path, export["stage"]), export["stage"]["sha256"])
    solved = replay.read_receipt(replay.check_link(solve_path, export["solve"]), export["solve"]["sha256"])
    safe = replay.read_receipt(replay.check_link(safe_path, stage["safe_receipt"]), stage["safe_receipt"]["sha256"])
    check_stage_identity(stage, original)
    require(safe["status"] == "safe_ready" and solved["status"] == "drives_accepted" and export["status"] == "completed_unverified"
            and solved["stage"] == export["stage"] and stage["safe_section"]["sha256"] == SECTION_SHA
            and safe["frequency_hz"] == FREQUENCY and same_artifact(safe["safe_modes"], stage["safe_modes"])
            and safe["trace_sha256"] == TRACE_SHA and safe["safe_section"]["sha256"] == SECTION_SHA
            and safe["section_receipt"]["sha256"] == SECTION_RECEIPT_SHA,
            "The accepted phases do not bind the original body, new modes and complete solves.")
    require(replay.file_sha(RUNNER) == RUNNER_SHA
            and all(phase["driver"]["sha256"] == RUNNER_SHA for phase in (safe, stage, solved, export))
            and all(phase["reviewed_sources"] == stage["reviewed_sources"]
                    and phase["superlu_library"] == stage["superlu_library"] for phase in (safe, solved, export)),
            "The phase execution or SuperLU source identity differs from the reviewed z4 runner.")
    require(all(phase[flag] is False for phase in (stage, export)
                for flag in ("spatial_convergence_established", "frequency_convergence_established")),
            "A finite z4 control claims spatial or frequency convergence.")
    array_links = stage["array_files"]
    arrays = {name: replay.check_link(stage_path.parent / file, array_links[name])
              for name, file in (("A", "A_AP_pnc.npz"), ("rhs", "rhs_AP_pnc.npy"))}
    body = {name: replay.check_link(ARCHIVE / f"{name}_AP_pnc.npz", original[name]) for name in ("K", "M")}
    mesh_path = replay.check_link(ROOT / original["mesh"]["path"], original["mesh"])
    modes_path = replay.check_link(safe_path.parent / PurePosixPath(stage["safe_modes"]["path"]).name, stage["safe_modes"])
    section_path = SECTION / "arrays_AP_pnc.npz"
    require(replay.file_sha(section_path) == SECTION_SHA, "The archived z4 SAFE operators changed.")
    with np.load(section_path, allow_pickle=False) as archive, np.load(ARCHIVE / "ports_AP_pnc.npz", allow_pickle=False) as ports:
        replay.check_link(ARCHIVE / "ports_AP_pnc.npz", original["port_modes"])
        section = {key: archive[key] for key in ("coordinates_yz_m", "component_dofs_xyz", "K0", "mass_kg_per_m", "Q", "G")}
        section.update(left_dofs=ports["left_dofs"], right_dofs=ports["right_dofs"])
        archived_keys = {"wavenumbers_rad_m": "k_rad_m", "powers_w": "signed_power_w"}
        require(all(np.array_equal(ports[key], archive[archived_keys.get(key, key)]) for key in FIELDS),
                "The saved section does not reproduce all original 1 GHz z4 modes exactly.")
    with np.load(modes_path, allow_pickle=False) as archive:
        modes = {key: archive[key] for key in (*FIELDS, "frequency_hz")}
    require(modes["frequency_hz"].item() == FREQUENCY, "The new z4 SAFE modes use another frequency.")
    incident, returning, port_metrics = replay_modes(modes, section)
    require(stage["incident_port_roots"] == incident and stage["backward_port_roots"] == returning,
            "The stage roots differ from the independently replayed signed channels.")

    public = export_path.parent / PurePosixPath(export["public_summary"]["path"]).parent.name
    summary_path = replay.check_link(public / "summary_AP_pnc.json", export["public_summary"])
    summary = read(summary_path)
    require(summary["frequency_hz"] == FREQUENCY and summary["incident_side"] == "left"
            and summary["hole_count"] == 26 and summary["ndof"] == DOFS and summary["port_dofs"] == PORT_DOFS
            and summary["power_balance_passed"] is True and summary["spatial_convergence_established"] is False,
            "The public z4 point scope or input power checks differ.")
    for name in ("selected_fields_AP_pnc.npz", "selected_vertex_shapes_AP_pnc.vtu", "scattering_AP_pnc.csv",
                 "scattering_channels_AP_pnc.csv", "port_modes_AP_pnc.csv", "configuration_AP_pnc.json"):
        require(replay.file_sha(public / name) == summary["output_sha256"][name], f"Public output changed: {name}.")
    report = solved["linear_solver"]
    require(report["relative_target"] == replay.TARGET
            and report["relative_residual_limit"] == report["backward_residual_limit"] == replay.RESIDUAL
            and report["factor"]["permc_spec"] == "COLAMD"
            and 2 * DOFS <= report["factor"]["factor_nonzeros"] <= np.iinfo(np.int32).max
            and all(report[key] == stage[key] for key in ("operator_csc_sha256", "rhs_array_sha256", "operator_nonzeros")),
            "Original equation gates, factor report or array identities differ.")
    require(all(summary["linear_solver"].get(key) == report[key] for key in
                ("relative_target", "relative_residual_limit", "backward_residual_limit",
                 "operator_csc_sha256", "rhs_array_sha256", "operator_nonzeros"))
            and read(public / "configuration_AP_pnc.json") == summary["configuration"] == stage["configuration"],
            "The public summary differs from the solved equations or exact staged configuration.")
    channels = report["channels"]
    require([row["incident_port_root"] for row in channels] == incident, "Four accepted drive identities are required.")
    vectors = np.empty((DOFS, 4), dtype=np.complex128, order="F")
    for column, row in enumerate(channels):
        require(row["column"] == column and row["target_met"] is True and row["info"] == 0,
                "A z4 channel lacks its accepted original-equation record.")
        path = replay.check_link(solve_path.parent / f"accepted-{column:03d}_AP_pnc.npy", row["accepted_vector"])
        vector = np.load(path, allow_pickle=False)
        require(vector.shape == (DOFS,) and vector.dtype == np.complex128 and np.isfinite(vector).all(),
                "An accepted z4 vector has invalid dimensions or values.")
        vectors[:, column] = vector
    available, required = psutil.virtual_memory().available, array_links["A"]["bytes"] + 512 * 2**20
    require(available >= required, f"Independent replay requires {required} B available; measured {available} B.")
    operator = load_npz(arrays["A"])
    rhs = np.load(arrays["rhs"], allow_pickle=False, mmap_mode="r")
    require(operator.shape == (DOFS, DOFS) and operator.format == "csc" and operator.dtype == np.complex128
            and operator.indices.dtype == operator.indptr.dtype == np.int32 and operator.has_canonical_format
            and rhs.shape == vectors.shape and rhs.dtype == np.complex128
            and replay.array_sha(operator.data, operator.indices, operator.indptr) == stage["operator_csc_sha256"]
            and replay.array_sha(rhs) == stage["rhs_array_sha256"], "The z4 original-equation arrays differ.")
    relative, backward, rhs_norm, matrix_norm = replay.check_original_equations(operator, rhs, vectors)
    del operator
    force = replay.body_action(body["K"], vectors, expected_dofs=DOFS)
    force -= (2 * np.pi * FREQUENCY)**2 * replay.body_action(body["M"], vectors, expected_dofs=DOFS)
    with np.load(public / "selected_fields_AP_pnc.npz", allow_pickle=False) as fields:
        replay.replay_fields(fields, mesh_path, section, modes, vectors, expected_dofs=DOFS,
                             incident_roots=incident, returning_roots=returning, trace_sha256=TRACE_SHA)
        blocks, totals, power_errors, rhs_error, boundary_error, coherent = replay.replay_ports(
            modes, fields, section["left_dofs"], section["right_dofs"], vectors, force, rhs, rhs_norm,
            incident_roots=incident, returning_roots=returning)
    replay.compare_public_tables(public, blocks, totals, incident_roots=incident, returning_roots=returning)
    r, t, direct_r, direct_t, net = totals
    same_r, same_t = (np.diag(blocks[key][1]) for key in ("reflection", "transmission"))
    representative = int(np.argmax(r))
    return {"author": "Athul Prem", "file_tag": "_AP_pnc", "status": "point_verified_within_saved_discretization",
            "frequency_hz": FREQUENCY, "problem_dofs": DOFS, "port_dofs": PORT_DOFS,
            "reference_planes_m": replay.PLANES, "public_summary": artifact(summary_path),
            "phase_receipts": {name: artifact(path) for name, path in (("safe", safe_path), ("stage", stage_path),
                                ("solve", solve_path), ("export", export_path))},
            "implementation": {"verifier": artifact(Path(__file__)), "shared_replay": artifact(Path(replay.__file__)),
                               "comparison": artifact(Path(comparison.__file__))},
            "matrix_infinity_norm": matrix_norm, "maximum_relative_residual": float(max(relative)),
            "maximum_backward_residual": float(max(backward)), "relative_target": replay.TARGET,
            "relative_residual_limit": replay.RESIDUAL, "backward_residual_limit": replay.RESIDUAL,
            "power_limit": replay.POWER, "port_replay": port_metrics,
            "maximum_power_errors": dict(zip(replay.POWER_CHECKS, power_errors.tolist(), strict=True)),
            "coherent_left_incidence": coherent,
            "channels": [{"incident_root": root, "reverse_root": returning[col], "relative_residual": relative[col],
                "backward_residual": backward[col], "rhs_relative_error": rhs_error[col],
                "body_boundary_relative_residual": boundary_error[col], "reflection": r[col], "transmission": t[col],
                "reflection_flux": direct_r[col], "transmission_flux": direct_t[col], "net_outward_power_ratio": net[col],
                "converted_reflection": r[col] - same_r[col], "converted_transmission": t[col] - same_t[col]}
                for col, root in enumerate(incident)],
            "representative_field": {"incident_root": incident[representative], "column": representative,
                 "field": artifact(public / "selected_fields_AP_pnc.npz"),
                 "accepted_vector": channels[representative]["accepted_vector"], "mesh": original["mesh"],
                 "units": "m per square root watt", "basis_order": 2},
            "available_memory_bytes_before_equations": available, "required_memory_bytes_before_equations": required,
            "elapsed_seconds": monotonic() - started,
            "spatial_convergence_established": False, "frequency_convergence_established": False}


def compare_verified_point(z4_report):
    """Bind two independently accepted points before invoking existing signed gauge matching."""
    z6 = replay.read_receipt(Z6_RECEIPT, Z6_RECEIPT_SHA)
    z4_public = Path(z4_report["public_summary"]["path"]).parent
    first, fixed, first_record = comparison.read_run(z4_public)
    second, other, second_record = comparison.read_run(Z6_PUBLIC)
    require(first_record["summary_sha256"] == z4_report["public_summary"]["sha256"]
            and second_record["summary_sha256"] == z6["public_summary_sha256"]
            and z4_report["status"] == z6["status"] == "point_verified_within_saved_discretization",
            "A compared result lacks its exact independently accepted summary.")
    fixed = physical_pair(first, second, fixed, other)
    require(first_record["power_balance_passed"] is second_record["power_balance_passed"] is True
            and first["configuration"]["mesh"]["thickness_layers"] == 4
            and second["configuration"]["mesh"]["thickness_layers"] == 6
            and first["port_dofs"] == PORT_DOFS and second["port_dofs"] == 975,
            "The pair differs in physical inputs, numerical kernels, accepted power or matched thickness profiles.")
    data_a = comparison.read_scattering(z4_public, first, fixed)
    data_b = comparison.read_scattering(Z6_PUBLIC, second, fixed)
    require(data_a[0].discretization_sha256 != data_b[0].discretization_sha256, "A thickness control needs distinct meshes.")
    matches, blocks, rows = comparison.compare_blocks(data_a, data_b)
    power = max(block["maximum_power_change"] for block in blocks.values())
    complex_change = max(block["maximum_complex_change"] for block in blocks.values())
    r = blocks["reflection"]
    reference_power, candidate_power = (np.asarray(r[f"{key}_power"]) for key in ("reference", "candidate"))
    first_rank, second_rank = np.argsort(-reference_power), np.argsort(-candidate_power)
    stronger, weaker = first_rank[:2], first_rank[2:]
    return {"status": "compared", "kind": "joint_body_port_thickness_sensitivity", "reference": first_record,
            "candidate": second_record, "z6_independent_replay": artifact(Z6_RECEIPT),
            "physical_inputs": fixed, "matched_channels": matches, "blocks": blocks, "rows": rows,
            "maximum_power_change": power, "maximum_complex_change": complex_change,
            "power_change_limit": replay.POWER, "complex_change_limit": replay.POWER,
            "power_passed": power < replay.POWER, "complex_passed": complex_change < replay.POWER,
            "mesh_pair_passed": max(power, complex_change) < replay.POWER,
            "reference_reflection_order_descending": first_rank.tolist(),
            "candidate_reflection_order_in_reference_basis_descending": second_rank.tolist(),
            "two_stronger_channels_preserved": bool(set(stronger) == set(second_rank[:2])),
            "reference_stronger_weaker_reflection_separation": float(min(reference_power[stronger]) - max(reference_power[weaker])),
            "candidate_separation_of_reference_groups": float(min(candidate_power[stronger]) - max(candidate_power[weaker])),
            "phase_adjustment_applied": False, "spatial_convergence_established": False,
            "frequency_convergence_established": False,
            "scope": "One same-frequency joint body/port thickness control at the original planes. "
                     "Signed roots and their modal gauges are matched independently in each direction."}


def physical_pair(first, second, fixed, other):
    require(all(value == other[key] for key, value in fixed.items() if key != "numerical_source_sha256")
            and fixed["frequency_hz"] == FREQUENCY and fixed["incident_side"] == "left",
            "The thickness pair changed physical inputs, frequency or reference incidence.")
    # Both bodies were assembled with the saved batched kernel, even when a new
    # frequency configuration omits an assembly control because it reuses K/M.
    modules = (*comparison.CORE, "assembly_batched", "layered")
    kernels = {}
    for module in modules:
        key = f"src/phononic/{module}_AP_pnc.py"
        value = first["implementation"]["source_sha256"].get(key)
        require(comparison.is_sha256(value) and value == second["implementation"]["source_sha256"].get(key),
                "The compared points have different physical numerical kernels.")
        kernels[key] = value
    require(first["implementation"]["packages"] == second["implementation"]["packages"],
            "The thickness points use different numerical packages.")
    return {**fixed, "numerical_source_sha256": kernels}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("stage", "safe", "solve", "export"):
        parser.add_argument(f"--{name}", type=Path, required=True)
    parser.add_argument("--compare-z6", action="store_true")
    args = parser.parse_args()
    result = replay_point(*(getattr(args, name).resolve() for name in ("stage", "safe", "solve", "export")))
    if args.compare_z6:
        result["thickness_comparison"] = compare_verified_point(result)
    print(json.dumps(result, indent=2, allow_nan=False, default=lambda value: value.item() if isinstance(value, np.generic) else value))


if __name__ == "__main__":
    main()
