"""Independently replay the archived h80 z6 point at 887.5 MHz. Athul Prem | _AP_pnc."""

from __future__ import annotations

import argparse
import csv
import hashlib
import importlib.util
from itertools import permutations
import json
import os
from pathlib import Path, PurePosixPath
import platform
import sys
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
sys.path.insert(0, str(ROOT))
from tools import replay_lawrence_device_point_AP_pnc as shared

BODY = ROOT / "results/runs/full_device_resource_profile/lawrence_thickness_AP_pnc"
BODY_RECEIPT = BODY / "thickness-h80-stage-1335309/stage_AP_pnc.json"
SECTION = BODY / "thickness-h80-safe-1335308/ports_AP_pnc.npz"
INTERVAL = ROOT / "results/runs/port_count_interval_AP_pnc/study"
LEDGER = ROOT / "verification/port_count_interval_AP_pnc.json"
RUNNER = ROOT / "tools/run_lawrence_frequency_point_AP_pnc.py"
RUNNER_SHA = "9b1a74fc50ec3e2fe1282ec05f5d21783be27b2a86cb81201f122426ff148045"
SHARED_SHA = "83c98faae6f07071394fc472a1f44d96897258e09b6266db60402da63e06a6e0"
BODY_STAGE_SHA = "95c1cfde8d51d77ab74fe9d2c424c1eda0822556a30664cbe581999f97e0ea5f"
INPUTS_SHA = "e14c59270fcf53d0b26ec41fe46db25ec0d208936101e1f456c50ecf9b05ed26"
POINT_SHA = "a53557ae5365570a39046b68f12eb0fd0e9e620d7df7d080561ec998e552e6e8"
LEDGER_SHA = "ada05174f12e937bc709b124de6ff97956381ceceb9f2f8595c863b91bdd517f"
MODES_SHA = "93e4e089e6b329244b204ca10e3612653824b1e569476a51ea1924654a83b81c"
RELATION_SHA = "6792e815654f5cc882310ab51a70f80ff5a0280e7f5db62df2ec1a30feaf8e65"
RELATION_PRODUCER_SHA = "877405161e5dddc6f2f8840a7f8249ad66bd6971cc75079dad276c341028806d"
RELATION_PRODUCER = ROOT / "results/runs/device_frequency_8875_AP_pnc/local_checks/compare_device_frequency_executed_AP_pnc.py"
FREQUENCY, DOFS, PORT_DOFS = 887_500_000, 616239, 975
INCIDENT = (1877, 1881, 1870, 1858, 1851)
RETURNING = (1878, 1882, 1871, 1859, 1852)
TRACE_SHA = shared.TRACE_SHA
PLANES = shared.PLANES
require = shared.require


def artifact(path):
    return {"path": str(path.resolve()), "bytes": path.stat().st_size, "sha256": shared.file_sha(path)}


def same_artifact(actual, expected):
    """Execution-platform paths may differ; filename, bytes and content must agree."""
    def name(link):
        return link.get("path", link.get("file", "")).replace("\\", "/").rsplit("/", 1)[-1]
    return name(actual) == name(expected) and all(actual[key] == expected[key] for key in ("bytes", "sha256"))


def digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()).hexdigest()


def frequency_core():
    """Configure a private instance of the reviewed gates, preserving archived defaults.

    The mature helper's frequency is module-local. A separate module namespace
    avoids changing its 100 MHz defaults or another verifier running in-process.
    No numerical formula, tolerance or accepted archive is changed.
    """
    path = Path(shared.__file__)
    require(shared.file_sha(path) == SHARED_SHA, "The reviewed independent replay gates changed.")
    spec = importlib.util.spec_from_file_location("_frequency_replay_gates_AP_pnc", path)
    core = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(core)
    core.FREQUENCY, core.OMEGA = FREQUENCY, 2 * np.pi * FREQUENCY
    core.INCIDENT, core.RETURNING = INCIDENT, RETURNING
    return core


def read_interval():
    inputs = shared.read_receipt(INTERVAL / "inputs_AP_pnc.json", INPUTS_SHA)
    point = shared.read_receipt(INTERVAL / f"point-{FREQUENCY}_AP_pnc.json", POINT_SHA)
    ledger = shared.read_receipt(LEDGER, LEDGER_SHA)
    identity = inputs["identity"]
    accepted = ledger["independent_verification"]
    require(inputs["identity_sha256"] == digest(identity) == point["identity_sha256"]
            and point["point_sha256"] == digest({key: value for key, value in point.items() if key != "point_sha256"})
            and point["status"] == "accepted" and point["frequency_hz"] == FREQUENCY
            and point["modes"]["sha256"] == MODES_SHA
            and accepted["status"] == "verified_finite_count_interval"
            and accepted["identity_sha256"] == inputs["identity_sha256"]
            and point in ledger["points"], "The accepted frequency sample or independent interval provenance differs.")
    return inputs, point


def check_stage_identity(stage, original, inputs, point):
    archive = inputs["identity"]["archive"]
    require(stage["status"] == "operator_staged" and stage["frequency_hz"] == FREQUENCY
            and stage["displacement_dofs"] == DOFS and stage["trace_sha256"] == TRACE_SHA
            and stage["reference_planes_m"] == list(PLANES)
            and stage["body_stage"]["sha256"] == BODY_STAGE_SHA
            and all(same_artifact(stage[key], original[key]) for key in ("mesh", "mesh_summary"))
            and all(same_artifact(stage["array_files"][key], original["array_files"][key]) for key in ("K", "M"))
            and same_artifact(stage["safe_section"], original["safe_modes"])
            and same_artifact(stage["safe_receipt"], original["safe_receipt"])
            and same_artifact(stage["safe_modes"], point["modes"])
            and stage["safe_section_array_sha256"] == archive["section_array_sha256"]
            and stage["interval_inputs"]["sha256"] == INPUTS_SHA
            and stage["interval_point"]["sha256"] == POINT_SHA
            and stage["interval_ledger"]["sha256"] == LEDGER_SHA
            and stage["incident_port_roots"] == list(INCIDENT)
            and stage["backward_port_roots"] == list(RETURNING)
            and stage["open_channels"] == point["metrics"]["open_channels"]
            and stage["cutoff_status"] == "solver_resolved_at_sample"
            and stage["degeneracy_status"] == "none_detected"
            and all(stage[key] is False for key in ("mesh_generated", "body_assembly_performed",
                    "safe_eigensolve_performed", "factorization_performed", "phase_adjustment_applied",
                    "spatial_convergence_established", "frequency_convergence_established")),
            "The point differs from its fixed body, accepted frequency modes, signed ordering or finite scope.")
    config = {**original["configuration"], "analysis": {**original["configuration"]["analysis"],
                                                       "frequency_GHz": FREQUENCY / 1e9}}
    require(stage["configuration"] == config, "The point changed physical inputs, incidence or resource configuration.")
    old, modes, current = archive["original_implementation"], archive["current_implementation"], stage["implementation"]
    require(stage["original_section_implementation"] == old
            and stage["section_mode_implementation"] == modes
            and original["implementation"] == old
            and old["source_sha256"] == modes["source_sha256"] == current["source_sha256"]
            and old["packages"] == modes["packages"] == current["packages"]
            and old["platform"].startswith("Linux") and old["python"] == "3.12.12"
            and modes["platform"].startswith("Windows") and modes["python"] == "3.12.14"
            and current["platform"].startswith("Linux"),
            "The original Linux section, Windows frequency modes or current Linux provenance was changed or conflated.")


def check_phase_identity(stage, solved, exported):
    phases = (stage, solved, exported)
    require(shared.file_sha(RUNNER) == RUNNER_SHA
            and all(phase["driver"]["sha256"] == RUNNER_SHA for phase in phases)
            and solved["status"] == "drives_accepted" and exported["status"] == "completed_unverified"
            and solved["stage"] == exported["stage"]
            and all(phase["reviewed_sources"] == stage["reviewed_sources"]
                    and phase["superlu_library"] == stage["superlu_library"]
                    and phase["implementation"] == stage["implementation"] for phase in phases)
            and all(exported[flag] is False for flag in ("spatial_convergence_established", "frequency_convergence_established")),
            "The returned phases differ in source, build, original stage or finite numerical scope.")
    for module, link in stage["reviewed_sources"].items():
        path = ROOT.joinpath(*module.split(".")).with_suffix(".py")
        shared.check_link(path, link)
    for name, expected in stage["implementation"]["source_sha256"].items():
        require(shared.file_sha(ROOT / name) == expected, f"A reviewed physical source changed: {name}.")
    report = solved["linear_solver"]
    count = report["factor"]["factor_nonzeros"]
    require(report["relative_target"] == shared.TARGET
            and report["relative_residual_limit"] == report["backward_residual_limit"] == shared.RESIDUAL
            and report["factor"]["permc_spec"] == "COLAMD"
            and type(count) is int and 2 * DOFS <= count <= np.iinfo(np.int32).max
            and all(report[key] == stage[key] for key in ("operator_csc_sha256", "rhs_array_sha256", "operator_nonzeros"))
            and [row["incident_port_root"] for row in report["channels"]] == list(INCIDENT),
            "The original-equation gates, aggregate factor count or accepted drive order differs.")
    return report


def compare_public_tables(public, blocks, totals):
    """Check all 50 complex propagating entries at the selected frequency."""
    with (public / "scattering_channels_AP_pnc.csv").open(newline="", encoding="utf-8") as stream:
        rows = list(csv.DictReader(stream))
    expected = {(kind, incoming, outgoing): (column, row)
                for kind, roots in (("reflection", RETURNING), ("transmission", INCIDENT))
                for column, incoming in enumerate(INCIDENT) for row, outgoing in enumerate(roots)}
    require(len(rows) == len(expected), "The public table does not contain exactly 50 propagating coefficients.")
    for row in rows:
        key = row["kind"], int(row["incident_port_index"]), int(row["outgoing_port_index"])
        require(key in expected, "A public propagating coefficient is duplicated or has another signed root.")
        column, index = expected.pop(key)
        coeff, power = blocks[key[0]]
        published = complex(float(row["amplitude_real"]), float(row["amplitude_imag"]))
        require(int(row["incident_mode_order"]) == column + 1
                and np.isclose(published, coeff[index, column], rtol=1e-8, atol=1e-10)
                and np.isclose(float(row["power_fraction"]), power[index, column], rtol=1e-8, atol=1e-10),
                "A public complex coefficient or channel power differs from independent replay.")
    with (public / "scattering_AP_pnc.csv").open(newline="", encoding="utf-8") as stream:
        rows = list(csv.DictReader(stream))
    require(len(rows) == len(INCIDENT), "The public table does not contain exactly five incident drives.")
    r, t, direct_r, direct_t, _ = totals
    metrics = {"reflection": r, "transmission": t, "reflection_flux": direct_r, "transmission_flux": direct_t,
               "power_balance": 1 - r - t, "flux_power_balance": 1 - direct_r - direct_t}
    for column, row in enumerate(rows):
        require(int(row["incident_mode_order"]) == column + 1 and float(row["frequency_GHz"]) == FREQUENCY / 1e9
                and all(np.isclose(float(row[key]), values[column], rtol=1e-8, atol=1e-10)
                        for key, values in metrics.items()), "A public frequency or incident-drive power row differs.")


def independent_relation_group(first, second, mass):
    """Five-channel scalar overlaps and exhaustive assignment, independent of align()."""
    a, b = first["displacements"], second["displacements"]
    require(a.shape == b.shape == (len(mass), 5) and mass.shape == (len(mass), len(mass)),
            "Independent relation requires five fields on the common saved section.")
    norm_a = [float(np.vdot(a[:, row], mass @ a[:, row]).real) for row in range(5)]
    norm_b = [float(np.vdot(b[:, row], mass @ b[:, row]).real) for row in range(5)]
    require(all(np.isfinite(value) and value > 0 for value in norm_a + norm_b), "Invalid independent mass norm.")
    cross = np.empty((5, 5), dtype=complex)
    for row in range(5):
        for column in range(5):
            cross[row, column] = np.vdot(a[:, row], mass @ b[:, column]) / np.sqrt(norm_a[row] * norm_b[column])
    require(np.isfinite(cross).all(), "Independent normalized overlap is nonfinite.")
    magnitude = abs(cross)
    order = max(permutations(range(5)), key=lambda candidate: sum(magnitude[row, column] for row, column in enumerate(candidate)))
    overlap = np.array([magnitude[row, column] for row, column in enumerate(order)])
    valid = bool(np.all(overlap >= 0.99))
    for row, column in enumerate(order):
        valid &= bool(overlap[row] > max(np.delete(magnitude[row], column)) + 1e-8
                      and overlap[row] > max(np.delete(magnitude[:, column], row)) + 1e-8)
    for fields in (first, second):
        waves = np.asarray(fields["k"])
        valid &= bool(np.all(np.diff(np.sort(waves)) > 1e-8 * np.max(abs(waves))))
    return {"status": "matched" if valid else "unresolved", "complex_overlap": cross,
            "order": list(order), "overlaps": overlap,
            "gauges": np.array([cross[row, column] / overlap[row] if overlap[row] else 0
                                for row, column in enumerate(order)])}


def check_relation_arrays(published, reference, candidate, mass):
    """Reconstruct every published signed overlap, assignment, gauge and stop decision."""
    groups, maxima, mappings = {}, {}, {}
    for name in ("incoming", "reflection"):
        measured = independent_relation_group(reference[name], candidate[name], mass)
        saved = published["matched_channels"][name]
        raw = np.asarray(saved["full_normalized_complex_overlap"])
        require(raw.shape == (5, 5, 2), "Published complex overlap layout differs.")
        complex_error = float(np.max(abs(raw[..., 0] + 1j * raw[..., 1] - measured["complex_overlap"])))
        absolute_error = float(np.max(abs(np.asarray(saved["full_mass_overlap_matrix"]) - abs(measured["complex_overlap"]))))
        require(max(complex_error, absolute_error) <= 2e-14
                and saved["reference_roots"] == np.asarray(reference[name]["roots"]).tolist()
                and saved["candidate_roots_unordered"] == np.asarray(candidate[name]["roots"]).tolist()
                and saved["status"] == measured["status"], "Published overlap values, signed roots or individual gate decision differ.")
        if measured["status"] == "matched":
            gauge = np.asarray(saved["candidate_gauge_complex"])
            phases = np.asarray(saved["candidate_gauge_phase_rad"])
            roots = np.asarray(candidate[name]["roots"])[measured["order"]].tolist()
            require(saved["candidate_order"] == measured["order"] and saved["candidate_roots"] == roots
                    and gauge.shape == (5, 2)
                    and phases.shape == (5,) and np.isfinite(phases).all()
                    and np.max(abs(np.exp(1j * phases) - measured["gauges"])) <= 2e-14
                    and np.max(abs(gauge[:, 0] + 1j * gauge[:, 1] - measured["gauges"])) <= 2e-14
                    and np.max(abs(np.asarray(saved["mass_overlaps"]) - np.minimum(measured["overlaps"], 1))) <= 2e-14,
                    "Published assignment or complex gauge differs from exhaustive independent matching.")
            mappings[name] = dict(zip(saved["reference_roots"], roots, strict=True))
        else:
            require("candidate_order" not in saved and "candidate_gauge_complex" not in saved,
                    "An unresolved signed relation publishes an accepted order or gauge.")
        groups[name] = measured["status"]
        maxima[name] = {"maximum_complex_overlap_difference": complex_error,
                        "maximum_absolute_overlap_difference": absolute_error,
                        "minimum_proposed_overlap": float(min(measured["overlaps"]))}
    pairing = bool(len(mappings) == 2 and all(mappings["reflection"][reverse]
                   == candidate["reverse_partners"][mappings["incoming"][root]]
                   for root, reverse in reference["reverse_partners"].items()))
    expected = "matched" if all(status == "matched" for status in groups.values()) and pairing else "unresolved"
    if len(mappings) == 2:
        for label, fields in (("reference", reference), ("candidate", candidate)):
            observed = published["reverse_partner_consistency"][f"{label}_forward_to_reverse"]
            require({int(key): int(value) for key, value in observed.items()} == fields["reverse_partners"],
                    "A published signed reverse-partner map differs from independently reconstructed wavenumbers.")
    require(published["status"] == expected
            and published["matched_channels"]["transmission"] == published["matched_channels"]["incoming"]
            and published["reverse_partner_consistency"]["status"] == ("consistent" if pairing else "unresolved"),
            "The published reciprocal pairing or overall correspondence stop decision differs.")
    return {"status": "independently_verified_signed_relation", "relation_status": expected,
            "signed_groups": groups, "overlap_checks": maxima, "reverse_partner_consistent": pairing}


def replay_relation(path):
    """Read only pinned section arrays; no body arrays, assembly, eigensolve or solve."""
    started = monotonic()
    inputs, point = read_interval()
    original = shared.read_receipt(BODY_RECEIPT, BODY_STAGE_SHA)
    section_receipt = shared.read_receipt(SECTION.parent / "thickness_h80_safe_AP_pnc.json", original["safe_receipt"]["sha256"])
    shared.check_link(SECTION, original["safe_modes"])
    modes_path = shared.check_link(INTERVAL / f"modes-{FREQUENCY}_AP_pnc.npz", point["modes"])
    published = shared.read_receipt(path, RELATION_SHA)
    require(published["content_sha256"] == digest({key: value for key, value in published.items() if key != "content_sha256"})
            and published["reference_frequency_hz"] == 1_000_000_000 and published["candidate_frequency_hz"] == FREQUENCY
            and published["trace_dofs"] == PORT_DOFS and published["trace_sha256"] == TRACE_SHA
            and published["section_array_sha256"] == inputs["identity"]["archive"]["section_array_sha256"]
            and published["body_solve_performed"] is False and published["safe_eigensolve_performed"] is False
            and published["continuous_branch_identity_established"] is False,
            "The relation receipt or its finite section scope differs.")
    pinned = {"safe_receipt": SECTION.parent / "thickness_h80_safe_AP_pnc.json", "section": SECTION,
              "interval_inputs": INTERVAL / "inputs_AP_pnc.json", "candidate_point": INTERVAL / f"point-{FREQUENCY}_AP_pnc.json",
              "candidate_modes": modes_path, "interval_verification": LEDGER}
    for role, actual in pinned.items():
        require(same_artifact(published["inputs"][role], artifact(actual)), f"The relation's published input differs: {role}.")
    require(published["producer"]["sha256"] == RELATION_PRODUCER_SHA
            and shared.file_sha(RELATION_PRODUCER) == RELATION_PRODUCER_SHA
            and RELATION_PRODUCER.stat().st_size == published["producer"]["bytes"],
            "The preserved, executed relation producer source differs.")

    def signed(archive, frequency, incident, returning):
        k, u, f, opened = (archive[key] for key in ("wavenumbers_rad_m", "displacements", "axial_forces", "propagating"))
        plus, minus = (archive[key] for key in ("positive_indices", "negative_indices"))
        require(float(archive["frequency_hz"]) == frequency
                and np.array_equal(plus[opened[plus]], incident) and np.array_equal(minus[opened[minus]], returning)
                and np.array_equal(np.sort(np.r_[plus, minus]), np.arange(2 * PORT_DOFS)),
                "The independent signed inventory differs from its accepted receipt.")
        result = {}
        for name, roots, sign in (("incoming", incident, 1), ("reflection", returning, -1)):
            wave = k[roots]
            power = np.pi * frequency * np.imag(np.sum(u[:, roots].conj() * f[:, roots], axis=0))
            require(np.all(wave.imag == 0) and np.isfinite(u[:, roots]).all() and np.isfinite(f[:, roots]).all()
                    and np.allclose(power, sign, rtol=0, atol=1e-7), "Independent signed transport is invalid.")
            result[name] = {"roots": roots, "k": wave.real, "displacements": u[:, roots]}
        partners = {}
        for root in incident:
            matches = [reverse for reverse in returning if np.isclose(k[root], -k[reverse], rtol=1e-12, atol=0)]
            require(len(matches) == 1, "Independent reciprocal wavenumber pairing is ambiguous.")
            partners[int(root)] = int(matches[0])
        require(len(set(partners.values())) == 5, "Independent reciprocal pairing is not bijective.")
        result["reverse_partners"] = partners
        return result

    with np.load(SECTION, allow_pickle=False) as section, np.load(modes_path, allow_pickle=False) as modes:
        mass = section["M"]
        reference = signed(section, 1_000_000_000, section_receipt["incident_port_roots"], section_receipt["backward_port_roots"])
        candidate = signed(modes, FREQUENCY, list(INCIDENT), list(RETURNING))
        result = check_relation_arrays(published, reference, candidate, mass)
    return {"author": "Athul Prem", "file_tag": "_AP_pnc", **result, "relation": artifact(path),
            "verifier": artifact(Path(__file__)), "frequency_hz": FREQUENCY,
            "body_arrays_loaded": False, "body_solve_performed": False, "safe_eigensolve_performed": False,
            "continuous_branch_identity_established": False, "elapsed_seconds": monotonic() - started}


def replay_point(stage_path, solve_path, export_path):
    started = monotonic()
    core = frequency_core()
    inputs, point = read_interval()
    original = shared.read_receipt(BODY_RECEIPT, BODY_STAGE_SHA)
    exported = json.loads(export_path.read_bytes())
    stage = shared.read_receipt(shared.check_link(stage_path, exported["stage"]), exported["stage"]["sha256"])
    solved = shared.read_receipt(shared.check_link(solve_path, exported["solve"]), exported["solve"]["sha256"])
    check_stage_identity(stage, original, inputs, point)
    report = check_phase_identity(stage, solved, exported)
    arrays = stage["array_files"]
    body = {key: shared.check_link(BODY_RECEIPT.parent / f"{key}_AP_pnc.npz", arrays[key]) for key in ("K", "M")}
    paths = {key: shared.check_link(stage_path.parent / name, arrays[key])
             for key, name in (("A", "A_AP_pnc.npz"), ("rhs", "rhs_AP_pnc.npy"))}
    mesh = shared.check_link(BODY.joinpath(*PurePosixPath(original["mesh"]["path"]).parts[-4:]), original["mesh"])
    shared.check_link(mesh.parent / "summary_AP_pnc.json", original["mesh_summary"])
    shared.check_link(SECTION, stage["safe_section"])
    shared.check_link(SECTION.parent / "thickness_h80_safe_AP_pnc.json", stage["safe_receipt"])
    modes_path = shared.check_link(INTERVAL / f"modes-{FREQUENCY}_AP_pnc.npz", stage["safe_modes"])
    for key, path in (("interval_inputs", INTERVAL / "inputs_AP_pnc.json"),
                      ("interval_point", INTERVAL / f"point-{FREQUENCY}_AP_pnc.json"), ("interval_ledger", LEDGER)):
        shared.check_link(path, stage[key])

    public = export_path.parent / PurePosixPath(exported["public_summary"]["path"]).parent.name
    summary_path = shared.check_link(public / "summary_AP_pnc.json", exported["public_summary"])
    summary = json.loads(summary_path.read_bytes())
    require(summary["frequency_hz"] == FREQUENCY and summary["incident_side"] == "left"
            and summary["hole_count"] == 26 and summary["ndof"] == DOFS and summary["port_dofs"] == PORT_DOFS
            and summary["power_balance_passed"] is True and summary["spatial_convergence_established"] is False
            and summary["configuration"] == stage["configuration"]
            and all(summary["linear_solver"].get(key) == report[key] for key in
                    ("relative_target", "relative_residual_limit", "backward_residual_limit",
                     "operator_csc_sha256", "rhs_array_sha256", "operator_nonzeros")),
            "The public point differs from its five solved drives or exact staged configuration.")
    field_name, vtu_name = "selected_fields_AP_pnc.npz", "selected_vertex_shapes_AP_pnc.vtu"
    for name in (field_name, vtu_name, "scattering_AP_pnc.csv", "scattering_channels_AP_pnc.csv",
                 "port_modes_AP_pnc.csv", "configuration_AP_pnc.json"):
        require(shared.file_sha(public / name) == summary["output_sha256"][name], f"Public output changed: {name}.")
    require(json.loads((public / "configuration_AP_pnc.json").read_bytes()) == stage["configuration"],
            "The public configuration differs from the staged physical inputs.")

    available = psutil.virtual_memory().available
    required = max(arrays[key]["bytes"] for key in ("A", "K", "M")) + 3 * DOFS * len(INCIDENT) * 16 + 512 * 2**20
    require(available >= required, f"Independent replay requires {required} B available; measured {available} B.")
    vectors = np.empty((DOFS, len(INCIDENT)), dtype=np.complex128, order="F")
    for column, row in enumerate(report["channels"]):
        require(row["column"] == column and row["target_met"] is True and row["info"] == 0,
                "An incident channel lacks an accepted original-equation record.")
        path = shared.check_link(solve_path.parent / f"accepted-{column:03d}_AP_pnc.npy", row["accepted_vector"])
        vector = np.load(path, allow_pickle=False)
        require(vector.shape == (DOFS,) and vector.dtype == np.complex128 and np.isfinite(vector).all(),
                "An accepted displacement vector has invalid dimensions, type or values.")
        vectors[:, column] = vector
    operator = load_npz(paths["A"])
    rhs = np.load(paths["rhs"], allow_pickle=False, mmap_mode="r")
    require(operator.shape == (DOFS, DOFS) and operator.format == "csc" and operator.has_canonical_format
            and operator.dtype == np.complex128 and operator.indices.dtype == operator.indptr.dtype == np.int32
            and operator.nnz == stage["operator_nonzeros"] and np.isfinite(operator.data).all()
            and rhs.shape == vectors.shape and rhs.dtype == np.complex128 and np.isfinite(rhs).all()
            and shared.array_sha(operator.data, operator.indices, operator.indptr) == stage["operator_csc_sha256"]
            and shared.array_sha(rhs) == stage["rhs_array_sha256"], "The original complex128 CSC equation arrays differ.")
    relative, backward, rhs_norm, matrix_norm = core.check_original_equations(operator, rhs, vectors)
    del operator
    force = core.body_action(body["K"], vectors)
    force -= core.OMEGA**2 * core.body_action(body["M"], vectors)
    with np.load(SECTION, allow_pickle=False) as section, np.load(modes_path, allow_pickle=False) as modes_archive, \
            np.load(public / field_name, allow_pickle=False) as fields:
        modes = dict(modes_archive)
        require(modes["frequency_hz"].item() == FREQUENCY, "The raw SAFE modes have another frequency.")
        core.replay_fields(fields, mesh, section, modes, vectors)
        blocks, totals, errors, rhs_error, boundary_error, coherent = core.replay_ports(
            modes, fields, section["left_dofs"], section["right_dofs"], vectors, force, rhs, rhs_norm)
    compare_public_tables(public, blocks, totals)
    r, t, direct_r, direct_t, net = totals
    same_r, same_t = (np.diag(blocks[key][1]) for key in ("reflection", "transmission"))
    representative = int(np.argmax(r))
    return {"author": "Athul Prem", "file_tag": "_AP_pnc", "status": "point_verified_within_saved_discretization",
            "frequency_hz": FREQUENCY, "problem_dofs": DOFS, "port_dofs": PORT_DOFS, "reference_planes_m": PLANES,
            "public_summary_sha256": exported["public_summary"]["sha256"], "public_summary": artifact(summary_path),
            "phase_receipts": {name: artifact(path) for name, path in (("stage", stage_path), ("solve", solve_path), ("export", export_path))},
            "implementation": {"verifier": artifact(Path(__file__)), "shared_replay": artifact(Path(shared.__file__))},
            "provenance": {"original_section_implementation": stage["original_section_implementation"],
                           "section_mode_implementation": stage["section_mode_implementation"],
                           "linux_execution_implementation": stage["implementation"]},
            "original_equations_passed": True, "physical_nodal_force_power_passed": True,
            "all_modal_coefficients_passed": True, "full_p2_field_identity_passed": True, "coherent_left_incidence_passed": True,
            "propagating_coefficients_checked": 2 * len(INCIDENT)**2,
            "full_directional_coefficients_checked": 2 * PORT_DOFS * len(INCIDENT),
            "matrix_infinity_norm": matrix_norm, "maximum_relative_residual": float(max(relative)),
            "maximum_backward_residual": float(max(backward)), "relative_target": core.TARGET,
            "relative_residual_limit": core.RESIDUAL, "backward_residual_limit": core.RESIDUAL, "power_limit": core.POWER,
            "factor_nonzeros_reported": report["factor"]["factor_nonzeros"],
            "maximum_power_errors": dict(zip(core.POWER_CHECKS, errors.tolist(), strict=True)), "coherent_left_incidence": coherent,
            "channels": [{"column": column, "incident_root": root, "reverse_root": RETURNING[column],
                "relative_residual": relative[column], "backward_residual": backward[column],
                "independent_rhs_relative_error": rhs_error[column], "independent_body_boundary_relative_residual": boundary_error[column],
                "reflection": r[column], "transmission": t[column], "reflection_flux": direct_r[column],
                "transmission_flux": direct_t[column], "net_outward_power_ratio": net[column],
                "converted_reflection": r[column] - same_r[column], "converted_transmission": t[column] - same_t[column]}
                for column, root in enumerate(INCIDENT)],
            "representative_field": {"selection": "largest reflected power in this point", "column": representative,
                "incident_root": INCIDENT[representative], "path": str(public / field_name), "field_sha256": summary["output_sha256"][field_name],
                "mesh_sha256": original["mesh"]["sha256"], "accepted_vector_sha256": report["channels"][representative]["accepted_vector"]["sha256"],
                "units": "m per square root watt", "basis_order": 2, "visualization_vtu": str(public / vtu_name)},
            "platform": platform.platform(), "python": platform.python_version(), "numpy": np.__version__, "scipy": scipy.__version__,
            "thread_environment": {key: os.environ.get(key) for key in THREADS}, "available_memory_bytes_before_replay": available,
            "required_memory_bytes_before_replay": required, "elapsed_seconds": monotonic() - started,
            "spatial_convergence_established": False, "frequency_convergence_established": False,
            "full_two_port_unitarity_established": False, "reciprocity_established": False}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--relation", type=Path, help="Independently replay a saved section relation instead of a body point.")
    for phase in ("stage", "solve", "export"):
        parser.add_argument(f"--{phase}", type=Path)
    args = parser.parse_args()
    phases = (args.stage, args.solve, args.export)
    if args.relation is not None:
        if any(phases):
            parser.error("Choose the section relation or all three body-point receipts.")
        result = replay_relation(args.relation.resolve())
    else:
        if not all(phases):
            parser.error("Independent body-point replay requires stage, solve and export receipts.")
        result = replay_point(*(path.resolve() for path in phases))
    print(json.dumps(result, indent=2, allow_nan=False))


if __name__ == "__main__":
    main()
