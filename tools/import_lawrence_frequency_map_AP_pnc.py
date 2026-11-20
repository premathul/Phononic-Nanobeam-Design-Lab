#!/usr/bin/env python3
"""Publish the archived Lawrence h80 z6 anchor as a read-only map. Author: Athul Prem | _AP_pnc."""

from __future__ import annotations

import argparse
from collections.abc import Sequence
import csv
import math
import os
from pathlib import Path, PurePosixPath
import sys
from tempfile import TemporaryDirectory
from urllib.parse import quote

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from tools import map_scattering_AP_pnc as native
from tools.compare_scattering_AP_pnc import require as _require
from phononic import workflow_AP_pnc as workflow
from phononic.report_style_AP_pnc import available_link, report_link, report_page, scroll_table
from phononic.resumable_scattering_AP_pnc import AUTHOR, THREADS, _exclusive, _read, _sha, _valid_residual


ARCHIVE = ROOT / "results/runs/full_device_resource_profile/lawrence_thickness_AP_pnc"
THICKNESS_EVIDENCE = ROOT / "verification/device_thickness_AP_pnc.json"
IN_PLANE_EVIDENCE = ROOT / "verification/device_h60_AP_pnc.json"
H100_H80_EVIDENCE = ROOT / "verification/lawrence_hpc_AP_pnc.json"
LINUX_RUNS = PurePosixPath("/home/usd.local/kunming.dong/Phononic_Nanobeam_Design_Lab_AP_pnc/runs")
ANCHOR_GHZ = 1.0
EXECUTION_MODE = "archived_lawrence_read_only"
PHASES = {
    "preflight": ("mesh_preflight", "thickness-h80-preflight-1335307", "thickness_h80_preflight_AP_pnc.json", "mesh_ready"),
    "safe": ("safe_section", "thickness-h80-safe-1335308", "thickness_h80_safe_AP_pnc.json", "safe_ready"),
    "stage": ("operator_stage", "thickness-h80-stage-1335309", "stage_AP_pnc.json", "operator_staged"),
    "solve": ("direct_solve", "thickness-h80-solve-1335321", "solve_AP_pnc.json", "drives_accepted"),
    "export": ("scattering_export", "thickness-h80-export-1335341", "export_AP_pnc.json", "completed_unverified"),
}
PUBLIC_FILES = (*native.POINT_FILES, "hole_pattern_AP_pnc.json")
EXECUTION_KEYS = ("slurm_job_id", "hostname", "implementation", "driver", "numerical_thread_environment")
SCOPE = ("Exploratory lossless 3D scattering at the archived 1 GHz point and original end planes. "
         "Equations and flux checks qualify the archived calculation. Spatial and frequency convergence remain unestablished.")


def _verify_file(path: Path, root: Path, sha256: str, size: int | None = None) -> int:
    resolved = path.resolve()
    if not resolved.is_relative_to(root) or not resolved.is_file():
        raise ValueError(f"Artifact is missing or outside its verified directory: {path}")
    actual_size = resolved.stat().st_size
    if (size is not None and (type(size) is not int or size != actual_size)) or _sha(resolved) != sha256:
        raise ValueError(f"Artifact size or SHA-256 changed: {path}")
    return actual_size


def _verify_archived_link(root: Path, relative: str, sha256: str, original: dict | None = None) -> dict:
    """Rebase only the intended POSIX artifact into the supplied archive directory."""
    portable = PurePosixPath(relative)
    if (portable.is_absolute() or not portable.parts or portable.as_posix() != relative
            or ".." in portable.parts or "\\" in relative or ":" in relative):
        raise ValueError(f"Invalid relative archive artifact path: {relative}")
    remote = (LINUX_RUNS / portable).as_posix()
    if original is not None and (original["path"] != remote or original["sha256"] != sha256
                                 or type(original["bytes"]) is not int or original["bytes"] <= 0):
        raise ValueError(f"Saved Linux artifact identity differs for {relative}.")
    size = _verify_file(root / relative, root, sha256, original["bytes"] if original is not None else None)
    return {"path": relative, "original_linux_path": remote, "sha256": sha256, "bytes": size}


def _require_link(original: dict, verified: dict) -> None:
    if original != {"path": verified["original_linux_path"], "sha256": verified["sha256"], "bytes": verified["bytes"]}:
        raise ValueError(f"Receipt chain differs for {verified['path']}.")


def _phase_receipts(root: Path, evidence: dict) -> tuple[dict, dict]:
    receipts, links = {}, {}
    for phase, (evidence_key, folder, filename, status) in PHASES.items():
        entry = evidence[evidence_key]
        relative = f"{folder}/{filename}"
        links[phase] = _verify_archived_link(root, relative, entry["receipt_sha256"])
        receipt = _read(root / relative)
        _require(receipt["mode"] == phase and receipt["status"] == status
                 and str(receipt["slurm_job_id"]) == entry["slurm_job_id"]
                 and entry["slurm_state"] == "COMPLETED" and entry["slurm_exit_code"] == "0:0",
                 f"Archived Lawrence {phase} receipt is inconsistent with the completed job.")
        receipts[phase] = receipt
    return receipts, links


def _acceptance(solve: dict, stage: dict, evidence: dict, roots: list[int]) -> tuple[list[dict], dict]:
    original = evidence["direct_solve"]["independent_original_equation_replay"]
    flux = evidence["scattering_export"]["independent_nodal_force_flux_replay"]
    export = evidence["scattering_export"]
    solver = solve["linear_solver"]
    channels = solver["channels"]
    _require(solver["operator_csc_sha256"] == stage["operator_csc_sha256"]
             == evidence["operator_stage"]["operator_csc_sha256"]
             and solver["rhs_array_sha256"] == stage["rhs_array_sha256"] == evidence["operator_stage"]["rhs_array_sha256"]
             and solver["relative_target"] == original["relative_l2_target"] == 1e-10
             and solver["relative_residual_limit"] == original["relative_l2_residual_gate"] == native.RESIDUAL_LIMIT
             and solver["backward_residual_limit"] == original["normwise_backward_residual_gate"] == native.RESIDUAL_LIMIT
             and [row["column"] for row in channels] == list(range(len(roots)))
             and [row["incident_port_root"] for row in channels] == roots
             and [row["incident_port_root"] for row in original["channels"]] == roots
             and original["accepted_drive_count"] == len(roots)
             and original["execution_platform"] == flux["execution_platform"] == "Windows"
             and all(export[key] is True for key in ("staged_operator_and_rhs_reconstructed_exactly",
                                                     "public_fields_exactly_match_accepted_vectors",
                                                     "public_safe_wavenumber_displacement_and_force_arrays_match_saved_z6_section"))
             and export["phase_adjustment_applied"] is False
             and 0 <= flux["maximum_flux_imbalance"] < 0.01,
             "Archived original-equation, export or independent-flux evidence differs.")
    accepted = []
    for saved, replay in zip(channels, original["channels"], strict=True):
        _require(type(saved["info"]) is int and saved["info"] == 0
                 and saved["target_met"] is True and saved["relative_residual"] <= solver["relative_target"]
                 and all(_valid_residual(row[key]) for row, key in (
                     (saved, "relative_residual"), (saved, "relative_backward_residual"),
                     (replay, "relative_l2_residual"), (replay, "normwise_backward_residual"))),
                 "Archived drive fails an original-equation residual gate.")
        accepted.append({"channel": saved["column"] + 1, "port_index": saved["incident_port_root"],
                         "relative_l2_residual": replay["relative_l2_residual"],
                         "normwise_backward_residual": replay["normwise_backward_residual"]})
    return accepted, {"relative_l2_residual_limit": native.RESIDUAL_LIMIT,
                      "normwise_backward_residual_limit": native.RESIDUAL_LIMIT,
                      "independent_original_equation_replay": original,
                      "independent_nodal_force_flux_replay": flux,
                      "power_balance_passed": True}


def _mesh_pair_evidence(thickness: dict) -> dict:
    in_plane_link = thickness["previous_results"]["h80_h60_device_record"]
    sha256 = _sha(IN_PLANE_EVIDENCE)
    _require(sha256 == in_plane_link["sha256"], "Archived in-plane mesh-pair evidence changed.")
    h60_record = _read(IN_PLANE_EVIDENCE)
    in_plane = h60_record["h80_h60_mesh_pair"]
    thickness_pair = thickness["h80_z4_z6_mesh_pair"]
    earlier = _read(H100_H80_EVIDENCE)["scattering_postprocessing"]["h100_h80_comparison"]
    previous = h60_record["reference"]
    identity = thickness["saved_in_plane_channel_phase_trends"]["sources"]["device_h100_to_h80"]
    _require(earlier["summary_sha256"] == identity["sha256"]
             and earlier["maximum_power_change"] == previous["h100_h80_device_maximum_power_change"]
             and earlier["maximum_complex_amplitude_change"] == previous["h100_h80_device_maximum_complex_change"]
             and earlier["mesh_pair_passed"] is previous["h100_h80_device_mesh_pair_passed"]
             and earlier["propagation_phase_removed"] is False,
             "Archived h100/h80 mesh comparison differs across verification records.")
    earlier_pair = {**earlier, "power_passed": earlier["power_gate_passed"],
                    "complex_passed": earlier["complex_gate_passed"],
                    "maximum_complex_change": earlier["maximum_complex_amplitude_change"],
                    "source_evidence": {"path": "verification/lawrence_hpc_AP_pnc.json", "sha256": _sha(H100_H80_EVIDENCE)}}
    _require(all(pair["input_power_checks_passed"] is True and pair["power_passed"] is True
                 and pair["complex_passed"] is False and pair["mesh_pair_passed"] is False
                 and 0 <= pair["maximum_power_change"] <= pair["power_change_limit"] == 0.01
                 and pair["maximum_complex_change"] > pair["complex_change_limit"] == 0.01
                 for pair in (earlier_pair, in_plane, thickness_pair)),
             "Archived in-plane or thickness mesh-pair gates differ.")
    return {"in_plane_h100_h80_z4": earlier_pair, "in_plane_h80_h60_z4": in_plane,
            "joint_body_port_thickness_h80_z4_z6": thickness_pair}


def _verify_phase_inputs(preflight: dict, safe: dict, stage: dict, evidence: dict) -> None:
    mesh_evidence, safe_evidence = evidence["mesh_preflight"], evidence["safe_section"]
    roots, backward_roots = safe_evidence["positive_propagating_roots"], safe_evidence["negative_propagating_roots"]
    planes = evidence["selected_control"]["fixed_physical_end_planes_m"]
    _require(len(roots) == len(backward_roots) == 5 and len(set(roots + backward_roots)) == 10
             and safe["incident_port_roots"] == stage["incident_port_roots"] == roots
             and safe["backward_port_roots"] == backward_roots
             and safe["frequency_hz"] == stage["frequency_hz"] == safe_evidence["frequency_hz"] == ANCHOR_GHZ * 1e9
             and safe["trace_sha256"] == stage["trace_sha256"] == safe_evidence["trace_sha256"]
             and preflight["mesh_hash"] == stage["mesh_hash"] == mesh_evidence["mesh_hash"]
             and all(preflight["configuration"][name] == safe["configuration"][name] == stage["configuration"][name]
                     for name in ("geometry", "mesh", "material", "analysis"))
             and stage["configuration"]["analysis"]["incident_side"] == "left"
             and planes == [preflight["geometry_verification"]["port_end_planes"][end]["x_m"] for end in ("xleft", "xright")],
             "Archived mesh, SAFE section, signed roots, configuration or reference planes differ.")


def _verify_public_execution(provenance: dict, records: dict) -> None:
    solve = provenance["linux_assembly_and_solve"]
    for phase, execution in (("stage", solve["assembly"]), ("safe", provenance["linux_safe"]),
                             ("solve", solve), ("export", provenance["linux_export"])):
        _require(all(execution[key] == records[phase][key] for key in EXECUTION_KEYS),
                 f"Public scattering provenance disagrees with the original {phase} receipt.")


def _verify_archive(root: Path) -> tuple[dict, dict, Path, dict]:
    evidence = _read(THICKNESS_EVIDENCE)
    _require(evidence["execution_and_provenance_review_passed"] is True,
             "The archived thickness evidence lacks the completed provenance review.")
    records, links = _phase_receipts(root, evidence)
    preflight, safe, stage, solve, export = (records[name] for name in PHASES)
    for record, field, phase in ((safe, "preflight_receipt", "preflight"), (stage, "preflight_receipt", "preflight"),
                                 (stage, "safe_receipt", "safe"), (solve, "stage", "stage"),
                                 (export, "stage", "stage"), (export, "solve", "solve")):
        _require_link(record[field], links[phase])
    mesh_evidence = evidence["mesh_preflight"]
    safe_evidence = evidence["safe_section"]
    stage_evidence = evidence["operator_stage"]
    export_evidence = evidence["scattering_export"]
    _verify_phase_inputs(preflight, safe, stage, evidence)
    mesh_hash = mesh_evidence["mesh_hash"]
    mesh_dir = f"{PHASES['preflight'][1]}/_meshes/{mesh_hash}"
    shared_artifacts = {
        "mesh_summary": _verify_archived_link(root, f"{mesh_dir}/{native.RUN_SUMMARY}",
                                              mesh_evidence["mesh_summary_sha256"], preflight["mesh_summary"]),
        "mesh": _verify_archived_link(root, f"{mesh_dir}/mesh_AP_pnc.npz", mesh_evidence["mesh_file_sha256"], preflight["mesh"]),
        "port_section": _verify_archived_link(root, f"{PHASES['safe'][1]}/ports_AP_pnc.npz",
                                              safe_evidence["ports_file_sha256"], safe["ports"]),
    }
    _require_link(safe["mesh"], shared_artifacts["mesh"])
    _require_link(stage["mesh"], shared_artifacts["mesh"])
    _require_link(stage["mesh_summary"], shared_artifacts["mesh_summary"])
    _require_link(stage["safe_modes"], shared_artifacts["port_section"])
    arrays = {}
    for role, name, filename in (("stiffness", "K", "K_AP_pnc.npz"), ("mass", "M", "M_AP_pnc.npz"),
                                 ("operator", "A", "A_AP_pnc.npz"), ("rhs", "rhs", "rhs_AP_pnc.npy")):
        arrays[role] = _verify_archived_link(root, f"{PHASES['stage'][1]}/{filename}",
                                            stage_evidence[f"{role}_file_sha256"], stage["array_files"][name])
    shared_artifacts.update(stiffness=arrays["stiffness"], mass=arrays["mass"])
    roots = safe_evidence["positive_propagating_roots"]
    backward_roots = safe_evidence["negative_propagating_roots"]
    config = stage["configuration"]
    planes = evidence["selected_control"]["fixed_physical_end_planes_m"]
    numerical_channels, numerical_checks = _acceptance(solve, stage, evidence, roots)
    vectors = []
    for index, (saved, replay) in enumerate(zip(solve["linear_solver"]["channels"],
                                                numerical_checks["independent_original_equation_replay"]["channels"], strict=True)):
        entry = saved["accepted_vector"]
        filename = f"accepted-{index:03d}_AP_pnc.npy"
        _require(entry["file"] == filename and entry == saved["vector"], "Unexpected archived accepted vector identity.")
        relative = f"{PHASES['solve'][1]}/{filename}"
        vectors.append(_verify_archived_link(root, relative, replay["accepted_vector_sha256"],
                                             {"path": (LINUX_RUNS / relative).as_posix(), "sha256": entry["sha256"],
                                              "bytes": entry["bytes"]}))
    public = f"{PHASES['export'][1]}/{config['name']}-{export_evidence['public_run_suffix']}"
    summary_link = _verify_archived_link(root, f"{public}/{native.RUN_SUMMARY}",
                                         export_evidence["public_summary_sha256"], export["public_summary"])
    summary = _read(root / summary_link["path"])
    provenance = summary["checkpointed_scattering"]
    for field, link in (("preflight_receipt", links["preflight"]), ("safe_receipt", links["safe"]),
                        ("assembly_stage", links["stage"]), ("solve_receipt", links["solve"]),
                        ("safe_modes", shared_artifacts["port_section"])):
        _require_link(provenance[field], link)
    _verify_public_execution(provenance, records)
    _require(summary["status"] == "completed_unverified" and summary["configuration"] == config
             and summary["frequency_hz"] == stage["frequency_hz"] and summary["incident_side"] == "left"
             and summary["incident_mode_indices"] == roots and summary["num_incident_channels"] == len(roots)
             and summary["port_dofs"] == safe["trace_dofs"] == safe_evidence["displacement_trace_dofs"]
             and summary["discrete_port_roots"] == safe["discrete_roots"] == safe_evidence["discrete_roots"]
             and summary["ndof"] == preflight["displacement_dofs"] == stage["displacement_dofs"]
             and summary["mesh_checkpoint"]["mesh_hash"] == mesh_hash
             and summary["mesh_checkpoint"]["mesh_sha256"] == shared_artifacts["mesh"]["sha256"]
             and summary["mesh_checkpoint"]["directory"] == (LINUX_RUNS / mesh_dir).as_posix()
             and summary["linear_solver"] == solve["linear_solver"] and summary["power_balance_passed"] is True
             and provenance["phase_adjustment_applied"] is False
             and set(summary["output_sha256"]) == set(PUBLIC_FILES),
             "Archived public scattering identity or power checks differ.")
    native.comparison.checked_lawrence_direct_accuracy(summary)
    outputs = {name: _verify_archived_link(root, f"{public}/{name}", summary["output_sha256"][name])
               for name in PUBLIC_FILES}
    shared = {"artifacts": shared_artifacts, "mesh_hash": mesh_hash, "body_dofs": summary["ndof"],
              "trace_dofs": safe["trace_dofs"], "trace_sha256": safe["trace_sha256"],
              "reference_planes_x_m": planes,
              "port_section_contents": "Frequency-independent K0, M, Q, G and trace layout; also contains 1 GHz modes."}
    archive = {"shared": shared, "configuration": config, "receipt_links": links,
               "archived_execution": {phase: {key: record[key] for key in EXECUTION_KEYS}
                                      for phase, record in records.items()},
               "verification_evidence": {"thickness": {"path": "verification/device_thickness_AP_pnc.json",
                                                        "sha256": _sha(THICKNESS_EVIDENCE)},
                                         "in_plane": evidence["previous_results"]["h80_h60_device_record"]},
               "mesh_pair_checks": _mesh_pair_evidence(evidence)}
    point = {"accepted": {"artifacts": {"operator": arrays["operator"], "rhs": arrays["rhs"],
                                       "accepted_vectors": vectors, "public_summary": summary_link, "public_outputs": outputs},
                          "operator_csc_sha256": stage["operator_csc_sha256"], "rhs_array_sha256": stage["rhs_array_sha256"]},
             "numerical_checks": numerical_checks, "incoming_channels": numerical_channels, "phase_adjustment_applied": False,
             "mode_identity": {"incident_roots": roots, "reflected_roots": backward_roots, "transmitted_roots": roots,
                               "cutoff_status": "No unresolved cutoff reported by the archived finite SAFE solve.",
                               "mode_completeness_established": False}}
    return archive, point, root / public, summary


def _csv_rows(path: Path) -> list[dict[str, str]]:
    with path.open(encoding="utf-8", newline="") as stream:
        return list(csv.DictReader(stream))


def _verified_channel_rows(public: Path, point: dict) -> list[dict]:
    """Use the compact, hash-verified tables for this single unmapped modal basis."""
    identity = point["mode_identity"]
    roots = identity["incident_roots"]
    modes = {}
    for row in _csv_rows(public / "port_modes_AP_pnc.csv"):
        if row["propagating"] != "True":
            continue
        root = int(row["port_index"])
        direction = row["direction"]
        k = float(row["k_real_rad_m"])
        signed_power = float(row["signed_power_W"])
        _require(root not in modes and direction in ("+x", "-x") and math.isfinite(k)
                 and float(row["k_imag_rad_m"]) == 0
                 and math.isclose(signed_power, 1. if direction == "+x" else -1., abs_tol=1e-8, rel_tol=0),
                 "Archived propagating mode lacks its signed unit-power normalization.")
        modes[root] = {"k_rad_m": k, "signed_power_w": signed_power, "direction": direction}
    outgoing = {"reflection": identity["reflected_roots"], "transmission": identity["transmitted_roots"]}
    forward, backward = set(roots), set(outgoing["reflection"])
    _require(set(modes) == forward | backward and set(outgoing["transmission"]) == forward
             and all(modes[root]["direction"] == "+x" for root in forward)
             and all(modes[root]["direction"] == "-x" for root in backward),
             "Archived propagating mode directions disagree with the accepted SAFE roots.")
    identity["signed_modes"] = {str(root): mode for root, mode in modes.items()}
    rows, powers = [], {}
    for saved in _csv_rows(public / "scattering_channels_AP_pnc.csv"):
        kind = saved["kind"]
        incident = int(saved["incident_port_index"])
        emitted = int(saved["outgoing_port_index"])
        order = int(saved["incident_mode_order"])
        key = (kind, incident, emitted)
        _require(0 < order <= len(roots) and roots[order - 1] == incident
                 and kind in outgoing and emitted in outgoing[kind] and key not in powers,
                 "Archived scattering table has an invalid signed channel identity.")
        real, imag, power = (float(saved[name]) for name in ("amplitude_real", "amplitude_imag", "power_fraction"))
        amplitude = math.hypot(real, imag)
        power_scale = abs(modes[emitted]["signed_power_w"] / modes[incident]["signed_power_w"])
        _require(all(math.isfinite(value) for value in (real, imag, power)) and power >= 0
                 and math.isclose(power, amplitude * amplitude * power_scale,
                                  rel_tol=1e-8, abs_tol=1e-15),
                 "Archived complex coefficients disagree with the modal power table.")
        powers[key] = power
        rows.append({"frequency_GHz": ANCHOR_GHZ, "segment": 1, "kind": kind, "incident_port_index": incident,
                     "outgoing_port_index": emitted, "incident_channel": order,
                     "outgoing_channel": outgoing[kind].index(emitted) + 1, "amplitude_real": real,
                     "amplitude_imag": imag, "power_fraction": power, "segment_amplitude_real": real,
                     "segment_amplitude_imag": imag})
    _require(len(powers) == len(roots) * sum(map(len, outgoing.values())),
             "Archived complex scattering table is incomplete.")
    totals = _csv_rows(public / "scattering_AP_pnc.csv")
    _require([int(row["incident_mode_order"]) for row in totals] == list(range(1, len(roots) + 1)),
             "Archived power table lacks its accepted drive order.")
    for accepted, saved in zip(point["incoming_channels"], totals, strict=True):
        root = accepted["port_index"]
        fractions = {name: float(saved[name]) for name in
                     ("transmission", "reflection", "transmission_flux", "reflection_flux", "flux_power_balance")}
        _require(float(saved["frequency_GHz"]) == ANCHOR_GHZ
                 and all(math.isfinite(value) for value in fractions.values())
                 and all(math.isclose(fractions[kind], math.fsum(powers[kind, root, out] for out in emitted),
                                      abs_tol=1e-8, rel_tol=0) for kind, emitted in outgoing.items())
                 and all(abs(fractions[kind] - fractions[f"{kind}_flux"]) < 0.01 for kind in outgoing)
                 and abs(fractions["flux_power_balance"]) < 0.01,
                 "Archived per-drive power or Linux flux checks differ.")
        accepted.update(k_rad_m=modes[root]["k_rad_m"], **fractions)
    return rows


def _report(result: dict, point: dict, output: Path, archived_report: Path) -> str:
    point_rows = "".join(
        f"<tr><td>{sample['frequency_GHz']:.8g}</td><td>{sample['status'].title()}</td></tr>"
        for sample in result["points"]
    )
    channel_rows = "".join(
        f"<tr><td>{row['port_index']}</td><td>{row['transmission']:.6g}</td><td>{row['reflection']:.6g}</td>"
        f"<td>{row['relative_l2_residual']:.3e}</td><td>{row['normwise_backward_residual']:.3e}</td>"
        f"<td>{row['flux_power_balance']:.3e}</td></tr>" for row in point["incoming_channels"]
    )
    labels = {"in_plane_h100_h80_z4": "In-plane h100/h80, four layers",
              "in_plane_h80_h60_z4": "In-plane h80/h60, four layers",
              "joint_body_port_thickness_h80_z4_z6": "Joint body/port thickness, h80 four/six layers"}
    mesh_rows = "".join(
        f"<tr><td>{labels[key]}</td><td>{pair['maximum_power_change']:.6g} · Pass</td>"
        f"<td>{pair['maximum_complex_change']:.6g} · Fail</td>"
        f"<td>{pair['maximum_same_channel_transmission_phase_change_rad']:.6g}</td></tr>"
        for key, pair in result["mesh_pair_checks"].items()
    )
    left, right = result["shared"]["reference_planes_x_m"]
    downloads = " · ".join((
        available_link(output, archived_report, "Original 3D field report (verified workspace archive)",
                       unavailable="workspace archive file unavailable"),
        report_link("Archived 1 GHz coefficients and channel powers (CSV)", native.TABLE, download=True),
        report_link("Read-only map and provenance (JSON)", native.MAP, download=True),
    ))
    point_table = scroll_table(f"<table><thead><tr><th>Frequency (GHz)</th><th>Point</th></tr></thead>"
                               f"<tbody>{point_rows}</tbody></table>", "Archived and pending frequencies")
    channel_table = scroll_table(f"""<table><thead><tr><th>Incident SAFE root</th><th>Transmission</th><th>Reflection</th>
<th>Relative L2 residual</th><th>Backward residual</th><th>Flux imbalance</th></tr></thead><tbody>{channel_rows}</tbody></table>""",
                                 "Archived anchor per-drive checks")
    mesh_table = scroll_table(f"""<table><thead><tr><th>Control</th><th>Power change (limit 0.01)</th>
<th>Complex change (limit 0.01)</th><th>Largest transmission phase change (rad)</th></tr></thead><tbody>{mesh_rows}</tbody></table>""",
                              "Saved mesh comparison checks")
    scope = (f"26-hole h80 six-layer device. Raw complex amplitudes retain the original modal basis at "
             f"x = {left * 1e6:.6g} and {right * 1e6:.6g} µm. Spatial and frequency convergence remain unresolved.")
    body = f"""
<p>One accepted archived sample. Pending frequencies have no computed response. Channel tracking requires adjacent accepted samples.</p>
<p class="sources">{downloads}</p>
{point_table}
<h2>Accepted 1 GHz point</h2>
{channel_table}
<p>Residuals come from the saved independent Windows original-equation replay. Per-drive flux values come from
the original Linux export. The independent Windows nodal-force replay measured maximum flux imbalance
{point['numerical_checks']['independent_nodal_force_flux_replay']['maximum_flux_imbalance']:.3e}.</p>
<h2>Saved mesh comparisons</h2>{mesh_table}
"""
    return report_page("Archived read-only 1 GHz anchor import", "lawrence_1ghz_anchor", body, scope=scope,
                       navigation=available_link(output, ROOT / "tools/README.md", "Simulation and comparison tools"))


def _publish(output: Path, result: dict, point: dict, rows: list[dict], archived_report: Path) -> dict:
    output.mkdir(parents=True, exist_ok=True)
    lock = ".frequency_map.lock"
    files = (native.MAP, native.TABLE, native.REPORT)
    if (output / lock).is_symlink():
        raise ValueError("The map ownership file must be inside the output directory.")
    with _exclusive(output / lock):
        if any(path.name not in (*files, lock) or path.is_symlink() for path in output.iterdir()):
            raise ValueError("Select an empty output directory or the same archived frequency map.")
        publication = output / native.MAP
        if publication.exists():
            saved = _read(publication)
            data_sha = saved.pop("map_data_sha256")
            if workflow.digest(saved) != data_sha:
                raise ValueError("Published archive map identity changed.")
            keys = result.keys() - {"import_environment"}
            if (result.keys() | {"output_sha256"}) != saved.keys() or any(saved[key] != result[key] for key in keys):
                raise ValueError("Existing archive map differs from the verified archive or requested points.")
            for filename in (native.TABLE, native.REPORT):
                _verify_file(output / filename, output, saved["output_sha256"][filename])
            return {**saved, "map_data_sha256": data_sha}
        with TemporaryDirectory(prefix="lawrence-map-", dir=output.parent) as temporary:
            staging = Path(temporary)
            workflow._write_csv(staging / native.TABLE, rows)
            (staging / native.REPORT).write_text(_report(result, point, output, archived_report), encoding="utf-8")
            result["output_sha256"] = {filename: _sha(staging / filename) for filename in (native.TABLE, native.REPORT)}
            result["map_data_sha256"] = workflow.digest(result)
            workflow.write_json(staging / native.MAP, result)
            for filename in (native.TABLE, native.REPORT):
                if (output / filename).exists():
                    _verify_file(output / filename, output, result["output_sha256"][filename])
            for filename in (native.TABLE, native.REPORT, native.MAP):
                if not (output / filename).exists():
                    (staging / filename).replace(output / filename)
    return result


def import_lawrence_frequency_map(archive_root: Path, output: Path, *,
                                  frequencies_GHz: Sequence[float] = (ANCHOR_GHZ,)) -> dict:
    """Verify the archived anchor and publish a map without any numerical workers."""
    frequencies = list(frequencies_GHz)
    if (ANCHOR_GHZ not in frequencies
            or any(not workflow._finite_number(f) or f <= 0 or not math.isfinite(f * 1e9) for f in frequencies)
            or any(b * 1e9 <= a * 1e9 for a, b in zip(frequencies, frequencies[1:]))):
        raise ValueError("Supply strictly increasing positive frequencies in GHz, including the 1 GHz anchor.")
    root, output = Path(archive_root).resolve(strict=True), Path(output).resolve()
    if output.is_relative_to(root):
        raise ValueError("The map output directory must be outside the read-only archive.")
    archive, accepted, public, _ = _verify_archive(root)
    points = [{"index": index, "frequency_GHz": frequency, "status": "pending", "attempts": [], **native.CONVERGENCE}
              for index, frequency in enumerate(frequencies, 1)]
    point = points[frequencies.index(ANCHOR_GHZ)]
    report_path = (public / native.REPORT).resolve()
    try:
        report_link = quote(Path(os.path.relpath(report_path, output)).as_posix(), safe="/")
    except ValueError:
        report_link = report_path.as_uri()
    point.update(accepted, status="accepted", segment=1, report=report_link,
                 track_from_previous={"status": "segment_start", "reason": "No adjacent accepted point to link."})
    rows = _verified_channel_rows(public, point)
    environment = workflow.source_identity()
    environment.update(thread_environment={key: os.getenv(key) for key in THREADS},
                       import_and_presentation_source_sha256={
                           Path(path).resolve().relative_to(ROOT).as_posix(): _sha(Path(path))
                           for path in (__file__, native.__file__, native.comparison.__file__)})
    try:
        archive_path = Path(os.path.relpath(root, output)).as_posix()
    except ValueError:
        archive_path = str(root)
    result = {**AUTHOR, **native.CONVERGENCE, **archive, "schema_version": 1, "execution_mode": EXECUTION_MODE,
              "computation_enabled": False, "archive_root": archive_path, "scope": SCOPE,
              "import_environment": environment, "frequencies_GHz": frequencies, "points": points,
              "status": "complete" if len(points) == 1 else "partial"}
    return _publish(output, result, point, rows, report_path)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--archive-root", type=Path, default=ARCHIVE, help="Local root of the archived Lawrence job outputs.")
    parser.add_argument("--output", type=Path, required=True, help="Directory for the read-only map and its report.")
    parser.add_argument("--frequencies-ghz", type=float, nargs="+", default=(ANCHOR_GHZ,),
                        help="Optional increasing list including 1 GHz. Other points remain pending.")
    args = parser.parse_args(argv)
    try:
        result = import_lawrence_frequency_map(args.archive_root, args.output, frequencies_GHz=args.frequencies_ghz)
    except (OSError, RuntimeError, ValueError, KeyError, TypeError) as exc:
        print(f"Archived frequency map could not be published: {exc}", file=sys.stderr)
        return 2
    print(f"Verified one archived point out of {len(result['points'])}. Report: {(args.output / native.REPORT).resolve()}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
