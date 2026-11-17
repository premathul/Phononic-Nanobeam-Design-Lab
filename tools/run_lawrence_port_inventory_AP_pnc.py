"""Inventory the archived h80 z6 SAFE section at five frequencies. Athul Prem | _AP_pnc."""

from __future__ import annotations

import argparse
import csv
from dataclasses import fields
from html import escape
import io
import json
import math
import os
from pathlib import Path
import shutil
import socket
import sys

import numpy as np
from skfem import Basis, ElementTriP2, ElementVector, MeshTri

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT / "src"), str(ROOT)]

from phononic import workflow_AP_pnc as workflow
from phononic.port_modes_AP_pnc import PortModes, PortSystem, solve_port_modes
from phononic.resumable_scattering_AP_pnc import AUTHOR, _array_digest, _exclusive, _link, _sha, _verified
from tools import lawrence_solver_AP_pnc as lawrence


ANCHOR_HZ = 1_000_000_000
FREQUENCIES_HZ = (100_000_000, ANCHOR_HZ, 2_800_000_000, 5_200_000_000, 6_000_000_000)
SAFE = "thickness_h80_safe_AP_pnc.json"
PORTS = "ports_AP_pnc.npz"
SAFE_SHA256 = "b5c0d73656e90aa5ba312939c0ef25633af5a775974f5fc778031584d55c78f3"
PORTS_SHA256 = "a13a58e47dee91e4c7d75bb9ec77feb7656ae674392c2f1d7485129da76d0e6a"
PORT_DOFS = 975
MODE_FIELDS = tuple(field.name for field in fields(PortModes) if field.name != "port")
RESIDUAL_LIMIT = 1e-8
FLUX_LIMIT_W = 1e-7
TRACTION_LIMIT = 1e-8
CONDITION_LIMIT = 1e10
SCOPE = ("Finite frequency samples on the archived 12-by-6 SAFE section. Root IDs are local to each frequency. "
         "Branch identity between points, cutoff locations, and spatial and frequency convergence remain unresolved.")
REPORT = "port_inventory_AP_pnc"


def read_archive(directory, implementation):
    """Reconstruct the section and load the original 1 GHz modes, without a volume mesh."""
    receipt_path, ports_path = directory / SAFE, directory / PORTS
    if _sha(receipt_path) != SAFE_SHA256 or _sha(ports_path) != PORTS_SHA256:
        raise ValueError("The archived SAFE receipt or section SHA-256 differs.")
    receipt = json.loads(receipt_path.read_bytes())
    port_link = receipt["ports"]
    original = receipt["implementation"]
    if (receipt["status"] != "safe_ready" or receipt["trace_dofs"] != PORT_DOFS
            or receipt["frequency_hz"] != ANCHOR_HZ or receipt["discrete_roots"] != 2 * PORT_DOFS
            or port_link["sha256"] != PORTS_SHA256 or port_link["bytes"] != ports_path.stat().st_size):
        raise ValueError("Unexpected archived section identity or 1 GHz receipt.")
    if any(implementation[key] != original[key] for key in ("source_sha256", "packages", "python")):
        raise ValueError("The archived numerical source, packages or Python version differs.")
    with np.load(ports_path, allow_pickle=False) as arrays:
        vertices = arrays["section_vertices_yz_m"]
        triangles = arrays["section_triangles"]
        mesh = MeshTri(np.ascontiguousarray(vertices.T), np.ascontiguousarray(triangles.T))
        basis = Basis(mesh, ElementVector(ElementTriP2(), dim=3), intorder=4)
        left, right = arrays["left_dofs"], arrays["right_dofs"]
        if (basis.N != PORT_DOFS or not np.array_equal(mesh.p.T, vertices)
                or not np.array_equal(mesh.t.T, triangles)
                or not np.array_equal(basis.doflocs.T, arrays["coordinates_yz_m"])
                or not np.array_equal(basis.split_indices(), arrays["component_dofs_xyz"])
                or any(ids.shape != (PORT_DOFS,) or np.unique(ids).size != PORT_DOFS for ids in (left, right))
                or np.intersect1d(left, right).size
                or workflow.digest([left.tolist(), right.tolist()]) != receipt["trace_sha256"]):
            raise ValueError("The reconstructed section basis or archived trace maps differ.")
        operators = {key: arrays[key] for key in ("K0", "M", "Q", "G")}
        if any(matrix.shape != (PORT_DOFS, PORT_DOFS) or matrix.dtype != np.float64
               or not np.isfinite(matrix).all() for matrix in operators.values()):
            raise ValueError("Invalid archived SAFE section operators.")
        speed = float(arrays["reference_speed_m_s"])
        if not math.isfinite(speed) or speed <= 0:
            raise ValueError("Invalid archived SAFE reference speed.")
        port = PortSystem(elastic_system=None, basis=basis, left_dofs=left, right_dofs=right,
                          reference_speed_m_s=speed, **operators)
        modes = read_modes(arrays, port)
    if (modes.frequency_hz != ANCHOR_HZ
            or float(modes.relative_residuals.max()) != receipt["maximum_port_relative_residual"]
            or not 0 <= modes.relative_residuals.max() <= RESIDUAL_LIMIT
            or any(ids[modes.propagating[ids]].tolist() != receipt[key] or len(receipt[key]) != 5
                   for ids, key in ((modes.positive_indices, "incident_port_roots"),
                                    (modes.negative_indices, "backward_port_roots")))):
        raise ValueError("The archived 1 GHz mode set differs from the acceptance receipt.")
    return port, modes, {
        "ports": {**port_link, "local_path": str(ports_path.resolve())},
        "safe_receipt": {"path": str(receipt_path.resolve()), "sha256": SAFE_SHA256,
                         "bytes": receipt_path.stat().st_size, "slurm_job_id": receipt["slurm_job_id"]},
        "trace_sha256": receipt["trace_sha256"], "configuration": receipt["configuration"],
    }


def read_modes(arrays, port):
    """Load a saved full mode set while using the shared archived operators and basis."""
    values = {name: arrays[name] for name in MODE_FIELDS}
    values["frequency_hz"] = float(values["frequency_hz"])
    return PortModes(port=port, **values)


def validate_modes(modes):
    """Replay the original quadratic eigen-equation, +x force and directional flux."""
    port = modes.port
    size, omega = int(port.basis.N), math.tau * modes.frequency_hz
    k, U, F = modes.wavenumbers_rad_m, modes.displacements, modes.axial_forces
    propagating = modes.propagating
    directions = (modes.positive_indices, modes.negative_indices)
    if (not math.isfinite(omega) or omega <= 0 or U.shape != (size, 2 * size) or F.shape != U.shape
            or any(vector.shape != (2 * size,) for vector in (k, propagating, modes.powers_w, modes.relative_residuals))
            or propagating.dtype != np.bool_ or any(ids.shape != (size,) or ids.dtype.kind not in "iu" for ids in directions)
            or not np.array_equal(np.sort(np.concatenate(directions)), np.arange(2 * size))
            or any(not np.isfinite(array).all() for array in (k, U, F, modes.powers_w, modes.relative_residuals))
            or not np.all((modes.relative_residuals >= 0) & (modes.relative_residuals <= RESIDUAL_LIMIT))
            or np.any(k[propagating].imag != 0) or np.any(k[~propagating].imag == 0)):
        raise ValueError("Invalid or incomplete archived or computed SAFE mode arrays.")

    A = port.K0 - omega**2 * port.M
    H = port.G.T - port.G
    a_norm, h_norm, q_norm = (np.linalg.norm(matrix, ord=np.inf) for matrix in (A, H, port.Q))
    residuals, traction, powers = np.empty(2 * size), np.empty(2 * size), np.empty(2 * size)
    replayed_forces = np.empty(U.shape, dtype=np.complex128)
    for first in range(0, 2 * size, 64):
        batch = slice(first, first + 64)
        wave = k[batch]
        displacement = U[:, batch]
        axial = port.Q @ displacement
        coupling = port.G @ displacement
        force = coupling + (1j * wave) * axial
        replayed_forces[:, batch] = force
        imbalance = A @ displacement + (1j * wave) * (H @ displacement) + wave**2 * axial
        scale = (a_norm + abs(wave) * h_norm + abs(wave)**2 * q_norm) * np.linalg.norm(displacement, axis=0)
        with np.errstate(divide="ignore", invalid="ignore"):
            residuals[batch] = np.linalg.norm(imbalance, axis=0) / scale
        force_scale = np.linalg.norm(coupling, axis=0) + abs(wave) * np.linalg.norm(axial, axis=0)
        traction[batch] = np.linalg.norm(F[:, batch] - force, axis=0) / np.maximum(force_scale, np.finfo(float).tiny)
        powers[batch] = omega / 2 * np.sum((displacement.conj() * force).imag, axis=0)
    maximum_residual, maximum_traction = float(residuals.max()), float(traction.max())
    power_difference = float(abs(powers - modes.powers_w).max())
    if not math.isfinite(maximum_residual) or maximum_residual > RESIDUAL_LIMIT:
        raise ValueError(f"Original SAFE eigenpair backward residual exceeds 1e-8: {maximum_residual:.3e}.")
    if not math.isfinite(maximum_traction) or maximum_traction > TRACTION_LIMIT:
        raise ValueError(f"Original SAFE traction replay exceeds 1e-8: {maximum_traction:.3e}.")
    if not math.isfinite(power_difference) or power_difference > FLUX_LIMIT_W:
        raise ValueError(f"Original SAFE modal power difference exceeds 1e-7 W: {power_difference:.3e} W.")

    # The solver coalesces a numerically coincident propagating subspace at its mean k.
    open_roots = np.flatnonzero(propagating)
    _, group_ids, counts = np.unique(k[open_roots].real, return_inverse=True, return_counts=True)
    groups = [open_roots[group_ids == i].tolist() for i in np.flatnonzero(counts > 1)]
    coincident = {root for group in groups for root in group}
    channels, signed = [], {}
    for word, sign, ids in zip(("positive", "negative"), (1, -1), directions, strict=True):
        opened = propagating[ids]
        if np.any(sign * powers[ids[opened]] <= 0) or np.any(sign * k[ids[~opened]].imag <= 0):
            raise ValueError(f"Inconsistent signed power or evanescent decay in the {word} basis.")
        directional = U[:, ids]
        condition = float(np.linalg.cond(directional / np.linalg.norm(directional, axis=0)))
        work = directional.conj().T @ replayed_forces[:, ids]
        flux = omega / (4j) * (work - work.conj().T)
        flux_error = float(abs(flux - np.diag(sign * opened.astype(float))).max())
        if (not math.isfinite(condition) or condition >= CONDITION_LIMIT
                or not math.isfinite(flux_error) or flux_error > FLUX_LIMIT_W):
            raise ValueError(f"Unaccepted {word} directional basis: condition {condition:.3e}, flux error {flux_error:.3e} W.")
        signed[f"{word}_open_channels"] = int(opened.sum())
        signed[f"{word}_flux_gram_error_w"] = flux_error
        signed[f"{word}_basis_condition"] = condition
        for root in ids[opened]:
            channels.append({"root_id": int(root), "direction": word,
                             "k_real_rad_m": float(k[root].real), "k_imag_rad_m": float(k[root].imag),
                             "signed_power_w": float(powers[root]), "eigen_backward_residual": float(residuals[root]),
                             "root_degeneracy": "coincident_real_roots" if root in coincident else "none_detected"})
    if signed["positive_open_channels"] != signed["negative_open_channels"]:
        raise ValueError("The two directions have unequal propagating channel counts.")
    return {**signed, "discrete_roots": 2 * size, "open_channels": channels,
            "maximum_eigen_backward_residual": maximum_residual,
            "maximum_solver_reported_eigen_backward_residual": float(modes.relative_residuals.max()),
            "maximum_relative_traction_error": maximum_traction, "maximum_modal_power_difference_w": power_difference,
            "cutoff_status": "solver_resolved_at_sample", "coincident_propagating_root_groups": groups,
            "degeneracy_status": "coincident_propagating_roots" if groups else "none_detected"}


def write_text(path, value):
    data = value.encode("utf-8")
    if not path.exists() or path.read_bytes() != data:
        temporary = path.with_suffix(path.suffix + ".tmp")
        temporary.write_bytes(data)
        temporary.replace(path)


def write_point(path, point):
    workflow.write_json(path, {**point, "point_sha256": workflow.digest(point)})


def publish(output, identity, points):
    summary = {**AUTHOR, "schema_version": 1, "scope": SCOPE,
               "status": "complete" if all(point["status"] in ("accepted", "archived") for point in points) else "incomplete",
               "spatially_validated": False, "frequency_validated": False,
               "identity": identity, "identity_sha256": workflow.digest(identity), "points": points,
               "gates": {"eigen_backward_residual": RESIDUAL_LIMIT, "directional_flux_error_w": FLUX_LIMIT_W,
                         "relative_traction_replay_error": TRACTION_LIMIT, "directional_basis_condition": CONDITION_LIMIT}}
    write_text(output / f"{REPORT}.json", json.dumps(summary, indent=2, allow_nan=False) + "\n")
    columns = ("frequency_hz", "status", "positive_open_channels", "negative_open_channels", "root_id", "direction",
               "k_real_rad_m", "k_imag_rad_m", "signed_power_w", "eigen_backward_residual", "maximum_eigen_backward_residual",
               "positive_flux_gram_error_w", "negative_flux_gram_error_w", "cutoff_status", "degeneracy_status",
               "root_degeneracy", "error")
    buffer = io.StringIO()
    writer = csv.DictWriter(buffer, fieldnames=columns, lineterminator="\n")
    writer.writeheader()
    for point in points:
        for channel in point.get("open_channels") or [{}]:
            row = {**point, **channel}
            writer.writerow({key: row.get(key) for key in columns})
    write_text(output / f"{REPORT}.csv", buffer.getvalue())

    def table_cell(value):
        return "" if value is None else escape(str(value))

    rows = []
    for point in points:
        backward = point.get("maximum_eigen_backward_residual")
        flux = (point.get("positive_flux_gram_error_w"), point.get("negative_flux_gram_error_w"))
        detail = point.get("error") or point.get("degeneracy_status", "")
        row = (f"{point['frequency_hz'] / 1e9:g}", point["status"], point.get("positive_open_channels"),
               point.get("negative_open_channels"), f"{backward:.3e}" if backward is not None else "",
               "/".join(f"{value:.3e}" for value in flux) if None not in flux else "", detail)
        rows.append("<tr>" + "".join(f"<td>{table_cell(value)}</td>" for value in row) + "</tr>")
    write_text(output / f"{REPORT}.html", f"""<!doctype html>
<html lang="en"><meta charset="utf-8"><title>SAFE port frequency inventory</title>
<style>body{{font:16px system-ui;margin:2rem;max-width:70rem}}table{{border-collapse:collapse}}td,th{{padding:.45rem .8rem;border-bottom:1px solid #ccc;text-align:left}}</style>
<h1>SAFE port frequency inventory</h1>
<p>Athul Prem | athul_simulation_pnc_port_inventory</p>
<p>{escape(SCOPE)}</p>
<p><a href="{REPORT}.csv">Open-channel data (CSV)</a> · <a href="{REPORT}.json">Provenance and checks (JSON)</a></p>
<table><thead><tr><th>Frequency (GHz)</th><th>Status</th><th>+x channels</th><th>−x channels</th>
<th>Max eigenpair backward residual</th><th>+x / −x flux error (W)</th><th>Diagnostic</th></tr></thead>
<tbody>{''.join(rows)}</tbody></table></html>
""")
    return summary


def run_inventory(output, port, archived_modes, provenance, *, resume=False, admit=None, execution=None):
    """Commit each frequency once. An explicit resume advances only untouched frequencies."""
    output = Path(output).resolve()
    if not resume:
        output.mkdir(parents=True)
    with _exclusive(output / "port_inventory.lock"):
        if archived_modes.port is not port or archived_modes.frequency_hz != ANCHOR_HZ:
            raise ValueError("The 1 GHz anchor must use the inventory section.")
        archived = validate_modes(archived_modes)
        identity = {"frequencies_hz": list(FREQUENCIES_HZ), "trace_dofs": int(port.basis.N),
                    "section_array_sha256": _array_digest(port.basis.mesh.p, port.basis.mesh.t, port.basis.doflocs,
                                                          port.left_dofs, port.right_dofs, port.K0, port.M, port.Q, port.G),
                    "reference_speed_m_s": port.reference_speed_m_s,
                    "anchor_modes_sha256": _array_digest(*(np.asarray(getattr(archived_modes, name)) for name in MODE_FIELDS)),
                    "provenance": provenance}
        checksum = workflow.digest(identity)
        manifest = output / "inventory_inputs_AP_pnc.json"
        inputs = {"identity": identity, "identity_sha256": checksum}
        if resume:
            if json.loads(manifest.read_bytes()) != inputs:
                raise ValueError("The inventory inputs or numerical implementation differ.")
        else:
            workflow.write_json(manifest, inputs)

        points, receipts = [], {}
        for freq in FREQUENCIES_HZ:
            receipt = output / f"point-{freq}_AP_pnc.json"
            receipts[freq] = receipt
            if freq == ANCHOR_HZ:
                point = {"frequency_hz": freq, "status": "archived", "eigensolve_performed": False, **archived}
            else:
                point = {"frequency_hz": freq, "status": "pending"}
            if receipt.exists():
                saved = json.loads(receipt.read_bytes())
                if (saved.pop("point_sha256", None) != workflow.digest(saved)
                        or saved["frequency_hz"] != freq or saved["identity_sha256"] != checksum):
                    raise ValueError(f"The frequency receipt identity differs: {receipt}")
                if saved["status"] == "archived" and freq == ANCHOR_HZ:
                    point = saved
                elif saved["status"] == "accepted" and freq != ANCHOR_HZ:
                    expected = f"modes-{freq}_AP_pnc.npz"
                    if saved["modes"]["path"] != expected:
                        raise ValueError(f"Unexpected mode artifact in {receipt}")
                    _verified(saved["modes"], output)
                    point = saved
                elif saved["status"] in ("running", "failed", "interrupted") and freq != ANCHOR_HZ:
                    point = saved
                else:
                    raise ValueError(f"Invalid frequency receipt state: {receipt}")
            points.append(point)
        # Reconcile an interrupted process only after every existing acceptance passed integrity checks.
        for point in points:
            receipt = receipts[point["frequency_hz"]]
            if point["status"] == "archived" and not receipt.exists():
                point["identity_sha256"] = checksum
                write_point(receipt, point)
            if point["status"] == "running":
                point.update(status="interrupted", error="The previous worker exited before committing the point.")
                write_point(receipt, point)
        publish(output, identity, points)

        for point in points:
            if point["status"] != "pending":
                continue
            freq = point["frequency_hz"]
            receipt = receipts[freq]
            artifact = output / f"modes-{freq}_AP_pnc.npz"
            temporary = artifact.with_suffix(".npz.tmp")
            if artifact.exists() or temporary.exists():
                raise ValueError(f"An unclaimed frequency artifact already exists: {artifact}")
            admission = admit(output, sum(entry["status"] == "pending" for entry in points)) if admit else None
            point.update(status="running", identity_sha256=checksum, execution=execution, admission=admission)
            write_point(receipt, point)
            try:
                with workflow.measured_resources() as resources:
                    modes = solve_port_modes(port, freq)
                    if modes.port is not port or modes.frequency_hz != freq:
                        raise ValueError("Returned SAFE modes differ from the requested frequency or section.")
                    metrics = validate_modes(modes)
                    with temporary.open("xb") as stream:
                        np.savez_compressed(stream, **{name: getattr(modes, name) for name in MODE_FIELDS})
                    del modes
                    temporary.replace(artifact)
                point.update(status="accepted", eigensolve_performed=True, resources=resources,
                             modes=_link(artifact, output), **metrics)
            except (RuntimeError, ValueError, np.linalg.LinAlgError) as error:
                message = str(error).lower()
                cutoff = "unresolved" if "cutoff" in message or "weakly decaying" in message else "not_established"
                degeneracy = "unresolved" if "too close" in message else "not_established"
                point.update(status="failed", error=f"{type(error).__name__}: {error}", resources=resources,
                             cutoff_status=cutoff, degeneracy_status=degeneracy)
            except KeyboardInterrupt as error:
                point.update(status="interrupted", error=f"{type(error).__name__}: {error}", resources=resources)
                write_point(receipt, point)
                publish(output, identity, points)
                raise
            write_point(receipt, point)
            publish(output, identity, points)
        return publish(output, identity, points)


def resource_admission(output, pending):
    quota = int(os.environ.get("SLURM_MEM_PER_NODE", "0")) * lawrence.MIB
    cpus = int(os.environ.get("SLURM_CPUS_PER_TASK", "0"))
    if quota != 4 * lawrence.GIB or cpus != 2:
        raise ValueError("The inventory requires the reviewed 4 GiB and two-CPU allocation.")
    snapshot = lawrence.memory_snapshot(quota, lawrence.cgroup_controls())
    # The measured 1 GHz process peak was 471976 KiB. Admit 1.5 GiB of fresh
    # headroom to cover replay, archive retention and frequency-specific workspaces.
    needed = 1536 * lawrence.MIB
    disk_needed = (pending + 1) * 64 * lawrence.MIB if pending else 0
    disk_available = shutil.disk_usage(output).free
    snapshot.update(required_memory_headroom_bytes=needed, disk_available_bytes=disk_available,
                    required_disk_bytes=disk_needed, slurm_memory_bytes=quota, slurm_cpus=cpus,
                    prior_safe_peak_rss_kib=471976, prior_safe_elapsed_seconds=15.29)
    if min(snapshot["node_available_bytes"], snapshot["job_headroom_bytes"]) < needed or disk_available < disk_needed:
        raise RuntimeError(f"SAFE inventory resource admission failed: {json.dumps(snapshot)}")
    return snapshot


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--safe-dir", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args()
    threads = {name: os.environ.get(name) for name in lawrence.THREADS}
    job = os.environ.get("SLURM_JOB_ID")
    if sys.platform != "linux" or not job or threads != lawrence.THREADS:
        raise ValueError("Use the reviewed single-threaded Lawrence Slurm batch script.")
    if args.output.resolve().is_relative_to(args.safe_dir.resolve()):
        raise ValueError("The inventory output must be outside the archived section directory.")
    resource_admission(args.output if args.resume else args.output.parent, 0)
    implementation = workflow.source_identity()
    backend = lawrence.solve_environment()
    port, anchor, archive = read_archive(args.safe_dir, implementation)
    provenance = {"archive": archive, "implementation": implementation, "driver_sha256": _sha(Path(__file__)),
                  "backend": {key: backend[key] for key in ("build_configs", "machine", "runner_sha256")},
                  "numerical_thread_environment": threads}
    execution = {"hostname": socket.gethostname(), "slurm_job_id": job, "cpu_model": backend["cpu_model"],
                 "available_cpus": backend["available_cpus"]}
    summary = run_inventory(args.output, port, anchor, provenance, resume=args.resume,
                            admit=resource_admission, execution=execution)
    print(f"{summary['status']}: {args.output.resolve() / (REPORT + '.json')}", flush=True)
    return 0 if summary["status"] == "complete" else 2


if __name__ == "__main__":
    sys.exit(main())
