"""Refine one sampled SAFE channel-count interval. Athul Prem | _AP_pnc.

This is a count-only finite-section experiment. It establishes no unique,
first or monotone cutoff, cross-frequency branch identity or body response.
Run each new point under an external 90-second process-tree supervisor.
"""
from __future__ import annotations

import argparse
from contextlib import contextmanager
import csv
import hashlib
import io
import json
import os
from pathlib import Path
import shutil
import sys
import threading
import time

THREADS = dict.fromkeys(("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS",
                         "BLIS_NUM_THREADS", "NUMEXPR_NUM_THREADS"), "1")
if __name__ == "__main__":
    os.environ.update(THREADS)

import numpy as np
import psutil

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT), str(ROOT / "src")]
from tools import run_lawrence_port_inventory_AP_pnc as inventory
from phononic import workflow_AP_pnc as workflow
from phononic.resumable_scattering_AP_pnc import _array_digest, _exclusive

AUTHOR = {"author": "Athul Prem", "file_tag": "_AP_pnc"}
INITIAL = (100_000_000, 1_000_000_000)
MAX_POINTS, TARGET_WIDTH_HZ = 4, 56_250_000
MIB = 2**20
CONTRACT = {"numerical_threads": 1, "sample_period_seconds": 0.05,
            "process_tree_rss_limit_bytes": 1024 * MIB, "available_reserve_bytes": 512 * MIB,
            "launch_available_bytes": 1536 * MIB, "point_wall_seconds": 90,
            "new_disk_limit_bytes": 512 * MIB, "external_process_tree_supervisor_required": True}
SCOPE = ("Sampled four/five-channel interval on the fixed archived section. No unique, first or monotone cutoff, "
         "cross-frequency branch identity, continuum completeness or device response is established.")
SAFE_DIR = ROOT / "results/runs/full_device_resource_profile/lawrence_thickness_AP_pnc/thickness-h80-safe-1335308"
INVENTORY_DIR = ROOT / "results/runs/port_frequency_inventory_AP_pnc/port-inventory-1335406"
PINNED = {
    "inventory_ledger": (ROOT / "verification/port_inventory_AP_pnc.json",
                         "d49f6a54848fc05a2bffdeda1141954a567e449ccbf2983b06c69533eae89a56"),
    "inventory_inputs": (INVENTORY_DIR / "inventory_inputs_AP_pnc.json",
                         "4fb564051f508ecbba2970dd98733b7e65a7fdb0c1f3b2f7793c32560600f0e6"),
    "lower_receipt": (INVENTORY_DIR / "point-100000000_AP_pnc.json",
                      "c7d81677fbe2d67649e943c609562a6407e24a97e6ebb4b878583e39d8806344"),
    "upper_receipt": (INVENTORY_DIR / "point-1000000000_AP_pnc.json",
                      "38fac1f8bedda024e65bbf3fab654734654793b61a049d86cf30944879b76b40"),
}
INVENTORY_SOURCE_SHA = "5bfb37a4c4320df4d8e9fd65421c3bf3595775c2dbe297916e06d196255de44c"


def require(condition, message):
    if not condition:
        raise ValueError(message)


def digest(value):
    return workflow.digest(value)


def sha(path):
    with Path(path).open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def link(path, base=None):
    path = Path(path).resolve()
    return {"path": str(path.relative_to(base.resolve())) if base else str(path),
            "sha256": sha(path), "bytes": path.stat().st_size}


def verified(path, expected):
    actual = link(path)
    require(all(actual[key] == expected[key] for key in ("bytes", "sha256")), "Artifact identity differs.")
    return Path(path)


def atomic_json(path, value):
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    temporary.replace(path)


def write_point(path, point):
    point = {key: value for key, value in point.items() if key != "point_sha256"}
    point["point_sha256"] = digest(point)
    atomic_json(path, point)
    return point


def count_decision(lower, upper, frequency, positive, negative):
    """Move one endpoint only for a symmetric accepted four/five count."""
    require(type(lower) is type(upper) is type(frequency) is int and lower < frequency < upper
            and 2 * frequency == lower + upper, "The requested frequency is not the exact dyadic midpoint.")
    if type(positive) is int and type(negative) is int and positive == negative == 4:
        return "replace_lower", (frequency, upper)
    if type(positive) is int and type(negative) is int and positive == negative == 5:
        return "replace_upper", (lower, frequency)
    return "stop_unresolved", (lower, upper)


def metrics_decision(lower, upper, frequency, metrics):
    decision, after = count_decision(lower, upper, frequency, metrics["positive_open_channels"],
                                     metrics["negative_open_channels"])
    if metrics["degeneracy_status"] != "none_detected" or metrics["coincident_propagating_root_groups"]:
        return "stop_unresolved", (lower, upper)
    return decision, after


def load_archived_inputs():
    """Reconstruct immutable arrays, independently recording the new environment.

    Passing the archived identity to read_archive verifies historical metadata;
    it does not assert that the running interpreter has that old identity.
    General live solver/cache compatibility checks remain unchanged.
    """
    current = workflow.source_identity()
    require(sha(Path(inventory.__file__)) == INVENTORY_SOURCE_SHA, "The frozen inventory helper changed.")
    records = {}
    for role, (path, expected) in PINNED.items():
        require(sha(path) == expected, f"Pinned {role} identity differs.")
        records[role] = link(path)
    receipt = json.loads((SAFE_DIR / inventory.SAFE).read_bytes())
    require(sha(SAFE_DIR / inventory.SAFE) == inventory.SAFE_SHA256, "The original SAFE receipt changed.")
    original = receipt["implementation"]
    require(current["source_sha256"] == original["source_sha256"], "Current physical numerical sources differ.")
    port, anchor, section = inventory.read_archive(SAFE_DIR, original)
    section_hash = _array_digest(port.basis.mesh.p, port.basis.mesh.t, port.basis.doflocs,
                                port.left_dofs, port.right_dofs, port.K0, port.M, port.Q, port.G)
    accepted_inputs = json.loads(Path(records["inventory_inputs"]["path"]).read_bytes())
    require(accepted_inputs["identity_sha256"] == digest(accepted_inputs["identity"])
            and accepted_inputs["identity"]["section_array_sha256"] == section_hash,
            "The accepted inventory section identity differs.")
    anchors = []
    for role, frequency, count in (("lower_receipt", INITIAL[0], 4), ("upper_receipt", INITIAL[1], 5)):
        point = json.loads(Path(records[role]["path"]).read_bytes())
        require(point["frequency_hz"] == frequency and point["status"] in ("accepted", "archived")
                and point["positive_open_channels"] == point["negative_open_channels"] == count
                and point["identity_sha256"] == accepted_inputs["identity_sha256"]
                and point["point_sha256"] == digest({k: v for k, v in point.items() if k != "point_sha256"}),
                "The accepted endpoint receipt differs.")
        modes_path = INVENTORY_DIR / point["modes"]["path"] if frequency == INITIAL[0] else SAFE_DIR / inventory.PORTS
        modes_link = link(modes_path)
        if frequency == INITIAL[0]:
            verified(modes_path, point["modes"])
        anchors.append({"frequency_hz": frequency, "count": count, "receipt": records[role], "modes": modes_link})
    del anchor
    archive = {"safe_receipt": link(SAFE_DIR / inventory.SAFE), "section_modes": link(SAFE_DIR / inventory.PORTS),
               "inventory_ledger": records["inventory_ledger"], "inventory_inputs": records["inventory_inputs"],
               "anchors": anchors, "section_array_sha256": section_hash, "trace_sha256": section["trace_sha256"],
               "original_implementation": original,
               "current_implementation": {k: v for k, v in current.items() if k != "git_commit"},
               "execution_provenance": {"git_commit": current["git_commit"]},
               "python_version_changed": original["python"] != current["python"],
               "reconstruction_policy": "immutable_archive_reconstruction_with_current_environment_recorded_separately"}
    return port, archive


@contextmanager
def monitored_point(output):
    """Observe resource boundaries; the external supervisor enforces hard exit."""
    process, finished, started = psutil.Process(), threading.Event(), time.monotonic()
    record = {"peak_sampled_process_tree_rss_bytes": 0, "minimum_available_bytes": psutil.virtual_memory().available,
              "sampling_interval_seconds": CONTRACT["sample_period_seconds"], "guard_observations": []}
    def sample():
        try:
            processes = [process, *process.children(recursive=True)]
            rss = sum(p.memory_info().rss for p in processes if p.is_running())
            available = psutil.virtual_memory().available
            record["peak_sampled_process_tree_rss_bytes"] = max(record["peak_sampled_process_tree_rss_bytes"], rss)
            record["minimum_available_bytes"] = min(record["minimum_available_bytes"], available)
            elapsed = time.monotonic() - started
            checks = ((rss > CONTRACT["process_tree_rss_limit_bytes"], "process_tree_rss"),
                      (available < CONTRACT["available_reserve_bytes"], "available_memory"),
                      (elapsed > CONTRACT["point_wall_seconds"], "wall_time"))
            for exceeded, name in checks:
                if exceeded and name not in record["guard_observations"]:
                    record["guard_observations"].append(name)
        except (psutil.Error, OSError) as error:
            record["guard_observations"].append(f"sampling_failed:{type(error).__name__}")
    def monitor():
        while not finished.wait(CONTRACT["sample_period_seconds"]):
            sample()
    sample()
    thread = threading.Thread(target=monitor, daemon=True)
    thread.start()
    try:
        yield record
    finally:
        finished.set()
        thread.join()
        sample()
        record["elapsed_seconds"] = time.monotonic() - started


def admission(output):
    output = Path(output)
    existing_parent = output
    while not existing_parent.exists():
        existing_parent = existing_parent.parent
    available, disk = psutil.virtual_memory().available, shutil.disk_usage(existing_parent).free
    used = sum(p.stat().st_size for p in output.iterdir() if p.is_file()) if output.is_dir() else 0
    require(available >= CONTRACT["launch_available_bytes"], "Less than 1.5 GiB available at point admission.")
    require(used + 64 * MIB <= CONTRACT["new_disk_limit_bytes"], "The next 64 MiB point would exceed the output budget.")
    require(disk >= 64 * MIB, "Insufficient free disk for the next 64 MiB point.")
    return {"available_bytes": available, "free_disk_bytes": disk, "existing_output_bytes": used,
            "required_new_disk_bytes": 64 * MIB}


def publish(output, identity_sha, points, *, verify_only=False):
    lower, upper = INITIAL
    for point in points:
        if point["status"] == "accepted":
            decision, after = metrics_decision(lower, upper, point["frequency_hz"], point["metrics"])
            require(point["decision"] == decision and point["interval_before_hz"] == [lower, upper]
                    and point["interval_after_hz"] == list(after), "Accepted interval decision differs.")
            lower, upper = after
    terminal = points and (points[-1]["status"] != "accepted" or points[-1]["decision"] == "stop_unresolved")
    complete = upper - lower <= TARGET_WIDTH_HZ and len(points) == MAX_POINTS and not terminal
    status = "complete" if complete else "unresolved" if terminal or len(points) >= MAX_POINTS else "pending"
    reason = ("target_sampled_width_reached" if complete else points[-1].get("stop_reason", "unexpected_count")
              if terminal else "four_point_budget_exhausted" if len(points) >= MAX_POINTS else "point_budget_pause")
    state = {**AUTHOR, "identity_sha256": identity_sha, "status": status, "stop_reason": reason,
             "lower_hz": lower, "upper_hz": upper, "width_hz": upper - lower,
             "target_width_hz": TARGET_WIDTH_HZ, "new_points_used": len(points),
             "points": [link(output / f"point-{p['frequency_hz']}_AP_pnc.json", output) for p in points],
             "cross_frequency_branch_identity": "unresolved", "scope": SCOPE}
    columns = ("frequency_hz", "status", "root_id", "direction", "k_real_rad_m", "k_imag_rad_m",
               "signed_power_w", "eigen_backward_residual", "receipt", "point_sha256", "error")
    stream = io.StringIO(newline="")
    writer = csv.DictWriter(stream, fieldnames=columns, extrasaction="ignore")
    writer.writeheader()
    for point in points:
        rows = point.get("metrics", {}).get("open_channels", []) or [{}]
        for row in rows:
            writer.writerow({**row, "frequency_hz": point["frequency_hz"], "status": point["status"],
                             "receipt": f"point-{point['frequency_hz']}_AP_pnc.json",
                             "point_sha256": point["point_sha256"], "error": point.get("error", "")})
    csv_bytes = stream.getvalue().encode("utf-8")
    state_path, csv_path = output / "interval_state_AP_pnc.json", output / "port_count_interval_AP_pnc.csv"
    if verify_only:
        require(json.loads(state_path.read_bytes()) == state and csv_path.read_bytes() == csv_bytes,
                "Published interval state or CSV differs from the immutable point receipts.")
    else:
        atomic_json(state_path, state)
        csv_path.write_bytes(csv_bytes)
    return state


def run_interval(output, port, archive, *, resume=False, max_new_points=1, solve=None, validate=None,
                 admit=admission, monitor=monitored_point):
    require(type(max_new_points) is int and 0 <= max_new_points <= MAX_POINTS, "Invocation point budget must be 0..4.")
    output = Path(output).resolve()
    provenance = archive.get("execution_provenance", {})
    archive = {k: v for k, v in archive.items() if k != "execution_provenance"}
    identity = {"schema_version": 1, "initial_interval_hz": list(INITIAL), "endpoint_counts": [4, 5],
                "target_width_hz": TARGET_WIDTH_HZ, "max_new_points": MAX_POINTS, "scope": SCOPE,
                "archive": archive, "producer": link(Path(__file__)), "resource_contract": CONTRACT}
    checksum = digest(identity)
    inputs = {"identity": identity, "identity_sha256": checksum, "initial_execution_provenance": provenance}
    if not resume:
        output.mkdir(parents=True)
    solve, validate = solve or inventory.solve_port_modes, validate or inventory.validate_modes
    with _exclusive(output / ".port_count_interval.lock"):
        manifest = output / "inputs_AP_pnc.json"
        if resume:
            saved = json.loads(manifest.read_bytes())
            require(saved["identity"] == identity and saved["identity_sha256"] == checksum,
                    "The saved interval inputs or implementation changed.")
        else:
            require(not any(p.name != ".port_count_interval.lock" for p in output.iterdir()), "Fresh output must be empty.")
            atomic_json(manifest, inputs)
        points, lower, upper = [], *INITIAL
        for index in range(1, MAX_POINTS + 1):
            midpoint = (lower + upper) // 2
            receipt_path = output / f"point-{midpoint}_AP_pnc.json"
            if not receipt_path.exists():
                break
            point = json.loads(receipt_path.read_bytes())
            require(point.get("point_sha256") == digest({k: v for k, v in point.items() if k != "point_sha256"})
                    and point["identity_sha256"] == checksum and point["index"] == index
                    and point["frequency_hz"] == midpoint and point["interval_before_hz"] == [lower, upper],
                    "Saved point receipt identity or midpoint differs.")
            if point["status"] == "accepted":
                require(point["modes"]["path"] == f"modes-{midpoint}_AP_pnc.npz", "Unexpected mode artifact path.")
                verified(output / point["modes"]["path"], point["modes"])
                decision, after = metrics_decision(lower, upper, midpoint, point["metrics"])
                require(point["decision"] == decision and point["interval_after_hz"] == list(after),
                        "Saved count decision changed.")
                lower, upper = after
            else:
                require(point["status"] in ("running", "failed", "interrupted"), "Invalid saved point state.")
            points.append(point)
            if point["status"] != "accepted" or point["decision"] == "stop_unresolved":
                break
        if points and points[-1]["status"] == "running":
            points[-1].update(status="interrupted", decision="stop_unresolved", stop_reason="previous_worker_interrupted")
            points[-1] = write_point(output / f"point-{points[-1]['frequency_hz']}_AP_pnc.json", points[-1])
            return publish(output, checksum, points)
        if resume and (output / "interval_state_AP_pnc.json").exists():
            publish(output, checksum, points, verify_only=True)
        if points and (points[-1]["status"] != "accepted" or points[-1]["decision"] == "stop_unresolved"):
            return publish(output, checksum, points, verify_only=True)
        if len(points) == MAX_POINTS or max_new_points == 0:
            state_path = output / "interval_state_AP_pnc.json"
            return publish(output, checksum, points, verify_only=True) if state_path.exists() else publish(output, checksum, points)
        for _ in range(min(max_new_points, MAX_POINTS - len(points))):
            midpoint = (lower + upper) // 2
            receipt_path = output / f"point-{midpoint}_AP_pnc.json"
            mode_path = output / f"modes-{midpoint}_AP_pnc.npz"
            temporary = mode_path.with_suffix(".npz.tmp")
            require(not mode_path.exists() and not temporary.exists(), "Unclaimed point artifact exists.")
            point = {"index": len(points) + 1, "frequency_hz": midpoint, "status": "running",
                     "identity_sha256": checksum, "interval_before_hz": [lower, upper],
                     "interval_after_hz": [lower, upper], "decision": "stop_unresolved", "admission": None,
                     "eigensolve_performed": False,
                     "execution_provenance": provenance,
                     "actual_environment": "inputs_AP_pnc.json:identity.archive.current_implementation"}
            point = write_point(receipt_path, point)
            points.append(point)
            publish(output, checksum, points)
            resources = {}
            try:
                point["admission"] = admit(output)
                point["eigensolve_performed"] = True
                point = write_point(receipt_path, point)
                with monitor(output) as resources:
                    modes = solve(port, midpoint)
                    require(modes.port is port and modes.frequency_hz == midpoint, "Returned section or frequency differs.")
                    metrics = validate(modes)
                    decision, after = metrics_decision(lower, upper, midpoint, metrics)
                    degenerate = (metrics["degeneracy_status"] != "none_detected"
                                  or bool(metrics["coincident_propagating_root_groups"]))
                    if degenerate:
                        decision, after = "stop_unresolved", (lower, upper)
                    with temporary.open("xb") as stream:
                        np.savez_compressed(stream, **{name: getattr(modes, name) for name in inventory.MODE_FIELDS})
                    del modes
                    temporary.replace(mode_path)
                require(not resources.get("guard_observations"), "A sampled resource boundary was exceeded.")
                require(sum(p.stat().st_size for p in output.iterdir() if p.is_file()) <= CONTRACT["new_disk_limit_bytes"],
                        "The bounded output exceeded 512 MiB.")
                point.update(status="accepted", decision=decision, interval_after_hz=list(after), metrics=metrics,
                             modes=link(mode_path, output), resources=resources)
                if decision == "stop_unresolved":
                    point["stop_reason"] = "root_degeneracy" if degenerate else "unexpected_directional_count"
                lower, upper = after
            except (Exception, KeyboardInterrupt) as error:
                point.update(status="interrupted" if isinstance(error, KeyboardInterrupt) else "failed",
                             decision="stop_unresolved", stop_reason="point_failed_or_interrupted",
                             error=f"{type(error).__name__}: {error}", resources=resources)
                if mode_path.exists():
                    point["retained_unaccepted_modes"] = link(mode_path, output)
                if temporary.exists():
                    point["retained_partial_modes"] = link(temporary, output)
            point = write_point(receipt_path, point)
            points[-1] = point
            state = publish(output, checksum, points)
            if point["status"] != "accepted" or point["decision"] == "stop_unresolved":
                return state
        return state


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--max-new-points", type=int, default=1)
    args = parser.parse_args()
    require(type(args.max_new_points) is int and 0 <= args.max_new_points <= 1,
            "The supervised CLI attempts at most one new point per invocation.")
    require(all(os.environ.get(key) == "1" for key in THREADS), "One numerical thread is required.")
    require(sys.platform == "win32", "This scoped reconstruction is reviewed for the local Windows environment.")
    admission(args.output)
    port, archive = load_archived_inputs()
    output = args.output.resolve()
    require(not output.is_relative_to(SAFE_DIR) and not output.is_relative_to(INVENTORY_DIR),
            "Output must be outside the accepted archives.")
    state = run_interval(output, port, archive, resume=args.resume, max_new_points=args.max_new_points)
    print(json.dumps(state, indent=2))
    return 2 if state["status"] == "unresolved" else 0


if __name__ == "__main__":
    sys.exit(main())
