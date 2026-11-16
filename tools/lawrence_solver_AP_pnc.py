"""Bounded CPU solve of the saved h80 incident drives. Athul Prem | _AP_pnc."""

import argparse
import hashlib
import io
import json
import math
import os
from pathlib import Path
import platform
import signal
import subprocess
import sys
import tempfile
import time
from contextlib import redirect_stdout


MIB = 2**20
GIB = 2**30
DOFS, NNZ = 426627, 32706643
INCIDENT_ROOTS = (1348, 1311, 1292, 1284, 1274)
RELATIVE_TARGET, RESIDUAL_LIMIT = 1e-10, 1e-9
WALL_LIMIT = 1200
RSS_LIMIT = 112 * GIB
JOB_RESERVE, NODE_RESERVE = 8 * GIB, 400 * MIB
INITIAL_FACTOR_BYTES = 30 * 40 * NNZ  # SciPy 1.18.1 SuperLU source estimate, not a peak bound.
ADMISSION_BYTES = INITIAL_FACTOR_BYTES + 2 * 655839372 + 512 * MIB
THREADS = dict.fromkeys(("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS",
                         "BLIS_NUM_THREADS", "NUMEXPR_NUM_THREADS"), "1")
PROGRESS = "progress_AP_pnc.json"
EXECUTION = "execution_AP_pnc.json"
SEED_DIR = "h80_first_drive_correction_double_fill4_r4m1rklb"
ARTIFACTS = {
    "A_AP_pnc.npz": (655840629, "2d1a0d342a293cd7f45874c396887d084e6ff3a42c5ba669a8f98c3c2554a56d"),
    "rhs_AP_pnc.npy": (34130288, "80800b866d8e4358f7a4dab8478e2bd21ab9cf9083619fa1bf6f37854dab96a1"),
    "stages_AP_pnc.json": (14841, "1c50971e27bfc52e63000c9d25df100e21bbe28b7fa47d7549f82cf465531a78"),
    "configuration_AP_pnc.json": (1416, "332190b8f4e70db8ff28c51ebe97317a5e94b2a11d70bd65e33d02e60ced2c76"),
    f"{SEED_DIR}/terminal-000.npy": (6826160, "6b6acc57fc3f6c208b55f2b353b38b8fa5da2ed085cd233f0ee6fd41b077dd30"),
    f"{SEED_DIR}/channels_AP_pnc.json": (15315, "9db270c34633ff479e574eba149bf0dd4c9d61eb4c0d27a835ea554fec8f0ff1"),
}
ARRAY_HASHES = {
    "operator_csc_sha256": "6c1ef8514f8d55ac0ac9ebd1011e4411117cda7b37211d41179b2ac5e3259d75",
    "rhs_array_sha256": "0e812751c2da8e1e531fba3c187967876f7d33e3ef251cdae9d3b9a4be6094f1",
    "rhs_column_sha256": "afe743a820e078694653d40fd83204bd3086f04388ca8cd0c66daaad97011355",
}
IDENTITY = {"author": "Athul Prem", "file_tag": "_AP_pnc", "schema_version": 1}


def file_sha(path):
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def write_record(path, record):
    temporary = path.with_suffix(".tmp")
    temporary.write_text(json.dumps(record, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    temporary.replace(path)


def write_vector(path, vector):
    import numpy as np

    temporary = path.with_suffix(".tmp")
    with temporary.open("wb") as stream:
        np.save(stream, vector, allow_pickle=False)
    temporary.replace(path)
    return {"file": path.name, "bytes": path.stat().st_size, "sha256": file_sha(path)}


def array_sha(*arrays):
    import numpy as np

    digest = hashlib.sha256()
    for array in arrays:
        digest.update(str((array.shape, array.dtype.str)).encode("ascii"))
        digest.update(memoryview(np.ascontiguousarray(array)).cast("B"))
    return digest.hexdigest()


def load_stage(directory):
    import numpy as np
    from scipy.sparse import load_npz

    for name, (size, digest) in ARTIFACTS.items():
        path = directory / name
        if not path.is_file() or path.stat().st_size != size or file_sha(path) != digest:
            raise ValueError(f"Saved artifact identity differs: {name}")
    stage = json.loads((directory / "stages_AP_pnc.json").read_bytes())
    lineage = json.loads((directory / SEED_DIR / "channels_AP_pnc.json").read_bytes())
    if (lineage["accepted"] or lineage["terminal"]["column"] != 0
            or any(lineage[key] != digest for key, digest in ARRAY_HASHES.items())):
        raise ValueError("Saved first-drive lineage differs.")
    matrix = load_npz(directory / "A_AP_pnc.npz")
    rhs = np.load(directory / "rhs_AP_pnc.npy", allow_pickle=False, mmap_mode="r")
    seed = np.load(directory / SEED_DIR / "terminal-000.npy", allow_pickle=False)
    complex_dtype = np.dtype(np.complex128)
    if (matrix.format != "csc" or matrix.shape != (DOFS, DOFS) or matrix.nnz != NNZ
            or matrix.dtype != complex_dtype or matrix.indices.dtype != np.int32
            or matrix.indptr.dtype != np.int32 or not matrix.has_canonical_format
            or rhs.shape != (DOFS, len(INCIDENT_ROOTS)) or rhs.dtype != complex_dtype
            or seed.shape != (DOFS,) or seed.dtype != complex_dtype
            or not np.isfinite(seed).all() or not np.isfinite(rhs).all()):
        raise ValueError("Saved matrix or vector layout differs.")
    if (array_sha(matrix.data, matrix.indices, matrix.indptr) != ARRAY_HASHES["operator_csc_sha256"]
            or array_sha(rhs) != ARRAY_HASHES["rhs_array_sha256"]
            or array_sha(rhs[:, 0]) != ARRAY_HASHES["rhs_column_sha256"]):
        raise ValueError("Saved equation array checksums differ.")
    return matrix, rhs, seed, stage["source"]


def infinity_norm(matrix):
    import numpy as np

    row_sums = np.zeros(matrix.shape[0], dtype=np.float64)
    for start in range(0, matrix.nnz, 2**20):
        part = slice(start, start + 2**20)
        np.add.at(row_sums, matrix.indices[part], np.abs(matrix.data[part]))
    norm = float(row_sums.max())
    if not math.isfinite(norm) or norm <= 0:
        raise ValueError("The original operator must have a finite positive infinity norm.")
    return norm


def residual_metrics(residual, b, vector, operator_norm):
    import numpy as np

    b_norm = float(np.linalg.norm(b))
    scale = operator_norm * float(np.max(abs(vector))) + float(np.max(abs(b)))
    if not (0 < b_norm < math.inf and 0 < scale < math.inf):
        raise ValueError("The original equation residual has an invalid normalization.")
    relative = float(np.linalg.norm(residual) / b_norm)
    backward = float(np.max(abs(residual)) / scale)
    if not (math.isfinite(relative) and math.isfinite(backward)):
        raise ValueError("The original equation residual is nonfinite.")
    return {"relative_residual": relative, "relative_backward_residual": backward}


def accepted(metrics):
    relative = metrics["relative_residual"]
    return (0 <= relative <= RELATIVE_TARGET and relative <= RESIDUAL_LIMIT
            and 0 <= metrics["relative_backward_residual"] <= RESIDUAL_LIMIT)


def solve_drives(matrix, rhs, first_seed, norm, publish, admit, max_updates=3):
    import numpy as np
    from scipy.sparse.linalg import splu

    for column in range(rhs.shape[1]):
        b = rhs[:, column].copy()
        current = first_seed if column == 0 else np.zeros_like(b)
        residual = b - matrix @ current
        publish("seed", {"column": column, **residual_metrics(residual, b, current, norm)})
        if column == 0:
            admit()
            started = time.monotonic()
            factor = splu(matrix.copy(), permc_spec="COLAMD")
            publish("factor", {"factor_seconds": time.monotonic() - started, "factor_nonzeros": int(factor.nnz)})
        for update in range(1, max_updates + 1):
            current = current + factor.solve(residual)
            if not np.isfinite(current).all():
                raise ValueError("Sparse correction produced a nonfinite candidate.")
            residual = b - matrix @ current
            metrics = residual_metrics(residual, b, current, norm)
            target_met = accepted(metrics)
            publish("candidate", {"column": column, "update": update, "target_met": target_met, **metrics}, current)
            if target_met:
                break
        else:
            return column
    return rhs.shape[1]


def replay_candidate(path, record, matrix, b, norm):
    import numpy as np

    replay = np.load(path, allow_pickle=False)
    if (file_sha(path) != record["sha256"] or replay.dtype != np.complex128
            or replay.shape != b.shape or not np.isfinite(replay).all()):
        raise ValueError("Saved candidate identity differs.")
    metrics = residual_metrics(matrix @ replay - b, b, replay, norm)
    if not accepted(metrics):
        raise ValueError("Saved candidate failed the original-equation replay.")
    return metrics


def smoke_check(attempt):
    import numpy as np
    from scipy.sparse import csc_matrix

    dense = np.array([[3 + 1j, 2j, 1], [1 - 2j, 4, 0], [2, 1j, 5 - 1j]])
    matrix = csc_matrix(dense)
    original = array_sha(matrix.data, matrix.indices, matrix.indptr)
    expected = np.array([[1 + 2j, 2j], [3, -2], [-1j, 0.5 + 3j]])
    rhs = dense @ expected
    reference = np.linalg.solve(dense, rhs)
    norm = infinity_norm(matrix)
    column_checks, factor_events = [], []
    with tempfile.TemporaryDirectory(dir=attempt) as scratch:
        def publish(event, data, vector=None):
            if event == "factor":
                factor_events.append(data)
            if event == "candidate" and data["target_met"]:
                column = data["column"]
                path = Path(scratch) / f"column-{column}.npy"
                replay = replay_candidate(path, write_vector(path, vector), matrix, rhs[:, column], norm)
                error = float(np.linalg.norm(vector - reference[:, column]) / np.linalg.norm(reference[:, column]))
                column_checks.append({"column": column, "dense_relative_error": error, **replay})

        count = solve_drives(matrix, rhs, expected[:, 0] + 1e-3j, norm, publish, lambda: None)
    if (count != 2 or [check["column"] for check in column_checks] != [0, 1] or len(factor_events) != 1
            or max(check["dense_relative_error"] for check in column_checks) > 1e-12
            or original != array_sha(matrix.data, matrix.indices, matrix.indptr)
            or accepted({"relative_residual": 2e-10, "relative_backward_residual": 1e-12})):
        raise RuntimeError("Complex smoke solution or matrix identity differs.")
    return {"columns": column_checks, "factorization_count": len(factor_events)}


def cgroup_controls(proc=Path("/proc/self")):
    """Return process memory controls from its own cgroup through mount ancestors."""
    groups = {}
    for line in (proc / "cgroup").read_text().splitlines():
        _, controllers, name = line.split(":", 2)
        if not controllers or "memory" in controllers.split(","):
            groups["cgroup2" if not controllers else "cgroup"] = Path(name)
    controls = []
    for line in (proc / "mountinfo").read_text().splitlines():
        prefix, suffix = line.split(" - ", 1)
        kind, _, options = suffix.split(maxsplit=2)
        if kind not in groups or (kind == "cgroup" and "memory" not in options.split(",")):
            continue
        root, mount = (Path(item.replace("\\040", " ")) for item in prefix.split()[3:5])
        path = groups[kind]
        if not path.is_relative_to(root):
            continue
        leaf = mount / path.relative_to(root)
        if kind == "cgroup2":
            limit, usage, peak, events = "memory.max", "memory.current", "memory.peak", "memory.events"
        else:
            limit, usage, peak, events = ("memory.limit_in_bytes", "memory.usage_in_bytes",
                                          "memory.max_usage_in_bytes", "memory.failcnt")
        for directory in (leaf, *leaf.parents):
            if not directory.is_relative_to(mount):
                break
            if (directory / limit).is_file() and (directory / usage).is_file():
                cap = (directory / limit).read_text().strip()
                cap = int(cap) if cap != "max" and 0 < int(cap) < 2**60 else None
                if cap is not None or is_job_group(directory):
                    controls.append((directory, cap, usage, peak, events))
    return controls


def is_job_group(directory):
    return f"job_{os.environ.get('SLURM_JOB_ID')}" in directory.parts


def proc_kib(path, key, default=None):
    for line in path.read_text().splitlines():
        if line.startswith(key + ":"):
            return int(line.split()[1]) * 1024
    if default is None:
        raise ValueError(f"Missing {key} in {path}")
    return default


def allocation():
    job = os.environ.get("SLURM_JOB_ID")
    if sys.platform != "linux" or not job:
        raise ValueError("The h80 correction requires a Linux Slurm allocation.")
    quota = int(os.environ.get("SLURM_MEM_PER_NODE", "0")) * MIB
    if quota < ADMISSION_BYTES + JOB_RESERVE:
        raise ValueError("Slurm node memory is missing or below the h80 initial factor admission budget.")
    return job, quota, cgroup_controls()


def memory_snapshot(quota, controls, worker_pid=None):
    node = proc_kib(Path("/proc/meminfo"), "MemAvailable")
    leader = proc_kib(Path("/proc/self/status"), "VmRSS")
    try:
        worker = 0 if worker_pid is None else proc_kib(Path(f"/proc/{worker_pid}/status"), "VmRSS", default=0)
    except (FileNotFoundError, ProcessLookupError):
        worker = 0
    aggregate_rss = leader + worker
    job_headroom = quota - aggregate_rss
    groups = []
    for directory, cap, usage, peak, events in controls:
        current = int((directory / usage).read_text())
        effective_cap = min(quota, cap) if is_job_group(directory) and cap is not None else cap or quota
        groups.append({"path": str(directory), "limit_bytes": cap, "effective_limit_bytes": effective_cap,
                       "current_bytes": current,
                       "peak": read_optional(directory / peak), "events": read_optional(directory / events)})
        job_headroom = min(job_headroom, effective_cap - current)
    return {"node_available_bytes": node, "job_headroom_bytes": job_headroom,
            "worker_rss_bytes": leader if worker_pid is None else worker, "cgroups": groups}


def read_optional(path):
    try:
        return path.read_text().strip()
    except FileNotFoundError:
        return None


def memory_block(snapshot, *, admission=False):
    needed = ADMISSION_BYTES if admission else 0
    if snapshot["worker_rss_bytes"] > RSS_LIMIT:
        return "Worker RSS exceeded the 112 GiB limit."
    if snapshot["node_available_bytes"] < NODE_RESERVE + needed:
        return "Node memory fell below the required reserve."
    if snapshot["job_headroom_bytes"] < JOB_RESERVE + needed:
        return "Job memory fell below the required reserve."
    return None


def parent_death_signal(parent_pid):
    if sys.platform == "linux":
        import ctypes

        if ctypes.CDLL(None, use_errno=True).prctl(1, signal.SIGTERM, 0, 0, 0) != 0:
            raise OSError(ctypes.get_errno(), "Could not configure worker shutdown on parent death.")
        if os.getppid() != parent_pid:
            raise RuntimeError("Worker supervisor exited before startup.")


def solve_environment():
    import numpy as np
    import scipy

    configs = {}
    for name, module in (("numpy", np), ("scipy", scipy)):
        with redirect_stdout(io.StringIO()) as stream:
            module.show_config()
        configs[name] = stream.getvalue()
    try:
        from threadpoolctl import threadpool_info
    except ImportError:
        pools = None
    else:
        pools = threadpool_info()
        if any(pool.get("num_threads", 1) > 1 for pool in pools):
            raise RuntimeError("Numerical library reports more than one thread.")
    identity = {"python": platform.python_version(), "platform": platform.platform(), "machine": platform.machine(),
                "node": platform.node(), "numpy": np.__version__, "scipy": scipy.__version__, "build_configs": configs,
                "threadpools": pools, "thread_limits": THREADS, "runner_sha256": file_sha(Path(__file__))}
    if sys.platform == "linux":
        cpu_model = next((line.partition(":")[2].strip() for line in Path("/proc/cpuinfo").read_text().splitlines()
                          if line.startswith("model name")), None)
        identity.update(cpu_model=cpu_model, available_cpus=len(os.sched_getaffinity(0)))
    return identity


def run_worker(args):
    parent_death_signal(args.parent_pid)
    report = {**IDENTITY, "status": "starting", "method": "complex128 SuperLU COLAMD residual correction",
              "relative_target": RELATIVE_TARGET, "relative_limit": RESIDUAL_LIMIT, "backward_limit": RESIDUAL_LIMIT,
              "candidate_history": [], "seed_metrics": {}, "best_candidates": {}, "accepted_channels": []}
    path = args.attempt_dir / PROGRESS

    def publish(event, data, vector=None):
        report.update(status=event)
        if event == "seed":
            report["seed_metrics"][str(data["column"])] = data
        elif vector is None:
            report[event] = data
        else:
            column = data["column"]
            path = args.attempt_dir / f"candidate-{column:03d}-{data['update']:03d}.npy"
            record = {**data, "incident_port_root": INCIDENT_ROOTS[column], **write_vector(path, vector)}
            report["candidate_history"].append(record)
            key = str(column)
            incumbent = report["best_candidates"].get(key, report["seed_metrics"][key])
            if data["relative_residual"] < incumbent["relative_residual"]:
                report["best_candidates"][key] = record
            if data["target_met"]:
                if column != len(report["accepted_channels"]):
                    raise RuntimeError("Accepted channel order differs from the staged right-hand sides.")
                replay = replay_candidate(path, record, matrix, rhs[:, column], norm)
                report["accepted_channels"].append({**record, "replay": replay})
                report["accepted_count"] = len(report["accepted_channels"])
        write_record(args.attempt_dir / PROGRESS, report)

    try:
        report["smoke"] = smoke_check(args.attempt_dir)
        report["solve_environment"] = solve_environment()
        if args.smoke_only:
            report["status"] = "smoke_passed"
            write_record(path, report)
            return 0
        job, quota, controls = allocation()

        def admit():
            snapshot = memory_snapshot(quota, controls)
            if reason := memory_block(snapshot, admission=True):
                raise MemoryError(reason)
            publish("factorization_admitted", snapshot)

        report.update(slurm_job_id=job, incident_port_roots=INCIDENT_ROOTS, frequency_hz=1e9,
                      accepted_count=0, total_drives=len(INCIDENT_ROOTS),
                      stage_artifacts={name: {"bytes": size, "sha256": digest}
                                       for name, (size, digest) in ARTIFACTS.items()}, original_array_sha256=ARRAY_HASHES)
        write_record(path, {**report, "status": "verifying_stage"})
        matrix, rhs, seed, report["windows_stage_source"] = load_stage(args.stage_dir)
        norm = infinity_norm(matrix)
        report["operator_infinity_norm"] = norm
        count = solve_drives(matrix, rhs, seed, norm, publish, admit)
        if array_sha(matrix.data, matrix.indices, matrix.indptr) != ARRAY_HASHES["operator_csc_sha256"]:
            raise ValueError("Correction changed the original operator arrays.")
        report["original_operator_preserved"] = True
        completed = count == len(INCIDENT_ROOTS)
        if count != len(report["accepted_channels"]):
            raise RuntimeError("Numerical and persisted accepted-drive counts differ.")
        report["status"] = "all_drives_passed" if completed else "target_not_reached"
        write_record(path, report)
        return 0 if completed else 1
    except Exception as exc:
        report.update(status="failed", error=f"{type(exc).__name__}: {exc}")
        write_record(path, report)
        raise


def stop_worker(child):
    if os.name != "posix":
        if child.poll() is None:
            child.kill()
        child.wait(timeout=5)
        return True

    def signal_group(sig):
        try:
            os.killpg(child.pid, sig)
        except ProcessLookupError:
            return False
        return True

    signal_group(signal.SIGTERM)
    until = time.monotonic() + 3
    while signal_group(0) and time.monotonic() < until:
        child.poll()
        time.sleep(0.05)
    signal_group(signal.SIGKILL)
    try:
        child.wait(timeout=5)
    except subprocess.TimeoutExpired:
        return False
    until = time.monotonic() + 2
    while signal_group(0) and time.monotonic() < until:
        time.sleep(0.05)
    return not signal_group(0)


def supervise(args):
    args.attempt_dir.mkdir()
    started = time.monotonic()
    report = {**IDENTITY, "status": "starting", "wall_limit_seconds": args.wall_seconds,
              "rss_limit_bytes": RSS_LIMIT, "node_reserve_bytes": NODE_RESERVE,
              "job_reserve_bytes": JOB_RESERVE, "initial_factor_estimate_bytes": INITIAL_FACTOR_BYTES,
              "admission_bytes": ADMISSION_BYTES, "sampled_peak_worker_rss_bytes": 0}
    execution_path = args.attempt_dir / EXECUTION
    child, canceled, reason = None, None, None
    quota, controls = None, []

    def request_stop(signum, _frame):
        nonlocal canceled
        canceled = signal.Signals(signum).name

    previous_handlers = {sig: signal.signal(sig, request_stop) for sig in (signal.SIGINT, signal.SIGTERM)}
    try:
        if not args.smoke_only:
            job, quota, controls = allocation()
            initial = memory_snapshot(quota, controls)
            report.update(slurm_job_id=job, slurm_memory_bytes=quota, initial_memory=initial,
                          admission_basis="Slurm quota and cgroup usage" if controls else
                          "Slurm quota and owned RSS only; no cgroup usage counters")
            if block := memory_block(initial, admission=True):
                raise MemoryError(block)
        command = [sys.executable, "-B", str(Path(__file__).resolve()), "--parent-pid", str(os.getpid()),
                   "--attempt-dir", str(args.attempt_dir)]
        command.extend(["--smoke-only"] if args.smoke_only else ["--stage-dir", str(args.stage_dir)])
        with (args.attempt_dir / "worker_AP_pnc.log").open("xb") as log:
            child = subprocess.Popen(command, stdout=log, stderr=subprocess.STDOUT, start_new_session=os.name == "posix")
            report.update(status="running", worker_pid=child.pid)
            write_record(execution_path, report)
            while child.poll() is None:
                if canceled:
                    reason = f"Supervisor received {canceled}."
                elif time.monotonic() - started >= args.wall_seconds:
                    reason = "Worker wall limit reached."
                elif not args.smoke_only:
                    current = memory_snapshot(quota, controls, child.pid)
                    report["last_memory"] = current
                    report["sampled_peak_worker_rss_bytes"] = max(report["sampled_peak_worker_rss_bytes"], current["worker_rss_bytes"])
                    reason = memory_block(current)
                if reason:
                    break
                time.sleep(0.2)
    except Exception as exc:
        reason = f"{type(exc).__name__}: {exc}"
    finally:
        report["worker_stopped"] = child is None or stop_worker(child)
        report["worker_exit_code"] = child.returncode if child else None
        report["elapsed_seconds"] = time.monotonic() - started
        if quota is not None:
            try:
                report["final_memory"] = memory_snapshot(quota, controls)
            except (OSError, ValueError) as exc:
                report["final_memory_error"] = str(exc)
        progress = args.attempt_dir / PROGRESS
        if progress.is_file():
            record = json.loads(progress.read_bytes())
            report.update(worker_status=record["status"], worker_progress_sha256=file_sha(progress))
            report.update({key: record[key] for key in ("accepted_count", "total_drives") if key in record})
        if not report["worker_stopped"]:
            report.update(status="blocked", reason="Owned worker process group did not exit.")
        elif reason:
            report.update(status="interrupted" if child else "blocked", reason=reason)
        else:
            report["status"] = report.get("worker_status", "failed")
            if child.returncode and report["status"] != "target_not_reached":
                report.update(status="failed", reason=f"Worker exited with code {child.returncode}.")
        write_record(execution_path, report)
        for sig, handler in previous_handlers.items():
            signal.signal(sig, handler)
    print(f"{report['status']}: {execution_path}", flush=True)
    return 0 if report["status"] in ("smoke_passed", "all_drives_passed") else 1


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument("--stage-dir", type=Path)
    mode.add_argument("--smoke-only", action="store_true")
    parser.add_argument("--attempt-dir", type=Path, required=True)
    parser.add_argument("--wall-seconds", type=int, default=WALL_LIMIT)
    parser.add_argument("--parent-pid", type=int, help=argparse.SUPPRESS)
    args = parser.parse_args()
    if not 0 < args.wall_seconds <= WALL_LIMIT:
        parser.error(f"--wall-seconds must be between 1 and {WALL_LIMIT}.")
    os.environ.update(THREADS)
    return run_worker(args) if args.parent_pid else supervise(args)


if __name__ == "__main__":
    raise SystemExit(main())
