#!/usr/bin/env python3
"""Run the prepared 168.75 nm endpoints sequentially. Author: Athul Prem | _AP_pnc."""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import os
from pathlib import Path
import re
import signal
import socket
import subprocess
import sys
from time import monotonic, sleep

import psutil

ROOT = Path(__file__).resolve().parents[1]
if __name__ == "__main__":
    sys.path.insert(0, str(ROOT))

from tools import transfer_frequency_preparation_AP_pnc as transfer

native = transfer.native
workflow = native.workflow
RESPONSES = (("995MHz", 0.995), ("1GHz", 1.0))
FREQUENCIES = [frequency for _, frequency in RESPONSES]
REPLAY = Path("results/runs/regional_mesh_AP_pnc/local_checks/replay_regional_flux_AP_pnc.py")
GIB = 2**30
WALL_SECONDS = 300
FLUX_WALL_SECONDS = 90
SAMPLE_INTERVAL = 0.1


def _slurm_job(job, host):
    try:
        line = subprocess.check_output(["scontrol", "-o", "show", "job", job], text=True, timeout=10)
    except (OSError, subprocess.SubprocessError) as exc:
        raise ValueError(f"Could not verify allocated resources for Slurm job {job}.") from exc
    fields = dict(re.findall(r"(?:^|\s)([A-Za-z]+)=([^\s]+)", line))
    if (fields.get("JobId") != job or fields.get("NumNodes") != "1" or fields.get("NumCPUs") != "1"
            or _host_label(fields.get("NodeList", "")) != _host_label(host)):
        raise ValueError("The Slurm job record does not bind this job and compute host to one node and one CPU.")
    return fields


def _host_label(host):
    return host.split(".", 1)[0].lower()


def _allocation():
    env = os.environ
    job = env.get("SLURM_JOB_ID", "")
    step = env.get("SLURM_STEP_ID") or env.get("SLURM_STEPID", "")
    tasks = env.get("SLURM_STEP_NUM_TASKS") or env.get("SLURM_NTASKS")
    cpus = env.get("SLURM_CPUS_PER_TASK") or env.get("SLURM_CPUS_ON_NODE")
    nodes = env.get("SLURM_JOB_NUM_NODES") or env.get("SLURM_NNODES", "1")
    memory_key = "SLURM_MEM_PER_NODE" if env.get("SLURM_MEM_PER_NODE") else "SLURM_MEM_PER_CPU"
    memory_mib = env.get(memory_key)
    partition = env.get("SLURM_JOB_PARTITION", "")
    node = env.get("SLURMD_NODENAME") or env.get("SLURM_JOB_NODELIST", "")
    host = socket.gethostname()
    threads = {name: env.get(name) for name in (*native.THREADS, "NUMEXPR_NUM_THREADS")}
    if (sys.platform != "linux" or not job.isdigit() or not (step.isdigit() or step == "interactive")
            or tasks != "1" or cpus not in (None, "1") or nodes != "1" or set(threads.values()) != {"1"}):
        raise ValueError("Run inside the interactive Lawrence compute step with one task, one CPU and one numerical thread.")
    if not memory_mib or not node or not partition or cpus is None:
        fields = _slurm_job(job, host)
        if partition and partition != fields.get("Partition"):
            raise ValueError("Slurm's job and step partitions differ.")
        partition = fields.get("Partition", "")
        node = node or fields["NodeList"]
        cpus = fields["NumCPUs"]
    if memory_mib:
        allocation = int(memory_mib) * 2**20
    else:
        for field in ("AllocTRES", "TRES"):
            match = re.search(r"(?:^|,)mem=(\d+)([KMGT]?)(?:,|$)", fields.get(field, ""))
            if match:
                count, unit = match.groups()
                powers = {"K": 10, "M": 20, "G": 30, "T": 40, "": 20}
                allocation = int(count) * 2**powers[unit]
                memory_key = f"scontrol:{field}"
                break
        else:
            raise ValueError("Slurm did not report the job's allocated memory in AllocTRES or TRES.")
    if not partition or _host_label(node) != _host_label(host) or allocation != 24 * GIB:
        raise ValueError("Run inside the interactive Lawrence compute step with one task, one CPU, "
                         "one numerical thread and 24 GiB of allocated memory.")
    return {"slurm_job_id": job, "slurm_partition": partition, "slurm_node": node,
            "slurm_step_id": step, "host_name": host, "slurm_cpus_per_task": int(cpus),
            "slurm_allocated_memory_bytes": allocation, "slurm_memory_source": memory_key,
            "resident_limit_bytes": allocation - 2 * GIB,
            "thread_environment": threads}


def _tree_rss(pid):
    try:
        process = psutil.Process(pid)
        processes = [process, *process.children(recursive=True)]
    except psutil.NoSuchProcess:
        return 0
    rss = 0
    for process in processes:
        try:
            rss += process.memory_info().rss
        except psutil.NoSuchProcess:
            continue
    return rss


def _kill_group(pid, sig):
    try:
        os.killpg(pid, sig)
    except ProcessLookupError:
        pass


def _guarded(argv, resident_limit, wall_seconds):
    """Bound a single Linux child and report wait4's measured per-child peak RSS."""
    started = monotonic()
    peak, guard, stopped_at = 0, None, None
    with subprocess.Popen(argv, cwd=ROOT, start_new_session=True) as worker:
        try:
            while True:
                peak = max(peak, _tree_rss(worker.pid))
                pid, status, usage = os.wait4(worker.pid, os.WNOHANG)
                if pid:
                    worker.returncode = os.waitstatus_to_exitcode(status)
                    break
                elapsed = monotonic() - started
                if guard is None and (peak > resident_limit or elapsed >= wall_seconds):
                    guard = "resident_limit" if peak > resident_limit else "wall_time"
                    stopped_at = monotonic()
                    _kill_group(worker.pid, signal.SIGINT)
                if stopped_at is not None and monotonic() - stopped_at >= 5:
                    _kill_group(worker.pid, signal.SIGKILL)
                sleep(SAMPLE_INTERVAL)
        except BaseException:
            _kill_group(worker.pid, signal.SIGKILL)
            raise
    child_peak = usage.ru_maxrss * 1024
    elapsed = monotonic() - started
    if guard is None and (child_peak > resident_limit or elapsed > wall_seconds):
        guard = "resident_limit" if child_peak > resident_limit else "wall_time"
    return {"elapsed_seconds": elapsed, "exit_code": worker.returncode, "guard": guard,
            "peak_sampled_process_tree_rss_sum_bytes": peak, "peak_child_rss_bytes": child_peak}


def _check_revision(expected, implementation, donor_revision):
    if not re.fullmatch(r"[0-9a-f]{40}", expected):
        raise ValueError("Pass the full original source revision recorded in the staged donor.")
    staged_git = implementation["git_commit"]
    if staged_git is not None:
        if staged_git != expected:
            raise ValueError("The staged Git HEAD differs from the requested revision.")
        return "git_head"
    if donor_revision != expected:
        raise ValueError("The requested packaged revision differs from the verified Windows donor revision.")
    return "packaged_revision_verified_against_windows_donor"


def _validate_response(output, index, receipt_path, expected):
    state = native.read_map_checkpoint(output)
    point = state["points"][index]
    if point["status"] != "accepted":
        raise ValueError(f"The selected point was not accepted at {output}.")
    native._read_accepted(point, state["inputs"]["configuration"], state["shared"], output)
    record = native._read(receipt_path)
    if (any(record[key] != value for key, value in expected.items())
            or record["exit_code"] != 0 or record["guard"] is not None
            or not 0 < record["elapsed_seconds"] <= record["wall_seconds"]
            or not 0 < record["peak_sampled_process_tree_rss_sum_bytes"] <= record["resident_limit_bytes"]
            or not 0 < record["peak_child_rss_bytes"] <= record["resident_limit_bytes"]
            or record["accepted_point_summary"] != point["accepted"]["summary"]):
        raise ValueError(f"The accepted point lacks a matching successful execution receipt: {receipt_path}.")


def _run_response(output, config, index, sample, allocation, provenance):
    phase = f"response_{sample}"
    receipt = output.parent / "local_checks" / f"{output.name}_{phase}_execution_AP_pnc.json"
    argv = [sys.executable, "-B", "-u", "tools/map_scattering_AP_pnc.py", "--config", config.relative_to(ROOT).as_posix(),
            "--frequencies-ghz", "0.995", "1", "--output", output.relative_to(ROOT).as_posix(),
            "--max-new-points", "1"]
    expected = {"phase": phase, "configuration_sha256": native._sha(config), **provenance}
    state = native.read_map_checkpoint(output)
    if state["points"][index]["status"] == "accepted":
        _validate_response(output, index, receipt, expected)
        print(f"Verified saved {output.name} {sample}", flush=True)
        return
    if receipt.exists():
        raise ValueError(f"An earlier successful execution receipt exists for an unaccepted point: {receipt}.")
    if any(point["status"] != "accepted" for point in state["points"][:index]):
        raise ValueError("A preceding point must be accepted before the next bounded response.")
    started = datetime.now(timezone.utc).isoformat()
    result = _guarded(argv, allocation["resident_limit_bytes"], WALL_SECONDS)
    record = {**native.AUTHOR, **expected, **allocation, **result, "argv": argv, "started_utc": started,
              "wall_seconds": WALL_SECONDS, "sample_interval_seconds": SAMPLE_INTERVAL,
              "runner_source_sha256": native._sha(Path(__file__).resolve())}
    point = native.read_map_checkpoint(output)["points"][index]
    if result["exit_code"] == 0 and result["guard"] is None and point["status"] == "accepted":
        record["accepted_point_summary"] = point["accepted"]["summary"]
        workflow.write_json(receipt, record)
        _validate_response(output, index, receipt, expected)
    else:
        count = len(point["attempts"])
        failure = receipt.with_name(f"{output.name}_{phase}_job{allocation['slurm_job_id']}_attempt{count}_execution_AP_pnc.json")
        if failure.exists():
            raise ValueError(f"A failure receipt already exists for the same job and attempt: {failure}.")
        workflow.write_json(failure, record)
        raise RuntimeError(f"The bounded response stopped or failed. Retained receipt: {failure}.")


def _check_flux(path, output, index):
    state = native.read_map_checkpoint(output)
    publication = native.read_saved_map(output)
    point = state["points"][index]
    result = native._read(path)
    if (result["status"] != "passed_saved_discretization_flux_replay"
            or result["map_checkpoint_sha256"] != publication["checkpoint_sha256"]
            or result["map_checkpoint_sha256"] != native._sha(output / native.CHECKPOINT)
            or result["point_summary"] != point["accepted"]["summary"]
            or result["frequency_hz"] != point["frequency_GHz"] * 1e9):
        raise ValueError(f"Independent flux receipt differs from the published point: {path}.")


def _replay_flux(output, index, sample, frequency, allocation):
    path = output.parent / f"{output.name}_{sample}_flux_AP_pnc.json"
    if not path.exists():
        # Stage replay output so an interrupted write leaves the checkpoint and its final path reusable.
        partial = path.with_name(f"{path.stem}_job{allocation['slurm_job_id']}_partial.json")
        argv = [sys.executable, "-B", "-u", REPLAY.as_posix(), output.relative_to(ROOT).as_posix(),
                str(frequency), "--output", partial.relative_to(ROOT).as_posix()]
        result = _guarded(argv, allocation["resident_limit_bytes"], FLUX_WALL_SECONDS)
        if result["exit_code"] != 0 or result["guard"] is not None:
            raise RuntimeError(f"Independent flux replay stopped or failed for {output.name} {sample}.")
        _check_flux(partial, output, index)
        partial.rename(path)
    _check_flux(path, output, index)


def run_campaign(study, revision, transfer_hashes):
    allocation = _allocation()
    implementation = workflow.source_identity()
    study = Path(study).resolve(strict=True)
    if not study.is_relative_to(ROOT) or not (ROOT / REPLAY).is_file():
        raise ValueError("Stage the study and the existing independent flux driver under the project root.")
    checks = study / "local_checks"
    checks.mkdir(exist_ok=True)
    cases = []
    with native._exclusive(checks / ".lawrence_nested_central.lock"):
        for role, thickness, transfer_sha in zip(("thick", "thin"), (200, 100), transfer_hashes, strict=True):
            stem = f"{role}{thickness}_central400_edge168p75_rim32"
            donor, output = study / stem, study / f"{stem}_linux"
            receipt, config = (study / f"{stem}_{item}_AP_pnc.json" for item in ("transfer", "configuration"))
            if not re.fullmatch(r"[0-9a-f]{64}", transfer_sha):
                raise ValueError(f"Pass the pinned SHA-256 of {receipt}.")
            saved_config = workflow.resolve_config(native._read(config))
            source = native.read_map_checkpoint(donor)
            basis = _check_revision(revision, implementation, source["source_git_commit"])
            mesh = saved_config["mesh"]
            if (workflow.digest(saved_config) != workflow.digest(source["inputs"]["configuration"])
                    or saved_config["geometry"]["thickness_nm"] != thickness or mesh["central_max_edge_nm"] != 168.75
                    or mesh["element_order"] != 2 or mesh["circle_resolution"] != 32 or mesh["thickness_layers"] != 2
                    or saved_config["analysis"]["incident_side"] != "left"
                    or saved_config["resources"].get("linear_solver", "direct") != "direct"):
                raise ValueError(f"The staged configuration differs from the selected direct P2 donor: {config}.")
            prepared = transfer.import_preparation(donor, receipt, transfer_sha, output, FREQUENCIES)
            origin = prepared["preparation_reuse"]["receipt"]
            provenance = {"source_revision": revision, "source_revision_basis": basis,
                          "host_platform": implementation["platform"],
                          "windows_donor_checkpoint_sha256": origin["donor_checkpoint_sha256"]}
            cases.append((output, config, provenance))
        for output, config, provenance in cases:
            for index, (sample, _) in enumerate(RESPONSES):
                _run_response(output, config, index, sample, allocation, provenance)
            # Publish both flux receipts against the final two-point checkpoint identity.
            for index, (sample, frequency) in enumerate(RESPONSES):
                _replay_flux(output, index, sample, frequency, allocation)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("study", type=Path)
    parser.add_argument("source_revision")
    parser.add_argument("thick_transfer_sha256")
    parser.add_argument("thin_transfer_sha256")
    args = parser.parse_args(argv)
    try:
        run_campaign(args.study, args.source_revision, (args.thick_transfer_sha256, args.thin_transfer_sha256))
    except (OSError, RuntimeError, ValueError, KeyError, TypeError) as exc:
        print(f"Lawrence campaign stopped: {exc}", file=sys.stderr)
        return 2
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
