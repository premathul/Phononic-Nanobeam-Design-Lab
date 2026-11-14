"""Bounded checkpointed matched-port scattering. Author: Athul Prem | _AP_pnc."""

from __future__ import annotations

from contextlib import contextmanager
from dataclasses import fields
import hashlib
import json
import math
import os
from pathlib import Path
from shutil import copyfile
import subprocess
import sys
import tempfile
from time import monotonic, sleep
from types import SimpleNamespace

import numpy as np
import psutil
from scipy.sparse import load_npz, save_npz

from . import workflow_AP_pnc as workflow


AUTHOR = {"author": "Athul Prem", "file_tag": "_AP_pnc"}
THREADS = dict.fromkeys(("OPENBLAS_NUM_THREADS", "OMP_NUM_THREADS", "MKL_NUM_THREADS", "BLIS_NUM_THREADS"), "1")
JOB_FILE = "job_AP_pnc.json"
JOURNAL_FILE = "channels_AP_pnc.json"
STAGE_FILE = "stage_AP_pnc.json"
STATUS_FILE = "execution_AP_pnc.json"
MIB = 2**20
PHYSICAL_FLOOR_MIB = 400
COMMIT_FLOOR_MIB = 512
WORKER_EXIT_TIMEOUT = 5


class _Busy(RuntimeError):
    """An existing attempt owns the checkpoint."""


class _OperatorSaved(Exception):
    """Exit the staging process before solving or postprocessing."""


class _MemoryBlocked(RuntimeError):
    """Available memory dropped before factorization could start."""


def _read(path):
    return json.loads(path.read_bytes())


def _sha(path):
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def _array_digest(*arrays):
    digest = hashlib.sha256()
    for array in arrays:
        digest.update(str((array.shape, array.dtype.str)).encode("ascii"))
        digest.update(memoryview(np.ascontiguousarray(array)).cast("B"))
    return digest.hexdigest()


def _link(path, job):
    return {"path": path.relative_to(job).as_posix(), "sha256": _sha(path), "bytes": path.stat().st_size}


def _verified(link, job):
    path = (job / link["path"]).resolve()
    if not path.is_relative_to(job) or not path.is_file() or path.stat().st_size != link["bytes"] or _sha(path) != link["sha256"]:
        raise ValueError(f"Checkpoint artifact changed or is missing: {link['path']}")
    return path


def _save_vector(path, vector):
    temporary = path.with_suffix(".tmp")
    with temporary.open("wb") as stream:
        np.save(stream, vector, allow_pickle=False)
    temporary.replace(path)


def _write_journal(path, state):
    for retry in range(6):
        try:
            workflow.write_json(path, state)
            return
        except PermissionError as exc:
            # A Windows reader can briefly deny the sharing mode needed by replace.
            if getattr(exc, "winerror", None) not in (5, 32) or retry == 5:
                raise
            sleep(0.02)


@contextmanager
def _exclusive(path):
    """An OS lock releases on process death and serializes competing resumes."""
    with path.open("a+b") as stream:
        if os.name == "nt":
            import msvcrt

            stream.seek(0)
            try:
                msvcrt.locking(stream.fileno(), msvcrt.LK_NBLCK, 1)
            except OSError as exc:
                raise _Busy("Another process owns this scattering checkpoint.") from exc
        else:
            import fcntl

            try:
                fcntl.flock(stream.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
            except BlockingIOError as exc:
                raise _Busy("Another process owns this scattering checkpoint.") from exc
        yield


def _numerical_identity(identity):
    return {key: identity[key] for key in ("source_sha256", "packages", "python", "platform")}


def _job_inputs(job):
    saved = _read(job / JOB_FILE)
    if saved["schema_version"] != 1 or saved["inputs_sha256"] != workflow.digest(saved["inputs"]):
        raise ValueError("Invalid scattering checkpoint inputs.")
    inputs = saved["inputs"]
    if inputs["numerical_identity"] != _numerical_identity(workflow.source_identity()) or inputs["threads"] != THREADS:
        raise ValueError("The checkpoint numerical implementation or environment differs from this installation.")
    return saved


def _solver_options(config):
    resources = config["resources"]
    return {"fill_factor": resources.get("ilu_fill_factor", 10.0),
            "single_precision": resources.get("ilu_precision", "double") == "single",
            **workflow.lgmres_options(resources)}


def _stage(attempt, saved):
    from .geometry_AP_pnc import BeamGeometry, Hole
    from .material_AP_pnc import CubicMaterial
    from .scattering_AP_pnc import solve_scattering

    inputs = saved["inputs"]
    config = inputs["configuration"]
    pattern = inputs["hole_pattern"]
    geometry = BeamGeometry(*(config["geometry"][key] * 1e-9 for key in workflow.DIMENSIONS_NM))
    holes = tuple(Hole(**entry) for entry in pattern["holes"]) if pattern else ()
    job = attempt.parent
    stage_dir = attempt / "stage"
    stage_dir.mkdir()
    settings = config["mesh"]
    resources = config["resources"]
    params = config["material"]
    material = CubicMaterial(C11=params["C11_Pa"], C12=params["C12_Pa"], C44=params["C44_Pa"],
                             rho=params["density_kg_m3"], rotation=params["crystal_rotation"])
    with workflow.measured_resources() as staging_resources:
        record = workflow._prepare_mesh_checkpoint(config, geometry, holes, pattern,
                                                   inputs["numerical_identity"], job.parents[1])
        if not record["within_body_budget"]:
            raise ValueError(f"Mesh has {record['ndof']} displacement DOFs, exceeding cap {resources['max_dofs']}.")
        original_mesh = Path(record["output_directory"])
        mesh_dir = stage_dir / "mesh"
        mesh_dir.mkdir()
        # Keep an independent copy of the admitted mesh and its original manifest.
        for name in ("summary_AP_pnc.json", "mesh_AP_pnc.npz"):
            copyfile(original_mesh / name, mesh_dir / name)
        mesh, _ = workflow.load_checkpoint(mesh_dir, record["mesh_hash"])
        with workflow.measured_resources() as assembly_resources:
            system = workflow._assemble_system(mesh, config, material)
        analysis = config["analysis"]
        frequency = analysis["frequency_GHz"] * 1e9
        port = workflow.assemble_port(system, material, max_dofs=resources["max_port_dofs"])
        modes = workflow.solve_port_modes(port, frequency)
        for name, matrix in (("K", system.K), ("M", system.M)):
            save_npz(stage_dir / f"{name}.npz", matrix, compressed=False)
        np.savez(stage_dir / "ports.npz", left_dofs=port.left_dofs, right_dofs=port.right_dofs,
                 **{field.name: getattr(modes, field.name) for field in fields(modes) if field.name != "port"})
        operator_info = {}

        def save_operator(A, rhs):
            save_npz(stage_dir / "A.npz", A, compressed=False)
            np.save(stage_dir / "rhs.npy", rhs, allow_pickle=False)
            operator_info.update(shape=list(rhs.shape), nnz=int(A.nnz),
                                 operator_storage_bytes=sum(a.nbytes for a in (A.data, A.indices, A.indptr)),
                                 index_bytes=A.indices.itemsize, working_dtype=A.dtype.name,
                                 operator_sha256=_array_digest(A.data, A.indices, A.indptr),
                                 rhs_sha256=_array_digest(rhs))
            raise _OperatorSaved

        try:
            solve_scattering(system, modes, frequency, analysis["incident_side"], linear_solver=save_operator)
        except _OperatorSaved:
            pass
        artifacts = {name: _link(stage_dir / name, job) for name in
                     ("A.npz", "rhs.npy", "K.npz", "M.npz", "ports.npz", "mesh/summary_AP_pnc.json", "mesh/mesh_AP_pnc.npz")}
    manifest = {**AUTHOR, "status": "operator_ready", "inputs_sha256": saved["inputs_sha256"],
                "artifacts": artifacts, "mesh_hash": record["mesh_hash"], "assembly_resources": assembly_resources,
                "staging_resources": staging_resources, **operator_info}
    workflow.write_json(stage_dir / STAGE_FILE, manifest)


def _stage_manifest(attempt, saved):
    state = _read(attempt / JOURNAL_FILE)
    manifest_path = _verified(state["stage"], attempt.parent)
    manifest = _read(manifest_path)
    if (state["inputs_sha256"] != saved["inputs_sha256"] or manifest["inputs_sha256"] != saved["inputs_sha256"]
            or manifest["status"] != "operator_ready" or state["total_channels"] != manifest["shape"][1]):
        raise ValueError("The staged operator or journal does not match the saved configuration.")
    return state, manifest


def _saved_channels(state, job, size):
    accepted, guesses = {}, {}
    committed = state["accepted"]
    if [item["column"] for item in committed] != list(range(len(committed))) or len(committed) > state["total_channels"]:
        raise ValueError("Accepted channel order is invalid.")
    active = state["active"]
    if active is not None and (active["column"] != len(committed) or active["column"] >= state["total_channels"]):
        raise ValueError("Active iterate does not belong to the next remaining channel.")
    for records, vectors in ((committed, accepted), ([] if active is None else [active], guesses)):
        for item in records:
            column = item["column"]
            vector = np.load(_verified(item["vector"], job), allow_pickle=False)
            if vector.shape != (size,) or vector.dtype != np.dtype("complex128") or not np.isfinite(vector).all():
                raise ValueError(f"Invalid saved vector for RHS {column + 1}.")
            vectors[column] = vector
    for item in committed:
        channel = item["diagnostics"]
        if (item["event"] != "accepted" or channel["column"] != item["column"]
                or type(channel["info"]) is not int or channel["info"] != 0
                or not all(_valid_residual(channel[key]) for key in ("relative_residual", "relative_backward_residual"))):
            raise ValueError("A saved channel lacks original solver acceptance.")
    return accepted, guesses


def _valid_residual(value):
    return type(value) in (float, int) and 0 <= value <= 1e-9


def _solve(attempt, saved):
    from .linear_lgmres_AP_pnc import solve_ilu_lgmres

    job = attempt.parent
    state, manifest = _stage_manifest(attempt, saved)
    options = _solver_options(saved["inputs"]["configuration"])
    links = manifest["artifacts"]
    paths = {name: _verified(link, job) for name, link in links.items()}
    A = load_npz(paths["A.npz"])
    rhs = np.load(paths["rhs.npy"], allow_pickle=False)
    if (A.format != "csc" or list(rhs.shape) != manifest["shape"] or A.shape != (rhs.shape[0],) * 2
            or A.dtype != np.dtype("complex128") or rhs.dtype != A.dtype
            or _array_digest(A.data, A.indices, A.indptr) != manifest["operator_sha256"]
            or _array_digest(rhs) != manifest["rhs_sha256"]):
        raise ValueError("The original CSC operator or RHS identity differs from the staged system.")
    accepted, guesses = _saved_channels(state, job, A.shape[0])
    report = {}

    def save_channel(event, column, vector, diagnostics):
        if event == "accepted":
            path = attempt / f"accepted-{column:03d}.npy"
            if path.exists():
                raise RuntimeError(f"An accepted vector already exists for RHS {column + 1}.")
        else:
            active = state["active"]
            slot = 1 if active is not None and active["vector"]["path"] == f"{attempt.name}/iterate-0.npy" else 0
            path = attempt / f"iterate-{slot}.npy"
        _save_vector(path, vector)
        entry = {"column": column, "vector": _link(path, job), "event": event, "diagnostics": diagnostics}
        if event == "accepted":
            state["accepted"].append(entry)
            state["active"] = None
        else:
            state["active"] = entry
        _write_journal(attempt / JOURNAL_FILE, state)

    try:
        if len(accepted) < manifest["shape"][1]:
            budget = _solver_budget(manifest, options, len(accepted))
            ram, commit = _memory_snapshot()
            input_mib = (manifest["operator_storage_bytes"] + rhs.nbytes * (1 + len(accepted) / rhs.shape[1])) / MIB
            if ram < PHYSICAL_FLOOR_MIB or commit < budget["required_commit_mib"] - input_mib:
                raise _MemoryBlocked("Available memory fell below the pre-factorization reserve. Resume when more memory is free.")
        solve_ilu_lgmres(A, rhs, report, **options, accepted_solutions=accepted,
                         initial_guesses=guesses, checkpoint_callback=save_channel)
    finally:
        workflow.write_json(attempt / "linear_solver_AP_pnc.json", report)


def _export(attempt, saved):
    from skfem import Dofs, ElementTetP1, ElementTetP2, ElementVector
    from .assembly_batched_AP_pnc import _DisplacementDofLayout
    from .elasticity_AP_pnc import ElasticSystem
    from .linear_lgmres_AP_pnc import solve_ilu_lgmres
    from .port_modes_AP_pnc import PortModes

    job = attempt.parent
    state, manifest = _stage_manifest(attempt, saved)
    links = manifest["artifacts"]
    config = saved["inputs"]["configuration"]
    ndof, total_channels = manifest["shape"]
    mesh_dir = _verified(links["mesh/summary_AP_pnc.json"], job).parent
    if _verified(links["mesh/mesh_AP_pnc.npz"], job).parent != mesh_dir:
        raise ValueError("The tagged mesh and its manifest differ.")
    mesh, mesh_record = workflow.load_checkpoint(mesh_dir, manifest["mesh_hash"])
    order = config["mesh"]["element_order"]
    element = ElementVector(ElementTetP1() if order == 1 else ElementTetP2())
    layout = _DisplacementDofLayout(mesh, element, Dofs(mesh, element), 2 * order)
    K, M = (load_npz(_verified(links[f"{name}.npz"], job)) for name in ("K", "M"))
    if layout.N != ndof or K.shape != (ndof, ndof) or M.shape != K.shape:
        raise ValueError("The staged stiffness and mass differ from the saved mesh DOFs.")
    system = ElasticSystem(mesh, layout, K, M)
    with np.load(_verified(links["ports.npz"], job), allow_pickle=False) as arrays:
        left, right = arrays["left_dofs"], arrays["right_dofs"]
        port = SimpleNamespace(elastic_system=system, basis=SimpleNamespace(N=len(left)), left_dofs=left, right_dofs=right)
        modes = PortModes(port=port, frequency_hz=float(arrays["frequency_hz"].item()),
                          **{field.name: arrays[field.name] for field in fields(PortModes)
                             if field.name not in ("port", "frequency_hz")})
    accepted, _ = _saved_channels(state, job, ndof)
    report = _read(attempt / "linear_solver_AP_pnc.json")
    if len(accepted) != total_channels or len(report["channels"]) != total_channels:
        raise ValueError("Every incident channel must be accepted before export.")

    def restore_solution(A, rhs):
        if (_array_digest(A.data, A.indices, A.indptr) != manifest["operator_sha256"]
                or _array_digest(rhs) != manifest["rhs_sha256"]):
            raise ValueError("The restored mesh, matrices or modes do not reproduce the original equations.")
        return solve_ilu_lgmres(A, rhs, **_solver_options(config), accepted_solutions=accepted)

    prepared = {"mesh_directory": mesh_dir, "mesh_checkpoint": (mesh, mesh_record), "system": system,
                "pattern": saved["inputs"]["hole_pattern"], "assembly_resources": manifest["assembly_resources"],
                "port_modes": modes, "solver_report": report, "linear_solver": restore_solution,
                "attempt": str(attempt), "provenance": {
                    "attempt": str(attempt), "inputs_sha256": saved["inputs_sha256"], "stage": state["stage"],
                    "original_operator_sha256": manifest["operator_sha256"], "original_rhs_sha256": manifest["rhs_sha256"],
                    "accepted": state["accepted"], "resumed_from": state["resumed_from"],
                }}
    summary = workflow.run_study(config, job.parents[1], _prepared_scattering=prepared)
    workflow.write_json(attempt / "result_AP_pnc.json", summary)


def _memory_snapshot():
    physical = psutil.virtual_memory().available / MIB
    if os.name != "nt":
        # Without a portable OS commit counter, budget against available physical RAM.
        return physical, physical
    from ctypes import Structure, WinDLL, WinError, byref, c_size_t, sizeof
    from ctypes.wintypes import DWORD

    class PerformanceInfo(Structure):
        _fields_ = [("cb", DWORD)] + [(name, c_size_t) for name in (
            "commit_total", "commit_limit", "commit_peak", "physical_total", "physical_available", "system_cache",
            "kernel_total", "kernel_paged", "kernel_nonpaged", "page_size",
        )] + [(name, DWORD) for name in ("handles", "processes", "threads")]

    info = PerformanceInfo(cb=sizeof(PerformanceInfo))
    if not WinDLL("psapi", use_last_error=True).GetPerformanceInfo(byref(info), info.cb):
        raise WinError()
    return physical, (info.commit_limit - info.commit_total) * info.page_size / MIB


def _solver_budget(manifest, options, accepted_channels, prior_peak_mib=0):
    dofs, channels = manifest["shape"]
    sparse_bytes = manifest["operator_storage_bytes"]
    rhs_bytes = 16 * dofs * channels
    remaining = channels - accepted_channels
    if not remaining:
        return {"required_commit_mib": COMMIT_FLOOR_MIB + (2 * sparse_bytes + 3 * rhs_bytes) / MIB,
                "required_physical_mib": PHYSICAL_FLOOR_MIB}
    factor_value_bytes = 8 if options["single_precision"] else 16
    index_bytes = manifest["index_bytes"]
    nnz = manifest["nnz"]
    factor_input = (factor_value_bytes + index_bytes) * nnz + index_bytes * (dofs + 1)
    # The measured h100 driver reserves 24 bytes/entry for mixed-precision factor buffers.
    mixed_buffer_bytes = 2 * (factor_value_bytes + index_bytes)
    factor = mixed_buffer_bytes * options["fill_factor"] * nnz
    # FGMRES keeps both the original and preconditioned vectors, including augmentation.
    krylov = (2 * options["inner_m"] + 4 * options["outer_k"] + 14) * dofs * 16
    margin = 1024 if factor >= 1024 * MIB else COMMIT_FLOOR_MIB
    required = (sparse_bytes + 3 * rhs_bytes + factor_input + factor + krylov) / MIB + margin
    return {"required_commit_mib": max(required, prior_peak_mib + COMMIT_FLOOR_MIB),
            "required_physical_mib": 1200 if factor >= 1024 * MIB else PHYSICAL_FLOOR_MIB}


_WATCH_PARENT = """
import sys, time
import psutil

def exact(pid, started):
    try:
        process = psutil.Process(int(pid))
        if process.create_time() == float(started) and process.status() != psutil.STATUS_ZOMBIE:
            return process
    except psutil.NoSuchProcess:
        pass

parent_pid, parent_start, worker_pid, worker_start, deadline = sys.argv[1:]
while (worker := exact(worker_pid, worker_start)) is not None:
    if exact(parent_pid, parent_start) is None or time.monotonic() >= float(deadline):
        try:
            worker.kill()
        except psutil.NoSuchProcess:
            pass
        break
    time.sleep(0.1)
"""


def _worker(phase, attempt, parent_pid, parent_start, deadline):
    job = attempt.parent
    with _exclusive(job / "worker.lock"):
        worker = psutil.Process()
        watch = subprocess.Popen([sys.executable, "-B", "-c", _WATCH_PARENT, str(parent_pid), str(parent_start),
                                  str(worker.pid), str(worker.create_time()), str(deadline + 1)],
                                 stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
                                 creationflags=subprocess.CREATE_NO_WINDOW if os.name == "nt" else 0)
        try:
            saved = _job_inputs(job)
            {"staging": _stage, "solving": _solve, "exporting": _export}[phase](attempt, saved)
        except Exception as exc:
            workflow.write_json(attempt / f"{phase}_error_AP_pnc.json", {
                "message": f"{type(exc).__name__}: {exc}",
                "invalid_checkpoint": isinstance(exc, (ValueError, KeyError, TypeError)),
                "blocked": isinstance(exc, _MemoryBlocked),
            })
            raise
        finally:
            if watch.poll() is None:
                watch.terminate()
            watch.wait()


def _worker_running(process):
    try:
        return process.is_running() and process.status() != psutil.STATUS_ZOMBIE
    except psutil.NoSuchProcess:
        return False
    except psutil.AccessDenied:
        return True


def _stop_phase_worker(child, descendants):
    deadline = monotonic() + WORKER_EXIT_TIMEOUT
    discovered = True
    if child.poll() is None:
        try:
            descendants.update(dict.fromkeys(psutil.Process(child.pid).children(recursive=True)))
        except psutil.NoSuchProcess:
            pass
        except psutil.AccessDenied:
            discovered = False
    for member in reversed(descendants):
        try:
            member.kill()
        except (psutil.NoSuchProcess, psutil.AccessDenied):
            pass
    try:
        if child.poll() is None:
            child.kill()
        exit_code = child.wait(timeout=max(0, deadline - monotonic()))
    except (OSError, subprocess.TimeoutExpired):
        return child.poll(), False

    # The Windows launcher can exit before the Python process releases its files.
    while any(_worker_running(member) for member in descendants):
        remaining = deadline - monotonic()
        if remaining <= 0:
            return exit_code, False
        sleep(min(0.05, remaining))
    return exit_code, discovered


def _monitor(phase, attempt, deadline, update, should_cancel):
    parent = psutil.Process()
    script_root = str(Path(__file__).resolve().parents[1])
    env = {**os.environ, **THREADS, "PYTHONPATH": os.pathsep.join(filter(None, (script_root, os.environ.get("PYTHONPATH"))))}
    command = [sys.executable, "-B", "-m", __spec__.name,
               phase, str(attempt), str(parent.pid), str(parent.create_time()), str(deadline)]
    initial_ram, initial_commit = _memory_snapshot()
    minimum_ram, minimum_commit = initial_ram, initial_commit
    peak_rss = peak_private = 0
    started = monotonic()
    status, message = "completed", ""
    with (attempt / f"{phase}_worker_AP_pnc.log").open("x", encoding="utf-8") as log:
        child = subprocess.Popen(command, env=env, cwd=workflow.ROOT, stdout=log, stderr=subprocess.STDOUT,
                                 creationflags=subprocess.CREATE_NO_WINDOW if os.name == "nt" else 0)
        descendants = {}
        try:
            while child.poll() is None:
                try:
                    process = psutil.Process(child.pid)
                    children = process.children(recursive=True)
                    descendants.update(dict.fromkeys(children))
                    samples = [member.memory_info() for member in (process, *children)]
                    peak_rss = max(peak_rss, sum(sample.rss for sample in samples) / MIB)
                    peak_private = max(peak_private, sum(getattr(sample, "private", sample.rss) for sample in samples) / MIB)
                except psutil.NoSuchProcess:
                    pass
                ram, commit = _memory_snapshot()
                minimum_ram, minimum_commit = min(ram, minimum_ram), min(commit, minimum_commit)
                update()
                if should_cancel():
                    status, message = "interrupted", "The attempt was stopped."
                elif monotonic() >= deadline:
                    status, message = "interrupted", "The attempt reached its wall-time limit."
                elif ram < PHYSICAL_FLOOR_MIB or commit < COMMIT_FLOOR_MIB:
                    status, message = "interrupted", "The attempt stopped because available memory fell below its reserve."
                else:
                    sleep(0.05)
                    continue
                break
        except KeyboardInterrupt:
            status, message = "interrupted", "The attempt was stopped."
        finally:
            exit_code, worker_stopped = _stop_phase_worker(child, descendants)
    error_path = attempt / f"{phase}_error_AP_pnc.json"
    invalid = False
    if not worker_stopped:
        status = "blocked"
        message = f"Could not confirm the {phase} worker stopped. Wait for it to exit before resuming."
    elif status == "completed" and exit_code:
        status = "interrupted" if monotonic() >= deadline else "failed"
        message = f"The {phase} worker exited with code {exit_code}. See its saved log."
        if error_path.exists():
            error = _read(error_path)
            message, invalid = error["message"], error["invalid_checkpoint"]
            if error["blocked"]:
                status = "blocked"
    return {"status": status, "message": message, "invalid_checkpoint": invalid, "exit_code": exit_code,
            "worker_stopped": worker_stopped,
            "elapsed_seconds": monotonic() - started, "peak_rss_mib": peak_rss, "peak_private_mib": peak_private,
            "minimum_physical_available_mib": minimum_ram, "minimum_commit_headroom_mib": minimum_commit}


def _response(attempt, status, message, *, phase=None, can_resume=True):
    journal_path = attempt / JOURNAL_FILE if attempt else None
    state = _read(journal_path) if journal_path and journal_path.is_file() else {}
    return {"status": status, "message": message, "resume_directory": str(attempt) if attempt else None,
            "accepted_channels": len(state.get("accepted", ())), "total_channels": state.get("total_channels"),
            "phase": phase, "can_resume": can_resume}


def _run_attempt(attempt, saved, prior, deadline, wall_seconds, progress_callback, should_cancel):
    job = attempt.parent
    if prior is None:
        state = {"stage": None, "accepted": [], "active": None, "total_channels": None,
                 "inputs_sha256": saved["inputs_sha256"]}
    else:
        state = _read(prior / JOURNAL_FILE)
        if state["inputs_sha256"] != saved["inputs_sha256"]:
            raise ValueError("The prior attempt uses different scientific inputs.")
        if state["stage"] is None:
            committed = prior / "stage" / STAGE_FILE
            if committed.is_file():
                state.update(stage=_link(committed, job), total_channels=_read(committed)["shape"][1])
    state["resumed_from"] = prior.name if prior else None
    request_path = attempt / JOURNAL_FILE
    _write_journal(request_path, state)
    execution = {**AUTHOR, "wall_seconds": wall_seconds, "status": "running", "phases": {}}
    execution_path = attempt / STATUS_FILE
    callback = progress_callback or (lambda _: None)
    cancel = should_cancel or (lambda: False)

    def finish(status, message, phase, *, can_resume=True):
        current = _response(attempt, status, message, phase=phase, can_resume=can_resume)
        if can_resume and _read(request_path)["stage"] is None:
            current["message"] += " Resume will restart staging from the saved inputs."
        execution.update(current)
        workflow.write_json(execution_path, execution)
        return current

    for phase in ("staging", "solving", "exporting"):
        if phase == "staging" and state["stage"] is not None:
            continue
        setup = _response(attempt, "running", f"Scattering {phase}.", phase=phase)
        execution.update(setup)
        workflow.write_json(execution_path, execution)
        callback(setup)
        if cancel() or monotonic() >= deadline:
            return finish("interrupted", "The attempt stopped before starting the next phase.", phase)
        if phase == "solving":
            state, manifest = _stage_manifest(attempt, saved)
            prior_peak = 0
            if prior and (prior / STATUS_FILE).is_file():
                previous = _read(prior / STATUS_FILE)
                prior_peak = max(previous.get("prior_solver_peak_private_mib", 0),
                                 previous.get("phases", {}).get("solving", {}).get("peak_private_mib", 0))
            execution["prior_solver_peak_private_mib"] = prior_peak
            budget = _solver_budget(manifest, _solver_options(saved["inputs"]["configuration"]), len(state["accepted"]), prior_peak)
            execution["solver_admission"] = budget
            ram, commit = _memory_snapshot()
            if ram < budget["required_physical_mib"] or commit < budget["required_commit_mib"]:
                message = (f"Insufficient memory for this staged solve. Available RAM {ram:.0f} MiB, "
                           f"commit headroom {commit:.0f} MiB. Required RAM {budget['required_physical_mib']:.0f} MiB, "
                           f"commit headroom {budget['required_commit_mib']:.0f} MiB.")
                return finish("blocked", message, phase)

        def update():
            nonlocal setup
            current = _response(attempt, "running", setup["message"], phase=phase)
            if current["accepted_channels"] != setup["accepted_channels"]:
                callback(current)
                setup = current

        result = _monitor(phase, attempt, deadline, update, cancel)
        execution["phases"][phase] = result
        if phase == "staging" and result["worker_stopped"]:
            # The stage may have committed just before the watchdog stopped the worker.
            committed = attempt / "stage" / STAGE_FILE
            if committed.is_file():
                state.update(stage=_link(committed, job), total_channels=_read(committed)["shape"][1])
                _write_journal(request_path, state)
        if result["status"] != "completed":
            can_resume = result["worker_stopped"] and not result["invalid_checkpoint"]
            return finish(result["status"], result["message"], phase, can_resume=can_resume)
        workflow.write_json(execution_path, execution)
    summary = _read(attempt / "result_AP_pnc.json")
    complete = finish("completed_unverified", "Scattering results were exported.", "exporting", can_resume=False)
    return {**summary, **complete}


def run_resumable_scattering(config=None, output_root=None, *, resume_from=None, wall_seconds=1800,
                             progress_callback=None, should_cancel=None) -> dict:
    """Start a fresh bounded single-frequency run or continue its latest saved attempt."""
    if (config is None) == (resume_from is None) or (resume_from is not None and output_root is not None):
        raise ValueError("Supply a configuration and optional output root, or a saved attempt directory.")
    if type(wall_seconds) not in (int, float) or not math.isfinite(wall_seconds) or wall_seconds <= 0:
        raise ValueError("wall_seconds must be positive and finite.")
    deadline = monotonic() + wall_seconds
    prior = None
    if resume_from is None:
        config = workflow.resolve_config(config)
        if config["analysis"]["kind"] != "scattering" or config["resources"].get("linear_solver") != "ilu_lgmres":
            raise ValueError("Checkpointing requires single-frequency scattering with linear_solver=ilu_lgmres.")
        _, _, pattern = workflow._resolve_geometry(config)
        root = Path(output_root or workflow.ROOT / "results" / "runs").resolve()
        checkpoint_root = root / "_checkpoints"
        checkpoint_root.mkdir(parents=True, exist_ok=True)
        job = Path(tempfile.mkdtemp(prefix=f"{config['name']}-", dir=checkpoint_root))
        identity = workflow.source_identity()
        inputs = {"configuration": config, "hole_pattern": pattern, "numerical_identity": _numerical_identity(identity),
                  "threads": THREADS}
        saved = {**AUTHOR, "schema_version": 1, "inputs": inputs, "inputs_sha256": workflow.digest(inputs),
                 "source_revision": identity["git_commit"]}
        workflow.write_json(job / JOB_FILE, saved)
    else:
        prior = Path(resume_from).resolve()
        job = prior.parent
        if not prior.is_dir() or not (prior / JOURNAL_FILE).is_file() or not (job / JOB_FILE).is_file():
            return _response(None, "failed", "Select a checkpoint attempt directory from a resumable scattering run.",
                             can_resume=False)
    attempt = prior
    try:
        with _exclusive(job / "supervisor.lock"):
            # A worker can outlive a supervisor killed by the host. Leave its files untouched.
            with _exclusive(job / "worker.lock"):
                pass
            saved = _job_inputs(job)
            occupied = sorted((path for path in job.glob("attempt-*") if path.is_dir() and path.name[8:].isdigit()),
                              key=lambda path: int(path.name[8:]))
            attempts = [path for path in occupied if (path / JOURNAL_FILE).is_file()]
            if prior is not None and (not attempts or prior != attempts[-1]):
                return _response(attempts[-1] if attempts else None, "blocked",
                                 "Select the latest attempt for this scattering checkpoint.", can_resume=bool(attempts))
            if prior is not None and (prior / "result_AP_pnc.json").is_file():
                summary = _read(prior / "result_AP_pnc.json")
                public = Path(summary["output_directory"])
                if not summary["output_sha256"] or not all(_sha(public / name) == sha for name, sha in summary["output_sha256"].items()):
                    raise ValueError("The exported scattering result has changed or is missing.")
                return {**summary, **_response(prior, "completed_unverified", "Scattering results were already exported.",
                                                phase="exporting", can_resume=False), "reused": True}
            next_index = int(occupied[-1].name[8:]) + 1 if occupied else 1
            attempt = job / f"attempt-{next_index:06d}"
            attempt.mkdir()
            return _run_attempt(attempt, saved, prior, deadline, wall_seconds, progress_callback, should_cancel)
    except _Busy as exc:
        return _response(attempt, "blocked", str(exc))
    except (OSError, ValueError, KeyError, TypeError) as exc:
        status = {"status": "failed", "message": f"{type(exc).__name__}: {exc}", "can_resume": False,
                  "resume_directory": str(attempt) if attempt else None, "accepted_channels": 0, "total_channels": None}
        if attempt is not None and attempt != prior:
            workflow.write_json(attempt / STATUS_FILE, {**AUTHOR, **status})
        return status


if __name__ == "__main__":
    _worker(sys.argv[1], Path(sys.argv[2]), int(sys.argv[3]), float(sys.argv[4]), float(sys.argv[5]))
