"""Replay accepted Lawrence vectors at the saved h80 ports. Athul Prem | _AP_pnc."""

import argparse
from dataclasses import fields
import json
import os
from pathlib import Path
import sys
from types import SimpleNamespace

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT), str(ROOT / "src")]

from tools import lawrence_solver_AP_pnc as solver


STAGE_DIRECTORY = ROOT / "results/runs/full_device_resource_profile/full_h80_operator_1r5fdlft"
RUNNER_SHA256 = "dd9e07d4955de6cf4cac97eec0485bfc34bd1d7f8173d4707718c2ea4ba89034"
REPLAY_METHOD = "lawrence_superlu_replay"
COORDINATOR = "src/phononic/resumable_scattering_AP_pnc.py"
REVIEWED_COORDINATOR_PAIR = (
    "68c12549b47a9ec98abf7b39d78ad597dd114da00e326adf8ebbec060947b53f",
    "dba6002d982647810f3fc714fdce73a0f34e306e06e8519df2d54778002205ec",
)


def read_json(path):
    return json.loads(path.read_bytes())


def artifact(path):
    path = path.resolve()
    label = path.relative_to(ROOT).as_posix() if path.is_relative_to(ROOT) else str(path)
    return {"path": label, "bytes": path.stat().st_size, "sha256": solver.file_sha(path)}


def verified(path, size, digest):
    if not path.is_file() or path.stat().st_size != size or solver.file_sha(path) != digest:
        raise ValueError(f"Artifact identity differs: {path}")
    return path


def read_receipt(attempt, job_id):
    """Bind the ordered candidate files to a stopped successful Lawrence job."""
    execution_path, progress_path = attempt / solver.EXECUTION, attempt / solver.PROGRESS
    execution, progress = (read_json(path) for path in (execution_path, progress_path))
    count = len(solver.INCIDENT_ROOTS)
    if (execution.get("worker_stopped") is not True or type(execution.get("worker_exit_code")) is not int
            or execution["worker_exit_code"] != 0 or execution.get("worker_progress_sha256") != solver.file_sha(progress_path)
            or any(record.get("status") != "all_drives_passed" or record.get("slurm_job_id") != job_id
                   or record.get("accepted_count") != count or record.get("total_drives") != count
                   for record in (execution, progress))):
        raise ValueError("Lawrence receipt does not verify a stopped, successful five-drive job.")
    environment = progress["solve_environment"]
    expected_artifacts = {name: {"bytes": size, "sha256": digest} for name, (size, digest) in solver.ARTIFACTS.items()}
    if (environment["runner_sha256"] != RUNNER_SHA256 or not environment["platform"].startswith("Linux")
            or environment["thread_limits"] != solver.THREADS or progress["frequency_hz"] != 1e9
            or progress["incident_port_roots"] != list(solver.INCIDENT_ROOTS)
            or progress["stage_artifacts"] != expected_artifacts
            or progress["original_array_sha256"] != solver.ARRAY_HASHES
            or progress.get("original_operator_preserved") is not True):
        raise ValueError("Lawrence solve provenance differs from the admitted saved equations.")
    channels = progress.get("accepted_channels", [])
    if [entry["column"] for entry in channels] != list(range(count)):
        raise ValueError("Lawrence receipt has missing or reordered incident drives.")
    paths = []
    for column, entry in enumerate(channels):
        root = solver.INCIDENT_ROOTS[column]
        update = entry["update"]
        if (type(update) is not int or not 1 <= update <= 3 or entry["incident_port_root"] != root
                or entry.get("target_met") is not True or not solver.accepted(entry["replay"])
                or entry["file"] != f"candidate-{column:03d}-{update:03d}.npy"):
            raise ValueError(f"Invalid accepted receipt for incident drive {column + 1}.")
        paths.append(verified(attempt / entry["file"], entry["bytes"], entry["sha256"]))
    return progress, paths


def check_replay_environment(staged, current, *, linux_replay=False):
    if linux_replay and (not staged["platform"].startswith("Windows") or not current["platform"].startswith("Linux")):
        raise ValueError("Explicit Linux replay requires a Windows stage and a Linux export environment.")
    environment_keys = ("packages",) if linux_replay else ("packages", "python", "platform")
    for key in environment_keys:
        if current[key] != staged[key]:
            raise ValueError(f"Local replay environment differs from the staged {key}.")
    expected, actual = (dict(identity["source_sha256"]) for identity in (staged, current))
    if (expected.get(COORDINATOR), actual.get(COORDINATOR)) == REVIEWED_COORDINATOR_PAIR:
        expected.pop(COORDINATOR)
        actual.pop(COORDINATOR)
    if actual != expected:
        raise ValueError("Local replay source differs from the original staged numerical implementation.")


def read_stage(stage_dir):
    stage_path = verified(stage_dir / "stages_AP_pnc.json", *solver.ARTIFACTS["stages_AP_pnc.json"])
    stage = read_json(stage_path)
    paths = {}
    for key in ("configuration", "K", "M", "port_modes", "mesh", "mesh_checkpoint_summary", "hole_pattern"):
        link = stage[key]
        relative = Path(link["path"])
        directory = stage_dir if key in ("configuration", "K", "M", "port_modes") else ROOT / relative.parent
        paths[key] = verified(directory / relative.name, link["bytes"], link["sha256"])
    return stage, paths


def replay_vectors(matrix, rhs, paths, channels):
    import numpy as np

    if (solver.array_sha(matrix.data, matrix.indices, matrix.indptr) != solver.ARRAY_HASHES["operator_csc_sha256"]
            or solver.array_sha(rhs) != solver.ARRAY_HASHES["rhs_array_sha256"]):
        raise ValueError("Restored matrices and ports do not reproduce the admitted original equations.")
    norm = solver.infinity_norm(matrix)
    solution = np.empty(rhs.shape, dtype=rhs.dtype)
    diagnostics = []
    for column, (path, channel) in enumerate(zip(paths, channels, strict=True)):
        b = rhs[:, column]
        vector = np.load(path, allow_pickle=False)
        if (solver.file_sha(path) != channel["sha256"] or vector.dtype != rhs.dtype
                or vector.shape != b.shape or not np.isfinite(vector).all()):
            raise ValueError(f"Accepted vector identity differs for drive {column + 1}.")
        metrics = solver.residual_metrics(matrix @ vector - b, b, vector, norm)
        if not solver.accepted(metrics):
            raise ValueError(f"Accepted vector failed original-equation replay for drive {column + 1}: {metrics}")
        diagnostics.append({"column": column, "incident_port_root": channel["incident_port_root"], **metrics})
        solution[:, column] = vector
    return solution, diagnostics


def export(attempt, job_id, output_root, stage_dir=STAGE_DIRECTORY, *, linux_replay=False):
    import numpy as np
    from scipy.sparse import load_npz
    from skfem import Dofs, ElementTetP2, ElementVector

    from phononic import workflow_AP_pnc as workflow
    from phononic.assembly_batched_AP_pnc import _DisplacementDofLayout
    from phononic.elasticity_AP_pnc import ElasticSystem
    from phononic.port_modes_AP_pnc import PortModes

    progress, vectors = read_receipt(attempt, job_id)
    stage, paths = read_stage(stage_dir)
    if progress["windows_stage_source"] != stage["source"]:
        raise ValueError("Lawrence assembly provenance differs from the local saved stage.")
    check_replay_environment(stage["source"], workflow.source_identity(), linux_replay=linux_replay)
    config = read_json(paths["configuration"])
    config["resources"] = {key: value for key, value in config["resources"].items()
                           if not key.startswith(("ilu_", "lgmres_"))}
    config["resources"]["linear_solver"] = "direct"
    mesh_dir = paths["mesh"].parent
    if paths["mesh_checkpoint_summary"].parent != mesh_dir:
        raise ValueError("Saved mesh and mesh manifest differ.")
    mesh, mesh_record = workflow.load_checkpoint(mesh_dir, stage["mesh_hash"])
    pattern = mesh_record["hole_pattern_provenance"]
    if pattern["source_sha256"] != stage["hole_pattern"]["sha256"]:
        raise ValueError("Saved hole provenance differs from the staged pattern.")
    element = ElementVector(ElementTetP2())
    layout = _DisplacementDofLayout(mesh, element, Dofs(mesh, element), 4)
    K, M = (load_npz(paths[name]) for name in ("K", "M"))
    if layout.N != solver.DOFS or K.shape != (layout.N, layout.N) or M.shape != K.shape:
        raise ValueError("Saved stiffness and mass do not match the h80 displacement layout.")
    system = ElasticSystem(mesh, layout, K, M)
    with np.load(paths["port_modes"], allow_pickle=False) as stored:
        left, right = stored["left_dofs"], stored["right_dofs"]
        coordinates = layout.doflocs
        if (left.shape != (675,) or right.shape != (675,)
                or not np.allclose(coordinates[0, left], coordinates[0].min(), atol=1e-20, rtol=0)
                or not np.allclose(coordinates[0, right], coordinates[0].max(), atol=1e-20, rtol=0)
                or not np.allclose(coordinates[1:, left], coordinates[1:, right], atol=1e-20, rtol=0)):
            raise ValueError("Saved port maps do not match the original end planes.")
        port = SimpleNamespace(elastic_system=system, basis=SimpleNamespace(N=675), left_dofs=left, right_dofs=right)
        modes = PortModes(port=port, frequency_hz=float(stored["frequency_hz"].item()),
                          **{field.name: stored[field.name] for field in fields(PortModes)
                             if field.name not in ("port", "frequency_hz")})
    roots = modes.positive_indices[modes.propagating[modes.positive_indices]]
    if roots.tolist() != list(solver.INCIDENT_ROOTS):
        raise ValueError("Saved port modes do not match the solved incident-drive order.")
    report = {"method": REPLAY_METHOD, "working_dtype": "complex128", "solver": progress["method"],
              "relative_target": solver.RELATIVE_TARGET, "relative_residual_limit": solver.RESIDUAL_LIMIT,
              "backward_residual_limit": solver.RESIDUAL_LIMIT}

    def restore_solution(matrix, rhs):
        solution, checks = replay_vectors(matrix, rhs, vectors, progress["accepted_channels"])
        report["channels"] = checks
        return solution

    stage_link = artifact(stage_dir / "stages_AP_pnc.json")
    execution, receipt = (artifact(attempt / name) for name in (solver.EXECUTION, solver.PROGRESS))
    provenance = {"method": REPLAY_METHOD, "slurm_job_id": job_id, "execution": execution, "receipt": receipt,
                  "windows_assembly_stage": stage_link, "windows_assembly_source": stage["source"],
                  "linux_solve_environment": progress["solve_environment"],
                  "original_array_sha256": solver.ARRAY_HASHES,
                  "accepted_vectors": progress["accepted_channels"], "exporter": artifact(Path(__file__))}
    if linux_replay:
        linux_environment = solver.solve_environment()
        linux_environment.pop("runner_sha256")
        provenance["linux_export_environment"] = {**linux_environment, "slurm_job_id": os.environ.get("SLURM_JOB_ID")}
    prepared = {"mesh_directory": mesh_dir, "mesh_checkpoint": (mesh, mesh_record), "system": system,
                "pattern": pattern, "assembly_resources": stage["stages"]["production_batched_assembly"],
                "port_modes": modes, "solver_report": report, "linear_solver": restore_solution,
                "attempt": f"lawrence-{job_id}-{stage_link['sha256']}-{receipt['sha256']}", "provenance": provenance}
    summary = workflow.run_study(config, output_root, _prepared_scattering=prepared)
    if summary.get("status") != "completed_unverified" or summary.get("power_balance_passed") is not True:
        raise ValueError(f"Local power and flux checks did not pass: {summary['output_directory']}")
    return summary


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--attempt-dir", type=Path, required=True)
    parser.add_argument("--slurm-job-id", required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--stage-dir", type=Path, default=STAGE_DIRECTORY)
    parser.add_argument("--linux-replay", action="store_true", help="Verify exact staged equations in a Linux export environment.")
    args = parser.parse_args()
    os.environ.update(solver.THREADS)
    result = export(args.attempt_dir, args.slurm_job_id, args.output_root, args.stage_dir, linux_replay=args.linux_replay)
    print(result["output_directory"])
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
