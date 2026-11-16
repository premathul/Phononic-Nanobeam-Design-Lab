"""Small saved-solution replay checks. Athul Prem | _AP_pnc."""

from copy import deepcopy
from pathlib import Path
from tempfile import TemporaryDirectory
import unittest
from unittest.mock import patch

import numpy as np
from scipy.sparse import csc_matrix

from tools import compare_scattering_AP_pnc as compare
from tools import export_lawrence_scattering_AP_pnc as exporter
from tools import lawrence_solver_AP_pnc as solver


class LawrenceExportTests(unittest.TestCase):
    def test_linux_replay_is_explicit_and_preserves_source_and_package_checks(self):
        staged = {"platform": "Windows-11", "python": "3.12.14", "packages": {"numpy": "2.5.4"},
                  "source_sha256": {"scattering.py": "original", exporter.COORDINATOR: exporter.REVIEWED_COORDINATOR_PAIR[0]}}
        current = deepcopy(staged)
        current.update(platform="Linux-x86_64", python="3.12.12")
        current["source_sha256"][exporter.COORDINATOR] = exporter.REVIEWED_COORDINATOR_PAIR[1]
        exporter.check_replay_environment(staged, current, linux_replay=True)
        with self.assertRaisesRegex(ValueError, "replay environment differs"):
            exporter.check_replay_environment(staged, current)
        with self.assertRaisesRegex(ValueError, "requires a Windows stage and a Linux export"):
            exporter.check_replay_environment(staged, staged, linux_replay=True)
        current["source_sha256"]["scattering.py"] = "modified"
        with self.assertRaisesRegex(ValueError, "replay source differs"):
            exporter.check_replay_environment(staged, current, linux_replay=True)
        current["packages"]["numpy"] = "2.5.5"
        with self.assertRaisesRegex(ValueError, "differs from the staged packages"):
            exporter.check_replay_environment(staged, current, linux_replay=True)

    def test_receipt_requires_linked_process_status_and_vector_order(self):
        zero = {"relative_residual": 0.0, "relative_backward_residual": 0.0}
        with TemporaryDirectory() as folder:
            attempt = Path(folder)
            common = {"status": "all_drives_passed", "slurm_job_id": "42", "accepted_count": 5, "total_drives": 5}
            execution = {**common, "worker_stopped": True, "worker_exit_code": 0}
            progress = {**common, "frequency_hz": 1e9, "incident_port_roots": list(solver.INCIDENT_ROOTS),
                        "stage_artifacts": {name: {"bytes": size, "sha256": digest}
                                            for name, (size, digest) in solver.ARTIFACTS.items()},
                        "original_array_sha256": solver.ARRAY_HASHES, "original_operator_preserved": True,
                        "solve_environment": {"runner_sha256": exporter.RUNNER_SHA256, "platform": "Linux-test",
                                              "thread_limits": solver.THREADS}, "accepted_channels": []}
            for column, root in enumerate(solver.INCIDENT_ROOTS):
                path = attempt / f"candidate-{column:03d}-001.npy"
                record = solver.write_vector(path, np.array([column + 1j]))
                progress["accepted_channels"].append({**record, "column": column, "incident_port_root": root,
                                                      "update": 1, "target_met": True, "replay": zero})

            def publish():
                progress_path = attempt / solver.PROGRESS
                solver.write_record(progress_path, progress)
                execution["worker_progress_sha256"] = solver.file_sha(progress_path)
                solver.write_record(attempt / solver.EXECUTION, execution)

            publish()
            metadata, vectors = exporter.read_receipt(attempt, "42")
            self.assertEqual([path.name for path in vectors], [row["file"] for row in metadata["accepted_channels"]])
            execution["worker_stopped"] = False
            publish()
            with self.assertRaisesRegex(ValueError, "stopped, successful"):
                exporter.read_receipt(attempt, "42")
            execution["worker_stopped"] = True
            progress["accepted_channels"][:2] = reversed(progress["accepted_channels"][:2])
            publish()
            with self.assertRaisesRegex(ValueError, "reordered incident drives"):
                exporter.read_receipt(attempt, "42")

    def test_replay_recomputes_complex_residuals_against_each_original_column(self):
        matrix = csc_matrix(np.array([[1 + 2j, 2], [3j, 5]]))
        vectors = np.array([[1, 2j], [-1j, 3 + 1j]])
        rhs = matrix @ vectors
        hashes = {"operator_csc_sha256": solver.array_sha(matrix.data, matrix.indices, matrix.indptr),
                  "rhs_array_sha256": solver.array_sha(rhs)}
        with TemporaryDirectory() as folder, patch.dict(solver.ARRAY_HASHES, hashes):
            paths = [Path(folder) / f"candidate-{column}.npy" for column in range(2)]
            records = [{"incident_port_root": root, **solver.write_vector(path, vector)}
                       for root, path, vector in zip(solver.INCIDENT_ROOTS, paths, vectors.T)]
            solution, checks = exporter.replay_vectors(matrix, rhs, paths, records)
            np.testing.assert_array_equal(solution, vectors)
            self.assertEqual([entry["column"] for entry in checks], [0, 1])
            self.assertTrue(all(solver.accepted(entry) for entry in checks))
            with self.assertRaisesRegex(ValueError, "original-equation replay"):
                exporter.replay_vectors(matrix, rhs, paths[::-1], records[::-1])
            with self.assertRaisesRegex(ValueError, "admitted original equations"):
                exporter.replay_vectors(matrix, rhs[:, ::-1], paths, records)

    def test_comparator_accepts_original_coordinate_proof_and_checks_staged_sources(self):
        core = {f"src/phononic/{module}_AP_pnc.py": "a" * 64 for module in (*compare.CORE, "assembly_batched")}
        columns = [{"column": i, "incident_port_root": root, "relative_residual": 1e-12,
                    "relative_backward_residual": 1e-13} for i, root in enumerate(solver.INCIDENT_ROOTS)]
        method = exporter.REPLAY_METHOD
        direct = {"configuration": {"resources": {"linear_solver": "direct"}}, "num_incident_channels": 5,
                  "incident_mode_indices": list(solver.INCIDENT_ROOTS), "implementation": {"source_sha256": core},
                  "linear_solver": {"method": method, "working_dtype": "complex128", "channels": deepcopy(columns)},
                  "checkpointed_scattering": {"method": method, "slurm_job_id": "42",
                                              "windows_assembly_source": {"source_sha256": deepcopy(core)}}}
        source = "459589f4e5de39c888966305e28dbd2b16f2947ec427ae9c131c69ae47c609df"
        iterative = {"configuration": {"resources": {"linear_solver": "ilu_lgmres", "ilu_precision": "single"}},
                     "num_incident_channels": 5, "linear_solver": {"method": "ilu_lgmres", "working_dtype": "complex128",
                     "factor_dtype": "complex64", "channels": [{**row, "info": 0} for row in columns]}}
        physical = {"numerical_source_sha256": core}
        previous = {"numerical_source_sha256": {**core, compare.LGMRES_SOURCE: source}}
        _, compatibility = compare.compatible_inputs(iterative, direct, previous, physical)
        self.assertEqual(compatibility["checks"]["candidate"]["slurm_job_id"], "42")
        direct["linear_solver"]["channels"][0]["relative_residual"] = 2e-10
        with self.assertRaisesRegex(ValueError, "residual limits"):
            compare.compatible_inputs(iterative, direct, previous, physical)
        direct["checkpointed_scattering"]["windows_assembly_source"]["source_sha256"].clear()
        with self.assertRaisesRegex(ValueError, "physical numerical kernels"):
            compare.compatible_inputs(iterative, direct, previous, physical)


if __name__ == "__main__":
    unittest.main()
