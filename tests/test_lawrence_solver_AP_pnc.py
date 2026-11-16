"""Acceptance checks for the saved-equation Lawrence pilot. Athul Prem | _AP_pnc."""

import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest
from unittest.mock import patch

import numpy as np
from scipy.sparse import csc_matrix

from tools import lawrence_solver_AP_pnc as runner


class LawrenceSolverTests(unittest.TestCase):
    def test_supervised_smoke_and_existing_attempt_are_safe(self):
        with tempfile.TemporaryDirectory() as root:
            attempt = Path(root) / "smoke"
            args = [sys.executable, "-B", runner.__file__, "--smoke-only", "--attempt-dir", str(attempt)]
            result = subprocess.run(args, capture_output=True, timeout=30)
            self.assertEqual(result.returncode, 0, result.stderr.decode(errors="replace"))
            record_path = attempt / runner.PROGRESS
            progress = json.loads(record_path.read_bytes())
            execution = json.loads((attempt / runner.EXECUTION).read_bytes())
            self.assertEqual(execution["status"], "smoke_passed")
            self.assertTrue(execution["worker_stopped"])
            self.assertEqual(execution["worker_progress_sha256"], runner.file_sha(record_path))
            columns = progress["smoke"]["columns"]
            self.assertEqual([column["column"] for column in columns], [0, 1])
            self.assertEqual(progress["smoke"]["factorization_count"], 1)
            for column in columns:
                self.assertLess(column["dense_relative_error"], 1e-12)
                self.assertTrue(runner.accepted(column))
            self.assertGreater(subprocess.run(args, capture_output=True, timeout=5).returncode, 0)
            self.assertEqual(execution["worker_progress_sha256"], runner.file_sha(record_path))

    def test_original_residual_and_candidate_checks_reject_false_passes(self):
        matrix = csc_matrix(np.array([[4j, 1.0], [1.0, 2j]]))
        vector = np.array([1.0, 2j])
        b = matrix @ vector
        norm = runner.infinity_norm(matrix)
        with tempfile.TemporaryDirectory() as root:
            path = Path(root) / "candidate.npy"
            record = runner.write_vector(path, vector + 1e-8)
            with self.assertRaisesRegex(ValueError, "original-equation"):
                runner.replay_candidate(path, record, matrix, b, norm)
            actual = runner.write_vector(path, vector)
            with self.assertRaisesRegex(ValueError, "identity"):
                runner.replay_candidate(path, record, matrix, b, norm)
            with self.assertRaisesRegex(ValueError, "original-equation"):
                runner.replay_candidate(path, actual, matrix, 2 * b, norm)
        self.assertFalse(runner.accepted({"relative_residual": 2e-10, "relative_backward_residual": 0}))
        self.assertFalse(runner.accepted({"relative_residual": 1e-11, "relative_backward_residual": 2e-9}))

    def test_failed_first_drive_stops_before_the_second(self):
        matrix = csc_matrix(np.eye(2, dtype=np.complex128))
        rhs = np.array([[1j, 2], [3, -1j]])
        events = []
        with patch("scipy.sparse.linalg.splu") as factor:
            factor.return_value.nnz = 2
            factor.return_value.solve.return_value = np.zeros(2, dtype=np.complex128)
            count = runner.solve_drives(matrix, rhs, np.zeros(2, dtype=np.complex128), 1,
                                        lambda event, data, *_: events.append((event, data)), lambda: None)
        self.assertEqual(count, 0)
        self.assertEqual(factor.call_count, 1)
        self.assertEqual(factor.return_value.solve.call_count, 3)
        self.assertEqual([data["column"] for event, data in events if event == "seed"], [0])

    def test_job_ancestor_usage_can_block_an_otherwise_admissible_worker(self):
        with tempfile.TemporaryDirectory() as root:
            root = Path(root)
            path, proc = root / "memory", root / "proc"
            proc.mkdir()
            group = path / "job_41" / "step_batch"
            group.mkdir(parents=True)
            (proc / "cgroup").write_text("0::/slurm/job_41/step_batch\n")
            (proc / "mountinfo").write_text(f"23 19 0:23 /slurm {path.as_posix()} rw - cgroup2 cgroup rw\n")
            for directory, quota, used in ((group, "max", 2 * runner.GIB),
                                           (group.parent, str(140 * runner.GIB), 121 * runner.GIB),
                                           (path, str(1500 * runner.GIB), 200 * runner.GIB)):
                (directory / "memory.max").write_text(quota)
                (directory / "memory.current").write_text(str(used))
            with patch.dict(os.environ, {"SLURM_JOB_ID": "41"}):
                controls = runner.cgroup_controls(proc)
                self.assertEqual([entry[0] for entry in controls], [group, group.parent, path])
                with patch.object(runner, "proc_kib", side_effect=[120 * runner.GIB, 20 * runner.MIB, 3 * runner.GIB]):
                    snapshot = runner.memory_snapshot(128 * runner.GIB, controls, worker_pid=42)
            self.assertEqual(snapshot["job_headroom_bytes"], 7 * runner.GIB)
            self.assertIn("Job memory", runner.memory_block(snapshot))

    def test_bad_stage_is_rejected_before_operator_load(self):
        with tempfile.TemporaryDirectory() as root:
            with patch("scipy.sparse.load_npz") as load:
                with self.assertRaisesRegex(ValueError, "Saved artifact identity"):
                    runner.load_stage(Path(root))
                load.assert_not_called()


if __name__ == "__main__":
    unittest.main()
