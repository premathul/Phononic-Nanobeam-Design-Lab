"""Direct uniform-guide solve checks. Author: Athul Prem | _AP_pnc."""

from pathlib import Path
from tempfile import TemporaryDirectory
import unittest
from unittest.mock import patch

import numpy as np
from scipy.sparse import csc_matrix

from tools import run_lawrence_uniform_h80_AP_pnc as runner


class LawrenceUniformTests(unittest.TestCase):
    def test_five_direct_drives_match_dense_reference_and_save_checked_vectors(self):
        dense = np.array([[3 + 1j, 2j, 1], [1 - 2j, 4, 0], [2, 1j, 5 - 1j]])
        matrix = csc_matrix(dense)
        random = np.random.default_rng(8721)
        rhs = random.normal(size=(3, 5)) + 1j * random.normal(size=(3, 5))
        original_rhs = rhs.copy()
        roots = [1348, 1311, 1292, 1284, 1274]
        with TemporaryDirectory() as directory, patch.object(runner, "direct_factor_admission", return_value={"admitted": True}):
            output = Path(directory)
            result = runner.solve_direct_original(matrix, rhs, roots, output, {})
            np.testing.assert_allclose(result, np.linalg.solve(dense, original_rhs), rtol=1e-14, atol=1e-14)
            np.testing.assert_array_equal(matrix.toarray(), dense)
            np.testing.assert_array_equal(rhs, original_rhs)
            receipt = runner.read_json(output / "direct_channels_AP_pnc.json")
            self.assertEqual(receipt["status"], "drives_accepted")
            channels = receipt["solver_report"]["channels"]
            self.assertEqual([row["incident_port_root"] for row in channels], roots)
            for column, row in enumerate(channels):
                saved = output / row["accepted_vector"]["file"]
                self.assertEqual(runner.artifact(saved)["sha256"], row["accepted_vector"]["sha256"])
                np.testing.assert_array_equal(np.load(saved, allow_pickle=False), result[:, column])

    def test_relative_residual_failure_retains_the_first_drive(self):
        # The first drive acts in an independent coordinate. The second loses
        # precision when two nearly equal equations cancel in the solve.
        matrix = csc_matrix([[1, 1, 0], [1, 1 + 1e-13, 0], [0, 0, 1]], dtype=np.complex128)
        rhs = np.array([[0, 0.3], [0, 0.7], [1 - 2j, 0]], dtype=np.complex128)
        with TemporaryDirectory() as directory, patch.object(runner, "direct_factor_admission", return_value={"admitted": True}):
            output = Path(directory)
            with self.assertRaisesRegex(RuntimeError, "Direct drive 2 failed"):
                runner.solve_direct_original(matrix, rhs, [1348, 1311], output, {})
            receipt = runner.read_json(output / "direct_channels_AP_pnc.json")
            self.assertEqual(receipt["status"], "failed")
            first, second = receipt["solver_report"]["channels"]
            self.assertGreater(second["relative_residual"], 1e-9)
            self.assertLessEqual(second["relative_backward_residual"], 1e-9)
            files = list(output.glob("*.npy"))
            self.assertEqual([path.name for path in files], [first["accepted_vector"]["file"]])
            np.testing.assert_array_equal(np.load(files[0], allow_pickle=False), rhs[:, 0])

    def test_unsafe_admission_precedes_factorization(self):
        matrix = csc_matrix([[1, 0], [0, 0]], dtype=np.complex128)
        rhs = np.ones((2, 1), dtype=np.complex128)
        with TemporaryDirectory() as directory, patch.object(runner, "direct_factor_admission", return_value={"admitted": False}):
            output = Path(directory)
            with self.assertRaisesRegex(RuntimeError, "does not fit the current direct factorization budget"):
                runner.solve_direct_original(matrix, rhs, [1348], output, {})
            receipt = runner.read_json(output / "direct_channels_AP_pnc.json")
            self.assertEqual(receipt["status"], "failed")
            self.assertNotIn("factor", receipt["solver_report"])
            self.assertEqual(list(output.glob("*.npy")), [])
