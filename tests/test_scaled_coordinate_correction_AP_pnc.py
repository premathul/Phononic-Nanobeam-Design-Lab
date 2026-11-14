"""Verify the scaled correction against dense equations. Author: Athul Prem | _AP_pnc."""

import unittest
from unittest.mock import patch
from weakref import ref

import numpy as np
from scipy.sparse import csc_matrix
from scipy.sparse.linalg import gcrotmk, spilu

from tools import scaled_coordinate_correction_AP_pnc as correction


class ScaledCoordinateCorrectionTests(unittest.TestCase):
    def setUp(self):
        self.dense = np.array([[7 + 2j, 19 - 8j, 0, -2], [-1 + 4j, 4, 2j, 0],
                               [0.5j, -1, 3 + 1j, 6], [0, 1 + 4j, -3j, 7]], dtype=np.complex128)
        self.A = csc_matrix(self.dense)
        self.b = np.array([1.2 - 0.7j, 3 + 0.4j, -2.3j, 2.7 + 0.8j], dtype=np.complex128)
        self.x0 = np.array([0.125 + 0.25j, -0.4j, -0.7, -0.25j], dtype=np.complex128)

    def test_dense_operator_factor_and_reconstruction_preserve_original_equations(self):
        row = np.abs(self.dense).max(axis=1) ** -0.5
        column = np.abs(self.dense).max(axis=0) ** -0.5
        self.assertFalse(np.allclose(row, column))
        expected_B = np.diag(row) @ self.dense @ np.diag(column)
        expected_x = np.linalg.solve(self.dense, self.b)
        r0 = self.b - self.dense @ self.x0
        rho = np.linalg.norm(row * r0)
        expected_q = row * r0 / rho
        expected_rtol = 1e-10 * np.linalg.norm(self.b) * row.min() / rho
        original_arrays = (self.A.data, self.A.indices, self.A.indptr, self.b, self.x0)
        original_bytes = [array.tobytes() for array in original_arrays]
        for array in original_arrays:
            array.flags.writeable = False
        factor_input_ref = None

        def check_factor_input(temporary_B, **settings):
            nonlocal factor_input_ref
            factor_input_ref = ref(temporary_B)
            self.assertEqual(temporary_B.format, "csc")
            self.assertEqual(temporary_B.dtype, np.dtype("complex128"))
            np.testing.assert_allclose(temporary_B.toarray(), expected_B, rtol=1e-14, atol=0)
            self.assertEqual(settings, {"fill_factor": 4.0, "drop_tol": 1e-4, "permc_spec": "COLAMD"})
            return spilu(temporary_B, **settings)

        def check_gcrot(B, q, **settings):
            self.assertIsNone(factor_input_ref())
            np.testing.assert_allclose(q, expected_q, rtol=1e-14, atol=0)
            np.testing.assert_allclose(expected_rtol, settings["rtol"], rtol=1e-14, atol=0)
            vector = np.array([1 + 2j, -3j, -2, 0.5 - 1j])
            np.testing.assert_allclose(B @ vector, expected_B @ vector, rtol=1e-14, atol=1e-14)
            # On this tiny system the incomplete factor solves the scaled equations to roundoff.
            np.testing.assert_allclose(settings["M"] @ vector, np.linalg.solve(expected_B, vector),
                                       rtol=1e-14, atol=1e-14)
            return gcrotmk(B, q, **settings)

        with (patch.object(correction, "spilu", new=check_factor_input),
              patch.object(correction, "gcrotmk", new=check_gcrot)):
            report, candidates = correction.solve_correction(self.A, self.b, self.x0, batch_entries=3)
        self.assertTrue(report["accepted"])
        self.assertEqual(report["solver"]["raw_info"], 0)
        candidate = candidates["terminal"]
        np.testing.assert_allclose(candidate, expected_x, rtol=1e-10, atol=1e-13)
        residual = self.b - self.dense @ candidate
        expected_relative = np.linalg.norm(residual) / np.linalg.norm(self.b)
        expected_backward = np.linalg.norm(residual, np.inf) / (
            np.linalg.norm(self.dense, np.inf) * np.linalg.norm(candidate, np.inf) + np.linalg.norm(self.b, np.inf))
        self.assertAlmostEqual(report["terminal"]["relative_residual"], expected_relative, delta=1e-14)
        self.assertAlmostEqual(report["terminal"]["relative_backward_residual"], expected_backward, delta=1e-14)
        self.assertEqual(report["candidate_storage_bytes"], (len(candidates["callbacks"]) + 1) * candidate.nbytes)
        np.testing.assert_array_equal(candidates["callbacks"][0], self.x0)
        self.assertEqual([array.tobytes() for array in original_arrays], original_bytes)

    def test_zero_transformed_residual_cannot_override_original_checks(self):
        A = csc_matrix(np.eye(2, dtype=np.complex128))
        seed = np.array([2**60, 2j], dtype=np.complex128)
        # Seed subtraction loses b[0] in floating point. The transformed solve can
        # be exact while the recovered original candidate still misses b[0].
        for lost_entry, relative_gate, backward_gate in ((1, False, False), (5e-10, True, True)):
            with self.subTest(lost_entry=lost_entry):
                b = np.array([lost_entry, 2j], dtype=np.complex128)
                # Exact Krylov breakdown can produce a 0/0 when forming a new basis vector.
                with np.errstate(invalid="ignore"):
                    report, candidates = correction.solve_correction(A, b, seed)
                terminal = report["terminal"]
                self.assertEqual(report["solver"]["raw_info"], 0)
                self.assertEqual(terminal["normalized_transformed_relative_residual"], 0)
                self.assertTrue(terminal["finite"])
                self.assertFalse(terminal["relative_target_passed"])
                self.assertEqual(terminal["relative_gate_passed"], relative_gate)
                self.assertEqual(terminal["backward_gate_passed"], backward_gate)
                self.assertFalse(report["accepted"])
                np.testing.assert_array_equal(candidates["terminal"], np.array([0, 2j]))

    def test_one_outer_update_truncated_before_tolerance_remains_unaccepted(self):
        n = 96
        rows = np.arange(n)
        columns = np.concatenate([(rows + shift) % n for shift in (0, 1, 11, 19, 37)])
        rng = np.random.default_rng(114)
        data = rng.standard_normal(4 * n) + 1j * rng.standard_normal(4 * n)
        A = csc_matrix((np.r_[np.full(n, 2.0), data], (np.tile(rows, 5), columns)), shape=(n, n))
        b = rng.standard_normal(n) + 1j * rng.standard_normal(n)
        x0 = np.zeros_like(b)
        report, candidates = correction.solve_correction(A, b, x0, maxiter=1)
        self.assertEqual(report["solver"]["raw_info"], 1)
        self.assertTrue(report["solver"]["returned_update_after_last_callback"])
        self.assertIsNone(report["solver"]["terminal_confirmation"])
        self.assertFalse(report["accepted"])
        self.assertGreater(report["terminal"]["relative_residual"], correction.ORIGINAL_LIMIT)
        self.assertIsNotNone(candidates["terminal"])
        self.assertEqual(len(candidates["callbacks"]), 1)

    def test_successful_last_update_requires_unchanged_original_confirmation(self):
        for changed in (False, True):
            with self.subTest(changed=changed):
                def check_confirmation(A, b, **settings):
                    if A is not self.A:
                        return gcrotmk(A, b, **settings)
                    self.assertIs(b, self.b)
                    self.assertIsNone(settings.get("M"))
                    self.assertEqual(settings["rtol"], correction.RELATIVE_TARGET)
                    candidate = settings["x0"].copy()
                    confirmed, info = gcrotmk(A, b, **settings)
                    np.testing.assert_array_equal(confirmed, candidate)
                    if changed:
                        confirmed[0] += 1e-14
                    return confirmed, info

                with patch.object(correction, "gcrotmk", side_effect=check_confirmation) as solve:
                    report, _ = correction.solve_correction(self.A, self.b, self.x0, maxiter=1)
                self.assertEqual(solve.call_count, 2)
                solver = report["solver"]
                self.assertEqual(solver["raw_info"], 1)
                self.assertTrue(solver["returned_update_after_last_callback"])
                self.assertEqual(solver["terminal_confirmation"], {
                    "info": 0, "callbacks": 1, "unchanged_start": True, "accepted": not changed,
                })
                self.assertEqual(report["accepted"], not changed)


if __name__ == "__main__":
    unittest.main()
