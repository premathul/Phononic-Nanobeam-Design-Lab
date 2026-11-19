"""Check the complex snapshot response against a closed-form case. Athul Prem | _AP_pnc."""

import unittest

import numpy as np
from scipy.sparse import csc_matrix

from tools.snapshot_response_AP_pnc import solve_snapshot_response


class SnapshotResponseTests(unittest.TestCase):
    def setUp(self):
        self.operator = np.array([[2 + 1j, 1 - 2j, 0], [3j, 4, 2], [-1, 1j, 3 - 2j]])
        self.action = csc_matrix(self.operator).__matmul__
        self.q = np.array([1, 1j, 0])
        self.w = np.array([0, 0, 1])

    def test_recovers_multiple_in_span_drives_with_dependent_and_zero_snapshots(self):
        snapshots = np.column_stack((self.q, 2j * self.q, self.w, np.zeros(3)))
        expected = np.column_stack(((1 + 1j) * self.q - self.w, 2 * self.w, np.zeros(3)))
        rhs = self.operator @ expected
        result = solve_snapshot_response(snapshots, self.action, rhs)
        self.assertEqual(result.rank, 2)
        self.assertEqual(result.action_rank, 2)
        orthonormal = np.column_stack((self.q / np.sqrt(2), self.w))
        reduced = orthonormal.conj().T @ self.operator @ orthonormal
        self.assertAlmostEqual(result.condition, np.linalg.cond(reduced), delta=1e-14)
        np.testing.assert_allclose(result.solution, expected, rtol=0, atol=1e-14)
        np.testing.assert_allclose(result.relative_residuals, 0, rtol=0, atol=1e-14)
        np.testing.assert_allclose(result.minimum_relative_residuals, 0, rtol=0, atol=1e-14)
        self.assertEqual(result.relative_residuals[2], 0)

    def test_residual_outside_retained_span_and_minimum_are_distinct(self):
        snapshots = np.column_stack((self.q, 1e-10 * self.w))
        image = self.operator @ self.q
        coefficient = 1.5 + 0.25j
        orthogonal_part = (0.5 - 0.25j) * self.w
        rhs = image * coefficient + orthogonal_part
        result = solve_snapshot_response(snapshots, self.action, rhs, rank_rtol=1e-8)
        self.assertEqual(result.rank, 1)
        self.assertEqual(result.action_rank, 1)
        self.assertAlmostEqual(result.condition, 1.0, delta=1e-15)
        np.testing.assert_allclose(result.solution, coefficient * self.q, rtol=0, atol=1e-14)

        # The Galerkin test vector is q, while a least-squares fit is the
        # Euclidean projection of the RHS onto its one-dimensional action image.
        expected_galerkin = np.linalg.norm(orthogonal_part) / np.linalg.norm(rhs)
        best_fit = image * np.vdot(image, rhs) / np.vdot(image, image)
        expected_minimum = np.linalg.norm(best_fit - rhs) / np.linalg.norm(rhs)
        self.assertAlmostEqual(result.relative_residuals[0], expected_galerkin, delta=1e-14)
        self.assertAlmostEqual(result.minimum_relative_residuals[0], expected_minimum, delta=1e-14)
        self.assertLess(result.minimum_relative_residuals[0], result.relative_residuals[0])

    def test_rejects_an_entirely_empty_span(self):
        with self.assertRaisesRegex(ValueError, "zero numerical rank"):
            solve_snapshot_response(np.zeros((3, 2)), self.action, np.ones(3))

    def test_rejects_singular_galerkin_projection(self):
        operator = np.array([[0, 1], [1, 0]])
        with self.assertRaises(np.linalg.LinAlgError):
            solve_snapshot_response(np.array([[1.0], [0.0]]), operator.__matmul__, np.ones(2))

    def test_reports_least_squares_rank_loss_even_if_galerkin_solve_succeeds(self):
        operator = np.diag([1.0, 1e-20])
        result = solve_snapshot_response(np.eye(2), operator.__matmul__, np.array([0.0, 1.0]))
        self.assertEqual(result.rank, 2)
        self.assertEqual(result.action_rank, 1)
        self.assertAlmostEqual(result.relative_residuals[0], 0.0, delta=1e-14)
        self.assertAlmostEqual(result.minimum_relative_residuals[0], 1.0, delta=1e-14)


if __name__ == "__main__":
    unittest.main()
