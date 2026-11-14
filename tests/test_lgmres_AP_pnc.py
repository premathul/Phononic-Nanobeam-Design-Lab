"""LGMRES scattering and stopping checks. Author: Athul Prem | _AP_pnc."""

from pathlib import Path
import sys
import unittest
from unittest.mock import Mock, patch

import numpy as np
from scipy.sparse import csc_matrix
from scipy.sparse.linalg import lgmres, spilu
from scipy.spatial.transform import Rotation

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from phononic.elasticity_AP_pnc import assemble_elasticity
from phononic.geometry_AP_pnc import BeamGeometry, make_straight_mesh
from phononic.linear_lgmres_AP_pnc import solve_ilu_lgmres
from phononic.material_AP_pnc import CubicMaterial
from phononic.port_modes_AP_pnc import assemble_port, solve_port_modes
from phononic.scattering_AP_pnc import solve_scattering


class LgmresScatteringTests(unittest.TestCase):
    def test_rotated_germanium_both_incidence_sides(self):
        mesh = make_straight_mesh(BeamGeometry(1e-6, 600e-9, 200e-9), (4, 2, 1))
        material = CubicMaterial(rotation=Rotation.from_euler("xyz", (12, 17, -8), degrees=True).as_matrix())
        system = assemble_elasticity(mesh, material)
        frequency = 1e9
        modes = solve_port_modes(assemble_port(system, material), frequency)
        for side in ("left", "right"):
            direct = solve_scattering(system, modes, frequency, side)
            for single_precision in (False, True):
                with self.subTest(side=side, single_precision=single_precision):
                    report = {}
                    iterative = solve_scattering(
                        system, modes, frequency, side,
                        linear_solver=lambda A, rhs: solve_ilu_lgmres(A, rhs, report, single_precision=single_precision),
                    )
                    for name in ("reflection_amplitudes", "transmission_amplitudes", "reflection_flux", "transmission_flux"):
                        np.testing.assert_allclose(getattr(iterative, name), getattr(direct, name), atol=1e-8, rtol=0)
                    np.testing.assert_allclose(iterative.flux_power_balance, 0, atol=1e-8)
                    self.assertEqual(report["method"], "ilu_lgmres")
                    self.assertEqual(report["working_dtype"], "complex128")
                    self.assertEqual(report["factor_dtype"], "complex64" if single_precision else "complex128")
                    self.assertEqual(len(report["channels"]), 4)
                    for channel in report["channels"]:
                        self.assertEqual(channel["info"], 0)
                        self.assertEqual(channel["iterations"], len(channel["relative_residual_history"]))
                        self.assertAlmostEqual(channel["relative_residual_history"][0], 1)
                        self.assertLessEqual(channel["relative_residual"], 1e-9)
                        self.assertLessEqual(channel["relative_backward_residual"], 1e-9)

    def test_single_precision_only_changes_ilu_input_and_inverse(self):
        operator = csc_matrix(np.array([[4 + 1.00000001j, 1], [2, 3 - 1j]], dtype=np.complex128))
        rhs = np.array([[2 + 1j], [-1 + 3j]], dtype=np.complex128)
        original = tuple(array.tobytes() for array in (operator.data, operator.indices, operator.indptr, rhs))

        def factor_in_single(A, **settings):
            self.assertEqual(A.dtype, np.dtype(np.complex64))
            factor = spilu(A, **settings)

            def solve_single(vector):
                self.assertEqual(vector.dtype, np.dtype(np.complex64))
                result = factor.solve(vector)
                self.assertEqual(result.dtype, np.dtype(np.complex64))
                return result

            return Mock(nnz=factor.nnz, solve=solve_single)

        def solve_in_double(A, b, **settings):
            self.assertIs(A, operator)
            self.assertEqual(b.dtype, np.dtype(np.complex128))
            self.assertEqual(settings["M"].matvec(b).dtype, np.dtype(np.complex128))
            return lgmres(A, b, **settings)

        with (patch("phononic.linear_lgmres_AP_pnc.spilu", side_effect=factor_in_single) as factorize,
              patch("phononic.linear_lgmres_AP_pnc.lgmres", side_effect=solve_in_double)):
            solution = solve_ilu_lgmres(operator, rhs, single_precision=True)
        factorize.assert_called_once()
        self.assertEqual(solution.dtype, np.dtype(np.complex128))
        np.testing.assert_allclose(operator @ solution, rhs, rtol=1e-10)
        self.assertEqual(tuple(array.tobytes() for array in (operator.data, operator.indices, operator.indptr, rhs)), original)

    def test_terminal_update_requires_unchanged_official_confirmation(self):
        operator = csc_matrix(np.eye(3, dtype=complex))
        rhs = np.ones((3, 2), dtype=complex)

        def last_update(A, b, **settings):
            if settings["maxiter"] == 1:
                self.assertIsNone(settings["M"])
                np.testing.assert_array_equal(settings["x0"], b)
                return lgmres(A, b, **settings)
            self.assertIsNotNone(settings["M"])
            settings["callback"](np.zeros_like(b))
            return b.copy(), settings["maxiter"]

        report = {}
        with (patch("phononic.linear_lgmres_AP_pnc.lgmres", side_effect=last_update) as solve,
              patch("phononic.linear_lgmres_AP_pnc.spilu", wraps=spilu) as factor):
            result = solve_ilu_lgmres(operator, rhs, report, fill_factor=6)
        np.testing.assert_array_equal(result, rhs)
        factor.assert_called_once()
        self.assertEqual(report["preconditioner"]["fill_factor"], 6)
        self.assertEqual(solve.call_count, 4)
        for channel in report["channels"]:
            self.assertEqual(channel["raw_info"], 30)
            self.assertEqual(channel["info"], 0)
            self.assertEqual(channel["relative_residual_history"], [1])
            self.assertEqual(channel["relative_residual"], 0)
            self.assertEqual(channel["terminal_confirmation"], {"info": 0, "callbacks": 1, "accepted": True})

    def test_rejects_bad_residual_even_with_success_status(self):
        operator = csc_matrix(np.eye(3, dtype=complex))
        rhs = np.ones((3, 2), dtype=complex)
        for answer, status in ((np.zeros(3), 0), (rhs[:, 0] * (1 + 5e-10), 30), (rhs[:, 0], 1)):
            with self.subTest(status=status):
                report = {}
                with (patch("phononic.linear_lgmres_AP_pnc.lgmres", return_value=(answer, status)) as solve,
                      self.assertRaisesRegex(RuntimeError, "failed for RHS 1")):
                    solve_ilu_lgmres(operator, rhs, report, single_precision=True)
                solve.assert_called_once()
                channel, = report["channels"]
                self.assertEqual(channel["raw_info"], status)
                self.assertIsNone(channel["terminal_confirmation"])
                np.testing.assert_allclose(channel["relative_residual"], abs(answer[0] - 1), rtol=1e-12)

    def test_rejects_confirmation_that_returns_a_changed_iterate(self):
        operator = csc_matrix(np.eye(3, dtype=complex))
        rhs = np.ones((3, 1), dtype=complex)

        def wrong_confirmation(_A, b, **settings):
            if settings["maxiter"] == 1:
                settings["callback"](settings["x0"])
                return b * (1 + 1e-12), 0
            return b.copy(), settings["maxiter"]

        report = {}
        with (patch("phononic.linear_lgmres_AP_pnc.lgmres", side_effect=wrong_confirmation),
              self.assertRaisesRegex(RuntimeError, "failed for RHS 1")):
            solve_ilu_lgmres(operator, rhs, report)
        channel, = report["channels"]
        self.assertEqual(channel["info"], 30)
        self.assertEqual(channel["terminal_confirmation"], {"info": 0, "callbacks": 1, "accepted": False})

    def test_interrupted_matched_port_response_resumes_only_unfinished_channels(self):
        mesh = make_straight_mesh(BeamGeometry(1e-6, 600e-9, 200e-9), (4, 2, 1))
        material = CubicMaterial(rotation=Rotation.from_euler("xyz", (12, 17, -8), degrees=True).as_matrix())
        system = assemble_elasticity(mesh, material)
        frequency = 1e9
        modes = solve_port_modes(assemble_port(system, material), frequency)
        settings = {"inner_m": 4, "outer_k": 2, "single_precision": True}

        def run(report, **kwargs):
            return solve_scattering(
                system, modes, frequency, "left",
                linear_solver=lambda A, rhs: solve_ilu_lgmres(A, rhs, report, **settings, **kwargs),
            )

        baseline = run({})
        accepted = {}
        active = {}

        class Interrupted(Exception):
            pass

        def interrupt(event, column, vector, channel):
            if event == "accepted":
                accepted[column] = vector
            if event == "iterate" and column == 1 and channel["iterations"] == 2:
                self.assertGreater(channel["relative_residual_history"][-1], 1e-9)
                self.assertGreater(np.linalg.norm(vector), 0)
                active[column] = vector
                raise Interrupted

        interrupted_report = {}
        with self.assertRaises(Interrupted):
            run(interrupted_report, checkpoint_callback=interrupt)
        self.assertEqual(list(accepted), [0])
        self.assertEqual(list(active), [1])
        self.assertIsNone(interrupted_report["channels"][1]["info"])
        active_bytes = active[1].tobytes()
        resume_report = {}
        events = []

        def checkpoint(event, column, vector, channel):
            if event == "iterate" and channel["iterations"] == 1:
                expected = active[column] if column in active else np.zeros_like(vector)
                np.testing.assert_array_equal(vector, expected)
            events.append((event, column))

        with patch("phononic.linear_lgmres_AP_pnc.lgmres", wraps=lgmres) as solve:
            resumed = run(resume_report, accepted_solutions=accepted, initial_guesses=active,
                          checkpoint_callback=checkpoint)
        self.assertEqual(solve.call_count, 3)
        self.assertEqual(active[1].tobytes(), active_bytes)
        self.assertEqual(events[-1], ("accepted", 3))
        self.assertNotIn(0, [column for _, column in events])
        reused = resume_report["channels"][0]
        self.assertTrue(reused["reused"])
        self.assertEqual(reused["iterations"], 0)
        self.assertTrue(resume_report["channels"][1]["warm_started"])
        for channel in resume_report["channels"]:
            self.assertEqual(channel["info"], 0)
            self.assertLessEqual(channel["relative_residual"], 1e-9)
            self.assertLessEqual(channel["relative_backward_residual"], 1e-9)
        for name in ("reflection_amplitudes", "transmission_amplitudes"):
            np.testing.assert_allclose(getattr(resumed, name), getattr(baseline, name), atol=1e-8, rtol=0)
        np.testing.assert_allclose(resumed.power_balance, 0, atol=1e-8, rtol=0)
        np.testing.assert_allclose(resumed.flux_power_balance, 0, atol=1e-8, rtol=0)

    def test_reused_vectors_are_verified_before_factorization(self):
        operator = csc_matrix(np.eye(3, dtype=complex))
        rhs = np.column_stack((np.ones(3), np.arange(3))).astype(complex)
        report = {}
        accepted = {1: rhs[:, 1], 0: rhs[:, 0]}
        with (patch("phononic.linear_lgmres_AP_pnc.spilu") as factor,
              patch("phononic.linear_lgmres_AP_pnc.lgmres") as solve):
            solution = solve_ilu_lgmres(operator, rhs, report, accepted_solutions=accepted)
            np.testing.assert_array_equal(solution, rhs)
            self.assertTrue(report["preconditioner"]["skipped"])
            self.assertEqual([channel["column"] for channel in report["channels"]], [0, 1])
            self.assertTrue(all(channel["reused"] and channel["iterations"] == 0 for channel in report["channels"]))
            for vector, message in ((rhs[:, 0], "original residual gates"),
                                    (np.full(3, np.nan), "finite vector"),
                                    (np.zeros((3, 1)), "finite vector")):
                with self.subTest(message=message), self.assertRaisesRegex(ValueError, message):
                    solve_ilu_lgmres(operator, rhs, accepted_solutions={1: vector})
            with self.assertRaisesRegex(ValueError, "finite vector"):
                solve_ilu_lgmres(operator, rhs, initial_guesses={1: np.full(3, np.inf)})
            with self.assertRaisesRegex(ValueError, "invalid RHS column"):
                solve_ilu_lgmres(operator, rhs, accepted_solutions={-1: rhs[:, 0]})
            with self.assertRaisesRegex(ValueError, "already accepted"):
                solve_ilu_lgmres(operator, rhs, accepted_solutions={0: rhs[:, 0]}, initial_guesses={0: rhs[:, 0]})
        factor.assert_not_called()
        solve.assert_not_called()

    def test_numerical_failure_checkpoints_latest_finite_vector_without_acceptance(self):
        operator = csc_matrix(np.eye(3, dtype=complex))
        rhs = np.ones((3, 1), dtype=complex)
        for finite_terminal in (True, False):
            with self.subTest(finite_terminal=finite_terminal):
                events = []

                def fail(_A, b, **settings):
                    iterate = np.zeros_like(b)
                    settings["callback"](iterate)
                    iterate[:] = b / 4
                    settings["callback"](iterate)
                    iterate[:] = b / 2 if finite_terminal else np.nan
                    return iterate, settings["maxiter"]

                report = {}
                with (patch("phononic.linear_lgmres_AP_pnc.lgmres", side_effect=fail),
                      self.assertRaisesRegex(RuntimeError, "failed for RHS 1")):
                    solve_ilu_lgmres(operator, rhs, report, maxiter=2,
                                     checkpoint_callback=lambda *args: events.append(args))
                self.assertEqual([event for event, *_ in events], ["iterate", "iterate", "terminal"])
                np.testing.assert_array_equal(events[0][2], 0)
                np.testing.assert_array_equal(events[1][2], 0.25)
                expected = 0.5 if finite_terminal else 0.25
                np.testing.assert_array_equal(events[2][2], expected)
                self.assertEqual(events[0][3]["relative_residual_history"], [1])
                self.assertEqual(events[2][3]["info"], 2)
                self.assertAlmostEqual(events[2][3]["relative_residual"], 1 - expected)

    def test_callback_failure_after_numerical_convergence_propagates(self):
        operator = csc_matrix(np.eye(3, dtype=complex))
        rhs = np.ones((3, 1), dtype=complex)
        report = {}
        events = []

        def checkpoint(event, *_):
            events.append(event)
            if event == "accepted":
                raise OSError("checkpoint write failed")

        with self.assertRaisesRegex(OSError, "checkpoint write failed"):
            solve_ilu_lgmres(operator, rhs, report, checkpoint_callback=checkpoint)
        self.assertEqual(events, ["iterate", "iterate", "accepted"])
        self.assertEqual(report["channels"][0]["raw_info"], 0)
        self.assertIsNone(report["channels"][0]["info"])


if __name__ == "__main__":
    unittest.main()
