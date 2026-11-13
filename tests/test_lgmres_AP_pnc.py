"""LGMRES scattering and stopping checks. Author: Athul Prem | _AP_pnc."""

from pathlib import Path
import sys
import unittest
from unittest.mock import patch

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
            with self.subTest(side=side):
                report = {}
                direct = solve_scattering(system, modes, frequency, side)
                iterative = solve_scattering(system, modes, frequency, side,
                                             linear_solver=lambda A, rhs: solve_ilu_lgmres(A, rhs, report))
                for name in ("reflection_amplitudes", "transmission_amplitudes", "reflection_flux", "transmission_flux"):
                    np.testing.assert_allclose(getattr(iterative, name), getattr(direct, name), atol=1e-8, rtol=0)
                np.testing.assert_allclose(iterative.flux_power_balance, 0, atol=1e-8)
                self.assertEqual(report["method"], "ilu_lgmres")
                self.assertEqual(len(report["channels"]), 4)
                for channel in report["channels"]:
                    self.assertEqual(channel["info"], 0)
                    self.assertEqual(channel["iterations"], len(channel["relative_residual_history"]))
                    self.assertAlmostEqual(channel["relative_residual_history"][0], 1)
                    self.assertLessEqual(channel["relative_residual"], 1e-9)
                    self.assertLessEqual(channel["relative_backward_residual"], 1e-9)

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
                    solve_ilu_lgmres(operator, rhs, report)
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


if __name__ == "__main__":
    unittest.main()
