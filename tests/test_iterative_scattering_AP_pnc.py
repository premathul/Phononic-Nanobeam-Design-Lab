"""Direct and iterative matched-port checks. Author: Athul Prem | _AP_pnc."""

from pathlib import Path
import sys
import unittest
from unittest.mock import patch

import numpy as np
from scipy.sparse import csc_matrix
from scipy.spatial.transform import Rotation

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from phononic.elasticity_AP_pnc import assemble_elasticity
from phononic.geometry_AP_pnc import BeamGeometry, make_straight_mesh
from phononic.linear_solvers_AP_pnc import solve_ilu_gmres
from phononic.material_AP_pnc import CubicMaterial
from phononic.port_modes_AP_pnc import assemble_port, solve_port_modes
from phononic.scattering_AP_pnc import solve_scattering


class IterativeScatteringTests(unittest.TestCase):
    def test_rotated_germanium_both_incidence_sides(self):
        geometry = BeamGeometry(1e-6, 600e-9, 200e-9)
        material = CubicMaterial(rotation=Rotation.from_euler("xyz", (12, 17, -8), degrees=True).as_matrix())
        system = assemble_elasticity(make_straight_mesh(geometry, (8, 2, 1)), material, max_dofs=3000)
        frequency = 1e9
        modes = solve_port_modes(assemble_port(system, material), frequency)
        for side in ("left", "right"):
            with self.subTest(side=side):
                report = {}
                direct = solve_scattering(system, modes, frequency, side)
                iterative = solve_scattering(system, modes, frequency, side,
                                             linear_solver=lambda A, rhs: solve_ilu_gmres(A, rhs, report))
                self.assertEqual(len(iterative.incident_mode_indices), 4)
                for name in ("reflection_amplitudes", "transmission_amplitudes"):
                    np.testing.assert_allclose(getattr(iterative, name), getattr(direct, name), atol=1e-8, rtol=0)
                field_error = np.linalg.norm(iterative.displacements - direct.displacements, axis=0)
                self.assertLess(np.max(field_error / np.linalg.norm(direct.displacements, axis=0)), 1e-8)
                for modal, flux in (("reflection", "reflection_flux"), ("transmission", "transmission_flux")):
                    np.testing.assert_allclose(getattr(iterative, modal), getattr(direct, modal), atol=1e-8, rtol=0)
                    np.testing.assert_allclose(getattr(iterative, modal), getattr(iterative, flux), atol=1e-8, rtol=0)
                np.testing.assert_allclose(iterative.power_balance, 0, atol=1e-8, rtol=0)
                np.testing.assert_allclose(iterative.flux_power_balance, 0, atol=1e-8, rtol=0)
                self.assertLess(np.max(iterative.relative_residuals), 1e-8)
                self.assertEqual(report["method"], "ilu_gmres")
                self.assertGreater(report["preconditioner"]["factor_nonzeros"], 0)
                self.assertEqual(len(report["channels"]), 4)
                for channel in report["channels"]:
                    self.assertEqual(channel["info"], 0)
                    self.assertGreater(channel["iterations"], 0)
                    self.assertEqual(channel["iterations"], len(channel["preconditioned_residual_history"]))
                    self.assertLessEqual(channel["relative_residual"], 1e-9)
                    self.assertLessEqual(channel["relative_backward_residual"], 1e-9)

    def test_rejects_failure_status_and_false_success(self):
        A = csc_matrix(np.eye(3, dtype=complex))
        rhs = np.ones((3, 2), dtype=complex)
        for answer, info in ((rhs[:, 0], 1), (np.zeros(3, dtype=complex), 0)):
            with self.subTest(info=info):
                report = {}
                with patch("phononic.linear_solvers_AP_pnc.gmres", return_value=(answer, info)) as gmres:
                    with self.assertRaisesRegex(RuntimeError, "failed for RHS 1"):
                        solve_ilu_gmres(A, rhs, report)
                self.assertEqual(gmres.call_count, 1)
                self.assertEqual(report["channels"][0]["info"], info)
                self.assertEqual(report["channels"][0]["relative_residual"], 0 if info else 1)

    def test_fill_factor_is_validated_and_passed_to_preconditioner(self):
        A = csc_matrix(np.array([[3 + 1j, 1], [1, 2 - 1j]]))
        rhs = np.array([[1 + 2j, 0], [3 + 4j, 0]])
        for value in (True, np.bool_(True), 0.99, float("inf"), float("nan"), "6"):
            with self.subTest(fill_factor=value):
                with self.assertRaisesRegex(ValueError, "fill_factor"):
                    solve_ilu_gmres(A, rhs, fill_factor=value)
        report = {}
        solution = solve_ilu_gmres(A, rhs, report, fill_factor=6.0)
        np.testing.assert_allclose(A @ solution, rhs, rtol=1e-12, atol=1e-12)
        self.assertEqual(report["preconditioner"]["fill_factor"], 6.0)
        self.assertEqual(report["channels"][1]["relative_residual"], 0)


if __name__ == "__main__":
    unittest.main()
