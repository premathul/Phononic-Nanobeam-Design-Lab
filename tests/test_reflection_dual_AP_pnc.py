"""Transpose duals of matched-port reflection. Author: Athul Prem | _AP_pnc."""

from pathlib import Path
import sys
import unittest

import numpy as np
from scipy.sparse import csc_matrix
from scipy.sparse.linalg import splu

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT), str(ROOT / "src")]

from phononic.elasticity_AP_pnc import assemble_elasticity
from phononic.geometry_AP_pnc import BeamGeometry, make_straight_mesh
from phononic.material_AP_pnc import CubicMaterial
from phononic.port_modes_AP_pnc import assemble_port, solve_port_modes
from phononic.scattering_AP_pnc import solve_scattering
from tools.reflection_dual_AP_pnc import recover_reflection_dual


class ReflectionDualTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        frequency = 1e9
        material = CubicMaterial()
        mesh = make_straight_mesh(BeamGeometry(1e-6, 600e-9, 200e-9), (8, 2, 1))
        system = assemble_elasticity(mesh, material, max_dofs=3000)
        port = assemble_port(system, material)
        modes = solve_port_modes(port, frequency)

        def forward(operator, rhs):
            cls.operator, cls.rhs = operator, rhs
            return splu(operator).solve(rhs)

        cls.fields = solve_scattering(system, modes, frequency, linear_solver=forward).displacements
        cls.left, cls.right = port.left_dofs, port.right_dofs
        cls.outgoing = modes.displacements[:, modes.negative_indices]
        cls.positions = np.flatnonzero(modes.propagating[modes.negative_indices])

    def test_native_duals_agree_with_independent_transpose_solve_and_signed_change(self):
        recovered = recover_reflection_dual(self.operator, self.rhs, self.fields,
                                            self.left, self.outgoing, self.positions)
        self.assertTrue(recovered.passed, recovered.metrics)
        independent_rows = np.linalg.inv(self.outgoing)[self.positions]
        target = np.zeros_like(self.rhs)
        target[self.left] = independent_rows.T
        direct_duals = splu(self.operator.T.tocsc()).solve(target)
        relative_error = np.linalg.norm(recovered.duals - direct_duals) / np.linalg.norm(direct_duals)
        self.assertLess(relative_error, 1e-8)

        # Change one interior stiffness without altering either original port.
        interior = np.setdiff1d(np.arange(self.operator.shape[0]), np.r_[self.left, self.right])
        dof = int(interior[len(interior) // 2])
        delta = csc_matrix(([self.operator[dof, dof]], ([dof], [dof])), shape=self.operator.shape)
        eps = 1e-4
        plus = splu(self.operator + eps * delta).solve(self.rhs)
        minus = splu(self.operator - eps * delta).solve(self.rhs)
        difference = independent_rows @ ((plus - minus)[self.left] / (2 * eps))
        derivative = -recovered.duals.T @ (delta @ self.fields)
        scale = np.linalg.norm(difference)
        self.assertGreater(scale, 1e-4)
        self.assertLess(np.linalg.norm(derivative - difference) / scale, 1e-6)

    def test_off_left_transpose_defect_fails_even_when_incident_span_fits(self):
        first, second = self.right[:2]
        delta = csc_matrix(([self.operator[first, first]], ([first], [second])), shape=self.operator.shape)
        asymmetric = self.operator + delta
        fields = splu(asymmetric).solve(self.rhs)
        recovered = recover_reflection_dual(asymmetric, self.rhs, fields,
                                            self.left, self.outgoing, self.positions)
        self.assertFalse(recovered.passed)
        self.assertLess(max(recovered.metrics["source_fit"]["relative_l2"]), 1e-8)
        self.assertGreater(max(recovered.metrics["transpose_residual"]["relative_l2"]), 1e-3)


if __name__ == "__main__":
    unittest.main()
