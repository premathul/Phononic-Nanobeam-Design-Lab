"""Check batched elastic assembly against the full basis. Author: Athul Prem | _AP_pnc."""

from pathlib import Path
import sys
import unittest

import numpy as np
from scipy.sparse import isspmatrix_csr
from scipy.spatial.transform import Rotation

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from phononic.assembly_batched_AP_pnc import assemble_elasticity_batched
from phononic.elasticity_AP_pnc import assemble_elasticity, solve_modes
from phononic.geometry_AP_pnc import BeamGeometry, make_straight_mesh
from phononic.material_AP_pnc import CubicMaterial
from phononic.port_modes_AP_pnc import assemble_port, solve_port_modes
from phononic.scattering_AP_pnc import solve_scattering


class BatchedAssemblyTests(unittest.TestCase):
    def setUp(self):
        self.mesh = make_straight_mesh(BeamGeometry(1e-6, 600e-9, 200e-9), (4, 2, 1))
        self.material = CubicMaterial(rotation=Rotation.from_rotvec([0.17, -0.38, 0.21]).as_matrix())

    def test_operators_layout_and_matched_response(self):
        for order in (1, 2):
            with self.subTest(order=order):
                full = assemble_elasticity(self.mesh, self.material, order)
                batched = assemble_elasticity_batched(self.mesh, self.material, order, batch_elements=11)
                for expected, actual in ((full.K, batched.K), (full.M, batched.M)):
                    self.assertTrue(isspmatrix_csr(actual))
                    scale = np.max(abs(expected.data))
                    np.testing.assert_allclose(actual.toarray() / scale, expected.toarray() / scale,
                                               rtol=1e-13, atol=1e-14)
                for name in ("doflocs", "element_dofs", "nodal_dofs"):
                    np.testing.assert_array_equal(getattr(batched.basis, name), getattr(full.basis, name))
                self.assertFalse(batched.basis.basis)
                with self.assertRaisesRegex(NotImplementedError, "no evaluated cell fields"):
                    batched.basis.interpolate(np.zeros(batched.basis.N))

                full_port = assemble_port(full, self.material)
                batched_port = assemble_port(batched, self.material)
                for name in ("left_dofs", "right_dofs"):
                    np.testing.assert_array_equal(getattr(batched_port, name), getattr(full_port, name))
                frequency = 1e9
                full_modes = solve_port_modes(full_port, frequency)
                batched_modes = solve_port_modes(batched_port, frequency)
                for side in ("left", "right"):
                    expected = solve_scattering(full, full_modes, frequency, side)
                    actual = solve_scattering(batched, batched_modes, frequency, side)
                    np.testing.assert_array_equal(actual.incident_mode_indices, expected.incident_mode_indices)
                    for name in ("reflection_amplitudes", "transmission_amplitudes"):
                        np.testing.assert_allclose(getattr(actual, name), getattr(expected, name), atol=1e-10, rtol=0)
                    self.assertLess(max(abs(actual.flux_power_balance)), 1e-8)

                expected = solve_modes(full, "clamped_free", num_modes=3)
                actual = solve_modes(batched, "clamped_free", num_modes=3)
                np.testing.assert_array_equal(actual.free_dofs, expected.free_dofs)
                np.testing.assert_allclose(actual.frequencies_hz, expected.frequencies_hz, rtol=1e-10)

    def test_rejects_invalid_batch_size_and_dof_cap(self):
        for batch_elements in (0, -1, 1.5, True):
            with self.subTest(batch_elements=batch_elements), self.assertRaisesRegex(ValueError, "batch_elements"):
                assemble_elasticity_batched(self.mesh, self.material, batch_elements=batch_elements)
        with self.assertRaisesRegex(ValueError, "exceeding cap 1"):
            assemble_elasticity_batched(self.mesh, self.material, max_dofs=1)


if __name__ == "__main__":
    unittest.main()
