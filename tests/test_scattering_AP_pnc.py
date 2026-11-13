"""Verification of 3D matched-port scattering. Author: Athul Prem | _AP_pnc."""

from pathlib import Path
import sys
import unittest

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from phononic.elasticity_AP_pnc import assemble_elasticity
from phononic.geometry_AP_pnc import BeamGeometry, make_straight_mesh
from phononic.material_AP_pnc import CubicMaterial
from phononic.port_modes_AP_pnc import assemble_port, solve_port_modes
from phononic.scattering_AP_pnc import solve_scattering


class ScatteringTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.geometry = BeamGeometry(1e-6, 600e-9, 200e-9)
        cls.material = CubicMaterial()
        mesh = make_straight_mesh(cls.geometry, (8, 2, 1))
        cls.system = assemble_elasticity(mesh, cls.material, max_dofs=3000)
        cls.frequency_hz = 1e9
        cls.modes = solve_port_modes(assemble_port(cls.system, cls.material), cls.frequency_hz)

    def test_matched_guide_for_both_directions(self):
        for side in ("left", "right"):
            with self.subTest(side=side):
                result = solve_scattering(self.system, self.modes, self.frequency_hz, side)
                self.assertEqual(len(result.incident_mode_indices), 4)
                self.assertEqual(result.displacements.shape, (self.system.basis.N, 4))
                self.assertGreater(len(result.reflection_all_mode_indices), 4)
                np.testing.assert_array_equal(result.incident_mode_indices, result.transmission_mode_indices)
                np.testing.assert_allclose(result.incident_powers_w, 1.0, rtol=1e-9)
                self.assertLess(np.max(result.reflection), 0.01)
                self.assertLess(np.max(abs(result.transmission - 1)), 0.01)
                np.testing.assert_allclose(result.reflection_flux, result.reflection, atol=1e-8, rtol=0)
                np.testing.assert_allclose(result.transmission_flux, result.transmission, atol=1e-8, rtol=0)
                np.testing.assert_allclose(result.reflection_gram, result.reflection_flux, atol=1e-8, rtol=0)
                np.testing.assert_allclose(result.transmission_gram, result.transmission_flux, atol=1e-8, rtol=0)
                np.testing.assert_allclose(result.flux_power_balance, 0, atol=1e-8, rtol=0)
                np.testing.assert_allclose(result.net_outward_power_ratio, 0, atol=1e-8, rtol=0)
                self.assertLess(np.max(result.relative_residuals), 1e-8)

                # In a uniform guide a mode propagates with exp(i*k*delta_x).
                # The axial tetrahedral discretization introduces a phase error.
                distance = self.geometry.length_m * (1 if side == "left" else -1)
                k = self.modes.wavenumbers_rad_m[result.incident_mode_indices]
                expected = np.diag(np.exp(1j * k * distance))
                np.testing.assert_allclose(result.transmission_amplitudes, expected, atol=0.05, rtol=0)

    def test_rejects_incompatible_frequency_and_port_mapping(self):
        with self.assertRaisesRegex(ValueError, "same frequency"):
            solve_scattering(self.system, self.modes, self.frequency_hz * 2)
        different_mesh = make_straight_mesh(self.geometry, (4, 2, 1))
        different_system = assemble_elasticity(different_mesh, self.material, max_dofs=3000)
        with self.assertRaisesRegex(ValueError, "DOF mapping"):
            solve_scattering(different_system, self.modes, self.frequency_hz)


if __name__ == "__main__":
    unittest.main()
