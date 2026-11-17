"""Independent equation and signed work checks. Athul Prem | _AP_pnc."""

import unittest
from unittest.mock import patch

import numpy as np
from scipy.sparse import csc_matrix

from tools import replay_lawrence_device_point_AP_pnc as replay


class DevicePointReplayTests(unittest.TestCase):
    def test_original_complex_residual_and_independent_infinity_norm(self):
        dense = np.array([[2 + 1j, -1j], [3j, 4]])
        operator = csc_matrix(dense)
        vector = np.array([[1 + 2j], [3 - 1j]])
        rhs = dense @ vector
        relative, backward, _, norm = replay.check_original_equations(operator, rhs, vector)
        self.assertAlmostEqual(norm, 7.0)
        self.assertLess(relative[0], 1e-15)
        self.assertLess(backward[0], 1e-15)
        with self.assertRaisesRegex(ValueError, "Original equation replay failed"):
            replay.check_original_equations(operator, rhs, vector * (1 + 2e-10))

    def test_traction_signs_independent_flux_and_original_rhs_support(self):
        modes = {"positive_indices": np.array([0]), "negative_indices": np.array([1]),
                 "propagating": np.array([True, True]), "wavenumbers_rad_m": np.array([3, -3]),
                 "displacements": np.array([[1, 1]]), "axial_forces": np.array([[1j, -1j]])}
        fields = {"incident_powers_w": np.array([1.0])}
        for name, root, coeff in (("reflection", 1, 0.6), ("transmission", 0, 0.8j)):
            fields[f"{name}_all_mode_indices"] = np.array([root])
            fields[f"{name}_amplitudes"] = fields[f"{name}_amplitudes_full"] = np.array([[coeff]])
        left, right = np.array([0]), np.array([2])
        vectors = np.array([[1.6], [0.4], [0.8j]])
        force = np.array([[-0.4j], [0.0], [-0.8]])
        rhs = np.array([[-2j], [0.0], [0.0]])
        with patch.multiple(replay, INCIDENT=(0,), RETURNING=(1,), OMEGA=2.0):
            result = replay.replay_ports(modes, fields, left, right, vectors, force, rhs, np.array([2.0]))
            np.testing.assert_allclose(np.asarray(result[1]).ravel(), [0.36, 0.64, 0.36, 0.64, 0.0], atol=1e-15)
            bad_rhs = rhs.copy()
            bad_rhs[1] = 1e-12
            with self.assertRaisesRegex(ValueError, "exact left support"):
                replay.replay_ports(modes, fields, left, right, vectors, force, bad_rhs, np.array([2.0]))
            force[1] = 0.1j
            with self.assertRaisesRegex(ValueError, "boundary forces fail to balance"):
                replay.replay_ports(modes, fields, left, right, vectors, force, rhs, np.array([2.0]))

    def test_modal_balance_does_not_mask_bad_physical_flux(self):
        powers = [np.array([value]) for value in (1.0, 0.36, 0.64, 0.34, 0.66, 0.0)]
        with self.assertRaisesRegex(ValueError, "physical-power checks failed"):
            replay.check_power(*powers)


if __name__ == "__main__":
    unittest.main()
