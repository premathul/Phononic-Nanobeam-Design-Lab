"""Observable-specific finite qualification. Athul Prem | _AP_pnc."""

from pathlib import Path
import sys
from types import SimpleNamespace
import unittest

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from phononic.scattering_qualification_AP_pnc import qualify_scattering


class QualificationTests(unittest.TestCase):
    def case(self, *, side="left", transmission=None):
        gram = np.diag([1., 1., -1., -1.])
        wave = np.array([1., 2., -1., -2.])
        modes = SimpleNamespace(wavenumbers_rad_m=wave,
                                flux_gram=lambda indices: gram[np.ix_(indices, indices)])
        incoming = np.array([0, 1] if side == "left" else [2, 3])
        returning = np.array([2, 3] if side == "left" else [0, 1])
        phase = np.exp(1j * wave[incoming] * (1 if side == "left" else -1))
        result = SimpleNamespace(incident_side=side, incident_mode_indices=incoming,
                                 reflection_mode_indices=returning, transmission_mode_indices=incoming,
                                 incident_powers_w=np.ones(2), reflection_amplitudes=np.zeros((2, 2), complex),
                                 transmission_amplitudes=np.diag(phase) if transmission is None else transmission)
        return result, modes

    def test_unit_column_budgets_do_not_certify_coherent_power(self):
        result, modes = self.case(transmission=np.array([[1., 1.], [0., 0.]], complex))
        np.testing.assert_array_equal(np.sum(abs(result.transmission_amplitudes)**2, axis=0), [1, 1])
        record = qualify_scattering(result, modes, length_m=1, uniform=False,
                                    power_balance_passed=True, matched_reference_passed=None)
        self.assertEqual(record["status"], "finite_checks_failed")
        self.assertEqual(record["coherent_power"]["status"], "failed")
        self.assertAlmostEqual(record["coherent_power"]["coefficient_defect_spectral_norm"], 1.)
        self.assertIsNone(record["eligible_as_uniform_reference"])

    def test_wrong_phase_keeps_power_but_fails_uniform_reference(self):
        result, modes = self.case(transmission=np.eye(2, dtype=complex))
        record = qualify_scattering(result, modes, length_m=1, uniform=True,
                                    power_balance_passed=True, matched_reference_passed=True)
        self.assertEqual(record["coherent_power"]["status"], "passed")
        self.assertFalse(record["eligible_as_uniform_reference"])
        self.assertFalse(record["uniform_reference"]["same_basis_complex_passed"])
        np.testing.assert_array_equal(result.transmission_amplitudes, np.eye(2))

    def test_original_plane_reference_handles_both_incident_sides(self):
        for side in ("left", "right"):
            with self.subTest(side=side):
                result, modes = self.case(side=side)
                record = qualify_scattering(result, modes, length_m=1, uniform=True,
                                            power_balance_passed=True, matched_reference_passed=True)
                self.assertTrue(record["eligible_as_uniform_reference"])
                self.assertEqual(record["status"], "finite_checks_passed")
                self.assertEqual(record["spatial_convergence"], "unestablished")
                failed = qualify_scattering(result, modes, length_m=1, uniform=True,
                                            power_balance_passed=True, matched_reference_passed=False)
                self.assertFalse(failed["eligible_as_uniform_reference"])

    def test_coefficient_identity_requires_checked_flux_basis(self):
        result, modes = self.case()
        modes.flux_gram = lambda indices: 2 * np.diag([1., 1., -1., -1.])[np.ix_(indices, indices)]
        record = qualify_scattering(result, modes, length_m=1, uniform=False,
                                    power_balance_passed=True, matched_reference_passed=None)
        self.assertLess(record["coherent_power"]["coefficient_defect_spectral_norm"], 1e-14)
        self.assertFalse(record["coherent_power"]["unit_flux_basis_passed"])
        self.assertEqual(record["coherent_power"]["status"], "failed")


if __name__ == "__main__":
    unittest.main()
