"""Check native snapshot actions against the production equations. Athul Prem | _AP_pnc."""

from math import tau
from pathlib import Path
import sys
from types import SimpleNamespace
import unittest

import numpy as np
from scipy.sparse import csc_matrix

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from phononic.port_modes_AP_pnc import PortModes
from phononic.scattering_AP_pnc import solve_scattering
from tools.snapshot_native_system_AP_pnc import NativeScatteringSystem, VolumeOperators


class NativeSnapshotTests(unittest.TestCase):
    def setUp(self):
        n = 8
        self.frequency = 2.0
        self.body = SimpleNamespace(
            K=csc_matrix(np.diag(np.arange(1., n + 1)) + np.full((n, n), 0.1)),
            M=csc_matrix(np.diag(np.linspace(0.01, 0.04, n))),
        )
        left_dofs, right_dofs = np.array([0, 3, 1]), np.array([6, 4, 5])
        left_roots, right_roots = np.array([4, 0, 2]), np.array([5, 1, 3])
        left_open, right_open = left_roots[:2], right_roots[:2]
        port = SimpleNamespace(elastic_system=self.body, left_dofs=left_dofs, right_dofs=right_dofs)
        displacement = np.empty((3, 6), dtype=complex)
        displacement[:, left_roots] = [[1, 0.2j, 0.6], [0.3, 1, 0.2j], [0.4j, 0.1, 1]]
        displacement[:, right_roots] = [[1, 0.2, 0.3j], [0.4j, 1, 0.5], [0.1, 0.5j, 1]]
        power = np.zeros(6)
        power[left_open], power[right_open] = -1, 1
        force = displacement * (2j * power / (tau * self.frequency * np.sum(abs(displacement)**2, axis=0)))
        force[:, [left_roots[-1], right_roots[-1]]] = 0.2 * displacement[:, [left_roots[-1], right_roots[-1]]]
        self.modes = PortModes(port, self.frequency, np.zeros(6), displacement, force,
                               right_roots, left_roots, power != 0, power, np.zeros(6))
        self.volume = VolumeOperators(self.body.K, self.body.M)
        self.archive = {
            "port_left_dofs": left_dofs,
            "port_right_dofs": right_dofs,
            "port_displacements": displacement,
            "port_axial_forces": force,
            "reflection_all_mode_indices": left_roots,
            "transmission_all_mode_indices": right_roots,
            "incident_mode_indices": right_open,
            "reflection_mode_indices": left_open,
            "transmission_mode_indices": right_open,
            "incident_powers_w": np.ones(2),
            "frequencies_hz": np.full(2, self.frequency),
        }
        self.system = NativeScatteringSystem.from_saved_fields(self.volume, self.frequency, self.archive)

    def test_action_forcing_and_native_coefficients_match_production(self):
        fields = np.arange(16).reshape(8, 2) * (0.01 + 0.033j)
        reflected = np.array([[0.12j, 0.03], [-0.15j, 0.23 - 0.03j], [0.8j, 0.33]])
        transmitted = np.array([[0.55 + 0.41j, -0.5j], [0.31, -0.34 + 0.57j], [-0.45, 0.35j]])
        displacement = self.modes.displacements
        fields[self.system.left_dofs] = (displacement[:, self.system.incident_roots]
                                        + displacement[:, self.modes.negative_indices] @ reflected)
        fields[self.system.right_dofs] = displacement[:, self.modes.positive_indices] @ transmitted

        def check_equations(operator, rhs):
            expected = operator @ fields
            np.testing.assert_allclose(self.system.apply(fields), expected, atol=1e-14, rtol=0)
            np.testing.assert_allclose(self.system.apply(fields[:, 0]), expected[:, 0], atol=1e-14, rtol=0)
            np.testing.assert_allclose(self.system.forcing(), rhs, atol=1e-15, rtol=0)
            residual = np.linalg.norm(expected - rhs, axis=0) / np.linalg.norm(rhs, axis=0)
            np.testing.assert_allclose(self.system.relative_residuals(fields, rhs), residual, atol=1e-13, rtol=0)
            return fields

        result = solve_scattering(self.body, self.modes, self.frequency, linear_solver=check_equations)
        amplitudes = self.system.outgoing(fields)
        for observed, expected, production in zip(amplitudes, (reflected[:2], transmitted[:2]),
                                                  (result.reflection_amplitudes, result.transmission_amplitudes),
                                                  strict=True):
            np.testing.assert_allclose(observed, expected, atol=1e-15, rtol=0)
            np.testing.assert_allclose(observed, production, atol=1e-15, rtol=0)
        for observed, production in zip(self.system.modal_power_fractions(*amplitudes),
                                        (result.reflection_modal_powers, result.transmission_modal_powers),
                                        strict=True):
            np.testing.assert_allclose(observed, production, atol=1e-15, rtol=0)

    def test_rejects_frequency_or_direction_mismatches(self):
        with self.assertRaisesRegex(ValueError, "stated native frequency"):
            NativeScatteringSystem.from_saved_fields(self.volume, 1.01 * self.frequency, self.archive)
        with self.assertRaisesRegex(ValueError, "complete outgoing basis"):
            NativeScatteringSystem.from_saved_fields(
                self.volume, self.frequency, {**self.archive, "reflection_mode_indices": self.system.incident_roots})


if __name__ == "__main__":
    unittest.main()
