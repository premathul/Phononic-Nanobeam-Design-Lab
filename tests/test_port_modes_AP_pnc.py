"""Matched cross-section mode benchmarks. Author: Athul Prem | _AP_pnc."""

import sys
import unittest
from pathlib import Path

import numpy as np
from scipy.linalg import eigh
from skfem import MeshTet

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from phononic.elasticity_AP_pnc import assemble_elasticity
from phononic.geometry_AP_pnc import BeamGeometry, make_straight_mesh
from phononic.lead_modes_AP_pnc import solve_bloch
from phononic.material_AP_pnc import CubicMaterial
from phononic.port_modes_AP_pnc import assemble_port, solve_port_modes


class PortModeTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.material = CubicMaterial()
        cls.geometry = BeamGeometry(600e-9, 500e-9, 200e-9)
        cls.system = assemble_elasticity(make_straight_mesh(cls.geometry, (3, 2, 1)), cls.material)
        cls.port = assemble_port(cls.system, cls.material)
        cls.modes = solve_port_modes(cls.port, 300e6)

    def test_end_face_basis_and_integrated_mass_match_for_p1_and_p2(self):
        mesh = self.system.mesh
        for order, expected_face_dofs in ((1, 18), (2, 45)):
            with self.subTest(order=order):
                system = assemble_elasticity(mesh, self.material, element_order=order)
                port = assemble_port(system, self.material)
                self.assertEqual(port.basis.N, expected_face_dofs)
                self.assertIs(port.elastic_system, system)
                for name, port_dofs in (("xleft", port.left_dofs), ("xright", port.right_dofs)):
                    expected = system.basis.get_dofs(name).all()
                    np.testing.assert_array_equal(np.sort(port_dofs), expected)
                    np.testing.assert_allclose(port.basis.doflocs, system.basis.doflocs[1:, port_dofs], atol=1e-22)
                    for face, volume in zip(port.basis.split_indices(), system.basis.split_indices()):
                        self.assertTrue(np.isin(port_dofs[face], volume).all())
                rigid_x = np.zeros(port.basis.N)
                rigid_x[port.basis.split_indices()[0]] = 1
                mass_per_length = self.material.rho * self.geometry.width_m * self.geometry.thickness_m
                np.testing.assert_allclose(rigid_x @ (port.M @ rigid_x), mass_per_length, rtol=2e-14)
        with self.assertRaisesRegex(ValueError, "dense solver cap"):
            assemble_port(self.system, self.material, max_dofs=44)

    def test_all_roots_flux_orthogonality_and_propagation_pairing(self):
        modes = self.modes
        n = self.port.basis.N
        self.assertEqual(modes.wavenumbers_rad_m.shape, (2 * n,))
        self.assertLess(modes.relative_residuals.max(), 1e-10)
        np.testing.assert_array_equal(np.sort(np.r_[modes.positive_indices, modes.negative_indices]), np.arange(2 * n))

        for indices, sign in ((modes.positive_indices, 1), (modes.negative_indices, -1)):
            self.assertEqual(indices.size, n)
            np.testing.assert_array_equal(modes.propagating[indices], np.arange(n) < 4)
            np.testing.assert_allclose(modes.flux_gram(indices), np.diag(sign * (np.arange(n) < 4)), atol=1e-10)
            np.testing.assert_allclose(modes.powers_w[indices], modes.flux_gram(indices).diagonal().real, atol=1e-12)
            self.assertTrue(np.all(sign * modes.wavenumbers_rad_m[indices[4:]].imag > 0))

        k_forward = modes.wavenumbers_rad_m[modes.positive_indices[:4]]
        k_backward = modes.wavenumbers_rad_m[modes.negative_indices[:4]]
        np.testing.assert_allclose(k_forward, -k_backward, rtol=1e-10)

    def test_signed_axial_power_agrees_with_dispersion_derivative(self):
        modes, port = self.modes, self.port
        omega = 2 * np.pi * modes.frequency_hz
        H = port.G.T - port.G
        for indices, sign in ((modes.positive_indices[:4], 1), (modes.negative_indices[:4], -1)):
            for index in indices:
                with self.subTest(index=index):
                    k = modes.wavenumbers_rad_m[index].real
                    U = modes.displacements[:, index]
                    # A centered slope of an independently solved real-k
                    # cross-section eigenproblem estimates d(omega)/dk.
                    dk = 2e-5 * abs(k)
                    frequencies = []
                    for point in (k - dk, k + dk):
                        stiffness = port.K0 + 1j * point * H + point**2 * port.Q
                        values = eigh(stiffness, port.M, eigvals_only=True)
                        angular_frequencies = np.sqrt(np.maximum(values, 0))
                        frequencies.append(angular_frequencies[np.argmin(abs(angular_frequencies - omega))])
                    group_velocity = (frequencies[1] - frequencies[0]) / (2 * dk)
                    energy_per_length = omega**2 / 2 * np.vdot(U, port.M @ U).real
                    np.testing.assert_allclose(group_velocity * energy_per_length, sign, rtol=2e-6)

    def test_wavenumbers_match_independent_three_dimensional_bloch_solve(self):
        for index in self.modes.positive_indices[:4]:
            k = self.modes.wavenumbers_rad_m[index].real
            dispersion = solve_bloch(self.system, k, num_modes=4)
            nearest = np.min(abs(dispersion.frequencies_hz / self.modes.frequency_hz - 1))
            self.assertLess(nearest, 0.01)

    def test_symmetry_degenerate_flexural_modes_are_flux_orthogonal(self):
        # A square with center-to-corner triangles preserves fourfold symmetry.
        yz = 0.25e-6 * np.array([[-1, 1, 1, -1, 0], [-1, -1, 1, 1, 0]])
        coordinates = np.vstack((np.repeat([0.0, 0.6e-6], 5), np.tile(yz, (1, 2))))
        tetrahedra = []
        for a, b, c in ((0, 1, 4), (1, 2, 4), (2, 3, 4), (0, 3, 4)):
            tetrahedra.extend(((a, b, c, c + 5), (a, b, b + 5, c + 5), (a, a + 5, b + 5, c + 5)))
        system = assemble_elasticity(MeshTet(coordinates, np.array(tetrahedra).T).oriented(), self.material)
        modes = solve_port_modes(assemble_port(system, self.material), 300e6)
        for indices, sign in ((modes.positive_indices, 1), (modes.negative_indices, -1)):
            flexural = indices[2:4]
            k = modes.wavenumbers_rad_m[flexural]
            self.assertTrue(modes.propagating[flexural].all())
            np.testing.assert_allclose(k[0], k[1], rtol=1e-10)
            np.testing.assert_allclose(modes.flux_gram(flexural), sign * np.eye(2), atol=1e-12)

    def test_cutoff_fails_explicitly_and_resolved_sides_change_channel_count(self):
        first_positive_squared = eigh(self.port.K0, self.port.M, eigvals_only=True, subset_by_index=(4, 4)).item()
        cutoff_hz = np.sqrt(first_positive_squared) / (2 * np.pi)
        with self.assertRaisesRegex(RuntimeError, "Unresolved lead cutoff"):
            solve_port_modes(self.port, cutoff_hz)
        below = solve_port_modes(self.port, cutoff_hz * 0.999)
        above = solve_port_modes(self.port, cutoff_hz * 1.001)
        self.assertEqual(np.count_nonzero(below.propagating), 8)
        self.assertEqual(np.count_nonzero(above.propagating), 10)


if __name__ == "__main__":
    unittest.main()
