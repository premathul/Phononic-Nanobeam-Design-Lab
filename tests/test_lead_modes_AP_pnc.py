"""Verification of 3D guide dispersion. Author: Athul Prem | _AP_pnc."""

import math
from pathlib import Path
import sys
import unittest

import numpy as np
from skfem import MeshTet

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from phononic.elasticity_AP_pnc import CubicMaterial, assemble_elasticity
from phononic.lead_modes_AP_pnc import solve_bloch


class LeadModeTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.period = 1e-6
        cls.c11, cls.c12, cls.c44, cls.density = 128.5e9, 48.3e9, 66.8e9, 5323.0
        cls.material = CubicMaterial(cls.c11, cls.c12, cls.c44, cls.density, np.eye(3))
        cls.mesh = MeshTet.init_tensor(
            np.linspace(0, cls.period, 5), np.linspace(-0.3e-6, 0.3e-6, 4),
            np.linspace(-0.2e-6, 0.2e-6, 3),
        )
        cls.system = assemble_elasticity(cls.mesh, cls.material)

    def test_gamma_has_four_rigid_modes(self):
        result = solve_bloch(self.system, 0.0)
        self.assertEqual(result.rigid_mode_count, 4)
        np.testing.assert_array_equal(result.frequencies_hz[:4], 0.0)
        self.assertTrue(np.all(result.frequencies_hz[4:] > 0.0))
        self.assertLess(np.max(result.relative_residuals), 1e-8)

        xyz = self.system.basis.doflocs
        translations = np.zeros((self.system.basis.N, 4))
        x, y, z = self.system.basis.split_indices()
        translations[x, 0] = 1.0
        translations[y, 1] = 1.0
        translations[z, 2] = 1.0
        translations[y, 3] = -xyz[2, y] / self.period
        translations[z, 3] = xyz[1, z] / self.period
        rigid = result.modes_full[:, :4]
        mass = self.system.M
        projected = rigid @ (rigid.conj().T @ (mass @ translations))
        error = translations - projected
        relative_error = np.linalg.norm(error, axis=0) / np.linalg.norm(translations, axis=0)
        self.assertLess(np.max(relative_error), 1e-8)

    def test_opposite_wavenumbers_and_face_phase(self):
        k = 0.7 / self.period
        forward = solve_bloch(self.system, k)
        backward = solve_bloch(self.system, -k)
        np.testing.assert_allclose(forward.frequencies_hz, backward.frequencies_hz, rtol=1e-9)
        self.assertEqual(forward.rigid_mode_count, 0)
        self.assertLess(np.max(forward.relative_residuals), 1e-8)
        np.testing.assert_allclose(forward.polarization_fractions.sum(axis=1), 1.0, atol=1e-10)
        modes = forward.modes_full
        np.testing.assert_allclose(modes.conj().T @ (self.system.M @ modes), np.eye(8), atol=1e-8)

        locations = self.system.basis.doflocs
        phase = np.exp(1j * k * self.period)
        for component in self.system.basis.split_indices():
            left = component[locations[0, component] == 0.0]
            right = component[locations[0, component] == self.period]
            order_left = np.lexsort(locations[1:, left])
            order_right = np.lexsort(locations[1:, right])
            np.testing.assert_allclose(
                modes[right[order_right]], phase * modes[left[order_left]], rtol=1e-12, atol=1e-12
            )

    def test_small_k_extension_uses_free_sidewall_rod_speed(self):
        # Cubic [100] axial modulus under uniaxial stress; the sidewalls are free.
        young = (self.c11 - self.c12) * (self.c11 + 2 * self.c12) / (self.c11 + self.c12)
        expected_speed = math.sqrt(young / self.density)
        for phase in (0.1, 0.2):
            with self.subTest(phase=phase):
                k = phase / self.period
                result = solve_bloch(self.system, k)
                extensional_mode = np.argmax(result.polarization_fractions[:4, 0])
                self.assertGreater(result.polarization_fractions[extensional_mode, 0], 0.99)
                speed = math.tau * result.frequencies_hz[extensional_mode] / k
                # Finite k is used, with a 0.5% window for discretization and long-wave errors.
                self.assertAlmostEqual(speed / expected_speed, 1.0, delta=0.005)

    def test_nonmatching_end_faces_are_rejected(self):
        points = self.mesh.p.copy()
        points[1, points[0] == self.period] += 0.01e-6
        mesh = MeshTet(points, self.mesh.t)
        system = assemble_elasticity(mesh, self.material, element_order=1)
        with self.assertRaisesRegex(ValueError, "must match one-to-one"):
            solve_bloch(system, 0.0)

    def test_matching_face_nodes_with_different_triangles_are_rejected(self):
        edges = [0.0, self.period]
        cube = MeshTet.init_tensor(edges, edges, edges)
        faces = cube.facets[:, cube.boundary_facets()].copy()
        right = np.all(cube.p[0, faces] == self.period, axis=0)
        vertices = np.flatnonzero(cube.p[0] == self.period)
        right_points = cube.p[:, vertices].copy()
        right_points[1] = self.period - right_points[1]
        reflected = np.argmin(
            np.linalg.norm(cube.p[:, :, None] - right_points[:, None, :], axis=0), axis=0
        )
        # Flip the right-face diagonal, then connect every face to the cube center.
        faces[:, right] = reflected[np.searchsorted(vertices, faces[:, right])]
        points = np.column_stack((cube.p, np.full(3, self.period / 2)))
        mesh = MeshTet(points, np.vstack((faces, np.full(faces.shape[1], cube.nvertices)))).oriented()
        system = assemble_elasticity(mesh, self.material, element_order=1)
        with self.assertRaisesRegex(ValueError, "matching triangulations"):
            solve_bloch(system, 0.0)


if __name__ == "__main__":
    unittest.main()
