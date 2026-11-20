"""Verify native regional bilinear work. Author: Athul Prem | _AP_pnc."""

from math import tau
from pathlib import Path
import sys
import unittest

import numpy as np
from scipy.spatial.transform import Rotation
from skfem import BilinearForm
from skfem.helpers import ddot, dot, sym_grad

from tools.regional_bilinear_work_AP_pnc import regional_bilinear_work

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from phononic.elasticity_AP_pnc import assemble_elasticity
from phononic.geometry_AP_pnc import BeamGeometry, make_straight_mesh
from phononic.material_AP_pnc import CubicMaterial


class RegionalBilinearWorkTests(unittest.TestCase):
    def test_complex_transpose_work_matches_native_regional_and_global_operators(self):
        mesh = make_straight_mesh(BeamGeometry(1.1e-6, 0.63e-6, 0.2e-6), (4, 2, 1))
        material = CubicMaterial(rotation=Rotation.from_rotvec([0.17, -0.38, 0.21]).as_matrix())
        system = assemble_elasticity(mesh, material, element_order=2)
        frequency = 3.25e9
        omega_squared = (tau * frequency) ** 2
        rng = np.random.default_rng(64329)
        field_size = (system.basis.N, 5)
        u = rng.standard_normal(field_size) + 1j * rng.standard_normal(field_size)
        z = rng.standard_normal(field_size) + 1j * rng.standard_normal(field_size)
        groups = np.resize([4, -1, 23], mesh.nelements)
        rng.shuffle(groups)

        result = regional_bilinear_work(mesh, material, u, z, groups, frequency, batch_elements=7)
        np.testing.assert_array_equal(result.groups, [-1, 4, 23])
        self.assertEqual(result.dynamic.shape, (3, 5, 5))

        stiffness = material.stiffness_tensor()

        @BilinearForm
        def elastic_form(u, v, _):
            stress = np.einsum("ijkl,kl...->ij...", stiffness, sym_grad(u))
            return ddot(stress, sym_grad(v))

        @BilinearForm
        def mass_form(u, v, _):
            return material.rho * dot(u, v)

        for index, group in enumerate(result.groups):
            with self.subTest(group=group):
                basis = system.basis.with_elements(np.flatnonzero(groups == group))
                K = elastic_form.assemble(basis)
                M = mass_form.assemble(basis)
                expected_stiffness = z.T @ (K @ u)
                expected_inertia = omega_squared * (z.T @ (M @ u))
                self.assertScaledEqual(result.stiffness[index], expected_stiffness)
                self.assertScaledEqual(result.inertia[index], expected_inertia)
                self.assertScaledEqual(result.dynamic[index], expected_stiffness - expected_inertia)

        self.assertScaledEqual(result.stiffness.sum(axis=0), z.T @ (system.K @ u))
        self.assertScaledEqual(result.inertia.sum(axis=0), omega_squared * (z.T @ (system.M @ u)))
        dynamic_operator = system.K - omega_squared * system.M
        self.assertScaledEqual(result.dynamic.sum(axis=0), z.T @ (dynamic_operator @ u))
        hermitian = z.conj().T @ (dynamic_operator @ u)
        self.assertGreater(np.linalg.norm(result.dynamic.sum(axis=0) - hermitian),
                           0.5 * np.linalg.norm(hermitian))

    def assertScaledEqual(self, actual, expected):
        scale = np.max(abs(expected))
        self.assertGreater(scale, 0)
        np.testing.assert_allclose(actual / scale, expected / scale, rtol=2e-13, atol=2e-13)


if __name__ == "__main__":
    unittest.main()
