"""Three-dimensional elasticity verification. Author: Athul Prem | _AP_pnc."""

import sys
import unittest
from dataclasses import replace
from pathlib import Path

import numpy as np
from scipy.spatial.transform import Rotation
from skfem import MeshTet

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from phononic.elasticity_AP_pnc import assemble_elasticity, solve_modes
from phononic.material_AP_pnc import CubicMaterial


def box_mesh(lengths, subdivisions):
    return MeshTet.init_tensor(*[
        np.linspace(0, length, n + 1) for length, n in zip(lengths, subdivisions)
    ])


def affine_field(basis, gradient, intercept):
    values = np.empty(basis.N)
    for axis, dofs in enumerate(basis.split_indices()):
        values[dofs] = gradient[axis] @ basis.doflocs[:, dofs] + intercept[axis]
    return values


def crystal_quadratic(material, strain):
    normal = np.diag(strain)
    shear = strain[np.triu_indices(3, k=1)]
    return (
        material.C11 * (normal @ normal)
        + 2 * material.C12 * (normal[0] * normal[1] + normal[1] * normal[2] + normal[2] * normal[0])
        + 4 * material.C44 * (shear @ shear)
    )


class CubicMaterialTests(unittest.TestCase):
    def test_tensor_symmetry_rotation_and_strain_energy(self):
        rotation = Rotation.from_rotvec([0.42, -0.13, 0.63]).as_matrix()
        material = CubicMaterial(rotation=rotation)
        tensor = material.stiffness_tensor()
        np.testing.assert_allclose(tensor, tensor.swapaxes(0, 1), atol=1e-4, rtol=1e-14)
        np.testing.assert_allclose(tensor, tensor.swapaxes(2, 3), atol=1e-4, rtol=1e-14)
        np.testing.assert_allclose(tensor, tensor.transpose(2, 3, 0, 1), atol=1e-4, rtol=1e-14)

        components = np.array([
            [[1, 0, 0], [0, 0, 0], [0, 0, 0]], [[0, 0, 0], [0, 1, 0], [0, 0, 0]],
            [[0, 0, 0], [0, 0, 0], [0, 0, 1]], [[0, 1, 0], [1, 0, 0], [0, 0, 0]],
            [[0, 0, 1], [0, 0, 0], [1, 0, 0]], [[0, 0, 0], [0, 0, 1], [0, 1, 0]],
        ])
        energy_matrix = np.einsum("aij,ijkl,bkl->ab", components, tensor, components)
        self.assertGreater(np.linalg.eigvalsh(energy_matrix).min(), 0.0)
        strain = np.array([[0.01, -0.03, 0.02], [-0.03, -0.07, 0.04], [0.02, 0.04, 0.05]])
        expected = crystal_quadratic(material, rotation.T @ strain @ rotation)
        actual = np.einsum("ij,ijkl,kl", strain, tensor, strain)
        np.testing.assert_allclose(actual, expected, rtol=2e-14)

        aligned = CubicMaterial().stiffness_tensor()
        self.assertEqual(aligned[0, 0, 0, 0], material.C11)
        self.assertEqual(aligned[0, 0, 1, 1], material.C12)
        self.assertEqual(aligned[0, 1, 0, 1], material.C44)
        self.assertEqual(aligned[0, 1, 1, 0], material.C44)

    def test_rejects_unstable_parameters_and_invalid_frames(self):
        for changes in ({"C44": 0}, {"C12": 150e9}, {"C12": -80e9}, {"rho": -1},
                        {"rotation": np.diag([1, 1, -1])}, {"rotation": 2 * np.eye(3)}):
            with self.subTest(changes=changes), self.assertRaises(ValueError):
                CubicMaterial(**changes)

    def test_rejects_boolean_and_other_nonreal_material_inputs(self):
        for name in ("C11", "C12", "C44", "rho"):
            for value in (True, np.bool_(True), "5323", 5323 + 0j):
                with self.subTest(name=name, value=value), self.assertRaisesRegex(ValueError, name):
                    CubicMaterial(**{name: value})

        mixed_rotation = [[True, 0, 0], [0, 1, 0], [0, 0, 1]]
        for rotation in (np.eye(3, dtype=bool), mixed_rotation):
            with self.subTest(rotation=rotation), self.assertRaisesRegex(ValueError, "Rotation"):
                CubicMaterial(rotation=rotation)
        numeric_rotation = [[1, 0, 0], [0, 1, 0], [0, 0, 1]]
        np.testing.assert_array_equal(CubicMaterial(rotation=numeric_rotation).rotation, np.eye(3))


class ElasticityTests(unittest.TestCase):
    def test_affine_patch_energy_consistent_mass_and_rigid_motion(self):
        lengths = np.array([0.9e-6, 0.4e-6, 0.2e-6])
        mesh = box_mesh(lengths, (2, 2, 2))
        rotation = Rotation.from_rotvec([0.17, -0.38, 0.21]).as_matrix()
        material = CubicMaterial(rotation=rotation)
        gradient = np.array([[0.07, -0.01, 0.05], [0.02, 0.04, -0.03], [0.03, 0.08, -0.02]])
        intercept = np.array([0.1e-6, -0.02e-6, 0.04e-6])
        strain = (gradient + gradient.T) / 2
        volume = np.prod(lengths)
        expected_stiffness = volume * crystal_quadratic(material, rotation.T @ strain @ rotation)
        mean = intercept + gradient @ (lengths / 2)
        variance = (gradient**2) @ (lengths**2 / 12)
        expected_mass = material.rho * volume * (mean @ mean + np.sum(variance))
        rotation_gradient = np.array([[0, -2, -1], [2, 0, -3], [1, 3, 0]])

        for order in (1, 2):
            with self.subTest(order=order):
                system = assemble_elasticity(mesh, material, element_order=order)
                u = affine_field(system.basis, gradient, intercept)
                np.testing.assert_allclose(u @ (system.K @ u), expected_stiffness, rtol=1e-12)
                np.testing.assert_allclose(u @ (system.M @ u), expected_mass, rtol=3e-14)
                force = system.K @ u
                interior = system.basis.complement_dofs(system.basis.get_dofs())
                self.assertGreater(len(interior), 0)
                self.assertLess(np.linalg.norm(force[interior]) / np.linalg.norm(force), 1e-13)
                for rigid_gradient, shift in ((np.zeros((3, 3)), intercept), (rotation_gradient, np.zeros(3))):
                    rigid = affine_field(system.basis, rigid_gradient, shift)
                    scale = np.asarray(abs(system.K).sum(axis=1)).max() * np.linalg.norm(rigid)
                    self.assertLess(np.linalg.norm(system.K @ rigid) / scale, 1e-14)

    def test_quadratic_displacement_integrates_degree_four_mass(self):
        lengths = np.array([0.9e-6, 0.4e-6, 0.2e-6])
        material = CubicMaterial()
        system = assemble_elasticity(box_mesh(lengths, (1, 1, 1)), material)
        u = np.zeros(system.basis.N)
        x_dofs = system.basis.split_indices()[0]
        u[x_dofs] = system.basis.doflocs[0, x_dofs] ** 2 / lengths[0]
        expected = material.rho * np.prod(lengths) * lengths[0]**2 / 5
        np.testing.assert_allclose(u @ (system.M @ u), expected, rtol=3e-14)

    def test_free_modes_mass_orthonormal_and_rigid_subspace(self):
        lengths = np.array([1e-6, 0.62e-6, 0.38e-6])
        system = assemble_elasticity(box_mesh(lengths, (2, 2, 2)), CubicMaterial())
        result = solve_modes(system, boundary="free", num_modes=8)
        self.assertEqual(result.rigid_mode_count, 6)
        np.testing.assert_array_equal(result.frequencies_hz[:6], 0)
        self.assertTrue(np.all(result.frequencies_hz[6:] > 0))
        self.assertLess(result.relative_residuals.max(), 1e-10)
        modes = result.modes_full
        np.testing.assert_allclose(modes.T @ (system.M @ modes), np.eye(8), atol=2e-12)
        np.testing.assert_allclose(result.polarization_fractions.sum(axis=1), 1.0, atol=1e-14)

        rigid_basis = modes[:, :6]
        for axis in np.eye(3):
            rotation = np.cross(axis, np.eye(3)).T
            for exact in (affine_field(system.basis, np.zeros((3, 3)), axis * lengths[0]),
                          affine_field(system.basis, rotation, np.zeros(3))):
                projection = rigid_basis @ (rigid_basis.T @ (system.M @ exact))
                error = projection - exact
                relative_error = np.sqrt((error @ (system.M @ error)) / (exact @ (system.M @ exact)))
                self.assertLess(relative_error, 1e-11)

        first = solve_modes(system, boundary="free", num_modes=1)
        self.assertEqual(first.rigid_mode_count, 1)
        self.assertEqual(first.frequencies_hz.item(), 0.0)

    def test_sparse_clamped_modes_match_dense_mass_whitening(self):
        mesh = box_mesh([1.15e-6, 0.72e-6, 0.46e-6], (3, 2, 2))
        system = assemble_elasticity(mesh, CubicMaterial())
        result = solve_modes(system, boundary="clamped_free", num_modes=6)
        free = result.free_dofs
        self.assertGreater(free.size, 256)
        K = system.K[free][:, free].toarray()
        M = system.M[free][:, free].toarray()
        L = np.linalg.cholesky(M)
        whitened = np.linalg.solve(L, np.linalg.solve(L, K).T)
        expected = np.sqrt(np.linalg.eigvalsh(whitened)[:6]) / (2 * np.pi)
        np.testing.assert_allclose(result.frequencies_hz, expected, rtol=3e-10)
        self.assertLess(result.relative_residuals.max(), 1e-10)

        fixed = np.setdiff1d(np.arange(system.basis.N), free)
        np.testing.assert_array_equal(system.basis.doflocs[0, fixed], 0.0)
        self.assertEqual(len(fixed), 3 * (2 * 2 + 1)**2)
        np.testing.assert_array_equal(result.modes_full[fixed], 0.0)

    def test_geometric_and_density_scaling_with_two_clamps(self):
        lengths = np.array([0.7e-6, 0.3e-6, 0.2e-6])
        material = CubicMaterial()
        system = assemble_elasticity(box_mesh(lengths, (2, 1, 1)), material)
        result = solve_modes(system, boundary="clamped_clamped", num_modes=4)
        resized = assemble_elasticity(box_mesh(2 * lengths, (2, 1, 1)), replace(material, rho=4 * material.rho))
        scaled = solve_modes(resized, boundary="clamped_clamped", num_modes=4)
        np.testing.assert_allclose(result.frequencies_hz, 4 * scaled.frequencies_hz, rtol=5e-13)
        fixed = np.setdiff1d(np.arange(system.basis.N), result.free_dofs)
        self.assertEqual(len(fixed), 2 * 3 * (2 + 1)**2)
        np.testing.assert_array_equal(result.modes_full[fixed], 0.0)

    def test_resource_cap_and_disconnected_mesh_fail_before_assembly(self):
        mesh = box_mesh([1e-6, 1e-6, 1e-6], (1, 1, 1))
        with self.assertRaisesRegex(ValueError, "exceeding cap"):
            assemble_elasticity(mesh, CubicMaterial(), max_dofs=30)

        separate_points = np.column_stack((mesh.p, mesh.p + 2e-6))
        separate_tets = np.column_stack((mesh.t, mesh.t + mesh.nvertices))
        with self.assertRaisesRegex(ValueError, "face-connected"):
            assemble_elasticity(MeshTet(separate_points, separate_tets), CubicMaterial())


if __name__ == "__main__":
    unittest.main()
