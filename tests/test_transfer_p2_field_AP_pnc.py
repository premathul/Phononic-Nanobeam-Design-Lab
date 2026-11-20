"""Exercise native P2 transfer on nonnested tetrahedra. Author: Athul Prem | _AP_pnc."""

import unittest

import numpy as np
from skfem import Basis, ElementTetP2, ElementVector, MeshTet

from tools.transfer_p2_field_AP_pnc import build_p2_transfer


SCALE_M = 1e-6


def thin_guide(x_coordinates, transverse=(0.0, 1.0)):
    mesh = MeshTet.init_tensor(SCALE_M * np.asarray(x_coordinates), SCALE_M * np.asarray(transverse),
                               SCALE_M * np.array([0.0, 0.1]))
    return mesh.with_boundaries({
        "xleft": lambda points: points[0] == 0.0,
        "xright": lambda points: points[0] == SCALE_M,
    })


def field(basis, break_x):
    """A globally continuous complex P2 field with a different polynomial in each slab."""
    x, y, z = basis.doflocs / SCALE_M
    kink = np.maximum(x - break_x, 0.0)
    components = np.array([
        2.0 + x + (0.5 + 3j) * y * z + (1.0j - 0.1) * kink,
        -4.0j + 7.0 * x * y - 2.0 * z**2 + (3.1 + 2.0j) * kink**2,
        5.0 + 2.3j * y**2 - 3.1j * x * z + (2.1 - 4j) * kink,
    ])
    values = components[np.arange(basis.N) % 3, np.arange(basis.N)]
    return np.column_stack((values, values.conj() * (2.3 - 4j)))


class P2FieldTransferTests(unittest.TestCase):
    def test_both_directions_preserve_native_components_and_exact_ports(self):
        left = thin_guide([0.0, 0.32, 1.0])
        right = thin_guide([0.0, 0.27, 0.58, 1.0])
        for source, target, break_x in ((left, right, 0.32), (right, left, 0.58)):
            with self.subTest(break_x=break_x):
                native_source = Basis(source, ElementVector(ElementTetP2()))
                native_target = Basis(target, ElementVector(ElementTetP2()))
                plan = build_p2_transfer(source, target, batch_size=3)
                original = field(native_source, break_x)
                transferred = plan.apply(original)
                np.testing.assert_allclose(transferred, field(native_target, break_x),
                                           rtol=0, atol=2e-14)
                np.testing.assert_array_equal(plan.apply(original[:, 1]), transferred[:, 1])
                with self.assertRaisesRegex(ValueError, "native xyz-interleaved"):
                    plan.apply(original.T)
                self.assertGreater(plan.diagnostics["multiple_parent_scalar_dofs"], 0)
                self.assertGreater(np.count_nonzero(plan.scalar_matrix.getnnz(axis=1) > 1), 0)
                edge_transfers = plan.scalar_matrix.getnnz(axis=1)[target.nvertices:]
                self.assertGreater(np.count_nonzero(edge_transfers > 1), 0)
                self.assertLessEqual(plan.diagnostics["maximum_reconstruction_error_m"],
                                     plan.diagnostics["coordinate_tolerance_m"])
                np.testing.assert_array_equal(
                    transferred.reshape(-1, 3, 2)[plan.shared_target],
                    original.reshape(-1, 3, 2)[plan.shared_source],
                )
                for port in ("xleft", "xright"):
                    source_port = native_source.get_dofs(port).all()
                    target_port = native_target.get_dofs(port).all()
                    source_order = np.lexsort((source_port % 3, *native_source.doflocs[:, source_port]))
                    target_order = np.lexsort((target_port % 3, *native_target.doflocs[:, target_port]))
                    np.testing.assert_array_equal(
                        transferred[target_port[target_order]], original[source_port[source_order]])
                    self.assertEqual(plan.diagnostics["ports"][port]["exact_vector_dofs"], len(target_port))

    def test_interior_values_follow_actual_parent_tetrahedra(self):
        vertices = np.eye(3, 4)
        tetrahedra = np.array([[4, 0, 1, 2], [4, 1, 2, 3], [4, 2, 3, 0], [4, 3, 0, 1]]).T
        source = MeshTet(SCALE_M * np.column_stack((vertices, [0.2, 0.2, 0.2])), tetrahedra)
        target = MeshTet(SCALE_M * np.column_stack((vertices, [0.18, 0.13, 0.21])), tetrahedra)
        random = np.random.default_rng(17)
        for donor, recipient in ((source, target), (target, source)):
            native_source = Basis(donor, ElementVector(ElementTetP2()))
            native_target = Basis(recipient, ElementVector(ElementTetP2()))
            plan = build_p2_transfer(donor, recipient, batch_size=1)
            field_values = random.normal(size=native_source.N) + 1j * random.normal(size=native_source.N)
            recipient_only = np.setdiff1d(np.arange(plan.scalar_matrix.shape[0]), plan.shared_target)
            points = native_target.doflocs[:, 3 * recipient_only]
            self.assertEqual(len(recipient_only), 5)
            self.assertGreater(np.unique(donor.element_finder()(*points)).size, 1)
            independent = native_source.interpolator(field_values)(points).T
            actual = plan.apply(field_values).reshape(-1, 3)[recipient_only]
            np.testing.assert_allclose(actual, independent, rtol=0, atol=3e-15)

    def test_different_named_port_triangulation_is_rejected(self):
        source = thin_guide([0.0, 1.0])
        target = thin_guide([0.0, 1.0], (0.0, 0.37, 1.0))
        with self.assertRaisesRegex(ValueError, "Physical xleft port triangulation"):
            build_p2_transfer(source, target)

    def test_outside_tetrahedron_within_axis_aligned_bounds_is_rejected(self):
        source = MeshTet(np.eye(3, 4) * SCALE_M, np.arange(4)[:, None])
        target = MeshTet((0.45 + 0.2 * np.eye(3, 4)) * SCALE_M, np.arange(4)[:, None])
        with self.assertRaisesRegex(ValueError, "lies outside the source solid"):
            build_p2_transfer(source, target)

    def test_intersecting_tetrahedra_with_distinct_traces_are_rejected(self):
        tetra = np.eye(3, 4) * SCALE_M
        source = MeshTet(np.hstack((tetra, tetra + 0.03 * SCALE_M)), np.arange(8).reshape(2, 4).T)
        target = MeshTet(0.08 * SCALE_M + 0.11 * tetra, np.arange(4)[:, None])
        with self.assertRaisesRegex(ValueError, "different P2 traces"):
            build_p2_transfer(source, target)


if __name__ == "__main__":
    unittest.main()
