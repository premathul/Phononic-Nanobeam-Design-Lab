"""Fixed-connectivity wall geometry tests. Athul Prem | _AP_pnc."""

from dataclasses import replace
import unittest

import numpy as np

from tools.fixed_wall_faceting_AP_pnc import make_fixed_wall_faceting_control
from phononic.geometry_AP_pnc import BeamGeometry, Hole
from phononic.layered_AP_pnc import make_layered_mesh


class FixedWallFacetingTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.geometry = BeamGeometry(1.6e-6, 5e-7, 1.5e-7)
        cls.hole = Hole(8e-7, 2.5e-7)
        cls.cuts = (3e-7, 1.3e-6)
        cls.mesh = cls.make_mesh(48)

    @classmethod
    def make_mesh(cls, circle_points):
        return make_layered_mesh(
            cls.geometry, (cls.hole,), 180e-9, port_mesh_size_m=125e-9, thickness_layers=2,
            circle_points=circle_points, lead_mesh_size_m=150e-9, patterned_x_interval_m=cls.cuts,
        )

    def make_control(self, mesh):
        return make_fixed_wall_faceting_control(mesh, self.geometry, (self.hole,), thickness_layers=2,
                                                patterned_x_interval_m=self.cuts)

    def test_projected_alternating_xy_vertices_and_volume(self):
        mesh = self.mesh
        original_points = mesh.p.copy()
        control, report = self.make_control(mesh)
        np.testing.assert_array_equal(mesh.p, original_points)
        np.testing.assert_array_equal(control.t, mesh.t)
        np.testing.assert_array_equal(control.p[2], mesh.p[2])
        for field in ("boundaries", "subdomains"):
            original_tags, control_tags = (getattr(value, field) for value in (mesh, control))
            self.assertEqual(original_tags.keys(), control_tags.keys())
            for tag in original_tags:
                np.testing.assert_array_equal(control_tags[tag], original_tags[tag])

        relative = mesh.p[:2] - [[self.hole.x_m], [0.0]]
        radius = self.hole.diameter_m / 2
        radial = np.linalg.norm(relative, axis=0) / radius
        angle = np.arctan2(relative[1], relative[0])
        angle_index = np.rint(48 * angle / (2 * np.pi)).astype(int) % 48
        expected_moved = np.isclose(radial, 1, atol=1e-9, rtol=0) & (angle_index % 2 == 1)
        np.testing.assert_array_equal(np.any(control.p != mesh.p, axis=0), expected_moved)
        np.testing.assert_allclose(np.linalg.norm(control.p[:2, expected_moved] - mesh.p[:2, expected_moved], axis=0),
                                   radius * (1 - np.cos(np.pi / 24)), rtol=1e-12, atol=0)
        self.assertEqual(report["moved_vertices_per_z_plane"], [24, 24, 24])
        self.assertEqual(report["touched_tetrahedra"], int(expected_moved[mesh.t].any(axis=0).sum()))
        self.assertEqual(report["control_physical_rim_sides"], 24)
        self.assertEqual(report["finite_element_rim_edges_per_ring"], 48)
        self.assertGreater(report["minimum_relative_signed_tet_determinant"], 0)
        expected_volume = radius**2 * (24 * np.sin(np.pi / 24) - 12 * np.sin(np.pi / 12)) * self.geometry.thickness_m
        np.testing.assert_allclose(report["measured_solid_volume_increase_m3"], expected_volume, rtol=1e-12, atol=0)

    def test_reject_actual_rim_count_from_mesh(self):
        with self.assertRaisesRegex(ValueError, "48-edge CAD rim"):
            self.make_control(self.make_mesh(32))

    def test_reject_incomplete_exterior_partition(self):
        tags = self.mesh.boundaries.copy()
        tags["free"] = tags["free"][:-1]
        with self.assertRaisesRegex(ValueError, "must partition"):
            self.make_control(replace(self.mesh, _boundaries=tags))


if __name__ == "__main__":
    unittest.main()
