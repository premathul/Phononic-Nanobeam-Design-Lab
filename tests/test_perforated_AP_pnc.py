"""Geometry and port checks for perforated beams. Author: Athul Prem | _AP_pnc."""

from importlib.util import find_spec
from math import cos, pi, sin
from pathlib import Path
import sys
import unittest

import numpy as np
from skfem import Dofs, ElementTetP2, ElementVector, FacetBasis

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from phononic.elasticity_AP_pnc import assemble_elasticity
from phononic.geometry_AP_pnc import BeamGeometry, Hole
from phononic.material_AP_pnc import CubicMaterial
from phononic.perforated_AP_pnc import make_perforated_mesh, volume_diagnostics
from phononic.port_modes_AP_pnc import assemble_port


def port_triangles(mesh, label):
    """Canonical transverse triangle coordinates for comparing separate meshes."""
    points = np.round(mesh.p[1:, :] * 1e9, 8).T
    return {tuple(sorted(tuple(points[vertex]) for vertex in face))
            for face in mesh.facets[:, mesh.boundaries[label]].T}


@unittest.skipUnless(find_spec("gmsh") and find_spec("meshio"), "requires gmsh and meshio")
class PerforatedMeshTests(unittest.TestCase):
    geometry = BeamGeometry(1e-6, 600e-9, 200e-9)
    holes = (Hole(300e-9, 180e-9), Hole(700e-9, 180e-9))

    def test_periodic_trace_matches_control_and_assembles(self):
        def build(holes):
            return make_perforated_mesh(self.geometry, holes, 150e-9,
                                        port_mesh_size_m=200e-9, circle_points=16, max_port_dofs=63)

        control = build(())
        mesh = build(self.holes)
        exterior = np.sort(np.concatenate(list(mesh.boundaries.values())))
        np.testing.assert_array_equal(exterior, mesh.boundary_facets())
        reference = port_triangles(control, "xleft")
        self.assertEqual(len(reference), 6)
        for label, outward in (("xleft", -1), ("xright", 1)):
            self.assertEqual(port_triangles(mesh, label), reference)
            face = FacetBasis(mesh, ElementTetP2(), facets=mesh.boundaries[label])
            self.assertAlmostEqual(face.dx.sum() / (self.geometry.width_m * self.geometry.thickness_m), 1.0)
            np.testing.assert_allclose(face.normals[0], outward, atol=1e-12, rtol=0)
        control_geometry = volume_diagnostics(control, self.geometry, ())
        self.assertLess(control_geometry["relative_solid_volume_error"], 1e-12)
        self.assertIsNone(control_geometry["relative_removed_volume_error"])

        material = CubicMaterial()
        system = assemble_elasticity(mesh, material, max_dofs=8000)
        port = assemble_port(system, material, max_dofs=63)
        self.assertEqual(port.basis.N, 63)
        np.testing.assert_allclose(system.basis.doflocs[1:, port.left_dofs],
                                   system.basis.doflocs[1:, port.right_dofs], atol=1e-20, rtol=0)

    def test_circle_resolution_is_independent_of_body_size(self):
        body_dofs = []
        for size, segments in ((145e-9, 16), (105e-9, 16), (105e-9, 24)):
            with self.subTest(size=size, segments=segments):
                mesh = make_perforated_mesh(self.geometry, self.holes, size,
                                            port_mesh_size_m=200e-9, circle_points=segments)
                body_dofs.append(Dofs(mesh, ElementVector(ElementTetP2())).N)
                diagnostics = volume_diagnostics(mesh, self.geometry, self.holes)
                self.assertEqual(diagnostics["rim_segments_min"], segments)
                self.assertEqual(diagnostics["rim_segments_max"], segments)
                polygon_area_ratio = segments * sin(2 * pi / segments) / (2 * pi)
                self.assertAlmostEqual(diagnostics["relative_removed_volume_error"],
                                       1 - polygon_area_ratio, places=12)
                self.assertAlmostEqual(diagnostics["max_radial_sagitta_m"] / 90e-9,
                                       1 - cos(pi / segments), places=12)
        self.assertLess(body_dofs[0], body_dofs[1])

    def test_rejects_port_budget_and_invalid_circle_divisions(self):
        with self.assertRaisesRegex(ValueError, "exceeding dense solver cap"):
            make_perforated_mesh(self.geometry, self.holes, 150e-9,
                                 port_mesh_size_m=200e-9, max_port_dofs=62)
        with self.assertRaisesRegex(ValueError, "multiple of 4"):
            make_perforated_mesh(self.geometry, self.holes, 150e-9,
                                 port_mesh_size_m=200e-9, circle_points=17)


if __name__ == "__main__":
    unittest.main()
