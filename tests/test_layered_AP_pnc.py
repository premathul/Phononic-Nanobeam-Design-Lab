"""Layered beam geometry and finite-element checks. Author: Athul Prem | _AP_pnc."""

from importlib.util import find_spec
from math import pi, sin
from pathlib import Path
import sys
import unittest

import numpy as np
from skfem import ElementTetP2, FacetBasis

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from phononic.elasticity_AP_pnc import assemble_elasticity
from phononic.geometry_AP_pnc import BeamGeometry, Hole
from phononic.layered_AP_pnc import make_layered_mesh
from phononic.material_AP_pnc import CubicMaterial
from phononic.perforated_AP_pnc import volume_diagnostics
from phononic.port_modes_AP_pnc import assemble_port


def face_coordinates(mesh, facets, axes):
    return {tuple(sorted(tuple(p) for p in face))
            for face in mesh.p[axes][:, mesh.facets[:, facets]].transpose(2, 1, 0)}


@unittest.skipUnless(find_spec("gmsh"), "requires gmsh")
class LayeredMeshTests(unittest.TestCase):
    geometry = BeamGeometry(1e-6, 600e-9, 200e-9)
    holes = (Hole(300e-9, 180e-9), Hole(700e-9, 180e-9))

    def mesh(self, h, layers, holes=None):
        return make_layered_mesh(self.geometry, self.holes if holes is None else holes, h * 1e-9,
                                 port_mesh_size_m=200e-9, thickness_layers=layers, circle_points=16)

    def test_layer_refinement_preserves_exact_footprint_and_polygon_volume(self):
        single = self.mesh(150, 1)
        double = self.mesh(150, 2)
        np.testing.assert_array_equal(np.unique(single.p[2]), [-100e-9, 100e-9])
        np.testing.assert_array_equal(np.unique(double.p[2]), [-100e-9, 0, 100e-9])
        vertices = single.p[:2, single.p[2] == -100e-9]
        for z in np.unique(double.p[2]):
            np.testing.assert_array_equal(double.p[:2, double.p[2] == z], vertices)
        floors = []
        for mesh in (single, double):
            exterior = np.sort(np.concatenate(list(mesh.boundaries.values())))
            np.testing.assert_array_equal(exterior, mesh.boundary_facets())
            np.testing.assert_array_equal(mesh.subdomains["solid"], np.arange(mesh.nelements))
            self.assertTrue(np.all(mesh.mapping().detA > 0))
            zs = np.searchsorted(np.unique(mesh.p[2]), mesh.p[2, mesh.t])
            np.testing.assert_array_equal(np.ptp(zs, axis=0), 1)
            floor = exterior[np.all(mesh.p[2, mesh.facets[:, exterior]] == -100e-9, axis=0)]
            floors.append(face_coordinates(mesh, floor, [0, 1]))
            self.assertEqual(mesh.nelements, 3 * len(floor) * (len(np.unique(mesh.p[2])) - 1))
            diagnostics = volume_diagnostics(mesh, self.geometry, self.holes)
            self.assertEqual(diagnostics["rim_segments_min"], 16)
            self.assertEqual(diagnostics["rim_segments_max"], 16)
            self.assertAlmostEqual(diagnostics["relative_removed_volume_error"], 1 - 16 * sin(2 * pi / 16) / (2 * pi), places=12)
        self.assertEqual(*floors)

    def test_xy_refinement_preserves_port_traces(self):
        coarse, fine = self.mesh(150, 2), self.mesh(105, 2)
        self.assertGreater(fine.nelements, coarse.nelements)
        reference = face_coordinates(coarse, coarse.boundaries["xleft"], [1, 2])
        self.assertEqual(len(reference), 12)
        for mesh in (coarse, fine):
            for side, sign in (("xleft", -1), ("xright", 1)):
                faces = mesh.boundaries[side]
                self.assertEqual(face_coordinates(mesh, faces, [1, 2]), reference)
                facet_basis = FacetBasis(mesh, ElementTetP2(), facets=faces)
                self.assertAlmostEqual(facet_basis.dx.sum() / (self.geometry.width_m * self.geometry.thickness_m), 1)
                np.testing.assert_allclose(facet_basis.normals[0], sign, atol=1e-12, rtol=0)

    def test_layered_affine_patch_and_port_assembly(self):
        mesh = self.mesh(150, 1, ())
        material = CubicMaterial()
        system = assemble_elasticity(mesh, material, max_dofs=2000)
        port = assemble_port(system, material, max_dofs=63)
        self.assertEqual(port.basis.N, 63)
        np.testing.assert_allclose(system.basis.doflocs[1:, port.left_dofs],
                                   system.basis.doflocs[1:, port.right_dofs], atol=1e-20, rtol=0)
        # For u=(x,0,0), epsilon_xx=1 and the stiffness energy is C11 times volume.
        displacement = np.zeros(system.basis.N)
        x_dofs = system.basis.split_indices()[0]
        displacement[x_dofs] = system.basis.doflocs[0, x_dofs]
        volume = self.geometry.length_m * self.geometry.width_m * self.geometry.thickness_m
        self.assertAlmostEqual((displacement @ (system.K @ displacement)) / (material.C11 * volume), 1, places=11)
        interior = system.basis.complement_dofs(system.basis.get_dofs())
        self.assertGreater(len(interior), 0)
        force = system.K @ displacement
        self.assertLess(np.linalg.norm(force[interior]) / np.linalg.norm(force), 1e-12)

    def test_regional_lead_sizes_preserve_central_elements_and_local_operators(self):
        interval = (150e-9, 850e-9)
        meshes = [make_layered_mesh(self.geometry, self.holes, 150e-9,
                                   port_mesh_size_m=200e-9, thickness_layers=2,
                                   circle_points=16, lead_mesh_size_m=lead_size,
                                   patterned_x_interval_m=interval)
                  for lead_size in (150e-9, 75e-9)]
        self.assertGreater(meshes[1].nelements, meshes[0].nelements)
        central = []
        for mesh in meshes:
            x = mesh.p[0, mesh.t]
            elements = np.flatnonzero((x.min(axis=0) >= interval[0])
                                      & (x.max(axis=0) <= interval[1]))
            self.assertGreater(len(elements), 0)
            # Local meshes contain only central element contributions; global
            # interface rows also receive lead contributions and may change.
            central.append(mesh.restrict(elements))
            for plane in interval:
                facets = np.flatnonzero(np.all(mesh.p[0, mesh.facets] == plane, axis=0))
                self.assertEqual(len(facets), 12)
                self.assertTrue(np.all(mesh.f2t[:, facets] >= 0))
            self.assertTrue(np.all(mesh.mapping().detA > 0))
            diagnostics = volume_diagnostics(mesh, self.geometry, self.holes)
            self.assertEqual(diagnostics["rim_segments_min"], 16)
            self.assertEqual(diagnostics["rim_segments_max"], 16)
            self.assertAlmostEqual(diagnostics["relative_removed_volume_error"],
                                   1 - 16 * sin(2 * pi / 16) / (2 * pi), places=12)
        np.testing.assert_array_equal(central[0].p, central[1].p)
        np.testing.assert_array_equal(central[0].t, central[1].t)
        systems = [assemble_elasticity(mesh, CubicMaterial(), max_dofs=10_000) for mesh in central]
        np.testing.assert_array_equal(systems[0].basis.doflocs, systems[1].basis.doflocs)
        for name in ("K", "M"):
            left, right = (getattr(system, name) for system in systems)
            np.testing.assert_array_equal(left.indptr, right.indptr)
            np.testing.assert_array_equal(left.indices, right.indices)
            np.testing.assert_array_equal(left.data, right.data)

        reference = self.mesh(150, 2)
        for mesh in meshes:
            np.testing.assert_array_equal(np.unique(mesh.p[2]), np.unique(reference.p[2]))
            for side in ("xleft", "xright"):
                self.assertEqual(face_coordinates(mesh, mesh.boundaries[side], [1, 2]),
                                 face_coordinates(reference, reference.boundaries[side], [1, 2]))

    def test_regional_uniform_mesh_affine_patch_and_port_assembly(self):
        mesh = make_layered_mesh(self.geometry, (), 180e-9, port_mesh_size_m=200e-9,
                                 thickness_layers=1, lead_mesh_size_m=90e-9,
                                 patterned_x_interval_m=(300e-9, 700e-9))
        material = CubicMaterial()
        system = assemble_elasticity(mesh, material, max_dofs=10_000)
        port = assemble_port(system, material, max_dofs=63)
        self.assertEqual(port.basis.N, 63)
        np.testing.assert_allclose(system.basis.doflocs[1:, port.left_dofs],
                                   system.basis.doflocs[1:, port.right_dofs], atol=1e-20, rtol=0)
        displacement = np.zeros(system.basis.N)
        x_dofs = system.basis.split_indices()[0]
        displacement[x_dofs] = system.basis.doflocs[0, x_dofs]
        volume = self.geometry.length_m * self.geometry.width_m * self.geometry.thickness_m
        self.assertAlmostEqual((displacement @ (system.K @ displacement)) / (material.C11 * volume),
                               1, places=11)
        interior = system.basis.complement_dofs(system.basis.get_dofs())
        force = system.K @ displacement
        self.assertLess(np.linalg.norm(force[interior]) / np.linalg.norm(force), 1e-12)

    def test_rejects_invalid_regional_mesh_inputs(self):
        valid = {"lead_mesh_size_m": 100e-9, "patterned_x_interval_m": (150e-9, 850e-9)}
        invalid = [({"lead_mesh_size_m": 100e-9}, "provided together"),
                   ({"patterned_x_interval_m": valid["patterned_x_interval_m"]}, "provided together")]
        invalid += [({**valid, "lead_mesh_size_m": value}, "positive and finite")
                    for value in (0, -1e-9, float("nan"), float("inf"), True)]
        invalid += [({**valid, "patterned_x_interval_m": value}, "patterned_x_interval_m")
                    for value in ((), (150e-9,), (150e-9, 850e-9, 900e-9), 5,
                                  (float("nan"), 850e-9), (True, 850e-9),
                                  (0, 850e-9), (150e-9, self.geometry.length_m),
                                  (850e-9, 150e-9), (300e-9, 850e-9),
                                  (self.holes[0].x_m - self.holes[0].diameter_m / 2, 850e-9))]
        for options, message in invalid:
            with self.subTest(options=options), self.assertRaisesRegex(ValueError, message):
                make_layered_mesh(self.geometry, self.holes, 150e-9,
                                  port_mesh_size_m=200e-9, thickness_layers=1, **options)

    def test_rejects_invalid_layers_and_port_budget(self):
        with self.assertRaisesRegex(ValueError, "thickness_layers"):
            self.mesh(150, 0)
        with self.assertRaisesRegex(ValueError, "105 DOFs"):
            make_layered_mesh(self.geometry, self.holes, 150e-9, port_mesh_size_m=200e-9,
                              thickness_layers=2, max_port_dofs=104)


if __name__ == "__main__":
    unittest.main()
