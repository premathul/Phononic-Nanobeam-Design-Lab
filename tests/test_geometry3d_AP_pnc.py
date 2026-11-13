"""Geometry verification for the 3D beam. Author: Athul Prem | _AP_pnc."""

from dataclasses import replace
from importlib.util import find_spec
from math import pi
from pathlib import Path
import sys
import unittest

import numpy as np
from skfem import Basis, ElementTetP1, FacetBasis

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from phononic.geometry_AP_pnc import (
    BeamGeometry,
    Hole,
    make_perforated_mesh,
    make_straight_mesh,
    read_hole_pattern,
    validate_hole_clearances,
)


class Geometry3DTests(unittest.TestCase):
    def assert_port_geometry(self, mesh, geometry):
        tagged = np.concatenate([mesh.boundaries[name] for name in ("xleft", "xright", "free")])
        np.testing.assert_array_equal(np.sort(tagged), mesh.boundary_facets())
        port_area = geometry.width_m * geometry.thickness_m
        for name, sign, x in [("xleft", -1, 0.0), ("xright", 1, geometry.length_m)]:
            port = FacetBasis(mesh, ElementTetP1(), facets=mesh.boundaries[name])
            self.assertAlmostEqual(port.dx.sum() / port_area, 1, places=12)
            expected = np.broadcast_to(np.array([sign, 0, 0])[:, None, None], port.normals.shape)
            np.testing.assert_allclose(port.normals, expected, atol=1e-12)
            np.testing.assert_allclose(port.global_coordinates()[0] / geometry.length_m,
                                       x / geometry.length_m, atol=1e-12)

    def test_straight_beam_si_volume_and_end_ports(self):
        for thickness_nm in (100, 200, 1000):
            with self.subTest(thickness_nm=thickness_nm):
                geometry = BeamGeometry(2e-6, 1e-6, thickness_nm * 1e-9)
                mesh = make_straight_mesh(geometry, cells=(4, 3, 2))
                expected_volume = geometry.length_m * geometry.width_m * geometry.thickness_m
                self.assertAlmostEqual(Basis(mesh, ElementTetP1()).dx.sum() / expected_volume,
                                       1, places=12)
                np.testing.assert_array_equal(mesh.orientation(), 1)
                np.testing.assert_array_equal(mesh.p.min(axis=1),
                                              [0, -geometry.width_m / 2, -geometry.thickness_m / 2])
                np.testing.assert_array_equal(mesh.p.max(axis=1),
                                              [geometry.length_m, geometry.width_m / 2,
                                               geometry.thickness_m / 2])
                self.assert_port_geometry(mesh, geometry)

    def test_reject_nonphysical_mesh_inputs(self):
        for length in (float("nan"), float("inf"), 0, -1e-6):
            with self.subTest(length=length), self.assertRaises(ValueError):
                BeamGeometry(length, 1e-6, 200e-9)
        geometry = BeamGeometry(2e-6, 1e-6, 200e-9)
        for cells in ((1, 1), (1, 0, 1), (1, 2.0, 1), (1, True, 1)):
            with self.subTest(cells=cells), self.assertRaises(ValueError):
                make_straight_mesh(geometry, cells=cells)
        with self.assertRaisesRegex(ValueError, "adjacent holes"):
            validate_hole_clearances(geometry, [Hole(300e-9, 200e-9), Hole(400e-9, 200e-9)])

    def test_historical_csv_and_example(self):
        pattern = read_hole_pattern(ROOT / "data/geometry/hole_geometry_derived_AP_pnc.csv")
        self.assertEqual(pattern.source_sha256,
                         "2c44d944ca9a161a5571a145d129972bb808e3e038f20765ea915f5ce0020771")
        self.assertEqual(len(pattern.holes), 26)
        self.assertAlmostEqual(pattern.holes[0].x_m * 1e9, -6700)
        self.assertAlmostEqual(pattern.holes[-1].x_m * 1e9, 6700)
        self.assertAlmostEqual(pattern.holes[0].diameter_m * 1e9, 250)
        self.assertEqual(pattern.holes[0].region, "Mirror")
        # Historical domain extends 1500 nm beyond the two outer hole rims.
        geometry = BeamGeometry(16650e-9, 1000e-9, 200e-9)
        placed = [replace(hole, x_m=hole.x_m + geometry.length_m / 2)
                  for hole in pattern.holes]
        clearances = validate_hole_clearances(geometry, placed)
        self.assertAlmostEqual(clearances["min_end_ligament_m"] * 1e9, 1500)
        self.assertAlmostEqual(clearances["min_side_ligament_m"] * 1e9, 362.5)
        self.assertAlmostEqual(clearances["min_interhole_gap_m"] * 1e9, 225)

        example = read_hole_pattern(ROOT / "data/examples/half_array_AP_pnc.csv")
        self.assertEqual(len(example.holes), 8)
        self.assertAlmostEqual(example.holes[-1].x_m * 1e9, -250)

    @unittest.skipUnless(find_spec("gmsh") and find_spec("meshio"), "requires gmsh and meshio")
    def test_two_cylindrical_through_holes_remove_expected_volume(self):
        geometry = BeamGeometry(1e-6, 600e-9, 200e-9)
        holes = [Hole(300e-9, 180e-9), Hole(700e-9, 180e-9)]
        box_volume = geometry.length_m * geometry.width_m * geometry.thickness_m
        cut_volume = 2 * pi * (90e-9)**2 * geometry.thickness_m
        coarse = make_perforated_mesh(geometry, holes, mesh_size_m=80e-9)
        fine = make_perforated_mesh(geometry, holes, mesh_size_m=40e-9)
        self.assert_port_geometry(fine, geometry)
        np.testing.assert_array_equal(fine.orientation(), 1)
        coarse_removed = box_volume - Basis(coarse, ElementTetP1()).dx.sum()
        fine_removed = box_volume - Basis(fine, ElementTetP1()).dx.sum()
        self.assertLess(abs(fine_removed / cut_volume - 1), 0.04)
        self.assertLess(abs(fine_removed - cut_volume), abs(coarse_removed - cut_volume))


if __name__ == "__main__":
    unittest.main()
