"""Conforming donor-patch geometry checks. Author: Athul Prem | _AP_pnc."""

import unittest

import numpy as np
from skfem import MeshTet

from tools.central_patch_refinement_AP_pnc import make_central_patch_control


def _extrude(xy, triangles):
    """Construct a two-layer native prism partition for a small exact fixture."""
    n = len(xy)
    z = np.array([-1., 0., 1.])
    a, b, c = np.asarray(triangles).T
    layer = np.hstack(([a, b, c, c + n], [a, b, b + n, c + n], [a, a + n, b + n, c + n]))
    points = np.column_stack((np.tile(xy, (len(z), 1)), np.repeat(z, n)))
    return MeshTet(points.T * 1e-7, np.hstack((layer, layer + n))).oriented()


def _cells(mesh, selected):
    return {tuple(sorted(map(tuple, cell))) for cell in mesh.p[:, mesh.t[:, selected]].transpose(2, 1, 0)}


class CentralPatchRefinementTests(unittest.TestCase):
    def setUp(self):
        xy = np.array([(x, y) for x in range(7) for y in (0., 1.)])
        triangles = [(i, i + offset, i + 3) for i in range(0, 12, 2) for offset in (1, 2)]
        self.coarse = _extrude(xy, triangles)
        points = xy.tolist()
        fine = triangles.copy()
        # Split a frozen/active interface edge and an independently usable
        # diagonal in the central patch.
        for a, b in ((4, 5), (6, 9)):
            midpoint = len(points)
            points.append(((xy[a] + xy[b]) / 2).tolist())
            for index, triangle in enumerate(fine.copy()):
                if a in triangle and b in triangle:
                    opposite, = set(triangle) - {a, b}
                    fine[index] = (*sorted((a, opposite)), midpoint)
                    fine.append((*sorted((b, opposite)), midpoint))
        self.fine = _extrude(np.asarray(points), fine)

    def test_patch_preserves_crossing_cells_and_reuses_exact_donor_cells(self):
        control, report = make_central_patch_control(self.coarse, self.fine, (1.5e-7, 4.5e-7))
        self.assertEqual(report["frozen_cross_cut_planar_triangles"], [2, 2])
        self.assertEqual(report["wholly_inside_coarse_triangles_frozen_by_conformity"], 1)
        self.assertEqual(report["replaced_coarse_planar_triangles"], 2)
        self.assertEqual(report["control_planar_triangles"], 14)
        self.assertEqual(report["mesh_tetrahedra"], 84)
        self.assertEqual(report["end_face_dofs"], {"xleft": 45, "xright": 45})
        np.testing.assert_array_equal(control.p[:, :self.coarse.nvertices], self.coarse.p)

        def in_changed_strip(mesh):
            x = mesh.p[0, mesh.t]
            return (x.min(axis=0) >= 3e-7) & (x.max(axis=0) <= 4e-7)

        selected = in_changed_strip(control)
        self.assertEqual(_cells(control, selected), _cells(self.fine, in_changed_strip(self.fine)))
        self.assertEqual(_cells(control, ~selected), _cells(self.coarse, ~in_changed_strip(self.coarse)))
        np.testing.assert_allclose(report["control_solid_volume_m3"], report["coarse_solid_volume_m3"],
                                   rtol=1e-14, atol=0)

    def test_complement_assigns_crossing_parents_and_reconstructs_fine_topology(self):
        cuts = (1.5e-7, 4.5e-7)
        central, central_report = make_central_patch_control(self.coarse, self.fine, cuts)
        remainder, remainder_report = make_central_patch_control(self.coarse, self.fine, cuts, region="complement")
        central_partition = central_report["coarse_parent_partition"]
        remainder_partition = remainder_report["coarse_parent_partition"]
        for key in ("central_parent_indices", "complement_parent_indices",
                    "central_active_mask_sha256", "complement_active_mask_sha256", "fine_donor_topology_sha256"):
            self.assertEqual(central_partition[key], remainder_partition[key])
        central_parents = set(central_partition["central_parent_indices"])
        remainder_parents = set(central_partition["complement_parent_indices"])
        self.assertFalse(central_parents & remainder_parents)
        self.assertEqual(central_parents | remainder_parents, set(range(12)))
        crossing = [{2, 3}, {8, 9}]
        self.assertEqual([set(ids) for ids in remainder_partition["nominal_cut_crossing_parent_indices"]], crossing)
        self.assertTrue(set.union(*crossing) <= remainder_parents)
        self.assertEqual(remainder_partition["replaced_cut_crossing_planar_triangles"], [1, 0])
        self.assertEqual(remainder_report["frozen_cross_cut_planar_triangles"], [1, 2])

        coarse_cells, fine_cells, central_cells, remainder_cells = (
            _cells(mesh, slice(None)) for mesh in (self.coarse, self.fine, central, remainder))
        removed_central, added_central = coarse_cells - central_cells, central_cells - coarse_cells
        removed_remainder, added_remainder = coarse_cells - remainder_cells, remainder_cells - coarse_cells
        self.assertFalse(removed_central & removed_remainder)
        self.assertFalse(added_central & added_remainder)
        self.assertEqual((coarse_cells - removed_central - removed_remainder) | added_central | added_remainder,
                         fine_cells)

    def test_reject_non_nested_donor_and_empty_patch(self):
        with self.assertRaisesRegex(ValueError, "share coarse vertex coordinates"):
            make_central_patch_control(self.fine, self.coarse, (1.5e-7, 4.5e-7))
        with self.assertRaisesRegex(ValueError, "No nested refinement"):
            make_central_patch_control(self.coarse, self.fine, (3.1e-7, 3.9e-7))
        with self.assertRaisesRegex(ValueError, "central or complement"):
            make_central_patch_control(self.coarse, self.fine, (1.5e-7, 4.5e-7), region="unknown")


if __name__ == "__main__":
    unittest.main()
