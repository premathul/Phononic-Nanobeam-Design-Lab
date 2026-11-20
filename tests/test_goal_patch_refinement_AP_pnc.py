"""Goal-selected refinement geometry checks. Author: Athul Prem | _AP_pnc."""

import unittest

import numpy as np

import test_central_patch_refinement_AP_pnc as fixtures
from tools.goal_patch_refinement_AP_pnc import make_goal_patch_control


class GoalPatchRefinementTests(unittest.TestCase):
    def test_centroid_selection_closes_split_edges_and_keeps_original_cells(self):
        fixture = fixtures.CentralPatchRefinementTests()
        fixture.setUp()
        centers = np.r_[np.linspace(0.1, 1.6, 12), 2.4, 3.6, np.linspace(4.4, 5.9, 12)] * 1e-7
        control, report = make_goal_patch_control(
            fixture.coarse, fixture.fine, (0.5e-7, 5.5e-7), centers, region="cavity_13_14"
        )
        parents = report["goal_parent_partition"]
        self.assertEqual(parents["hole_bins"], [13, 14])
        self.assertEqual(parents["selected_parent_indices"], [4, 5, 6, 7])
        self.assertEqual(parents["selected_refinable_parent_indices"], [4, 6, 7])
        self.assertEqual(parents["selected_parent_indices_frozen_by_conformity"], [4])
        self.assertEqual(parents["replaced_parent_indices"], [6, 7])
        self.assertEqual(report["retained_coarse_tetrahedra"], 60)
        self.assertEqual(report["inserted_fine_donor_tetrahedra"], 24)
        self.assertGreater(report["unchanged_patch_frontier_p2_dofs"], 0)

        def in_replaced_strip(mesh):
            x = mesh.p[0, mesh.t]
            return (x.min(axis=0) >= 3e-7) & (x.max(axis=0) <= 4e-7)

        selected = in_replaced_strip(control)
        self.assertEqual(fixtures._cells(control, selected),
                         fixtures._cells(fixture.fine, in_replaced_strip(fixture.fine)))
        self.assertEqual(fixtures._cells(control, ~selected),
                         fixtures._cells(fixture.coarse, ~in_replaced_strip(fixture.coarse)))
        self.assertEqual(report["end_face_dofs"], {"xleft": 45, "xright": 45})

    def test_outer_taper_bins_are_restricted_to_original_central_parents(self):
        fixture = fixtures.CentralPatchRefinementTests()
        fixture.setUp()
        centers = np.r_[np.linspace(0.1, 1.15, 8), 2.05, np.linspace(2.7, 3.0, 8),
                        3.59, np.linspace(4.0, 5.9, 8)] * 1e-7
        _, report = make_goal_patch_control(
            fixture.coarse, fixture.fine, (1.5e-7, 4.5e-7), centers, region="outer_taper_09_18"
        )
        parents = report["goal_parent_partition"]
        self.assertEqual(parents["hole_bins"], [9, 18])
        self.assertEqual(parents["bins_parent_count_outside_original_central_mask"], 2)
        self.assertEqual(parents["selected_parent_indices"], [6, 7])
        self.assertEqual(parents["replaced_parent_indices"], [6, 7])


if __name__ == "__main__":
    unittest.main()
