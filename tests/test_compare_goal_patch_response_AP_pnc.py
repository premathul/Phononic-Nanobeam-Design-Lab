"""Verify goal-pair prediction bookkeeping. Author: Athul Prem | _AP_pnc."""

import unittest

import numpy as np

from tools import compare_goal_patch_response_AP_pnc as goal


def edge(delta, offset):
    rows = []
    for drive, change in enumerate(delta, 1):
        rows.append({
            "reference": {"drive": drive, "native_root_id": 10 + drive, "transmission": 0.5},
            "candidate": {"drive": drive, "native_root_id": offset + drive, "transmission": 0.5 + change / 100},
            "candidate_minus_reference_percentage_points": {"transmission": change},
        })
    return {"drive_power_fractions": rows}


class GoalPatchComparisonTests(unittest.TestCase):
    def setUp(self):
        self.work = {
            "indices": [4, 9, 17, 30, 44],
            "hole_numbers": [13, 13, 14, 14, 18],
            "delta_T_percentage_points_by_drive": [
                [1, 2, 3, 4, 5], [100, 100, 100, 100, 100], [40, 30, 20, 10, 0],
                [0, 0, -1, 0, 1], [0, 0, -2, 0, 8],
            ],
        }
        self.cavity = {"selected_parent_indices": [4, 9, 30], "replaced_parent_indices": [4, 30],
                       "hole_bins": [13, 14]}

    def test_sums_only_replaced_parents_and_preserves_native_drive_powers(self):
        central, pair = edge([8, 8, 8, 8, 8], 50), edge([-1, 4, -3, 2, 6], 100)
        compared = goal._compare_prediction(self.work, self.cavity, pair, central, added_body_dofs=2500)
        selection = compared["selection"]
        self.assertEqual(compared["added_body_dofs"], 2500)
        self.assertEqual(selection["selected_parent_count"], 3)
        self.assertEqual(selection["replaced_parent_count"], 2)
        self.assertEqual(selection["full_bin_unreplaced_parent_indices"], [9, 17])
        rows = compared["per_drive"]
        self.assertEqual([row["prediction_percentage_points"] for row in rows], [1, 2, 2, 4, 6])
        self.assertEqual([row["measured_change_percentage_points"] for row in rows], [-1, 4, -3, 2, 6])
        drive3 = rows[2]
        self.assertEqual(drive3["pair_native_root"], 103)
        self.assertAlmostEqual(drive3["unreplaced_prediction_percentage_points"], 120)
        self.assertAlmostEqual(drive3["full_bin_work_percentage_points"], 122)
        self.assertAlmostEqual(drive3["coarse_transmission_fraction"], 0.5)
        self.assertAlmostEqual(drive3["predicted_transmission_fraction"], 0.52)
        self.assertAlmostEqual(drive3["measured_transmission_fraction"], 0.47)
        self.assertAlmostEqual(drive3["prediction_percentage_points_per_1000_added_body_dofs"], 0.8)
        self.assertAlmostEqual(drive3["measured_change_percentage_points_per_1000_added_body_dofs"], -1.2)
        self.assertAlmostEqual(drive3["full_central_minus_pair_percentage_points"], 11)
        self.assertFalse(drive3["sign_matches"])
        self.assertTrue(rows[4]["sign_matches"])
        self.assertAlmostEqual(compared["equal_incoherent_mean"]["prediction_percentage_points"], 3.0)
        self.assertAlmostEqual(compared["equal_incoherent_mean"]["measured_change_percentage_points"], 1.6)
        self.assertAlmostEqual(
            compared["equal_incoherent_mean"]["prediction_percentage_points_per_1000_added_body_dofs"], 1.2)
        self.assertAlmostEqual(
            compared["equal_incoherent_mean"]["measured_change_percentage_points_per_1000_added_body_dofs"], 0.64)

    def test_invalid_parent_masks_fail_closed(self):
        invalid_masks = [
            {"selected_parent_indices": [4, 9, 30, 43], "replaced_parent_indices": [4, 43]},
            {"replaced_parent_indices": [17, 30]},
            {"selected_parent_indices": [4, 9, 30, 44], "replaced_parent_indices": [4, 44]},
            {"replaced_parent_indices": [4, 4]},
        ]
        for invalid in invalid_masks:
            with self.subTest(invalid=invalid), self.assertRaises(ValueError):
                goal._parent_predictions(self.work, {**self.cavity, **invalid})
        bad = {**self.work, "delta_T_percentage_points_by_drive": np.full((5, 5), np.nan)}
        with self.assertRaisesRegex(ValueError, "Invalid five-drive parent work"):
            goal._parent_predictions(bad, self.cavity)

    def test_signed_ranking_and_magnitude_are_tested_separately(self):
        central = edge([8, 8, 8, 8, 8], 50)
        outer_mask = {"selected_parent_indices": [44], "replaced_parent_indices": [44], "hole_bins": [9, 18]}
        actual = {"cavity": (self.cavity, [0, 0, -5, 0, 2], 2000),
                  "outer": (outer_mask, [0, 0, 3, 0, 1], 500)}
        compared = {region: goal._compare_prediction(self.work, mask, edge(response, 100), central,
                                                     added_body_dofs=added)
                    for region, (mask, response, added) in actual.items()}
        drive3, drive5 = goal._drive_checks(compared)
        self.assertEqual(drive3["predicted_order_by_signed_gain"], ["cavity", "outer"])
        self.assertEqual(drive3["measured_order_by_signed_gain"], ["outer", "cavity"])
        self.assertFalse(drive3["signed_gain_ranking_matches"])
        self.assertIsNone(drive3["predicted_order_by_magnitude"])
        self.assertIsNone(drive3["magnitude_ranking_matches"])
        self.assertEqual(drive5["sign_matches_by_control"], {"cavity": True, "outer": True})
        self.assertFalse(drive5["signed_gain_ranking_matches"])
        self.assertFalse(drive5["magnitude_ranking_matches"])
        self.assertEqual(drive3["predicted_order_by_magnitude_per_1000_added_body_dofs"], ["outer", "cavity"])
        self.assertEqual(drive3["measured_order_by_magnitude_per_1000_added_body_dofs"], ["outer", "cavity"])
        self.assertTrue(drive3["magnitude_ranking_matches_per_1000_added_body_dofs"])
        self.assertEqual(drive5["measured_order_by_signed_gain"], ["cavity", "outer"])
        self.assertEqual(drive5["measured_order_by_signed_gain_per_1000_added_body_dofs"], ["outer", "cavity"])
        self.assertTrue(drive5["signed_gain_ranking_matches_per_1000_added_body_dofs"])

    def test_nonpositive_added_dof_count_is_rejected(self):
        with self.assertRaisesRegex(ValueError, "must add body DOFs"):
            goal._compare_prediction(self.work, self.cavity, {}, {}, added_body_dofs=0)


if __name__ == "__main__":
    unittest.main()
