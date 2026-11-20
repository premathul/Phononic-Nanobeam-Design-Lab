"""Check parent-to-bin work aggregation. Author: Athul Prem | _AP_pnc."""

import unittest

import numpy as np

from tools.localize_reflection_change_AP_pnc import HOLE_BINS, HOLE_PAIRS, _validate_result, _work_rows


class ReflectionWorkTests(unittest.TestCase):
    def test_signed_work_aggregates_parents_into_holes_and_mirror_pairs(self):
        # Keep independent incident and reflected columns, an unchanged group,
        # and repeated hole ownership so the aggregation can detect swaps.
        rng = np.random.default_rng(763)
        holes = np.array([13, 9, 13, 18, 14])
        change = rng.standard_normal((1 + len(holes), 5, 5)) + 1j * rng.standard_normal((1 + len(holes), 5, 5))
        weights = rng.standard_normal((5, 5)) + 1j * rng.standard_normal((5, 5))
        signed, bins, pairs = _work_rows(change, weights, holes)
        np.testing.assert_allclose(signed, -100 * np.einsum("rj,prj->pj", weights, change).real)
        np.testing.assert_allclose(bins[0]["delta_T_percentage_points_by_drive"], signed[0])
        np.testing.assert_allclose(bins[HOLE_BINS.index(13)]["delta_T_percentage_points_by_drive"], signed[[1, 3]].sum(axis=0))
        np.testing.assert_allclose(pairs[HOLE_PAIRS.index((9, 18))]["delta_T_percentage_points_by_drive"], signed[[2, 4]].sum(axis=0))
        np.testing.assert_array_equal(bins[HOLE_BINS.index(12)]["delta_T_percentage_points_by_drive"], np.zeros(5))
        result = {"parent_work": {"indices": [0, 1, 7, 9, 12], "hole_numbers": holes.tolist(),
                                  "delta_T_percentage_points_by_drive": signed[1:].tolist()},
                  "partition": {"coarse_parent_count": 13, "changed_parent_count": len(holes)},
                  "incident_roots": [1, 2, 4, 8, 16], "reflected_roots": [3, 5, 6, 9, 15],
                  "bins": bins, "paired_hole_bins": pairs, "closure": {"synthetic": 0.0},
                  "end_correction_delta_T_percentage_points": np.zeros(5).tolist(),
                  "measured_delta_T_percentage_points": signed.sum(axis=0).tolist()}
        _validate_result(result)
        result["parent_work"]["delta_T_percentage_points_by_drive"][2][3] += 0.01
        with self.assertRaisesRegex(ValueError, "fails the measured transmission closure"):
            _validate_result(result)


if __name__ == "__main__":
    unittest.main()
