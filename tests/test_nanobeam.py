import math
import sys
import unittest
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from nanobeam import Hole, diatomic_branches, inspect_geometry, read_holes


class GeometryTests(unittest.TestCase):
    def test_known_adjacent_spacing(self):
        result = inspect_geometry([Hole(-6700, 250), Hole(-6150, 250)], 600)
        self.assertEqual(result["adjacent"][0]["pitch_nm"], 550)
        self.assertEqual(result["adjacent"][0]["edge_gap_nm"], 300)
        self.assertFalse(result["issues"])

    def test_overlap_and_sidewall(self):
        result = inspect_geometry([Hole(0, 700), Hole(300, 200)], 600)
        self.assertEqual(len(result["issues"]), 2)

    def test_example_partial_array(self):
        holes = read_holes(Path(__file__).resolve().parents[1] / "examples" / "half_array.csv")
        result = inspect_geometry(holes, 600)
        self.assertEqual(result["hole_count"], 8)
        self.assertTrue(all(row["edge_gap_nm"] > 0 for row in result["adjacent"]))


class ChainTests(unittest.TestCase):
    def test_acoustic_zero_and_zone_edge(self):
        a, b, spring, cell = 1e-18, 2e-18, 1., 1e-6
        low, high = diatomic_branches(np.array([0., math.pi / cell]), cell, a, b, spring)
        self.assertAlmostEqual(low[0], 0., delta=1e-4)
        self.assertAlmostEqual(high[0], math.sqrt(2*spring*(1/a+1/b))/(2*math.pi), delta=1e-4)
        self.assertLess(low[-1], high[-1])


if __name__ == "__main__":
    unittest.main()
