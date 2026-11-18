"""Saved comparison presentation tests. Author: Athul Prem | _AP_pnc."""

import math
import unittest

from tools.inspect_scattering_comparison_AP_pnc import comparison_lines


class InspectScatteringComparisonTests(unittest.TestCase):
    def test_aligned_pair_and_relative_power_use_the_saved_reference_order(self):
        summary = {
            "schema_version": 1,
            "status": "compared",
            "physical_inputs": {"frequency_hz": 1e9, "incident_side": "left"},
            "matched_channels": {
                "incoming": {"reference_roots": [17, 35], "candidate_roots": [45, 27]},
                "reflection": {"reference_roots": [9, 23], "candidate_roots": [19, 33]},
                "transmission": {"reference_roots": [17, 35], "candidate_roots": [45, 27]},
            },
            "blocks": {
                "reflection": {
                    "reference": [[[0, 0], [1e-10, 0]], [[0, 0], [0, 0]]],
                    "candidate_aligned": [[[0.125, 0], [2e-10, 0]], [[0, 0], [0, 0]]],
                    "reference_power": [0, 1e-20],
                    "candidate_power": [0.015625, 4e-20],
                },
                "transmission": {
                    "reference": [[[0, 0], [1, 0]], [[-0.99, 0], [0, 0]]],
                    "candidate_aligned": [[[0, 0], [1, 0]], [[0, -0.98], [0, 0]]],
                    "reference_power": [0.99**2, 1],
                    "candidate_power": [0.98**2, 1],
                },
            },
            "maximum_power_change": 0.99**2 - 0.98**2,
            "maximum_complex_change": math.hypot(0.99, 0.98),
        }

        lines = comparison_lines(summary)
        self.assertIn("  Incident channel 1, reference root 17, candidate root 45", lines)
        self.assertIn("  Outgoing transmission channel 2, reference root 35, candidate root 27", lines)
        self.assertIn("  Aligned magnitudes: reference 0.99, candidate 0.98", lines)
        self.assertIn("  Absolute magnitude-only mismatch: 0.01", lines)
        self.assertIn("  Wrapped phase change (candidate minus reference): +90 deg", lines)
        reflection = [line.split() for line in lines if line.startswith("Reflection")]
        self.assertEqual(reflection[0], ["Reflection", "1", "17", "45", "0", "0.015625", "0.015625", "+1.5625",
                                         "undefined", "(zero", "reference)"])
        self.assertEqual(reflection[1], ["Reflection", "2", "35", "27", "1e-20", "4e-20", "3e-20", "+3e-18", "+300%"])
        transmission = next(line.split() for line in lines if line.startswith("Transmission"))
        self.assertEqual(transmission[-3:], ["0.0197", "-1.97", "-2.009999%"])


if __name__ == "__main__":
    unittest.main()
