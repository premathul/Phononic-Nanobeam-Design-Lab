"""Offline checks of the uniform-guide diagnostic. Author: Athul Prem | _AP_pnc."""

import csv
from hashlib import sha256
import json
from pathlib import Path
import unittest

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
EVIDENCE = ROOT / "verification/wave_resolution_diagnostic_AP_pnc.json"


def unpack(values):
    pairs = np.asarray(values)
    return pairs[..., 0] + 1j * pairs[..., 1]


class WaveResolutionEvidenceTests(unittest.TestCase):
    def test_uniform_guide_power_phase_and_fixed_port_controls(self):
        data = json.loads(EVIDENCE.read_bytes())
        self.assertEqual(data["status"], "completed_diagnostic")
        self.assertEqual(data["frequency_hz"], 1e9)
        self.assertEqual(data["element_order"], 2)
        np.testing.assert_allclose([data["length_m"], data["width_m"], data["thickness_m"]],
                                   np.array([1200, 1000, 200]) * 1e-9, rtol=1e-14, atol=0)
        self.assertEqual(data["material"]["temperature_policy"], "fixed_property_control")
        np.testing.assert_array_equal(data["material"]["rotation"], np.eye(3))
        source = "tools/diagnose_wave_resolution_AP_pnc.py"
        self.assertEqual(sha256((ROOT / source).read_bytes()).hexdigest(), data["implementation"]["source_sha256"][source])
        table = EVIDENCE.with_suffix(".csv")
        self.assertEqual(sha256(table.read_bytes()).hexdigest(), data["channel_csv_sha256"])
        with table.open(encoding="utf-8", newline="") as stream:
            csv_rows = list(csv.DictReader(stream))
        self.assertEqual(len(csv_rows), 20)
        planned = {
            "axial_200_thickness_1": [6, 5, 1], "axial_100_thickness_1": [12, 5, 1],
            "axial_200_thickness_2": [6, 5, 2], "axial_100_thickness_2": [12, 5, 2],
        }
        self.assertEqual(set(data["cases"]), set(planned))
        wavevectors = {}
        for name, cells in planned.items():
            with self.subTest(case=name):
                case = data["cases"][name]
                self.assertEqual(case["status"], "solved")
                self.assertEqual(case["cells"], cells)
                nx, ny, nz = cells
                self.assertEqual(case["displacement_dofs"], 3 * (2 * nx + 1) * (2 * ny + 1) * (2 * nz + 1))
                self.assertEqual(case["port_dofs"], 3 * (2 * ny + 1) * (2 * nz + 1))
                channels = case["channels"]
                self.assertEqual(len(channels), 5)
                self.assertTrue(all(row["incident_side"] == "left" for row in channels))
                self.assertEqual([row["safe_root_index"] for row in channels], case["incident_root_indices"])
                self.assertEqual(case["incident_root_indices"], case["transmission_root_indices"])
                k = np.asarray([row["wavenumber_rad_m"] for row in channels])
                wavevectors[name] = k
                R, T = (unpack(case[f"{quantity}_matrix"]) for quantity in ("reflection", "transmission"))
                self.assertEqual(R.shape, (5, 5))
                self.assertEqual(T.shape, (5, 5))
                # Propagating amplitudes are normalized to one watt in both directions.
                np.testing.assert_allclose([row["incident_power_W"] for row in channels], 1, atol=1e-12, rtol=0)
                for matrix, coefficient in ((R, "R"), (T, "T")):
                    powers = (abs(matrix) ** 2).sum(axis=0)
                    np.testing.assert_allclose(powers, [row[coefficient] for row in channels], atol=1e-12, rtol=0)
                    np.testing.assert_allclose(powers, [row[f"{coefficient}_flux"] for row in channels], atol=1e-12, rtol=0)
                np.testing.assert_allclose((abs(R) ** 2 + abs(T) ** 2).sum(axis=0), 1, atol=1e-12, rtol=0)
                expected = np.exp(1j * k * data["length_m"])
                np.testing.assert_allclose(unpack(case["expected_transmission_diagonal"]), expected, atol=1e-13, rtol=0)
                phase = np.angle(np.diag(T) / expected)
                np.testing.assert_allclose(phase, [row["phase_error_rad"] for row in channels], atol=1e-13, rtol=0)
                self.assertAlmostEqual(max(abs(phase)), case["maximum_absolute_phase_error_rad"], places=12)
                self.assertAlmostEqual(abs(T - np.diag(expected)).max(), case["maximum_transmission_complex_error"], places=12)
                rows = [row for row in csv_rows if row["case"] == name]
                self.assertEqual([int(row["mode_index"]) for row in rows], [1, 2, 3, 4, 5])
                np.testing.assert_allclose([float(row["phase_error_rad"]) for row in rows], phase, atol=1e-13, rtol=0)

        for layers in (1, 2):
            first, second = (f"axial_{spacing}_thickness_{layers}" for spacing in (200, 100))
            np.testing.assert_allclose(wavevectors[first], wavevectors[second], atol=0, rtol=1e-12)
            self.assertEqual(data["comparisons"][f"{first}_to_{second}"]["maximum_absolute_safe_phase_change_rad"], 0)
        finest = data["cases"]["axial_100_thickness_2"]
        interpretation = data["interpretation"]
        self.assertEqual(interpretation["finest_case"], "axial_100_thickness_2")
        self.assertEqual(interpretation["diagnostic_reference"], 0.01)
        self.assertGreater(finest["maximum_absolute_phase_error_rad"], 0.01)
        self.assertGreater(finest["maximum_transmission_complex_error"], 0.01)
        self.assertFalse(interpretation["finest_phase_error_rad_below_reference"])
        self.assertFalse(interpretation["finest_transmission_complex_error_below_reference"])


if __name__ == "__main__":
    unittest.main()
