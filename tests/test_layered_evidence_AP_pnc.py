"""Offline checks for the layered mesh controls. Author: Athul Prem | _AP_pnc."""

import csv
from hashlib import sha256
import json
from pathlib import Path
import unittest

import numpy as np


EVIDENCE = Path(__file__).resolve().parents[1] / "verification/layered_mesh_acceptance_AP_pnc.json"


def unpack(values):
    pairs = np.asarray(values)
    return pairs[..., 0] + 1j * pairs[..., 1]


class LayeredEvidenceTests(unittest.TestCase):
    def test_saved_controls_and_scattering_comparisons(self):
        record = json.loads(EVIDENCE.read_bytes())
        self.assertEqual(record["status"], "completed_diagnostic")
        table = EVIDENCE.with_suffix(".csv")
        self.assertEqual(sha256(table.read_bytes()).hexdigest(), record["channel_csv_sha256"])
        with table.open(encoding="utf-8", newline="") as stream:
            csv_rows = list(csv.DictReader(stream))
        self.assertEqual(len(csv_rows), 20)
        planned = {"uniform_h105": 7935, "two_hole_h105": 13485, "two_hole_h80": 17415,
                   "uniform_h60": 21915, "two_hole_h60": 26025}
        self.assertEqual(set(record["cases"]), set(planned))
        fixed_k = [row["wavenumber_rad_m"] for row in record["cases"]["uniform_h105"]["channels"]]
        for name, ndof in planned.items():
            with self.subTest(case=name):
                case = record["cases"][name]
                config = case["configuration"]
                self.assertEqual(config["analysis"]["frequency_GHz"], 1)
                self.assertEqual(config["analysis"]["incident_side"], "left")
                self.assertEqual(config["material"]["temperature_policy"], "fixed_property_control")
                self.assertEqual(config["mesh"]["kind"], "layered")
                self.assertEqual(config["mesh"]["element_order"], 2)
                self.assertEqual(config["mesh"]["thickness_layers"], 2)
                self.assertEqual(config["mesh"]["circle_resolution"], 48)
                self.assertEqual(config["mesh"]["port_size_nm"], 150)
                self.assertEqual(case["metrics"]["ndof"], ndof)
                self.assertEqual(case["metrics"]["port_dofs"], 135)
                self.assertEqual(case["execution"]["status"], "completed")
                self.assertTrue(case["metrics"]["power_balance_passed"])
                rows = [row for row in csv_rows if row["case"] == name]
                channels = case["channels"]
                self.assertEqual([int(row["incident_mode_order"]) for row in rows], [1, 2, 3, 4])
                self.assertEqual([row["wavenumber_rad_m"] for row in channels], fixed_k)
                self.assertEqual(case["root_indices"]["incident"], case["root_indices"]["transmission"])
                for quantity in ("reflection", "transmission"):
                    amplitude = unpack(case["amplitudes"][quantity])
                    self.assertEqual(amplitude.shape, (4, 4))
                    power = (abs(amplitude)**2).sum(axis=0)
                    for values in ([row[quantity] for row in channels],
                                   [row[f"{quantity}_flux"] for row in channels],
                                   [float(row[quantity]) for row in rows]):
                        np.testing.assert_allclose(power, values, atol=1e-12, rtol=0)

        for name, reference in record["uniform_same_port_reference"].items():
            with self.subTest(uniform=name):
                case = record["cases"][name]
                self.assertIsNone(case["hole_pattern_provenance"])
                R, T = (unpack(case["amplitudes"][quantity]) for quantity in ("reflection", "transmission"))
                expected = np.exp(1j * np.asarray(fixed_k) * case["configuration"]["geometry"]["length_nm"] * 1e-9)
                np.testing.assert_allclose(unpack(reference["expected_transmission_amplitudes"]), np.diag(expected),
                                           atol=1e-13, rtol=0)
                phase = np.angle(np.diag(T) / expected)
                np.testing.assert_allclose(phase, reference["same_channel_phase_error_rad"], atol=1e-13, rtol=0)
                self.assertAlmostEqual(max(abs(phase)), reference["maximum_absolute_phase_error_rad"], places=12)
                self.assertAlmostEqual(abs(T - np.diag(expected)).max(), reference["maximum_transmission_complex_error"], places=12)
                self.assertAlmostEqual(abs(R).max(), reference["maximum_reflection_amplitude"], places=12)
                self.assertEqual(reference["same_port_complex_reference_limit"], 0.01)
                self.assertGreater(abs(T - np.diag(expected)).max(), 0.01)
                self.assertFalse(reference["same_port_complex_reference_passed"])

        comparisons = record["two_hole_comparison"]
        self.assertEqual(set(comparisons), {"h105_to_h80", "h80_to_h60"})
        for name, comparison in comparisons.items():
            self.check_matched_comparison(comparison, mesh_pair=True)
            self.assertTrue(comparison["power_passed"])
            self.assertEqual(comparison["complex_passed"], name == "h80_to_h60")
        for item in record["cross_mesh_h50_checks"].values():
            self.check_matched_comparison(item["comparison"], mesh_pair=False)
            self.assertTrue(item["comparison"]["power_passed"])
            self.assertFalse(item["comparison"]["complex_passed"])

    def check_matched_comparison(self, comparison, *, mesh_pair):
        self.assertEqual(comparison["power_change_limit" if mesh_pair else "power_limit"], 0.01)
        self.assertEqual(comparison["complex_change_limit" if mesh_pair else "complex_limit"], 0.01)
        power_changes, complex_changes = [], []
        for block in comparison["blocks"].values():
            a, b = (unpack(block[key]) for key in ("reference", "candidate_aligned"))
            power = (abs(b)**2 - abs(a)**2).sum(axis=0)
            np.testing.assert_allclose(power, block["power_change"], atol=1e-13, rtol=0)
            power_changes.append(max(abs(power)))
            complex_changes.append(abs(b - a).max())
        self.assertAlmostEqual(max(power_changes), comparison["maximum_power_change"], places=12)
        self.assertAlmostEqual(max(complex_changes), comparison["maximum_complex_change"], places=12)
        self.assertEqual(comparison["power_passed"], max(power_changes) < 0.01)
        self.assertEqual(comparison["complex_passed"], max(complex_changes) < 0.01)
        if mesh_pair:
            self.assertTrue(comparison["input_power_checks_passed"])
            self.assertEqual(comparison["mesh_pair_passed"], max(power_changes + complex_changes) < 0.01)


if __name__ == "__main__":
    unittest.main()
