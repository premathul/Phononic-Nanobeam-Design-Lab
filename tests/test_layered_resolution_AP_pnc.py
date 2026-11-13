"""Offline checks of the layered resolution controls. Author: Athul Prem | _AP_pnc."""

import csv
from hashlib import sha256
import json
from pathlib import Path
import unittest

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
EVIDENCE = ROOT / "verification/layered_resolution_followup_AP_pnc.json"


def unpack(values):
    pairs = np.asarray(values)
    return pairs[..., 0] + 1j * pairs[..., 1]


class LayeredResolutionEvidenceTests(unittest.TestCase):
    def test_saved_pairs_preserve_scope_and_reproduce_power_and_complex_changes(self):
        record = json.loads(EVIDENCE.read_bytes())
        self.assertEqual(record["status"], "completed_diagnostic")
        prior = record["previous_evidence"]
        source = (ROOT / prior["path"]).read_bytes()
        self.assertEqual(sha256(source).hexdigest(), prior["sha256"])
        baseline = json.loads(source)["cases"][prior["baseline_case"]]
        table = EVIDENCE.with_suffix(".csv")
        self.assertEqual(sha256(table.read_bytes()).hexdigest(), record["channel_csv_sha256"])
        with table.open(encoding="utf-8", newline="") as stream:
            csv_rows = list(csv.DictReader(stream))
        self.assertEqual(len(csv_rows), 16)
        planned = {
            "two_hole_h80_port100": ({"port_size_nm": 100}, 17715, 195, False, True),
            "two_hole_h80_z3": ({"thickness_layers": 3}, 24381, 189, True, False),
            "two_hole_h80_z4": ({"thickness_layers": 4}, 31347, 243, True, True),
            "two_hole_h80_z4_port100": ({"thickness_layers": 4, "port_size_nm": 100}, 31887, 351, False, True),
        }
        self.assertEqual(set(record["cases"]), set(planned))
        self.assertEqual(set(record["comparisons"]), {*planned, "port100_thickness_z2_to_z4"})
        self.assertEqual(record["selected_case"], "two_hole_h80_z4_port100")
        for label, (mesh_change, body_dofs, port_dofs, same_footprint, expected_pass) in planned.items():
            with self.subTest(case=label):
                case = record["cases"][label]
                config = case["configuration"]
                self.assertEqual(config["mesh"], {**baseline["configuration"]["mesh"], **mesh_change})
                for section in ("geometry", "material", "analysis"):
                    self.assertEqual(config[section], baseline["configuration"][section])
                self.assertEqual(case["footprint_triangles_match_reference"], same_footprint)
                self.assertEqual(case["metrics"]["ndof"], body_dofs)
                self.assertEqual(case["metrics"]["port_dofs"], port_dofs)
                self.assertTrue(case["metrics"]["power_balance_passed"])
                self.assertEqual(case["execution"]["status"], "completed")
                self.assertEqual(case["geometry_diagnostics"]["rim_segments_min"], 48)
                self.assertEqual(case["geometry_diagnostics"]["rim_segments_max"], 48)
                rows = [row for row in csv_rows if row["case"] == label]
                self.assertEqual([int(row["incident_mode_order"]) for row in rows], [1, 2, 3, 4])
                self.assertEqual(len(case["channels"]), 4)
                np.testing.assert_allclose([row["incident_power_w"] for row in case["channels"]], 1, rtol=0, atol=1e-12)
                comparison_record = record["comparisons"][label]
                reference_label = comparison_record["reference_case"]
                reference_case = baseline if reference_label == prior["baseline_case"] else record["cases"][reference_label]
                comparison = comparison_record["comparison"]
                self.assertEqual(comparison["comparison_kind"], "mesh_sensitivity")
                self.assertTrue(comparison["input_power_checks_passed"])
                self.assertEqual(comparison["power_change_limit"], 0.01)
                self.assertEqual(comparison["complex_change_limit"], 0.01)
                power_changes, complex_changes = [], []
                for quantity in ("reflection", "transmission"):
                    amplitudes = unpack(case["amplitudes"][quantity])
                    powers = (abs(amplitudes)**2).sum(axis=0)
                    self.assertEqual(amplitudes.shape, (4, 4))
                    for values in ([row[quantity] for row in case["channels"]],
                                   [row[f"{quantity}_flux"] for row in case["channels"]],
                                   [float(row[quantity]) for row in rows]):
                        np.testing.assert_allclose(powers, values, rtol=0, atol=1e-12)
                    block = comparison["blocks"][quantity]
                    reference = unpack(block["reference"])
                    candidate = unpack(block["candidate_aligned"])
                    np.testing.assert_array_equal(reference, unpack(reference_case["amplitudes"][quantity]))
                    np.testing.assert_allclose(abs(candidate), abs(amplitudes), rtol=0, atol=1e-13)
                    change = (abs(candidate)**2 - abs(reference)**2).sum(axis=0)
                    np.testing.assert_allclose(change, block["power_change"], rtol=0, atol=1e-13)
                    power_changes.append(max(abs(change)))
                    complex_changes.append(abs(candidate - reference).max())
                    if quantity == "transmission":
                        phase = np.angle(np.diag(candidate) / np.diag(reference))
                        np.testing.assert_allclose(phase, block["same_channel_phase_change_rad"], rtol=0, atol=1e-13)
                self.assertAlmostEqual(max(power_changes), comparison["maximum_power_change"], places=12)
                self.assertAlmostEqual(max(complex_changes), comparison["maximum_complex_change"], places=12)
                self.assertTrue(comparison["power_passed"])
                self.assertLess(max(power_changes), 0.01)
                self.assertEqual(max(complex_changes) < 0.01, expected_pass)
                self.assertEqual(comparison["complex_passed"], expected_pass)
                self.assertEqual(comparison["mesh_pair_passed"], expected_pass)

                cross = record["cross_mesh_h50_checks"][label]["comparison"]
                powers, complex_values = [], []
                for block in cross["blocks"].values():
                    a, b = (unpack(block[key]) for key in ("reference", "candidate_aligned"))
                    powers.append(max(abs((abs(b)**2 - abs(a)**2).sum(axis=0))))
                    complex_values.append(abs(b - a).max())
                self.assertAlmostEqual(max(powers), cross["maximum_power_change"], places=12)
                self.assertAlmostEqual(max(complex_values), cross["maximum_complex_change"], places=12)
                self.assertTrue(cross["power_passed"])
                self.assertEqual(cross["complex_passed"], label != "two_hole_h80_port100")
                self.assertEqual(max(complex_values) < 0.01, cross["complex_passed"])

        joint = record["comparisons"]["port100_thickness_z2_to_z4"]
        self.assertTrue(joint["same_footprint_triangles"])
        self.assertEqual(joint["reference_case"], "two_hole_h80_port100")
        self.assertEqual(joint["candidate_case"], record["selected_case"])
        comparison = joint["comparison"]
        max_power = max(block["maximum_power_change"] for block in comparison["blocks"].values())
        max_complex = max(abs(unpack(block["candidate_aligned"]) - unpack(block["reference"])).max()
                          for block in comparison["blocks"].values())
        self.assertAlmostEqual(max_power, comparison["maximum_power_change"], places=12)
        self.assertAlmostEqual(max_complex, comparison["maximum_complex_change"], places=12)
        self.assertLess(max_power, 0.01)
        self.assertGreater(max_complex, 0.01)
        self.assertTrue(comparison["power_passed"])
        self.assertFalse(comparison["complex_passed"])
        self.assertFalse(comparison["mesh_pair_passed"])

    def test_saved_uniform_responses_against_common_safe_reference(self):
        common = json.loads(EVIDENCE.read_bytes())["uniform_common_reference"]
        self.assertEqual(common["status"], "completed_diagnostic")
        original, safe = (
            self.read_evidence(common[key])
            for key in ("original_uniform_evidence", "safe_reference_evidence")
        )
        safe_case = common["safe_reference_evidence"]["case"]
        self.assertEqual(safe_case, "8x4")
        section = safe["cases"][safe_case]
        self.assertEqual(section["trace_dofs"], common["safe_reference_evidence"]["trace_dofs"])
        self.assertEqual(section["trace_dofs"], 459)
        self.assertEqual(safe["frequency_hz"], 1e9)
        self.assertEqual(safe["physical_section_nm"], {"width": 600, "thickness": 200})
        self.assertEqual(common["reference_phase_qualification"]["status"], "passed")

        self.assertEqual(set(common["cases"]), {"uniform_h105", "uniform_h60"})
        for label, expected_pass in (("uniform_h105", False), ("uniform_h60", True)):
            with self.subTest(case=label):
                case = common["cases"][label]
                run = original["cases"][label]
                same_port = original["uniform_same_port_reference"][label]
                self.assertEqual(case["original_run_summary_sha256"], run["source"]["summary"]["sha256"])
                self.assertEqual(case["original_run_fields_sha256"], run["source"]["fields"]["sha256"])
                self.assertEqual(run["configuration"]["analysis"]["incident_side"], "left")
                length = case["propagation_length_m"]
                self.assertEqual(length, run["configuration"]["geometry"]["length_nm"] * 1e-9)
                self.assertEqual(length, same_port["length_m"])
                self.assertEqual(case["power_error_limit"], 0.01)
                self.assertEqual(case["complex_error_limit"], 0.01)

                match = case["matched_channels"]
                positive = section["propagating"]["positive"]
                self.assertEqual(match["incoming"]["reference_roots"], positive["root_indices"])
                self.assertEqual(match["incoming"], match["transmission"])
                k = np.array(positive["k_rad_m"])
                np.testing.assert_array_equal(k, match["incoming"]["reference_wavenumber_rad_m"])
                native_k = {row["incident_root"]: row["wavenumber_rad_m"] for row in run["channels"]}
                matched_native_k = np.array([native_k[root] for root in match["incoming"]["candidate_roots"]])
                np.testing.assert_array_equal(matched_native_k, match["incoming"]["candidate_wavenumber_rad_m"])
                native = case["native_port_diagnostic"]
                np.testing.assert_allclose((matched_native_k - k) * length,
                                           native["native_minus_common_propagation_phase_rad"], atol=1e-13, rtol=0)

                indices = run["root_indices"]
                incident_order = [indices["incident"].index(root) for root in match["incoming"]["candidate_roots"]]
                incident_gauge = np.exp(1j * np.asarray(match["incoming"]["candidate_gauge_phase_rad"]))
                ideal_transmission = np.diag(np.exp(1j * k * length))
                max_power, max_complex = [], []
                for quantity, ideal in (("reflection", np.zeros_like(ideal_transmission)),
                                        ("transmission", ideal_transmission)):
                    block = case["blocks"][quantity]
                    order = [indices[quantity].index(root) for root in match[quantity]["candidate_roots"]]
                    gauge = np.exp(1j * np.asarray(match[quantity]["candidate_gauge_phase_rad"]))
                    raw = unpack(run["amplitudes"][quantity])
                    aligned = gauge[:, None] * raw[np.ix_(order, incident_order)] / incident_gauge[None, :]
                    np.testing.assert_allclose(aligned, unpack(block["candidate_aligned"]), atol=1e-13, rtol=0)
                    np.testing.assert_allclose(ideal, unpack(block["reference"]), atol=1e-13, rtol=0)
                    power_change = (abs(aligned)**2 - abs(ideal)**2).sum(axis=0)
                    max_power.append(max(abs(power_change)))
                    max_complex.append(abs(aligned - ideal).max())
                    if quantity == "transmission":
                        phase = np.angle(np.diag(aligned) / np.diag(ideal))
                        np.testing.assert_allclose(phase, block["same_channel_phase_change_rad"], atol=1e-13, rtol=0)
                        self.assertAlmostEqual(max(abs(phase)), case["maximum_absolute_phase_error_rad"], places=12)
                        native_ideal = np.diag(np.exp(1j * matched_native_k * length))
                        native_error = abs(aligned - native_ideal).max()
                        self.assertAlmostEqual(native_error, native["maximum_native_transmission_complex_error"], places=12)
                        self.assertAlmostEqual(native_error, same_port["maximum_transmission_complex_error"], places=12)
                        self.assertGreater(native_error, 0.01)

                self.assertAlmostEqual(max(max_power), case["maximum_power_error"], places=12)
                self.assertAlmostEqual(max(max_complex), case["maximum_complex_error"], places=12)
                self.assertLess(max(max_power), 0.01)
                self.assertEqual(max(max_complex) < 0.01, expected_pass)
                self.assertTrue(case["power_passed"])
                self.assertEqual(case["complex_passed"], expected_pass)
                self.assertTrue(run["metrics"]["power_balance_passed"])
                self.assertTrue(case["input_power_balance_passed"])
                self.assertEqual(case["uniform_response_passed"], expected_pass)

    def read_evidence(self, provenance):
        payload = (ROOT / provenance["path"]).read_bytes()
        self.assertEqual(sha256(payload).hexdigest(), provenance["sha256"])
        return json.loads(payload)


if __name__ == "__main__":
    unittest.main()
