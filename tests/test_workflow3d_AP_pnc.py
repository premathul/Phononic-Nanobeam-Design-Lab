"""End-to-end scientific output and retry checks. Author: Athul Prem | _AP_pnc."""
import copy
import csv
import json
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from phononic.workflow_AP_pnc import resolve_config, run_study


class Workflow3D(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        self.output = Path(self.temporary.name)
        self.config = json.loads((ROOT / "configs/straight_beam_AP_pnc.json").read_text())
        self.config["mesh"]["cells"] = [2, 2, 1]
        self.config["analysis"]["num_modes"] = 4

    def test_output_can_be_reconstructed_and_cache_checks_files(self):
        result = run_study(self.config, self.output)
        directory = Path(result["output_directory"])
        self.assertFalse(result["reused"])
        self.assertEqual(result["status"], "completed_unverified")
        self.assertFalse(result["calibrated_scattering_available"])
        with np.load(directory / "selected_fields_AP_pnc.npz", allow_pickle=False) as fields:
            self.assertEqual(int(fields["element_order"]), 2)
            self.assertEqual(fields["element_dofs"].shape, (30, result["mesh_tetrahedra"]))
            self.assertEqual(fields["mass_normalized_modes"].shape, (result["ndof"], 4))
            np.testing.assert_array_equal(np.unique(fields["dof_component"]), [0, 1, 2])
            self.assertTrue(np.all(fields["frequencies_hz"] > 0))
        self.assertTrue(run_study(self.config, self.output)["reused"])
        (directory / "modes_AP_pnc.csv").write_text("changed result\n")
        with self.assertRaisesRegex(ValueError, "changed or are missing"):
            run_study(self.config, self.output)

    def test_failed_attempt_is_retained_and_bad_material_creates_no_output(self):
        with patch("phononic.workflow_AP_pnc.solve_modes", side_effect=RuntimeError("test solve failure")):
            with self.assertRaisesRegex(RuntimeError, "test solve failure"):
                run_study(self.config, self.output)
        failed_directory = next(self.output.iterdir())
        original_failure = (failed_directory / "summary_AP_pnc.json").read_bytes()
        self.assertFalse((failed_directory / ".run.lock").exists())
        recovered = run_study(self.config, self.output)
        self.assertNotEqual(Path(recovered["output_directory"]), failed_directory)
        self.assertEqual((failed_directory / "summary_AP_pnc.json").read_bytes(), original_failure)
        before = set(self.output.iterdir())
        bad = copy.deepcopy(self.config)
        bad["material"]["density_kg_m3"] = True
        with self.assertRaises(ValueError):
            run_study(bad, self.output)
        self.assertEqual(set(self.output.iterdir()), before)
        bad = copy.deepcopy(self.config)
        bad["material"]["crystal_rotation"][0][0] = True
        with self.assertRaises(ValueError):
            resolve_config(bad)

    def test_scattering_exports_power_channels_and_reconstructible_responses(self):
        config = json.loads((ROOT / "configs/straight_scattering_AP_pnc.json").read_text())
        config["geometry"].update(length_nm=1000, width_nm=600)
        config["mesh"]["cells"] = [6, 2, 1]
        config["analysis"]["frequency_GHz"] = 1.0
        result = run_study(config, self.output)
        directory = Path(result["output_directory"])
        self.assertEqual(result["num_incident_channels"], 4)
        self.assertTrue(result["matched_reference_passed"])
        self.assertFalse(result["spatial_convergence_established"])
        self.assertFalse(result["calibrated_scattering_available"])
        with (directory / "scattering_AP_pnc.csv").open(newline="") as stream:
            rows = list(csv.DictReader(stream))
        np.testing.assert_allclose([float(row["transmission"]) for row in rows], 1, atol=0.001)
        np.testing.assert_allclose([float(row["flux_power_balance"]) for row in rows], 0, atol=1e-10)
        with (directory / "scattering_channels_AP_pnc.csv").open(newline="") as stream:
            channels = list(csv.DictReader(stream))
        self.assertEqual(len(channels), 32)
        for column, row in enumerate(rows, 1):
            transmitted = sum(float(item["power_fraction"]) for item in channels
                              if item["kind"] == "transmission" and int(item["incident_mode_order"]) == column)
            self.assertAlmostEqual(transmitted, float(row["transmission"]), places=12)
        with np.load(directory / "selected_fields_AP_pnc.npz", allow_pickle=False) as fields:
            self.assertNotIn("mass_normalized_modes", fields.files)
            self.assertEqual(fields["displacements_per_sqrt_w"].shape, (result["ndof"], 4))
            np.testing.assert_allclose(fields["incident_powers_w"], 1, atol=1e-10)
            self.assertEqual(fields["port_displacements"].shape, (45, 90))
            self.assertEqual(fields["transmission_amplitudes_full"].shape, (45, 4))
            port = fields["port_displacements"]
            incident_trace = (port[:, fields["incident_mode_indices"]]
                              + port[:, fields["reflection_all_mode_indices"]] @ fields["reflection_amplitudes_full"])
            transmitted_trace = port[:, fields["transmission_all_mode_indices"]] @ fields["transmission_amplitudes_full"]
            response = fields["displacements_per_sqrt_w"]
            scale = float(np.max(abs(response)))
            np.testing.assert_allclose(response[fields["port_left_dofs"]], incident_trace, rtol=1e-10, atol=scale * 1e-12)
            np.testing.assert_allclose(response[fields["port_right_dofs"]], transmitted_trace, rtol=1e-10, atol=scale * 1e-12)
        html = (directory / "report_AP_pnc.html").read_text(encoding="utf-8")
        self.assertIn('lang="en"', html)
        self.assertIn("Incident channel", html)
        self.assertNotIn("mass-normalized eigenvectors", html)
        self.assertNotRegex(html, r"[\u3400-\u9fff]")
        self.assertTrue(run_study(config, self.output)["reused"])
        invalid = copy.deepcopy(config)
        invalid["analysis"]["num_modes"] = 4
        with self.assertRaisesRegex(ValueError, "omit num_modes"):
            resolve_config(invalid)
        invalid = copy.deepcopy(config)
        invalid["resources"]["max_port_dofs"] = 44
        with self.assertRaisesRegex(ValueError, "45 port DOFs"):
            resolve_config(invalid)


if __name__ == "__main__":
    unittest.main()
