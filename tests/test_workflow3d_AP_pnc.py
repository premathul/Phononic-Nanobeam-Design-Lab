"""End-to-end scientific output and retry checks. Author: Athul Prem | _AP_pnc."""
import copy
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


if __name__ == "__main__":
    unittest.main()
