"""Saved scattering comparison checks. Author: Athul Prem | _AP_pnc."""

import json
from pathlib import Path
import sys
from tempfile import TemporaryDirectory
import unittest
from unittest.mock import patch

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "tools"))

from compare_scattering_AP_pnc import compare_runs, file_sha
from phononic.workflow_AP_pnc import run_study, write_json


class ScatteringComparisonTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        temporary = TemporaryDirectory()
        cls.addClassCleanup(temporary.cleanup)
        cls.output = Path(temporary.name)
        config = json.loads((ROOT / "configs/straight_scattering_AP_pnc.json").read_bytes())
        config["geometry"].update(length_nm=1000, width_nm=600)
        config["mesh"].update(cells=[6, 3, 2], element_order=1)
        cls.reference = Path(run_study(config, cls.output)["output_directory"])
        config["mesh"].update(cells=[8, 3, 2], element_order=2)
        cls.candidate = Path(run_study(config, cls.output)["output_directory"])

    def test_unequal_traces_compare_and_reuse_with_checksum(self):
        output = self.output / "unequal-trace"
        first = compare_runs(self.reference, self.candidate, output)
        self.assertEqual(first["status"], "compared")
        self.assertEqual(first["comparison_kind"], "mesh_sensitivity")
        self.assertNotEqual(first["discretization_sha256"]["reference"], first["discretization_sha256"]["candidate"])
        self.assertFalse(first["reused"])
        self.assertLess(first["reference"]["port_dofs"], first["candidate"]["port_dofs"])
        for group in first["matched_channels"].values():
            self.assertGreaterEqual(min(group["mass_overlaps"]), 0.99)
        self.assertTrue(Path(first["report_path"]).is_file())
        self.assertGreater(first["maximum_complex_change"], 0)
        self.assertTrue(compare_runs(self.reference, self.candidate, output)["reused"])
        reverse = compare_runs(self.candidate, self.reference, output)
        self.assertAlmostEqual(first["maximum_power_change"], reverse["maximum_power_change"], places=11)
        self.assertAlmostEqual(first["maximum_complex_change"], reverse["maximum_complex_change"], places=11)
        report = Path(first["report_path"])
        report.write_text("changed comparison", encoding="utf-8")
        with self.assertRaisesRegex(ValueError, "missing or changed"):
            compare_runs(self.reference, self.candidate, output)
        self.assertEqual(report.read_text(encoding="utf-8"), "changed comparison")

    def test_signed_channel_gauge_and_permutation_leave_comparison_invariant(self):
        summary, fields = self.copy_run(self.reference, "gauge")
        candidate = fields.parent
        with np.load(fields, allow_pickle=False) as saved:
            arrays = {name: saved[name] for name in saved.files}
        incoming = arrays["incident_mode_indices"]
        reflected = arrays["reflection_mode_indices"]
        transmitted = arrays["transmission_mode_indices"]
        phases = np.ones(len(arrays["port_wavenumbers_rad_m"]), dtype=complex)
        phases[incoming] = np.exp(1j * np.linspace(0.3, 1.7, len(incoming)))
        phases[reflected] = np.exp(-1j * np.linspace(1.1, 2.7, len(reflected)))
        for field in ("port_displacements", "port_axial_forces"):
            arrays[field] *= phases
        for name, roots in (("reflection", reflected), ("transmission", transmitted)):
            order = np.roll(np.arange(len(roots)), 1)
            arrays[f"{name}_amplitudes"] = (arrays[f"{name}_amplitudes"] * phases[incoming][None, :]
                                             / phases[roots][:, None])[order]
            arrays[f"{name}_mode_indices"] = roots[order]
        np.savez_compressed(fields, **arrays)
        self.save_summary(summary, fields.parent, fields.name)
        result = compare_runs(self.reference, candidate, self.output / "gauge-comparison")
        self.assertEqual(result["comparison_kind"], "repeatability")
        self.assertEqual(result["discretization_sha256"]["reference"], result["discretization_sha256"]["candidate"])
        self.assertTrue(result["input_power_checks_passed"])
        self.assertTrue(result["power_passed"] and result["complex_passed"])
        self.assertFalse(result["mesh_pair_passed"])
        self.assertIn("Same discretization; mesh convergence not assessed", Path(result["report_path"]).read_text(encoding="utf-8"))
        self.assertLess(result["maximum_power_change"], 1e-12)
        self.assertLess(result["maximum_complex_change"], 1e-12)
        np.testing.assert_allclose(result["blocks"]["transmission"]["same_channel_phase_change_rad"], 0, atol=1e-12)

    def test_failed_input_power_balance_prevents_mesh_acceptance(self):
        summary, fields = self.copy_run(self.candidate, "failed-balance")
        summary["power_balance_passed"] = False
        write_json(fields.parent / "summary_AP_pnc.json", summary)
        # Isolate the input-balance veto by supplying changes below both limits.
        original = compare_runs(self.reference, self.candidate, self.output / "original-balance-comparison")
        for block in original["blocks"].values():
            block["maximum_power_change"] = 0.001
            block["maximum_complex_change"] = 0.002
        with patch("compare_scattering_AP_pnc.compare_blocks",
                   return_value=(original["matched_channels"], original["blocks"], [{
                       "quantity": "transmission", "incident_channel": 1,
                       "reference_power": 0.998, "candidate_power": 0.999,
                       "power_change": 0.001, "maximum_complex_change": 0.002,
                       "same_channel_transmission_phase_change_rad": 0.001,
                   }])):
            result = compare_runs(self.reference, fields.parent, self.output / "failed-balance-comparison")
        self.assertEqual(result["comparison_kind"], "mesh_sensitivity")
        self.assertTrue(result["reference"]["power_balance_passed"])
        self.assertFalse(result["candidate"]["power_balance_passed"])
        self.assertFalse(result["input_power_checks_passed"])
        self.assertTrue(result["power_passed"] and result["complex_passed"])
        self.assertFalse(result["mesh_pair_passed"])
        report = Path(result["report_path"]).read_text(encoding="utf-8")
        self.assertIn("reference passed; candidate failed", report)
        self.assertIn("Mesh-pair checks failed", report)

    def test_changed_physical_input_is_rejected_before_comparing(self):
        summary, fields = self.copy_run(self.reference, "different")
        summary["configuration"]["material"]["C44_Pa"] *= 1.05
        write_json(fields.parent / "configuration_AP_pnc.json", summary["configuration"])
        self.save_summary(summary, fields.parent, "configuration_AP_pnc.json")
        with self.assertRaisesRegex(ValueError, "differ in material"):
            compare_runs(self.reference, fields.parent, self.output / "different-comparison")
        self.assertFalse((self.output / "different-comparison").exists())

    def copy_run(self, source: Path, name: str):
        directory = self.output / name
        directory.mkdir()
        summary = json.loads((source / "summary_AP_pnc.json").read_bytes())
        for filename in summary["output_sha256"]:
            (directory / filename).write_bytes((source / filename).read_bytes())
        return summary, directory / "selected_fields_AP_pnc.npz"

    @staticmethod
    def save_summary(summary: dict, directory: Path, changed_file: str) -> None:
        summary["output_sha256"][changed_file] = file_sha(directory / changed_file)
        write_json(directory / "summary_AP_pnc.json", summary)


if __name__ == "__main__":
    unittest.main()
