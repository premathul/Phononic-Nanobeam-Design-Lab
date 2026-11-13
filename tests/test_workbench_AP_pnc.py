"""Small desktop workflow checks. Author: Athul Prem | _AP_pnc."""

import json
import os
from pathlib import Path
import sys
from tempfile import TemporaryDirectory
import time
import unittest
from unittest.mock import patch

import psutil

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
try:
    from PySide6.QtCore import Qt
    from PySide6.QtTest import QTest
    from PySide6.QtWidgets import QApplication
except ImportError:
    raise unittest.SkipTest("PySide6 Essentials is optional for core tests.") from None

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from workbench_AP_pnc import PRESETS, Workbench


class WorkbenchTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.app = QApplication.instance() or QApplication([])

    def setUp(self):
        temporary = TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        self.output = Path(temporary.name)
        self.window = Workbench(self.output)
        self.addCleanup(self.window.close)
        self.window.show()

    def wait_for(self, condition, seconds=15):
        deadline = time.monotonic() + seconds
        while not condition() and time.monotonic() < deadline:
            QTest.qWait(25)
        self.assertTrue(condition(), self.window.status.text())

    def test_tiny_preflight_then_run_reuses_exact_mesh_and_resets_on_load(self):
        self.window.fields["cells"].setText("2, 2, 1")
        self.window.fields["num_modes"].setText("3")
        with patch("workbench_AP_pnc.QDesktopServices.openUrl") as open_url:
            QTest.mouseClick(self.window.mesh_button, Qt.MouseButton.LeftButton)
            self.wait_for(lambda: self.window.action is None)
            self.assertIn("Mesh ready", self.window.status.text())
            self.assertIn("225 body DOFs", self.window.status.text())
            self.assertTrue(self.window.folder_button.isEnabled())
            self.assertFalse(self.window.report_button.isEnabled())
            mesh = json.loads(self.window.response_path.read_bytes())["summary"]
            self.assertEqual(mesh["status"], "mesh_ready")

            QTest.mouseClick(self.window.run_button, Qt.MouseButton.LeftButton)
            self.wait_for(lambda: self.window.action is None)
            self.assertIn("Computed (unverified)", self.window.status.text())
            self.assertTrue(self.window.report_button.isEnabled())
            run = json.loads(self.window.response_path.read_bytes())["summary"]
            self.assertEqual(run["mesh_checkpoint"]["mesh_sha256"], mesh["mesh_sha256"])
            self.assertTrue(self.window.report_path.is_file())
            open_url.assert_not_called()

            self.window.preset.setCurrentIndex(2)
            self.assertFalse(self.window.report_button.isEnabled())
            self.assertFalse(self.window.folder_button.isEnabled())
            self.assertTrue(self.window.fields["frequency_GHz"].isVisible())
            self.assertFalse(self.window.fields["cells"].isVisible())
            preset_path = ROOT / "configs" / PRESETS[2][1]
            baseline = json.loads(preset_path.read_bytes())
            self.window.fields["width_nm"].setText("620")
            edited = self.window.edited_config()
            self.assertEqual(edited["geometry"]["width_nm"], 620)
            self.assertEqual(edited["geometry"]["pattern"], baseline["geometry"]["pattern"])
            self.assertEqual(edited["material"], baseline["material"])

    def test_invalid_input_never_starts_a_worker(self):
        self.window.fields["thickness_nm"].setText("nan")
        self.assertFalse(self.window.start_job("mesh"))
        self.assertIn("Thickness (nm) must be a positive number", self.window.status.text())
        self.assertIsNone(self.window.action)
        self.assertFalse((self.output / "_workbench").exists())

    def test_malformed_analysis_load_preserves_prior_inputs_and_result(self):
        report = self.output / "report_AP_pnc.html"
        report.touch()
        self.window._show_result("completed_unverified", {"output_directory": str(self.output), "analysis_kind": "modes"})
        previous = self.window.edited_config()
        provenance = self.window.provenance.text()
        for field, message in (("kind", "finite-mode, Bloch, or scattering"), ("boundary", "supported boundary condition")):
            with self.subTest(field=field):
                malformed = self.window.edited_config()
                malformed["geometry"]["length_nm"] = 500
                malformed["analysis"][field] = []
                path = self.output / "malformed_AP_pnc.json"
                path.write_text(json.dumps(malformed), encoding="utf-8")
                self.assertFalse(self.window.load_configuration(path))
                self.assertIn(message, self.window.status.text())
                self.assertEqual(self.window.edited_config(), previous)
                self.assertEqual(self.window.provenance.text(), provenance)
                self.assertEqual(self.window.preset.currentIndex(), 0)
                self.assertEqual(self.window.report_path, report)
                self.assertTrue(self.window.report_button.isEnabled())

    def test_stop_waits_for_worker_descendants_and_keeps_job_record(self):
        script = self.output / "synthetic_worker_AP_pnc.py"
        script.write_text(
            "import json, pathlib, subprocess, sys, time\n"
            "path = pathlib.Path(sys.argv[sys.argv.index('--response') + 1]).parent\n"
            "child = subprocess.Popen([sys.executable, '-c', 'import time; time.sleep(20)'])\n"
            "(path / 'child_AP_pnc.json').write_text(json.dumps(child.pid))\n"
            "print('Synthetic worker started', flush=True)\n"
            "time.sleep(20)\n",
            encoding="utf-8",
        )
        with patch("workbench_AP_pnc.WORKER", script):
            self.assertTrue(self.window.start_job("run"))
            directory = self.window.response_path.parent
            child_file = directory / "child_AP_pnc.json"
            self.wait_for(child_file.is_file)
            child = psutil.Process(json.loads(child_file.read_bytes()))
            self.addCleanup(lambda: child.kill() if child.is_running() and child.status() != psutil.STATUS_ZOMBIE else None)
            self.assertFalse(self.window.close())
            self.assertIn("Use Stop", self.window.status.text())
            QTest.mouseClick(self.window.stop_button, Qt.MouseButton.LeftButton)
            self.wait_for(lambda: self.window.action is None)
        self.assertIn("Study cancelled", self.window.status.text())
        self.assertTrue(self.window.run_button.isEnabled())
        self.assertTrue((directory / "configuration_AP_pnc.json").is_file())
        self.assertIn("Synthetic worker started", (directory / "worker_AP_pnc.log").read_text(encoding="utf-8"))
        self.assertFalse(child.is_running() and child.status() != psutil.STATUS_ZOMBIE)


if __name__ == "__main__":
    unittest.main()
