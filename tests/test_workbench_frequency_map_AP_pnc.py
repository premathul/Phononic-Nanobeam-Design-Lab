"""Desktop frequency-map workflow boundaries. Author: Athul Prem | _AP_pnc."""

import hashlib
import json
import os
from pathlib import Path
import sys
from tempfile import TemporaryDirectory
import time
import unittest
from unittest.mock import patch

import numpy as np

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
try:
    from PySide6.QtCore import QProcess, Qt
    from PySide6.QtTest import QTest
    from PySide6.QtWidgets import QApplication
except ImportError:
    raise unittest.SkipTest("PySide6 Essentials is optional for core tests.") from None

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT), str(ROOT / "src")]
from tools import map_scattering_AP_pnc as mapper
from workbench_AP_pnc import Workbench


class WorkbenchFrequencyMapTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.app = QApplication.instance() or QApplication([])

    def setUp(self):
        temporary = TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        self.output = Path(temporary.name).resolve()
        self.window = Workbench(self.output)
        self.addCleanup(self.window.close)
        self.window.show()

    def wait_for(self, condition, seconds=15):
        deadline = time.monotonic() + seconds
        while not condition() and time.monotonic() < deadline:
            QTest.qWait(25)
        self.assertTrue(condition(), self.window.status.text())

    def synthetic_worker(self, *, stoppable=False):
        path = self.output / "synthetic_frequency_worker_AP_pnc.py"
        path.write_text(
            "import json, pathlib, sys, time\n"
            "args = sys.argv[1:]\n"
            "action = args[args.index('--action') + 1]\n"
            "response = pathlib.Path(args[args.index('--response') + 1])\n"
            "(response.parent / 'observed_args_AP_pnc.json').write_text(json.dumps(args))\n"
            "directory = pathlib.Path(args[args.index('--map-directory') + 1])\n"
            "directory.mkdir(parents=True, exist_ok=True)\n"
            "report = directory / 'report_AP_pnc.html'\n"
            "report.write_text('Saved frequency map')\n"
            "summary = {'status': 'partial', 'output_directory': str(directory), 'map_directory': str(directory),\n"
            "           'report_path': str(report), 'accepted_points': 1, 'total_points': 2, 'failed_points': 0,\n"
            "           'can_resume': True, 'spatial_convergence_established': False,\n"
            "           'frequency_convergence_established': False,\n"
            "           'points': [{'status': 'accepted'}, {'status': 'pending'}]}\n"
            + (
                "progress = {**summary, 'status': 'running', 'phase': 'solving', 'current_frequency_GHz': 0.30}\n"
                "response.with_suffix('.progress.json').write_text(json.dumps({'action': action, 'status': 'running', 'summary': progress}))\n"
                "deadline = time.monotonic() + 20\n"
                "while not response.with_suffix('.cancel').exists() and time.monotonic() < deadline: time.sleep(0.025)\n"
                "while not (response.parent / 'release_AP_pnc').exists() and time.monotonic() < deadline: time.sleep(0.025)\n"
                "summary['stop_requested'] = True\n" if stoppable else ""
            )
            + "response.write_text(json.dumps({'action': action, 'status': 'partial', 'summary': summary}))\n",
            encoding="utf-8",
        )
        return path

    def test_invalid_frequency_inputs_and_point_budget_create_no_job(self):
        self.assertFalse(self.window.map_button.isVisible())
        self.window.preset.setCurrentIndex(1)
        self.assertTrue(self.window.map_button.isVisible())
        self.assertEqual(self.window.map_max_points.text(), "1")
        for text in ("", "nan", "0", "0.30, 0.25", "0.25, 0.25", "0.25, inf"):
            with self.subTest(frequencies=text):
                self.window.map_frequencies.setText(text)
                self.assertFalse(self.window.start_job("map"))
                self.assertIsNone(self.window.action)
                self.assertFalse((self.output / "_workbench").exists())
        self.window.map_frequencies.setText("0.25, 0.30")
        for bound in ("0", "-1", "1.5", "nan", ""):
            with self.subTest(bound=bound):
                self.window.map_max_points.setText(bound)
                self.assertFalse(self.window.start_job("map"))
                self.assertFalse((self.output / "_workbench").exists())

    def test_26hole_map_request_preserves_geometry_resources_and_resume_uses_saved_inputs(self):
        self.window.preset.setCurrentIndex(4)
        baseline = self.window.edited_config()
        self.window.map_frequencies.setText("0.1, 0.8875, 1")
        self.window.map_max_points.setText("1")
        worker = self.synthetic_worker()
        with patch("workbench_AP_pnc.WORKER", worker):
            QTest.mouseClick(self.window.map_button, Qt.MouseButton.LeftButton)
            self.assertEqual(self.window.action, "map")
            self.assertFalse(self.window.map_button.isEnabled())
            self.assertFalse(self.window.resume_map_button.isEnabled())
            self.assertFalse(self.window.start_job("map"))
            self.wait_for(lambda: self.window.action is None)
            initial_job = self.window.response_path.parent
            args = json.loads((initial_job / "observed_args_AP_pnc.json").read_bytes())
            submitted = json.loads((initial_job / "configuration_AP_pnc.json").read_bytes())
            self.assertEqual(submitted, baseline)
            self.assertEqual(submitted["geometry"]["pattern"], {
                "csv": "data/geometry/hole_geometry_derived_AP_pnc.csv", "x_offset_nm": 8325,
            })
            self.assertEqual(submitted["geometry"]["length_nm"], 16650)
            self.assertEqual(submitted["geometry"]["thickness_nm"], 200)
            self.assertEqual(submitted["resources"], {"max_dofs": 60000, "max_port_dofs": 400})
            frequency_start = args.index("--frequencies-ghz") + 1
            self.assertEqual([float(value) for value in args[frequency_start:frequency_start + 3]], [0.1, 0.8875, 1.0])
            self.assertEqual(args[args.index("--max-new-points") + 1], "1")
            directory = Path(args[args.index("--map-directory") + 1])
            self.assertEqual(self.window.result_directory, directory)
            self.assertFalse(self.window.compare_button.isEnabled())
            self.assertTrue(self.window.report_button.isEnabled())
            self.assertIn("1/2", self.window.status.text())
            self.assertIn("convergence", self.window.status.text().lower())

            self.window.fields["frequency_GHz"].setText("invalid unsaved value")
            self.window.map_frequencies.setText("invalid unsaved frequencies")
            self.window.map_max_points.setText("2")
            self.assertTrue(self.window.start_job("map-resume", map_directory=directory))
            self.wait_for(lambda: self.window.action is None)
        resume_job = self.window.response_path.parent
        resume_args = json.loads((resume_job / "observed_args_AP_pnc.json").read_bytes())
        self.assertEqual(resume_args[resume_args.index("--map-directory") + 1], str(directory))
        self.assertEqual(resume_args[resume_args.index("--max-new-points") + 1], "2")
        self.assertNotIn("--config", resume_args)
        self.assertNotIn("--frequencies-ghz", resume_args)
        self.assertFalse((resume_job / "configuration_AP_pnc.json").exists())
        self.assertEqual(self.window.fields["frequency_GHz"].text(), "invalid unsaved value")
        self.assertEqual(self.window.map_frequencies.text(), "invalid unsaved frequencies")

    def test_map_stop_keeps_worker_until_point_boundary_and_blocks_competing_launch(self):
        self.window.preset.setCurrentIndex(1)
        self.window.map_frequencies.setText("0.25, 0.30")
        worker = self.synthetic_worker(stoppable=True)
        with patch("workbench_AP_pnc.WORKER", worker):
            self.assertTrue(self.window.start_job("map"))
            self.wait_for(lambda: "1/2" in self.window.status.text())
            QTest.mouseClick(self.window.stop_button, Qt.MouseButton.LeftButton)
            self.assertTrue(self.window.response_path.with_suffix(".cancel").exists())
            QTest.qWait(200)
            self.assertEqual(self.window.action, "map")
            self.assertEqual(self.window.process.state(), QProcess.ProcessState.Running)
            self.assertIsNone(self.window.stop_deadline)
            self.assertFalse(self.window.start_job("run"))
            self.assertFalse(self.window.map_button.isEnabled())
            self.assertFalse(self.window.close())
            (self.window.response_path.parent / "release_AP_pnc").touch()
            self.wait_for(lambda: self.window.action is None)
        self.assertIn("1/2", self.window.status.text())
        self.assertTrue(self.window.resume_map_button.isEnabled())
        self.assertTrue(self.window.report_button.isEnabled())

    def test_saved_map_open_reads_only_publication_and_preserves_current_inputs(self):
        self.window.preset.setCurrentIndex(1)
        directory = self.output / "saved imported map"
        directory.mkdir()
        report = directory / mapper.REPORT
        table = directory / mapper.TABLE
        report.write_text("Saved imported report", encoding="utf-8")
        table.write_text("frequency_GHz,status\n1,accepted\n", encoding="utf-8")
        result = {
            "schema_version": 1, "status": "partial", "execution_mode": "imported_read_only",
            "spatial_convergence_established": False, "frequency_convergence_established": False,
            "points": [{"index": 1, "frequency_GHz": 1, "status": "accepted"},
                       {"index": 2, "frequency_GHz": 2, "status": "pending"}],
            "output_sha256": {path.name: hashlib.sha256(path.read_bytes()).hexdigest() for path in (report, table)},
        }
        result["map_data_sha256"] = mapper.workflow.digest(result)
        (directory / mapper.MAP).write_text(json.dumps(result), encoding="utf-8")
        baseline = self.window.edited_config()
        preserved = {path: (path.read_bytes(), path.stat().st_mtime_ns) for path in directory.iterdir()}
        with (
            patch.object(mapper, "map_scattering", side_effect=AssertionError("Computed while opening saved map")),
            patch.object(mapper.workflow, "source_identity", side_effect=AssertionError("Compared current sources while viewing")),
            patch.object(np, "load", side_effect=AssertionError("Loaded scientific arrays while opening saved map")),
            patch("workbench_AP_pnc.QFileDialog.getExistingDirectory", return_value=str(directory)),
            patch("workbench_AP_pnc.QDesktopServices.openUrl", return_value=True) as open_url,
        ):
            QTest.mouseClick(self.window.saved_map_button, Qt.MouseButton.LeftButton)
            self.assertEqual(Path(open_url.call_args.args[0].toLocalFile()), report)
        self.assertEqual(self.window.result_directory, directory)
        self.assertEqual(self.window.report_path, report)
        self.assertEqual(self.window.edited_config(), baseline)
        self.assertIsNone(self.window.action)
        self.assertFalse((self.output / "_workbench").exists())
        self.assertFalse(self.window.compare_button.isEnabled())
        self.assertTrue(self.window.report_button.isEnabled())
        self.assertEqual({path: (path.read_bytes(), path.stat().st_mtime_ns) for path in directory.iterdir()}, preserved)
        report.write_text("Changed report", encoding="utf-8")
        self.assertFalse(self.window.open_saved_map(directory))
        self.assertIn("changed", self.window.status.text().lower())
        self.assertEqual(self.window.edited_config(), baseline)


if __name__ == "__main__":
    unittest.main()
