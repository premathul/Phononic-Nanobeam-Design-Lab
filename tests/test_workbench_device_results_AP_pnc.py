"""Saved-device browser desktop wiring. Author: Athul Prem | _AP_pnc."""

from copy import deepcopy
import csv
import json
import os
from pathlib import Path
import sys
from tempfile import TemporaryDirectory
import unittest
from unittest.mock import patch

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
try:
    from PySide6.QtCore import QProcess, Qt
    from PySide6.QtTest import QTest
    from PySide6.QtWidgets import QApplication
except ImportError:
    raise unittest.SkipTest("PySide6 Essentials is optional for core tests.") from None

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from workbench_AP_pnc import Workbench
from tools import map_scattering_AP_pnc as mapper


class WorkbenchDeviceResultsTests(unittest.TestCase):
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
        self.app.processEvents()

    def snapshot(self):
        return {
            "config": deepcopy(self.window.config),
            "edited_config": self.window.edited_config(),
            "fields": {name: field.text() for name, field in self.window.fields.items()},
            "preset": self.window.preset.currentIndex(),
            "state": tuple(getattr(self.window, name) for name in (
                "action", "response_path", "result_directory", "report_path",
                "scattering_directory", "resume_directory", "resume_source",
            )),
        }

    def assert_no_worker(self):
        self.assertEqual(self.window.process.state(), QProcess.ProcessState.NotRunning)
        self.assertFalse((self.output / "_workbench").exists())

    def test_open_device_results_publishes_the_thin_grid_from_the_desktop_button(self):
        before = self.snapshot()
        with patch("workbench_AP_pnc.QDesktopServices.openUrl", return_value=True) as open_url:
            QTest.mouseClick(self.window.device_results_button, Qt.MouseButton.LeftButton)
        open_url.assert_called_once()
        report = Path(open_url.call_args.args[0].toLocalFile()).resolve()
        self.assertEqual(report, self.output / "saved_device_results_AP_pnc/report_AP_pnc.html")
        self.assertIn('id="thin-grid-mesh"', report.read_text(encoding="utf-8"))
        with report.with_name("thin_mesh_coefficients_AP_pnc.csv").open(newline="") as stream:
            self.assertEqual(sum(1 for _ in csv.DictReader(stream)), 450)
        self.assertEqual(self.snapshot(), before)
        self.assert_no_worker()

    def test_thin_grid_configuration_downloads_open_in_the_workbench(self):
        comparison = json.loads((ROOT / "verification/thin_mesh_frequency_AP_pnc.json").read_bytes())
        native_maps = {(level["central_max_edge_nm"], sample["frequency_hz"]): ROOT / sample["archive"]["map"]["path"]
                       for level in comparison["mesh_levels"] for sample in level["samples"]}
        with patch("workbench_AP_pnc.QDesktopServices.openUrl", return_value=True) as open_url:
            QTest.mouseClick(self.window.device_results_button, Qt.MouseButton.LeftButton)
        open_url.assert_called_once()
        report = Path(open_url.call_args.args[0].toLocalFile()).read_text(encoding="utf-8")
        _, marker, payload = report.partition('<script type="application/json" id="thin-grid-data">')
        self.assertTrue(marker)
        grid = json.loads(payload.split("</script>", 1)[0])

        downloads = self.output / "Browser downloads"
        downloads.mkdir()
        self.assertFalse(downloads.is_relative_to(ROOT))
        opened = set()
        with (
            patch.object(self.window, "start_job", side_effect=AssertionError("Loading started a job")),
            patch.object(self.window.process, "start", side_effect=AssertionError("Loading started a process")),
        ):
            for level in grid["levels"]:
                target_nm = level["central_max_edge_nm"]
                for point in level["points"]:
                    frequency_hz = point["frequency_hz"]
                    key = target_nm, frequency_hz
                    with self.subTest(target_nm=target_nm, frequency_hz=frequency_hz):
                        self.assertNotIn(key, opened)
                        opened.add(key)
                        native = json.loads(native_maps[key].read_bytes())
                        expected = mapper._point_config(native["inputs"]["configuration"], frequency_hz / 1e9)
                        path = downloads / f"thin_{target_nm}nm_{frequency_hz}Hz_AP_pnc.json"
                        # The browser download uses this serialized payload verbatim.
                        path.write_bytes(point["configuration_json"].encode("utf-8"))
                        with patch("workbench_AP_pnc.QFileDialog.getOpenFileName",
                                   return_value=(str(path), "JSON (*.json)")) as choose:
                            QTest.mouseClick(self.window.load_button, Qt.MouseButton.LeftButton)
                        choose.assert_called_once()
                        self.assertIn("Configuration ready", self.window.status.text())
                        self.assertEqual(float(self.window.fields["frequency_GHz"].text()), frequency_hz / 1e9)
                        self.assertEqual(float(self.window.regional_central_edge.text()), target_nm)
                        edited = self.window.edited_config()
                        self.assertEqual(mapper.workflow.digest(edited), mapper.workflow.digest(expected))
                        resolved = mapper.workflow.resolve_config(edited)
                        self.assertEqual(mapper.workflow.digest(resolved), mapper.workflow.digest(expected))
                        _, holes, pattern = mapper.workflow._resolve_geometry(resolved)
                        self.assertEqual(len(holes), 26)
                        self.assertEqual(pattern, native["inputs"]["hole_pattern"])
        self.assertEqual(opened, set(native_maps))
        self.assertEqual(len(opened), 9)
        self.assertIn((126.5625, 997_500_000), opened)
        self.assert_no_worker()

    def test_button_opens_device_report_and_preserves_current_study(self):
        current = self.output / "existing scattering result"
        current.mkdir()
        self.window.result_directory = current
        self.window.scattering_directory = current
        self.window.report_path = current / "report_AP_pnc.html"
        self.window.report_path.write_text("Existing result", encoding="utf-8")
        self.window.fields["width_nm"].setText("625")
        self.window._set_result_buttons(False)
        report_identity = self.window.current_report.text()
        before = self.snapshot()

        publication = self.output / "saved_device_results_AP_pnc"
        publication.mkdir()
        report = publication / "report_AP_pnc.html"
        report.write_text("Saved device results", encoding="utf-8")
        dataset = {"points": [{"frequency_GHz": 0.1}, {"frequency_GHz": 0.8875}, {"frequency_GHz": 1.0}]}
        with patch("workbench_AP_pnc.device_results.load_dataset", return_value=dataset) as load, \
                patch("workbench_AP_pnc.device_results.publish_dataset", return_value={
                    "report_path": str(report), "output_directory": str(publication),
                }) as publish, \
                patch.object(self.window, "start_job", side_effect=AssertionError("Browser launched a worker")), \
                patch.object(self.window.process, "start", side_effect=AssertionError("Browser started a process")), \
                patch("workbench_AP_pnc.QDesktopServices.openUrl", return_value=True) as open_url:
            self.assertEqual(self.window.device_results_button.text(), "Browse device results")
            QTest.mouseClick(self.window.device_results_button, Qt.MouseButton.LeftButton)

        load.assert_called_once_with()
        publish.assert_called_once_with(dataset, publication)
        open_url.assert_called_once()
        self.assertEqual(Path(open_url.call_args.args[0].toLocalFile()).resolve(), report)
        self.assertEqual(self.snapshot(), before)
        self.assertIn(current.name, report_identity)
        self.assertEqual(self.window.current_report.text(), report_identity)
        self.assertEqual((current / "report_AP_pnc.html").read_text(encoding="utf-8"), "Existing result")
        self.assert_no_worker()

    def test_missing_saved_source_reports_failure_without_launching_or_mutating(self):
        before = self.snapshot()
        with patch("workbench_AP_pnc.device_results.load_dataset", side_effect=FileNotFoundError("Saved source absent")), \
                patch("workbench_AP_pnc.device_results.publish_dataset") as publish, \
                patch.object(self.window, "start_job", side_effect=AssertionError("Browser launched a worker")), \
                patch.object(self.window.process, "start", side_effect=AssertionError("Browser started a process")), \
                patch("workbench_AP_pnc.QDesktopServices.openUrl") as open_url:
            self.assertFalse(self.window.open_device_results())

        publish.assert_not_called()
        open_url.assert_not_called()
        self.assertIn("Saved source absent", self.window.status.text())
        self.assertEqual(self.snapshot(), before)
        self.assert_no_worker()

    def test_system_open_failure_keeps_the_prepared_report_and_error_message(self):
        report = self.output / "report_AP_pnc.html"
        report.write_text("Prepared saved results", encoding="utf-8")
        before = self.snapshot()
        with patch("workbench_AP_pnc.device_results.load_dataset", return_value={}), \
                patch("workbench_AP_pnc.device_results.publish_dataset", return_value={"report_path": str(report)}), \
                patch("workbench_AP_pnc.QDesktopServices.openUrl", return_value=False):
            self.assertFalse(self.window.open_device_results())
        self.assertIn("Could not open", self.window.status.text())
        self.assertIn(str(report), self.window.status.text())
        self.assertEqual(report.read_text(encoding="utf-8"), "Prepared saved results")
        self.assertEqual(self.snapshot(), before)
        self.assert_no_worker()

    def test_active_job_disables_browser_and_refuses_direct_open(self):
        self.window.action = "run"
        self.window._set_busy(True)
        try:
            before = self.snapshot()
            with patch("workbench_AP_pnc.device_results.load_dataset") as load, \
                    patch("workbench_AP_pnc.device_results.publish_dataset") as publish, \
                    patch("workbench_AP_pnc.QDesktopServices.openUrl") as open_url:
                self.assertFalse(self.window.device_results_button.isEnabled())
                self.assertFalse(self.window.open_device_results())
            load.assert_not_called()
            publish.assert_not_called()
            open_url.assert_not_called()
            self.assertEqual(self.snapshot(), before)
        finally:
            self.window.action = None
            self.window._set_busy(False)
        self.assert_no_worker()


if __name__ == "__main__":
    unittest.main()
