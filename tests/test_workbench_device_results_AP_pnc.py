"""Saved-device browser desktop wiring. Author: Athul Prem | _AP_pnc."""

from copy import deepcopy
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

    def test_button_opens_device_report_and_preserves_current_study(self):
        current = self.output / "existing scattering result"
        current.mkdir()
        self.window.result_directory = current
        self.window.scattering_directory = current
        self.window.report_path = current / "report_AP_pnc.html"
        self.window.report_path.write_text("Existing result", encoding="utf-8")
        self.window.fields["width_nm"].setText("625")
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
            self.assertEqual(self.window.device_results_button.text(), "Open device results")
            QTest.mouseClick(self.window.device_results_button, Qt.MouseButton.LeftButton)

        load.assert_called_once_with()
        publish.assert_called_once_with(dataset, publication)
        open_url.assert_called_once()
        self.assertEqual(Path(open_url.call_args.args[0].toLocalFile()).resolve(), report)
        self.assertEqual(self.snapshot(), before)
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
