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
            self.assertFalse(self.window.compare_button.isEnabled())
            mesh = json.loads(self.window.response_path.read_bytes())["summary"]
            self.assertEqual(mesh["status"], "mesh_ready")

            QTest.mouseClick(self.window.run_button, Qt.MouseButton.LeftButton)
            self.wait_for(lambda: self.window.action is None)
            self.assertIn("Computed (unverified)", self.window.status.text())
            self.assertTrue(self.window.report_button.isEnabled())
            self.assertFalse(self.window.compare_button.isEnabled())
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

    def test_layered_mesh_controls_keep_physical_thickness_and_other_settings(self):
        config = json.loads((ROOT / "configs/perforated_beam_AP_pnc.json").read_bytes())
        config["mesh"].update(kind="layered", thickness_layers=2)
        path = self.output / "layered_AP_pnc.json"
        path.write_text(json.dumps(config), encoding="utf-8")
        self.assertTrue(self.window.load_configuration(path))
        self.assertTrue(self.window.fields["thickness_layers"].isVisible())
        self.assertFalse(self.window.fields["cells"].isVisible())
        self.assertIn("also set the port thickness resolution", self.window.provenance.text())
        self.assertEqual(self.window.fields["mesh_size_nm"].accessibleName(), "In-plane mesh size (nm)")
        self.window.fields["thickness_layers"].setText("3")
        self.window.fields["mesh_size_nm"].setText("120")
        edited = self.window.edited_config()
        self.assertEqual(edited["mesh"]["thickness_layers"], 3)
        self.assertEqual(edited["mesh"]["mesh_size_nm"], 120)
        self.assertEqual(edited["geometry"], config["geometry"])
        self.assertEqual(edited["material"], config["material"])
        self.assertEqual(edited["mesh"]["port_size_nm"], config["mesh"]["port_size_nm"])
        self.window.fields["thickness_layers"].setText("0")
        self.assertFalse(self.window.start_job("mesh"))
        self.assertIn("Thickness layers must be a positive whole number", self.window.status.text())
        self.assertFalse((self.output / "_workbench").exists())
        self.window.preset.setCurrentIndex(2)
        self.assertFalse(self.window.fields["thickness_layers"].isVisible())
        self.assertEqual(self.window.fields["mesh_size_nm"].accessibleName(), "Body mesh size (nm)")

    def test_open_saved_run_uses_selected_folder_without_launching_or_changing_files(self):
        directory = self.output.resolve() / "moved scattering result"
        directory.mkdir()
        config = json.loads((ROOT / "configs" / PRESETS[1][1]).read_bytes())
        config_path = directory / "configuration_AP_pnc.json"
        config_path.write_text(json.dumps(config), encoding="utf-8")
        report = directory / "report_AP_pnc.html"
        report.write_text("Saved report", encoding="utf-8")
        summary_path = directory / "summary_AP_pnc.json"
        summary = {"status": "completed_unverified", "analysis_kind": "scattering", "power_balance_passed": True,
                   "configuration": config, "output_directory": str(self.output / "former location")}
        summary_path.write_text(json.dumps(summary), encoding="utf-8")
        saved_bytes = {path: path.read_bytes() for path in (config_path, report, summary_path)}
        with patch("workbench_AP_pnc.QFileDialog.getExistingDirectory", return_value=str(directory)):
            QTest.mouseClick(self.window.saved_run_button, Qt.MouseButton.LeftButton)
        self.assertIsNone(self.window.action)
        self.assertFalse((self.output / "_workbench").exists())
        self.assertEqual(self.window.result_directory, directory)
        self.assertEqual(self.window.report_path, report)
        self.assertEqual(self.window.scattering_directory, directory)
        self.assertTrue(self.window.compare_button.isEnabled())
        self.assertEqual(self.window.edited_config(), config)
        self.assertEqual({path: path.read_bytes() for path in saved_bytes}, saved_bytes)

        failed = self.output / "failed run"
        failed.mkdir()
        (failed / config_path.name).write_bytes(saved_bytes[config_path])
        (failed / summary_path.name).write_text(json.dumps({**summary, "status": "failed"}), encoding="utf-8")
        self.assertFalse(self.window.open_saved_run(failed))
        self.assertIn("Choose a completed simulation run", self.window.status.text())
        self.assertEqual(self.window.edited_config(), config)
        self.assertEqual(self.window.report_path, report)
        self.assertEqual(self.window.scattering_directory, directory)

        modes = self.output.resolve() / "saved finite modes"
        modes.mkdir()
        mode_config = json.loads((ROOT / "configs" / PRESETS[0][1]).read_bytes())
        (modes / config_path.name).write_text(json.dumps(mode_config), encoding="utf-8")
        (modes / summary_path.name).write_text(json.dumps({
            "status": "completed_unverified", "analysis_kind": "modes", "configuration": mode_config,
        }), encoding="utf-8")
        (modes / report.name).touch()
        self.assertTrue(self.window.open_saved_run(modes))
        self.assertTrue(self.window.report_button.isEnabled())
        self.assertFalse(self.window.compare_button.isEnabled())
        self.assertIsNone(self.window.scattering_directory)

    def test_compare_protocol_keeps_scattering_candidate_and_opens_comparison_report(self):
        candidate = self.output.resolve() / "saved candidate"
        reference = self.output.resolve() / "saved reference"
        comparison = self.output.resolve() / "saved comparison"
        for directory in (candidate, reference, comparison):
            directory.mkdir()
            (directory / "report_AP_pnc.html").touch()
        self.window._show_result("completed_unverified", {
            "output_directory": str(candidate), "analysis_kind": "scattering", "power_balance_passed": True,
        })
        self.assertTrue(self.window.compare_button.isEnabled())

        script = self.output / "synthetic_comparison_worker_AP_pnc.py"
        script.write_text(
            "import json, pathlib, sys\n"
            "args = sys.argv[1:]\n"
            "response = pathlib.Path(args[args.index('--response') + 1])\n"
            "(response.parent / 'observed_args_AP_pnc.json').write_text(json.dumps(args))\n"
            "response.write_text(json.dumps({'action': 'compare', 'status': 'compared', 'summary': {"
            f"'output_directory': {str(comparison)!r}, 'report_path': {str(comparison / 'report_AP_pnc.html')!r}, "
            "'maximum_power_change': 0.00234, 'maximum_complex_change': 0.138, "
            "'power_passed': True, 'complex_passed': False, 'mesh_pair_passed': False, "
            "'comparison_kind': 'mesh_sensitivity', 'input_power_checks_passed': True, "
            "'reference': {'power_balance_passed': True}, 'candidate': {'power_balance_passed': True}}}))\n",
            encoding="utf-8",
        )
        with patch("workbench_AP_pnc.WORKER", script), patch("workbench_AP_pnc.QFileDialog.getExistingDirectory", return_value=str(reference)), patch("workbench_AP_pnc.QDesktopServices.openUrl") as open_url:
            QTest.mouseClick(self.window.compare_button, Qt.MouseButton.LeftButton)
            self.assertEqual(self.window.action, "compare")
            self.assertFalse(self.window.compare_button.isEnabled())
            self.wait_for(lambda: self.window.action is None)
            self.assertIn("0.00234 (passed)", self.window.status.text())
            self.assertIn("0.138 (failed)", self.window.status.text())
            self.assertIn("Mesh-pair check failed for these runs", self.window.status.text())
            self.assertEqual(self.window.scattering_directory, candidate)
            self.assertTrue(self.window.compare_button.isEnabled())
            self.assertEqual(self.window.result_directory, comparison)
            QTest.mouseClick(self.window.report_button, Qt.MouseButton.LeftButton)
            self.assertEqual(Path(open_url.call_args.args[0].toLocalFile()), comparison / "report_AP_pnc.html")

        job = self.window.response_path.parent
        args = json.loads((job / "observed_args_AP_pnc.json").read_bytes())
        for flag, value in (("--action", "compare"), ("--reference", str(reference)), ("--candidate", str(candidate))):
            self.assertEqual(args[args.index(flag) + 1], value)
        self.assertNotIn("--config", args)
        self.assertFalse((job / "configuration_AP_pnc.json").exists())
        request = json.loads((job / "comparison_request_AP_pnc.json").read_bytes())
        self.assertEqual((request["reference"], request["candidate"]), (str(reference), str(candidate)))
        self.window.preset.setCurrentIndex(1)
        self.assertFalse(self.window.compare_button.isEnabled())
        self.assertFalse(self.window.report_button.isEnabled())
        self.assertIsNone(self.window.scattering_directory)

    def test_comparison_status_separates_repeatability_and_input_power_checks(self):
        report = self.output / "report_AP_pnc.html"
        report.touch()
        summary = {
            "output_directory": str(self.output), "report_path": str(report),
            "maximum_power_change": 0, "maximum_complex_change": 0,
            "power_passed": True, "complex_passed": True, "mesh_pair_passed": False,
            "comparison_kind": "repeatability", "input_power_checks_passed": True,
            "reference": {"power_balance_passed": True}, "candidate": {"power_balance_passed": True},
        }
        self.window._show_result("compared", summary)
        self.assertIn("Repeatability", self.window.status.text())
        self.assertIn("same discretization", self.window.status.text())
        self.assertNotIn("Mesh-pair", self.window.status.text())

        summary.update(comparison_kind="mesh_sensitivity", input_power_checks_passed=False,
                       candidate={"power_balance_passed": False})
        self.window._show_result("compared", summary)
        self.assertIn("maximum power change 0 (passed)", self.window.status.text())
        self.assertIn("reference passed, candidate failed", self.window.status.text())
        self.assertIn("Mesh-pair check failed for these runs", self.window.status.text())

    def test_scattering_resume_routes_saved_inputs_and_displays_recoverable_status(self):
        attempt = self.output.resolve() / "preserved attempt"
        attempt.mkdir()
        self.assertFalse(self.window.save_progress.isVisible())
        self.window.preset.setCurrentIndex(1)
        self.assertTrue(self.window.save_progress.isVisible())
        self.assertFalse(self.window.save_progress.isChecked())
        script = self.output / "synthetic_resumable_worker_AP_pnc.py"
        script.write_text(
            "import json, pathlib, sys\n"
            "args = sys.argv[1:]\n"
            "action = args[args.index('--action') + 1]\n"
            "response = pathlib.Path(args[args.index('--response') + 1])\n"
            "(response.parent / 'observed_args_AP_pnc.json').write_text(json.dumps(args))\n"
            "status = 'interrupted' if action == 'run-resumable' else 'blocked'\n"
            "summary = {'status': status, 'accepted_channels': 1, 'total_channels': 4, "
            f"'resume_directory': {str(attempt)!r}, 'can_resume': True, 'message': 'Budget exhausted.'}}\n"
            "response.write_text(json.dumps({'action': action, 'status': status, 'summary': summary}))\n"
            "sys.exit(1 if action == 'resume' else 0)\n",
            encoding="utf-8",
        )
        with patch("workbench_AP_pnc.WORKER", script):
            self.window.save_progress.setChecked(True)
            self.window.wall_seconds.setText("nan")
            QTest.mouseClick(self.window.run_button, Qt.MouseButton.LeftButton)
            self.assertIn("Wall-time limit (s) must be a finite positive number", self.window.status.text())
            self.assertFalse((self.output / "_workbench").exists())

            self.window.wall_seconds.setText("5.5")
            QTest.mouseClick(self.window.run_button, Qt.MouseButton.LeftButton)
            self.assertEqual(self.window.action, "run-resumable")
            self.assertFalse(self.window.resume_button.isEnabled())
            self.wait_for(lambda: self.window.action is None)
            self.assertIn("Scattering interrupted", self.window.status.text())
            self.assertIn("Accepted incident channels: 1/4", self.window.status.text())
            self.assertIn("Budget exhausted", self.window.status.text())
            self.assertEqual(self.window.result_directory, attempt)
            self.assertFalse(self.window.report_button.isEnabled())
            self.assertFalse(self.window.compare_button.isEnabled())
            new_job = self.window.response_path.parent
            new_config = json.loads((new_job / "configuration_AP_pnc.json").read_bytes())
            self.assertEqual(new_config["resources"]["linear_solver"], "ilu_lgmres")
            self.assertNotIn("linear_solver", self.window.edited_config()["resources"])
            new_args = json.loads((new_job / "observed_args_AP_pnc.json").read_bytes())
            self.assertEqual(new_args[new_args.index("--wall-seconds") + 1], "5.5")

            self.window.fields["frequency_GHz"].setText("invalid unsaved field")
            self.window.wall_seconds.setText("25")
            with patch("workbench_AP_pnc.QFileDialog.getExistingDirectory", return_value=str(attempt)) as choose:
                QTest.mouseClick(self.window.resume_button, Qt.MouseButton.LeftButton)
            self.assertEqual(choose.call_args.args[2], str(attempt))
            self.wait_for(lambda: self.window.action is None)
        self.assertIn("Scattering blocked", self.window.status.text())
        resume_job = self.window.response_path.parent
        resume_args = json.loads((resume_job / "observed_args_AP_pnc.json").read_bytes())
        self.assertEqual(resume_args[resume_args.index("--resume-from") + 1], str(attempt))
        self.assertEqual(resume_args[resume_args.index("--wall-seconds") + 1], "25.0")
        self.assertNotIn("--config", resume_args)
        self.assertFalse((resume_job / "configuration_AP_pnc.json").exists())
        self.assertEqual(self.window.fields["frequency_GHz"].text(), "invalid unsaved field")

    def test_resumable_stop_requests_checkpoint_and_uses_saved_progress_without_final_response(self):
        self.window.preset.setCurrentIndex(1)
        attempt = self.output.resolve() / "preserved attempt"
        attempt.mkdir()
        script = self.output / "synthetic_stopped_worker_AP_pnc.py"
        script.write_text(
            "import json, pathlib, sys, time\n"
            "response = pathlib.Path(sys.argv[sys.argv.index('--response') + 1])\n"
            "summary = {'status': 'running', 'phase': 'solving', 'accepted_channels': 1, 'total_channels': 4, "
            f"'resume_directory': {str(attempt)!r}}}\n"
            "response.with_suffix('.progress.json').write_text(json.dumps("
            "{'action': 'run-resumable', 'status': 'running', 'summary': summary}))\n"
            "for i in range(200):\n"
            "    if response.with_suffix('.cancel').exists(): sys.exit(3)\n"
            "    time.sleep(0.05)\n",
            encoding="utf-8",
        )
        with patch("workbench_AP_pnc.WORKER", script):
            self.assertTrue(self.window.start_job("run-resumable"))
            self.wait_for(lambda: "Accepted incident channels: 1/4" in self.window.status.text())
            QTest.mouseClick(self.window.stop_button, Qt.MouseButton.LeftButton)
            self.wait_for(lambda: self.window.action is None)
        self.assertTrue(self.window.response_path.with_suffix(".cancel").is_file())
        self.assertFalse(self.window.response_path.exists())
        self.assertEqual(self.window.resume_directory, attempt)
        self.assertIn("Resume checks the saved checkpoints", self.window.status.text())
        self.assertIn("Last reported accepted incident channels: 1/4", self.window.status.text())
        self.assertTrue(self.window.folder_button.isEnabled())
        self.assertTrue(self.window.resume_button.isEnabled())

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
            "pending = path / 'child_AP_pnc.tmp'\n"
            "pending.write_text(json.dumps(child.pid))\n"
            "pending.replace(path / 'child_AP_pnc.json')\n"
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
