"""Headless workbench worker integration tests. Author: Athul Prem | _AP_pnc."""

import json
from pathlib import Path
import subprocess
import sys
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch


ROOT = Path(__file__).resolve().parents[1]
WORKER = ROOT / "tools" / "workbench_worker_AP_pnc.py"


class WorkbenchWorkerTests(unittest.TestCase):
    def setUp(self):
        temporary = tempfile.TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        # The worker returns resolved paths. Windows TEMP can use an 8.3 alias.
        self.directory = Path(temporary.name).resolve()
        self.output = self.directory / "results"
        self.config = json.loads((ROOT / "configs" / "straight_beam_AP_pnc.json").read_bytes())
        self.config["mesh"]["cells"] = [2, 2, 1]
        self.config["analysis"]["num_modes"] = 1

    def job(self, name, config):
        directory = self.output / "_workbench" / name
        directory.mkdir(parents=True)
        config_path = directory / "configuration_AP_pnc.json"
        config_path.write_text(json.dumps(config), encoding="utf-8")
        return config_path, directory / "response_AP_pnc.json"

    def execute(self, action, config_path, response_path):
        return subprocess.run(
            [sys.executable, "-B", str(WORKER), "--action", action, "--config", str(config_path),
             "--output-root", str(self.output), "--response", str(response_path)],
            cwd=self.directory, capture_output=True, encoding="utf-8", timeout=45,
        )

    def test_mesh_and_solve_use_the_same_checkpoint_and_return_exact_paths(self):
        mesh_config, mesh_response = self.job("mesh", self.config)
        completed = self.execute("mesh", mesh_config, mesh_response)
        self.assertEqual(completed.returncode, 0, completed.stderr)
        mesh_result = json.loads(mesh_response.read_bytes())
        self.assertEqual(mesh_result["action"], "mesh")
        self.assertEqual(mesh_result["status"], "mesh_ready")
        mesh_summary = mesh_result["summary"]
        self.assertEqual(mesh_summary["ndof"], 225)
        mesh_directory = Path(mesh_summary["output_directory"])
        self.assertTrue(mesh_directory.is_relative_to(self.output / "_meshes"))
        self.assertTrue((mesh_directory / "mesh_AP_pnc.npz").is_file())
        self.assertEqual(list(self.output.glob("*/summary_AP_pnc.json")), [])

        run_config, run_response = self.job("run", self.config)
        completed = self.execute("run", run_config, run_response)
        self.assertEqual(completed.returncode, 0, completed.stderr)
        run_result = json.loads(run_response.read_bytes())
        self.assertEqual(run_result["action"], "run")
        self.assertEqual(run_result["status"], "completed_unverified")
        solved_summary = run_result["summary"]
        solved_directory = Path(solved_summary["output_directory"])
        self.assertTrue(solved_directory.is_relative_to(self.output))
        saved = json.loads((solved_directory / "summary_AP_pnc.json").read_bytes())
        self.assertEqual(saved["run_hash"], solved_summary["run_hash"])
        self.assertEqual(saved["mesh_checkpoint"]["mesh_sha256"], mesh_summary["mesh_sha256"])
        self.assertFalse(solved_summary["reused"])

        cached_config, cached_response = self.job("cached", self.config)
        completed = self.execute("run", cached_config, cached_response)
        self.assertEqual(completed.returncode, 0, completed.stderr)
        cached = json.loads(cached_response.read_bytes())["summary"]
        self.assertTrue(cached["reused"])
        self.assertEqual(cached["output_directory"], str(solved_directory))
        self.assertEqual(cached["run_hash"], solved_summary["run_hash"])

    def test_invalid_configuration_writes_failure_and_existing_response_is_preserved(self):
        self.config["geometry"]["width_nm"] = 0
        config_path, response_path = self.job("invalid", self.config)
        failed = self.execute("mesh", config_path, response_path)
        self.assertEqual(failed.returncode, 1)
        original_response = response_path.read_bytes()
        result = json.loads(original_response)
        self.assertEqual(result["action"], "mesh")
        self.assertEqual(result["status"], "failed")
        self.assertIn("width_nm must be positive", result["error"])
        self.assertNotIn("summary", result)
        self.assertFalse((self.output / "_meshes").exists())

        self.config["geometry"]["width_nm"] = 1000
        config_path.write_text(json.dumps(self.config), encoding="utf-8")
        refused = self.execute("mesh", config_path, response_path)
        self.assertEqual(refused.returncode, 1)
        self.assertIn("response already exists", refused.stderr)
        self.assertEqual(response_path.read_bytes(), original_response)
        self.assertFalse((self.output / "_meshes").exists())
        self.assertEqual(list(response_path.parent.glob(".response-*.tmp")), [])

    def test_comparison_uses_saved_folders_and_returns_scientific_failure_as_compared(self):
        sys.path.insert(0, str(ROOT / "tools"))
        self.addCleanup(sys.path.remove, str(ROOT / "tools"))
        from workbench_worker_AP_pnc import main

        reference, candidate = self.output / "reference", self.output / "candidate"
        job = self.output / "_workbench" / "compare"
        job.mkdir(parents=True)
        response = job / "response_AP_pnc.json"
        comparison = {
            "status": "compared", "output_directory": str(self.output / "comparison"),
            "report_path": str(self.output / "comparison" / "report_AP_pnc.html"),
            "reference": {"path": str(reference), "power_balance_passed": True},
            "candidate": {"path": str(candidate), "power_balance_passed": True},
            "maximum_power_change": 0.00234, "maximum_complex_change": 0.138,
            "comparison_kind": "mesh_sensitivity", "input_power_checks_passed": True,
            "power_passed": True, "complex_passed": False, "mesh_pair_passed": False,
        }
        args = ["--action", "compare", "--output-root", str(self.output)]
        with patch.dict(sys.modules, {"compare_scattering_AP_pnc": SimpleNamespace(compare_runs=None)}) as modules:
            with patch.object(modules["compare_scattering_AP_pnc"], "compare_runs", return_value=comparison) as compare:
                self.assertEqual(main([*args, "--reference", str(reference), "--candidate", str(candidate),
                                       "--response", str(response)]), 0)
                compare.assert_called_once_with(reference, candidate, self.output)
                self.assertEqual(json.loads(response.read_bytes()), {
                    "action": "compare", "status": "compared", "summary": comparison,
                })

                response = job / "missing_candidate_AP_pnc.json"
                self.assertEqual(main([*args, "--reference", str(reference), "--response", str(response)]), 1)
                compare.assert_called_once()
                failure = json.loads(response.read_bytes())
                self.assertEqual((failure["action"], failure["status"]), ("compare", "failed"))
                self.assertIn("--candidate", failure["error"])


if __name__ == "__main__":
    unittest.main()
