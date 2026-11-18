"""Bounded workbench frequency-map checks. Author: Athul Prem | _AP_pnc."""

from copy import deepcopy
import json
import os
from pathlib import Path
import sys
from tempfile import TemporaryDirectory
import unittest
from unittest.mock import patch

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT), str(ROOT / "src")]

from tools import map_scattering_AP_pnc as mapper
from tools.workbench_worker_AP_pnc import main


class WorkbenchFrequencyWorkerTests(unittest.TestCase):
    def setUp(self):
        temporary = TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        self.directory = Path(temporary.name).resolve()
        self.output = self.directory / "results"
        self.config = json.loads((ROOT / "configs/straight_scattering_AP_pnc.json").read_bytes())
        self.config["geometry"].update(length_nm=1000, width_nm=600)
        self.config["mesh"]["cells"] = [4, 2, 1]
        self.config["analysis"].update(frequency_GHz=0.25, incident_side="right")
        self.config_path = self.directory / "configuration_AP_pnc.json"
        self.config_path.write_text(json.dumps(self.config), encoding="utf-8")
        self.map_directory = self.directory / "bounded map"
        environment = patch.dict(os.environ, dict.fromkeys(mapper.THREADS, "1"))
        environment.start()
        self.addCleanup(environment.stop)

    def execute(self, name, action, *arguments):
        job = self.directory / name
        job.mkdir()
        response = job / "response_AP_pnc.json"
        code = main(["--action", action, "--output-root", str(self.output),
                     "--map-directory", str(self.map_directory), "--response", str(response), *arguments])
        return code, json.loads(response.read_bytes()), response

    def test_bounded_first_point_and_resume_preserve_accepted_work_and_saved_settings(self):
        workflow = mapper.workflow
        solve = workflow.solve_scattering

        def stop_during_first_point(*args, **kwargs):
            # A Stop received inside a point must leave that point's equations,
            # physical gates and accepted export intact before the boundary.
            if args[2] == 250e6:
                (self.directory / "start" / "response_AP_pnc.cancel").touch()
            return solve(*args, **kwargs)

        with (
            patch.object(workflow, "build_mesh", wraps=workflow.build_mesh) as mesh,
            patch.object(workflow, "_assemble_system", wraps=workflow._assemble_system) as assembly,
            patch.object(workflow, "assemble_port", wraps=workflow.assemble_port) as port,
            patch.object(workflow, "solve_port_modes", wraps=workflow.solve_port_modes) as modes,
            patch.object(workflow, "solve_scattering", side_effect=stop_during_first_point) as scattering,
        ):
            preflight = workflow.preflight_mesh(self.config, self.output)
            mesh.assert_called_once()
            code, first, _ = self.execute("start", "map", "--config", str(self.config_path),
                                         "--frequencies-ghz", "0.25", "0.30", "--max-new-points", "1")
            self.assertEqual(code, 0, first)
            self.assertEqual(first["status"], "partial")
            self.assertTrue(first["summary"]["stop_requested"])
            saved = mapper.read_map_checkpoint(self.map_directory)
            self.assertEqual(saved["inputs"]["frequencies_GHz"], [0.25, 0.30])
            self.assertEqual(saved["inputs"]["configuration"], workflow.resolve_config(self.config))
            self.assertEqual([point["status"] for point in saved["points"]], ["accepted", "pending"])
            self.assertEqual(saved["shared"]["mesh_hash"], preflight["mesh_hash"])
            self.assertEqual(saved["shared"]["artifacts"]["mesh"]["sha256"], preflight["mesh_sha256"])
            accepted = self.map_directory / saved["points"][0]["accepted"]["summary"]["path"]
            files = (accepted, accepted.with_name("selected_fields_AP_pnc.npz"))
            preserved = {path: (path.read_bytes(), path.stat().st_mtime_ns) for path in files}
            first_identity = saved["inputs_sha256"]
            self.config_path.unlink()

            code, resumed, _ = self.execute("resume", "map-resume", "--max-new-points", "1")
            self.assertEqual(code, 0, resumed)
            self.assertEqual(resumed["status"], "complete")
            completed = mapper.read_saved_map(self.map_directory)
            self.assertEqual(completed["inputs_sha256"], first_identity)
            self.assertEqual([point["status"] for point in completed["points"]], ["accepted", "accepted"])
            self.assertFalse(completed["spatial_convergence_established"])
            self.assertFalse(completed["frequency_convergence_established"])
            self.assertEqual({path: (path.read_bytes(), path.stat().st_mtime_ns) for path in files}, preserved)
            self.assertEqual([call.args[1] for call in modes.call_args_list], [250e6, 300e6])
            self.assertEqual([call.args[2] for call in scattering.call_args_list], [250e6, 300e6])
            mesh.assert_called_once()
            assembly.assert_called_once()
            port.assert_called_once()

        published_files = tuple(self.map_directory / name for name in (mapper.MAP, mapper.TABLE, mapper.REPORT))
        published = {path: (path.read_bytes(), path.stat().st_mtime_ns) for path in published_files}
        with (
            patch.object(np, "load", side_effect=AssertionError("Loaded scientific arrays while viewing a map")),
            patch.object(workflow, "source_identity", side_effect=AssertionError("Revalidated historical source for viewing")),
            patch.object(mapper, "map_scattering", side_effect=AssertionError("Started a map while viewing")),
        ):
            self.assertEqual(mapper.read_saved_map(self.map_directory), completed)
        self.assertEqual({path: (path.read_bytes(), path.stat().st_mtime_ns) for path in published_files}, published)

    def test_worker_validates_bound_and_saved_only_resume_before_computation(self):
        invalid = (
            ("missing-bound", "map", ["--config", str(self.config_path), "--frequencies-ghz", "0.25"]),
            ("zero-bound", "map", ["--config", str(self.config_path), "--frequencies-ghz", "0.25", "--max-new-points", "0"]),
            ("resume-config", "map-resume", ["--config", str(self.config_path), "--max-new-points", "1"]),
            ("resume-frequencies", "map-resume", ["--frequencies-ghz", "0.25", "--max-new-points", "1"]),
        )
        with patch.object(mapper, "map_scattering", side_effect=AssertionError("Computed invalid map inputs")) as solve:
            for name, action, arguments in invalid:
                with self.subTest(name=name):
                    code, response, _ = self.execute(name, action, *arguments)
                    self.assertEqual(code, 1)
                    self.assertEqual(response["status"], "failed")
                    self.assertIn("error", response)
            solve.assert_not_called()
        self.assertFalse(self.map_directory.exists())

    def test_resume_refuses_imported_read_only_map_and_changed_checkpoint(self):
        self.map_directory.mkdir()
        imported = {"status": "partial", "source": "imported_lawrence", "points": []}
        (self.map_directory / mapper.MAP).write_text(json.dumps(imported), encoding="utf-8")
        preserved = (self.map_directory / mapper.MAP).read_bytes()
        with patch.object(mapper, "map_scattering", side_effect=AssertionError("Resumed an imported read-only map")) as solve:
            code, response, _ = self.execute("imported", "map-resume", "--max-new-points", "1")
            self.assertEqual(code, 1)
            self.assertEqual(response["status"], "failed")
            solve.assert_not_called()
        self.assertEqual((self.map_directory / mapper.MAP).read_bytes(), preserved)

        state = {"schema_version": 1, "inputs": {"configuration": deepcopy(self.config),
                                                   "frequencies_GHz": [0.25, 0.30]}, "points": []}
        state["inputs_sha256"] = mapper.workflow.digest(state["inputs"])
        state["checkpoint_data_sha256"] = mapper.workflow.digest(state)
        state["inputs"]["frequencies_GHz"] = [0.25, 0.35]
        checkpoint = self.map_directory / mapper.CHECKPOINT
        checkpoint.write_text(json.dumps(state), encoding="utf-8")
        original = checkpoint.read_bytes()
        with self.assertRaisesRegex(ValueError, "(?i)changed|checksum|invalid|checkpoint"):
            mapper.read_map_checkpoint(self.map_directory)
        self.assertEqual(checkpoint.read_bytes(), original)


if __name__ == "__main__":
    unittest.main()
