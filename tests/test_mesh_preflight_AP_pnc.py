"""Public mesh preflight and checkpoint checks. Author: Athul Prem | _AP_pnc."""

import contextlib
import io
import json
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from phononic.workflow_AP_pnc import main, preflight_mesh, run_study


class MeshPreflightTests(unittest.TestCase):
    def setUp(self):
        temporary = tempfile.TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        self.output = Path(temporary.name)

    def test_public_preflight_reuses_exact_tagged_gmsh_mesh_after_body_budget_changes(self):
        config = json.loads((ROOT / "configs/perforated_beam_AP_pnc.json").read_bytes())
        config["mesh"].update(element_order=1, mesh_size_nm=240, port_size_nm=250, circle_resolution=8)
        config["resources"]["max_dofs"] = 1
        config_file = self.output / "config_AP_pnc.json"
        config_file.write_text(json.dumps(config), encoding="utf-8")
        argv = ["--config", str(config_file), "--output-root", str(self.output)]

        output = io.StringIO()
        with contextlib.redirect_stdout(output), patch("phononic.workflow_AP_pnc.assemble_elasticity") as assembly:
            self.assertEqual(main([*argv, "--mesh-only"]), 0)
            assembly.assert_not_called()
        self.assertIn("exceeds the configured solve budget", output.getvalue())
        self.assertIn("No mechanical assembly or solve was run", output.getvalue())
        directory = next((self.output / "_meshes").iterdir())
        summary_path = directory / "summary_AP_pnc.json"
        checkpoint_bytes = summary_path.read_bytes()
        checkpoint = json.loads(checkpoint_bytes)
        mesh_path = directory / "mesh_AP_pnc.npz"
        mesh_bytes = mesh_path.read_bytes()
        self.assertEqual(checkpoint["status"], "mesh_ready")
        self.assertGreater(checkpoint["ndof"], 1)
        self.assertGreater(checkpoint["geometry_diagnostics"]["analytic_removed_volume_m3"], 0)
        self.assertEqual(checkpoint["hole_count"], 2)
        self.assertEqual(set(checkpoint["boundary_facet_counts"]), {"xleft", "xright", "free"})
        self.assertEqual(len(list(directory.iterdir())), 2)

        with patch("phononic.workflow_AP_pnc.build_mesh") as builder, patch("phononic.workflow_AP_pnc.assemble_elasticity") as assembly:
            with self.assertRaisesRegex(ValueError, "displacement DOFs, exceeding cap 1"):
                run_study(config, self.output)
            builder.assert_not_called()
            assembly.assert_not_called()
        failed = next(self.output.glob("*/summary_AP_pnc.json"))
        failure = json.loads(failed.read_bytes())
        self.assertEqual(failure["status"], "failed")
        self.assertEqual(failure["ndof"], checkpoint["ndof"])

        with patch("phononic.workflow_AP_pnc.build_mesh") as builder, contextlib.redirect_stdout(io.StringIO()):
            self.assertEqual(main([*argv, "--max-dofs", str(checkpoint["ndof"])]), 0)
            builder.assert_not_called()
        finished = next(json.loads(path.read_bytes()) for path in self.output.glob("*/summary_AP_pnc.json")
                        if path != failed)
        self.assertEqual(finished["status"], "completed_unverified")
        self.assertEqual(finished["mesh_checkpoint"]["mesh_sha256"], checkpoint["mesh_sha256"])
        self.assertEqual(finished["port_dofs"], checkpoint["end_face_dofs"]["xleft"])
        with np.load(mesh_path, allow_pickle=False) as saved:
            with np.load(Path(finished["output_directory"]) / "selected_fields_AP_pnc.npz", allow_pickle=False) as solved:
                np.testing.assert_array_equal(solved["points_m"], saved["points_m"].T)
                np.testing.assert_array_equal(solved["tetrahedra"], saved["tetrahedra"].T)
        self.assertEqual(summary_path.read_bytes(), checkpoint_bytes)
        self.assertEqual(mesh_path.read_bytes(), mesh_bytes)

        config["resources"]["max_port_dofs"] = checkpoint["end_face_dofs"]["xleft"] - 1
        message = rf"Port has {checkpoint['end_face_dofs']['xleft']} DOFs, exceeding dense solver cap"
        with self.assertRaisesRegex(ValueError, message):
            preflight_mesh(config, self.output / "cold")
        with patch("phononic.workflow_AP_pnc.build_mesh") as builder:
            with self.assertRaisesRegex(ValueError, message):
                preflight_mesh(config, self.output)
            builder.assert_not_called()
        self.assertEqual(summary_path.read_bytes(), checkpoint_bytes)
        self.assertEqual(mesh_path.read_bytes(), mesh_bytes)

    def test_checkpoint_identity_ignores_analysis_and_detects_changed_mesh_bytes(self):
        config = json.loads((ROOT / "configs/straight_beam_AP_pnc.json").read_bytes())
        config["mesh"]["cells"] = [2, 2, 1]
        config["resources"]["max_dofs"] = 1
        first = preflight_mesh(config, self.output)
        self.assertEqual(first["ndof"], 225)
        self.assertEqual(first["end_face_dofs"], {"xleft": 45, "xright": 45})
        self.assertFalse(first["within_body_budget"])
        config["analysis"]["num_modes"] = 1
        config["resources"]["max_dofs"] = 225
        second = preflight_mesh(config, self.output)
        self.assertTrue(second["reused"] and second["within_body_budget"])
        self.assertEqual(second["output_directory"], first["output_directory"])
        completed = run_study(config, self.output)
        self.assertFalse(completed["reused"])
        self.assertTrue(run_study(config, self.output)["reused"])
        result_summary = Path(completed["output_directory"]) / "summary_AP_pnc.json"
        completed_bytes = result_summary.read_bytes()
        mesh_path = Path(first["output_directory"]) / "mesh_AP_pnc.npz"
        mesh_path.write_bytes(b"changed mesh bytes")
        with patch("phononic.workflow_AP_pnc.build_mesh") as builder, patch("phononic.workflow_AP_pnc.assemble_elasticity") as assembly:
            with self.assertRaisesRegex(ValueError, "Mesh checkpoint changed or is invalid"):
                run_study(config, self.output)
            builder.assert_not_called()
            assembly.assert_not_called()
        self.assertEqual(mesh_path.read_bytes(), b"changed mesh bytes")
        self.assertEqual(result_summary.read_bytes(), completed_bytes)

    def test_preflight_after_completed_uncached_mesh_uses_separate_run_identity(self):
        config = json.loads((ROOT / "configs/straight_beam_AP_pnc.json").read_bytes())
        config["mesh"]["cells"] = [2, 2, 1]
        first = run_study(config, self.output)
        self.assertNotIn("mesh_checkpoint", first)
        ready = preflight_mesh(config, self.output)
        with patch("phononic.workflow_AP_pnc.build_mesh") as builder:
            second = run_study(config, self.output)
            builder.assert_not_called()
        self.assertFalse(second["reused"])
        self.assertNotEqual(first["run_hash"], second["run_hash"])
        self.assertNotEqual(first["mechanical_operator_hash"], second["mechanical_operator_hash"])
        self.assertEqual(second["mesh_checkpoint"]["mesh_sha256"], ready["mesh_sha256"])


if __name__ == "__main__":
    unittest.main()
