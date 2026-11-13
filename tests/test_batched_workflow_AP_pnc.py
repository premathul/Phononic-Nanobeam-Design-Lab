"""Public batched-assembly workflow check. Author: Athul Prem | _AP_pnc."""

import json
from pathlib import Path
import sys
from tempfile import TemporaryDirectory
import unittest
from unittest.mock import patch

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "tools"))
from compare_scattering_AP_pnc import file_sha, read_run
from phononic.workflow_AP_pnc import preflight_mesh, resolve_config, run_study


class BatchedWorkflowTests(unittest.TestCase):
    def setUp(self):
        self.config = json.loads((ROOT / "configs/straight_scattering_AP_pnc.json").read_bytes())
        self.config["geometry"].update(length_nm=1000, width_nm=600)
        self.config["mesh"]["cells"] = [4, 2, 1]

    def test_batched_scattering_exports_global_dofs_and_reuses_exact_mesh(self):
        config = self.config
        resources = config["resources"]
        for invalid in (True, 0):
            resources["assembly_batch_elements"] = invalid
            with self.subTest(invalid=invalid), self.assertRaisesRegex(ValueError, "assembly_batch_elements"):
                resolve_config(config)

        with TemporaryDirectory() as temporary:
            output = Path(temporary)
            resources["assembly_batch_elements"] = 7
            mesh = preflight_mesh(config, output)
            with patch("phononic.workflow_AP_pnc.build_mesh", side_effect=AssertionError("mesh rebuilt")):
                first = run_study(config, output)
                self.assertEqual(first["assembly_method"], "batched")
                self.assertGreater(first["assembly_resources"]["elapsed_seconds"], 0)
                self.assertTrue(first["power_balance_passed"])
                with patch("phononic.assembly_batched_AP_pnc.assemble_elasticity_batched",
                           side_effect=AssertionError("cached result reassembled")):
                    reused = run_study(config, output)
                self.assertTrue(reused["reused"])
                self.assertEqual(reused["run_hash"], first["run_hash"])

                resources["assembly_batch_elements"] = 11
                self.assertTrue(preflight_mesh(config, output)["reused"])
                second = run_study(config, output)
                self.assertFalse(second["reused"])
                self.assertNotEqual(second["run_hash"], first["run_hash"])
                self.assertNotEqual(second["mechanical_operator_hash"], first["mechanical_operator_hash"])
                self.assertEqual(second["mesh_checkpoint"], first["mesh_checkpoint"])
                self.assertEqual(second["mesh_checkpoint"]["mesh_sha256"], mesh["mesh_sha256"])

                del resources["assembly_batch_elements"]
                standard = run_study(config, output)
                resources.update(assembly_batch_elements=7, linear_solver="ilu_gmres", ilu_fill_factor=6)
                iterative = run_study(config, output)
                resources["linear_solver"] = "ilu_lgmres"
                resources["ilu_precision"] = "single"
                augmented = run_study(config, output)
                with patch("phononic.linear_lgmres_AP_pnc.solve_ilu_lgmres",
                           side_effect=AssertionError("cached augmented result solved again")):
                    self.assertTrue(run_study(config, output)["reused"])
            self.assertEqual(standard["assembly_method"], "standard")
            self.assertEqual(standard["mesh_checkpoint"], first["mesh_checkpoint"])
            self.assertEqual(iterative["mesh_checkpoint"], first["mesh_checkpoint"])
            self.assertNotEqual(iterative["run_hash"], first["run_hash"])
            self.assertTrue(iterative["power_balance_passed"])
            self.assertEqual(augmented["mesh_checkpoint"], first["mesh_checkpoint"])
            self.assertTrue(augmented["power_balance_passed"])
            self.assertNotEqual(augmented["run_hash"], iterative["run_hash"])

            first_path = Path(first["output_directory"])
            source = "src/phononic/assembly_batched_AP_pnc.py"
            _, batched_inputs, _ = read_run(first_path)
            _, standard_inputs, _ = read_run(Path(standard["output_directory"]))
            self.assertEqual(batched_inputs["numerical_source_sha256"][source], file_sha(ROOT / source))
            self.assertNotIn(source, standard_inputs["numerical_source_sha256"])

            with (np.load(first_path / "selected_fields_AP_pnc.npz", allow_pickle=False) as fields,
                  np.load(Path(standard["output_directory"]) / "selected_fields_AP_pnc.npz", allow_pickle=False) as control):
                self.assertEqual(fields["element_dofs"].shape, (30, mesh["mesh_tetrahedra"]))
                np.testing.assert_array_equal(np.unique(fields["element_dofs"]), np.arange(mesh["ndof"]))
                for name in ("element_dofs", "nodal_dofs", "dof_coordinates_m", "dof_component",
                             "port_left_dofs", "port_right_dofs"):
                    np.testing.assert_array_equal(fields[name], control[name])

            iterative_path = Path(iterative["output_directory"])
            saved, iterative_inputs, _ = read_run(iterative_path)
            source = "src/phononic/linear_solvers_AP_pnc.py"
            self.assertEqual(iterative_inputs["numerical_source_sha256"][source], file_sha(ROOT / source))
            self.assertNotIn(source, batched_inputs["numerical_source_sha256"])
            report = saved["linear_solver"]
            self.assertEqual(report["method"], "ilu_gmres")
            self.assertEqual(report["preconditioner"]["fill_factor"], 6)
            self.assertEqual(report["gmres"]["rtol"], 1e-10)
            self.assertGreater(saved["scattering_resources"]["elapsed_seconds"], 0)
            self.assertEqual(len(report["channels"]), saved["num_incident_channels"])
            for channel in report["channels"]:
                self.assertEqual(channel["info"], 0)
                self.assertGreater(channel["iterations"], 0)
                self.assertLessEqual(channel["relative_residual"], report["relative_residual_limit"])
                self.assertLessEqual(channel["relative_backward_residual"], report["backward_residual_limit"])
            with (np.load(iterative_path / "selected_fields_AP_pnc.npz", allow_pickle=False) as fields,
                  np.load(first_path / "selected_fields_AP_pnc.npz", allow_pickle=False) as control):
                for group in ("incident", "reflection", "transmission"):
                    np.testing.assert_array_equal(fields[f"{group}_mode_indices"], control[f"{group}_mode_indices"])
                for block in ("reflection_amplitudes", "transmission_amplitudes"):
                    np.testing.assert_allclose(fields[block], control[block], atol=1e-8, rtol=0)

            augmented_path = Path(augmented["output_directory"])
            saved, augmented_inputs, _ = read_run(augmented_path)
            source = "src/phononic/linear_lgmres_AP_pnc.py"
            self.assertEqual(augmented_inputs["numerical_source_sha256"][source], file_sha(ROOT / source))
            self.assertNotIn(source, iterative_inputs["numerical_source_sha256"])
            self.assertEqual(saved["linear_solver"]["method"], "ilu_lgmres")
            self.assertEqual(saved["linear_solver"]["working_dtype"], "complex128")
            self.assertEqual(saved["linear_solver"]["factor_dtype"], "complex64")
            self.assertEqual(saved["linear_solver"]["lgmres"]["outer_k"], 3)
            self.assertEqual(len(saved["linear_solver"]["channels"]), saved["num_incident_channels"])
            for channel in saved["linear_solver"]["channels"]:
                self.assertEqual(channel["info"], 0)
                self.assertLessEqual(channel["relative_residual"], 1e-9)
                self.assertLessEqual(channel["relative_backward_residual"], 1e-9)
            with (np.load(augmented_path / "selected_fields_AP_pnc.npz", allow_pickle=False) as fields,
                  np.load(first_path / "selected_fields_AP_pnc.npz", allow_pickle=False) as control):
                for block in ("reflection_amplitudes", "transmission_amplitudes"):
                    np.testing.assert_allclose(fields[block], control[block], atol=1e-8, rtol=0)

    def test_iterative_failure_preserves_diagnostics_and_rejects_invalid_options(self):
        config = self.config
        resources = config["resources"]
        for options, error in (({"linear_solver": "cg"}, "linear_solver must be"),
                               ({"ilu_fill_factor": 6}, "requires linear_solver"),
                               ({"linear_solver": "ilu_gmres", "ilu_fill_factor": True}, "must be finite and at least 1"),
                               ({"linear_solver": "ilu_gmres", "ilu_fill_factor": 0}, "must be finite and at least 1"),
                               ({"linear_solver": "ilu_lgmres", "ilu_precision": "half"}, "ilu_precision must be"),
                               ({"linear_solver": "ilu_gmres", "ilu_precision": "single"}, "ilu_precision requires")):
            with self.subTest(options=options), self.assertRaisesRegex(ValueError, error):
                resolve_config({**config, "resources": {**resources, **options}})

        resources.update(assembly_batch_elements=7, linear_solver="ilu_gmres", ilu_fill_factor=6)

        def fail_gmres(_operator, rhs, *, callback, **_options):
            callback(0.5)
            return np.zeros_like(rhs), 10

        with TemporaryDirectory() as temporary:
            output = Path(temporary)
            with (patch("phononic.linear_solvers_AP_pnc.gmres", side_effect=fail_gmres),
                  self.assertRaisesRegex(RuntimeError, "ILU-GMRES failed for RHS 1")):
                run_study(config, output)
            directory, = output.iterdir()
            summary = json.loads((directory / "summary_AP_pnc.json").read_bytes())
            self.assertEqual(summary["status"], "failed")
            self.assertIn("ILU-GMRES failed for RHS 1", summary["error"])
            self.assertGreater(summary["scattering_resources"]["elapsed_seconds"], 0)
            channel, = summary["linear_solver"]["channels"]
            self.assertEqual(channel["info"], 10)
            self.assertEqual(channel["preconditioned_residual_history"], [0.5])
            self.assertAlmostEqual(channel["relative_residual"], 1)
            self.assertFalse((directory / "selected_fields_AP_pnc.npz").exists())
            self.assertFalse((directory / ".run.lock").exists())


if __name__ == "__main__":
    unittest.main()
