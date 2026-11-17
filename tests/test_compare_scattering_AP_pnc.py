"""Saved scattering comparison checks. Author: Athul Prem | _AP_pnc."""

from copy import deepcopy
import json
from pathlib import Path
import sys
from tempfile import TemporaryDirectory
import unittest
from unittest.mock import patch

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "tools"))

from compare_scattering_AP_pnc import (
    LAWRENCE_DEVICE_DIRECT,
    LAWRENCE_THICKNESS_DIRECT,
    LAWRENCE_UNIFORM_DIRECT,
    LGMRES_SOURCE,
    compare_runs,
    compatible_inputs,
    file_sha,
    read_run,
)
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

    def test_verified_precision_change_retains_physical_and_residual_gates(self):
        config = json.loads((self.reference / "configuration_AP_pnc.json").read_bytes())
        config["resources"].update(linear_solver="ilu_lgmres", ilu_fill_factor=12, ilu_precision="double")
        double = Path(run_study(config, self.output)["output_directory"])
        config["resources"]["ilu_precision"] = "single"
        single = Path(run_study(config, self.output)["output_directory"])
        current = compare_runs(double, single, self.output / "current-precision")
        self.assertTrue(current["complex_passed"])
        self.assertEqual(current["comparison_kind"], "repeatability")
        self.assertFalse(current["mesh_pair_passed"])
        self.assertEqual(current["linear_solver_compatibility"]["checks"]["candidate"]["precision"], "single")

        # Exercise the archived h200 report schema with the unchanged default
        # double-precision numerical path, without repeating that large solve.
        legacy, fields = self.copy_run(double, "legacy-lgmres")
        legacy["implementation"]["source_sha256"][LGMRES_SOURCE] = "543f0be835f2f0f54502eae170584d531bd3126766175d7229a3818360d0aa67"
        for name in ("working_dtype", "factor_dtype"):
            legacy["linear_solver"].pop(name)
        for row in legacy["linear_solver"]["channels"]:
            row["relative_true_residual"] = row.pop("relative_residual")
        write_json(fields.parent / "summary_AP_pnc.json", legacy)
        result = compare_runs(fields.parent, single, self.output / "legacy-precision")
        self.assertLess(result["maximum_complex_change"], 1e-8)
        self.assertTrue(result["input_power_checks_passed"])
        self.assertIsNotNone(result["linear_solver_compatibility"])
        self.assertNotIn(LGMRES_SOURCE, result["physical_inputs"]["numerical_source_sha256"])

        bad, bad_fields = self.copy_run(single, "invalid-precision")
        for field, value in (("relative_residual", 1e-5), ("relative_backward_residual", 1e-5), ("info", 30)):
            altered = json.loads(json.dumps(bad))
            altered["linear_solver"]["channels"][0][field] = value
            write_json(bad_fields.parent / "summary_AP_pnc.json", altered)
            with self.subTest(field=field), self.assertRaisesRegex(ValueError, "residuals|did not converge"):
                compare_runs(fields.parent, bad_fields.parent, self.output / "rejected-precision")
        for source in (LGMRES_SOURCE, "src/phononic/material_AP_pnc.py"):
            altered = json.loads(json.dumps(bad))
            altered["implementation"]["source_sha256"][source] = "0" * 64
            write_json(bad_fields.parent / "summary_AP_pnc.json", altered)
            with self.subTest(source=source), self.assertRaisesRegex(ValueError, "verified LGMRES|physical numerical kernels"):
                compare_runs(fields.parent, bad_fields.parent, self.output / "rejected-precision")
        altered = json.loads(json.dumps(bad))
        altered["linear_solver"]["channels"][0].update(relative_residual=1e-5, relative_true_residual=1e-12)
        write_json(bad_fields.parent / "summary_AP_pnc.json", altered)
        with self.assertRaisesRegex(ValueError, "unambiguous relative residual"):
            compare_runs(fields.parent, bad_fields.parent, self.output / "rejected-precision")

    def test_lawrence_uniform_direct_compatibility_checks_provenance_and_original_residuals(self):
        evidence = json.loads((ROOT / "verification/uniform_h80_AP_pnc.json").read_bytes())
        self.check_lawrence_direct_compatibility(LAWRENCE_UNIFORM_DIRECT, evidence["solve"]["driver_sha256"])

    def test_lawrence_h60_uniform_refinement_retains_fixed_ports_and_original_planes(self):
        driver = file_sha(ROOT / "tools/run_lawrence_uniform_h80_AP_pnc.py")
        self.check_lawrence_direct_compatibility(LAWRENCE_UNIFORM_DIRECT, driver, uniform_refinement=True)

    def test_lawrence_device_direct_compatibility_with_saved_windows_assembly(self):
        driver = file_sha(ROOT / "tools/solve_lawrence_device_refinement_AP_pnc.py")
        self.check_lawrence_direct_compatibility(LAWRENCE_DEVICE_DIRECT, driver)

    def test_lawrence_thickness_direct_compatibility_with_new_safe_basis(self):
        driver = file_sha(ROOT / "tools/solve_lawrence_thickness_AP_pnc.py")
        self.check_lawrence_direct_compatibility(LAWRENCE_THICKNESS_DIRECT, driver)

    def check_lawrence_direct_compatibility(self, method, driver, *, uniform_refinement=False):
        reference, physical, _ = read_run(self.reference)
        direct, other, _ = read_run(self.candidate)
        device = method != LAWRENCE_UNIFORM_DIRECT
        new_safe = method == LAWRENCE_THICKNESS_DIRECT
        scope = "thickness" if new_safe else "device" if device else "uniform"
        source = "459589f4e5de39c888966305e28dbd2b16f2947ec427ae9c131c69ae47c609df"
        helpers = file_sha(ROOT / "tools/lawrence_solver_AP_pnc.py")
        roots = [1348, 1311, 1292, 1284, 1274]
        direct_roots = [1940, 1877, 1867, 1855, 1837] if new_safe else roots
        relative = 2e-11 if device else 2e-10
        channels = [{"column": column, "incident_port_root": root, "info": 0,
                     "relative_residual": relative, "relative_backward_residual": 2e-12}
                    for column, root in enumerate(roots)]
        reference.update(num_incident_channels=5, incident_mode_indices=roots)
        if device:
            replay_method = "lawrence_superlu_replay"
            reference["configuration"]["resources"]["linear_solver"] = "direct"
            reference["linear_solver"] = {"method": replay_method, "working_dtype": "complex128", "channels": channels}
            reference["checkpointed_scattering"] = {
                "method": replay_method, "slurm_job_id": "12344", "windows_assembly_source": reference["implementation"],
            }
            holes = [(index * 500e-9, 100e-9) for index in range(26)]
            physical["placed_holes_m"] = other["placed_holes_m"] = holes
            reference["hole_count"] = direct["hole_count"] = len(holes)
        else:
            reference["configuration"]["resources"].update(linear_solver="ilu_lgmres", ilu_precision="single")
            reference["linear_solver"] = {"method": "ilu_lgmres", "working_dtype": "complex128", "factor_dtype": "complex64",
                                          "channels": channels}
            physical["numerical_source_sha256"][LGMRES_SOURCE] = source
        direct["configuration"]["resources"]["linear_solver"] = "direct"
        direct["configuration"]["mesh"]["kind"] = "layered"
        direct["implementation"]["platform"] = "Linux-test"
        direct_channels = [
            dict(row, incident_port_root=root, accepted_vector={"file": f"accepted-{row['column']:03d}_AP_pnc.npy",
                                                               "bytes": 1024, "sha256": "d" * 64})
            for row, root in zip(channels, direct_roots, strict=True)
        ]
        direct.update(num_incident_channels=5, incident_mode_indices=direct_roots,
                      linear_solver={"method": method, "working_dtype": "complex128",
                                     "operator_csc_sha256": "b" * 64, "rhs_array_sha256": "c" * 64,
                                     "relative_residual_limit": 1e-9, "backward_residual_limit": 1e-9,
                                     "channels": direct_channels},
                      checkpointed_scattering={"method": method, "reused_preflight_mesh": True,
                                               "safe_eigensolve_performed": new_safe,
                                               "direct_solver_helpers": {"sha256": helpers},
                                               "linux_assembly_and_solve": {"implementation": direct["implementation"],
                                                                            "driver": {"sha256": driver},
                                                                            "slurm_job_id": "12345"}})
        if device:
            direct["linear_solver"]["relative_target"] = 1e-10
            provenance = direct["checkpointed_scattering"]
            solve = provenance["linux_assembly_and_solve"]
            provenance["linux_export"] = dict(solve, slurm_job_id="12346")
            solve["implementation"] = dict(direct["implementation"], platform="Linux-solve-test")
            solve["assembly"] = {"implementation": dict(direct["implementation"], platform="Linux-assembly-test"),
                                 "driver": {"sha256": driver}, "slurm_job_id": "12343"}
            provenance["phase_adjustment_applied"] = False
            for key in ("assembly_stage", "solve_receipt", "preflight_receipt"):
                provenance[key] = {"sha256": "1" * 64}
        if new_safe:
            direct["configuration"]["mesh"].update(mesh_size_nm=80, element_order=2, thickness_layers=6, circle_resolution=32)
            direct["port_dofs"] = 975
            safe_driver = file_sha(ROOT / "tools/run_lawrence_thickness_AP_pnc.py")
            provenance.update(linux_safe={"implementation": dict(direct["implementation"], platform="Linux-safe-test"),
                                          "driver": {"sha256": safe_driver}, "slurm_job_id": "12347"},
                              safe_receipt={"sha256": "2" * 64}, safe_modes={"sha256": "3" * 64})
        if uniform_refinement:
            mesh = direct["configuration"]["mesh"]
            mesh.update(mesh_size_nm=60, element_order=2, thickness_layers=4, port_size_nm=1000 / 12)
            dimensions = {"length_nm": 3000, "width_nm": 1000, "thickness_nm": 200}
            direct["configuration"]["geometry"] = dimensions
            direct.update(frequency_hz=1e9, incident_side="left", port_dofs=675)
            physical.update(dimensions_nm=dimensions, frequency_hz=1e9)
            other.update(dimensions_nm=dimensions, frequency_hz=1e9)
            provenance = direct["checkpointed_scattering"]
            evidence = json.loads((ROOT / "verification/uniform_h80_AP_pnc.json").read_bytes())
            provenance.update(end_planes_m=[0.0, 3e-6], preflight_receipt={"sha256": "1" * 64},
                              preflight_driver={"sha256": driver},
                              reused_safe_artifact=evidence["native_ports"]["staged_artifact"])
            archived = deepcopy(direct)
            archived["configuration"]["mesh"]["mesh_size_nm"] = 80
            archived_driver = evidence["solve"]["driver_sha256"]
            archived["checkpointed_scattering"]["linux_assembly_and_solve"]["driver"]["sha256"] = archived_driver
            _, pair = compatible_inputs(archived, direct, other, other)
            self.assertEqual(pair["checks"]["reference"]["source_sha256"], archived_driver)
            self.assertEqual(pair["checks"]["candidate"]["source_sha256"], driver)
        inputs, compatible = compatible_inputs(reference, direct, physical, other)
        self.assertEqual(compatible["checks"]["candidate"], {
            "method": method, "slurm_job_id": "12345", "source_sha256": driver,
            "maximum_relative_residual": relative, "maximum_backward_residual": 2e-12,
        })
        if device:
            self.assertEqual(compatible["checks"]["reference"]["method"], replay_method)
        self.assertNotIn(LGMRES_SOURCE, inputs["numerical_source_sha256"])
        provenance = ("checkpointed_scattering",)
        linux = (*provenance, "linux_assembly_and_solve")
        channel = ("linear_solver", "channels", 0)
        cases = (
            ((*linux, "driver", "sha256"), "0" * 64, "reviewed driver"),
            ((*provenance, "direct_solver_helpers", "sha256"), "0" * 64, "reviewed residual and memory helper"),
            ((*linux, "implementation"), {}, "matching Linux assembly"),
            (("implementation", "packages", "scipy"), "different", "matching numerical package"),
            (("implementation", "source_sha256", "src/phononic/layered_AP_pnc.py"), "0" * 64, "matching layered mesh"),
            ((*channel, "incident_port_root"), 1, f"five Lawrence {scope} channels"),
            ((*channel, "info"), True, "did not finish successfully"),
            ((*channel, "accepted_vector", "file"), "accepted-001_AP_pnc.npy", "accepted vector identity"),
            ((*channel, "relative_residual"), 2e-10 if device else 2e-9, "both original-equation residual limits"),
            ((*channel, "relative_backward_residual"), float("nan"), "both original-equation residual limits"),
            (("hole_count",), 0 if device else 26, "holes and five channels"),
            ((*provenance, "method"), "unreviewed_method", "reviewed Lawrence direct-solve method"),
        )
        if device:
            cases += (
                (("linear_solver", "relative_target"), 1e-9, "original-equation relative target"),
                ((*linux, "assembly", "driver", "sha256"), "0" * 64, "reviewed driver"),
                ((*linux, "assembly", "implementation", "platform"), "Windows-test", "matching Linux assembly"),
                ((*provenance, "linux_export", "implementation"), {}, "matching Linux assembly"),
                ((*provenance, "solve_receipt"), {}, "linked phase receipts"),
                ((*provenance, "phase_adjustment_applied"), True, "unchanged reference-plane phases"),
            )
        if new_safe:
            cases += (
                (("configuration", "mesh", "thickness_layers"), 4, "six layers"),
                (("port_dofs",), 675, "975 port DOFs"),
                ((*provenance, "safe_eigensolve_performed"), False, "mesh and SAFE section"),
                ((*provenance, "safe_modes"), {}, "linked phase receipts"),
                ((*provenance, "linux_safe", "driver", "sha256"), driver, "reviewed driver"),
            )
        if uniform_refinement:
            cases += (
                (("configuration", "mesh", "mesh_size_nm"), 40, "h60 or h80 P2 guide"),
                (("configuration", "mesh", "thickness_layers"), 6, "four thickness layers"),
                (("port_dofs",), 975, "675-DOF ports"),
                ((*provenance, "end_planes_m"), [0.0, 4e-6], "original end planes"),
                ((*provenance, "reused_safe_artifact", "sha256"), "0" * 64, "archived SAFE modes"),
                ((*provenance, "preflight_driver", "sha256"), "0" * 64, "same reviewed driver"),
            )
        for key_path, value, message in cases:
            altered = deepcopy(direct)
            entry = altered
            for key in key_path[:-1]:
                entry = entry[key]
            entry[key_path[-1]] = value
            with self.subTest(key_path=key_path), self.assertRaisesRegex(ValueError, message):
                compatible_inputs(reference, altered, physical, other)
        altered = deepcopy(direct)
        altered["linear_solver"]["channels"][0]["relative_residual"] = 2e-9
        with self.assertRaisesRegex(ValueError, "both original-equation residual limits"):
            compatible_inputs(direct, altered, other, other)

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
