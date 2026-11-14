"""Verify portable full-device scattering evidence. Author: Athul Prem | _AP_pnc."""

import csv
from hashlib import sha256
import io
import json
from math import ceil, floor
from pathlib import Path
import sys
import unittest

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
LEDGER = ROOT / "verification/solver_resources_AP_pnc.json"
sys.path.insert(0, str(ROOT / "src"))


def complex_block(scattering, direction):
    coefficients = np.asarray(scattering[f"{direction}_amplitudes"])
    return coefficients[..., 0] + 1j * coefficients[..., 1]


class SolverResourceEvidenceTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.ledger = json.loads(LEDGER.read_bytes())

    def test_portable_modal_power_matches_boundary_flux(self):
        for case in ("full_device_lgmres_public_replay", "full_device_h160_lgmres_public_replay",
                     "full_device_h130_lgmres_public_replay", "full_device_h100_lgmres_public_replay"):
            with self.subTest(case=case):
                self.check_modal_power(self.ledger[case])

    def check_modal_power(self, result):
        self.assertEqual(result["status"], "completed_unverified")
        scattering = result["scattering"]
        self.assertEqual((scattering["frequency_hz"], scattering["incident_side"]), (1e9, "left"))
        groups = scattering["groups"]
        checks = result["physical_checks"]
        channels = checks["channels"]
        roots = groups["incoming"]["root_indices"]
        self.assertEqual(len(set(roots)), 5)
        self.assertEqual([row["incident_root_index"] for row in channels], roots)
        self.assertEqual(groups["transmission"]["root_indices"], roots)
        self.assertTrue(set(roots).isdisjoint(groups["reflection"]["root_indices"]))
        incident = np.asarray(groups["incoming"]["signed_power_w"])
        self.assertEqual(incident.shape, (5,))
        np.testing.assert_allclose(incident, 1, atol=1e-8, rtol=0)
        totals, fluxes = [], []
        for direction, sign in (("reflection", -1), ("transmission", 1)):
            signed = np.asarray(groups[direction]["signed_power_w"])
            self.assertEqual(signed.shape, (5,))
            np.testing.assert_allclose(signed, sign, atol=1e-8, rtol=0)
            amplitudes = complex_block(scattering, direction)
            self.assertEqual(amplitudes.shape, (5, 5))
            self.assertTrue(np.isfinite(amplitudes).all())
            power = (np.abs(signed) @ (np.abs(amplitudes) ** 2)) / incident
            boundary_flux = np.asarray([row[f"{direction}_flux"] for row in channels])
            np.testing.assert_allclose(power, [row[direction] for row in channels], atol=1e-12, rtol=0)
            np.testing.assert_allclose(power, boundary_flux, atol=1e-8, rtol=0)
            totals.append(power)
            fluxes.append(boundary_flux)
        modal_balance = 1 - sum(totals)
        flux_balance = 1 - sum(fluxes)
        np.testing.assert_allclose(modal_balance, [row["power_balance"] for row in channels], atol=1e-12, rtol=0)
        np.testing.assert_allclose(flux_balance, [row["flux_power_balance"] for row in channels], atol=1e-12, rtol=0)
        self.assertAlmostEqual(np.max(np.abs(modal_balance)), checks["max_power_imbalance"], places=12)
        self.assertAlmostEqual(np.max(np.abs(flux_balance)), checks["maximum_flux_power_imbalance"], places=12)
        disagreement = np.max(np.abs(np.asarray(totals) - np.asarray(fluxes)))
        self.assertAlmostEqual(disagreement, checks["maximum_modal_flux_difference"], places=12)
        passed = (max(np.max(np.abs(modal_balance)), np.max(np.abs(flux_balance)), disagreement,
                      checks["maximum_net_outward_power_ratio"]) < 0.01
                  and max(row["relative_residual"] for row in channels) < 1e-8
                  and checks["maximum_relative_port_residual"] < 1e-8)
        self.assertTrue(passed)
        self.assertEqual(checks["power_balance_passed"], passed)
        self.assertFalse(checks["spatial_convergence_established"])

    def test_terminal_status_is_distinct_from_residual_failure(self):
        first = self.ledger["full_device_lgmres_initial"]
        channel, = first["channels"]
        self.assertEqual(first["status"], "failed")
        self.assertEqual(channel["info"], first["lgmres_settings"]["maxiter"])
        self.assertGreater(channel["last_callback_relative_true_residual"], first["lgmres_settings"]["rtol"])
        self.assertLessEqual(channel["relative_true_residual"], first["lgmres_settings"]["rtol"])
        self.assertLessEqual(channel["relative_backward_residual"], first["backward_residual_limit"])
        self.assertTrue(channel["residual_gates_passed"])
        self.assertFalse(first["candidate_vector_retained"])
        self.assertNotIn("solution", first["raw_files"])
        self.assertFalse(first["scattering_computed"])
        gmres, = self.ledger["full_device_solve_replay"]["iterative_solver"]["channels"]
        self.assertGreater(gmres["relative_residual"], first["relative_residual_limit"])

        corrected = self.ledger["full_device_lgmres_checkpointed"]
        self.assertEqual(corrected["status"], "solved")
        self.assertEqual(len(corrected["channels"]), 5)
        for row in corrected["channels"]:
            self.assertEqual(row["info"], 0)
            self.assertLessEqual(row["relative_true_residual"], corrected["lgmres_settings"]["rtol"])
            self.assertLessEqual(row["relative_backward_residual"], corrected["backward_residual_limit"])

    def test_h160_continuation_reuses_failed_iterate(self):
        stage = self.ledger["full_device_h160_operator"]
        self.assertTrue(stage["port_reuse_proof"]["all_local_safe_arrays_bit_identical"])
        self.assertTrue(stage["port_reuse_proof"]["all_local_physical_rhs_values_bit_identical"])
        solve = self.ledger["full_device_h160_lgmres_fill8"]
        self.assertEqual(solve["status"], "failed")
        column, = solve["channels"]
        self.assertGreater(column["relative_true_residual"], solve["relative_residual_limit"])
        self.assertGreater(column["relative_backward_residual"], solve["backward_residual_limit"])
        self.assertEqual(column["latest_checkpoint"]["stage"], "solver_return")
        self.assertNotIn("solution", column)
        self.assertFalse(solve["scattering_computed"])
        comparison = self.ledger["full_device_h200_h160_comparison"]
        self.assertEqual(comparison["status"], "unevaluated")
        self.assertIsNone(comparison["mesh_pair_passed"])
        self.assertNotIn("maximum_complex_change", comparison)
        self.assertNotIn("maximum_power_change", comparison)

        continued = self.ledger["full_device_h160_lgmres_mixed_fill12"]
        self.assertEqual(continued["status"], "solved")
        self.assertEqual(continued["factor_input_dtype"], "complex64")
        self.assertEqual(continued["outer_dtype"], "complex128")
        self.assertEqual(continued["operator_csc_sha256"], solve["operator_csc_sha256"])
        self.assertEqual(continued["rhs_array_sha256"], solve["rhs_array_sha256"])
        self.assertEqual(continued["initial_iterate"]["artifact"]["sha256"], column["latest_checkpoint"]["sha256"])
        self.assertEqual([row["column"] for row in continued["channels"]], list(range(5)))
        for i, row in enumerate(continued["channels"]):
            self.assertEqual(row["info"], 0)
            self.assertEqual(row["status"], "accepted")
            self.assertLessEqual(row["relative_true_residual"], continued["lgmres_settings"]["rtol"])
            self.assertLessEqual(row["relative_backward_residual"], continued["backward_residual_limit"])
            expected_initial = 1.0 if i else column["relative_true_residual"]
            self.assertEqual(row["initial_relative_true_residual"], expected_initial)

    def test_h100_stage_preserves_geometry_and_exact_ports(self):
        preflight = self.ledger["h100_candidate_preflight"]
        prior = self.ledger["full_device_h130_operator"]
        operator = self.ledger["full_device_h100_operator"]
        self.assertEqual(preflight["status"], "mesh_ready")
        self.assertFalse(preflight["within_body_budget"])
        self.assertEqual(preflight["hole_count"], 26)
        self.assertEqual(preflight["end_face_dofs"], {"xleft": 675, "xright": 675})
        self.assertEqual(operator["status"], "operator_ready")
        self.assertFalse(operator["factorization_attempted"])
        self.assertEqual(operator["raw_files"]["mesh"], preflight["raw_files"]["mesh"])
        self.assertEqual(operator["mesh_hash"], preflight["mesh_hash"])
        self.assertNotEqual(operator["mesh_hash"], prior["mesh_hash"])
        self.assertEqual(operator["mesh_comparison"], {
            "exact_yz_triangulations": True, "exact_z_planes": True,
            "tetrahedra": preflight["mesh_tetrahedra"], "vertices": preflight["mesh_vertices"], "ndof": preflight["ndof"],
        })
        self.assertGreater(preflight["ndof"], prior["mesh_comparison"]["ndof"])
        config = operator["configuration"]
        for section in ("geometry", "material", "analysis"):
            self.assertEqual(config[section], prior["configuration"][section])
        self.assertEqual(config["mesh"], prior["configuration"]["mesh"] | {"mesh_size_nm": 100})
        self.assertEqual(operator["port_modes"]["source_modes"], prior["raw_files"]["port_modes"])
        self.assertEqual(operator["port_modes"]["propagating_channels_per_direction"], 5)
        self.assertFalse(operator["port_modes"]["phase_adjustment_applied"])
        proof = operator["port_reuse_proof"]
        self.assertEqual(proof["reference_stage"], prior["raw_files"]["stages"])
        self.assertTrue(proof["all_local_safe_arrays_bit_identical"])
        self.assertTrue(proof["all_local_physical_rhs_values_bit_identical"])
        self.assertEqual(proof["rhs_nonzero_rows_per_channel"], [675] * 5)
        nnz = operator["operator"]["nonzero_storage"]
        self.assertEqual(operator["operator"]["array_bytes"], 20 * nnz + 4 * (preflight["ndof"] + 1))
        self.assertEqual(operator["predicted_initial_ilu_buffers_mib"]["complex64_fill12"], 24 * 12 * nnz / 2**20)
        run = operator["run"]
        self.assertEqual(run["worker_exit_code"], 0)
        self.assertGreaterEqual(run["minimum_available_mib"], run["available_floor_mib"])
        self.assertGreaterEqual(run["minimum_commit_headroom_mib"], run["commit_headroom_floor_mib"])

    def test_h80_preflight_replays_the_resource_decision(self):
        from phononic.resumable_scattering_AP_pnc import _solver_budget

        preflight = self.ledger["h80_candidate_preflight"]
        h100 = self.ledger["h100_candidate_preflight"]
        config = preflight["configuration"]
        self.assertEqual(preflight["status"], "mesh_ready")
        self.assertGreater(preflight["ndof"], h100["ndof"])
        self.assertGreater(preflight["ndof"], config["resources"]["max_dofs"])
        self.assertFalse(preflight["within_body_budget"])
        self.assertEqual(config["mesh"], h100["settings"] | {"mesh_size_nm": 80})
        self.assertEqual(preflight["end_face_dofs"], h100["end_face_dofs"])
        self.assertEqual(preflight["geometry_diagnostics"], h100["geometry_diagnostics"])
        self.assertEqual(preflight["raw_files"]["mesh"]["sha256"], preflight["mesh_sha256"])
        bounds = preflight["boundary_verification"]
        self.assertTrue(bounds["z_planes_exact_h100_match"])
        self.assertTrue(bounds["hole_wall_connectivity_exact_h100_match"])
        self.assertEqual(bounds["maximum_h100_hole_wall_vertex_change_m"], 0)
        self.assertEqual(set(bounds["ports"]), set(h100["end_face_dofs"]))
        self.assertTrue(all(port["exact_h100_yz_match"] for port in bounds["ports"].values()))

        forecast = preflight["resource_forecast"]
        for status in ("staging_performed", "factorization_performed", "scattering_computed"):
            self.assertFalse(forecast[status])
        snapshot = forecast["host_snapshot"]
        cases = forecast["historical_cases"]
        stages = [self.ledger[case["operator_case"]] for case in cases]
        solves = [self.ledger[case["solve_case"]] for case in cases]
        rates = [stage["operator"]["nonzero_storage"] / solve["body_dofs"] for stage, solve in zip(stages, solves)]
        reference = self.ledger[forecast["solver_reference_case"]]
        peak = reference["run"]["peak_tree_private_commit_mib"]
        options = {"single_precision": reference["factor_input_dtype"] == "complex64",
                   "fill_factor": reference["ilu_settings"]["fill_factor"], **reference["lgmres_settings"]}
        dofs, channels = preflight["ndof"], forecast["expected_incident_drives"]
        index_bytes = forecast["csc_index_bytes_assumed"]
        self.assertEqual(channels, len(reference["channels"]))
        for endpoint, nnz in (("low_estimate", floor(dofs * min(rates))),
                              ("high_estimate", ceil(dofs * max(rates)))):
            estimate = forecast[endpoint]
            storage = (16 + index_bytes) * nnz + index_bytes * (dofs + 1)
            self.assertEqual(estimate["operator_nnz"], nnz)
            self.assertEqual(estimate["operator_storage_bytes"], storage)
            manifest = {"shape": [dofs, channels], "nnz": nnz,
                        "operator_storage_bytes": storage, "index_bytes": index_bytes}
            budget = _solver_budget(manifest, options, 0, peak)
            for key, value in budget.items():
                self.assertAlmostEqual(value, estimate[key])
            self.assertGreater(budget["required_commit_mib"], snapshot["commit_headroom_mib"])

        stage = forecast["operator_only_stage"]
        high_storage = forecast["high_estimate"]["operator_storage_bytes"]
        for observed, reserve, estimated, measured in (
            ("physical_available_mib", "physical_reserve_mib", "estimated_peak_tree_rss_mib", "peak_tree_rss_mib"),
            ("commit_headroom_mib", "commit_reserve_mib", "estimated_peak_tree_private_commit_mib",
             "peak_tree_private_commit_mib"),
        ):
            planning_peak = max(old["run"][measured] / old["operator"]["array_bytes"] for old in stages) * high_storage
            self.assertAlmostEqual(stage[estimated], planning_peak)
            self.assertGreater(snapshot[observed], planning_peak + stage[reserve])
        self.assertGreater(snapshot["disk_free_gib"] * 1024, stage["estimated_checkpoint_mib"])
        self.assertEqual(stage["safe_reference"], self.ledger["full_device_h100_operator"]["raw_files"]["port_modes"])
        self.assertEqual(forecast["solve_admission"], "deferred_until_actual_storage_and_sufficient_live_memory")

        table = LEDGER.with_suffix(".csv").read_bytes()
        self.assertEqual(sha256(table).hexdigest(), self.ledger["csv_sha256"])
        row, = [row for row in csv.DictReader(io.StringIO(table.decode("utf-8")))
                if row["case"] == "h80_candidate_preflight"]
        self.assertEqual((row["phase"], row["status"], int(row["body_dofs"])), ("mesh_only", "mesh_ready", dofs))
        self.assertEqual(float(row["elapsed_seconds"]), preflight["mesh_resources"]["elapsed_seconds"])
        self.assertEqual(row["operator_nnz"], "")
        self.assertEqual(row["factor_nnz"], "")

    def test_h80_operator_replays_actual_storage_admission(self):
        from phononic.resumable_scattering_AP_pnc import _solver_budget, _solver_options

        stage = self.ledger["full_device_h80_operator"]
        preflight = self.ledger["h80_candidate_preflight"]
        reference = self.ledger["full_device_h100_operator"]
        config = stage["configuration"]
        ndof = preflight["ndof"]
        self.assertEqual(stage["status"], "operator_ready")
        for status in ("mesh_generated", "safe_eigensolve_attempted", "factorization_attempted",
                       "solve_attempted", "scattering_computed", "spatial_convergence_established"):
            self.assertFalse(stage[status])
        raw = stage["raw_files"]
        for key, saved in (("mesh", "mesh"), ("mesh_checkpoint_summary", "mesh_summary"),
                           ("preflight", "preflight"), ("preflight_configuration", "configuration")):
            self.assertEqual(raw[key], preflight["raw_files"][saved])
        self.assertEqual(stage["mesh_hash"], preflight["mesh_hash"])
        self.assertEqual(raw["h100_stage"], reference["raw_files"]["stages"])
        for key in ("geometry", "material", "analysis", "mesh"):
            self.assertEqual(config[key], preflight["configuration"][key])
        self.assertEqual(config["resources"]["max_dofs"], ndof)
        self.assertEqual(config["resources"]["max_dofs"], stage["stage_only_body_dof_cap"])
        self.assertEqual(config["resources"]["assembly_batch_elements"], 2048)
        self.assertFalse(preflight["within_body_budget"])
        for key in ("exact_yz_triangulations", "exact_x_end_planes", "exact_z_planes",
                    "exact_hole_walls_from_saved_preflight"):
            self.assertTrue(stage["mesh_comparison"][key])

        proof = stage["port_reuse_proof"]
        self.assertEqual(proof["reference_port_modes"], reference["raw_files"]["port_modes"])
        self.assertEqual(proof["reference_rhs"], reference["raw_files"]["rhs"])
        self.assertTrue(proof["all_local_safe_arrays_bit_identical"])
        self.assertTrue(proof["all_local_physical_rhs_values_bit_identical"])
        self.assertFalse(proof["phase_adjustment_applied"])
        self.assertEqual(proof["complete_discrete_roots"], 2 * preflight["end_face_dofs"]["xleft"])
        h100_scattering = self.ledger["full_device_h100_lgmres_public_replay"]["scattering"]
        self.assertEqual(proof["incoming_root_indices"], h100_scattering["groups"]["incoming"]["root_indices"])
        channels = len(proof["incoming_root_indices"])
        for side, port in proof["trace_mapping"].items():
            self.assertTrue(port["matches_h100_in_saved_safe_order"])
            self.assertEqual(port["trace_dofs"], preflight["end_face_dofs"][side])
            self.assertEqual(port["end_plane_m"], preflight["boundary_verification"]["ports"][side]["x_m"])
        self.assertEqual(proof["rhs_nonzero_rows_per_channel"], [preflight["end_face_dofs"]["xleft"]] * channels)
        self.assertEqual(proof["rhs_shape"], [ndof, channels])
        self.assertEqual(proof["rhs_array_bytes"], ndof * channels * np.dtype("complex128").itemsize)
        self.assertEqual(stage["checkpoint_bytes"], sum(raw[key]["bytes"] for key in ("K", "M", "A", "rhs", "port_modes")))
        for key, record, fmt, dtype in (
            ("K", "stiffness", "csr", "float64"),
            ("M", "mass", "csr", "float64"),
            ("A", "operator", "csc", "complex128"),
        ):
            matrix = stage[record]
            value_bytes = np.dtype(dtype).itemsize
            index_bytes = np.dtype(matrix["index_dtype"]).itemsize
            self.assertEqual(matrix["index_dtype"], "int32")
            self.assertEqual((matrix["format"], matrix["data_dtype"], matrix["shape"]), (fmt, dtype, [ndof, ndof]))
            storage = (value_bytes + index_bytes) * matrix["nonzero_storage"] + index_bytes * (ndof + 1)
            self.assertEqual(matrix["array_bytes"], storage)
            self.assertGreaterEqual(raw[key]["bytes"], storage)

        run = stage["run"]
        self.assertEqual(run["worker_exit_code"], 0)
        self.assertLessEqual(run["elapsed_seconds"], run["wall_cap_seconds"])
        self.assertLessEqual(run["peak_tree_rss_mib"], run["rss_cap_mib"])
        self.assertLessEqual(run["peak_tree_private_commit_mib"], run["private_commit_cap_mib"])
        self.assertGreaterEqual(run["minimum_available_mib"], run["available_floor_mib"])
        self.assertGreaterEqual(run["minimum_commit_headroom_mib"], run["commit_headroom_floor_mib"])
        admission = stage["solve_admission"]
        self.assertEqual(admission["measured_stage_peak_private_commit_mib"], run["peak_tree_private_commit_mib"])
        historical = self.ledger[admission["solver_reference_case"]]
        for key in ("ilu_settings", "lgmres_settings", "factor_input_dtype", "outer_dtype",
                    "relative_residual_limit", "backward_residual_limit"):
            self.assertEqual(admission[key], historical[key])
        options = _solver_options(config)
        self.assertEqual(options, {"single_precision": True, "fill_factor": 12.0,
                                   "inner_m": 100, "outer_k": 10, "maxiter": 60})
        matrix = stage["operator"]
        manifest = {"shape": proof["rhs_shape"], "nnz": matrix["nonzero_storage"],
                    "operator_storage_bytes": matrix["array_bytes"],
                    "index_bytes": np.dtype(matrix["index_dtype"]).itemsize}
        self.assertEqual(admission["accepted_channels"], 0)
        ilu_bytes = (np.dtype(admission["factor_input_dtype"]).itemsize + manifest["index_bytes"]) * 2
        self.assertAlmostEqual(admission["initial_ilu_buffer_mib"],
                               ilu_bytes * options["fill_factor"] * manifest["nnz"] / 2**20)
        h100_peak = historical["run"]["peak_tree_private_commit_mib"]
        self.assertEqual(admission["h100_solve_peak_private_mib"], h100_peak)
        budget = _solver_budget(manifest, options, 0, h100_peak)
        for key, value in budget.items():
            self.assertAlmostEqual(admission[key], value)
        scale = matrix["array_bytes"] / reference["operator"]["array_bytes"]
        self.assertAlmostEqual(admission["h100_scaled_solve_peak_private_mib"], h100_peak * scale)
        snapshot = admission["host_snapshot"]
        self.assertGreaterEqual(snapshot["physical_available_mib"], budget["required_physical_mib"])
        self.assertLess(snapshot["commit_headroom_mib"], budget["required_commit_mib"])
        self.assertAlmostEqual(admission["commit_shortfall_mib"],
                               budget["required_commit_mib"] - snapshot["commit_headroom_mib"])
        self.assertEqual(admission["status"], "blocked_insufficient_commit")

        table = LEDGER.with_suffix(".csv").read_bytes()
        self.assertEqual(sha256(table).hexdigest(), self.ledger["csv_sha256"])
        rows = {row["phase"]: row for row in csv.DictReader(io.StringIO(table.decode("utf-8")))
                if row["case"] == "full_device_h80_operator"}
        self.assertEqual(set(rows), {"production_batched_assembly", "operator_stage_total"})
        for row in rows.values():
            self.assertEqual(row["status"], stage["status"])
            self.assertEqual(int(row["body_dofs"]), ndof)
            self.assertEqual(int(row["operator_nnz"]), matrix["nonzero_storage"])
            self.assertEqual(row["factor_nnz"], "")
        self.assertEqual(float(rows["operator_stage_total"]["elapsed_seconds"]), run["elapsed_seconds"])
        assembly = stage["phases"]["production_batched_assembly"]
        self.assertEqual(float(rows["production_batched_assembly"]["elapsed_seconds"]), assembly["elapsed_seconds"])

    def test_h80_first_drive_pilot_replays_memory_and_residual_gates(self):
        from phononic.resumable_scattering_AP_pnc import _solver_budget, _solver_options

        pilot = self.ledger["full_device_h80_lgmres_fill8_pilot"]
        stage = self.ledger[pilot["reference_operator_case"]]
        self.assertEqual(stage["solve_admission"]["status"], "blocked_insufficient_commit")
        for key in ("A", "rhs", "stages"):
            self.assertEqual(pilot["raw_files"][key], stage["raw_files"][key])
        for key in ("source_sha256", "packages", "python", "platform"):
            self.assertEqual(pilot["source"][key], stage["source"][key])
        self.assertEqual(pilot["incident_columns"], [0])
        self.assertEqual(pilot["incident_root_index"], stage["port_reuse_proof"]["incoming_root_indices"][0])
        self.assertEqual(pilot["total_device_channels"], stage["port_reuse_proof"]["rhs_shape"][1])
        self.assertFalse(pilot["scattering_computed"])
        self.assertFalse(pilot["spatial_convergence_established"])

        options = _solver_options(stage["configuration"]) | {"fill_factor": 8.0}
        self.assertEqual(pilot["solver_options"], options)
        matrix = stage["operator"]
        manifest = {"shape": [pilot["body_dofs"], 1], "nnz": matrix["nonzero_storage"],
                    "operator_storage_bytes": matrix["array_bytes"],
                    "index_bytes": np.dtype(matrix["index_dtype"]).itemsize}
        self.assertEqual(pilot["admission_manifest"], manifest)
        self.assertEqual(pilot["admission"], _solver_budget(manifest, options, 0))
        admission = pilot["admission"]
        run = pilot["run"]
        self.assertGreaterEqual(run["initial_memory"]["physical_available_mib"], admission["required_physical_mib"])
        self.assertGreaterEqual(run["initial_memory"]["commit_headroom_mib"], admission["required_commit_mib"])
        loaded_equation_mib = (matrix["array_bytes"] + pilot["body_dofs"] * np.dtype("complex128").itemsize) / 2**20
        self.assertAlmostEqual(pilot["required_pre_factor_commit_headroom_mib"],
                               admission["required_commit_mib"] - loaded_equation_mib)
        self.assertGreaterEqual(pilot["before_factorization"]["commit_headroom_mib"],
                                pilot["required_pre_factor_commit_headroom_mib"])
        respects_limits = (run["minimum_available_mib"] >= run["available_floor_mib"]
                           and run["minimum_commit_headroom_mib"] >= run["commit_headroom_floor_mib"]
                           and run["peak_tree_rss_mib"] <= run["rss_cap_mib"]
                           and run["peak_tree_private_commit_mib"] <= run["private_commit_cap_mib"])
        self.assertEqual(pilot["memory_limits_respected"], respects_limits)
        self.assertEqual(run["status"], "aborted_memory_limit")
        self.assertEqual(run["status"], pilot["status"])
        self.assertGreater(run["peak_tree_rss_mib"], run["rss_cap_mib"])
        self.assertLessEqual(run["peak_tree_private_commit_mib"], run["private_commit_cap_mib"])
        self.assertGreaterEqual(run["minimum_available_mib"], run["available_floor_mib"])
        self.assertGreaterEqual(run["minimum_commit_headroom_mib"], run["commit_headroom_floor_mib"])
        self.assertTrue(pilot["factorization_attempted"])
        self.assertEqual(pilot["factor_dtype"], "complex64")
        self.assertEqual(pilot["equation_dtype"], "complex128")
        factorization = pilot["factorization"]
        self.assertEqual(factorization["fill_factor"], options["fill_factor"])
        self.assertGreater(factorization["factor_nonzeros"], 0)
        self.assertGreater(factorization["elapsed_seconds"], 0)

        kept = pilot["retained_checkpoint"]
        self.assertEqual(kept["artifact"], pilot["raw_files"]["retained_vector"])
        self.assertEqual(kept["column"], 0)
        self.assertEqual(kept["event"], "iterate")
        diagnostics = kept["diagnostics"]
        history = diagnostics["relative_residual_history"]
        self.assertEqual(len(history), diagnostics["iterations"])
        self.assertAlmostEqual(history[0], 1.0)
        observed = kept["independent_original_equation_checks"]
        saved = diagnostics["relative_residual"]
        self.assertTrue(np.isclose(observed["relative_residual"], history[-1] if saved is None else saved,
                                   rtol=1e-10, atol=1e-15))
        passed = (observed["relative_residual"] <= pilot["relative_residual_limit"]
                  and observed["relative_backward_residual"] <= pilot["backward_residual_limit"])
        self.assertEqual(kept["original_residual_gates_passed"], passed)
        self.assertFalse(passed)
        self.assertIsNone(diagnostics["info"])
        self.assertEqual(pilot["accepted_channels"], 0)
        self.assertEqual(run["accepted_channels"], pilot["accepted_channels"])

        table = LEDGER.with_suffix(".csv").read_bytes()
        self.assertEqual(sha256(table).hexdigest(), self.ledger["csv_sha256"])
        row, = [row for row in csv.DictReader(io.StringIO(table.decode("utf-8")))
                if row["case"] == "full_device_h80_lgmres_fill8_pilot"]
        self.assertEqual((row["phase"], row["status"]), ("first_drive_pilot", pilot["status"]))
        self.assertEqual(int(row["factor_nnz"]), factorization["factor_nonzeros"])
        self.assertEqual(float(row["max_relative_true_residual"]), observed["relative_residual"])
        self.assertEqual(float(row["max_relative_backward_residual"]), observed["relative_backward_residual"])

    def test_h80_continuation_reuses_first_drive_with_original_equation_gates(self):
        from phononic.resumable_scattering_AP_pnc import _solver_budget

        continuation = self.ledger["full_device_h80_lgmres_fill8_continuation"]
        previous = self.ledger[continuation["previous_attempt_case"]]
        prior = continuation["resumed_from"]
        for key, previous_key in (("execution", "execution"), ("journal", "channel_journal"),
                                  ("profile", "profile"), ("seed", "retained_vector")):
            self.assertEqual(prior[key], previous["raw_files"][previous_key])
        for key in ("A", "rhs", "stages"):
            self.assertEqual(continuation["raw_files"][key], previous["raw_files"][key])
        for key in ("equation_dtype", "factor_dtype", "solver_options", "admission_manifest",
                    "relative_tolerance", "absolute_tolerance", "relative_residual_limit", "backward_residual_limit"):
            self.assertEqual(continuation[key], previous[key])
        self.assertEqual(continuation["incident_columns"], [0])
        self.assertTrue(prior["preconditioner_rebuild_required"])
        seed = previous["retained_checkpoint"]
        self.assertEqual(prior["original_csc_digest"], seed["operator_csc_sha256"])
        self.assertEqual(prior["original_rhs_column_digest"], seed["input_rhs_column_sha256"])
        self.assertEqual(prior["saved_outer_callbacks"], seed["diagnostics"]["iterations"])
        initial_check = continuation["independent_initial_equation_checks"]
        self.assertEqual(initial_check, seed["independent_original_equation_checks"])
        self.assertAlmostEqual(initial_check["relative_residual"], continuation["verified_initial_relative_residual"])

        run = continuation["run"]
        prior_run = previous["run"]
        self.assertEqual(prior["previous_wall_seconds"], prior_run["elapsed_seconds"])
        self.assertEqual(prior["previous_peak_private_commit_mib"], prior_run["peak_tree_private_commit_mib"])
        self.assertEqual(run["wall_cap_seconds"], floor(prior_run["wall_cap_seconds"] - prior["previous_wall_seconds"]))
        self.assertEqual(continuation["total_pilot_elapsed_seconds"], prior["previous_wall_seconds"] + run["elapsed_seconds"])
        self.assertEqual(run["rss_cap_mib"], 4608)
        for key in ("available_floor_mib", "commit_headroom_floor_mib", "private_commit_cap_mib"):
            self.assertEqual(run[key], prior_run[key])
        budget = _solver_budget(continuation["admission_manifest"], continuation["solver_options"], 0,
                                prior["previous_peak_private_commit_mib"])
        self.assertEqual(continuation["admission"], budget)
        self.assertEqual(run["admission"], budget)
        self.assertGreaterEqual(run["initial_memory"]["physical_available_mib"], budget["required_physical_mib"])
        self.assertGreaterEqual(run["initial_memory"]["commit_headroom_mib"], budget["required_commit_mib"])
        respects_limits = (run["minimum_available_mib"] >= run["available_floor_mib"]
                           and run["minimum_commit_headroom_mib"] >= run["commit_headroom_floor_mib"]
                           and run["peak_tree_rss_mib"] <= run["rss_cap_mib"]
                           and run["peak_tree_private_commit_mib"] <= run["private_commit_cap_mib"])
        self.assertEqual(continuation["memory_limits_respected"], respects_limits)
        self.assertEqual(continuation["status"], "aborted_memory_limit")
        self.assertEqual(continuation["status"], run["status"])
        self.assertLess(run["minimum_available_mib"], run["available_floor_mib"])
        self.assertGreaterEqual(run["minimum_commit_headroom_mib"], run["commit_headroom_floor_mib"])
        self.assertLessEqual(run["peak_tree_rss_mib"], run["rss_cap_mib"])
        self.assertLessEqual(run["peak_tree_private_commit_mib"], run["private_commit_cap_mib"])
        self.assertEqual(continuation["accepted_channels"], run["accepted_channels"])
        self.assertFalse(continuation["scattering_computed"])
        self.assertFalse(continuation["spatial_convergence_established"])

        kept = continuation["retained_checkpoint"]
        self.assertEqual(kept["column"], 0)
        self.assertEqual(kept["event"], "iterate")
        self.assertEqual(kept["artifact"], continuation["raw_files"]["retained_vector"])
        diagnostic = kept["diagnostics"]
        history = diagnostic["relative_residual_history"]
        self.assertTrue(diagnostic["warm_started"])
        self.assertAlmostEqual(history[0], initial_check["relative_residual"])
        self.assertEqual(len(history), diagnostic["iterations"])
        checked = kept["independent_original_equation_checks"]
        expected = history[-1] if diagnostic["relative_residual"] is None else diagnostic["relative_residual"]
        self.assertTrue(np.isclose(checked["relative_residual"], expected, rtol=1e-10, atol=1e-15))
        passed = (checked["relative_residual"] <= continuation["relative_residual_limit"]
                  and checked["relative_backward_residual"] <= continuation["backward_residual_limit"])
        self.assertEqual(kept["original_residual_gates_passed"], passed)
        self.assertFalse(passed)
        self.assertEqual(continuation["accepted_channels"], 0)
        self.assertLess(checked["relative_residual"], initial_check["relative_residual"])
        self.assertLess(checked["relative_backward_residual"], initial_check["relative_backward_residual"])
        self.assertEqual(continuation["factorization"]["factor_nonzeros"], previous["factorization"]["factor_nonzeros"])

        table = LEDGER.with_suffix(".csv").read_bytes()
        self.assertEqual(sha256(table).hexdigest(), self.ledger["csv_sha256"])
        row, = [row for row in csv.DictReader(io.StringIO(table.decode("utf-8")))
                if row["case"] == "full_device_h80_lgmres_fill8_continuation"]
        self.assertEqual((row["phase"], row["status"]), ("first_drive_continuation", run["status"]))
        self.assertEqual(float(row["elapsed_seconds"]), run["elapsed_seconds"])
        self.assertEqual(float(row["max_relative_true_residual"]), checked["relative_residual"])
        self.assertEqual(float(row["max_relative_backward_residual"]), checked["relative_backward_residual"])

    def test_h80_inner50_continuation_reduces_workspace_with_same_equations(self):
        from phononic.resumable_scattering_AP_pnc import _solver_budget

        result = self.ledger["full_device_h80_lgmres_fill8_inner50_continuation"]
        previous = self.ledger[result["previous_attempt_case"]]
        starting = result["resumed_from"]
        for artifact, name in (("execution", "execution"), ("journal", "channel_journal"),
                               ("profile", "profile"), ("seed", "retained_vector")):
            self.assertEqual(starting[artifact], previous["raw_files"][name])
        self.assertEqual(starting["earlier_pilot"], previous["resumed_from"])
        self.assertTrue(starting["preconditioner_rebuild_required"])
        for key in ("A", "rhs", "stages"):
            self.assertEqual(result["raw_files"][key], previous["raw_files"][key])
        for key in ("equation_dtype", "factor_dtype", "admission_manifest", "relative_tolerance",
                    "absolute_tolerance", "relative_residual_limit", "backward_residual_limit"):
            self.assertEqual(result[key], previous[key])
        before = previous["solver_options"]
        after = result["solver_options"]
        self.assertEqual(after, before | {"inner_m": 50, "maxiter": 120})
        self.assertEqual(result["nominal_inner_step_budget"], before["inner_m"] * before["maxiter"])
        self.assertEqual(result["nominal_inner_step_budget"], after["inner_m"] * after["maxiter"])
        expected_reduction = (2 * (before["inner_m"] - after["inner_m"])
                              * result["body_dofs"] * np.dtype("complex128").itemsize / 2**20)
        self.assertAlmostEqual(result["modeled_krylov_workspace_reduction_mib"], expected_reduction)
        prior_peak = result["prior_h80_solver_peak_private_commit_mib"]
        self.assertEqual(prior_peak, previous["run"]["peak_tree_private_commit_mib"])
        budget = _solver_budget(result["admission_manifest"], after, 0, prior_peak)
        self.assertEqual(result["admission"], budget)
        self.assertAlmostEqual(previous["admission"]["required_commit_mib"] - budget["required_commit_mib"],
                               expected_reduction)

        run = result["run"]
        prior_run = previous["run"]
        for key in ("rss_cap_mib", "private_commit_cap_mib", "available_floor_mib", "commit_headroom_floor_mib"):
            self.assertEqual(run[key], prior_run[key])
        self.assertEqual(run["wall_cap_seconds"], floor(1800 - previous["total_pilot_elapsed_seconds"]))
        self.assertAlmostEqual(result["total_pilot_elapsed_seconds"],
                               previous["total_pilot_elapsed_seconds"] + run["elapsed_seconds"])
        self.assertEqual(result["comparison_case_for_sampled_peak_differences"], result["previous_attempt_case"])
        for quantity in ("rss", "private_commit"):
            peak_key = f"peak_tree_{quantity}_mib"
            self.assertAlmostEqual(result[f"sampled_peak_tree_{quantity}_reduction_mib"],
                                   prior_run[peak_key] - run[peak_key])
        self.assertGreaterEqual(run["initial_memory"]["physical_available_mib"], budget["required_physical_mib"])
        self.assertGreaterEqual(run["initial_memory"]["commit_headroom_mib"], budget["required_commit_mib"])
        self.assertEqual(result["status"], "aborted_wall_time_limit")
        self.assertEqual(result["status"], run["status"])
        self.assertGreaterEqual(run["elapsed_seconds"], run["wall_cap_seconds"])
        respects_limits = (run["minimum_available_mib"] >= run["available_floor_mib"]
                           and run["minimum_commit_headroom_mib"] >= run["commit_headroom_floor_mib"]
                           and run["peak_tree_rss_mib"] <= run["rss_cap_mib"]
                           and run["peak_tree_private_commit_mib"] <= run["private_commit_cap_mib"])
        self.assertEqual(result["memory_limits_respected"], respects_limits)
        self.assertTrue(respects_limits)
        self.assertGreater(result["sampled_peak_tree_rss_reduction_mib"], 0)
        self.assertGreater(result["sampled_peak_tree_private_commit_reduction_mib"], 0)

        initial = result["independent_initial_equation_checks"]
        self.assertEqual(initial, previous["retained_checkpoint"]["independent_original_equation_checks"])
        kept = result["retained_checkpoint"]
        self.assertEqual(kept["artifact"], result["raw_files"]["retained_vector"])
        self.assertEqual(kept["column"], 0)
        self.assertEqual(kept["event"], "iterate")
        diagnostics = kept["diagnostics"]
        history = diagnostics["relative_residual_history"]
        self.assertTrue(diagnostics["warm_started"])
        self.assertAlmostEqual(history[0], initial["relative_residual"])
        self.assertEqual(len(history), diagnostics["iterations"])
        observed = kept["independent_original_equation_checks"]
        passed = (observed["relative_residual"] <= result["relative_residual_limit"]
                  and observed["relative_backward_residual"] <= result["backward_residual_limit"])
        self.assertEqual(kept["original_residual_gates_passed"], passed)
        self.assertFalse(passed)
        self.assertEqual(result["accepted_channels"], 0)
        self.assertLess(observed["relative_residual"], initial["relative_residual"])
        self.assertLess(observed["relative_backward_residual"], initial["relative_backward_residual"])
        self.assertEqual(result["accepted_channels"], run["accepted_channels"])
        self.assertFalse(result["scattering_computed"])
        self.assertFalse(result["spatial_convergence_established"])

        table = LEDGER.with_suffix(".csv").read_bytes()
        self.assertEqual(sha256(table).hexdigest(), self.ledger["csv_sha256"])
        row, = [row for row in csv.DictReader(io.StringIO(table.decode("utf-8")))
                if row["case"] == "full_device_h80_lgmres_fill8_inner50_continuation"]
        self.assertEqual((row["phase"], row["status"]), ("first_drive_inner50_continuation", run["status"]))
        self.assertEqual(float(row["max_relative_true_residual"]), observed["relative_residual"])
        self.assertEqual(float(row["max_relative_backward_residual"]), observed["relative_backward_residual"])

    def test_h80_long_continuation_replays_checkpoint_lineage_and_gates(self):
        from phononic.resumable_scattering_AP_pnc import _solver_budget

        key = "full_device_h80_lgmres_fill8_inner50_long_continuation"
        result = self.ledger[key]
        previous = self.ledger[result["previous_attempt_case"]]
        operator = self.ledger[result["reference_operator_case"]]
        for field, raw_name in (("execution", "execution"), ("journal", "channel_journal"),
                                ("profile", "profile"), ("seed", "retained_vector")):
            self.assertEqual(result["resumed_from"][field], previous["raw_files"][raw_name])
        self.assertEqual(result["resumed_from"]["saved_outer_callbacks"],
                         previous["retained_checkpoint"]["diagnostics"]["iterations"])
        self.assertTrue(result["resumed_from"]["preconditioner_rebuild_required"])
        for name in ("A", "rhs", "stages"):
            self.assertEqual(result["raw_files"][name], operator["raw_files"][name])
        for key_name in ("source_sha256", "packages", "python", "platform"):
            self.assertEqual(result["source"][key_name], operator["source"][key_name])
        for name, alias in (("operator_csc_sha256", "original_csc_digest"), ("rhs_array_sha256", "original_rhs_digest"),
                            ("rhs_column_sha256", "original_rhs_column_digest")):
            self.assertEqual(result["equation_array_sha256"][name], previous["resumed_from"][alias])

        options = result["solver_options"]
        self.assertEqual(options, previous["solver_options"] | {"maxiter": 400})
        self.assertEqual((result["equation_dtype"], result["factor_dtype"]), ("complex128", "complex64"))
        self.assertEqual((result["relative_tolerance"], result["absolute_tolerance"]), (1e-10, 0))
        self.assertEqual((result["relative_residual_limit"], result["backward_residual_limit"]), (1e-9, 1e-9))
        budget = _solver_budget(result["admission_manifest"], options, 0,
                                result["prior_h80_solver_peak_private_commit_mib"])
        self.assertEqual(result["admission"], previous["admission"])
        self.assertEqual(result["admission"], budget)
        run = result["run"]
        self.assertEqual(run["wall_cap_seconds"], 7200)
        self.assertLess(run["elapsed_seconds"], run["wall_cap_seconds"])
        self.assertEqual(result["status"], run["status"])
        self.assertEqual(run["status"], "iteration_limit")
        self.assertEqual(run["worker_exit_code"], 1)
        self.assertGreaterEqual(run["initial_memory"]["physical_available_mib"], budget["required_physical_mib"])
        self.assertGreaterEqual(run["initial_memory"]["commit_headroom_mib"], budget["required_commit_mib"])
        self.assertGreaterEqual(result["before_factorization"]["commit_headroom_mib"],
                                result["required_pre_factor_commit_headroom_mib"])
        self.assertTrue(result["memory_limits_respected"])
        for measured, limit, compare in (
                ("peak_tree_rss_mib", "rss_cap_mib", self.assertLessEqual),
                ("peak_tree_private_commit_mib", "private_commit_cap_mib", self.assertLessEqual),
                ("minimum_available_mib", "available_floor_mib", self.assertGreaterEqual),
                ("minimum_commit_headroom_mib", "commit_headroom_floor_mib", self.assertGreaterEqual)):
            self.assertEqual(run[limit], previous["run"][limit])
            compare(run[measured], run[limit])

        kept = result["retained_checkpoint"]
        self.assertEqual(kept["artifact"], result["raw_files"]["retained_vector"])
        self.assertEqual((kept["event"], kept["column"]), ("terminal", 0))
        self.assertEqual(result["incident_columns"], [0])
        self.assertEqual(result["incident_root_index"], previous["incident_root_index"])
        diagnostics = kept["diagnostics"]
        self.assertTrue(diagnostics["warm_started"])
        self.assertEqual(diagnostics["info"], options["maxiter"])
        self.assertEqual(diagnostics["raw_info"], options["maxiter"])
        self.assertIsNone(diagnostics["terminal_confirmation"])
        terminal = kept["independent_original_equation_checks"]
        for field, limit, passed in (("relative_residual", "relative_residual_limit", False),
                                     ("relative_backward_residual", "backward_residual_limit", True)):
            np.testing.assert_allclose(terminal[field], diagnostics[field], rtol=1e-12, atol=0)
            self.assertEqual(terminal[field] <= result[limit], passed)
        self.assertFalse(kept["relative_residual_passed"])
        self.assertTrue(kept["backward_residual_passed"])
        self.assertFalse(kept["original_residual_gates_passed"])
        self.assertEqual((result["accepted_channels"], run["accepted_channels"], result["total_device_channels"]), (0, 0, 5))
        self.assertFalse(result["scattering_computed"])
        self.assertFalse(result["spatial_convergence_established"])

        history = result["relative_residual_history_summary"]
        initial = result["independent_initial_equation_checks"]
        prior_terminal = previous["retained_checkpoint"]["independent_original_equation_checks"]
        np.testing.assert_allclose(initial["relative_residual"], prior_terminal["relative_residual"], rtol=1e-12, atol=0)
        np.testing.assert_allclose(initial["relative_backward_residual"], prior_terminal["relative_backward_residual"], rtol=1e-12, atol=0)
        np.testing.assert_allclose(initial["relative_residual"], history["samples_by_callback"]["1"], rtol=1e-12, atol=0)
        self.assertEqual(history["callback_count"], diagnostics["iterations"])
        self.assertEqual(history["callback_count"], options["maxiter"])
        self.assertAlmostEqual(history["terminal_improvement_factor"], initial["relative_residual"] / terminal["relative_residual"])
        self.assertGreater(history["minimum"], result["relative_residual_limit"])
        self.assertGreater(history["last_25_range"][0], history["minimum"])
        previous_vector = result["preterminal_checkpoint"]
        self.assertEqual(previous_vector["callback"], history["callback_count"])
        self.assertEqual(previous_vector["artifact"], result["raw_files"]["preterminal_vector"])
        np.testing.assert_allclose(previous_vector["independent_original_equation_checks"]["relative_residual"],
                                   history["samples_by_callback"][str(history["callback_count"])], rtol=1e-12, atol=0)

        table = LEDGER.with_suffix(".csv").read_bytes()
        self.assertEqual(sha256(table).hexdigest(), self.ledger["csv_sha256"])
        row, = [row for row in csv.DictReader(io.StringIO(table.decode("utf-8"))) if row["case"] == key]
        self.assertEqual((row["phase"], row["status"]), ("first_drive_long_continuation", run["status"]))
        self.assertEqual(int(row["max_gmres_iterations"]), history["callback_count"])
        self.assertEqual(float(row["max_relative_true_residual"]), terminal["relative_residual"])
        self.assertEqual(float(row["max_relative_backward_residual"]), terminal["relative_backward_residual"])

    def test_h80_residual_correction_replays_original_acceptance_and_memory(self):
        from phononic.resumable_scattering_AP_pnc import _solver_budget

        key = "full_device_h80_gcrotmk_residual_correction"
        result = self.ledger[key]
        prior = self.ledger[result["previous_attempt_case"]]
        operator = self.ledger[result["reference_operator_case"]]
        raw = result["raw_files"]
        self.assertEqual(raw["seed"], prior["retained_checkpoint"]["artifact"])
        for name in ("A", "rhs", "stages"):
            self.assertEqual(raw[name], operator["raw_files"][name])
        for name in ("source_sha256", "packages", "python", "platform"):
            self.assertEqual(result["source"][name], operator["source"][name])
        self.assertEqual(result["equation_array_sha256"], prior["equation_array_sha256"])
        self.assertEqual(result["incident_columns"], [0])
        self.assertEqual(result["incident_root_index"], prior["incident_root_index"])
        self.assertEqual((result["equation_dtype"], result["preconditioner_side"]), ("complex128", "flexible_right"))
        factor = result["factorization"]
        self.assertEqual((factor["dtype"], factor["fill_factor"], factor["drop_tol"], factor["permc_spec"]),
                         ("complex64", 8.0, 1e-4, "COLAMD"))

        options = result["gcrotmk_settings"]
        self.assertEqual((options["m"], options["k"], options["maxiter"], options["atol"]), (25, 5, 8, 0))
        vectors = result["vector_accounting"]
        inner = 2 * (2 * (options["m"] + options["k"]) + 1)
        recycled = 2 * options["k"]
        self.assertEqual(vectors["overlapping_inner_vectors_upper_bound"], inner)
        self.assertEqual(vectors["recycled_vectors_upper_bound"], recycled)
        self.assertEqual(vectors["budgeted_vectors"], inner + recycled + vectors["scratch_vector_slots"])
        self.assertEqual(vectors["scratch_vector_slots"], 22)
        self.assertEqual(vectors["bytes_per_complex128_vector"], 16 * result["body_dofs"])
        budget = _solver_budget(result["admission_manifest"], result["admission_budget_profile"], 0,
                                prior["run"]["peak_tree_private_commit_mib"])
        self.assertEqual(result["admission"], budget)
        self.assertEqual(result["admission"], prior["admission"])
        run = result["run"]
        self.assertEqual((run["status"], run["worker_exit_code"], run["wall_cap_seconds"]), ("iteration_limit", 1, 600))
        self.assertEqual(result["status"], run["status"])
        self.assertLess(run["elapsed_seconds"], run["wall_cap_seconds"])
        self.assertGreaterEqual(run["initial_memory"]["physical_available_mib"], budget["required_physical_mib"])
        self.assertGreaterEqual(run["initial_memory"]["commit_headroom_mib"], budget["required_commit_mib"])
        self.assertGreaterEqual(result["before_factorization"]["commit_headroom_mib"],
                                result["required_pre_factor_commit_headroom_mib"])
        self.assertTrue(result["memory_limits_respected"])
        for observed, limit, check in (
                ("peak_tree_rss_mib", "rss_cap_mib", self.assertLessEqual),
                ("peak_tree_private_commit_mib", "private_commit_cap_mib", self.assertLessEqual),
                ("minimum_available_mib", "available_floor_mib", self.assertGreaterEqual),
                ("minimum_commit_headroom_mib", "commit_headroom_floor_mib", self.assertGreaterEqual)):
            self.assertEqual(run[limit], prior["run"][limit])
            check(run[observed], run[limit])

        target = result["relative_tolerance"]
        relative_gate = result["relative_residual_limit"]
        backward_gate = result["backward_residual_limit"]
        self.assertEqual((target, relative_gate, backward_gate), (1e-10, 1e-9, 1e-9))
        seed = result["seed_original_equation_checks"]
        terminal = result["terminal_candidate"]
        final = terminal["independent_original_equation_checks"]
        self.assertEqual(terminal["artifact"], raw["terminal_vector"])
        self.assertNotEqual(raw["terminal_vector"]["sha256"], raw["final_callback_vector"]["sha256"])
        self.assertNotEqual(raw["terminal_vector"]["sha256"], raw["seed"]["sha256"])
        self.assertTrue(terminal["returned_update_after_last_callback"])
        self.assertIsNone(terminal["terminal_confirmation"])
        for name in ("raw_info", "info", "iterations"):
            self.assertEqual(terminal[name], options["maxiter"])
        b_norm = result["original_rhs_norm_2"]
        for vector, reference in ((seed, prior["retained_checkpoint"]["independent_original_equation_checks"]),
                                  (final, terminal)):
            backward_scale = (result["original_operator_inf_norm"] * vector["solution_norm_inf"]
                              + result["original_rhs_norm_inf"])
            np.testing.assert_allclose(
                [vector["residual_norm_2"] / b_norm, vector["residual_norm_inf"] / backward_scale],
                [reference["relative_residual"], reference["relative_backward_residual"]], rtol=1e-12, atol=0,
            )
        np.testing.assert_allclose(result["normalized_correction_rhs_norm"], 1, rtol=1e-12, atol=0)
        np.testing.assert_allclose(options["rtol"] * seed["relative_residual"], target, rtol=1e-12, atol=0)
        np.testing.assert_allclose(terminal["correction_normalized_relative_residual"] * seed["relative_residual"],
                                   terminal["correction_relative_residual"], rtol=1e-8, atol=0)
        self.assertGreater(terminal["correction_normalized_relative_residual"], options["rtol"])
        np.testing.assert_allclose(terminal["correction_relative_residual"], final["relative_residual"], rtol=1e-8, atol=0)
        self.assertEqual(terminal["solver_relative_target_passed"], final["relative_residual"] <= target)
        self.assertEqual(terminal["relative_residual_gate_passed"], final["relative_residual"] <= relative_gate)
        self.assertEqual(terminal["backward_residual_gate_passed"], final["relative_backward_residual"] <= backward_gate)
        self.assertFalse(terminal["solver_relative_target_passed"])
        self.assertFalse(terminal["relative_residual_gate_passed"])
        self.assertTrue(terminal["backward_residual_gate_passed"])
        history = result["callback_original_relative_residuals"]
        self.assertEqual(len(history), terminal["iterations"])
        np.testing.assert_allclose(history[0], seed["relative_residual"], rtol=1e-12, atol=0)
        self.assertLess(final["relative_residual"], min(history))
        self.assertAlmostEqual(result["original_relative_residual_reduction_fraction"],
                               1 - final["relative_residual"] / seed["relative_residual"])
        self.assertEqual((result["accepted_channels"], run["accepted_channels"], result["total_device_channels"]), (0, 0, 5))
        self.assertFalse(result["scattering_computed"])
        self.assertFalse(result["spatial_convergence_established"])

        normal, last, limited = (result["algebraic_checks"][name] for name in ("normal", "last_update", "limited"))
        for small, raw_info in ((normal, 0), (last, 1)):
            self.assertEqual((small["status"], small["raw_info"], small["info"], small["accepted_count"]),
                             ("first_drive_accepted", raw_info, 0, 1))
            self.assertLessEqual(small["direct_relative_residual"], target)
            self.assertLessEqual(small["relative_dense_solution_error"], target)
        self.assertIsNone(normal["terminal_confirmation"])
        self.assertEqual(last["terminal_confirmation"], {"callbacks": 1, "unchanged_start": True,
                                                        "accepted": True, "info": 0})
        self.assertEqual((limited["status"], limited["raw_info"], limited["info"], limited["accepted_count"]),
                         ("iteration_limit", 1, 1, 0))
        self.assertGreater(limited["direct_relative_residual"], target)
        self.assertIsNone(limited["terminal_confirmation"])

        table = LEDGER.with_suffix(".csv").read_bytes()
        self.assertEqual(sha256(table).hexdigest(), self.ledger["csv_sha256"])
        row, = [row for row in csv.DictReader(io.StringIO(table.decode("utf-8"))) if row["case"] == key]
        self.assertEqual((row["phase"], row["status"]), ("first_drive_residual_correction", run["status"]))
        self.assertEqual(int(row["max_gmres_iterations"]), terminal["iterations"])
        self.assertEqual(float(row["max_relative_true_residual"]), final["relative_residual"])
        self.assertEqual(float(row["max_relative_backward_residual"]), final["relative_backward_residual"])

    def test_h80_fill9_correction_replays_admission_lineage_and_original_gates(self):
        key = "full_device_h80_gcrotmk_fill9_residual_correction"
        result = self.ledger[key]
        prior = self.ledger[result["previous_attempt_case"]]
        operator = self.ledger[result["reference_operator_case"]]
        raw = result["raw_files"]
        self.assertEqual(raw["seed"], prior["raw_files"]["terminal_vector"])
        for name in ("A", "rhs", "stages"):
            self.assertEqual(raw[name], operator["raw_files"][name])
        self.assertEqual(raw["prior_driver"], prior["raw_files"]["driver"])
        self.assertEqual(raw["algebraic_check"], prior["raw_files"]["algebraic_check"])
        self.assertEqual(result["algebraic_check_functions_unchanged"], ["original_gates_pass", "solve_correction"])
        self.assertEqual(result["equation_array_sha256"], prior["equation_array_sha256"])
        self.assertEqual(result["numerical_thread_environment"], prior["numerical_thread_environment"])
        self.assertEqual(result["incident_columns"], [0])
        self.assertEqual(result["incident_root_index"], prior["incident_root_index"])
        self.assertEqual((result["equation_dtype"], result["preconditioner_side"]), ("complex128", "flexible_right"))

        staged, source = operator["source"], result["source"]
        coordinator = "src/phononic/resumable_scattering_AP_pnc.py"
        self.assertEqual(result["source_compatibility_exception"], {
            "path": coordinator,
            "staged_sha256": "68c12549b47a9ec98abf7b39d78ad597dd114da00e326adf8ebbec060947b53f",
            "current_sha256": "4dc59e08d0cbadcfe2bca554e1be08b01a9dd56c4e660f995189171527f45a2b",
            "staged_git_commit": staged["git_commit"], "current_git_commit": source["git_commit"],
            "review": "PR #21 worker shutdown only",
        })
        exception = result["source_compatibility_exception"]
        self.assertEqual(staged["source_sha256"][coordinator], exception["staged_sha256"])
        self.assertEqual(source["source_sha256"], staged["source_sha256"] | {coordinator: exception["current_sha256"]})
        for name in ("python", "platform", "packages"):
            self.assertEqual(source[name], staged[name])

        factor, options = result["factorization"], result["gcrotmk_settings"]
        self.assertEqual((factor["dtype"], factor["fill_factor"], factor["drop_tol"], factor["permc_spec"]),
                         ("complex64", 9.0, 1e-4, "COLAMD"))
        self.assertEqual((options["m"], options["k"], options["maxiter"], options["atol"]), (15, 3, 13, 0))
        inner = 2 * (2 * (options["m"] + options["k"]) + 1)
        recycled = 2 * options["k"]
        vector_count = inner + recycled + 22
        dofs, nnz = result["body_dofs"], result["operator_nnz"]
        vector_bytes = 16 * dofs
        self.assertEqual(vector_count, 102)
        self.assertEqual(result["vector_accounting"], {
            "budgeted_vectors": vector_count, "overlapping_inner_vectors_upper_bound": inner,
            "recycled_vectors_upper_bound": recycled, "scratch_vector_slots": 22,
            "bytes_per_complex128_vector": vector_bytes,
        })
        memory_bytes = {
            "operator_storage_bytes": 20 * nnz + 4 * (dofs + 1),
            "rhs_workspace_bytes": 3 * vector_bytes,
            "factor_input_bytes": 12 * nnz + 4 * (dofs + 1),
            "factor_buffer_bytes": 24 * 9 * nnz,
            "krylov_workspace_bytes": vector_count * vector_bytes,
        }
        run = result["run"]
        earlier = self.ledger[prior["previous_attempt_case"]]
        prior_peak = max(prior["run"]["peak_tree_private_commit_mib"], earlier["run"]["peak_tree_private_commit_mib"])
        prior_floor = prior_peak + run["commit_headroom_floor_mib"]
        modeled = sum(memory_bytes.values()) / 2**20 + 1024
        required_commit = max(modeled, prior_floor)
        loaded = (memory_bytes["operator_storage_bytes"] + vector_bytes) / 2**20
        self.assertEqual(result["admission_estimate"], memory_bytes | {
            "large_factor_margin_mib": 1024, "prior_peak_with_reserve_mib": prior_floor,
            "modeled_commit_mib": modeled, "estimated_loaded_equation_mib": loaded,
            "required_pre_factor_commit_headroom_mib": required_commit - loaded,
        })
        self.assertEqual(result["admission"], {"required_commit_mib": required_commit, "required_physical_mib": 1200})
        self.assertEqual(result["required_pre_factor_commit_headroom_mib"], required_commit - loaded)
        self.assertGreaterEqual(result["before_factorization"]["commit_headroom_mib"], required_commit - loaded)
        self.assertGreaterEqual(result["before_factorization"]["physical_available_mib"], run["available_floor_mib"])
        self.assertGreaterEqual(run["initial_memory"]["commit_headroom_mib"], required_commit)
        self.assertGreaterEqual(run["initial_memory"]["physical_available_mib"], 1200)
        self.assertGreaterEqual(run["initial_disk_free_mib"], run["required_disk_free_mib"])
        self.assertEqual((run["status"], run["worker_exit_code"], run["wall_cap_seconds"]), ("iteration_limit", 1, 600))
        self.assertEqual(run["status"], result["status"])
        self.assertTrue(run["worker_stopped"])
        self.assertLess(run["elapsed_seconds"], run["wall_cap_seconds"])
        for observed, limit, check in (
                ("peak_tree_rss_mib", "rss_cap_mib", self.assertLessEqual),
                ("peak_tree_private_commit_mib", "private_commit_cap_mib", self.assertLessEqual),
                ("minimum_available_mib", "available_floor_mib", self.assertGreaterEqual),
                ("minimum_commit_headroom_mib", "commit_headroom_floor_mib", self.assertGreaterEqual)):
            self.assertEqual(run[limit], prior["run"][limit])
            check(run[observed], run[limit])

        target, relative_gate, backward_gate = (result[name] for name in (
            "relative_tolerance", "relative_residual_limit", "backward_residual_limit"))
        self.assertEqual((target, relative_gate, backward_gate), (1e-10, 1e-9, 1e-9))
        seed = result["seed_original_equation_checks"]
        terminal = result["terminal_candidate"]
        final = terminal["independent_original_equation_checks"]
        self.assertEqual(terminal["artifact"], raw["terminal_vector"])
        self.assertEqual(raw["terminal_vector"]["sha256"], raw["best_vector"]["sha256"])
        self.assertNotEqual(raw["terminal_vector"]["sha256"], raw["final_callback_vector"]["sha256"])
        self.assertTrue(terminal["returned_update_after_last_callback"])
        self.assertIsNone(terminal["terminal_confirmation"])
        self.assertEqual([terminal[name] for name in ("raw_info", "info", "iterations")], [options["maxiter"]] * 3)
        for candidate, reference in ((seed, prior["terminal_candidate"]["independent_original_equation_checks"]),
                                     (final, terminal)):
            backward_scale = (result["original_operator_inf_norm"] * candidate["solution_norm_inf"]
                              + result["original_rhs_norm_inf"])
            relative = candidate["residual_norm_2"] / result["original_rhs_norm_2"]
            backward = candidate["residual_norm_inf"] / backward_scale
            np.testing.assert_allclose([relative, backward],
                                       [reference["relative_residual"], reference["relative_backward_residual"]],
                                       rtol=1e-12, atol=0)
        np.testing.assert_allclose(result["normalized_correction_rhs_norm"], 1, rtol=1e-12, atol=0)
        np.testing.assert_allclose(options["rtol"] * seed["relative_residual"], target, rtol=1e-12, atol=0)
        np.testing.assert_allclose(terminal["correction_normalized_relative_residual"] * seed["relative_residual"],
                                   terminal["correction_relative_residual"], rtol=5e-8, atol=0)
        np.testing.assert_allclose(terminal["correction_relative_residual"], final["relative_residual"], rtol=5e-8, atol=0)
        self.assertEqual(terminal["solver_relative_target_passed"], final["relative_residual"] <= target)
        self.assertEqual(terminal["relative_residual_gate_passed"], final["relative_residual"] <= relative_gate)
        self.assertEqual(terminal["backward_residual_gate_passed"], final["relative_backward_residual"] <= backward_gate)
        self.assertFalse(terminal["solver_relative_target_passed"])
        self.assertFalse(terminal["relative_residual_gate_passed"])
        self.assertTrue(terminal["backward_residual_gate_passed"])
        history = result["callback_original_relative_residuals"]
        self.assertEqual(len(history), terminal["iterations"])
        np.testing.assert_allclose(history[0], seed["relative_residual"], rtol=1e-12, atol=0)
        self.assertLess(final["relative_residual"], min(history))
        self.assertAlmostEqual(result["original_relative_residual_reduction_fraction"],
                               1 - final["relative_residual"] / seed["relative_residual"])
        self.assertEqual((result["accepted_channels"], run["accepted_channels"], result["total_device_channels"]), (0, 0, 5))
        self.assertFalse(result["scattering_computed"])
        self.assertFalse(result["spatial_convergence_established"])

        table = LEDGER.with_suffix(".csv").read_bytes()
        self.assertEqual(sha256(table).hexdigest(), self.ledger["csv_sha256"])
        row, = [row for row in csv.DictReader(io.StringIO(table.decode("utf-8"))) if row["case"] == key]
        self.assertEqual((row["phase"], row["status"]), ("first_drive_residual_correction", run["status"]))
        self.assertEqual(int(row["max_gmres_iterations"]), terminal["iterations"])
        self.assertEqual(float(row["max_relative_true_residual"]), final["relative_residual"])
        self.assertEqual(float(row["max_relative_backward_residual"]), final["relative_backward_residual"])

    def test_h80_double_fill4_correction_preserves_equations_admission_and_gates(self):
        key = "full_device_h80_gcrotmk_double_fill4_residual_correction"
        result = self.ledger[key]
        prior = self.ledger["full_device_h80_gcrotmk_fill9_residual_correction"]
        operator = self.ledger["full_device_h80_operator"]
        self.assertEqual(result["previous_attempt_case"], "full_device_h80_gcrotmk_fill9_residual_correction")
        self.assertEqual(result["reference_operator_case"], "full_device_h80_operator")
        raw = result["raw_files"]
        for name in ("A", "rhs", "stages"):
            self.assertEqual(raw[name], operator["raw_files"][name])
        self.assertEqual(raw["seed"], prior["raw_files"]["terminal_vector"])
        self.assertEqual(raw["prior_driver"], prior["raw_files"]["driver"])
        self.assertEqual(raw["algebraic_check"], prior["raw_files"]["algebraic_check"])
        self.assertEqual(result["equation_array_sha256"], prior["equation_array_sha256"])
        self.assertEqual(result["source"]["source_sha256"], prior["source"]["source_sha256"])
        exception = result["source_compatibility_exception"]
        self.assertEqual(exception, prior["source_compatibility_exception"] | {"current_git_commit": result["source"]["git_commit"]})
        self.assertEqual((result["frequency_hz"], result["incident_side"], result["incident_columns"]), (1e9, "left", [0]))
        self.assertEqual(result["incident_root_index"], prior["incident_root_index"])

        factor, options = result["factorization"], result["gcrotmk_settings"]
        self.assertTrue(result["factorization_attempted"])
        self.assertEqual((factor["dtype"], factor["fill_factor"], factor["drop_tol"], factor["permc_spec"]),
                         ("complex128", 4.0, 1e-4, "COLAMD"))
        self.assertEqual((result["equation_dtype"], result["preconditioner_side"]), ("complex128", "flexible_right"))
        self.assertEqual((options["m"], options["k"], options["maxiter"], options["atol"]), (25, 5, 8, 0))
        target, relative_gate, backward_gate = (result[name] for name in (
            "relative_tolerance", "relative_residual_limit", "backward_residual_limit"))
        self.assertEqual((target, relative_gate, backward_gate), (1e-10, 1e-9, 1e-9))
        small = result["small_precision_check"]
        self.assertEqual((small["status"], small["shape"], small["correction_outcome"]), ("passed", [4, 4], "first_drive_accepted"))
        self.assertEqual(small["native_application_dtype"], small["correction_dtype"])
        self.assertEqual(small["native_application_dtype"], factor["dtype"])
        self.assertLessEqual(small["native_application_relative_dense_error"], 1e-14)
        self.assertLessEqual(small["relative_dense_solution_error"], target)
        self.assertLessEqual(small["original_relative_residual"], target)
        self.assertLessEqual(small["original_backward_residual"], backward_gate)
        self.assertEqual(small["original_csc_sha256_before"], small["original_csc_sha256_after"])

        inner = 2 * (2 * (options["m"] + options["k"]) + 1)
        recycled = 2 * options["k"]
        vector_count = inner + recycled + 22
        dofs, nnz = result["body_dofs"], result["operator_nnz"]
        vector_bytes = 16 * dofs
        self.assertEqual(result["vector_accounting"], {
            "budgeted_vectors": vector_count, "overlapping_inner_vectors_upper_bound": inner,
            "recycled_vectors_upper_bound": recycled, "scratch_vector_slots": 22,
            "bytes_per_complex128_vector": vector_bytes,
        })
        csc_bytes = 20 * nnz + 4 * (dofs + 1)
        memory_bytes = {"operator_storage_bytes": csc_bytes, "factor_input_bytes": csc_bytes,
                        "factor_buffer_bytes": int(2 * 20 * factor["fill_factor"] * nnz),
                        "rhs_workspace_bytes": 3 * vector_bytes, "krylov_workspace_bytes": vector_count * vector_bytes}
        run = result["run"]
        modeled = sum(memory_bytes.values()) / 2**20 + 1024
        prior_floor = prior["run"]["peak_tree_private_commit_mib"] + run["commit_headroom_floor_mib"]
        required_commit = max(modeled, prior_floor)
        loaded = (csc_bytes + vector_bytes) / 2**20
        self.assertEqual(result["admission_estimate"], memory_bytes | {
            "large_factor_margin_mib": 1024, "prior_peak_with_reserve_mib": prior_floor,
            "modeled_commit_mib": modeled, "estimated_loaded_equation_mib": loaded,
            "required_pre_factor_commit_headroom_mib": required_commit - loaded,
        })
        self.assertEqual(result["admission"], {"required_commit_mib": required_commit, "required_physical_mib": 1200})
        self.assertEqual(result["required_pre_factor_commit_headroom_mib"], required_commit - loaded)
        self.assertGreaterEqual(run["initial_memory"]["commit_headroom_mib"], required_commit)
        self.assertGreaterEqual(run["initial_memory"]["physical_available_mib"], 1200)
        self.assertGreaterEqual(result["before_factorization"]["commit_headroom_mib"], required_commit - loaded)
        self.assertGreaterEqual(result["before_factorization"]["physical_available_mib"], run["available_floor_mib"])
        self.assertEqual((run["status"], run["worker_exit_code"], run["wall_cap_seconds"]), ("iteration_limit", 1, 600))
        self.assertEqual(result["status"], run["status"])
        self.assertTrue(run["worker_stopped"])
        self.assertLess(run["elapsed_seconds"], run["wall_cap_seconds"])
        for observed, limit, check in (
                ("peak_tree_rss_mib", "rss_cap_mib", self.assertLessEqual),
                ("peak_tree_private_commit_mib", "private_commit_cap_mib", self.assertLessEqual),
                ("minimum_available_mib", "available_floor_mib", self.assertGreaterEqual),
                ("minimum_commit_headroom_mib", "commit_headroom_floor_mib", self.assertGreaterEqual)):
            self.assertEqual(run[limit], prior["run"][limit])
            check(run[observed], run[limit])

        seed, terminal = result["seed_original_equation_checks"], result["terminal_candidate"]
        best, callback = result["best_candidate"], result["final_callback_candidate"]
        self.assertEqual(seed, prior["terminal_candidate"]["independent_original_equation_checks"])
        self.assertEqual(terminal["artifact"]["sha256"], best["artifact"]["sha256"])
        self.assertNotEqual(terminal["artifact"]["sha256"], callback["artifact"]["sha256"])
        self.assertTrue(terminal["returned_update_after_last_callback"])
        self.assertEqual([terminal[name] for name in ("raw_info", "info", "iterations")], [options["maxiter"]] * 3)
        for candidate in (terminal, best, callback):
            measured = candidate["independent_original_equation_checks"]
            relative = measured["residual_norm_2"] / result["original_rhs_norm_2"]
            scale = result["original_operator_inf_norm"] * measured["solution_norm_inf"] + result["original_rhs_norm_inf"]
            backward = measured["residual_norm_inf"] / scale
            np.testing.assert_allclose([relative, backward], [candidate["relative_residual"], candidate["relative_backward_residual"]],
                                       rtol=1e-12, atol=0)
            self.assertEqual(candidate["solver_relative_target_passed"], relative <= target)
            self.assertEqual(candidate["relative_residual_gate_passed"], relative <= relative_gate)
            self.assertEqual(candidate["backward_residual_gate_passed"], backward <= backward_gate)
        final = terminal["independent_original_equation_checks"]
        self.assertFalse(terminal["relative_residual_gate_passed"])
        self.assertTrue(terminal["backward_residual_gate_passed"])
        history = result["callback_original_relative_residuals"]
        self.assertEqual(len(history), terminal["iterations"])
        np.testing.assert_allclose([history[0], history[-1], options["rtol"] * seed["relative_residual"]],
                                   [seed["relative_residual"], callback["relative_residual"], target], rtol=1e-12, atol=0)
        self.assertLess(final["relative_residual"], min(history))
        self.assertAlmostEqual(result["original_relative_residual_reduction_fraction"],
                               1 - final["relative_residual"] / seed["relative_residual"])
        self.assertEqual((result["accepted_channels"], run["accepted_channels"], result["total_device_channels"]), (0, 0, 5))
        self.assertFalse(result["scattering_computed"])
        self.assertFalse(result["spatial_convergence_established"])

        table = LEDGER.with_suffix(".csv").read_bytes()
        self.assertEqual(sha256(table).hexdigest(), self.ledger["csv_sha256"])
        row, = [row for row in csv.DictReader(io.StringIO(table.decode("utf-8"))) if row["case"] == key]
        self.assertEqual((row["phase"], row["status"]), ("first_drive_residual_correction", run["status"]))
        self.assertEqual(int(row["factor_nnz"]), factor["factor_nonzeros"])
        self.assertEqual(int(row["max_gmres_iterations"]), terminal["iterations"])
        self.assertEqual(float(row["max_relative_true_residual"]), final["relative_residual"])
        self.assertEqual(float(row["max_relative_backward_residual"]), final["relative_backward_residual"])

    def test_h100_recovery_reuses_the_failed_first_drive(self):
        operator = self.ledger["full_device_h100_operator"]
        first_attempt = self.ledger["full_device_h100_lgmres_mixed_fill12"]
        recovery = self.ledger["full_device_h100_lgmres_recovery_fill12"]
        first, = first_attempt["channels"]
        self.assertEqual(first_attempt["status"], "failed")
        self.assertEqual((first["column"], first["status"]), (0, "failed"))
        self.assertEqual(first["raw_info"], first_attempt["lgmres_settings"]["maxiter"])
        self.assertGreater(first["relative_true_residual"], first_attempt["relative_residual_limit"])
        self.assertGreater(first["relative_backward_residual"], first_attempt["backward_residual_limit"])
        self.assertNotIn("solution", first_attempt["raw_files"])
        self.assertFalse(first_attempt["scattering_computed"])
        self.assertEqual(first["latest_checkpoint"]["stage"], "solver_return")
        self.assertEqual(recovery["admission"]["stage"], operator["raw_files"]["stages"])
        self.assertEqual(recovery["admission"]["continuation"], {
            "replay": first_attempt["raw_files"]["replay"], "execution": first_attempt["raw_files"]["execution"],
            "checkpoint": first["latest_checkpoint"], "original_relative_residual": first["relative_true_residual"],
        })
        self.assertEqual(recovery["admission"]["lgmres_settings"], first_attempt["lgmres_settings"] | {
            "inner_m": 100, "outer_k": 10, "maxiter": 60,
        })
        self.assertEqual(recovery["admission"]["ilu_settings"], first_attempt["ilu_settings"])
        start = recovery["initial_iterate"]
        self.assertEqual(start["checkpoint"], first["latest_checkpoint"])
        np.testing.assert_allclose(start["recomputed_relative_residual"], first["relative_true_residual"], rtol=1e-12, atol=0)
        self.assertNotEqual(recovery["settings_sha256"], first_attempt["settings_sha256"])
        for field in ("operator_csc_sha256", "rhs_array_sha256", "relative_residual_limit", "backward_residual_limit"):
            self.assertEqual(recovery[field], first_attempt[field])
        for row in recovery["channels"]:
            self.assertEqual(row["initial_iterate_sha256"], None if row["column"] else start["displacement_sha256"])
        self.assertEqual(recovery["run"]["status"], recovery["status"])
        self.assertEqual(recovery["status"], "aborted_wall_time_limit")
        self.assertEqual(recovery["worker_status"], "running")
        self.assertGreaterEqual(recovery["run"]["elapsed_seconds"], recovery["admission"]["wall_cap_seconds"])
        self.assertNotIn("solution", recovery["raw_files"])
        self.assertFalse(recovery["scattering_computed"])
        self.assertEqual([(row["column"], row["status"]) for row in recovery["channels"]], [(0, "accepted"), (1, "running")])
        accepted, active = recovery["channels"]
        self.assertEqual(accepted["info"], 0)
        self.assertLessEqual(accepted["relative_true_residual"], recovery["relative_residual_limit"])
        self.assertLessEqual(accepted["relative_backward_residual"], recovery["backward_residual_limit"])
        self.assertNotIn("solution", active)
        self.assertIn("last_callback_relative_true_residual", active)

    def test_h100_continuation_reuses_accepted_and_active_checkpoints(self):
        recovery = self.ledger["full_device_h100_lgmres_recovery_fill12"]
        continuation = self.ledger["full_device_h100_lgmres_continuation_fill12"]
        accepted, active = recovery["channels"]
        plan = continuation["admission"]["continuation"]
        self.assertEqual(plan["replay"], recovery["raw_files"]["replay"])
        self.assertEqual(plan["execution"], recovery["raw_files"]["execution"])
        self.assertEqual(plan["prior_budget"], recovery["raw_files"]["budget"])
        self.assertEqual(plan["accepted_columns"], [accepted])
        self.assertEqual(plan["active_checkpoint"], active.get("checkpoint_after_last_replay_update", active["latest_checkpoint"]))
        self.assertEqual(continuation["admission"]["lgmres_settings"], recovery["lgmres_settings"])
        self.assertEqual(continuation["admission"]["ilu_settings"], recovery["ilu_settings"])
        self.assertEqual(continuation["admission"]["stage"], recovery["admission"]["stage"])
        start = continuation["initial_iterate"]
        self.assertEqual(start["checkpoint"], plan["active_checkpoint"])
        np.testing.assert_allclose(start["recomputed_relative_residual"], plan["active_checkpoint"]["relative_residual"], rtol=1e-10, atol=0)
        retained = continuation["channels"][0]
        self.assertTrue(retained["reused_solution"])
        self.assertEqual(retained["solution"], accepted["solution"])
        self.assertEqual(retained["latest_checkpoint"], accepted["latest_checkpoint"])
        self.assertLessEqual(retained["rechecked_relative_true_residual"], continuation["relative_residual_limit"])
        self.assertLessEqual(retained["rechecked_relative_backward_residual"], continuation["backward_residual_limit"])
        for row in continuation["channels"][1:]:
            self.assertEqual(row["initial_iterate_sha256"], start["displacement_sha256"] if row["column"] == 1 else None)
        for field in ("operator_csc_sha256", "rhs_array_sha256", "relative_residual_limit", "backward_residual_limit"):
            self.assertEqual(continuation[field], recovery[field])
        self.assertEqual(continuation["run"]["status"], continuation["status"])
        self.assertEqual(continuation["status"], "solved")
        self.assertEqual([row["column"] for row in continuation["channels"]], list(range(5)))
        public = self.ledger["full_device_h100_lgmres_public_replay"]
        self.assertEqual(public["raw_files"]["solution"], continuation["raw_files"]["solution"])
        self.assertEqual(public["raw_files"]["raw_solver"], continuation["raw_files"]["replay"])
        checks = public["physical_checks"]
        for replay_name, channel_name, limit in (
                ("replay_relative_residuals", "relative_true_residual", "relative_residual_limit"),
                ("replay_backward_residuals", "relative_backward_residual", "backward_residual_limit")):
            for replayed, row in zip(checks[replay_name], continuation["channels"], strict=True):
                self.assertEqual(row["status"], "accepted")
                self.assertEqual(row["info"], 0)
                self.assertLessEqual(row[channel_name], continuation[limit])
                self.assertLessEqual(replayed, continuation[limit])
                np.testing.assert_allclose(replayed, row[channel_name], atol=0, rtol=1e-10)
        table = LEDGER.with_suffix(".csv").read_bytes()
        self.assertEqual(sha256(table).hexdigest(), self.ledger["csv_sha256"])
        totals = {row["case"]: row for row in csv.DictReader(io.StringIO(table.decode("utf-8")))
                  if row["phase"] == "solve_replay_total"}
        for key in ("full_device_h100_lgmres_mixed_fill12", "full_device_h100_lgmres_recovery_fill12",
                    "full_device_h100_lgmres_continuation_fill12"):
            attempt = self.ledger[key]
            self.assertEqual(totals[key]["status"], attempt["status"])
            self.assertEqual(int(totals[key]["body_dofs"]), attempt["body_dofs"])

    def test_h80_saved_operator_diagnostic_replays_residual_scaling_and_scope(self):
        key = "full_device_h80_residual_scaling_diagnostic"
        result = self.ledger[key]
        prior = self.ledger[result["previous_attempt_case"]]
        stage = self.ledger[result["reference_operator_case"]]
        self.assertEqual(result["previous_attempt_case"], "full_device_h80_gcrotmk_double_fill4_residual_correction")
        self.assertEqual(result["reference_operator_case"], "full_device_h80_operator")
        self.assertEqual(result["operator"], stage["operator"])
        raw = result["raw_files"]
        self.assertEqual(raw["seed"], prior["raw_files"]["terminal_vector"])
        for name in ("A", "rhs", "stages", "port_modes"):
            self.assertEqual(raw[name], stage["raw_files"][name])
        self.assertEqual(result["equation_array_sha256"], prior["equation_array_sha256"])
        source = result["source"]
        self.assertEqual({k: v for k, v in source.items() if k != "git_commit"},
                         {k: v for k, v in prior["source"].items() if k != "git_commit"})
        self.assertEqual(result["source_compatibility_exception"],
                         prior["source_compatibility_exception"] | {"current_git_commit": source["git_commit"]})

        original = result["measurement"]["original"]
        independent = prior["terminal_candidate"]["independent_original_equation_checks"]
        np.testing.assert_allclose(
            [original["residual_l2"], original["residual_linf"], original["solution_linf"],
             original["operator_linf"], original["rhs_l2"], original["rhs_linf"]],
            [independent["residual_norm_2"], independent["residual_norm_inf"], independent["solution_norm_inf"],
             prior["original_operator_inf_norm"], prior["original_rhs_norm_2"], prior["original_rhs_norm_inf"]],
            rtol=1e-12, atol=0,
        )
        scale = original["operator_linf"] * original["solution_linf"] + original["rhs_linf"]
        relative = original["residual_l2"] / original["rhs_l2"]
        backward = original["residual_linf"] / scale
        np.testing.assert_allclose(
            [scale, relative, backward],
            [original["backward_denominator"], original["relative_residual"], original["relative_backward_residual"]],
            rtol=1e-12, atol=0,
        )
        tolerance = result["relative_tolerance"]
        relative_gate = result["relative_residual_limit"]
        backward_gate = result["backward_residual_limit"]
        self.assertEqual((tolerance, relative_gate, backward_gate), (1e-10, 1e-9, 1e-9))
        self.assertEqual(result["original_gate_results"], {
            "solver_relative_target_met": relative <= tolerance,
            "relative_residual_limit_met": relative <= relative_gate,
            "backward_residual_limit_met": backward <= backward_gate,
        })
        self.assertGreater(relative, relative_gate)
        self.assertLess(backward, backward_gate)
        self.assertTrue(result["original_arrays_unchanged"])
        self.assertEqual((result["accepted_channels"], result["total_device_channels"]), (0, 5))
        self.assertEqual((result["frequency_hz"], result["incident_side"], result["incident_columns"]), (1e9, "left", [0]))
        self.assertEqual(result["incident_root_index"], prior["incident_root_index"])
        self.assertFalse(any(result[name] for name in (
            "factorization_attempted", "solve_attempted", "scattering_computed", "spatial_convergence_established")))

        measurement = result["measurement"]
        groups = measurement["partition"]
        n = result["operator"]["shape"][0]
        trace_dofs = stage["port_reuse_proof"]["trace_mapping"]["xleft"]["trace_dofs"]
        self.assertEqual({name: item["row_count"] for name, item in groups.items()},
                         {"left_trace": trace_dofs, "right_trace": trace_dofs, "other_dofs": n - 2 * trace_dofs})
        self.assertEqual({name: item["rhs_nonzero_rows"] for name, item in groups.items()},
                         {"left_trace": trace_dofs, "right_trace": 0, "other_dofs": 0})
        for item in groups.values():
            relative_l2 = item["relative_residual_l2"]
            self.assertAlmostEqual(item["population_fraction"], item["row_count"] / n, places=15)
            np.testing.assert_allclose(item["residual_energy_fraction"], (relative_l2 / relative) ** 2,
                                       rtol=1e-12, atol=0)
            np.testing.assert_allclose(item["residual_rms"],
                                       relative_l2 * original["rhs_l2"] / np.sqrt(item["row_count"]), rtol=1e-12, atol=0)
        np.testing.assert_allclose(sum(item["residual_energy_fraction"] for item in groups.values()), 1,
                                   rtol=1e-12, atol=0)
        concentration = measurement["residual_concentration"]
        self.assertEqual([entry["row_count"] for entry in concentration],
                         [ceil(n * item) for item in (0.001, 0.01, 0.05, 0.1)])
        energies = [entry["residual_energy_fraction"] for entry in concentration]
        self.assertEqual(energies, sorted(energies))
        self.assertLess(energies[-1], 1)
        for distribution in measurement["row_column_distributions"].values():
            quantiles = [distribution[name] for name in ("min", "p01", "median", "p99", "max")]
            self.assertEqual(quantiles, sorted(quantiles))
            self.assertEqual(distribution["zero_count"], 0)
            self.assertAlmostEqual(distribution["max_over_positive_min"], quantiles[-1] / quantiles[0], places=11)

        arithmetic = measurement["arithmetic"]
        u = arithmetic["binary64_unit_roundoff"]
        self.assertEqual(u, np.finfo(np.float64).eps / 2)
        proxy = arithmetic["arithmetic_proxy_l2_over_rhs_l2"]
        np.testing.assert_allclose(
            [proxy, proxy / tolerance, proxy / relative_gate, proxy / relative],
            [np.linalg.norm([item["arithmetic_proxy_l2_over_rhs_l2"] for item in groups.values()]),
             arithmetic["arithmetic_proxy_over_solver_target"], arithmetic["arithmetic_proxy_over_original_relative_gate"],
             arithmetic["arithmetic_proxy_over_measured_relative_residual"]],
            rtol=1e-12, atol=0,
        )
        selected = measurement["selected_rows"]
        self.assertEqual(len(selected), 16)
        self.assertEqual(len({row["row_zero_based"] for row in selected}), len(selected))
        selected_norms = []
        for row in selected:
            with self.subTest(row=row["row_zero_based"]):
                self.assertIn(row["row_zero_based"], range(n))
                self.assertIn(row["group"], groups)
                r = complex(row["ordinary_residual_real"], row["ordinary_residual_imag"])
                exact = complex(row["exact_residual_real_rounded_to_float"], row["exact_residual_imag_rounded_to_float"])
                gamma_input = 8 * (row["stored_entries"] + 1) * u
                np.testing.assert_allclose(
                    [abs(r), abs(r - exact), row["cancellation_activity"] * gamma_input / (1 - gamma_input)],
                    [row["abs_original_residual"], row["abs_ordinary_minus_exact_residual"], row["arithmetic_proxy"]],
                    rtol=1e-12, atol=4e-29,
                )
                selected_norms.append(abs(r))
        np.testing.assert_allclose(selected_norms, sorted(selected_norms, reverse=True), rtol=1e-15, atol=0)
        self.assertAlmostEqual(selected_norms[0] / original["residual_linf"], 1, places=13)
        self.assertAlmostEqual(measurement["selected_residual_energy_fraction"],
                               (np.linalg.norm(selected_norms) / original["residual_l2"]) ** 2, places=14)
        np.testing.assert_allclose(measurement["selected_exact_error_l2_over_rhs_l2"],
                                   np.linalg.norm([row["abs_ordinary_minus_exact_residual"] for row in selected])
                                   / original["rhs_l2"], rtol=1e-12, atol=0)

        run = result["run"]
        self.assertEqual(result["status"], "diagnostic_completed")
        self.assertEqual((run["status"], run["worker_exit_code"]), ("completed", 0))
        self.assertTrue(run["worker_stopped"])
        self.assertEqual(run["wall_cap_seconds"], 300)
        self.assertLess(run["elapsed_seconds"], run["wall_cap_seconds"])
        for observed, limit, check in (
                ("peak_tree_rss_mib", "rss_cap_mib", self.assertLessEqual),
                ("peak_tree_private_commit_mib", "private_commit_cap_mib", self.assertLessEqual),
                ("minimum_available_mib", "available_floor_mib", self.assertGreaterEqual),
                ("minimum_commit_headroom_mib", "commit_headroom_floor_mib", self.assertGreaterEqual)):
            self.assertEqual(run[limit], prior["run"][limit])
            check(run[observed], run[limit])
        model = sum(result["admission_storage_bytes"].values()) / 2**20 + result["runtime_import_margin_mib"]
        self.assertEqual(result["modeled_worker_mib"], model)
        for required, floor, measured in (
                ("required_physical_mib", "available_floor_mib", "physical_available_mib"),
                ("required_commit_mib", "commit_headroom_floor_mib", "commit_headroom_mib")):
            self.assertEqual(result["admission"][required], model + run[floor])
            self.assertGreaterEqual(run["initial_memory"][measured], result["admission"][required])
            self.assertGreaterEqual(result["before_operator_load_memory"][measured], result["admission"][required])
        dense = result["dense_check"]
        self.assertEqual(dense["status"], "passed")
        self.assertEqual(dense["known_exact_product_cancellation_residual"], 2 ** -54)
        self.assertTrue(dense["partition_check_passed"])
        self.assertTrue(dense["selected_row_check_passed"])
        replay = result["independent_selected_row_check"]
        self.assertEqual((replay["status"], replay["row_count"]), ("passed", len(selected)))
        self.assertEqual(replay["rounded_exact_residual_max_abs_difference"], 0)
        self.assertEqual(replay["ordinary_residual_max_abs_difference"], 0)

        table = LEDGER.with_suffix(".csv").read_bytes()
        self.assertEqual(sha256(table).hexdigest(), self.ledger["csv_sha256"])
        row, = [row for row in csv.DictReader(io.StringIO(table.decode("utf-8"))) if row["case"] == key]
        self.assertEqual((row["phase"], row["status"]), ("saved_operator_residual_scaling", result["status"]))
        self.assertEqual(int(row["body_dofs"]), n)
        self.assertEqual(int(row["operator_nnz"]), result["operator"]["nonzero_storage"])
        self.assertEqual(float(row["elapsed_seconds"]), run["elapsed_seconds"])
        self.assertEqual(float(row["sampled_phase_rss_mib"]), run["peak_tree_rss_mib"])
        self.assertEqual(float(row["max_relative_true_residual"]), original["relative_residual"])
        self.assertEqual(float(row["max_relative_backward_residual"]), original["relative_backward_residual"])

    def test_h80_scaled_factor_probe_checks_same_seed_candidates_and_guards(self):
        key = "full_device_h80_gcrotmk_scaled_double_fill4_residual_correction"
        result = self.ledger[key]
        control = self.ledger[result["control_case"]]
        prior = self.ledger[result["seed_case"]]
        stage = self.ledger[result["reference_operator_case"]]
        raw = result["raw_files"]
        for name in ("A", "rhs", "stages"):
            self.assertEqual(raw[name], stage["raw_files"][name])
        self.assertEqual(raw["seed"], control["raw_files"]["seed"])
        self.assertEqual(raw["seed"], prior["raw_files"]["terminal_vector"])
        self.assertEqual(raw["control_terminal"], control["raw_files"]["terminal_vector"])
        self.assertEqual(result["equation_array_sha256"], control["equation_array_sha256"])
        self.assertEqual(result["numerical_thread_environment"], control["numerical_thread_environment"])
        self.assertEqual((result["frequency_hz"], result["incident_side"], result["incident_columns"]), (1e9, "left", [0]))
        self.assertEqual(result["incident_root_index"], control["incident_root_index"])

        source, compatibility = result["source"], result["source_compatibility_exception"]
        path = "src/phononic/resumable_scattering_AP_pnc.py"
        staged_sha = "68c12549b47a9ec98abf7b39d78ad597dd114da00e326adf8ebbec060947b53f"
        current_sha = "dba6002d982647810f3fc714fdce73a0f34e306e06e8519df2d54778002205ec"
        self.assertEqual(compatibility["path"], path)
        self.assertEqual(compatibility["staged_sha256"], stage["source"]["source_sha256"][path])
        self.assertEqual((compatibility["staged_sha256"], compatibility["current_sha256"]), (staged_sha, current_sha))
        self.assertEqual(compatibility["staged_git_commit"], stage["source"]["git_commit"])
        self.assertEqual(compatibility["current_git_commit"], source["git_commit"])
        self.assertEqual(source["source_sha256"], stage["source"]["source_sha256"] | {path: current_sha})
        for name in ("packages", "python", "platform"):
            self.assertEqual(source[name], stage["source"][name])

        factor = result["factorization"]
        for name in ("dtype", "fill_factor", "drop_tol", "permc_spec"):
            self.assertEqual(factor[name], control["factorization"][name])
        self.assertEqual(result["equation_dtype"], "complex128")
        self.assertEqual(result["preconditioner_side"], "flexible_right")
        self.assertEqual(result["gcrotmk_settings"], control["gcrotmk_settings"])
        target, relative_gate, backward_gate = (result[name] for name in (
            "relative_tolerance", "relative_residual_limit", "backward_residual_limit"))
        self.assertEqual((target, relative_gate, backward_gate), (1e-10, 1e-9, 1e-9))

        estimate, admission, run = result["admission_estimate"], result["admission"], result["run"]
        baseline = control["admission_estimate"]
        for name in ("operator_storage_bytes", "rhs_workspace_bytes", "factor_input_bytes", "factor_buffer_bytes",
                     "krylov_workspace_bytes", "large_factor_margin_mib"):
            self.assertEqual(estimate[name], baseline[name])
        dofs = result["body_dofs"]
        workspace = (estimate["scaling_application_bytes"], estimate["scaling_vectors_bytes"],
                     estimate["scaling_batch_workspace_bytes"])
        self.assertEqual(workspace, (2 * 16 * dofs, 2 * 8 * dofs, 64 * 2**20))
        self.assertLessEqual(estimate["allocated_scaling_batch_bound_bytes"], workspace[-1])
        self.assertTrue(estimate["scaled_factor_fill_is_empirical"])
        self.assertEqual(estimate["modeled_commit_mib"], baseline["modeled_commit_mib"] + sum(workspace) / 2**20)
        peak = max(prior["run"]["peak_tree_private_commit_mib"], control["run"]["peak_tree_private_commit_mib"])
        required_commit = max(estimate["modeled_commit_mib"], peak + run["commit_headroom_floor_mib"])
        self.assertEqual(admission, {"required_commit_mib": required_commit, "required_physical_mib": 1200})
        self.assertGreaterEqual(run["initial_memory"]["commit_headroom_mib"], required_commit)
        self.assertGreaterEqual(run["initial_memory"]["physical_available_mib"], admission["required_physical_mib"])
        required_pre_factor = required_commit - (estimate["operator_storage_bytes"] + 16 * dofs) / 2**20
        self.assertEqual(estimate["required_pre_factor_commit_headroom_mib"], required_pre_factor)
        self.assertGreaterEqual(result["before_factorization"]["commit_headroom_mib"], required_pre_factor)
        self.assertGreaterEqual(result["before_factorization"]["physical_available_mib"], run["available_floor_mib"])
        self.assertEqual(run["wall_cap_seconds"], control["run"]["wall_cap_seconds"])
        self.assertLess(run["elapsed_seconds"], run["wall_cap_seconds"])
        self.assertTrue(run["worker_stopped"])
        self.assertGreaterEqual(run["initial_disk_free_mib"], run["required_disk_free_mib"])
        for observed, limit, check in (
                ("peak_tree_rss_mib", "rss_cap_mib", self.assertLessEqual),
                ("peak_tree_private_commit_mib", "private_commit_cap_mib", self.assertLessEqual),
                ("minimum_available_mib", "available_floor_mib", self.assertLess),
                ("minimum_commit_headroom_mib", "commit_headroom_floor_mib", self.assertGreaterEqual)):
            self.assertEqual(run[limit], control["run"][limit])
            check(run[observed], run[limit])

        scaling = factor["factor_scaling"]
        self.assertEqual((scaling["steps"], scaling["max_norm"], scaling["scale_exponent"]),
                         (1, "complex_modulus", -0.5))
        self.assertEqual(scaling["matrix"], "D_r A D_c")
        self.assertEqual(scaling["application"], "D_c ILU(D_r A D_c)^-1 D_r")
        small = result["small_scaling_check"]
        self.assertEqual((small["status"], small["correction_outcome"]), ("passed", "first_drive_accepted"))
        self.assertTrue(small["unequal_row_column_scales"])
        self.assertGreater(small["swapped_map_relative_dense_error"], 0.01)
        for name in ("scaled_matrix_relative_dense_error", "native_application_relative_dense_error",
                     "mapped_application_relative_dense_error"):
            self.assertLessEqual(small[name], 1e-14)
        self.assertEqual(small["original_csc_sha256_before"], small["original_csc_sha256_after"])

        candidates = result["candidates"]
        self.assertEqual(len({entry["artifact"]["path"] for entry in candidates}), len(candidates))
        self.assertEqual([entry["event"] for entry in candidates], ["iterate"] * 3)
        self.assertEqual([entry["iteration"] for entry in candidates], [1, 2, 3])
        self.assertEqual(candidates[0]["artifact"]["sha256"], raw["seed"]["sha256"])
        self.assertFalse(result["terminal_candidate_available"])
        self.assertFalse(result["completed_probe_comparison_available"])
        seed = result["seed_original_equation_checks"]
        control_terminal = result["control_original_equation_checks"]
        for measured, previous in (
                (seed, prior["terminal_candidate"]["independent_original_equation_checks"]),
                (control_terminal, control["terminal_candidate"]["independent_original_equation_checks"])):
            np.testing.assert_allclose([measured[name] for name in previous], list(previous.values()), rtol=1e-12, atol=0)
        self.assertAlmostEqual(seed["relative_residual"] * result["gcrotmk_settings"]["rtol"] / target, 1, places=12)
        for candidate in candidates:
            measured = candidate["independent_original_equation_checks"]
            relative = measured["residual_norm_2"] / result["original_rhs_norm_2"]
            scale = result["original_operator_inf_norm"] * measured["solution_norm_inf"] + result["original_rhs_norm_inf"]
            backward = measured["residual_norm_inf"] / scale
            np.testing.assert_allclose([relative, backward], [measured["relative_residual"], measured["relative_backward_residual"]],
                                       rtol=1e-12, atol=0)
            self.assertEqual(candidate["solver_relative_target_passed"], relative <= target)
            self.assertEqual(candidate["relative_residual_gate_passed"], relative <= relative_gate)
            self.assertEqual(candidate["backward_residual_gate_passed"], backward <= backward_gate)
            control_relative = control["callback_original_relative_residuals"][candidate["iteration"] - 1]
            self.assertEqual(candidate["control_callback_relative_residual"], control_relative)
            self.assertAlmostEqual(candidate["relative_residual_change_from_control_fraction"],
                                   relative / control_relative - 1)
        self.assertTrue(all(entry["relative_residual_change_from_control_fraction"] > 0 for entry in candidates[1:]))
        best = min(candidates, key=lambda entry: entry["independent_original_equation_checks"]["relative_residual"])
        best_checks = best["independent_original_equation_checks"]
        self.assertAlmostEqual(result["best_saved_relative_residual_reduction_fraction"],
                               1 - best_checks["relative_residual"] / seed["relative_residual"])
        self.assertEqual(result["best_saved_candidate"], best["artifact"])
        self.assertGreater(best_checks["relative_residual"], control_terminal["relative_residual"])
        self.assertEqual(result["best_retained_vector"], raw["control_terminal"])
        self.assertFalse(best["relative_residual_gate_passed"])
        self.assertTrue(best["backward_residual_gate_passed"])
        self.assertEqual((result["status"], run["status"]), ("aborted_memory_limit", "aborted_memory_limit"))
        self.assertEqual((result["accepted_channels"], run["accepted_channels"], result["total_device_channels"]), (0, 0, 5))
        self.assertFalse(result["scattering_computed"])
        self.assertFalse(result["spatial_convergence_established"])

        table = LEDGER.with_suffix(".csv").read_bytes()
        self.assertEqual(sha256(table).hexdigest(), self.ledger["csv_sha256"])
        row, = [row for row in csv.DictReader(io.StringIO(table.decode("utf-8"))) if row["case"] == key]
        self.assertEqual((row["phase"], row["status"]), ("first_drive_scaled_factor_correction", run["status"]))
        for name, value in (
                ("body_dofs", dofs), ("operator_nnz", result["operator_nnz"]),
                ("elapsed_seconds", run["elapsed_seconds"]), ("sampled_phase_rss_mib", run["peak_tree_rss_mib"]),
                ("factor_nnz", factor["factor_nonzeros"]), ("ilu_fill_factor", factor["fill_factor"]),
                ("max_gmres_iterations", candidates[-1]["iteration"]), ("max_relative_true_residual", best_checks["relative_residual"]),
                ("max_relative_backward_residual", best_checks["relative_backward_residual"])):
            self.assertEqual(float(row[name]), value)

    def test_full_device_body_mesh_pair_replays_matching_and_gates(self):
        for case in ("full_device_h200_h160_mixed_comparison", "full_device_h160_h130_mixed_comparison",
                     "full_device_h130_h100_mixed_comparison"):
            with self.subTest(case=case):
                self.check_body_mesh_pair(self.ledger[case])

    def test_h100_pair_fails_complex_response_with_unchanged_mode_identities(self):
        pair = self.ledger["full_device_h130_h100_mixed_comparison"]
        reference, candidate = (self.ledger[pair[f"{side}_case"]] for side in ("reference", "candidate"))
        self.assertEqual(reference["scattering"]["groups"], candidate["scattering"]["groups"])
        self.assertTrue(pair["input_power_checks_passed"])
        self.assertTrue(pair["power_passed"])
        self.assertFalse(pair["complex_passed"])
        self.assertFalse(pair["mesh_pair_passed"])

    def check_body_mesh_pair(self, comparison):
        reference, candidate = (self.ledger[comparison[f"{name}_case"]] for name in ("reference", "candidate"))
        a, b = reference["scattering"], candidate["scattering"]
        for key in ("frequency_hz", "incident_side", "dimensions_nm"):
            self.assertEqual(a[key], b[key])
        planes = comparison["reference_planes_m"]
        self.assertEqual(planes, {"xleft": 0, "xright": a["dimensions_nm"]["length_nm"] * 1e-9})
        self.assertFalse(comparison["phase_adjustment_applied"])
        compatibility = comparison["linear_solver_compatibility"]
        if compatibility is not None:
            self.assertEqual(compatibility["relative_residual_limit"], 1e-9)
            self.assertEqual(compatibility["backward_residual_limit"], 1e-9)
        for side, case in (("reference", reference), ("candidate", candidate)):
            sources = dict(case["scattering"]["numerical_source_sha256"])
            if compatibility is not None:
                checked = compatibility["checks"][side]
                self.assertEqual(checked["precision"], case["configuration"]["resources"].get("ilu_precision", "double"))
                self.assertEqual(sources.pop("src/phononic/linear_lgmres_AP_pnc.py"), checked["source_sha256"])
                self.assertLessEqual(checked["maximum_relative_residual"], compatibility["relative_residual_limit"])
                self.assertLessEqual(checked["maximum_backward_residual"], compatibility["backward_residual_limit"])
            self.assertEqual(sources, comparison["physical_inputs"]["numerical_source_sha256"])

        def match(group):
            mapping = comparison["matched_channels"][group]
            a_roots = a["groups"][group]["root_indices"]
            b_roots = b["groups"][group]["root_indices"]
            self.assertEqual(mapping["reference_roots"], a_roots)
            self.assertEqual(sorted(mapping["candidate_roots"]), sorted(b_roots))
            order = [b_roots.index(root) for root in mapping["candidate_roots"]]
            np.testing.assert_array_equal(a["groups"][group]["wavenumbers_rad_m"],
                                          np.asarray(b["groups"][group]["wavenumbers_rad_m"])[order])
            np.testing.assert_allclose(mapping["mass_overlaps"], 1, atol=1e-12, rtol=0)
            return order, np.exp(1j * np.asarray(mapping["candidate_gauge_phase_rad"]))

        incident_order, incident_gauge = match("incoming")
        maximum_power_change, maximum_complex_change = 0.0, 0.0
        for name in ("reflection", "transmission"):
            outgoing_order, outgoing_gauge = match(name)
            reference_block = complex_block(a, name)
            ordered = complex_block(b, name)[np.ix_(outgoing_order, incident_order)]
            candidate_block = outgoing_gauge[:, None] * ordered / incident_gauge
            reference_power = np.sum(abs(reference_block) ** 2, axis=0)
            candidate_power = np.sum(abs(candidate_block) ** 2, axis=0)
            power_change = np.max(abs(candidate_power - reference_power))
            complex_change = np.max(abs(candidate_block - reference_block))
            self.assertAlmostEqual(power_change, comparison["block_maximum_changes"][name]["maximum_power_change"], places=12)
            self.assertAlmostEqual(complex_change, comparison["block_maximum_changes"][name]["maximum_complex_change"], places=12)
            maximum_power_change = max(maximum_power_change, power_change)
            maximum_complex_change = max(maximum_complex_change, complex_change)
            if name == "transmission":
                incoming_roots = a["groups"]["incoming"]["root_indices"]
                transmitted_roots = a["groups"]["transmission"]["root_indices"]
                rows = [transmitted_roots.index(root) for root in incoming_roots]
                columns = range(len(rows))
                phase = np.angle(candidate_block[rows, columns] / reference_block[rows, columns])
                np.testing.assert_allclose(phase, comparison["same_channel_transmission_phase_change_rad"], atol=1e-12, rtol=0)

        self.assertAlmostEqual(maximum_power_change, comparison["maximum_power_change"], places=12)
        self.assertAlmostEqual(maximum_complex_change, comparison["maximum_complex_change"], places=12)
        self.assertEqual(comparison["power_change_limit"], 0.01)
        self.assertEqual(comparison["complex_change_limit"], 0.01)
        power_passed = maximum_power_change < comparison["power_change_limit"]
        complex_passed = maximum_complex_change < comparison["complex_change_limit"]
        self.assertEqual(comparison["power_passed"], power_passed)
        self.assertEqual(comparison["complex_passed"], complex_passed)
        input_power_passed = all(case["physical_checks"]["power_balance_passed"] for case in (reference, candidate))
        self.assertEqual(comparison["input_power_checks_passed"], input_power_passed)
        mesh_pair = len(set(comparison["discretization_sha256"].values())) == 2
        self.assertTrue(mesh_pair)
        self.assertEqual(comparison["mesh_pair_passed"], mesh_pair and input_power_passed and power_passed and complex_passed)

    def test_uniform_body_diagnostic_replays_native_phase(self):
        diagnostic = self.ledger["full_device_uniform_h160_h130_diagnostic"]
        self.check_uniform_body_diagnostic(diagnostic, [diagnostic["cases"][size] for size in ("160", "130")])

    def test_h100_uniform_body_diagnostic_replays_saved_h130(self):
        prior = self.ledger["full_device_uniform_h160_h130_diagnostic"]
        diagnostic = self.ledger["full_device_uniform_h130_h100_diagnostic"]
        reference = prior["cases"]["130"]
        candidate = diagnostic["cases"]["100"]
        self.assertEqual(diagnostic["reference_case"], {
            "entry": "full_device_uniform_h160_h130_diagnostic", "size_nm": "130",
            "public_summary": reference["raw_files"]["public_summary"],
        })
        self.assertEqual(diagnostic["reused_safe_sources"], prior["reused_safe_sources"])
        self.assertEqual(candidate["cached_port_provenance"]["source"], prior["reused_safe_sources"])
        self.assertEqual(candidate["cached_port_provenance"]["trace_dofs"], 675)
        self.assertEqual(candidate["preflight"]["end_face_dofs"], {"xleft": 675, "xright": 675})
        self.assertEqual(candidate["preflight"]["hole_count"], 0)
        self.assertEqual(candidate["preflight"]["ndof"], candidate["ndof"])
        self.assertEqual(candidate["preflight"]["mesh_sha256"], candidate["mesh_checkpoint"]["mesh_sha256"])
        self.assertTrue(candidate["mesh_checkpoint"]["reused"])
        self.assertGreater(candidate["ndof"], reference["ndof"])
        for section in ("geometry", "material", "analysis"):
            self.assertEqual(candidate["configuration"][section], reference["configuration"][section])
        mesh = dict(candidate["configuration"]["mesh"])
        self.assertEqual(mesh.pop("mesh_size_nm"), 100)
        reference_mesh = dict(reference["configuration"]["mesh"])
        self.assertEqual(reference_mesh.pop("mesh_size_nm"), 130)
        self.assertEqual(mesh, reference_mesh)
        self.check_uniform_body_diagnostic(diagnostic, [reference, candidate])
        limit = diagnostic["finite_diagnostic_limit"]
        for name, phase in (
            ("native", candidate["same_channel_native_phase_error_rad"]),
            ("pair", diagnostic["same_channel_phase_difference_rad"]),
        ):
            maximum = np.max(np.abs(phase))
            self.assertAlmostEqual(maximum, diagnostic[f"maximum_{name}_phase_error_rad"], places=12)
            self.assertEqual(diagnostic[f"{name}_phase_passed"], maximum < limit)
        self.assertEqual(diagnostic["native_complex_passed"], candidate["maximum_native_complex_error"] < limit)
        self.assertEqual(candidate["run"]["status"], "completed")
        self.assertEqual(candidate["run"]["exit_code"], 0)
        timeout = diagnostic["initial_timed_out_attempt"]
        self.assertEqual(timeout["status"], "stopped_timeout")
        self.assertFalse(timeout["scattering_exported"])
        self.assertEqual(timeout["mesh_sha256"], candidate["mesh_checkpoint"]["mesh_sha256"])
        self.assertEqual(timeout["mechanical_operator_hash"], candidate["mechanical_operator_hash"])
        self.assertEqual(timeout["run"]["limits"]["wall_seconds"], 106)
        self.assertGreaterEqual(timeout["run"]["elapsed_seconds"], 106)
        self.assertEqual(candidate["run"]["limits"], timeout["run"]["limits"] | {"wall_seconds": 360})

        table = LEDGER.with_suffix(".csv").read_bytes()
        self.assertEqual(sha256(table).hexdigest(), self.ledger["csv_sha256"])
        rows = csv.DictReader(io.StringIO(table.decode("utf-8")))
        resource, = [row for row in rows if row["case"] == "uniform_3um_layered_h100"]
        self.assertEqual(int(resource["body_dofs"]), candidate["ndof"])
        self.assertEqual(resource["status"], candidate["status"])
        self.assertEqual(float(resource["elapsed_seconds"]), candidate["run"]["elapsed_seconds"])

    def test_h100_uniform_body_diagnostic_replays_common_safe_reference(self):
        diagnostic = self.ledger["full_device_uniform_h130_h100_diagnostic"]
        reference = self.ledger["full_device_uniform_h160_h130_diagnostic"]["cases"]["130"]
        common = diagnostic["common_safe_reference"]
        reference_file = LEDGER.parent / "full_device_port_resolution_AP_pnc.json"
        reference_bytes = reference_file.read_bytes()
        self.assertEqual(sha256(reference_bytes).hexdigest(), common["source"]["sha256"])
        section = json.loads(reference_bytes)["cases"][common["section"]]
        self.assertEqual(common["section"], "20x6")
        self.assertEqual(common["trace_dofs"], section["trace_dofs"])
        wave = np.asarray(section["propagating"]["positive"]["k_rad_m"])
        np.testing.assert_array_equal(common["reference_wavenumbers_rad_m"], wave)
        self.assertEqual(common["reference_roots"], section["propagating"]["positive"]["root_indices"])
        length = diagnostic["length_m"]
        ideal = np.diag(np.exp(1j * wave * length))
        drift = np.max(abs(np.asarray(diagnostic["wavenumbers_rad_m"]) - wave)) * length
        self.assertAlmostEqual(common["maximum_native_port_phase_drift_rad"], drift, places=12)
        self.assertEqual(common["phase_qualification"]["status"], "passed")
        self.assertEqual(set(common["phase_qualification"]["pairs"]), {"width", "thickness"})
        for size, case in (("130", reference), ("100", diagnostic["cases"]["100"])):
            with self.subTest(size=size):
                summary = common["cases"][size]
                r, t = (complex_block(case["scattering"], name) for name in ("reflection", "transmission"))
                phase = np.angle(np.diag(t) / np.diag(ideal))
                np.testing.assert_allclose(phase, summary["same_channel_phase_error_rad"], rtol=0, atol=1e-12)
                self.assertAlmostEqual(np.max(abs(phase)), summary["maximum_phase_error_rad"], places=12)
                complex_error = max(np.max(abs(r)), np.max(abs(t - ideal)))
                power_error = max(np.max(np.sum(abs(r)**2, axis=0)), np.max(abs(np.sum(abs(t)**2, axis=0) - 1)))
                self.assertAlmostEqual(complex_error, summary["maximum_complex_error"], places=12)
                self.assertAlmostEqual(power_error, summary["maximum_power_error"], places=12)
                self.assertEqual(summary["uniform_response_passed"], max(complex_error, power_error) < 0.01)
                self.assertGreaterEqual(summary["minimum_mode_mass_overlap"], 0.99)

    def check_uniform_body_diagnostic(self, diagnostic, cases):
        self.assertEqual(diagnostic["status"], "diagnostic_completed")
        self.assertFalse(diagnostic["phase_adjustment_applied"])
        h200 = self.ledger["full_device_lgmres_public_replay"]
        mode_checkpoint = self.ledger["full_device_operator"]["raw_files"]["port_modes"]
        self.assertEqual(diagnostic["reused_safe_sources"]["modes"]["sha256"], mode_checkpoint["sha256"])
        self.assertEqual(diagnostic["reused_safe_sources"]["fields"], h200["raw_files"]["selected_fields_AP_pnc.npz"])
        roots = diagnostic["roots"]
        wave = np.asarray(diagnostic["wavenumbers_rad_m"])
        length = diagnostic["length_m"]
        native = np.exp(1j * wave * length)
        phase_limit = diagnostic["finite_diagnostic_limit"]
        self.assertEqual(phase_limit, 0.01)
        reflection, transmission = [], []
        for case in cases:
            with self.subTest(size=case["configuration"]["mesh"]["mesh_size_nm"]):
                self.check_modal_power(case)
                scattering = case["scattering"]
                self.assertEqual(scattering["groups"], h200["scattering"]["groups"])
                self.assertEqual(scattering["groups"]["incoming"]["root_indices"], roots)
                np.testing.assert_array_equal(scattering["groups"]["incoming"]["wavenumbers_rad_m"], wave)
                self.assertEqual(case["configuration"]["geometry"]["length_nm"] * 1e-9, length)
                self.assertEqual(scattering["frequency_hz"], diagnostic["frequency_hz"])
                r, t = (complex_block(scattering, name) for name in ("reflection", "transmission"))
                reflection.append(r)
                transmission.append(t)
                phase = np.angle(np.diag(t) / native)
                np.testing.assert_allclose(phase, case["same_channel_native_phase_error_rad"], atol=1e-12, rtol=0)
                error = np.max(abs(t - np.diag(native)))
                self.assertAlmostEqual(error, case["maximum_native_complex_error"], places=12)
                self.assertAlmostEqual(np.max(abs(r)), case["maximum_reflection_amplitude"], places=12)
                checks = case["physical_checks"]
                matched_power = (max(np.max(np.sum(abs(r) ** 2, axis=0)),
                                     np.max(abs(np.sum(abs(t) ** 2, axis=0) - 1))) < 0.01
                                 and checks["power_balance_passed"])
                self.assertEqual(checks["matched_reference_passed"], matched_power)
                linear = case["linear_solver"]
                for channel in linear["channels"]:
                    self.assertEqual(channel["info"], 0)
                    self.assertLessEqual(channel["relative_residual"], linear["relative_residual_limit"])
                    self.assertLessEqual(channel["relative_backward_residual"], linear["backward_residual_limit"])

        phase_difference = np.angle(np.diag(transmission[1]) / np.diag(transmission[0]))
        np.testing.assert_allclose(phase_difference, diagnostic["same_channel_phase_difference_rad"], atol=1e-12, rtol=0)
        np.testing.assert_allclose(phase_difference / (length * 1e6),
                                   diagnostic["same_channel_phase_difference_rad_per_um"], atol=1e-12, rtol=0)
        power_change, complex_change = 0.0, 0.0
        for name, (first, second) in (("reflection", reflection), ("transmission", transmission)):
            block_change = np.max(abs(second - first))
            self.assertAlmostEqual(block_change, diagnostic[f"maximum_{name}_complex_change"], places=12)
            complex_change = max(complex_change, block_change)
            power_change = max(power_change, np.max(abs(np.sum(abs(second) ** 2 - abs(first) ** 2, axis=0))))
        self.assertAlmostEqual(power_change, diagnostic["maximum_power_change"], places=12)
        self.assertEqual(diagnostic["pair_power_passed"], power_change < phase_limit)
        self.assertEqual(diagnostic["pair_complex_passed"], complex_change < phase_limit)


if __name__ == "__main__":
    unittest.main()
