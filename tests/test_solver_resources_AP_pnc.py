"""Verify portable full-device scattering evidence. Author: Athul Prem | _AP_pnc."""

import csv
from hashlib import sha256
import io
import json
from pathlib import Path
import unittest

import numpy as np


LEDGER = Path(__file__).resolve().parents[1] / "verification/solver_resources_AP_pnc.json"


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
