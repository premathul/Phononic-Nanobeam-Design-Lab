"""Verify portable full-device scattering evidence. Author: Athul Prem | _AP_pnc."""

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
        for case in ("full_device_lgmres_public_replay", "full_device_h160_lgmres_public_replay"):
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

    def test_full_device_body_mesh_pair_replays_matching_and_gates(self):
        comparison = self.ledger["full_device_h200_h160_mixed_comparison"]
        reference, candidate = (self.ledger[comparison[f"{name}_case"]] for name in ("reference", "candidate"))
        a, b = reference["scattering"], candidate["scattering"]
        for key in ("frequency_hz", "incident_side", "dimensions_nm"):
            self.assertEqual(a[key], b[key])
        planes = comparison["reference_planes_m"]
        self.assertEqual(planes, {"xleft": 0, "xright": a["dimensions_nm"]["length_nm"] * 1e-9})
        self.assertFalse(comparison["phase_adjustment_applied"])
        compatibility = comparison["linear_solver_compatibility"]
        self.assertEqual(compatibility["relative_residual_limit"], 1e-9)
        self.assertEqual(compatibility["backward_residual_limit"], 1e-9)
        for side, case, precision in (("reference", reference, "double"), ("candidate", candidate, "single")):
            checked = compatibility["checks"][side]
            self.assertEqual(checked["precision"], precision)
            sources = dict(case["scattering"]["numerical_source_sha256"])
            self.assertEqual(sources.pop("src/phononic/linear_lgmres_AP_pnc.py"), checked["source_sha256"])
            self.assertEqual(sources, comparison["physical_inputs"]["numerical_source_sha256"])
            self.assertLessEqual(checked["maximum_relative_residual"], compatibility["relative_residual_limit"])
            self.assertLessEqual(checked["maximum_backward_residual"], compatibility["backward_residual_limit"])

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


if __name__ == "__main__":
    unittest.main()
