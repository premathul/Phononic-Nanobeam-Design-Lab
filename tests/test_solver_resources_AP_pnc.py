"""Verify the portable full-device scattering result. Author: Athul Prem | _AP_pnc."""

import json
from pathlib import Path
import unittest

import numpy as np


LEDGER = Path(__file__).resolve().parents[1] / "verification/solver_resources_AP_pnc.json"


class SolverResourceEvidenceTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.ledger = json.loads(LEDGER.read_bytes())

    def test_portable_modal_power_matches_boundary_flux(self):
        result = self.ledger["full_device_lgmres_public_replay"]
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
            coefficients = np.asarray(scattering[f"{direction}_amplitudes"])
            self.assertEqual(coefficients.shape, (5, 5, 2))
            self.assertTrue(np.isfinite(coefficients).all())
            amplitudes = coefficients[..., 0] + 1j * coefficients[..., 1]
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


if __name__ == "__main__":
    unittest.main()
