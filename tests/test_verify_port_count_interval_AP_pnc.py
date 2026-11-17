"""Independent analytic count-interval checks. Athul Prem | _AP_pnc."""

from copy import deepcopy
import csv
import json
from pathlib import Path
from tempfile import TemporaryDirectory
from unittest import TestCase
from unittest.mock import patch

import numpy as np

from tools import verify_port_count_interval_AP_pnc as verifier


LENGTH_M, SPEED_M_S = 1e-6, 5000.0
OMEGA_REFERENCE = 2 * np.pi * 1e9


def analytic_section():
    """Five diagonal branches; the fifth opens at 0.62 GHz.

    Nonzero symmetric G changes nodal traction while cancelling from the
    quadratic dispersion. This detects an omitted traction coupling term.
    """
    return {"K0": np.diag([0., 0., 0., 0., 0.62**2]),
            "M": np.eye(5) / OMEGA_REFERENCE**2,
            "Q": np.diag(np.arange(1., 6.)) * LENGTH_M**2,
            "G": np.diag(np.arange(1., 6.)) * LENGTH_M * 0.1}


def analytic_modes(operators, frequency_hz):
    ratio = frequency_hz / 1e9
    q = np.arange(1., 6.)
    values = np.sqrt(((ratio**2 - np.diag(operators["K0"])) / q).astype(complex)) / LENGTH_M
    waves = np.column_stack((values, -values)).ravel()
    vectors = np.repeat(np.eye(5, dtype=complex), 2, axis=1)
    opened = waves.imag == 0
    raw_power = np.pi * frequency_hz * waves.real * np.repeat(q, 2) * LENGTH_M**2
    reference_power = SPEED_M_S * ratio**2 / 2
    vectors /= np.sqrt(np.where(opened, abs(raw_power), reference_power))
    force = operators["G"] @ vectors + (operators["Q"] @ vectors) * (1j * waves)
    direction = np.where(opened, np.sign(raw_power), np.sign(waves.imag))
    return {"frequency_hz": np.asarray(float(frequency_hz)), "wavenumbers_rad_m": waves,
            "displacements": vectors, "axial_forces": force,
            "positive_indices": np.flatnonzero(direction > 0), "negative_indices": np.flatnonzero(direction < 0),
            "propagating": opened, "powers_w": np.where(opened, direction, 0.), "relative_residuals": np.zeros(10)}


def replay(operators, arrays):
    return verifier.replay_modes(operators, arrays, float(arrays["frequency_hz"]),
                                 length_m=LENGTH_M, reference_speed_m_s=SPEED_M_S, batch_size=3)


def identity():
    return {"schema_version": 1, "initial_interval_hz": verifier.INITIAL_INTERVAL,
            "endpoint_counts": [4, 5], "target_width_hz": verifier.TARGET_WIDTH_HZ,
            "max_new_points": verifier.MAX_NEW_POINTS, "resource_contract": verifier.RESOURCE_CONTRACT}


def signed_receipt(document, lower, upper, index, count, *, status="accepted", degeneracy=False):
    frequency = (lower + upper) // 2
    metrics = {"positive_open_channels": count, "negative_open_channels": count,
               "degeneracy_status": "coincident_propagating_roots" if degeneracy else "none_detected",
               "coincident_propagating_root_groups": [[0, 2]] if degeneracy else [],
               "cutoff_status": "solver_resolved_at_sample"}
    decision = "replace_lower" if count == 4 else "replace_upper" if count == 5 else "stop_unresolved"
    if status != "accepted" or degeneracy:
        decision = "stop_unresolved"
    after = [frequency, upper] if decision == "replace_lower" else [lower, frequency] if decision == "replace_upper" else [lower, upper]
    point = {"index": index, "frequency_hz": frequency, "status": status,
             "identity_sha256": verifier.digest(document), "interval_before_hz": [lower, upper],
             "interval_after_hz": after, "decision": decision, "metrics": metrics,
             "actual_environment": "inputs_AP_pnc.json:identity.archive.current_implementation",
             "eigensolve_performed": True,
             "resources": {"guard_observations": [], "peak_sampled_process_tree_rss_bytes": 100 * 2**20,
                           "minimum_available_bytes": 2**30, "elapsed_seconds": 1.0, "sampling_interval_seconds": 0.05},
             "admission": {"available_bytes": 2 * 2**30, "free_disk_bytes": 2**30,
                           "required_new_disk_bytes": 64 * 2**20, "existing_output_bytes": 0}}
    if decision == "stop_unresolved":
        point["stop_reason"] = "root_degeneracy" if degeneracy else "unexpected_directional_count" if status == "accepted" else "point_failed_or_interrupted"
    if status != "accepted":
        point["error"] = "Synthetic interrupted or failed point."
    point["point_sha256"] = verifier.digest(point)
    return point


def resign(point):
    point["point_sha256"] = verifier.digest({key: value for key, value in point.items() if key != "point_sha256"})


def state_for(document, points):
    lower, upper = points[-1]["interval_after_hz"] if points else verifier.INITIAL_INTERVAL
    unresolved = bool(points and points[-1]["decision"] == "stop_unresolved")
    complete = upper - lower <= verifier.TARGET_WIDTH_HZ and not unresolved
    return {"identity_sha256": verifier.digest(document), "status": "complete" if complete else "unresolved" if unresolved else "pending",
            "stop_reason": "target_sampled_width_reached" if complete else points[-1]["stop_reason"] if unresolved else "point_budget_pause",
            "lower_hz": lower, "upper_hz": upper, "width_hz": upper - lower,
            "target_width_hz": verifier.TARGET_WIDTH_HZ, "new_points_used": len(points),
            "cross_frequency_branch_identity": "unresolved"}


def artifact(path, base):
    return {"path": path.relative_to(base).as_posix(), "sha256": verifier.sha256(path), "bytes": path.stat().st_size}


def write_csv(directory, points):
    columns = ("frequency_hz", "status", "root_id", "direction", "k_real_rad_m", "k_imag_rad_m",
               "signed_power_w", "eigen_backward_residual", "receipt", "point_sha256", "error")
    path = directory / "port_count_interval_AP_pnc.csv"
    with path.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=columns)
        writer.writeheader()
        for point in points:
            for channel in point.get("metrics", {}).get("open_channels", []) or [{}]:
                writer.writerow({**channel, "frequency_hz": point["frequency_hz"], "status": point["status"],
                    "receipt": f"point-{point['frequency_hz']}_AP_pnc.json", "point_sha256": point["point_sha256"],
                    "error": point.get("error", "")})
    return path


class VerifyPortCountIntervalTests(TestCase):
    def test_closed_form_full_roots_original_force_flux_and_pairs(self):
        operators = analytic_section()
        for frequency, count in ((100_000_000, 4), (550_000_000, 4), (775_000_000, 5), (1_000_000_000, 5)):
            actual = replay(operators, analytic_modes(operators, frequency))
            self.assertEqual(actual["discrete_roots"], 10)
            self.assertEqual(actual["positive_open_channels"], count)
            self.assertEqual(actual["negative_open_channels"], count)
            self.assertLess(actual["maximum_eigen_backward_residual"], 1e-15)
            self.assertLess(actual["maximum_relative_traction_error"], 1e-15)
            self.assertLess(actual["positive_flux_gram_error_w"], 1e-14)
            self.assertEqual(actual["root_partner_checks"]["reciprocity"]["maximum_mismatch_rad_m"], 0)

    def test_wrong_traction_equilibrium_signed_direction_and_missing_root_fail(self):
        operators = analytic_section()
        arrays = analytic_modes(operators, 550_000_000)
        wrong = deepcopy(arrays)
        wrong["axial_forces"] -= operators["G"] @ wrong["displacements"]
        with self.assertRaisesRegex(ValueError, "traction"):
            replay(operators, wrong)
        wrong = deepcopy(arrays)
        wrong["wavenumbers_rad_m"][0] *= 1.01
        with self.assertRaisesRegex(ValueError, "eigenpair"):
            replay(operators, wrong)
        wrong = deepcopy(arrays)
        wrong["positive_indices"], wrong["negative_indices"] = wrong["negative_indices"], wrong["positive_indices"]
        with self.assertRaisesRegex(ValueError, "signed direction"):
            replay(operators, wrong)
        wrong = deepcopy(arrays)
        wrong["positive_indices"][0] = wrong["negative_indices"][0]
        with self.assertRaisesRegex(ValueError, "incomplete signed"):
            replay(operators, wrong)

    def test_cutoff_vanishing_power_and_nonbijective_pairs_fail(self):
        operators = analytic_section()
        arrays = analytic_modes(operators, 550_000_000)
        arrays["wavenumbers_rad_m"][0] = 0
        with self.assertRaisesRegex(ValueError, "cutoff"):
            replay(operators, arrays)
        arrays = analytic_modes(operators, 550_000_000)
        with self.assertRaisesRegex(ValueError, "vanishing power"):
            verifier.replay_modes(operators, arrays, 550_000_000, length_m=LENGTH_M,
                                  reference_speed_m_s=1e20)
        with self.assertRaisesRegex(ValueError, "pairing"):
            verifier.pair_roots(np.array([1., 1., -1., -2.]), 1.)

    def test_four_exact_dyadic_decisions_and_terminal_rejection(self):
        document, points = identity(), []
        lower, upper = verifier.INITIAL_INTERVAL
        for index, count in enumerate((4, 5, 5, 4), 1):
            point = signed_receipt(document, lower, upper, index, count)
            points.append(point)
            lower, upper = point["interval_after_hz"]
        state = state_for(document, points)
        result = verifier.verify_history(document, state, points)
        self.assertEqual((result["lower_hz"], result["upper_hz"]), (606_250_000, 662_500_000))
        self.assertTrue(result["target_width_reached"])
        wrong = deepcopy(points)
        wrong[1]["decision"] = "replace_lower"
        resign(wrong[1])
        with self.assertRaisesRegex(ValueError, "decision"):
            verifier.verify_history(document, state, wrong)
        state["status"] = "pending"
        with self.assertRaisesRegex(ValueError, "status"):
            verifier.verify_history(document, state, points)

    def test_unexpected_count_cutoff_and_degeneracy_stop_without_moving_interval(self):
        document = identity()
        for count, status, degeneracy in ((6, "accepted", False), (4, "failed", False),
                                           (5, "interrupted", False), (4, "accepted", True)):
            point = signed_receipt(document, *verifier.INITIAL_INTERVAL, 1, count,
                                   status=status, degeneracy=degeneracy)
            state = state_for(document, [point])
            result = verifier.verify_history(document, state, [point])
            self.assertTrue(result["stopped_unresolved"])
            self.assertEqual(result["width_hz"], 900_000_000)
            wrong = deepcopy(point)
            wrong["decision"] = "replace_lower"
            resign(wrong)
            with self.assertRaisesRegex(ValueError, "decision"):
                verifier.verify_history(document, state, [wrong])

    def test_receipt_resource_and_artifact_tampering_fail(self):
        document = identity()
        point = signed_receipt(document, *verifier.INITIAL_INTERVAL, 1, 4)
        state = state_for(document, [point])
        wrong = deepcopy(point)
        wrong["frequency_hz"] += 1
        with self.assertRaisesRegex(ValueError, "midpoint"):
            verifier.verify_history(document, state, [wrong])
        wrong = deepcopy(point)
        wrong["resources"]["peak_sampled_process_tree_rss_bytes"] = 2 * 2**30
        resign(wrong)
        with self.assertRaisesRegex(ValueError, "resource"):
            verifier.verify_history(document, state, [wrong])
        with TemporaryDirectory() as temporary:
            base = Path(temporary)
            path = base / "sample.txt"
            path.write_text("first", encoding="utf-8")
            link = artifact(path, base)
            path.write_text("other", encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "Artifact identity"):
                verifier.linked(link, base)

    def test_complete_synthetic_file_audit_and_csv_corruption(self):
        operators, document = analytic_section(), identity()
        with TemporaryDirectory() as temporary:
            base, points = Path(temporary), []
            lower, upper = verifier.INITIAL_INTERVAL
            for index, count in enumerate((4, 5, 5, 4), 1):
                point = signed_receipt(document, lower, upper, index, count)
                frequency = point["frequency_hz"]
                arrays = analytic_modes(operators, frequency)
                point["metrics"].update(replay(operators, arrays))
                mode_path = base / f"modes-{frequency}_AP_pnc.npz"
                np.savez_compressed(mode_path, **arrays)
                point["modes"] = artifact(mode_path, base)
                resign(point)
                path = base / f"point-{frequency}_AP_pnc.json"
                path.write_text(json.dumps(point), encoding="utf-8")
                points.append(point)
                lower, upper = point["interval_after_hz"]
            state = state_for(document, points)
            state["points"] = [artifact(base / f"point-{point['frequency_hz']}_AP_pnc.json", base) for point in points]
            (base / "inputs_AP_pnc.json").write_text(json.dumps({"identity": document, "identity_sha256": verifier.digest(document)}), encoding="utf-8")
            (base / "interval_state_AP_pnc.json").write_text(json.dumps(state), encoding="utf-8")
            csv_path = write_csv(base, points)
            with patch.object(verifier, "audit_archived_inputs", return_value=base / "synthetic-original.npz"), \
                 patch.object(verifier, "load_original_operators", return_value=(operators, LENGTH_M, SPEED_M_S)):
                result = verifier.verify_interval(base)
            self.assertEqual(result["checked_new_roots"], 40)
            self.assertEqual(result["checked_csv_rows"], 36)
            self.assertEqual(result["replayed_archived_roots"], 0)
            csv_path.write_text(csv_path.read_text(encoding="utf-8").replace("positive", "negative", 1), encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "CSV rows"):
                verifier.verify_csv(base, points)

    def test_endpoint_ledger_pin_and_independent_module_boundary(self):
        with TemporaryDirectory() as temporary:
            base = Path(temporary)
            path = base / "verification/port_inventory_AP_pnc.json"
            path.parent.mkdir()
            path.write_text("{}", encoding="utf-8")
            document = {"archive": {"inventory_ledger": artifact(path, base)}}
            with self.assertRaisesRegex(ValueError, "ledger identity"):
                verifier.audit_archived_inputs(document, base)
        source = Path(verifier.__file__).read_text(encoding="utf-8")
        self.assertNotIn("from tools import refine_port_count", source)
        self.assertNotIn("import phononic", source)
        self.assertNotIn("validate_modes(", source)
