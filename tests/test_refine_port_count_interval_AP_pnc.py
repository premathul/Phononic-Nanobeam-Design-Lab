"""Synthetic interval/recovery checks. Athul Prem | _AP_pnc."""
from contextlib import contextmanager
import json
from pathlib import Path
from tempfile import TemporaryDirectory
from types import SimpleNamespace
from unittest import TestCase
from unittest.mock import patch

import numpy as np

from tools import refine_port_count_interval_AP_pnc as refine


def read(path):
    return json.loads(path.read_bytes())


def snapshot(path):
    return path.read_bytes(), path.stat().st_mtime_ns


@contextmanager
def quiet_monitor(output):
    yield {"guard_observations": [], "elapsed_seconds": 0.001}


def fake_modes(port, frequency):
    # Opaque tiny arrays exercise persistence without a scientific solve.
    fields = {name: np.array([1.0]) for name in refine.inventory.MODE_FIELDS}
    fields.update(port=port, frequency_hz=frequency)
    return SimpleNamespace(**fields)


def metrics(count=4, *, negative=None, degeneracy=False):
    return {"positive_open_channels": count, "negative_open_channels": count if negative is None else negative,
            "degeneracy_status": "coincident_propagating_roots" if degeneracy else "none_detected",
            "coincident_propagating_root_groups": [[0, 1]] if degeneracy else [],
            "open_channels": [{"root_id": 0, "direction": "positive", "k_real_rad_m": 1.0,
                               "k_imag_rad_m": 0.0, "signed_power_w": 1.0, "eigen_backward_residual": 0.0}]}


class PortCountIntervalTests(TestCase):
    def setUp(self):
        self.port = object()
        self.archive = {"section": "synthetic", "current_implementation": {"python": "fixture"},
                        "execution_provenance": {"git_commit": "initial"}}

    def run_fixture(self, output, **kwargs):
        options = dict(solve=fake_modes, validate=lambda modes: metrics(4), admit=lambda path: {}, monitor=quiet_monitor)
        options.update(kwargs)
        return refine.run_interval(output, self.port, self.archive, **options)

    def test_exact_dyadic_decisions_and_unexpected_counts(self):
        self.assertEqual(refine.count_decision(100, 1000, 550, 4, 4), ("replace_lower", (550, 1000)))
        self.assertEqual(refine.count_decision(100, 1000, 550, 5, 5), ("replace_upper", (100, 550)))
        for positive, negative in ((3, 3), (6, 6), (4, 5), (4.0, 4), (True, True)):
            self.assertEqual(refine.count_decision(100, 1000, 550, positive, negative),
                             ("stop_unresolved", (100, 1000)))
        with self.assertRaisesRegex(ValueError, "midpoint"):
            refine.count_decision(100, 1000, 551, 4, 4)

    def test_four_step_budget_and_accepted_reuse_preserve_bytes_and_mtime(self):
        with TemporaryDirectory() as directory:
            output = Path(directory) / "study"
            calls = []
            counts = iter((4, 5, 4, 5))
            def solve(port, frequency):
                calls.append(frequency)
                return fake_modes(port, frequency)
            for index in range(4):
                state = self.run_fixture(output, resume=index > 0, solve=solve,
                                         validate=lambda modes: metrics(next(counts)))
            self.assertEqual(calls, [550_000_000, 775_000_000, 662_500_000, 718_750_000])
            self.assertEqual((state["lower_hz"], state["upper_hz"], state["width_hz"]),
                             (662_500_000, 718_750_000, 56_250_000))
            self.assertEqual(state["status"], "complete")
            retained = sorted(p for p in output.iterdir() if p.is_file())
            before = {p.name: snapshot(p) for p in retained}
            # Repository metadata may change after merge without changing numerics.
            self.archive["execution_provenance"]["git_commit"] = "merged"
            self.run_fixture(output, resume=True, max_new_points=0,
                             solve=lambda *args: self.fail("An accepted point was recomputed."))
            self.assertEqual({p.name: snapshot(p) for p in retained}, before)
            self.run_fixture(output, resume=True, max_new_points=4,
                             solve=lambda *args: self.fail("A fifth point was attempted."))
            self.assertEqual({p.name: snapshot(p) for p in retained}, before)

    def test_unexpected_nonmonotone_count_stops_and_retains_full_modes(self):
        with TemporaryDirectory() as directory:
            output = Path(directory) / "study"
            first = self.run_fixture(output)
            first_receipt = output / "point-550000000_AP_pnc.json"
            frozen = snapshot(first_receipt)
            state = self.run_fixture(output, resume=True, max_new_points=4, validate=lambda modes: metrics(3))
            self.assertEqual(state["status"], "unresolved")
            self.assertEqual(state["new_points_used"], 2)
            self.assertEqual((state["lower_hz"], state["upper_hz"]), (first["lower_hz"], first["upper_hz"]))
            point = read(output / "point-775000000_AP_pnc.json")
            self.assertEqual((point["status"], point["decision"]), ("accepted", "stop_unresolved"))
            self.assertTrue((output / point["modes"]["path"]).exists())
            self.run_fixture(output, resume=True, solve=lambda *args: self.fail("Unexpected count was retried."))
            self.assertEqual(snapshot(first_receipt), frozen)

    def test_valid_degenerate_modes_stop_without_moving_endpoint_and_reuse(self):
        with TemporaryDirectory() as directory:
            output = Path(directory) / "study"
            state = self.run_fixture(output, validate=lambda modes: metrics(5, degeneracy=True))
            self.assertEqual(state["status"], "unresolved")
            self.assertEqual(state["stop_reason"], "root_degeneracy")
            self.assertEqual((state["lower_hz"], state["upper_hz"]), refine.INITIAL)
            point = read(output / "point-550000000_AP_pnc.json")
            self.assertEqual(point["status"], "accepted")
            self.assertIn("modes", point)
            self.run_fixture(output, resume=True, solve=lambda *args: self.fail("Degenerate point was retried."))

    def test_cutoff_failure_and_interruption_are_terminal(self):
        for exception in (RuntimeError("Unresolved lead cutoff."), KeyboardInterrupt("stopped")):
            with self.subTest(exception=type(exception).__name__), TemporaryDirectory() as directory:
                output = Path(directory) / "study"
                def fail(*args):
                    raise exception
                state = self.run_fixture(output, max_new_points=4, solve=fail)
                self.assertEqual(state["status"], "unresolved")
                self.assertEqual(state["new_points_used"], 1)
                self.assertEqual((state["lower_hz"], state["upper_hz"]), refine.INITIAL)
                point = read(output / "point-550000000_AP_pnc.json")
                self.assertEqual(point["status"], "interrupted" if isinstance(exception, KeyboardInterrupt) else "failed")
                self.assertIn("error", point)
                self.run_fixture(output, resume=True, solve=lambda *args: self.fail("Stopped point was retried."))

    def test_external_worker_interruption_marks_running_point_and_never_retries(self):
        with TemporaryDirectory() as directory:
            output = Path(directory) / "study"
            self.run_fixture(output)
            receipt = output / "point-550000000_AP_pnc.json"
            point = read(receipt)
            point.update(status="running", decision="stop_unresolved", interval_after_hz=list(refine.INITIAL))
            refine.write_point(receipt, point)
            state = self.run_fixture(output, resume=True, solve=lambda *args: self.fail("Running point was retried."))
            self.assertEqual(state["status"], "unresolved")
            self.assertEqual(read(receipt)["status"], "interrupted")

    def test_changed_input_receipt_modes_state_and_csv_are_refused_before_solve(self):
        with TemporaryDirectory() as directory:
            output = Path(directory) / "study"
            self.run_fixture(output)
            blocked = lambda *args: self.fail("A corrupted study reached a solve.")
            for name in ("point-550000000_AP_pnc.json", "modes-550000000_AP_pnc.npz",
                         "interval_state_AP_pnc.json", "port_count_interval_AP_pnc.csv"):
                with self.subTest(name=name):
                    path, original = output / name, (output / name).read_bytes()
                    if name == "interval_state_AP_pnc.json":
                        changed = read(path)
                        changed["lower_hz"] += 1
                        path.write_text(json.dumps(changed), encoding="utf-8")
                    else:
                        path.write_bytes(original + b"corrupted")
                    with self.assertRaises((ValueError, RuntimeError)):
                        self.run_fixture(output, resume=True, max_new_points=0, solve=blocked)
                    path.write_bytes(original)
            self.archive["current_implementation"]["python"] = "changed"
            with self.assertRaisesRegex(ValueError, "inputs|implementation"):
                self.run_fixture(output, resume=True, solve=blocked)

    def test_resource_guard_and_disk_admission_fail_before_acceptance(self):
        @contextmanager
        def limited(output):
            yield {"guard_observations": ["available_memory"]}
        with TemporaryDirectory() as directory:
            output = Path(directory) / "study"
            state = self.run_fixture(output, monitor=limited)
            point = read(output / "point-550000000_AP_pnc.json")
            self.assertEqual(point["status"], "failed")
            self.assertIn("retained_unaccepted_modes", point)
            self.assertEqual(state["status"], "unresolved")
            with patch.object(refine.psutil, "virtual_memory", return_value=SimpleNamespace(available=2**31)), \
                    patch.object(refine.shutil, "disk_usage", return_value=SimpleNamespace(free=63 * refine.MIB)):
                with self.assertRaisesRegex(ValueError, "disk"):
                    refine.admission(output)

    def test_point_admission_refusal_is_retained_without_a_solve_or_retry(self):
        with TemporaryDirectory() as directory:
            output = Path(directory) / "study"
            def refuse(path):
                raise ValueError("Insufficient available memory.")
            state = self.run_fixture(output, admit=refuse,
                                     solve=lambda *args: self.fail("Refused point reached a solve."))
            point = read(output / "point-550000000_AP_pnc.json")
            self.assertEqual(state["status"], "unresolved")
            self.assertEqual(point["status"], "failed")
            self.assertFalse(point["eigensolve_performed"])
            self.run_fixture(output, resume=True, solve=lambda *args: self.fail("Refused point was retried."))

    def test_budget_validation_and_frozen_archive_rejection_precede_array_read(self):
        with TemporaryDirectory() as directory:
            for count in (-1, 5, True):
                with self.assertRaisesRegex(ValueError, "budget"):
                    self.run_fixture(Path(directory) / "study", max_new_points=count)
        with patch.object(refine, "sha", return_value="changed"), \
                patch.object(refine.workflow, "source_identity", return_value={}), \
                patch.object(refine.np, "load", side_effect=AssertionError("Unqualified archive reached array load.")):
            with self.assertRaisesRegex(ValueError, "frozen inventory"):
                refine.load_archived_inputs()
