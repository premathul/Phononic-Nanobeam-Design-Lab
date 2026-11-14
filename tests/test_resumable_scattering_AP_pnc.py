"""Small checkpointed-scattering acceptance. Author: Athul Prem | _AP_pnc."""

import json
from pathlib import Path
from shutil import copytree
import subprocess
import sys
from tempfile import TemporaryDirectory
from time import monotonic, sleep
import unittest
from unittest.mock import Mock, patch

import numpy as np
import psutil
from scipy.spatial.transform import Rotation

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from phononic import resumable_scattering_AP_pnc as resume


def small_guide():
    config = json.loads((ROOT / "configs" / "straight_beam_AP_pnc.json").read_bytes())
    config["geometry"] = {"length_nm": 1000, "width_nm": 600, "thickness_nm": 200}
    config["mesh"]["cells"] = [4, 2, 1]
    config["material"]["crystal_rotation"] = Rotation.from_euler("xyz", (12, 17, -8), degrees=True).as_matrix().tolist()
    config["analysis"] = {"kind": "scattering", "boundary": "matched_ports", "frequency_GHz": 1, "incident_side": "left"}
    config["resources"].update(linear_solver="ilu_lgmres", ilu_precision="single", assembly_batch_elements=8,
                               lgmres_inner_m=4, lgmres_outer_k=2)
    return config


def pause_after_durable_iterate(phase, attempt, parent_pid, parent_start, deadline):
    from phononic import linear_lgmres_AP_pnc as linear

    original = linear.solve_ilu_lgmres

    def solve(*args, checkpoint_callback, **kwargs):
        def save(event, column, vector, diagnostics):
            checkpoint_callback(event, column, vector, diagnostics)
            if event == "iterate" and column == 1 and diagnostics["iterations"] == 2:
                # The supervisor will observe this actual solver checkpoint and stop the worker.
                sleep(60)

        return original(*args, checkpoint_callback=save, **kwargs)

    with patch.object(linear, "solve_ilu_lgmres", solve):
        resume._worker(phase, Path(attempt), int(parent_pid), float(parent_start), float(deadline))


class ResumableScatteringTests(unittest.TestCase):
    def test_interrupted_phase_awaits_sampled_and_new_descendants(self):
        supervisor = Mock(pid=100, **{"create_time.return_value": 1.0})
        launcher = Mock(pid=101, **{"poll.return_value": None, "wait.return_value": 1})
        sampled_worker = Mock(**{"is_running.return_value": True, "status.return_value": psutil.STATUS_RUNNING})
        new_worker = Mock(**{"is_running.return_value": True, "status.return_value": psutil.STATUS_RUNNING})
        worker_process = Mock(**{"children.side_effect": [[sampled_worker], [new_worker]]})
        for process in (worker_process, sampled_worker):
            process.memory_info.return_value = Mock(rss=1024, private=2048)

        def complete_shutdown(_):
            launcher.wait.assert_called_once()
            for worker in (sampled_worker, new_worker):
                worker.kill.assert_called_once()
                worker.is_running.return_value = False

        with (
            TemporaryDirectory() as directory,
            patch.object(resume.psutil, "Process", side_effect=[supervisor, worker_process, worker_process]),
            patch.object(resume.subprocess, "Popen", return_value=launcher),
            patch.object(resume, "_memory_snapshot", return_value=(1024, 2048)),
            patch.object(resume, "sleep", side_effect=complete_shutdown) as wait_for_exit,
        ):
            result = resume._monitor("solving", Path(directory), monotonic() + 30, lambda: None, lambda: True)

        self.assertEqual(result["status"], "interrupted")
        self.assertTrue(result["worker_stopped"])
        wait_for_exit.assert_called_once()

    def test_watchdog_kills_only_worker_when_supervisor_exits(self):
        def sleeping_process():
            interpreter = getattr(sys, "_base_executable", sys.executable)
            return subprocess.Popen([interpreter, "-c", "import time; time.sleep(20)"])

        supervisor = sleeping_process()
        worker = sleeping_process()
        watchdog = None
        try:
            deadline = monotonic() + 15
            supervisor_proc, worker_proc = psutil.Process(supervisor.pid), psutil.Process(worker.pid)
            watchdog = subprocess.Popen([sys.executable, "-B", "-c", resume._WATCH_PARENT,
                                         str(supervisor.pid), str(supervisor_proc.create_time()),
                                         str(worker.pid), str(worker_proc.create_time()), str(deadline)])
            supervisor.kill()
            supervisor.wait(timeout=5)
            self.assertNotEqual(worker.wait(timeout=5), 0)
            self.assertEqual(watchdog.wait(timeout=5), 0)
            self.assertLess(monotonic(), deadline)
        finally:
            for process in (watchdog, worker, supervisor):
                if process is not None and process.poll() is None:
                    try:
                        for child in psutil.Process(process.pid).children(recursive=True):
                            child.kill()
                    except psutil.NoSuchProcess:
                        pass
                    process.kill()
                    process.wait(timeout=5)

    def test_real_interruption_and_continuation(self):
        popen = subprocess.Popen
        attempt = None

        def checkpoint_ready():
            state = resume._read(attempt / resume.JOURNAL_FILE) if attempt else {}
            active = state.get("active")
            return active is not None and active["column"] == 1 and active["diagnostics"]["iterations"] == 2

        def progress(update):
            nonlocal attempt
            attempt = Path(update["resume_directory"])

        def launch(command, *args, **kwargs):
            if "-m" in command and resume.__name__ in command and "solving" in command:
                command = [sys.executable, "-B", str(Path(__file__).resolve()), "pause", *command[-5:]]
            return popen(command, *args, **kwargs)

        with TemporaryDirectory() as directory:
            output_root = Path(directory)
            config = small_guide()
            with patch.object(resume.subprocess, "Popen", side_effect=launch):
                partial = resume.run_resumable_scattering(config, output_root, wall_seconds=30,
                                                           progress_callback=progress, should_cancel=checkpoint_ready)
            self.assertEqual(partial["status"], "interrupted", partial.get("message"))
            self.assertEqual((partial["accepted_channels"], partial["total_channels"]), (1, 4))
            self.assertTrue(partial["can_resume"])
            journal_path = attempt / resume.JOURNAL_FILE
            original_journal = journal_path.read_bytes()
            journal = json.loads(original_journal)
            self.assertEqual(journal["active"]["column"], 1)
            active = resume._verified(journal["active"]["vector"], attempt.parent)
            self.assertGreater(np.linalg.norm(np.load(active, allow_pickle=False)), 0)

            with resume._exclusive(attempt.parent / "worker.lock"):
                blocked = resume.run_resumable_scattering(resume_from=attempt, wall_seconds=30)
            self.assertEqual(blocked["status"], "blocked")
            self.assertEqual(blocked["resume_directory"], str(attempt))

            damaged_job = attempt.parent.with_name("damaged-copy")
            # Copy only immutable data and the incomplete attempt after the worker exits.
            copytree(attempt.parent, damaged_job, ignore=lambda *_: {"supervisor.lock", "worker.lock"})
            damaged = damaged_job / attempt.name
            vector = damaged_job / journal["accepted"][0]["vector"]["path"]
            with vector.open("ab") as stream:
                stream.write(b"damaged")
            rejected = resume.run_resumable_scattering(resume_from=damaged, wall_seconds=30)
            self.assertEqual(rejected["status"], "failed", rejected.get("message"))
            self.assertFalse(rejected["can_resume"])
            self.assertIn("Checkpoint artifact changed", rejected["message"])

            (attempt.parent / "attempt-000002").mkdir()
            directory_before = set(attempt.parent.iterdir())
            resumed = resume.run_resumable_scattering(resume_from=attempt, wall_seconds=30)
            self.assertEqual(resumed["status"], "completed_unverified", resumed.get("message"))
            resumed_attempt = Path(resumed["resume_directory"])
            self.assertEqual(resumed_attempt.name, "attempt-000003")
            self.assertEqual(set(attempt.parent.iterdir()) - directory_before, {resumed_attempt})
            self.assertEqual(journal_path.read_bytes(), original_journal)
            run = resume._read(resumed_attempt / resume.STATUS_FILE)
            self.assertEqual(set(run["phases"]), {"solving", "exporting"})
            self.assertEqual(resumed["checkpointed_scattering"]["stage"], journal["stage"])
            self.assertEqual(resumed["checkpointed_scattering"]["accepted"][0], journal["accepted"][0])
            html = (Path(resumed["output_directory"]) / "report_AP_pnc.html").read_text(encoding="utf-8")
            self.assertIn("Cached export time (s)", html)
            self.assertIn("Export process memory (MiB)", html)
            channels = resumed["linear_solver"]["channels"]
            self.assertTrue(channels[0]["reused"])
            self.assertEqual(channels[0]["iterations"], 0)
            self.assertTrue(channels[1]["warm_started"])
            self.assertAlmostEqual(channels[1]["relative_residual_history"][0],
                                   journal["active"]["diagnostics"]["relative_residual_history"][-1], places=14)
            self.assertFalse(any(channel.get("warm_started", False) for channel in channels[2:]))

            reference = resume.run_resumable_scattering(config, output_root, wall_seconds=30)
            self.assertEqual(reference["status"], "completed_unverified", reference.get("message"))
            for completed in (resumed, reference):
                self.assertTrue(completed["power_balance_passed"])
                self.assertTrue(completed["matched_reference_passed"])
                self.assertLess(completed["maximum_flux_power_imbalance"], 1e-8)
                self.assertLess(completed["maximum_modal_flux_difference"], 1e-8)
                for channel in completed["linear_solver"]["channels"]:
                    self.assertLessEqual(channel["relative_residual"], 1e-9)
                    self.assertLessEqual(channel["relative_backward_residual"], 1e-9)

            def field_path(result):
                return Path(result["output_directory"]) / "selected_fields_AP_pnc.npz"

            with np.load(field_path(resumed), allow_pickle=False) as actual, np.load(field_path(reference), allow_pickle=False) as expected:
                for name in ("reflection_amplitudes", "transmission_amplitudes"):
                    np.testing.assert_allclose(actual[name], expected[name], atol=1e-8, rtol=0)

    def test_offline_h100_admission(self):
        ledger = json.loads((ROOT / "verification" / "solver_resources_AP_pnc.json").read_bytes())
        h100 = ledger["full_device_h100_lgmres_continuation_fill12"]
        policy, execution = h100["admission"], h100["run"]
        ndof, nnz, size = (policy[key] for key in ("body_dofs", "operator_nonzeros", "operator_storage_bytes"))
        self.assertEqual(size, 20 * nnz + 4 * (ndof + 1))
        channels = policy["remaining_channels"] + policy["reused_channels"]
        manifest = {"shape": [ndof, channels], "nnz": nnz, "operator_storage_bytes": size, "index_bytes": 4}
        options = {"single_precision": policy["factor_input_dtype"] == "complex64",
                   "fill_factor": policy["ilu_settings"]["fill_factor"], **policy["lgmres_settings"]}
        peak = execution["peak_tree_private_commit_mib"]
        partial = resume._solver_budget(manifest, options, 1, peak)
        all_accepted = resume._solver_budget(manifest, options, channels, peak)
        self.assertGreaterEqual(partial["required_commit_mib"], peak + resume.COMMIT_FLOOR_MIB)
        self.assertLess(all_accepted["required_commit_mib"], peak)


if __name__ == "__main__":
    if len(sys.argv) > 1 and sys.argv[1] == "pause":
        pause_after_durable_iterate(*sys.argv[2:])
    else:
        unittest.main()
