"""Fixed-device stage and direct replay checks. Author: Athul Prem | _AP_pnc."""

import json
from math import tau
from pathlib import Path
from tempfile import TemporaryDirectory
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import numpy as np
from scipy.sparse import csc_matrix, csr_matrix, load_npz, save_npz
from scipy.sparse.linalg import splu

from tools import solve_lawrence_device_refinement_AP_pnc as driver


class LawrenceDeviceTests(unittest.TestCase):
    def setUp(self):
        scratch = TemporaryDirectory()
        self.addCleanup(scratch.cleanup)
        self.root = Path(scratch.name)
        self.stage = self.root / "stage"
        self.stage.mkdir()
        self.matrix = csc_matrix([[3 + 1j, 2j, 1], [1 - 2j, 4, 0], [2, 1j, 5 - 1j]])
        values = np.arange(1, 16, dtype=float).reshape(3, 5)
        self.rhs = values + 1j * values[::-1] / 3
        self.execution = {"driver": {"sha256": "driver"}, "direct_solver_helpers": {"sha256": "solver"},
                          "implementation": {key: key for key in driver.NUMERICAL_KEYS}}
        files = {name: self.stage / driver.ARRAY_FILES[name] for name in ("A", "rhs")}
        save_npz(files["A"], self.matrix, compressed=False)
        np.save(files["rhs"], self.rhs, allow_pickle=False)
        self.manifest = {**self.execution, "status": "operator_staged", "incident_port_roots": driver.ROOTS,
                         "displacement_dofs": self.matrix.shape[0], "operator_nonzeros": self.matrix.nnz,
                         "safe_eigensolve_performed": False, "factorization_performed": False,
                         "array_files": {name: driver.artifact(path) for name, path in files.items()},
                         **driver.equation_hashes(self.matrix, self.rhs)}
        self.manifest_path = self.stage / "stage_AP_pnc.json"
        driver.workflow.write_json(self.manifest_path, self.manifest)
        for name, replacement in (("cgroup_controls", []), ("memory_snapshot", {}), ("memory_block", None)):
            mocked = patch.object(driver.lawrence, name, return_value=replacement)
            mocked.start()
            self.addCleanup(mocked.stop)
        mocked = patch.object(driver, "direct_factor_admission", return_value={"admitted": True, "slurm_node_quota_bytes": 1})
        mocked.start()
        self.addCleanup(mocked.stop)

    def args(self, name, prior=None):
        output = self.root / name
        output.mkdir()
        return SimpleNamespace(output=output, stage_dir=self.stage, stage_sha256=driver.lawrence.file_sha(self.manifest_path),
                               prior_solve_dir=prior,
                               prior_solve_sha256=driver.lawrence.file_sha(prior / "solve_AP_pnc.json") if prior else None)

    def test_direct_replay_preserves_equations_and_skips_accepted_prefix(self):
        prior = self.args("prior")
        original = driver.lawrence.solve_drives

        def interrupt_after_two(operator, rhs, *args):
            self.assertEqual(original(operator, rhs[:, :2], *args), 2)
            raise KeyboardInterrupt

        with patch.object(driver.lawrence, "solve_drives", side_effect=interrupt_after_two):
            with self.assertRaises(KeyboardInterrupt):
                driver.solve(prior, dict(self.execution))
        prior_path = prior.output / "solve_AP_pnc.json"
        receipt = json.loads(prior_path.read_bytes())
        self.assertEqual([row["column"] for row in receipt["linear_solver"]["channels"]], [0, 1])
        resumed = self.args("resumed", prior.output)
        record = dict(self.execution)
        with patch("scipy.sparse.linalg.splu", wraps=splu) as factor:
            driver.solve(resumed, record)
        self.assertEqual(factor.call_count, 1)
        self.assertEqual(record["status"], "drives_accepted")
        self.assertEqual([row["column"] for row in record["candidate_history"] if row["event"] == "seed"], [2, 3, 4])
        self.assertEqual([row["column"] for row in record["linear_solver"]["channels"] if row.get("reused")], [0, 1])

        result = np.linalg.solve(self.matrix.toarray(), self.rhs)
        norm = driver.lawrence.infinity_norm(self.matrix)
        receipt_path = resumed.output / "solve_AP_pnc.json"
        driver.workflow.write_json(receipt_path, record)
        link = driver.artifact(receipt_path)
        stage_link = driver.artifact(self.manifest_path)
        replayed = list(driver.replay_channels((record, link), stage_link, self.matrix, self.rhs, norm, complete=True))
        np.testing.assert_allclose(np.column_stack([vector for _, _, vector in replayed]), result, rtol=1e-13, atol=1e-13)
        self.assertEqual(driver.equation_hashes(self.matrix, self.rhs),
                         {name: self.manifest[name] for name in ("operator_csc_sha256", "rhs_array_sha256")})
        accepted = resumed.output / "accepted-000_AP_pnc.npy"
        replacement = driver.lawrence.write_vector(accepted, result[:, 0] + 1e-6)
        record["linear_solver"]["channels"][0]["accepted_vector"] = replacement
        with self.assertRaisesRegex(ValueError, "original-equation replay"):
            list(driver.replay_channels((record, link), stage_link, self.matrix, self.rhs, norm, complete=True))

    def test_staging_checks_refuse_corrupt_or_changed_equations(self):
        args = self.args("rejected")
        rhs_path = self.stage / driver.ARRAY_FILES["rhs"]
        np.save(rhs_path, self.rhs * 2, allow_pickle=False)
        with patch("scipy.sparse.linalg.splu") as factor:
            with self.assertRaisesRegex(ValueError, "Saved h60 array changed: rhs"):
                driver.solve(args, dict(self.execution))
            factor.assert_not_called()
        with self.assertRaisesRegex(ValueError, "exact staged h60 equations"):
            driver.check_equations(self.matrix, self.rhs * 2, self.manifest)

    def test_scattering_callback_rebuilds_the_captured_equations(self):
        channels = len(driver.ROOTS)
        section = np.eye(channels)
        stiffness = csr_matrix(np.block([[6 * section, -section], [-section, 6 * section]]))
        mass = csr_matrix(np.eye(2 * channels))
        stage = {}
        frequency = 0.125

        def scattering(k, m, callback):
            system = driver.ElasticSystem(None, None, k, m)
            port = SimpleNamespace(elastic_system=system, left_dofs=np.arange(channels), right_dofs=np.arange(channels, 2 * channels))
            modes = driver.PortModes(port, frequency, np.r_[np.ones(channels), -np.ones(channels)],
                                     np.hstack((section, section)), np.hstack((1j * section, -1j * section)),
                                     np.arange(channels), np.arange(channels, 2 * channels),
                                     np.ones(2 * channels, dtype=bool),
                                     tau * frequency * np.r_[np.ones(channels), -np.ones(channels)] / 2,
                                     np.zeros(2 * channels))
            return driver.solve_scattering(system, modes, frequency, linear_solver=callback)

        def capture(operator, rhs):
            stage.update(displacement_dofs=2 * channels, operator_nonzeros=operator.nnz, **driver.equation_hashes(operator, rhs))
            raise driver.OperatorCaptured

        with self.assertRaises(driver.OperatorCaptured):
            scattering(stiffness, mass, capture)
        k_path, m_path = (self.root / f"{name}.npz" for name in ("K", "M"))
        save_npz(k_path, stiffness)
        save_npz(m_path, mass)

        def restore(operator, rhs):
            driver.check_equations(operator, rhs, stage)
            return np.linalg.solve(operator.toarray(), rhs)

        result = scattering(load_npz(k_path), load_npz(m_path), restore)
        np.testing.assert_allclose(result.reflection + result.transmission, 1, rtol=0, atol=1e-14)
        np.testing.assert_allclose(result.flux_power_balance, 0, rtol=0, atol=1e-14)
