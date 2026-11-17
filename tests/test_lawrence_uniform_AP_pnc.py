"""Direct uniform-guide solve checks. Author: Athul Prem | _AP_pnc."""

from contextlib import redirect_stderr, redirect_stdout
from io import StringIO
from pathlib import Path
from tempfile import TemporaryDirectory
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import numpy as np
from scipy.sparse import csc_matrix, load_npz

from tools import run_lawrence_uniform_h80_AP_pnc as runner


class LawrenceUniformTests(unittest.TestCase):
    def test_preflight_keeps_h80_default_and_routes_explicit_h60_separately(self):
        identity = {"source_sha256": {"test": "source"}}
        with TemporaryDirectory() as directory:
            root = Path(directory)
            for selection, size in (([], 80), (["--mesh-size-nm", "60"], 60)):
                with self.subTest(mesh_size_nm=size):
                    output = root / str(size)
                    output.mkdir()
                    reference = SimpleNamespace(config={"mesh": {"mesh_size_nm": size}},
                                                provenance={"safe_eigensolve_performed": False})
                    arguments = ["uniform", "preflight", "--h100", str(root / "h100"), "--ports", str(root / "ports.npz"),
                                 "--output", str(output), *selection]
                    with (patch("sys.argv", arguments), redirect_stdout(StringIO()),
                          patch.object(runner.workflow, "source_identity", return_value=identity),
                          patch.object(runner, "SOURCE_MAP_SHA", runner.workflow.digest(identity["source_sha256"])),
                          patch.object(runner, "read_reference", return_value=reference) as load_reference,
                          patch.object(runner, "preflight", return_value={"status": "mesh_ready", "case_max_dofs": 42})):
                        runner.main()
                    parameters = load_reference.call_args.kwargs
                    self.assertEqual(parameters.get("mesh_size_nm", 80), size)
                    receipt = runner.read_json(output / f"uniform_h{size}_AP_pnc.json")
                    self.assertEqual(receipt["status"], "mesh_ready")
                    self.assertEqual(receipt["configuration"], reference.config)
                    self.assertEqual(receipt["cached_port_provenance"], reference.provenance)

    def test_h60_rejects_the_legacy_h80_preflight_exception_before_assembly(self):
        config = {"mesh": {"mesh_size_nm": 60}}
        reference = SimpleNamespace(config=config, provenance={"safe_eigensolve_performed": False})
        identity = {"source_sha256": {"test": "source"}, "packages": {}, "python": "3.12.12"}
        execution = {"driver": {"sha256": "new-driver"}, "implementation": identity}
        ready = {"status": "mesh_ready", "configuration": config, "cached_port_provenance": reference.provenance,
                 "driver": {"sha256": runner.LEGACY_PREFLIGHT_DRIVER_SHA}, "implementation": identity}
        with TemporaryDirectory() as directory:
            output = Path(directory)
            path = output / "uniform_h60_AP_pnc.json"
            runner.workflow.write_json(path, ready)
            # Even an exact approved legacy digest cannot authorize the new target.
            with (patch.object(runner, "LEGACY_PREFLIGHT_RECEIPT_SHA", runner.artifact(path)["sha256"]),
                  patch.object(runner.workflow, "_assemble_system") as assemble):
                with self.assertRaisesRegex(ValueError, "independent h60 preflight"):
                    runner.solve(reference, output, output, 42, execution, direct=True)
                assemble.assert_not_called()

    def test_cli_refuses_unsupported_h60_iterative_solve(self):
        arguments = ["uniform", "solve", "--mesh-size-nm", "60", "--h100", "h100", "--ports", "ports.npz",
                     "--output", "solve", "--preflight-dir", "preflight", "--max-dofs", "42"]
        with (patch("sys.argv", arguments), redirect_stderr(StringIO()) as errors,
              patch.object(runner, "read_reference") as load_reference):
            with self.assertRaises(SystemExit) as exit_status:
                runner.main()
        self.assertEqual(exit_status.exception.code, 2)
        self.assertIn("solve-direct", errors.getvalue())
        load_reference.assert_not_called()

    def test_five_direct_drives_match_dense_reference_and_save_checked_vectors(self):
        dense = np.array([[3 + 1j, 2j, 1], [1 - 2j, 4, 0], [2, 1j, 5 - 1j]])
        matrix = csc_matrix(dense)
        random = np.random.default_rng(8721)
        rhs = random.normal(size=(3, 5)) + 1j * random.normal(size=(3, 5))
        original_rhs = rhs.copy()
        roots = [1348, 1311, 1292, 1284, 1274]
        with TemporaryDirectory() as directory, patch.object(runner, "direct_factor_admission", return_value={"admitted": True}):
            output = Path(directory)
            result = runner.solve_direct_original(matrix, rhs, roots, output, {})
            np.testing.assert_allclose(result, np.linalg.solve(dense, original_rhs), rtol=1e-14, atol=1e-14)
            np.testing.assert_array_equal(matrix.toarray(), dense)
            np.testing.assert_array_equal(rhs, original_rhs)
            receipt = runner.read_json(output / "direct_channels_AP_pnc.json")
            self.assertEqual(receipt["status"], "drives_accepted")
            channels = receipt["solver_report"]["channels"]
            self.assertEqual([row["incident_port_root"] for row in channels], roots)
            for column, row in enumerate(channels):
                saved = output / row["accepted_vector"]["file"]
                self.assertEqual(runner.artifact(saved)["sha256"], row["accepted_vector"]["sha256"])
                np.testing.assert_array_equal(np.load(saved, allow_pickle=False), result[:, column])

    def test_relative_residual_failure_retains_the_first_drive(self):
        # The first drive acts in an independent coordinate. The second loses
        # precision when two nearly equal equations cancel in the solve.
        matrix = csc_matrix([[1, 1, 0], [1, 1 + 1e-13, 0], [0, 0, 1]], dtype=np.complex128)
        rhs = np.array([[0, 0.3], [0, 0.7], [1 - 2j, 0]], dtype=np.complex128)
        with TemporaryDirectory() as directory, patch.object(runner, "direct_factor_admission", return_value={"admitted": True}):
            output = Path(directory)
            with self.assertRaisesRegex(RuntimeError, "Direct drive 2 failed"):
                runner.solve_direct_original(matrix, rhs, [1348, 1311], output, {})
            receipt = runner.read_json(output / "direct_channels_AP_pnc.json")
            self.assertEqual(receipt["status"], "failed")
            first, second = receipt["solver_report"]["channels"]
            self.assertGreater(second["relative_residual"], 1e-9)
            self.assertLessEqual(second["relative_backward_residual"], 1e-9)
            files = list(output.glob("*.npy"))
            self.assertEqual([path.name for path in files], [first["accepted_vector"]["file"]])
            np.testing.assert_array_equal(np.load(files[0], allow_pickle=False), rhs[:, 0])

    def test_unsafe_admission_precedes_factorization(self):
        matrix = csc_matrix([[1, 0], [0, 0]], dtype=np.complex128)
        rhs = np.ones((2, 1), dtype=np.complex128)
        system = SimpleNamespace(K=csc_matrix([[2.0, 0], [0, 1.0]]), M=csc_matrix(np.eye(2)))

        with (TemporaryDirectory() as directory,
              patch.object(runner, "disk_usage", return_value=SimpleNamespace(free=2**31)),
              patch.object(runner, "direct_factor_admission", return_value={"admitted": False}),
              patch.object(runner, "splu") as factor):
            output = Path(directory)
            with self.assertRaisesRegex(RuntimeError, "does not fit the current direct factorization budget"):
                runner.solve_direct_original(matrix, rhs, [1348], output, {}, system=system)
            factor.assert_not_called()
            receipt = runner.read_json(output / "direct_channels_AP_pnc.json")
            self.assertEqual(receipt["status"], "failed")
            self.assertNotIn("factor", receipt["solver_report"])
            self.assertEqual(list(output.glob("accepted-*.npy")), [])
            for name, expected in (("K", system.K), ("M", system.M), ("A", matrix)):
                path = output / f"{name}_AP_pnc.npz"
                np.testing.assert_array_equal(load_npz(path).toarray(), expected.toarray())
                self.assertEqual(receipt["solver_report"]["array_files"][name], runner.artifact(path))
            np.testing.assert_array_equal(np.load(output / "rhs_AP_pnc.npy", allow_pickle=False), rhs)
