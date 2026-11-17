"""Verification of the staged h80 thickness control. Athul Prem | _AP_pnc."""

from pathlib import Path
from tempfile import TemporaryDirectory
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import numpy as np
from scipy.sparse import csc_matrix, save_npz
from skfem import MeshTet

from tools import run_lawrence_thickness_AP_pnc as thickness
from tools import solve_lawrence_thickness_AP_pnc as compute
from phononic.geometry_AP_pnc import BeamGeometry, make_straight_mesh
from phononic.material_AP_pnc import CubicMaterial


class LawrenceThicknessTests(unittest.TestCase):
    def test_footprint_retains_coordinates_and_diagonals_across_thickness_layers(self):
        geometry = BeamGeometry(1e-6, 600e-9, 200e-9)
        reference = make_straight_mesh(geometry, (4, 2, 2))
        candidate = make_straight_mesh(geometry, (4, 2, 3))
        reference_xy = thickness.footprint(reference)
        self.assertEqual(len(reference_xy), 16)
        self.assertEqual(thickness.footprint(candidate), reference_xy)
        points = candidate.p.copy()
        vertex = np.flatnonzero((points[0] > 0) & (points[2] == points[2].min()))[0]
        points[0, vertex] = np.nextafter(points[0, vertex], np.inf)
        self.assertNotEqual(thickness.footprint(MeshTet(points, candidate.t)), reference_xy)

    def test_p2_layout_supplies_a_safe_solve_without_body_matrices(self):
        geometry = BeamGeometry(1e-6, 600e-9, 200e-9)
        mesh = make_straight_mesh(geometry, (4, 2, 1))
        layout = thickness.fixed_port.displacement_layout(mesh)
        port = thickness.assemble_port(SimpleNamespace(mesh=mesh, basis=layout), CubicMaterial(), max_dofs=45)
        modes = thickness.solve_port_modes(port, 1e9)
        self.assertEqual(modes.displacements.shape, (45, 90))
        self.assertLess(np.max(modes.relative_residuals), 1e-8)
        np.testing.assert_array_equal(layout.doflocs[1:, port.left_dofs], port.basis.doflocs)
        np.testing.assert_array_equal(layout.doflocs[1:, port.right_dofs], port.basis.doflocs)

    def test_direct_replay_resumes_the_incident_roots_from_the_approved_stage(self):
        operator = csc_matrix([[4 + 2j, 1], [-1j, 5 - 2j]])
        rhs = np.array([[1j, 2, 5], [-1, 3j, 2j]])
        roots = [21, 12, 17]
        with TemporaryDirectory() as directory:
            root = Path(directory)
            arrays = {"A": root / "A.npz", "rhs": root / "rhs.npy"}
            save_npz(arrays["A"], operator)
            np.save(arrays["rhs"], rhs, allow_pickle=False)
            staged = {"incident_port_roots": roots, "displacement_dofs": 2, "operator_nonzeros": operator.nnz,
                      "array_files": {key: compute.artifact(path) for key, path in arrays.items()},
                      **compute.equation_hashes(operator, rhs)}
            stage_link = {"sha256": "reviewed-stage"}
            identity = {"driver": {"sha256": "reviewed-driver"},
                        "implementation": {key: key for key in thickness.NUMERICAL_KEYS}}
            prior, resumed = root / "prior", root / "resumed"
            prior.mkdir()
            resumed.mkdir()
            original = compute.lawrence.solve_drives

            def finish_prefix_then_interrupt(matrix, drives, *arguments):
                self.assertEqual(original(matrix, drives[:, :2], *arguments), 2)
                raise KeyboardInterrupt

            with (patch.object(compute, "load_stage", return_value=(staged, stage_link)),
                  patch.object(compute, "direct_factor_admission", return_value={"admitted": True, "slurm_node_quota_bytes": 1}),
                  patch.object(compute.lawrence, "cgroup_controls", return_value=[]),
                  patch.object(compute.lawrence, "memory_snapshot", return_value={}),
                  patch.object(compute.lawrence, "memory_block", return_value=None)):
                with patch.object(compute.lawrence, "solve_drives", side_effect=finish_prefix_then_interrupt):
                    with self.assertRaises(KeyboardInterrupt):
                        compute.solve(SimpleNamespace(output=prior, prior_solve_dir=None), dict(identity))
                prior_sha = compute.lawrence.file_sha(prior / "solve_AP_pnc.json")
                args = SimpleNamespace(output=resumed, prior_solve_dir=prior, prior_solve_sha256=prior_sha)
                record = dict(identity)
                compute.solve(args, record)
            self.assertEqual(record["status"], "drives_accepted")
            channels = record["linear_solver"]["channels"]
            self.assertEqual([row["incident_port_root"] for row in channels], roots)
            self.assertEqual([row["incident_port_root"] for row in channels if row.get("reused")], roots[:2])
            self.assertEqual([row["incident_port_root"] for row in record["candidate_history"] if row["event"] == "seed"], roots[2:])
            norm = compute.lawrence.infinity_norm(operator)
            receipt = resumed / "solve_AP_pnc.json"
            replayed = compute.replay_channels(record, receipt, roots, stage_link, operator, rhs, norm, complete=True)
            values = np.column_stack([vector for _, vector in replayed])
            np.testing.assert_allclose(values, np.linalg.solve(operator.toarray(), rhs), rtol=1e-14, atol=1e-14)
            with self.assertRaisesRegex(ValueError, "changed identity or order"):
                list(compute.replay_channels(record, receipt, roots[::-1], stage_link, operator, rhs, norm, complete=True))
            with self.assertRaisesRegex(ValueError, "admitted exact equations"):
                compute.validate_equations(operator, rhs * 2, staged, replay=True)
