"""Saved-equation and checkpoint checks for 100 MHz execution. Athul Prem | _AP_pnc."""

import json
from math import tau
import os
from pathlib import Path
from tempfile import TemporaryDirectory
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import numpy as np
from scipy.sparse import csc_matrix, load_npz, save_npz
from skfem import MeshTet

from tools import run_lawrence_low_band_AP_pnc as low_band
from phononic.port_modes_AP_pnc import PortModes


class LawrenceLowBandTests(unittest.TestCase):
    def test_phase_entry_admits_only_the_current_reviewed_helper(self):
        with TemporaryDirectory() as directory:
            root = Path(directory)
            output = root / "stage"

            def run_stage(args, record):
                self.assertEqual(record["slurm_job_id"], "123")
                self.assertEqual(record["numerical_thread_environment"], low_band.lawrence.THREADS)
                record["status"] = "operator_staged"

            command = ["low-band", "stage", "--body-stage-dir", str(root / "body"),
                       "--safe-dir", str(root / "safe"), "--inventory-dir", str(root / "inventory"),
                       "--output", str(output)]
            artifact = low_band.artifact
            uniform_path = Path(low_band.uniform.__file__).resolve()

            def archived_uniform(path):
                link = artifact(path)
                if path.resolve() == uniform_path:
                    link["sha256"] = low_band.prep.UNIFORM_HELPER_SHA256
                return link

            with (patch.object(low_band, "sys", SimpleNamespace(platform="linux")),
                  patch.dict(os.environ, {**low_band.lawrence.THREADS, "SLURM_JOB_ID": "123"}),
                  patch.object(low_band.lawrence, "solve_environment", return_value={"fixture": "Slurm backend"}),
                  patch.object(low_band, "stage", side_effect=run_stage) as stage_call):
                with patch("sys.argv", command):
                    low_band.main()
                rejected_output = root / "rejected"
                with (patch("sys.argv", command[:-1] + [str(rejected_output)]),
                      patch.object(low_band, "artifact", side_effect=archived_uniform)):
                    with self.assertRaisesRegex(ValueError, "reviewed archive or prepared-equation helper changed"):
                        low_band.main()
                stage_call.assert_called_once()
            receipt = json.loads((output / "stage_AP_pnc.json").read_bytes())
            self.assertEqual(receipt["status"], "operator_staged")
            self.assertEqual(receipt["reviewed_sources"][low_band.uniform.__name__], artifact(uniform_path))
            self.assertEqual(receipt["reviewed_sources"][low_band.uniform.__name__]["sha256"], low_band.UNIFORM_HELPER_SHA)
            self.assertFalse(rejected_output.exists())

    def test_preparation_keeps_archived_inputs_and_writes_only_frequency_equations(self):
        with TemporaryDirectory() as directory:
            root = Path(directory)
            saved_body = root / "archived"
            output = root / "stage"
            saved_body.mkdir()
            output.mkdir()
            stiffness = csc_matrix(np.diag(np.arange(4.0, 12.0)))
            mass = csc_matrix(np.eye(8) / (tau * low_band.FREQUENCY_HZ) ** 2)
            omega = tau * low_band.FREQUENCY_HZ
            traction = 2j / omega
            expected = stiffness.toarray() - np.eye(8) * (1 + traction)
            paths = {key: saved_body / low_band.thickness.MATRIX_FILES[key] for key in ("K", "M", "A")}
            save_npz(paths["K"], stiffness)
            save_npz(paths["M"], mass)
            save_npz(paths["A"], csc_matrix(expected * (1 + 1j)))
            files = {key: low_band.artifact(path) for key, path in paths.items()}
            mtimes = {key: path.stat().st_mtime_ns for key, path in paths.items()}
            body = {"configuration": {"analysis": {"frequency_GHz": 1}}, "operator_csc_bytes": 256,
                    "operator_nonzeros": 8, "array_files": files, "displacement_dofs": 8,
                    **dict.fromkeys(("mesh_directory", "mesh_hash", "mesh", "mesh_summary",
                                     "preflight_receipt", "assembly_resources"))}
            system = SimpleNamespace(K=stiffness, M=mass)
            port = SimpleNamespace(elastic_system=system, left_dofs=np.arange(4), right_dofs=np.arange(4, 8))
            displacements = np.zeros((4, 1950), dtype=complex)
            displacements[:, low_band.ROOTS] = displacements[:, low_band.BACKWARD_ROOTS] = np.eye(4)
            forces = displacements.copy()
            forces[:, low_band.ROOTS] *= traction
            forces[:, low_band.BACKWARD_ROOTS] *= -traction
            modes = SimpleNamespace(port=port, frequency_hz=low_band.FREQUENCY_HZ,
                                    positive_indices=np.array(low_band.ROOTS), negative_indices=np.array(low_band.BACKWARD_ROOTS),
                                    displacements=displacements, axial_forces=forces,
                                    propagating=np.ones(1950, dtype=bool))
            args = SimpleNamespace(output=output, body_stage_dir=saved_body, safe_dir=saved_body, inventory_dir=saved_body)
            record = {"implementation": {}}
            with (patch.object(low_band, "read_body_stage", return_value=(body, {})),
                  patch.object(low_band, "admit_workspace", return_value={}),
                  patch.object(low_band, "read_inventory", return_value=(port, {})),
                  patch.object(low_band, "restore_system", return_value=(system, modes, {})),
                  patch.object(low_band.workflow, "resolve_config", side_effect=lambda value: json.loads(json.dumps(value))),
                  patch("phononic.scattering_AP_pnc.splu", side_effect=AssertionError("body factorization requested"))):
                low_band.stage(args, record)
            self.assertEqual(record["status"], "operator_staged")
            self.assertEqual(record["configuration"]["analysis"]["frequency_GHz"], 0.1)
            self.assertEqual(body["configuration"]["analysis"]["frequency_GHz"], 1)
            for key, path in paths.items():
                self.assertEqual(low_band.artifact(path), files[key])
                self.assertEqual(path.stat().st_mtime_ns, mtimes[key])
            self.assertEqual({path.name for path in output.iterdir()}, {"A_AP_pnc.npz", "rhs_AP_pnc.npy", "stage_AP_pnc.json"})
            np.testing.assert_allclose(load_npz(output / "A_AP_pnc.npz").toarray(), expected, rtol=0, atol=2e-15)
            rhs = np.load(output / "rhs_AP_pnc.npy", allow_pickle=False)
            np.testing.assert_allclose(rhs[:4], -2 * traction * np.eye(4), rtol=0, atol=1e-22)
            np.testing.assert_array_equal(rhs[4:], 0)
            exposure = record["factor_index_exposure"]
            self.assertTrue(exposure["matches_archived_1_ghz_csc_graph"])
            self.assertEqual(exposure["initial_factor_slots_per_store"], 30 * 8)
            self.assertEqual(exposure["archived_aggregate_report_headroom"], 300_551_043)
            self.assertFalse(exposure["new_factor_index_capacity_established"])
            self.assertTrue(all(record[name] is False for name in ("mesh_generated", "body_assembly_performed",
                                                                   "safe_eigensolve_performed", "factorization_performed")))

    def test_prepared_graph_inspection_checks_structure_and_width(self):
        with TemporaryDirectory() as directory:
            before = Path(directory) / "before.npz"
            matrix = csc_matrix(np.eye(2, dtype=complex))
            save_npz(before, csc_matrix([[1j, 1], [0, 0]]))
            exposure = low_band.factor_index_exposure(matrix, before, 2)
            self.assertFalse(exposure["matches_archived_1_ghz_csc_graph"])
            matrix.indptr = matrix.indptr.astype(np.int64)
            with self.assertRaisesRegex(ValueError, "int32 interface"):
                low_band.factor_index_exposure(matrix, before, 2)

    def test_factor_count_refusal_checkpoint_and_resume(self):
        with TemporaryDirectory() as directory:
            root = Path(directory)
            matrix = csc_matrix(np.diag([2.0 + 2j, 5.0 - 1j]))
            rhs = np.array([[1j, 2, 5, 1], [-1, 3j, 2j, 2]])
            paths = {"A": root / "A.npz", "rhs": root / "rhs.npy"}
            save_npz(paths["A"], matrix)
            np.save(paths["rhs"], rhs, allow_pickle=False)
            staged = {"incident_port_roots": list(low_band.ROOTS), "displacement_dofs": 2,
                      "operator_nonzeros": matrix.nnz, "array_files": {key: low_band.artifact(path) for key, path in paths.items()},
                      "factor_index_exposure": low_band.factor_index_exposure(matrix, paths["A"], matrix.nnz),
                      **low_band.equations.equation_hashes(matrix, rhs)}
            stage_link = {"sha256": "reviewed-exact-stage"}
            identity = {"driver": {"sha256": "reviewed-driver"}, "superlu_library": {"sha256": "reviewed-binary"},
                        "reviewed_sources": {}, "implementation": {key: key for key in low_band.prep.NUMERICAL_KEYS}}

            def drive_fixture(limit, factor_count):
                def drives(operator, columns, seed, norm, publish, admit):
                    admit()
                    publish("factor", {"factor_seconds": 0, "factor_nonzeros": factor_count})
                    for column, b in enumerate(columns.T[:limit]):
                        vector = b / operator.diagonal()
                        metrics = low_band.lawrence.residual_metrics(b - operator @ vector, b, vector, norm)
                        publish("candidate", {"column": column, "update": 1, "target_met": True, **metrics}, vector)
                    if limit < columns.shape[1]:
                        raise KeyboardInterrupt()
                    return columns.shape[1]
                return drives

            refused, prior, resumed, reused = (root / name for name in ("refused", "prior", "resumed", "reused"))
            for output in (refused, prior, resumed, reused):
                output.mkdir()
            with (patch.object(low_band, "load_stage", return_value=(staged, stage_link)),
                  patch.object(low_band, "admit_workspace", return_value={}),
                  patch.object(low_band, "direct_factor_admission", return_value={"admitted": True, "slurm_node_quota_bytes": 1}),
                  patch.object(low_band.lawrence, "cgroup_controls", return_value=[]),
                  patch.object(low_band.lawrence, "memory_snapshot", return_value={}),
                  patch.object(low_band.lawrence, "memory_block", return_value=None)):
                with patch.object(low_band.lawrence, "solve_drives", side_effect=drive_fixture(4, -1)):
                    with self.assertRaisesRegex(RuntimeError, "invalid signed aggregate factor count"):
                        low_band.solve(SimpleNamespace(output=refused, prior_solve_dir=None), dict(identity))
                refused_record = json.loads((refused / "solve_AP_pnc.json").read_bytes())
                self.assertEqual(refused_record["linear_solver"]["channels"], [])
                self.assertEqual(list(refused.glob("accepted-*.npy")), [])
                with patch.object(low_band.lawrence, "solve_drives", side_effect=drive_fixture(2, 4)):
                    with self.assertRaises(KeyboardInterrupt):
                        low_band.solve(SimpleNamespace(output=prior, prior_solve_dir=None), dict(identity))
                prior_path = prior / "solve_AP_pnc.json"
                prior_bytes = prior_path.read_bytes()
                prior_time = prior_path.stat().st_mtime_ns
                args = SimpleNamespace(output=resumed, prior_solve_dir=prior, prior_solve_sha256=low_band.lawrence.file_sha(prior_path))
                record = dict(identity)
                with patch.object(low_band.lawrence, "solve_drives", side_effect=drive_fixture(2, 4)):
                    low_band.solve(args, record)
                complete_path = resumed / "solve_AP_pnc.json"
                low_band.workflow.write_json(complete_path, record)
                complete_link = low_band.artifact(complete_path)
                reuse = SimpleNamespace(output=reused, prior_solve_dir=resumed, prior_solve_sha256=complete_link["sha256"])
                reuse_record = dict(identity)
                with patch.object(low_band.lawrence, "solve_drives", side_effect=AssertionError("body factorization requested")):
                    low_band.solve(reuse, reuse_record)
            self.assertEqual(record["status"], "drives_accepted")
            channels = record["linear_solver"]["channels"]
            self.assertEqual([row["incident_port_root"] for row in channels], list(low_band.ROOTS))
            self.assertEqual([row["column"] for row in channels if row.get("reused")], [0, 1])
            self.assertEqual(prior_path.read_bytes(), prior_bytes)
            self.assertEqual(prior_path.stat().st_mtime_ns, prior_time)
            for column, row in enumerate(channels):
                vector = np.load(resumed / row["accepted_vector"]["file"], allow_pickle=False)
                np.testing.assert_allclose(matrix @ vector, rhs[:, column], rtol=1e-14, atol=1e-14)
            self.assertEqual(reuse_record["status"], "drives_accepted")
            reused_report = reuse_record["linear_solver"]
            self.assertEqual(reused_report["factor"], record["linear_solver"]["factor"])
            self.assertEqual(reused_report["factor_reused_from"], complete_link)
            self.assertTrue(all(row["reused"] for row in reused_report["channels"]))

    def test_cached_export_replays_exact_equations_and_links_all_fields(self):
        with TemporaryDirectory() as directory:
            root = Path(directory)
            mesh = MeshTet(np.array([[0, 0, low_band.END_PLANES_M[1], low_band.END_PLANES_M[1]],
                                     [0, 1e-6, 0, 0], [0, 0, 0, 2e-7]]), np.arange(4).reshape(4, 1))
            layout = low_band.prep.fixed_port.displacement_layout(mesh)
            left = layout.nodal_dofs[:, :2].ravel()[:4]
            right = layout.nodal_dofs[:, 2:].ravel()[:4]
            omega = tau * low_band.FREQUENCY_HZ
            scale = 2 / omega
            # Each matched pair has D = [0 -c; -c 0], with the exact traces [1, i].
            dynamic = np.eye(layout.N)
            dynamic[np.r_[left, right], np.r_[left, right]] = 0
            dynamic[left, right] = dynamic[right, left] = -scale
            mass = np.eye(layout.N) * (scale / (omega * omega))
            system = low_band.ElasticSystem(mesh, layout, csc_matrix(dynamic + scale * np.eye(layout.N)), csc_matrix(mass))
            port = SimpleNamespace(elastic_system=system, basis=SimpleNamespace(N=4), left_dofs=left, right_dofs=right)
            positive, negative = np.array(low_band.ROOTS), np.array(low_band.BACKWARD_ROOTS)
            displacement = np.zeros((4, 1950), dtype=complex)
            displacement[:, positive] = displacement[:, negative] = np.eye(4)
            forces = np.zeros_like(displacement)
            forces[:, positive], forces[:, negative] = 1j * scale * np.eye(4), -1j * scale * np.eye(4)
            powers, wavenumbers = np.zeros(1950), np.zeros(1950, dtype=complex)
            powers[positive], powers[negative] = 1, -1
            wavenumbers[positive], wavenumbers[negative] = np.arange(1, 5), -np.arange(1, 5)
            modes = PortModes(port, low_band.FREQUENCY_HZ, wavenumbers, displacement, forces, positive, negative,
                              powers != 0, powers, np.zeros(1950))
            vectors = np.zeros((layout.N, 4), dtype=complex)
            vectors[left], vectors[right] = np.eye(4), 1j * np.eye(4)
            report = {"method": low_band.thickness.METHOD, "channels": []}
            stage_link = {"sha256": "fixture-stage"}

            def capture(operator, rhs):
                norm = low_band.lawrence.infinity_norm(operator)
                for column, incident in enumerate(positive):
                    vector = vectors[:, column]
                    name = f"accepted-{column:03d}_AP_pnc.npy"
                    metrics = low_band.lawrence.residual_metrics(rhs[:, column] - operator @ vector, rhs[:, column], vector, norm)
                    self.assertTrue(low_band.lawrence.accepted(metrics))
                    report["channels"].append({"column": column, "incident_port_root": int(incident), "info": 0,
                                               "accepted_vector": low_band.lawrence.write_vector(root / name, vector), **metrics})
                staged.update(operator_nonzeros=operator.nnz, **low_band.equations.equation_hashes(operator, rhs))
                raise low_band.equations.OperatorCaptured()

            config = json.loads((low_band.ROOT / "configs/perforated_layered_AP_pnc.json").read_text())
            config["analysis"]["frequency_GHz"] = 0.1
            phase = dict.fromkeys(low_band.thickness.EXECUTION_KEYS, {})
            phase.update(superlu_library={}, reviewed_sources={})
            staged = {**phase, "mesh_directory": str(root), "displacement_dofs": layout.N,
                      "incident_port_roots": list(low_band.ROOTS), "configuration": config, "assembly_resources": {},
                      "reference_planes_m": list(low_band.END_PLANES_M), "safe_modes": {"bytes": 1},
                      "array_files": {"rhs": {"bytes": vectors.nbytes}},
                      **dict.fromkeys(("body_stage", "inventory_inputs", "inventory_point", "open_channels", "safe_section",
                                       "safe_receipt", "trace_sha256", "cutoff_status", "degeneracy_status"), {})}
            solved = {**phase, "stage": stage_link, "linear_solver": report, "status": "drives_accepted"}
            solve_link = {"path": str(root / "solve_AP_pnc.json"), "sha256": "fixture-solve"}
            mesh_record = {"ndof": layout.N, "mesh_hash": "mesh-hash", "mesh_sha256": "mesh-sha",
                           "mesh_vertices": 4, "mesh_tetrahedra": 1, "end_face_dofs": {"xleft": 4, "xright": 4},
                           "boundary_facet_counts": {},
                           "hole_pattern_provenance": {"source_sha256": "fixture", "x_offset_m": 0, "holes": [None] * 26}}
            with self.assertRaises(low_band.equations.OperatorCaptured):
                low_band.solve_scattering(system, modes, low_band.FREQUENCY_HZ, linear_solver=capture)
            with (patch.object(low_band, "load_stage", return_value=(staged, stage_link)),
                  patch.object(low_band, "admit_workspace", return_value={}),
                  patch.object(low_band.thickness, "read_phase", return_value=(solved, solve_link)),
                  patch.object(low_band, "restore_system", return_value=(system, modes, mesh_record)),
                  patch("phononic.scattering_AP_pnc.splu", side_effect=AssertionError("body factorization requested"))):
                output = root / "export"
                args = SimpleNamespace(output=output, solve_dir=root, solve_sha256="fixture-solve")
                record = dict(phase)
                low_band.export(args, record)
                for key in ("operator_csc_sha256", "rhs_array_sha256"):
                    args.output = root / key
                    verified = staged[key]
                    staged[key] = "changed"
                    with self.assertRaisesRegex(ValueError, "reconstructed operator or drives differ"):
                        low_band.export(args, dict(phase))
                    staged[key] = verified
            summary = json.loads(Path(record["public_summary"]["path"]).read_bytes())
            self.assertEqual(record["status"], "completed_unverified")
            self.assertTrue(summary["power_balance_passed"])
            for key in ("max_power_imbalance", "maximum_flux_power_imbalance",
                        "maximum_net_outward_power_ratio", "maximum_modal_flux_difference"):
                self.assertLess(summary[key], 1e-12)
            provenance = summary["checkpointed_scattering"]
            self.assertEqual(provenance["prepared_frequency_stage"], stage_link)
            self.assertEqual(provenance["solve_receipt"], solve_link)
            self.assertEqual(provenance["reference_planes_m"], list(low_band.END_PLANES_M))
            self.assertTrue(all(provenance[name] is False for name in low_band.EXPLORATORY))
            self.assertEqual(set(record["public_fields"]), {"selected_fields_AP_pnc.npz", "selected_vertex_shapes_AP_pnc.vtu"})
            for name, link in record["public_fields"].items():
                self.assertEqual(link["sha256"], summary["output_sha256"][name])
            with np.load(record["public_fields"]["selected_fields_AP_pnc.npz"]["path"], allow_pickle=False) as fields:
                np.testing.assert_array_equal(fields["displacements_per_sqrt_w"], vectors)
                np.testing.assert_array_equal(fields["incident_mode_indices"], positive)
                np.testing.assert_allclose(fields["reflection_amplitudes"], 0, atol=1e-15)
                np.testing.assert_allclose(fields["transmission_amplitudes"], 1j * np.eye(4), atol=1e-15)


if __name__ == "__main__":
    unittest.main()
