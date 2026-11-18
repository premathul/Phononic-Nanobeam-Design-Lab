"""Saved-equation and checkpoint checks for 887.5 MHz execution. Athul Prem | _AP_pnc."""

import json
from copy import deepcopy
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

from tools import run_lawrence_frequency_point_AP_pnc as frequency_point
from phononic.port_modes_AP_pnc import PortModes


class LawrenceFrequencyPointTests(unittest.TestCase):
    def test_interval_preserves_distinct_identities_and_rejects_changed_inputs(self):
        root = Path("fixture").resolve()
        source = {"physical.py": "unchanged"}
        original = {"source_sha256": source, "packages": {"numpy": "fixed"}, "python": "3.12.12", "platform": "Linux-fixture"}
        producer = {**original, "python": "3.12.14", "platform": "Windows-fixture"}
        configuration = dict.fromkeys(("geometry", "mesh", "material", "analysis"), {})
        safe_link = {"path": str(root / frequency_point.inventory.SAFE), "sha256": "safe", "bytes": 1}
        section_link = {"path": str(root / frequency_point.inventory.PORTS), "sha256": "section", "bytes": 2}
        modes_link = {"path": str(root / f"modes-{frequency_point.FREQUENCY_HZ}_AP_pnc.npz"),
                      "sha256": frequency_point.INTERVAL_MODES_SHA, "bytes": 3}
        port = SimpleNamespace(basis=SimpleNamespace(mesh=SimpleNamespace(p=None, t=None), doflocs=None),
                               **dict.fromkeys(("left_dofs", "right_dofs", "K0", "M", "Q", "G")))
        archive = {"original_implementation": original, "current_implementation": producer,
                   "section_array_sha256": "section-arrays", "trace_sha256": "trace",
                   "section_modes": section_link, "safe_receipt": safe_link}
        identity = {"archive": archive, "producer": {"sha256": frequency_point.INTERVAL_PRODUCER_SHA}}
        inputs = {"identity": identity, "identity_sha256": frequency_point.workflow.digest(identity)}
        metrics = {"discrete_roots": 1950, "positive_open_channels": 5, "negative_open_channels": 5,
                   "cutoff_status": "solver_resolved_at_sample", "degeneracy_status": "none_detected",
                   "coincident_propagating_root_groups": [],
                   "open_channels": [{"root_id": root_id, "direction": side}
                                     for side, roots in (("positive", frequency_point.ROOTS), ("negative", frequency_point.BACKWARD_ROOTS))
                                     for root_id in roots]}
        point = {"frequency_hz": frequency_point.FREQUENCY_HZ, "status": "accepted",
                 "identity_sha256": inputs["identity_sha256"], "metrics": metrics,
                 "actual_environment": "inputs_AP_pnc.json:identity.archive.current_implementation",
                 "modes": {**modes_link, "path": Path(modes_link["path"]).name},
                 "execution_provenance": {"git_commit": "producer-revision"}}
        point["point_sha256"] = frequency_point.workflow.digest(point)
        ledger = {"status": "independently_verified_finite_count_interval", "inputs": inputs, "points": [point],
                  "state": {"status": "complete", "lower_hz": 831_250_000, "upper_hz": frequency_point.FREQUENCY_HZ}}
        body = {"implementation": original, "safe_modes": section_link, "safe_receipt": safe_link,
                "trace_sha256": "trace", "configuration": configuration}
        records = {"ledger.json": ledger, "inputs_AP_pnc.json": inputs,
                   f"point-{frequency_point.FREQUENCY_HZ}_AP_pnc.json": point}
        artifacts = {Path(link["path"]).name: link for link in (safe_link, section_link, modes_link)}

        def read_fixture(path, sha):
            expected = {"ledger.json": frequency_point.INTERVAL_LEDGER_SHA,
                        "inputs_AP_pnc.json": frequency_point.INTERVAL_INPUTS_SHA,
                        f"point-{frequency_point.FREQUENCY_HZ}_AP_pnc.json": frequency_point.INTERVAL_POINT_SHA}
            self.assertEqual(sha, expected[path.name])
            return deepcopy(records[path.name]), {"path": str(path), "sha256": sha}

        with (patch.object(frequency_point.equations, "read_receipt", side_effect=read_fixture),
              patch.object(frequency_point, "artifact", side_effect=lambda path: deepcopy(artifacts[path.name])),
              patch.object(frequency_point, "_array_digest", return_value="section-arrays"),
              patch.object(frequency_point.inventory, "read_archive", return_value=(port, None, {"trace_sha256": "trace", "configuration": configuration})) as read_archive):
            actual_port, section = frequency_point.read_interval(root, root / "ledger.json", root, body, {"implementation": original})
            self.assertIs(actual_port, port)
            self.assertEqual(section["section_mode_implementation"], producer)
            self.assertEqual(section["original_section_implementation"], original)
            self.assertNotEqual(section["section_mode_implementation"]["python"], original["python"])
            self.assertEqual(section["cross_frequency_branch_identity"], "unresolved")
            self.assertEqual(read_archive.call_args.args[1], original)
            for key, changed, message in (("status", "failed", "selected accepted sample"),
                                          ("frequency_hz", 887_500_001, "selected accepted sample")):
                with self.subTest(key=key):
                    before = point[key]
                    point[key] = changed
                    with self.assertRaisesRegex(ValueError, message):
                        frequency_point.read_interval(root, root / "ledger.json", root, body, {"implementation": original})
                    point[key] = before
            artifacts[Path(modes_link["path"]).name]["sha256"] = "changed"
            with self.assertRaisesRegex(ValueError, "full SAFE modes differ"):
                frequency_point.read_interval(root, root / "ledger.json", root, body, {"implementation": original})

    def test_interval_receipt_pin_refuses_changed_bytes_before_array_loading(self):
        with TemporaryDirectory() as directory:
            root = Path(directory)
            ledger = root / "ledger.json"
            ledger.write_text("{}", encoding="utf-8")
            with patch.object(frequency_point.inventory, "read_archive", side_effect=AssertionError("array load requested")):
                with self.assertRaisesRegex(ValueError, "Receipt SHA-256 differs"):
                    frequency_point.read_interval(root, ledger, root, {}, {})

    def test_prepared_phase_refuses_wrong_frequency_channel_order_or_graph(self):
        record = {"reviewed_sources": {}, "superlu_library": {}}
        staged = {**record, "frequency_hz": frequency_point.FREQUENCY_HZ,
                  "incident_port_roots": list(frequency_point.ROOTS), "backward_port_roots": list(frequency_point.BACKWARD_ROOTS),
                  "reference_planes_m": list(frequency_point.END_PLANES_M), "operator_nonzeros": 10,
                  "interval_inputs": {"sha256": frequency_point.INTERVAL_INPUTS_SHA},
                  "interval_point": {"sha256": frequency_point.INTERVAL_POINT_SHA},
                  "interval_ledger": {"sha256": frequency_point.INTERVAL_LEDGER_SHA},
                  "safe_modes": {"sha256": frequency_point.INTERVAL_MODES_SHA},
                  "factor_index_exposure": {"matches_archived_1_ghz_csc_graph": True,
                      "signed_index_maximum": frequency_point.INDEX_MAX, "operator_nonzeros": 10,
                      "initial_factor_slots_per_store": 300}}
        with patch.object(frequency_point.thickness, "load_stage", return_value=(staged, {})):
            self.assertEqual(frequency_point.load_stage(None, record), (staged, {}))
            for key, replacement in (("frequency_hz", 1_000_000_000),
                                     ("incident_port_roots", list(reversed(frequency_point.ROOTS))),
                                     ("safe_modes", {"sha256": "changed"}),
                                     ("factor_index_exposure", {**staged["factor_index_exposure"], "matches_archived_1_ghz_csc_graph": False})):
                with self.subTest(key=key):
                    accepted = staged[key]
                    staged[key] = replacement
                    with self.assertRaisesRegex(ValueError, "prepared frequency, graph exposure"):
                        frequency_point.load_stage(None, record)
                    staged[key] = accepted

    def test_phase_entry_admits_only_the_current_reviewed_helper(self):
        implementation = json.loads((frequency_point.ROOT / "data/saved_device_AP_pnc/h80_z6_08875/summary_AP_pnc.json").read_bytes())["implementation"]
        self.assertEqual(frequency_point.workflow.digest(implementation["source_sha256"]), frequency_point.prep.fixed_port.SOURCE_MAP_SHA256)
        with TemporaryDirectory() as directory:
            root = Path(directory)
            output = root / "stage"

            def run_stage(args, record):
                self.assertEqual(record["slurm_job_id"], "123")
                self.assertEqual(record["numerical_thread_environment"], frequency_point.lawrence.THREADS)
                record["status"] = "operator_staged"

            command = ["frequency-point", "stage", "--body-stage-dir", str(root / "body"),
                       "--safe-dir", str(root / "safe"), "--interval-dir", str(root / "interval"), "--interval-ledger", str(root / "ledger.json"),
                       "--output", str(output)]
            artifact = frequency_point.artifact
            helper_path = Path(frequency_point.low_band.__file__).resolve()

            def changed_helper(path):
                link = artifact(path)
                if path.resolve() == helper_path:
                    link["sha256"] = "changed"
                return link

            with (patch.object(frequency_point, "sys", SimpleNamespace(platform="linux")),
                  patch.object(frequency_point.workflow, "source_identity", return_value=implementation),
                  patch.dict(os.environ, {**frequency_point.lawrence.THREADS, "SLURM_JOB_ID": "123"}),
                  patch.object(frequency_point.lawrence, "solve_environment", return_value={"fixture": "Slurm backend"}),
                  patch.object(frequency_point, "stage", side_effect=run_stage) as stage_call):
                with patch("sys.argv", command):
                    frequency_point.main()
                rejected_output = root / "rejected"
                with (patch("sys.argv", command[:-1] + [str(rejected_output)]),
                      patch.object(frequency_point, "artifact", side_effect=changed_helper)):
                    with self.assertRaisesRegex(ValueError, "reviewed low-band helper changed"):
                        frequency_point.main()
                stage_call.assert_called_once()
            receipt = json.loads((output / "stage_AP_pnc.json").read_bytes())
            self.assertEqual(receipt["status"], "operator_staged")
            self.assertEqual(receipt["reviewed_sources"][frequency_point.low_band.__name__], artifact(helper_path))
            self.assertEqual(receipt["reviewed_sources"][frequency_point.low_band.__name__]["sha256"], frequency_point.LOW_BAND_DRIVER_SHA)
            self.assertFalse(rejected_output.exists())

    def test_preparation_keeps_archived_inputs_and_writes_only_frequency_equations(self):
        with TemporaryDirectory() as directory:
            root = Path(directory)
            saved_body = root / "archived"
            output = root / "stage"
            saved_body.mkdir()
            output.mkdir()
            stiffness = csc_matrix(np.diag(np.arange(4.0, 14.0)))
            mass = csc_matrix(np.eye(10) / (tau * frequency_point.FREQUENCY_HZ) ** 2)
            omega = tau * frequency_point.FREQUENCY_HZ
            traction = 2j / omega
            expected = stiffness.toarray() - np.eye(10) * (1 + traction)
            paths = {key: saved_body / frequency_point.thickness.MATRIX_FILES[key] for key in ("K", "M", "A")}
            save_npz(paths["K"], stiffness)
            save_npz(paths["M"], mass)
            save_npz(paths["A"], csc_matrix(expected * (1 + 1j)))
            files = {key: frequency_point.artifact(path) for key, path in paths.items()}
            mtimes = {key: path.stat().st_mtime_ns for key, path in paths.items()}
            body = {"configuration": {"analysis": {"frequency_GHz": 1}}, "operator_csc_bytes": 320,
                    "operator_nonzeros": 10, "array_files": files, "displacement_dofs": 10,
                    **dict.fromkeys(("mesh_directory", "mesh_hash", "mesh", "mesh_summary",
                                     "preflight_receipt", "assembly_resources"))}
            system = SimpleNamespace(K=stiffness, M=mass)
            port = SimpleNamespace(elastic_system=system, left_dofs=np.arange(5), right_dofs=np.arange(5, 10))
            displacements = np.zeros((5, 1950), dtype=complex)
            displacements[:, frequency_point.ROOTS] = displacements[:, frequency_point.BACKWARD_ROOTS] = np.eye(5)
            forces = displacements.copy()
            forces[:, frequency_point.ROOTS] *= traction
            forces[:, frequency_point.BACKWARD_ROOTS] *= -traction
            modes = SimpleNamespace(port=port, frequency_hz=frequency_point.FREQUENCY_HZ,
                                    positive_indices=np.array(frequency_point.ROOTS), negative_indices=np.array(frequency_point.BACKWARD_ROOTS),
                                    displacements=displacements, axial_forces=forces,
                                    propagating=np.ones(1950, dtype=bool))
            args = SimpleNamespace(output=output, body_stage_dir=saved_body, safe_dir=saved_body, interval_dir=saved_body, interval_ledger=saved_body / "ledger.json")
            record = {"implementation": {}}
            with (patch.object(frequency_point, "read_body_stage", return_value=(body, {})),
                  patch.object(frequency_point, "admit_workspace", return_value={}),
                  patch.object(frequency_point, "read_interval", return_value=(port, {})),
                  patch.object(frequency_point, "restore_system", return_value=(system, modes, {})),
                  patch.object(frequency_point.workflow, "resolve_config", side_effect=lambda value: json.loads(json.dumps(value))),
                  patch("phononic.scattering_AP_pnc.splu", side_effect=AssertionError("body factorization requested"))):
                frequency_point.stage(args, record)
            self.assertEqual(record["status"], "operator_staged")
            self.assertEqual(record["configuration"]["analysis"]["frequency_GHz"], 0.8875)
            self.assertEqual(body["configuration"]["analysis"]["frequency_GHz"], 1)
            for key, path in paths.items():
                self.assertEqual(frequency_point.artifact(path), files[key])
                self.assertEqual(path.stat().st_mtime_ns, mtimes[key])
            self.assertEqual({path.name for path in output.iterdir()}, {"A_AP_pnc.npz", "rhs_AP_pnc.npy", "stage_AP_pnc.json"})
            np.testing.assert_allclose(load_npz(output / "A_AP_pnc.npz").toarray(), expected, rtol=0, atol=2e-15)
            rhs = np.load(output / "rhs_AP_pnc.npy", allow_pickle=False)
            np.testing.assert_allclose(rhs[:5], -2 * traction * np.eye(5), rtol=0, atol=1e-22)
            np.testing.assert_array_equal(rhs[5:], 0)
            exposure = record["factor_index_exposure"]
            self.assertTrue(exposure["matches_archived_1_ghz_csc_graph"])
            self.assertEqual(exposure["initial_factor_slots_per_store"], 30 * 10)
            self.assertEqual(exposure["archived_aggregate_report_headroom"], 300_551_043)
            self.assertFalse(exposure["new_factor_index_capacity_established"])
            self.assertTrue(all(record[name] is False for name in ("mesh_generated", "body_assembly_performed",
                                                                   "safe_eigensolve_performed", "factorization_performed")))

    def test_prepared_graph_inspection_checks_structure_and_width(self):
        with TemporaryDirectory() as directory:
            before = Path(directory) / "before.npz"
            matrix = csc_matrix(np.eye(2, dtype=complex))
            save_npz(before, csc_matrix([[1j, 1], [0, 0]]))
            exposure = frequency_point.factor_index_exposure(matrix, before, 2)
            self.assertFalse(exposure["matches_archived_1_ghz_csc_graph"])
            matrix.indptr = matrix.indptr.astype(np.int64)
            with self.assertRaisesRegex(ValueError, "int32 interface"):
                frequency_point.factor_index_exposure(matrix, before, 2)

    def test_factor_count_refusal_checkpoint_and_resume(self):
        with TemporaryDirectory() as directory:
            root = Path(directory)
            matrix = csc_matrix(np.diag([2.0 + 2j, 5.0 - 1j]))
            rhs = np.array([[1j, 2, 5, 1, 3], [-1, 3j, 2j, 2, 1j]])
            paths = {"A": root / "A.npz", "rhs": root / "rhs.npy"}
            save_npz(paths["A"], matrix)
            np.save(paths["rhs"], rhs, allow_pickle=False)
            staged = {"incident_port_roots": list(frequency_point.ROOTS), "displacement_dofs": 2,
                      "operator_nonzeros": matrix.nnz, "array_files": {key: frequency_point.artifact(path) for key, path in paths.items()},
                      "factor_index_exposure": frequency_point.factor_index_exposure(matrix, paths["A"], matrix.nnz),
                      **frequency_point.equations.equation_hashes(matrix, rhs)}
            stage_link = {"sha256": "reviewed-exact-stage"}
            identity = {"driver": {"sha256": "reviewed-driver"}, "superlu_library": {"sha256": "reviewed-binary"},
                        "reviewed_sources": {}, "implementation": {key: key for key in frequency_point.prep.NUMERICAL_KEYS}}

            def drive_fixture(limit, factor_count):
                def drives(operator, columns, seed, norm, publish, admit):
                    admit()
                    publish("factor", {"factor_seconds": 0, "factor_nonzeros": factor_count})
                    for column, b in enumerate(columns.T[:limit]):
                        vector = b / operator.diagonal()
                        metrics = frequency_point.lawrence.residual_metrics(b - operator @ vector, b, vector, norm)
                        publish("candidate", {"column": column, "update": 1, "target_met": True, **metrics}, vector)
                    if limit < columns.shape[1]:
                        raise KeyboardInterrupt()
                    return columns.shape[1]
                return drives

            refused, prior, resumed, reused = (root / name for name in ("refused", "prior", "resumed", "reused"))
            for output in (refused, prior, resumed, reused):
                output.mkdir()
            with (patch.object(frequency_point, "load_stage", return_value=(staged, stage_link)),
                  patch.object(frequency_point, "admit_workspace", return_value={}),
                  patch.object(frequency_point, "direct_factor_admission", return_value={"admitted": True, "slurm_node_quota_bytes": 1}),
                  patch.object(frequency_point.lawrence, "cgroup_controls", return_value=[]),
                  patch.object(frequency_point.lawrence, "memory_snapshot", return_value={}),
                  patch.object(frequency_point.lawrence, "memory_block", return_value=None)):
                with patch.object(frequency_point.lawrence, "solve_drives", side_effect=drive_fixture(4, -1)):
                    with self.assertRaisesRegex(RuntimeError, "invalid signed aggregate factor count"):
                        frequency_point.solve(SimpleNamespace(output=refused, prior_solve_dir=None), dict(identity))
                refused_record = json.loads((refused / "solve_AP_pnc.json").read_bytes())
                self.assertEqual(refused_record["linear_solver"]["channels"], [])
                self.assertEqual(list(refused.glob("accepted-*.npy")), [])
                with patch.object(frequency_point.lawrence, "solve_drives", side_effect=drive_fixture(2, 4)):
                    with self.assertRaises(KeyboardInterrupt):
                        frequency_point.solve(SimpleNamespace(output=prior, prior_solve_dir=None), dict(identity))
                prior_path = prior / "solve_AP_pnc.json"
                prior_bytes = prior_path.read_bytes()
                prior_time = prior_path.stat().st_mtime_ns
                args = SimpleNamespace(output=resumed, prior_solve_dir=prior, prior_solve_sha256=frequency_point.lawrence.file_sha(prior_path))
                record = dict(identity)
                with patch.object(frequency_point.lawrence, "solve_drives", side_effect=drive_fixture(3, 4)):
                    frequency_point.solve(args, record)
                complete_path = resumed / "solve_AP_pnc.json"
                frequency_point.workflow.write_json(complete_path, record)
                complete_link = frequency_point.artifact(complete_path)
                reuse = SimpleNamespace(output=reused, prior_solve_dir=resumed, prior_solve_sha256=complete_link["sha256"])
                reuse_record = dict(identity)
                with patch.object(frequency_point.lawrence, "solve_drives", side_effect=AssertionError("body factorization requested")):
                    frequency_point.solve(reuse, reuse_record)
            self.assertEqual(record["status"], "drives_accepted")
            channels = record["linear_solver"]["channels"]
            self.assertEqual([row["incident_port_root"] for row in channels], list(frequency_point.ROOTS))
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
            mesh = MeshTet(np.array([[0, 0, frequency_point.END_PLANES_M[1], frequency_point.END_PLANES_M[1]],
                                     [0, 1e-6, 0, 0], [0, 0, 0, 2e-7]]), np.arange(4).reshape(4, 1))
            layout = frequency_point.prep.fixed_port.displacement_layout(mesh)
            left = layout.nodal_dofs[:, :2].ravel()[:5]
            right = layout.nodal_dofs[:, 2:].ravel()[:5]
            omega = tau * frequency_point.FREQUENCY_HZ
            scale = 2 / omega
            # Each matched pair has D = [0 -c; -c 0], with the exact traces [1, i].
            dynamic = np.eye(layout.N)
            dynamic[np.r_[left, right], np.r_[left, right]] = 0
            dynamic[left, right] = dynamic[right, left] = -scale
            mass = np.eye(layout.N) * (scale / (omega * omega))
            system = frequency_point.ElasticSystem(mesh, layout, csc_matrix(dynamic + scale * np.eye(layout.N)), csc_matrix(mass))
            port = SimpleNamespace(elastic_system=system, basis=SimpleNamespace(N=5), left_dofs=left, right_dofs=right)
            positive, negative = np.array(frequency_point.ROOTS), np.array(frequency_point.BACKWARD_ROOTS)
            displacement = np.zeros((5, 1950), dtype=complex)
            displacement[:, positive] = displacement[:, negative] = np.eye(5)
            forces = np.zeros_like(displacement)
            forces[:, positive], forces[:, negative] = 1j * scale * np.eye(5), -1j * scale * np.eye(5)
            powers, wavenumbers = np.zeros(1950), np.zeros(1950, dtype=complex)
            powers[positive], powers[negative] = 1, -1
            wavenumbers[positive], wavenumbers[negative] = np.arange(1, 6), -np.arange(1, 6)
            modes = PortModes(port, frequency_point.FREQUENCY_HZ, wavenumbers, displacement, forces, positive, negative,
                              powers != 0, powers, np.zeros(1950))
            vectors = np.zeros((layout.N, 5), dtype=complex)
            vectors[left], vectors[right] = np.eye(5), 1j * np.eye(5)
            report = {"method": frequency_point.thickness.METHOD, "channels": []}
            stage_link = {"sha256": "fixture-stage"}

            def capture(operator, rhs):
                norm = frequency_point.lawrence.infinity_norm(operator)
                for column, incident in enumerate(positive):
                    vector = vectors[:, column]
                    name = f"accepted-{column:03d}_AP_pnc.npy"
                    metrics = frequency_point.lawrence.residual_metrics(rhs[:, column] - operator @ vector, rhs[:, column], vector, norm)
                    self.assertTrue(frequency_point.lawrence.accepted(metrics))
                    report["channels"].append({"column": column, "incident_port_root": int(incident), "info": 0,
                                               "accepted_vector": frequency_point.lawrence.write_vector(root / name, vector), **metrics})
                staged.update(operator_nonzeros=operator.nnz, **frequency_point.equations.equation_hashes(operator, rhs))
                raise frequency_point.equations.OperatorCaptured()

            config = json.loads((frequency_point.ROOT / "configs/perforated_layered_AP_pnc.json").read_text())
            config["analysis"]["frequency_GHz"] = 0.8875
            phase = dict.fromkeys(frequency_point.thickness.EXECUTION_KEYS, {})
            phase.update(superlu_library={}, reviewed_sources={})
            staged = {**phase, "mesh_directory": str(root), "displacement_dofs": layout.N,
                      "incident_port_roots": list(frequency_point.ROOTS), "configuration": config, "assembly_resources": {},
                      "reference_planes_m": list(frequency_point.END_PLANES_M), "safe_modes": {"bytes": 1},
                      "array_files": {"rhs": {"bytes": vectors.nbytes}},
                      **dict.fromkeys(("body_stage", "interval_inputs", "interval_point", "interval_ledger", "open_channels", "safe_section",
                                       "original_section_implementation", "section_mode_implementation",
                                       "section_mode_execution_provenance", "cross_frequency_branch_identity",
                                       "safe_receipt", "trace_sha256", "cutoff_status", "degeneracy_status"), {})}
            solved = {**phase, "stage": stage_link, "linear_solver": report, "status": "drives_accepted"}
            solve_link = {"path": str(root / "solve_AP_pnc.json"), "sha256": "fixture-solve"}
            mesh_record = {"ndof": layout.N, "mesh_hash": "mesh-hash", "mesh_sha256": "mesh-sha",
                           "mesh_vertices": 4, "mesh_tetrahedra": 1, "end_face_dofs": {"xleft": 5, "xright": 5},
                           "boundary_facet_counts": {},
                           "hole_pattern_provenance": {"source_sha256": "fixture", "x_offset_m": 0, "holes": [None] * 26}}
            with self.assertRaises(frequency_point.equations.OperatorCaptured):
                frequency_point.solve_scattering(system, modes, frequency_point.FREQUENCY_HZ, linear_solver=capture)
            with (patch.object(frequency_point, "load_stage", return_value=(staged, stage_link)),
                  patch.object(frequency_point, "admit_workspace", return_value={}),
                  patch.object(frequency_point.thickness, "read_phase", return_value=(solved, solve_link)),
                  patch.object(frequency_point, "restore_system", return_value=(system, modes, mesh_record)),
                  patch("phononic.scattering_AP_pnc.splu", side_effect=AssertionError("body factorization requested"))):
                output = root / "export"
                args = SimpleNamespace(output=output, solve_dir=root, solve_sha256="fixture-solve")
                record = dict(phase)
                frequency_point.export(args, record)
                for key in ("operator_csc_sha256", "rhs_array_sha256"):
                    args.output = root / key
                    verified = staged[key]
                    staged[key] = "changed"
                    with self.assertRaisesRegex(ValueError, "reconstructed operator or drives differ"):
                        frequency_point.export(args, dict(phase))
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
            self.assertEqual(provenance["reference_planes_m"], list(frequency_point.END_PLANES_M))
            self.assertTrue(all(provenance[name] is False for name in frequency_point.EXPLORATORY))
            self.assertEqual(set(record["public_fields"]), {"selected_fields_AP_pnc.npz", "selected_vertex_shapes_AP_pnc.vtu"})
            for name, link in record["public_fields"].items():
                self.assertEqual(link["sha256"], summary["output_sha256"][name])
            with np.load(record["public_fields"]["selected_fields_AP_pnc.npz"]["path"], allow_pickle=False) as fields:
                np.testing.assert_array_equal(fields["displacements_per_sqrt_w"], vectors)
                np.testing.assert_array_equal(fields["incident_mode_indices"], positive)
                np.testing.assert_allclose(fields["reflection_amplitudes"], 0, atol=1e-15)
                np.testing.assert_allclose(fields["transmission_amplitudes"], 1j * np.eye(5), atol=1e-15)


if __name__ == "__main__":
    unittest.main()
