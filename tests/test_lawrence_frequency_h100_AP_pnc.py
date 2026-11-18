"""Bounded h100 frequency reuse checks. Athul Prem | _AP_pnc."""
from copy import deepcopy
import io
import json
from math import tau
import os
from pathlib import Path
from tempfile import TemporaryDirectory
from types import SimpleNamespace
import unittest
from unittest.mock import patch
from contextlib import redirect_stderr

import numpy as np
from scipy.sparse import csc_matrix, load_npz, save_npz
from skfem import MeshTet

from tools import run_lawrence_frequency_h100_AP_pnc as runner
from tools import run_lawrence_low_band_h100_AP_pnc as original
from phononic.port_modes_AP_pnc import PortModes, PortSystem


def link(sha, size=1):
    return {"path": "original-provider-path", "bytes": size, "sha256": sha}


def provider_fixture():
    safe_link, proof_link = link(runner.SAFE_SHA), link(runner.PREBODY_SHA)
    safe = {"status": "safe_ready", "driver": link(runner.PROVIDER_RUNNER_SHA), "slurm_job_id": "1335641",
            "frequency_hz": runner.FREQUENCY_HZ, "displacement_dofs": 426627, "discrete_roots": 1350,
            "positive_open_channels": 5, "negative_open_channels": 5, "incident_port_roots": list(runner.ROOTS),
            "backward_port_roots": list(runner.BACKWARD_ROOTS), "trace_sha256": runner.PROVIDER_TRACE_SHA,
            "safe_modes": link(runner.MODES_SHA), "safe_section": link(runner.z4.SECTION_SHA),
            "section_receipt": link(runner.z4.SECTION_RECEIPT_SHA), "body_stage": link(runner.z4.BODY_STAGE_SHA),
            "archived_safe_modes": link("provider-anchor"), "cutoff_status": "solver_resolved_at_sample",
            "degeneracy_status": "none_detected", "coincident_propagating_root_groups": [],
            "reference_planes_m": list(runner.low_band.END_PLANES_M), "open_channels": [],
            **dict.fromkeys(("body_assembly_performed", "section_assembly_performed", "mesh_generated",
                             "anchor_eigensolve_performed", "factorization_performed"), False)}
    proof = {"status": "independently_verified_safe", "frequency_hz": runner.FREQUENCY_HZ,
             "problem_dofs": 426627, "port_dofs": 675, "all_discrete_roots_replayed": 1350,
             "positive_open_channels": 5, "negative_open_channels": 5, "incident_port_roots": list(runner.ROOTS),
             "backward_port_roots": list(runner.BACKWARD_ROOTS), "safe_receipt": safe_link,
             "safe_modes": safe["safe_modes"], "relation_status": "matched", "assigned_root_square_consistent": True,
             "relation": link(runner.RELATION_SHA)}
    accepted = {"status": "point_verified_within_saved_discretization", "frequency_hz": runner.FREQUENCY_HZ,
                "problem_dofs": 426627, "port_dofs": 675, "reference_planes_m": list(runner.low_band.END_PLANES_M),
                "phase_receipts": {"safe": safe_link}, "prebody_verification": proof_link,
                "coherent_left_incidence": {"incident_roots": list(runner.ROOTS),
                                             "reflected_roots": list(runner.BACKWARD_ROOTS)}}
    return safe, safe_link, proof, proof_link, accepted


def phase_fields():
    return {**dict.fromkeys(runner.thickness.EXECUTION_KEYS, {}), "reviewed_sources": {}, "superlu_library": {}}


class LawrenceFrequencyH100Tests(unittest.TestCase):
    def test_isolated_reuse_preserves_imported_modules_and_source_identity(self):
        before = (original.FREQUENCY_HZ, original.ROOTS, original.read_inputs, original.workflow,
                  original.__file__, runner.workflow.run_study)
        core = runner.execution_core()
        self.assertEqual(core.FREQUENCY_HZ, runner.FREQUENCY_HZ)
        self.assertEqual(core.ROOTS, runner.ROOTS)
        for name in ("restore_system", "stage", "solve", "export"):
            self.assertEqual(getattr(core, name).__code__.co_code, getattr(original, name).__code__.co_code)
        self.assertEqual(core.__file__, original.__file__)
        self.assertIsNot(core.workflow, original.workflow)
        self.assertEqual(before, (original.FREQUENCY_HZ, original.ROOTS, original.read_inputs, original.workflow,
                                  original.__file__, runner.workflow.run_study))
        with patch.object(runner, "HELPER_SHA", "different"):
            with self.assertRaisesRegex(ValueError, "frozen h100 execution helper"):
                runner.execution_core()

    def test_provider_acceptance_rejects_frequency_order_and_provenance_changes(self):
        fields = provider_fixture()
        runner.validate_provider(*fields)
        for index, key, value in ((0, "frequency_hz", 1e9), (0, "trace_sha256", runner.TRACE_SHA),
                                   (0, "incident_port_roots", sorted(runner.ROOTS)),
                                   (0, "degeneracy_status", "unresolved"),
                                   (0, "body_assembly_performed", True),
                                   (2, "all_discrete_roots_replayed", 1349),
                                   (2, "assigned_root_square_consistent", False),
                                   (2, "safe_modes", link("different")),
                                   (4, "prebody_verification", link("different"))):
            with self.subTest(key=key):
                changed = list(deepcopy(fields))
                changed[index][key] = value
                with self.assertRaises(ValueError):
                    runner.validate_provider(*changed)

    def test_input_reader_accepts_archived_schema_without_solve_flag_and_distinct_traces(self):
        with TemporaryDirectory() as directory:
            root = Path(directory)
            config = {"geometry": {"thickness_nm": 200}, "material": {"rotation": "identity"},
                      "mesh": {"mesh_size_nm": 100, "thickness_layers": 4},
                      "analysis": {"kind": "scattering", "frequency_GHz": 1, "incident_side": "left"},
                      "resources": {"assembly_batch_elements": 2048}}
            config_path = root / "configuration_AP_pnc.json"
            runner.workflow.write_json(config_path, config)
            source = {"source_sha256": {}, "packages": {}}
            body = {"status": "operator_ready", "factorization_attempted": False,
                    "mesh_comparison": {"ndof": runner.BODY_DOFS}, "source": source, "mesh_hash": "mesh",
                    "configuration": {"path": str(config_path)}, "K": link("K"), "M": link("M"),
                    "port_modes": link("h100-map"), "mesh": link("mesh"),
                    "A": {**runner.REFERENCE_OPERATOR, "nnz": runner.OPERATOR_NNZ},
                    "stages": {"production_batched_assembly": {}}}
            self.assertNotIn("solve_attempted", body)
            safe, safe_link, proof, proof_link, accepted = provider_fixture()
            safe["configuration"] = deepcopy(config)
            safe["configuration"]["mesh"]["mesh_size_nm"] = 80
            safe["configuration"]["analysis"]["frequency_GHz"] = .8875
            safe["implementation"] = source
            arguments = SimpleNamespace(body_stage_dir=root, mesh_dir=root, safe_dir=root,
                                        reference_operator=root / "A.npz", safe_verification=root / "proof.json",
                                        reference_point=root / "accepted.json")

            def archived(path, expected):
                return {**expected, "path": str(path)}

            with (patch.object(runner.equations, "read_receipt", return_value=(body, link(runner.BODY_STAGE_SHA))),
                  patch.object(runner, "frozen_core", return_value=SimpleNamespace(verify_body_sources=lambda *_: {})),
                  patch.object(runner.z4, "archive_link", side_effect=archived),
                  patch.object(runner, "read_acceptance", return_value=(safe, safe_link, proof, proof_link,
                                                                          link(runner.REFERENCE_POINT_SHA))),
                  patch.object(runner.thickness, "read_link", side_effect=lambda item: Path(item["path"])),
                  patch.object(runner.low_band.prep, "NUMERICAL_KEYS", ("source_sha256", "packages"))):
                result, _ = runner.read_inputs(arguments, {"implementation": source})
                self.assertEqual(result["trace_sha256"], runner.TRACE_SHA)
                self.assertEqual(result["safe_provider_trace_sha256"], runner.PROVIDER_TRACE_SHA)
                self.assertNotEqual(result["trace_sha256"], result["safe_provider_trace_sha256"])
                self.assertEqual(result["archived_safe_modes"]["sha256"], "h100-map")
                self.assertEqual(result["safe_reference_ports"]["sha256"], "provider-anchor")
                self.assertEqual(result["configuration"]["analysis"]["frequency_GHz"], .8875)
                self.assertEqual(config["analysis"]["frequency_GHz"], 1)
                safe["configuration"]["geometry"]["thickness_nm"] = 201
                with self.assertRaisesRegex(ValueError, "physical inputs"):
                    runner.read_inputs(arguments, {"implementation": source})

    def test_frozen_restore_applies_h100_global_maps_to_shared_section_modes(self):
        with TemporaryDirectory() as directory:
            root, count, size = Path(directory), 5, 12
            core = runner.execution_core()
            left, right = np.arange(count), np.arange(size - count, size)
            roots, returning = tuple(range(count)), tuple(range(count, 2 * count))
            provider = PortSystem(None, SimpleNamespace(N=count), left + 100, right + 100,
                                  *(np.eye(count) for _ in range(4)), 1.)
            anchor = {key: np.array([1.]) for key in core.z4.inventory.MODE_FIELDS}
            anchor["frequency_hz"] = np.array(1e9)
            anchor_path = root / "anchor.npz"
            np.savez(anchor_path, **anchor, left_dofs=left, right_dofs=right)
            accepted_path = root / "accepted.npz"
            np.savez(accepted_path, frequency_hz=runner.FREQUENCY_HZ)
            maps_sha = runner.workflow.digest([left.tolist(), right.tolist()])
            mesh_record = {"hole_count": 26, "mesh_tetrahedra": 61368,
                           "end_face_dofs": {"xleft": count, "xright": count}}
            mapping_checks = []

            def verify(port, *_):
                np.testing.assert_array_equal(port.left_dofs, left)
                np.testing.assert_array_equal(port.right_dofs, right)
                mapping_checks.append(True)

            def modes(_, port):
                return SimpleNamespace(frequency_hz=runner.FREQUENCY_HZ, positive_indices=np.array(roots),
                                       negative_indices=np.array(returning), propagating=np.ones(2 * count, dtype=bool),
                                       port=port)

            core.BODY_DOFS, core.PORT_DOFS, core.TRACE_SHA = size, count, maps_sha
            core.ROOTS, core.BACKWARD_ROOTS = roots, returning
            core.z4 = SimpleNamespace(**{**vars(core.z4),
                "read_section": lambda *_: (provider, SimpleNamespace(**anchor)), "verify_trace": verify,
                "fixed_port": SimpleNamespace(displacement_layout=lambda _: SimpleNamespace(N=size)),
                "inventory": SimpleNamespace(MODE_FIELDS=tuple(anchor), read_modes=modes)})
            core.workflow.load_checkpoint = lambda *_: (None, mesh_record)
            core.thickness = SimpleNamespace(**{**vars(core.thickness), "read_link": lambda item: Path(item["path"])})
            core.load_npz = lambda _: csc_matrix(np.eye(size))
            core.ElasticSystem = lambda *args: SimpleNamespace()
            staged = {"mesh_directory": str(root), "mesh_hash": "mesh", "mesh": {"path": str(root / "mesh_AP_pnc.npz")},
                      "mesh_summary": {"path": str(root / "summary_AP_pnc.json")}, "safe_section": {},
                      "safe_reference_ports": {}, "archived_safe_modes": {"path": str(anchor_path)},
                      "safe_modes": {"path": str(accepted_path)}, "array_files": {key: {"path": key} for key in ("K", "M")}}
            _, result, _ = core.restore_system(staged)
            self.assertTrue(mapping_checks)
            np.testing.assert_array_equal(result.port.left_dofs, left)
            np.testing.assert_array_equal(provider.left_dofs, left + 100)
            core.TRACE_SHA = "wrong-body-map"
            with self.assertRaisesRegex(ValueError, "trace order differs"):
                core.restore_system(staged)

    def test_capture_preserves_original_equations_and_refuses_changed_graph(self):
        count, size = 5, 10
        roots, returning = tuple(range(0, size, 2)), tuple(range(1, size, 2))
        omega = tau * runner.FREQUENCY_HZ
        stiffness = csc_matrix(np.diag(np.arange(4., 14.)))
        system = SimpleNamespace(K=stiffness, M=csc_matrix(np.eye(size) / omega**2))
        port = SimpleNamespace(elastic_system=system, left_dofs=np.arange(count), right_dofs=np.arange(count, size))
        displacement = np.zeros((count, size), dtype=complex)
        displacement[:, roots] = displacement[:, returning] = np.eye(count)
        traction = 2j / omega
        force = displacement.copy()
        force[:, roots], force[:, returning] = traction * np.eye(count), -traction * np.eye(count)
        modes = SimpleNamespace(port=port, frequency_hz=runner.FREQUENCY_HZ, positive_indices=np.array(roots),
                                negative_indices=np.array(returning), displacements=displacement,
                                axial_forces=force, propagating=np.ones(size, dtype=bool))
        graph = runner.lawrence.array_sha(stiffness.indices, stiffness.indptr)
        for changed in (False, True):
            with self.subTest(changed_graph=changed), TemporaryDirectory() as directory:
                root = Path(directory)
                reference = stiffness.astype(complex).toarray()
                if changed:
                    reference[0, 1] = 1
                save_npz(root / "A_AP_pnc.npz", csc_matrix(reference))
                (root / "stage").mkdir()
                core = runner.execution_core()
                core.read_inputs = lambda *_: ({"displacement_dofs": size, "incident_port_roots": list(roots),
                                                "array_files": {}}, {"nnz": size})
                core.restore_system = lambda _: (system, modes, {})
                core.low_band = SimpleNamespace(**{**vars(core.low_band), "admit_workspace": lambda *_: {}})
                core.z4 = SimpleNamespace(**{**vars(core.z4), "archive_link": lambda *_: {}})
                record = {}
                with (patch.object(runner, "OPERATOR_NNZ", size), patch.object(runner, "OPERATOR_GRAPH_SHA", graph),
                      patch("phononic.scattering_AP_pnc.splu", side_effect=AssertionError("factorization requested"))):
                    if changed:
                        with self.assertRaisesRegex(ValueError, "operator graph differs"):
                            core.stage(SimpleNamespace(output=root / "stage", body_stage_dir=root), record)
                        self.assertFalse((root / "stage/A_AP_pnc.npz").exists())
                        continue
                    core.stage(SimpleNamespace(output=root / "stage", body_stage_dir=root), record)
                self.assertEqual(record["status"], "operator_staged")
                expected = stiffness.toarray() - np.eye(size) * (1 + traction)
                np.testing.assert_allclose(load_npz(root / "stage/A_AP_pnc.npz").toarray(), expected, rtol=0, atol=2e-15)
                rhs = np.load(root / "stage/rhs_AP_pnc.npy", allow_pickle=False)
                np.testing.assert_allclose(rhs[:count], -2 * traction * np.eye(count), rtol=0, atol=1e-22)
                np.testing.assert_array_equal(rhs[count:], 0)
                self.assertFalse(record["phase_adjustment_applied"])

    def test_direct_solver_recovers_only_accepted_prefix_and_preserves_gates(self):
        with TemporaryDirectory() as directory:
            root = Path(directory)
            operator = csc_matrix(np.diag([2 + 1j, 3 - 2j]))
            rhs = np.array([[1j, 2, 5, 1, 8], [-1, 3j, 2j, 4, 3j]], dtype=complex)
            a_path, rhs_path = root / "A.npz", root / "rhs.npy"
            save_npz(a_path, operator)
            np.save(rhs_path, rhs, allow_pickle=False)
            staged = {"displacement_dofs": 2, "incident_port_roots": list(runner.ROOTS), "operator_nonzeros": operator.nnz,
                      "array_files": {"A": runner.artifact(a_path), "rhs": runner.artifact(rhs_path)},
                      "factor_index_exposure": {"operator_graph_sha256": runner.lawrence.array_sha(operator.indices, operator.indptr)},
                      **runner.equations.equation_hashes(operator, rhs)}
            core = runner.execution_core()
            core.load_stage = lambda *_: (staged, {})
            core.low_band = SimpleNamespace(**{**vars(core.low_band), "admit_workspace": lambda *_: {}})
            core.direct_factor_admission = lambda _: {"admitted": True, "slurm_node_quota_bytes": 1}
            actual_solve = runner.lawrence.solve_drives
            core.lawrence = SimpleNamespace(**{**vars(core.lawrence), "cgroup_controls": lambda: [],
                "memory_snapshot": lambda *_: {}, "memory_block": lambda _: None})
            core.z4 = SimpleNamespace(**{**vars(core.z4), "read_phase": lambda path, *_: (
                json.loads((path / "solve_AP_pnc.json").read_bytes()), {})})
            prior, resumed, reused = (root / name for name in ("prior", "resumed", "reused"))
            for path in (prior, resumed, reused):
                path.mkdir()

            def interrupt(matrix, drives, *args):
                self.assertEqual(actual_solve(matrix, drives[:, :2], *args), 2)
                raise KeyboardInterrupt

            core.lawrence.solve_drives = interrupt
            with self.assertRaises(KeyboardInterrupt):
                core.solve(SimpleNamespace(output=prior, prior_solve_dir=None), {})
            core.lawrence.solve_drives = actual_solve
            record = {}
            core.solve(SimpleNamespace(output=resumed, prior_solve_dir=prior, prior_solve_sha256="sha"), record)
            runner.workflow.write_json(resumed / "solve_AP_pnc.json", record)
            core.lawrence.solve_drives = lambda *_: self.fail("Complete accepted-prefix reuse requested a factor.")
            complete = {}
            core.solve(SimpleNamespace(output=reused, prior_solve_dir=resumed, prior_solve_sha256="sha"), complete)
            self.assertEqual([row["incident_port_root"] for row in record["linear_solver"]["channels"]], list(runner.ROOTS))
            self.assertEqual([row["column"] for row in record["linear_solver"]["channels"] if row.get("reused")], [0, 1])
            self.assertEqual(record["linear_solver"]["relative_target"], 1e-10)
            self.assertEqual(record["linear_solver"]["relative_residual_limit"], 1e-9)
            self.assertEqual(record["linear_solver"]["backward_residual_limit"], 1e-9)
            self.assertEqual(complete["status"], "drives_accepted")

    def test_cached_export_publishes_exact_provenance_and_full_p2_fields_without_solving(self):
        with TemporaryDirectory() as directory:
            root, count = Path(directory), 5
            core = runner.execution_core()
            mesh = MeshTet(np.array([[0, 0, 16.65e-6, 16.65e-6], [0, 1e-6, 0, 0], [0, 0, 0, 2e-7]]),
                           np.arange(4).reshape(4, 1))
            layout = runner.z4.fixed_port.displacement_layout(mesh)
            left, right = (layout.nodal_dofs[:, selection].ravel()[:count] for selection in (slice(0, 2), slice(2, 4)))
            omega, scale = tau * runner.FREQUENCY_HZ, 2 / (tau * runner.FREQUENCY_HZ)
            dynamic = np.eye(layout.N)
            dynamic[np.r_[left, right], np.r_[left, right]] = 0
            dynamic[left, right] = dynamic[right, left] = -scale
            system = original.ElasticSystem(mesh, layout, csc_matrix(dynamic + scale * np.eye(layout.N)),
                                            csc_matrix(np.eye(layout.N) * scale / omega**2))
            roots, returning = tuple(range(0, 2 * count, 2)), tuple(range(1, 2 * count, 2))
            displacement, forces = np.zeros((count, 2 * count), complex), np.zeros((count, 2 * count), complex)
            displacement[:, roots] = displacement[:, returning] = np.eye(count)
            forces[:, roots], forces[:, returning] = 1j * scale * np.eye(count), -1j * scale * np.eye(count)
            port = SimpleNamespace(elastic_system=system, basis=SimpleNamespace(N=count), left_dofs=left, right_dofs=right)
            powers, waves = np.empty(2 * count), np.empty(2 * count, complex)
            powers[list(roots)], powers[list(returning)] = 1, -1
            waves[list(roots)], waves[list(returning)] = np.arange(1, count + 1), -np.arange(1, count + 1)
            modes = PortModes(port, runner.FREQUENCY_HZ, waves, displacement, forces, np.array(roots), np.array(returning),
                              np.ones(2 * count, bool), powers, np.zeros(2 * count))
            vectors = np.zeros((layout.N, count), complex)
            vectors[left], vectors[right] = np.eye(count), 1j * np.eye(count)
            config = json.loads((runner.ROOT / "configs/perforated_layered_AP_pnc.json").read_bytes())
            config["analysis"]["frequency_GHz"] = .8875
            staged = {**dict.fromkeys(("body_stage", "body_assembly_source", "body_source_compatibility", "safe_receipt",
                "section_receipt", "open_channels", "safe_section", "safe_modes", "safe_reference_ports", "trace_sha256",
                "cutoff_status", "degeneracy_status", *runner.EXTRA_KEYS), {}), **phase_fields(),
                "configuration": config, "mesh_directory": str(root), "assembly_resources": {},
                "displacement_dofs": layout.N, "incident_port_roots": list(roots),
                "reference_planes_m": list(runner.low_band.END_PLANES_M), "reused_execution_helper": runner.frozen_helper()}
            report = {"method": runner.thickness.METHOD, "channels": []}

            def capture(operator, rhs):
                staged.update(operator_nonzeros=operator.nnz, **runner.equations.equation_hashes(operator, rhs))
                norm = runner.lawrence.infinity_norm(operator)
                for column, root_id in enumerate(roots):
                    vector = vectors[:, column]
                    metrics = runner.lawrence.residual_metrics(operator @ vector - rhs[:, column], rhs[:, column], vector, norm)
                    report["channels"].append({"column": column, "incident_port_root": root_id, "info": 0, **metrics,
                        "accepted_vector": runner.lawrence.write_vector(root / f"accepted-{column:03d}_AP_pnc.npy", vector)})
                raise runner.equations.OperatorCaptured

            with self.assertRaises(runner.equations.OperatorCaptured):
                core.solve_scattering(system, modes, runner.FREQUENCY_HZ, linear_solver=capture)
            stage_path = root / "stage_AP_pnc.json"
            runner.workflow.write_json(stage_path, staged)
            stage_link = runner.artifact(stage_path)
            solve_link = {"path": str(root / "solve_AP_pnc.json"), "sha256": "accepted-solve"}
            solved = {**phase_fields(), "stage": stage_link, "status": "drives_accepted", "linear_solver": report}
            mesh_record = {"ndof": layout.N, "mesh_hash": "mesh", "mesh_sha256": "mesh-sha", "mesh_vertices": 4,
                           "mesh_tetrahedra": 1, "end_face_dofs": {"xleft": count, "xright": count},
                           "boundary_facet_counts": {}, "hole_pattern_provenance": {
                               "source_sha256": "fixture", "x_offset_m": 0, "holes": [None] * 26}}
            core.ROOTS, core.BACKWARD_ROOTS = roots, returning
            core.load_stage = lambda *_: (staged, stage_link)
            core.z4 = SimpleNamespace(**{**vars(core.z4), "read_phase": lambda *_: (solved, solve_link)})
            core.low_band = SimpleNamespace(**{**vars(core.low_band), "admit_workspace": lambda *_: {}})
            core.restore_system = lambda _: (system, modes, mesh_record)
            with patch("phononic.scattering_AP_pnc.splu", side_effect=AssertionError("factorization requested")):
                record = phase_fields()
                core.export(SimpleNamespace(output=root / "export", solve_dir=root, solve_sha256="sha"), record)
                staged["rhs_array_sha256"] = "changed"
                with self.assertRaisesRegex(ValueError, "reconstructed operator or drives differ"):
                    core.export(SimpleNamespace(output=root / "refused", solve_dir=root, solve_sha256="sha"), phase_fields())
            summary = json.loads(Path(record["public_summary"]["path"]).read_bytes())
            self.assertTrue(summary["power_balance_passed"])
            self.assertEqual(summary["frequency_hz"], runner.FREQUENCY_HZ)
            provenance = summary["checkpointed_scattering"]
            self.assertEqual(provenance["reused_execution_helper"], runner.frozen_helper())
            self.assertEqual(provenance["prepared_frequency_stage"], stage_link)
            self.assertFalse(provenance["phase_adjustment_applied"])
            self.assertEqual(provenance["reference_planes_m"], list(runner.low_band.END_PLANES_M))
            for key in runner.EXTRA_KEYS:
                self.assertEqual(provenance[key], staged[key])
            with np.load(record["public_fields"]["selected_fields_AP_pnc.npz"]["path"], allow_pickle=False) as fields:
                np.testing.assert_array_equal(fields["displacements_per_sqrt_w"], vectors)
                np.testing.assert_array_equal(fields["incident_mode_indices"], roots)
                np.testing.assert_allclose(fields["reflection_amplitudes"], 0, atol=1e-15)
                np.testing.assert_allclose(fields["transmission_amplitudes"], 1j * np.eye(count), atol=1e-15)

    def test_entry_checks_resource_ceilings_and_bounded_job_cgroup(self):
        implementation = json.loads((runner.ROOT / "data/saved_device_AP_pnc/h80_z6_08875/summary_AP_pnc.json").read_bytes())["implementation"]
        self.assertEqual(runner.workflow.digest(implementation["source_sha256"]), runner.z4.fixed_port.SOURCE_MAP_SHA256)
        with TemporaryDirectory() as directory:
            root = Path(directory)
            environment = {**runner.lawrence.THREADS, "SLURM_JOB_ID": "123", "SLURM_CPUS_PER_TASK": "2",
                           "SLURM_MEM_PER_NODE": "49152", "AP_PNC_WORKER_SECONDS": "850"}
            snapshot = {"job_headroom_bytes": 47 * runner.lawrence.GIB, "node_available_bytes": 100 * runner.lawrence.GIB,
                        "worker_rss_bytes": 100 * runner.lawrence.MIB, "cgroups": []}
            bounded = [(Path("/job_123"), 48 * runner.lawrence.GIB, "usage", "peak", "events")]
            proxy = SimpleNamespace(**{**vars(runner.lawrence), "cgroup_controls": lambda: bounded,
                "is_job_group": lambda _: True, "memory_snapshot": lambda *_: snapshot, "solve_environment": lambda: {}})
            with (patch.object(runner, "sys", SimpleNamespace(platform="linux")), patch.dict(os.environ, environment),
                  patch.object(runner.workflow, "source_identity", return_value=implementation),
                  patch.object(runner, "lawrence", proxy), patch.object(runner, "allocation_seconds", return_value=900)):
                runner.run_phase(SimpleNamespace(mode="solve", output=root / "solve"),
                                 lambda _, record: record.update(status="drives_accepted"))
                record = json.loads((root / "solve/solve_AP_pnc.json").read_bytes())
                self.assertEqual(record["driver"], runner.artifact(Path(runner.__file__)))
                self.assertEqual(record["reused_execution_helper"], runner.frozen_helper())
                self.assertEqual(record["resource_ceiling"], runner.RESOURCE_CEILINGS["solve"])
                self.assertFalse(record["automatic_retry"])
                for changes in ({"SLURM_CPUS_PER_TASK": "3"}, {"SLURM_MEM_PER_NODE": "65536"},
                                {"AP_PNC_WORKER_SECONDS": "851"}, {"OPENBLAS_NUM_THREADS": "2"}):
                    with self.subTest(changes=changes), patch.dict(os.environ, changes):
                        with self.assertRaisesRegex(ValueError, "resource ceilings"):
                            runner.run_phase(SimpleNamespace(mode="solve", output=root / "refused"), lambda *_: self.fail())
                with patch.object(runner, "allocation_seconds", return_value=901):
                    with self.assertRaisesRegex(ValueError, "bounded Slurm allocation"):
                        runner.run_phase(SimpleNamespace(mode="solve", output=root / "refused"), lambda *_: self.fail())
                proxy.cgroup_controls = lambda: []
                with self.assertRaisesRegex(ValueError, "memory cgroup"):
                    runner.run_phase(SimpleNamespace(mode="solve", output=root / "refused"), lambda *_: self.fail())
                self.assertFalse((root / "refused").exists())

    def test_cli_refuses_existing_outputs_and_incomplete_prefix_recovery(self):
        with TemporaryDirectory() as directory:
            root = Path(directory)
            output = root / "existing"
            output.mkdir()
            (output / "accepted-000_AP_pnc.npy").write_text("preserved", encoding="utf-8")
            argv = ["runner", "solve", "--stage-dir", str(root / "stage"), "--stage-sha256", "sha",
                    "--output", str(output)]
            with patch("sys.argv", argv), redirect_stderr(io.StringIO()), self.assertRaises(SystemExit):
                runner.main()
            with (patch("sys.argv", [*argv[:-2], "--prior-solve-dir", str(root / "prior"), "--output", str(root / "new")]),
                  redirect_stderr(io.StringIO()), self.assertRaises(SystemExit)):
                runner.main()


if __name__ == "__main__":
    unittest.main()
