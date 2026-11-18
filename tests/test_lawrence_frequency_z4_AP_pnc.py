"""Bounded 887.5 MHz four-layer execution checks. Athul Prem | _AP_pnc."""

from copy import deepcopy
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

from tools import run_lawrence_frequency_z4_AP_pnc as z4
from phononic.port_modes_AP_pnc import PortModes, PortSystem


def phase_fields():
    return {**dict.fromkeys(z4.thickness.EXECUTION_KEYS, {}), "reviewed_sources": {}, "superlu_library": {}}


def point_fields(roots=None, returning=None, *, dofs=z4.BODY_DOFS, port_dofs=z4.PORT_DOFS):
    roots = [12, 8, 6, 4, 2] if roots is None else roots
    returning = [13, 9, 7, 5, 3] if returning is None else returning
    fields = dict.fromkeys(("configuration", "body_stage", "body_assembly_source", "mesh_directory", "mesh_hash", "mesh",
                           "mesh_summary", "archived_safe_modes", "assembly_resources", "safe_section", "safe_modes",
                           "section_receipt", "safe_section_array_sha256"))
    return {**fields, "array_files": {}, "frequency_hz": z4.FREQUENCY_HZ, "displacement_dofs": dofs,
            "incident_port_roots": roots, "backward_port_roots": returning, "discrete_roots": 2 * port_dofs,
            "positive_open_channels": 5, "negative_open_channels": 5, "cutoff_status": "solver_resolved_at_sample",
            "degeneracy_status": "none_detected", "coincident_propagating_root_groups": [],
            "trace_sha256": z4.TRACE_SHA, "reference_planes_m": list(z4.low_band.END_PLANES_M),
            "open_channels": [{"direction": side, "root_id": root} for side, selected in
                              (("positive", roots), ("negative", returning)) for root in selected]}


def admission_fixture(root, ready):
    safe = root / "safe_AP_pnc.json"
    safe.write_text("saved SAFE fixture", encoding="utf-8")
    modes = root / "modes.npz"
    modes.write_text("saved raw modes fixture", encoding="utf-8")
    safe_link, mode_link = z4.artifact(safe), z4.artifact(modes)
    ready["safe_modes"] = mode_link
    relation = {"status": "signed_section_relation_accepted", "frequency_hz": z4.FREQUENCY_HZ,
                "positive_open_channels": 5, "negative_open_channels": 5, "overlap_gate": 0.99,
                "candidate_safe_receipt": {**safe_link, "path": "original-Windows-safe-path"},
                "candidate_safe_modes": mode_link, "root_permutation_square": {"status": "consistent"}}
    relation_path = root / "relation.json"
    z4.workflow.write_json(relation_path, relation)
    verification = {"status": "independently_verified_safe", "frequency_hz": z4.FREQUENCY_HZ,
                    "problem_dofs": ready["displacement_dofs"], "port_dofs": ready["discrete_roots"] // 2,
                    "all_discrete_roots_replayed": ready["discrete_roots"],
                    "positive_open_channels": 5, "negative_open_channels": 5,
                    "incident_port_roots": ready["incident_port_roots"], "backward_port_roots": ready["backward_port_roots"],
                    "safe_receipt": safe_link, "safe_modes": {**mode_link, "path": "original-Windows-modes-path"},
                    "relation_status": "matched", "relation": z4.artifact(relation_path),
                    "assigned_root_square_consistent": True}
    verification_path = root / "verified.json"
    z4.workflow.write_json(verification_path, verification)
    return safe_link, relation, verification, SimpleNamespace(
        safe_verification=verification_path, safe_verification_sha256=z4.artifact(verification_path)["sha256"],
        relation=relation_path, relation_sha256=z4.artifact(relation_path)["sha256"])


class LawrenceFrequencyZ4Tests(unittest.TestCase):
    def test_entry_checks_real_helpers_and_bounded_slurm_allocation(self):
        with TemporaryDirectory() as directory:
            root = Path(directory)
            argv = ["z4", "stage", "--safe-dir", str(root / "safe"), "--safe-sha256", "safe",
                    "--reference-operator", str(root / "original.npz"), "--safe-verification", str(root / "verified.json"),
                    "--safe-verification-sha256", "verified", "--relation", str(root / "relation.json"),
                    "--relation-sha256", "relation", "--output", str(root / "stage")]
            environment = {**z4.lawrence.THREADS, "SLURM_JOB_ID": "123", "SLURM_CPUS_PER_TASK": "2",
                           "SLURM_MEM_PER_NODE": "16384"}
            with (patch.object(z4, "sys", SimpleNamespace(platform="linux")), patch.dict(os.environ, environment),
                  patch.object(z4.lawrence, "solve_environment", return_value={}),
                  patch.object(z4, "stage", side_effect=lambda _, record: record.update(status="operator_staged")) as stage,
                  patch("sys.argv", argv)):
                z4.main()
            stage.assert_called_once()
            record = json.loads((root / "stage/stage_AP_pnc.json").read_bytes())
            self.assertEqual(record["reviewed_sources"][z4.archived_z4.__name__]["sha256"], z4.Z4_RUNNER_SHA)
            self.assertEqual(record["driver"], z4.artifact(Path(z4.__file__)))
            self.assertEqual(record["resource_ceiling"], z4.RESOURCE_CEILINGS["stage"])
            self.assertFalse(record["automatic_retry"])
            with patch.object(z4, "Z4_RUNNER_SHA", "changed"):
                with self.assertRaisesRegex(ValueError, "archive helper changed"):
                    z4.checked_sources()
            for changed in ({"SLURM_CPUS_PER_TASK": "3"}, {"SLURM_MEM_PER_NODE": "65536"}):
                with (patch.object(z4, "sys", SimpleNamespace(platform="linux")),
                      patch.dict(os.environ, {**environment, **changed})):
                    with self.assertRaisesRegex(ValueError, "single-threaded Linux"):
                        z4.run_phase(SimpleNamespace(mode="stage", output=root / "refused"), lambda *_: self.fail())
            self.assertFalse((root / "refused").exists())

    def test_safe_preserves_full_new_modes_and_stops_on_changed_channel_count(self):
        size = 6

        def analytic_port(fifth_cutoff):
            stiffness = np.diag([(tau * cutoff) ** 2 for cutoff in (0, 1e8, 2e8, 3e8, fifth_cutoff, 1.2e9)])
            basis = SimpleNamespace(N=size, mesh=SimpleNamespace(p=np.eye(2), t=np.zeros((3, 1), dtype=int)))
            port = PortSystem(None, basis, np.arange(size), np.arange(size, 2 * size),
                              stiffness, np.eye(size), np.eye(size), np.zeros((size, size)), 1.0)

            def modes(frequency):
                omega = tau * frequency
                k = np.sqrt((omega * omega - stiffness.diagonal()).astype(complex))
                waves = np.r_[k, -k]
                propagating = waves.imag == 0
                amplitudes = np.ones(2 * size)
                amplitudes[propagating] = np.sqrt(2 / (omega * abs(waves[propagating])))
                displacement = np.tile(np.eye(size), 2) * amplitudes
                force = 1j * waves * displacement
                powers = omega / 2 * np.sum((displacement.conj() * force).imag, axis=0)
                return PortModes(port, frequency, waves, displacement, force, np.arange(size), np.arange(size, 2 * size),
                                 propagating, powers, np.zeros(2 * size))
            return port, modes

        for cutoff, accepted in ((5e8, True), (9.5e8, False)):
            with self.subTest(cutoff=cutoff), TemporaryDirectory() as directory:
                root = Path(directory)
                port, modes = analytic_port(cutoff)
                body = {"archived_safe_modes": {}, "displacement_dofs": z4.BODY_DOFS,
                        "reference_planes_m": list(z4.low_band.END_PLANES_M)}
                receipt = {"status": "solved", "case": "12x4", "arrays": {"sha256": z4.SECTION_SHA}}
                args = SimpleNamespace(output=root, body_stage_dir=root, mesh_dir=root, section_dir=root)
                with (patch.object(z4, "PORT_DOFS", size), patch.object(z4.low_band, "admit_workspace", return_value={}),
                      patch.object(z4, "read_body", return_value=body),
                      patch.object(z4.equations, "read_receipt", return_value=(receipt, {})),
                      patch.object(z4, "archive_link", return_value={}),
                      patch.object(z4, "read_section", return_value=(port, modes(1e9))),
                      patch.object(z4, "restore_mesh", return_value=(None, None, None)), patch.object(z4, "verify_trace"),
                      patch.object(z4.lawrence, "INCIDENT_ROOTS", tuple(range(5))),
                      patch.object(z4, "_array_digest", return_value="section-arrays"),
                      patch.object(z4, "solve_port_modes", side_effect=lambda _, frequency: modes(frequency)) as solve):
                    record = {}
                    if accepted:
                        z4.safe(args, record)
                        self.assertEqual(record["status"], "safe_ready")
                        self.assertEqual(record["incident_port_roots"], list(range(5)))
                        self.assertEqual(record["backward_port_roots"], list(range(6, 11)))
                    else:
                        with self.assertRaisesRegex(ValueError, "exactly five resolved"):
                            z4.safe(args, record)
                        self.assertNotEqual(record["status"], "safe_ready")
                    solve.assert_called_once_with(port, z4.FREQUENCY_HZ)
                self.assertTrue(record["safe_eigensolve_performed"])
                self.assertFalse(record["anchor_eigensolve_performed"])
                self.assertFalse(record["body_assembly_performed"])
                with np.load(root / f"modes-{z4.FREQUENCY_HZ}_AP_pnc.npz", allow_pickle=False) as arrays:
                    self.assertEqual(arrays["displacements"].shape, (size, 2 * size))
                    self.assertEqual(float(arrays["frequency_hz"]), z4.FREQUENCY_HZ)

    def test_prebody_binds_independent_full_safe_and_signed_relation(self):
        with TemporaryDirectory() as directory:
            root = Path(directory)
            ready = point_fields()
            safe, relation, verification, args = admission_fixture(root, ready)

            def check():
                return z4.prebody_checks(ready, safe, args.safe_verification, args.safe_verification_sha256,
                                         args.relation, args.relation_sha256)

            links = check()
            self.assertEqual(links["safe_verification"], z4.artifact(args.safe_verification))
            self.assertEqual(links["signed_relation"], z4.artifact(args.relation))
            for key, changed in (("all_discrete_roots_replayed", 1349), ("relation_status", "unresolved"),
                                 ("assigned_root_square_consistent", False),
                                 ("incident_port_roots", list(reversed(ready["incident_port_roots"]))),
                                 ("safe_modes", {"bytes": 1, "sha256": "different"})):
                with self.subTest(key=key):
                    bad = {**verification, key: changed}
                    z4.workflow.write_json(args.safe_verification, bad)
                    args.safe_verification_sha256 = z4.artifact(args.safe_verification)["sha256"]
                    with self.assertRaisesRegex(ValueError, "independent full SAFE"):
                        check()
            z4.workflow.write_json(args.safe_verification, verification)
            args.safe_verification_sha256 = z4.artifact(args.safe_verification)["sha256"]
            for key, changed in (("overlap_gate", 0.98), ("root_permutation_square", {"status": "inconsistent"}),
                                 ("candidate_safe_receipt", {"bytes": 1, "sha256": "other"})):
                with self.subTest(key=key):
                    z4.workflow.write_json(args.relation, {**relation, key: changed})
                    args.relation_sha256 = z4.artifact(args.relation)["sha256"]
                    verification["relation"] = z4.artifact(args.relation)
                    z4.workflow.write_json(args.safe_verification, verification)
                    args.safe_verification_sha256 = z4.artifact(args.safe_verification)["sha256"]
                    with self.assertRaisesRegex(ValueError, "accepted signed section relation"):
                        check()

    def test_stage_uses_saved_matrices_five_dynamic_roots_and_exact_graph(self):
        size, count = 10, 5
        roots, returning = list(range(0, size, 2)), list(range(1, size, 2))
        omega = tau * z4.FREQUENCY_HZ
        stiffness = csc_matrix(np.diag(np.arange(4.0, 14.0)))
        system = SimpleNamespace(K=stiffness, M=csc_matrix(np.eye(size) / omega**2))
        port = SimpleNamespace(elastic_system=system, left_dofs=np.arange(count), right_dofs=np.arange(count, size))
        displacement = np.zeros((count, size), dtype=complex)
        displacement[:, roots] = displacement[:, returning] = np.eye(count)
        traction = 2j / omega
        force = displacement.copy()
        force[:, roots] *= traction
        force[:, returning] *= -traction
        modes = SimpleNamespace(port=port, frequency_hz=z4.FREQUENCY_HZ, positive_indices=np.array(roots),
                                negative_indices=np.array(returning), displacements=displacement,
                                axial_forces=force, propagating=np.ones(size, dtype=bool))
        for changed_graph in (False, True):
            with self.subTest(changed_graph=changed_graph), TemporaryDirectory() as directory:
                root = Path(directory)
                output = root / "stage"
                output.mkdir()
                ready = point_fields(roots, returning, dofs=size, port_dofs=count)
                safe, _, _, evidence = admission_fixture(root, ready)
                reference = stiffness.toarray().astype(complex)
                if changed_graph:
                    reference[0, 1] = 1
                saved = root / "original.npz"
                save_npz(saved, csc_matrix(reference))
                args = SimpleNamespace(output=output, safe_dir=root, safe_sha256="safe", reference_operator=saved,
                                       **vars(evidence))
                with (patch.multiple(z4, BODY_DOFS=size, PORT_DOFS=count),
                      patch.object(z4, "read_phase", return_value=(ready, safe)),
                      patch.object(z4.low_band, "admit_workspace", return_value={}),
                      patch.object(z4, "archive_link", return_value={}),
                      patch.object(z4, "restore_system", return_value=(system, modes, {})),
                      patch("phononic.scattering_AP_pnc.splu", side_effect=AssertionError("factorization requested"))):
                    record = {}
                    if changed_graph:
                        with self.assertRaisesRegex(ValueError, "operator graph differs"):
                            z4.stage(args, record)
                        self.assertFalse((output / "A_AP_pnc.npz").exists())
                        continue
                    z4.stage(args, record)
                self.assertEqual(record["status"], "operator_staged")
                self.assertEqual(record["incident_port_roots"], roots)
                expected = stiffness.toarray() - np.eye(size) * (1 + traction)
                np.testing.assert_allclose(load_npz(output / "A_AP_pnc.npz").toarray(), expected, rtol=0, atol=2e-15)
                rhs = np.load(output / "rhs_AP_pnc.npy", allow_pickle=False)
                np.testing.assert_allclose(rhs[:count], -2 * traction * np.eye(count), rtol=0, atol=1e-22)
                np.testing.assert_array_equal(rhs[count:], 0)
                self.assertTrue(record["factor_index_exposure"]["matches_archived_1_ghz_csc_graph"])
                self.assertEqual(record["safe_verification"], z4.artifact(evidence.safe_verification))

    def test_point_rejects_unresolved_degenerate_or_reordered_signed_inventory(self):
        point = point_fields()
        for key, value in (("frequency_hz", 1e9), ("cutoff_status", "unresolved"),
                           ("degeneracy_status", "coincident_propagating_roots"),
                           ("coincident_propagating_root_groups", [[2, 4]]),
                           ("incident_port_roots", [12, 8, 6, 4, 4]),
                           ("reference_planes_m", [0.0, 3e-6]),
                           ("open_channels", list(reversed(point["open_channels"])))):
            with self.subTest(key=key), self.assertRaises(ValueError):
                z4.validate_point({**point, key: value})

    def test_load_stage_rechecks_proof_artifacts_and_factor_index_admission(self):
        with TemporaryDirectory() as directory:
            root = Path(directory)
            staged = point_fields()
            safe, _, _, args = admission_fixture(root, staged)
            staged.update(safe_receipt=safe, safe_verification=z4.artifact(args.safe_verification),
                          signed_relation=z4.artifact(args.relation), factorization_performed=False,
                          operator_nonzeros=3, factor_index_exposure={
                              "matches_archived_1_ghz_csc_graph": True, "signed_index_maximum": z4.low_band.INDEX_MAX,
                              "operator_nonzeros": 3, "initial_factor_slots_per_store": 90})
            arguments = SimpleNamespace(stage_dir=root, stage_sha256="stage")
            with patch.object(z4, "read_phase", side_effect=lambda *_: (deepcopy(staged), {})):
                self.assertEqual(z4.load_stage(arguments, {})[0], staged)
                staged["factor_index_exposure"]["matches_archived_1_ghz_csc_graph"] = False
                with self.assertRaisesRegex(ValueError, "reviewed index exposure"):
                    z4.load_stage(arguments, {})
                staged["factor_index_exposure"]["matches_archived_1_ghz_csc_graph"] = True
                staged["factor_index_exposure"]["initial_factor_slots_per_store"] = z4.low_band.INDEX_MAX + 1
                with self.assertRaisesRegex(ValueError, "reviewed index exposure"):
                    z4.load_stage(arguments, {})
                staged["factor_index_exposure"]["initial_factor_slots_per_store"] = 90
                args.safe_verification.write_text("changed after preparation", encoding="utf-8")
                with self.assertRaisesRegex(ValueError, "SHA-256 differs"):
                    z4.load_stage(arguments, {})

    def test_direct_recovery_replays_accepted_prefix_and_preserves_five_root_order(self):
        with TemporaryDirectory() as directory:
            root = Path(directory)
            operator = csc_matrix([[4 + 2j, 1], [-1j, 5 - 2j]])
            rhs = np.array([[1j, 2, 5, 1, 8], [-1, 3j, 2j, 4, 3j]], dtype=complex)
            roots = [7, 4, 1, 9, 3]
            a_path, rhs_path = root / "A.npz", root / "rhs.npy"
            save_npz(a_path, operator)
            np.save(rhs_path, rhs, allow_pickle=False)
            staged = {"displacement_dofs": 2, "incident_port_roots": roots, "operator_nonzeros": operator.nnz,
                      "array_files": {"A": z4.artifact(a_path), "rhs": z4.artifact(rhs_path)},
                      "factor_index_exposure": z4.factor_index_exposure(operator, a_path),
                      **z4.equations.equation_hashes(operator, rhs)}
            prior, resumed, reused = (root / label for label in ("prior", "resumed", "reused"))
            for output in (prior, resumed, reused):
                output.mkdir()
            original = z4.lawrence.solve_drives

            def stop_after_prefix(matrix, drives, *args):
                self.assertEqual(original(matrix, drives[:, :2], *args), 2)
                raise KeyboardInterrupt

            with (patch.object(z4, "load_stage", return_value=(staged, {})),
                  patch.object(z4.low_band, "admit_workspace", return_value={}),
                  patch.object(z4, "direct_factor_admission", return_value={"admitted": True, "slurm_node_quota_bytes": 1}),
                  patch.object(z4.lawrence, "cgroup_controls", return_value=[]),
                  patch.object(z4.lawrence, "memory_snapshot", return_value={}),
                  patch.object(z4.lawrence, "memory_block", return_value=None),
                  patch.object(z4, "read_phase", side_effect=lambda path, *args: (
                      json.loads((path / "solve_AP_pnc.json").read_bytes()), {}))):
                with patch.object(z4.lawrence, "solve_drives", side_effect=stop_after_prefix):
                    with self.assertRaises(KeyboardInterrupt):
                        z4.solve(SimpleNamespace(output=prior, prior_solve_dir=None), {})
                record = {}
                z4.solve(SimpleNamespace(output=resumed, prior_solve_dir=prior, prior_solve_sha256="sha"), record)
                z4.workflow.write_json(resumed / "solve_AP_pnc.json", record)
                with patch.object(z4.lawrence, "solve_drives", side_effect=AssertionError("factorization requested")):
                    complete = {}
                    z4.solve(SimpleNamespace(output=reused, prior_solve_dir=resumed, prior_solve_sha256="sha"), complete)
            channels = record["linear_solver"]["channels"]
            self.assertEqual([row["incident_port_root"] for row in channels], roots)
            self.assertEqual([row["column"] for row in channels if row.get("reused")], [0, 1])
            self.assertEqual(complete["status"], "drives_accepted")
            self.assertEqual(complete["linear_solver"]["factor"], record["linear_solver"]["factor"])
            self.assertTrue(all(row["reused"] for row in complete["linear_solver"]["channels"]))

    def test_cached_export_preserves_full_p2_fields_and_rejects_changed_equations(self):
        with TemporaryDirectory() as directory:
            root = Path(directory)
            count = 5
            mesh = MeshTet(np.array([[0, 0, 16.65e-6, 16.65e-6], [0, 1e-6, 0, 0], [0, 0, 0, 2e-7]]),
                           np.arange(4).reshape(4, 1))
            layout = z4.fixed_port.displacement_layout(mesh)
            left, right = (layout.nodal_dofs[:, selected].ravel()[:count] for selected in (slice(0, 2), slice(2, 4)))
            omega, scale = tau * z4.FREQUENCY_HZ, 2 / (tau * z4.FREQUENCY_HZ)
            dynamic = np.eye(layout.N)
            dynamic[np.r_[left, right], np.r_[left, right]] = 0
            dynamic[left, right] = dynamic[right, left] = -scale
            system = z4.ElasticSystem(mesh, layout, csc_matrix(dynamic + scale * np.eye(layout.N)),
                                     csc_matrix(np.eye(layout.N) * scale / omega**2))
            roots, returning = list(range(0, 2 * count, 2)), list(range(1, 2 * count, 2))
            displacement, forces = np.zeros((count, 2 * count), dtype=complex), np.zeros((count, 2 * count), dtype=complex)
            displacement[:, roots] = displacement[:, returning] = np.eye(count)
            forces[:, roots], forces[:, returning] = 1j * scale * np.eye(count), -1j * scale * np.eye(count)
            port = SimpleNamespace(elastic_system=system, basis=SimpleNamespace(N=count), left_dofs=left, right_dofs=right)
            powers, waves = np.empty(2 * count), np.empty(2 * count, dtype=complex)
            powers[roots], powers[returning] = 1, -1
            waves[roots], waves[returning] = np.arange(1, count + 1), -np.arange(1, count + 1)
            modes = PortModes(port, z4.FREQUENCY_HZ, waves, displacement, forces, np.array(roots), np.array(returning),
                              np.ones(2 * count, dtype=bool), powers, np.zeros(2 * count))
            vectors = np.zeros((layout.N, count), dtype=complex)
            vectors[left], vectors[right] = np.eye(count), 1j * np.eye(count)
            config = json.loads((z4.ROOT / "configs/perforated_layered_AP_pnc.json").read_text())
            config["analysis"]["frequency_GHz"] = z4.FREQUENCY_HZ / 1e9
            staged = {**point_fields(roots, returning, dofs=layout.N, port_dofs=count), **phase_fields(),
                      "configuration": config, "mesh_directory": str(root), "safe_receipt": {}, "assembly_resources": {},
                      "safe_verification": {}, "signed_relation": {}}
            report = {"method": z4.thickness.METHOD, "channels": []}

            def capture(operator, rhs):
                staged.update(operator_nonzeros=operator.nnz, **z4.equations.equation_hashes(operator, rhs))
                norm = z4.lawrence.infinity_norm(operator)
                for column, root_id in enumerate(roots):
                    vector = vectors[:, column]
                    metrics = z4.lawrence.residual_metrics(operator @ vector - rhs[:, column], rhs[:, column], vector, norm)
                    report["channels"].append({"column": column, "incident_port_root": root_id, "info": 0, **metrics,
                        "accepted_vector": z4.lawrence.write_vector(root / f"accepted-{column:03d}_AP_pnc.npy", vector)})
                raise z4.equations.OperatorCaptured

            with self.assertRaises(z4.equations.OperatorCaptured):
                z4.solve_scattering(system, modes, z4.FREQUENCY_HZ, linear_solver=capture)
            stage_link = {"sha256": "exact-stage"}
            solve_link = {"path": str(root / "solve_AP_pnc.json"), "sha256": "accepted-solve"}
            solved = {**phase_fields(), "stage": stage_link, "status": "drives_accepted", "linear_solver": report}
            mesh_record = {"ndof": layout.N, "mesh_hash": "mesh", "mesh_sha256": "mesh-sha", "mesh_vertices": 4,
                           "mesh_tetrahedra": 1, "end_face_dofs": {"xleft": count, "xright": count},
                           "boundary_facet_counts": {}, "hole_pattern_provenance": {
                               "source_sha256": "fixture", "x_offset_m": 0, "holes": [None] * 26}}
            with (patch.object(z4, "load_stage", return_value=(staged, stage_link)),
                  patch.object(z4, "read_phase", return_value=(solved, solve_link)),
                  patch.object(z4.low_band, "admit_workspace", return_value={}),
                  patch.object(z4, "restore_system", return_value=(system, modes, mesh_record)),
                  patch("phononic.scattering_AP_pnc.splu", side_effect=AssertionError("factorization requested"))):
                record = phase_fields()
                z4.export(SimpleNamespace(output=root / "export", solve_dir=root, solve_sha256="sha"), record)
                for key in ("operator_csc_sha256", "rhs_array_sha256"):
                    before = staged[key]
                    staged[key] = "changed"
                    with self.assertRaisesRegex(ValueError, "reconstructed operator or drives differ"):
                        z4.export(SimpleNamespace(output=root / key, solve_dir=root, solve_sha256="sha"), phase_fields())
                    staged[key] = before
            summary = json.loads(Path(record["public_summary"]["path"]).read_bytes())
            self.assertTrue(summary["power_balance_passed"])
            self.assertEqual(summary["frequency_hz"], z4.FREQUENCY_HZ)
            for key in ("mesh_generated", "body_assembly_performed", "section_assembly_performed",
                        "safe_eigensolve_performed", "phase_adjustment_applied"):
                self.assertFalse(summary["checkpointed_scattering"][key])
            with np.load(record["public_fields"]["selected_fields_AP_pnc.npz"]["path"], allow_pickle=False) as fields:
                np.testing.assert_array_equal(fields["displacements_per_sqrt_w"], vectors)
                np.testing.assert_array_equal(fields["incident_mode_indices"], roots)
                np.testing.assert_allclose(fields["reflection_amplitudes"], 0, atol=1e-15)
                np.testing.assert_allclose(fields["transmission_amplitudes"], 1j * np.eye(count), atol=1e-15)


if __name__ == "__main__":
    unittest.main()
