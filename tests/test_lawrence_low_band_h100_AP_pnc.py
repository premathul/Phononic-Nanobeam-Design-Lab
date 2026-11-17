"""Cached h100 point acceptance and recovery. Athul Prem | _AP_pnc."""

from copy import deepcopy
from dataclasses import replace
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

from tools import run_lawrence_low_band_h100_AP_pnc as h100
from phononic.port_modes_AP_pnc import PortModes, PortSystem


STAGED_KEYS = ("configuration", "body_stage", "body_assembly_source", "body_source_compatibility", "mesh_directory",
               "mesh_hash", "mesh", "mesh_summary", "array_files", "archived_safe_modes", "assembly_resources",
               "displacement_dofs", "safe_receipt", "safe_section", "safe_modes", "safe_reference_ports",
               "section_receipt", "trace_sha256", "frequency_hz", "incident_port_roots", "backward_port_roots",
               "open_channels", "cutoff_status", "degeneracy_status", "reference_planes_m")


def phase_fields():
    return {**dict.fromkeys(h100.thickness.EXECUTION_KEYS, {}), "reviewed_sources": {}, "superlu_library": {}}


def matched_fixture():
    """An exactly matched four-channel finite system with known displacement."""
    mesh = MeshTet(np.array([[0, 0, 16.65e-6, 16.65e-6], [0, 1e-6, 0, 0], [0, 0, 0, 2e-7]]),
                   np.arange(4).reshape(4, 1))
    layout = h100.z4.fixed_port.displacement_layout(mesh)
    left, right = (layout.nodal_dofs[:, selected].ravel()[:4] for selected in (slice(0, 2), slice(2, 4)))
    omega, size = tau * h100.FREQUENCY_HZ, layout.N
    scale = 2 / omega
    dynamic = np.eye(size)
    dynamic[np.r_[left, right], np.r_[left, right]] = 0
    dynamic[left, right] = dynamic[right, left] = -scale
    system = h100.ElasticSystem(mesh, layout, csc_matrix(dynamic + scale * np.eye(size)),
                               csc_matrix(np.eye(size) * scale / omega**2))
    roots, backward = h100.ROOTS, h100.BACKWARD_ROOTS
    total = max(*roots, *backward) + 1
    displacement, forces = np.zeros((4, total), dtype=complex), np.zeros((4, total), dtype=complex)
    displacement[:, roots] = displacement[:, backward] = np.eye(4)
    forces[:, roots], forces[:, backward] = 1j * scale * np.eye(4), -1j * scale * np.eye(4)
    port = SimpleNamespace(elastic_system=system, basis=SimpleNamespace(N=4), left_dofs=left, right_dofs=right)
    powers, waves = np.zeros(total), np.zeros(total, dtype=complex)
    powers[list(roots)], powers[list(backward)] = 1, -1
    waves[list(roots)], waves[list(backward)] = np.arange(1, 5), -np.arange(1, 5)
    modes = PortModes(port, h100.FREQUENCY_HZ, waves, displacement, forces, np.array(roots), np.array(backward),
                      powers != 0, powers, np.zeros(total))
    vectors = np.zeros((size, 4), dtype=complex)
    vectors[left], vectors[right] = np.eye(4), 1j * np.eye(4)
    config = json.loads((h100.ROOT / "configs/perforated_layered_AP_pnc.json").read_text())
    config["analysis"]["frequency_GHz"] = 0.1
    record = {"ndof": size, "mesh_hash": "mesh", "mesh_sha256": "mesh-sha", "mesh_vertices": 4,
              "mesh_tetrahedra": 1, "end_face_dofs": {"xleft": 4, "xright": 4}, "boundary_facet_counts": {},
              "hole_pattern_provenance": {"source_sha256": "fixture", "x_offset_m": 0, "holes": [None] * 26}}
    return system, modes, vectors, config, record


class LawrenceLowBandH100Tests(unittest.TestCase):
    def test_actual_entry_checks_current_frozen_helpers(self):
        with TemporaryDirectory() as directory:
            root = Path(directory)
            command = ["h100", "stage", "--body-stage-dir", str(root / "body"), "--mesh-dir", str(root / "mesh"),
                       "--safe-dir", str(root / "safe"), "--output", str(root / "stage")]
            with (patch.object(h100, "sys", SimpleNamespace(platform="linux")),
                  patch.dict(os.environ, {**h100.lawrence.THREADS, "SLURM_JOB_ID": "123"}),
                  patch.object(h100.lawrence, "solve_environment", return_value={"single_thread": True}),
                  patch.object(h100, "stage", side_effect=lambda _, record: record.update(status="operator_staged")) as operation,
                  patch("sys.argv", command)):
                h100.main()
            operation.assert_called_once()
            record = json.loads((root / "stage/stage_AP_pnc.json").read_bytes())
            self.assertEqual(record["reviewed_sources"][h100.z4.__name__]["sha256"], h100.Z4_RUNNER_SHA)
            self.assertEqual(record["driver"], h100.artifact(Path(h100.__file__)))
            with patch.object(h100, "Z4_RUNNER_SHA", "changed"):
                with self.assertRaisesRegex(ValueError, "cached-execution helper changed"):
                    h100.checked_sources()

    def test_historical_source_compatibility_accepts_only_declared_pairs(self):
        original = {"source_sha256": {"physical_kernel": "same"}, "packages": {"numpy": "fixture"}, "platform": "Windows"}
        current = {**deepcopy(original), "platform": "Linux"}
        for path, (first, second) in h100.SOURCE_PAIRS.items():
            if first is not None:
                original["source_sha256"][path] = first
            current["source_sha256"][path] = second
        accepted = h100.verify_body_sources(original, current)
        self.assertEqual(accepted["historical_execution_source_pairs"], {key: list(value) for key, value in h100.SOURCE_PAIRS.items()})
        wrong = deepcopy(current)
        wrong["source_sha256"]["physical_kernel"] = "changed"
        with self.assertRaisesRegex(ValueError, "physical numerical kernels"):
            h100.verify_body_sources(original, wrong)
        wrong = deepcopy(current)
        wrong["source_sha256"]["src/phononic/workflow_AP_pnc.py"] = "changed"
        with self.assertRaisesRegex(ValueError, "historical h100 execution-source pair"):
            h100.verify_body_sources(original, wrong)

    def test_input_reader_retains_the_saved_batched_assembly_method(self):
        current = h100.workflow.source_identity()
        current["platform"] = "Linux"
        original = deepcopy(current)
        original["platform"] = "Windows"
        for path, (first, _) in h100.SOURCE_PAIRS.items():
            if first is None:
                original["source_sha256"].pop(path)
            else:
                original["source_sha256"][path] = first
        with TemporaryDirectory() as directory:
            root = Path(directory)
            config = json.loads((h100.ROOT / "configs/perforated_layered_AP_pnc.json").read_bytes())
            config["mesh"]["mesh_size_nm"] = 100
            config["resources"]["assembly_batch_elements"] = 2048
            (root / "configuration_AP_pnc.json").write_text(json.dumps(config), encoding="utf-8")
            link = {"path": "fixture", "sha256": "a" * 64, "bytes": 1}
            body = {"source": original, "status": "operator_ready", "factorization_attempted": False,
                    "mesh_comparison": {"ndof": h100.BODY_DOFS}, "mesh_hash": "mesh", "stages": {"production_batched_assembly": {}},
                    **{key: dict(link) for key in ("K", "M", "port_modes", "mesh", "A")}}
            body["configuration"] = {**link, "path": "configuration_AP_pnc.json"}
            shared = deepcopy(config)
            shared["mesh"]["mesh_size_nm"] = 80
            shared["analysis"]["frequency_GHz"] = 0.1
            safe = {"status": "safe_ready", "driver": {"sha256": h100.Z4_RUNNER_SHA}, "slurm_job_id": "1335500",
                    "frequency_hz": h100.FREQUENCY_HZ, "incident_port_roots": list(h100.ROOTS),
                    "backward_port_roots": list(h100.BACKWARD_ROOTS), "trace_sha256": h100.z4.TRACE_SHA,
                    "safe_modes": {**link, "sha256": h100.SAFE_MODES_SHA}, "safe_section": {**link, "sha256": h100.z4.SECTION_SHA},
                    "section_receipt": link, "archived_safe_modes": link, "configuration": shared,
                    "implementation": current, "open_channels": [], "cutoff_status": "resolved", "degeneracy_status": "none"}

            def relocated(path, expected):
                return {**expected, "path": str(path)}

            with (patch.object(h100.equations, "read_receipt", side_effect=[(body, {}), (safe, {})]),
                  patch.object(h100.z4, "archive_link", side_effect=relocated),
                  patch.object(h100.thickness, "read_link", return_value=root)):
                inputs, _ = h100.read_inputs(SimpleNamespace(body_stage_dir=root, mesh_dir=root, safe_dir=root),
                                             {"implementation": current})
            resources = inputs["configuration"]["resources"]
            self.assertEqual(resources["assembly_batch_elements"], 2048)
            self.assertEqual(resources["linear_solver"], "direct")
            self.assertFalse(any(key.startswith(("ilu_", "lgmres_")) for key in resources))

    def test_stage_captures_new_equations_with_no_eigensolve_or_factor(self):
        system, modes, vectors, config, mesh_record = matched_fixture()
        with TemporaryDirectory() as directory:
            root = Path(directory)
            output = root / "stage"
            output.mkdir()
            original = root / "A_AP_pnc.npz"
            save_npz(original, system.K.astype(complex))
            inputs = {**dict.fromkeys(STAGED_KEYS), "configuration": config, "displacement_dofs": system.basis.N,
                      "incident_port_roots": list(h100.ROOTS), "backward_port_roots": list(h100.BACKWARD_ROOTS), "array_files": {}}
            with (patch.object(h100.low_band, "admit_workspace", return_value={}),
                  patch.object(h100, "read_inputs", return_value=(inputs, {"nnz": system.K.nnz})),
                  patch.object(h100.z4, "archive_link", return_value={}),
                  patch.object(h100, "restore_system", return_value=(system, modes, mesh_record)),
                  patch.object(h100.z4, "solve_port_modes", side_effect=AssertionError("SAFE eigensolve requested")),
                  patch("phononic.scattering_AP_pnc.splu", side_effect=AssertionError("factorization requested"))):
                record = {}
                h100.stage(SimpleNamespace(output=output, body_stage_dir=root), record)
            operator = load_npz(output / "A_AP_pnc.npz")
            rhs = np.load(output / "rhs_AP_pnc.npy", allow_pickle=False)
            np.testing.assert_allclose(operator @ vectors, rhs, rtol=1e-15, atol=1e-24)
            self.assertFalse(record["factor_index_exposure"]["archived_h100_direct_factor_available"])
            self.assertNotIn("archived_aggregate_factor_nonzeros", record["factor_index_exposure"])
            self.assertTrue(all(record[name] is False for name in ("mesh_generated", "body_assembly_performed",
                                "section_assembly_performed", "safe_eigensolve_performed", "factorization_performed")))

    def test_restore_reuses_frequency_arrays_with_original_h100_maps(self):
        system, modes, _, _, mesh_record = matched_fixture()
        layout, size = system.basis, system.basis.N
        left, right = modes.port.left_dofs, modes.port.right_dofs
        source_port = PortSystem(None, modes.port.basis, left + 100, right + 100,
                                 np.zeros((4, 4)), np.eye(4), np.eye(4), np.zeros((4, 4)), 1.0)
        anchor = replace(modes, port=source_port, frequency_hz=1e9)
        fields = {name: getattr(anchor, name) for name in h100.z4.inventory.MODE_FIELDS}
        maps_checked = []

        def verify_maps(port, mesh, basis):
            np.testing.assert_array_equal(port.left_dofs, left)
            np.testing.assert_array_equal(port.right_dofs, right)
            self.assertIs(mesh, system.mesh)
            self.assertEqual(basis.N, size)
            maps_checked.append(True)

        with TemporaryDirectory() as directory:
            # Match the production stage's canonical path, including Windows
            # temporary directories supplied through an 8.3 short-name alias.
            root = Path(directory).resolve()
            for key, matrix in (("K", system.K), ("M", system.M)):
                save_npz(root / f"{key}.npz", matrix)
            np.savez(root / "anchor.npz", left_dofs=left, right_dofs=right, **fields)
            np.savez(root / "accepted.npz", **{name: getattr(modes, name) for name in h100.z4.inventory.MODE_FIELDS})
            for name in ("mesh_AP_pnc.npz", "summary_AP_pnc.json"):
                (root / name).write_bytes(b"saved-mesh-fixture")
            staged = {"mesh_directory": str(root), "mesh_hash": "mesh",
                      "mesh": h100.artifact(root / "mesh_AP_pnc.npz"),
                      "mesh_summary": h100.artifact(root / "summary_AP_pnc.json"),
                      "safe_section": {}, "safe_reference_ports": {},
                      "archived_safe_modes": h100.artifact(root / "anchor.npz"),
                      "safe_modes": h100.artifact(root / "accepted.npz"),
                      "array_files": {key: h100.artifact(root / f"{key}.npz") for key in ("K", "M")}}
            mesh_record.update(mesh_tetrahedra=61368, hole_count=26)
            digest = h100.workflow.digest([left.tolist(), right.tolist()])
            with (patch.multiple(h100, BODY_DOFS=size, PORT_DOFS=4, TRACE_SHA=digest),
                  patch.object(h100.workflow, "load_checkpoint", return_value=(system.mesh, mesh_record)),
                  patch.object(h100.z4, "read_section", return_value=(source_port, anchor)),
                  patch.object(h100.z4, "verify_trace", side_effect=verify_maps),
                  patch.object(h100.z4, "solve_port_modes", side_effect=AssertionError("SAFE eigensolve requested"))):
                restored, reused, _ = h100.restore_system(staged)
                np.testing.assert_array_equal(reused.displacements, modes.displacements)
                np.testing.assert_array_equal(reused.axial_forces, modes.axial_forces)
                np.testing.assert_array_equal(reused.port.left_dofs, left)
                self.assertIs(reused.port.elastic_system, restored)
                self.assertEqual(maps_checked, [True])
                fields["axial_forces"] = fields["axial_forces"] * 2
                np.savez(root / "anchor.npz", left_dofs=left, right_dofs=right, **fields)
                staged["archived_safe_modes"] = h100.artifact(root / "anchor.npz")
                with self.assertRaisesRegex(ValueError, "archived 1 GHz mode arrays differ"):
                    h100.restore_system(staged)

    def test_direct_partial_and_complete_reuse_keep_accepted_order_and_gates(self):
        with TemporaryDirectory() as directory:
            root = Path(directory)
            operator = csc_matrix([[4 + 2j, 1], [-1j, 5 - 2j]])
            rhs = np.array([[1j, 2, 5, 1], [-1, 3j, 2j, 4]])
            a_path, rhs_path = root / "A.npz", root / "rhs.npy"
            save_npz(a_path, operator)
            np.save(rhs_path, rhs, allow_pickle=False)
            staged = {"displacement_dofs": 2, "incident_port_roots": list(h100.ROOTS), "operator_nonzeros": operator.nnz,
                      "array_files": {"A": h100.artifact(a_path), "rhs": h100.artifact(rhs_path)},
                      "factor_index_exposure": h100.factor_index_exposure(operator, a_path, operator.nnz),
                      **h100.equations.equation_hashes(operator, rhs)}
            prior, resumed, reused = (root / name for name in ("prior", "resumed", "reused"))
            for output in (prior, resumed, reused):
                output.mkdir()
            original = h100.lawrence.solve_drives

            def stop_after_prefix(matrix, drives, *args):
                self.assertEqual(original(matrix, drives[:, :2], *args), 2)
                raise KeyboardInterrupt

            def read_previous(path, *args):
                return json.loads((path / "solve_AP_pnc.json").read_bytes()), {}

            with (patch.object(h100, "load_stage", return_value=(staged, {})),
                  patch.object(h100.low_band, "admit_workspace", return_value={}),
                  patch.object(h100, "direct_factor_admission", return_value={"admitted": True, "slurm_node_quota_bytes": 1}),
                  patch.object(h100.lawrence, "cgroup_controls", return_value=[]),
                  patch.object(h100.lawrence, "memory_snapshot", return_value={}),
                  patch.object(h100.lawrence, "memory_block", return_value=None),
                  patch.object(h100.z4, "read_phase", side_effect=read_previous)):
                with patch.object(h100.lawrence, "solve_drives", side_effect=stop_after_prefix):
                    with self.assertRaises(KeyboardInterrupt):
                        h100.solve(SimpleNamespace(output=prior, prior_solve_dir=None), {})
                record = {}
                h100.solve(SimpleNamespace(output=resumed, prior_solve_dir=prior, prior_solve_sha256="sha"), record)
                h100.workflow.write_json(resumed / "solve_AP_pnc.json", record)
                with patch.object(h100.lawrence, "solve_drives", side_effect=AssertionError("factorization requested")):
                    complete = {}
                    h100.solve(SimpleNamespace(output=reused, prior_solve_dir=resumed, prior_solve_sha256="sha"), complete)
            self.assertEqual(record["status"], "drives_accepted")
            channels = record["linear_solver"]["channels"]
            self.assertEqual([row["incident_port_root"] for row in channels], list(h100.ROOTS))
            self.assertEqual([row["column"] for row in channels if row.get("reused")], [0, 1])
            self.assertEqual(complete["linear_solver"]["factor"], record["linear_solver"]["factor"])
            self.assertTrue(all(row["reused"] for row in complete["linear_solver"]["channels"]))
            self.assertEqual(record["linear_solver"]["relative_target"], 1e-10)
            self.assertEqual(record["linear_solver"]["relative_residual_limit"], 1e-9)
            self.assertEqual(record["linear_solver"]["backward_residual_limit"], 1e-9)

    def test_cached_export_checks_exact_equations_and_full_field_identity(self):
        system, modes, vectors, config, mesh_record = matched_fixture()
        with TemporaryDirectory() as directory:
            root = Path(directory)
            staged = {**dict.fromkeys(STAGED_KEYS), **phase_fields(), "configuration": config,
                      "displacement_dofs": system.basis.N, "incident_port_roots": list(h100.ROOTS),
                      "backward_port_roots": list(h100.BACKWARD_ROOTS), "mesh_directory": str(root),
                      "assembly_resources": {}}
            report = {"method": h100.thickness.METHOD, "channels": []}

            def capture(operator, rhs):
                staged.update(operator_nonzeros=operator.nnz, **h100.equations.equation_hashes(operator, rhs))
                norm = h100.lawrence.infinity_norm(operator)
                for column, root_id in enumerate(h100.ROOTS):
                    vector = vectors[:, column]
                    metrics = h100.lawrence.residual_metrics(operator @ vector - rhs[:, column], rhs[:, column], vector, norm)
                    report["channels"].append({"column": column, "incident_port_root": root_id, "info": 0, **metrics,
                        "accepted_vector": h100.lawrence.write_vector(root / f"accepted-{column:03d}_AP_pnc.npy", vector)})
                raise h100.equations.OperatorCaptured

            with self.assertRaises(h100.equations.OperatorCaptured):
                h100.solve_scattering(system, modes, h100.FREQUENCY_HZ, linear_solver=capture)
            stage_link = {"sha256": "exact-stage"}
            solve_link = {"path": str(root / "solve_AP_pnc.json"), "sha256": "accepted-solve"}
            solved = {**phase_fields(), "stage": stage_link, "status": "drives_accepted", "linear_solver": report}
            with (patch.object(h100, "load_stage", return_value=(staged, stage_link)),
                  patch.object(h100.z4, "read_phase", return_value=(solved, solve_link)),
                  patch.object(h100.low_band, "admit_workspace", return_value={}),
                  patch.object(h100, "restore_system", return_value=(system, modes, mesh_record)),
                  patch("phononic.scattering_AP_pnc.splu", side_effect=AssertionError("factorization requested"))):
                record = phase_fields()
                h100.export(SimpleNamespace(output=root / "export", solve_dir=root, solve_sha256="sha"), record)
                for key in ("operator_csc_sha256", "rhs_array_sha256"):
                    original_hash = staged[key]
                    staged[key] = "changed"
                    with self.assertRaisesRegex(ValueError, "reconstructed operator or drives differ"):
                        h100.export(SimpleNamespace(output=root / key, solve_dir=root, solve_sha256="sha"), phase_fields())
                    staged[key] = original_hash
            summary = json.loads(Path(record["public_summary"]["path"]).read_bytes())
            self.assertTrue(summary["power_balance_passed"])
            self.assertEqual(summary["checkpointed_scattering"]["prepared_frequency_stage"], stage_link)
            for key in ("max_power_imbalance", "maximum_flux_power_imbalance", "maximum_net_outward_power_ratio",
                        "maximum_modal_flux_difference"):
                self.assertLess(summary[key], 1e-12)
            with np.load(record["public_fields"]["selected_fields_AP_pnc.npz"]["path"], allow_pickle=False) as fields:
                np.testing.assert_array_equal(fields["displacements_per_sqrt_w"], vectors)
                np.testing.assert_array_equal(fields["incident_mode_indices"], h100.ROOTS)
                np.testing.assert_allclose(fields["reflection_amplitudes"], 0, atol=1e-15)
                np.testing.assert_allclose(fields["transmission_amplitudes"], 1j * np.eye(4), atol=1e-15)


if __name__ == "__main__":
    unittest.main()
