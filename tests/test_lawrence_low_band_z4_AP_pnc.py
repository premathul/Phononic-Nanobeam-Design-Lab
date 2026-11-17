"""Cached four-layer frequency-point checks. Athul Prem | _AP_pnc."""

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
from skfem import Basis, ElementTriP2, ElementVector, MeshTet, MeshTri

from tools import run_lawrence_low_band_z4_AP_pnc as z4
from phononic.port_modes_AP_pnc import PortModes, PortSystem


def phase_fields():
    return {**dict.fromkeys(z4.thickness.EXECUTION_KEYS, {}), "reviewed_sources": {}, "superlu_library": {}}


def stage_fields():
    keys = ("configuration", "body_stage", "body_assembly_source", "mesh_directory", "mesh_hash", "mesh", "mesh_summary",
            "array_files", "archived_safe_modes", "assembly_resources", "displacement_dofs", "safe_section", "safe_modes",
            "section_receipt", "trace_sha256", "safe_section_array_sha256", "frequency_hz", "incident_port_roots",
            "backward_port_roots", "open_channels", "cutoff_status", "degeneracy_status", "reference_planes_m")
    return dict.fromkeys(keys)


class LawrenceLowBandZ4Tests(unittest.TestCase):
    def test_entry_checks_real_current_helpers_before_stage(self):
        with TemporaryDirectory() as directory:
            root = Path(directory)
            args = ["z4", "stage", "--safe-dir", str(root / "safe"), "--safe-sha256", "safe-sha",
                    "--reference-operator", str(root / "original.npz"), "--output", str(root / "stage")]
            with (patch.object(z4, "sys", SimpleNamespace(platform="linux")),
                  patch.dict(os.environ, {**z4.lawrence.THREADS, "SLURM_JOB_ID": "123"}),
                  patch.object(z4.lawrence, "solve_environment", return_value={"Slurm": "single thread"}),
                  patch.object(z4, "stage", side_effect=lambda _, record: record.update(status="operator_staged")) as operation,
                  patch("sys.argv", args)):
                z4.main()
            operation.assert_called_once()
            record = json.loads((root / "stage/stage_AP_pnc.json").read_bytes())
            self.assertEqual(record["status"], "operator_staged")
            self.assertEqual(record["reviewed_sources"][z4.low_band.__name__]["sha256"], z4.LOW_BAND_SHA)
            self.assertEqual(record["driver"], z4.artifact(Path(z4.__file__)))
            with patch.object(z4, "LOW_BAND_SHA", "unreviewed"):
                with self.assertRaisesRegex(ValueError, "execution helper changed"):
                    z4.checked_sources()

    def test_saved_section_reconstruction_rejects_changed_anchor_and_order(self):
        mesh = MeshTri()
        basis = Basis(mesh, ElementVector(ElementTriP2(), dim=3), intorder=4)
        size = basis.N
        left, right = np.arange(size), np.arange(size, 2 * size)
        port_values = {"left_dofs": left, "right_dofs": right, "frequency_hz": 1e9,
                       "wavenumbers_rad_m": np.r_[np.arange(1, size + 1), -np.arange(1, size + 1)].astype(complex),
                       "displacements": np.tile(np.eye(size), 2).astype(complex),
                       "axial_forces": 1j * np.tile(np.eye(size), 2),
                       "positive_indices": np.arange(size), "negative_indices": np.arange(size, 2 * size),
                       "propagating": np.ones(2 * size, dtype=bool), "powers_w": np.r_[np.ones(size), -np.ones(size)],
                       "relative_residuals": np.zeros(2 * size)}
        section = {"section_vertices_yz_m": mesh.p.T, "section_triangles": mesh.t.T,
                   "coordinates_yz_m": basis.doflocs.T, "component_dofs_xyz": np.asarray(basis.split_indices()),
                   "mass_kg_per_m": np.eye(size), "K0": np.zeros((size, size)), "Q": np.eye(size), "G": np.zeros((size, size)),
                   "reference_speed_m_s": np.sqrt(128500000000.0 / 5323)}
        aliases = {"wavenumbers_rad_m": "k_rad_m", "powers_w": "signed_power_w"}
        section.update({aliases.get(key, key): port_values[key] for key in z4.inventory.MODE_FIELDS if key != "frequency_hz"})
        with TemporaryDirectory() as directory:
            root = Path(directory)
            ports, saved = root / "ports.npz", root / "section.npz"
            np.savez(ports, **port_values)
            np.savez(saved, **section)
            trace = z4.workflow.digest([left.tolist(), right.tolist()])
            with patch.multiple(z4, PORT_DOFS=size, TRACE_SHA=trace):
                port, modes = z4.read_section(z4.artifact(saved), z4.artifact(ports))
                np.testing.assert_array_equal(port.K0, section["K0"])
                np.testing.assert_array_equal(modes.displacements, port_values["displacements"])
                section["k_rad_m"] = section["k_rad_m"] * 2
                np.savez(saved, **section)
                with self.assertRaisesRegex(ValueError, "archived 1 GHz SAFE array differs"):
                    z4.read_section(z4.artifact(saved), z4.artifact(ports))
                section["k_rad_m"] = port_values["wavenumbers_rad_m"]
                section["coordinates_yz_m"] = section["coordinates_yz_m"][::-1]
                np.savez(saved, **section)
                with self.assertRaisesRegex(ValueError, "coordinate/component order differs"):
                    z4.read_section(z4.artifact(saved), z4.artifact(ports))

    def test_safe_phase_replays_anchor_and_solves_only_new_frequency(self):
        size = 5
        stiffness = np.diag([0.0] * 4 + [(tau * 2e8) ** 2])
        port = PortSystem(None, SimpleNamespace(N=size), np.arange(size), np.arange(size, 2 * size),
                          stiffness, np.eye(size), np.eye(size), np.zeros((size, size)), 1.0)

        def analytic_modes(frequency):
            omega = tau * frequency
            k = np.sqrt((omega * omega - stiffness.diagonal()).astype(complex))
            waves = np.r_[k, -k]
            propagating = waves.imag == 0
            amplitudes = np.ones(2 * size)
            amplitudes[propagating] = np.sqrt(2 / (omega * abs(waves[propagating])))
            displacement = np.tile(np.eye(size), 2) * amplitudes
            force = 1j * waves * displacement
            powers = (omega / 2) * np.sum((displacement.conj() * force).imag, axis=0)
            return PortModes(port, frequency, waves, displacement, force, np.arange(size), np.arange(size, 2 * size),
                             propagating, powers, np.zeros(2 * size))

        anchor = analytic_modes(1e9)
        with TemporaryDirectory() as directory:
            root = Path(directory)
            output = root / "safe"
            output.mkdir()
            args = SimpleNamespace(output=output, body_stage_dir=root, mesh_dir=root, section_dir=root)
            body = {"archived_safe_modes": {}}
            receipt = {"status": "solved", "case": "12x4", "arrays": {"sha256": z4.SECTION_SHA}}
            with (patch.object(z4.low_band, "admit_workspace", return_value={}),
                  patch.object(z4, "read_body", return_value=body),
                  patch.object(z4.equations, "read_receipt", return_value=(receipt, {})),
                  patch.object(z4, "archive_link", return_value={}),
                  patch.object(z4, "read_section", return_value=(port, anchor)),
                  patch.object(z4, "restore_mesh", return_value=(None, None, None)),
                  patch.object(z4, "verify_trace"),
                  patch.object(z4.lawrence, "INCIDENT_ROOTS", tuple(range(size))),
                  patch.object(z4, "_array_digest", return_value="section-array-hash"),
                  patch.object(z4, "solve_port_modes", side_effect=lambda p, frequency: analytic_modes(frequency)) as solve):
                # The production section basis has a real MeshTri, omitted by this analytic mode fixture.
                port.basis.mesh = SimpleNamespace(p=np.eye(2), t=np.zeros((3, 1), dtype=int))
                record = {}
                z4.safe(args, record)
            solve.assert_called_once_with(port, z4.FREQUENCY_HZ)
            self.assertEqual(record["status"], "safe_ready")
            self.assertEqual(record["incident_port_roots"], [0, 1, 2, 3])
            self.assertEqual(record["backward_port_roots"], [5, 6, 7, 8])
            self.assertEqual(record["archived_1ghz_replay"]["positive_open_channels"], 5)
            self.assertTrue(record["safe_eigensolve_performed"])
            self.assertFalse(record["anchor_eigensolve_performed"])
            self.assertFalse(record["section_assembly_performed"])
            self.assertFalse(record["body_assembly_performed"])
            with np.load(output / "modes-100000000_AP_pnc.npz", allow_pickle=False) as arrays:
                self.assertEqual(float(arrays["frequency_hz"]), z4.FREQUENCY_HZ)
                self.assertEqual(arrays["displacements"].shape, (size, 2 * size))

    def test_stage_uses_saved_matrices_and_dynamic_signed_roots(self):
        with TemporaryDirectory() as directory:
            root = Path(directory)
            output = root / "stage"
            output.mkdir()
            size = 8
            roots, backwards = [0, 2, 4, 6], [1, 3, 5, 7]
            omega = tau * z4.FREQUENCY_HZ
            stiffness = csc_matrix(np.diag(np.arange(4.0, 12.0)))
            mass = csc_matrix(np.eye(size) / omega**2)
            traction = 2j / omega
            system = SimpleNamespace(K=stiffness, M=mass)
            port = SimpleNamespace(elastic_system=system, left_dofs=np.arange(4), right_dofs=np.arange(4, 8))
            displacement = np.zeros((4, 8), dtype=complex)
            displacement[:, roots] = displacement[:, backwards] = np.eye(4)
            force = displacement.copy()
            force[:, roots] *= traction
            force[:, backwards] *= -traction
            modes = SimpleNamespace(port=port, frequency_hz=z4.FREQUENCY_HZ, positive_indices=np.array(roots),
                                    negative_indices=np.array(backwards), displacements=displacement,
                                    axial_forces=force, propagating=np.ones(8, dtype=bool))
            saved = root / "original.npz"
            save_npz(saved, stiffness.astype(complex))
            ready = {**stage_fields(), "displacement_dofs": size, "incident_port_roots": roots,
                     "backward_port_roots": backwards, "array_files": {}}
            args = SimpleNamespace(output=output, safe_dir=root, safe_sha256="safe", reference_operator=saved)
            with (patch.object(z4, "read_phase", return_value=(ready, {})),
                  patch.object(z4.low_band, "admit_workspace", return_value={}),
                  patch.object(z4, "archive_link", return_value={}),
                  patch.object(z4, "restore_system", return_value=(system, modes, {})),
                  patch("phononic.scattering_AP_pnc.splu", side_effect=AssertionError("factorization requested"))):
                record = {}
                z4.stage(args, record)
            self.assertEqual(record["status"], "operator_staged")
            self.assertEqual(record["incident_port_roots"], roots)
            self.assertEqual(record["factor_index_exposure"]["archived_aggregate_factor_nonzeros"], 803_307_165)
            expected = stiffness.toarray() - np.eye(size) * (1 + traction)
            np.testing.assert_allclose(load_npz(output / "A_AP_pnc.npz").toarray(), expected, rtol=0, atol=2e-15)
            rhs = np.load(output / "rhs_AP_pnc.npy", allow_pickle=False)
            np.testing.assert_allclose(rhs[:4], -2 * traction * np.eye(4), rtol=0, atol=1e-22)
            np.testing.assert_array_equal(rhs[4:], 0)

    def test_direct_checkpoint_recovery_preserves_dynamic_order(self):
        with TemporaryDirectory() as directory:
            root = Path(directory)
            operator = csc_matrix([[4 + 2j, 1], [-1j, 5 - 2j]])
            rhs = np.array([[1j, 2, 5, 1], [-1, 3j, 2j, 4]])
            roots = [7, 4, 1, 9]
            a_path, rhs_path = root / "A.npz", root / "rhs.npy"
            save_npz(a_path, operator)
            np.save(rhs_path, rhs, allow_pickle=False)
            staged = {"displacement_dofs": 2, "incident_port_roots": roots,
                      "operator_nonzeros": operator.nnz, "array_files": {"A": z4.artifact(a_path), "rhs": z4.artifact(rhs_path)},
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
                  patch.object(z4, "read_phase", side_effect=lambda path, *args: (json.loads((path / "solve_AP_pnc.json").read_bytes()), {}))):
                with patch.object(z4.lawrence, "solve_drives", side_effect=stop_after_prefix):
                    with self.assertRaises(KeyboardInterrupt):
                        z4.solve(SimpleNamespace(output=prior, prior_solve_dir=None), {})
                args = SimpleNamespace(output=resumed, prior_solve_dir=prior, prior_solve_sha256="sha")
                record = {}
                z4.solve(args, record)
                z4.workflow.write_json(resumed / "solve_AP_pnc.json", record)
                with patch.object(z4.lawrence, "solve_drives", side_effect=AssertionError("factorization requested")):
                    complete = {}
                    z4.solve(SimpleNamespace(output=reused, prior_solve_dir=resumed, prior_solve_sha256="sha"), complete)
            self.assertEqual(record["status"], "drives_accepted")
            channels = record["linear_solver"]["channels"]
            self.assertEqual([row["incident_port_root"] for row in channels], roots)
            self.assertEqual([row["column"] for row in channels if row.get("reused")], [0, 1])
            self.assertEqual(complete["status"], "drives_accepted")
            self.assertEqual(complete["linear_solver"]["factor"], record["linear_solver"]["factor"])
            self.assertTrue(all(row["reused"] for row in complete["linear_solver"]["channels"]))

    def test_cached_export_restores_all_fields_and_rejects_changed_equations(self):
        with TemporaryDirectory() as directory:
            root = Path(directory)
            mesh = MeshTet(np.array([[0, 0, 16.65e-6, 16.65e-6], [0, 1e-6, 0, 0], [0, 0, 0, 2e-7]]),
                           np.arange(4).reshape(4, 1))
            layout = fixed_layout = z4.fixed_port.displacement_layout(mesh)
            left, right = (fixed_layout.nodal_dofs[:, selected].ravel()[:4] for selected in (slice(0, 2), slice(2, 4)))
            omega = tau * z4.FREQUENCY_HZ
            scale = 2 / omega
            dynamic = np.eye(layout.N)
            dynamic[np.r_[left, right], np.r_[left, right]] = 0
            dynamic[left, right] = dynamic[right, left] = -scale
            system = z4.ElasticSystem(mesh, layout, csc_matrix(dynamic + scale * np.eye(layout.N)),
                                     csc_matrix(np.eye(layout.N) * scale / omega**2))
            roots, backward = [0, 2, 4, 6], [1, 3, 5, 7]
            displacement, forces = np.zeros((4, 8), dtype=complex), np.zeros((4, 8), dtype=complex)
            displacement[:, roots] = displacement[:, backward] = np.eye(4)
            forces[:, roots], forces[:, backward] = 1j * scale * np.eye(4), -1j * scale * np.eye(4)
            port = SimpleNamespace(elastic_system=system, basis=SimpleNamespace(N=4), left_dofs=left, right_dofs=right)
            powers, waves = np.empty(8), np.empty(8, dtype=complex)
            powers[roots], powers[backward] = 1, -1
            waves[roots], waves[backward] = np.arange(1, 5), -np.arange(1, 5)
            modes = PortModes(port, z4.FREQUENCY_HZ, waves, displacement, forces, np.array(roots), np.array(backward),
                              np.ones(8, dtype=bool), powers, np.zeros(8))
            vectors = np.zeros((layout.N, 4), dtype=complex)
            vectors[left], vectors[right] = np.eye(4), 1j * np.eye(4)
            config = json.loads((z4.ROOT / "configs/perforated_layered_AP_pnc.json").read_text())
            config["analysis"]["frequency_GHz"] = 0.1
            staged = {**stage_fields(), **phase_fields(), "configuration": config, "displacement_dofs": layout.N,
                      "incident_port_roots": roots, "backward_port_roots": backward, "mesh_directory": str(root),
                      "safe_receipt": {}, "assembly_resources": {}}
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
                           "mesh_tetrahedra": 1, "end_face_dofs": {"xleft": 4, "xright": 4}, "boundary_facet_counts": {},
                           "hole_pattern_provenance": {"source_sha256": "fixture", "x_offset_m": 0, "holes": [None] * 26}}
            with (patch.object(z4, "load_stage", return_value=(staged, stage_link)),
                  patch.object(z4, "read_phase", return_value=(solved, solve_link)),
                  patch.object(z4.low_band, "admit_workspace", return_value={}),
                  patch.object(z4, "restore_system", return_value=(system, modes, mesh_record)),
                  patch("phononic.scattering_AP_pnc.splu", side_effect=AssertionError("factorization requested"))):
                record = phase_fields()
                z4.export(SimpleNamespace(output=root / "export", solve_dir=root, solve_sha256="sha"), record)
                for name in ("operator_csc_sha256", "rhs_array_sha256"):
                    digest = staged[name]
                    staged[name] = "changed"
                    with self.assertRaisesRegex(ValueError, "reconstructed operator or drives differ"):
                        z4.export(SimpleNamespace(output=root / name, solve_dir=root, solve_sha256="sha"), phase_fields())
                    staged[name] = digest
            summary = json.loads(Path(record["public_summary"]["path"]).read_bytes())
            self.assertTrue(summary["power_balance_passed"])
            provenance = summary["checkpointed_scattering"]
            self.assertEqual(provenance["body_assembly_source"], staged["body_assembly_source"])
            self.assertEqual(provenance["prepared_frequency_stage"], stage_link)
            self.assertTrue(all(provenance[name] is False for name in ("mesh_generated", "body_assembly_performed",
                               "section_assembly_performed", "safe_eigensolve_performed", "phase_adjustment_applied")))
            for key in ("max_power_imbalance", "maximum_flux_power_imbalance", "maximum_net_outward_power_ratio",
                        "maximum_modal_flux_difference"):
                self.assertLess(summary[key], 1e-12)
            with np.load(record["public_fields"]["selected_fields_AP_pnc.npz"]["path"], allow_pickle=False) as fields:
                np.testing.assert_array_equal(fields["displacements_per_sqrt_w"], vectors)
                np.testing.assert_array_equal(fields["incident_mode_indices"], roots)
                np.testing.assert_allclose(fields["reflection_amplitudes"], 0, atol=1e-15)
                np.testing.assert_allclose(fields["transmission_amplitudes"], 1j * np.eye(4), atol=1e-15)


if __name__ == "__main__":
    unittest.main()
