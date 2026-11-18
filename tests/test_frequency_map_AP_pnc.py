"""Bounded scattering frequency-map verification. Author: Athul Prem | _AP_pnc."""

from copy import deepcopy
import csv
import json
from pathlib import Path
import sys
from tempfile import TemporaryDirectory
import unittest
from unittest.mock import patch

import numpy as np
from scipy.sparse.linalg import splu

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "tools"))
sys.path.insert(0, str(ROOT / "src"))

import map_scattering_AP_pnc as mapper
from compare_scattering_AP_pnc import Modes
from phononic.elasticity_AP_pnc import assemble_elasticity
from phononic.geometry_AP_pnc import BeamGeometry, make_straight_mesh
from phononic.material_AP_pnc import CubicMaterial
from phononic.port_modes_AP_pnc import assemble_port, solve_port_modes
from phononic.scattering_AP_pnc import solve_scattering


FREQUENCIES_GHZ = [0.25, 0.30, 0.35]
DIMENSIONS_NM = (1000, 600, 200)
LENGTH_M = DIMENSIONS_NM[0] * 1e-9
CELLS = (4, 2, 1)


def small_guide():
    config = json.loads((ROOT / "configs/straight_scattering_AP_pnc.json").read_bytes())
    config["geometry"].update(zip(("length_nm", "width_nm", "thickness_nm"), DIMENSIONS_NM, strict=True))
    config["mesh"].update(cells=list(CELLS))
    config["analysis"].update(incident_side="right")
    return config


def independent_scattering(frequency_hz):
    """Solve each point on its own uncached body, port section, and modes."""
    material = CubicMaterial()
    mesh = make_straight_mesh(BeamGeometry(*(size * 1e-9 for size in DIMENSIONS_NM)), CELLS)
    system = assemble_elasticity(mesh, material)
    modes = solve_port_modes(assemble_port(system, material), frequency_hz)
    equations = {}

    def capture_and_solve(operator, rhs):
        equations.update(operator=operator, rhs=rhs)
        return splu(operator).solve(rhs)

    result = solve_scattering(system, modes, frequency_hz, "right", linear_solver=capture_and_solve)
    return system, modes, result, equations


class FrequencyMapTests(unittest.TestCase):
    def test_cli_prepare_only_reuses_checked_mesh_and_resumes(self):
        workflow = mapper.workflow
        with TemporaryDirectory() as temporary:
            root = Path(temporary)
            config = small_guide()
            config_path = root / "config_AP_pnc.json"
            workflow.write_json(config_path, config)
            mesh_cache = root / "mesh-check"
            cached_mesh = workflow.preflight_mesh(config, mesh_cache)
            output = root / "map"
            with (
                patch.object(workflow, "build_mesh", side_effect=AssertionError("recreated a checked mesh")),
                patch.object(workflow, "_assemble_system", wraps=workflow._assemble_system) as assemble,
                patch.object(workflow, "assemble_port", wraps=workflow.assemble_port) as section,
                patch.object(mapper, "_run_point", wraps=mapper._run_point) as solve,
                patch("builtins.print"),
            ):
                args = ["--config", str(config_path), "--frequencies-ghz", "0.25", "--output", str(output)]
                self.assertEqual(mapper.main([*args, "--max-new-points", "0", "--mesh-cache-root", str(mesh_cache)]), 0)
                prepared = mapper.read_saved_map(output)
                self.assertEqual(prepared["status"], "partial")
                self.assertFalse(prepared["stop_requested"])
                self.assertEqual([(p["status"], p["attempts"]) for p in prepared["points"]], [("pending", [])])
                self.assertFalse((output / "point-001").exists())
                solve.assert_not_called()
                shared = prepared["shared"]
                self.assertEqual(shared["mesh_hash"], cached_mesh["mesh_hash"])
                self.assertEqual(shared["artifacts"]["mesh"]["sha256"], cached_mesh["mesh_sha256"])
                self.assertEqual(set(shared["artifacts"]), set(mapper.SHARED_FILES))
                for artifact in shared["artifacts"].values():
                    self.assertTrue((output / artifact["path"]).is_file())

                self.assertEqual(mapper.main([*args, "--max-new-points", "1"]), 0)
                completed = mapper.read_saved_map(output)
                self.assertEqual(completed["status"], "complete")
                self.assertEqual(completed["shared"], shared)
                self.assertEqual([p["status"] for p in completed["points"][0]["attempts"]], ["accepted"])
                assemble.assert_called_once()
                section.assert_called_once()
                solve.assert_called_once()

    def test_adjacent_matching_respects_direction_permutation_and_complex_gauge(self):
        mass = np.diag([2.0, 3.0, 5.0])
        reference, candidate, expected = {}, {}, {}
        for name, sign, roots, new_roots, order, gauges in (
            ("incoming", -1, [7, 11, 19], [40, 51, 62], [2, 0, 1], [0.3, -1.2, 2.2]),
            ("reflection", 1, [29, 37, 43], [21, 24, 31], [1, 2, 0], [2.1, 0.7, -1.9]),
        ):
            shape = np.eye(3)[None, :, :]
            wave = sign * np.array([1.0, 3.0, 4.0]) * 1e6
            reference[name] = Modes(np.array(roots), wave, shape)
            candidate[name] = Modes(np.array(new_roots)[order], 1.05 * wave[order],
                                    (shape * np.exp(1j * np.array(gauges)))[..., order])
            expected[name] = (roots, new_roots, gauges)
        reference["transmission"] = reference["incoming"]
        candidate["transmission"] = candidate["incoming"]
        expected["transmission"] = expected["incoming"]

        matched = mapper.match_modes(reference, candidate, mass)
        self.assertEqual(matched["status"], "matched")
        for name, (old_roots, new_roots, gauges) in expected.items():
            with self.subTest(group=name):
                group = matched["groups"][name]
                self.assertEqual(group["reference_roots"], old_roots)
                self.assertEqual(group["candidate_roots"], new_roots)
                np.testing.assert_allclose(group["mass_overlaps"], 1, atol=1e-14, rtol=0)
                np.testing.assert_allclose(group["candidate_gauge_phase_rad"], gauges, atol=1e-14, rtol=0)

        truncated = {name: Modes(group.roots[:-1], group.k[:-1], group.displacement[..., :-1])
                     for name, group in candidate.items()}
        changed_count = mapper.match_modes(reference, truncated, mass)
        self.assertEqual(changed_count["status"], "unresolved")
        self.assertIn("count", changed_count["groups"]["incoming"]["reason"].lower())

        # Each old mode has the same mass overlap (> 0.99) with both candidates.
        # An overlap threshold alone cannot physically distinguish the branches.
        first = np.array([[1, 1], [-0.02, 0.02], [0, 0]])
        second = np.array([[1, 1], [0, 0], [-0.02, 0.02]])
        ambiguous_reference, ambiguous_candidate = {}, {}
        for name in reference:
            old, new = reference[name], candidate[name]
            scale = np.sqrt(mass.diagonal())[:, None]
            ambiguous_reference[name] = Modes(old.roots[:2], old.k[:2], (first / scale)[None, :, :])
            ambiguous_candidate[name] = Modes(new.roots[:2], new.k[:2], (second / scale)[None, :, :])
        ambiguous = mapper.match_modes(ambiguous_reference, ambiguous_candidate, mass)
        self.assertEqual(ambiguous["status"], "unresolved")
        self.assertRegex(ambiguous["groups"]["incoming"]["reason"], "(?i)ambigu|competing|distinguish")

    def test_interrupted_three_frequency_map_reuses_shared_and_accepted_work(self):
        def interrupt_second_frequency(port, frequency):
            if frequency == 300e6:
                raise KeyboardInterrupt("interrupt before the second SAFE eigensolve")
            return solve_port_modes(port, frequency)

        def report_unresolved_cutoff(port, frequency):
            if frequency == 300e6:
                raise RuntimeError("Unresolved lead cutoff in the injected test point.")
            return solve_port_modes(port, frequency)

        config = small_guide()
        with TemporaryDirectory() as temporary:
            output = Path(temporary) / "map"
            workflow = mapper.workflow
            with (
                patch.object(workflow, "build_mesh", wraps=workflow.build_mesh) as mesh,
                patch.object(workflow, "_assemble_system", wraps=workflow._assemble_system) as assemble,
                patch.object(workflow, "assemble_port", wraps=workflow.assemble_port) as section,
                patch.object(workflow, "solve_port_modes", side_effect=interrupt_second_frequency) as eigen,
                patch.object(workflow, "solve_scattering", wraps=workflow.solve_scattering) as scatter,
            ):
                with self.assertRaises(KeyboardInterrupt):
                    mapper.map_scattering(config, FREQUENCIES_GHZ, output)
                interrupted = json.loads((output / "map_AP_pnc.json").read_bytes())
                first, second, third = interrupted["points"]
                self.assertEqual([first["status"], second["status"], third["status"]],
                                 ["accepted", "interrupted", "pending"])
                self.assertEqual([call.args[1] for call in eigen.call_args_list], [250e6, 300e6])
                self.assertEqual([call.args[2] for call in scatter.call_args_list], [250e6])
                accepted_summary = output / first["accepted"]["summary"]["path"]
                preserved_summary = accepted_summary.read_bytes()

                eigen.side_effect = report_unresolved_cutoff
                with_gap = mapper.map_scattering(config, FREQUENCIES_GHZ, output)
                self.assertEqual([point["status"] for point in with_gap["points"]], ["accepted", "failed", "accepted"])
                failure = with_gap["points"][1]["attempts"][-1]
                self.assertEqual(failure["failure_kind"], "unresolved_cutoff")
                self.assertEqual(with_gap["points"][2]["track_from_previous"]["status"], "segment_start")
                self.assertEqual([with_gap["points"][index]["segment"] for index in (0, 2)], [1, 2])

                eigen.side_effect = solve_port_modes
                mapper.map_scattering(config, FREQUENCIES_GHZ, output)
                self.assertEqual([call.args[1] for call in eigen.call_args_list], [250e6, 300e6, 300e6, 350e6, 300e6])
                self.assertEqual([call.args[2] for call in scatter.call_args_list], [250e6, 350e6, 300e6])
                mesh.assert_called_once()
                assemble.assert_called_once()
                section.assert_called_once()
                self.assertEqual(accepted_summary.read_bytes(), preserved_summary)

            completed = json.loads((output / "map_AP_pnc.json").read_bytes())
            points = completed["points"]
            self.assertEqual([point["frequency_GHz"] for point in points], FREQUENCIES_GHZ)
            self.assertEqual([point["status"] for point in points], ["accepted"] * 3)
            self.assertEqual([point["track_from_previous"]["status"] for point in points],
                             ["segment_start", "matched", "matched"])
            self.assertEqual([point["segment"] for point in points], [1, 1, 1])
            self.assertFalse(completed["spatial_convergence_established"])
            self.assertFalse(completed["frequency_convergence_established"])
            self.assertEqual(points[1]["attempts"][:2], [second["attempts"][0], failure])
            self.assertEqual(len(points[1]["attempts"]), 3)
            self.assertTrue((output / "report_AP_pnc.html").is_file())
            with (output / "scattering_map_AP_pnc.csv").open(newline="", encoding="utf-8") as stream:
                map_rows = list(csv.DictReader(stream))
            self.assertEqual(len(map_rows), 3 * 2 * 4 * 4)

            extensional_wavenumbers = []
            operator_hashes, rhs_hashes = set(), set()
            for point in points:
                frequency = point["frequency_GHz"] * 1e9
                with self.subTest(frequency_hz=frequency):
                    system, modes, reference, equations = independent_scattering(frequency)
                    summary_path = output / point["accepted"]["summary"]["path"]
                    summary = json.loads(summary_path.read_bytes())
                    self.assertEqual(summary["frequency_hz"], frequency)
                    self.assertEqual(summary["incident_side"], "right")
                    self.assertEqual(summary["num_incident_channels"], 4)
                    self.assertTrue(summary["power_balance_passed"])
                    self.assertTrue(summary["matched_reference_passed"])
                    self.assertFalse(summary["spatial_convergence_established"])
                    provenance = summary["checkpointed_scattering"]
                    self.assertEqual(provenance["reference_planes_x_m"], [0., LENGTH_M])
                    operator_hashes.add(provenance["original_equations"]["operator_csc_sha256"])
                    rhs_hashes.add(provenance["original_equations"]["rhs_sha256"])
                    rows = [row for row in map_rows if float(row["frequency_GHz"]) == point["frequency_GHz"]]
                    self.check_compact_rows(rows, reference)
                    with np.load(summary_path.with_name("selected_fields_AP_pnc.npz"), allow_pickle=False) as saved:
                        np.testing.assert_array_equal(saved["incident_mode_indices"], reference.incident_mode_indices)
                        roots = reference.incident_mode_indices
                        wavenumbers = saved["port_wavenumbers_rad_m"]
                        self.assertEqual(len(wavenumbers), 90)
                        np.testing.assert_allclose(wavenumbers[roots], modes.wavenumbers_rad_m[roots], rtol=1e-9)
                        extensional_wavenumbers.append(abs(wavenumbers[roots[0]]))
                        self.check_selected_field(saved["displacements_per_sqrt_w"], system, modes, reference, equations)
                        for quantity in ("reflection", "transmission"):
                            np.testing.assert_allclose(saved[f"{quantity}_amplitudes"],
                                                       getattr(reference, f"{quantity}_amplitudes"), atol=1e-9, rtol=0)
                        signed_k = modes.wavenumbers_rad_m[roots]
                        phase_reference = np.diag(np.exp(-1j * signed_k * LENGTH_M))
                        np.testing.assert_allclose(saved["transmission_amplitudes"], phase_reference, atol=0.025, rtol=0)
            self.assertTrue(np.all(np.diff(extensional_wavenumbers) > 0))
            self.assertEqual(len(operator_hashes), len(points))
            self.assertEqual(len(rhs_hashes), len(points))
            shared_mass = output / completed["shared"]["artifacts"]["mass"]["path"]
            self.check_reuse_refuses_changed_inputs(config, output, accepted_summary, shared_mass)

    def check_reuse_refuses_changed_inputs(self, config, output, accepted_summary, shared_mass):
        workflow = mapper.workflow
        mismatched_config = deepcopy(config)
        mismatched_config["material"]["C44_Pa"] *= 1.03
        mismatched_source = workflow.source_identity()
        mismatched_source["source_sha256"]["src/phononic/material_AP_pnc.py"] = "0" * 64
        with (
            patch.object(workflow, "build_mesh", side_effect=AssertionError("recreated a saved mesh")),
            patch.object(workflow, "_assemble_system", side_effect=AssertionError("reassembled saved body operators")),
            patch.object(workflow, "assemble_port", side_effect=AssertionError("reassembled the saved SAFE section")),
            patch.object(workflow, "solve_port_modes", side_effect=AssertionError("resolved saved SAFE modes")),
            patch.object(workflow, "solve_scattering", side_effect=AssertionError("resolved an accepted frequency")),
        ):
            mapper.map_scattering(config, FREQUENCIES_GHZ, output)
            with self.assertRaisesRegex(ValueError, "(?i)checkpoint differs|configuration changed"):
                mapper.map_scattering(mismatched_config, FREQUENCIES_GHZ, output)
            with (
                patch.object(workflow, "source_identity", return_value=mismatched_source),
                self.assertRaisesRegex(ValueError, "(?i)identity|source|changed"),
            ):
                mapper.map_scattering(config, FREQUENCIES_GHZ, output)

            accepted_fields = accepted_summary.with_name("selected_fields_AP_pnc.npz")
            for path in (accepted_fields, shared_mass, output / "scattering_map_AP_pnc.csv", output / "map_AP_pnc.json"):
                with self.subTest(changed_artifact=path.name):
                    original = path.read_bytes()
                    if path.suffix == ".json":
                        map_data = json.loads(original)
                        map_data["spatial_convergence_established"] = True
                        tampered = json.dumps(map_data).encode("utf-8")
                    else:
                        tampered = original + b"\nchanged after completion\n"
                    path.write_bytes(tampered)
                    try:
                        with self.assertRaisesRegex(ValueError, "(?i)checksum|changed|corrupt|invalid"):
                            mapper.map_scattering(config, FREQUENCIES_GHZ, output)
                        self.assertEqual(path.read_bytes(), tampered)
                    finally:
                        path.write_bytes(original)

    def check_compact_rows(self, rows, reference):
        self.assertEqual(len(rows), 2 * 4 * 4)
        incident = reference.incident_mode_indices.tolist()
        for row in rows:
            quantity = row["kind"]
            roots = getattr(reference, f"{quantity}_mode_indices").tolist()
            index = (roots.index(int(row["outgoing_port_index"])), incident.index(int(row["incident_port_index"])))
            raw = complex(float(row["amplitude_real"]), float(row["amplitude_imag"]))
            power = float(row["power_fraction"])
            aligned = complex(float(row["segment_amplitude_real"]), float(row["segment_amplitude_imag"]))
            self.assertAlmostEqual(raw, getattr(reference, f"{quantity}_amplitudes")[index], delta=1e-9)
            self.assertAlmostEqual(power, getattr(reference, f"{quantity}_modal_powers")[index], delta=1e-9)
            self.assertAlmostEqual(abs(aligned)**2, power, delta=1e-12)
            if quantity == "transmission" and int(row["incident_port_index"]) == int(row["outgoing_port_index"]):
                self.assertEqual(row["incident_channel"], row["outgoing_channel"])
                self.assertAlmostEqual(aligned, raw, delta=1e-12)

    def check_selected_field(self, displacement, system, modes, reference, equations):
        operator, rhs = equations["operator"], equations["rhs"]
        remainder = operator @ displacement - rhs
        relative = np.linalg.norm(remainder, axis=0) / np.linalg.norm(rhs, axis=0)
        operator_norm = np.max(np.asarray(abs(operator).sum(axis=1)))
        backward = np.max(abs(remainder), axis=0) / (
            operator_norm * np.max(abs(displacement), axis=0) + np.max(abs(rhs), axis=0)
        )
        self.assertLess(np.max(relative), 1e-9)
        self.assertLess(np.max(backward), 1e-9)

        omega = 2 * np.pi * modes.frequency_hz
        nodal_force = (system.K - omega**2 * system.M) @ displacement
        left, right = modes.port.left_dofs, modes.port.right_dofs
        roots = reference.incident_mode_indices
        incident_u = modes.displacements[:, roots]
        incident_force = modes.axial_forces[:, roots]

        def outward_power(dof_field, dof_force):
            velocity = -1j * omega * dof_field
            return -0.5 * np.sum(dof_force * velocity.conj(), axis=0).real

        drive_power = -outward_power(incident_u, incident_force)
        reflected = outward_power(displacement[right] - incident_u, nodal_force[right] - incident_force) / drive_power
        transmitted = outward_power(displacement[left], nodal_force[left]) / drive_power
        np.testing.assert_allclose(reflected, reference.reflection, atol=1e-8, rtol=0)
        np.testing.assert_allclose(transmitted, reference.transmission, atol=1e-8, rtol=0)
        np.testing.assert_allclose(transmitted + reflected, 1.0, atol=1e-8, rtol=0)


if __name__ == "__main__":
    unittest.main()
