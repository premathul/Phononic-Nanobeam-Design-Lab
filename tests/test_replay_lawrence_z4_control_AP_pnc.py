"""Explicit-profile replay and signed thickness matching. Athul Prem | _AP_pnc."""

from copy import deepcopy
from pathlib import Path
import unittest
from unittest.mock import patch

import numpy as np
from skfem import Basis, ElementTriP1, MeshTri
from skfem.models.poisson import mass

from tools import compare_scattering_AP_pnc as comparison
from tools import replay_lawrence_device_point_AP_pnc as replay
from tools import replay_lawrence_z4_control_AP_pnc as control


class Z4ControlReplayTests(unittest.TestCase):
    @staticmethod
    def scalar_pencil():
        """Four uncoupled quadratic channels have exact signed roots and work."""
        wave = np.arange(1.0, 5.0)
        omega = 2 * np.pi * control.FREQUENCY
        U = np.tile(np.eye(4) * np.sqrt(2 / (omega * wave)), (1, 2)).astype(complex)
        roots = np.r_[wave, -wave]
        modes = {"wavenumbers_rad_m": roots.astype(complex), "displacements": U,
                 "axial_forces": 1j * roots * U, "positive_indices": np.arange(4),
                 "negative_indices": np.arange(4, 8), "propagating": np.ones(8, dtype=bool),
                 "powers_w": np.r_[np.ones(4), -np.ones(4)], "relative_residuals": np.zeros(8)}
        section = {"coordinates_yz_m": np.zeros((4, 2)), "K0": -np.diag(wave**2),
                   "mass_kg_per_m": np.zeros((4, 4)), "Q": np.eye(4), "G": np.zeros((4, 4))}
        return modes, section

    def test_saved_quadratic_roots_traction_and_flux_require_independent_agreement(self):
        modes, section = self.scalar_pencil()
        incident, returning, metrics = control.replay_modes(modes, section)
        self.assertEqual((incident, returning), (list(range(4)), list(range(4, 8))))
        self.assertEqual(metrics["roots_checked"], 8)
        self.assertLess(metrics["maximum_original_backward_residual"], 1e-15)
        wrong = deepcopy(modes)
        wrong["axial_forces"][:, 0] *= 1.0001
        with self.assertRaisesRegex(ValueError, "traction or signed power"):
            control.replay_modes(wrong, section)
        wrong = deepcopy(modes)
        wrong["negative_indices"] = np.arange(4)
        with self.assertRaisesRegex(ValueError, "incomplete or invalid"):
            control.replay_modes(wrong, section)

    def test_explicit_profile_projection_preserves_default_z6_identity(self):
        modes, _ = self.scalar_pencil()
        incident, returning = tuple(range(4)), tuple(range(4, 8))
        U, F = modes["displacements"], modes["axial_forces"]
        reflection = np.diag([0.1, 0.2, 0.3, 0.4])
        transmission = np.diag(np.sqrt(1 - np.diag(reflection)**2) * np.exp(1j * np.arange(4) / 3))
        vectors = np.vstack((U[:, incident] + U[:, returning] @ reflection, U[:, incident] @ transmission))
        forces = np.vstack((-F[:, incident] - F[:, returning] @ reflection, F[:, incident] @ transmission))
        rhs = np.vstack((-2 * F[:, incident], np.zeros((4, 4))))
        fields = {"incident_powers_w": np.ones(4)}
        for name, roots, block in (("reflection", returning, reflection), ("transmission", incident, transmission)):
            fields[f"{name}_all_mode_indices"] = np.array(roots)
            fields[f"{name}_amplitudes_full"] = fields[f"{name}_amplitudes"] = block
        original_identity = replay.INCIDENT, replay.RETURNING, replay.DOFS
        result = replay.replay_ports(modes, fields, np.arange(4), np.arange(4, 8), vectors, forces,
                                     rhs, np.linalg.norm(rhs, axis=0), incident_roots=incident, returning_roots=returning)
        self.assertEqual(original_identity, (replay.INCIDENT, replay.RETURNING, replay.DOFS))
        np.testing.assert_allclose(result[1][0], np.diag(reflection)**2, atol=1e-15)
        self.assertLess(max(result[2]), 1e-14)
        self.assertEqual(result[-1]["incident_roots"], incident)
        self.assertLess(result[-1]["coherent_coefficient_power_defect_spectral_norm"], 1e-14)

    @staticmethod
    def trace_pair():
        coordinates = np.array([[0, 1, 0, 1], [0, 0, 1, 1]], dtype=float)
        mesh = MeshTri(coordinates, np.array([[0, 0], [1, 3], [3, 2]]))
        coarse = Basis(mesh, ElementTriP1(), intorder=2)
        fine = Basis(mesh.refined(), ElementTriP1(), intorder=2)

        def modes(basis, roots):
            y, z = basis.doflocs
            fields = np.zeros((3, basis.N, 4), dtype=complex)
            fields[0, :, 0], fields[1, :, 1], fields[2, :, 2], fields[0, :, 3] = 1, 1, 1, y + 2 * z
            return comparison.Modes(np.asarray(roots), np.arange(1.0, 5.0), fields)

        reference_modes = {name: modes(coarse, np.arange(4) + offset)
                           for name, offset in (("incoming", 0), ("reflection", 4), ("transmission", 0))}
        candidate_modes = {name: modes(fine, np.arange(4) + offset)
                           for name, offset in (("incoming", 10), ("reflection", 14), ("transmission", 10))}
        amplitudes = np.array([0.1, 0.2, 0.3, 0.4])
        reference_blocks = {"reflection": np.diag(amplitudes), "transmission": np.diag(np.sqrt(1 - amplitudes**2))}
        order = np.array([2, 0, 3, 1])
        phases = {name: np.exp(1j * (np.arange(4) + shift)) for name, shift in
                  (("incoming", 0.3), ("reflection", 1.1), ("transmission", 0.3))}
        for name, original in candidate_modes.items():
            candidate_modes[name] = comparison.Modes(original.roots[order], original.k[order],
                                                     original.displacement[..., order] * phases[name][order])
        candidate_blocks = {name: (block * phases["incoming"][None, :] / phases[name][:, None])[np.ix_(order, order)]
                            for name, block in reference_blocks.items()}
        first = comparison.Section(coarse, np.zeros((3, coarse.N), dtype=int), mass.assemble(coarse).toarray(), "coarse")
        second = comparison.Section(fine, np.zeros((3, fine.N), dtype=int), mass.assemble(fine).toarray(), "fine")
        return (first, reference_modes, reference_blocks), (second, candidate_modes, candidate_blocks)

    def test_four_channel_unequal_trace_permutation_and_complex_gauge(self):
        first, second = self.trace_pair()
        matches, blocks, rows = comparison.compare_blocks(first, second)
        self.assertEqual(matches["incoming"]["candidate_roots"], list(range(10, 14)))
        self.assertEqual(len(rows), 8)
        self.assertTrue(all(min(group["mass_overlaps"]) > 0.999999 for group in matches.values()))
        self.assertLess(max(block["maximum_complex_change"] for block in blocks.values()), 1e-14)
        changed = deepcopy(second)
        changed[2]["transmission"] *= np.exp(0.02j)
        _, blocks, _ = comparison.compare_blocks(first, changed)
        self.assertGreater(blocks["transmission"]["maximum_complex_change"], 0.01)
        self.assertLess(blocks["transmission"]["maximum_power_change"], 1e-14)

    def test_archived_body_identity_and_known_source_exception_are_narrow(self):
        config = {"geometry": {"length_nm": 16650, "thickness_nm": 200}, "mesh": {"thickness_layers": 4},
                  "material": {"density_kg_m3": 5323}, "analysis": {"kind": "scattering", "frequency_GHz": 1}}
        link = {"sha256": "a" * 64, "bytes": 24}
        original = {"configuration": link, "mesh": link, "mesh_checkpoint_summary": link, "K": link, "M": link,
                    "source": {"source_sha256": {control.COORDINATOR: control.COORDINATOR_PAIR[0],
                                               "src/phononic/elasticity_AP_pnc.py": "b" * 64},
                               "packages": {"numpy": "fixture"}}}
        current = deepcopy(original["source"])
        current["source_sha256"][control.COORDINATOR] = control.COORDINATOR_PAIR[1]
        stage = {"status": "operator_staged", "frequency_hz": control.FREQUENCY, "displacement_dofs": control.DOFS,
                 "trace_sha256": control.TRACE_SHA, "body_stage": {"sha256": control.STAGE_SHA},
                 "mesh": original["mesh"], "mesh_summary": original["mesh_checkpoint_summary"],
                 "array_files": {key: original[key] for key in ("K", "M")}, "configuration": deepcopy(config),
                 "implementation": current, **dict.fromkeys(("mesh_generated", "body_assembly_performed",
                    "factorization_performed", "phase_adjustment_applied"), False)}
        stage["configuration"]["analysis"]["frequency_GHz"] = 0.1
        with (patch.object(replay, "check_link", return_value=Path("fixture")), patch.object(control, "read", return_value=config)):
            control.check_stage_identity(stage, original)
            wrong = deepcopy(stage)
            wrong["configuration"]["geometry"]["thickness_nm"] = 201
            with self.assertRaisesRegex(ValueError, "physical inputs"):
                control.check_stage_identity(wrong, original)
            wrong = deepcopy(stage)
            wrong["implementation"]["source_sha256"][control.COORDINATOR] = "0" * 64
            with self.assertRaisesRegex(ValueError, "numerical kernels"):
                control.check_stage_identity(wrong, original)

    def test_pair_uses_actual_assembly_kernels_despite_resource_metadata(self):
        kernels = {f"src/phononic/{name}_AP_pnc.py": "a" * 64
                   for name in (*comparison.CORE, "assembly_batched", "layered")}
        first = {"implementation": {"source_sha256": kernels, "packages": {"numpy": "fixture"}}}
        second = deepcopy(first)
        fixed = {"frequency_hz": control.FREQUENCY, "incident_side": "left",
                 "dimensions_nm": {"thickness_nm": 200}, "numerical_source_sha256": {"assembly": "saved"}}
        other = {**fixed, "numerical_source_sha256": {}}
        pair = control.physical_pair(first, second, fixed, other)
        self.assertEqual(pair["numerical_source_sha256"], kernels)
        wrong = deepcopy(second)
        wrong["implementation"]["source_sha256"]["src/phononic/assembly_batched_AP_pnc.py"] = "b" * 64
        with self.assertRaisesRegex(ValueError, "numerical kernels"):
            control.physical_pair(first, wrong, fixed, other)
        wrong = deepcopy(other)
        wrong["dimensions_nm"]["thickness_nm"] = 201
        with self.assertRaisesRegex(ValueError, "physical inputs"):
            control.physical_pair(first, second, fixed, wrong)

    def test_verified_pair_adapter_rejects_unbound_summary(self):
        kernels = {f"src/phononic/{name}_AP_pnc.py": "a" * 64
                   for name in (*comparison.CORE, "assembly_batched", "layered")}
        first = {"implementation": {"source_sha256": kernels, "packages": {}}, "port_dofs": 675,
                 "configuration": {"mesh": {"thickness_layers": 4}}}
        second = {**first, "port_dofs": 975, "configuration": {"mesh": {"thickness_layers": 6}}}
        fixed = {"frequency_hz": control.FREQUENCY, "incident_side": "left", "numerical_source_sha256": kernels}
        first_record, second_record = ({"summary_sha256": digest * 64, "power_balance_passed": True} for digest in ("b", "c"))
        accepted = {"status": "point_verified_within_saved_discretization", "public_summary_sha256": "c" * 64}
        report = {"status": accepted["status"], "public_summary": {"sha256": "b" * 64, "path": "fixture/summary_AP_pnc.json"}}
        data = self.trace_pair()
        with (patch.object(replay, "read_receipt", return_value=accepted),
              patch.object(control, "artifact", return_value={"sha256": "fixture"}),
              patch.object(comparison, "read_run", side_effect=lambda path: (first, fixed, first_record)
                           if Path(path).name == "fixture" else (second, fixed, second_record)),
              patch.object(comparison, "read_scattering", side_effect=lambda path, *args: data[0]
                           if Path(path).name == "fixture" else data[1])):
            result = control.compare_verified_point(report)
            self.assertTrue(result["mesh_pair_passed"])
            self.assertEqual(len(result["matched_channels"]["incoming"]["mass_overlaps"]), 4)
            self.assertTrue(result["two_stronger_channels_preserved"])
            wrong = deepcopy(report)
            wrong["public_summary"]["sha256"] = "d" * 64
            with self.assertRaisesRegex(ValueError, "exact independently accepted summary"):
                control.compare_verified_point(wrong)


if __name__ == "__main__":
    unittest.main()
