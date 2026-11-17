"""Pinned h100 provenance and fixed-port comparison checks. Athul Prem | _AP_pnc."""

from contextlib import nullcontext
from copy import deepcopy
from pathlib import Path
from unittest.mock import patch
import unittest

import numpy as np
from skfem import Basis, ElementTriP1, MeshTri
from skfem.models.poisson import mass

from tools import compare_scattering_AP_pnc as comparison
from tools import replay_lawrence_h100_control_AP_pnc as control


class H100ControlReplayTests(unittest.TestCase):
    @staticmethod
    def identities():
        old = {"src/phononic/elasticity_AP_pnc.py": "a" * 64}
        new = dict(old)
        for key, (archived, current) in control.SOURCE_CHANGES.items():
            if archived is not None:
                old[key] = archived
            new[key] = current
        return {"source_sha256": old, "packages": {"numpy": "fixture"}, "platform": "Windows-fixture"}, {
                "source_sha256": new, "packages": {"numpy": "fixture"}, "platform": "Linux-fixture"}

    def test_exact_historical_source_changes_reject_other_kernels_or_packages(self):
        old, current = self.identities()
        changes = control.verify_sources(old, current)
        self.assertEqual(set(changes), set(control.SOURCE_CHANGES))
        wrong = deepcopy(current)
        wrong["source_sha256"]["src/phononic/elasticity_AP_pnc.py"] = "b" * 64
        with self.assertRaisesRegex(ValueError, "unreviewed source change"):
            control.verify_sources(old, wrong)
        wrong = deepcopy(current)
        wrong["packages"]["numpy"] = "different"
        with self.assertRaisesRegex(ValueError, "different numerical packages"):
            control.verify_sources(old, wrong)
        wrong = deepcopy(current)
        wrong["source_sha256"].pop("src/phononic/resumable_scattering_AP_pnc.py")
        with self.assertRaisesRegex(ValueError, "unreviewed source change"):
            control.verify_sources(old, wrong)

    def test_archived_body_configuration_trace_and_matrix_are_fixed(self):
        old, current = self.identities()
        link = {"bytes": 123, "sha256": "a" * 64}
        original = {"source": old, "mesh": link, "K": link, "M": link}
        configuration = {"geometry": {"thickness_nm": 200}, "mesh": {"mesh_size_nm": 100, "thickness_layers": 4},
                         "material": {"density_kg_m3": 5323}, "analysis": {"frequency_GHz": 1, "incident_side": "left"}}
        stage = {"status": "operator_staged", "frequency_hz": control.FREQUENCY,
                 "displacement_dofs": control.DOFS, "trace_sha256": control.TRACE_SHA,
                 "body_stage": {"sha256": control.STAGE_SHA}, "mesh": link,
                 "body_assembly_source": old,
                 "body_source_compatibility": {"historical_execution_source_pairs": {
                     key: list(pair) for key, pair in control.SOURCE_CHANGES.items()}},
                 "mesh_summary": {"sha256": control.MESH_SUMMARY_SHA}, "array_files": {"K": link, "M": link},
                 "implementation": current, "configuration": deepcopy(configuration),
                 **dict.fromkeys(("mesh_generated", "body_assembly_performed", "section_assembly_performed",
                    "safe_eigensolve_performed", "factorization_performed", "phase_adjustment_applied"), False)}
        stage["configuration"]["analysis"]["frequency_GHz"] = 0.1
        control.check_body(stage, original, configuration)
        wrong = deepcopy(stage)
        wrong["trace_sha256"] = "b" * 64
        with self.assertRaisesRegex(ValueError, "exact archived body"):
            control.check_body(wrong, original, configuration)
        wrong = deepcopy(stage)
        wrong["array_files"]["K"] = {"bytes": 124, "sha256": "a" * 64}
        with self.assertRaisesRegex(ValueError, "exact archived body"):
            control.check_body(wrong, original, configuration)
        wrong = deepcopy(stage)
        wrong["configuration"]["geometry"]["thickness_nm"] = 201
        with self.assertRaisesRegex(ValueError, "physical inputs"):
            control.check_body(wrong, original, configuration)

    def test_shared_safe_uses_h100_global_map_and_rejects_mode_substitution(self):
        roots, reverse = [0, 1, 2, 3], [4, 5, 6, 7]
        mode_link = {"bytes": 32, "sha256": control.MODES_SHA, "path": "modes-100000000_AP_pnc.npz"}
        mapping_link = {"bytes": 12, "sha256": "c" * 64}
        stage = {"safe_receipt": {"sha256": control.H80_SAFE_SHA}, "safe_modes": mode_link,
                 "safe_section": {"sha256": control.z4.SECTION_SHA}, "archived_safe_modes": mapping_link,
                 "section_receipt": {"sha256": control.z4.SECTION_RECEIPT_SHA}, "safe_reference_ports": mapping_link,
                 "incident_port_roots": roots, "backward_port_roots": reverse}
        safe = {"status": "safe_ready", "frequency_hz": control.FREQUENCY, "safe_modes": mode_link,
                "archived_safe_modes": mapping_link,
                "incident_port_roots": roots, "backward_port_roots": reverse}
        accepted = {"status": "point_verified_within_saved_discretization",
                    "phase_receipts": {"safe": {"sha256": control.H80_SAFE_SHA}},
                    "coherent_left_incidence": {"incident_roots": roots, "reflected_roots": reverse}}
        section = {"coordinates_yz_m": np.zeros((4, 2)), "component_dofs_xyz": np.array([[0], [1], [2]])}
        mapping = {"left_dofs": np.arange(4) + 100, "right_dofs": np.arange(4) + 200}
        modes = {key: np.zeros(8) for key in control.z4.FIELDS}
        modes.update(frequency_hz=np.asarray(control.FREQUENCY), positive_indices=np.array(roots),
                     negative_indices=np.array(reverse), propagating=np.ones(8, dtype=bool))
        with (patch.object(control.np, "load", side_effect=[nullcontext(section), nullcontext(mapping), nullcontext(modes)]),
              patch.object(control.replay, "check_link", return_value=Path("fixture"))):
            mapped, _, incident, returning = control.shared_ports(stage, {"port_modes": mapping_link}, safe, accepted)
        np.testing.assert_array_equal(mapped["left_dofs"], mapping["left_dofs"])
        self.assertEqual((incident, returning), (roots, reverse))
        wrong = deepcopy(stage)
        wrong["safe_modes"] = {**mode_link, "sha256": "d" * 64}
        with self.assertRaisesRegex(ValueError, "independently accepted h80 SAFE"):
            control.shared_ports(wrong, {"port_modes": mapping_link}, safe, accepted)

    @staticmethod
    def data_pair():
        mesh = MeshTri(np.array([[0, 1, 0, 1], [0, 0, 1, 1]], dtype=float), np.array([[0, 0], [1, 3], [3, 2]]))
        basis = Basis(mesh, ElementTriP1(), intorder=2)
        field = np.zeros((3, basis.N, 4), dtype=complex)
        field[0, :, 0], field[1, :, 1], field[2, :, 2], field[0, :, 3] = 1, 1, 1, basis.doflocs[0] + basis.doflocs[1]
        first_modes, second_modes = {}, {}
        permutation = np.array([2, 0, 3, 1])
        phases = {name: np.exp(1j * (np.arange(4) + shift)) for name, shift in
                  (("incoming", 0.3), ("reflection", 1.1), ("transmission", 0.3))}
        for name, offset in (("incoming", 0), ("reflection", 4), ("transmission", 0)):
            roots = np.arange(4) + offset
            k = np.arange(1.0, 5.0) * (-1 if name == "reflection" else 1)
            first_modes[name] = comparison.Modes(roots, k, field)
            second_modes[name] = comparison.Modes((roots + 10)[permutation], k[permutation],
                                                   field[..., permutation] * phases[name][permutation])
        powers = np.array([0.02, 0.015, 0.0002, 0.00015])
        first_blocks = {"reflection": np.diag(np.sqrt(powers)), "transmission": np.diag(np.sqrt(1 - powers))}
        second_blocks = {name: (block * phases["incoming"][None, :] / phases[name][:, None])[np.ix_(permutation, permutation)]
                         for name, block in first_blocks.items()}
        mass_matrix = mass.assemble(basis).toarray()
        sections = [comparison.Section(basis, np.zeros((3, basis.N), dtype=int), mass_matrix, label) for label in ("h100", "h80")]
        return (sections[0], first_modes, first_blocks), (sections[1], second_modes, second_blocks)

    def test_signed_pair_gauges_and_weak_relative_reflection_keep_absolute_gates(self):
        first, second = self.data_pair()
        result = control.compare_blocks(first, second)
        self.assertTrue(result["mesh_pair_passed"])
        self.assertTrue(result["two_stronger_channels_preserved"])
        self.assertLess(result["maximum_complex_change"], 1e-14)
        changed = deepcopy(second)
        # Candidate channel 13 is stored at row/column 2 after the permutation.
        changed[2]["reflection"][2, 2] *= np.sqrt(1.1)
        result = control.compare_blocks(first, changed)
        diagnostic = result["weak_reflection_diagnostic"]
        self.assertIsNone(diagnostic["relative_acceptance_gate"])
        self.assertAlmostEqual(diagnostic["channels"][3]["signed_relative_change"], 0.1)
        self.assertTrue(result["mesh_pair_passed"])
        changed[2]["transmission"] *= np.exp(0.02j)
        self.assertFalse(control.compare_blocks(first, changed)["complex_passed"])

    def test_comparison_rejects_an_unbound_summary_before_loading_fields(self):
        accepted = {"status": "point_verified_within_saved_discretization", "public_summary": {"sha256": "c" * 64}}
        report = {"status": accepted["status"], "public_summary": {"path": "fixture/summary_AP_pnc.json", "sha256": "b" * 64}}
        with (patch.object(control.replay, "read_receipt", return_value=accepted),
              patch.object(comparison, "read_run", side_effect=[({}, {}, {"summary_sha256": "d" * 64}),
                                                              ({}, {}, {"summary_sha256": "c" * 64})]),
              patch.object(comparison, "read_scattering", side_effect=AssertionError("unverified fields loaded"))):
            with self.assertRaisesRegex(ValueError, "exact independently accepted summary"):
                control.compare_verified_point(report)


if __name__ == "__main__":
    unittest.main()
