"""Independent finite matching and coefficient fixtures. Athul Prem | _AP_pnc."""
from copy import deepcopy
from pathlib import Path
from tempfile import TemporaryDirectory
from unittest import TestCase
from unittest.mock import patch
import json

import numpy as np

from tools import compare_device_frequency_AP_pnc as comparison


def mode_fixture():
    """Signed roots, permutations and gauges deliberately differ from sorted root IDs."""
    roots, reverse = np.array([21, 4, 17, 8, 30]), np.array([3, 27, 2, 29, 11])
    candidate_roots, candidate_reverse = np.array([5, 25, 9, 12, 1]), np.array([26, 6, 22, 7, 24])
    mass = np.diag([2., 3., 5., 7., 11.])
    fields = np.diag(1 / np.sqrt(np.diag(mass))).astype(complex)
    permutation, reverse_permutation = np.array([3, 1, 4, 0, 2]), np.array([4, 2, 0, 3, 1])
    gauge, reverse_gauge = np.exp(1j * np.array([.4, -.8, 1.4, 2.3, -.2])), np.exp(1j * np.array([-.7, 1.2, 2.1, -.4, .8]))
    waves = np.array([2., 3., 5., 7., 11.])
    first = {"incoming": {"roots": roots, "k": waves, "displacements": fields},
             "reflection": {"roots": reverse, "k": -waves, "displacements": fields.conj()},
             "reverse_partners": dict(zip(map(int, roots), map(int, reverse), strict=True))}
    second = {"incoming": {"roots": candidate_roots, "k": 1.1 * waves[permutation],
                           "displacements": fields[:, permutation] * gauge},
              "reflection": {"roots": candidate_reverse, "k": -1.1 * waves[reverse_permutation],
                             "displacements": fields[:, reverse_permutation] * reverse_gauge}}
    second["reverse_partners"] = {
        int(candidate_roots[i]): int(candidate_reverse[np.flatnonzero(reverse_permutation == original)[0]])
        for i, original in enumerate(permutation)}
    return first, second, mass, permutation, gauge


def response_fixture(modes, reflection, transmission, frequency):
    roots = modes["incoming"]["roots"].tolist()
    reverse = [modes["reverse_partners"][root] for root in roots]
    summary = {"frequency_hz": frequency, "incident_side": "left", "incident_mode_indices": roots,
               "num_incident_channels": 5, "power_balance_passed": True, "run_hash": "synthetic"}
    totals, coefficients = [], []
    for column, incoming in enumerate(roots):
        totals.append({"incident_mode_order": column + 1, "frequency_GHz": frequency / 1e9,
                       "reflection": sum(abs(reflection[:, column])**2), "transmission": sum(abs(transmission[:, column])**2)})
        for kind, outgoing, matrix in (("reflection", reverse, reflection), ("transmission", roots, transmission)):
            for row, root in enumerate(outgoing):
                amplitude = matrix[row, column]
                coefficients.append({"kind": kind, "incident_mode_order": column + 1, "incident_port_index": incoming,
                    "outgoing_port_index": root, "amplitude_real": amplitude.real,
                    "amplitude_imag": amplitude.imag, "power_fraction": abs(amplitude)**2})
    return summary, totals, coefficients, roots, reverse


class DeviceFrequencyTests(TestCase):
    def test_all_complex_coefficients_align_with_independent_input_output_gauges(self):
        first, second, mass, permutation, _ = mode_fixture()
        relation = comparison.relation_from_modes(first, second, mass)
        # Unit column powers provide a deterministic fixture; coherent unitarity is not claimed.
        a = np.eye(5, dtype=complex) * .1j
        a[0, 1], a[1, 0] = .03 + .04j, -.03 + .04j
        t = np.diag(np.sqrt(1 - np.sum(abs(a)**2, axis=0)))
        forward = relation["matched_channels"]["incoming"]
        reflected = relation["matched_channels"]["reflection"]
        # The response CSV stores reflected roots in reciprocal-forward order.
        back_phase_by_root = dict(zip(reflected["candidate_roots"], reflected["candidate_gauge_phase_rad"], strict=True))
        forward_phase_by_root = dict(zip(forward["candidate_roots"], forward["candidate_gauge_phase_rad"], strict=True))
        croot = second["incoming"]["roots"].tolist()
        back_phase = np.array([back_phase_by_root[second["reverse_partners"][root]] for root in croot])
        in_phase = np.array([forward_phase_by_root[root] for root in croot])
        b = a[np.ix_(permutation, permutation)] * np.exp(1j * (in_phase[None, :] - back_phase[:, None]))
        bt = t[np.ix_(permutation, permutation)] * np.exp(1j * (in_phase[None, :] - in_phase[:, None]))
        inputs = [response_fixture(first, a, t, comparison.REFERENCE_HZ),
                  response_fixture(second, b, bt, comparison.CANDIDATE_HZ)]
        responses = [comparison.parse_response(summary, totals[::-1], coefficients[::-1], roots, reverse, frequency)
                     for (summary, totals, coefficients, roots, reverse), frequency in
                     zip(inputs, (comparison.REFERENCE_HZ, comparison.CANDIDATE_HZ), strict=True)]
        selected = ((int(first["incoming"]["roots"][0]), int(first["reflection"]["roots"][1])),
                    (int(first["incoming"]["roots"][1]), int(first["reflection"]["roots"][0])))
        result, rows = comparison.compare_responses(*responses, relation, selected_pair=selected)
        self.assertEqual(len(result["coefficients"]), 50)
        self.assertEqual(len(rows), 28)
        self.assertLess(max(row["absolute_complex_change"] for row in result["coefficients"]), 1e-15)
        self.assertAlmostEqual(result["selected_pair_two_drive_power_sum"]["reference_power"], .005)
        self.assertLess(max(row["absolute_power_change"] for row in rows), 1e-15)
        self.assertEqual(result["per_drive"][4]["quantity"], "total_converted_power")
        with TemporaryDirectory() as temporary:
            path = comparison.write_outputs(temporary, {"phase": "device_frequency_comparison", "status": "synthetic", **result}, rows)
            self.assertTrue(path.is_file())
            self.assertTrue((Path(temporary) / "device_frequency_comparison_AP_pnc.csv").is_file())

    def test_response_corruption_and_wrong_replay_receipts_reject(self):
        first, _, _, _, _ = mode_fixture()
        values = response_fixture(first, np.eye(5, dtype=complex) * .1j,
                                  np.eye(5, dtype=complex) * np.sqrt(.99), comparison.CANDIDATE_HZ)
        summary, totals, coefficients, roots, reverse = values
        for mutation in ("duplicate", "wrong_root", "wrong_power"):
            changed = deepcopy(coefficients)
            if mutation == "duplicate":
                changed[-1] = changed[0]
            elif mutation == "wrong_root":
                changed[0]["outgoing_port_index"] = roots[0]
            else:
                changed[0]["power_fraction"] += .01
            with self.assertRaises(ValueError):
                comparison.parse_response(summary, totals, changed, roots, reverse, comparison.CANDIDATE_HZ)
        with TemporaryDirectory() as temporary:
            path = Path(temporary) / "summary.json"
            path.write_text("{}", encoding="utf-8")
            verification = {"status": "point_verified_within_saved_discretization", "frequency_hz": comparison.CANDIDATE_HZ,
                            "public_summary_sha256": comparison.sha(path),
                            "public_summary": {"path": "summary_AP_pnc.json", "sha256": comparison.sha(path), "bytes": path.stat().st_size},
                            "implementation": {}}
            source_directory = Path(temporary) / "tools"
            source_directory.mkdir()
            for role, filename in (("verifier", "replay_lawrence_frequency_point_AP_pnc.py"),
                                   ("shared_replay", "replay_lawrence_device_point_AP_pnc.py")):
                source = source_directory / filename
                source.write_text(role, encoding="utf-8")
                verification["implementation"][role] = comparison.link(source, temporary)
            for key in ("original_equations_passed", "physical_nodal_force_power_passed", "all_modal_coefficients_passed",
                        "full_p2_field_identity_passed", "coherent_left_incidence_passed"):
                verification[key] = True
            with patch.object(comparison, "REPLAY_SOURCE_SHA256", verification["implementation"]["verifier"]["sha256"]), \
                    patch.object(comparison, "SHARED_REPLAY_SHA256", verification["implementation"]["shared_replay"]["sha256"]):
                comparison.validate_candidate_verification(verification, path, temporary)
                changed = {**verification, "full_p2_field_identity_passed": False}
                with self.assertRaises(ValueError):
                    comparison.validate_candidate_verification(changed, path, temporary)
                with self.assertRaises(ValueError):
                    comparison.validate_candidate_verification({**verification, "public_summary_sha256": "0" * 64}, path, temporary)
                changed = deepcopy(verification)
                changed["implementation"]["verifier"]["sha256"] = "0" * 64
                with self.assertRaisesRegex(ValueError, "reviewed source"):
                    comparison.validate_candidate_verification(changed, path, temporary)
                changed = deepcopy(verification)
                changed["public_summary"]["bytes"] += 1
                with self.assertRaisesRegex(ValueError, "public-summary artifact"):
                    comparison.validate_candidate_verification(changed, path, temporary)

    def test_recomputed_relation_digest_and_replay_receipt_do_not_replace_accepted_bytes(self):
        with TemporaryDirectory() as temporary:
            path = Path(temporary) / "relation.json"
            relation = {"matched_channels": {"incoming": {"candidate_roots": [1, 2, 3, 4, 5],
                                                           "candidate_gauge_phase_rad": [0.] * 5}}}
            relation["content_sha256"] = comparison.digest(relation)
            path.write_text(json.dumps(relation), encoding="utf-8")
            accepted = comparison.sha(path)
            relation["matched_channels"]["incoming"]["candidate_roots"][:2] = [2, 1]
            relation["matched_channels"]["incoming"]["candidate_gauge_phase_rad"][0] = .2
            relation["content_sha256"] = comparison.digest({key: value for key, value in relation.items() if key != "content_sha256"})
            path.write_text(json.dumps(relation), encoding="utf-8")
            with patch.object(comparison, "ACCEPTED_RELATION_SHA256", accepted):
                with self.assertRaisesRegex(ValueError, "accepted executed relation artifact"):
                    comparison.load_relation(path, temporary)
            with self.assertRaisesRegex(ValueError, "point-replay receipt SHA-256"):
                comparison.build_comparison(path, temporary, path, temporary, verification_sha256=accepted)

    def test_relative_power_floor_is_explicit_and_unresolved_relation_stops(self):
        changes = comparison.power_change(0., 2e-14)
        self.assertTrue(changes["denominator_floor_applied"])
        self.assertEqual(changes["relative_denominator"], 1e-12)
        self.assertAlmostEqual(changes["relative_power_change"], .02)
        with self.assertRaisesRegex(ValueError, "Unresolved endpoint"):
            comparison.compare_responses({}, {}, {"status": "unresolved"})

    def test_signed_assignment_uses_fields_and_keeps_complex_gauge(self):
        first, second, mass, permutation, gauge = mode_fixture()
        result = comparison.relation_from_modes(first, second, mass)
        self.assertEqual(result["status"], "matched")
        group = result["matched_channels"]["incoming"]
        expected = np.argsort(permutation)
        self.assertEqual(group["candidate_order"], expected.tolist())
        self.assertEqual(group["candidate_roots"], second["incoming"]["roots"][expected].tolist())
        actual = np.asarray(group["candidate_gauge_complex"])
        np.testing.assert_allclose(actual[:, 0] + 1j * actual[:, 1], gauge[expected], atol=1e-15)
        np.testing.assert_allclose(group["full_mass_overlap_matrix"], np.eye(5)[:, permutation], atol=1e-15)
        self.assertEqual(result["reverse_partner_consistency"]["status"], "consistent")

    def test_weak_or_ambiguous_fields_preserve_full_diagnostic_and_break(self):
        first, second, mass, _, _ = mode_fixture()
        second["incoming"]["displacements"][:, 0] = (first["incoming"]["displacements"][:, 0]
                                                      + first["incoming"]["displacements"][:, 1]) / np.sqrt(2)
        result = comparison.relation_from_modes(first, second, mass)
        self.assertEqual(result["status"], "unresolved")
        group = result["matched_channels"]["incoming"]
        self.assertEqual(np.asarray(group["full_mass_overlap_matrix"]).shape, (5, 5))
        self.assertIn("reason", group)
        self.assertNotIn("candidate_order", group)

    def test_degenerate_signed_wavenumbers_and_wrong_reverse_map_break(self):
        first, second, mass, _, _ = mode_fixture()
        second["incoming"]["k"][1] = second["incoming"]["k"][0]
        self.assertEqual(comparison.relation_from_modes(first, second, mass)["status"], "unresolved")
        first, second, mass, _, _ = mode_fixture()
        a, b = list(second["reverse_partners"])[:2]
        second["reverse_partners"][a], second["reverse_partners"][b] = second["reverse_partners"][b], second["reverse_partners"][a]
        self.assertEqual(comparison.relation_from_modes(first, second, mass)["status"], "unresolved")

    def test_saved_partition_rejects_missing_channel_and_invalid_signed_flux(self):
        with TemporaryDirectory() as temporary:
            path = Path(temporary) / "modes.npz"
            dofs = 5
            values = dict(frequency_hz=comparison.CANDIDATE_HZ,
                wavenumbers_rad_m=np.r_[np.arange(1, 6), -np.arange(1, 6)].astype(complex),
                propagating=np.ones(10, dtype=bool), powers_w=np.r_[np.ones(5), -np.ones(5)],
                relative_residuals=np.zeros(10), positive_indices=np.arange(5), negative_indices=np.arange(5, 10),
                displacements=np.tile(np.eye(5, dtype=complex), (1, 2)))
            values["axial_forces"] = values["displacements"] * (1j * values["powers_w"] / (np.pi * comparison.CANDIDATE_HZ))
            np.savez(path, **values)
            result = comparison.read_signed_modes(path, comparison.CANDIDATE_HZ, np.arange(5), np.arange(5, 10), dofs=dofs)
            self.assertEqual(result["reverse_partners"], dict(zip(range(5), range(5, 10), strict=True)))
            changed = deepcopy(values)
            changed["propagating"][0] = False
            np.savez(path, **changed)
            with self.assertRaisesRegex(ValueError, "complete five-channel"):
                comparison.read_signed_modes(path, comparison.CANDIDATE_HZ, np.arange(5), np.arange(5, 10), dofs=dofs)
            changed = deepcopy(values)
            changed["axial_forces"][0, 0] *= -1
            np.savez(path, **changed)
            with self.assertRaisesRegex(ValueError, "axial-force"):
                comparison.read_signed_modes(path, comparison.CANDIDATE_HZ, np.arange(5), np.arange(5, 10), dofs=dofs)
