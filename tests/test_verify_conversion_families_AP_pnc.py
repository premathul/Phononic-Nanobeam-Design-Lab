"""Independent five-channel adapter checks. Athul Prem | _AP_pnc."""

from copy import deepcopy
import json
from pathlib import Path
from tempfile import TemporaryDirectory
import unittest
from unittest.mock import patch

import numpy as np

from tools import verify_conversion_families_AP_pnc as verify


def section_fixture():
    vertices = np.array([[0., 0.], [1e-6, 0.], [1e-6, 2e-7], [0., 2e-7]])
    triangles = np.array([[0, 1, 2], [0, 2, 3]])
    nodes = np.vstack((vertices, [(vertices[a] + vertices[b]) / 2 for a, b in ((0, 1), (1, 2), (0, 2), (2, 3), (0, 3))]))
    components = np.arange(3 * len(nodes)).reshape(-1, 3).T
    fields = np.zeros((3 * len(nodes), 5), dtype=complex)
    for component in range(3):
        fields[components[component], component] = 1
    fields[components[1], 3], fields[components[2], 3] = -(nodes[:, 1] - 1e-7), nodes[:, 0] - 5e-7
    fields[components[0], 4] = nodes[:, 0] - 5e-7
    return dict(vertices_yz_m=vertices, triangles=triangles, coordinates_yz_m=np.repeat(nodes, 3, axis=0),
                component_dofs_xyz=components, displacements=fields, density_kg_m3=5323.)


def response_fixture():
    roots, reverse = [1, 3, 5, 7, 9], [18, 10, 16, 12, 14]
    modes = [dict(root=root, reverse_root=partner) for root, partner in zip(roots, reverse)]
    R = np.eye(5, dtype=complex) * .1
    R[1, 4], R[4, 1] = .2j, -.2
    T = np.diag(np.sqrt(1 - np.sum(abs(R)**2, axis=0)))
    totals = [dict(incident_mode_order=str(i + 1), frequency_GHz="1", reflection=str(float(sum(abs(R[:, i])**2))),
                   transmission=str(float(sum(abs(T[:, i])**2)))) for i in range(5)]
    coefficients = []
    for kind, block, outgoing in (("reflection", R, reverse), ("transmission", T, roots)):
        for i in range(5):
            for j in range(5):
                coefficients.append(dict(kind=kind, incident_mode_order=str(j + 1), incident_port_index=str(roots[j]),
                    outgoing_port_index=str(outgoing[i]), amplitude_real=str(block[i, j].real),
                    amplitude_imag=str(block[i, j].imag), power_fraction=str(abs(block[i, j])**2)))
    summary = dict(frequency_hz=1e9, incident_side="left", incident_mode_indices=roots, num_incident_channels=5,
                   power_balance_passed=True, run_hash="synthetic")
    return summary, totals, coefficients, modes


def pair_publication(response):
    modes = response["rows"]
    lookup = {(r["incident_root"], r["outgoing_root"]): r for r in response["coefficients"] if r["kind"] == "reflection"}
    ranked = []
    for i in range(5):
        for j in range(i + 1, 5):
            first, second = modes[i], modes[j]
            directions = [lookup[first["root"], second["reverse_root"]], lookup[second["root"], first["reverse_root"]]]
            ranked.append(dict(forward_roots=[first["root"], second["root"]], reverse_roots=[first["reverse_root"], second["reverse_root"]],
                               directions=directions, two_direction_power_sum=sum(row["power_fraction"] for row in directions)))
    ranked.sort(key=lambda row: row["two_direction_power_sum"], reverse=True)
    return dict(selection="Largest sum of the two off-diagonal reflected powers; descriptive ranking only.",
                all_offdiagonal_reflection_zero=False, selected_pair=ranked[0], ranked_pairs=ranked)


class IndependentConversionFamiliesTests(unittest.TestCase):
    def test_reverse_partners_use_wavenumbers_instead_of_root_adjacency(self):
        waves, powers, propagating = np.zeros(20, dtype=complex), np.zeros(20), np.zeros(20, dtype=bool)
        positive, negative = [1, 3, 5, 7, 9], [18, 10, 16, 12, 14]
        waves[positive], waves[negative] = np.arange(1, 6), -np.arange(1, 6)
        powers[positive], powers[negative], propagating[positive + negative] = 1, -1, True
        self.assertEqual(verify.reverse_partners(waves, powers, positive, negative[::-1], propagating), negative)
        bad = waves.copy()
        bad[negative[-1]] = bad[negative[0]]
        with self.assertRaisesRegex(ValueError, "absent or ambiguous"):
            verify.reverse_partners(bad, powers, positive, negative, propagating)

    def test_five_mode_signed_match_checks_permutation_and_complex_gauge(self):
        source, target = section_fixture(), section_fixture()
        permutation = [4, 2, 0, 3, 1]
        phases = np.exp(1j * np.array([.3, -1.2, 2.1, np.pi, -.8]))
        target["displacements"] = target["displacements"][:, permutation] * phases
        order = np.argsort(permutation)
        roots_a, roots_b = [1, 3, 5, 7, 9], [21, 23, 25, 27, 29]
        waves_a, waves_b = np.arange(1., 6.), np.arange(11., 16.)
        accepted = dict(reference_roots=roots_a, candidate_roots=np.asarray(roots_b)[order].tolist(), mass_overlaps=[1.] * 5,
                        candidate_gauge_phase_rad=(np.angle(phases[order]) + 2 * np.pi).tolist(),
                        reference_wavenumber_rad_m=waves_a.tolist(), candidate_wavenumber_rad_m=waves_b[order].tolist())
        actual_order, overlaps, gauge = verify.signed_match(source, target, accepted, roots_a, roots_b, waves_a, waves_b)
        np.testing.assert_array_equal(actual_order, order)
        np.testing.assert_allclose(overlaps, 1, atol=1e-14)
        np.testing.assert_allclose(gauge, phases[order], atol=1e-14)
        bad = deepcopy(accepted)
        bad["candidate_gauge_phase_rad"][0] += .02
        with self.assertRaisesRegex(ValueError, "comparison gauge"):
            verify.signed_match(source, target, bad, roots_a, roots_b, waves_a, waves_b)

    def test_complete_signed_coefficients_recover_conversion_under_csv_permutation(self):
        summary, totals, coefficients, modes = response_fixture()
        result = verify.parse_response_tables(summary, totals[::-1], coefficients[::-1], modes)
        self.assertEqual(len(result["coefficients"]), 50)
        np.testing.assert_allclose([r["converted_R"] for r in result["rows"]], [0, .04, 0, 0, .04], atol=1e-15)
        self.assertTrue(all(row["converted_T"] == 0 for row in result["rows"]))
        np.testing.assert_allclose(result["blocks"]["reflection"][[1, 4], [4, 1]], [.2j, -.2])

    def test_coefficient_adapter_rejects_wrong_role_duplicate_nonfinite_and_false_power(self):
        summary, totals, coefficients, modes = response_fixture()
        for field, value in (("outgoing_port_index", "3"), ("incident_port_index", "18"),
                             ("amplitude_real", "nan"), ("power_fraction", ".7")):
            bad = deepcopy(coefficients)
            bad[0][field] = value
            with self.assertRaises(ValueError):
                verify.parse_response_tables(summary, totals, bad, modes)
        bad = deepcopy(coefficients)
        bad[-1] = bad[0]
        with self.assertRaisesRegex(ValueError, "Repeated"):
            verify.parse_response_tables(summary, totals, bad, modes)
        bad_summary = dict(summary, incident_mode_indices=summary["incident_mode_indices"][::-1])
        with self.assertRaisesRegex(ValueError, "summary/root"):
            verify.parse_response_tables(bad_summary, totals, coefficients, modes)

    def test_dominant_pair_checks_both_directions_and_all_ranked_pairs(self):
        response = verify.parse_response_tables(*response_fixture())
        expected = pair_publication(response)
        document = dict(dominant_conversion_pair={"h80_z4": expected})
        selected = verify.verify_dominant_pair(document, {"h80_z4": response})["h80_z4"]
        self.assertEqual(selected["forward_roots"], [3, 9])
        self.assertAlmostEqual(selected["two_direction_power_sum"], .08)
        for field in ("forward_roots", "two_direction_power_sum", "directions"):
            bad = deepcopy(document)
            pair = bad["dominant_conversion_pair"]["h80_z4"]["selected_pair"]
            if field == "forward_roots":
                pair[field][0] = 1
            elif field == "directions":
                pair[field][0]["amplitude_imag"] += .01
            else:
                pair[field] += .01
            with self.assertRaisesRegex(ValueError, "dominant converted pair"):
                verify.verify_dominant_pair(bad, {"h80_z4": response})

    def test_manifest_rejects_accepted_artifacts_swapped_between_roles(self):
        with TemporaryDirectory() as temporary:
            root = Path(temporary).resolve()
            for filename in ("first.json", "second.json"):
                (root / filename).write_text(json.dumps({"distinct": filename}), encoding="utf-8")
            pins = {role: (filename, verify.quadrature.sha256(root / filename)) for role, filename in
                    (("first_role", "first.json"), ("second_role", "second.json"))}
            document = {"inputs": [dict(role=role, path=filename, sha256=digest, bytes=(root / filename).stat().st_size)
                                     for role, (filename, digest) in pins.items()]}
            with patch.dict(verify.ROLE_PINS, pins, clear=True):
                self.assertEqual(set(verify.verified_inputs(document, root)), set(pins))
                bad = deepcopy(document)
                bad["inputs"][0].update(path=document["inputs"][1]["path"], sha256=document["inputs"][1]["sha256"],
                                        bytes=document["inputs"][1]["bytes"])
                with self.assertRaisesRegex(ValueError, "scientific role identity"):
                    verify.verified_inputs(bad, root)


if __name__ == "__main__":
    unittest.main()
