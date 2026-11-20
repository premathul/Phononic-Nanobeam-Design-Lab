"""Test signed grid alignment and finite-difference bookkeeping. Author: Athul Prem | _AP_pnc."""

import unittest
from copy import deepcopy
from types import SimpleNamespace

import numpy as np

from tools import compare_thin_mesh_frequency_AP_pnc as grid


def permutations():
    random = np.random.default_rng(25)
    native_orders = {(m, f): {g: random.permutation(5).tolist() for g in grid.native.GROUPS}
                     for m in grid.AXIS for f in grid.AXIS}
    matches = {(a, b): {g: {"candidate_order": [native_orders[b][g].index(root) for root in native_orders[a][g]]}
                       for g in grid.native.GROUPS} for a, b in grid._edges()}
    return native_orders, matches


class ThinMeshFrequencyTests(unittest.TestCase):
    def test_grid_uses_1ghz_anchor_and_requires_commuting_signed_matches(self):
        native_orders, matches = permutations()
        orders = grid._channel_orders(matches)
        self.assertEqual(len(orders), 9)
        for point, groups in native_orders.items():
            for group, local in groups.items():
                self.assertEqual(orders[point][group],
                                 [local.index(root) for root in native_orders[grid.ANCHOR][group]])
        wrong = deepcopy(matches)
        last_square = wrong[(2, 1), (2, 2)]["reflection"]["candidate_order"]
        last_square[0], last_square[1] = last_square[1], last_square[0]
        with self.assertRaisesRegex(ValueError, "signed reflection.*depends on the path"):
            grid._channel_orders(wrong)

    def test_power_interactions_use_native_matched_powers_and_original_gate(self):
        native_orders, matches = permutations()
        orders = grid._channel_orders(matches)
        anchors = native_orders[grid.ANCHOR]["incoming"]
        points = {}
        for (m, f), groups in native_orders.items():
            rows = []
            for i, label in enumerate(groups["incoming"]):
                drive = anchors.index(label) + 1
                fraction = 0.4 + 0.05 * drive + (0.02 * f if drive == 3 else 0)
                if drive == 3:
                    fraction += [0.0, 0.04, 0.08][m] * f**2
                rows.append({"drive": i + 1, "native_root_id": label, "k_rad_m": label + 1.0,
                             "transmission": fraction, "reflection": 1 - fraction})
            points[m, f] = SimpleNamespace(record={"native_drives": rows}, evidence={"frequency_hz": grid.FREQUENCIES[f]})
        layers = grid._power_levels(points, orders, [{"central_max_edge_nm": a} for a in grid.TARGETS])
        self.assertEqual([row["native_root_id"] for row in layers[1]["matched_drives"][2]["samples"]],
                         [anchors[2]] * 3)
        amplitudes = {(m, f): {"transmission": np.array([0.003, 0.012, 0.021, 0.004, 0.003]),
                               "reflection": np.zeros(5)} for m in range(2) for f in range(3)}
        increments = grid._spatial_increments(layers, amplitudes)
        row = increments[0]["per_drive"][2]
        self.assertAlmostEqual(row["endpoint_change_difference_percentage_points"]["transmission"], 16.0)
        self.assertAlmostEqual(row["midpoint_departure_difference_percentage_points"]["transmission"], -4.0)
        self.assertAlmostEqual(row["midpoint_departure_difference_percentage_points"]["reflection"], 4.0)
        self.assertAlmostEqual(increments[0]["equal_incoherent_mean"]["endpoint_change_difference_percentage_points"]
                               ["transmission"], 3.2)
        early, middle, late = increments[0]["transmission_gates"]
        self.assertTrue(early["all_drives_passed"])
        self.assertFalse(early["all_drives_complex_amplitude_passed"])
        self.assertEqual(middle["failed_reference_drives"], [3])
        self.assertAlmostEqual(late["maximum_absolute_transmission_change_fraction"], 0.16)
        self.assertAlmostEqual(late["maximum_absolute_complex_amplitude_change"], 0.021)

    def test_complex_spatial_gate_applies_incoming_and_outgoing_gauges_once(self):
        _, matches = permutations()
        orders = grid._channel_orders(matches)
        random = np.random.default_rng(26)
        gauges = {key: {group: random.random(5) for group in grid.native.GROUPS} for key in orders}
        for (a, b), groups in matches.items():
            for group, row in groups.items():
                row["candidate_gauge_phase_rad"] = (gauges[b][group][row["candidate_order"]] - gauges[a][group]).tolist()
        canonical = {q: random.random((5, 5)) + 1j * random.random((5, 5)) for q in grid.QUANTITIES}
        addition = np.diag([0.0, 0.001, 0.012, 0.0, 0.003])
        points = {}
        for key, groups in orders.items():
            incoming = np.argsort(groups["incoming"])
            blocks = {}
            for quantity in grid.QUANTITIES:
                matrix = canonical[quantity] + (key[0] * addition if quantity == "transmission" else 0)
                outgoing = np.argsort(groups[quantity])
                blocks[quantity] = (matrix[np.ix_(outgoing, incoming)]
                                    * np.exp(1j * gauges[key]["incoming"] - 1j * gauges[key][quantity][:, None]))
            points[key] = SimpleNamespace(blocks=blocks)
        results = grid._spatial_amplitudes(points, matches, orders)
        self.assertEqual(len(results), 6)
        for row in results.values():
            np.testing.assert_allclose(row["transmission"], addition.diagonal(), rtol=0, atol=1e-15)
            np.testing.assert_allclose(row["reflection"], np.zeros(5), rtol=0, atol=1e-15)

    def test_best_constant_offset_uses_sample_extrema_and_detects_unresolved_shape(self):
        def summary(values):
            return grid._power_summary([{"frequency_hz": f, "transmission_fraction": t, "reflection_fraction": 1 - t}
                                        for f, t in zip(grid.FREQUENCIES, values, strict=True)])

        before = summary([0.4, 0.4, 0.4])
        after = summary([0.47, 0.45, 0.5])
        fit = grid._change(before, after)["constant_transmission_offset_test"]
        self.assertAlmostEqual(fit["best_constant_offset_percentage_points"], 7.5)
        self.assertAlmostEqual(fit["minimum_possible_max_absolute_residual_percentage_points"], 2.5)
        self.assertEqual(fit["minimum_difference_frequencies_hz"], [997_500_000])
        self.assertEqual(fit["maximum_difference_frequencies_hz"], [1_000_000_000])
        self.assertFalse(fit["residual_floor_meets_spatial_transmission_limit"])


if __name__ == "__main__":
    unittest.main()
