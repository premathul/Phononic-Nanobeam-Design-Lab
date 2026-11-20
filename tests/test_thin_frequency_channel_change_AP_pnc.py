"""Verify signed temporal power changes. Author: Athul Prem | _AP_pnc."""

from types import SimpleNamespace
import unittest

import numpy as np

from tools import map_scattering_AP_pnc as native
from tools.compare_scattering_AP_pnc import Modes
from tools.thin_frequency_channel_change_AP_pnc import frequency_local_block_changes


def _sample(frequency, physical):
    rng = np.random.default_rng(912 + frequency)
    order = {group: rng.permutation(5) for group in native.GROUPS}
    if frequency == 2:
        order["incoming"] = np.arange(5)
    phases = {group: rng.uniform(-3, 3, size=5) for group in native.GROUPS}
    # Match the same +x eigenbasis between the two ends inside each sample.
    phases["transmission"] = phases["incoming"]
    roots_in = 10 + 20 * frequency + np.arange(5)
    k = (1 + 0.01 * frequency) * np.arange(1, 6, dtype=float)
    modal_watts = {group: rng.uniform(0.4, 1.9, size=5) for group in native.GROUPS}
    modal_watts["transmission"] = modal_watts["incoming"]
    modes, blocks, modal_powers = {}, {}, {}
    for group in native.GROUPS:
        native_to_channel = np.argsort(order[group])
        root_ids = roots_in if group != "reflection" else roots_in + 9
        signed_k = k if group != "reflection" else -k
        vectors = (np.eye(5, dtype=complex) * np.exp(1j * phases[group]))[:, native_to_channel]
        modes[group] = Modes(root_ids[native_to_channel], signed_k[native_to_channel], vectors[None, :, :])
        if group != "incoming":
            ratio = modal_watts[group][:, None] / modal_watts["incoming"]
            native_amplitude = physical[group] * np.exp(1j * (phases["incoming"] - phases[group][:, None]))
            actual = native_amplitude / np.sqrt(ratio)
            index = np.ix_(native_to_channel, np.argsort(order["incoming"]))
            blocks[group] = actual[index]
            modal_powers[group] = (abs(actual)**2 * ratio)[index]
    drives = [{"native_root_id": int(root), **{kind: float(modal_powers[kind][:, i].sum()) for kind in modal_powers}}
              for i, root in enumerate(modes["incoming"].roots)]
    return SimpleNamespace(modes=modes, blocks=blocks, modal_powers=modal_powers,
                           record={"native_drives": drives, "mesh_hash": "same", "reference_planes_x_m": [0, 1e-6]})


class ThinFrequencyChannelChangeTests(unittest.TestCase):
    def setUp(self):
        rng = np.random.default_rng(287)
        self.physical = [{kind: rng.normal(size=(5, 5)) * 0.1 + 1j * rng.normal(size=(5, 5)) * 0.1
                          for kind in ("transmission", "reflection")} for _ in range(3)]
        self.physical[0]["transmission"][2, 2] = 0.34
        self.physical[1]["transmission"][2, 2] = 0.34 * np.exp(-0.44j)
        self.physical[0]["transmission"][4, 4] = 0.0
        self.points = {(0, slot): _sample(slot, values) for slot, values in enumerate(self.physical)}
        self.order = {}
        for key, point in self.points.items():
            anchor = native.match_modes(self.points[0, 2].modes, point.modes, np.eye(5))
            self.assertEqual(anchor["status"], "matched")
            self.order[key] = {name: group["candidate_order"] for name, group in anchor["groups"].items()}
        self.matches = {((0, k), (0, k + 1)): native.match_modes(self.points[0, k].modes, self.points[0, k + 1].modes,
                                                                 np.eye(5))["groups"] for k in range(2)}

    def test_selected_transmission_and_reflected_conversions_match_independent_powers(self):
        record = frequency_local_block_changes(self.points, self.matches, self.order, section_mass=np.eye(5))
        self.assertEqual(len(record["edges"]), 2)
        for edge_index, edge in enumerate(record["edges"]):
            t, r = (100 * (abs(self.physical[edge_index + 1][kind])**2 - abs(self.physical[edge_index][kind])**2)
                    for kind in ("transmission", "reflection"))
            focus = np.zeros((5, 5))
            for term in edge["focus_transmission_terms"]:
                i, j = term["outgoing_drive"] - 1, term["incident_drive"] - 1
                self.assertIn(i, (2, 4))
                self.assertIn(j, (2, 4))
                self.assertAlmostEqual(term["delta_T_percentage_points"], t[i, j], places=12)
                focus[i, j] = t[i, j]
            convert = r - np.diag(r.diagonal())
            np.testing.assert_allclose(edge["reflection_conversion_delta_percentage_points_by_outgoing_and_incident"],
                                       convert, atol=1e-12, rtol=0)
            for j, power in enumerate(edge["drives"]):
                self.assertAlmostEqual(power["delta_T_percentage_points"], t[:, j].sum(), places=12)
                self.assertAlmostEqual(power["focused_delta_T_percentage_points"], focus[:, j].sum(), places=12)
                self.assertAlmostEqual(power["remaining_delta_T_percentage_points"], (t - focus)[:, j].sum(), places=12)
                self.assertAlmostEqual(power["delta_R_percentage_points"], r[:, j].sum(), places=12)
                self.assertAlmostEqual(power["same_channel_delta_R_percentage_points"], r[j, j], places=12)
                self.assertAlmostEqual(power["converted_delta_R_percentage_points"], convert[:, j].sum(), places=12)
        pure_phase, *_, zero_amplitude = record["edges"][0]["focus_transmission_terms"]
        self.assertAlmostEqual(pure_phase["wrapped_phase_change_degrees"], np.rad2deg(-0.44), places=12)
        self.assertIsNone(zero_amplitude["wrapped_phase_change_degrees"])

    def test_inconsistent_saved_power_and_mode_match_fail(self):
        row = self.points[0, 0].record["native_drives"][0]
        saved = row["reflection"]
        row["reflection"] += 0.01
        with self.assertRaisesRegex(ValueError, "does not reproduce the saved native totals"):
            frequency_local_block_changes(self.points, self.matches, self.order, section_mass=np.eye(5))
        row["reflection"] = saved
        self.matches[(0, 0), (0, 1)]["reflection"]["candidate_order"].reverse()
        with self.assertRaisesRegex(ValueError, "edge reflection phase match differs"):
            frequency_local_block_changes(self.points, self.matches, self.order, section_mass=np.eye(5))

    def test_opposite_wavenumber_without_conjugate_trace_fails(self):
        modes = self.points[0, 0].modes
        returning = modes["reflection"]
        modes["reflection"] = Modes(returning.roots, returning.k, np.roll(returning.displacement, 1, axis=-1))
        with self.assertRaisesRegex(ValueError, "Conjugate return traces and opposite-wavenumber partners disagree"):
            frequency_local_block_changes(self.points, self.matches, self.order, section_mass=np.eye(5))


if __name__ == "__main__":
    unittest.main()
