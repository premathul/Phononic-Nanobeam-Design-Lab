"""Offline uniform-reference checks. Author: Athul Prem | _AP_pnc."""

import copy
import json
from pathlib import Path
import sys
from tempfile import TemporaryDirectory
import unittest
from unittest.mock import patch

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "tools"))

from compare_uniform_reference_AP_pnc import (
    DEFAULT_REFERENCE, Modes, compare_uniform_blocks, compare_uniform_reference, read_reference,
)


class UniformReferenceTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        record = json.loads(DEFAULT_REFERENCE.read_bytes())
        cls.fixed = {"dimensions_nm": {"length_nm": 1800, "width_nm": 600, "thickness_nm": 200},
                     "material": record["cases"]["8x4"]["inputs"]["material"], "frequency_hz": 1e9,
                     "numerical_source_sha256": record["numerical_source_sha256"]}
        cls.reference, _, cls.qualification = read_reference(DEFAULT_REFERENCE, "8x4", cls.fixed)

    def test_known_uniform_phase_and_full_reflection_survive_signed_channel_reordering(self):
        section, modes = self.reference
        length = 1.8e-6
        for side, direction, forward, backward in (("left", 1, "positive", "negative"),
                                                   ("right", -1, "negative", "positive")):
            with self.subTest(side=side):
                incident, reflected = modes[forward], modes[backward]
                n = len(incident.roots)
                native_k = incident.k + direction * np.arange(1, n + 1) * 1e4
                in_order, ref_order, out_order = (np.roll(np.arange(n), shift) for shift in (1, 2, 3))
                in_phase = np.exp(1j * np.arange(n) * 0.45)
                ref_phase = np.exp(-1j * np.arange(n) * 0.31)
                candidate = {
                    "incoming": Modes(incident.roots[in_order], native_k[in_order],
                                      (incident.displacement * in_phase)[:, :, in_order]),
                    "reflection": Modes(reflected.roots[ref_order], reflected.k[ref_order],
                                        (reflected.displacement * ref_phase)[:, :, ref_order]),
                    "transmission": Modes(incident.roots[out_order], native_k[out_order],
                                          (incident.displacement * in_phase)[:, :, out_order]),
                }
                t = np.diag(np.exp(1j * incident.k * direction * length))
                r = np.zeros((n, n), dtype=complex)
                r[1, 0] = 0.02j
                measured = {"transmission": (t * in_phase[None, :] / in_phase[:, None])[np.ix_(out_order, in_order)],
                            "reflection": (r * in_phase[None, :] / ref_phase[:, None])[np.ix_(ref_order, in_order)]}
                matches, blocks, _, native = compare_uniform_blocks(self.reference, (section, candidate, measured), side, length)
                self.assertGreater(min(matches["reflection"]["mass_overlaps"]), 0.999999)
                self.assertAlmostEqual(blocks["reflection"]["maximum_complex_change"], 0.02)
                self.assertAlmostEqual(blocks["reflection"]["maximum_power_change"], 0.0004)
                self.assertLess(blocks["transmission"]["maximum_complex_change"], 1e-12)
                np.testing.assert_allclose(blocks["transmission"]["same_channel_phase_change_rad"], 0, atol=1e-12)
                expected_phase = n * 1e4 * length
                self.assertAlmostEqual(native["maximum_native_phase_drift_rad"], expected_phase)
                self.assertAlmostEqual(native["maximum_native_transmission_complex_error"], 2 * np.sin(expected_phase / 2))

    def test_port_qualification_is_recomputed_from_independent_pairs_at_run_length(self):
        self.assertEqual(self.qualification["status"], "passed")
        pairs = self.qualification["pairs"]
        self.assertEqual(pairs["width"]["pair"], "6x4_to_8x4")
        self.assertEqual(pairs["thickness"]["pair"], "8x2_to_8x4")
        fixed = copy.deepcopy(self.fixed)
        fixed["dimensions_nm"]["length_nm"] *= 2
        _, _, longer = read_reference(DEFAULT_REFERENCE, "8x4", fixed)
        self.assertEqual(longer["status"], "failed")
        for axis in pairs:
            self.assertAlmostEqual(longer["pairs"][axis]["maximum_phase_indicator_rad"],
                                   2 * pairs[axis]["maximum_phase_indicator_rad"])

    def test_physical_and_source_mismatches_rejected(self):
        for section, key, value, error in (
            ("dimensions_nm", "thickness_nm", 220, "section width or thickness differs"),
            ("material", "C44_Pa", 67e9, "material or crystal orientation differs"),
            ("numerical_source_sha256", "src/phononic/port_modes_AP_pnc.py", "0" * 64, "source differs"),
        ):
            with self.subTest(key=key):
                fixed = copy.deepcopy(self.fixed)
                fixed[section][key] = value
                with self.assertRaisesRegex(ValueError, error):
                    read_reference(DEFAULT_REFERENCE, "8x4", fixed)
        with patch("compare_uniform_reference_AP_pnc.read_run", return_value=({}, {"placed_holes_m": [(6.25e-7, 2.5e-7)]}, {})):
            with self.assertRaisesRegex(ValueError, "uniform scattering run with no hole pattern"):
                compare_uniform_reference(ROOT / "synthetic_patterned_run")

    def test_qualification_rejects_incompatible_predecessors(self):
        with TemporaryDirectory() as temporary:
            reference = Path(temporary) / "mismatched_port_AP_pnc.json"
            for case, section, key, value, error in (
                ("8x2", None, "frequency_hz", 2e9, "frequency differs"),
                ("6x4", "material", "C44_Pa", 67e9, "material or crystal orientation differs"),
                ("8x2", "geometry_nm", "width_nm", 650, "section width or thickness differs"),
                ("6x4", None, "element_order", 1, "requires standalone P2 section inputs"),
                ("8x2", None, "holes", [1], "requires standalone P2 section inputs"),
            ):
                with self.subTest(case=case, key=key):
                    data = json.loads(DEFAULT_REFERENCE.read_bytes())
                    inputs = data["cases"][case]["inputs"]
                    (inputs if section is None else inputs[section])[key] = value
                    reference.write_text(json.dumps(data), encoding="utf-8")
                    with self.assertRaisesRegex(ValueError, f"SAFE case {case} {error}"):
                        read_reference(reference, "8x4", self.fixed)


if __name__ == "__main__":
    unittest.main()
