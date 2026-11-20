"""Check endpoint-only candidate comparisons using the compact archive. Athul Prem | _AP_pnc."""

import json
from pathlib import Path
from shutil import copyfile
from tempfile import TemporaryDirectory
import unittest

from tools import compare_snapshot_midpoint_AP_pnc as replay
from tools import read_thin_device_grid_AP_pnc as grid


class CompareSnapshotMidpointTests(unittest.TestCase):
    def setUp(self):
        temporary = TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        self.root = Path(temporary.name).resolve()
        source = grid.load_thin_grid(replay.ROOT)
        paths = [Path(row["path"]).relative_to(replay.ROOT) for row in source["sources"]]
        for path in (*paths, *(replay.RECEIPTS / name for name in replay.FILES)):
            copy = self.root / path
            copy.parent.mkdir(parents=True, exist_ok=True)
            copyfile(replay.ROOT / path, copy)

    def _change_receipt(self, mutate):
        path = self.root / replay.RECEIPTS / replay.FILES[0]
        saved = json.loads(path.read_bytes())
        mutate(saved)
        path.write_text(json.dumps(saved, allow_nan=False), encoding="utf-8")

    def test_reproduces_direct_gates_and_beats_linear_power_on_compact_sources(self):
        result = replay.compare(self.root)
        self.assertEqual(result["frequency_hz"], 997_500_000)
        self.assertTrue(result["all_spatial_gates_agree"])
        expected = ((0.21420145417, 0.28552550366), (0.62257986956, 0.74123672821),
                    (0.75365468019, 1.28009065478))
        for level, (snapshot, linear) in zip(result["mesh_levels"], expected, strict=True):
            errors = level["maximum_absolute_transmission_error_percentage_points"]
            self.assertAlmostEqual(errors["snapshot"], snapshot, delta=1e-8)
            self.assertAlmostEqual(errors["linear_endpoint"], linear, delta=1e-8)
        maxima = (0.75593306196, 0.22141318336)
        for edge, maximum in zip(result["spatial_edges"], maxima, strict=True):
            for kind in ("direct", "snapshot"):
                rows = [(row["reference_drive"], row[kind]) for row in edge["drives"]]
                self.assertEqual([drive for drive, row in rows if not row["total_power_gate_passed"]], [3, 5])
                self.assertFalse(any(row["complex_amplitude_gate_passed"] for _, row in rows))
            computed = max(abs(value) for row in edge["drives"]
                           for value in row["snapshot"]["maximum_absolute_complex_change_by_block"].values())
            self.assertAlmostEqual(computed, maximum, delta=1e-8)

    def test_rejects_an_incorrect_training_source_identity(self):
        self._change_receipt(lambda row: row["inputs"]["endpoints"][0]["original_equations"].update(rhs_sha256="bad"))
        with self.assertRaisesRegex(ValueError, "inputs differ from the accepted"):
            replay.compare(self.root)

    def test_rejects_a_reordered_native_reflection_basis(self):
        self._change_receipt(lambda row: row["native_coefficients"]["reflection_roots_by_row"].reverse())
        with self.assertRaisesRegex(ValueError, "coefficient root order"):
            replay.compare(self.root)

    def test_rejects_a_coefficient_change_inconsistent_with_reported_power(self):
        self._change_receipt(lambda row: row["native_coefficients"]["transmission"][0].__setitem__(0, [0.0, 0.0]))
        with self.assertRaisesRegex(ValueError, "transmission metrics disagree"):
            replay.compare(self.root)
