"""Verify the saved layered SAFE section study without solving. Author: Athul Prem | _AP_pnc."""

import csv
import json
from pathlib import Path
import sys
import unittest

import numpy as np
from skfem import Basis, ElementTriP2, MeshTri
from skfem.models.poisson import mass

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "tools"))
from compare_scattering_AP_pnc import Modes, Section, align, file_sha, interpolation
from validate_perforated_AP_pnc import unpack


class LayeredPortEvidenceTests(unittest.TestCase):
    def test_saved_port_refinement(self):
        path = ROOT / "verification/layered_port_resolution_AP_pnc.json"
        data = json.loads(path.read_bytes())
        self.assertEqual(data["frequency_hz"], 1e9)
        self.assertEqual(data["diagnostic_length_m"], 1.8e-6)
        self.assertEqual(data["physical_section_nm"], {"width": 600, "thickness": 200})
        baseline = data["previous_evidence"]
        self.assertEqual(file_sha(ROOT / baseline["path"]), baseline["sha256"])
        for file, expected in data["numerical_source_sha256"].items():
            self.assertEqual(file_sha(ROOT / file), expected)
        self.assertEqual(set(data["cases"]), {"4x2", "4x3", "4x4", "6x2", "8x2", "6x4", "8x4"})
        cases = {key: self.section(record) for key, record in data["cases"].items()}

        csv_path = path.with_suffix(".csv")
        self.assertEqual(file_sha(csv_path), data["channel_csv_sha256"])
        with csv_path.open(newline="", encoding="utf-8") as stream:
            rows = list(csv.DictReader(stream))
        indexed = {(row["comparison"], row["direction"], int(row["mode_order"])): row for row in rows}
        self.assertEqual(len(rows), len(indexed))
        self.assertEqual(len(rows), 8 * len(data["comparisons"]))
        for name, comparison in data["comparisons"].items():
            coarse_key, fine_key = name.split("_to_")
            coarse, old_modes = cases[coarse_key]
            fine, new_modes = cases[fine_key]
            probe = interpolation(coarse, fine)
            for side, saved in comparison.items():
                old, new = old_modes[side], new_modes[side]
                indices, _, details = align(old, new, fine.mass, probe, True)
                difference = new.k[indices] - old.k
                phase = abs(difference) * data["diagnostic_length_m"]
                np.testing.assert_allclose(saved["delta_k_rad_m"], difference, rtol=0, atol=1e-8)
                np.testing.assert_allclose(saved["mass_overlaps"], details["mass_overlaps"], rtol=0, atol=1e-10)
                self.assertAlmostEqual(saved["max_phase_indicator_rad"], max(phase), places=12)
                self.assertEqual(saved["largest_change_reference_root"], old.roots[phase.argmax()])
                energy = np.sum((old.displacement.conj() * (coarse.mass @ old.displacement)).real, axis=1)
                np.testing.assert_allclose(saved["reference_kinetic_fractions_xyz"],
                                           (energy / energy.sum(axis=0)).T, rtol=0, atol=1e-10)
                for i, phase_i in enumerate(phase):
                    row = indexed[name, side, i + 1]
                    self.assertEqual(int(row["reference_root"]), old.roots[i])
                    self.assertEqual(int(row["refined_root"]), new.roots[indices[i]])
                    for field, value in (("reference_k_rad_m", old.k[i]), ("refined_k_rad_m", new.k[indices[i]]),
                                         ("delta_k_rad_m", difference[i]), ("mass_overlap", details["mass_overlaps"][i]),
                                         ("phase_indicator_rad", phase_i)):
                        np.testing.assert_allclose(float(row[field]), value, rtol=1e-12, atol=1e-10)
        self.assertTrue(data["optional_case_run"])
        for key, pair in (("width_nz2", "6x2_to_8x2"), ("thickness_ny6", "6x2_to_6x4"),
                          ("width_nz4", "4x4_to_6x4")):
            value = max(v["max_phase_indicator_rad"] for v in data["comparisons"][pair].values())
            self.assertEqual(data["optional_trigger_values_rad"][key], value)
        self.assertGreater(max(data["optional_trigger_values_rad"].values()), data["optional_trigger_limit_rad"])

    def section(self, record):
        ny, nz = record["width_cells"], record["thickness_layers"]
        section = record["section"]
        basis = Basis(MeshTri(np.asarray(section["vertices_yz_m"]).T, np.asarray(section["triangles"]).T),
                      ElementTriP2(), intorder=4)
        np.testing.assert_allclose(basis.doflocs.T, section["scalar_coordinates_yz_m"], rtol=0, atol=1e-20)
        self.assertEqual(record["trace_dofs"], 3 * basis.N)
        self.assertEqual(basis.N, (2 * ny + 1) * (2 * nz + 1))
        self.assertAlmostEqual(basis.dx.sum() / (600e-9 * 200e-9), 1, places=12)
        self.assertLess(record["max_relative_residual"], 1e-8)
        section_basis = Section(basis, np.empty((0, 0)), mass.assemble(basis).toarray() * 5323, "")
        modes = {}
        for side, sign in (("positive", 1), ("negative", -1)):
            fields = record["propagating"][side]
            roots, k = np.asarray(fields["root_indices"]), np.asarray(fields["k_rad_m"])
            u = unpack(record["propagating"][fields["conjugate_of"]]["displacement"]).conj() if "conjugate_of" in fields else unpack(fields["displacement"])
            self.assertEqual(u.shape, (3, basis.N, 4))
            self.assertTrue(np.isfinite(u).all() and np.all(k * sign > 0))
            self.assertEqual(len(set(roots)), 4)
            np.testing.assert_allclose(fields["signed_power_w"], sign, atol=1e-9, rtol=0)
            modes[side] = Modes(roots, k, u)
        return section_basis, modes


if __name__ == "__main__":
    unittest.main()
