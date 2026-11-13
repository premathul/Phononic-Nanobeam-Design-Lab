"""Verify the saved full-device SAFE section study. Author: Athul Prem | _AP_pnc."""

import csv
import json
from pathlib import Path
import sys
import unittest

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "tools"))

from check_device_followup_AP_pnc import port_section, propagating
from check_phase_refinement_AP_pnc import check_recorded_source
from compare_scattering_AP_pnc import align, file_sha, interpolation
from compare_uniform_reference_AP_pnc import check_section_inputs, read_reference, read_section


EVIDENCE = ROOT / "verification/full_device_port_resolution_AP_pnc.json"
SECTION_NM = {"width": 1000, "thickness": 200}
LENGTHS = {
    "one_lead_1p5um": (1500, "phase_indicator_1p5um_rad"),
    "both_leads_3um": (3000, "phase_indicator_3um_rad"),
    "uniform_equivalent_16p65um": (16650, "phase_indicator_16p65um_rad"),
}


class FullDevicePortResolutionEvidenceTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.data = json.loads(EVIDENCE.read_bytes())
        cls.prior = cls.read_evidence(cls.data["previous_evidence"])
        cls.historical = cls.read_evidence(cls.prior["reference_26hole_refinement"])
        saved = cls.historical["levels"]["160"]
        source = saved["configuration"]["material"]
        material = {key: source[key] for key in (
            "C11_Pa", "C12_Pa", "C44_Pa", "density_kg_m3", "crystal_rotation", "temperature_policy",
        )}
        cls.fixed = {
            "dimensions_nm": {key: saved["configuration"]["geometry"][key]
                              for key in ("length_nm", "width_nm", "thickness_nm")},
            "material": material,
            "frequency_hz": cls.historical["frequency_hz"],
            "numerical_source_sha256": saved["implementation"]["source_sha256"],
        }
        cls.cases = {
            label: read_section(case, SECTION_NM["width"], SECTION_NM["thickness"], material["density_kg_m3"])
            for label, case in cls.data["cases"].items()
        }

    @staticmethod
    def read_evidence(provenance):
        path = ROOT / provenance["path"]
        if file_sha(path) != provenance["sha256"]:
            raise AssertionError(f"Evidence checksum differs: {path.name}")
        return json.loads(path.read_bytes())

    def test_saved_channels_and_phase_indicators_reproduce_from_fields(self):
        data = self.data
        self.assertEqual(data["status"], "diagnostic_completed")
        self.assertEqual(data["frequency_hz"], self.fixed["frequency_hz"])
        self.assertEqual(data["physical_section_nm"], SECTION_NM)
        self.assertEqual(set(data["diagnostic_lengths_m"]), set(LENGTHS))
        for label, (length_nm, _) in LENGTHS.items():
            self.assertAlmostEqual(data["diagnostic_lengths_m"][label] / 1e-9, length_nm, places=8)
        for source, checksum in data["numerical_source_sha256"].items():
            check_recorded_source(source, checksum)
            if source in self.fixed["numerical_source_sha256"]:
                self.assertEqual(checksum, self.fixed["numerical_source_sha256"][source])
        for label, record in data["cases"].items():
            check_section_inputs(label, record, self.fixed)
            source = record["source"]["source_sha256"]
            for name, checksum in data["numerical_source_sha256"].items():
                if record["mesh_family"] == "layered" or not name.endswith("/layered_AP_pnc.py"):
                    self.assertEqual(source[name], checksum)
            basis, modes = self.cases[label]
            ny, nz = record["width_cells"], record["thickness_layers"]
            self.assertEqual(basis.basis.N, (2 * ny + 1) * (2 * nz + 1))
            self.assertEqual({side: len(group.roots) for side, group in modes.items()}, {"positive": 5, "negative": 5})
            if record["mesh_family"] == "layered":
                self.check_layered_grid(basis.basis.mesh, ny, nz)

        csv_path = EVIDENCE.with_suffix(".csv")
        self.assertEqual(file_sha(csv_path), data["channel_csv_sha256"])
        with csv_path.open(newline="", encoding="utf-8") as stream:
            rows = list(csv.DictReader(stream))
        indexed = {(row["comparison"], row["direction"], int(row["mode_order"])): row for row in rows}
        self.assertEqual(len(rows), len(indexed))
        self.assertEqual(len(rows), 10 * len(data["comparisons"]))
        for pair, comparison in data["comparisons"].items():
            first, second = pair.split("_to_")
            coarse, old_modes = self.cases[first]
            fine, new_modes = self.cases[second]
            project_old = coarse.basis.N <= fine.basis.N
            source, target = (coarse, fine) if project_old else (fine, coarse)
            probe = interpolation(source, target)
            self.assertEqual(set(comparison), {"positive", "negative"})
            for direction, saved in comparison.items():
                with self.subTest(pair=pair, direction=direction):
                    old, new = old_modes[direction], new_modes[direction]
                    order, _, details = align(old, new, target.mass, probe, project_old)
                    self.assertEqual(saved["reference_roots"], details["reference_roots"])
                    self.assertEqual(saved["candidate_roots"], details["candidate_roots"])
                    np.testing.assert_allclose(saved["mass_overlaps"], details["mass_overlaps"], rtol=0, atol=1e-10)
                    delta = new.k[order] - old.k
                    np.testing.assert_allclose(saved["delta_k_rad_m"], delta, rtol=0, atol=1e-8)
                    channel_rows = [indexed[pair, direction, i] for i in range(1, len(delta) + 1)]
                    self.assertEqual([int(row["reference_root"]) for row in channel_rows], old.roots.tolist())
                    self.assertEqual([int(row["refined_root"]) for row in channel_rows], new.roots[order].tolist())
                    for key, (_, csv_column) in LENGTHS.items():
                        phase = abs(delta) * data["diagnostic_lengths_m"][key]
                        np.testing.assert_allclose(saved["phase_indicators_rad"][key], phase, rtol=0, atol=1e-12)
                        self.assertAlmostEqual(saved["max_phase_indicators_rad"][key], max(phase), places=12)
                        np.testing.assert_allclose([float(row[csv_column]) for row in channel_rows], phase, rtol=0, atol=1e-12)

    def test_historical_bridge_and_independent_axis_qualification(self):
        self.check_historical_ports()
        data = self.data
        for previous_pair, previous in self.prior["SAFE_comparisons"].items():
            first, second = previous_pair.split("_to_")
            current = data["comparisons"][f"gmsh{first}_to_gmsh{second}"]
            for direction, saved in previous.items():
                self.assertEqual(current[direction]["reference_roots"], saved["old_root_indices"])
                self.assertEqual(current[direction]["candidate_roots"], saved["new_root_indices"])
                self.assertAlmostEqual(max(abs(np.asarray(current[direction]["delta_k_rad_m"]))),
                                       saved["max_abs_delta_k_rad_m"], places=8)

        for key in ("economical_reference_qualification", "reference_qualification"):
            with self.subTest(qualification=key):
                self.check_reference_qualification(data[key])

    def check_reference_qualification(self, qualification):
        data = self.data
        endpoint = qualification["endpoint"]
        self.assertEqual(set(qualification["pairs"]), {"width", "thickness"})
        self.assertEqual(qualification["phase_indicator_limit_rad"], 0.01)
        for axis, pair in qualification["pairs"].items():
            first, second = pair.split("_to_")
            self.assertEqual(second, endpoint)
            before, after = (data["cases"][key] for key in (first, second))
            self.assertEqual(before["mesh_family"], "layered")
            self.assertEqual(after["mesh_family"], "layered")
            changed, fixed = ("width_cells", "thickness_layers") if axis == "width" else ("thickness_layers", "width_cells")
            self.assertEqual(before[fixed], after[fixed])
            self.assertLess(before[changed], after[changed])

        for key, (length_nm, _) in LENGTHS.items():
            with self.subTest(length=key):
                fixed = {**self.fixed, "dimensions_nm": {**self.fixed["dimensions_nm"], "length_nm": length_nm}}
                _, _, replay = read_reference(EVIDENCE, endpoint, fixed)
                self.assertEqual(replay["phase_indicator_limit_rad"], 0.01)
                maxima = []
                for axis, pair in qualification["pairs"].items():
                    self.assertEqual(replay["pairs"][axis]["pair"], pair)
                    phase = {direction: side["max_phase_indicators_rad"][key]
                             for direction, side in data["comparisons"][pair].items()}
                    for direction, value in phase.items():
                        self.assertAlmostEqual(replay["pairs"][axis]["phase_indicators_rad"][direction], value, places=12)
                    self.assertAlmostEqual(qualification["max_phase_indicators_rad"][key][axis],
                                           max(phase.values()), places=12)
                    maxima.append(max(phase.values()))
                expected = "passed" if max(maxima) < 0.01 else "failed"
                self.assertEqual(replay["status"], expected)
                self.assertEqual(qualification["status_by_length"][key], expected)

    def check_layered_grid(self, mesh, ny, nz):
        extent_m = np.array([SECTION_NM["width"], SECTION_NM["thickness"]])[:, None] * 1e-9
        scaled = (mesh.p / extent_m + 0.5) * np.array([[ny], [nz]])
        grid = np.rint(scaled).astype(int)
        np.testing.assert_allclose(scaled, grid, rtol=0, atol=1e-10)
        self.assertEqual(set(map(tuple, grid.T)), {(i, j) for i in range(ny + 1) for j in range(nz + 1)})
        triangles = [tuple(sorted(map(tuple, grid[:, nodes].T))) for nodes in mesh.t.T]
        expected = {triangle for i in range(ny) for j in range(nz) for triangle in (
            ((i, j), (i + 1, j), (i + 1, j + 1)), ((i, j), (i, j + 1), (i + 1, j + 1)),
        )}
        self.assertEqual(len(triangles), len(expected))
        self.assertEqual(set(triangles), expected)

    def check_historical_ports(self):
        old = self.historical["levels"]["160"]["port_modes"]
        cases = {"gmsh99": (self.historical["port_geometry"], {
            "positive": old["incoming"], "negative": old["reflection"],
        })}
        for count, record in self.prior["SAFE_ports"].items():
            cases[f"gmsh{count}"] = (record["port_geometry"], record["propagating"])
        for label, (geometry, old_fields) in cases.items():
            with self.subTest(historical_case=label):
                data = self.data["cases"][label]
                count = int(label.removeprefix("gmsh"))
                self.assertEqual(data["trace_dofs"], count)
                basis, _, maps, _ = port_section(geometry, data["width_cells"], data["thickness_layers"])
                original_modes = propagating(old_fields, count)
                converted, saved_modes = self.cases[label]
                np.testing.assert_array_equal(basis.doflocs, converted.basis.doflocs)
                for direction, (roots, k, fields) in original_modes.items():
                    expected = np.empty((3, basis.N, len(roots)), dtype=complex)
                    for component, (selected, order) in enumerate(maps):
                        expected[component, order] = fields[selected]
                    actual = saved_modes[direction]
                    np.testing.assert_array_equal(roots, actual.roots)
                    np.testing.assert_array_equal(k, actual.k)
                    np.testing.assert_array_equal(expected, actual.displacement)


if __name__ == "__main__":
    unittest.main()
