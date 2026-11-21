"""Verify the portable thin-grid reader against compact sources. Author: Athul Prem | _AP_pnc."""

from copy import deepcopy
import csv
from hashlib import sha256
import json
from pathlib import Path
from tempfile import TemporaryDirectory
import unittest
from unittest.mock import patch

import numpy as np

from tools import read_thin_device_grid_AP_pnc as reader


REPOSITORY = Path(__file__).resolve().parents[1]


def pin(path, root):
    raw = (root / path).read_bytes()
    return {"path": path, "sha256": sha256(raw).hexdigest(), "bytes": len(raw)}


def write_json(path, record):
    path.write_text(json.dumps(record, allow_nan=False) + "\n", encoding="utf-8")


def decode_gram(change, kind):
    return np.array([[complex(entry["real"], entry["imag"]) for entry in row]
                     for row in change[f"delta_{kind}_gram_fraction"]])


class OptionalThinGridTests(unittest.TestCase):
    def test_missing_comparison_is_optional_but_present_corrupt_record_fails(self):
        with TemporaryDirectory() as directory:
            root = Path(directory)
            self.assertIsNone(reader.load_thin_grid(root))
            evidence = root / reader.COMPARISON["path"]
            evidence.parent.mkdir()
            evidence.write_text("{}", encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "source identity changed"):
                reader.load_thin_grid(root)


class ThinGridTests(unittest.TestCase):
    def setUp(self):
        temporary = TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        self.root = Path(temporary.name).resolve() / "portable thin archive"
        self.comparison = self.root / reader.COMPARISON["path"]
        self.copy(reader.COMPARISON["path"])
        self.report = json.loads(self.comparison.read_bytes())
        self.map_paths = {sample["archive"]["map"]["path"] for level in self.report["mesh_levels"]
                          for sample in level["samples"]}
        for path in self.map_paths:
            self.copy(path)
            self.copy(Path(path).with_name(reader.CSV_FILE))

    def copy(self, relative):
        target = self.root / relative
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_bytes((REPOSITORY / relative).read_bytes())

    def repin(self):
        write_json(self.comparison, self.report)
        return patch.object(reader, "COMPARISON", pin(reader.COMPARISON["path"], self.root))

    def endpoint_match(self, mesh):
        return next(edge["groups"] for edge in self.report["channel_correspondence"]["signed_matches"]
                    if edge["reference_point"] == [mesh, 997_500_000] and edge["candidate_point"] == [mesh, 1_000_000_000])

    def test_all_sources_drives_and_coefficients_load_without_raw_archives(self):
        grid = reader.load_thin_grid(self.root)
        points = [point for level in grid["levels"] for point in level["points"]]
        self.assertEqual([len(level["points"]) for level in grid["levels"]], [3, 3, 3])
        self.assertEqual(len(grid["sources"]), 13)
        self.assertEqual(len({source["path"] for source in grid["sources"]}), 13)
        self.assertEqual(sum(len(point["drives"]) for point in points), 45)
        self.assertEqual(sum(len(point["coefficients"]) for point in points), 450)
        self.assertEqual(grid["reference_planes_x_m"], [0.0, 1.665e-5])
        self.assertFalse(grid["spatial_convergence_established"])
        self.assertFalse(grid["frequency_convergence_established"])
        self.assertFalse(grid["computation_enabled"])
        self.assertEqual([len(point["neighboring_spatial_gates"]) for point in points], [1] * 3 + [2] * 3 + [1] * 3)
        differs_from_segment_gauge = False
        for point in points:
            with Path(point["source_coefficients"]).open(encoding="utf-8", newline="") as stream:
                rows = [row for row in csv.DictReader(stream)
                        if float(row["frequency_GHz"]) * 1e9 == point["frequency_hz"]]
            by_port = {row["incident_port_index"]: row["reference_drive"] for row in point["drives"]}
            for raw, coefficient in zip(rows, point["coefficients"], strict=True):
                for key in ("amplitude_real", "amplitude_imag", "power_fraction"):
                    self.assertEqual(coefficient[key], float(raw[key]))
                differs_from_segment_gauge |= coefficient["amplitude_real"] != float(raw["segment_amplitude_real"])
                self.assertEqual(coefficient["outgoing_port_index"], int(raw["outgoing_port_index"]))
                self.assertEqual(coefficient["kind"], raw["kind"])
                self.assertEqual(coefficient["reference_drive"], by_port[int(raw["incident_port_index"])])
            for gate in point["neighboring_spatial_gates"]:
                self.assertEqual([r["reference_drive"] for r in gate["drives"] if not r["total_power_gate_passed"]], [3, 5])
        self.assertTrue(differs_from_segment_gauge)

    def test_coherent_bounds_use_all_native_rows_and_the_signed_incident_phase(self):
        grid = reader.load_thin_grid(self.root)
        changes = grid["coherent_endpoint_changes"]
        self.assertEqual([row["central_max_edge_nm"] for row in changes], [225, 168.75, 126.5625])
        expected_bounds = [(0.541, 0.901), (4.624, 5.738), (9.383, 9.905)]
        for change, level, bounds in zip(changes, self.report["mesh_levels"], expected_bounds, strict=True):
            with self.subTest(mesh=change["central_max_edge_nm"]):
                self.assertEqual((change["lower_frequency_hz"], change["upper_frequency_hz"]), (997_500_000, 1_000_000_000))
                self.assertEqual(change["reference_drives"], [3, 5])
                self.assertEqual(change["outgoing_rows_per_block"], 5)
                self.assertEqual(change["provenance"]["comparison"], pin(reader.COMPARISON["path"], self.root))
                groups = self.endpoint_match(level["central_max_edge_nm"])
                incoming = groups["incoming"]
                tables, selected_roots = [], []
                for index, endpoint in ((1, "lower"), (2, "upper")):
                    sample = level["samples"][index]
                    csv_path = Path(sample["archive"]["map"]["path"]).with_name(reader.CSV_FILE).as_posix()
                    self.assertEqual(change["provenance"][f"{endpoint}_source_coefficients"], pin(csv_path, self.root))
                    with (self.root / csv_path).open(encoding="utf-8", newline="") as stream:
                        tables.append({(r["kind"], int(r["outgoing_port_index"]), int(r["incident_port_index"])):
                                       complex(float(r["amplitude_real"]), float(r["amplitude_imag"]))
                                       for r in csv.DictReader(stream) if float(r["frequency_GHz"]) * 1e9 == sample["frequency_hz"]})
                    roots = [level["matched_drives"][col]["samples"][index]["native_root_id"] for col in (2, 4)]
                    self.assertEqual(change[f"{endpoint}_incident_native_roots"], roots)
                    selected_roots.append(roots)
                slots = [incoming["reference_roots"].index(root) for root in selected_roots[0]]
                phases = np.array(incoming["candidate_gauge_phase_rad"])[slots]
                np.testing.assert_array_equal(change["candidate_incident_gauge_phase_rad"], phases)
                np.testing.assert_array_equal(change["incident_mass_overlaps"], np.array(incoming["mass_overlaps"])[slots])
                differences = []
                for kind in ("transmission", "reflection"):
                    match = groups[kind]
                    outgoing = (match["reference_roots"], match["candidate_roots"])
                    before, after = (np.array([[table[kind, out, inc] for inc in roots] for out in outputs])
                                     for table, outputs, roots in zip(tables, outgoing, selected_roots, strict=True))
                    after = np.exp(1j * np.array(match["candidate_gauge_phase_rad"]))[:, None] * after / np.exp(1j * phases)
                    expected = after.conj().T @ after - before.conj().T @ before
                    differences.append(expected)
                    np.testing.assert_allclose(decode_gram(change, kind), expected, atol=5e-16, rtol=0)
                    eig = 100 * np.linalg.eigvalsh(expected)
                    np.testing.assert_allclose(change[f"{kind}_eigenvalue_bounds_percentage_points"], eig, atol=1e-13, rtol=0)
                    if kind == "transmission":
                        np.testing.assert_array_equal(eig.round(3), bounds)
                        focus = [outgoing[0].index(root) for root in selected_roots[0]]
                        a, b = before[focus], after[focus]
                        self.assertGreater(np.linalg.norm(expected - (b.conj().T @ b - a.conj().T @ a)), 1e-5)
                    else:
                        np.testing.assert_array_equal(eig.round(3), [-bounds[1], -bounds[0]])
                self.assertAlmostEqual(change["closure_discrepancy_percentage_points"],
                                       100 * np.linalg.norm(sum(differences), ord=2), delta=6e-14)

    def test_coherent_gram_survives_upper_native_gauge_and_root_changes(self):
        grid = reader.load_thin_grid(self.root)
        lower, upper = grid["levels"][0]["points"][1:]
        native = deepcopy(upper)
        groups = self.endpoint_match(225)
        match = deepcopy(groups["incoming"])
        incoming_phases = dict(zip(match["candidate_roots"], (.37, .1, -.63, 1.7, .29), strict=True))
        match["candidate_gauge_phase_rad"] = [phase + incoming_phases[root] for root, phase in
                                             zip(match["candidate_roots"], match["candidate_gauge_phase_rad"], strict=True)]
        match["candidate_roots"] = [root + 1000 for root in match["candidate_roots"]]
        match["candidate_order"] = [4, 3, 2, 1, 0]
        lower_inventory = {kind: groups[kind]["reference_roots"] for kind in reader.KINDS}
        upper_inventory = {kind: [root + 1000 for root in reversed(groups[kind]["candidate_roots"])] for kind in reader.KINDS}
        native["coefficients"].reverse()
        for row in native["coefficients"]:
            incident, outgoing = row["incident_port_index"], row["outgoing_port_index"]
            outgoing_phase = .37 * upper_inventory[row["kind"]].index(outgoing + 1000)
            value = complex(row["amplitude_real"], row["amplitude_imag"]) * np.exp(1j * (incoming_phases[incident] - outgoing_phase))
            row.update(incident_port_index=incident + 1000, outgoing_port_index=outgoing + 1000,
                       amplitude_real=value.real, amplitude_imag=value.imag)
        change = reader._coherent_endpoint_change(lower, native, lower_inventory, upper_inventory, match)
        original = grid["coherent_endpoint_changes"][0]
        self.assertEqual(change["upper_incident_native_roots"], [root + 1000 for root in original["upper_incident_native_roots"]])
        for kind in reader.KINDS:
            np.testing.assert_allclose(decode_gram(change, kind), decode_gram(original, kind), atol=5e-16, rtol=0)

    def test_present_comparison_requires_its_native_csv_hash(self):
        relative = Path(min(self.map_paths)).with_name(reader.CSV_FILE)
        with (self.root / relative).open("ab") as stream:
            stream.write(b"\n")
        with self.assertRaisesRegex(ValueError, "source identity changed"):
            reader.load_thin_grid(self.root)

    def test_portable_configurations_reproduce_exact_native_bytes(self):
        grid = reader.load_thin_grid(self.root)
        hole_csv = {"path": "data/geometry/hole_geometry_derived_AP_pnc.csv",
                    "sha256": "2c44d944ca9a161a5571a145d129972bb808e3e038f20765ea915f5ce0020771"}
        windows_points = []
        for level in grid["levels"]:
            for point in level["points"]:
                source = json.loads(Path(point["source_map"]).read_bytes())["inputs"]["configuration"]
                source["analysis"]["frequency_GHz"] = point["frequency_hz"] / 1e9
                config_json = point["configuration_json"]
                self.assertEqual(point["configuration"], source)
                self.assertEqual(json.loads(config_json), source)
                self.assertEqual(sha256(config_json.encode("utf-8")).hexdigest(), point["configuration_sha256"])
                self.assertTrue(config_json.endswith("\n"))
                self.assertEqual(point["hole_csv"], hole_csv)
                if "\r\n" in config_json:
                    windows_points.append((level["central_max_edge_nm"], point["frequency_hz"]))
        self.assertEqual(windows_points, [(225.0, 995_000_000), (225.0, 1_000_000_000)])
        self.assertFalse((self.root / hole_csv["path"]).exists())
        self.assertEqual(len(grid["sources"]), 13)

    def test_consistently_rehashed_map_cannot_change_native_configuration(self):
        name = self.report["mesh_levels"][0]["samples"][0]["archive"]["map"]["path"]
        path = self.root / name
        record = json.loads(path.read_bytes())
        record["inputs"]["configuration"]["resources"]["max_dofs"] += 1
        record["inputs_sha256"] = reader._digest(record["inputs"])
        del record["map_data_sha256"]
        record["map_data_sha256"] = reader._digest(record)
        write_json(path, record)
        for sample in self.report["mesh_levels"][0]["samples"]:
            if sample["archive"]["map"]["path"] == name:
                sample["archive"]["map"] = pin(name, self.root)
        with self.repin(), self.assertRaisesRegex(ValueError, "differ from the accepted configuration"):
            reader.load_thin_grid(self.root)

    def test_consistently_rehashed_power_row_must_still_sum_to_native_total(self):
        name = min(self.map_paths)
        table = self.root / Path(name).with_name(reader.CSV_FILE)
        with table.open(encoding="utf-8", newline="") as stream:
            parser = csv.DictReader(stream)
            rows = list(parser)
            columns = parser.fieldnames
        first = rows[0]
        for key, factor in (("amplitude_real", 0.5), ("amplitude_imag", 0.5), ("power_fraction", 0.25)):
            first[key] = str(float(first[key]) * factor)
        with table.open("w", encoding="utf-8", newline="") as stream:
            writer = csv.DictWriter(stream, fieldnames=columns)
            writer.writeheader()
            writer.writerows(rows)
        path = self.root / name
        data = json.loads(path.read_bytes())
        data["output_sha256"][reader.CSV_FILE] = sha256(table.read_bytes()).hexdigest()
        del data["map_data_sha256"]
        raw = json.dumps(data, sort_keys=True, separators=(",", ":"), allow_nan=False).encode("utf-8")
        data["map_data_sha256"] = sha256(raw).hexdigest()
        write_json(path, data)
        for level in self.report["mesh_levels"]:
            for sample in level["samples"]:
                if sample["archive"]["map"]["path"] == name:
                    sample["archive"]["map"] = pin(name, self.root)
        with self.repin(), self.assertRaisesRegex(ValueError, "row sums differ from the accepted"):
            reader.load_thin_grid(self.root)

    def test_locally_valid_swapped_drives_must_follow_signed_reference_matches(self):
        first, second = self.report["mesh_levels"][0]["matched_drives"][:2]
        first["samples"][0], second["samples"][0] = second["samples"][0], first["samples"][0]
        with self.repin(), self.assertRaisesRegex(ValueError, "do not follow the published signed channel matches"):
            reader.load_thin_grid(self.root)


if __name__ == "__main__":
    unittest.main()
