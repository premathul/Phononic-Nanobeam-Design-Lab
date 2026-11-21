"""Check saved frequency-figure samples and provenance. Author: Athul Prem | _AP_pnc."""

import csv
from pathlib import Path
from tempfile import TemporaryDirectory
import unittest

from tools import plot_thin_frequency_traces_AP_pnc as plot
from tools.read_thin_device_grid_AP_pnc import load_thin_grid


ROOT = Path(__file__).resolve().parents[1]


class ThinFrequencyFigureTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.grid = load_thin_grid(ROOT)
        cls.rows = plot._plotted_rows(cls.grid)

    def test_saved_csv_reproduces_with_native_identities(self):
        with TemporaryDirectory() as output:
            pdf, csv_path, png, coherent_path = plot.render(Path(output))
            self.assertTrue(pdf.read_bytes().startswith(b"%PDF-"))
            self.assertTrue(png.read_bytes().startswith(b"\x89PNG\r\n\x1a\n"))
            self.assertEqual(csv_path.read_bytes(), (ROOT / "figures" / "device_AP_pnc" / csv_path.name).read_bytes())
            self.assertEqual(coherent_path.read_bytes(), (ROOT / "figures" / "device_AP_pnc" / coherent_path.name).read_bytes())
            with csv_path.open(encoding="utf-8", newline="") as stream:
                saved = list(csv.DictReader(stream))
        self.assertEqual(len(saved), 18)
        points = {(level["central_max_edge_nm"], point["frequency_hz"]): point
                  for level in self.grid["levels"] for point in level["points"]}
        sources = {Path(item["path"]).relative_to(ROOT).as_posix(): item["sha256"]
                   for item in self.grid["sources"]}
        identities = set()
        for row in saved:
            mesh, frequency, label = (float(row["central_max_edge_nm"]), int(row["frequency_hz"]),
                                      int(row["reference_drive"]))
            self.assertIn(label, (3, 5))
            identities.add((mesh, frequency, label))
            point = points[mesh, frequency]
            native = next(drive for drive in point["drives"] if drive["reference_drive"] == label)
            self.assertEqual(int(row["native_incident_root"]), native["incident_port_index"])
            self.assertEqual(int(row["native_drive"]), native["native_drive"])
            self.assertEqual(float(row["transmission_fraction"]), native["transmission"])
            self.assertEqual(float(row["reflection_fraction"]), native["reflection"])
            self.assertEqual(row["configuration_sha256"], point["configuration_sha256"])
            for stem, source in (("source_map", point["source_map"]), ("source_coefficients", point["source_coefficients"]),
                                 ("comparison_record", self.grid["source_comparison"])):
                relative = Path(source).relative_to(ROOT).as_posix()
                self.assertEqual(row[f"{stem}_path"], relative)
                self.assertEqual(row[f"{stem}_sha256"], sources[relative])
        self.assertEqual(len(identities), 18)

    def test_coherent_export_preserves_matrices_and_phase_correspondence(self):
        rows = plot._coherent_rows(self.grid)
        self.assertEqual(len(rows), 3)
        for row, change in zip(rows, self.grid["coherent_endpoint_changes"], strict=True):
            self.assertEqual(row["outgoing_rows_per_block"], 5)
            self.assertEqual(row["lower_frequency_hz"], 997_500_000)
            self.assertEqual(row["upper_frequency_hz"], 1_000_000_000)
            for index, drive in enumerate((3, 5)):
                self.assertEqual(row[f"candidate_incident_gauge_phase_rad_drive_{drive}"],
                                 change["candidate_incident_gauge_phase_rad"][index])
            for kind in ("transmission", "reflection"):
                for i, matrix_row in enumerate(change[f"delta_{kind}_gram_fraction"], 1):
                    for j, value in enumerate(matrix_row, 1):
                        for part in ("real", "imag"):
                            self.assertEqual(row[f"delta_{kind}_gram_{i}{j}_{part}_fraction"], value[part])
            for stem, pin in change["provenance"].items():
                self.assertEqual(row[f"{stem}_sha256"], pin["sha256"])
                self.assertEqual(row[f"{stem}_path"], pin["path"])

    def test_only_saved_markers_are_plotted_on_common_axes(self):
        figure = plot._plot_figure(self.grid, self.rows)
        endpoint_changes = (
            (0.7475703233390363, 5.240139263693955, 9.779567406729495),
            (0.6941248438833203, 5.121596720999566, 9.508498335541216),
        )
        for axis, drive, changes in zip(figure.axes, (3, 5), endpoint_changes, strict=True):
            self.assertEqual(axis.get_title(loc="left"), f"Reference drive {drive}")
            self.assertEqual(axis.get_ylim(), (0, 100))
            self.assertEqual(list(axis.get_xticks()), [995, 997.5, 1000])
            self.assertEqual([line.get_label() for line in axis.lines], ["225", "168.75", "126.5625"])
            for line, mesh, marker, change in zip(axis.lines, self.grid["central_max_edges_nm"],
                                                  ("o", "s", "^"), changes, strict=True):
                samples = [row for row in self.rows if (row["central_max_edge_nm"], row["reference_drive"]) == (mesh, drive)]
                self.assertEqual(list(line.get_xdata()), [995, 997.5, 1000])
                values = list(line.get_ydata())
                self.assertEqual(values, [100 * sample["transmission_fraction"] for sample in samples])
                self.assertEqual(line.get_marker(), marker)
                self.assertAlmostEqual(values[-1] - values[-2], change, places=12)


if __name__ == "__main__":
    unittest.main()
