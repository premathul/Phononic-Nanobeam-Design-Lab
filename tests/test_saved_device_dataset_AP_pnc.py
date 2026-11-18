"""Portable saved-device browser checks. Author: Athul Prem | _AP_pnc."""

from contextlib import contextmanager
import csv
from hashlib import sha256
from html.parser import HTMLParser
import json
import math
from pathlib import Path
from tempfile import TemporaryDirectory
import unittest
from unittest.mock import patch
from urllib.parse import unquote, urlparse

from tools import browse_device_results_AP_pnc as browser


FREQUENCIES = (100000000, 887500000, 1000000000)
ROOTS = ((41, 7, 29, 3), (29, 41, 7, 3, 11), (11, 7, 41, 29, 3))


def write_json(path, value):
    path.write_text(json.dumps(value, indent=2, allow_nan=False), encoding="utf-8")


def digest(path):
    return sha256(path.read_bytes()).hexdigest()


def write_csv(path, rows):
    with path.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=rows[0])
        writer.writeheader()
        writer.writerows(rows)


def make_fixture(root):
    """Make three exact native-basis samples, without constructing a solver model."""
    manifest = {"schema_version": 1, "title": "Saved 26-hole study", "points": [], "controls": []}
    expected = {}
    for sample, (frequency, incoming) in enumerate(zip(FREQUENCIES, ROOTS)):
        point_id = f"sample-{sample}"
        directory = root / point_id
        directory.mkdir()
        outgoing = tuple(value + 100 for value in incoming)
        config = {
            "schema_version": 1,
            "name": point_id,
            "geometry": {"length_nm": 16650, "width_nm": 1000, "thickness_nm": 200},
            "mesh": {"kind": "layered", "thickness_layers": 6, "circle_resolution": 32},
            "material": {"density_kg_m3": 5323, "C11_GPa": 128.5, "C12_GPa": 48.3, "C44_GPa": 66.8},
            "analysis": {"kind": "scattering", "boundary": "matched_ports", "frequency_GHz": frequency / 1e9},
            "resources": {"max_dofs": 60000},
        }
        write_json(directory / "configuration_AP_pnc.json", config)
        drives, coefficients, amplitudes = [], [], {}
        for column, root_id in enumerate(incoming):
            reflection = complex(0.1 * (column + 1), 0.05 * (sample + 1))
            reflection_power = abs(reflection) ** 2
            transmission = complex(0, math.sqrt(1 - reflection_power))
            drives.append({
                "incident_mode_order": column + 1, "frequency_GHz": frequency / 1e9,
                "transmission": abs(transmission) ** 2, "reflection": reflection_power,
                "power_balance": 0.0, "transmission_flux": abs(transmission) ** 2,
                "reflection_flux": reflection_power, "flux_power_balance": 0.0,
                "relative_residual": 2e-12,
            })
            for kind, roots, diagonal in (("reflection", outgoing, reflection),
                                          ("transmission", incoming, transmission)):
                for row, outgoing_root in enumerate(roots):
                    amplitude = diagonal if row == column else 0j
                    amplitudes[(kind, root_id, outgoing_root)] = amplitude
                    coefficients.append({
                        "kind": kind, "incident_mode_order": column + 1,
                        "incident_port_index": root_id, "outgoing_port_index": outgoing_root,
                        "amplitude_real": amplitude.real, "amplitude_imag": amplitude.imag,
                        "power_fraction": abs(amplitude) ** 2,
                    })
        # CSV ordering is unrelated to the mode IDs or the frequency's incoming ordering.
        write_csv(directory / "scattering_AP_pnc.csv", list(reversed(drives)))
        write_csv(directory / "scattering_channels_AP_pnc.csv", list(reversed(coefficients)))
        modes = [
            {"port_index": root_id, "direction": direction, "propagating": True,
             "k_real_rad_m": sign * (1e6 + root_id), "k_imag_rad_m": 0.0,
             "signed_power_W": sign, "relative_residual": 3e-15}
            for roots, direction, sign in ((incoming, "+x", 1), (outgoing, "-x", -1))
            for root_id in roots
        ]
        write_csv(directory / "port_modes_AP_pnc.csv", sorted(modes, key=lambda row: row["port_index"]))
        (directory / "report_AP_pnc.html").write_text("<p>Existing field report</p>", encoding="utf-8")
        # These are link targets only. Access to any field/mesh array is forbidden below.
        (directory / "fields_AP_pnc.npz").write_bytes(b"unread scientific field sentinel")
        summary = {
            "status": "completed_unverified", "analysis_kind": "scattering",
            "frequency_hz": frequency, "incident_side": "left", "configuration": config,
            "incident_mode_indices": list(incoming), "num_incident_channels": len(incoming),
            "power_balance_passed": True,
            "output_sha256": {
                name: digest(directory / name) for name in (
                    "configuration_AP_pnc.json", "scattering_AP_pnc.csv",
                    "scattering_channels_AP_pnc.csv", "port_modes_AP_pnc.csv",
                )
            },
        }
        summary_path = directory / "summary_AP_pnc.json"
        write_json(summary_path, summary)
        evidence_path = directory / "evidence_AP_pnc.json"
        write_json(evidence_path, {
            "status": "point_verified_within_saved_discretization", "frequency_hz": frequency,
            "public_summary_sha256": digest(summary_path),
        })
        manifest["points"].append({
            "id": point_id, "frequency_hz": frequency,
            "summary": {"path": str(summary_path.relative_to(root)), "sha256": digest(summary_path)},
            "evidence": {"path": str(evidence_path.relative_to(root)), "sha256": digest(evidence_path)},
        })
        expected[point_id] = {"drives": drives, "amplitudes": amplitudes, "configuration": config}
    control = root / "spatial_control_AP_pnc.json"
    write_json(control, {"maximum_complex_change": 0.017530141984, "complex_passed": False})
    control_report = root / "spatial_control_AP_pnc.html"
    control_report.write_text("<p>Unresolved spatial comparison</p>", encoding="utf-8")
    manifest["controls"].append({
        "label": "Saved thickness sensitivity",
        "source": {"path": control.name, "sha256": digest(control)},
        "report_path": control_report.name,
        "interpretation": "Complex-response check failed; this finite result is spatially unresolved.",
    })
    manifest_path = root / "manifest_AP_pnc.json"
    write_json(manifest_path, manifest)
    return manifest_path, manifest, expected


def snapshot(root):
    return {str(path.relative_to(root)): (path.stat().st_mtime_ns, path.read_bytes())
            for path in root.rglob("*") if path.is_file()}


@contextmanager
def forbid_computation_and_arrays():
    original_open = Path.open

    def small_file_only(path, *args, **kwargs):
        if path.suffix.lower() in (".npz", ".npy", ".vtu", ".msh"):
            raise AssertionError(f"Saved-data browser opened scientific arrays: {path.name}")
        return original_open(path, *args, **kwargs)

    with patch.object(Path, "open", small_file_only), patch(
        "subprocess.Popen", side_effect=AssertionError("Saved-data browser launched a worker")
    ):
        yield


class LinkParser(HTMLParser):
    def __init__(self):
        super().__init__()
        self.links = []

    def handle_starttag(self, tag, attrs):
        if tag == "a":
            self.links.extend(value for name, value in attrs if name == "href")


class SavedDeviceDatasetTests(unittest.TestCase):
    def setUp(self):
        temporary = TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        self.root = Path(temporary.name).resolve() / "portable archive"
        self.root.mkdir()
        self.manifest_path, self.manifest, self.expected = make_fixture(self.root)

    def load(self):
        return browser.load_dataset(self.manifest_path, root=self.root)

    def test_three_native_samples_preserve_exact_values_and_scientific_status(self):
        before = snapshot(self.root)
        with forbid_computation_and_arrays():
            dataset = self.load()
        self.assertFalse(dataset["computation_enabled"])
        self.assertFalse(dataset["spatial_convergence_established"])
        self.assertFalse(dataset["frequency_convergence_established"])
        self.assertEqual([point["frequency_hz"] for point in dataset["points"]], list(FREQUENCIES))
        self.assertEqual([len(point["drives"]) for point in dataset["points"]], [4, 5, 5])
        self.assertEqual(sum(len(point["coefficients"]) for point in dataset["points"]), 132)
        for index, point in enumerate(dataset["points"]):
            expected = self.expected[point["id"]]
            self.assertEqual(point["incident_roots"], list(ROOTS[index]))
            self.assertEqual(point["configuration"], expected["configuration"])
            self.assertEqual(point["recorded_evidence_status"], "point_verified_within_saved_discretization")
            actual_drives = {int(row["incident_mode_order"]): row for row in point["drives"]}
            for source in expected["drives"]:
                row = actual_drives[source["incident_mode_order"]]
                self.assertEqual(row["incident_port_index"], ROOTS[index][source["incident_mode_order"] - 1])
                for quantity in ("transmission", "reflection", "transmission_flux", "reflection_flux", "relative_residual"):
                    self.assertEqual(row[quantity], source[quantity])
            actual = {
                (row["kind"], row["incident_port_index"], row["outgoing_port_index"]):
                complex(row["amplitude_real"], row["amplitude_imag"])
                for row in point["coefficients"]
            }
            self.assertEqual(actual, expected["amplitudes"])
        self.assertEqual(snapshot(self.root), before)

    def test_publication_exports_all_native_rows_and_resolves_existing_links(self):
        before = snapshot(self.root)
        output = self.root.parent / "browser output"
        with forbid_computation_and_arrays():
            dataset = self.load()
            publication = browser.publish_dataset(dataset, output)
        report = Path(publication["report_path"])
        self.assertEqual(report.parent, output)
        with (output / "device_drives_AP_pnc.csv").open(encoding="utf-8-sig", newline="") as stream:
            drives = list(csv.DictReader(stream))
        with (output / "device_coefficients_AP_pnc.csv").open(encoding="utf-8-sig", newline="") as stream:
            coefficients = list(csv.DictReader(stream))
        self.assertEqual(len(drives), 14)
        self.assertEqual(len(coefficients), 132)
        expected_amplitudes = {
            (point_id, kind, incoming, outgoing): amplitude
            for point_id, expected in self.expected.items()
            for (kind, incoming, outgoing), amplitude in expected["amplitudes"].items()
        }
        actual_amplitudes = {
            (row["point_id"], row["kind"], int(row["incident_port_index"]), int(row["outgoing_port_index"])):
            complex(float(row["amplitude_real"]), float(row["amplitude_imag"]))
            for row in coefficients
        }
        self.assertEqual(actual_amplitudes, expected_amplitudes)
        portable = json.loads((output / "device_results_AP_pnc.json").read_bytes())
        self.assertFalse(portable["computation_enabled"])
        self.assertFalse(portable["spatial_convergence_established"])
        html = report.read_text(encoding="utf-8")
        for term in ("Athul Prem", "0.8875", "0.1", "1", "sparse", "native", "unresolved"):
            self.assertIn(term.lower(), html.lower())
        parser = LinkParser()
        parser.feed(html)
        linked = []
        for href in parser.links:
            parsed = urlparse(href)
            if not parsed.scheme and parsed.path:
                path = (report.parent / unquote(parsed.path)).resolve()
                self.assertTrue(path.is_file(), f"Unusable report link: {href}")
                linked.append(path)
        for name in ("device_drives_AP_pnc.csv", "device_coefficients_AP_pnc.csv"):
            self.assertIn(output / name, linked)
        for point_id in self.expected:
            self.assertIn(self.root / point_id / "report_AP_pnc.html", linked)
            self.assertIn(self.root / point_id / "configuration_AP_pnc.json", linked)
        self.assertEqual(snapshot(self.root), before)

    def test_missing_or_changed_small_sources_are_refused(self):
        for mutation in ("missing", "changed"):
            with self.subTest(mutation=mutation):
                source = self.root / self.manifest["points"][0]["evidence"]["path"]
                original = source.read_bytes()
                if mutation == "missing":
                    source.unlink()
                else:
                    source.write_bytes(original + b" ")
                try:
                    with self.assertRaises((ValueError, OSError)):
                        self.load()
                finally:
                    source.write_bytes(original)

    def test_csv_tampering_is_rejected_against_accepted_summary(self):
        source = self.root / "sample-1" / "scattering_channels_AP_pnc.csv"
        source.write_bytes(source.read_bytes().replace(b"0.0", b"0.1", 1))
        with self.assertRaises((ValueError, OSError)):
            self.load()


if __name__ == "__main__":
    unittest.main()
