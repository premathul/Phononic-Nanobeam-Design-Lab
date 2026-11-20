"""Portable saved-device browser checks. Author: Athul Prem | _AP_pnc."""

from contextlib import contextmanager
import csv
from hashlib import sha256
from html.parser import HTMLParser
import json
import math
from pathlib import Path
import shutil
import subprocess
import sys
from tempfile import TemporaryDirectory
import unittest
from unittest.mock import patch
from urllib.parse import unquote, urlparse

from tools import browse_device_results_AP_pnc as browser


FREQUENCIES = (100000000, 887500000, 1000000000)
ROOTS = ((41, 7, 29, 3), (29, 41, 7, 3, 11), (11, 7, 41, 29, 3))
REPOSITORY = Path(__file__).resolve().parents[1]
# The accepted archive publication was captured before portable packaging.
ACCEPTED_EXPORT_SHA256 = {
    "device_drives_AP_pnc.csv": "997d6ccb1f97445347aab662cf4d6480959f5b9b77738bd3114050fad860f3d3",
    "device_coefficients_AP_pnc.csv": "c7dde89c518493bb0de07395df4e5ed34d3fd67640cbf9b28cdc133b96c5c949",
}

# Isolated Python has neither site packages nor a copied solver. Refuse imports,
# array reads and worker launches explicitly as well, so absence is not a skip.
PORTABLE_CLI = r'''
import importlib.abc
from pathlib import Path
import runpy
import sys

class NoNumericalDependencies(importlib.abc.MetaPathFinder):
    def find_spec(self, fullname, path=None, target=None):
        if fullname.split(".")[0] in {
            "phononic", "simulate_AP_pnc", "numpy", "scipy", "gmsh", "skfem", "meshio", "PySide6"
        }:
            raise AssertionError("Saved-data browser imported a numerical dependency: " + fullname)

def refuse_computation(event, args):
    if event in {"subprocess.Popen", "os.system", "os.spawn", "os.posix_spawn"}:
        raise AssertionError("Saved-data browser launched a worker")
    if event == "open" and isinstance(args[0], (str, bytes)):
        filename = args[0].decode() if isinstance(args[0], bytes) else args[0]
        if Path(filename).suffix.lower() in {".npz", ".npy", ".vtu", ".msh"}:
            raise AssertionError("Saved-data browser opened scientific arrays: " + filename)

sys.meta_path.insert(0, NoNumericalDependencies())
sys.addaudithook(refuse_computation)
sys.argv = sys.argv[1:]
runpy.run_path(sys.argv[0], run_name="__main__")
'''


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
        self.dataset_parts = []
        self.in_dataset = False

    def handle_starttag(self, tag, attrs):
        if tag == "a":
            self.links.extend(value for name, value in attrs if name == "href")
        if tag == "script" and dict(attrs).get("id") == "dataset":
            self.in_dataset = True

    def handle_data(self, data):
        if self.in_dataset:
            self.dataset_parts.append(data)

    def handle_endtag(self, tag):
        if tag == "script":
            self.in_dataset = False


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


class PortableAcceptedDeviceTests(unittest.TestCase):
    """Reproduce the accepted tables from the shipped clone inputs alone."""

    def setUp(self):
        temporary = TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        self.root = Path(temporary.name).resolve() / "clean clone with spaces"
        self.root.mkdir()
        for relative in (
            "tools/browse_device_results_AP_pnc.py",
            "src/phononic/report_style_AP_pnc.py",
            "configs/saved_device_dataset_AP_pnc.json",
            "data/geometry/hole_geometry_derived_AP_pnc.csv",
            "verification/device_thickness_AP_pnc.json",
        ):
            target = self.root / relative
            target.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(REPOSITORY / relative, target)
        shutil.copytree(REPOSITORY / "data/saved_device_AP_pnc", self.root / "data/saved_device_AP_pnc")
        self.manifest = json.loads((self.root / "configs/saved_device_dataset_AP_pnc.json").read_bytes())
        self.assertFalse((self.root / "results").exists())
        self.assertFalse((self.root / "simulate_AP_pnc.py").exists())
        self.inputs = snapshot(self.root)
        self.output = self.root / "results/runs/saved_device_results_AP_pnc"

    def publish(self):
        return subprocess.run(
            [sys.executable, "-I", "-S", "-B", "-c", PORTABLE_CLI,
             str(self.root / "tools/browse_device_results_AP_pnc.py")],
            cwd=self.root, capture_output=True, text=True, timeout=30,
        )

    def assert_inputs_unchanged(self):
        after = snapshot(self.root)
        for name, contents_and_mtime in self.inputs.items():
            self.assertEqual(after[name], contents_and_mtime, f"Browser changed input: {name}")

    def assert_resolves(self, href):
        parsed = urlparse(href)
        self.assertFalse(parsed.scheme, f"Portable link must be relative: {href}")
        self.assertTrue(parsed.path)
        target = (self.output / unquote(parsed.path)).resolve()
        self.assertTrue(target.is_relative_to(self.root), f"Link escapes clone: {href}")
        self.assertTrue(target.is_file(), f"Missing portable link target: {href}")
        return target

    def test_clean_clone_reproduces_all_accepted_rows_and_available_links(self):
        result = self.publish()
        self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
        self.assertIn("3 saved points, 14 drives and 132 native coefficients", result.stdout)
        for filename, expected in ACCEPTED_EXPORT_SHA256.items():
            self.assertEqual(digest(self.output / filename), expected, filename)
        for filename, expected_count in ((browser.DRIVES, 14), (browser.COEFFICIENTS, 132)):
            with (self.output / filename).open(encoding="utf-8", newline="") as stream:
                self.assertEqual(len(list(csv.DictReader(stream))), expected_count)

        parser = LinkParser()
        html = (self.output / browser.REPORT).read_text(encoding="utf-8")
        for title in ("Selected drive coefficients (CSV)", "All 3 saved samples", "All 14 drive powers (CSV)",
                      "All 132 native coefficients (CSV)", "All samples, inputs and evidence (JSON)"):
            self.assertIn(title, html)
        self.assertLess(html.index('id="frequency"'), html.index("Accepted within the saved discretization"))
        self.assertLess(html.index("Accepted within the saved discretization"), html.index("Sources for the selected sample"))
        self.assertEqual(html.count("(archive unavailable)"), 9)
        parser.feed(html)
        for href in parser.links:
            self.assert_resolves(href)
        view = json.loads("".join(parser.dataset_parts))
        self.assertFalse(view["computation_enabled"])
        self.assertFalse(view["spatial_convergence_established"])
        self.assertFalse(view["frequency_convergence_established"])
        self.assertEqual([point["frequency_hz"] for point in view["points"]], list(FREQUENCIES))
        self.assertEqual([len(point["drives"]) for point in view["points"]], [4, 5, 5])
        self.assertEqual([len(point["coefficients"]) for point in view["points"]], [32, 50, 50])

        for point in view["points"]:
            self.assert_resolves(point["summary_href"])
            self.assert_resolves(point["evidence_href"])
            available = []
            archive = []
            for artifact in point["artifacts"]:
                if artifact["available"]:
                    available.append(self.assert_resolves(artifact["href"]))
                    self.assertFalse(artifact["archive_only"])
                    self.assertFalse(artifact["missing_in_clone"])
                else:
                    archive.append(Path(artifact["path"]).name)
                    self.assertTrue(artifact["archive_only"])
                    self.assertTrue(artifact["missing_in_clone"])
                    self.assertNotIn("href", artifact)
                    self.assertFalse(Path(artifact["path"]).exists())
            self.assertEqual({path.name for path in available},
                             {"configuration_AP_pnc.json", "hole_geometry_derived_AP_pnc.csv"})
            self.assertEqual(set(archive), {"report_AP_pnc.html", "selected_fields_AP_pnc.npz",
                                           "selected_vertex_shapes_AP_pnc.vtu"})

        self.assertEqual(len(view["controls"]), 9)
        self.assertEqual([control["interpretation"] for control in view["controls"]],
                         [control["interpretation"] for control in self.manifest["controls"]])
        for control in view["controls"]:
            self.assertTrue(Path(control["source"]).is_file())
            self.assertFalse(control["available"])
            self.assertTrue(control["archive_only"])
            self.assertTrue(control["missing_in_clone"])
        self.assertIn("archive-only", html.lower())
        self.assert_inputs_unchanged()

    def test_repeat_publication_preserves_accepted_inputs_and_publication_bytes_and_mtimes(self):
        first = self.publish()
        self.assertEqual(first.returncode, 0, first.stdout + first.stderr)
        before = snapshot(self.output)
        second = self.publish()
        self.assertEqual(second.returncode, 0, second.stdout + second.stderr)
        self.assertEqual(snapshot(self.output), before)
        self.assert_inputs_unchanged()

    def test_optional_paired_evidence_changes_only_the_default_html(self):
        legacy = self.publish()
        self.assertEqual(legacy.returncode, 0, legacy.stdout + legacy.stderr)
        before = {name: (self.output / name).read_bytes() for name in browser.FILES}
        self.assertNotIn(b'id="paired-title"', before[browser.REPORT])
        for pin in browser.PAIRED_SOURCES.values():
            shutil.copy2(REPOSITORY / pin["path"], self.root / pin["path"])

        latest = self.publish()
        self.assertEqual(latest.returncode, 0, latest.stdout + latest.stderr)
        for name in (browser.DATA, browser.DRIVES, browser.COEFFICIENTS):
            self.assertEqual((self.output / name).read_bytes(), before[name])
        html = (self.output / browser.REPORT).read_text(encoding="utf-8")
        self.assertIn('id="paired-title"', html)
        self.assertIn("32 rim edges per hole", html)
        self.assertIn("five equal-power", html)
        self.assertIn("0 and 16.65 µm", html)
        for label in ("225 nm baseline", "168.75 nm taper/cavity patch within the 225 nm mesh",
                      "168.75 nm across the central region", "995", "997.5", "1000"):
            self.assertIn(f'<th scope="row">{label}</th>', html)
        parser = LinkParser()
        parser.feed(html)
        linked = {self.assert_resolves(href).relative_to(self.root).as_posix() for href in parser.links}
        self.assertTrue({pin["path"] for pin in browser.PAIRED_SOURCES.values()} <= linked)
        self.assertEqual([point["frequency_hz"] for point in json.loads("".join(parser.dataset_parts))["points"]],
                         list(FREQUENCIES))

        default = self.root / "configs/saved_device_dataset_AP_pnc.json"
        paired = browser.load_dataset(default, root=self.root)["paired_thickness_tables"]
        self.assertEqual([[round(row[3], 4) for row in table["rows"]] for table in paired],
                         [[-28.0624, -23.8988, -22.6482], [-26.2463, -24.7457, -22.6482]])
        custom = self.root / "custom_AP_pnc.json"
        shutil.copy2(default, custom)
        self.assertNotIn("paired_thickness_tables", browser.load_dataset(custom, root=self.root))

        published = snapshot(self.output)
        source = self.root / browser.PAIRED_SOURCES["frequency"]["path"]
        source.write_bytes(source.read_bytes() + b" ")
        failed = self.publish()
        self.assertEqual(failed.returncode, 2, failed.stdout + failed.stderr)
        self.assertIn("Saved source identity changed", failed.stderr)
        self.assertEqual(snapshot(self.output), published)
        self.assert_inputs_unchanged()

    def test_goal_patch_evidence_adds_only_available_columns_to_the_report(self):
        legacy = self.publish()
        self.assertEqual(legacy.returncode, 0, legacy.stdout + legacy.stderr)
        before = {name: (self.output / name).read_bytes() for name in browser.FILES}
        self.assertNotIn(b'id="goal-patch-title"', before[browser.REPORT])
        paths = {"localization": self.root / "verification/regional_work_fixture_AP_pnc.json",
                 "comparison": self.root / "verification/goal_patch_fixture_AP_pnc.json"}
        planes = [0.0, 1.665e-5]
        roots = [268, 263, 258, 242, 240]
        scope = {"frequency_hz": 1e9, "spatial_convergence_established": False,
                 "frequency_convergence_established": False}
        work_by_hole = {9: (0.4, 0.2), 13: (1.05, 2.0), 14: (2.15, 1.5), 18: (0.6, 0.7)}
        write_json(paths["localization"], {**scope, "cases": {"thin_first": {"result": {
            "original_planes_m": planes, "incident_roots": roots,
            "mode_alignment": {"incoming": {"reference_roots": roots, "candidate_roots": roots}},
            "bins": [{"hole": hole, "delta_T_percentage_points_by_drive": [0, 0, drive_three, 0, drive_five]}
                     for hole, (drive_three, drive_five) in work_by_hole.items()],
        }}}})
        pins = {"localization": {"path": paths["localization"].relative_to(self.root).as_posix(),
                                 "sha256": digest(paths["localization"])}}
        manifest = self.root / "configs/saved_device_dataset_AP_pnc.json"

        with patch.object(browser, "GOAL_PATCH_SOURCES", pins), forbid_computation_and_arrays():
            localization = browser.load_dataset(manifest, root=self.root)
            publication = browser.publish_dataset(localization, self.output)
        report = Path(publication["report_path"])
        html = report.read_text(encoding="utf-8")
        self.assertIn('id="goal-patch-title"', html)
        self.assertIn('+3.200', html)
        self.assertIn('+0.900', html)
        self.assertNotIn("Patch solve ΔT", html)

        comparison_cases = {}
        for region, values in (
            ("cavity_13_14", {3: (3.2, 2.5, 3.1), 5: (3.5, 3.0, 3.7)}),
            ("outer_taper_09_18", {3: (1.0, 0.5, 0.9), 5: (0.9, 0.4, -0.2)}),
        ):
            rows = []
            for drive in range(1, 6):
                full, restricted, measured = values.get(drive, (0.0, 0.0, 0.0))
                rows.append({"reference_drive": drive, "full_bin_work_percentage_points": full,
                             "prediction_percentage_points": restricted, "measured_change_percentage_points": measured})
            comparison_cases[region] = {"per_drive": rows}
        write_json(paths["comparison"], {**scope, "schema_version": 1, "status": "compared_saved_goal_patch_controls",
                                         "physical_thickness_nm": 100, "incident_side": "left",
                                         "reference_planes_x_m": planes, "cases": comparison_cases,
                                         "provenance": {"parent_work": pins["localization"]}})
        pins["comparison"] = {"path": paths["comparison"].relative_to(self.root).as_posix(),
                              "sha256": digest(paths["comparison"])}

        with patch.object(browser, "GOAL_PATCH_SOURCES", pins), forbid_computation_and_arrays():
            compared = browser.load_dataset(manifest, root=self.root)
            browser.publish_dataset(compared, self.output)
        html = report.read_text(encoding="utf-8")
        self.assertIn("Patch solve ΔT (pp)", html)
        self.assertIn("<td>+2.500</td><td>+3.100</td>", html)
        self.assertIn("<td>+0.400</td><td>-0.200</td>", html)
        self.assertIn("Spatial and frequency convergence remain unresolved.", html)
        parser = LinkParser()
        parser.feed(html)
        self.assertTrue(set(paths.values()) <= {self.assert_resolves(href) for href in parser.links})
        for name in (browser.DATA, browser.DRIVES, browser.COEFFICIENTS):
            self.assertEqual((self.output / name).read_bytes(), before[name])

        paths["localization"].unlink()
        with patch.object(browser, "GOAL_PATCH_SOURCES", pins), forbid_computation_and_arrays():
            remaining = browser.load_dataset(manifest, root=self.root)["goal_patch_table"]
            self.assertEqual(remaining["rows"], compared["goal_patch_table"]["rows"])
            self.assertEqual(set(remaining["sources"]), {"comparison"})
            paths["comparison"].write_bytes(paths["comparison"].read_bytes() + b" ")
            with self.assertRaisesRegex(ValueError, "Saved source identity changed"):
                browser.load_dataset(manifest, root=self.root)

    def test_optional_finer_factorial_updates_only_html_and_refuses_changed_evidence(self):
        legacy = self.publish()
        self.assertEqual(legacy.returncode, 0, legacy.stdout + legacy.stderr)
        before = {name: (self.output / name).read_bytes() for name in browser.FILES}
        self.assertNotIn(b'id="finer-factorial-title"', before[browser.REPORT])
        relative = browser.FINER_FACTORIAL_SOURCE["path"]
        source = self.root / relative
        shutil.copy2(REPOSITORY / relative, source)

        result = self.publish()
        self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
        html = (self.output / browser.REPORT).read_text(encoding="utf-8")
        _, section = html.split('id="finer-factorial-title"', 1)
        section = section.split("</section>", 1)[0]
        for drive in range(1, 6):
            self.assertIn(f'<th scope="row">{drive}</th>', section)
        self.assertIn("<td>43.9553</td><td>56.7291</td><td>45.1172</td><td>59.8453</td><td>+1.9543</td>", section)
        self.assertIn("<td>44.1129</td><td>56.4771</td><td>45.1627</td><td>59.4515</td><td>+1.9245</td>", section)
        parser = LinkParser()
        parser.feed(html)
        self.assertIn(source, {self.assert_resolves(href) for href in parser.links})
        for name in (browser.DATA, browser.DRIVES, browser.COEFFICIENTS):
            self.assertEqual((self.output / name).read_bytes(), before[name])

        published = snapshot(self.output)
        evidence = json.loads(source.read_bytes())
        first = evidence["factorial_power_comparison"]["per_drive"][0]
        first["signed_effects_percentage_points"]["transmission"]["mixed_difference"] += 1
        write_json(source, evidence)
        failed = self.publish()
        self.assertEqual(failed.returncode, 2, failed.stdout + failed.stderr)
        self.assertIn("Saved source identity changed", failed.stderr)
        self.assertEqual(snapshot(self.output), published)
        with patch.object(browser, "FINER_FACTORIAL_SOURCE", {"path": relative, "sha256": digest(source)}):
            with self.assertRaisesRegex(ValueError, "transmission or mixed change is inconsistent"):
                browser.load_dataset(self.root / "configs/saved_device_dataset_AP_pnc.json", root=self.root)

        source.unlink()
        restored = self.publish()
        self.assertEqual(restored.returncode, 0, restored.stdout + restored.stderr)
        self.assertEqual((self.output / browser.REPORT).read_bytes(), before[browser.REPORT])

    def test_changed_pinned_resolved_input_is_refused_before_publication(self):
        summary = self.root / self.manifest["points"][0]["summary"]["path"]
        configuration = summary.parent / "configuration_AP_pnc.json"
        configuration.write_bytes(configuration.read_bytes() + b" ")
        result = self.publish()
        self.assertEqual(result.returncode, 2, result.stdout + result.stderr)
        self.assertIn("Saved source identity changed", result.stderr)
        self.assertFalse(self.output.exists())

    def test_portable_thin_grid_exports_all_native_rows_and_upgrades_existing_view(self):
        def copy(relative):
            target = self.root / relative
            target.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(REPOSITORY / relative, target)
            return target

        self.assertEqual(self.publish().returncode, 0)
        legacy = snapshot(self.output)
        for module in ("read_thin_device_grid_AP_pnc.py", "thin_device_grid_view_AP_pnc.py"):
            copy(Path("tools") / module)
        proof = copy(browser.THIN_GRID_COMPARISON)
        comparison = json.loads(proof.read_bytes())
        native = {}
        table_paths = set()
        for level in comparison["mesh_levels"]:
            for point in level["samples"]:
                publication = Path(point["archive"]["map"]["path"])
                table = publication.with_name("scattering_map_AP_pnc.csv")
                if table in table_paths:
                    continue
                table_paths.add(table)
                copy(publication)
                with copy(table).open(newline="") as stream:
                    for row in csv.DictReader(stream):
                        key = (level["central_max_edge_nm"], round(float(row["frequency_GHz"]) * 1e9),
                               int(row["incident_port_index"]), row["kind"], int(row["outgoing_port_index"]))
                        self.assertNotIn(key, native)
                        native[key] = (float(row["amplitude_real"]), float(row["amplitude_imag"]),
                                       float(row["power_fraction"]), table.as_posix())
        result = self.publish()
        self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
        self.assertIn("9 thin-device mesh and frequency points", result.stdout)
        with (self.output / browser.GRID_COEFFICIENTS).open(newline="") as stream:
            coefficients = list(csv.DictReader(stream))
        with (self.output / browser.GRID_DRIVES).open(newline="") as stream:
            drives = list(csv.DictReader(stream))
        self.assertEqual((len(table_paths), len(drives), len(coefficients)), (6, 45, 450))
        matched = {(float(d["central_max_edge_nm"]), int(d["frequency_hz"]), int(d["incident_port_index"])): d
                   for d in drives}
        for row in coefficients:
            source = (float(row["central_max_edge_nm"]), int(row["frequency_hz"]), int(row["incident_port_index"]))
            drive = matched[source]
            self.assertEqual((row["reference_drive"], row["native_drive"]), (drive["reference_drive"], drive["native_drive"]))
            self.assertEqual((float(row["amplitude_real"]), float(row["amplitude_imag"]),
                              float(row["power_fraction"]), row["source_csv"]),
                             native.pop((*source, row["kind"], int(row["outgoing_port_index"]))))
        self.assertFalse(native)
        published = snapshot(self.output)
        dataset = browser.load_dataset(self.root / "configs/saved_device_dataset_AP_pnc.json", root=self.root)
        geometry = dataset["thin_grid"]["levels"][0]["points"][0]["hole_csv"]
        geometry_path = self.root / geometry["path"]
        href = browser._href(geometry_path, self.output)
        self.assertEqual(self.assert_resolves(href), geometry_path)
        self.assertIn(f'id="thin-grid-export-hole-geometry" href="{href}"',
                      (self.output / browser.REPORT).read_text(encoding="utf-8"))
        geometry_bytes = geometry_path.read_bytes()
        geometry_path.write_bytes(geometry_bytes + b" ")
        with self.assertRaisesRegex(ValueError, "source identity changed"):
            browser.publish_dataset(dataset, self.output)
        self.assertEqual(snapshot(self.output), published)
        geometry_path.write_bytes(geometry_bytes)
        for name in (browser.DATA, browser.DRIVES, browser.COEFFICIENTS):
            self.assertEqual(published[name], legacy[name])  # bytes and modification times
        self.assertEqual(self.publish().returncode, 0)
        self.assertEqual(snapshot(self.output), published)
        table = self.root / next(iter(table_paths))
        table.write_bytes(table.read_bytes() + b" ")
        self.assertNotEqual(self.publish().returncode, 0)
        self.assertEqual(snapshot(self.output), published)
        proof.unlink()  # Open the same generated directory with only the original archive.
        self.assertEqual(self.publish().returncode, 0)
        self.assertEqual({p.name for p in self.output.iterdir()}, set(browser.FILES))
        self.assertEqual((self.output / browser.REPORT).read_bytes(), legacy[browser.REPORT][1])


if __name__ == "__main__":
    unittest.main()
