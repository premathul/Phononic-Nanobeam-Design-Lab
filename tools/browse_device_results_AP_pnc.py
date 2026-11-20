#!/usr/bin/env python3
"""Browse and export saved 26-hole responses without computing. Athul Prem | _AP_pnc."""

from __future__ import annotations

import argparse
import csv
from hashlib import sha256
from html import escape
import io
import json
import math
import os
from pathlib import Path
import sys
from tempfile import TemporaryDirectory
from urllib.parse import quote
import webbrowser


ROOT = Path(__file__).resolve().parents[1]
if __name__ == "__main__":
    sys.path.insert(0, str(ROOT))

from src.phononic.report_style_AP_pnc import REPORT_CSS


DEFAULT_MANIFEST = ROOT / "configs/saved_device_dataset_AP_pnc.json"
THIN_GRID_COMPARISON = Path("verification/thin_mesh_frequency_AP_pnc.json")
FINER_FACTORIAL_SOURCE = {
    "path": "verification/finer_taper_patch_response_1GHz_AP_pnc.json",
    "sha256": "0d3f9e5854025b578c87f1ac4c6f77b6daed73784bccd0ffd423b2636092e111",
}
PAIRED_SOURCES = {
    "spatial": {"path": "verification/paired_taper_contrast_1GHz_AP_pnc.json",
                "sha256": "0e5b36447f120e14930d4e2dcc5916558c5b992213fd44476c06988c5bc6922b"},
    "frequency": {"path": "verification/paired_nested_midpoint_AP_pnc.json",
                  "sha256": "fd2f29826370ea050929a03a684cebdece4826391bcf02918d6b1e67a25dcd52"},
}
GOAL_PATCH_SOURCES = {
    "localization": {"path": "verification/regional_reflection_work_1GHz_AP_pnc.json",
                     "sha256": "d035eabcaedbc851ec02b01fd22d2776964021882fe2899139527b1cfd00e05d"},
    "comparison": {"path": "verification/goal_patch_response_1GHz_AP_pnc.json",
                   "sha256": "29904ca1c8097f1e1bcc2f2d0dd2993e49af0ebba825041acf5cc385307fb7a6"},
}
GOAL_PATCH_REGIONS = (
    ("cavity_13_14", "Cavity · holes 13 and 14", (13, 14)),
    ("outer_taper_09_18", "Outer taper · holes 9 and 18", (9, 18)),
)
GOAL_PATCH_DRIVES = (3, 5)
REPORT = "report_AP_pnc.html"
DATA = "device_results_AP_pnc.json"
DRIVES = "device_drives_AP_pnc.csv"
COEFFICIENTS = "device_coefficients_AP_pnc.csv"
FILES = (REPORT, DATA, DRIVES, COEFFICIENTS)
GRID_DRIVES = "thin_mesh_drives_AP_pnc.csv"
GRID_COEFFICIENTS = "thin_mesh_coefficients_AP_pnc.csv"
GRID_FILES = (GRID_DRIVES, GRID_COEFFICIENTS)
SCOPE = ("Accepted finite computations at three sparse sampled frequencies, using the original native modal bases and "
         "reference planes. Root IDs are local to each frequency. No interpolation, new mode matching, phase "
         "adjustment or scientific computation is performed. Spatial and frequency convergence remain unresolved; "
         "passing finite low-band controls does not establish global convergence.")
LIMIT_BYTES = 4 * 2**20


def _require(condition, message):
    if not condition:
        raise ValueError(message)


def _json_bytes(value):
    return (json.dumps(value, indent=2, ensure_ascii=False, allow_nan=False) + "\n").encode("utf-8")


def _read_small(path, expected=None):
    _require(path.is_file() and path.suffix in (".json", ".csv") and path.stat().st_size <= LIMIT_BYTES,
             f"Missing or oversized compact data source: {path}")
    raw = path.read_bytes()
    actual = sha256(raw).hexdigest()
    _require(expected is None or actual == expected, f"Saved source identity changed: {path}")
    return raw.decode("utf-8-sig"), {"path": str(path), "sha256": actual, "bytes": len(raw)}


def _source(root, entry, sources):
    path = (root / entry["path"]).resolve()
    _require(path.is_relative_to(root), "A saved data source escapes the workspace.")
    _require(isinstance(entry.get("sha256"), str) and len(entry["sha256"]) == 64,
             "Each compact source requires its accepted SHA-256 identity.")
    content, record = _read_small(path, entry["sha256"])
    sources.append(record)
    return path, content


def _number(value):
    number = float(value)
    _require(math.isfinite(number), "A saved table contains a non-finite number.")
    return number


def _table(directory, name, hashes, sources):
    path = directory / name
    text, record = _read_small(path, hashes[name])
    sources.append(record)
    return list(csv.DictReader(io.StringIO(text)))


def _paired_row(label, thin, thick, contrast):
    thin, thick, contrast = map(_number, (thin, thick, contrast))
    _require(0 <= thin <= 100 and 0 <= thick <= 100
             and math.isclose(thin - thick, contrast, rel_tol=0, abs_tol=1e-10),
             "The saved paired transmission contrast differs from its two thicknesses.")
    return (label, thin, thick, contrast)


def _paired_tables(root):
    """Read the published paired evidence when it is present in the default archive."""
    tables = []
    for key, pin in PAIRED_SOURCES.items():
        source = root / pin["path"]
        if not source.exists() and not source.is_symlink():
            continue
        path, text = _source(root, pin, [])
        record = json.loads(text)
        _require(record["schema_version"] == 1 and record["physical_thickness_nm"] == {"thin": 100, "thick": 200}
                 and record["incident_side"] == "left" and record["reference_planes_x_m"] == [0.0, 1.665e-5]
                 and record["spatial_convergence_established"] is False and record["frequency_convergence_established"] is False,
                 "The paired device evidence uses different thicknesses, planes or convergence labels.")
        if key == "spatial":
            _require(record["frequency_hz"] == 1e9 and record["equal_power_input_count"] == 5 and record["hole_count"] == 26
                     and record["coarse_to_fine_edge_targets_nm"] == [225.0, 168.75],
                     "The paired 1 GHz evidence uses different spatial settings.")
            stages = (("baseline", "225 nm baseline"), ("central_only", "168.75 nm taper/cavity patch within the 225 nm mesh"),
                      ("full_refinement", "168.75 nm across the central region"))
            means = record["cases"]
            contrasts = record["paired_thickness_contrasts_percentage_points_100nm_minus_200nm"]
            rows = [_paired_row(label, 100 * means["thin"][stage]["equal_incoherent_power_fractions"]["transmission"],
                                100 * means["thick"][stage]["equal_incoherent_power_fractions"]["transmission"],
                                contrasts[stage]["transmission"]) for stage, label in stages]
            caption, heading = "Mesh comparison at 1 GHz", "Central edge target"
        else:
            frequencies = record["frequencies_hz"]
            mesh = record["nested_mesh"]
            _require(record["status"] == "compared_saved_nested_midpoints" and record["incident_count_per_thickness"] == 5
                     and frequencies == [995000000, 997500000, 1000000000]
                     and mesh["central_max_edge_nm"] == 168.75 and mesh["circle_resolution"] == 32,
                     "The paired frequency evidence uses different samples or spatial settings.")
            nested = record["nested_rim32"]
            means = nested["equal_incoherent_mean_power"]
            rows = [_paired_row(f"{frequency / 1e6:g}", thin["transmission"], thick["transmission"], contrast["transmission"])
                    for frequency, thin, thick, contrast in zip(frequencies, means["thin"]["sample_values_times_100"],
                                                               means["thick"]["sample_values_times_100"],
                                                               nested["thin_minus_thick"]["sample_values_times_100"], strict=True)]
            caption, heading = "Frequency samples at the 168.75 nm central edge target", "Frequency (MHz)"
        tables.append({"caption": caption, "heading": heading, "rows": rows, "source": str(path)})
    if len(tables) == 2:
        _require(all(math.isclose(a, b, rel_tol=0, abs_tol=1e-10)
                     for a, b in zip(tables[0]["rows"][-1][1:], tables[1]["rows"][-1][1:], strict=True)),
                 "The paired tables disagree at their shared 1 GHz point.")
    return tables


def _goal_patch_table(root):
    """Read the compact 1 GHz work and independent patch comparisons when present."""
    records, paths = {}, {}
    for key, pin in GOAL_PATCH_SOURCES.items():
        path = root / pin["path"]
        if not path.exists() and not path.is_symlink():
            continue
        source, text = _source(root, pin, [])
        record = json.loads(text)
        _require(record["frequency_hz"] == 1e9 and record["spatial_convergence_established"] is False
                 and record["frequency_convergence_established"] is False,
                 "The goal-patch evidence has a different frequency or convergence scope.")
        records[key], paths[key] = record, str(source)
    if not records:
        return None

    rows = {}
    if "localization" in records:
        case = records["localization"]["cases"]["thin_first"]["result"]
        alignment = case["mode_alignment"]["incoming"]
        roots = case["incident_roots"]
        _require(case["original_planes_m"] == [0.0, 1.665e-5] and len(set(roots)) == 5
                 and alignment["reference_roots"] == alignment["candidate_roots"] == roots,
                 "The work localization has different end planes or native drive order.")
        work = {item["hole"]: item["delta_T_percentage_points_by_drive"] for item in case["bins"]}
        for region, _, holes in GOAL_PATCH_REGIONS:
            _require(all(len(work[hole]) == 5 for hole in holes), "The saved axial work bins are incomplete.")
            for drive in GOAL_PATCH_DRIVES:
                rows[region, drive] = {"full_bin_work": math.fsum(_number(work[hole][drive - 1]) for hole in holes)}

    if "comparison" in records:
        comparison = records["comparison"]
        _require(comparison["schema_version"] == 1 and comparison["status"] == "compared_saved_goal_patch_controls"
                 and comparison["physical_thickness_nm"] == 100 and comparison["incident_side"] == "left"
                 and comparison["reference_planes_x_m"] == [0.0, 1.665e-5],
                 "The patch responses differ in saved status, thickness, incidence or original end planes.")
        used_work = comparison["provenance"]["parent_work"]
        _require(all(used_work[key] == GOAL_PATCH_SOURCES["localization"][key] for key in ("path", "sha256")),
                 "The patch comparison used a different work-localization record.")
        for region, _, _ in GOAL_PATCH_REGIONS:
            drives = comparison["cases"][region]["per_drive"]
            by_order = {row["reference_drive"]: row for row in drives}
            _require(len(drives) == 5 and set(by_order) == set(range(1, 6)),
                     "The patch responses have an incomplete native drive order.")
            for drive in GOAL_PATCH_DRIVES:
                saved = by_order[drive]
                full = _number(saved["full_bin_work_percentage_points"])
                row = rows.setdefault((region, drive), {"full_bin_work": full})
                _require(math.isclose(row["full_bin_work"], full, rel_tol=0, abs_tol=1e-10),
                         "The patch comparison and localization disagree on the work in the full bins.")
                row.update(full_bin_work=full, replaced_parent_work=_number(saved["prediction_percentage_points"]),
                           patch_solve_change=_number(saved["measured_change_percentage_points"]))

    return {"sources": paths, "rows": [{"patch": label, "drive": drive, **rows[region, drive]}
                                       for region, label, _ in GOAL_PATCH_REGIONS for drive in GOAL_PATCH_DRIVES]}


def _finer_factorial_table(root):
    source = root / FINER_FACTORIAL_SOURCE["path"]
    if not source.exists() and not source.is_symlink():
        return None
    source, text = _source(root, FINER_FACTORIAL_SOURCE, [])
    record = json.loads(text)
    _require(record["schema_version"] == 1 and record["status"] == "compared_saved_finer_taper_factorial"
             and record["frequency_hz"] == 1e9 and record["physical_thickness_nm"] == 100 and record["incident_side"] == "left"
             and record["reference_planes_x_m"] == [0.0, 1.665e-5] and record["central_mesh_targets_nm"] == [168.75, 126.5625]
             and record["spatial_convergence_established"] is False and record["frequency_convergence_established"] is False
             and record["physical_mesh_complement_coverage"]["recombined_tetrahedra_match_full_refinement"] is True,
             "The finer regional evidence uses different inputs, coverage or convergence labels.")
    drives = record["factorial_power_comparison"]["per_drive"]
    baseline = record["cases"]["baseline"]["native_drive_order"]
    roots = [row["native_root_id"] for row in baseline]
    _require([row["reference_drive"] for row in drives] == [row["drive"] for row in baseline] == list(range(1, 6))
             and [row["native_root_id_by_case"]["baseline"] for row in drives] == roots and len(set(roots)) == 5,
             "The finer regional evidence does not match the five baseline drives.")
    rows = []
    for row in drives:
        transmission = row["power_fractions"]["transmission"]
        coarse, center, exterior, both = (
            _number(transmission[key]) for key in ("baseline", "taper_patch", "complement", "full_refinement")
        )
        mixed = _number(row["signed_effects_percentage_points"]["transmission"]["mixed_difference"])
        _require(all(0 <= power <= 1 for power in (coarse, center, exterior, both))
                 and math.isclose(mixed, 100 * math.fsum((both, -center, -exterior, coarse)), rel_tol=0, abs_tol=1e-10),
                 "The finer regional transmission or mixed change is inconsistent.")
        rows.append((row["reference_drive"], 100 * coarse, 100 * center, 100 * exterior, 100 * both, mixed))
    return {"source": str(source), "rows": rows}


def load_dataset(manifest: Path = DEFAULT_MANIFEST, *, root: Path = ROOT) -> dict:
    """Read pinned compact sources; archive-only fields/reports are checked for existence."""
    root = Path(root).resolve()
    manifest = Path(manifest).resolve()
    text, manifest_record = _read_small(manifest)
    selection = json.loads(text)
    _require(selection["schema_version"] == 1 and selection["points"], "Select a version 1 saved-device dataset.")
    sources, points = [manifest_record], []
    for entry in selection["points"]:
        summary_path, text = _source(root, entry["summary"], sources)
        summary = json.loads(text)
        evidence_path, text = _source(root, entry["evidence"], sources)
        evidence = json.loads(text)
        frequency = _number(entry["frequency_hz"])
        roots = summary["incident_mode_indices"]
        n = len(roots)
        _require(n > 0 and len(set(roots)) == n and all(type(value) is int for value in roots)
                 and summary["status"] == "completed_unverified" and summary["analysis_kind"] == "scattering"
                 and summary["frequency_hz"] == frequency and summary["incident_side"] == "left"
                 and summary["num_incident_channels"] == n and summary["power_balance_passed"] is True,
                 "The selected saved response lacks its completed, power-checked incident inventory.")
        if "scattering_export" in evidence:
            accepted_summary = evidence["scattering_export"]["public_summary_sha256"]
        else:
            _require(evidence["status"] == "point_verified_within_saved_discretization"
                     and evidence["frequency_hz"] == frequency, "The recorded point evidence differs.")
            accepted_summary = evidence["public_summary_sha256"]
        _require(accepted_summary == entry["summary"]["sha256"], "Recorded acceptance refers to another public summary.")
        directory, hashes = summary_path.parent, summary["output_sha256"]
        archive_directory = (root / entry.get("archive_directory", directory)).resolve()
        _require(archive_directory.is_relative_to(root), "An original run directory escapes the workspace.")
        config_text, config_record = _read_small(directory / "configuration_AP_pnc.json", hashes["configuration_AP_pnc.json"])
        sources.append(config_record)
        configuration = json.loads(config_text)
        _require(configuration == summary["configuration"], "Saved resolved inputs differ from their summary.")
        modes = _table(directory, "port_modes_AP_pnc.csv", hashes, sources)
        open_modes = {int(row["port_index"]): row for row in modes if row["propagating"] == "True"}
        forward = [key for key, row in open_modes.items() if row["direction"] == "+x"]
        backward = [key for key, row in open_modes.items() if row["direction"] == "-x"]
        _require(set(forward) == set(roots) and len(backward) == n and len(open_modes) == 2 * n,
                 "Saved signed open-channel inventory differs from the accepted response.")
        for mode in open_modes.values():
            _require(_number(mode["k_imag_rad_m"]) == 0 and math.isclose(
                _number(mode["signed_power_W"]), 1 if mode["direction"] == "+x" else -1, abs_tol=1e-7),
                "Saved channels lack their recorded signed unit-power basis.")
        drives = []
        totals = _table(directory, "scattering_AP_pnc.csv", hashes, sources)
        _require(len(totals) == n, "The saved drive table is incomplete.")
        orders = set()
        for row in totals:
            order = int(row["incident_mode_order"])
            _require(1 <= order <= n and order not in orders
                     and math.isclose(_number(row["frequency_GHz"]) * 1e9, frequency, rel_tol=1e-14),
                     "The saved drive order or frequency differs.")
            orders.add(order)
            drives.append({"point_id": entry["id"], "frequency_GHz": frequency / 1e9,
                           "incident_port_index": roots[order - 1], "incident_mode_order": order,
                           **{key: _number(value) for key, value in row.items()
                              if key not in ("frequency_GHz", "incident_mode_order")}})
        drives.sort(key=lambda row: row["incident_mode_order"])
        coefficients, identities = [], set()
        for row in _table(directory, "scattering_channels_AP_pnc.csv", hashes, sources):
            kind = row["kind"]
            order, incoming, outgoing = (int(row[key]) for key in
                                         ("incident_mode_order", "incident_port_index", "outgoing_port_index"))
            identity = kind, incoming, outgoing
            _require(kind in ("reflection", "transmission") and 1 <= order <= n and roots[order - 1] == incoming
                     and outgoing in (backward if kind == "reflection" else roots) and identity not in identities,
                     "A saved coefficient has a duplicated or invalid frequency-local signed identity.")
            identities.add(identity)
            real, imag, power = (_number(row[key]) for key in ("amplitude_real", "amplitude_imag", "power_fraction"))
            _require(power >= 0 and math.isclose(power, real * real + imag * imag, rel_tol=1e-8, abs_tol=1e-15),
                     "Saved complex amplitudes and their recorded unit-power fractions differ.")
            coefficients.append({"point_id": entry["id"], "frequency_GHz": frequency / 1e9, "kind": kind,
                                 "incident_mode_order": order, "incident_port_index": incoming,
                                 "outgoing_port_index": outgoing, "amplitude_real": real,
                                 "amplitude_imag": imag, "power_fraction": power})
        _require(len(coefficients) == 2 * n * n, "The saved complex-coefficient table is incomplete.")
        for drive in drives:
            for kind in ("reflection", "transmission"):
                total = math.fsum(row["power_fraction"] for row in coefficients
                                  if row["kind"] == kind and row["incident_port_index"] == drive["incident_port_index"])
                _require(math.isclose(total, drive[kind], abs_tol=1e-10, rel_tol=0),
                         "Saved channel powers do not join to the per-drive totals.")
        artifacts = []
        for filename, label, description in (
            (REPORT, "Original 3D field report", "Interactive archived visualization; display amplitude is arbitrary."),
            ("selected_fields_AP_pnc.npz", "Full quadratic-field archive (NPZ)",
             "All accepted driven complex displacement fields and quadratic-element DOF mappings; displacement in m/sqrt(W)."),
            ("selected_vertex_shapes_AP_pnc.vtu", "Vertex visualization (VTU)",
             "Shape-normalized real/imaginary vertex samples for at most the first four drives; not the complete quadratic field."),
            ("configuration_AP_pnc.json", "Resolved configuration", "Exact saved inputs for this accepted computation."),
        ):
            archive_only = filename != "configuration_AP_pnc.json" and "archive_directory" in entry
            path = (archive_directory if archive_only else directory) / filename
            available = path.is_file()
            artifacts.append({"label": label, "description": description, "path": str(path),
                              "available": available, "archive_only": archive_only,
                              "missing_in_clone": archive_only and not available,
                              "recorded_sha256": hashes.get(filename),
                              "full_field_bytes_reverified": False})
        hole_pattern = summary.get("hole_pattern_provenance")
        if hole_pattern is not None:
            path, _ = _source(root, {"path": hole_pattern["source_csv"],
                                    "sha256": hole_pattern["source_sha256"]}, sources)
            artifacts.append({"label": "Exact hole geometry table (CSV)",
                              "description": "Pinned source table used for the saved hole pattern and placement.",
                              "path": str(path), "available": True, "archive_only": False,
                              "missing_in_clone": False, "recorded_sha256": hole_pattern["source_sha256"],
                              "full_field_bytes_reverified": False})
        points.append({"id": entry["id"], "frequency_hz": frequency, "configuration": configuration,
                       "incident_roots": roots, "reflected_roots": backward,
                       "reference_planes_m": evidence.get("reference_planes_m", [0.0, configuration["geometry"]["length_nm"] * 1e-9]),
                       "recorded_evidence_status": evidence["status"], "point_status": "accepted_within_saved_discretization",
                       "source_summary": str(summary_path), "source_evidence": str(evidence_path),
                       "archive_directory": str(archive_directory),
                       "spatial_convergence_established": False, "frequency_convergence_established": False,
                       "drives": drives, "coefficients": coefficients, "artifacts": artifacts})
    _require(all(isinstance(point["id"], str) and point["id"].strip() for point in points)
             and len({point["id"] for point in points}) == len(points), "Saved point IDs must be nonempty and distinct.")
    controls = []
    for entry in selection.get("controls", []):
        source, text = _source(root, entry["source"], sources)
        report = (root / entry["report_path"]).resolve()
        _require(report.is_relative_to(root), "A comparison report escapes the workspace.")
        controls.append({"label": entry["label"], "interpretation": entry["interpretation"],
                         "recorded_status": json.loads(text).get("status", "recorded_finite_control"), "source": str(source),
                         "report": str(report), "available": report.is_file(), "archive_only": True,
                         "missing_in_clone": not report.is_file()})
    dataset = {"author": "Athul Prem", "file_tag": "_AP_pnc", "schema_version": 1,
               "artifact_kind": "saved_device_results_browser", "title": selection["title"], "scope": SCOPE,
               "computation_enabled": False, "spatial_convergence_established": False,
               "frequency_convergence_established": False, "points": points, "controls": controls, "sources": sources}
    if manifest == (root / DEFAULT_MANIFEST.relative_to(ROOT)).resolve():
        comparison = root / THIN_GRID_COMPARISON
        if comparison.exists() or comparison.is_symlink():
            from tools.read_thin_device_grid_AP_pnc import load_thin_grid
            dataset["thin_grid"] = load_thin_grid(root)
        if paired := _paired_tables(root):
            dataset["paired_thickness_tables"] = paired
        if goal_patch := _goal_patch_table(root):
            dataset["goal_patch_table"] = goal_patch
        if finer_factorial := _finer_factorial_table(root):
            dataset["finer_factorial_table"] = finer_factorial
    return dataset


def _href(path, output):
    try:
        return quote(Path(os.path.relpath(path, output)).as_posix(), safe="/")
    except ValueError:
        return Path(path).as_uri()


def _csv_bytes(rows):
    stream = io.StringIO(newline="")
    fields = list(dict.fromkeys(key for row in rows for key in row))
    writer = csv.DictWriter(stream, fieldnames=fields)
    writer.writeheader()
    writer.writerows(rows)
    return stream.getvalue().encode("utf-8")


def _thin_grid_exports(grid):
    """Export original-plane native values with the matched incident drive as a separate column."""
    drives, coefficients = [], []
    source_root = Path(grid["source_comparison"]).parent.parent
    drive_fields = ("reference_drive", "native_drive", "incident_port_index", "wavenumber_rad_m", "transmission", "reflection")
    coefficient_fields = ("incident_port_index", "kind", "outgoing_port_index", "amplitude_real", "amplitude_imag", "power_fraction")
    for level in grid["levels"]:
        for point in level["points"]:
            source = Path(point["source_coefficients"]).relative_to(source_root).as_posix()
            context = {"central_max_edge_nm": level["central_max_edge_nm"], "frequency_hz": int(point["frequency_hz"])}
            native_drives = {drive["incident_port_index"]: drive for drive in point["drives"]}
            for drive in point["drives"]:
                drives.append({**context, **{key: drive[key] for key in drive_fields}, "source_csv": source})
            for row in point["coefficients"]:
                drive = native_drives[row["incident_port_index"]]
                coefficients.append({**context, "reference_drive": drive["reference_drive"], "native_drive": drive["native_drive"],
                                     **{key: row[key] for key in coefficient_fields}, "source_csv": source})
    return {GRID_DRIVES: _csv_bytes(drives), GRID_COEFFICIENTS: _csv_bytes(coefficients)}


def _paired_section(tables, output):
    if not tables:
        return ""
    parts = [('<section aria-labelledby="paired-title"><h2 id="paired-title">Paired 100 and 200 nm devices</h2>'
              '<p>Both 26-hole devices use 32 rim edges per hole and original end planes at 0 and 16.65 µm. '
              'Mean transmission (T) averages five equal-power, mutually incoherent left-incident modes for each thickness.</p>')]
    for table in tables:
        rows = "".join(f'<tr><th scope="row">{escape(label)}</th><td>{thin:.4f}</td><td>{thick:.4f}</td>'
                       f'<td>{contrast:+.4f}</td></tr>' for label, thin, thick, contrast in table["rows"])
        source = escape(_href(table["source"], output), quote=True)
        parts.append(f'<div style="overflow-x:auto"><table><caption>{escape(table["caption"])}</caption><thead><tr>'
                     f'<th scope="col">{escape(table["heading"])}</th><th scope="col">100 nm mean T (%)</th>'
                     f'<th scope="col">200 nm mean T (%)</th><th scope="col">100 − 200 nm (pp)</th></tr></thead>'
                     f'<tbody>{rows}</tbody></table></div><p><a href="{source}">Verification evidence for {escape(table["caption"].lower())}</a></p>')
    parts.append('<p>Differences are in percentage points (pp). Spatial and frequency convergence remain unresolved.</p></section>')
    return "".join(parts)


def _goal_patch_section(table, output):
    if not table:
        return ""
    compared = "comparison" in table["sources"]
    headings = ('<th scope="col">Work on replaced parents (pp)</th><th scope="col">Patch solve ΔT (pp)</th>'
                if compared else "")
    rows = []
    for row in table["rows"]:
        patch_values = (f'<td>{row["replaced_parent_work"]:+.3f}</td><td>{row["patch_solve_change"]:+.3f}</td>'
                        if compared else "")
        rows.append(f'<tr><th scope="row">{escape(row["patch"])}</th><td>{row["drive"]}</td>'
                    f'<td>{row["full_bin_work"]:+.3f}</td>{patch_values}</tr>')
    descriptions = (
        "Full-bin work assigns part of the transmission change from the joint taper and cavity refinement to the "
        "regions around those holes. "
        + ("The other columns show work restricted to the tetrahedra replaced in each conforming patch and the "
           "change from solving that patch independently. " if compared else "")
        + "The work estimates depend on the transfer of fields between the original meshes. "
        "Spatial and frequency convergence remain unresolved."
    )
    links = []
    for key, label in (("localization", "Regional work evidence"), ("comparison", "Independent patch comparisons")):
        if key in table["sources"]:
            source = escape(_href(table["sources"][key], output), quote=True)
            links.append(f'<a href="{source}">{label}</a>')
    return ('<section aria-labelledby="goal-patch-title"><h2 id="goal-patch-title">Where refinement changes transmission</h2>'
            '<p>100 nm device at 1 GHz. The local mesh target decreases from 225 to 168.75 nm. '
            'Drive numbers follow the 225 nm mesh. Changes are in percentage points (pp).</p>'
            '<div style="overflow-x:auto"><table><thead><tr><th scope="col">Patch</th><th scope="col">Drive</th>'
            f'<th scope="col">Full-bin work (pp)</th>{headings}</tr></thead><tbody>{"".join(rows)}</tbody></table></div>'
            f'<p>{descriptions}</p><p>{" · ".join(links)}</p></section>')


def _finer_factorial_section(table, output):
    if not table:
        return ""
    rows = "".join(f'<tr><th scope="row">{drive}</th><td>{coarse:.4f}</td><td>{center:.4f}</td>'
                   f'<td>{exterior:.4f}</td><td>{both:.4f}</td><td>{mixed:+.4f}</td></tr>'
                   for drive, coarse, center, exterior, both, mixed in table["rows"])
    source = escape(_href(table["source"], output), quote=True)
    return ('<section aria-labelledby="finer-factorial-title"><h2 id="finer-factorial-title">Finer regional refinement at 1 GHz</h2>'
            '<p>The 100 nm device uses edge targets of 168.75 nm for the baseline and 126.5625 nm in the refined regions. '
            'Drive numbers follow the baseline. The mixed change is the joint transmission change minus the sum of the '
            'two isolated changes, in percentage points (pp).</p>'
            '<div style="overflow-x:auto"><table><thead><tr><th scope="col">Drive</th><th scope="col">Baseline T (%)</th>'
            '<th scope="col">Center-only T (%)</th><th scope="col">Exterior-only T (%)</th>'
            '<th scope="col">Both refined T (%)</th><th scope="col">Mixed ΔT (pp)</th></tr></thead>'
            f'<tbody>{rows}</tbody></table></div><p><a href="{source}">Finer regional verification evidence</a></p></section>')


def _point_sources(point):
    """Keep available native links in the document for readers without JavaScript."""
    artifacts = []
    for artifact in point["artifacts"]:
        label = escape(artifact["label"])
        if artifact["available"]:
            href = escape(artifact["href"], quote=True)
            label = f'<a href="{href}">{label}</a>'
            availability = " (archive-only)" if artifact["archive_only"] else ""
        else:
            availability = " (archive unavailable)" if artifact["archive_only"] else " (unavailable)"
        artifacts.append(f'<li>{label}{availability}. {escape(artifact["description"])}</li>')
    summary = escape(point["summary_href"], quote=True)
    evidence = escape(point["evidence_href"], quote=True)
    return (f'<article data-point-sources="{escape(point["id"], quote=True)}"><h4>{point["frequency_hz"] / 1e9:g} GHz</h4>'
            f'<ul><li><a href="{summary}">Exact saved summary</a></li>'
            f'<li><a href="{evidence}">Independent acceptance evidence</a></li>{"".join(artifacts)}</ul></article>')


_SELECTION_SCRIPT = r'''<script>
const savedSelection = (() => {
    "use strict";
    const views = [];
    const fragment = (pointId, root) => "#" + new URLSearchParams({point: pointId, root});
    function select(pointId, root) {
        const hash = fragment(pointId, root);
        if (location.hash !== hash) history.pushState(null, "", hash);
        for (const view of views) view.notice.textContent = "";
    }
    function restore(initial = false) {
        for (const view of views) view.notice.textContent = "";
        if (!location.hash) {
            for (const view of views) view.show(...view.defaultSelection);
            return;
        }
        const parameters = new URLSearchParams(location.hash.slice(1));
        const pointIds = parameters.getAll("point");
        const roots = parameters.getAll("root");
        const view = pointIds.length === 1 && views.find(item => item.points.has(pointIds[0]));
        const point = view && view.points.get(pointIds[0]);
        const root = point && roots.length === 1 && [...parameters].length === 2
            ? point.roots.find(candidate => String(candidate) === roots[0]) : undefined;
        const target = view || views[0];
        if (root !== undefined) {
            target.show(pointIds[0], root);
        } else if (point) {
            target.show(pointIds[0], point.defaultRoot);
            target.notice.textContent = "Selection link has an invalid incident root. Showing local root " +
                point.defaultRoot + " at the linked sample.";
        } else {
            target.show(...target.defaultSelection);
            target.notice.textContent = "Selection link has an invalid point ID. Showing the default sample and incident root.";
        }
        if (initial) requestAnimationFrame(() => target.section.scrollIntoView());
    }
    function start() {
        // Section scripts initialize their own defaults before the fragment is restored.
        window.addEventListener("hashchange", () => restore());
        if (location.hash) restore(true);
    }
    function csvCell(value) {
        const text = Object.is(value, -0) ? "-0.0" : String(value);
        return /[",\r\n]/.test(text) ? '"' + text.replaceAll('"', '""') + '"' : text;
    }
    return {register: view => views.push(view), fragment, select, start, csvCell};
})();
</script>'''


_H80_INTERACTION = r'''<script>
(() => {
    "use strict";
    const id = name => document.getElementById(name);
    const data = JSON.parse(id("dataset").textContent);
    const frequency = id("frequency");
    const incident = id("incident");
    const points = new Map(data.points.map(point => [point.id, point]));
    const sources = document.querySelectorAll("[data-point-sources]");
    const fmt = x => Number(x).toPrecision(12);

    function option(value, label) {
        const element = document.createElement("option");
        element.value = value;
        element.textContent = label;
        return element;
    }
    function showDrive(point, root) {
        const drive = point.drives.find(row => row.incident_port_index === root);
        id("summary").textContent = "Transmission " + fmt(drive.transmission) + " · Reflection " + fmt(drive.reflection);
        const rows = point.coefficients.filter(row => row.incident_port_index === root);
        id("coefficients").replaceChildren(...rows.map(row => {
            const tr = document.createElement("tr");
            for (const value of [row.kind, row.outgoing_port_index, fmt(row.amplitude_real),
                                 fmt(row.amplitude_imag), fmt(row.power_fraction)]) {
                const td = document.createElement("td");
                td.textContent = value;
                tr.append(td);
            }
            return tr;
        }));
        const columns = Object.keys(rows[0]);
        const records = rows.map(row => columns.map(column => savedSelection.csvCell(row[column])).join(","));
        const download = id("h80-selected-coefficients");
        download.href = "data:text/csv;charset=utf-8," +
            encodeURIComponent([columns.join(","), ...records].join("\r\n") + "\r\n");
        download.download = "point_" + point.id.replace(/[^0-9A-Za-z._-]/g, "_") + "_root" + root + "_AP_pnc.csv";
        id("h80-selection-link").href = savedSelection.fragment(point.id, root);
    }
    function showPoint(pointId, root) {
        const point = points.get(pointId);
        frequency.value = pointId;
        incident.replaceChildren(...point.incident_roots.map(value => option(value, value)));
        incident.value = String(root);
        id("h80-inventory").textContent = point.incident_roots.length + " propagating channels per direction at " +
            point.frequency_hz / 1e9 + " GHz.";
        id("planes").textContent = "Original reference planes at x = " +
            point.reference_planes_m.map(x => Number((x * 1e6).toPrecision(12))).join(" and ") +
            " µm. Left incidence.";
        for (const source of sources) source.hidden = source.dataset.pointSources !== pointId;
        showDrive(point, root);
    }
    frequency.append(...data.points.map(point => option(point.id, point.frequency_hz / 1e9 + " GHz")));
    const defaultSelection = [data.points[0].id, data.points[0].incident_roots[0]];
    frequency.addEventListener("change", () => {
        const point = points.get(frequency.value);
        const root = point.incident_roots[0];
        showPoint(point.id, root);
        savedSelection.select(point.id, root);
    });
    incident.addEventListener("change", () => {
        const point = points.get(frequency.value);
        const root = Number(incident.value);
        showDrive(point, root);
        savedSelection.select(point.id, root);
    });
    savedSelection.register({
        section: id("h80-samples"),
        notice: id("h80-selection-notice"),
        defaultSelection,
        points: new Map(data.points.map(point => [point.id, {
            roots: point.incident_roots, defaultRoot: point.incident_roots[0]
        }])),
        show: showPoint
    });
    showPoint(...defaultSelection);
})();
</script>'''


def _report(dataset, output, paired_tables=(), goal_patch_table=None, thin_section="", finer_factorial_table=None):
    view = json.loads(json.dumps(dataset))
    for point in view["points"]:
        for artifact in point["artifacts"]:
            if artifact["available"]:
                artifact["href"] = _href(artifact["path"], output)
        point["evidence_href"] = _href(point["source_evidence"], output)
        point["summary_href"] = _href(point["source_summary"], output)
    point_sources = "".join(_point_sources(point) for point in view["points"])
    controls = "".join(f'<tr><th scope="row">{escape(row["label"])}</th><td>{escape(row["interpretation"])}</td><td>'
                       + (f'<a href="{escape(_href(row["report"], output), quote=True)}">Saved report (archive-only)</a>'
                          if row["available"] else "Report archive unavailable")
                       + f' · <a href="{escape(_href(row["source"], output), quote=True)}">Recorded evidence</a></td></tr>'
                       for row in view["controls"])
    payload = json.dumps(view, ensure_ascii=False, allow_nan=False).replace("<", "\\u003c").replace("&", "\\u0026")
    supplements = (_finer_factorial_section(finer_factorial_table, output)
                   + _goal_patch_section(goal_patch_table, output) + _paired_section(paired_tables, output))
    title = "Saved 26-hole device results" if thin_section or supplements else dataset["title"]
    h80_title = "Archived h80, six-layer samples" if thin_section or supplements else "Saved samples"
    point_count = len(view["points"])
    drive_count = sum(len(point["drives"]) for point in view["points"])
    coefficient_count = sum(len(point["coefficients"]) for point in view["points"])
    return f'''<!doctype html><html lang="en"><meta charset="utf-8"><meta name="viewport" content="width=device-width">
<title>Saved device results | Athul Prem</title><style>
{REPORT_CSS}
main{{width:min(70rem,calc(100% - 2rem))}}
section{{background:white;padding:1.2rem;border:1px solid var(--line);border-radius:10px;margin:1rem 0}}
.sample-selectors{{display:flex;flex-wrap:wrap;gap:.75rem 1.5rem}}
.sample-selectors label{{display:block;font-weight:600}}
select{{font:inherit;padding:.45rem;color:inherit}}
.summary{{font-size:1.2rem;font-weight:600}}
.muted{{color:var(--muted)}}
.export-links{{display:flex;flex-wrap:wrap;gap:.35rem 1.1rem;margin:.65rem 0}}
.selection-notice:not(:empty){{border-left:3px solid var(--accent);padding:.5rem .75rem}}
.table-scroll table{{min-width:34rem}}
</style><body><main><header><h1>{escape(title)}</h1><p>Athul Prem | athul_simulation_pnc_saved_device_results</p>
<p>Use a selection link to reopen a sample and its incident root. Browser Back and Forward retrace selections.</p>
</header>{_SELECTION_SCRIPT}{thin_section}
<section id="h80-samples" aria-labelledby="h80-title"><h2 id="h80-title">{h80_title}</h2>
<div class="sample-selectors">
<label for="frequency">Frequency <select id="frequency"></select></label>
<label for="incident">Incident SAFE root <select id="incident"></select></label></div>
<p id="h80-selection-notice" class="selection-notice" role="status"></p>
<p id="summary" class="summary" aria-live="polite"></p>
<p>Accepted within the saved discretization. Spatial and frequency convergence remain unresolved.</p>
<p id="h80-inventory"></p><p id="planes"></p>
<div class="table-scroll" id="coeffwrap" role="region" aria-labelledby="h80-coefficients-title" tabindex="0">
<table><caption id="h80-coefficients-title">Outgoing coefficients for the selected drive</caption>
<thead><tr><th scope="col">Output</th><th scope="col">Local SAFE root</th><th scope="col">Real amplitude</th>
<th scope="col">Imaginary amplitude</th><th scope="col">Power fraction</th></tr></thead><tbody id="coefficients"></tbody></table></div>
<p>Power fractions are normalized to the incident power. Complex amplitudes retain their native phases at the original end planes.
Mode roots are local to each frequency.</p>
<nav class="export-links" aria-label="Selected saved drive">
<a id="h80-selected-coefficients" download>Selected drive coefficients (CSV)</a>
<a id="h80-selection-link">Link to selected drive</a></nav>
<h3>All {point_count} saved samples</h3>
<nav class="export-links" aria-label="Downloads for all saved samples">
<a href="{DRIVES}" download>All {drive_count} drive powers (CSV)</a>
<a href="{COEFFICIENTS}" download>All {coefficient_count} native coefficients (CSV)</a>
<a href="{DATA}" download>All samples, inputs and evidence (JSON)</a></nav>
<details><summary>Sources for the selected sample</summary>{point_sources}</details>
<noscript><p>Enable JavaScript to select a drive. The all-sample exports and sources above remain available.</p></noscript>
</section>{supplements}
<section aria-labelledby="h80-controls-title"><h2 id="h80-controls-title">Saved endpoint relation and finite controls</h2>
<p>The 0.1 GHz sample has four channels per direction. The 0.8875 and 1 GHz samples have five.
The recorded relation links those two endpoints only. The browser displays the sampled data without interpolation.</p>
<div class="table-scroll" role="region" aria-label="Recorded endpoint and finite controls" tabindex="0">
<table><thead><tr><th scope="col">Saved comparison</th><th scope="col">Recorded interpretation</th>
<th scope="col">Sources</th></tr></thead><tbody>{controls}</tbody></table></div>
<p class="muted">Original field reports, quadratic fields, vertex visualizations and comparison reports are archive-only.
Links are provided for available files. The browser checks for their presence without reading or re-verifying large files.</p>
<details><summary>Reproduction guidance</summary>
<p>The compact bundle includes pinned tables, original resolved inputs, hole geometry and acceptance evidence.
Read the original paths in those records as archival provenance. Recomputing a point requires measured resource admission
and the recorded numerical environment.</p>
<p>Use <strong>Browse device results</strong> in the workbench or run
<code>python -B tools/browse_device_results_AP_pnc.py --open</code> from the project directory to rebuild this report.</p>
</details></section></main>
<script type="application/json" id="dataset">{payload}</script>{_H80_INTERACTION}
<script>savedSelection.start();</script></body></html>'''.encode("utf-8")


def publish_dataset(dataset: dict, output: Path) -> dict:
    """Publish the read-only report and the available datasets' CSV exports."""
    output = Path(output).resolve()
    for point in dataset["points"]:
        _require(not output.is_relative_to(Path(point["source_summary"]).parent),
                 "The browser output must be outside each compact source directory.")
        _require(not output.is_relative_to(Path(point.get("archive_directory", Path(point["source_summary"]).parent))),
                 "The browser output must be outside each original run directory.")
    drives = [row for point in dataset["points"] for row in point["drives"]]
    coefficients = [row for point in dataset["points"] for row in point["coefficients"]]
    archive_dataset = {key: value for key, value in dataset.items()
                       if key not in ("paired_thickness_tables", "goal_patch_table", "thin_grid", "finer_factorial_table")}
    content = {DATA: _json_bytes(archive_dataset),
               DRIVES: _csv_bytes(drives), COEFFICIENTS: _csv_bytes(coefficients)}
    grid = dataset.get("thin_grid")
    thin_section = ""
    if grid:
        from tools.thin_device_grid_view_AP_pnc import render_thin_device_grid
        geometry = grid["levels"][0]["points"][0]["hole_csv"]
        for level in grid["levels"]:
            for point in level["points"]:
                _require(point["hole_csv"] == geometry, "The thin-device grid requires one shared hole geometry table.")
                _require(not output.is_relative_to(Path(point["source_map"]).parent),
                         "Choose a browser output outside each native mesh and frequency map.")
        geometry_path, _ = _source(Path(grid["source_comparison"]).parent.parent, geometry, [])
        content.update(_thin_grid_exports(grid))
        thin_section = render_thin_device_grid(grid, output, href=_href, drive_csv=GRID_DRIVES,
                                               coefficient_csv=GRID_COEFFICIENTS,
                                               hole_geometry_csv_href=_href(geometry_path, output))
    content[REPORT] = _report(archive_dataset, output, dataset.get("paired_thickness_tables", ()),
                              dataset.get("goal_patch_table"), thin_section, dataset.get("finer_factorial_table"))
    if output.exists() and any(output.iterdir()):
        _require(all(path.name in FILES + GRID_FILES and path.is_file() and not path.is_symlink() for path in output.iterdir())
                 and (output / DATA).is_file(), "Choose an empty directory or an existing saved-device browser.")
        old = json.loads((output / DATA).read_text(encoding="utf-8"))
        _require(old.get("artifact_kind") == "saved_device_results_browser", "Existing files are not a generated device browser.")
    output.parent.mkdir(parents=True, exist_ok=True)
    updates = {name: data for name, data in content.items()
               if not (output / name).is_file() or (output / name).read_bytes() != data}
    obsolete = [output / name for name in GRID_FILES if name not in content and (output / name).is_file()]
    if updates:
        with TemporaryDirectory(prefix="device-browser-", dir=output.parent) as temporary:
            staging = Path(temporary)
            for name, value in updates.items():
                (staging / name).write_bytes(value)
            output.mkdir(exist_ok=True)
            # Install exports before the report that links them.
            for name in updates:
                (staging / name).replace(output / name)
    for path in obsolete:
        path.unlink(missing_ok=True)
    return {"output_directory": str(output), "report_path": str(output / REPORT), "reused": not updates and not obsolete,
            "thin_grid_point_count": sum(len(level["points"]) for level in grid["levels"]) if grid else 0,
            "point_count": len(dataset["points"]), "drive_count": len(drives), "coefficient_count": len(coefficients)}


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, default=DEFAULT_MANIFEST)
    parser.add_argument("--output", type=Path, default=ROOT / "results/runs/saved_device_results_AP_pnc")
    parser.add_argument("--open", action="store_true", help="Open the generated local report.")
    args = parser.parse_args(argv)
    try:
        result = publish_dataset(load_dataset(args.manifest), args.output)
    except (OSError, ValueError, KeyError, TypeError) as exc:
        print(f"Saved device results could not be opened: {exc}", file=sys.stderr)
        return 2
    print(f"Opened {result['point_count']} saved points, {result['drive_count']} drives and "
          f"{result['coefficient_count']} native coefficients from the selected archive.")
    if result["thin_grid_point_count"]:
        print(f"Opened {result['thin_grid_point_count']} thin-device mesh and frequency points.")
    print(f"Report: {result['report_path']}")
    if args.open:
        webbrowser.open(Path(result["report_path"]).as_uri())
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
