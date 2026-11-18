#!/usr/bin/env python3
"""Browse and export three saved device responses without computing. Athul Prem | _AP_pnc."""

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
DEFAULT_MANIFEST = ROOT / "configs/saved_device_dataset_AP_pnc.json"
REPORT = "report_AP_pnc.html"
DATA = "device_results_AP_pnc.json"
DRIVES = "device_drives_AP_pnc.csv"
COEFFICIENTS = "device_coefficients_AP_pnc.csv"
FILES = (REPORT, DATA, DRIVES, COEFFICIENTS)
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
    _require(len({point["id"] for point in points}) == len(points), "Saved point IDs must be distinct.")
    controls = []
    for entry in selection.get("controls", []):
        source, text = _source(root, entry["source"], sources)
        report = (root / entry["report_path"]).resolve()
        _require(report.is_relative_to(root), "A comparison report escapes the workspace.")
        controls.append({"label": entry["label"], "interpretation": entry["interpretation"],
                         "recorded_status": json.loads(text).get("status", "recorded_finite_control"), "source": str(source),
                         "report": str(report), "available": report.is_file(), "archive_only": True,
                         "missing_in_clone": not report.is_file()})
    return {"author": "Athul Prem", "file_tag": "_AP_pnc", "schema_version": 1,
            "artifact_kind": "saved_device_results_browser", "title": selection["title"], "scope": SCOPE,
            "computation_enabled": False, "spatial_convergence_established": False,
            "frequency_convergence_established": False, "points": points, "controls": controls, "sources": sources}


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


def _report(dataset, output):
    view = json.loads(json.dumps(dataset))
    for point in view["points"]:
        for artifact in point["artifacts"]:
            if artifact["available"]:
                artifact["href"] = _href(artifact["path"], output)
        point["evidence_href"] = _href(point["source_evidence"], output)
        point["summary_href"] = _href(point["source_summary"], output)
    sample_rows = "".join(
        f'<tr><td>{point["frequency_hz"] / 1e9:g} GHz</td><td>{len(point["incident_roots"])}</td><td>'
        + " · ".join(
            (f'<a href="{escape(artifact["href"], quote=True)}">{escape(artifact["label"])}</a>'
             + (" (archive-only)" if artifact["archive_only"] else "")) if artifact["available"] else
            escape(artifact["label"]) + (" — archive-only; missing from this clone" if artifact["missing_in_clone"]
                                         else " — unavailable")
            for artifact in point["artifacts"] if Path(artifact["path"]).suffix in (".html", ".json", ".csv"))
        + '</td></tr>' for point in view["points"])
    controls = "".join(f'<tr><td>{escape(row["label"])}</td><td>{escape(row["interpretation"])}</td><td>'
                       + (f'<a href="{escape(_href(row["report"], output), quote=True)}">Saved report (archive-only)</a> · '
                          if row["available"] else "Report archive-only; missing from this clone · ")
                       + f'<a href="{escape(_href(row["source"], output), quote=True)}">Recorded evidence</a></td></tr>'
                       for row in view["controls"])
    payload = json.dumps(view, ensure_ascii=False, allow_nan=False).replace("<", "\\u003c").replace("&", "\\u0026")
    return ('''<!doctype html><html lang="en"><meta charset="utf-8"><meta name="viewport" content="width=device-width">
<title>Saved device results | Athul Prem</title><style>
body{font:16px system-ui,sans-serif;max-width:1120px;margin:2rem auto;padding:0 1rem;color:#172b39;background:#f8fafb}
h1{font-size:1.8rem}section{background:white;padding:1.2rem;border:1px solid #dce4e9;border-radius:10px;margin:1rem 0}
label{display:inline-block;margin:0 1.5rem .8rem 0}select{font:inherit;padding:.45rem}table{border-collapse:collapse;width:100%}
th,td{text-align:left;padding:.6rem;border-bottom:1px solid #e3e9ed}pre{white-space:pre-wrap;overflow-wrap:anywhere}
.summary{font-size:1.2rem;font-weight:600}.muted{color:#465c6d}a{color:#075ba4}#coeffwrap{overflow-x:auto}
</style><h1>''' + escape(dataset["title"]) + '''</h1><p>Athul Prem | athul_simulation_pnc_saved_device_results</p><p>'''
            + escape(SCOPE) + '''</p><table><thead><tr><th>Saved sample</th><th>Incident channels</th><th>Saved files</th></tr></thead><tbody>'''
            + sample_rows + '''</tbody></table><p><a href="''' + DRIVES + '''">Export per-drive powers (CSV)</a> ·
<a href="''' + COEFFICIENTS + '''">Export native complex coefficients (CSV)</a> · <a href="''' + DATA + '''">Inputs, status and sources (JSON)</a></p>
<section><label>Frequency <select id="frequency" aria-label="Frequency"></select></label>
<label>Incident SAFE root <select id="incident" aria-label="Incident SAFE root"></select></label>
<p id="summary" class="summary"></p><p id="status" class="muted"></p><p id="planes"></p>
<div id="coeffwrap"><table><thead><tr><th>Output</th><th>Local SAFE root</th><th>Real amplitude</th>
<th>Imaginary amplitude</th><th>Power fraction</th></tr></thead><tbody id="coefficients"></tbody></table></div>
<p>Amplitudes use each archived unit-power modal basis. Phase is retained at the original end planes.
Changing frequency does not associate roots or interpolate a spectrum.</p></section>
<section><h2>Fields and resolved inputs</h2><ul id="artifacts"></ul><p id="evidence"></p>
<details><summary>Resolved configuration and reproduction guidance</summary><pre id="configuration"></pre>
<p>The linked configuration records the original geometry, nominal material, mesh, frequency and resource inputs.
Recomputing a device point requires measured resource admission and the archived numerical environment;
this browser only reads compact saved data. The compact review bundle includes exact saved tables, configurations,
hole geometry and acceptance/control records; it is not an executable reproduction bundle. Original paths inside
these records retain archival provenance. NPZ, VTU and field/control reports remain archive-only and are linked
only when present in this workspace; their large contents are not loaded or reverified here.</p>
<p>Rebuild this browser from the project directory using standard-library Python with <code>python -B tools/browse_device_results_AP_pnc.py</code>,
or use <strong>Open device results</strong> in the workbench. The JSON export retains the exact compact source identities.</p>
</details></section><section><h2>Accepted endpoint relation and finite controls</h2>
<p>The 0.1 GHz inventory has four channels per direction; the other points have five. Its roots remain separate.
The linked 0.8875/1 GHz relation is a sampled endpoint correspondence, not continuous branch identity.
The browser reuses existing comparisons and does not calculate new scientific differences.</p>
<table><thead><tr><th>Saved comparison</th><th>Recorded interpretation</th><th>Open</th></tr></thead><tbody>'''
            + controls + '''</tbody></table></section><script type="application/json" id="dataset">''' + payload + '''</script><script>
const data=JSON.parse(document.getElementById('dataset').textContent);
const frequency=document.getElementById('frequency'), incident=document.getElementById('incident');
const fmt=x=>Number(x).toPrecision(12);
function link(label,href){const a=document.createElement('a');a.textContent=label;a.href=href;return a;}
function showDrive(){
 const p=data.points[Number(frequency.value)],root=Number(incident.value),d=p.drives.find(r=>r.incident_port_index===root);
 document.getElementById('summary').textContent='Transmission '+fmt(d.transmission)+' · Reflection '+fmt(d.reflection);
 const tbody=document.getElementById('coefficients');tbody.replaceChildren();
 for(const row of p.coefficients.filter(r=>r.incident_port_index===root)){
  const tr=document.createElement('tr');for(const value of [row.kind,row.outgoing_port_index,fmt(row.amplitude_real),fmt(row.amplitude_imag),fmt(row.power_fraction)]){
   const td=document.createElement('td');td.textContent=value;tr.append(td);}tbody.append(tr);}
}
function showPoint(){
 const p=data.points[Number(frequency.value)];incident.replaceChildren();
 for(const root of p.incident_roots){const o=document.createElement('option');o.value=root;o.textContent=root;incident.append(o);}
 document.getElementById('status').textContent='Accepted within the saved discretization · Spatial convergence unresolved · Frequency convergence unresolved';
 document.getElementById('planes').textContent='Reference planes x = '+p.reference_planes_m.map(x=>fmt(x*1e6)).join(' and ')+' µm; left incidence.';
 document.getElementById('configuration').textContent=JSON.stringify(p.configuration,null,2);
 const list=document.getElementById('artifacts');list.replaceChildren();
 for(const a of p.artifacts){const li=document.createElement('li');if(a.available){li.append(link(a.label,a.href));if(a.archive_only)li.append(document.createTextNode(' — archive-only'));}
  else li.textContent=a.label+(a.missing_in_clone?' — archive-only; missing from this clone':' — unavailable');
  li.append(document.createTextNode(': '+a.description));list.append(li);}
 const proof=document.getElementById('evidence');proof.replaceChildren(link('Exact saved summary',p.summary_href),document.createTextNode(' · '),link('Recorded independent acceptance',p.evidence_href));
 showDrive();
}
data.points.forEach((p,i)=>{const o=document.createElement('option');o.value=i;o.textContent=(p.frequency_hz/1e9)+' GHz';frequency.append(o);});
frequency.addEventListener('change',showPoint);incident.addEventListener('change',showDrive);showPoint();
</script></html>''').encode("utf-8")


def publish_dataset(dataset: dict, output: Path) -> dict:
    """Publish four derived browser files, preserving source data and unchanged publications."""
    output = Path(output).resolve()
    for point in dataset["points"]:
        _require(not output.is_relative_to(Path(point["source_summary"]).parent),
                 "The browser output must be outside each compact source directory.")
        _require(not output.is_relative_to(Path(point.get("archive_directory", Path(point["source_summary"]).parent))),
                 "The browser output must be outside each original run directory.")
    drives = [row for point in dataset["points"] for row in point["drives"]]
    coefficients = [row for point in dataset["points"] for row in point["coefficients"]]
    content = {REPORT: _report(dataset, output), DATA: _json_bytes(dataset),
               DRIVES: _csv_bytes(drives), COEFFICIENTS: _csv_bytes(coefficients)}
    if output.exists() and any(output.iterdir()):
        _require(all(path.name in FILES and path.is_file() and not path.is_symlink() for path in output.iterdir())
                 and (output / DATA).is_file(), "Choose an empty directory or an existing saved-device browser.")
        old = json.loads((output / DATA).read_text(encoding="utf-8"))
        _require(old.get("artifact_kind") == "saved_device_results_browser", "Existing files are not a generated device browser.")
    output.parent.mkdir(parents=True, exist_ok=True)
    reused = output.exists() and all((output / name).is_file() and (output / name).read_bytes() == value
                                     for name, value in content.items())
    if not reused:
        with TemporaryDirectory(prefix="device-browser-", dir=output.parent) as temporary:
            staging = Path(temporary)
            for name, value in content.items():
                (staging / name).write_bytes(value)
            output.mkdir(exist_ok=True)
            for name in FILES:
                (staging / name).replace(output / name)
    return {"output_directory": str(output), "report_path": str(output / REPORT), "reused": reused,
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
          f"{result['coefficient_count']} native coefficients. Report: {result['report_path']}")
    if args.open:
        webbrowser.open(Path(result["report_path"]).as_uri())
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
