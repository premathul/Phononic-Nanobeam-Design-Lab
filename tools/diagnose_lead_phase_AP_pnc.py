"""Audit a bounded saved phase-only model. Athul Prem | _AP_pnc.

Only the immutable accepted comparison is read; no arrays or physics are run.
The reverse-triangle bound permits an independent phase for every coefficient,
so it also bounds every diagonal, reflectionless, unmixed lead-phase model.
"""
from __future__ import annotations

import argparse
import cmath
import csv
import hashlib
from html import escape
from io import StringIO
import json
import math
from pathlib import Path
import platform
import sys
from time import monotonic

import psutil

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from phononic.report_style_AP_pnc import available_link, report_link, report_page, scroll_table

AUTHOR = {"author": "Athul Prem", "file_tag": "_AP_pnc"}
COMPARISON_PATH = Path("results/runs/device_frequency_inplane_8875_AP_pnc/comparison/frequency_inplane_comparison_AP_pnc.json")
COMPARISON_SHA = "107d1c62817349d0a6900650d10dd3826ee492ca4bc36abc8195ee3c3e7911c7"
MODES_SHA = "a2cf1204490983c766d84292bfc85a0182da9997e56fbf2e7281251e11864b32"
SECTION_SHA = "2a2aa9347921ac6940f851f2eb53bc06480a28a6cf67c27a000faa357b1404e8"
SAFE_SHA = "7d3dda2761e3f25d2f6470259cd18306688e2e9ce7afc5f2d9c9ee92ea5eae20"
ROOTS = (1302, 1346, 1292, 1286, 1282)
REVERSE = (1303, 1347, 1293, 1287, 1283)
PAIR = ((1302, 1283), (1282, 1303))
FREQUENCY = 887_500_000
LIMIT = .01
JSON_NAME, CSV_NAME, HTML_NAME = "lead_phase_AP_pnc.json", "lead_phase_AP_pnc.csv", "report_AP_pnc.html"
SOURCE_ROLES = ("h100_8875_summary", "h100_8875_coefficients", "h80_8875_summary", "h80_8875_coefficients")
CSV_COLUMNS = ("kind", "incident_root", "outgoing_root", "reference_real", "reference_imag", "candidate_real", "candidate_imag",
               "reference_power", "candidate_power", "reference_magnitude", "candidate_magnitude", "magnitude_lower_bound",
               "raw_complex_delta_real", "raw_complex_delta_imag", "absolute_raw_complex_change", "diagnostic_phase_change_rad",
               "selected_directed_pair")
LIMITATIONS = (
    "This is an algebraic residual lower bound for the two saved finite responses, not a numerical floor, error estimate, uncertainty bound or convergence result.",
    "The tested null is diagonal, reflectionless, unmixed, unit-modulus lead propagation. Reflective or mixing lead errors and their interaction with the pattern are not excluded.",
    "The bound cannot identify the patterned region as the cause or locate a discretization error. The global mesh increment changes both lead interiors and pattern.",
    "No phase fitting, de-embedding, uniform-guide subtraction, field projection, new solve, assembly, remesh or eigensolve is performed.",
    "The regional-energy observation concerns one drive's integrated total field; it does not determine traveling-wave phase or weak-conversion accuracy.",
    "Saved 1 GHz uniform controls do not establish uniform propagation error at 0.8875 GHz. Original device planes and all failed spatial criteria remain unchanged.",
)


def require(condition, message):
    if not condition:
        raise ValueError(message)


def sha(path):
    with Path(path).open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def artifact(path, workspace):
    path, workspace = Path(path).resolve(strict=True), Path(workspace).resolve()
    require(path.is_relative_to(workspace), "Artifact escapes the workspace.")
    return {"path": path.relative_to(workspace).as_posix(), "sha256": sha(path), "bytes": path.stat().st_size}


def number(value, label):
    require(isinstance(value, (int, float)) and not isinstance(value, bool) and math.isfinite(value), f"Nonfinite or invalid {label}.")
    return float(value)


def amplitude(value):
    require(isinstance(value, list) and len(value) == 2, "Invalid complex amplitude layout.")
    return complex(number(value[0], "real amplitude"), number(value[1], "imaginary amplitude"))


def close(first, second, label):
    require(math.isclose(first, second, rel_tol=2e-12, abs_tol=2e-15), f"Inconsistent saved {label}.")


def validate_comparison(value):
    require(value.get("schema_version") == 1 and value.get("phase") == "device_inplane_comparison"
            and value.get("status") == "compared_finite_inplane_increment"
            and value.get("frequency_hz") == FREQUENCY and value.get("reference_resolution") == "h100_z4"
            and value.get("candidate_resolution") == "h80_z4", "Accepted comparison scope differs.")
    require(value.get("input_power_checks_passed") is True and value.get("power_passed") is True
            and value.get("complex_passed") is False and value.get("mesh_pair_passed") is False
            and value.get("power_change_limit") == LIMIT and value.get("complex_change_limit") == LIMIT,
            "The original power/failed complex gates differ.")
    for key in ("phase_adjustment_applied", "field_matching_recomputed", "spatial_convergence_established",
                "frequency_convergence_established", "numerical_floor_established"):
        require(value.get(key) is False, "A saved phase or scientific-limit flag differs.")
    require(value.get("other_prior_failed_spatial_comparisons_remain_unresolved") is True,
            "Prior failed spatial comparisons must remain unresolved.")
    basis = value.get("shared_basis", {})
    require(basis.get("status") == "identical_saved_basis" and basis.get("incoming_roots") == list(ROOTS)
            and basis.get("transmitted_roots") == list(ROOTS) and basis.get("reflected_roots") == list(REVERSE)
            and basis.get("candidate_gauge_complex") == [[1., 0.]]*5
            and basis.get("candidate_gauge_phase_rad") == [0.]*5 and basis.get("field_matching_recomputed") is False,
            "Exact signed basis or identity gauges differ.")
    for role, expected in (("safe_modes", MODES_SHA), ("safe_section", SECTION_SHA), ("safe_receipt", SAFE_SHA)):
        require(basis.get(role, {}).get("sha256") == expected, "The exact saved SAFE basis identity differs.")
    for role in SOURCE_ROLES:
        link = value.get("inputs", {}).get(role, {})
        require(isinstance(link.get("path"), str) and isinstance(link.get("bytes"), int) and link["bytes"] > 0
                and isinstance(link.get("sha256"), str) and len(link["sha256"]) == 64
                and all(c in "0123456789abcdef" for c in link["sha256"]), "Missing original summary/CSV source identity.")
    rows = value.get("coefficients")
    require(isinstance(rows, list) and len(rows) == 50, "Exactly 50 coefficient rows are required.")
    expected = {(kind, incoming, outgoing) for kind, outgoing_roots in (("reflection", REVERSE), ("transmission", ROOTS))
                for incoming in ROOTS for outgoing in outgoing_roots}
    seen, entries = set(), []
    for row in rows:
        key = row.get("kind"), row.get("incident_root"), row.get("outgoing_root")
        require(key in expected and key not in seen and all(isinstance(i, int) and not isinstance(i, bool) for i in key[1:]),
                "Duplicate, missing or invalid signed coefficient identity.")
        seen.add(key)
        a, b = amplitude(row["reference_raw_amplitude"]), amplitude(row["candidate_raw_amplitude"])
        require(row["candidate_gauge_aligned_amplitude"] == row["candidate_raw_amplitude"], "Identity-gauge amplitudes differ.")
        pa, pb = number(row["reference_power"], "reference power"), number(row["candidate_power"], "candidate power")
        require(pa >= 0 and pb >= 0 and math.isfinite(abs(a)**2) and math.isfinite(abs(b)**2), "Invalid nonnegative coefficient powers.")
        close(pa, abs(a)**2, "reference amplitude/power")
        close(pb, abs(b)**2, "candidate amplitude/power")
        delta = b-a
        phase = cmath.phase(b/a) if a != 0 and b != 0 else None
        for change_name in ("raw_complex_change", "gauge_aligned_complex_change"):
            saved = row[change_name]
            saved_delta = amplitude(saved["complex_delta"])
            close(saved_delta.real, delta.real, "complex delta real")
            close(saved_delta.imag, delta.imag, "complex delta imaginary")
            close(number(saved["absolute_complex_change"], "absolute complex change"), abs(delta), "absolute complex change")
            if phase is None:
                require(saved["phase_change_rad"] is None, "Zero-amplitude phase must be undefined.")
            else:
                close(number(saved["phase_change_rad"], "phase change"), phase, "diagnostic phase")
        entries.append({"kind": key[0], "incident_root": key[1], "outgoing_root": key[2],
                        "reference_raw_amplitude": [a.real, a.imag], "candidate_raw_amplitude": [b.real, b.imag],
                        "reference_power": pa, "candidate_power": pb, "reference_magnitude": abs(a), "candidate_magnitude": abs(b),
                        "magnitude_lower_bound": abs(abs(b)-abs(a)), "raw_complex_delta": [delta.real, delta.imag],
                        "absolute_raw_complex_change": abs(delta), "diagnostic_phase_change_rad": phase,
                        "selected_directed_pair": key[0] == "reflection" and key[1:] in PAIR})
    require(seen == expected, "The complete 10 by 5 signed block is required.")
    for incoming in ROOTS:
        for role in ("reference_power", "candidate_power"):
            require(abs(math.fsum(row[role] for row in entries if row["incident_root"] == incoming)-1) < LIMIT,
                    "A complete unit-drive coefficient power sum fails its existing 0.01 gate.")
    close(number(value["maximum_complex_change"], "original maximum complex change"),
          max(row["absolute_raw_complex_change"] for row in entries), "original maximum complex change")
    require(value["maximum_complex_change"] > LIMIT, "The original complex-response failure was changed.")
    entries.sort(key=lambda r: (0 if r["kind"] == "reflection" else 1, ROOTS.index(r["incident_root"]),
                               (REVERSE if r["kind"] == "reflection" else ROOTS).index(r["outgoing_root"])))
    return entries


def audit(value):
    entries = validate_comparison(value)
    maximum = max(row["magnitude_lower_bound"] for row in entries)
    frobenius = math.hypot(*(row["magnitude_lower_bound"] for row in entries))
    return {"entries": entries, "aggregate": {
        "maximum_magnitude_lower_bound": maximum, "frobenius_magnitude_lower_bound": frobenius,
        "maximum_bound_entries": [{k: row[k] for k in ("kind", "incident_root", "outgoing_root")}
                                  for row in entries if row["magnitude_lower_bound"] == maximum],
        "existing_complex_change_limit": LIMIT, "bound_strictly_exceeds_existing_limit": maximum > LIMIT,
        "verdict": "phase-only model excluded at existing tolerance" if maximum > LIMIT else "not excluded",
        "frobenius_acceptance_gate_applied": False},
        "selected_pair": [row for row in entries if row["selected_directed_pair"]]}


def csv_text(entries):
    stream = StringIO(newline="")
    writer = csv.DictWriter(stream, fieldnames=CSV_COLUMNS, lineterminator="\n")
    writer.writeheader()
    for entry in entries:
        row = {key: entry[key] for key in CSV_COLUMNS if key in entry}
        for prefix, key in (("reference", "reference_raw_amplitude"), ("candidate", "candidate_raw_amplitude"),
                            ("raw_complex_delta", "raw_complex_delta")):
            row[prefix+"_real"], row[prefix+"_imag"] = entry[key]
        writer.writerow(row)
    return stream.getvalue()


def report_text(record, output, workspace):
    def source_link(link, label):
        target = (workspace/link["path"]).resolve()
        require(target.is_relative_to(workspace), "A report source link escapes the workspace.")
        if target.is_file() and target.stat().st_size == link["bytes"] and sha(target) == link["sha256"]:
            return available_link(output, target, label)
        return f'<span class="availability">{escape(label)} (verified archive source unavailable)</span>'
    agg = record["aggregate"]
    table = "".join("<tr><td>"+str(r["incident_root"])+"→"+str(r["outgoing_root"])+"</td><td>"+format(r["reference_magnitude"], ".12g")
                    +"</td><td>"+format(r["candidate_magnitude"], ".12g")+"</td><td>"+format(r["magnitude_lower_bound"], ".12g")+"</td></tr>"
                    for r in record["selected_pair"])
    sources = " · ".join(source_link(link, name.replace("_8875_", " ").replace("_", " "))
                         for name, link in record["original_response_sources"].items())
    downloads = " · ".join((report_link("Complete audit JSON", JSON_NAME, download=True),
                            report_link("All 50 coefficient rows (CSV)", CSV_NAME, download=True),
                            source_link(record["accepted_comparison"], "Accepted h100/h80 comparison (JSON)")))
    navigation = available_link(output, ROOT / "verification/README.md", "Verification records")
    table = scroll_table(f"""<table><thead><tr><th>Directed reflected pair</th><th>|S h100|</th><th>|S h80|</th>
<th>Magnitude lower bound</th></tr></thead><tbody>{table}</tbody></table>""", "Selected reflected pairs and phase-only bounds")
    scope = ("Finite algebraic exclusion test on the saved 10 × 5 left-incident block at the original x = 0 and 16.65 µm planes. "
             "The two responses retain the exact shared signed SAFE columns and native gauge.")
    body = f'''
<p><strong>{escape(agg['verdict'])}.</strong> For the accepted 887.5 MHz h100/h80 z4 responses, the maximum magnitude-only lower bound is
{agg['maximum_magnitude_lower_bound']:.12g}, versus the unchanged 0.01 complex-response limit. The Frobenius lower bound {agg['frobenius_magnitude_lower_bound']:.12g} is diagnostic.</p>
<p class="sources">{downloads}</p>
<p>For every coefficient, |b − exp(iθ)a| ≥ ||b| − |a||. Even granting an independent phase to every entry cannot lower the maximum residual below this bound.
Every diagonal, reflectionless, unmixed lead-phase model is a subset of that larger phase freedom. No phases are fitted or applied.</p>
{table}
<h2>Original archive sources</h2><p class="sources">{sources}</p>
<h2>Limits</h2><ul>{''.join('<li>'+escape(s)+'</li>' for s in LIMITATIONS)}</ul>'''
    return report_page("Saved lead phase-only audit at 887.5 MHz", "lead_phase_8875", body,
                       navigation=navigation, scope=scope)


def run(workspace, comparison_path, output):
    start = monotonic()
    workspace, output = Path(workspace).resolve(), Path(output).resolve()
    require(output.is_relative_to(workspace) and not output.exists(), "Use a fresh output directory inside the workspace; no overwrites.")
    process, peak = psutil.Process(), 0
    def guard():
        nonlocal peak
        family = [process, *process.children(recursive=True)]
        rss = 0
        for worker in family:
            try:
                rss += worker.memory_info().rss
            except psutil.NoSuchProcess:
                pass
        peak = max(peak, rss)
        require(monotonic()-start < 30 and peak <= 256*2**20 and psutil.virtual_memory().available >= 512*2**20,
                "Audit exceeds its 30 s / 256 MiB process-tree / 512 MiB reserve admission.")
    guard()
    comparison_path = Path(comparison_path).resolve(strict=True)
    require(comparison_path.is_relative_to(workspace) and comparison_path.is_file() and comparison_path.suffix == ".json"
            and comparison_path.stat().st_size <= 4*2**20, "Only a bounded comparison JSON inside the workspace is permitted.")
    require(sha(comparison_path) == COMPARISON_SHA, "The immutable accepted comparison SHA-256 differs.")
    value = json.loads(comparison_path.read_bytes())
    calculated = audit(value)
    guard()
    record = {**AUTHOR, "schema_version": 1, "status": "finite_phase_only_audit_completed",
              "research_question": "Can diagonal, reflectionless, unmixed lead propagation phase alone account for the saved h100/h80 complex-response discrepancy?",
              "scope": {"frequency_hz": FREQUENCY, "reference_resolution": "h100_z4", "candidate_resolution": "h80_z4",
                        "incident_side": "left", "block_shape": [10, 5], "coefficient_count": 50,
                        "reference_planes_m": [0., 1.665e-5], "native_incident_gauge_preserved": True,
                        "new_solve_performed": False, "new_assembly_performed": False, "arrays_read": False,
                        "phase_fitting_performed": False, "phase_adjustment_applied": False,
                        "deembedding_performed": False, "uniform_phase_subtraction_performed": False,
                        "spatial_convergence_established": False, "numerical_floor_established": False},
              "accepted_comparison": artifact(comparison_path, workspace), "shared_basis": value["shared_basis"],
              "original_response_sources": {key: value["inputs"][key] for key in SOURCE_ROLES},
              "preserved_original_acceptance": {key: value[key] for key in ("maximum_power_change", "maximum_complex_change", "power_change_limit",
                  "complex_change_limit", "power_passed", "complex_passed", "mesh_pair_passed", "pair_specific_targets",
                  "other_prior_failed_spatial_comparisons_remain_unresolved")},
              "method": {"inequality": "abs(b-exp(i*theta)*a) >= abs(abs(b)-abs(a))",
                         "phase_freedom": "Independent unit-modulus phase per coefficient; a superset of diagonal input/output phases.",
                         "tested_model": "Diagonal, reflectionless, unmixed, unit-modulus lead propagation.",
                         "not_excluded_meaning": "The bound does not exclude the model; it never proves that model."},
              **calculated, "limitations": list(LIMITATIONS), "producer": artifact(__file__, workspace),
              "numerical_environment": {"python": platform.python_version(), "platform": platform.platform(), "psutil": psutil.__version__}}
    csv_payload = csv_text(record["entries"])
    html_payload = report_text(record, output, workspace)
    record["publication"] = {"json": JSON_NAME, "csv": CSV_NAME, "html": HTML_NAME,
                             "csv_sha256": hashlib.sha256(csv_payload.encode()).hexdigest()}
    guard()
    record["resources"] = {"wall_limit_s": 30, "rss_limit_bytes": 256*2**20, "available_memory_reserve_bytes": 512*2**20,
                           "output_limit_bytes": 2*2**20, "elapsed_s": monotonic()-start, "sampled_peak_process_tree_rss_bytes": peak,
                           "sampling_note": "Admission/checkpoint samples are not an exact high-water mark; the root execution receipt supplies external monitoring."}
    json_payload = json.dumps(record, indent=2, allow_nan=False)+"\n"
    payloads = {JSON_NAME: json_payload, CSV_NAME: csv_payload, HTML_NAME: html_payload}
    require(sum(len(payload.encode()) for payload in payloads.values()) <= 2*2**20, "Audit outputs exceed the admitted 2 MiB ceiling.")
    output.mkdir(parents=True, exist_ok=False)
    for name, payload in payloads.items():
        with (output/name).open("x", encoding="utf-8", newline="") as stream:
            stream.write(payload)
    guard()
    return record


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--workspace", type=Path, default=ROOT)
    parser.add_argument("--comparison", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args(argv)
    result = run(args.workspace, args.comparison or args.workspace/COMPARISON_PATH, args.output)
    print(json.dumps({"status": result["status"], **result["aggregate"]}, allow_nan=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
