"""Read the pinned thin-device grid and native coefficients. Author: Athul Prem | _AP_pnc."""

from __future__ import annotations

import cmath
from copy import deepcopy
import csv
from hashlib import sha256
import io
import json
import math
from pathlib import Path


COMPARISON = {"path": "verification/thin_mesh_frequency_AP_pnc.json",
              "sha256": "8ece9fce1492bf9faa46f4b54484450e5fcb85c63e374a5bf7d98cb93d092ec0",
              "bytes": 203315}
FREQUENCIES = (995_000_000, 997_500_000, 1_000_000_000)
TARGETS = (225.0, 168.75, 126.5625)
# Native byte digests of configuration_AP_pnc.json from the accepted point
# summaries pinned by COMPARISON, in frequency order.
CONFIGURATION_SHA256 = {
    225.0: (
        "2677faf9f424cb36a846de93248678a5af6ce70c06f12da288e2b7ca9e202bc4",
        "979a6b44282aed7deb2c6bc4f477fdec33107a95b8f8c1d8b18cd569a5607171",
        "2fee272712a2f3eaf920ab7627584e87fbed0fa3a46c9d086568673a730c95d3",
    ),
    168.75: (
        "a3d27d606bb758ba53a8caec884d7fd42efc1dc702e3629da5ec078e49b99ac4",
        "9317b00310d3f3f63d16f26eaf38c46a25452aa86a80e86581a6e46a435cde62",
        "b94c242ed15f646149e70086b596bc6cf915facbaa46283609c5fa7df96e216e",
    ),
    126.5625: (
        "cc387f514fb76e3eeb8437270cf6a3621d24f792f4b7b09d312ee407c42e4159",
        "cf8eeac1b72765d3e9ef94d9e3203b06a50c9181421f58f5f901072429b0e7ac",
        "b78bbdda519a79983eb69f9611556420763b9f3ce72726212fcdabd95badb337",
    ),
}
PLANES = [0.0, 1.665e-5]
DRIVES = list(range(1, 6))
COHERENT_DRIVES = (3, 5)
KINDS = ("reflection", "transmission")
CSV_FILE = "scattering_map_AP_pnc.csv"
CSV_HEADER = (
    "frequency_GHz", "segment", "kind", "incident_port_index", "outgoing_port_index", "incident_channel",
    "outgoing_channel", "amplitude_real", "amplitude_imag", "power_fraction", "segment_amplitude_real",
    "segment_amplitude_imag",
)
LIMIT_BYTES = 4 * 2**20
LIMIT = 0.01
ROUNDOFF = 1e-10


def _require(condition, message):
    if not condition:
        raise ValueError(message)


def _source(root, pin, sources):
    relative = Path(pin["path"])
    path = (root / relative).resolve()
    _require(not relative.is_absolute() and ".." not in relative.parts and path.is_relative_to(root)
             and path.suffix in (".json", ".csv"), "A compact source path escapes the archive or has an invalid type.")
    with path.open("rb") as stream:
        raw = stream.read(LIMIT_BYTES + 1)
    digest = sha256(raw).hexdigest()
    _require(len(raw) <= LIMIT_BYTES and len(raw) == pin.get("bytes", len(raw)) and digest == pin["sha256"],
             f"Saved compact source identity changed: {path}")
    sources.append({"path": str(path), "sha256": digest, "bytes": len(raw)})
    return path, raw.decode("utf-8-sig")


def _digest(value):
    return sha256(json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False).encode("utf-8")).hexdigest()


def _finite(value):
    result = float(value)
    _require(math.isfinite(result), "A saved response contains a non-finite value.")
    return result


def _matches(first, second):
    return math.isclose(_finite(first), _finite(second), rel_tol=0, abs_tol=ROUNDOFF)


def _unresolved(row):
    return row["spatial_convergence_established"] is False and row["frequency_convergence_established"] is False


def _group_roots(roots):
    return len(roots) == 5 and all(type(root) is int and root >= 0 for root in roots) and len(set(roots)) == 5


def _channel_inventory(comparison):
    positions = {(m, f) for m in TARGETS for f in FREQUENCIES}
    grid_edges = {((m, a), (m, b)) for m in TARGETS for a, b in zip(FREQUENCIES, FREQUENCIES[1:])} | {
        ((a, f), (b, f)) for a, b in zip(TARGETS, TARGETS[1:]) for f in FREQUENCIES
    }
    correspondence = comparison["channel_correspondence"]
    _require(correspondence["commuting_adjacent_grid_squares"] == 4
             and correspondence["reference_central_max_edge_nm"] == TARGETS[0]
             and correspondence["reference_frequency_hz"] == FREQUENCIES[-1],
             "The accepted reference channel or signed grid topology changed.")
    inventory, incoming_edges = {}, {}
    for edge in correspondence["signed_matches"]:
        a, b = tuple(edge["reference_point"]), tuple(edge["candidate_point"])
        _require((a, b) in grid_edges and (a, b) not in incoming_edges, "A signed match has an invalid or duplicate edge.")
        for group in ("incoming", *KINDS):
            match = edge["groups"][group]
            candidate, reference, permutation = (match[key] for key in ("candidate_roots", "reference_roots", "candidate_order"))
            overlaps = match["mass_overlaps"]
            _require(match["status"] == "matched" and _group_roots(reference) and _group_roots(candidate)
                     and all(type(slot) is int for slot in permutation) and sorted(permutation) == list(range(5))
                     and len(overlaps) == 5
                     and all(comparison["minimum_required_signed_mass_overlap"] <= _finite(x) <= 1 for x in overlaps),
                     "The signed match does not identify five supported modal roots.")
            native_candidate = [root for _, root in sorted(zip(permutation, candidate, strict=True))]
            for point, roots in ((a, reference), (b, native_candidate)):
                groups = inventory.setdefault(point, {})
                _require(groups.setdefault(group, roots) == roots, "Signed edges disagree on a point's native channel order.")
            if group == "incoming":
                phases = match["candidate_gauge_phase_rad"]
                _require(len(phases) == len(reference) and all(math.isfinite(float(phase)) for phase in phases),
                         "A signed incident match has an invalid gauge.")
                incoming_edges[a, b] = match
    _require(inventory.keys() == positions and incoming_edges.keys() == grid_edges
             and all(set(g["incoming"]) == set(g["transmission"]) and not set(g["incoming"]) & set(g["reflection"])
                     for g in inventory.values()), "The fixed signed modal grid is incomplete.")
    return inventory, incoming_edges


def _read_map(root, pin, sources):
    _require(Path(pin["path"]).name == "map_AP_pnc.json", "Expected a native frequency-map publication.")
    path, text = _source(root, pin, sources)
    data = json.loads(text)
    _require(data["schema_version"] == 1 and data["status"] == "complete" and _unresolved(data)
             and data["map_data_sha256"] == _digest({key: val for key, val in data.items() if key != "map_data_sha256"})
             and data["inputs_sha256"] == _digest(data["inputs"])
             and data["shared"]["reference_planes_x_m"] == PLANES, "Native map publication identity or scope changed.")
    csv_pin = {"path": Path(pin["path"]).with_name(CSV_FILE).as_posix(), "sha256": data["output_sha256"][CSV_FILE]}
    csv_path, text = _source(root, csv_pin, sources)
    settings = data["inputs"]["frequencies_GHz"]
    points = data["points"]
    _require(settings and [(row["index"], row["frequency_GHz"]) for row in points] == list(enumerate(settings, 1))
             and all(row["status"] == "accepted" and _unresolved(row) for row in points),
             "The map has an incomplete or unaccepted native frequency plan.")
    by_frequency = {}
    for point in points:
        frequency = point["frequency_GHz"] * 1e9
        _require(frequency in FREQUENCIES and frequency not in by_frequency, "A map has a duplicate or unsupported frequency.")
        by_frequency[int(frequency)] = point
    table = csv.DictReader(io.StringIO(text, newline=""))
    _require(table.fieldnames == list(CSV_HEADER), "The native coefficient table has an unexpected header.")
    rows = {f: [] for f in by_frequency}
    for row in table:
        frequency = _finite(row["frequency_GHz"]) * 1e9
        _require(None not in row and None not in row.values() and frequency in rows,
                 "A coefficient has an unexpected column count or frequency.")
        rows[frequency].append(row)
    return data, by_frequency, rows, str(path), str(csv_path)


def _drives(level, f, frequency, native, inventory):
    accepted = native["incoming_channels"]
    matched = level["matched_drives"]
    _require(len(accepted) == len(matched) == 5 and [row["reference_drive"] for row in matched] == DRIVES
             and [row["port_index"] for row in accepted] == inventory["incoming"]
             and sorted(row["channel"] for row in accepted) == DRIVES, "The native or reference drive inventory changed.")
    drives = []
    for drive in matched:
        sample = drive["samples"][f]
        native_drive = sample["native_drive"]
        _require(len(drive["samples"]) == len(FREQUENCIES) and sample["frequency_hz"] == frequency
                 and type(native_drive) is int and native_drive in DRIVES, "A reference drive has an invalid native slot.")
        row = accepted[native_drive - 1]
        t, r = (_finite(row[k]) for k in ("transmission", "reflection"))
        _require(row["port_index"] == sample["native_root_id"] and _matches(row["k_rad_m"], sample["wavenumber_rad_m"])
                 and t >= 0 and r >= 0 and abs(t + r - 1) < LIMIT
                 and _matches(t, sample["transmission_fraction"]) and _matches(r, sample["reflection_fraction"]),
                 "A matched reference drive differs from the native root or accepted power totals.")
        drives.append({"reference_drive": drive["reference_drive"], "native_drive": native_drive,
                       "incident_port_index": row["port_index"], "wavenumber_rad_m": _finite(row["k_rad_m"]),
                       "transmission": t, "reflection": r})
    _require(sorted(row["native_drive"] for row in drives) == DRIVES, "The matched native drive order is not one-to-one.")
    return drives


def _coefficients(native, rows, drives, inventory, frequency):
    _require(len(rows) == 50, "A saved point must contain both complete native five-by-five coefficient blocks.")
    by_root = {row["incident_port_index"]: row for row in drives}
    channels = {row["port_index"]: row["channel"] for row in native["incoming_channels"]}
    totals = {(q, root): [] for q in KINDS for root in by_root}
    roots_by_label = {q: {} for q in KINDS}
    result, seen = [], set()
    for row in rows:
        q = row["kind"]
        a, b = int(row["incident_port_index"]), int(row["outgoing_port_index"])
        identity = q, a, b
        _require(q in KINDS and a in by_root and b in inventory[q] and identity not in seen
                 and channels[a] == int(row["incident_channel"]) and int(row["segment"]) == native["segment"],
                 "A coefficient row has a duplicated or invalid native frequency-local identity.")
        seen.add(identity)
        label = int(row["outgoing_channel"])
        _require(label in DRIVES and roots_by_label[q].setdefault(label, b) == b,
                 "A coefficient has an invalid or inconsistent outgoing tracking label.")
        real, imag, power = (_finite(row[k]) for k in ("amplitude_real", "amplitude_imag", "power_fraction"))
        _require(power >= 0 and math.isclose(power, real * real + imag * imag, rel_tol=1e-8, abs_tol=1e-15),
                 "A raw native amplitude differs from its recorded power.")
        _finite(row["segment_amplitude_real"])
        _finite(row["segment_amplitude_imag"])
        totals[q, a].append(power)
        drive = by_root[a]
        result.append({"frequency_hz": frequency, "reference_drive": drive["reference_drive"],
                       "native_drive": drive["native_drive"], "kind": q, "incident_port_index": a,
                       "outgoing_port_index": b, "amplitude_real": real, "amplitude_imag": imag, "power_fraction": power})
    _require(all(set(labels.values()) == set(inventory[q]) for q, labels in roots_by_label.items())
             and all(len(powers) == 5 and _matches(math.fsum(powers), by_root[root][q])
                     for (q, root), powers in totals.items()),
             "Native outgoing roots or row sums differ from the accepted per-drive totals.")
    return result


def _read_point(level, comparison, sample_index, inventory, data):
    f = FREQUENCIES[sample_index]
    sample = level["samples"][sample_index]
    record, by_frequency, rows, source_map, source_coefficients = data
    native = by_frequency[f]
    archive = sample["archive"]
    cfg = deepcopy(record["inputs"]["configuration"])
    cfg["analysis"]["frequency_GHz"] = native["frequency_GHz"]
    mesh = level["central_max_edge_nm"]
    cfg_hash = CONFIGURATION_SHA256[mesh][sample_index]
    cfg_json = json.dumps(cfg, indent=2, ensure_ascii=False, allow_nan=False) + "\n"
    # The native writer used the solve host's newline convention.
    if sha256(cfg_json.encode("utf-8")).hexdigest() != cfg_hash:
        cfg_json = cfg_json.replace("\n", "\r\n")
        _require(sha256(cfg_json.encode("utf-8")).hexdigest() == cfg_hash,
                 f"The published inputs differ from the accepted configuration at {mesh:g} nm, {f} Hz.")
    hole_pattern = record["inputs"]["hole_pattern"]
    source = Path(hole_pattern["source_csv"])
    source_hash = hole_pattern["source_sha256"]
    _require(source.as_posix() == hole_pattern["source_csv"] == cfg["geometry"]["pattern"]["csv"]
             and not source.is_absolute() and not source.drive and ".." not in source.parts
             and source.suffix == ".csv"
             and len(source_hash) == 64 and all(c in "0123456789abcdef" for c in source_hash)
             and hole_pattern["x_offset_m"] == cfg["geometry"]["pattern"]["x_offset_nm"] * 1e-9,
             "The published hole CSV identity differs from the point configuration.")
    directory = Path(archive["map"]["path"]).parent
    summary = native["accepted"]["summary"]
    _require(sample["frequency_hz"] == f and record["checkpoint_sha256"] == archive["checkpoint"]["sha256"]
             and (directory / "checkpoint_AP_pnc.json").as_posix() == archive["checkpoint"]["path"]
             and {**summary, "path": (directory / summary["path"]).as_posix()} == archive["point_summary"]
             and record["shared"]["mesh_hash"] == level["mesh_hash"]
             and cfg["geometry"]["thickness_nm"] == 100 and cfg["geometry"]["length_nm"] * 1e-9 == PLANES[1]
             and cfg["analysis"]["incident_side"] == "left" and cfg["analysis"]["boundary"] == "matched_ports"
             and len(hole_pattern["holes"]) == 26
             and cfg["mesh"] == {**comparison["mesh_parameters_common_to_levels"],
                                 "central_max_edge_nm": mesh},
             "The native point does not bind the archived geometry, mesh, checkpoint and summary.")
    drives = _drives(level, sample_index, f, native, inventory)
    mean = level["equal_incoherent_mean"]["samples"][sample_index]
    _require(mean["frequency_hz"] == f and all(_matches(math.fsum(d[q] for d in drives) / 5, mean[f"{q}_fraction"])
                                               for q in KINDS), "The published mean differs from the five native powers.")
    return {"frequency_hz": f, "source_map": source_map, "source_coefficients": source_coefficients,
            "configuration": cfg, "configuration_json": cfg_json, "configuration_sha256": cfg_hash,
            "hole_csv": {"path": source.as_posix(), "sha256": source_hash},
            "drives": drives, "coefficients": _coefficients(native, rows[f], drives, inventory, f),
            "neighboring_spatial_gates": []}


def _spatial_gates(comparison, points):
    expected = list(zip(TARGETS, TARGETS[1:]))
    increments = comparison["spatial_increments"]
    key_names = ("coarser_central_max_edge_nm", "finer_central_max_edge_nm")
    _require([tuple(edge[key] for key in key_names) for edge in increments] == expected,
             "The saved spatial comparisons do not cover the two adjacent mesh steps.")
    for edge, (a, b) in zip(increments, expected, strict=True):
        rows = edge["per_drive"]
        _require([row["reference_drive"] for row in rows] == DRIVES
                 and all(len(row["sample_changes_percentage_points"]) == len(FREQUENCIES) for row in rows),
                 "The saved finite spatial comparison is incomplete.")
        for i, f in enumerate(FREQUENCIES):
            before, after = points[a, f], points[b, f]
            step = []
            for row, x, y in zip(rows, before["drives"], after["drives"], strict=True):
                change = row["sample_changes_percentage_points"][i]
                power = {q: _finite(change[q]) for q in KINDS}
                amplitude = {q: _finite(change["maximum_absolute_complex_amplitude_change_by_block"][q]) for q in KINDS}
                _require(change["frequency_hz"] == f and all(_matches(power[q], 100 * (y[q] - x[q])) for q in KINDS)
                         and all(value >= 0 for value in amplitude.values())
                         and change["transmission_gate_passed"] is (abs(power["transmission"]) / 100 < LIMIT)
                         and change["total_power_gate_passed"] is (max(map(abs, power.values())) / 100 < LIMIT)
                         and change["complex_amplitude_gate_passed"] is (max(amplitude.values()) < LIMIT),
                         "A saved per-drive spatial gate differs from the published powers or limits.")
                step.append({"reference_drive": row["reference_drive"], **change})
            gates = {"coarser_central_max_edge_nm": a, "finer_central_max_edge_nm": b, "drives": step}
            for point in (before, after):
                point["neighboring_spatial_gates"].append(gates)


def _two_drive_grams(point, inventory, incident_roots, factors):
    coefficients = {(row["kind"], row["incident_port_index"], row["outgoing_port_index"]):
                    complex(row["amplitude_real"], row["amplitude_imag"]) for row in point["coefficients"]}
    grams = {}
    for kind in KINDS:
        first, second = ([factor * coefficients[kind, root, out] for out in inventory[kind]]
                         for root, factor in zip(incident_roots, factors, strict=True))
        products = [a.conjugate() * b for a, b in zip(first, second, strict=True)]
        cross = complex(math.fsum(value.real for value in products), math.fsum(value.imag for value in products))
        grams[kind] = math.fsum(abs(a)**2 for a in first), math.fsum(abs(b)**2 for b in second), cross
    return grams


def _hermitian_bounds(first, second, cross):
    center = (first + second) / 2
    radius = math.hypot((first - second) / 2, abs(cross))
    return center - radius, center + radius


def _coherent_endpoint_change(lower, upper, lower_inventory, upper_inventory, match):
    lower_roots = [lower["drives"][drive - 1]["incident_port_index"] for drive in COHERENT_DRIVES]
    indices = [match["reference_roots"].index(root) for root in lower_roots]
    upper_roots = [match["candidate_roots"][i] for i in indices]
    phases = [float(match["candidate_gauge_phase_rad"][i]) for i in indices]
    overlaps = [float(match["mass_overlaps"][i]) for i in indices]
    # The matched upper incident modes are rephased by exp(-i * phase).
    # Unitary outgoing row permutations and phases cancel in each Gram matrix.
    before = _two_drive_grams(lower, lower_inventory, lower_roots, (1, 1))
    after = _two_drive_grams(upper, upper_inventory, upper_roots, [cmath.rect(1, -phase) for phase in phases])
    deltas = {kind: tuple(b - a for a, b in zip(before[kind], after[kind], strict=True)) for kind in KINDS}
    closure = _hermitian_bounds(*(a + b for a, b in zip(*deltas.values(), strict=True)))
    result = {"lower_frequency_hz": lower["frequency_hz"], "upper_frequency_hz": upper["frequency_hz"],
              "reference_drives": list(COHERENT_DRIVES), "lower_incident_native_roots": lower_roots,
              "upper_incident_native_roots": upper_roots, "candidate_incident_gauge_phase_rad": phases,
              "incident_mass_overlaps": overlaps, "outgoing_rows_per_block": len(DRIVES),
              "closure_discrepancy_percentage_points": 100 * max(map(abs, closure))}
    for kind, (first, second, cross) in deltas.items():
        matrix = ((first, cross), (cross.conjugate(), second))
        result[f"delta_{kind}_gram_fraction"] = [[{"real": v.real, "imag": v.imag} for v in row] for row in matrix]
        result[f"{kind}_eigenvalue_bounds_percentage_points"] = [100 * v for v in _hermitian_bounds(first, second, cross)]
    return result


def _load(root, sources):
    comparison_path, content = _source(root, COMPARISON, sources)
    comparison = json.loads(content)
    metadata = {"physical_thickness_nm": 100, "thickness_layers": 2, "circle_resolution": 32, "incident_side": "left",
                "reference_planes_x_m": PLANES, "frequencies_hz": list(FREQUENCIES), "central_max_edges_nm": list(TARGETS),
                "spatial_transmission_limit_fraction": LIMIT, "spatial_total_power_limit_fraction": LIMIT,
                "spatial_complex_amplitude_limit": LIMIT}
    _require(comparison["schema_version"] == 1 and comparison["status"] == "compared_saved_thin_mesh_frequency"
             and comparison["published_1ghz_spatial_gates_reproduced"] is True
             and comparison["minimum_required_signed_mass_overlap"] == 0.99
             and _unresolved(comparison) and all(comparison[k] == v for k, v in metadata.items()),
             "The thin-device grid has different samples, planes, limits or convergence scope.")
    inventory, incoming_edges = _channel_inventory(comparison)
    levels = comparison["mesh_levels"]
    _require([level["central_max_edge_nm"] for level in levels] == list(TARGETS)
             and all(len(level["samples"]) == len(FREQUENCIES) for level in levels),
             "The grid does not contain the three fixed mesh levels and frequency samples.")
    cache, requested, output, points = {}, {}, [], {}
    for level in levels:
        mesh = level["central_max_edge_nm"]
        dofs = level["body_dofs"]
        _require(type(dofs) is int and dofs > 0, "Invalid native body DOF count.")
        for index, frequency in enumerate(FREQUENCIES):
            sample = level["samples"][index]
            pin = sample["archive"]["map"]
            name = pin["path"]
            if name not in cache:
                cache[name] = (pin, _read_map(root, pin, sources))
            original_pin, data = cache[name]
            _require(pin == original_pin, "The grid assigns two identities to one native map.")
            requested.setdefault(name, []).append(frequency)
            points[mesh, frequency] = _read_point(level, comparison, index, inventory[mesh, frequency], data)
        output.append({"central_max_edge_nm": mesh, "body_dofs": dofs,
                       "points": [points[mesh, frequency] for frequency in FREQUENCIES]})
    _require(len(cache) == 6 and all(frequencies == list(cache[name][1][1]) for name, frequencies in requested.items()),
             "The grid and native publications do not cover exactly the same nine point records.")
    reference = points[TARGETS[0], FREQUENCIES[-1]]["drives"]
    _require([row["incident_port_index"] for row in reference]
             == inventory[TARGETS[0], FREQUENCIES[-1]]["incoming"]
             == comparison["channel_correspondence"]["reference_native_incident_roots"],
             "Reference drive labels differ from the native 225 nm, 1 GHz order.")
    for (a, b), match in incoming_edges.items():
        roots = dict(zip(match["reference_roots"], match["candidate_roots"], strict=True))
        _require(all(roots[x["incident_port_index"]] == y["incident_port_index"]
                     for x, y in zip(points[a]["drives"], points[b]["drives"], strict=True)),
                 "Reference drive labels do not follow the published signed channel matches.")
    _spatial_gates(comparison, points)
    pins = {item["path"]: {**item, "path": Path(item["path"]).relative_to(root).as_posix()} for item in sources}
    coherent_changes = []
    for mesh in TARGETS:
        a, b = (mesh, FREQUENCIES[-2]), (mesh, FREQUENCIES[-1])
        lower, upper = points[a], points[b]
        change = _coherent_endpoint_change(lower, upper, inventory[a], inventory[b], incoming_edges[a, b])
        provenance = {"comparison": pins[str(comparison_path)],
                      "lower_source_coefficients": pins[lower["source_coefficients"]],
                      "upper_source_coefficients": pins[upper["source_coefficients"]]}
        coherent_changes.append({"central_max_edge_nm": mesh, **change, "provenance": provenance})
    return {**metadata, "source_comparison": str(comparison_path), "sources": sources, "levels": output,
            "coherent_endpoint_changes": coherent_changes,
            "scope": comparison["scope"], "computation_enabled": False,
            "spatial_convergence_established": False, "frequency_convergence_established": False}


def load_thin_grid(root: Path) -> dict | None:
    """Read only 13 compact pinned files. Return None when the optional evidence is absent."""
    root = Path(root).resolve()
    path = root / COMPARISON["path"]
    try:
        path.lstat()
    except FileNotFoundError:
        return None
    try:
        return _load(root, [])
    except (ArithmeticError, OSError, KeyError, IndexError, TypeError, ValueError) as exc:
        raise ValueError(f"Invalid saved thin mesh-frequency evidence. {exc}") from exc
