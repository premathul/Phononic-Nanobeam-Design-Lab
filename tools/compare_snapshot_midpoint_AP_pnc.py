"""Compare the saved midpoint predictions using compact data. Author: Athul Prem | _AP_pnc."""

from __future__ import annotations

import argparse
import cmath
from dataclasses import dataclass
from hashlib import sha256
import json
from math import fsum, isclose, isfinite
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
if __name__ == "__main__":
    sys.path.insert(0, str(ROOT))

from tools import read_thin_device_grid_AP_pnc as grid


RECEIPTS = Path("verification/snapshot_midpoint_AP_pnc")
FILES = ("edge225_AP_pnc.json", "edge168p75_AP_pnc.json", "edge126p5625_AP_pnc.json")
MIDPOINT = grid.FREQUENCIES[1]
GROUPS = ("incoming", *grid.KINDS)
EQUATION_LIMIT = 1e-9
ROUND_OFF = grid.ROUNDOFF


@dataclass(frozen=True)
class MidpointResponse:
    direct: dict
    candidate: dict
    candidate_power: dict
    direct_power: dict
    column_by_drive: dict


def _read_receipt(path):
    with path.open("rb") as stream:
        raw = stream.read(grid.LIMIT_BYTES + 1)
    grid._require(len(raw) <= grid.LIMIT_BYTES, f"Snapshot receipt is too large: {path}.")
    return json.loads(raw), {"sha256": sha256(raw).hexdigest(), "bytes": len(raw), "filename": path.name}


def _check_inputs(receipt, direct, published, sources):
    level = published["central_max_edge_nm"]
    grid._require(
        receipt["author"] == "Athul Prem"
        and receipt["file_tag"] == "_AP_pnc"
        and receipt["central_max_edge_nm"] == level
        and receipt["body_dofs"] == published["body_dofs"]
        and receipt["training_frequencies_hz"] == list(grid.FREQUENCIES[::2])
        and receipt["prediction_frequency_hz"] == MIDPOINT
        and receipt["reference_planes_x_m"] == grid.PLANES
        and receipt["output_comparison_limit"] == grid.LIMIT
        and receipt["relative_equation_residual_limit"] == EQUATION_LIMIT,
        f"The snapshot metadata differs from the checked {level:g} nm grid.")
    inputs = receipt["inputs"]
    shared = inputs["level"]
    grid._require(inputs["grid_comparison"] == shared["comparison"] == grid.COMPARISON
                  and shared["mesh_hash"] == published["mesh_hash"]
                  and shared["reference_planes_x_m"] == grid.PLANES
                  and shared["same_mesh_shared_inputs_by_map"] == published["shared_inputs_by_map"],
                  f"The snapshot preparation differs from the checked {level:g} nm grid.")
    endpoints = inputs["endpoints"]
    grid._require(len(endpoints) == 2, "The snapshot receipt must identify two endpoints.")
    samples = endpoints[0], inputs["midpoint"], endpoints[1]
    for actual, expected, point in zip(samples, published["samples"], direct, strict=True):
        columns = [d["reference_drive"] for d in sorted(point["drives"], key=lambda row: row["native_drive"])]
        grid._require(all(actual[key] == value for key, value in expected.items())
                      and actual["native_coefficients"] == sources[point["source_coefficients"]]
                      and actual["reference_drives_in_native_order"] == columns,
                      f"The snapshot inputs differ from the accepted {level:g} nm point at {point['frequency_hz']} Hz.")


def _blocks(receipt, point, roots):
    saved = receipt["native_coefficients"]
    grid._require(saved["incident_roots_by_column"] == roots["incoming"]
                  and all(saved[f"{q}_roots_by_row"] == roots[q] for q in grid.KINDS),
                  "The midpoint coefficient root order differs from the checked signed inventory.")
    rows = {(row["kind"], row["outgoing_port_index"], row["incident_port_index"]): row
            for row in point["coefficients"]}
    direct, candidate, weights = {}, {}, {}
    for q in grid.KINDS:
        pairs = saved[q]
        grid._require(len(pairs) == 5 and all(len(row) == 5 for row in pairs)
                      and all(len(entry) == 2 for row in pairs for entry in row),
                      f"The candidate {q} block must have five rows and five [real, imaginary] columns.")
        candidate[q] = [[complex(grid._finite(entry[0]), grid._finite(entry[1])) for entry in row] for row in pairs]
        source = [[rows[q, out_root, in_root] for in_root in roots["incoming"]] for out_root in roots[q]]
        direct[q] = [[complex(entry["amplitude_real"], entry["amplitude_imag"]) for entry in row] for row in source]
        # Recover the native flux-to-amplitude ratio from the nonzero CSV entries.
        # An exactly zero entry gives no ratio and uses the analytic unit power normalization.
        weights[q] = [[row["power_fraction"] / abs(amplitude)**2 if amplitude else 1.0
                       for row, amplitude in zip(source_row, direct_row, strict=True)]
                      for source_row, direct_row in zip(source, direct[q], strict=True)]
        grid._require(all(isclose(value, 1.0, rel_tol=0, abs_tol=2e-8) for row in weights[q] for value in row),
                      f"The checked native {q} amplitudes do not have unit power normalization.")
    return direct, candidate, weights


def _response(receipt, direct_points, roots):
    point = direct_points[1]
    direct, candidate, weights = _blocks(receipt, point, roots)
    by_root = {row["incident_port_index"]: row for row in point["drives"]}
    rows = receipt["drives"]
    grid._require([row["native_incident_root"] for row in rows] == roots["incoming"],
                  "The snapshot drive rows do not follow the checked native incident order.")
    power, direct_power = {q: {} for q in grid.KINDS}, {q: {} for q in grid.KINDS}
    column_by_drive, differences = {}, {}
    for col, (row, root) in enumerate(zip(rows, roots["incoming"], strict=True)):
        source = by_root[root]
        drive = source["reference_drive"]
        grid._require(row["reference_drive"] == drive, "A candidate drive has the wrong signed grid identity.")
        column_by_drive[drive] = col
        complex_errors, power_errors = {}, {}
        for q in grid.KINDS:
            predicted = fsum(abs(a[col])**2 * w[col] for a, w in zip(candidate[q], weights[q], strict=True))
            error = 100 * (predicted - source[q])
            complex_error = max(abs(a[col] - b[col]) for a, b in zip(candidate[q], direct[q], strict=True))
            checks = {"candidate_power_fraction": predicted, "direct_power_fraction": source[q],
                      "power_error_percentage_points": error, "maximum_absolute_complex_error": complex_error}
            grid._require(all(isclose(row[q][key], value, rel_tol=0, abs_tol=100 * ROUND_OFF if
                                     key == "power_error_percentage_points" else ROUND_OFF)
                              for key, value in checks.items()),
                          f"Reported {q} metrics disagree with native coefficients for drive {drive}.")
            power[q][drive], direct_power[q][drive] = predicted, source[q]
            complex_errors[q], power_errors[q] = complex_error, error
        residual = row["relative_equation_residual"]
        best = row["minimum_relative_equation_residual_in_trial_span"]
        budget = 1 - fsum(power[q][drive] for q in grid.KINDS)
        grid._require(isfinite(residual) and residual >= 0 and 0 <= best <= residual + ROUND_OFF
                      and isclose(row["power_budget_defect"], budget, rel_tol=0, abs_tol=ROUND_OFF)
                      and row["total_power_comparison_passed"] is (max(map(abs, power_errors.values())) < 100 * grid.LIMIT)
                      and row["complex_comparison_passed"] is (max(complex_errors.values()) < grid.LIMIT)
                      and row["relative_equation_check_passed"] is (residual <= EQUATION_LIMIT),
                      f"Reported power budget, residual or gate differs for drive {drive}.")
        differences[drive] = {"reference_drive": drive, "native_incident_root": root,
                              "snapshot_error_percentage_points": power_errors["transmission"]}
    for lower, mid, upper in zip(*(item["drives"] for item in direct_points), strict=True):
        drive = mid["reference_drive"]
        chord = (lower["transmission"] + upper["transmission"]) / 2
        differences[drive]["linear_endpoint_error_percentage_points"] = 100 * (chord - mid["transmission"])
    replay = receipt["held_out_direct_solution_replay"]
    grid._require(0 <= replay["maximum_relative_equation_residual"] <= EQUATION_LIMIT
                  and 0 <= replay["maximum_absolute_native_coefficient_difference"] < 1e-9,
                  "The archived direct solution replay fails its recorded equation or coefficient limit.")
    return MidpointResponse(direct, candidate, power, direct_power, column_by_drive), [
        differences[drive] for drive in grid.DRIVES]


def _spatial_amplitudes(before, after, match, col):
    # Same reference-gauge rotation and candidate permutation as the nine-point spatial replay.
    incoming = match["incoming"]["candidate_order"][col]
    incoming_phase = cmath.rect(1.0, match["incoming"]["candidate_gauge_phase_rad"][col])
    maxima = {}
    for q in grid.KINDS:
        order = match[q]["candidate_order"]
        phases = match[q]["candidate_gauge_phase_rad"]
        maxima[q] = max(abs(cmath.rect(1.0, phase) * after[q][row][incoming] / incoming_phase - before[q][i][col])
                        for i, (row, phase) in enumerate(zip(order, phases, strict=True)))
    return maxima


def _spatial_stats(names, before, after, match, drive):
    blocks, powers = names
    col = before.column_by_drive[drive]
    grid._require(match["incoming"]["candidate_order"][col] == after.column_by_drive[drive],
                  "The spatial matching edge does not preserve this reference drive.")
    increments = {q: getattr(after, powers)[q][drive] - getattr(before, powers)[q][drive] for q in grid.KINDS}
    maxima = _spatial_amplitudes(getattr(before, blocks), getattr(after, blocks), match, col)
    return {"power_changes_percentage_points": {q: 100 * value for q, value in increments.items()},
            "maximum_absolute_complex_change_by_block": maxima,
            "transmission_gate_passed": abs(increments["transmission"]) < grid.LIMIT,
            "total_power_gate_passed": max(map(abs, increments.values())) < grid.LIMIT,
            "complex_amplitude_gate_passed": max(maxima.values()) < grid.LIMIT}


def _spatial_edge(before, after, match, pinned):
    results = []
    for published in pinned["drives"]:
        drive = published["reference_drive"]
        direct = _spatial_stats(("direct", "direct_power"), before, after, match, drive)
        predicted = _spatial_stats(("candidate", "candidate_power"), before, after, match, drive)
        gates = ("transmission_gate_passed", "total_power_gate_passed", "complex_amplitude_gate_passed")
        grid._require(all(direct[key] == published[key] for key in gates)
                      and all(isclose(direct["power_changes_percentage_points"][q], published[q],
                                      rel_tol=0, abs_tol=ROUND_OFF)
                              and isclose(direct["maximum_absolute_complex_change_by_block"][q],
                                          published["maximum_absolute_complex_amplitude_change_by_block"][q],
                                          rel_tol=0, abs_tol=ROUND_OFF) for q in grid.KINDS),
                      f"The signed direct replay differs from the checked spatial increments for drive {drive}.")
        results.append({"reference_drive": drive, "direct": direct, "snapshot": predicted,
                        "gate_agreement": {key: direct[key] == predicted[key] for key in gates}})
    edge = {key: pinned[key] for key in ("coarser_central_max_edge_nm", "finer_central_max_edge_nm")}
    edge["drives"] = results
    edge["all_gates_agree"] = all(all(row["gate_agreement"].values()) for row in results)
    return edge


def compare(root=ROOT, receipt_directory=None):
    """Reproduce the midpoint power and signed-amplitude comparisons from small checked files."""
    root = Path(root).resolve()
    receipt_directory = root / RECEIPTS if receipt_directory is None else Path(receipt_directory)
    compact = grid.load_thin_grid(root)
    grid._require(compact is not None, "The checked nine-point comparison is required.")
    _, content = grid._source(root, grid.COMPARISON, [])
    comparison = json.loads(content)
    inventory, _ = grid._channel_inventory(comparison)
    sources = {row["path"]: {**row, "path": Path(row["path"]).relative_to(root).as_posix()}
               for row in compact["sources"]}
    responses, levels = {}, []
    for filename, level, pinned in zip(FILES, compact["levels"], comparison["mesh_levels"], strict=True):
        receipt, pin = _read_receipt(receipt_directory / filename)
        target = level["central_max_edge_nm"]
        direct_points = level["points"]
        _check_inputs(receipt, direct_points, pinned, sources)
        responses[target], rows = _response(receipt, direct_points, inventory[target, MIDPOINT])
        errors = {name: max(abs(row[f"{name}_error_percentage_points"]) for row in rows)
                  for name in ("snapshot", "linear_endpoint")}
        levels.append({"central_max_edge_nm": target, "receipt": pin, "drives": rows,
                       "maximum_absolute_transmission_error_percentage_points": errors})
    edges = []
    for edge in comparison["channel_correspondence"]["signed_matches"]:
        (coarse, fa), (fine, fb) = edge["reference_point"], edge["candidate_point"]
        if fa == fb == MIDPOINT:
            pinned, = (row for row in compact["levels"][grid.TARGETS.index(coarse)]["points"][1]["neighboring_spatial_gates"]
                       if row["finer_central_max_edge_nm"] == fine)
            edges.append(_spatial_edge(responses[coarse], responses[fine], edge["groups"], pinned))
    grid._require(len(edges) == 2, "Both adjacent checked midpoint spatial edges are required.")
    return {"author": "Athul Prem", "file_tag": "_AP_pnc", "frequency_hz": MIDPOINT,
            "reference_planes_x_m": grid.PLANES, "direct_comparison": grid.COMPARISON,
            "mesh_levels": levels, "spatial_edges": edges,
            "all_spatial_gates_agree": all(edge["all_gates_agree"] for edge in edges)}


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--receipt-directory", type=Path, help="Directory containing the three candidate receipts.")
    args = parser.parse_args(argv)
    try:
        result = compare(receipt_directory=args.receipt_directory)
    except (ArithmeticError, KeyError, OSError, TypeError, ValueError) as exc:
        parser.exit(1, f"Midpoint comparison failed. {exc}\n")
    print(json.dumps(result, indent=2, allow_nan=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
