#!/usr/bin/env python3
"""Inspect a saved matched-port comparison. Author: Athul Prem | _AP_pnc."""

import argparse
import cmath
import json
import math
from pathlib import Path
import sys


def comparison_lines(summary: dict) -> list[str]:
    if summary["schema_version"] != 1 or summary["status"] != "compared":
        raise ValueError("Expected a completed scattering comparison summary (schema 1).")

    matches = summary["matched_channels"]
    incoming = list(zip(matches["incoming"]["reference_roots"], matches["incoming"]["candidate_roots"], strict=True))
    changes = []
    power_rows = []
    for quantity in ("reflection", "transmission"):
        block = summary["blocks"][quantity]
        outgoing = list(zip(matches[quantity]["reference_roots"], matches[quantity]["candidate_roots"], strict=True))
        for row, (reference, candidate, outgoing_roots) in enumerate(
            zip(block["reference"], block["candidate_aligned"], outgoing, strict=True), start=1
        ):
            for column, (a, b, incoming_roots) in enumerate(zip(reference, candidate, incoming, strict=True), start=1):
                first, second = complex(*a), complex(*b)
                change = abs(second - first)
                if not math.isfinite(change):
                    raise ValueError("The saved complex amplitude change is not finite.")
                changes.append((change, quantity, column, row, incoming_roots, outgoing_roots, first, second))

        for channel, (reference_power, candidate_power, roots) in enumerate(
            zip(block["reference_power"], block["candidate_power"], incoming, strict=True), start=1
        ):
            if not all(math.isfinite(power) and power >= 0 for power in (reference_power, candidate_power)):
                raise ValueError("Saved total power must be finite and nonnegative.")
            delta = candidate_power - reference_power
            relative = f"{100 * (delta / reference_power):+.8g}%" if reference_power else "undefined (zero reference)"
            power_rows.append(
                f"{quantity.title():<12} {channel:>5} {roots[0]:>8} {roots[1]:>8} "
                f"{reference_power:>12.8g} {candidate_power:>12.8g} {abs(delta):>12.8g} {100 * delta:>+14.8g}  {relative}"
            )

    change, quantity, column, row, incident, outgoing, first, second = max(changes, key=lambda entry: entry[0])
    if not math.isclose(change, summary["maximum_complex_change"], rel_tol=1e-12):
        raise ValueError("The aligned coefficients disagree with the saved maximum complex amplitude change.")
    phase = "undefined (zero magnitude)"
    if first and second:
        phase_change = math.remainder(cmath.phase(second) - cmath.phase(first), math.tau)
        phase = f"{math.degrees(phase_change):+.8g} deg"

    physical = summary["physical_inputs"]
    return [
        f"{physical['frequency_hz'] / 1e9:g} GHz, {physical['incident_side']} incidence",
        f"Largest complex amplitude change at the original planes: {change:.8g}",
        f"  Incident channel {column}, reference root {incident[0]}, candidate root {incident[1]}",
        f"  Outgoing {quantity} channel {row}, reference root {outgoing[0]}, candidate root {outgoing[1]}",
        f"  Aligned magnitudes: reference {abs(first):.8g}, candidate {abs(second):.8g}",
        f"  Absolute magnitude-only mismatch: {abs(abs(second) - abs(first)):.8g}",
        f"  Wrapped phase change (candidate minus reference): {phase}",
        "",
        f"Maximum absolute total power change: {summary['maximum_power_change']:.8g}",
        "Total power per incident drive (fractions of incident power)",
        "Relative change is the signed change divided by reference power. pp = percentage points.",
        "Quantity     Drive Ref root Cand root    Ref power   Cand power     |Change|    Signed (pp)  Relative (%)",
        *power_rows,
    ]


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("comparison", type=Path, help="Saved comparison directory or summary_AP_pnc.json file")
    args = parser.parse_args(argv)
    path = args.comparison
    if path.is_dir():
        path /= "summary_AP_pnc.json"
    try:
        lines = comparison_lines(json.loads(path.read_text(encoding="utf-8")))
    except (OSError, ValueError, KeyError, TypeError) as exc:
        print(f"Saved scattering comparison could not be inspected: {exc}", file=sys.stderr)
        return 2
    print("\n".join(lines))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
