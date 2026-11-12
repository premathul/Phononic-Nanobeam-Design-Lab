"""Nanobeam geometry checks and a deliberately simplified 1D chain model.

Author: Athul Prem | _AP_pnc.

The chain represents longitudinal displacement of lumped masses connected by
springs. It is NOT an elastic band-structure solver for a perforated beam.
"""
from __future__ import annotations

import argparse
import csv
import json
import math
from dataclasses import dataclass
from pathlib import Path

import numpy as np


@dataclass(frozen=True)
class Hole:
    x_nm: float
    diameter_nm: float
    region: str = ""


def read_holes(path: str | Path) -> list[Hole]:
    with open(path, newline="", encoding="utf-8") as stream:
        rows = list(csv.DictReader(stream))
    if not rows:
        raise ValueError("CSV must contain at least one hole")
    holes = [Hole(float(row["x_nm"]), float(row["diameter_nm"]),
                  row.get("region", "")) for row in rows]
    return holes


def inspect_geometry(holes: list[Hole], beam_width_nm: float,
                     min_edge_gap_nm: float = 0.0,
                     center_tolerance_nm: float = 1e-6) -> dict:
    """Check circular holes centered on beam centerline, sorted by x.

    edge_gap_nm is the x-direction clearance between adjacent circles.
    Side clearance assumes every hole center has transverse coordinate y=0.
    """
    if not holes or not math.isfinite(beam_width_nm) or beam_width_nm <= 0:
        raise ValueError("Nonempty holes and positive finite beam width required")
    if min_edge_gap_nm < 0 or center_tolerance_nm < 0:
        raise ValueError("Tolerances cannot be negative")
    if any(not math.isfinite(h.x_nm) or not math.isfinite(h.diameter_nm)
           or h.diameter_nm <= 0 for h in holes):
        raise ValueError("Hole coordinates and diameters must be finite; diameters positive")
    ordered = sorted(holes, key=lambda h: h.x_nm)
    issues = []
    adjacent = []
    for i, hole in enumerate(ordered):
        side_clearance = (beam_width_nm - hole.diameter_nm) / 2
        if side_clearance < 0:
            issues.append(f"hole {i + 1}: circular hole exceeds beam width")
        if i:
            previous = ordered[i - 1]
            pitch = hole.x_nm - previous.x_nm
            gap = pitch - (previous.diameter_nm + hole.diameter_nm) / 2
            adjacent.append({"left_hole": i, "right_hole": i + 1,
                             "pitch_nm": pitch, "edge_gap_nm": gap})
            if gap < min_edge_gap_nm:
                issues.append(f"holes {i}–{i + 1}: edge gap {gap:.3f} nm "
                              f"below required {min_edge_gap_nm:.3f} nm")
    # Symmetry check only when both corresponding centers are explicitly present.
    by_position = {round(h.x_nm / max(center_tolerance_nm, 1e-12)): h
                   for h in ordered}
    symmetry = []
    for hole in ordered:
        if hole.x_nm >= -center_tolerance_nm:
            continue
        other = by_position.get(round(-hole.x_nm / max(center_tolerance_nm, 1e-12)))
        if other is not None and abs(hole.diameter_nm - other.diameter_nm) > center_tolerance_nm:
            symmetry.append({"x_nm": hole.x_nm,
                             "diameter_mismatch_nm": abs(hole.diameter_nm - other.diameter_nm)})
    return {"hole_count": len(ordered), "beam_width_nm": beam_width_nm,
            "adjacent": adjacent, "issues": issues,
            "paired_mirror_diameter_mismatches": symmetry,
            "note": "Coordinates represent circle centers on y=0; substrate and anchors are not modeled."}


def diatomic_branches(k_rad: np.ndarray, lattice_m: float,
                      mass_a_kg: float, mass_b_kg: float,
                      spring_n_m: float) -> tuple[np.ndarray, np.ndarray]:
    """Return acoustic and optical frequencies in Hz of an infinite alternating chain.

    Omega^2 = K(1/mA+1/mB) ± K sqrt((1/mA+1/mB)^2
              -4 sin^2(ka/2)/(mA*mB)). Cell length a contains two masses.
    """
    if min(lattice_m, mass_a_kg, mass_b_kg, spring_n_m) <= 0:
        raise ValueError("Cell length, masses and spring constant must be positive")
    k_rad = np.asarray(k_rad, dtype=float)
    if not np.all(np.isfinite(k_rad)):
        raise ValueError("Wave vectors must be finite")
    inverse_sum = 1 / mass_a_kg + 1 / mass_b_kg
    discriminant = inverse_sum**2 - 4 * np.sin(k_rad * lattice_m / 2)**2 / (mass_a_kg * mass_b_kg)
    root = np.sqrt(np.maximum(discriminant, 0))
    acoustic = np.sqrt(np.maximum(spring_n_m * (inverse_sum - root), 0)) / (2 * np.pi)
    optical = np.sqrt(np.maximum(spring_n_m * (inverse_sum + root), 0)) / (2 * np.pi)
    return acoustic, optical


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    geometry = sub.add_parser("geometry", help="Check circle spacing and beam side clearance")
    geometry.add_argument("csv_file")
    geometry.add_argument("--beam-width-nm", type=float, required=True)
    geometry.add_argument("--min-edge-gap-nm", type=float, default=0.0)
    bands = sub.add_parser("chain", help="Illustrative alternating mass-spring dispersion")
    bands.add_argument("--cell-nm", type=float, default=1000.0)
    bands.add_argument("--mass-a-kg", type=float, default=1e-18)
    bands.add_argument("--mass-b-kg", type=float, default=2e-18)
    bands.add_argument("--spring-n-m", type=float, default=1.0)
    bands.add_argument("--points", type=int, default=101)
    args = parser.parse_args()
    if args.command == "geometry":
        report = inspect_geometry(read_holes(args.csv_file), args.beam_width_nm,
                                  args.min_edge_gap_nm)
        print(json.dumps(report, indent=2))
        if report["issues"]:
            raise SystemExit(1)
    else:
        if args.points < 2:
            parser.error("--points must be at least 2")
        cell = args.cell_nm * 1e-9
        if cell <= 0:
            parser.error("--cell-nm must be positive")
        k = np.linspace(0, np.pi / cell, args.points)
        low, high = diatomic_branches(k, cell, args.mass_a_kg,
                                       args.mass_b_kg, args.spring_n_m)
        print("ka_over_pi,acoustic_hz,optical_hz")
        for q, f0, f1 in zip(k * cell / np.pi, low, high):
            print(f"{q:.8g},{f0:.8g},{f1:.8g}")


if __name__ == "__main__":
    main()
