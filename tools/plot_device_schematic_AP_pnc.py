"""Draw the saved straight-device geometry. Athul Prem | _AP_pnc."""

from __future__ import annotations

import argparse
import csv
from hashlib import sha256
from itertools import groupby
import json
from math import isclose
from pathlib import Path

import matplotlib as mpl
from matplotlib.figure import Figure
from matplotlib.patches import Circle, Rectangle


ROOT = Path(__file__).resolve().parents[1]
SAVED_SUMMARY = ROOT / "data/saved_device_AP_pnc/h80_z6_1/summary_AP_pnc.json"
REGIONAL_CONFIG = ROOT / "configs/historical_26hole_regional_AP_pnc.json"
REGIONAL_EVIDENCE = ROOT / "verification/regional_lead_1GHz_AP_pnc.json"
IDENTIFIER = "athul_simulation_pnc_device_schematic"
BLUE = "#176B87"
ORANGE = "#B65324"
INK = "#202F38"
MUTED = "#4B5B64"
SOLID = "#DCEEF3"
MESH_DASH = (0, (2.5, 2.0))


def _read_geometry():
    saved = json.loads(SAVED_SUMMARY.read_bytes())
    regional_bytes = REGIONAL_CONFIG.read_bytes()
    regional = json.loads(regional_bytes)
    evidence = json.loads(REGIONAL_EVIDENCE.read_bytes())
    geometry = regional["geometry"]
    length, width, thickness = (geometry[key] for key in ("length_nm", "width_nm", "thickness_nm"))
    if (geometry != saved["configuration"]["geometry"]
            or sha256(regional_bytes).hexdigest() != evidence["preset"]["sha256"]
            or not all(isclose(actual, expected * 1e-9, rel_tol=1e-12) for actual, expected in
                       zip(evidence["reference_planes_x_m"], (0, length), strict=True))):
        raise ValueError("The saved and regional configurations must share the plotted footprint and end planes.")

    pattern = geometry["pattern"]
    source = ROOT / pattern["csv"]
    provenance = saved["hole_pattern_provenance"]
    if sha256(source.read_bytes()).hexdigest() != provenance["source_sha256"]:
        raise ValueError("The hole table differs from the saved geometry source.")
    with source.open(newline="", encoding="utf-8-sig") as table:
        holes = []
        for row in csv.DictReader(table):
            source_position = row.pop("position_nm")
            holes.append(dict(row, source_position_nm=source_position,
                              x_nm=float(source_position) + pattern["x_offset_nm"], y_nm=0))
    if len(holes) != 26:
        raise ValueError("The straight device requires 26 holes.")
    for hole, reference in zip(holes, provenance["holes"], strict=True):
        if (not isclose(hole["x_nm"] * 1e-9, reference["x_m"], rel_tol=1e-12)
                or not isclose(float(hole["diameter_nm"]) * 1e-9, reference["diameter_m"], rel_tol=1e-12)
                or hole["region"] != reference["region"]):
            raise ValueError(f"Hole {hole['hole_number']} differs from the saved geometry.")

    cuts = tuple(regional["mesh"]["patterned_x_interval_nm"])
    return (length, width, thickness), cuts, holes


def _bracket(ax, left, right, y, color, tick=0.11):
    ax.plot([left, left, right, right], [y - tick, y, y, y - tick], color=color, linewidth=0.85)


def _plan(fig, length_nm, width_nm, cuts_nm, holes):
    ax = fig.add_axes((0.093, 0.409, 0.856, 0.349))
    length, width = length_nm / 1000, width_nm / 1000
    diameters = [float(hole["diameter_nm"]) for hole in holes]
    centers = [hole["x_nm"] / 1000 for hole in holes]
    start = centers[0] - diameters[0] / 2000
    stop = centers[-1] + diameters[-1] / 2000
    ax.set(xlim=(-0.30, length + 0.30), ylim=(-1.08, 1.66), aspect="equal")
    ax.add_patch(Rectangle((0, -width / 2), length, width, facecolor=SOLID, edgecolor=BLUE, linewidth=1.05))
    for x, diameter in zip(centers, diameters, strict=True):
        ax.add_patch(Circle((x, 0), diameter / 2000, facecolor="white", edgecolor=BLUE, linewidth=0.8))

    for cut in cuts_nm:
        ax.vlines(cut / 1000, -width / 2, width / 2, color=MUTED, linewidth=1.15, linestyle=MESH_DASH)
    for x, label, align in [(0, "Left port", "left"), (length, "Right port", "right")]:
        ax.vlines(x, -width / 2 - 0.11, width / 2, color=ORANGE, linewidth=1.6)
        ax.text(x, -width / 2 - 0.22, label, ha=align, va="top", color=ORANGE, fontsize=8)

    for left, right, label in [
        (0, start, f"Left lead\n{start:.2f} µm"),
        (start, stop, f"Patterned region · {stop - start:.2f} µm\n"
                      f"{len(holes)} through-holes · Ø {min(diameters):g}–{max(diameters):g} nm"),
        (stop, length, f"Right lead\n{length - stop:.2f} µm"),
    ]:
        _bracket(ax, left, right, 0.84, MUTED)
        ax.text((left + right) / 2, 0.99, label, ha="center", va="bottom", color=INK, fontsize=8.6)

    labels = {"Mirror transition": "Mirror", "Central cavity": "Cavity"}
    regions = [labels.get(hole["region"], hole["region"]) for hole in holes]
    bounds = [start] + [(a + b) / 2 for a, b in zip(centers[:-1], centers[1:])] + [stop]
    for label, group in groupby(enumerate(regions), key=lambda pair: pair[1]):
        indices = [index for index, _ in group]
        left, right = bounds[indices[0]], bounds[indices[-1] + 1]
        ax.hlines(-0.62, left + 0.055, right - 0.055, color=BLUE, linewidth=0.85)
        ax.text((left + right) / 2, -0.72, label, color=BLUE, ha="center", va="top", fontsize=8)

    ax.set_xticks([0, 4, 8, 12, length], ["0", "4", "8", "12", f"{length:.2f}"])
    ax.set_yticks([-width / 2, 0, width / 2], [f"−{width / 2:g}", "0", f"{width / 2:g}"])
    ax.set_xlabel("Lab $x$ (µm)", labelpad=4)
    ax.set_ylabel("Lab $y$ (µm)", labelpad=4)
    ax.spines[["top", "right"]].set_visible(False)
    ax.spines["left"].set_bounds(-width / 2, width / 2)
    ax.spines["bottom"].set_bounds(0, length)
    ax.tick_params(length=3, width=0.7, pad=3, color=MUTED)


def _section(fig, width_nm, thickness_nm):
    width, thickness = width_nm / 1000, thickness_nm / 1000
    ax = fig.add_axes((0.110, 0.065, 0.36, 0.217), anchor="W")
    ax.set(xlim=(-0.68, 1.16), ylim=(-0.29, 0.39), aspect="equal")
    ax.axis("off")
    ax.add_patch(Rectangle((-width / 2, -thickness / 2), width, thickness,
                           facecolor=SOLID, edgecolor=BLUE, linewidth=1.05))
    dim = dict(arrowstyle="<->", color=MUTED, linewidth=0.8, shrinkA=0, shrinkB=0, mutation_scale=8)
    ax.vlines([-width / 2, width / 2], thickness / 2 + 0.02, 0.25, color=MUTED, linewidth=0.7)
    ax.annotate("", (width / 2, 0.21), (-width / 2, 0.21), arrowprops=dim)
    ax.text(0, 0.26, f"{width:.2f} µm", ha="center", va="bottom", fontsize=8.6)
    ax.hlines([-thickness / 2, thickness / 2], width / 2 + 0.02, 0.73, color=MUTED, linewidth=0.7)
    ax.annotate("", (0.68, thickness / 2), (0.68, -thickness / 2), arrowprops=dim)
    ax.text(0.77, 0, f"{thickness_nm:g} nm", ha="left", va="center", fontsize=8.6)
    direction = dict(arrowstyle="->", color=INK, linewidth=0.8, shrinkA=0, shrinkB=0, mutation_scale=8)
    origin = (-0.53, -0.24)
    ax.annotate("", (-0.31, origin[1]), origin, arrowprops=direction)
    ax.annotate("", (origin[0], -0.12), origin, arrowprops=direction)
    ax.text(-0.27, origin[1], "$y$", ha="left", va="center", fontsize=8.6)
    ax.text(-0.59, -0.13, "$z$", ha="right", va="center", fontsize=8.6)


def _plane_legend(fig, length_nm, cuts_nm):
    ax = fig.add_axes((0.543, 0.073, 0.414, 0.228))
    ax.set(xlim=(0, 1), ylim=(0, 1))
    ax.axis("off")
    left_cut, right_cut = (value / 1000 for value in cuts_nm)
    for y, color, style, weight, label, coordinates in [
        (0.73, ORANGE, "-", 1.6, "Physical port reference planes", f"$x$ = 0 and {length_nm / 1000:.2f} µm"),
        (0.25, MUTED, MESH_DASH, 1.15, "Regional mesh cuts", f"$x$ = {left_cut:.2f} and {right_cut:.2f} µm"),
    ]:
        ax.hlines(y, 0.00, 0.11, color=color, linewidth=weight, linestyle=style)
        ax.text(0.154, y + 0.02, label, va="center", fontsize=9)
        ax.text(0.154, y - 0.22, coordinates, va="center", color=MUTED, fontsize=8.5)


def render(output_dir: Path) -> tuple[Path, Path, Path]:
    """Write the vector schematic and exact placed hole table from tracked inputs."""
    (length, width, thickness), cuts, holes = _read_geometry()
    output_dir.mkdir(parents=True, exist_ok=True)
    pdf_path = output_dir / f"{IDENTIFIER}_AP_pnc.pdf"
    png_path = pdf_path.with_suffix(".png")
    csv_path = output_dir / "device_geometry_AP_pnc.csv"
    with mpl.rc_context({"font.size": 9, "text.color": INK, "axes.labelcolor": INK,
                         "axes.edgecolor": MUTED, "axes.linewidth": 0.7, "xtick.color": INK,
                         "ytick.color": INK, "pdf.fonttype": 42}):
        fig = Figure(figsize=(7.30, 3.82), facecolor="white")
        fig.text(0.061, 0.951, f"Athul Prem | {IDENTIFIER}", fontsize=8, color=MUTED)
        fig.text(0.061, 0.870, "Straight 26-hole germanium nanobeam", fontsize=16, fontweight="bold")
        fig.text(0.061, 0.795, "(a)  Plan view", fontsize=10, fontweight="bold")
        fig.text(0.061, 0.299, "(b)  Port cross-section", fontsize=10, fontweight="bold")
        _plan(fig, length, width, cuts, holes)
        _section(fig, width, thickness)
        _plane_legend(fig, length, cuts)
        fig.savefig(pdf_path, metadata={
            "Author": "Athul Prem",
            "Title": "Straight 26-hole germanium nanobeam",
            "Subject": f"Saved {thickness:g} nm thick device and original reference planes",
            "CreationDate": None,
        })
        fig.savefig(png_path, dpi=240, metadata={"Author": "Athul Prem", "Title": "Straight 26-hole germanium nanobeam"})
    fields = ["hole_number", "source_position_nm", "x_nm", "y_nm", "diameter_nm", "region"]
    with csv_path.open("w", newline="", encoding="utf-8") as table:
        writer = csv.DictWriter(table, fieldnames=fields, lineterminator="\n")
        writer.writeheader()
        writer.writerows(holes)
    return pdf_path, csv_path, png_path


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", required=True, type=Path, help="Output directory")
    for path in render(parser.parse_args().output):
        print(path)


if __name__ == "__main__":
    main()
