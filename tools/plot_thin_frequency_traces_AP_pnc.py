"""Plot the saved 100 nm device frequency traces. Author: Athul Prem | _AP_pnc."""

from __future__ import annotations

import argparse
import csv
from pathlib import Path

import matplotlib as mpl
from matplotlib.figure import Figure

if __package__:
    from .read_thin_device_grid_AP_pnc import load_thin_grid
else:
    from read_thin_device_grid_AP_pnc import load_thin_grid


ROOT = Path(__file__).resolve().parents[1]
IDENTIFIER = "thin_frequency_traces_AP_pnc"
ATTRIBUTION = f"Athul Prem | athul_simulation_pnc_{IDENTIFIER}"
TITLE = "Transmission near 1 GHz"
DRIVES = (3, 5)
COLORS = ("#176B87", "#B65324", "#6B4C9A")
MARKERS = ("o", "s", "^")
INK = "#192E42"
MUTED = "#4B5A68"


def _plotted_rows(grid: dict) -> list[dict]:
    source_hashes = {item["path"]: item["sha256"] for item in grid["sources"]}
    comparison = grid["source_comparison"]
    rows = []
    for level in grid["levels"]:
        for point in level["points"]:
            frequency = point["frequency_hz"]
            native_map = point["source_map"]
            coefficients = point["source_coefficients"]
            for drive in point["drives"]:
                if drive["reference_drive"] not in DRIVES:
                    continue
                rows.append({
                    "central_max_edge_nm": level["central_max_edge_nm"],
                    "frequency_hz": frequency,
                    "frequency_mhz": frequency / 1e6,
                    "reference_drive": drive["reference_drive"],
                    "native_drive": drive["native_drive"],
                    "native_incident_root": drive["incident_port_index"],
                    "transmission_fraction": drive["transmission"],
                    "transmission_percent": 100 * drive["transmission"],
                    "reflection_fraction": drive["reflection"],
                    "source_map_path": Path(native_map).relative_to(ROOT).as_posix(),
                    "source_map_sha256": source_hashes[native_map],
                    "source_coefficients_path": Path(coefficients).relative_to(ROOT).as_posix(),
                    "source_coefficients_sha256": source_hashes[coefficients],
                    "configuration_sha256": point["configuration_sha256"],
                    "comparison_record_path": Path(comparison).relative_to(ROOT).as_posix(),
                    "comparison_record_sha256": source_hashes[comparison],
                })
    return rows


def _plot_figure(grid: dict, rows: list[dict]) -> Figure:
    figure = Figure(figsize=(8.3, 5.05), facecolor="white")
    axes = figure.subplots(1, 2, sharex=True, sharey=True)
    left, right = 0.102, 0.957
    figure.subplots_adjust(left=left, right=right, top=0.688, bottom=0.282, wspace=0.16)
    figure.text(left, 0.968, ATTRIBUTION, fontsize=8.2, color=MUTED, va="top")
    figure.text(left, 0.924, TITLE, fontsize=17, weight="bold", va="top")
    figure.text(
        left, 0.855,
        f"Straight 26-hole Ge beam · {grid['physical_thickness_nm']} nm thick · "
        f"{grid['thickness_layers']} quadratic thickness layers · "
        f"{grid['circle_resolution']} facets per hole · {grid['incident_side']} incidence",
        fontsize=9, va="top",
    )
    meshes = grid["central_max_edges_nm"]
    frequencies_mhz = [value / 1e6 for value in grid["frequencies_hz"]]
    plane_left, plane_right = (value * 1e6 for value in grid["reference_planes_x_m"])
    figure.text(
        left, 0.82, f"Reference drives use the {meshes[0]:g} nm mesh at {frequencies_mhz[-1] / 1000:g} GHz. "
        f"Original end planes at x = {plane_left:g} and {plane_right:g} µm.",
        fontsize=8.7, color=MUTED, va="top",
    )

    change_labels = []
    for axis, drive in zip(axes, DRIVES, strict=True):
        axis.set_title(f"Reference drive {drive}", loc="left", fontsize=10, weight="bold", pad=7)
        axis.set_xticks(frequencies_mhz, [f"{frequency:g}" for frequency in frequencies_mhz])
        axis.set_xlim(frequencies_mhz[0] - 0.45, frequencies_mhz[-1] + 0.45)
        axis.set_yticks(range(0, 101, 20))
        axis.set_ylim(0, 100)
        axis.grid(axis="y", color="#DFE6EC", linewidth=0.7)
        axis.set_axisbelow(True)
        axis.tick_params(axis="both", length=3, width=0.7, color="#81919F")
        deltas = []
        for mesh, color, marker in zip(meshes, COLORS, MARKERS, strict=True):
            samples = [row for row in rows if row["reference_drive"] == drive and row["central_max_edge_nm"] == mesh]
            axis.plot(
                [row["frequency_mhz"] for row in samples], [row["transmission_percent"] for row in samples],
                label=f"{mesh:.15g}", color=color, marker=marker, linewidth=1.3, markersize=5.3,
                markeredgecolor="white", markeredgewidth=0.65,
            )
            deltas.append(100 * (samples[-1]["transmission_fraction"] - samples[-2]["transmission_fraction"]))
        change_labels.append(f"Drive {drive}  " + " / ".join(f"{delta:+.2f}" for delta in deltas))

    axes[0].set_ylabel("Transmitted / incident power (%)", labelpad=7)
    figure.supxlabel("Frequency (MHz)", y=0.208, fontsize=9)
    figure.text(left, 0.768, "Central edge ceiling (nm)", fontsize=8.6, color=MUTED, va="center")
    handles, labels = axes[0].get_legend_handles_labels()
    figure.legend(handles, labels, loc="center left", bbox_to_anchor=(0.29, 0.768),
                  frameon=False, ncols=3, columnspacing=1.6, borderaxespad=0)
    figure.text(
        left, 0.144, "Transmission rises from 997.5 to 1000 MHz on all three meshes. The increase depends on mesh.",
        fontsize=9, weight="bold", va="top",
    )
    figure.text(
        left, 0.108, "   ·   ".join(change_labels) + "  (percentage points, legend order)",
        fontsize=8.5, va="top",
    )
    figure.text(
        left, 0.073, "Markers show saved samples. Lines guide the eye. A converged slope and linewidth remain unresolved.",
        fontsize=8.3, color=MUTED, va="top",
    )
    return figure


def render(output_dir: Path) -> tuple[Path, Path, Path]:
    """Write the vector PDF, PNG preview and 18 plotted samples."""
    grid = load_thin_grid(ROOT)
    if grid is None:
        raise ValueError("The saved thin-device grid is required to plot its frequency traces.")
    rows = _plotted_rows(grid)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    pdf_path = output_dir / f"athul_simulation_pnc_{IDENTIFIER}.pdf"
    png_path = pdf_path.with_suffix(".png")
    csv_path = output_dir / f"{IDENTIFIER}.csv"
    with mpl.rc_context({
        "font.family": "DejaVu Sans", "font.size": 9, "pdf.fonttype": 42,
        "axes.spines.top": False, "axes.spines.right": False, "axes.edgecolor": "#81919F",
        "axes.labelcolor": INK, "xtick.color": MUTED, "ytick.color": MUTED, "text.color": INK,
    }):
        figure = _plot_figure(grid, rows)
        metadata = {"Title": ATTRIBUTION, "Author": "Athul Prem"}
        figure.savefig(pdf_path, metadata={**metadata, "CreationDate": None})
        figure.savefig(png_path, dpi=240, metadata=metadata)

    with csv_path.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]), lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)
    return pdf_path, csv_path, png_path


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, required=True)
    for path in render(parser.parse_args().output_dir):
        print(path.resolve())
