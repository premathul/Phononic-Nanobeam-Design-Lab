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
    figure = Figure(figsize=(8.3, 6.0), facecolor="white")
    axes = figure.subplots(1, 2, sharex=True, sharey=True)
    left, right = 0.102, 0.957
    figure.subplots_adjust(left=left, right=right, top=0.70, bottom=0.37, wspace=0.16)
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
    figure.supxlabel("Frequency (MHz)", y=0.305, fontsize=9)
    figure.text(left, 0.768, "Central edge ceiling (nm)", fontsize=8.6, color=MUTED, va="center")
    handles, labels = axes[0].get_legend_handles_labels()
    figure.legend(handles, labels, loc="center left", bbox_to_anchor=(0.29, 0.768),
                  frameon=False, ncols=3, columnspacing=1.6, borderaxespad=0)
    figure.text(
        left, 0.253, "997.5 → 1000 MHz: positive ΔT on each saved mesh; magnitude remains unconverged.",
        fontsize=9, weight="bold", va="top",
    )
    figure.text(
        left, 0.219, "   ·   ".join(change_labels) + "  (percentage points, legend order)",
        fontsize=8.5, va="top",
    )
    figure.text(
        left, 0.184, "Every fixed normalized coherent mixture of drives 3 and 5 has ΔT > 0 at these endpoints.",
        fontsize=8.5, va="top",
    )
    bounds = [row["transmission_eigenvalue_bounds_percentage_points"] for row in grid["coherent_endpoint_changes"]]
    figure.text(
        left, 0.151, "Δ(T†T) eigenvalue bounds: " + " / ".join(f"[{a:+.3f}, {b:+.3f}]" for a, b in bounds)
        + " pp (legend order)", fontsize=8.5, va="top",
    )
    figure.text(
        left, 0.118, "Incident phases matched; all five outgoing rows retained. Reflection has the opposite intervals.",
        fontsize=8.3, color=MUTED, va="top",
    )
    figure.text(
        left, 0.085, "Markers are saved samples; lines guide the eye. Two endpoints establish no linewidth or converged slope.",
        fontsize=8.3, color=MUTED, va="top",
    )
    return figure


def _coherent_rows(grid: dict) -> list[dict]:
    """Export exact endpoint Gram changes, phase correspondence and input identities."""
    rows = []
    for change in grid["coherent_endpoint_changes"]:
        row = {key: change[key] for key in ("central_max_edge_nm", "lower_frequency_hz", "upper_frequency_hz",
                                           "outgoing_rows_per_block", "closure_discrepancy_percentage_points")}
        for index, drive in enumerate(change["reference_drives"]):
            row[f"reference_drive_{index + 1}"] = drive
            for stem in ("lower_incident_native_roots", "upper_incident_native_roots",
                         "candidate_incident_gauge_phase_rad", "incident_mass_overlaps"):
                row[f"{stem}_drive_{drive}"] = change[stem][index]
        for kind in ("transmission", "reflection"):
            for label, value in zip(("min", "max"), change[f"{kind}_eigenvalue_bounds_percentage_points"], strict=True):
                row[f"{kind}_{label}_percentage_points"] = value
            for i, matrix_row in enumerate(change[f"delta_{kind}_gram_fraction"], 1):
                for j, value in enumerate(matrix_row, 1):
                    for part in ("real", "imag"):
                        row[f"delta_{kind}_gram_{i}{j}_{part}_fraction"] = value[part]
        row["left_reference_plane_m"], row["right_reference_plane_m"] = grid["reference_planes_x_m"]
        for stem, pin in change["provenance"].items():
            row[f"{stem}_path"] = pin["path"]
            row[f"{stem}_sha256"] = pin["sha256"]
        rows.append(row)
    return rows


def render(output_dir: Path) -> tuple[Path, Path, Path, Path]:
    """Write the PDF, PNG, 18 plotted samples and three coherent endpoint bounds."""
    grid = load_thin_grid(ROOT)
    if grid is None:
        raise ValueError("The saved thin-device grid is required to plot its frequency traces.")
    rows = _plotted_rows(grid)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    pdf_path = output_dir / f"athul_simulation_pnc_{IDENTIFIER}.pdf"
    png_path = pdf_path.with_suffix(".png")
    csv_path = output_dir / f"{IDENTIFIER}.csv"
    coherent_path = output_dir / "thin_coherent_endpoint_bounds_AP_pnc.csv"
    with mpl.rc_context({
        "font.family": "DejaVu Sans", "font.size": 9, "pdf.fonttype": 42,
        "axes.spines.top": False, "axes.spines.right": False, "axes.edgecolor": "#81919F",
        "axes.labelcolor": INK, "xtick.color": MUTED, "ytick.color": MUTED, "text.color": INK,
    }):
        figure = _plot_figure(grid, rows)
        metadata = {"Title": ATTRIBUTION, "Author": "Athul Prem"}
        figure.savefig(pdf_path, metadata={**metadata, "CreationDate": None})
        figure.savefig(png_path, dpi=240, metadata=metadata)

    for path, table in ((csv_path, rows), (coherent_path, _coherent_rows(grid))):
        with path.open("w", encoding="utf-8", newline="") as stream:
            writer = csv.DictWriter(stream, fieldnames=list(table[0]), lineterminator="\n")
            writer.writeheader()
            writer.writerows(table)
    return pdf_path, csv_path, png_path, coherent_path


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, required=True)
    for path in render(parser.parse_args().output_dir):
        print(path.resolve())
