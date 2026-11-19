"""Render saved 887.5 MHz reflection powers, reciprocity and mesh sensitivity.

Athul Prem | _AP_pnc
"""

from __future__ import annotations

import argparse
import csv
from hashlib import sha256
import json
from math import isclose
from pathlib import Path

import matplotlib
from matplotlib.figure import Figure
from matplotlib.lines import Line2D

from weak_conversion_diagnostics_AP_pnc import load_diagnostics


ROOT = Path(__file__).resolve().parents[1]
IDENTIFIER = "weak_conversion_8875"
TITLE = "Reflected mode conversion at 887.5 MHz"
ATTRIBUTION = f"Athul Prem | athul_simulation_pnc_{IDENTIFIER}"
INPLANE_SOURCE = "data/saved_device_AP_pnc/controls/frequency_inplane_8875_AP_pnc.json"
THICKNESS_SOURCE = "data/saved_device_AP_pnc/controls/frequency_thickness_8875_AP_pnc.json"
MESHES = ("h100 z4", "h80 z4", "h80 z6")
DIRECTIONS = ("1 → 5", "5 → 1")
COLORS = ("#176B87", "#B65324")
MARKERS = ("o", "D")
OFFSETS = (-0.075, 0.075)
INK = "#192E42"
MUTED = "#4B5A68"
DIAGNOSTIC_COLOR = "#34537E"


def _load_rows() -> list[dict]:
    manifest = json.loads((ROOT / "configs/saved_device_dataset_AP_pnc.json").read_bytes())
    hashes = {item["source"]["path"]: item["source"]["sha256"] for item in manifest["controls"]}
    controls = {}
    for source in (INPLANE_SOURCE, THICKNESS_SOURCE):
        raw = (ROOT / source).read_bytes()
        if sha256(raw).hexdigest() != hashes[source]:
            raise ValueError(f"Saved control differs from the dataset manifest: {source}")
        controls[source] = json.loads(raw)
        if controls[source]["frequency_hz"] != 887_500_000:
            raise ValueError(f"Unexpected frequency in {source}")

    inplane = controls[INPLANE_SOURCE]
    thickness = controls[THICKNESS_SOURCE]
    meshes = (
        (MESHES[0], 100, 4, inplane, INPLANE_SOURCE, "h100", "reference_h100", "h100_8875"),
        (MESHES[1], 80, 4, inplane, INPLANE_SOURCE, "h80", "candidate_h80", "h80_8875"),
        (MESHES[2], 80, 6, thickness, THICKNESS_SOURCE, "z6", "candidate", "z6_8875"),
    )
    rows = []
    for mesh, h_nm, layers, control, source, resolution, response_key, input_key in meshes:
        response = control["raw_responses"][response_key]
        lookup = {(r["incident_root"], r["outgoing_root"]): r["power"]
                  for r in response["coefficients"] if r["kind"] == "reflection"}
        for direction, selected in enumerate(control["selected_pair"], 1):
            roots = (selected["incident_root"], selected["outgoing_root"]) if control is inplane else (
                selected["candidate_incident_root"], selected["candidate_outgoing_root"])
            power = lookup[roots]
            targets = control["pair_specific_targets"]
            anchor = next(r for r in targets["suppression_retention_rows"]
                          if r["direction"] == direction and r["resolution"] == resolution)
            if power != anchor["power_8875"] or power <= 0:
                raise ValueError(f"Inconsistent saved 887.5 MHz power for {mesh}, direction {direction}")
            coefficient = control["inputs"][f"{input_key}_coefficients"]
            rows.append({
                "mesh": mesh,
                "global_inplane_target_nm": h_nm,
                "quadratic_thickness_layers": layers,
                "physical_thickness_nm": 200,
                "direction": DIRECTIONS[direction - 1],
                "incident_root_887500000_hz": roots[0],
                "reflected_root_887500000_hz": roots[1],
                "power_fraction_887500000_hz": power,
                "power_fraction_1000000000_hz": anchor["power_one_ghz"],
                "power_ratio_887500000_to_1000000000_hz": anchor["ratio"],
                "power_ratio_target": anchor["limit"],
                "power_fraction_target_887500000_hz": anchor["limit"] * anchor["power_one_ghz"],
                "suppression_target_passed": anchor["passed"],
                "finer_mesh_denominator": "",
                "finer_mesh_power_fraction": "",
                "absolute_power_fraction_increment": "",
                "relative_absolute_power_increment": "",
                "relative_absolute_power_increment_target": "",
                "mesh_sensitivity_target_passed": "",
                "whole_block_maximum_absolute_power_increment": "",
                "whole_block_absolute_power_increment_target": "",
                "whole_block_power_target_passed": "",
                "whole_block_maximum_complex_amplitude_increment": "",
                "whole_block_complex_amplitude_increment_target": "",
                "whole_block_complex_target_passed": "",
                "portable_control_source": source,
                "native_887500000_hz_coefficient_archive_source": coefficient["path"],
                "native_887500000_hz_coefficient_archive_sha256": coefficient["sha256"],
                "mesh_sensitivity_source": "",
            })

    for first, control, source in ((0, inplane, INPLANE_SOURCE),
                                   (2, thickness, THICKNESS_SOURCE)):
        for offset, gate in enumerate(control["pair_specific_targets"]["directed_increment_rows"]):
            coarse, fine = rows[first + offset], rows[first + 2 + offset]
            difference = abs(coarse["power_fraction_887500000_hz"] - fine["power_fraction_887500000_hz"])
            relative = difference / fine["power_fraction_887500000_hz"]
            if (fine["power_fraction_887500000_hz"] != gate["exact_denominator"]
                    or not isclose(relative, gate["relative_absolute_power_change"], rel_tol=1e-12)):
                raise ValueError(f"Saved weak-conversion target does not match {coarse['mesh']} and {fine['mesh']}")
            coarse.update(
                finer_mesh_denominator=fine["mesh"],
                finer_mesh_power_fraction=gate["exact_denominator"],
                absolute_power_fraction_increment=difference,
                relative_absolute_power_increment=relative,
                relative_absolute_power_increment_target=gate["limit"],
                mesh_sensitivity_target_passed=gate["passed"],
                whole_block_maximum_absolute_power_increment=control["maximum_power_change"],
                whole_block_absolute_power_increment_target=control["power_change_limit"],
                whole_block_power_target_passed=control["power_passed"],
                whole_block_maximum_complex_amplitude_increment=control["maximum_complex_change"],
                whole_block_complex_amplitude_increment_target=control["complex_change_limit"],
                whole_block_complex_target_passed=control["complex_passed"],
                mesh_sensitivity_source=source,
            )
    return rows


def _write_csv(path: Path, rows: list[dict]) -> None:
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]), lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)


def _scientific_label(value: float) -> str:
    mantissa, exponent = f"{value:.5e}".split("e")
    return rf"${mantissa}\times10^{{{int(exponent)}}}$"


def _annotate(ax, x: float, y: float, label: str) -> None:
    ax.annotate(label, (x, y), xytext=(0, 7), textcoords="offset points",
                ha="center", fontsize=7.8, color=MUTED)


def _plot_figure(rows: list[dict], diagnostics: list[dict]) -> Figure:
    fig = Figure(figsize=(8.35, 6.8), facecolor="white")
    (powers, reciprocity), (cross_ratio, sensitivity) = fig.subplots(2, 2)
    left, right = .112, .965
    fig.subplots_adjust(left=left, right=right, bottom=.17, top=.812, wspace=.39, hspace=.59)
    fig.text(left, .955, ATTRIBUTION, fontsize=8.2, color=MUTED)
    fig.text(left, .913, TITLE, fontsize=17, fontweight="bold", color=INK)
    fig.text(left, .878, "26-hole germanium beam · original end planes", fontsize=8.9, color=MUTED)
    fig.legend(
        [Line2D([], [], color=color, marker=marker, linestyle="none", markersize=5.3)
         for color, marker in zip(COLORS, MARKERS, strict=True)],
        [f"Mode {direction}" for direction in DIRECTIONS],
        frameon=False, ncols=2, loc="upper right", bbox_to_anchor=(right, .910),
        handletextpad=.28, columnspacing=.95, borderaxespad=0,
    )

    titles = ("(a) Directed reflected power", "(b) Same-mesh reciprocity",
              "(c) Reflection-block cross-ratio", "(d) Sensitivity to mesh refinement")
    for ax, title in zip((powers, reciprocity, cross_ratio, sensitivity), titles, strict=True):
        ax.set_title(title, loc="left", fontweight="bold", color=INK, pad=8)
        ax.set_axisbelow(True)
        ax.grid(axis="y", which="major", color="#DFE6EC", linewidth=.7)
        ax.tick_params(axis="both", which="major", length=3)
        ax.tick_params(axis="y", which="minor", length=0)
    for ax in (powers, reciprocity, cross_ratio):
        ax.set_xlim(-.48, 2.48)
        ax.set_xticks(range(len(MESHES)), MESHES)

    powers.set_ylabel("Reflected / incident power")
    powers.set_yscale("log")
    powers.set_ylim(4.9e-8, 3.4e-5)
    powers.set_yticks([1e-7, 1e-6, 1e-5])
    powers.text(1, 2.07e-5, "Dashed marks show 1% of each 1 GHz power",
                ha="center", fontsize=7.6, color=MUTED)

    reciprocity.set_ylabel(r"$|P_{1\to5} - P_{5\to1}|\,/\,\overline{P}$  ($\times10^{-9}$)")
    reciprocity.set_ylim(0, 4.09e-9)
    reciprocity.set_yticks([0, 1e-9, 2e-9, 3e-9, 4e-9], ["0", "1", "2", "3", "4"])
    cross_ratio.set_ylabel(r"$|R_{12}R_{21}\,/\,(R_{11}R_{22})|$")
    cross_ratio.set_ylim(0, 5.0e-3)
    cross_ratio.set_yticks([0, 1e-3, 2e-3, 3e-3, 4e-3, 5e-3],
                          ["0", "0.001", "0.002", "0.003", "0.004", "0.005"])
    for mesh_index, values in enumerate(diagnostics):
        _annotate(powers, mesh_index, values["mean_power"], _scientific_label(values["mean_power"]))
        residual = values["reciprocity_relative_difference"]
        ratio = values["cross_ratio_abs"]
        for ax, value, label in ((reciprocity, residual, _scientific_label(residual)),
                                 (cross_ratio, ratio, f"{ratio:.6g}")):
            ax.plot(mesh_index, value, marker="o", linestyle="none", markersize=5,
                    color=DIAGNOSTIC_COLOR, zorder=4)
            _annotate(ax, mesh_index, value, label)

    sensitivity.set_ylabel(r"$|\Delta P|\,/\,P_{\mathrm{finer}}$ (%)")
    sensitivity.set_yscale("log")
    sensitivity.set_ylim(5.7, 3700)
    sensitivity.set_yticks([10, 100, 1000], ["10", "100", "1,000"])
    sensitivity.set_xlim(-.48, 1.48)
    sensitivity.set_xticks([0, 1], ["100 → 80 nm\n4 layers fixed", "4 → 6 layers\n80 nm fixed"])
    target = rows[0]["relative_absolute_power_increment_target"] * 100
    sensitivity.axhline(target, color="#667480", linestyle=(0, (4, 2)), linewidth=.9)
    sensitivity.text(.5, 13.1, f"{target:g}% target", ha="center", fontsize=7.6, color=MUTED)

    for index, row in enumerate(rows):
        mesh, direction = divmod(index, len(DIRECTIONS))
        x = mesh + OFFSETS[direction]
        marker = dict(marker=MARKERS[direction], color=COLORS[direction],
                      linestyle="none", markersize=5.3, zorder=4)
        powers.plot(x, row["power_fraction_887500000_hz"], **marker)
        powers.hlines(row["power_fraction_target_887500000_hz"], x - .068, x + .068,
                      colors=COLORS[direction], linestyles=(0, (2.5, 1.5)), linewidth=1.05)
        if row["relative_absolute_power_increment"] != "":
            change = row["relative_absolute_power_increment"] * 100
            sensitivity.plot(x, change, **marker)
            if direction == 0:
                _annotate(sensitivity, mesh, change, f"{change:,.2f}%")

    fig.text(left, .097, "The directions agree on each mesh. Power varies nonmonotonically between meshes.",
             fontsize=9.5, fontweight="bold", color=INK)
    fig.text(left, .070,
             "Power labels in (a) are reciprocal means. h is the global in-plane target (nm). z counts quadratic layers through 200 nm.",
             fontsize=7.7, color=MUTED)
    fig.text(left, .048,
             "Both diagnostics in (b) and (c) are dimensionless. Block indices 1 and 2 correspond to modes 1 and 5.",
             fontsize=7.7, color=MUTED)
    fig.text(left, .026,
             "The ratio cancels independent complex row and column factors. Mode mixing and the origin of the error remain unresolved.",
             fontsize=7.7, color=MUTED)
    return fig


def render(output_dir: Path) -> tuple[Path, Path, Path, Path]:
    """Write the vector PDF, PNG preview and both numerical tables."""
    rows = _load_rows()
    diagnostics = load_diagnostics(ROOT)
    for diagnostic, forward, reverse in zip(diagnostics, rows[::2], rows[1::2], strict=True):
        if (diagnostic["mesh"] != forward["mesh"]
                or diagnostic["power_1_to_5"] != forward["power_fraction_887500000_hz"]
                or diagnostic["power_5_to_1"] != reverse["power_fraction_887500000_hz"]):
            raise ValueError(f"Reciprocity and directed power records disagree for {forward['mesh']}")

    output_dir.mkdir(parents=True, exist_ok=True)
    stem = f"athul_simulation_pnc_{IDENTIFIER}_AP_pnc"
    pdf_path = output_dir / f"{stem}.pdf"
    png_path = output_dir / f"{stem}.png"
    csv_path = output_dir / f"{IDENTIFIER}_AP_pnc.csv"
    diagnostics_path = output_dir / "weak_conversion_diagnostics_8875_AP_pnc.csv"
    _write_csv(csv_path, rows)
    _write_csv(diagnostics_path, diagnostics)

    with matplotlib.rc_context({
        "font.family": "DejaVu Sans", "font.size": 8.7, "pdf.fonttype": 42,
        "axes.spines.top": False, "axes.spines.right": False, "axes.edgecolor": "#81919F",
        "axes.labelcolor": INK, "xtick.color": MUTED, "ytick.color": MUTED, "text.color": INK,
    }):
        fig = _plot_figure(rows, diagnostics)
        fig.savefig(pdf_path, metadata={"Title": TITLE, "Author": "Athul Prem", "CreationDate": None})
        fig.savefig(png_path, dpi=240, metadata={"Title": TITLE, "Author": "Athul Prem"})
    return pdf_path, csv_path, png_path, diagnostics_path


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, required=True)
    for path in render(parser.parse_args().output_dir):
        print(path.resolve())
