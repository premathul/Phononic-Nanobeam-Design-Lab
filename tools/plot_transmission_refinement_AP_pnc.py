"""Plot saved 1 GHz regional mesh comparisons. Author: Athul Prem | _AP_pnc."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
from pathlib import Path

import matplotlib as mpl
from matplotlib.figure import Figure
from matplotlib.ticker import FixedLocator, FuncFormatter


ROOT = Path(__file__).resolve().parents[1]
IDENTIFIER = "transmission_refinement_1GHz"
LEAD_RECORD = Path("verification/regional_lead_1GHz_AP_pnc.json")
LAYER_RECORD = Path("verification/regional_thickness_1GHz_AP_pnc.json")
COLORS = ("#176B87", "#B65324")
MARKERS = ("o", "D")


def _read_record(path: Path) -> tuple[dict, str]:
    content = (ROOT / path).read_bytes()
    return json.loads(content), hashlib.sha256(content).hexdigest()


def _comparison_rows() -> list[dict]:
    lead, lead_hash = _read_record(LEAD_RECORD)
    layer, layer_hash = _read_record(LAYER_RECORD)
    preset, preset_hash = _read_record(Path(lead["preset"]["path"]))
    lead_geometry = lead["geometry_control"]
    lead_mesh = lead_geometry["lead_targets_nm"]
    lead_layers = lead_geometry["thickness_layers"]
    refined_layers = layer["candidate_additional_overrides"]["mesh.thickness_layers"]
    thickness = layer["physical_thickness_nm"]
    if (
        lead["preset"] != layer["preset"]
        or preset_hash != lead["preset"]["sha256"]
        or (lead["frequency_GHz"], layer["frequency_GHz"]) != (1, 1)
        or (lead["incidence"], layer["incident_side"]) != ("left", "left")
        or lead["reference_planes_x_m"] != layer["reference_planes_x_m"]
        or lead_geometry["hole_count"] != layer["identical_footprint"]["holes"]
        or preset["geometry"]["thickness_nm"] != thickness
        or preset["mesh"]["thickness_layers"] != lead_layers
        or layer["baseline_overrides"]["mesh.lead_mesh_size_nm"] != lead_mesh[1]
    ):
        raise ValueError("The saved records no longer describe the same 1 GHz control geometry.")

    lead_pair = lead["comparison"]["maximum_complex_change_pair"]
    layer_pair = layer["maximum_complex_pair"]
    lead_roots = [lead_pair[side]["port_index"] for side in ("incident", "outgoing")]
    cases = (
        ("lead_only", LEAD_RECORD, lead_hash, lead["comparison"], lead_pair["block"],
         lead_mesh, [lead_layers] * 2,
         [lead_pair[side]["channel"] for side in ("incident", "outgoing")],
         lead_roots, lead_roots, lead_pair["reference_amplitude"]["magnitude"],
         lead_pair["candidate_aligned_amplitude"]["magnitude"],
         lead_pair["wrapped_phase_change_rad"]),
        ("thickness_layers_and_port", LAYER_RECORD, layer_hash, layer["comparison"],
         layer_pair["block"], [lead_mesh[1]] * 2, [lead_layers, refined_layers],
         [layer_pair["incident_channel"], layer_pair["outgoing_channel"]],
         layer_pair["one_layer_roots"], layer_pair["two_layer_roots"],
         layer_pair["one_layer_magnitude"], layer_pair["two_layer_magnitude"],
         layer_pair["wrapped_phase_change_rad"]),
    )
    rows = []
    for (name, source, source_hash, comparison, block, lead_targets, layer_counts,
         channels, reference_roots, candidate_roots, reference, candidate, phase) in cases:
        magnitude_change = candidate - reference
        # Chord length of the aligned complex difference from the two radii and their phase separation.
        complex_change = math.hypot(
            magnitude_change, 2 * math.sqrt(reference * candidate) * math.sin(phase / 2)
        )
        if block != "transmission" or not math.isclose(
            complex_change, comparison["maximum_complex_change"], rel_tol=1e-12
        ):
            raise ValueError(f"The selected transmission pair does not reproduce the complex maximum in {source}.")

        rows.append({
            "refinement": name,
            "frequency_GHz": lead["frequency_GHz"],
            "physical_thickness_nm": thickness,
            "central_mesh_target_nm": preset["mesh"]["mesh_size_nm"],
            "reference_lead_mesh_target_nm": lead_targets[0],
            "candidate_lead_mesh_target_nm": lead_targets[1],
            "reference_quadratic_thickness_layers": layer_counts[0],
            "candidate_quadratic_thickness_layers": layer_counts[1],
            "left_reference_plane_m": lead["reference_planes_x_m"][0],
            "right_reference_plane_m": lead["reference_planes_x_m"][1],
            "incident_side": "left",
            "outgoing_side": "right",
            "incident_channel": channels[0],
            "outgoing_channel": channels[1],
            "reference_incident_root": reference_roots[0],
            "reference_outgoing_root": reference_roots[1],
            "candidate_incident_root": candidate_roots[0],
            "candidate_outgoing_root": candidate_roots[1],
            "reference_coefficient_magnitude": reference,
            "candidate_aligned_coefficient_magnitude": candidate,
            "signed_magnitude_change": magnitude_change,
            "wrapped_phase_change_rad": phase,
            "wrapped_phase_change_deg": math.degrees(phase),
            "maximum_absolute_per_drive_power_change_fraction": comparison["maximum_power_change"],
            "power_change_limit_fraction": comparison["power_change_limit"],
            "power_passed": comparison["power_passed"],
            "maximum_complex_coefficient_change": comparison["maximum_complex_change"],
            "complex_coefficient_change_limit": comparison["complex_change_limit"],
            "complex_passed": comparison["complex_passed"],
            "source_path": source.as_posix(),
            "source_sha256": source_hash,
            "preset_path": lead["preset"]["path"],
            "preset_sha256": preset_hash,
        })
    return rows


def _signed(value: float, precision: int) -> str:
    return f"{value:+.{precision}f}".replace("-", "−")


def _point(axis, index, value, label, color, marker, offset=7):
    axis.plot(index, value, marker=marker, color=color, markersize=5.7, zorder=3)
    axis.annotate(label, (index, value), xytext=(0, offset), textcoords="offset points",
                  ha="center", va="bottom" if offset > 0 else "top", fontsize=8.6, color=color)


def _gate(axis, value, label):
    axis.axhline(value, color="#707985", linestyle=(0, (4, 2)), linewidth=0.9)
    axis.annotate(label, (0.97, value), xycoords=axis.get_yaxis_transform(),
                  xytext=(0, 4), textcoords="offset points", ha="right", fontsize=7.6,
                  color="#4B5563")


def _plot(rows, output_path):
    lead, layer = rows
    power_limit_pp = lead["power_change_limit_fraction"] * 100
    complex_limit = lead["complex_coefficient_change_limit"]
    if (power_limit_pp != layer["power_change_limit_fraction"] * 100
            or complex_limit != layer["complex_coefficient_change_limit"]):
        raise ValueError("The figure requires common power and complex limits for the two comparisons.")

    with mpl.rc_context({"font.family": "DejaVu Sans", "font.size": 9,
                         "pdf.fonttype": 42, "axes.unicode_minus": True,
                         "text.color": "#161E2A", "axes.labelcolor": "#161E2A",
                         "xtick.color": "#323C4A", "ytick.color": "#323C4A"}):
        figure = Figure(figsize=(7.3, 5.95), facecolor="white")
        grid = figure.add_gridspec(2, 2, left=0.122, right=0.957, top=0.736, bottom=0.138,
                                   hspace=0.81, wspace=0.35)
        magnitude, phase, power, complex_change = (figure.add_subplot(cell) for cell in grid)

        figure.text(0.095, 0.966, f"Athul Prem | athul_simulation_pnc_{IDENTIFIER}",
                    fontsize=8, color="#4B5563", va="top")
        figure.text(0.095, 0.927, "Transmission mesh sensitivity at 1 GHz", fontsize=16,
                    weight="bold", va="top")
        figure.text(0.095, 0.877,
                    f"Straight 26-hole Ge device · {lead['physical_thickness_nm']} nm physical thickness"
                    f" · {lead['central_mesh_target_nm']} nm central mesh target", fontsize=9.4, va="top")

        lead_from, lead_to = (lead[f"{stage}_lead_mesh_target_nm"] for stage in ("reference", "candidate"))
        layer_from, layer_to = (layer[f"{stage}_quadratic_thickness_layers"]
                                for stage in ("reference", "candidate"))
        figure.text(0.095, 0.826, f"●  Lead only · {lead_from} → {lead_to} nm", fontsize=9.2,
                    color=COLORS[0], weight="bold", va="top")
        figure.text(0.55, 0.826, f"◆  Thickness + port · {layer_from} → {layer_to} layers", fontsize=9.2,
                    color=COLORS[1], weight="bold", va="top")
        for x, row, fixed_mesh in zip((0.095, 0.55), rows,
                                     (f"{layer_from} quadratic layer", f"{lead_to} nm lead target"), strict=True):
            figure.text(x, 0.796,
                        f"{fixed_mesh} · selected T {row['incident_channel']} → {row['outgoing_channel']}",
                        fontsize=8.4, va="top")

        panels = (
            (magnitude, "A  Transmission magnitude", "Δ|S| (×10⁻³)"),
            (phase, "B  Transmission phase", "Wrapped Δphase (°)"),
            (power, "C  Maximum per-drive power change", "Max |ΔT| or |ΔR| (pp)"),
            (complex_change, "D  Maximum complex change", "Max |ΔS| (dimensionless)"),
        )
        for axis, title, ylabel in panels:
            axis.set_title(title, loc="left", fontsize=9.5, weight="bold", pad=7)
            axis.set_ylabel(ylabel, fontsize=8.5, labelpad=7)
            axis.set_xlim(-0.46, 1.46)
            axis.set_xticks((0, 1), ("Lead only", "Layers + port"), fontsize=8.4)
            axis.spines[["top", "right"]].set_visible(False)
            axis.spines[["left", "bottom"]].set_color("#AEB8C0")
            axis.tick_params(axis="both", which="major", length=3, width=0.6, color="#AEB8C0")
            axis.set_axisbelow(True)
            axis.grid(axis="y", color="#E2E8ED", linewidth=0.65)

        magnitude.axhline(0, color="#707985", linewidth=0.8)
        magnitude.set(ylim=(-0.49, 1.52), yticks=(-0.4, 0.0, 0.4, 0.8, 1.2))
        magnitude.set_xticks((0, 1), [
            f"{row['reference_coefficient_magnitude']:.6f} → "
            f"{row['candidate_aligned_coefficient_magnitude']:.6f}" for row in rows
        ], fontsize=7.9)
        phase.set(ylim=(0, 61), yticks=(0, 15, 30, 45, 60))
        power.set(ylim=(0, 1.24), yticks=(0, 0.25, 0.5, 0.75, 1))
        complex_change.set_yscale("log")
        complex_change.set_ylim(0.006, 2.25)
        complex_change.yaxis.set_major_locator(FixedLocator((0.01, 0.1, 1)))
        complex_change.yaxis.set_major_formatter(FuncFormatter(lambda value, _: f"{value:g}"))
        complex_change.yaxis.set_minor_locator(FixedLocator([]))
        _gate(power, power_limit_pp, f"{power_limit_pp:g} pp power limit")
        _gate(complex_change, complex_limit, f"{complex_limit:g} complex limit")

        for index, (row, color, marker) in enumerate(zip(rows, COLORS, MARKERS, strict=True)):
            delta = row["signed_magnitude_change"] * 1e3
            _point(magnitude, index, delta, _signed(delta, 4), color, marker,
                   offset=7 if delta > 0 else -7)
            angle = row["wrapped_phase_change_deg"]
            _point(phase, index, angle, f"{_signed(angle, 2)}°", color, marker)
            power_pp = row["maximum_absolute_per_drive_power_change_fraction"] * 100
            _point(power, index, power_pp, f"{power_pp:.4f}", color, marker)
            change = row["maximum_complex_coefficient_change"]
            _point(complex_change, index, change, f"{change:.5f}", color, marker)

        figure.text(0.095, 0.066,
                    "A–B show the aligned transmission coefficient with the largest complex change in each pair.",
                    fontsize=7.8, va="top")
        figure.text(0.095, 0.042,
                    "End planes at 0 and 16.65 µm · five left-incident drives · convergence unresolved · pp = percentage points",
                    fontsize=7.8, va="top")
        figure.savefig(output_path, metadata={"Author": "Athul Prem", "CreationDate": None,
                                              "Title": "Transmission mesh sensitivity at 1 GHz"})
        figure.savefig(output_path.with_suffix(".png"), dpi=240,
                       metadata={"Author": "Athul Prem", "Title": "Transmission mesh sensitivity at 1 GHz"})


def render(output_dir: Path) -> tuple[Path, Path, Path]:
    """Write the vector PDF and its plotted values as a two-row CSV."""
    rows = _comparison_rows()
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    pdf_path = output_dir / f"athul_simulation_pnc_{IDENTIFIER}_AP_pnc.pdf"
    data_path = output_dir / f"{IDENTIFIER}_AP_pnc.csv"
    _plot(rows, pdf_path)
    with data_path.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    return pdf_path, data_path, pdf_path.with_suffix(".png")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    for path in render(args.output_dir):
        print(path.resolve())


if __name__ == "__main__":
    main()
