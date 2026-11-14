"""Offline report for 3D elastic modes and scattering responses.

Author: Athul Prem | _AP_pnc
"""

from __future__ import annotations

from html import escape
import json
from pathlib import Path
from string import Template

import numpy as np
from skfem import MeshTet


_FIELD_LABELS = {
    "mode_order": "Mode order",
    "frequency_hz": "Frequency (Hz)",
    "frequency_GHz": "Frequency (GHz)",
    "k_rad_m": "Wavenumber (rad/m)",
    "kL_over_pi": "kL/π",
    "relative_residual": "Relative residual",
    "rigid_mode": "Rigid-body mode",
    "kinetic_fraction_x": "Kinetic energy fraction (x)",
    "kinetic_fraction_y": "Kinetic energy fraction (y)",
    "kinetic_fraction_z": "Kinetic energy fraction (z)",
    "incident_mode_order": "Incident channel",
    "transmission": "Transmission (T)",
    "reflection": "Reflection (R)",
    "power_balance": "Power imbalance (1 − R − T)",
    "transmission_flux": "Transmission from port flux",
    "reflection_flux": "Reflection from port flux",
    "flux_power_balance": "Power imbalance from port flux",
}

_VALUE_LABELS = {
    "clamped_free": "Clamped at one end",
    "clamped_clamped": "Clamped at both ends",
    "free": "Free at both ends",
    "free_free": "Free at both ends",
    "periodic_x": "Bloch periodicity along x",
    "bloch_periodic": "Bloch periodicity along x",
    "matched_ports": "Matched waveguide ports at both ends",
    "left": "Left end",
    "right": "Right end",
    "fixed_property_control": "Fixed-property control",
    "nominal fixed-property control; no temperature calibration": "Nominal fixed properties, no temperature calibration",
    "completed_unverified": "Computed, verification pending",
    "verified_within_model": "Verified within the model",
    "failed": "Failed",
    "interrupted": "Interrupted",
}


def _display(value: object) -> str:
    if value is None:
        return "—"
    if isinstance(value, (float, np.floating)):
        return escape(f"{value:.8g}")
    if isinstance(value, (bool, np.bool_)):
        return "Yes" if value else "No"
    return escape(_VALUE_LABELS.get(str(value), str(value)))


def _result_table(rows: list[dict]) -> str:
    if not rows:
        return "<p>No results to display.</p>"
    columns = list(dict.fromkeys(column for row in rows for column in row))
    headers = "".join(f'<th scope="col">{escape(_FIELD_LABELS.get(key, key))}</th>' for key in columns)
    body = "".join("<tr>" + "".join(f"<td>{_display(row.get(key))}</td>" for key in columns) + "</tr>"
                   for row in rows)
    return ('<div class="table-scroll" tabindex="0" aria-label="Scrollable results table">'
            f'<table><thead><tr>{headers}</tr></thead><tbody>{body}</tbody></table></div>')


def write_report(
    output: Path,
    mesh: MeshTet,
    frequencies_hz: np.ndarray,
    vertex_modes: np.ndarray,
    summary: dict,
    rows: list[dict],
) -> Path:
    """Write one self-contained report into an existing or new output directory.

    The viewer follows the frequency and vertex displacement array order. For
    scattering, each field is the harmonic response to one unit-power incident
    channel at the same drive frequency. The table may contain other k samples
    when viewing a Bloch report.
    """
    frequencies = np.asarray(frequencies_hz, dtype=float)
    modes = np.asarray(vertex_modes)
    points = mesh.p.T
    if frequencies.ndim != 1 or len(frequencies) == 0 or not np.all(np.isfinite(frequencies)):
        raise ValueError("report requires a nonempty vector of finite frequencies")
    if modes.shape != (len(frequencies), len(points), 3) or not np.all(np.isfinite(modes)):
        raise ValueError("vertex_modes must contain finite xyz displacements for each field and vertex")
    if not len(points) or not np.all(np.isfinite(points)):
        raise ValueError("report requires nonempty finite mesh coordinates")

    # Retain boundary vertices only. Interior tetrahedral edges would obscure
    # the surface and needlessly enlarge the offline report.
    surface = mesh.facets[:, mesh.boundary_facets()]
    surface_vertices, local_faces = np.unique(surface, return_inverse=True)
    faces = local_faces.reshape(surface.shape).T
    edges = np.concatenate([faces[:, [0, 1]], faces[:, [1, 2]], faces[:, [2, 0]]])
    edges = np.unique(np.sort(edges, axis=1), axis=0)
    surface_modes = modes[:, surface_vertices]
    # Rescale each field for viewing while preserving relative displacement and
    # phase across its vertices.
    amplitudes = np.max(np.linalg.norm(surface_modes, axis=2), axis=1)
    amplitudes[amplitudes == 0] = 1.0
    relative_modes = surface_modes / amplitudes[:, None, None]
    surface_points = points[surface_vertices]
    low, high = surface_points.min(axis=0), surface_points.max(axis=0)
    span = float((high - low).max())
    if span <= 0 or not np.isfinite(span):
        raise ValueError("report requires a nondegenerate mesh extent")

    analysis_kind = summary["analysis_kind"]
    bloch = analysis_kind == "bloch"
    scattering = analysis_kind == "scattering"
    complex_fields = (bloch or scattering) and np.iscomplexobj(relative_modes)
    viewer_data = {
        "points": np.round((surface_points - (low + high) / 2) / span, 9).tolist(),
        "edges": edges.tolist(),
        "real": np.round(relative_modes.real, 7).tolist(),
        "imag": np.round(relative_modes.imag, 7).tolist() if complex_fields else None,
    }
    # Script-safe JSON even if a future version includes text in the payload.
    payload = json.dumps(viewer_data, ensure_ascii=True, allow_nan=False, separators=(",", ":"))
    payload = payload.replace("&", "\\u0026").replace("<", "\\u003c").replace(">", "\\u003e")
    options = "".join(f'<option value="{index}">Mode {index + 1} · {frequency / 1e9:.7g} GHz</option>'
                      for index, frequency in enumerate(frequencies))
    scope = "3D elastic finite-element results. Transmission and reflection require a separate scattering calculation."
    figure_id = "mode_shapes"
    shape_title = "Mode shape"
    selection_label = "Mode"
    table_title = "Frequencies"
    amplitude_note = "The viewer scales mass-normalized eigenvectors to show relative displacement."

    geometry = summary.get("geometry_nm", {})
    dimension = " × ".join(_display(geometry.get(axis)) for axis in ("length_nm", "width_nm", "thickness_nm"))
    statistics = {
        "Length × width × thickness (nm)": dimension,
        "Boundary conditions": _display(summary.get("boundary")),
        "Degrees of freedom": _display(summary.get("ndof")),
        ("Cached export time (s)" if "checkpointed_scattering" in summary else "Elapsed time (s)"):
            _display(summary.get("elapsed_seconds")),
        ("Export process memory (MiB)" if "checkpointed_scattering" in summary else "Peak process memory (MiB)"):
            _display(summary.get("peak_process_memory_mib")),
        "Material model": _display(summary.get("material_status")),
        "Run status": _display(summary.get("status")),
    }
    hole_count = summary.get("hole_count", 0)
    diagnostics = summary.get("geometry_diagnostics")
    if diagnostics:
        statistics["Through-holes"] = _display(hole_count)
        statistics["Relative solid volume error"] = _display(diagnostics.get("relative_solid_volume_error"))
        if hole_count:
            statistics["Relative removed volume error"] = _display(diagnostics.get("relative_removed_volume_error"))
            pattern = summary.get("hole_pattern_provenance", {})
            statistics["Hole pattern source"] = _display(pattern.get("source_csv"))
    if scattering:
        incident_indices = summary["incident_mode_indices"]
        if len(incident_indices) != len(frequencies):
            raise ValueError("incident_mode_indices must identify every scattering field")
        options = "".join(
            f'<option value="{index}">Channel {index + 1} · Port mode index {_display(mode_id)}</option>'
            for index, mode_id in enumerate(incident_indices)
        )
        title = "Perforated-beam scattering with matched ports" if hole_count else "Straight-beam scattering with matched ports"
        scope = "3D lossless elasticity with fixed material properties. Temperature effects and physical dissipation remain uncalibrated."
        figure_id = "scattering_response"
        shape_title = "Steady-state response"
        selection_label = "Incident channel"
        table_title = "Transmission, reflection, and power balance"
        sample_note = ("Each incident channel uses a mathematical normalization of 1 W. "
                       "Port mode indices start at 0.")
        amplitude_note = ("The viewer scales each response separately to show relative displacement. "
                          "The physical drive and displacement amplitude depend on the device and excitation.")
        table_note = ("T and R give transmitted and reflected power as fractions of incident power. "
                      "Port flux uses displacement and force at the end faces as an independent check. "
                      "Power imbalance should approach zero for this lossless model.")
        frequency_ghz = summary.get("frequency_hz", frequencies[0]) / 1e9
        condition = summary.get("port_basis_condition")
        statistics.update({
            "Drive frequency (GHz)": _display(frequency_ghz),
            "Incident side": _display(summary.get("incident_side")),
            "Incident channels": _display(summary.get("num_incident_channels")),
            "Maximum absolute power imbalance": _display(summary.get("max_power_imbalance")),
            "Maximum reflection": _display(summary.get("maximum_reflection")),
            "Port basis condition numbers (left / right)": (
                " / ".join(_display(value) for value in condition) if condition is not None else "—"
            ),
        })
        if "power_balance_passed" in summary:
            statistics["Power balance check passed"] = _display(summary["power_balance_passed"])
        if hole_count:
            scope += (" The holes scatter and convert guided waves. Transmission need not approach one. "
                      "A closed power budget alone does not establish mesh convergence or a band gap.")
            table_note += " The hole walls are traction free. No intrinsic absorption is included."
        else:
            statistics["Matched-guide power check passed"] = _display(summary.get("matched_reference_passed"))
            if summary.get("matched_reference_passed"):
                scope += " This case passed the matched-guide power check. Phase accuracy and mesh convergence require separate verification."
            else:
                scope += " This case has yet to pass the matched-guide power check."
    elif bloch:
        title = "Perforated-beam Bloch modes" if hole_count else "Straight-beam Bloch modes"
        k = summary.get("field_k_rad_m")
        sample_note = (f"The viewer shows Bloch modes at k = {_display(k)} rad/m."
                       if k is not None else "The viewer shows Bloch modes at one sampled wavenumber.")
        table_note = ("The table includes all computed wavenumbers. Each wavenumber has its own frequency order. "
                      "Tracking mode identity across wavenumbers requires a separate analysis.")
    else:
        title = "Finite perforated-beam eigenmodes" if hole_count else "Finite straight-beam eigenmodes"
        sample_note = "The modes reflect the specified end boundary conditions."
        table_note = ""
    if complex_fields:
        sample_note += " The phase slider displays Re(u e⁻ⁱᵠ), starting at φ = 0°."
    statistics_html = "".join(f"<div><dt>{label}</dt><dd>{value}</dd></div>"
                              for label, value in statistics.items())
    phase_control = ('<label>Phase <output id="phase-value">0°</output>'
                     '<input id="phase" type="range" min="0" max="360" value="0" step="5"></label>'
                     if complex_fields else "")
    html = _PAGE.substitute(title=title, statistics=statistics_html, sample_note=sample_note,
                            phase_control=phase_control, options=options, payload=payload,
                            scope=scope, figure_id=figure_id, shape_title=shape_title,
                            selection_label=selection_label, amplitude_note=amplitude_note,
                            table_title=table_title,
                            table_note=f'<p class="note">{table_note}</p>' if table_note else "",
                            table=_result_table(rows))
    output = Path(output)
    output.mkdir(parents=True, exist_ok=True)
    report = output / "report_AP_pnc.html"
    temporary = report.with_suffix(".html.tmp")
    temporary.write_text(html, encoding="utf-8")
    temporary.replace(report)
    return report


_PAGE = Template(r"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>$title | Athul Prem</title>
<style>
  :root { color-scheme: light; --ink: #17243a; --muted: #43536a; --line: #d5dfeb;
    --accent: #116461; --canvas: #112037; }
  * { box-sizing: border-box; }
  body { margin: 0; background: #edf2f8; color: var(--ink); line-height: 1.55;
    font-family: "Microsoft YaHei UI", "PingFang SC", "Noto Sans CJK SC", system-ui, sans-serif; }
  main { width: min(1120px, calc(100% - 36px)); margin: 36px auto 48px; }
  h1, h2, p, dl { margin: 0; }
  h1 { font-size: clamp(1.65rem, 4vw, 2.6rem); margin: 5px 0 8px; }
  h2 { font-size: 1.22rem; }
  .author { color: var(--accent); font-size: .85rem; font-weight: 650; overflow-wrap: anywhere; }
  .scope { color: var(--muted); }
  .card { background: white; border: 1px solid var(--line); border-radius: 12px;
    margin-top: 18px; padding: 20px; }
  .stats { display: grid; grid-template-columns: repeat(auto-fit, minmax(175px, 1fr)); gap: 14px 20px; }
  dt { color: var(--muted); font-size: .82rem; }
  dd { margin: 3px 0 0; font-weight: 650; overflow-wrap: anywhere; }
  .controls { display: flex; flex-wrap: wrap; align-items: end; gap: 12px 18px; margin: 14px 0; }
  label { color: var(--muted); font-size: .85rem; }
  label select, label input { display: block; margin-top: 4px; }
  select, button { padding: 6px 8px; color: var(--ink); border: 1px solid #8395ac;
    background: white; border-radius: 5px; font: inherit; }
  input[type="range"] { width: 130px; accent-color: var(--accent); min-height: 24px; }
  :focus-visible { outline: 2px solid #167c78; outline-offset: 3px; }
  canvas { display: block; width: 100%; height: clamp(270px, 48vw, 440px); background: var(--canvas);
    border-radius: 8px; cursor: grab; touch-action: none; }
  canvas:active { cursor: grabbing; }
  .legend { display: flex; flex-wrap: wrap; gap: 6px 20px; margin: 10px 0 5px;
    font-size: .85rem; color: var(--muted); }
  .legend span::before { content: ""; display: inline-block; width: 18px; height: 3px;
    vertical-align: middle; border-radius: 2px; margin-right: 6px; }
  .base::before { background: #7487a4; }
  .shape::before { background: #188278; }
  .note { color: var(--muted); font-size: .86rem; }
  .table-scroll { margin-top: 10px; overflow: auto; max-height: 500px; }
  table { width: 100%; border-collapse: collapse; font-size: .85rem; font-variant-numeric: tabular-nums; }
  th, td { text-align: left; padding: 8px 10px; border-bottom: 1px solid var(--line); }
  th { position: sticky; top: 0; background: #e9f1f7; white-space: nowrap; }
  tbody tr:nth-child(even) { background: #f7fafc; }
  @media (max-width: 530px) { main { width: calc(100% - 20px); margin-top: 20px; }
    .card { padding: 13px; } .stats { grid-template-columns: 1fr 1fr; } }
</style>
</head>
<body>
<main>
  <header>
    <p class="author">Athul Prem | athul_simulation_pnc_$figure_id</p>
    <h1>$title</h1>
    <p class="scope">$scope</p>
  </header>
  <section class="card" aria-label="Simulation parameters">
    <dl class="stats">$statistics</dl>
  </section>
  <section class="card" aria-labelledby="shape-title">
    <h2 id="shape-title">$shape_title</h2>
    <div class="controls">
      <label>$selection_label <select id="mode">$options</select></label>
      <label>View <select id="view"><option value="iso">3D</option><option value="top">Top (x–y)</option>
        <option value="side">Side (x–z)</option><option value="end">End (y–z)</option>
        <option value="custom" disabled>Manual rotation</option></select></label>
      <label>Display scale <output id="scale-value">1.0×</output>
        <input id="scale" type="range" min="0" max="3" value="1" step="0.1"></label>
      $phase_control
    </div>
    <canvas id="beam" tabindex="0" aria-label="3D surface wireframe. Drag or use the arrow keys to rotate.">3D surface wireframe.</canvas>
    <div class="legend"><span class="base">Undeformed geometry</span><span class="shape">Relative displacement</span></div>
    <p class="note">$sample_note Drag or use the arrow keys to rotate. $amplitude_note</p>
    <noscript><p>Enable JavaScript to use the interactive view. The results table is available without it.</p></noscript>
  </section>
  <section class="card" aria-labelledby="table-title">
    <h2 id="table-title">$table_title</h2>
    $table_note
    $table
  </section>
</main>
<script id="viewer-data" type="application/json">$payload</script>
<script>
"use strict";
const data = JSON.parse(document.getElementById("viewer-data").textContent);
const canvas = document.getElementById("beam");
const context = canvas.getContext("2d");
const mode = document.getElementById("mode");
const view = document.getElementById("view");
const scale = document.getElementById("scale");
const phase = document.getElementById("phase");
const camera = { yaw: 0.42, pitch: 0.56 };
const presets = { iso: [0.42, 0.56], top: [0, Math.PI / 2], side: [0, 0], end: [Math.PI / 2, 0] };
let pointer = null;

function draw() {
  if (!context) return;
  const {width, height} = canvas.getBoundingClientRect();
  const density = window.devicePixelRatio || 1;
  canvas.width = Math.round(width * density);
  canvas.height = Math.round(height * density);
  context.setTransform(density, 0, 0, density, 0, 0);
  const cy = Math.cos(camera.yaw), sy = Math.sin(camera.yaw);
  const cp = Math.cos(camera.pitch), sp = Math.sin(camera.pitch);
  const phaseDeg = phase ? Number(phase.value) : 0;
  const phaseRad = phaseDeg * Math.PI / 180;
  const phaseCos = Math.cos(phaseRad), phaseSin = Math.sin(phaseRad);
  const real = data.real[Number(mode.value)];
  const imag = data.imag && data.imag[Number(mode.value)];
  const magnification = 0.14 * Number(scale.value);
  document.getElementById("scale-value").value = Number(scale.value).toFixed(1) + "×";
  if (phase) document.getElementById("phase-value").value = phaseDeg + "°";

  function project([x, y, z]) {
    return [cy * x - sy * y, cp * z - sp * (sy * x + cy * y)];
  }
  const undeformed = data.points.map(project);
  const deformed = data.points.map((point, index) => project(point.map((value, axis) => {
    const re = real[index][axis] * phaseCos;
    const im = imag ? imag[index][axis] * phaseSin : 0;
    return value + magnification * (re + im);
  })));
  let minX = Infinity, maxX = -Infinity, minY = Infinity, maxY = -Infinity;
  for (const nodes of [undeformed, deformed]) {
    for (const [x, y] of nodes) {
      minX = Math.min(minX, x); maxX = Math.max(maxX, x);
      minY = Math.min(minY, y); maxY = Math.max(maxY, y);
    }
  }
  const padding = 52;
  const zoom = Math.min((width - padding) / (maxX - minX), (height - padding) / (maxY - minY));
  const centerX = (minX + maxX) / 2;
  const centerY = (minY + maxY) / 2;

  function drawEdges(nodes, color) {
    const xy = nodes.map(([x, y]) => [width / 2 + (x - centerX) * zoom,
      height / 2 - (y - centerY) * zoom]);
    context.beginPath();
    for (const [i, j] of data.edges) {
      context.moveTo(xy[i][0], xy[i][1]);
      context.lineTo(xy[j][0], xy[j][1]);
    }
    context.strokeStyle = color;
    context.lineWidth = 0.85;
    context.stroke();
  }

  drawEdges(undeformed, "rgba(159,181,211,0.39)");
  if (magnification > 0) drawEdges(deformed, "rgba(100,238,205,0.83)");

  context.font = "12px system-ui";
  for (const [axis, vector, color] of [["x", [1,0,0], "#ffc389"], ["y", [0,1,0], "#7fe4be"],
    ["z", [0,0,1], "#a8c2ff"]]) {
    const [dx, dy] = project(vector);
    if (Math.hypot(dx, dy) < 0.1) continue;
    context.strokeStyle = context.fillStyle = color;
    context.beginPath();
    context.moveTo(42, height - 39);
    context.lineTo(42 + 19 * dx, height - 39 - 19 * dy);
    context.stroke();
    context.fillText(axis, 39 + 27 * dx, height - 35 - 27 * dy);
  }
}

for (const control of [mode, scale, phase]) {
  if (control) control.addEventListener("input", draw);
}
view.addEventListener("change", () => {
  [camera.yaw, camera.pitch] = presets[view.value];
  draw();
});
function rotate(dx, dy) {
  camera.yaw += dx;
  camera.pitch = Math.max(-Math.PI / 2, Math.min(Math.PI / 2, camera.pitch + dy));
  view.value = "custom";
  draw();
}
canvas.addEventListener("pointerdown", event => {
  canvas.setPointerCapture(event.pointerId);
  pointer = [event.clientX, event.clientY];
});
canvas.addEventListener("pointermove", event => {
  if (!pointer) return;
  rotate((event.clientX - pointer[0]) * 0.01, (event.clientY - pointer[1]) * 0.01);
  pointer = [event.clientX, event.clientY];
});
canvas.addEventListener("lostpointercapture", () => { pointer = null; });
canvas.addEventListener("keydown", event => {
  const delta = { ArrowLeft: [-0.12, 0], ArrowRight: [0.12, 0], ArrowUp: [0, 0.12], ArrowDown: [0, -0.12] };
  if (!(event.key in delta)) return;
  event.preventDefault();
  rotate(...delta[event.key]);
});
window.addEventListener("resize", draw);
draw();
</script>
</body>
</html>
""")
