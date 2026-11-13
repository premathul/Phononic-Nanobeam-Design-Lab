"""Offline report for computed 3D beam mode shapes.

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
    "mode_order": "频率序号",
    "frequency_hz": "频率 (Hz)",
    "frequency_GHz": "频率 (GHz)",
    "k_rad_m": "波数 (rad/m)",
    "kL_over_pi": "kL/π",
    "relative_residual": "相对残差",
    "rigid_mode": "刚体模态",
    "kinetic_fraction_x": "x 向动能占比",
    "kinetic_fraction_y": "y 向动能占比",
    "kinetic_fraction_z": "z 向动能占比",
}

_VALUE_LABELS = {
    "clamped_free": "一端固定、一端自由 (clamped-free)",
    "free_free": "两端自由 (free-free)",
    "bloch_periodic": "Bloch 周期边界",
    "fixed_property_control": "固定材料参数对照",
    "nominal fixed-property control; no temperature calibration": "名义材料参数固定，温度效应待标定",
    "completed_unverified": "已计算，待验证",
    "verified_within_model": "已通过模型内验证",
}


def _display(value: object) -> str:
    if value is None:
        return "—"
    if isinstance(value, (float, np.floating)):
        return escape(f"{value:.8g}")
    if isinstance(value, (bool, np.bool_)):
        return "是" if value else "否"
    return escape(_VALUE_LABELS.get(str(value), str(value)))


def _result_table(rows: list[dict]) -> str:
    if not rows:
        return "<p>无结果行。</p>"
    columns = list(dict.fromkeys(column for row in rows for column in row))
    headers = "".join(f'<th scope="col">{escape(_FIELD_LABELS.get(key, key))}</th>' for key in columns)
    body = "".join("<tr>" + "".join(f"<td>{_display(row.get(key))}</td>" for key in columns) + "</tr>"
                   for row in rows)
    return ('<div class="table-scroll" tabindex="0" aria-label="可滚动的频率结果表">'
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

    Mode rows in the shape viewer follow the frequencies and vertex-mode array
    order. The frequency table is independent and may contain other k samples.
    """
    frequencies = np.asarray(frequencies_hz, dtype=float)
    modes = np.asarray(vertex_modes)
    points = mesh.p.T
    if frequencies.ndim != 1 or len(frequencies) == 0 or not np.all(np.isfinite(frequencies)):
        raise ValueError("report requires a nonempty vector of finite frequencies")
    if modes.shape != (len(frequencies), len(points), 3) or not np.all(np.isfinite(modes)):
        raise ValueError("vertex_modes must contain finite xyz displacements for each mode and vertex")
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
    # Remove the units of each mass-normalized mode while retaining the
    # relative displacement and phase at every displayed vertex.
    amplitudes = np.max(np.linalg.norm(surface_modes, axis=2), axis=1)
    amplitudes[amplitudes == 0] = 1.0
    relative_modes = surface_modes / amplitudes[:, None, None]
    surface_points = points[surface_vertices]
    low, high = surface_points.min(axis=0), surface_points.max(axis=0)
    span = float((high - low).max())
    if span <= 0 or not np.isfinite(span):
        raise ValueError("report requires a nondegenerate mesh extent")

    bloch = summary["analysis_kind"] == "bloch"
    complex_modes = bloch and np.iscomplexobj(relative_modes)
    viewer_data = {
        "points": np.round((surface_points - (low + high) / 2) / span, 9).tolist(),
        "edges": edges.tolist(),
        "real": np.round(relative_modes.real, 7).tolist(),
        "imag": np.round(relative_modes.imag, 7).tolist() if complex_modes else None,
    }
    # Script-safe JSON even if a future version includes text in the payload.
    payload = json.dumps(viewer_data, ensure_ascii=True, allow_nan=False, separators=(",", ":"))
    payload = payload.replace("&", "\\u0026").replace("<", "\\u003c").replace(">", "\\u003e")
    options = "".join(f'<option value="{index}">模态 {index + 1} · {frequency / 1e9:.7g} GHz</option>'
                      for index, frequency in enumerate(frequencies))

    geometry = summary.get("geometry_nm", {})
    dimension = " × ".join(_display(geometry.get(axis)) for axis in ("length_nm", "width_nm", "thickness_nm"))
    statistics = {
        "长 × 宽 × 厚 (nm)": dimension,
        "边界条件": _display(summary.get("boundary")),
        "自由度": _display(summary.get("ndof")),
        "耗时 (s)": _display(summary.get("elapsed_seconds")),
        "进程峰值内存 (MiB)": _display(summary.get("peak_process_memory_mib")),
        "材料状态": _display(summary.get("material_status")),
        "计算状态": _display(summary.get("status")),
    }
    statistics_html = "".join(f"<div><dt>{label}</dt><dd>{value}</dd></div>"
                              for label, value in statistics.items())
    if bloch:
        title = "直梁 Bloch 模态"
        k = summary.get("field_k_rad_m")
        sample_note = (f"显示波数 k = {_display(k)} rad/m 的 Bloch 模态形状。"
                       if k is not None else "显示一个 Bloch 波数样本的模态形状。")
        if complex_modes:
            sample_note += " 相位滑块显示 Re(u e⁻ⁱᵠ)，初始相位 φ = 0°。"
        table_note = "表中列出所有已计算波数。每个波数内独立按频率排序，跨波数的模态身份尚未追踪。"
    else:
        title = "有限直梁本征模态"
        sample_note = "显示给定端部边界条件下的有限梁本征模态。"
        table_note = ""
    phase_control = ('<label>相位 <output id="phase-value">0°</output>'
                     '<input id="phase" type="range" min="0" max="360" value="0" step="5"></label>'
                     if complex_modes else "")
    html = _PAGE.substitute(title=title, statistics=statistics_html, sample_note=sample_note,
                            phase_control=phase_control, options=options, payload=payload,
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
<html lang="zh-CN">
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
    <p class="author">Athul Prem | athul_simulation_pnc_mode_shapes</p>
    <h1>$title</h1>
    <p class="scope">三维弹性有限元结果。透射率与反射率 (T/R) 尚未校准。</p>
  </header>
  <section class="card" aria-label="计算参数">
    <dl class="stats">$statistics</dl>
  </section>
  <section class="card" aria-labelledby="shape-title">
    <h2 id="shape-title">模态形状</h2>
    <div class="controls">
      <label>选择模态 <select id="mode">$options</select></label>
      <label>视角 <select id="view"><option value="iso">三维</option><option value="top">俯视 x–y</option>
        <option value="side">侧视 x–z</option><option value="end">端视 y–z</option>
        <option value="custom" disabled>手动旋转</option></select></label>
      <label>显示放大 <output id="scale-value">1.0×</output>
        <input id="scale" type="range" min="0" max="3" value="1" step="0.1"></label>
      $phase_control
    </div>
    <canvas id="beam" tabindex="0" aria-label="三维表面线框图。拖动或使用方向键旋转。">三维表面线框图。</canvas>
    <div class="legend"><span class="base">原几何</span><span class="shape">相对形状</span></div>
    <p class="note">$sample_note 拖动或按方向键旋转。位移来自质量归一化本征向量，按相对形状放大显示，实际受驱位移幅值未知。</p>
    <noscript><p>模态图需要启用 JavaScript。频率表可直接阅读。</p></noscript>
  </section>
  <section class="card" aria-labelledby="table-title">
    <h2 id="table-title">频率结果</h2>
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
