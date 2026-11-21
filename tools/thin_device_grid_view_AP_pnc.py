"""Offline selector for saved thin-device responses. Author: Athul Prem | _AP_pnc."""

from __future__ import annotations

from collections.abc import Callable
from html import escape
import json
from pathlib import Path


_MESH_TRACES = (
    ("#176B87", '<circle r="4.5"/>'),
    ("#B65324", '<rect x="-4.3" y="-4.3" width="8.6" height="8.6"/>'),
    ("#6B4C9A", '<path d="M 0,-5.5 L 5.3,4.4 H -5.3 Z"/>'),
)


def _transmission_panel(drive, frequencies, traces):
    """Plot the three mesh traces on shared frequency and power scales."""
    left, right, top, bottom = 58, 372, 17, 192
    start, end = frequencies[0], frequencies[-1]
    x = {frequency: left + (right - left) * (frequency - start) / (end - start) for frequency in frequencies}

    def y(percent):
        return bottom - (bottom - top) * percent / 100

    ticks = []
    for percent in range(0, 101, 20):
        height = y(percent)
        ticks.append(f'<path class="thin-grid-plot-guide" d="M {left} {height:g} H {right}"/>'
                     f'<text x="{left - 9}" y="{height:g}" dy=".35em" text-anchor="end">{percent}</text>')
    for frequency, position in x.items():
        ticks.append(f'<path class="thin-grid-plot-axis" d="M {position:g} {bottom} v 5"/>'
                     f'<text x="{position:g}" y="{bottom + 22}" text-anchor="middle">{frequency / 1e6:g}</text>')
    curves = []
    for mesh, color, marker, samples in traces:
        coordinates = [(x[frequency], y(100 * sample["transmission"]))
                       for frequency, sample in zip(frequencies, samples, strict=True)]
        line = " ".join(f"{position:.3f},{height:.3f}" for position, height in coordinates)
        points = "".join(
            f'<g transform="translate({position:.3f} {height:.3f})">{marker}</g>'
            for position, height in coordinates
        )
        curves.append(f'<g data-central-max-edge-nm="{mesh:.15g}" stroke="{color}" fill="white" stroke-width="2.1">'
                      f'<polyline points="{line}" fill="none"/>{points}</g>')
    title = f"thin-grid-transmission-drive-{drive}"
    return (f'<div><h4>Reference drive {drive}</h4>'
            f'<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 396 250" role="img" aria-labelledby="{title}">'
            f'<title id="{title}">Reference drive {drive} transmission on three central meshes</title>'
            f'{"".join(ticks)}<path class="thin-grid-plot-axis" d="M {left} {top} V {bottom} H {right}"/>'
            f'<text x="{(left + right) / 2:g}" y="241" text-anchor="middle">Frequency (MHz)</text>'
            '<text x="18" y="105" text-anchor="middle" transform="rotate(-90 18 105)">Transmission (%)</text>'
            f'{"".join(curves)}</svg></div>')


def _transmission_comparison(levels, frequencies):
    mesh_styles = list(zip(levels, _MESH_TRACES, strict=True))
    legend, panels, rows = [], [], []
    for level, (color, marker) in mesh_styles:
        mesh = level["central_max_edge_nm"]
        legend.append(f'<li><svg width="35" height="15" viewBox="0 0 35 15" aria-hidden="true" '
                      f'stroke="{color}" stroke-width="2.1" fill="white"><path d="M 1 7.5 H 34"/>'
                      f'<g transform="translate(17.5 7.5)">{marker}</g></svg>{mesh:.15g} nm</li>')
    for drive in (3, 5):
        traces = []
        for level, (color, marker) in mesh_styles:
            mesh = level["central_max_edge_nm"]
            samples = [next(row for row in point["drives"] if row["reference_drive"] == drive)
                       for point in level["points"]]
            traces.append((mesh, color, marker, samples))
            cells = "".join(f'<td>{100 * sample["transmission"]:.4f}<br><small>Root {sample["incident_port_index"]}</small></td>'
                            for sample in samples)
            change = 100 * (samples[-1]["transmission"] - samples[-2]["transmission"])
            rows.append(f'<tr><th scope="row">{drive}</th><td>{mesh:.15g}</td>{cells}<td>{change:+.2f}</td></tr>')
        panels.append(_transmission_panel(drive, frequencies, traces))
    headings = "".join(f'<th scope="col">{frequency / 1e6:g} MHz</th>' for frequency in frequencies)
    return ('<figure class="thin-grid-transmission-comparison" id="thin-grid-transmission-comparison" '
            'aria-labelledby="thin-grid-transmission-comparison-title">'
            '<h3 id="thin-grid-transmission-comparison-title">Transmission across three central meshes</h3>'
            '<div class="thin-grid-mesh-legend"><span>Central mesh ceiling</span>'
            f'<ul aria-label="Central mesh ceilings">{"".join(legend)}</ul></div>'
            f'<div class="thin-grid-transmission-panels">{"".join(panels)}</div>'
            '<figcaption>Both drives transmit a larger fraction of incident power at 1000 MHz than at 997.5 MHz '
            'on all three meshes. Lines connect three saved samples as visual guides.</figcaption>'
            '</figure><div class="thin-grid-table-wrap" role="region" tabindex="0" aria-labelledby="thin-grid-transmission-table-title">'
            '<table><caption id="thin-grid-transmission-table-title">Transmission (%) and frequency-local incident roots. '
            'ΔT subtracts the 997.5 MHz transmission from the 1000 MHz transmission, in percentage points (pp).</caption>'
            '<thead><tr><th scope="col">Reference drive</th><th scope="col">Mesh ceiling (nm)</th>'
            f'{headings}<th scope="col">997.5 → 1000 MHz ΔT (pp)</th></tr></thead><tbody>{"".join(rows)}</tbody></table></div>')


def _coherent_endpoint_table(changes, evidence_href):
    first = changes[0]
    lower_mhz = first["lower_frequency_hz"] / 1e6
    upper_mhz = first["upper_frequency_hz"] / 1e6
    drives = " and ".join(map(str, first["reference_drives"]))
    outgoing_rows = first["outgoing_rows_per_block"]
    rows = []
    for change in changes:
        mesh = change["central_max_edge_nm"]
        t_min, t_max = change["transmission_eigenvalue_bounds_percentage_points"]
        r_min, r_max = change["reflection_eigenvalue_bounds_percentage_points"]
        rows.append(f'<tr><th scope="row">{mesh:.15g}</th><td>{t_min:+.3f} to {t_max:+.3f}</td>'
                    f'<td>{r_min:+.3f} to {r_max:+.3f}</td></tr>')
    sign = ""
    if all(change["transmission_eigenvalue_bounds_percentage_points"][0] > 0
           and change["reflection_eigenvalue_bounds_percentage_points"][1] < 0 for change in changes):
        sign = " Transmission increases and reflection decreases for every mixture on all three saved meshes."
    evidence = ""
    if evidence_href is not None:
        evidence = f'<p><a href="{escape(evidence_href, quote=True)}" download>Coherent endpoint bounds and provenance (CSV)</a></p>'
    return ('<div class="thin-grid-table-wrap" role="region" tabindex="0" aria-labelledby="thin-grid-coherent-table-title">'
            '<table class="thin-grid-coherent-table" aria-describedby="thin-grid-coherent-description">'
            '<caption id="thin-grid-coherent-table-title">Coherent power-change bounds from '
            f'{lower_mhz:g} to {upper_mhz:g} MHz</caption>'
            '<thead><tr><th scope="col">Mesh (nm)</th><th scope="col">ΔT (pp)</th><th scope="col">ΔR (pp)</th></tr></thead>'
            f'<tbody>{"".join(rows)}</tbody></table></div>'
            '<p id="thin-grid-coherent-description">The bounds cover every fixed, unit-power coherent mixture of '
            f'reference drives {drives}. The same complex weights apply at both endpoints after matching incident-mode '
            f'phases. Each power sums over all {outgoing_rows} outgoing rows in its block.{sign} '
            'The bounds vary with the mesh. The continuum response magnitude, spectral slope and linewidth remain unresolved.</p>'
            + evidence)


def render_thin_device_grid(
    grid: dict | None,
    output: Path,
    *,
    href: Callable[[str, Path], str],
    drive_csv: str,
    coefficient_csv: str,
    hole_geometry_csv_href: str | None = None,
) -> str:
    """Render the validated comparison grid as one section of the saved report."""
    if grid is None:
        return ""
    repo_root = Path(grid["source_comparison"]).parent.parent
    levels = []
    for level in grid["levels"]:
        points = [{
            "id": f'thin100-{level["central_max_edge_nm"]:.15g}nm-{int(point["frequency_hz"])}Hz',
            "frequency_hz": point["frequency_hz"],
            "configuration_json": point["configuration_json"],
            "csv_href": href(point["source_coefficients"], output),
            "map_href": href(point["source_map"], output),
            "source_csv": Path(point["source_coefficients"]).relative_to(repo_root).as_posix(),
            "drives": point["drives"],
            "coefficients": point["coefficients"],
            "neighboring_spatial_gates": point["neighboring_spatial_gates"],
        } for point in level["points"]]
        levels.append({"central_max_edge_nm": level["central_max_edge_nm"], "points": points})
    data = json.dumps({"levels": levels, "frequencies_hz": grid["frequencies_hz"]},
                      separators=(",", ":"), allow_nan=False).replace("<", "\\u003c").replace("&", "\\u0026")
    comparison = escape(href(grid["source_comparison"], output), quote=True)
    coherent_evidence = repo_root / "figures/device_AP_pnc/thin_coherent_endpoint_bounds_AP_pnc.csv"
    coherent_evidence_href = href(str(coherent_evidence), output) if coherent_evidence.is_file() else None
    planes = " and ".join(f"{x * 1e6:g}" for x in grid["reference_planes_x_m"])
    hole_geometry_link = ""
    if hole_geometry_csv_href is not None:
        geometry_href = escape(hole_geometry_csv_href, quote=True)
        hole_geometry_link = f'\n <a id="thin-grid-export-hole-geometry" href="{geometry_href}" download>Hole geometry (CSV)</a>'
    return f'''<style>
#thin-grid .thin-grid-controls{{display:grid;gap:.75rem;grid-template-columns:repeat(auto-fit,minmax(min(100%,12rem),1fr))}}
#thin-grid .thin-grid-controls label{{display:block;margin:0;font-weight:600}}
#thin-grid select{{display:block;width:100%;margin-top:.25rem;box-sizing:border-box;color:inherit}}
#thin-grid .thin-grid-powers{{display:flex;flex-wrap:wrap;gap:.5rem 1.6rem;padding:.75rem 1rem;background:#edf4f8;border-radius:6px;font-variant-numeric:tabular-nums}}
#thin-grid .thin-grid-links{{display:flex;flex-wrap:wrap;gap:.35rem 1.1rem;margin:.65rem 0}}
#thin-grid .thin-grid-table-wrap{{overflow-x:auto}}
#thin-grid table{{min-width:34rem;font-variant-numeric:tabular-nums}}
#thin-grid .thin-grid-coherent-table{{min-width:0;max-width:32rem}}
#thin-grid .thin-grid-coherent-table :is(th,td){{padding:.4rem .45rem}}
#thin-grid caption{{font-weight:600;text-align:left;padding:.5rem 0}}
#thin-grid td{{white-space:nowrap}}
#thin-grid .thin-grid-gates{{padding-left:1.25rem}}
#thin-grid .thin-grid-gates li+li{{margin-top:.35rem}}
#thin-grid .thin-grid-transmission-comparison{{margin:1rem 0}}
#thin-grid .thin-grid-mesh-legend,#thin-grid .thin-grid-mesh-legend ul{{display:flex;align-items:center;flex-wrap:wrap;gap:.3rem .95rem}}
#thin-grid .thin-grid-mesh-legend ul{{margin:0;padding:0;list-style:none}}
#thin-grid .thin-grid-mesh-legend li{{display:flex;align-items:center;gap:.35rem;white-space:nowrap}}
#thin-grid .thin-grid-transmission-panels{{display:grid;gap:.5rem 1rem;grid-template-columns:repeat(auto-fit,minmax(min(100%,23rem),1fr))}}
#thin-grid .thin-grid-transmission-panels h4{{margin:.65rem 0 .15rem;text-align:center}}
#thin-grid .thin-grid-transmission-panels svg{{display:block;width:100%;font-family:inherit;font-size:15px;fill:currentColor}}
#thin-grid .thin-grid-plot-guide{{stroke:var(--line);fill:none}}
#thin-grid .thin-grid-plot-axis{{stroke:currentColor;fill:none}}
#thin-grid .thin-grid-transmission-comparison figcaption{{margin:.6rem 0}}
</style>
<section id="thin-grid" aria-labelledby="thin-grid-title">
<h2 id="thin-grid-title">100 nm thick device</h2>
<p>All nine results use {grid["thickness_layers"]} quadratic thickness layers, {grid["circle_resolution"]} segments per hole rim, and left incidence.</p>
<div class="thin-grid-controls">
 <label for="thin-grid-mesh">Central mesh ceiling<select id="thin-grid-mesh"></select></label>
 <label for="thin-grid-frequency">Frequency<select id="thin-grid-frequency"></select></label>
 <label for="thin-grid-drive">Reference drive<select id="thin-grid-drive" aria-describedby="thin-grid-drive-guide"></select></label>
</div>
<p id="thin-grid-drive-guide" class="muted">Drive numbers follow the 225 nm mesh at 1000 MHz. Each native root belongs to the selected sample.</p>
<p id="thin-grid-selection-notice" class="selection-notice" role="status"></p>
<div class="thin-grid-powers" aria-live="polite" aria-atomic="true">
 <span>Transmission <strong id="thin-grid-transmission"></strong></span>
 <span>Reflection <strong id="thin-grid-reflection"></strong></span>
</div>
<p>The saved solutions passed their equation and flux checks. Spatial and frequency convergence remain unresolved.</p>
<p>Original reference planes at x = {planes} µm. The amplitudes use the native unit-power bases at these planes.</p>
{_transmission_comparison(levels, grid["frequencies_hz"])}
{_coherent_endpoint_table(grid["coherent_endpoint_changes"], coherent_evidence_href)}
<div class="thin-grid-table-wrap" role="region" aria-labelledby="thin-grid-table-caption" tabindex="0">
 <table>
  <caption id="thin-grid-table-caption">Outgoing coefficients for the selected drive</caption>
  <thead><tr><th scope="col">Output</th><th scope="col">Outgoing native root</th>
   <th scope="col">Real amplitude</th><th scope="col">Imaginary amplitude</th><th scope="col">Power fraction</th></tr></thead>
  <tbody id="thin-grid-coefficients"></tbody>
 </table>
</div>
<p>Adjacent-mesh differences for the selected drive and frequency. Limits are {100 * grid["spatial_total_power_limit_fraction"]:g} percentage point for total transmission or reflection and {grid["spatial_complex_amplitude_limit"]:g} for any complex coefficient.</p>
<ul id="thin-grid-spatial-gates" class="thin-grid-gates"></ul>
<nav aria-label="Selected thin-device drive and sample" class="thin-grid-links">
 <a id="thin-grid-export-selection" download>Selected drive coefficients (CSV)</a>
 <a id="thin-grid-export-configuration" download>Selected configuration (JSON)</a>
 <a id="thin-grid-selection-link">Link to selected drive</a>
</nav>
<h3>All nine saved thin-device samples</h3>
<nav aria-label="Downloads for all saved thin-device samples" class="thin-grid-links">
 <a id="thin-grid-export-drives" href="{escape(drive_csv, quote=True)}" download>All 45 drive powers (CSV)</a>
 <a id="thin-grid-export-coefficients" href="{escape(coefficient_csv, quote=True)}" download>All 450 coefficients (CSV)</a>
</nav>
<details><summary>Sources for the selected sample</summary>
<nav class="thin-grid-links" aria-label="Native inputs and evidence">
 <a id="thin-grid-native-csv">Native source table (CSV)</a>
 <a id="thin-grid-native-map">Native map publication (JSON)</a>
 <a id="thin-grid-comparison" href="{comparison}">Verification evidence</a>{hole_geometry_link}
</nav></details>
<noscript><p>Enable JavaScript to select a sample. The full CSV exports are available above.</p></noscript>
</section>
<script type="application/json" id="thin-grid-data">{data}</script>
''' + _INTERACTION


_INTERACTION = r'''<script>
(() => {
    "use strict";
    const id = name => document.getElementById("thin-grid-" + name);
    const grid = JSON.parse(id("data").textContent);
    const mesh = id("mesh");
    const frequency = id("frequency");
    const drive = id("drive");
    const points = new Map(grid.levels.flatMap(level => level.points.map(point => [point.id, {level, point}])));
    const coefficientBody = id("coefficients");
    const gateList = id("spatial-gates");
    const selectedExport = id("export-selection");
    const configurationExport = id("export-configuration");
    const columns = [
        "central_max_edge_nm", "frequency_hz", "reference_drive", "native_drive", "incident_port_index",
        "kind", "outgoing_port_index", "amplitude_real", "amplitude_imag", "power_fraction", "source_csv"
    ];
    const percent = value => (100 * value).toFixed(4) + "%";
    const decimal = value => value === 0 ? (Object.is(value, -0) ? "−0" : "0") : value.toPrecision(10);
    const delta = value => (value > 0 ? "+" : "") + value.toFixed(3);
    const filenamePart = value => String(value).replace(".", "p");

    function option(value, label) {
        const item = document.createElement("option");
        item.value = String(value);
        item.textContent = label;
        return item;
    }

    function selectedPoint() {
        const level = grid.levels.find(item => item.central_max_edge_nm === Number(mesh.value));
        const point = level.points.find(item => item.frequency_hz === Number(frequency.value));
        return {level, point};
    }

    function showDrive(level, point, selected) {
        id("transmission").textContent = percent(selected.transmission);
        id("reflection").textContent = percent(selected.reflection);
        const coefficients = point.coefficients.filter(row => row.reference_drive === selected.reference_drive);
        coefficientBody.replaceChildren(...coefficients.map(row => {
            const tr = document.createElement("tr");
            const kind = row.kind === "reflection" ? "Reflection" : "Transmission";
            for (const value of [kind, row.outgoing_port_index, decimal(row.amplitude_real),
                                 decimal(row.amplitude_imag), decimal(row.power_fraction)]) {
                const cell = document.createElement("td");
                cell.textContent = value;
                tr.append(cell);
            }
            return tr;
        }));

        gateList.replaceChildren(...point.neighboring_spatial_gates.map(edge => {
            const row = edge.drives.find(item => item.reference_drive === selected.reference_drive);
            const li = document.createElement("li");
            const power = row.total_power_gate_passed ? "Power within limit" : "Power limit exceeded";
            const complex = row.complex_amplitude_gate_passed ? "Complex change within limit" : "Complex limit exceeded";
            const maximum = Math.max(...Object.values(row.maximum_absolute_complex_amplitude_change_by_block));
            li.textContent = edge.coarser_central_max_edge_nm + " → " + edge.finer_central_max_edge_nm +
                " nm · ΔT " + delta(row.transmission) + " pp · ΔR " + delta(row.reflection) +
                " pp · max complex change " + maximum.toPrecision(4) + " · " + power + " · " + complex;
            return li;
        }));

        const records = coefficients.map(row => {
            const record = {central_max_edge_nm: level.central_max_edge_nm, source_csv: point.source_csv, ...row};
            return columns.map(column => savedSelection.csvCell(record[column])).join(",");
        });
        selectedExport.href = "data:text/csv;charset=utf-8," +
            encodeURIComponent([columns.join(","), ...records].join("\r\n") + "\r\n");
        selectedExport.download = "thin_mesh_" + filenamePart(level.central_max_edge_nm) + "nm_" +
            filenamePart(point.frequency_hz / 1e6) + "MHz_drive" + selected.reference_drive + "_AP_pnc.csv";
        id("selection-link").href = savedSelection.fragment(point.id, selected.incident_port_index);
    }

    function showPoint(pointId, root) {
        const {level, point} = points.get(pointId);
        const selected = point.drives.find(item => item.incident_port_index === root);
        mesh.value = String(level.central_max_edge_nm);
        frequency.value = String(point.frequency_hz);
        drive.replaceChildren(...point.drives.map(item =>
            option(item.reference_drive, "Drive " + item.reference_drive + " · native root " + item.incident_port_index)));
        drive.value = String(selected.reference_drive);
        configurationExport.href = "data:application/json;charset=utf-8," + encodeURIComponent(point.configuration_json);
        configurationExport.download = "thin_mesh_" + filenamePart(level.central_max_edge_nm) + "nm_" +
            filenamePart(point.frequency_hz / 1e6) + "MHz_configuration_AP_pnc.json";
        id("native-csv").href = point.csv_href;
        id("native-map").href = point.map_href;
        showDrive(level, point, selected);
    }

    function changePoint() {
        const {point} = selectedPoint();
        const selected = point.drives.find(item => item.reference_drive === Number(drive.value));
        showPoint(point.id, selected.incident_port_index);
        savedSelection.select(point.id, selected.incident_port_index);
    }

    mesh.append(...grid.levels.map(level => option(level.central_max_edge_nm, level.central_max_edge_nm + " nm")));
    frequency.append(...grid.frequencies_hz.map(hz => option(hz, hz / 1e6 + " MHz")));
    const defaultPoint = grid.levels.at(-1).points.at(-1);
    const defaultRoot = point => point.drives.find(item => item.reference_drive === 3).incident_port_index;
    const defaultSelection = [defaultPoint.id, defaultRoot(defaultPoint)];
    mesh.addEventListener("change", changePoint);
    frequency.addEventListener("change", changePoint);
    drive.addEventListener("change", () => {
        const {level, point} = selectedPoint();
        const selected = point.drives.find(item => item.reference_drive === Number(drive.value));
        showDrive(level, point, selected);
        savedSelection.select(point.id, selected.incident_port_index);
    });
    savedSelection.register({
        section: document.getElementById("thin-grid"),
        notice: id("selection-notice"),
        defaultSelection,
        points: new Map([...points].map(([pointId, {point}]) => [pointId, {
            roots: point.drives.map(item => item.incident_port_index), defaultRoot: defaultRoot(point)
        }])),
        show: showPoint
    });
    showPoint(...defaultSelection);
})();
</script>'''
