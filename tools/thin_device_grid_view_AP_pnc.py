"""Offline selector for saved thin-device responses. Author: Athul Prem | _AP_pnc."""

from __future__ import annotations

from collections.abc import Callable
from html import escape
import json
from pathlib import Path


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
#thin-grid caption{{font-weight:600;text-align:left;padding:.5rem 0}}
#thin-grid td{{white-space:nowrap}}
#thin-grid .thin-grid-gates{{padding-left:1.25rem}}
#thin-grid .thin-grid-gates li+li{{margin-top:.35rem}}
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
