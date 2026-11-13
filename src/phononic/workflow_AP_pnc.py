"""Small reproducible 3D study runner. Author: Athul Prem | _AP_pnc.

Finite-body modes, Bloch samples and matched-port scattering are distinct
analyses. Numerical solutions require separate mesh and model validation.
"""
from __future__ import annotations

import argparse
from contextlib import contextmanager
import copy
import csv
import hashlib
from importlib.metadata import version
import json
import math
from pathlib import Path
import platform
import re
import subprocess
import sys
import threading
import time
import webbrowser

import meshio
import numpy as np
import psutil

from .geometry_AP_pnc import BeamGeometry, make_straight_mesh
from .material_AP_pnc import CubicMaterial
from .elasticity_AP_pnc import assemble_elasticity, solve_modes
from .lead_modes_AP_pnc import solve_bloch
from .port_modes_AP_pnc import assemble_port, solve_port_modes
from .scattering_AP_pnc import solve_scattering

ROOT = Path(__file__).resolve().parents[2]
PACKAGES = ("numpy", "scipy", "scikit-fem", "meshio", "gmsh", "matplotlib", "psutil")


def write_json(path: Path, value: dict) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    def numpy_scalar(item):
        if isinstance(item, np.generic):
            return item.item()
        raise TypeError(f"Unsupported JSON value: {type(item).__name__}")
    temporary.write_text(json.dumps(value, indent=2, ensure_ascii=False, allow_nan=False, default=numpy_scalar) + "\n", encoding="utf-8")
    temporary.replace(path)


def digest(value: object) -> str:
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()).hexdigest()


def resolve_config(config: dict) -> dict:
    """Keep the first public configuration deliberately small and explicit."""
    config = copy.deepcopy(config)
    allowed = {"author", "file_tag", "schema_version", "name", "geometry", "mesh", "material", "analysis", "resources"}
    if set(config) - allowed:
        raise ValueError(f"Unknown configuration fields: {sorted(set(config) - allowed)}")
    if config.get("schema_version") != 1:
        raise ValueError("schema_version must be 1")
    config.setdefault("name", "straight_beam")
    if not re.fullmatch(r"[A-Za-z0-9_-]{1,80}", config["name"]):
        raise ValueError("name must use 1-80 letters, digits, underscores or hyphens")
    required = {
        "geometry": {"length_nm", "width_nm", "thickness_nm"},
        "mesh": {"cells", "element_order"},
        "material": {"name", "C11_Pa", "C12_Pa", "C44_Pa", "density_kg_m3", "crystal_rotation", "temperature_policy", "temperature_K"},
        "analysis": {"kind", "boundary", "num_modes", "k_points", "frequency_GHz", "incident_side"},
        "resources": {"max_dofs", "max_port_dofs"},
    }
    for section, keys in required.items():
        if section not in config or not isinstance(config[section], dict):
            raise ValueError(f"Missing configuration object: {section}")
        if set(config[section]) - keys:
            raise ValueError(f"Unknown {section} fields: {sorted(set(config[section]) - keys)}")
    for key in required["geometry"]:
        value = config["geometry"][key]
        if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value) or value <= 0:
            raise ValueError(f"{key} must be positive and finite (nm)")
    cells = config["mesh"]["cells"]
    if len(cells) != 3 or any(type(n) is not int or n <= 0 for n in cells):
        raise ValueError("mesh.cells must contain three positive integers")
    order = config["mesh"]["element_order"]
    if type(order) is not int or order not in (1, 2):
        raise ValueError("element_order must be 1 or 2")
    maximum = config["resources"]["max_dofs"]
    if type(maximum) is not int or maximum <= 0:
        raise ValueError("max_dofs must be a positive integer")
    predicted = 3 * math.prod(order * n + 1 for n in cells)
    if predicted > maximum:
        raise ValueError(f"Requested grid has {predicted} displacement DOFs; configured maximum is {maximum}")
    material = config["material"]
    if material["temperature_policy"] != "fixed_property_control":
        raise ValueError("Only fixed_property_control is implemented; temperature-dependent material data are not installed")
    temperature = material["temperature_K"]
    if isinstance(temperature, bool) or not isinstance(temperature, (float, int)) or not math.isfinite(temperature) or temperature <= 0:
        raise ValueError("temperature_K must be positive and finite")
    CubicMaterial(C11=material["C11_Pa"], C12=material["C12_Pa"], C44=material["C44_Pa"],
                  rho=material["density_kg_m3"], rotation=material["crystal_rotation"])
    analysis = config["analysis"]
    if analysis["kind"] == "scattering":
        if analysis["boundary"] != "matched_ports":
            raise ValueError("Scattering requires boundary=matched_ports")
        if "num_modes" in analysis or "k_points" in analysis:
            raise ValueError("Scattering finds all roots of the discretized port; omit num_modes and k_points")
        if "frequency_GHz" not in analysis:
            raise ValueError("Set analysis.frequency_GHz or pass --frequency-ghz for scattering")
        frequency = analysis["frequency_GHz"]
        if isinstance(frequency, bool) or not isinstance(frequency, (float, int)) or not math.isfinite(frequency) or frequency <= 0:
            raise ValueError("frequency_GHz must be positive and finite")
        analysis.setdefault("incident_side", "left")
        if analysis["incident_side"] not in ("left", "right"):
            raise ValueError("incident_side must be left or right")
        config["resources"].setdefault("max_port_dofs", 200)
        maximum_port = config["resources"]["max_port_dofs"]
        if type(maximum_port) is not int or maximum_port <= 0:
            raise ValueError("max_port_dofs must be a positive integer")
        predicted_port = 3 * math.prod(order * n + 1 for n in cells[1:])
        if predicted_port > maximum_port:
            raise ValueError(f"Requested section has {predicted_port} port DOFs; configured maximum is {maximum_port}")
        return config
    if analysis["kind"] not in ("modes", "bloch"):
        raise ValueError("analysis.kind must be modes, bloch or scattering")
    if "frequency_GHz" in analysis or "incident_side" in analysis:
        raise ValueError("frequency_GHz and incident_side belong to scattering analysis")
    if analysis["boundary"] not in ("clamped_free", "clamped_clamped", "free", "periodic_x"):
        raise ValueError("Unsupported boundary condition")
    if analysis["kind"] == "bloch":
        if analysis["boundary"] != "periodic_x":
            raise ValueError("Bloch analysis requires boundary=periodic_x")
        analysis.setdefault("k_points", 9)
        if type(analysis["k_points"]) is not int or not 2 <= analysis["k_points"] <= 101:
            raise ValueError("k_points must be an integer between 2 and 101 for this pilot runner")
    elif analysis["boundary"] == "periodic_x" or "k_points" in analysis:
        raise ValueError("k_points and periodic_x belong to Bloch analysis")
    if type(analysis["num_modes"]) is not int or not 1 <= analysis["num_modes"] <= 64:
        raise ValueError("num_modes must be an integer between 1 and 64")
    return config


def source_identity() -> dict:
    paths = sorted(Path(__file__).parent.glob("*.py")) + [ROOT / "simulate_AP_pnc.py"]
    hashes = {p.relative_to(ROOT).as_posix(): hashlib.sha256(p.read_bytes()).hexdigest() for p in paths}
    try:
        revision = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True, stderr=subprocess.DEVNULL).strip()
    except (OSError, subprocess.CalledProcessError):
        revision = None
    return {"git_commit": revision, "source_sha256": hashes,
            "packages": {name: version(name) for name in PACKAGES},
            "python": platform.python_version(), "platform": platform.platform()}


@contextmanager
def measured_resources():
    process = psutil.Process()
    record = {"peak_process_memory_mib": process.memory_info().rss / 2**20,
              "system_available_memory_mib_at_start": psutil.virtual_memory().available / 2**20}
    stop = threading.Event()
    def sample():
        while not stop.wait(0.05):
            record["peak_process_memory_mib"] = max(record["peak_process_memory_mib"], process.memory_info().rss / 2**20)
    thread = threading.Thread(target=sample, daemon=True)
    start = time.perf_counter()
    thread.start()
    try:
        yield record
    finally:
        stop.set()
        thread.join()
        memory = process.memory_info()
        record["peak_process_memory_mib"] = max(record["peak_process_memory_mib"], memory.rss / 2**20)
        record["process_lifetime_peak_working_set_mib"] = getattr(memory, "peak_wset", memory.rss) / 2**20
        record["elapsed_seconds"] = time.perf_counter() - start


def run_study(config: dict, output_root: Path | None = None) -> dict:
    config = resolve_config(config)
    identity = source_identity()
    numerical_identity = {key: identity[key] for key in ("source_sha256", "packages", "python", "platform")}
    run_hash = digest({"configuration": config, "implementation": numerical_identity})
    mechanical = copy.deepcopy(config)
    for key in ("name", "author", "file_tag"):
        mechanical.pop(key, None)
    mechanical["material"].pop("temperature_K")
    operator_hash = digest({"configuration": mechanical, "implementation": numerical_identity})
    output_root = Path(output_root or ROOT / "results" / "runs")
    stem = f"{config['name']}-{run_hash[:12]}"
    output = output_root / stem
    attempt = 1
    while output.exists():
        summary_path = output / "summary_AP_pnc.json"
        if summary_path.exists():
            previous = json.loads(summary_path.read_text(encoding="utf-8"))
            if previous.get("run_hash") == run_hash and previous.get("status") == "completed_unverified":
                expected = previous.get("output_sha256", {})
                if expected and all((output / name).is_file() and hashlib.sha256((output / name).read_bytes()).hexdigest() == value
                                    for name, value in expected.items()):
                    return {**previous, "output_directory": str(output.resolve()), "reused": True}
                raise ValueError(f"Saved result files changed or are missing: {output}")
        old_lock = output / ".run.lock"
        if old_lock.exists():
            owner = json.loads(old_lock.read_text(encoding="utf-8"))
            try:
                process = psutil.Process(owner["pid"])
                active = process.create_time() == owner["process_start"]
            except psutil.NoSuchProcess:
                active = False
            if active:
                raise ValueError(f"This study is already running: {output}")
        # A failed or interrupted attempt is evidence. Preserve it and use a new directory.
        attempt += 1
        output = output_root / f"{stem}-attempt{attempt}"
    output.mkdir(parents=True, exist_ok=False)
    summary_path = output / "summary_AP_pnc.json"
    lock = output / ".run.lock"
    summary = {"author": "Athul Prem", "file_tag": "_AP_pnc", "run_hash": run_hash,
               "mechanical_operator_hash": operator_hash, "status": "running", "configuration": config,
               "implementation": identity, "analysis_kind": config["analysis"]["kind"],
               "geometry_nm": config["geometry"], "boundary": config["analysis"]["boundary"],
               "material_status": "nominal fixed-property control; no temperature calibration",
               "calibrated_scattering_available": False, "experimental_validation": False,
               "mode_completeness_established": False, "spatial_convergence_established": False,
               "output_directory": str(output.resolve())}
    try:
        with lock.open("x", encoding="utf-8") as stream:
            process = psutil.Process()
            json.dump({"pid": process.pid, "process_start": process.create_time()}, stream)
        write_json(summary_path, summary)
        write_json(output / "configuration_AP_pnc.json", config)
        with measured_resources() as measured:
            dims = config["geometry"]
            geometry = BeamGeometry(*(dims[key] * 1e-9 for key in ("length_nm", "width_nm", "thickness_nm")))
            params = config["material"]
            material = CubicMaterial(C11=params["C11_Pa"], C12=params["C12_Pa"], C44=params["C44_Pa"],
                                     rho=params["density_kg_m3"], rotation=params["crystal_rotation"])
            mesh = make_straight_mesh(geometry, config["mesh"]["cells"])
            system = assemble_elasticity(mesh, material, element_order=config["mesh"]["element_order"],
                                        max_dofs=config["resources"]["max_dofs"])
            summary.update({"ndof": int(system.K.shape[0]), "mesh_vertices": int(mesh.nvertices), "mesh_tetrahedra": int(mesh.nelements)})
            print(f"3D solid: {mesh.nelements} tetrahedra, {system.K.shape[0]} displacement DOFs", flush=True)
            analysis = config["analysis"]
            rows = []
            scattering = analysis["kind"] == "scattering"
            extra_fields = {}
            if scattering:
                frequency = analysis["frequency_GHz"] * 1e9
                port = assemble_port(system, material, max_dofs=config["resources"]["max_port_dofs"])
                port_modes = solve_port_modes(port, frequency)
                result = solve_scattering(system, port_modes, frequency, incident_side=analysis["incident_side"])
                rows = _scattering_rows(result)
                _write_csv(output / "scattering_AP_pnc.csv", rows)
                _write_csv(output / "scattering_channels_AP_pnc.csv", _channel_rows(result))
                _write_csv(output / "port_modes_AP_pnc.csv", _port_rows(port_modes))
                fields = result.displacements
                frequencies = np.full(fields.shape[1], frequency)
                max_power_imbalance = float(np.max(abs(result.power_balance)))
                power_disagreement = float(max(np.max(abs(result.reflection - result.reflection_flux)),
                                               np.max(abs(result.transmission - result.transmission_flux))))
                matched = bool(np.max(abs(result.reflection)) < 0.01
                               and np.max(abs(result.transmission - 1)) < 0.01
                               and max_power_imbalance < 0.01 and power_disagreement < 0.01
                               and np.max(abs(result.flux_power_balance)) < 0.01
                               and np.max(abs(result.net_outward_power_ratio)) < 0.01
                               and np.max(result.relative_residuals) < 1e-8
                               and np.max(port_modes.relative_residuals) < 1e-8)
                summary.update({"frequency_hz": frequency, "incident_side": analysis["incident_side"],
                                "incident_mode_indices": result.incident_mode_indices.tolist(),
                                "num_incident_channels": len(result.incident_mode_indices),
                                "port_dofs": port.basis.N, "discrete_port_roots": len(port_modes.powers_w),
                                "maximum_relative_port_residual": float(np.max(port_modes.relative_residuals)),
                                "port_basis_condition": list(result.port_basis_condition),
                                "maximum_reflection": float(np.max(result.reflection)),
                                "max_power_imbalance": max_power_imbalance,
                                "maximum_flux_power_imbalance": float(np.max(abs(result.flux_power_balance))),
                                "maximum_net_outward_power_ratio": float(np.max(abs(result.net_outward_power_ratio))),
                                "maximum_modal_flux_difference": power_disagreement,
                                "matched_reference_passed": matched,
                                "calibrated_scattering_available": False,
                                "scattering_power_normalization_available": True,
                                "matched_reference_scope": "Power-only check of this uniform guide; phase and mesh convergence require separate evidence",
                                "field_normalization": "Response per square root of incident watt; visualization has arbitrary amplitude; VTU samples vertices"})
                extra_fields = {"displacements_per_sqrt_w": fields,
                                "incident_mode_indices": result.incident_mode_indices,
                                "incident_powers_w": result.incident_powers_w,
                                "reflection_mode_indices": result.reflection_mode_indices,
                                "transmission_mode_indices": result.transmission_mode_indices,
                                "reflection_amplitudes": result.reflection_amplitudes,
                                "transmission_amplitudes": result.transmission_amplitudes,
                                "reflection_all_mode_indices": result.reflection_all_mode_indices,
                                "transmission_all_mode_indices": result.transmission_all_mode_indices,
                                "reflection_amplitudes_full": result.reflection_amplitudes_full,
                                "transmission_amplitudes_full": result.transmission_amplitudes_full,
                                "port_wavenumbers_rad_m": port_modes.wavenumbers_rad_m,
                                "port_displacements": port_modes.displacements,
                                "port_axial_forces": port_modes.axial_forces,
                                "port_left_dofs": port.left_dofs, "port_right_dofs": port.right_dofs}
            elif analysis["kind"] == "modes":
                selected = solve_modes(system, boundary=analysis["boundary"], num_modes=analysis["num_modes"])
            else:
                wavevectors = np.linspace(0, np.pi / geometry.length_m, analysis["k_points"])
                for index, k in enumerate(wavevectors):
                    print(f"Bloch sample {index + 1}/{len(wavevectors)}: kL/pi={k * geometry.length_m / np.pi:.4f}", flush=True)
                    result = solve_bloch(system, float(k), num_modes=analysis["num_modes"])
                    # Keep only one representative full field; all scalar rows are retained.
                    selected = result
                    rows.extend(_mode_rows(result, float(k), geometry.length_m))
                    _write_csv(output / "modes_AP_pnc.csv", rows)
                    summary["completed_k_samples"] = index + 1
                    write_json(summary_path, summary)
                summary["field_k_rad_m"] = float(wavevectors[-1])
                summary["branch_order"] = "frequency sorted independently at each k; not mode tracked"
            if analysis["kind"] == "modes":
                rows = _mode_rows(selected, None, geometry.length_m)
            if not scattering:
                _write_csv(output / "modes_AP_pnc.csv", rows)
                fields, frequencies = selected.modes_full, selected.frequencies_hz
                extra_fields = {"mass_normalized_modes": fields}
                summary.update({"maximum_relative_eigen_residual": max(row["relative_residual"] for row in rows),
                                "selected_rigid_mode_count": selected.rigid_mode_count,
                                "field_normalization": "Mass normalized; visualization has arbitrary amplitude; VTU samples vertices"})
            vertex_modes = fields[system.basis.nodal_dofs, :].transpose(2, 1, 0)
            dof_component = np.empty(system.basis.N, dtype=np.int8)
            for component, indices in enumerate(system.basis.split_indices()):
                dof_component[indices] = component
            np.savez_compressed(output / "selected_fields_AP_pnc.npz", points_m=mesh.p.T, tetrahedra=mesh.t.T,
                                dof_coordinates_m=system.basis.doflocs.T,
                                dof_component=dof_component, element_dofs=system.basis.element_dofs,
                                nodal_dofs=system.basis.nodal_dofs, element_order=config["mesh"]["element_order"],
                                frequencies_hz=frequencies, **extra_fields)
            # VTU deliberately samples P2 displacement at vertices, with shape normalized per mode.
            point_data = {}
            for index, field in enumerate(vertex_modes[:4]):
                scale = np.max(np.linalg.norm(field, axis=1))
                normalized = field / scale if scale else field
                label = "incident" if scattering else "mode"
                point_data[f"{label}_{index+1}_relative_shape_real"] = normalized.real
                if np.iscomplexobj(normalized):
                    point_data[f"{label}_{index+1}_relative_shape_imag"] = normalized.imag
            meshio.write_points_cells(output / "selected_vertex_shapes_AP_pnc.vtu", mesh.p.T,
                                      [("tetra", mesh.t.T)], point_data=point_data)
            summary.update({"maximum_relative_solver_residual": max(row["relative_residual"] for row in rows),
                            "solved_frequency_rows": len(rows), "selected_frequencies_hz": frequencies.tolist()})
        summary.update(measured)
        summary["status"] = "completed_unverified"
        from .report_AP_pnc import write_report
        write_report(output, mesh, frequencies, vertex_modes, summary, rows)
        summary["status"] = "completed_unverified"
        summary["numerical_interpretation"] = "Solved 3D model; convergence and physical adequacy require separate evidence"
        summary["output_sha256"] = {p.name: hashlib.sha256(p.read_bytes()).hexdigest() for p in output.iterdir()
                                    if p.is_file() and p not in (summary_path, lock) and p.suffix != ".tmp"}
        summary["output_bytes"] = sum(p.stat().st_size for p in output.iterdir() if p.is_file() and p != lock)
        write_json(summary_path, summary)
        return {**summary, "reused": False}
    except BaseException as exc:
        summary.update({"status": "interrupted" if isinstance(exc, KeyboardInterrupt) else "failed",
                        "error": f"{type(exc).__name__}: {exc}"})
        try:
            write_json(summary_path, summary)
        except OSError:
            # Preserve the original exception when a full/read-only disk also prevents diagnostics.
            pass
        raise
    finally:
        lock.unlink(missing_ok=True)


def _mode_rows(result, k: float | None, length: float) -> list[dict]:
    rows = []
    for index, frequency in enumerate(result.frequencies_hz):
        row = {"mode_order": index + 1, "frequency_hz": float(frequency), "frequency_GHz": float(frequency / 1e9),
               "relative_residual": float(result.relative_residuals[index]),
               "rigid_mode": index < result.rigid_mode_count}
        if k is not None:
            row.update({"k_rad_m": k, "kL_over_pi": k * length / np.pi})
        row.update({f"kinetic_fraction_{axis}": float(result.polarization_fractions[index, component])
                    for component, axis in enumerate("xyz")})
        rows.append(row)
    return rows


def _scattering_rows(result) -> list[dict]:
    fields = ("transmission", "reflection", "power_balance", "transmission_flux", "reflection_flux", "flux_power_balance")
    return [{"incident_mode_order": column + 1, "frequency_GHz": result.frequency_hz / 1e9,
             **{name: float(getattr(result, name)[column]) for name in fields},
             "relative_residual": float(result.relative_residuals[column])}
            for column in range(len(result.incident_mode_indices))]


def _channel_rows(result) -> list[dict]:
    rows = []
    for kind in ("reflection", "transmission"):
        amplitudes = getattr(result, f"{kind}_amplitudes")
        powers = getattr(result, f"{kind}_modal_powers")
        for outgoing, root in enumerate(getattr(result, f"{kind}_mode_indices")):
            for incident, incident_root in enumerate(result.incident_mode_indices):
                value = amplitudes[outgoing, incident]
                rows.append({"kind": kind, "incident_mode_order": incident + 1,
                             "incident_port_index": int(incident_root), "outgoing_port_index": int(root),
                             "amplitude_real": float(value.real), "amplitude_imag": float(value.imag),
                             "power_fraction": float(powers[outgoing, incident])})
    return rows


def _port_rows(modes) -> list[dict]:
    positive = set(modes.positive_indices.tolist())
    return [{"port_index": index, "k_real_rad_m": float(k.real), "k_imag_rad_m": float(k.imag),
             "direction": "+x" if index in positive else "-x", "propagating": bool(modes.propagating[index]),
             "signed_power_W": float(modes.powers_w[index]), "relative_residual": float(modes.relative_residuals[index])}
            for index, k in enumerate(modes.wavenumbers_rad_m)]


def _write_csv(path: Path, rows: list[dict]) -> None:
    temporary = path.with_suffix(".csv.tmp")
    with temporary.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    temporary.replace(path)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="3D germanium beam modes, dispersion and matched-port scattering")
    parser.add_argument("--config", type=Path, default=ROOT / "configs" / "straight_beam_AP_pnc.json")
    parser.add_argument("--analysis", choices=("modes", "bloch", "scattering"))
    parser.add_argument("--thickness-nm", type=float)
    parser.add_argument("--cells", nargs=3, type=int, metavar=("NX", "NY", "NZ"))
    parser.add_argument("--boundary", choices=("clamped_free", "clamped_clamped", "free", "periodic_x", "matched_ports"))
    parser.add_argument("--modes", type=int)
    parser.add_argument("--k-points", type=int)
    parser.add_argument("--frequency-ghz", type=float)
    parser.add_argument("--incident-side", choices=("left", "right"))
    parser.add_argument("--max-port-dofs", type=int)
    parser.add_argument("--output-root", type=Path)
    parser.add_argument("--open", action="store_true", help="Open the offline interactive result page")
    args = parser.parse_args(argv)
    try:
        config = json.loads(args.config.read_text(encoding="utf-8-sig"))
        if args.analysis:
            previous = config["analysis"]
            if args.analysis == "scattering":
                config["analysis"] = {"kind": "scattering", "boundary": "matched_ports",
                                      "incident_side": previous.get("incident_side", "left")}
                if "frequency_GHz" in previous:
                    config["analysis"]["frequency_GHz"] = previous["frequency_GHz"]
            else:
                config["analysis"] = {"kind": args.analysis, "boundary": "periodic_x" if args.analysis == "bloch" else "clamped_free",
                                      "num_modes": previous.get("num_modes", 8)}
                if args.analysis == "bloch":
                    config["analysis"]["k_points"] = previous.get("k_points", 9)
        for value, section, key in ((args.thickness_nm, "geometry", "thickness_nm"), (args.cells, "mesh", "cells"),
                                    (args.boundary, "analysis", "boundary"), (args.modes, "analysis", "num_modes"),
                                    (args.k_points, "analysis", "k_points"),
                                    (args.frequency_ghz, "analysis", "frequency_GHz"),
                                    (args.incident_side, "analysis", "incident_side"),
                                    (args.max_port_dofs, "resources", "max_port_dofs")):
            if value is not None:
                config[section][key] = value
        summary = run_study(config, args.output_root)
        output = Path(summary["output_directory"])
        print(f"{'Reused' if summary['reused'] else 'Completed'}: {output}")
        print(f"Max relative solve residual: {summary['maximum_relative_solver_residual']:.3g}")
        if summary["analysis_kind"] == "scattering":
            print(f"Matched-guide power check: {summary['matched_reference_passed']}; max R={summary['maximum_reflection']:.6g}")
        print(f"Solve time {summary['elapsed_seconds']:.2f} s; sampled process peak {summary['peak_process_memory_mib']:.1f} MiB")
        print("Model status: computed; phase accuracy, spatial convergence and physical adequacy require separate evidence.")
        if args.open:
            webbrowser.open((output / "report_AP_pnc.html").as_uri())
        return 0
    except (ValueError, KeyError, OSError, RuntimeError) as exc:
        print(f"Study could not complete: {exc}", file=sys.stderr)
        return 1
