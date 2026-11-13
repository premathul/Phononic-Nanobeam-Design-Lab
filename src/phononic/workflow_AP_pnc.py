"""Small reproducible 3D study runner. Author: Athul Prem | _AP_pnc.

Finite-body modes and frequency-sorted Bloch samples are distinct analyses.
Neither analysis supplies calibrated scattering or a complete mode inventory.
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
        "analysis": {"kind", "boundary", "num_modes", "k_points"},
        "resources": {"max_dofs"},
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
    if analysis["kind"] not in ("modes", "bloch"):
        raise ValueError("analysis.kind must be modes or bloch")
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
            if analysis["kind"] == "modes":
                selected = solve_modes(system, boundary=analysis["boundary"], num_modes=analysis["num_modes"])
                samples = [(None, selected)]
            else:
                samples = []
                wavevectors = np.linspace(0, np.pi / geometry.length_m, analysis["k_points"])
                for index, k in enumerate(wavevectors):
                    print(f"Bloch sample {index + 1}/{len(wavevectors)}: kL/pi={k * geometry.length_m / np.pi:.4f}", flush=True)
                    result = solve_bloch(system, float(k), num_modes=analysis["num_modes"])
                    # Keep only one representative full field; all scalar rows are retained.
                    samples = [(float(k), result)]
                    rows.extend(_mode_rows(result, float(k), geometry.length_m))
                    _write_csv(output / "modes_AP_pnc.csv", rows)
                    summary["completed_k_samples"] = index + 1
                    write_json(summary_path, summary)
                selected = samples[0][1]
                summary["field_k_rad_m"] = float(wavevectors[-1])
                summary["branch_order"] = "frequency sorted independently at each k; not mode tracked"
            if analysis["kind"] == "modes":
                rows = _mode_rows(selected, None, geometry.length_m)
            _write_csv(output / "modes_AP_pnc.csv", rows)
            vertex_modes = selected.modes_full[system.basis.nodal_dofs, :].transpose(2, 1, 0)
            dof_component = np.empty(system.basis.N, dtype=np.int8)
            for component, indices in enumerate(system.basis.split_indices()):
                dof_component[indices] = component
            np.savez_compressed(output / "selected_fields_AP_pnc.npz", points_m=mesh.p.T, tetrahedra=mesh.t.T,
                                dof_coordinates_m=system.basis.doflocs.T, mass_normalized_modes=selected.modes_full,
                                dof_component=dof_component, element_dofs=system.basis.element_dofs,
                                nodal_dofs=system.basis.nodal_dofs, element_order=config["mesh"]["element_order"],
                                frequencies_hz=selected.frequencies_hz)
            # VTU deliberately samples P2 displacement at vertices, with shape normalized per mode.
            point_data = {}
            for index, field in enumerate(vertex_modes[:4]):
                scale = np.max(np.linalg.norm(field, axis=1))
                normalized = field / scale if scale else field
                point_data[f"mode_{index+1}_relative_shape_real"] = normalized.real
                if np.iscomplexobj(normalized):
                    point_data[f"mode_{index+1}_relative_shape_imag"] = normalized.imag
            meshio.write_points_cells(output / "selected_vertex_shapes_AP_pnc.vtu", mesh.p.T,
                                      [("tetra", mesh.t.T)], point_data=point_data)
            summary.update({"maximum_relative_eigen_residual": max(row["relative_residual"] for row in rows),
                            "solved_frequency_rows": len(rows), "selected_frequencies_hz": selected.frequencies_hz.tolist(),
                            "selected_rigid_mode_count": selected.rigid_mode_count,
                            "field_normalization": "mass normalized; shape visualization has arbitrary amplitude; VTU samples vertices"})
        summary.update(measured)
        summary["status"] = "completed_unverified"
        from .report_AP_pnc import write_report
        write_report(output, mesh, selected.frequencies_hz, vertex_modes, summary, rows)
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


def _write_csv(path: Path, rows: list[dict]) -> None:
    temporary = path.with_suffix(".csv.tmp")
    with temporary.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    temporary.replace(path)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="3D germanium beam modes and guided-wave dispersion")
    parser.add_argument("--config", type=Path, default=ROOT / "configs" / "straight_beam_AP_pnc.json")
    parser.add_argument("--analysis", choices=("modes", "bloch"))
    parser.add_argument("--thickness-nm", type=float)
    parser.add_argument("--cells", nargs=3, type=int, metavar=("NX", "NY", "NZ"))
    parser.add_argument("--boundary", choices=("clamped_free", "clamped_clamped", "free", "periodic_x"))
    parser.add_argument("--modes", type=int)
    parser.add_argument("--k-points", type=int)
    parser.add_argument("--output-root", type=Path)
    parser.add_argument("--open", action="store_true", help="Open the offline interactive result page")
    args = parser.parse_args(argv)
    try:
        config = json.loads(args.config.read_text(encoding="utf-8-sig"))
        if args.analysis:
            config["analysis"]["kind"] = args.analysis
            config["analysis"]["boundary"] = "periodic_x" if args.analysis == "bloch" else "clamped_free"
            if args.analysis == "modes":
                config["analysis"].pop("k_points", None)
        for value, section, key in ((args.thickness_nm, "geometry", "thickness_nm"), (args.cells, "mesh", "cells"),
                                    (args.boundary, "analysis", "boundary"), (args.modes, "analysis", "num_modes"),
                                    (args.k_points, "analysis", "k_points")):
            if value is not None:
                config[section][key] = value
        summary = run_study(config, args.output_root)
        output = Path(summary["output_directory"])
        print(f"{'Reused' if summary['reused'] else 'Completed'}: {output}")
        print(f"Max relative eigen residual: {summary['maximum_relative_eigen_residual']:.3g}")
        print(f"Solve time {summary['elapsed_seconds']:.2f} s; sampled process peak {summary['peak_process_memory_mib']:.1f} MiB")
        print("Model status: computed; spatial convergence and calibrated power scattering are separate validation steps.")
        if args.open:
            webbrowser.open((output / "report_AP_pnc.html").as_uri())
        return 0
    except (ValueError, KeyError, OSError, RuntimeError) as exc:
        print(f"Study could not complete: {exc}", file=sys.stderr)
        return 1
