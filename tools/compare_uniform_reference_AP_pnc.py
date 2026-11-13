#!/usr/bin/env python3
"""Compare a saved uniform guide against a common SAFE port. Author: Athul Prem | _AP_pnc."""

from __future__ import annotations

import argparse
import csv
from importlib.metadata import version
import json
from pathlib import Path
import platform
import sys
from tempfile import TemporaryDirectory

import numpy as np
from skfem import Basis, ElementTriP2, MeshTri
from skfem.models.poisson import mass

import compare_scattering_AP_pnc as saved
from compare_scattering_AP_pnc import (
    AUTHOR, K_GAP_TOL, LIMIT, ROOT, Modes, Section, align, check_outputs,
    compare_blocks, file_sha, interpolation, read_run, read_scattering, require,
)
from phononic.workflow_AP_pnc import digest, write_json

TABLE = "uniform_reference_AP_pnc.csv"
DEFAULT_REFERENCE = ROOT / "verification/layered_port_resolution_AP_pnc.json"
SECTION_SOURCES = ("elasticity", "geometry", "material", "port_modes")


def read_section(record: dict, width_nm: float, thickness_nm: float, density: float) -> tuple[Section, dict[str, Modes]]:
    size = np.array([width_nm, thickness_nm]) * 1e-9
    mesh = record["section"]
    basis = Basis(MeshTri(np.asarray(mesh["vertices_yz_m"]).T, np.asarray(mesh["triangles"]).T),
                  ElementTriP2(), intorder=4)
    require(np.isclose(basis.dx.sum(), size.prod(), rtol=1e-10, atol=0)
            and np.allclose(basis.mesh.p.min(axis=1) / size, -0.5, rtol=0, atol=1e-10)
            and np.allclose(basis.mesh.p.max(axis=1) / size, 0.5, rtol=0, atol=1e-10),
            "Saved SAFE port area or extent differs from the configured section.")
    n = 3 * basis.N
    require(n == record["trace_dofs"]
            and np.allclose(mesh["scalar_coordinates_yz_m"], basis.doflocs.T, atol=1e-20, rtol=0)
            and 0 <= record["max_relative_residual"] < 1e-8,
            "Invalid saved SAFE trace or eigensolver residual.")
    section = Section(basis, np.empty((0, 0), dtype=int), mass.assemble(basis).toarray() * density, "")
    modes = {}
    for sign, direction in ((1, "positive"), (-1, "negative")):
        fields = record["propagating"][direction]
        roots, k = (np.asarray(fields[key]) for key in ("root_indices", "k_rad_m"))
        require(roots.ndim == 1 and roots.size > 0 and roots.dtype.kind in "iu" and len(np.unique(roots)) == roots.size
                and roots.min() >= 0 and roots.max() < 2 * n and k.shape == roots.shape and np.isfinite(k).all()
                and np.all(k != 0), f"Unresolved SAFE {direction} propagating roots.")
        require(np.all(np.diff(np.sort(k)) > K_GAP_TOL * max(abs(k))), f"Unresolved degenerate SAFE {direction} branches.")
        power = np.asarray(fields["signed_power_w"])
        require(power.shape == roots.shape and np.allclose(power, sign, rtol=0, atol=1e-8),
                f"Invalid saved SAFE {direction} signed unit-power normalization.")
        if "conjugate_of" in fields:
            require(direction == "negative" and fields["conjugate_of"] == "positive"
                    and np.allclose(k, -modes["positive"].k, rtol=1e-12, atol=0),
                    "Invalid time-reversed SAFE field reference.")
            u = modes["positive"].displacement.conj()
        else:
            pairs = np.asarray(fields["displacement"])
            require(pairs.shape == (3, basis.N, len(roots), 2) and np.isfinite(pairs).all(),
                    f"Invalid saved SAFE {direction} displacement fields.")
            u = pairs[..., 0] + 1j * pairs[..., 1]
        modes[direction] = Modes(roots, k, u)
    require(np.intersect1d(modes["positive"].roots, modes["negative"].roots).size == 0,
            "SAFE propagating direction groups overlap.")
    return section, modes


def check_section_inputs(case: str, record: dict, fixed: dict) -> None:
    inputs = record.get("inputs", {})
    require(inputs.get("holes") == [] and inputs.get("element_order") == 2,
            f"SAFE case {case} requires standalone P2 section inputs without holes.")
    dimensions = fixed["dimensions_nm"]
    require(all(inputs["geometry_nm"][key] == dimensions[key] for key in ("width_nm", "thickness_nm")),
            f"SAFE case {case} section width or thickness differs from the run.")
    require(inputs["frequency_hz"] == fixed["frequency_hz"], f"SAFE case {case} frequency differs from the run.")
    require(inputs["material"] == fixed["material"], f"SAFE case {case} material or crystal orientation differs from the run.")


def reference_qualification(data: dict, endpoint: str, reference, fixed: dict) -> dict:
    cases = data["cases"]
    ny, nz = cases[endpoint]["width_cells"], cases[endpoint]["thickness_layers"]
    pairs = {}
    for key in data["comparisons"]:
        first, second = key.split("_to_")
        if second != endpoint:
            continue
        old_ny, old_nz = cases[first]["width_cells"], cases[first]["thickness_layers"]
        if old_nz == nz and old_ny < ny:
            axis, previous = "width", old_ny
        elif old_ny == ny and old_nz < nz:
            axis, previous = "thickness", old_nz
        else:
            continue
        if axis not in pairs or previous > pairs[axis]["coarser_subdivisions"]:
            pairs[axis] = {"pair": key, "coarser_subdivisions": previous}
    section, modes = reference
    dimensions = fixed["dimensions_nm"]
    length_m = dimensions["length_nm"] * 1e-9
    for pair in pairs.values():
        coarse_case = pair["pair"].split("_to_")[0]
        coarse = cases[coarse_case]
        check_section_inputs(coarse_case, coarse, fixed)
        old_section, old_modes = read_section(coarse, dimensions["width_nm"], dimensions["thickness_nm"],
                                              fixed["material"]["density_kg_m3"])
        probe = interpolation(old_section, section)
        indicators = {}
        for direction, new in modes.items():
            old = old_modes[direction]
            require(len(old.roots) == len(new.roots), "SAFE refinement changed the propagating channel count.")
            indices, _, _ = align(old, new, section.mass, probe, True)
            indicators[direction] = float(max(abs(new.k[indices] - old.k)) * length_m)
        pair.update(phase_indicators_rad=indicators, maximum_phase_indicator_rad=max(indicators.values()))
    if set(pairs) == {"width", "thickness"}:
        status = "passed" if all(pair["maximum_phase_indicator_rad"] < LIMIT for pair in pairs.values()) else "failed"
    else:
        status = "unevaluated"
    return {"status": status, "length_m": length_m, "phase_indicator_limit_rad": LIMIT, "pairs": pairs,
            "scope": "Successive SAFE section differences at this propagation distance. No continuum error bound."}


def read_reference(path: Path, case: str, fixed: dict) -> tuple[tuple[Section, dict[str, Modes]], dict, dict]:
    data = json.loads(path.read_bytes())
    require(data["status"] == "diagnostic_completed" and data["schema_version"] == 1,
            "Select a completed portable SAFE refinement record.")
    require(case in data["cases"], f"SAFE reference case is missing: {case}.")
    record = data["cases"][case]
    check_section_inputs(case, record, fixed)
    dimensions = fixed["dimensions_nm"]
    width_nm, thickness_nm = dimensions["width_nm"], dimensions["thickness_nm"]
    require(data["physical_section_nm"] == {"width": width_nm, "thickness": thickness_nm},
            "SAFE reference section width or thickness differs from the run.")
    require(data["frequency_hz"] == fixed["frequency_hz"], "SAFE reference frequency differs from the run.")
    sources = data["numerical_source_sha256"]
    for module in SECTION_SOURCES:
        key = f"src/phononic/{module}_AP_pnc.py"
        require(sources[key] == fixed["numerical_source_sha256"][key], f"SAFE reference source differs from the run: {key}.")
    reference = read_section(record, width_nm, thickness_nm, fixed["material"]["density_kg_m3"])
    quality = reference_qualification(data, case, reference, fixed)
    provenance = {"path": str(path), "sha256": file_sha(path), "case": case, "trace_dofs": record["trace_dofs"],
                  "inputs": record["inputs"], "source": record["source"], "numerical_source_sha256": sources,
                  "numerical_environment": data["numerical_environment"]}
    return reference, provenance, quality


def compare_uniform_blocks(reference, candidate, incident_side: str, length_m: float):
    section, modes = reference
    forward, backward = ("positive", "negative") if incident_side == "left" else ("negative", "positive")
    incident, reflected = modes[forward], modes[backward]
    direction = 1 if incident_side == "left" else -1
    distance = direction * length_m
    n = len(incident.roots)
    require(n == len(reflected.roots), "The reference requires the same propagating count in both directions.")
    ideal = np.diag(np.exp(1j * incident.k * distance))
    reference_data = (section, {"incoming": incident, "reflection": reflected, "transmission": incident},
                      {"reflection": np.zeros((n, n), dtype=complex), "transmission": ideal})
    matches, blocks, rows = compare_blocks(reference_data, candidate)
    measured = np.asarray(blocks["transmission"]["candidate_aligned"])
    transmission = measured[:, :, 0] + 1j * measured[:, :, 1]
    native_k = np.asarray(matches["incoming"]["candidate_wavenumber_rad_m"])
    phase_drift = (native_k - incident.k) * distance
    native = np.diag(np.exp(1j * native_k * distance))
    native_error = np.max(abs(transmission - native), axis=0)
    for row in rows:
        index = row["incident_channel"] - 1
        is_transmission = row["quantity"] == "transmission"
        row["native_minus_common_k_rad_m"] = float(native_k[index] - incident.k[index]) if is_transmission else None
        row["native_minus_common_propagation_phase_rad"] = float(phase_drift[index]) if is_transmission else None
        row["maximum_native_complex_error"] = float(native_error[index]) if is_transmission else None
    diagnostic = {"native_minus_common_propagation_phase_rad": phase_drift.tolist(),
                  "maximum_native_phase_drift_rad": float(max(abs(phase_drift))),
                  "maximum_native_transmission_complex_error": float(max(native_error))}
    return matches, blocks, rows, diagnostic


def compare_uniform_reference(run: Path, reference: Path = DEFAULT_REFERENCE, output_root: Path | None = None,
                              reference_case: str = "8x4") -> dict:
    run, reference = Path(run).resolve(), Path(reference).resolve()
    summary, fixed, run_record = read_run(run)
    require(not fixed["placed_holes_m"] and run_record["hole_pattern_provenance"] is None,
            "Select a uniform scattering run with no hole pattern.")
    reference_data, provenance, quality = read_reference(reference, reference_case, fixed)
    length_m = fixed["dimensions_nm"]["length_nm"] * 1e-9
    implementation = {"tool_sha256": file_sha(Path(__file__)), "comparison_helper_sha256": file_sha(Path(saved.__file__)),
                      "python": platform.python_version(),
                      "packages": {name: version(name) for name in ("numpy", "scipy", "scikit-fem")}}
    identity = {"run": run_record, "reference": provenance, "reference_phase_qualification": quality,
                "implementation": implementation}
    comparison_hash = digest(identity)
    output = Path(output_root or ROOT / "results/runs/comparisons").resolve() / f"uniform-reference-{comparison_hash[:16]}"
    if output.exists():
        result = json.loads((output / saved.FILES[0]).read_bytes())
        require(result["comparison_hash"] == comparison_hash and result["status"] == "compared"
                and all(result[key] == value for key, value in identity.items()), "Saved uniform comparison identity differs.")
        check_outputs(output, result["output_sha256"], (TABLE,))
        return {**result, "output_directory": str(output), "reused": True}

    matches, blocks, rows, native = compare_uniform_blocks(reference_data, read_scattering(run, summary, fixed),
                                                          fixed["incident_side"], length_m)
    power = max(block["maximum_power_change"] for block in blocks.values())
    complex_error = max(block["maximum_complex_change"] for block in blocks.values())
    result = {**AUTHOR, **identity, "schema_version": 1, "status": "compared", "comparison_hash": comparison_hash,
              "physical_inputs": fixed, "propagation_length_m": length_m,
              "matched_channels": matches, "blocks": blocks, "native_port_diagnostic": native,
              "power_error_limit": LIMIT, "complex_error_limit": LIMIT, "maximum_power_error": power,
              "maximum_complex_error": complex_error, "power_passed": power < LIMIT, "complex_passed": complex_error < LIMIT,
              "input_power_balance_passed": run_record["power_balance_passed"],
              "uniform_response_passed": run_record["power_balance_passed"] is True and max(power, complex_error) < LIMIT,
              "scope": "Uniform-guide complex response at its original end planes, compared with zero reflection and "
                       "unit-power transmission propagated by one finite SAFE section. Reference refinement is a separate check. "
                       "Results apply to this uniform guide, frequency and incidence side."}
    output.parent.mkdir(parents=True, exist_ok=True)
    with TemporaryDirectory(prefix="uniform-reference-", dir=output.parent) as temporary:
        staging = Path(temporary)
        with (staging / TABLE).open("w", newline="", encoding="utf-8") as stream:
            writer = csv.DictWriter(stream, fieldnames=rows[0].keys())
            writer.writeheader()
            writer.writerows(rows)
        result["output_sha256"] = {TABLE: file_sha(staging / TABLE)}
        write_json(staging / saved.FILES[0], result)
        staging.rename(output)
    return {**result, "output_directory": str(output), "reused": False}


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", type=Path, required=True)
    parser.add_argument("--reference", type=Path, default=DEFAULT_REFERENCE)
    parser.add_argument("--reference-case", default="8x4", help="Section within the portable SAFE record (default: 8x4)")
    parser.add_argument("--output-root", type=Path)
    args = parser.parse_args(argv)
    try:
        result = compare_uniform_reference(args.run, args.reference, args.output_root, args.reference_case)
    except (OSError, ValueError, KeyError, TypeError, IndexError) as exc:
        print(f"Uniform reference comparison could not complete: {exc}", file=sys.stderr)
        return 2
    print(f"Compared: {result['output_directory']}")
    print(f"Uniform response: {'passed' if result['uniform_response_passed'] else 'failed'} "
          f"(max complex error {result['maximum_complex_error']:.6g}, power error {result['maximum_power_error']:.6g}).")
    print(f"SAFE reference phase refinement: {result['reference_phase_qualification']['status']}.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
