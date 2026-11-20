"""Compare fixed-frequency waves on three meshes of the periodic mirror cell.

Athul Prem | _AP_pnc
"""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import sys
from tempfile import NamedTemporaryFile

if __name__ == "__main__":
    os.environ.update(dict.fromkeys(("OPENBLAS_NUM_THREADS", "OMP_NUM_THREADS", "MKL_NUM_THREADS", "BLIS_NUM_THREADS"), "1"))

import numpy as np
from scipy.optimize import linear_sum_assignment
from scipy.spatial import cKDTree

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT), str(ROOT / "src")]

from phononic.lead_modes_AP_pnc import solve_bloch
from tools import map_scattering_AP_pnc as native
from tools import run_mirror_cell_phase_AP_pnc as cell
from tools.compare_axial_device_power_AP_pnc import _match_evidence
from tools.compare_scattering_AP_pnc import MIN_OVERLAP, K_GAP_TOL, require

STUDY = ROOT / "results/runs/mirror_cell_phase_AP_pnc"
OUTPUT = ROOT / "verification/mirror_cell_phase_AP_pnc.json"
TARGETS = (225., 168.75, 126.5625)
FREQUENCY = 1e9
PERIOD_M = 550e-9


def trace_order(source_coordinates, source_components, target_coordinates, target_components):
    """Return source row indices in target order for the same physical P2 trace."""
    require(source_coordinates.shape == target_coordinates.shape and source_coordinates.shape[1] == 2,
            "The transverse traces differ in size.")
    span = np.ptp(target_coordinates, axis=0)
    result = np.empty(len(target_components), dtype=int)
    for component in range(3):
        src = np.flatnonzero(source_components == component)
        dst = np.flatnonzero(target_components == component)
        require(len(src) == len(dst) and len(src) > 0, "The Cartesian trace components differ.")
        distance, order = cKDTree(source_coordinates[src] / span).query(target_coordinates[dst] / span)
        require(np.max(distance) < 1e-10 and len(np.unique(order)) == len(dst),
                "A transverse P2 node differs between the paired traces.")
        result[dst] = src[order]
    return result


def overlaps(first, second, mass):
    mf, ms = mass @ first, mass @ second
    a = np.sum(first.conj() * mf, axis=0).real
    b = np.sum(second.conj() * ms, axis=0).real
    require(np.all(a > 0) and np.all(b > 0), "A transverse field has invalid modal mass.")
    values = np.abs(first.conj().T @ ms) / np.sqrt(a[:, None] * b)
    require(np.isfinite(values).all() and np.max(values, initial=0.) <= 1 + 1e-10,
            "An overlap exceeds its mass-normalized bound.")
    return np.minimum(1., values)


def compare_edge(coarse, fine):
    a, b = coarse["arrays"], fine["arrays"]
    order = trace_order(b["trace_dof_yz_m"], b["trace_dof_components"],
                        a["trace_dof_yz_m"], a["trace_dof_components"])
    m = a["trace_mass"]
    require(np.allclose(m, b["trace_mass"][np.ix_(order, order)], rtol=0,
                        atol=1e-12 * np.max(np.abs(m))), "The trace mass or triangulation changed.")
    rows = []
    for direction in ("right", "left"):
        ai, bi = (np.flatnonzero(d["directions"] == direction) for d in (a, b))
        sims = overlaps(a["traces"][:, ai], b["traces"][order][:, bi], m)
        used_a, used_b = set(), set()
        for i, j in zip(*linear_sum_assignment(-sims), strict=True):
            ia, jb = ai[i], bi[j]
            used_a.add(ia)
            used_b.add(jb)
            score, competitor = _match_evidence(sims, i, j)
            matched = score >= MIN_OVERLAP and score > competitor + K_GAP_TOL
            row = {"direction": direction, "reference_root": int(a["root_indices"][ia]),
                   "candidate_root": int(b["root_indices"][jb]), "mass_overlap": score,
                   "largest_competing_overlap": competitor, "status": "matched" if matched else "unresolved"}
            if matched:
                phase_a, phase_b = (float(np.angle(d["lambda_factors"][index])) for d, index in ((a, ia), (b, jb)))
                delta = float(np.angle(b["lambda_factors"][jb] / a["lambda_factors"][ia]))
                row.update(reference_phase_rad=phase_a, candidate_phase_rad=phase_b,
                           phase_change_rad_per_period=delta, phase_change_degrees_per_period=float(np.degrees(delta)))
            rows.append(row)
        for data, indices, used, key in ((a, ai, used_a, "reference_root"), (b, bi, used_b, "candidate_root")):
            rows.extend({"direction": direction, key: int(data["root_indices"][idx]), "status": "unpaired"}
                        for idx in indices if idx not in used)
    return {"reference_target_nm": coarse["target_nm"], "candidate_target_nm": fine["target_nm"], "modes": rows}


def forward_frequency_check(thickness, case, root):
    """Use the existing real-wavevector 3D eigensolver on the finest finite cell."""
    data = case["arrays"]
    index, = np.flatnonzero(data["root_indices"] == root)
    k = float(data["k_rad_m"][index])
    system = cell.assemble_case(thickness, case["target_nm"])
    independent = solve_bloch(system, k, num_modes=12)
    nearest = int(np.argmin(abs(independent.frequencies_hz - FREQUENCY)))
    # These indices bind the rebuilt cell to the saved face before comparing fields.
    face = data["left_dofs"]
    require(np.allclose(system.basis.doflocs[1:, face].T, data["trace_dof_yz_m"], atol=1e-17, rtol=0)
            and np.all(np.abs(system.basis.doflocs[0, face]) < 1e-17), "The rebuilt face has different coordinates.")
    frequency = float(independent.frequencies_hz[nearest])
    trace = independent.modes_full[face, nearest:nearest + 1]
    similarity = float(overlaps(data["traces"][:, index:index + 1], trace, data["trace_mass"])[0, 0])
    error = abs(frequency / FREQUENCY - 1)
    require(error < 1e-7 and similarity >= MIN_OVERLAP, "The independent real-k calculation did not recover the 1 GHz field.")
    return {"root": root, "k_rad_m": k, "recovered_frequency_hz": frequency, "relative_frequency_error": error,
            "mass_overlap_on_left_face": similarity, "frequency_eigenpair_backward_residual":
                float(independent.relative_residuals[nearest]), "frequencies_requested": 12}


def read_case(thickness, target):
    folder = cell.case_directory(STUDY, thickness, target)
    summary_path, array_path = folder / "summary_AP_pnc.json", folder / "traces_AP_pnc.npz"
    summary = native._read(summary_path)
    inputs = summary["inputs"]
    require(native._verified(summary["trace_arrays"], folder) == array_path
            and native.workflow.digest(inputs) == summary["inputs_sha256"]
            and inputs["frequency_hz"] == FREQUENCY and inputs["geometry"]["period_nm"] == 550
            and inputs["geometry"]["thickness_nm"] == thickness and inputs["mesh"]["central_target_nm"] == target
            and inputs["mesh"]["thickness_layers"] == 2, "The saved cell inputs or arrays differ.")
    source_hashes = inputs["numerical_identity"]["source_sha256"]
    for name in ("tools/bloch_factors_AP_pnc.py", "tools/run_mirror_cell_phase_AP_pnc.py",
                 "src/phononic/lead_modes_AP_pnc.py", "src/phononic/refinement/central_AP_pnc.py"):
        require(native._sha(ROOT / name) == source_hashes[name], f"The cell source changed: {name}.")
    for key in ("hole_table", "source_device_config"):
        native._verified(inputs[key], ROOT)
    keys = ("trace_mass", "trace_dof_yz_m", "trace_dof_components", "traces", "root_indices",
            "lambda_factors", "k_rad_m", "signed_flux_w", "directions", "left_dofs")
    with np.load(array_path, allow_pickle=False) as stored:
        arrays = {key: stored[key] for key in keys}
        require(stored["inputs_sha256"].item() == summary["inputs_sha256"], "Saved cell inputs differ.")
    require(np.allclose(arrays["k_rad_m"], np.angle(arrays["lambda_factors"]) / PERIOD_M,
                        atol=1e-5, rtol=1e-12), "A saved Bloch phase differs from its principal wavenumber.")
    for direction, sign in (("right", 1), ("left", -1)):
        selected = arrays["directions"] == direction
        require(np.all(sign * arrays["signed_flux_w"][selected] > 0), "A wave has the wrong flux sign.")
    return {"target_nm": target, "summary": native._link(summary_path, ROOT), "arrays": arrays,
            "status": summary["status"], "mesh": summary["mesh"], "inputs": inputs}


def compare(thickness):
    levels = [read_case(thickness, h) for h in TARGETS]
    baseline = levels[0]["inputs"]
    for case in levels:
        require(case["inputs"] == {**baseline, "mesh": {**baseline["mesh"], "central_target_nm": case["target_nm"]}}
                and all(case["mesh"][key] == levels[0]["mesh"][key] for key in
                        ("frozen_rim_sha256", "translated_end_triangles_sha256")),
                "The mesh comparison changed additional physical inputs, end meshes, or hole rims.")
    require(all(a["mesh"]["planar_triangles"] < b["mesh"]["planar_triangles"]
                for a, b in zip(levels[:-1], levels[1:], strict=True)), "Cells must be distinct mesh levels.")
    edges = [compare_edge(a, b) for a, b in zip(levels[:-1], levels[1:], strict=True)]
    matched = [r for r in edges[-1]["modes"] if r["direction"] == "right" and r["status"] == "matched"]
    check = None
    if matched:
        largest = max(matched, key=lambda r: abs(r["phase_change_rad_per_period"]))
        check = forward_frequency_check(thickness, levels[-1], largest["candidate_root"])
    return {"physical_thickness_nm": thickness, "cases": [
                {"source": v["summary"], "case_status": v["status"], "central_target_nm": v["target_nm"],
                 "body_dofs": v["mesh"]["quadratic_vector_dofs"], "signed_mode_counts":
                     {s: int(np.count_nonzero(v["arrays"]["directions"] == s)) for s in ("right", "left", "unresolved")}}
                for v in levels],
            "adjacent_mesh_changes": edges, "independent_check_of_largest_matched_rightward_change": check}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    result = {**native.AUTHOR, **native.CONVERGENCE, "schema_version": 1, "frequency_hz": FREQUENCY,
              "cell_period_m": PERIOD_M, "axial_planes_m": [0., PERIOD_M],
              "minimum_mass_overlap": MIN_OVERLAP, "minimum_competitor_margin": K_GAP_TOL,
              "by_thickness": {str(nm): compare(nm) for nm in (100, 200)},
              "source": native._link(Path(__file__).resolve(), ROOT),
              "scope": "Each phase is the principal argument of the finite-cell Bloch multiplier per period. "
                       "Each mesh change is measured modulo 2 pi with direction set by signed right-face power. "
                       "The periodic mirror omits the tapers, central cavity and finite-device end leads. "
                       "These phase changes do not assign a phase error to the 26-hole device."}
    body = json.dumps(result, indent=2, allow_nan=False) + "\n"
    if args.output is None:
        print(body, end="")
        return
    output = args.output.resolve()
    require(output == OUTPUT or not output.exists(), "Choose a new file or the mirror-cell comparison.")
    require(not output.is_relative_to(STUDY), "Keep comparison output outside the original cell archive.")
    with NamedTemporaryFile("w", encoding="utf-8", newline="\n", dir=output.parent, delete=False) as temp:
        temp.write(body)
    Path(temp.name).replace(output)


if __name__ == "__main__":
    main()
