"""Localize a measured two-mesh reflection change at 1 GHz. Athul Prem | _AP_pnc."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import sys
from tempfile import NamedTemporaryFile

if __name__ == "__main__":
    os.environ.update(dict.fromkeys(("OPENBLAS_NUM_THREADS", "OMP_NUM_THREADS", "MKL_NUM_THREADS",
                                     "BLIS_NUM_THREADS", "NUMEXPR_NUM_THREADS"), "1"))

import numpy as np
from scipy.linalg import solve
from skfem import Dofs, ElementTetP2, ElementVector

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT), str(ROOT / "src")]
from tools import central_patch_refinement_AP_pnc as patch
from tools import map_scattering_AP_pnc as native
from tools import reflection_dual_AP_pnc as dual
from tools import regional_bilinear_work_AP_pnc as work
from tools import run_split_device_AP_pnc as split_reader
from tools import transfer_p2_field_AP_pnc as transfer
from tools.compare_scattering_AP_pnc import align, require

STUDY = ROOT / "results/runs/physical_thickness_device_AP_pnc"
OUTPUT = ROOT / "verification/regional_reflection_work_1GHz_AP_pnc.json"
CASES = {
    "thin_first": ("thin100_central400_edge225_rim32", "thin100_taper_patch225to168p75_1GHz"),
    "thin_finer": ("thin100_central400_edge168p75_rim32_linux", "thin100_taper_patch168p75to126p5625_1GHz"),
    "thick_first": ("thick200_central400_edge225_rim32", "thick200_taper_patch225to168p75_1GHz"),
}
ABSOLUTE_CLOSURE = 1e-7
SCHEMA_VERSION = 2
HOLE_BINS = (0, *range(9, 19))
HOLE_PAIRS = tuple((hole, 27 - hole) for hole in range(9, 14))
SCOPE = ("Signed contributions to measured finite changes on fixed physical geometry, conditional on "
         "the two P2 interpolation operators and grouping convention. Continuum error and local power flow "
         "remain undetermined by this calculation.")


def _read_case(directory):
    state = native.read_map_checkpoint(directory)
    cfg, shared = state["inputs"]["configuration"], state["shared"]
    point, = [p for p in state["points"] if p["frequency_GHz"] == 1 and p["status"] == "accepted"]
    folder, summary = native._read_accepted(point, cfg, shared, directory)
    flux = native._read(directory.with_name(directory.name + "_1GHz_flux_AP_pnc.json"))
    require(flux["status"] == "passed_saved_discretization_flux_replay" and flux["fields_sha256"] ==
            summary["output_sha256"][native.FIELDS] and flux["map_checkpoint_sha256"] == native._sha(directory / native.CHECKPOINT)
            and flux["point_summary"] == point["accepted"]["summary"] and summary["hole_count"] == 26
            and summary["incident_side"] == "left" and cfg["mesh"]["thickness_layers"] == 2
            and cfg["mesh"]["element_order"] == 2, "Expected the independently checked native 26-hole point.")
    records = shared["artifacts"]
    mesh_path = native._verified(records["mesh"], directory)
    manifest_path = native._verified(records["mesh_summary"], directory)
    require(manifest_path == mesh_path.with_name(native.RUN_SUMMARY), "Native mesh and manifest paths disagree.")
    mesh, manifest = native.workflow.load_checkpoint(mesh_path.parent, shared["mesh_hash"])
    section_path = native._verified(records["port_section"], directory)
    with np.load(section_path, allow_pickle=False) as arrays:
        section = {key: arrays[key] for key in ("M", "K0", "Q", "G", "points_m", "triangles", "left_dofs", "right_dofs")}
    field_names = ("displacements_per_sqrt_w", "port_displacements", "port_axial_forces", "port_left_dofs", "port_right_dofs",
                   "reflection_all_mode_indices", "transmission_all_mode_indices", "incident_mode_indices",
                   "reflection_mode_indices", "reflection_amplitudes", "transmission_amplitudes", "incident_powers_w")
    with np.load(folder / native.FIELDS, allow_pickle=False) as archive:
        dofs = Dofs(mesh, ElementVector(ElementTetP2()))
        require(np.array_equal(mesh.p.T, archive["points_m"]) and np.array_equal(mesh.t.T, archive["tetrahedra"])
                and np.array_equal(dofs.element_dofs, archive["element_dofs"]) and
                np.array_equal(dofs.nodal_dofs, archive["nodal_dofs"]), "Field and original P2 mesh ordering differ.")
        data = {k: archive[k] for k in field_names}
        xyz, components = archive["dof_coordinates_m"], archive["dof_component"]
        trace = {}
        for side in ("left", "right"):
            ids = data[f"port_{side}_dofs"]
            require(np.array_equal(ids, section[f"{side}_dofs"]), "Archived end-face indices differ.")
            trace[side] = (xyz[ids], components[ids])
    modes, _, powers = native._point_modes(folder, summary)
    return {"directory": directory, "mesh": mesh, "manifest": manifest, "configuration": cfg, "section": section,
            "trace": trace, "data": data, "modes": modes, "power": powers, "planes": shared["reference_planes_x_m"],
            "holes": state["inputs"]["hole_pattern"]["holes"],
            "pins": {"checkpoint": native._link(directory / native.CHECKPOINT, ROOT),
                     "point": native._link(folder / native.RUN_SUMMARY, ROOT), "mesh": native._link(mesh_path, ROOT),
                     "mesh_summary": native._link(manifest_path, ROOT),
                     "section": native._link(section_path, ROOT), "fields": native._link(folder / native.FIELDS, ROOT),
                     "flux": native._link(directory.with_name(directory.name + "_1GHz_flux_AP_pnc.json"), ROOT)}}


def _columns(mesh):
    xy, ids = np.unique(mesh.p[:2].T, axis=0, return_inverse=True)
    corners = np.sort(ids[mesh.t.T], axis=1)
    middle = np.where(corners[:, 1] > corners[:, 0], corners[:, 1], corners[:, 2])
    tri = np.column_stack((corners[:, 0], middle, corners[:, 3]))
    require(np.all(np.diff(tri, axis=1) > 0) and np.all(np.count_nonzero(np.diff(corners, axis=1), axis=1) == 2),
            "Every tetrahedron must span three distinct planar vertices.")
    triangles, owners = np.unique(tri, axis=0, return_inverse=True)
    return xy[triangles], owners


def _groups(coarse, fine):
    coarse_mesh, fine_mesh = coarse["mesh"], fine["mesh"]
    _, c_xy, c_footprint, c_idx = patch._footprint(coarse_mesh)
    c_tri = c_xy[c_footprint]
    # Hybrid meshes append inserted vertices after the original body. Recover
    # their footprint from coordinates and keep the original coarse ordinals.
    require(np.array_equal(coarse_mesh.p, fine_mesh.p[:, :coarse_mesh.nvertices]),
            "The hybrid mesh changed original coordinate indices.")
    f_tri, f_idx = _columns(fine_mesh)
    parents = patch._descendant_parents(c_tri, f_tri)
    changed = np.bincount(parents, minlength=len(c_tri)) > 1
    changed_ids = np.flatnonzero(changed)
    diagnostics = fine["manifest"]["geometry_diagnostics"]["taper_patch_refinement"]
    require(changed_ids.size == diagnostics["replaced_coarse_planar_triangles"],
            "Replaced parent count differs from the original refinement.")
    if "coarse_parent_partition" in diagnostics:
        require(np.array_equal(changed_ids, diagnostics["coarse_parent_partition"]["replaced_parent_indices"]),
                "Replaced parent IDs differ from the original refinement.")
    unchanged_coarse = patch._tetrahedron_rows(coarse_mesh.t[:, ~changed[c_idx]])
    unchanged_fine = patch._tetrahedron_rows(fine_mesh.t[:, ~changed[parents[f_idx]]])
    require(np.array_equal(unchanged_coarse, unchanged_fine), "Unchanged coarse and hybrid tetrahedra differ.")
    centers = np.array([hole["x_m"] for hole in coarse["holes"]])
    edges = (centers[1:] + centers[:-1]) / 2
    x = c_tri[:, :, 0]
    holes = np.searchsorted(edges, np.sort(x, axis=1).mean(axis=1)) + 1
    changed_holes = holes[changed_ids]
    require(np.isin(changed_holes, HOLE_BINS[1:]).all(), "Changed parents extend outside the central ten holes.")
    labels = np.where(changed, np.arange(len(c_tri)) + 1, 0)
    partition = {"coarse_parent_count": len(c_tri), "changed_parent_count": len(changed_ids),
                 "changed_parents_crossing_an_axial_bin_edge": int(np.count_nonzero(changed &
                     (np.searchsorted(edges, x.min(axis=1)) != np.searchsorted(edges, x.max(axis=1))))),
                 "ownership": "Zero-based original native bottom-footprint triangle order. All children "
                              "inherit the coarse planar triangle centroid's nearest axial hole center. "
                              "Group zero contains unchanged parents. Bin boundaries bisect adjacent hole centers."}
    return labels[c_idx], labels[parents[f_idx]], changed_ids, changed_holes, partition


def _end_forces(case, u, incident_columns):
    data = case["data"]
    us, fs = data["port_displacements"], data["port_axial_forces"]
    plus, minus = data["transmission_all_mode_indices"], data["reflection_all_mode_indices"]
    inc = data["incident_mode_indices"]
    left = u[data["port_left_dofs"]] - us[:, inc] @ incident_columns
    fleft = -fs[:, minus] @ solve(us[:, minus], left) - fs[:, inc] @ incident_columns
    fright = fs[:, plus] @ solve(us[:, plus], u[data["port_right_dofs"]])
    return fleft, fright


def _complex(a):
    return {"real": a.real.tolist(), "imag": a.imag.tolist()}


def _work_rows(parent_change, power_weight, parent_holes):
    def power_points(amplitudes):
        return -100 * np.sum((power_weight * amplitudes).real, axis=-2)

    def row(amplitude):
        values = power_points(amplitude)
        return {"delta_T_percentage_points_by_drive": values.tolist(),
                "equal_power_mean_delta_T_percentage_points": float(np.mean(values)),
                "reflection_amplitude_change": _complex(amplitude)}

    holes = np.r_[0, parent_holes]
    by_hole = {hole: parent_change[holes == hole].sum(axis=0) for hole in HOLE_BINS}
    bins = [{"hole": hole, **row(by_hole[hole])} for hole in HOLE_BINS]
    pairs = [{"holes": [left, right], **row(by_hole[left] + by_hole[right])} for left, right in HOLE_PAIRS]
    return power_points(parent_change), bins, pairs


def measure(coarse, fine, recovered):
    a, b = coarse["data"], fine["data"]
    require(coarse["configuration"]["material"] == fine["configuration"]["material"] and
            coarse["configuration"]["geometry"] == fine["configuration"]["geometry"] and coarse["holes"] == fine["holes"]
            and coarse["planes"] == fine["planes"] == [0.0, 16650e-9] and
            all(np.array_equal(x, y) for side in ("left", "right") for x, y in
                zip(coarse["trace"][side], fine["trace"][side], strict=True)), "Native physical inputs or traces differ.")
    for k in ("points_m", "triangles", "M", "K0", "G", "Q"):
        require(np.array_equal(coarse["section"][k], fine["section"][k]), "Native lead operators or topology differ.")
    mass = fine["section"]["M"]
    matches = {k: align(fine["modes"][k], coarse["modes"][k], mass, None, False)
               for k in ("incoming", "reflection")}
    order, phase, _ = matches["incoming"]
    out_order, out_phase, _ = matches["reflection"]
    change_input = np.eye(len(order))[:, order] / phase
    uc = a["displacements_per_sqrt_w"] @ change_input
    zf = recovered.duals
    forward = transfer.build_p2_transfer(coarse["mesh"], fine["mesh"])
    backward = transfer.build_p2_transfer(fine["mesh"], coarse["mesh"])
    pu, zc = forward.apply(uc), backward.apply(zf)
    rc = out_phase[:, None] * a["reflection_amplitudes"][np.ix_(out_order, order)] / phase
    rf = b["reflection_amplitudes"]
    trace_amp = recovered.extraction @ (pu[b["port_left_dofs"]] - b["port_displacements"][:, b["incident_mode_indices"]])
    require(np.max(abs(rc - trace_amp)) < ABSOLUTE_CLOSURE, "Coarse field and native reflection block disagree in the fine gauge.")
    cg, fg, changed_ids, changed_holes, ownership = _groups(coarse, fine)
    mat = coarse["configuration"]["material"]
    material = native.workflow.CubicMaterial(C11=mat["C11_Pa"], C12=mat["C12_Pa"], C44=mat["C44_Pa"],
                                            rho=mat["density_kg_m3"], rotation=mat["crystal_rotation"])
    wc = work.regional_bilinear_work(coarse["mesh"], material, uc, zc, cg, 1e9)
    wf = work.regional_bilinear_work(fine["mesh"], material, pu, zf, fg, 1e9)
    require(np.array_equal(wc.groups, np.r_[0, changed_ids + 1]) and np.array_equal(wc.groups, wf.groups),
            "Coarse and refined parent work groups differ.")
    contribution = wc.dynamic - wf.dynamic
    def boundary(case, u, z, inc):
        forces = _end_forces(case, u, inc)
        return sum(z[case["data"][f"port_{side}_dofs"]].T @ force
                   for side, force in zip(("left", "right"), forces, strict=True))
    port_term = boundary(fine, pu, zf, np.eye(len(order))) - boundary(coarse, uc, zc, change_input)
    amplitude_residual = rf - rc - contribution.sum(axis=0) - port_term
    power_out = fine["modes"]["reflection"].roots
    omega = 2 * np.pi * 1e9
    out_w = abs((omega / 2) * np.sum(b["port_displacements"][:, power_out].conj()
                                    * b["port_axial_forces"][:, power_out], axis=0).imag)
    power_weight = (rf + rc).conj() * out_w[:, None] / b["incident_powers_w"]
    # T and R sum to one for each propagating incident mode in this lossless system.
    signed_t, bins, pairs = _work_rows(contribution, power_weight, changed_holes)
    t_delta = 100 * (fine["power"]["transmission"].sum(axis=0) - coarse["power"]["transmission"].sum(axis=0)[order])
    r_delta = 100 * (fine["power"]["reflection"].sum(axis=0) - coarse["power"]["reflection"].sum(axis=0)[order])
    boundary_t = -100 * (power_weight * port_term).real.sum(axis=0)
    closure_t = t_delta - signed_t.sum(axis=0) - boundary_t
    closure = {"maximum_complex_amplitude_residual": float(np.max(abs(amplitude_residual))),
               "maximum_total_T_residual_percentage_points": float(np.max(abs(closure_t))),
               "maximum_measured_delta_T_plus_delta_R_percentage_points": float(np.max(abs(t_delta + r_delta))),
               "maximum_unchanged_parent_amplitude": float(np.max(abs(contribution[0]))),
               "maximum_port_correction_amplitude": float(np.max(abs(port_term)))}
    require(all(v < ABSOLUTE_CLOSURE for v in closure.values()), f"Finite difference identity failed: {closure}")
    parent_work = {"indices": changed_ids.tolist(), "hole_numbers": changed_holes.tolist(),
                   "delta_T_percentage_points_by_drive": signed_t[1:].tolist()}
    return {"original_planes_m": fine["planes"], "incident_roots": b["incident_mode_indices"].tolist(),
            "reflected_roots": b["reflection_mode_indices"].tolist(), "mode_alignment": {k: v[2] for k,v in matches.items()},
            "dual_equations": recovered.metrics, "coarse_to_refined_transfer": forward.diagnostics,
            "refined_to_coarse_transfer": backward.diagnostics, "partition": ownership, "closure": closure,
            "measured_delta_T_percentage_points": t_delta.tolist(), "measured_delta_R_percentage_points": r_delta.tolist(),
            "end_correction_delta_T_percentage_points": boundary_t.tolist(), "bins": bins,
            "parent_work": parent_work, "paired_hole_bins": pairs}


def _validate_result(result):
    parents = result["parent_work"]
    indices, holes, work_by_parent = (np.asarray(parents[key]) for key in (
        "indices", "hole_numbers", "delta_T_percentage_points_by_drive"))
    count = result["partition"]["changed_parent_count"]
    drive_count = len(result["incident_roots"])
    bins, pairs = result["bins"], result["paired_hole_bins"]
    closure = result["closure"]
    require(drive_count == len(result["reflected_roots"]) == 5 and count > 0
            and indices.size == count and indices.shape == holes.shape == (count,)
            and work_by_parent.shape == (count, drive_count) and np.isfinite(work_by_parent).all()
            and indices.dtype.kind in "iu" and indices[0] >= 0 and np.all(indices[1:] > indices[:-1])
            and indices[-1] < result["partition"]["coarse_parent_count"]
            and holes.dtype.kind in "iu" and np.isin(holes, HOLE_BINS[1:]).all()
            and tuple(row["hole"] for row in bins) == HOLE_BINS
            and tuple(tuple(row["holes"]) for row in pairs) == HOLE_PAIRS
            and closure and all(0 <= value < ABSOLUTE_CLOSURE for value in closure.values()),
            "The work receipt contains inconsistent parent ordinals, hole bins, drives, or closure.")
    by_hole = {row["hole"]: np.asarray(row["delta_T_percentage_points_by_drive"]) for row in bins}
    require(all(np.allclose(by_hole[hole], work_by_parent[holes == hole].sum(axis=0), atol=ABSOLUTE_CLOSURE, rtol=0)
                for hole in HOLE_BINS[1:])
            and all(np.allclose(row["delta_T_percentage_points_by_drive"], by_hole[left] + by_hole[right],
                                atol=ABSOLUTE_CLOSURE, rtol=0) for row, (left, right) in zip(pairs, HOLE_PAIRS, strict=True))
            and np.allclose(np.sum(list(by_hole.values()), axis=0) + result["end_correction_delta_T_percentage_points"],
                            result["measured_delta_T_percentage_points"], atol=ABSOLUTE_CLOSURE, rtol=0),
            "Parent, hole, or pair work fails the measured transmission closure.")


def _receipt(output, header, identity):
    if not output.exists():
        return {**header, "cases": {}}
    saved = native._read(output)
    require(saved.keys() == header.keys() | {"cases"} and all(saved[key] == value for key, value in header.items())
            and isinstance(saved["cases"], dict) and saved["cases"].keys() <= CASES.keys(),
            "The existing receipt has a different schema or scope.")
    for name, entry in saved["cases"].items():
        require(entry["inputs"]["code_environment"] == identity
                and entry["result_sha256"] == native.workflow.digest(entry["result"]),
                "The existing work uses a different environment or its result changed.")
        for role, directory_name in zip(("coarse", "refined"), CASES[name], strict=True):
            directory = STUDY / directory_name
            links = entry["inputs"][role]
            require(native._verified(links["checkpoint"], ROOT) == directory / native.CHECKPOINT,
                    "The work receipt points to the wrong native checkpoint.")
            for kind, link in links.items():
                if kind != "checkpoint":
                    native._verified(link, ROOT)
        _validate_result(entry["result"])
    return saved


def run(case, output):
    paths = [STUDY / name for name in CASES[case]]
    coarse, fine = [_read_case(p) for p in paths]
    origin = native.workflow.source_identity()
    identity = native._numerical_identity(origin)
    for module in (patch, dual, work, transfer, native, native.comparison, split_reader):
        file = Path(module.__file__).resolve()
        identity["source_sha256"][file.relative_to(ROOT).as_posix()] = native._sha(file)
    identity["source_sha256"][Path(__file__).resolve().relative_to(ROOT).as_posix()] = native._sha(Path(__file__))
    identity["thread_environment"] = {key: os.getenv(key) for key in native.THREADS}
    inputs = {"coarse": coarse["pins"], "refined": fine["pins"], "code_environment": identity}
    header = {**native.AUTHOR, **native.CONVERGENCE, "schema_version": SCHEMA_VERSION, "frequency_hz": 1e9,
              "absolute_closure_limit": ABSOLUTE_CLOSURE, "scope": SCOPE}
    with native._exclusive(output.with_suffix(".lock")):
        result = _receipt(output, header, identity)
        if case in result["cases"]:
            old = result["cases"][case]
            require(old["inputs"] == inputs, "The existing record belongs to different native inputs.")
            return old["result"]
        with native.workflow.measured_resources() as resources:
            recovered = dual.recover_saved_reflection_dual(paths[1], 1e9)
            require(recovered.passed, f"Fine transpose equations failed: {recovered.metrics}")
            measured = measure(coarse, fine, recovered)
        _validate_result(measured)
        result["cases"][case] = {"inputs": inputs, "result": measured, "result_sha256": native.workflow.digest(measured),
                                 "resources": resources, "source_git_commit": origin["git_commit"]}
        with NamedTemporaryFile("w", encoding="utf-8", newline="\n", dir=output.parent, delete=False) as stream:
            stream.write(json.dumps(result, indent=2, allow_nan=False) + "\n")
        Path(stream.name).replace(output)
        return measured


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--case", choices=CASES, required=True)
    parser.add_argument("--output", type=Path, default=OUTPUT)
    args = parser.parse_args()
    destination = args.output.resolve()
    require(not destination.is_relative_to(STUDY) and destination.suffix == ".json",
            "Choose a JSON receipt outside the saved device maps.")
    record = run(args.case, destination)
    print(json.dumps({"case": args.case, "closure": record["closure"], "output": str(destination)}))
