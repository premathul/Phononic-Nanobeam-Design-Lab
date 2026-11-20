#!/usr/bin/env python3
"""Compare the saved 1 GHz regional mesh responses. Author: Athul Prem | _AP_pnc."""

from __future__ import annotations

import argparse
from dataclasses import dataclass
import json
from math import fsum
from pathlib import Path
import runpy
import sys
from tempfile import NamedTemporaryFile

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if __name__ == "__main__":
    sys.path.insert(0, str(ROOT))

from tools import central_patch_refinement_AP_pnc as patch
from tools import compare_axial_nested_mesh_AP_pnc as nested
from tools import compare_fixed_wall_response_AP_pnc as wall
from tools import compare_scattering_AP_pnc as scattering
from tools import compare_thin_layer_response_AP_pnc as layer
from tools import map_scattering_AP_pnc as native
from tools.compare_scattering_AP_pnc import require


STUDY = ROOT / "results/runs/physical_thickness_device_AP_pnc"
PRIOR = ROOT / "verification/axial_device_nested_central_AP_pnc.json"
FINER_PRIOR = ROOT / "verification/thin_device_nested_response_AP_pnc.json"
OUTPUT = ROOT / "verification/taper_patch_response_1GHz_AP_pnc.json"
FINER_OUTPUT = ROOT / "verification/finer_taper_patch_response_1GHz_AP_pnc.json"
FREQUENCY_GHZ = 1.0
MAPS = {
    "baseline": ("thin100_central400_edge225_rim32", 2, 225.0),
    "taper_patch": ("thin100_taper_patch225to168p75_1GHz", 1, 225.0),
    "full_refinement": ("thin100_central400_edge168p75_rim32_linux", 2, 168.75),
}
EDGES = (("baseline", "taper_patch"), ("taper_patch", "full_refinement"), ("baseline", "full_refinement"))
COMPLEMENT_MAP = ("thin100_complement225to168p75_1GHz", 1, 225.0)
COMPLEMENT_EDGES = (("baseline", "complement"), ("complement", "full_refinement"))
FINER_MAPS = {
    "finer_center": ("thin100_taper_patch168p75to126p5625_1GHz", 1, 168.75),
    "finer_full": ("thin100_central400_edge126p5625_rim32_linux", 1, 126.5625),
}
FINER_COMPLEMENT_MAP = ("thin100_complement168p75to126p5625_1GHz", 1, 168.75)
CENTRAL_REGION = ("central", "nested_coarse_mirror_fine_taper_and_cavity", "thin100_taper_patch_AP_pnc")
COMPLEMENT_REGION = ("complement", "nested_coarse_taper_and_cavity_fine_complement", "thin100_complement_patch_AP_pnc")
CONTROL_REGIONS = {"taper_patch": CENTRAL_REGION, "complement": COMPLEMENT_REGION,
                   "finer_center": CENTRAL_REGION, "finer_complement": COMPLEMENT_REGION}
ORIGINAL_CENTRAL_REFINER_SHA256 = "ab330477d1c205d0fc727c734a1453bdcd264aea56f06b660fe42af443a331c4"
FINER_CENTER_MESH_ARRAY_SHA256 = "45953b964bb93719f67cf79922b358e724c04417031ea394ab5d50998f1b3c67"
SECTION_MATRICES = ("K0", "M", "Q", "G", "reference_speed_m_s")
SECTION_TOLERANCE = 1e-12
GAUGE_CLOSURE_TOLERANCE = 1e-10
REGIONAL_EDGES = {
    "central_at_coarse_exterior": ("baseline", "taper_patch"),
    "central_at_refined_exterior": ("complement", "full_refinement"),
    "complement_at_coarse_center": ("baseline", "complement"),
    "complement_at_refined_center": ("taper_patch", "full_refinement"),
}


@dataclass(frozen=True)
class Response:
    record: dict
    config: dict
    pattern: dict
    checkpoint: dict
    manifest: dict
    mesh: object
    summary: dict
    fixed: dict
    section: dict
    trace: dict
    modal_data: tuple
    transmission_eigenchannels: dict
    reflection_eigenvalues: np.ndarray


def _same_pin(first, second):
    return all(first[key] == second[key] for key in ("sha256", "bytes"))


def _read_section_and_trace(section_path, field_path, mesh, planes, port_dofs):
    with np.load(section_path, allow_pickle=False) as data:
        arrays = {key: data[key] for key in ("points_m", "triangles", "left_dofs", "right_dofs", *SECTION_MATRICES)}
    with np.load(field_path, allow_pickle=False) as fields:
        require(np.array_equal(fields["points_m"], mesh.p.T) and np.array_equal(fields["tetrahedra"], mesh.t.T),
                f"Saved field vertices or tetrahedra differ from the accepted mesh in {field_path}.")
        coordinates, components = fields["dof_coordinates_m"], fields["dof_component"]
        trace = {}
        for end, plane in zip(("left", "right"), planes, strict=True):
            dofs = arrays[f"{end}_dofs"]
            require(dofs.shape == (port_dofs,) and dofs.dtype.kind in "iu" and dofs.min() >= 0
                    and dofs.max() < len(coordinates) and len(np.unique(dofs)) == port_dofs
                    and np.array_equal(dofs, fields[f"port_{end}_dofs"])
                    and np.all(coordinates[dofs, 0] == plane),
                    f"The {end} section DOF map differs from its archived field trace at {field_path}.")
            trace[end] = (coordinates[dofs, 1:], components[dofs])
    require(all(np.array_equal(a, b) for a, b in zip(trace["left"], trace["right"], strict=True)),
            f"The local component and coordinate trace order differs between ends at {field_path}.")
    return arrays, trace


def _read_response(study, role, prior, read_point, *, finer_archive=None):
    maps = {**MAPS, "complement": COMPLEMENT_MAP, **FINER_MAPS, "finer_complement": FINER_COMPLEMENT_MAP}
    name, index, target = maps[role]
    directory = (study / name).resolve()
    flux_path = (study / f"{name}_1GHz_flux_AP_pnc.json").resolve()
    require(directory.parent == flux_path.parent == study, f"Unexpected archive location for {role}.")
    require((directory / native.CHECKPOINT).is_file() and flux_path.is_file(),
            f"The {role} response requires a native checkpoint in {directory} and its flux replay at {flux_path}.")
    checkpoint = native.read_map_checkpoint(directory)
    point, = (row for row in checkpoint["points"] if row["frequency_GHz"] == FREQUENCY_GHZ)
    require(point["index"] == index, f"Expected point {index} for {role}.")
    record, config, pattern = read_point(directory, flux_path, FREQUENCY_GHZ)
    run_dir, summary, fixed, _, section_path = nested._channel_input(record)
    require(config["mesh"] == {**prior["parent_mesh"], "central_max_edge_nm": target}
            and record["physical_thickness_nm"] == 100 and record["reference_planes_x_m"] == prior["reference_planes_x_m"]
            and config["resources"].get("linear_solver", "direct") == "direct"
            and config["analysis"].get("incident_side", "left") == prior["incident_side"] == "left"
            and summary["num_incident_channels"] == len(record["native_drives"]) == layer.CHANNEL_COUNT,
            f"Unexpected saved device, section, frequency or solver settings at {directory}.")
    if role not in CONTROL_REGIONS:
        if finer_archive is None:
            published = next(row for row in prior["levels"] if row["central_max_edge_nm"] == target)
            archive = published["samples"]["1GHz"]["thin"]["archive"]
        else:
            archive = finer_archive
        require(all(_same_pin(record["archive"][key], pin) for key, pin in archive.items()),
                f"The {role} input differs from its published 1 GHz archive.")
    else:
        require(checkpoint["inputs"]["frequencies_GHz"] == [FREQUENCY_GHZ]
                and "preparation_reuse" not in checkpoint and config["name"] == CONTROL_REGIONS[role][2],
                f"Expected the native single-point {role} response.")
        if role == "finer_center":
            require(record["body_dofs"] == 163425, "The finer central device has a different displacement space.")

    flux = native._read(flux_path)
    require([flux[key] for key in ("replay_source_sha256", "regional_replay_source_sha256")]
            == [pin["sha256"] for pin in prior["provenance"]["independent_flux_sources"]],
            f"The independent flux replay sources differ from the accepted controls at {flux_path}.")
    shared = checkpoint["shared"]
    manifest_path = native._verified(shared["artifacts"]["mesh_summary"], directory)
    mesh, manifest = native.workflow.load_checkpoint(manifest_path.parent, shared["mesh_hash"])
    require(manifest["mesh_sha256"] == record["archive"]["mesh"]["sha256"] and manifest["ndof"] == record["body_dofs"],
            f"The accepted mesh differs from the preflight at {directory}.")
    record["archive"]["mesh_summary"] = native._link(manifest_path, ROOT)
    record["point_index"] = index
    record["original_equation_identity"] = {key: flux["original_equations_saved"][key]
                                            for key in ("operator_csc_sha256", "rhs_sha256")}
    identity = checkpoint["inputs"]["numerical_identity"]
    assembly = checkpoint.get("preparation_reuse", {}).get("receipt", {}).get("donor_numerical_identity", identity)
    record["platform"] = {"assembly": assembly["platform"], "solve": identity["platform"]}
    section, trace = _read_section_and_trace(section_path, native._verified(record["archive"][native.FIELDS], ROOT),
                                           mesh, record["reference_planes_x_m"], record["port_dofs"])
    data = scattering.read_scattering(run_dir, summary, fixed)
    eigen = layer._transmission_eigenchannels(record, data, flux, prior["original_numerical_limits"])
    reflected = np.linalg.svd(data[2]["reflection"], compute_uv=False)**2
    require(abs(reflected.sum() - sum(row["reflection"] for row in record["native_drives"])) < layer.NATIVE_POWER_ROUNDOFF,
            f"The reflection eigenvalue sum differs from the native power totals at {directory}.")
    return Response(record, config, pattern, checkpoint, manifest, mesh, summary, fixed, section, trace, data, eigen, reflected)


def _section_match(reference, candidate):
    a, b = reference.section, candidate.section
    require(all(np.array_equal(a[key], b[key]) for key in ("points_m", "triangles"))
            and all(np.array_equal(x, y) for end in reference.trace for x, y in zip(reference.trace[end], candidate.trace[end], strict=True)),
            "The native section coordinates, triangles or local P2 trace order differ across meshes.")
    size = reference.record["port_dofs"]
    differences = {}
    for key in SECTION_MATRICES:
        shape = () if key == "reference_speed_m_s" else (size, size)
        require(a[key].shape == b[key].shape == shape and np.isfinite(a[key]).all() and np.isfinite(b[key]).all(),
                f"Invalid section matrix or scalar {key}.")
        scale = float(np.max(abs(a[key])))
        error = float(np.max(abs(a[key] - b[key])))
        require(error <= SECTION_TOLERANCE * scale, f"The matched lead section changed in {key}.")
        differences[key] = error / scale if scale else 0.0
    return differences


def _surface_signature(mesh):
    faces, rim = patch._surface(mesh)
    signature = {}
    for name, ids in (("xleft", faces["xleft"]), ("xright", faces["xright"]), ("hole_rims", rim)):
        if name != "hole_rims":
            require(np.array_equal(np.sort(ids), np.sort(mesh.boundaries[name])), f"Invalid native boundary tag {name}.")
        points = mesh.p.T[mesh.facets[:, ids].T]
        signature[name] = frozenset(tuple(sorted(map(tuple, face))) for face in points)
        require(len(signature[name]) == len(ids) > 0, f"The accepted mesh has empty or duplicate {name} facets.")
    return signature


def _control_geometry(baseline, candidate, full, role):
    region, method, _ = CONTROL_REGIONS[role]
    donors = candidate.checkpoint["taper_patch_donors"]
    diag = candidate.manifest["geometry_diagnostics"]["taper_patch_refinement"]
    control = candidate.pattern["taper_patch_refinement"]
    cut_planes = [(baseline.pattern["holes"][i]["x_m"] + baseline.pattern["holes"][j]["x_m"]) / 2
                  for i, j in ((7, 8), (17, 18))]
    require(control == candidate.manifest["mesh_inputs"]["taper_patch_refinement"]
            and control["donors"] == donors and native.workflow.digest(candidate.manifest["mesh_inputs"]) == candidate.record["mesh_hash"]
            and diag["method"] == method and diag.get("region", "central") == region
            and control.get("refined_region", "central") == candidate.checkpoint.get("refined_region", "central") == region
            and diag["nominal_patch_cuts_x_m"] == donors["nominal_mirror_cuts_x_m"] == cut_planes
            and diag["reference_planes_x_m"] == candidate.record["reference_planes_x_m"]
            and donors["hole_pattern_sha256"] == baseline.pattern["source_sha256"]
            and {key: value for key, value in candidate.pattern.items() if key != "taper_patch_refinement"}
            == baseline.pattern == full.pattern,
            f"The {role} checkpoint does not bind the declared donor geometry and taper cuts.")
    refiner_pin = native._link(Path(patch.__file__).resolve(), ROOT)
    archived_sha = diag["control_source_sha256"]
    historical = role == "taper_patch" and archived_sha == ORIGINAL_CENTRAL_REFINER_SHA256
    require(candidate.checkpoint["inputs"]["numerical_identity"]["source_sha256"][refiner_pin["path"]] == archived_sha
            and (historical or archived_sha == refiner_pin["sha256"]),
            f"The {role} checkpoint uses a different footprint refiner.")
    for name, case in (("coarse", baseline), ("fine", full)):
        require(donors[name]["mesh_hash"] == case.record["mesh_hash"]
                and donors[name]["source_git_commit"] == case.record["source_git_commit"]
                and donors[name]["mesh"] == case.checkpoint["shared"]["artifacts"]["mesh"]
                and donors[name]["mesh_summary"] == case.checkpoint["shared"]["artifacts"]["mesh_summary"]
                and _same_pin(donors[name]["checkpoint"], case.record["archive"]["checkpoint"])
                and native._array_digest(case.mesh.p, case.mesh.t) == diag[f"{name}_coordinates_and_connectivity_sha256"],
                f"The hybrid footprint refers to a different {name} donor.")
    require(native._array_digest(candidate.mesh.p, candidate.mesh.t)
            == diag["control_coordinates_and_connectivity_sha256"] == control["mesh_geometry_connectivity_sha256"],
            "The accepted hybrid mesh coordinates or connectivity differ from the footprint check.")
    if role == "finer_center":
        require(control["mesh_geometry_connectivity_sha256"] == FINER_CENTER_MESH_ARRAY_SHA256,
                "The finer central mesh differs from its preflight array hash.")
    if region == "central":
        def exterior_tetrahedra(mesh):
            x = mesh.p[0, mesh.t]
            return mesh.p[:, mesh.t[:, (x.min(axis=0) < cut_planes[0]) | (x.max(axis=0) > cut_planes[1])]]

        require(np.array_equal(exterior_tetrahedra(baseline.mesh), exterior_tetrahedra(candidate.mesh)),
                "The taper patch changes a coarse cell crossing or outside its nominal cuts.")
    surface = _surface_signature(baseline.mesh)
    require(all(_surface_signature(case.mesh) == surface for case in (candidate, full))
            and len(surface["hole_rims"]) == diag["unchanged_rim_facets"],
            "The 26-hole rim or original end face counts or coordinates differ across meshes.")
    coarse_cells, control_cells, fine_cells = (_tetrahedron_keys(case.mesh) for case in (baseline, candidate, full))
    removed = np.setdiff1d(coarse_cells, control_cells, assume_unique=True)
    added = np.setdiff1d(control_cells, coarse_cells, assume_unique=True)
    require(not np.intersect1d(removed, fine_cells, assume_unique=True).size
            and not np.setdiff1d(added, fine_cells, assume_unique=True).size
            and removed.size == diag["replaced_coarse_planar_triangles"] * 3 * (len(diag["z_planes_m"]) - 1)
            and added.size == diag["inserted_fine_donor_tetrahedra"],
            f"The {role} tetrahedra do not match their declared coarse-to-fine donor replacements.")
    metrics = {key: diag[key] for key in ("nominal_patch_cuts_x_m", "changed_axial_extent_x_m", "z_planes_m",
                                      "coarse_planar_triangles", "control_planar_triangles", "fine_donor_planar_triangles",
                                      "replaced_coarse_planar_triangles", "retained_coarse_tetrahedra",
                                      "inserted_fine_donor_tetrahedra", "unchanged_patch_frontier_p2_dofs")}
    return {**metrics, "refined_region": region, "archived_refiner_source_sha256": archived_sha,
        **({"identical_outer_tetrahedra_to_baseline": True} if region == "central" else {}),
        "identical_rim_and_end_face_counts": {name: len(faces) for name, faces in surface.items()},
        "exact_removed_coarse_and_added_fine_tetrahedra": True,
        "exact_field_and_mesh_coordinates_and_connectivity": True,
        "identical_section_geometry_and_local_trace_order": True,
    }


def _tetrahedron_keys(mesh):
    """Physical tetrahedra independent of volume numbering and local orientation."""
    cells = mesh.p.T[mesh.t.T].astype("<f8", copy=False)
    order = np.lexsort((cells[..., 2], cells[..., 1], cells[..., 0]), axis=1)
    cells = np.ascontiguousarray(np.take_along_axis(cells, order[..., None], axis=1))
    cells[cells == 0] = 0
    keys = np.unique(cells.reshape(len(cells), -1).view("V96").ravel())
    require(len(keys) == mesh.nelements, "Duplicate physical tetrahedra in a refinement case.")
    return keys


def _complement_coverage(cases):
    baseline, central, complement, full = (_tetrahedron_keys(cases[role].mesh)
                                           for role in ("baseline", "taper_patch", "complement", "full_refinement"))
    coverage = {}
    for name, pairs in (
        ("removed_coarse_tetrahedra", ((baseline, central), (baseline, complement), (baseline, full))),
        ("added_fine_tetrahedra", ((central, baseline), (complement, baseline), (full, baseline))),
    ):
        first, second, expected = (np.setdiff1d(a, b, assume_unique=True) for a, b in pairs)
        recombined = np.union1d(first, second)
        require(not np.intersect1d(first, second, assume_unique=True).size
                and np.array_equal(recombined, expected),
                f"The two controls do not form the complete disjoint set of {name}.")
        coverage[name] = {"central": len(first), "complement": len(second), "full": len(expected),
                          "recombined_and_full_array_sha256": native._array_digest(recombined)}
    diag = cases["complement"].manifest["geometry_diagnostics"]["taper_patch_refinement"]
    partition = diag["coarse_parent_partition"]
    central_ids, complement_ids = (set(partition[f"{role}_parent_indices"]) for role in ("central", "complement"))
    replaced = set(partition["replaced_parent_indices"])
    tets_per_parent = 3 * (len(diag["z_planes_m"]) - 1)
    require(central_ids.isdisjoint(complement_ids)
            and central_ids | complement_ids == set(range(diag["coarse_planar_triangles"]))
            and replaced <= complement_ids and len(replaced) == diag["replaced_coarse_planar_triangles"]
            and len(replaced) * tets_per_parent == coverage["removed_coarse_tetrahedra"]["complement"]
            and diag["inserted_fine_donor_tetrahedra"] == coverage["added_fine_tetrahedra"]["complement"]
            and partition["recombined_tetrahedra"] == len(full)
            and all(scattering.is_sha256(partition[key]) for key in (
                "central_active_mask_sha256", "complement_active_mask_sha256", "fine_donor_topology_sha256")),
            "The complement mesh preflight does not identify a complete coarse partition and fine recombination.")
    return {**coverage, "recombined_tetrahedra_match_full_refinement": True,
            "full_fine_physical_tetrahedron_set_sha256": native._array_digest(full),
            "coarse_parent_counts": {"central": len(central_ids), "complement": len(complement_ids)},
            "refined_cut_crossing_planar_triangles": partition["replaced_cut_crossing_planar_triangles"]}


def _matched_edge(reference, candidate):
    scattering.compatible_inputs(reference.summary, candidate.summary, reference.fixed, candidate.fixed)
    require(all(reference.config[key] == candidate.config[key] for key in ("geometry", "material", "analysis")),
            "The fixed geometry, material or analysis changed across the mesh edge.")
    require(reference.checkpoint["inputs"]["numerical_identity"]["packages"]
            == candidate.checkpoint["inputs"]["numerical_identity"]["packages"], "Numerical package versions differ.")
    section_changes = _section_match(reference, candidate)
    matches, blocks, rows = scattering.compare_blocks(reference.modal_data, candidate.modal_data)
    return matches, blocks, rows, section_changes


def _edge(reference, candidate):
    matches, blocks, rows, section_changes = _matched_edge(reference, candidate)
    drives, means = wall.drive_powers(reference.record, candidate.record, blocks, matches["incoming"])
    for index, drive in enumerate(drives, 1):
        drive["maximum_absolute_complex_change"] = {
            row["quantity"]: row["maximum_complex_change"] for row in rows if row["incident_channel"] == index
        }
    power = max(block["maximum_power_change"] for block in blocks.values())
    amplitude = max(block["maximum_complex_change"] for block in blocks.values())
    return {
        "signed_mode_matches": matches,
        "section_relative_maximum_entry_changes": section_changes,
        "drive_power_fractions": drives,
        "equal_incoherent_mean_power_fractions": means,
        "transmission_eigenvalue_changes": layer._eigenchannel_changes(reference.transmission_eigenchannels,
                                                                         candidate.transmission_eigenchannels),
        "signed_reflection_eigenvalue_changes_percentage_points_descending_order": (
            100 * (candidate.reflection_eigenvalues - reference.reflection_eigenvalues)).tolist(),
        "per_block_maxima": {name: nested._block_increment(block, matches["incoming"], matches[name])
                             for name, block in blocks.items()},
        "maximum_absolute_total_drive_power_change": power,
        "total_drive_power_increment_passed": power < scattering.LIMIT,
        "maximum_absolute_complex_amplitude_change": amplitude,
        "complex_amplitude_increment_passed": amplitude < scattering.LIMIT,
    }


def _case_provenance(case):
    record = case.record
    facts = {key: record[key] for key in ("point_index", "body_dofs", "port_dofs", "mesh_hash", "source_git_commit", "platform",
                                        "original_equation_identity", "maximum_numerical_checks", "independent_flux_checks")}
    facts["archive"] = {key: record["archive"][key] for key in ("checkpoint", "point_summary", "mesh", "mesh_summary",
                                                               "port_section", native.FIELDS, "independent_flux_replay")}
    return facts


def _case_facts(case):
    return {**_case_provenance(case), "native_drives": case.record["native_drives"],
            "transmission_eigenchannels": case.transmission_eigenchannels,
            "reflection_eigenvalues_descending": case.reflection_eigenvalues.tolist()}


def _matched_roots(edge, group):
    match = edge["signed_mode_matches"][group]
    return dict(zip(match["reference_roots"], match["candidate_roots"], strict=True))


def _require_path_matches(edges, first, middle, last):
    for group in scattering.GROUPS:
        initial = _matched_roots(edges[first, middle], group)
        subsequent = _matched_roots(edges[middle, last], group)
        direct = _matched_roots(edges[first, last], group)
        require({root: subsequent[mid] for root, mid in initial.items()} == direct,
                f"Successive signed {group} matches via {middle} disagree with the direct {first}-to-{last} match.")


def _signed_power_effects(fractions):
    coarse, central, complement, full = (fractions[role] for role in ("baseline", "taper_patch", "complement", "full_refinement"))
    return {**{effect: 100 * (fractions[b] - fractions[a]) for effect, (a, b) in REGIONAL_EDGES.items()},
            "mixed_difference": 100 * fsum((full, -central, -complement, coarse))}


def _factorial(cases, edges):
    baseline = cases["baseline"].record["native_drives"]
    mappings = {role: _matched_roots(edges["baseline", role], "incoming") for role in cases if role != "baseline"}
    native = {role: {row["native_root_id"]: row for row in case.record["native_drives"]} for role, case in cases.items()}
    drives = []
    for a in baseline:
        root = a["native_root_id"]
        rows = {"baseline": a, **{role: native[role][lookup[root]] for role, lookup in mappings.items()}}
        powers = {key: {role: row[key] for role, row in rows.items()} for key in ("transmission", "reflection")}
        drives.append({"native_root_id_by_case": {role: row["native_root_id"] for role, row in rows.items()},
                       "reference_drive": a["drive"], "power_fractions": powers,
                       "signed_effects_percentage_points": {key: _signed_power_effects(values) for key, values in powers.items()}})
    means = {key: {role: fsum(row[key] for row in saved.values()) / len(baseline) for role, saved in native.items()}
             for key in ("transmission", "reflection")}
    return {"per_drive": drives, "equal_incoherent_mean_power_fractions": means,
            "signed_equal_incoherent_mean_effects_percentage_points": {key: _signed_power_effects(values) for key, values in means.items()}}


def _published_finer(prior):
    published = native._read(FINER_PRIOR)
    require(_same_pin(published["provenance"]["prior_nested_comparison"], native._link(PRIOR, ROOT))
            and published["reference_planes_x_m"] == prior["reference_planes_x_m"]
            and published["original_numerical_limits"] == prior["original_numerical_limits"]
            and published["incident_side"] == prior["incident_side"]
            and published["physical_thickness_nm"] == 100 and published["frequency_hz"] == FREQUENCY_GHZ * 1e9,
            "The published finer global endpoint belongs to a different device hierarchy.")
    return published


def _finer_comparison(study, prior, reader, initial_cases, initial_edges):
    published = _published_finer(prior)
    inherited = initial_cases["full_refinement"]
    require(all(_same_pin(inherited.record["archive"][key], pin)
                for key, pin in published["levels"]["168.75"]["archive"].items()),
            "The shared 168.75 nm response differs from its published archive.")
    cases = {"full_refinement": inherited,
             "finer_center": _read_response(study, "finer_center", prior, reader),
             "finer_full": _read_response(study, "finer_full", prior, reader, finer_archive=published["levels"]["126.5625"]["archive"])}
    operators = {case.record["original_equation_identity"]["operator_csc_sha256"]
                 for case in (*initial_cases.values(), cases["finer_center"], cases["finer_full"])}
    require(len(operators) == len(initial_cases) + 2, "The finer saved responses must have distinct original volume equations.")
    geometry = _control_geometry(inherited, cases["finer_center"], cases["finer_full"], "finer_center")
    pairs = (("full_refinement", "finer_center"), ("finer_center", "finer_full"), ("full_refinement", "finer_full"))
    edges = {(a, b): _edge(cases[a], cases[b]) for a, b in pairs}
    _require_path_matches(edges, "full_refinement", "finer_center", "finer_full")

    earlier = initial_edges["baseline", "taper_patch"]
    later = edges["full_refinement", "finer_center"]
    baseline_to_finer = _matched_roots(initial_edges["baseline", "full_refinement"], "incoming")
    later_by_root = {row["reference"]["native_root_id"]: row for row in later["drive_power_fractions"]}
    trend = []
    for first in earlier["drive_power_fractions"]:
        second = later_by_root[baseline_to_finer[first["reference"]["native_root_id"]]]
        trend.append({"reference_drive": first["reference"]["drive"],
                      "coarse_step_incident_root": first["reference"]["native_root_id"],
                      "finer_step_incident_root": second["reference"]["native_root_id"],
                      "signed_change_percentage_points": {
                          "225_to_168.75_nm_center": first["candidate_minus_reference_percentage_points"],
                          "168.75_to_126.5625_nm_center": second["candidate_minus_reference_percentage_points"],
                      }})
    return {
        "baseline_case": "full_refinement", "central_mesh_targets_nm": [168.75, 126.5625],
        "fixed_geometry": geometry,
        "cases": {role: _case_facts(cases[role]) for role in FINER_MAPS},
        "increments": {f"{a}_to_{b}": edge for (a, b), edge in edges.items()},
        "central_refinement_trend": {"per_drive": trend, "equal_incoherent_mean_signed_change_percentage_points": {
            "225_to_168.75_nm_center": earlier["equal_incoherent_mean_power_fractions"]["candidate_minus_reference_percentage_points"],
            "168.75_to_126.5625_nm_center": later["equal_incoherent_mean_power_fractions"]["candidate_minus_reference_percentage_points"],
        }},
        "published_finer_global_endpoint": native._link(FINER_PRIOR, ROOT),
        "scope": "The finer patch refines the center from the 168.75 nm baseline while preserving its outer cells. "
                 "The comparison has no finer complementary response. The two central increments describe finite changes "
                 "with their respective outer meshes fixed. They retain the original end phases and do not establish convergence.",
    }


def _gauge_closure(edges, middle):
    _require_path_matches(edges, "baseline", middle, "full_refinement")
    defects = {}
    for group in scattering.GROUPS:
        first, second, direct = (
            edges[pair]["signed_mode_matches"][group]
            for pair in (("baseline", middle), (middle, "full_refinement"), ("baseline", "full_refinement"))
        )
        first_phase, second_phase, direct_phase = (
            np.exp(1j * np.asarray(match["candidate_gauge_phase_rad"])) for match in (first, second, direct)
        )
        next_index = {root: i for i, root in enumerate(second["reference_roots"])}
        order = [next_index[root] for root in first["candidate_roots"]]
        defect = float(np.max(abs(first_phase * second_phase[order] - direct_phase)))
        require(defect <= GAUGE_CLOSURE_TOLERANCE, f"The {group} coefficient gauges do not close via {middle}.")
        defects[group] = defect
    return defects


def _signed_complex_effects(base_edges):
    changes, maxima = {}, {}
    for quantity in scattering.GROUPS[1:]:
        base = base_edges["full_refinement"][quantity]["reference"]
        require(all(np.array_equal(block[quantity]["reference"], base) for block in base_edges.values()),
                f"The {quantity} coefficient reference changes between paths.")
        values = {"baseline": np.asarray(base), **{
            role: np.asarray(block[quantity]["candidate_aligned"]) for role, block in base_edges.items()
        }}
        differences = {effect: values[b] - values[a] for effect, (a, b) in REGIONAL_EDGES.items()}
        differences["mixed_difference"] = (differences["central_at_refined_exterior"]
                                            - differences["central_at_coarse_exterior"])
        changes[quantity] = {effect: delta.tolist() for effect, delta in differences.items()}
        maxima[quantity] = {effect: np.max(np.hypot(delta[..., 0], delta[..., 1]), axis=0).tolist()
                            for effect, delta in differences.items()}
    return {"signed_coefficient_effects_re_im": changes, "maximum_absolute_effect_by_reference_drive": maxima}


def _compare_factorial_steps(coarser, finer):
    earlier = coarser["factorial_power_comparison"]
    bridge = _matched_roots(coarser["increments"]["baseline_to_full_refinement"], "incoming")
    finer_by_root = {row["native_root_id_by_case"]["baseline"]: row for row in finer["per_drive"]}

    def differences(first, second):
        return {kind: {effect: second[kind][effect] - value for effect, value in result.items()}
                for kind, result in first.items()}

    rows = []
    for prior in earlier["per_drive"]:
        old_root = prior["native_root_id_by_case"]["baseline"]
        new_root = bridge[old_root]
        require(new_root == prior["native_root_id_by_case"]["full_refinement"],
                "The earlier factorial and its signed bridge disagree on the shared drive.")
        current = finer_by_root[new_root]
        rows.append({"coarser_reference_drive": prior["reference_drive"], "coarser_reference_root": old_root,
                     "finer_reference_drive": current["reference_drive"], "finer_reference_root": new_root,
                     "finer_minus_coarser_effects_percentage_points": differences(
                         prior["signed_effects_percentage_points"], current["signed_effects_percentage_points"])})
    key = "signed_equal_incoherent_mean_effects_percentage_points"
    return {"coarser_central_mesh_targets_nm": [225.0, 168.75], "per_drive": rows,
            "finer_minus_coarser_mean_effects_percentage_points": differences(earlier[key], finer[key])}


def _factorial_edges(cases):
    edges, aligned = {}, {}
    for a, b in (*REGIONAL_EDGES.values(), ("baseline", "full_refinement")):
        reference, candidate = cases[a], cases[b]
        matches, blocks, _, section_changes = _matched_edge(reference, candidate)
        wall.drive_powers(reference.record, candidate.record, blocks, matches["incoming"])
        power = max(block["maximum_power_change"] for block in blocks.values())
        amplitude = max(block["maximum_complex_change"] for block in blocks.values())
        edges[a, b] = {
            "signed_mode_matches": matches, "section_relative_maximum_entry_changes": section_changes,
            "maximum_absolute_total_drive_power_change": power, "total_drive_power_increment_passed": power < scattering.LIMIT,
            "maximum_absolute_complex_amplitude_change": amplitude, "complex_amplitude_increment_passed": amplitude < scattering.LIMIT,
        }
        if a == "baseline":
            aligned[b] = blocks
    return edges, aligned


def replay_finer_factorial(study=STUDY):
    study = Path(study).resolve()
    require(study.is_relative_to(ROOT), "Keep the native maps and flux receipts under this checkout for portable evidence paths.")
    prior = native._read(PRIOR)
    published = _published_finer(prior)
    coarser = native._read(OUTPUT)
    require(_same_pin(coarser["provenance"]["published_donor_comparison"], native._link(PRIOR, ROOT))
            and _same_pin(coarser["finer_center_comparison"]["published_finer_global_endpoint"], native._link(FINER_PRIOR, ROOT)),
            "The earlier regional comparison belongs to a different donor hierarchy.")
    reader_pin = prior["provenance"]["acceptance_reader"]
    reader = runpy.run_path(str(native._verified(reader_pin, ROOT)))["read_point"]
    roles = {"baseline": "full_refinement", "taper_patch": "finer_center", "complement": "finer_complement",
             "full_refinement": "finer_full"}
    cases = {role: _read_response(study, archive_role, prior, reader,
                                  finer_archive=published["levels"]["126.5625"]["archive"] if role == "full_refinement" else None)
             for role, archive_role in roles.items()}
    baseline, full = cases["baseline"], cases["full_refinement"]
    for case, archive in (
        (baseline, published["levels"]["168.75"]["archive"]),
        (baseline, coarser["cases"]["full_refinement"]["archive"]),
        (cases["taper_patch"], coarser["finer_center_comparison"]["cases"]["finer_center"]["archive"]),
    ):
        require(all(_same_pin(case.record["archive"][key], pin) for key, pin in archive.items()),
                "A reused finer factorial case differs from the original accepted archive.")
    require(len({case.record["original_equation_identity"]["operator_csc_sha256"] for case in cases.values()}) == len(cases),
            "Every finer factorial case must have distinct original volume equations.")
    geometry = {role: _control_geometry(baseline, cases[role], full, roles[role]) for role in ("taper_patch", "complement")}
    coverage = _complement_coverage(cases)

    edges, aligned = _factorial_edges(cases)
    gauge_closure = {role: _gauge_closure(edges, role) for role in ("taper_patch", "complement")}
    factorial = _factorial(cases, edges)
    case_provenance = {role: _case_provenance(case) for role, case in cases.items()}
    for role, case in cases.items():
        case_provenance[role]["archive"].update({key: case.record["archive"][key] for key in ("map", "map_coefficients")})
        case_provenance[role]["native_drive_order"] = [
            {key: drive[key] for key in ("drive", "native_root_id", "k_rad_m")} for drive in case.record["native_drives"]
        ]

    return {**native.AUTHOR, **native.CONVERGENCE, "schema_version": 1, "status": "compared_saved_finer_taper_factorial",
            "frequency_hz": FREQUENCY_GHZ * 1e9, "physical_thickness_nm": 100, "incident_side": "left",
            "central_mesh_targets_nm": [168.75, 126.5625], "reference_planes_x_m": prior["reference_planes_x_m"],
            "original_numerical_limits": prior["original_numerical_limits"],
            "comparison_limits": {"power_change_fraction": scattering.LIMIT, "complex_amplitude_change": scattering.LIMIT,
                                  "minimum_signed_mass_overlap": scattering.MIN_OVERLAP, "unit_phase_closure_defect": GAUGE_CLOSURE_TOLERANCE,
                                  "section_relative_maximum_entry_difference": SECTION_TOLERANCE},
            "cases": case_provenance, "control_geometry": geometry, "physical_mesh_complement_coverage": coverage,
            "factorial_power_comparison": factorial,
            "increments": {f"{a}_to_{b}": edge for (a, b), edge in edges.items()},
            "factorial_complex_comparison": {
                "baseline_root_order": {group: match["reference_roots"] for group, match in
                                        edges["baseline", "full_refinement"]["signed_mode_matches"].items()},
                "maximum_unit_phase_closure_defect": gauge_closure,
                "coefficient_layout": "Matrices have outgoing roots as rows, incident roots as columns, and [real, imaginary] entries.",
                **_signed_complex_effects(aligned)},
            "coarser_factorial_comparison": _compare_factorial_steps(coarser, factorial),
            "scope": "All four responses use the original reference planes and two quadratic thickness layers. "
                     "The control meshes partition the physical tetrahedron replacements. Each effect subtracts the two matching "
                     "corners with the other region held fixed. The mixed difference subtracts the central increment at coarse "
                     "exterior from the central increment at refined exterior. The equal incoherent mean uses five equal-power "
                     "inputs. Coefficient effects share the baseline port gauges and retain physical end-plane phases. The finer "
                     "and coarser partitions each use their own nested donor pair. These are finite changes at one frequency. "
                     "Spatial and frequency convergence remain unresolved.",
            "provenance": {"coarser_factorial": native._link(OUTPUT, ROOT), "published_donor_comparison": native._link(PRIOR, ROOT),
                           "published_finer_global_endpoint": native._link(FINER_PRIOR, ROOT), "acceptance_reader": reader_pin,
                           "comparison_source": native._link(Path(__file__).resolve(), ROOT),
                           "matching_source": native._link(Path(scattering.__file__).resolve(), ROOT),
                           "geometry_verification_source": native._link(Path(patch.__file__).resolve(), ROOT),
                           "reporting_sources": [native._link(Path(module.__file__).resolve(), ROOT)
                                                 for module in (nested, wall, layer)]}}


def replay(study=STUDY, *, include_complement=False, include_finer_center=False):
    study = Path(study).resolve()
    require(study.is_relative_to(ROOT), "Keep the native maps and flux receipts under this checkout for portable evidence paths.")
    prior = native._read(PRIOR)
    reader_pin = prior["provenance"]["acceptance_reader"]
    reader = runpy.run_path(str(native._verified(reader_pin, ROOT)))["read_point"]
    roles = (*MAPS, "complement") if include_complement else tuple(MAPS)
    cases = {role: _read_response(study, role, prior, reader) for role in roles}
    baseline, full = cases["baseline"], cases["full_refinement"]
    controls = {role: _control_geometry(baseline, cases[role], full, role) for role in CONTROL_REGIONS if role in cases}
    require(len({case.record["original_equation_identity"]["operator_csc_sha256"] for case in cases.values()}) == len(cases),
            "The accepted responses must have distinct volume equations.")
    pairs = EDGES + COMPLEMENT_EDGES if include_complement else EDGES
    edges = {(a, b): _edge(cases[a], cases[b]) for a, b in pairs}
    for role in controls:
        _require_path_matches(edges, "baseline", role, "full_refinement")
    scope = "The taper patch retains coarse exterior cells and refines compatible central cells. "
    if include_complement:
        coverage = _complement_coverage(cases)
        factorial = {"physical_mesh_complement_coverage": coverage, **_factorial(cases, edges)}
        scope += ("The complementary control refines the rest of the coarse-to-fine changes. The mixed difference is the change "
                  "in one region's finite refinement increment when the other is refined. Nonzero mixed differences make a single "
                  "additive assignment of the observed response change to the two regions path-dependent. ")
    else:
        factorial = None
        scope += ("The patch-to-full edge measures the remaining change along this mesh path. The complementary response on the "
                  "coarse center is unmeasured in this comparison. Independent regional effects and their interaction remain unresolved. ")
    finer = _finer_comparison(study, prior, reader, cases, edges) if include_finer_center else None
    return {**native.AUTHOR, **native.CONVERGENCE, "schema_version": 1, "status": "compared_saved_taper_patch_control",
            "frequency_hz": FREQUENCY_GHZ * 1e9, "physical_thickness_nm": 100, "incident_side": "left",
            "reference_planes_x_m": prior["reference_planes_x_m"], "original_numerical_limits": prior["original_numerical_limits"],
            "comparison_limits": {"power_change_fraction": scattering.LIMIT, "complex_amplitude_change": scattering.LIMIT,
                                  "minimum_signed_mass_overlap": scattering.MIN_OVERLAP,
                                  "section_relative_maximum_entry_difference": SECTION_TOLERANCE},
            "fixed_geometry": controls["taper_patch"], "cases": {role: _case_facts(case) for role, case in cases.items()},
            "increments": {f"{a}_to_{b}": edge for (a, b), edge in edges.items()},
            **({"complement_geometry": controls["complement"], "factorial_power_comparison": factorial} if include_complement else {}),
            **({"finer_center_comparison": finer} if include_finer_center else {}),
            "scope": scope + "Each result is a finite mesh response at one frequency. Spatial and frequency convergence remain unresolved. "
                     "Complex coefficients retain the original end-plane phases. Transmission and reflection eigenvalues are ranked "
                     "by strength within each saved block. The equal incoherent mean uses five equal-power inputs.",
            "provenance": {"published_donor_comparison": native._link(PRIOR, ROOT), "acceptance_reader": reader_pin,
                           "comparison_source": native._link(Path(__file__).resolve(), ROOT),
                           "matching_source": native._link(Path(scattering.__file__).resolve(), ROOT),
                           "geometry_verification_source": native._link(Path(patch.__file__).resolve(), ROOT),
                           "reporting_sources": [native._link(Path(module.__file__).resolve(), ROOT)
                                                 for module in (nested, wall, layer)]}}


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--study-root", type=Path, default=STUDY)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--include-complement", action="store_true", help="Require and compare both complementary regional refinements.")
    parser.add_argument("--include-finer-center", action="store_true", help="Include the finer 168.75 to 126.5625 nm central patch.")
    parser.add_argument("--finer-factorial", action="store_true", help="Compare the complete 168.75 to 126.5625 nm factorial separately.")
    args = parser.parse_args(argv)
    try:
        output = (args.output or (FINER_OUTPUT if args.finer_factorial else OUTPUT)).resolve()
        require(not output.is_relative_to(args.study_root.resolve()) and output not in (PRIOR, FINER_PRIOR),
                "Place the comparison JSON outside the saved archive and prior evidence.")
        if args.finer_factorial:
            require(output != OUTPUT and not (args.include_complement or args.include_finer_center),
                    "The finer factorial uses separate evidence and takes no other comparison flags.")
            result = replay_finer_factorial(args.study_root)
        else:
            require(output != FINER_OUTPUT, "The finer factorial output belongs to --finer-factorial.")
            result = replay(args.study_root, include_complement=args.include_complement, include_finer_center=args.include_finer_center)
        payload = json.dumps(result, indent=2, allow_nan=False) + "\n"
        with NamedTemporaryFile(mode="w", encoding="utf-8", newline="\n", prefix=f".{output.name}.",
                                suffix=".tmp", dir=output.parent, delete=False) as stream:
            temporary = Path(stream.name)
            stream.write(payload)
        try:
            temporary.replace(output)
        finally:
            temporary.unlink(missing_ok=True)
    except (ArithmeticError, KeyError, OSError, RuntimeError, TypeError, ValueError) as exc:
        parser.exit(1, f"Taper patch comparison failed. {exc}\n")
    print(output)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
