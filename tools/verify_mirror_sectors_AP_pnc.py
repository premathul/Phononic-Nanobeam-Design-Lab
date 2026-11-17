"""Independently verify saved vector-mirror contents. Athul Prem | _AP_pnc.

Explicit P2 evaluation on a rectangular-cell common refinement uses the frozen
nine-point Duffy rule. No new diagnostic core, saved mass, assembler or solver
is imported. Only selected saved section fields are loaded by the CLI.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
from pathlib import Path
import sys

if __name__ == "__main__":
    os.environ.update(dict.fromkeys(("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS",
                                    "BLIS_NUM_THREADS", "NUMEXPR_NUM_THREADS"), "1"))

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from tools import verify_port_polarization_AP_pnc as frozen

TOLERANCE = 1e-10
GEOMETRY_RELATIVE_TOLERANCE = 1e-12
REPRESENTATION_RELATIVE_TOLERANCE = 64 * np.finfo(float).eps
FREQUENCY_HZ = 1_000_000_000
SECTORS = np.array([[1, 1], [1, -1], [-1, 1], [-1, -1]])
SIGNED_ROLES = ("forward_a", "forward_b", "reflected_b", "reflected_a")
ROOTS = {"z4": [1311, 1274, 1275, 1312], "z6": [1877, 1837, 1838, 1878]}
ALL_FORWARD = {"z4": [1348, 1311, 1292, 1284, 1274], "z6": [1940, 1877, 1867, 1855, 1837]}
ALL_BACKWARD = {"z4": [1349, 1312, 1293, 1285, 1275], "z6": [1941, 1878, 1868, 1856, 1838]}
QUADRATURE_SHA256 = "d5cdd47c7df2de902224de6f7c70299862187fa372a0eecaaa38e358e4693779"
BASE = "results/runs/full_device_resource_profile/"
Z4 = BASE + "lawrence_h80_AP_pnc/h80-export-1335045/historical_26hole_layered_h80_ny12_z4-3385249d64d1/"
Z6 = BASE + "lawrence_thickness_AP_pnc/thickness-h80-export-1335341/historical_26hole_layered_h80_ny12_z6-585ccac1d4f2/"
ROLE_PINS = {
    "metric_source": ("tools/diagnose_port_polarization_AP_pnc.py", "367eb1fa5ed0296df548716cb8eb02e422dd639a23f8ad434dc0e357af5d1e79"),
    "matching_source": ("tools/compare_scattering_AP_pnc.py", "148f7efa58ad668a7a57be9db68a3241ffb67ed94067499ce6dcd4a0f8c59313"),
    "device_thickness_verification": ("verification/device_thickness_AP_pnc.json", "81f1e195956c1986d067fc397ea1836c284d9e645073e032d323772918532f05"),
    "thickness_comparison": (BASE + "lawrence_thickness_AP_pnc/comparisons/scattering-b06d34228c560e43/summary_AP_pnc.json", "8a5aea4238d3f09bdd7e97f82cd73d2b74f6a64846ca946633aec08d0b6dede3"),
    "z4_safe_receipt": ("results/runs/full_device_port_resolution/12x4/attempt_lpnbd9kr/checkpoint_AP_pnc.json", "677c64a816ea42a8b37e1358d5549d63ecaeec7623a9366139ba64d301bbea4e"),
    "z6_safe_receipt": (BASE + "lawrence_thickness_AP_pnc/thickness-h80-safe-1335308/thickness_h80_safe_AP_pnc.json", "b5c0d73656e90aa5ba312939c0ef25633af5a775974f5fc778031584d55c78f3"),
    "z4_section": ("results/runs/full_device_port_resolution/12x4/attempt_lpnbd9kr/arrays_AP_pnc.npz", "2a2aa9347921ac6940f851f2eb53bc06480a28a6cf67c27a000faa357b1404e8"),
    "z6_section": (BASE + "lawrence_thickness_AP_pnc/thickness-h80-safe-1335308/ports_AP_pnc.npz", "a13a58e47dee91e4c7d75bb9ec77feb7656ae674392c2f1d7485129da76d0e6a"),
    "h80_z4_summary": (Z4 + "summary_AP_pnc.json", "f65f61eaf62d69fa0574fe8566ba088fd7e090a019685bdac3db95261b61619f"),
    "h80_z6_summary": (Z6 + "summary_AP_pnc.json", "040a219f54446219b6a610cf08382ff7e93a6dcb01f31f3be810977651301c4e"),
    "h80_z4_scattering": (Z4 + "scattering_AP_pnc.csv", "5b718ef9bbdf526540f027af433342209bed3dd8f26b9258ee2bb42158074fc0"),
    "h80_z6_scattering": (Z6 + "scattering_AP_pnc.csv", "54cb1ff9cf3d69cef63f3c9858ec7587efee7cec4fe9324751f7094ea323598a"),
    "h80_z4_coefficients": (Z4 + "scattering_channels_AP_pnc.csv", "3f8094e029e3abd3c705eb72f0e9f5c360f6e12a00e77fe48f17ce5fb01c9978"),
    "h80_z6_coefficients": (Z6 + "scattering_channels_AP_pnc.csv", "bca65a396d45a15c0f3b8ba58a5d85e9add2b1a7b24b3ec111bbebb0310d6014"),
    "conversion_ledger": ("verification/one_ghz_conversion_AP_pnc.json", "f9a72bdcb173c31b96fc0d6d21e589ba9a3a94aa3c9d57877bfd7d6fd90459fd"),
    "conversion_source": ("tools/diagnose_conversion_families_AP_pnc.py", "01d0ba73879138b5fcb1524dd393ac9138be2e0d1a16c83fa5364dd3e9552b99"),
}
PHYSICAL_SOURCES = {
    "src/phononic/geometry_AP_pnc.py": "e43125e4252c748965daa1a1ba4d9b75d3fc573c2bd0ba4731a6dcfe8385db15",
    "src/phononic/material_AP_pnc.py": "6b6db34f0ac0126f6047d78ebeb23de0f07ad738c3d400e5ba81247288fe47b2",
    "src/phononic/layered_AP_pnc.py": "954ce31756410b3767b40e412bcbf299000b4e3e63569eeb7685ca17f6fc5b1a",
    "src/phononic/port_modes_AP_pnc.py": "729f8cf72cd9eb8c78b50e9e9a4d16eea876543156b2f53ef837998e5ca0c91b",
    "src/phononic/scattering_AP_pnc.py": "d3b2bbe9059ec615a63976a54c6e254696fc17d86838f07dd3710b199b53aeab",
}


def require(condition, message):
    if not condition:
        raise ValueError(message)


def sha256(path):
    with Path(path).open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def close(actual, expected, label, tolerance=TOLERANCE):
    a, b = np.asarray(actual), np.asarray(expected)
    require(a.shape == b.shape and a.dtype.kind in "fciu" and b.dtype.kind in "fciu"
            and np.isfinite(a).all() and np.isfinite(b).all()
            and np.all(np.abs(a - b) <= tolerance), f"Independent {label} differs.")
    return float(np.max(np.abs(a - b), initial=0))


def exact_tree(actual, expected, label):
    """Pinned retained science is copied exactly, without a new loose gate."""
    require(actual == expected, f"Retained {label} differs.")


def rectangular_refinement(vertices, triangles):
    """Audit full Cartesian cells and independently add both cell diagonals."""
    vertices, cells = np.asarray(vertices), np.asarray(triangles)
    require(vertices.ndim == 2 and vertices.shape[1] == 2 and vertices.dtype.kind in "fiu"
            and np.isfinite(vertices).all() and len(vertices) >= 4
            and cells.ndim == 2 and cells.shape[1] == 3 and cells.dtype.kind in "iu"
            and len(cells) > 0 and cells.min() >= 0 and cells.max() < len(vertices), "Invalid section geometry.")
    require(len(np.unique(vertices, axis=0)) == len(vertices)
            and len(np.unique(np.sort(cells, axis=1), axis=0)) == len(cells), "Duplicate section geometry.")
    bounds = np.stack((vertices.min(axis=0), vertices.max(axis=0)), axis=1)
    spans = bounds[:, 1] - bounds[:, 0]
    require(np.all(spans > 0), "Degenerate rectangular grid.")
    require(np.all(abs(bounds.sum(axis=1)) <= REPRESENTATION_RELATIVE_TOLERANCE * spans),
            "The geometric mirror center differs.")
    # Group raw coordinates only for topology: they are never replaced in P2
    # fields or quadrature. Connected roundoff clusters allow opposite signed
    # representation errors along one prescribed Cartesian grid line.
    axes, index_columns, grid_residuals = [], [], []
    for axis, span in enumerate(spans):
        tolerance = REPRESENTATION_RELATIVE_TOLERANCE * span
        sorted_values = np.sort(vertices[:, axis])
        clusters = np.split(sorted_values, np.flatnonzero(np.diff(sorted_values) > 2 * tolerance) + 1)
        centers = np.array([np.mean(group) for group in clusters])
        index = abs(vertices[:, axis, None] - centers).argmin(axis=1)
        prescribed = np.linspace(bounds[axis, 0], bounds[axis, 1], len(centers))
        residual = float(np.max(abs(vertices[:, axis] - prescribed[index])))
        require(len(centers) >= 2 and residual <= tolerance,
                "The section grid differs from its prescribed uniform subdivision.")
        axes.append(centers)
        index_columns.append(index)
        grid_residuals.append(residual)
    ny, nz = (len(a) - 1 for a in axes)
    require(len(vertices) == (ny + 1) * (nz + 1) and len(cells) == 2 * ny * nz,
            "The section does not cover the complete rectangular grid.")
    indices = np.column_stack(index_columns)
    lookup = {tuple(index): i for i, index in enumerate(indices)}
    require(len(lookup) == len(vertices)
            and all((iy, iz) in lookup for iy in range(ny + 1) for iz in range(nz + 1)),
            "Missing or duplicate Cartesian grid vertex.")
    mismatch = []
    for axis, count in enumerate((ny, nz)):
        partners = indices.copy()
        partners[:, axis] = count - partners[:, axis]
        partner_ids = [lookup[tuple(index)] for index in partners]
        error = float(np.max(abs(vertices[:, axis] + vertices[partner_ids, axis])))
        require(error <= REPRESENTATION_RELATIVE_TOLERANCE * spans[axis],
                "The reflected grid differs beyond roundoff representation.")
        mismatch.append(error)
    groups = {}
    for tri in cells:
        local_indices = indices[tri]
        low, high = local_indices.min(axis=0), local_indices.max(axis=0)
        require(np.all(high - low == 1), "A section triangle crosses a rectangular cell.")
        groups.setdefault(tuple(low), []).append(set(tri.tolist()))
    refinement = []
    for iy in range(ny):
        for iz in range(nz):
            corner_ids = [lookup[p] for p in ((iy, iz), (iy + 1, iz), (iy + 1, iz + 1), (iy, iz + 1))]
            corners = vertices[corner_ids]
            ids = set(corner_ids)
            pair = groups.get((iy, iz), [])
            require(len(pair) == 2 and pair[0] | pair[1] == ids and len(pair[0] & pair[1]) == 2,
                    "Invalid or incomplete cell triangulation.")
            center = corners.mean(axis=0)
            shared = vertices[list(pair[0] & pair[1])]
            require(np.all(abs(shared.mean(axis=0) - center) <= REPRESENTATION_RELATIVE_TOLERANCE * spans),
                    "Cell triangles do not share a diagonal.")
            refinement.extend(np.array([corners[i], corners[(i + 1) % 4], center]) for i in range(4))
    refined = np.asarray(refinement)
    areas = abs(np.linalg.det(np.stack((refined[:, 1] - refined[:, 0], refined[:, 2] - refined[:, 0]), axis=2))) / 2
    area = float(areas.sum())
    require(np.all(areas > 0) and abs(area / np.prod(spans) - 1) <= GEOMETRY_RELATIVE_TOLERANCE,
            "The refined area differs from the physical rectangle.")
    return refined, {"shape_cells_yz": [ny, nz], "bounds_yz_m": bounds.tolist(),
                     "area_m2": area, "refinement_triangles": len(refined),
                     "maximum_axis_reflection_mismatch_m": mismatch,
                     "maximum_grid_coordinate_residual_m": max(grid_residuals)}


def prepare_section(vertices_yz_m, triangles, coordinates_yz_m, component_dofs_xyz, displacements,
                    density_kg_m3):
    vertices, cells = np.asarray(vertices_yz_m), np.asarray(triangles)
    refined, grid = rectangular_refinement(vertices, cells)
    coordinates, components, values = map(np.asarray, (coordinates_yz_m, component_dofs_xyz, displacements))
    n = len(coordinates)
    require(coordinates.shape == (n, 2) and coordinates.dtype.kind in "fiu" and np.isfinite(coordinates).all()
            and n % 3 == 0 and components.shape == (3, n // 3) and components.dtype.kind in "iu"
            and np.array_equal(np.sort(components.ravel()), np.arange(n))
            and values.ndim == 2 and values.shape[0] == n and values.shape[1] > 0
            and values.dtype.kind in "fciu" and np.isfinite(values).all()
            and np.isfinite(density_kg_m3) and density_kg_m3 > 0,
            "Invalid section field, density or component map.")
    local = frozen.p2_element_dofs(vertices, cells, coordinates, components)
    corners = vertices[cells]
    matrix = np.stack((corners[:, 1] - corners[:, 0], corners[:, 2] - corners[:, 0]), axis=2)
    inverse = np.linalg.inv(matrix)
    grid["mass_per_length_kg_m"] = float(density_kg_m3 * grid["area_m2"])
    grid["quadrature_points"] = 9 * len(refined)
    return {"vertices": vertices, "triangles": cells, "local": local, "corners": corners,
            "inverse": inverse, "values": values, "refined": refined,
            "density": float(density_kg_m3), "grid": grid}


def evaluate_p2(section, points):
    """Barycentric search and explicit six-node P2 functions, including edges."""
    points = np.asarray(points)
    require(points.ndim == 2 and points.shape[1] == 2 and points.dtype.kind in "fiu"
            and np.isfinite(points).all(), "Invalid field sample coordinates.")
    affine = np.einsum("tij,ptj->pti", section["inverse"], points[:, None] - section["corners"][None, :, 0])
    bary = np.concatenate((1 - affine.sum(axis=2, keepdims=True), affine), axis=2)
    inside = (bary >= -1e-12).all(axis=2) & (bary <= 1 + 1e-12).all(axis=2)
    require(inside.any(axis=1).all(), "A reflected or sampled point is outside the section.")
    chosen = inside.argmax(axis=1)
    b = bary[np.arange(len(points)), chosen]
    shape = np.column_stack((b * (2 * b - 1), 4 * b[:, 0] * b[:, 1],
                             4 * b[:, 1] * b[:, 2], 4 * b[:, 0] * b[:, 2]))
    values = section["values"][section["local"][chosen]]
    return np.einsum("pn,pcnm->pcm", shape, values)


def integrate_mirrors(**arguments):
    """Integrate four projector fields and the independent correlation formula."""
    section = prepare_section(**arguments)
    bary, weights, _ = frozen.triangle_quadrature()
    tri = section["refined"]
    determinants = abs(np.linalg.det(np.stack((tri[:, 1] - tri[:, 0], tri[:, 2] - tri[:, 0]), axis=2)))
    points = np.einsum("qn,tnd->tqd", bary, tri).reshape(-1, 2)
    measure = (determinants[:, None] * weights * section["density"]).ravel()
    # Identity, y, z, yz: all actions are complex-linear polar-vector maps.
    reflected = []
    for sy, sz in ((1, 1), (-1, 1), (1, -1), (-1, -1)):
        field = evaluate_p2(section, points * [sy, sz])
        reflected.append(field * np.array([1, sy, sz])[None, :, None])
    group = np.asarray(reflected)
    norm = np.einsum("p,pcm,pcm->m", measure, group[0].conj(), group[0]).real
    require(np.all(norm > 0) and np.isfinite(norm).all(), "A field has zero integrated displacement norm.")
    section["mode_mass_norms"] = norm
    coefficients = np.column_stack((np.ones(4), SECTORS[:, 0], SECTORS[:, 1], SECTORS.prod(axis=1))) / 4
    projected = np.einsum("sg,gpcm->spcm", coefficients, group)
    gram = np.einsum("p,spcm,tpcm->mst", measure, projected.conj(), projected) / norm[:, None, None]
    correlations = np.einsum("p,pcm,gpcm->mg", measure, group[0].conj(), group[1:]) / norm[:, None]
    from_correlations = (1 + correlations.real @ coefficients[:, 1:].T * 4) / 4
    reflected_norms = np.einsum("p,gpcm,gpcm->mg", measure, group[1:].conj(), group[1:]).real
    relative_norm_errors = reflected_norms / norm[:, None] - 1
    rows = []
    for i in range(len(norm)):
        w = np.diag(gram[i]).real
        offdiagonal = gram[i] - np.diag(np.diag(gram[i]))
        close(w, from_correlations[i], "projector/correlation weights")
        close(correlations[i].imag, np.zeros(3), "real mirror correlations")
        close(relative_norm_errors[i], np.zeros(3), "reflection norm identities")
        close(offdiagonal, np.zeros((4, 4)), "projector orthogonality")
        close(w.sum(), 1, "weight partition")
        require(np.all(w >= -TOLERANCE) and np.all(w <= 1 + TOLERANCE), "Invalid sector weight range.")
        rows.append({"displacement_mass_norm": float(norm[i]), "weights_pp_pm_mp_mm": w.tolist(),
                     "correlation_weights_pp_pm_mp_mm": from_correlations[i].tolist(),
                     "mirror_correlations_y_z_yz_real": correlations[i].real.tolist(),
                     "mirror_correlations_y_z_yz_imag": correlations[i].imag.tolist(),
                     "normalized_projector_gram_real": gram[i].real.tolist(),
                     "normalized_projector_gram_imag": gram[i].imag.tolist(),
                     "weight_sum_residual": float(abs(w.sum() - 1)),
                     "maximum_orthogonality_residual": float(abs(offdiagonal).max()),
                     "reflection_norm_relative_errors_y_z_yz": relative_norm_errors[i].tolist(),
                     "descriptive_largest_sector": SECTORS[int(w.argmax())].tolist(),
                     "largest_sector_remainder": float(1 - w.max())})
    return section["grid"], rows, section


def verified_inputs(document, root=ROOT):
    root, roles = Path(root).resolve(), {}
    require(len(document["inputs"]) == len(ROLE_PINS), "Scientific role count differs.")
    for item in document["inputs"]:
        role = item["role"]
        require(role in ROLE_PINS and role not in roles, "Unknown or duplicate scientific role.")
        relative, digest = ROLE_PINS[role]
        path = (root / item["path"]).resolve()
        require(path == (root / relative).resolve() and path.is_relative_to(root)
                and item["sha256"] == digest and path.stat().st_size == item["bytes"]
                and sha256(path) == digest, f"Accepted scientific role identity differs: {role}.")
        roles[role] = path
    return roles


def audit_metadata(records, root=ROOT):
    """Intended continuum symmetry follows pinned source/metadata, not mesh parity."""
    ledger, comparison = records["conversion_ledger"], records["thickness_comparison"]
    require(ledger["status"] == "independently_verified_finite_conversion_pair_diagnosis"
            and ledger["frequency_hz"] == FREQUENCY_HZ
            and comparison["complex_passed"] is False and comparison["mesh_pair_passed"] is False
            and comparison["power_change_limit"] == comparison["complex_change_limit"] == .01,
            "The accepted conversion ledger or failed overall status differs.")
    old_roles = set(ROLE_PINS) - {"conversion_ledger", "conversion_source"}
    require(len(ledger["inputs"]) == len(old_roles)
            and {item["role"] for item in ledger["inputs"]} == old_roles,
            "The accepted conversion ledger input roles differ.")
    for item in ledger["inputs"]:
        relative, digest = ROLE_PINS[item["role"]]
        require(item["path"] == relative and item["sha256"] == digest
                and item["bytes"] == (Path(root) / relative).stat().st_size,
                "The accepted conversion ledger role link differs.")
    for key in ("input_power_checks_passed", "power_change_limit", "complex_change_limit", "maximum_power_change",
                "maximum_complex_change", "power_passed", "complex_passed", "mesh_pair_passed"):
        exact_tree(ledger["preserved_scientific_status"][key], comparison[key], "comparison " + key)
    require(ledger["preserved_scientific_status"]["reference_planes_m"] == [0, 1.665e-5],
            "The original reference planes differ.")
    for name in ("z4", "z6"):
        summary = records[f"h80_{name}_summary"]
        c = summary["configuration"]
        require(c["geometry"]["length_nm"] == 16650 and c["geometry"]["width_nm"] == 1000
                and c["geometry"]["thickness_nm"] == 200 and c["mesh"]["kind"] == "layered"
                and c["mesh"]["circle_resolution"] == 32
                and c["mesh"]["thickness_layers"] == int(name[1:])
                and c["analysis"]["kind"] == "scattering" and c["analysis"]["boundary"] == "matched_ports"
                and c["analysis"]["frequency_GHz"] == 1 and c["analysis"]["incident_side"] == "left",
                "The physical geometry, boundaries or frequency differs.")
        material = c["material"]
        require(material["C11_Pa"] == 128.5e9 and material["C12_Pa"] == 48.3e9
                and material["C44_Pa"] == 66.8e9 and material["density_kg_m3"] == 5323
                and material["crystal_rotation"] == np.eye(3).tolist()
                and material["temperature_policy"] == "fixed_property_control", "The aligned cubic material differs.")
        for path, digest in PHYSICAL_SOURCES.items():
            require(summary["implementation"]["source_sha256"][path] == digest
                    and sha256(Path(root) / path) == digest, "The audited physical source identity differs.")
        holes = summary["hole_pattern_provenance"]
        require(summary["hole_count"] == len(holes["holes"]) == 26
                and holes["source_sha256"] == "2c44d944ca9a161a5571a145d129972bb808e3e038f20765ea915f5ce0020771"
                and all(set(h) == {"x_m", "diameter_m", "region"} and 0 < h["diameter_m"] < 1e-6
                        and 0 < h["x_m"] < 1.665e-5 for h in holes["holes"]), "The centered-hole metadata differs.")
        record = comparison["reference" if name == "z4" else "candidate"]
        require(record["run_hash"] == summary["run_hash"]
                and record["summary_sha256"] == ROLE_PINS[f"h80_{name}_summary"][1]
                and summary["incident_mode_indices"] == ALL_FORWARD[name]
                and summary["frequency_hz"] == FREQUENCY_HZ and summary["incident_side"] == "left"
                and summary["power_balance_passed"] is True, "The response run/root identity differs.")
        for suffix, filename in (("scattering", "scattering_AP_pnc.csv"), ("coefficients", "scattering_channels_AP_pnc.csv")):
            digest = ROLE_PINS[f"h80_{name}_{suffix}"][1]
            require(summary["output_sha256"][filename] == record["output_sha256"][filename] == digest,
                    "The accepted response CSV role differs.")
    z4, z6, thickness = records["z4_safe_receipt"], records["z6_safe_receipt"], records["device_thickness_verification"]
    require(z4["status"] == "solved" and z4["trace_dofs"] == 675
            and z4["inputs"]["frequency_hz"] == FREQUENCY_HZ
            and z4["arrays"]["sha256"] == ROLE_PINS["z4_section"][1], "The z4 receipt/section identity differs.")
    require(z6["status"] == "safe_ready" and z6["trace_dofs"] == 975 and z6["frequency_hz"] == FREQUENCY_HZ
            and z6["ports"]["sha256"] == ROLE_PINS["z6_section"][1]
            and z6["incident_port_roots"] == ALL_FORWARD["z6"]
            and z6["backward_port_roots"] == ALL_BACKWARD["z6"]
            and thickness["safe_section"]["receipt_sha256"] == ROLE_PINS["z6_safe_receipt"][1]
            and thickness["safe_section"]["ports_file_sha256"] == ROLE_PINS["z6_section"][1]
            and thickness["scattering_export"]["public_summary_sha256"] == ROLE_PINS["h80_z6_summary"][1]
            and thickness["h80_z4_z6_mesh_pair"]["comparison_summary_sha256"] == ROLE_PINS["thickness_comparison"][1],
            "The z6 receipt/section/comparison identity differs.")
    exact_tree(ledger["preserved_scientific_status"]["maximum_same_channel_transmission_phase_change_rad"],
               thickness["h80_z4_z6_mesh_pair"]["maximum_same_channel_transmission_phase_change_rad"],
               "retained transmission phase change")
    # Cartesian cubic tensor nonzero terms contain even multiplicities of each
    # reflected index. Centered holes/free surfaces are explicit in pinned code.
    return {"intended_continuum_y_mirror": True, "intended_continuum_z_mirror": True,
            "identity_crystal_rotation": True, "aligned_cubic_tensor": True,
            "centered_circular_through_holes": True, "traction_free_lateral_surfaces": True,
            "unchanged_x_reference_planes_m": [0, 1.665e-5], "discrete_body_symmetry_proved": False,
            "discrete_port_symmetry_proved": False, "radiation_map_symmetry_proved": False}


def load_section(path, name):
    require(path.stat().st_size <= 64 * 1024**2, "Section verifier refuses an oversized archive.")
    roots, n = ROOTS[name], 675 if name == "z4" else 975
    with np.load(path, allow_pickle=False) as arrays:
        waves = arrays["k_rad_m" if name == "z4" else "wavenumbers_rad_m"]
        powers = arrays["signed_power_w" if name == "z4" else "powers_w"]
        prop, pos, neg = (arrays[k] for k in ("propagating", "positive_indices", "negative_indices"))
        residual = arrays["relative_residuals"]
        require(waves.shape == powers.shape == prop.shape == residual.shape == (2 * n,)
                and prop.dtype == np.bool_ and pos.shape == neg.shape == (n,)
                and pos.dtype.kind in "iu" and neg.dtype.kind in "iu"
                and np.array_equal(np.sort(np.r_[pos, neg]), np.arange(2 * n))
                and pos[prop[pos]].tolist() == ALL_FORWARD[name] and neg[prop[neg]].tolist() == ALL_BACKWARD[name],
                "The complete signed mode partition differs.")
        if name == "z6":
            require(float(arrays["frequency_hz"]) == FREQUENCY_HZ, "The saved section frequency differs.")
        require(np.isfinite(waves[roots]).all() and np.all(waves[roots].imag == 0)
                and np.all(waves[roots[:2]].real > 0) and np.all(waves[roots[2:]].real < 0)
                and np.isfinite(residual[roots]).all() and np.all((residual[roots] >= 0) & (residual[roots] <= 1e-8)),
                "The selected signed root metadata differs.")
        close(powers[roots], [1, 1, -1, -1], "signed physical power", 1e-7)
        for forward, reverse in zip(roots[:2], roots[:1:-1]):
            matches = np.asarray(ALL_BACKWARD[name])[abs(waves[ALL_BACKWARD[name]] + waves[forward]) <= abs(waves[forward]) * 1e-12]
            require(matches.tolist() == [reverse], "A signed reverse-wavenumber partner differs.")
        arguments = {k: arrays[v] for k, v in {"vertices_yz_m": "section_vertices_yz_m", "triangles": "section_triangles",
                     "coordinates_yz_m": "coordinates_yz_m", "component_dofs_xyz": "component_dofs_xyz"}.items()}
        displacement = arrays["displacements"]
        require(displacement.shape == (n, 2 * n) and displacement.dtype == np.complex128,
                "The saved displacement array dimensions/type differs.")
        arguments["displacements"] = displacement[:, roots]
    require(len(arguments["coordinates_yz_m"]) == n, "The saved section DOF count differs.")
    arguments["density_kg_m3"] = 5323.
    metadata = [dict(root=root, signed_role=role, wavenumber_rad_m=float(waves[root].real),
                     signed_power_w=float(powers[root]), relative_residual=float(residual[root]))
                for root, role in zip(roots, SIGNED_ROLES)]
    return arguments, metadata


def compare_mode(actual, expected, label):
    maximum = 0.
    for key, value in expected.items():
        if key == "descriptive_largest_sector":
            require(actual[key] == value, f"The {label} descriptive sector differs.")
        elif key == "displacement_mass_norm":
            close(actual[key] / value, 1, label + " mass norm", GEOMETRY_RELATIVE_TOLERANCE)
        else:
            maximum = max(maximum, close(actual[key], value, label + " " + key))
    return maximum


def verify_profiles(profile, section, roots):
    y = np.linspace(-5e-7, 5e-7, 129)
    close(profile["coordinates_y_m"], y, "profile physical coordinates", 1e-18)
    require(profile["z_m"] == 0 and len(profile["modes"]) == 2
            and profile["units"] == "m/sqrt(W)" and profile["normalization_applied"] is False
            and profile["phase_rotation_applied"] is False, "The midline profile metadata differs.")
    expected = evaluate_p2(section, np.column_stack((y, np.zeros(129))))[:, 2, :2]
    maximum = 0.
    for i, row in enumerate(profile["modes"]):
        require(row["root"] == roots[i], "The profile signed root differs.")
        field = np.asarray(row["uz_real"]) + 1j * np.asarray(row["uz_imag"])
        scale = float(np.sqrt(section["mode_mass_norms"][i] / section["grid"]["mass_per_length_kg_m"]))
        require(np.isfinite(scale) and scale > 0, "The full-field RMS is invalid.")
        maximum = max(maximum, close(field / scale, expected[:, i] / scale, "raw complex uz profile"))
    return maximum


def verify_diagnosis(document, root=ROOT):
    require(document["status"] == "diagnosed_within_saved_sections" and document["schema_version"] == 1
            and document["frequency_hz"] == FREQUENCY_HZ and document["sector_order"] == SECTORS.tolist(),
            "The mirror diagnosis identity differs.")
    require(sha256(Path(frozen.__file__)) == QUADRATURE_SHA256, "The frozen quadrature source differs.")
    roles = verified_inputs(document, root)
    require(document["sources"] == [{"path": "tools/diagnose_mirror_sectors_AP_pnc.py",
            "sha256": sha256(Path(root) / "tools/diagnose_mirror_sectors_AP_pnc.py")}],
            "The producing source identity differs.")
    records = {key: json.loads(path.read_bytes()) for key, path in roles.items() if path.suffix == ".json"}
    audit = audit_metadata(records, root)
    for key, value in audit.items():
        exact_tree(document["physical_audit"][key], value, "physical audit " + key)
    source_list = document["physical_audit"]["source_identities"]
    require(len(source_list) == len(PHYSICAL_SOURCES)
            and {r["path"]: r["sha256"] for r in source_list} == PHYSICAL_SOURCES,
            "The physical audit source identities differ.")
    ledger, comparison = records["conversion_ledger"], records["thickness_comparison"]
    exact_tree(document["accepted_conversion"]["selected_conversion_pairs"], ledger["selected_conversion_pairs"], "conversion pair")
    exact_tree(document["accepted_conversion"]["aligned_coefficient_changes_for_selected_pair"],
               ledger["aligned_coefficient_changes_for_selected_pair"], "conversion increments")
    exact_tree(document["preserved_scientific_status"], ledger["preserved_scientific_status"], "failed overall scientific status")
    require(document["preserved_scientific_status"]["complex_passed"] is False
            and document["preserved_scientific_status"]["mesh_pair_passed"] is False,
            "The failed overall response status must remain failed.")
    maximum, profile_maximum, expected_rows = 0., 0., {}
    for name in ("z4", "z6"):
        arguments, metadata = load_section(roles[name + "_section"], name)
        grid, rows, section = integrate_mirrors(**arguments)
        actual = document["sections"][name]
        require(len(actual["modes"]) == 4, "All four signed participant fields are required per section.")
        for key, value in grid.items():
            if key in ("area_m2", "mass_per_length_kg_m"):
                close(actual["grid"][key] / value, 1, name + " " + key, GEOMETRY_RELATIVE_TOLERANCE)
            elif key == "bounds_yz_m":
                close(actual["grid"][key], value, name + " bounds", 1e-18)
            elif key == "quadrature_points":
                exact_tree(actual["grid"][key], 6 * grid["refinement_triangles"], name + " producer quadrature points")
            elif key.startswith("maximum_"):
                close(actual["grid"][key], value, name + " representation residual", 1e-21)
            else:
                exact_tree(actual["grid"][key], value, name + " grid " + key)
        close(grid["bounds_yz_m"], [[-5e-7, 5e-7], [-1e-7, 1e-7]], name + " physical bounds", 1e-18)
        require(grid["shape_cells_yz"] == [12, int(name[1:])], "The accepted section grid differs.")
        positive = {row["root"]: row for row in ledger["sections"][name]["modes"]}
        for i, (published, row, meta) in enumerate(zip(actual["modes"], rows, metadata)):
            for key, value in meta.items():
                exact_tree(published[key], value, name + " signed " + key)
            maximum = max(maximum, compare_mode(published, row, f"{name} root {meta['root']}"))
            bounds = positive[meta["root"]]["template_scores_tx_ty_tz_rx"] if i < 2 else None
            bounds = np.asarray(bounds)[[0, 2, 1, 3]].tolist() if bounds is not None else None
            exact_tree(published["template_sector_lower_bounds_pp_pm_mp_mm"], bounds, "template-sector bounds")
            if bounds is not None:
                require(np.all(np.asarray(row["weights_pp_pm_mp_mm"]) >= np.asarray(bounds) - TOLERANCE),
                        "The template-sector lower bound is violated.")
        profile_maximum = max(profile_maximum, verify_profiles(actual["forward_profiles"], section, ROOTS[name][:2]))
        expected_rows[name] = rows
        pair = ledger["selected_conversion_pairs"]["h80_" + name]
        joins = document["directed_pair_joins"][name]
        require(len(joins) == 2, "Both signed conversion directions are required.")
        for j, (published, direction) in enumerate(zip(joins, pair["directions"])):
            for key in ("incident_root", "outgoing_root", "amplitude_real", "amplitude_imag", "power_fraction"):
                exact_tree(published[key], direction[key], "directed conversion " + key)
            maximum = max(maximum, close(published["incident_weights_pp_pm_mp_mm"], rows[j]["weights_pp_pm_mp_mm"], "incident sector join"),
                          close(published["outgoing_weights_pp_pm_mp_mm"], rows[j + 2]["weights_pp_pm_mp_mm"], "outgoing sector join"))
    require(len(document["matched_rows"]) == 4, "All four finite signed increments are required.")
    for i, row in enumerate(document["matched_rows"]):
        require(row["signed_role"] == SIGNED_ROLES[i] and row["z4_root"] == ROOTS["z4"][i]
                and row["z6_root"] == ROOTS["z6"][i], "The finite signed increment root differs.")
        delta = np.asarray(expected_rows["z6"][i]["weights_pp_pm_mp_mm"]) - expected_rows["z4"][i]["weights_pp_pm_mp_mm"]
        maximum = max(maximum, close(row["z6_minus_z4_weights_pp_pm_mp_mm"], delta, "finite signed sector increment"))
    return {"author": "Athul Prem", "file_tag": "_AP_pnc", "status": "independently_verified_saved_mirror_content",
            "method": "Explicit P2 fields on independently audited rectangular-cell crossed-diagonal refinement; frozen nine-point Duffy quadrature.",
            "section_rows_checked": 8, "forward_profiles_checked": 4, "profile_samples_per_mode": 129,
            "implementation_agreement_tolerance": TOLERANCE, "area_and_mass_relative_tolerance": GEOMETRY_RELATIVE_TOLERANCE,
            "maximum_dimensionless_metric_difference": maximum, "maximum_normalized_complex_profile_difference": profile_maximum,
            "representation_relative_tolerance": REPRESENTATION_RELATIVE_TOLERANCE,
            "quadrature_exactness_limit": "Quartic products on the common refinement, subject to measured roundoff-scale axis representation mismatches; raw coordinates retained.",
            "physical_source_sha256": PHYSICAL_SOURCES, "physical_audit": audit,
            "maximum_complex_change": comparison["maximum_complex_change"], "spatial_convergence_established": False,
            "frequency_convergence_established": False, "parity_purity_threshold_defined": False,
            "power_error_floor_inferred": False, "body_arrays_loaded": False, "saved_mass_loaded": False,
            "assembly_or_eigensolve_performed": False}


def verify_publication_csv(document, directory):
    filename = "mirror_sectors_AP_pnc.csv"
    path = Path(directory) / filename
    require(sha256(path) == document["output_sha256"][filename], "The mirror CSV checksum differs.")
    with path.open(newline="", encoding="utf-8") as stream:
        rows = list(csv.DictReader(stream))
    expected = [(name, row) for name in ("z4", "z6") for row in document["sections"][name]["modes"]]
    require(len(rows) == len(expected) == 8, "The mirror CSV must contain eight signed rows.")
    for actual, (name, row) in zip(rows, expected):
        require(actual["section"] == name and int(actual["root"]) == row["root"] and actual["signed_role"] == row["signed_role"],
                "The mirror CSV signed identity differs.")
        for key in ("wavenumber_rad_m", "signed_power_w", "largest_sector_remainder"):
            close(float(actual[key]), row[key], "mirror CSV " + key)
        close([float(actual["weight_" + suffix]) for suffix in ("pp", "pm", "mp", "mm")], row["weights_pp_pm_mp_mm"], "mirror CSV weights")
        require([int(actual["largest_sector_y"]), int(actual["largest_sector_z"])] == row["descriptive_largest_sector"],
                "The mirror CSV descriptive sector differs.")
    return len(rows)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--diagnosis", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    require(not args.output.exists() and args.output.resolve() != args.diagnosis.resolve(), "Verification requires a fresh output file.")
    document = json.loads(args.diagnosis.read_bytes())
    result = verify_diagnosis(document)
    result["publication_csv_rows_checked"] = verify_publication_csv(document, args.diagnosis.parent)
    result["diagnosis"] = {"path": str(args.diagnosis.resolve()), "bytes": args.diagnosis.stat().st_size,
                           "sha256": sha256(args.diagnosis)}
    result["verifier_sha256"] = sha256(Path(__file__))
    result["frozen_quadrature_sha256"] = QUADRATURE_SHA256
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("x", encoding="utf-8") as stream:
        json.dump(result, stream, indent=2, allow_nan=False)
        stream.write("\n")
    print(json.dumps({"status": result["status"], "output": str(args.output.resolve()),
                      "maximum_dimensionless_metric_difference": result["maximum_dimensionless_metric_difference"]}))


if __name__ == "__main__":
    main()
