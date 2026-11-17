"""Measure eight saved 1 GHz polar-vector mirror sectors. Athul Prem | _AP_pnc.

The four fractions are displacement mass content, not scattering power.  A
common refinement splits both rectangular-cell diagonals before quadrature;
neither a nodal reflection nor the original mass matrix is used.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
from pathlib import Path
import sys
from time import monotonic

if __name__ == "__main__":
    os.environ.update(dict.fromkeys(("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS",
                                    "BLIS_NUM_THREADS", "NUMEXPR_NUM_THREADS"), "1"))

import numpy as np
from skfem import Basis, ElementTriP2, ElementVector, MeshTri
from skfem.quadrature import get_quadrature_tri

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT), str(ROOT / "src")]
from tools import diagnose_conversion_families_AP_pnc as accepted

AUTHOR = {"author": "Athul Prem", "file_tag": "_AP_pnc"}
FREQUENCY_HZ = 1_000_000_000
DENSITY = 5323.0
TOLERANCE = 1e-10
GEOMETRY_TOLERANCE = 1e-12
SECTORS = ((1, 1), (1, -1), (-1, 1), (-1, -1))
SIGNED_ROLES = ("forward_a", "forward_b", "reflected_b", "reflected_a")
PARTICIPANTS = {"z4": (1311, 1274, 1275, 1312), "z6": (1877, 1837, 1838, 1878)}
JSON_NAME = "mirror_sectors_AP_pnc.json"
CSV_NAME = "mirror_sectors_AP_pnc.csv"
ROLE_PINS = {**accepted.ROLE_PINS,
    "conversion_ledger": ("verification/one_ghz_conversion_AP_pnc.json",
        "f9a72bdcb173c31b96fc0d6d21e589ba9a3a94aa3c9d57877bfd7d6fd90459fd"),
    "conversion_source": ("tools/diagnose_conversion_families_AP_pnc.py",
        "01d0ba73879138b5fcb1524dd393ac9138be2e0d1a16c83fa5364dd3e9552b99"),
}
PHYSICAL_SOURCES = ("src/phononic/geometry_AP_pnc.py", "src/phononic/material_AP_pnc.py",
                    "src/phononic/layered_AP_pnc.py", "src/phononic/port_modes_AP_pnc.py",
                    "src/phononic/scattering_AP_pnc.py")


def require(condition, message):
    if not condition:
        raise ValueError(message)


def file_sha(path):
    with Path(path).open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def bind_input(workspace, role, inputs):
    """Bind a scientific role to its own exact archive path and digest."""
    require(role in ROLE_PINS and role not in {row["role"] for row in inputs}, "Unknown or duplicate input role.")
    relative, digest = ROLE_PINS[role]
    root = Path(workspace).resolve()
    path = (root / relative).resolve(strict=True)
    require(path.is_relative_to(root) and path.is_file() and file_sha(path) == digest,
            f"The accepted identity differs for role {role}.")
    inputs.append({"role": role, "path": relative, "sha256": digest, "bytes": path.stat().st_size})
    return path


def validate_grid(vertices, triangles, shape_cells_yz):
    """Prove a centered uniform rectangle tiling, retaining its raw coordinates.

    The 64-epsilon guards bound coordinate representation error, separately
    from the 1e-12 relative area check. No scientific coordinates are snapped.
    """
    vertices, triangles = np.asarray(vertices), np.asarray(triangles)
    require(vertices.ndim == 2 and vertices.shape[1] == 2 and vertices.dtype.kind == "f"
            and np.isfinite(vertices).all() and triangles.ndim == 2 and triangles.shape[1] == 3
            and triangles.dtype.kind in "iu" and len(triangles) > 0
            and triangles.min() >= 0 and triangles.max() < len(vertices), "Invalid section topology.")
    require(len(shape_cells_yz) == 2 and all(type(n) is int and n > 0 for n in shape_cells_yz),
            "Invalid rectangular grid dimensions.")
    ny, nz = shape_cells_yz
    bounds = np.stack((vertices.min(axis=0), vertices.max(axis=0)), axis=1)
    spans = np.ptp(bounds, axis=1)
    roundoff = 64 * np.finfo(float).eps * spans
    require(np.all(spans > 0) and np.all(abs(bounds.sum(axis=1)) <= roundoff),
            "The rectangular section is not centered on the physical mirrors.")
    require(len(vertices) == (ny + 1) * (nz + 1) and len(triangles) == 2 * ny * nz,
            "The rectangular grid has missing or extra vertices/triangles.")
    indices, residuals = [], []
    for axis, count in enumerate((ny, nz)):
        expected = np.linspace(bounds[axis, 0], bounds[axis, 1], count + 1)
        distance = abs(vertices[:, axis, None] - expected)
        index = distance.argmin(axis=1)
        error = distance[np.arange(len(vertices)), index]
        require(np.all(error <= roundoff[axis]), "The section is not a uniform rectangular grid.")
        indices.append(index)
        residuals.append(error)
    indices = np.column_stack(indices)
    lookup = {tuple(index): i for i, index in enumerate(indices)}
    require(len(lookup) == len(vertices), "Duplicate or missing rectangular grid point.")
    mismatch = []
    for axis, count in enumerate((ny, nz)):
        reflected = indices.copy()
        reflected[:, axis] = count - reflected[:, axis]
        partners = np.array([lookup[tuple(index)] for index in reflected])
        error = abs(vertices[:, axis] + vertices[partners, axis])
        require(np.max(error) <= roundoff[axis], "The grid is physically asymmetric under reflection.")
        mismatch.append(float(np.max(error)))
    require(len(np.unique(np.sort(triangles, axis=1), axis=0)) == len(triangles),
            "Duplicate section triangle.")
    groups = {}
    for triangle in triangles:
        ij = indices[triangle]
        start = ij.min(axis=0)
        require(np.array_equal(ij.max(axis=0) - start, [1, 1]),
                "A section triangle does not belong to one rectangular cell.")
        groups.setdefault(tuple(start), []).append(set(triangle.tolist()))
    refinement = []
    for iy in range(ny):
        for iz in range(nz):
            corners = [lookup[index] for index in ((iy, iz), (iy + 1, iz),
                                                   (iy + 1, iz + 1), (iy, iz + 1))]
            group = groups.get((iy, iz), [])
            require(len(group) == 2 and group[0] | group[1] == set(corners),
                    "The rectangular cell is not covered by two triangles.")
            common = group[0] & group[1]
            require(common in ({corners[0], corners[2]}, {corners[1], corners[3]}),
                    "The cell triangles do not share a diagonal.")
            points = vertices[corners]
            center = points.mean(axis=0)
            refinement.extend(np.stack((points[i], points[(i + 1) % 4], center)) for i in range(4))
    refinement = np.asarray(refinement)
    edges = refinement[:, 1:] - refinement[:, :1]
    determinants = abs(edges[:, 0, 0] * edges[:, 1, 1] - edges[:, 0, 1] * edges[:, 1, 0])
    area = float(determinants.sum() / 2)
    require(np.all(determinants > 0) and np.isclose(area, np.prod(spans), rtol=GEOMETRY_TOLERANCE, atol=0),
            "The common refinement has invalid area.")
    grid = {"shape_cells_yz": [ny, nz], "bounds_yz_m": bounds.tolist(), "area_m2": area,
            "mass_per_length_kg_m": DENSITY * area, "refinement_triangles": len(refinement),
            "quadrature_points": 6 * len(refinement),
            "maximum_grid_coordinate_residual_m": float(np.max(residuals)),
            "maximum_axis_reflection_mismatch_m": mismatch,
            "method": "Raw cell corners and their center; both diagonals split; degree-4 six-point triangle quadrature. Exact P2 products up to the recorded roundoff geometry representation."}
    return refinement, determinants, grid


def section_basis(vertices, triangles, coordinates, components):
    """Match the raw saved scalar/vector P2 topology and Cartesian order."""
    coordinates, components = np.asarray(coordinates), np.asarray(components)
    mesh = MeshTri(np.ascontiguousarray(vertices.T), np.ascontiguousarray(triangles.T))
    scalar = Basis(mesh, ElementTriP2(), intorder=4)
    vector = Basis(mesh, ElementVector(ElementTriP2(), dim=3), intorder=4)
    require(np.array_equal(mesh.p.T, vertices) and np.array_equal(mesh.t.T, triangles)
            and np.array_equal(vector.doflocs.T, coordinates)
            and np.array_equal(vector.split_indices(), components),
            "The saved P2 topology/coordinates/component order differ.")
    return scalar


def measure_sectors(*, vertices_yz_m, triangles, coordinates_yz_m, component_dofs_xyz,
                    displacements, shape_cells_yz, density_kg_m3=DENSITY):
    """Integrate complex-linear polar-vector projectors on a common refinement."""
    refinement, determinants, grid = validate_grid(vertices_yz_m, triangles, shape_cells_yz)
    scalar = section_basis(np.asarray(vertices_yz_m), np.asarray(triangles),
                           coordinates_yz_m, component_dofs_xyz)
    fields, components = np.asarray(displacements), np.asarray(component_dofs_xyz)
    require(fields.ndim == 2 and fields.shape[0] == 3 * scalar.N and fields.shape[1] > 0
            and fields.dtype.kind in "fc" and np.isfinite(fields).all(), "Invalid section displacement fields.")
    require(np.isfinite(density_kg_m3) and density_kg_m3 > 0, "Invalid section density.")
    grid["mass_per_length_kg_m"] = float(density_kg_m3 * grid["area_m2"])
    q, weights = get_quadrature_tri(4)
    require(q.shape == (2, 6) and weights.shape == (6,), "Expected six-point degree-4 quadrature.")
    barycentric = np.vstack((1 - q.sum(axis=0), q)).T
    points = np.einsum("qv,tvd->tqd", barycentric, refinement).reshape(-1, 2)
    measure = (density_kg_m3 * determinants[:, None] * weights).ravel()
    scales = np.max(abs(fields), axis=0)
    require(np.all(scales > 0), "A saved displacement field is zero.")
    normalized = fields / scales
    reflected_fields = []
    for coordinate_signs, vector_signs in (((1, 1), (1, 1, 1)), ((-1, 1), (1, -1, 1)),
                                          ((1, -1), (1, 1, -1)), ((-1, -1), (1, -1, -1))):
        probes = scalar.probes((points * coordinate_signs).T)
        values = np.stack([probes @ normalized[ids] for ids in components], axis=1)
        reflected_fields.append(values * np.asarray(vector_signs)[None, :, None])
    original, ry, rz, ryz = reflected_fields
    norm = np.einsum("q,qcm,qcm->m", measure, original.conj(), original).real
    require(np.all(np.isfinite(norm) & (norm > 0)), "Invalid integrated displacement norm.")
    projections = np.stack([(original + a * ry + b * rz + a * b * ryz) / 4 for a, b in SECTORS])
    gram = np.einsum("q,sqcm,tqcm->mst", measure, projections.conj(), projections) / norm[:, None, None]
    direct = np.diagonal(gram, axis1=1, axis2=2).real
    correlations = np.stack([np.einsum("q,qcm,qcm->m", measure, original.conj(), reflected) / norm
                             for reflected in (ry, rz, ryz)], axis=1)
    moment_weights = np.stack([(1 + a * correlations[:, 0].real + b * correlations[:, 1].real
                                + a * b * correlations[:, 2].real) / 4 for a, b in SECTORS], axis=1)
    reflection_errors = np.stack([np.einsum("q,qcm,qcm->m", measure, values.conj(), values).real / norm - 1
                                 for values in (ry, rz, ryz)], axis=1)
    off_diagonal = gram.copy()
    off_diagonal[:, np.arange(4), np.arange(4)] = 0
    orthogonality = np.max(abs(off_diagonal), axis=(1, 2))
    sum_errors = direct.sum(axis=1) - 1
    require(np.all(direct >= -TOLERANCE) and np.all(direct <= 1 + TOLERANCE)
            and np.max(abs(sum_errors)) <= TOLERANCE and np.max(orthogonality) <= TOLERANCE
            and np.max(abs(reflection_errors)) <= TOLERANCE
            and np.max(abs(correlations.imag)) <= TOLERANCE
            and np.max(abs(direct - moment_weights)) <= TOLERANCE,
            "Mirror projector norm/correlation/orthogonality checks failed.")
    modes = []
    for index in range(fields.shape[1]):
        largest = int(np.argmax(direct[index]))
        modes.append({"displacement_mass_norm": float(norm[index] * scales[index]**2),
            "weights_pp_pm_mp_mm": direct[index].tolist(),
            "correlation_weights_pp_pm_mp_mm": moment_weights[index].tolist(),
            "mirror_correlations_y_z_yz_real": correlations[index].real.tolist(),
            "mirror_correlations_y_z_yz_imag": correlations[index].imag.tolist(),
            "normalized_projector_gram_real": gram[index].real.tolist(),
            "normalized_projector_gram_imag": gram[index].imag.tolist(),
            "weight_sum_residual": float(sum_errors[index]),
            "maximum_orthogonality_residual": float(orthogonality[index]),
            "reflection_norm_relative_errors_y_z_yz": reflection_errors[index].tolist(),
            "descriptive_largest_sector": list(SECTORS[largest]),
            "largest_sector_remainder": float(1 - direct[index, largest])})
    return {"grid": grid, "modes": modes}, scalar


def require_participants(name, roots, signed_roles):
    require(name in PARTICIPANTS and list(roots) == list(PARTICIPANTS[name])
            and list(signed_roles) == list(SIGNED_ROLES), "The eight actual signed participant identities differ.")


def read_participants(path, name, ndof):
    """Read selected signed fields only; never body arrays or a mass matrix."""
    require(Path(path).stat().st_size <= 64 * 1024**2, "The section archive exceeds 64 MiB.")
    roots = np.asarray(PARTICIPANTS[name])
    with np.load(path, allow_pickle=False) as archive:
        if name == "z6":
            require(float(archive["frequency_hz"]) == FREQUENCY_HZ, "The section frequency differs.")
        waves = archive["k_rad_m" if name == "z4" else "wavenumbers_rad_m"]
        powers = archive["signed_power_w" if name == "z4" else "powers_w"]
        positive, negative = archive["positive_indices"], archive["negative_indices"]
        propagating, residuals = archive["propagating"], archive["relative_residuals"]
        fields = archive["displacements"]
        require(waves.shape == powers.shape == propagating.shape == residuals.shape == (2 * ndof,)
                and fields.shape == (ndof, 2 * ndof) and fields.dtype == np.complex128
                and positive.shape == negative.shape == (ndof,)
                and np.array_equal(positive[propagating[positive]], accepted.FIXED_ROOTS[name])
                and np.array_equal(negative[propagating[negative]], accepted.FIXED_REVERSE[name]),
                "The saved signed mode inventory differs.")
        require(np.isfinite(waves[roots]).all() and np.all(waves[roots].imag == 0)
                and np.all(waves[roots[:2]].real > 0) and np.all(waves[roots[2:]].real < 0)
                and np.allclose(powers[roots], [1, 1, -1, -1], rtol=0, atol=1e-7)
                and np.all(np.isfinite(residuals[roots]) & (residuals[roots] >= 0) & (residuals[roots] <= 1e-8))
                and np.isfinite(fields[:, roots]).all(), "The actual signed participant fields differ.")
        require(np.allclose(waves[roots[:2]].real, -waves[roots[[3, 2]]].real, rtol=1e-12, atol=0),
                "The selected reverse partners differ.")
        return fields[:, roots].copy(), waves[roots].real, powers[roots], residuals[roots]


def physical_audit(documents, workspace):
    """Audit intended continuum mirrors, without asserting discrete symmetry."""
    identities = []
    for source in PHYSICAL_SOURCES:
        first, second = [documents["h80_" + name + "_summary"]["implementation"]["source_sha256"][source]
                         for name in ("z4", "z6")]
        require(first == second == file_sha(Path(workspace) / source), "The physical source identity differs.")
        identities.append({"path": source, "sha256": first})
    for name in ("z4", "z6"):
        summary = documents["h80_" + name + "_summary"]
        config = summary["configuration"]
        geometry, material = config["geometry"], config["material"]
        require(geometry["length_nm"] == 16650 and geometry["width_nm"] == 1000
                and geometry["thickness_nm"] == 200 and config["mesh"]["kind"] == "layered"
                and config["mesh"]["circle_resolution"] == 32
                and config["analysis"]["boundary"] == "matched_ports"
                and material["crystal_rotation"] == np.eye(3).astype(int).tolist()
                and [material[key] for key in ("C11_Pa", "C12_Pa", "C44_Pa", "density_kg_m3")]
                    == [128.5e9, 48.3e9, 66.8e9, DENSITY], "The intended mirror geometry/material differs.")
        holes = summary["hole_pattern_provenance"]["holes"]
        require(len(holes) == summary["hole_count"] == 26
                and all(set(hole) == {"x_m", "diameter_m", "region"}
                        and np.isfinite([hole["x_m"], hole["diameter_m"]]).all()
                        and hole["diameter_m"] > 0 for hole in holes),
                "The centered through-hole provenance differs.")
    return {"intended_continuum_y_mirror": True, "intended_continuum_z_mirror": True,
        "identity_crystal_rotation": True, "aligned_cubic_tensor": True,
        "centered_circular_through_holes": True, "traction_free_lateral_surfaces": True,
        "unchanged_x_reference_planes_m": [0., 1.665e-5], "discrete_body_symmetry_proved": False,
        "discrete_port_symmetry_proved": False, "radiation_map_symmetry_proved": False,
        "source_identities": identities,
        "evidence": ["Pinned layered source centers each circular hole at y=0 and extrudes the footprint over centered z planes; circle32 geometry is retained.",
            "Identity-oriented cubic stiffness has an even number of each reflected coordinate index in every nonzero component; constant scalar density preserves both mirrors.",
            "Pinned physical assembly/scattering sources retain free lateral surfaces and matched x end planes.",
            "These are intended continuum symmetries. Archived body/port triangulation and the discrete radiation map have not been shown to commute with reflection."]}


def forward_profiles(scalar, fields, components, roots):
    y = np.linspace(-0.5e-6, 0.5e-6, 129)
    sampled = scalar.probes(np.stack((y, np.zeros_like(y)))) @ fields[np.asarray(components)[2], :2]
    return {"coordinates_y_m": y.tolist(), "z_m": 0., "units": "m/sqrt(W)",
        "normalization_applied": False, "phase_rotation_applied": False,
        "modes": [{"root": int(root), "uz_real": sampled[:, i].real.tolist(),
                   "uz_imag": sampled[:, i].imag.tolist()} for i, root in enumerate(roots[:2])]}


def diagnose(workspace=ROOT):
    started = monotonic()
    inputs, paths = [], {}
    for role in ROLE_PINS:
        paths[role] = bind_input(workspace, role, inputs)
    require(paths["conversion_source"] == Path(accepted.__file__).resolve(), "The frozen helper import differs.")
    documents = {role: json.loads(paths[role].read_bytes()) for role in accepted.METADATA_ROLES}
    accepted.validate_metadata(documents)
    ledger = json.loads(paths["conversion_ledger"].read_bytes())
    require(ledger["status"] == "independently_verified_finite_conversion_pair_diagnosis"
            and ledger["frequency_hz"] == FREQUENCY_HZ, "The accepted conversion ledger differs.")
    for record in ledger["inputs"]:
        require(record["role"] in ROLE_PINS and (record["path"], record["sha256"]) == ROLE_PINS[record["role"]],
                "An accepted ledger input has the wrong role.")
    audit = physical_audit(documents, workspace)
    sections, joins = {}, {}
    for name, nz in (("z4", 4), ("z6", 6)):
        with np.load(paths[name + "_section"], allow_pickle=False) as archive:
            vertices, triangles, coordinates, components = [archive[key] for key in
                ("section_vertices_yz_m", "section_triangles", "coordinates_yz_m", "component_dofs_xyz")]
        fields, waves, powers, residuals = read_participants(paths[name + "_section"], name, len(coordinates))
        value, scalar = measure_sectors(vertices_yz_m=vertices, triangles=triangles, coordinates_yz_m=coordinates,
            component_dofs_xyz=components, displacements=fields, shape_cells_yz=(12, nz))
        require(np.allclose(np.asarray(value["grid"]["bounds_yz_m"]), [[-.5e-6, .5e-6], [-.1e-6, .1e-6]],
                            rtol=GEOMETRY_TOLERANCE, atol=0), "The accepted section dimensions differ.")
        roots = PARTICIPANTS[name]
        require_participants(name, roots, SIGNED_ROLES)
        old_modes = {row["root"]: row for row in ledger["sections"][name]["modes"]}
        for i, row in enumerate(value["modes"]):
            root = roots[i]
            row.update(root=root, signed_role=SIGNED_ROLES[i], wavenumber_rad_m=float(waves[i]),
                       signed_power_w=float(powers[i]), relative_residual=float(residuals[i]),
                       template_sector_lower_bounds_pp_pm_mp_mm=None)
            if i < 2:
                old = old_modes[root]
                bounds = np.asarray(old["template_scores_tx_ty_tz_rx"])[[0, 2, 1, 3]]
                require(np.all(np.asarray(row["weights_pp_pm_mp_mm"]) >= bounds - TOLERANCE)
                        and np.isclose(row["displacement_mass_norm"], old["displacement_mass_norm"],
                                       rtol=TOLERANCE, atol=0), "The analytic template-sector lower bounds/norm differ.")
                row["template_sector_lower_bounds_pp_pm_mp_mm"] = bounds.tolist()
        value["forward_profiles"] = forward_profiles(scalar, fields, components, roots)
        sections[name] = value
        by_root = {row["root"]: row for row in value["modes"]}
        directions = ledger["selected_conversion_pairs"]["h80_" + name]["directions"]
        require([(row["incident_root"], row["outgoing_root"]) for row in directions]
                == [(roots[0], roots[2]), (roots[1], roots[3])], "The directed conversion participants differ.")
        joins[name] = [{**row, "incident_weights_pp_pm_mp_mm": by_root[row["incident_root"]]["weights_pp_pm_mp_mm"],
                       "outgoing_weights_pp_pm_mp_mm": by_root[row["outgoing_root"]]["weights_pp_pm_mp_mm"]}
                      for row in directions]
    matched = [{"signed_role": role, "z4_root": PARTICIPANTS["z4"][i], "z6_root": PARTICIPANTS["z6"][i],
                "z6_minus_z4_weights_pp_pm_mp_mm": (np.asarray(sections["z6"]["modes"][i]["weights_pp_pm_mp_mm"])
                    - sections["z4"]["modes"][i]["weights_pp_pm_mp_mm"]).tolist()}
               for i, role in enumerate(SIGNED_ROLES)]
    return {**AUTHOR, "schema_version": 1, "status": "diagnosed_within_saved_sections",
        "question": "Which geometric polar-vector mirror sectors contain the eight actual signed participants in the saved 1 GHz conversion pair?",
        "frequency_hz": FREQUENCY_HZ, "sector_order": [list(sector) for sector in SECTORS], "inputs": inputs,
        "sources": [{"path": "tools/diagnose_mirror_sectors_AP_pnc.py", "sha256": file_sha(__file__)}],
        "physical_audit": audit, "sections": sections, "directed_pair_joins": joins, "matched_rows": matched,
        "accepted_conversion": {key: ledger[key] for key in ("selected_conversion_pairs", "aligned_coefficient_changes_for_selected_pair")},
        "preserved_scientific_status": ledger["preserved_scientific_status"], "numerical_tolerance": TOLERANCE,
        "geometry_relative_agreement_tolerance": GEOMETRY_TOLERANCE,
        "definitions": {"Ry": "diag(1,-1,1) u(-y,z)", "Rz": "diag(1,1,-1) u(y,-z)",
            "projector": "P_ab=(I+a*Ry)(I+b*Rz)/4; complex-linear, no conjugation",
            "weight": "integral rho*abs(P_ab*u)^2 dA / integral rho*abs(u)^2 dA; displacement mass content, not scattering power",
            "correlation_check": "w_ab=(1+a*Re<c_y>+b*Re<c_z>+a*b*Re<c_yz>)/4, normalized by displacement mass norm",
            "descriptive_largest_sector": "Argmax of measured weights in sector_order; first index breaks exact ties, with no physical dominance threshold",
            "template_sector_lower_bounds": "Weights ordered pp,pm,mp,mm are at least accepted Tx,Tz,Ty,Rx template scores for forward roots",
            "profile": "Original complex uz at physical z=0; 129 common uniform y coordinates in SI; no gauge rotation or normalization"},
        "scope": {"body_arrays_loaded": False, "saved_mass_matrix_used": False, "new_safe_or_body_solve": False,
            "discrete_operator_symmetry_proved": False, "conversion_mechanism_established": False,
            "continuous_branch_identified": False, "cross_frequency_tracking_performed": False,
            "spatial_or_frequency_convergence_established": False, "scattering_phase_adjustment_applied": False},
        "limits": ["Joint weights and their finite z4/z6 increments describe these saved fields, with no physical purity threshold or uncertainty bound.",
            "Compatible measured sectors remove one possible parity obstruction only conditional on operator symmetry; they do not explain conversion strength.",
            "Different or mixed sectors motivate checking discrete geometry/port/radiation symmetry and modal degeneracy; section fields alone cannot locate symmetry breaking.",
            "The common refinement integrates P2 products exactly up to the reported raw-coordinate roundoff representation and floating-point arithmetic.",
            "All accepted failed 1 GHz complex-response spatial comparisons remain failed; material calibration, faceting and frequency limits remain unresolved."],
        "elapsed_seconds": monotonic() - started}


def publish(document, output):
    output = Path(output).resolve()
    require(not output.exists(), "Select a fresh output directory; existing results must be preserved.")
    output.mkdir(parents=True)
    columns = ["section", "root", "signed_role", "wavenumber_rad_m", "signed_power_w",
               "weight_pp", "weight_pm", "weight_mp", "weight_mm", "largest_sector_y", "largest_sector_z",
               "largest_sector_remainder"]
    with (output / CSV_NAME).open("x", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=columns)
        writer.writeheader()
        for section, value in document["sections"].items():
            for row in value["modes"]:
                record = {key: row[key] for key in columns if key in row}
                record.update(section=section, **dict(zip(columns[5:9], row["weights_pp_pm_mp_mm"])),
                    largest_sector_y=row["descriptive_largest_sector"][0], largest_sector_z=row["descriptive_largest_sector"][1])
                writer.writerow(record)
    result = {**document, "output_sha256": {CSV_NAME: file_sha(output / CSV_NAME)}}
    with (output / JSON_NAME).open("x", encoding="utf-8") as stream:
        json.dump(result, stream, indent=2, allow_nan=False)
        stream.write("\n")
    return output / JSON_NAME


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", required=True, type=Path, help="Fresh directory for JSON and eight-row CSV")
    args = parser.parse_args()
    require(not args.output.exists(), "Select a fresh output directory.")
    result = diagnose()
    print(json.dumps({"status": result["status"], "json": str(publish(result, args.output)), "rows": 8,
                      "elapsed_seconds": result["elapsed_seconds"]}))


if __name__ == "__main__":
    main()
