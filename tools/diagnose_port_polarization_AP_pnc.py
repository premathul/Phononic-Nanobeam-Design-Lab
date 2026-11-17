"""Diagnose saved 0.1 GHz lead fields without a solve. Athul Prem | _AP_pnc.

The four template scores are dimensionless mass-inner-product diagnostics,
not effective masses, power fractions or a physical branch classification.
Only archived section arrays and accepted response metadata are read.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
from importlib.metadata import version
import json
import math
import os
from pathlib import Path
import platform
import sys
from time import monotonic

if __name__ == "__main__":
    os.environ.update(dict.fromkeys(("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS",
                                    "BLIS_NUM_THREADS", "NUMEXPR_NUM_THREADS"), "1"))

import numpy as np
from skfem import Basis, ElementTriP2, ElementVector, MeshTri

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT), str(ROOT / "src")]

from tools.compare_scattering_AP_pnc import Modes, Section, align, interpolation


AUTHOR = {"author": "Athul Prem", "file_tag": "_AP_pnc"}
FREQUENCY_HZ = 100_000_000
DENSITY_KG_M3 = 5323.0
NUMERICAL_TOLERANCE = 1e-10
TEMPLATES = ("Tx", "Ty", "Tz", "Rx")
Z4_ROOTS = (1348, 1346, 1344, 1294)
Z6_ROOTS = (1948, 1916, 1940, 1873)
JSON_NAME = "port_polarization_AP_pnc.json"
CSV_NAME = "port_polarization_AP_pnc.csv"

# Immutable accepted evidence, rather than a live solver/cache compatibility rule.
PINNED_JSON = {
    "low_band_inplane_verification": ("verification/low_band_inplane_AP_pnc.json",
        "97dc09f33e85ede34cab6ef3eae0367b8bb8b1dec3610d36ea5f6d4c2e786d04"),
    "low_band_thickness_verification": ("verification/low_band_thickness_AP_pnc.json",
        "28df92da08af195c5ced321e05d17c14e664a2bf5da5a1594f6ea13a6e096bca"),
    "low_band_device_verification": ("verification/low_band_device_AP_pnc.json",
        "46df6b610e46867b7bded8f03576d3239504bb92b0a232f9259cae52d5845ddf"),
    "port_inventory_verification": ("verification/port_inventory_AP_pnc.json",
        "d49f6a54848fc05a2bffdeda1141954a567e449ccbf2983b06c69533eae89a56"),
    "z4_safe_receipt": ("results/runs/low_band_thickness_AP_pnc/h80-z4-low-band-safe-1335500/safe_AP_pnc.json",
        "59907315426ecf8b240ca8c37a40fe6cb8b019f9b79a941ce7fc65065c771cbe"),
    "z6_safe_receipt": ("results/runs/full_device_resource_profile/lawrence_thickness_AP_pnc/"
        "thickness-h80-safe-1335308/thickness_h80_safe_AP_pnc.json",
        "b5c0d73656e90aa5ba312939c0ef25633af5a775974f5fc778031584d55c78f3"),
    "thickness_comparison": ("results/runs/low_band_thickness_AP_pnc/comparison/summary_AP_pnc.json",
        "0e93087580b622b0a587c94e4a072f405603129ec00bbf91daf329a4784c1591"),
    "inplane_comparison": ("results/runs/low_band_inplane_AP_pnc/comparison/summary_AP_pnc.json",
        "c8512b0e55c7d0ead6aeab891f93633c6735c3114e6994106f96cd812c1161df"),
}
SECTION_PATHS = {
    "z4": "results/runs/full_device_port_resolution/12x4/attempt_lpnbd9kr/arrays_AP_pnc.npz",
    "z6": "results/runs/full_device_resource_profile/lawrence_thickness_AP_pnc/"
          "thickness-h80-safe-1335308/ports_AP_pnc.npz",
}
MODE_PATHS = {
    "z4": "results/runs/low_band_thickness_AP_pnc/h80-z4-low-band-safe-1335500/modes-100000000_AP_pnc.npz",
    "z6": "results/runs/port_frequency_inventory_AP_pnc/port-inventory-1335406/modes-100000000_AP_pnc.npz",
}
POINT_PATHS = {
    "h100_z4": "results/runs/low_band_inplane_AP_pnc/h100-z4-low-band-export-1335527/"
               "historical_26hole_layered_h100_ny12_z4-d80b611a700a",
    "h80_z4": "results/runs/low_band_thickness_AP_pnc/h80-z4-low-band-export-1335512/"
              "historical_26hole_layered_h80_ny12_z4-a80dd9c4d431",
    "h80_z6": "results/runs/low_band_device_AP_pnc/h80-z6-low-band-export-1335483/"
              "historical_26hole_layered_h80_ny12_z6-bb55473510a2",
}


def require(condition, message):
    if not condition:
        raise ValueError(message)


def file_sha(path):
    with Path(path).open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def checked_input(inputs, role, path, sha256, size=None):
    """Read only an artifact whose immutable accepted identity matches."""
    path = Path(path).resolve()
    require(path.is_file() and (size is None or path.stat().st_size == size)
            and file_sha(path) == sha256, f"Accepted artifact identity differs: {role}.")
    require(role not in {item["role"] for item in inputs}, f"Duplicate input role: {role}.")
    inputs.append({"role": role, "path": str(path), "bytes": path.stat().st_size, "sha256": sha256})
    return path


def diagnose_modes(*, coordinates_yz_m, component_dofs_xyz, mass_kg_per_m, displacements):
    """Return phase/scale-invariant diagnostics in a consistent vector mass metric.

    Component maps must share an identical scalar coordinate order and identical,
    uncoupled Cartesian mass blocks. The four centered rigid templates are
    orthogonal in this metric. Their scores plus the unexplained remainder
    partition the mass norm; only the three Cartesian fractions partition
    Cartesian kinetic energy. Neither partition measures scattering power.
    """
    coordinates, components = np.asarray(coordinates_yz_m), np.asarray(component_dofs_xyz)
    matrix, fields = np.asarray(mass_kg_per_m), np.asarray(displacements)
    n = len(coordinates)
    require(coordinates.shape == (n, 2) and coordinates.dtype.kind == "f"
            and np.isfinite(coordinates).all(), "Invalid transverse coordinates.")
    require(components.ndim == 2 and components.shape[0] == 3 and components.shape[1] * 3 == n
            and components.dtype.kind in "iu" and np.array_equal(np.sort(components.ravel()), np.arange(n)),
            "The Cartesian component map is not a complete unique partition.")
    scalar_coordinates = coordinates[components[0]]
    require(all(np.array_equal(coordinates[ids], scalar_coordinates) for ids in components),
            "Cartesian components do not share the exact scalar coordinate order.")
    require(len(np.unique(scalar_coordinates, axis=0)) == len(scalar_coordinates),
            "Scalar transverse coordinates contain duplicates.")
    require(matrix.shape == (n, n) and matrix.dtype.kind in "fc" and np.isfinite(matrix).all(),
            "Invalid consistent section mass matrix.")
    scale = float(np.max(abs(matrix)))
    require(scale > 0 and np.max(abs(matrix - matrix.conj().T)) <= NUMERICAL_TOLERANCE * scale,
            "The section mass matrix is not Hermitian.")
    scalar_mass = matrix[np.ix_(components[0], components[0])]
    for a in range(3):
        for b in range(3):
            block = matrix[np.ix_(components[a], components[b])]
            expected = scalar_mass if a == b else 0
            require(np.max(abs(block - expected)) <= NUMERICAL_TOLERANCE * scale,
                    "The mass matrix does not have equal uncoupled Cartesian blocks.")
    require(np.max(abs(matrix.imag)) <= NUMERICAL_TOLERANCE * scale,
            "Physical consistent mass must be real.")
    require(fields.ndim == 2 and fields.shape[0] == n and fields.shape[1] > 0
            and fields.dtype.kind in "fc" and np.isfinite(fields).all(), "Invalid saved displacement fields.")
    one = np.ones(len(scalar_coordinates))
    mass_length = float(np.real(one @ scalar_mass @ one))
    require(math.isfinite(mass_length) and mass_length > 0, "Nonpositive rigid-translation mass norm.")
    centroid = np.real(one @ scalar_mass @ scalar_coordinates) / mass_length
    templates = np.zeros((n, 4), dtype=float)
    for c, ids in enumerate(components):
        templates[ids, c] = 1
    centered = scalar_coordinates - centroid
    templates[components[1], 3] = -centered[:, 1]
    templates[components[2], 3] = centered[:, 0]
    mass_templates = matrix @ templates
    gram = templates.conj().T @ mass_templates
    norms = gram.diagonal().real
    require(np.all(np.isfinite(norms) & (norms > 0)), "A rigid template has nonpositive mass norm.")
    normalized_gram = gram / np.sqrt(np.outer(norms, norms))
    gram_error = float(np.max(abs(normalized_gram - np.eye(4))))
    require(gram_error <= NUMERICAL_TOLERANCE, "Centered rigid templates are not mass orthogonal.")

    # Rescaling before inner products prevents harmless modal normalization from
    # degrading the dimensionless scores. The original dimensional norm is kept.
    field_scales = np.max(abs(fields), axis=0)
    require(np.all(field_scales > 0), "A displacement field is zero.")
    normalized_fields = fields / field_scales
    mass_fields = matrix @ normalized_fields
    field_products = normalized_fields.conj() * mass_fields
    mass_norms = np.sum(field_products, axis=0)
    require(np.all(np.isfinite(mass_norms)) and np.all(mass_norms.real > 0)
            and np.all(abs(mass_norms.imag) <= NUMERICAL_TOLERANCE * mass_norms.real),
            "A displacement field has a nonpositive or nonreal mass norm.")
    component_fractions = np.stack([field_products[ids].sum(axis=0).real / mass_norms.real
                                    for ids in components], axis=1)
    scores = (abs(templates.conj().T @ mass_fields)**2
              / (norms[:, None] * mass_norms.real[None, :])).T
    projection = scores.sum(axis=1)
    remainder = 1 - projection
    require(np.isfinite(component_fractions).all() and np.isfinite(scores).all()
            and np.all(component_fractions >= -NUMERICAL_TOLERANCE)
            and np.all(component_fractions <= 1 + NUMERICAL_TOLERANCE)
            and np.all(abs(component_fractions.sum(axis=1) - 1) <= NUMERICAL_TOLERANCE),
            "Cartesian kinetic fractions do not form a nonnegative unit partition.")
    require(np.all(scores >= 0) and np.all(scores <= 1 + NUMERICAL_TOLERANCE)
            and np.all(remainder >= -NUMERICAL_TOLERANCE) and np.all(remainder <= 1 + NUMERICAL_TOLERANCE),
            "Rigid-template projection is outside the unit mass norm.")
    original_norms = mass_norms.real * field_scales**2
    require(np.all(np.isfinite(original_norms) & (original_norms > 0)),
            "The original dimensional displacement mass norm cannot be represented.")
    modes = []
    for i in range(fields.shape[1]):
        order = np.argsort(scores[i])
        largest = TEMPLATES[int(order[-1])] if scores[i, order[-1]] - scores[i, order[-2]] > NUMERICAL_TOLERANCE else "tied"
        modes.append({"displacement_mass_norm": float(original_norms[i]),
                      "component_fractions_xyz": component_fractions[i].tolist(),
                      "template_scores_tx_ty_tz_rx": scores[i].tolist(),
                      "template_projection_fraction": float(projection[i]),
                      "template_remainder_fraction": float(remainder[i]),
                      "descriptive_largest_template": largest})
    return {"mass_centroid_yz_m": centroid.tolist(), "mass_per_length_kg_m": mass_length,
            "normalized_template_gram_real": normalized_gram.real.tolist(),
            "normalized_template_gram_imag": normalized_gram.imag.tolist(),
            "maximum_template_gram_error": gram_error, "modes": modes}


def read_section(path, name):
    """Check the archived P2 order without assembling any matrix."""
    expected_dofs = 675 if name == "z4" else 975
    with np.load(path, allow_pickle=False) as archive:
        data = {key: archive[key] for key in ("section_vertices_yz_m", "section_triangles",
                                             "coordinates_yz_m", "component_dofs_xyz")}
        data["mass_kg_per_m"] = archive["mass_kg_per_m" if name == "z4" else "M"]
    vertices, triangles = data["section_vertices_yz_m"], data["section_triangles"]
    require(vertices.ndim == 2 and vertices.shape[1] == 2 and np.isfinite(vertices).all()
            and triangles.ndim == 2 and triangles.shape[1] == 3 and triangles.dtype.kind in "iu"
            and triangles.min() >= 0 and triangles.max() < len(vertices), "Invalid archived section topology.")
    mesh = MeshTri(np.ascontiguousarray(vertices.T), np.ascontiguousarray(triangles.T))
    vector = Basis(mesh, ElementVector(ElementTriP2(), dim=3), intorder=4)
    scalar = Basis(mesh, ElementTriP2(), intorder=4)
    require(vector.N == expected_dofs and np.array_equal(mesh.p.T, vertices)
            and np.array_equal(mesh.t.T, triangles)
            and np.array_equal(vector.doflocs.T, data["coordinates_yz_m"])
            and np.array_equal(vector.split_indices(), data["component_dofs_xyz"]),
            "The archived P2 coordinates or Cartesian order differ from their topology.")
    bounds = np.array([1e-6, 200e-9])
    require(np.allclose(vertices.min(axis=0) / bounds, -0.5, rtol=0, atol=NUMERICAL_TOLERANCE)
            and np.allclose(vertices.max(axis=0) / bounds, 0.5, rtol=0, atol=NUMERICAL_TOLERANCE)
            and math.isclose(float(scalar.dx.sum()), float(np.prod(bounds)), rel_tol=NUMERICAL_TOLERANCE),
            "The archived section does not have the accepted physical width, thickness and area.")
    ids = data["component_dofs_xyz"]
    section = Section(scalar, ids, data["mass_kg_per_m"][np.ix_(ids[0], ids[0])], file_sha(path))
    return data, section


def read_modes(path, roots, expected_dofs):
    """Read four selected saved fields and their signed root metadata."""
    roots = np.asarray(roots, dtype=int)
    with np.load(path, allow_pickle=False) as archive:
        frequency = float(archive["frequency_hz"])
        k, powers, propagating = (archive[key] for key in ("wavenumbers_rad_m", "powers_w", "propagating"))
        positive, negative = archive["positive_indices"], archive["negative_indices"]
        residuals = archive["relative_residuals"]
        require(frequency == FREQUENCY_HZ and k.shape == powers.shape == propagating.shape == residuals.shape == (2 * expected_dofs,)
                and propagating.dtype == np.bool_ and positive.shape == negative.shape == (expected_dofs,)
                and positive.dtype.kind in "iu" and negative.dtype.kind in "iu"
                and np.array_equal(np.sort(np.r_[positive, negative]), np.arange(2 * expected_dofs)),
                "Saved mode dimensions, frequency or signed root partition differ.")
        reverse = negative[propagating[negative]]
        require(np.array_equal(positive[propagating[positive]], roots) and len(reverse) == 4
                and np.all(np.isfinite(k[roots])) and np.all(k[roots].imag == 0) and np.all(k[roots].real > 0)
                and np.allclose(k[reverse], -k[roots], rtol=1e-12, atol=0)
                and np.allclose(powers[roots], 1, rtol=0, atol=1e-7)
                and np.allclose(powers[reverse], -1, rtol=0, atol=1e-7)
                and np.all(np.isfinite(residuals[np.r_[roots, reverse]]))
                and np.all((residuals[np.r_[roots, reverse]] >= 0) & (residuals[np.r_[roots, reverse]] <= 1e-8)),
                "Saved propagating order, reverse partners, normalization or accepted residuals differ.")
        full = archive["displacements"]
        require(full.shape == (expected_dofs, 2 * expected_dofs) and full.dtype == np.complex128,
                "Saved displacement columns do not have the accepted SAFE dimensions/type.")
        selected = full[:, roots].copy()
    return {"roots": roots, "reverse_roots": reverse, "k": k[roots].real,
            "powers": powers[roots], "displacements": selected}


def read_response_rows(path, summary, modes):
    roots = modes["roots"].tolist()
    require(summary["frequency_hz"] == FREQUENCY_HZ and summary["incident_side"] == "left"
            and summary["incident_mode_indices"] == roots and summary["num_incident_channels"] == 4
            and summary["power_balance_passed"] is True, "Accepted response root/frequency/power identity differs.")
    with Path(path).open(newline="", encoding="utf-8") as stream:
        rows = list(csv.DictReader(stream))
    require(len(rows) == 4 and [int(row["incident_mode_order"]) for row in rows] == [1, 2, 3, 4],
            "Accepted scattering CSV row order differs.")
    result = []
    for i, row in enumerate(rows):
        reflection, transmission = float(row["reflection"]), float(row["transmission"])
        require(float(row["frequency_GHz"]) * 1e9 == FREQUENCY_HZ
                and all(math.isfinite(value) and value >= 0 for value in (reflection, transmission))
                and abs(reflection + transmission - 1) < 0.01, "Invalid accepted CSV frequency or R/T power budget.")
        result.append({"root": roots[i], "reverse_root": int(modes["reverse_roots"][i]),
                       "R": reflection, "T": transmission})
    return result


def check_comparison_rows(comparison, reference, candidate):
    require(comparison["status"] == "compared" and comparison["mesh_pair_passed"] is True
            and comparison["phase_adjustment_applied"] is False
            and comparison["spatial_convergence_established"] is False
            and comparison["frequency_convergence_established"] is False, "Accepted finite comparison status differs.")
    for quantity, key in (("reflection", "R"), ("transmission", "T")):
        saved = [row for row in comparison["rows"] if row["quantity"] == quantity]
        require(len(saved) == 4, "An accepted comparison is missing response rows.")
        for row, a, b in zip(saved, reference, candidate, strict=True):
            require(row["reference_incident_root"] == a["root"] and row["candidate_incident_root"] == b["root"]
                    and math.isclose(row["reference_power"], a[key], rel_tol=0, abs_tol=NUMERICAL_TOLERANCE)
                    and math.isclose(row["candidate_power"], b[key], rel_tol=0, abs_tol=NUMERICAL_TOLERANCE),
                    "Accepted comparison roots or R/T rows disagree with their original CSVs.")


def diagnose(workspace=ROOT):
    """Create the bounded diagnosis in memory. No archive is modified."""
    started = monotonic()
    workspace, inputs, documents = Path(workspace).resolve(), [], {}
    for role, (relative, digest) in PINNED_JSON.items():
        path = checked_input(inputs, role, workspace / relative, digest)
        documents[role] = json.loads(path.read_bytes())
    z4_safe = documents["z4_safe_receipt"]
    inventory = documents["port_inventory_verification"]
    low_band = documents["low_band_device_verification"]
    inplane, thickness = documents["inplane_comparison"], documents["thickness_comparison"]
    require(documents["low_band_inplane_verification"]["independent_replay"]["accepted_shared_safe"]["sha256"]
            == PINNED_JSON["z4_safe_receipt"][1], "h100 does not reuse the exact accepted z4 SAFE receipt.")
    require(z4_safe["frequency_hz"] == FREQUENCY_HZ and z4_safe["incident_port_roots"] == list(Z4_ROOTS),
            "Accepted z4 SAFE receipt differs.")
    links = {
        "z4_section": z4_safe["safe_section"], "z4_modes": z4_safe["safe_modes"],
        "z6_section": inventory["archived_section"]["mode_archive"],
        "z6_modes": low_band["scientific_inputs"]["accepted_safe_modes"],
    }
    sections, raw, matching_sections = {}, {}, {}
    for name, roots in (("z4", Z4_ROOTS), ("z6", Z6_ROOTS)):
        paths = {}
        for kind, relatives in (("section", SECTION_PATHS), ("modes", MODE_PATHS)):
            role, link = f"{name}_{kind}", links[f"{name}_{kind}"]
            paths[kind] = checked_input(inputs, role, workspace / relatives[name], link["sha256"], link["bytes"])
        data, matching_sections[name] = read_section(paths["section"], name)
        raw[name] = read_modes(paths["modes"], roots, len(data["coordinates_yz_m"]))
        sections[name] = diagnose_modes(coordinates_yz_m=data["coordinates_yz_m"],
            component_dofs_xyz=data["component_dofs_xyz"], mass_kg_per_m=data["mass_kg_per_m"],
            displacements=raw[name]["displacements"])
        sections[name].update({"area_m2": sections[name]["mass_per_length_kg_m"] / DENSITY_KG_M3,
                              "density_kg_m3": DENSITY_KG_M3,
                              "displacement_dofs": len(data["coordinates_yz_m"]),
                              "mode_order": list(roots)})
        require(math.isclose(sections[name]["area_m2"], 2e-13, rel_tol=NUMERICAL_TOLERANCE),
                "The saved rigid mass norm does not recover the accepted section area.")
        for i, mode in enumerate(sections[name]["modes"]):
            mode.update({"root": int(roots[i]), "reverse_root": int(raw[name]["reverse_roots"][i]),
                         "wavenumber_rad_m": float(raw[name]["k"][i]), "signed_power_w": float(raw[name]["powers"][i])})
        del data

    a, b = matching_sections["z4"], matching_sections["z6"]
    def matching_modes(name, section):
        data = raw[name]
        return Modes(data["roots"], data["k"], data["displacements"][section.trace_to_scalar])
    order, _, match = align(matching_modes("z4", a), matching_modes("z6", b), b.mass, interpolation(a, b), True)
    accepted = thickness["matched_channels"]["incoming"]
    require(match["reference_roots"] == accepted["reference_roots"]
            and match["candidate_roots"] == accepted["candidate_roots"]
            and np.allclose(match["mass_overlaps"], accepted["mass_overlaps"], rtol=0, atol=NUMERICAL_TOLERANCE)
            and np.array_equal(order, np.arange(4)), "Replayed signed section correspondence differs from the accepted comparison.")
    for name, record, prefix in (("z4", accepted, "reference"), ("z6", accepted, "candidate")):
        require(np.array_equal(raw[name]["k"], record[f"{prefix}_wavenumber_rad_m"]),
                "Accepted matched wavenumbers differ from the saved frequency-specific modes.")
    shared = inplane["matched_channels"]["incoming"]
    require(shared["reference_roots"] == shared["candidate_roots"] == list(Z4_ROOTS)
            and shared["reference_wavenumber_rad_m"] == shared["candidate_wavenumber_rad_m"] == raw["z4"]["k"].tolist(),
            "The h100/h80 join does not have identical z4 incident roots and wavenumbers.")

    response_records = {"h100_z4": inplane["reference"], "h80_z4": inplane["candidate"], "h80_z6": thickness["candidate"]}
    require(response_records["h80_z4"]["summary_sha256"] == thickness["reference"]["summary_sha256"],
            "The two accepted controls do not share the exact h80 z4 response.")
    responses = {}
    for point, relative in POINT_PATHS.items():
        section = point.split("_")[-1]
        record, directory = response_records[point], workspace / relative
        summary_path = checked_input(inputs, f"{point}_summary", directory / "summary_AP_pnc.json", record["summary_sha256"])
        summary = json.loads(summary_path.read_bytes())
        require(summary["run_hash"] == record["run_hash"] and summary["configuration"]["material"]["density_kg_m3"] == DENSITY_KG_M3,
                "Accepted run identity or density differs.")
        digest = record["output_sha256"]["scattering_AP_pnc.csv"]
        require(summary["output_sha256"]["scattering_AP_pnc.csv"] == digest, "CSV identity differs between accepted records.")
        path = checked_input(inputs, f"{point}_scattering", directory / "scattering_AP_pnc.csv", digest)
        rows = read_response_rows(path, summary, raw[section])
        for row, mode in zip(rows, sections[section]["modes"], strict=True):
            row.update(mode)
        responses[point] = {"section": section, "run_hash": record["run_hash"], "rows": rows}
    check_comparison_rows(thickness, responses["h80_z4"]["rows"], responses["h80_z6"]["rows"])
    check_comparison_rows(inplane, responses["h100_z4"]["rows"], responses["h80_z4"]["rows"])
    matched_rows = []
    for i, (z4, z6) in enumerate(zip(sections["z4"]["modes"], sections["z6"]["modes"], strict=True)):
        matched_rows.append({"z4_root": z4["root"], "z6_root": z6["root"],
            "mass_overlap": match["mass_overlaps"][i], "accepted_mass_overlap": accepted["mass_overlaps"][i],
            "z6_minus_z4_component_fractions_xyz": (np.array(z6["component_fractions_xyz"]) - z4["component_fractions_xyz"]).tolist(),
            "z6_minus_z4_template_scores_tx_ty_tz_rx": (np.array(z6["template_scores_tx_ty_tz_rx"]) - z4["template_scores_tx_ty_tz_rx"]).tolist(),
            "z6_minus_z4_template_remainder_fraction": z6["template_remainder_fraction"] - z4["template_remainder_fraction"]})
    return {**AUTHOR, "schema_version": 1, "status": "diagnosed_within_saved_sections",
        "question": "Which mass-weighted lead-field templates associate with the persistent 0.1 GHz reflection groups?",
        "frequency_hz": FREQUENCY_HZ, "inputs": inputs, "sections": sections,
        "responses": responses, "matched_rows": matched_rows,
        "numerical_tolerance": NUMERICAL_TOLERANCE,
        "definitions": {"component_fractions_xyz": "Re(u_c^H M_c u_c)/(u^H M u), lab axes x/y/z",
            "template_order": list(TEMPLATES), "templates": ["(1,0,0)", "(0,1,0)", "(0,0,1)", "(0,-(z-z_c),y-y_c)"],
            "template_score": "abs(q^H M u)^2 / ((q^H M q)*(u^H M u))",
            "template_projection_fraction": "Sum of the four scores after checking normalized template Gram=I",
            "template_remainder_fraction": "1 - template_projection_fraction; content outside these four rigid templates",
            "mass_centroid": "Consistent scalar-mass integral of (y,z) divided by section mass per unit length",
            "displacement_mass_norm": "Original u^H M u; for saved m/sqrt(W) fields, units kg*m/W",
            "roundoff_policy": "Raw values retained, including boundary excursions no larger than numerical_tolerance",
            "descriptive_largest_template": "Numerically largest tested template; tied if indistinguishable within numerical_tolerance. No physical dominance threshold."},
        "scope": {"body_arrays_loaded": False, "body_fields_loaded": False, "matrix_assembly_performed": False,
            "safe_eigensolve_performed": False, "body_solve_performed": False, "new_frequency_point_computed": False,
            "phase_adjustment_applied": False, "spatial_convergence_established": False, "frequency_convergence_established": False,
            "physical_branch_classification_established": False, "reflection_mechanism_established": False,
            "unknown_or_mixed_content_retained": True},
        "limits": ["A finite-section association, not a causal explanation or continuum polarization assignment.",
            "Template scores and Cartesian kinetic fractions are not scattering-power fractions or commercial effective masses.",
            "Physical Ge properties are nominal fixed-property controls, not measured mK/4 K sample data.",
            "Failed 1 GHz spatial checks, hole faceting and frequency sampling remain unresolved; no 1 GHz modes are analyzed.",
            "The archived global in-plane increment also changes lead interiors; uniform-guide phase is not subtracted."],
        "sources": [{"path": str(Path(__file__).resolve()), "sha256": file_sha(__file__)},
                    {"path": str((ROOT / "tools/compare_scattering_AP_pnc.py").resolve()),
                     "sha256": file_sha(ROOT / "tools/compare_scattering_AP_pnc.py")}],
        "environment": {"python": platform.python_version(), "numpy": np.__version__,
            "scipy": version("scipy"), "scikit_fem": version("scikit-fem"), "platform": platform.platform(),
            "numerical_thread_environment": {name: os.environ.get(name) for name in (
                "OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "BLIS_NUM_THREADS", "NUMEXPR_NUM_THREADS")}},
        "elapsed_seconds": monotonic() - started}


def publish(document, output):
    """Publish only a new compact JSON/CSV pair, preserving every source artifact."""
    output = Path(output).resolve()
    require(not any((output / name).exists() for name in (JSON_NAME, CSV_NAME)),
            "Diagnostic output already exists; preserve it and select a new destination.")
    output.mkdir(parents=True, exist_ok=True)
    columns = ["point", "section", "root", "reverse_root", "wavenumber_rad_m", "signed_power_w", "R", "T",
               "kinetic_x", "kinetic_y", "kinetic_z", "score_tx", "score_ty", "score_tz", "score_rx",
               "template_projection_fraction", "template_remainder_fraction", "descriptive_largest_template"]
    with (output / CSV_NAME).open("x", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=columns)
        writer.writeheader()
        for point, response in document["responses"].items():
            for row in response["rows"]:
                values = {key: row[key] for key in columns if key in row}
                values.update({"point": point, "section": response["section"]})
                values.update(zip(("kinetic_x", "kinetic_y", "kinetic_z"), row["component_fractions_xyz"]))
                values.update(zip(("score_tx", "score_ty", "score_tz", "score_rx"), row["template_scores_tx_ty_tz_rx"]))
                writer.writerow(values)
    document = {**document, "output_sha256": {CSV_NAME: file_sha(output / CSV_NAME)}}
    with (output / JSON_NAME).open("x", encoding="utf-8") as stream:
        json.dump(document, stream, indent=2, allow_nan=False)
        stream.write("\n")
    return output / JSON_NAME


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    result = diagnose()
    path = publish(result, args.output)
    print(json.dumps({"status": result["status"], "json": str(path), "rows": 12,
                      "elapsed_seconds": result["elapsed_seconds"]}))


if __name__ == "__main__":
    main()
