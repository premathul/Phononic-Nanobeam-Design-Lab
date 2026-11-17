"""Independently integrate saved P2 lead fields. Athul Prem | _AP_pnc.

The verifier uses explicit triangle shape functions and tensor Gauss quadrature.
It does not import the diagnostic, a finite-element assembler, or a solver.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import itertools
import json
from pathlib import Path

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
TOLERANCE = 1e-10
GEOMETRY_RELATIVE_TOLERANCE = 1e-12
TEMPLATES = ("Tx", "Ty", "Tz", "Rx")
LEDGER_HASHES = {
    "verification/low_band_inplane_AP_pnc.json": "97dc09f33e85ede34cab6ef3eae0367b8bb8b1dec3610d36ea5f6d4c2e786d04",
    "verification/low_band_thickness_AP_pnc.json": "28df92da08af195c5ced321e05d17c14e664a2bf5da5a1594f6ea13a6e096bca",
    "verification/low_band_device_AP_pnc.json": "46df6b610e46867b7bded8f03576d3239504bb92b0a232f9259cae52d5845ddf",
    "verification/port_inventory_AP_pnc.json": "d49f6a54848fc05a2bffdeda1141954a567e449ccbf2983b06c69533eae89a56",
}
PINNED_METADATA_HASHES = {
    "z4_safe_receipt": "59907315426ecf8b240ca8c37a40fe6cb8b019f9b79a941ce7fc65065c771cbe",
    "z6_safe_receipt": "b5c0d73656e90aa5ba312939c0ef25633af5a775974f5fc778031584d55c78f3",
    "thickness_comparison": "0e93087580b622b0a587c94e4a072f405603129ec00bbf91daf329a4784c1591",
    "inplane_comparison": "c8512b0e55c7d0ead6aeab891f93633c6735c3114e6994106f96cd812c1161df",
}


def triangle_quadrature():
    """Nine Duffy points, exact for products of two quadratic P2 fields.

    Map (r,s) in [0,1]^2 to barycentric coordinates
    (1-r, r(1-s), rs); the triangle Jacobian contributes r.
    A degree-four triangle polynomial becomes degree at most five in r
    and four in s, so three-point Gauss-Legendre in each variable is exact.
    Weights sum to 1/2 and must be multiplied by the physical determinant.
    """
    points, weights = np.polynomial.legendre.leggauss(3)
    points, weights = (points + 1) / 2, weights / 2
    barycentric, quadrature_weights = [], []
    for r, wr in zip(points, weights):
        for s, ws in zip(points, weights):
            barycentric.append((1 - r, r * (1 - s), r * s))
            quadrature_weights.append(wr * ws * r)
    barycentric = np.asarray(barycentric)
    shape = np.column_stack((barycentric * (2 * barycentric - 1),
                             4 * barycentric[:, 0] * barycentric[:, 1],
                             4 * barycentric[:, 1] * barycentric[:, 2],
                             4 * barycentric[:, 0] * barycentric[:, 2]))
    return barycentric, np.asarray(quadrature_weights), shape


def _finite_array(value, name, dimensions):
    result = np.asarray(value)
    if result.ndim != dimensions or result.dtype.kind not in "fciu" or not np.isfinite(result).all():
        raise ValueError(f"Invalid finite {name} array.")
    return result


def describe_scores(scores):
    order = np.argsort(scores)
    return TEMPLATES[int(order[-1])] if scores[order[-1]] - scores[order[-2]] > TOLERANCE else "tied"


def p2_element_dofs(vertices, cells, coordinates, components):
    """Locate vertex and edge-midpoint DOFs without element-library numbering."""
    tolerance = 64 * np.finfo(float).eps * max(float(np.ptp(vertices, axis=0).max()),
                                             float(np.abs(vertices).max()))
    local_dofs = np.empty((len(cells), 3, 6), dtype=int)
    used = set()
    for index, triangle in enumerate(vertices[cells]):
        nodes = np.vstack((triangle, (triangle[[0, 1, 0]] + triangle[[1, 2, 2]]) / 2))
        for component, ids in enumerate(components):
            distance = np.max(np.abs(nodes[:, None, :] - coordinates[ids][None, :, :]), axis=2)
            matches = distance <= tolerance
            if not np.all(matches.sum(axis=1) == 1):
                raise ValueError("P2 node/component coordinate map is missing or ambiguous.")
            found = ids[matches.argmax(axis=1)]
            local_dofs[index, component] = found
            used.update(found.tolist())
    if used != set(range(len(coordinates))):
        raise ValueError("The P2 topology does not use every mapped component DOF.")
    return local_dofs


def integrate_section(vertices_yz_m, triangles, coordinates_yz_m, component_dofs_xyz,
                      displacements, density_kg_m3):
    """Integrate fields directly, using only topology, DOF maps and density."""
    vertices = _finite_array(vertices_yz_m, "vertices", 2)
    cells = _finite_array(triangles, "triangles", 2)
    coordinates = _finite_array(coordinates_yz_m, "coordinates", 2)
    components = _finite_array(component_dofs_xyz, "component map", 2)
    values = _finite_array(displacements, "displacements", 2)
    ndof = len(coordinates)
    if (vertices.shape[1:] != (2,) or coordinates.shape[1:] != (2,)
            or np.iscomplexobj(vertices) or np.iscomplexobj(coordinates)
            or cells.shape[1:] != (3,) or cells.dtype.kind not in "iu"
            or components.shape != (3, ndof // 3) or components.dtype.kind not in "iu"
            or ndof % 3 or values.shape[0] != ndof or values.shape[1] < 1
            or len(vertices) < 3 or len(cells) < 1
            or cells.min() < 0 or cells.max() >= len(vertices)
            or not np.array_equal(np.sort(components.ravel()), np.arange(ndof))
            or not np.isfinite(density_kg_m3) or density_kg_m3 <= 0):
        raise ValueError("Invalid section dimensions, component map, topology or density.")
    if (np.unique(np.sort(cells, axis=1), axis=0).shape != cells.shape
            or np.unique(vertices, axis=0).shape != vertices.shape):
        raise ValueError("Repeated section triangle or vertex.")
    corners = vertices[cells]
    edges = corners[:, 1:] - corners[:, :1]
    determinants = np.abs(edges[:, 0, 0] * edges[:, 1, 1] - edges[:, 0, 1] * edges[:, 1, 0])
    scale = float(np.ptp(vertices, axis=0).max())
    if scale <= 0 or np.any(determinants <= np.finfo(float).eps * scale**2):
        raise ValueError("Degenerate section triangle.")
    area = float(determinants.sum() / 2)
    centroid = np.sum(determinants[:, None] * corners.mean(axis=1), axis=0) / (2 * area)
    local_dofs = p2_element_dofs(vertices, cells, coordinates, components)
    barycentric, weights, shape = triangle_quadrature()
    mode_mass = np.zeros(values.shape[1])
    component_mass = np.zeros((3, values.shape[1]))
    gram = np.zeros((4, 4))
    overlaps = np.zeros((4, values.shape[1]), dtype=complex)
    displacement_gram = np.zeros((values.shape[1], values.shape[1]), dtype=complex)
    for triangle, determinant, dofs in zip(corners, determinants, local_dofs):
        xyz = np.einsum("qn,cnm->qcm", shape, values[dofs])
        yz = barycentric @ triangle - centroid
        templates = np.zeros((len(weights), 3, 4))
        templates[:, np.arange(3), np.arange(3)] = 1
        templates[:, 1, 3], templates[:, 2, 3] = -yz[:, 1], yz[:, 0]
        measure = density_kg_m3 * determinant * weights
        components_here = np.einsum("q,qcm->cm", measure, np.abs(xyz)**2)
        component_mass += components_here
        mode_mass += components_here.sum(axis=0)
        gram += np.einsum("q,qci,qcj->ij", measure, templates, templates)
        overlaps += np.einsum("q,qci,qcm->im", measure, templates, xyz)
        displacement_gram += np.einsum("q,qci,qcj->ij", measure, xyz.conj(), xyz)
    if np.any(mode_mass <= 0) or not np.isfinite(mode_mass).all():
        raise ValueError("A saved mode has zero or invalid integrated displacement norm.")
    norms = np.sqrt(np.diag(gram))
    normalized_gram = gram / np.outer(norms, norms)
    scaled_overlaps = overlaps / norms[:, None]
    scores = np.abs(scaled_overlaps)**2 / mode_mass
    # Solve the full Gram problem rather than assuming template orthogonality.
    projection = np.einsum("im,im->m", scaled_overlaps.conj(),
                           np.linalg.solve(normalized_gram, scaled_overlaps)).real / mode_mass
    if (np.any(projection < -TOLERANCE) or np.any(projection > 1 + TOLERANCE)
            or not np.isfinite(scores).all()):
        raise ValueError("Invalid integrated projection or template Gram.")
    return {
        "mass_centroid_yz_m": centroid.tolist(), "area_m2": area,
        "mass_per_length_kg_m": density_kg_m3 * area,
        "normalized_template_gram_real": normalized_gram.tolist(),
        "normalized_template_gram_imag": np.zeros((4, 4)).tolist(),
        "displacement_mass_gram_real": displacement_gram.real.tolist(),
        "displacement_mass_gram_imag": displacement_gram.imag.tolist(),
        "modes": [{"displacement_mass_norm": float(mode_mass[index]),
                   "component_fractions_xyz": (component_mass[:, index] / mode_mass[index]).tolist(),
                   "template_scores_tx_ty_tz_rx": scores[:, index].tolist(),
                   "template_projection_fraction": float(projection[index]),
                   "template_remainder_fraction": float(1 - projection[index]),
                   "descriptive_largest_template": describe_scores(scores[:, index])}
                  for index in range(values.shape[1])],
    }


def interpolate_fields(source, target):
    """Evaluate the coarse P2 field at richer-section DOFs by barycentric search."""
    vertices = np.asarray(source["vertices_yz_m"])
    cells = np.asarray(source["triangles"])
    dofs = p2_element_dofs(vertices, cells, np.asarray(source["coordinates_yz_m"]),
                           np.asarray(source["component_dofs_xyz"]))
    values = np.asarray(source["displacements"])
    result = np.empty((len(target["coordinates_yz_m"]), values.shape[1]), dtype=complex)
    triangles = vertices[cells]
    transforms = np.stack((triangles[:, 1] - triangles[:, 0], triangles[:, 2] - triangles[:, 0]), axis=2)
    inverses = np.linalg.inv(transforms)
    for component, target_ids in enumerate(target["component_dofs_xyz"]):
        for target_id in target_ids:
            point = np.asarray(target["coordinates_yz_m"])[target_id]
            coordinates = np.einsum("tij,tj->ti", inverses, point - triangles[:, 0])
            barycentric = np.column_stack((1 - coordinates.sum(axis=1), coordinates))
            inside = np.flatnonzero((barycentric >= -1e-12).all(axis=1)
                                    & (barycentric <= 1 + 1e-12).all(axis=1))
            if not len(inside):
                raise ValueError("A richer-section coordinate is outside the source triangles.")
            index = inside[0]
            bary = barycentric[index]
            shape = np.r_[bary * (2 * bary - 1), 4 * bary[0] * bary[1],
                           4 * bary[1] * bary[2], 4 * bary[0] * bary[2]]
            result[target_id] = shape @ values[dofs[index, component]]
    return result


def independent_mode_match(source, target):
    """Replay the accepted richer-section mass correspondence independently."""
    first = interpolate_fields(source, target)
    args = dict(target)
    args["displacements"] = np.column_stack((first, target["displacements"]))
    integrated = integrate_section(**args)
    gram = (np.asarray(integrated["displacement_mass_gram_real"])
            + 1j * np.asarray(integrated["displacement_mass_gram_imag"]))
    count = first.shape[1]
    cross = gram[:count, count:] / np.sqrt(np.outer(np.diag(gram[:count, :count]).real,
                                                  np.diag(gram[count:, count:]).real))
    choices = list(itertools.permutations(range(count)))
    chosen = max(choices, key=lambda order: np.abs(cross[np.arange(count), order]).sum())
    overlaps = np.abs(cross[np.arange(count), chosen])
    competitors = np.abs(cross).copy()
    competitors[np.arange(count), chosen] = 0
    if (np.any(overlaps < .99) or np.any(overlaps <= competitors.max(axis=1) + 1e-8)
            or np.any(overlaps <= competitors[:, chosen].max(axis=0) + 1e-8)):
        raise ValueError("The independent signed mode correspondence is ambiguous.")
    return np.asarray(chosen), overlaps


def compare_metrics(actual, expected):
    """Reject altered diagnostic numbers; return maximum dimensionless error."""
    maximum = 0.0
    for name in ("normalized_template_gram_real", "normalized_template_gram_imag"):
        difference = np.abs(np.asarray(actual[name]) - np.asarray(expected[name]))
        if difference.shape != (4, 4) or not np.isfinite(difference).all() or difference.max() > TOLERANCE:
            raise ValueError(f"Independent quadrature disagrees with {name}.")
        maximum = max(maximum, float(difference.max()))
    for name in ("area_m2", "mass_per_length_kg_m"):
        if not np.isclose(actual[name], expected[name], rtol=GEOMETRY_RELATIVE_TOLERANCE, atol=0):
            raise ValueError(f"Independent quadrature disagrees with {name}.")
    scale = np.sqrt(expected["area_m2"])
    if not np.allclose(actual["mass_centroid_yz_m"], expected["mass_centroid_yz_m"], rtol=0,
                       atol=GEOMETRY_RELATIVE_TOLERANCE * scale):
        raise ValueError("Independent quadrature disagrees with mass centroid.")
    if len(actual["modes"]) != len(expected["modes"]):
        raise ValueError("The diagnostic mode count differs.")
    for first, second in zip(actual["modes"], expected["modes"]):
        for name in ("component_fractions_xyz", "template_scores_tx_ty_tz_rx",
                     "template_projection_fraction", "template_remainder_fraction"):
            a, b = np.asarray(first[name]), np.asarray(second[name])
            if a.shape != b.shape or not np.isfinite(a).all() or np.max(np.abs(a - b)) > TOLERANCE:
                raise ValueError(f"Independent quadrature disagrees with {name}.")
            maximum = max(maximum, float(np.max(np.abs(a - b))))
        if not np.isclose(first["displacement_mass_norm"], second["displacement_mass_norm"],
                          rtol=TOLERANCE, atol=0):
            raise ValueError("Independent quadrature disagrees with displacement mass norm.")
        if first["descriptive_largest_template"] != describe_scores(second["template_scores_tx_ty_tz_rx"]):
            raise ValueError("The descriptive largest-template annotation differs.")
    return maximum


def sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _hash_strings(value):
    if isinstance(value, dict):
        return set().union(*(_hash_strings(item) for item in value.values())) if value else set()
    if isinstance(value, list):
        return set().union(*(_hash_strings(item) for item in value)) if value else set()
    if isinstance(value, str) and len(value) == 64 and all(char in "0123456789abcdef" for char in value):
        return {value}
    return set()


def verified_inputs(document, root=ROOT):
    """Bind every input to the pinned accepted ledgers and its actual bytes."""
    accepted_hashes = set(LEDGER_HASHES.values())
    for name, digest in LEDGER_HASHES.items():
        path = root / name
        if sha256(path) != digest:
            raise ValueError("An accepted verification ledger identity differs.")
        accepted_hashes.update(_hash_strings(json.loads(path.read_bytes())))
    roles = {}
    for item in document["inputs"]:
        path = (root / item["path"]).resolve()
        if (not path.is_relative_to(root.resolve()) or item["role"] in roles
                or path.stat().st_size != item["bytes"] or sha256(path) != item["sha256"]
                or item["sha256"] not in accepted_hashes):
            raise ValueError("A scientific input differs from its accepted identity or manifest.")
        roles[item["role"]] = path
    return roles


def bind_scientific_roles(roles, root=ROOT):
    """Bind roles to exact archived relationships, including each response point."""
    def require_identity(role, digest, size=None):
        path = roles.get(role)
        if (path is None or sha256(path) != digest or (size is not None and path.stat().st_size != size)):
            raise ValueError(f"Accepted scientific role identity differs: {role}.")
        return path

    for role, digest in PINNED_METADATA_HASHES.items():
        require_identity(role, digest)
    documents = {role: json.loads(roles[role].read_bytes()) for role in PINNED_METADATA_HASHES}
    ledger_names = {"low_band_inplane_verification": "verification/low_band_inplane_AP_pnc.json",
                    "low_band_thickness_verification": "verification/low_band_thickness_AP_pnc.json",
                    "low_band_device_verification": "verification/low_band_device_AP_pnc.json",
                    "port_inventory_verification": "verification/port_inventory_AP_pnc.json"}
    for role, name in ledger_names.items():
        require_identity(role, LEDGER_HASHES[name])
        documents[role] = json.loads((root / name).read_bytes())
    z4 = documents["z4_safe_receipt"]
    shared = documents["low_band_inplane_verification"]["independent_replay"]["accepted_shared_safe"]
    if shared["sha256"] != PINNED_METADATA_HASHES["z4_safe_receipt"]:
        raise ValueError("The accepted h100 response does not reuse the exact z4 SAFE receipt.")
    links = {"z4_section": z4["safe_section"], "z4_modes": z4["safe_modes"],
             "z6_section": documents["port_inventory_verification"]["archived_section"]["mode_archive"],
             "z6_modes": documents["low_band_device_verification"]["scientific_inputs"]["accepted_safe_modes"]}
    for role, link in links.items():
        require_identity(role, link["sha256"], link["bytes"])
    inplane, thickness = documents["inplane_comparison"], documents["thickness_comparison"]
    records = {"h100_z4": inplane["reference"], "h80_z4": inplane["candidate"], "h80_z6": thickness["candidate"]}
    if records["h80_z4"]["summary_sha256"] != thickness["reference"]["summary_sha256"]:
        raise ValueError("Accepted comparisons do not share the exact h80 z4 response.")
    for point, record in records.items():
        require_identity(f"{point}_summary", record["summary_sha256"])
        require_identity(f"{point}_scattering", record["output_sha256"]["scattering_AP_pnc.csv"])
    return records


def load_section(section_path, modes_path, roots):
    """Load section topology and four saved columns only, never body arrays."""
    if max(section_path.stat().st_size, modes_path.stat().st_size) > 64 * 1024**2:
        raise ValueError("Section-only verifier refuses an oversized archive.")
    keys = {"vertices_yz_m": "section_vertices_yz_m", "triangles": "section_triangles",
            "coordinates_yz_m": "coordinates_yz_m", "component_dofs_xyz": "component_dofs_xyz"}
    with np.load(section_path, allow_pickle=False) as arrays:
        arguments = {name: arrays[key] for name, key in keys.items()}
    with np.load(modes_path, allow_pickle=False) as arrays:
        if float(arrays["frequency_hz"]) != 1e8:
            raise ValueError("The diagnosis must use the saved 0.1 GHz modes.")
        waves = arrays["wavenumbers_rad_m"][roots]
        powers = arrays["powers_w"][roots]
        if (not arrays["propagating"][roots].all() or not np.all(waves.imag == 0)
                or not np.all(waves.real > 0) or not np.allclose(powers, 1, rtol=0, atol=1e-8)
                or not set(roots).issubset(arrays["positive_indices"].tolist())
                or not np.isfinite(arrays["relative_residuals"][roots]).all()
                or np.any(arrays["relative_residuals"][roots] < 0)
                or np.any(arrays["relative_residuals"][roots] > 1e-8)):
            raise ValueError("The saved signed roots are not the accepted incoming modes.")
        arguments["displacements"] = arrays["displacements"][:, roots]
    arguments["density_kg_m3"] = 5323.0
    return arguments, waves.real, powers


def verify_response_rows(responses, sections, roles, records):
    """Join incident CSV order to accepted signed roots, without field loading."""
    result = {}
    for name, section_name in (("h100_z4", "z4"), ("h80_z4", "z4"), ("h80_z6", "z6")):
        summary = json.loads(roles[f"{name}_summary"].read_bytes())
        record = records[name]
        expected_modes = sections[section_name]["modes"]
        if (summary["frequency_hz"] != 1e8 or summary["incident_side"] != "left"
                or summary["incident_mode_indices"] != [mode["root"] for mode in expected_modes]
                or summary["num_incident_channels"] != 4 or summary["power_balance_passed"] is not True
                or summary["run_hash"] != record["run_hash"] or responses[name]["run_hash"] != record["run_hash"]
                or summary["output_sha256"]["scattering_AP_pnc.csv"] != record["output_sha256"]["scattering_AP_pnc.csv"]
                or sha256(roles[f"{name}_scattering"]) != summary["output_sha256"]["scattering_AP_pnc.csv"]):
            raise ValueError("The accepted response summary/root/run/CSV identity differs.")
        with roles[f"{name}_scattering"].open(newline="", encoding="utf-8") as stream:
            rows = list(csv.DictReader(stream))
        actual_rows = responses[name]["rows"]
        if len(rows) != 4 or len(actual_rows) != 4:
            raise ValueError("Response join must contain all four accepted columns.")
        for index, (source, joined, mode) in enumerate(zip(rows, actual_rows, expected_modes)):
            if (int(source["incident_mode_order"]) != index + 1 or float(source["frequency_GHz"]) != .1
                    or not np.isfinite([joined["R"], joined["T"]]).all()
                    or joined["root"] != mode["root"] or joined["reverse_root"] != mode["root"] + 1
                    or abs(joined["R"] - float(source["reflection"])) > TOLERANCE
                    or abs(joined["T"] - float(source["transmission"])) > TOLERANCE):
                raise ValueError("The accepted signed-root/response join differs.")
            for key in ("component_fractions_xyz", "template_scores_tx_ty_tz_rx",
                        "template_projection_fraction", "template_remainder_fraction"):
                if not np.allclose(joined[key], mode[key], rtol=0, atol=TOLERANCE):
                    raise ValueError("Joined polarization metrics differ from their exact section modes.")
        result[name] = {"checked_rows": 4, "R": [float(row["reflection"]) for row in rows]}
    return result


def verify_diagnosis(document, root=ROOT):
    """Independently replay the bounded four-channel diagnosis and data joins."""
    if document["frequency_hz"] != 1e8 or document["numerical_tolerance"] != TOLERANCE:
        raise ValueError("The bounded frequency or implementation-agreement gate differs.")
    for key in ("body_arrays_loaded", "body_fields_loaded", "matrix_assembly_performed", "safe_eigensolve_performed",
                "body_solve_performed", "new_frequency_point_computed", "phase_adjustment_applied",
                "spatial_convergence_established", "frequency_convergence_established",
                "physical_branch_classification_established", "reflection_mechanism_established"):
        if document["scope"][key] is not False:
            raise ValueError("The diagnosis overstates its saved-section scope.")
    roles = verified_inputs(document, root)
    records = bind_scientific_roles(roles, root)
    arguments, sections = {}, {}
    maximum = 0.0
    roots = {"z4": [1348, 1346, 1344, 1294], "z6": [1948, 1916, 1940, 1873]}
    for name in ("z4", "z6"):
        args, waves, powers = load_section(roles[f"{name}_section"], roles[f"{name}_modes"], roots[name])
        expected = integrate_section(**args)
        if (not np.isclose(expected["area_m2"], 2e-13, rtol=GEOMETRY_RELATIVE_TOLERANCE, atol=0)
                or not np.isclose(expected["mass_per_length_kg_m"], 1.0646e-9,
                                  rtol=GEOMETRY_RELATIVE_TOLERANCE, atol=0)):
            raise ValueError("The section area or nominal-Ge mass per length differs.")
        for index, mode in enumerate(expected["modes"]):
            mode.update(root=roots[name][index], wavenumber_rad_m=float(waves[index]),
                        signed_power_w=float(powers[index]))
        actual = document["sections"][name]
        maximum = max(maximum, compare_metrics(actual, expected))
        for first, second in zip(actual["modes"], expected["modes"]):
            if (first["root"] != second["root"] or first["wavenumber_rad_m"] != second["wavenumber_rad_m"]
                    or first["signed_power_w"] != second["signed_power_w"]):
                raise ValueError("Diagnostic roots, wavenumbers or signed powers differ from the saved modes.")
        arguments[name], sections[name] = args, expected
    permutation, overlaps = independent_mode_match(arguments["z4"], arguments["z6"])
    comparison = json.loads(roles["thickness_comparison"].read_bytes())["matched_channels"]["incoming"]
    if (comparison["reference_roots"] != roots["z4"]
            or comparison["candidate_roots"] != [roots["z6"][index] for index in permutation]
            or not np.allclose(comparison["mass_overlaps"], overlaps, rtol=0, atol=TOLERANCE)):
        raise ValueError("Independent correspondence disagrees with the accepted signed matching.")
    if len(document["matched_rows"]) != 4:
        raise ValueError("The matched diagnosis does not include four channels.")
    for index, row in enumerate(document["matched_rows"]):
        first, second = sections["z4"]["modes"][index], sections["z6"]["modes"][permutation[index]]
        if (row["z4_root"] != first["root"] or row["z6_root"] != second["root"]
                or not np.isfinite(row["mass_overlap"]) or abs(row["mass_overlap"] - overlaps[index]) > TOLERANCE):
            raise ValueError("A diagnostic match or overlap differs.")
        if (abs(row["accepted_mass_overlap"] - comparison["mass_overlaps"][index]) > TOLERANCE
                or not np.isfinite(row["accepted_mass_overlap"])):
            raise ValueError("The retained accepted overlap differs.")
        for key in ("component_fractions_xyz", "template_scores_tx_ty_tz_rx"):
            expected = np.asarray(second[key]) - first[key]
            actual = np.asarray(row[f"z6_minus_z4_{key}"])
            if actual.shape != expected.shape or not np.isfinite(actual).all() or np.max(abs(actual - expected)) > TOLERANCE:
                raise ValueError("A diagnostic thickness increment differs.")
        remainder = second["template_remainder_fraction"] - first["template_remainder_fraction"]
        if (not np.isfinite(row["z6_minus_z4_template_remainder_fraction"])
                or abs(row["z6_minus_z4_template_remainder_fraction"] - remainder) > TOLERANCE):
            raise ValueError("A diagnostic remainder increment differs.")
    responses = verify_response_rows(document["responses"], sections, roles, records)
    return {"author": "Athul Prem", "file_tag": "_AP_pnc", "status": "independently_verified_saved_section_diagnosis",
            "method": "Explicit P2 triangle functions with 3x3 Gauss-Legendre and a Duffy map; no assembler or eigensolver.",
            "implementation_agreement_tolerance": TOLERANCE,
            "area_and_mass_relative_tolerance": GEOMETRY_RELATIVE_TOLERANCE,
            "maximum_dimensionless_metric_difference": maximum,
            "independently_matched_overlaps": overlaps.tolist(), "responses": responses,
            "section_rows_checked": 8, "response_rows_checked": 12,
            "spatial_convergence_established": False, "physical_polarization_uncertainty_established": False,
            "body_arrays_loaded": False, "assembly_or_eigensolve_performed": False,
            "scope": "Finite saved-mode displacement-family association only; scores are neither transmitted power nor uncertainty bounds."}


def verify_publication_csv(document, directory):
    name = "port_polarization_AP_pnc.csv"
    path = directory / name
    if sha256(path) != document["output_sha256"][name]:
        raise ValueError("The published diagnostic CSV checksum differs.")
    with path.open(newline="", encoding="utf-8") as stream:
        rows = list(csv.DictReader(stream))
    expected_rows = [(name, response["section"], row) for name, response in document["responses"].items()
                     for row in response["rows"]]
    if len(rows) != len(expected_rows):
        raise ValueError("The diagnostic CSV row count differs.")
    for actual, (point, section, expected) in zip(rows, expected_rows):
        if (actual["point"] != point or actual["section"] != section
                or int(actual["root"]) != expected["root"] or int(actual["reverse_root"]) != expected["reverse_root"]
                or actual["descriptive_largest_template"] != expected["descriptive_largest_template"]):
            raise ValueError("The diagnostic CSV root/annotation join differs.")
        values = {name: expected[name] for name in ("wavenumber_rad_m", "signed_power_w", "R", "T",
                                                    "template_projection_fraction", "template_remainder_fraction")}
        values.update(zip(("kinetic_x", "kinetic_y", "kinetic_z"), expected["component_fractions_xyz"]))
        values.update(zip(("score_tx", "score_ty", "score_tz", "score_rx"), expected["template_scores_tx_ty_tz_rx"]))
        for name, value in values.items():
            number = float(actual[name])
            if not np.isfinite(number) or abs(number - value) > TOLERANCE:
                raise ValueError("The diagnostic CSV numeric value differs.")
    return len(rows)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--diagnosis", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists() or args.output.resolve() == args.diagnosis.resolve():
        raise ValueError("Independent verification requires a new output path.")
    document = json.loads(args.diagnosis.read_bytes())
    result = verify_diagnosis(document)
    result["publication_csv_rows_checked"] = verify_publication_csv(document, args.diagnosis.parent)
    result["diagnosis"] = {"path": str(args.diagnosis.resolve()), "bytes": args.diagnosis.stat().st_size,
                           "sha256": sha256(args.diagnosis)}
    result["verifier_sha256"] = sha256(Path(__file__))
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    print(json.dumps({"status": result["status"], "output": str(args.output.resolve()),
                      "maximum_dimensionless_metric_difference": result["maximum_dimensionless_metric_difference"]}))


if __name__ == "__main__":
    main()
