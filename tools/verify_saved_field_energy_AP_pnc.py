"""Independent saved P2 energy and tetra10 verification. Athul Prem | _AP_pnc.

The verifier uses explicit quadratic tetrahedron functions, a 4^3 Duffy rule,
and case-based plane clipping. It imports neither the producer nor its convex
hull geometry or quadrature implementation. No assembly or solve is performed.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import platform
from time import monotonic
from math import factorial
from importlib.metadata import version

if __name__ == "__main__":
    os.environ.update(dict.fromkeys(("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS",
                                    "BLIS_NUM_THREADS", "NUMEXPR_NUM_THREADS"), "1"))

import numpy as np
import psutil

AUTHOR = {"author": "Athul Prem", "file_tag": "_AP_pnc"}
COMPARISON_SHA = "107d1c62817349d0a6900650d10dd3826ee492ca4bc36abc8195ee3c3e7911c7"
REPLAY_SHAS = {"h100": "ac9d1ccd9d5346102d97d03b2dac7e47fc3cb319f99d67d8a1d33bba087fc2b9",
               "h80": "17bcad69b7d567331489d32a798f6755df292272473df78e9338ba259683fd34"}
FREQUENCY, ROOT, COLUMN = 887_500_000, 1302, 0
CUTS = (0., 1.5e-6, 15.15e-6, 16.65e-6)
NAMES = ("left_lead", "patterned_region", "right_lead")
EDGES = ((0, 1), (1, 2), (0, 2), (0, 3), (1, 3), (2, 3))
ENERGY_RTOL, VOLUME_RTOL = 1e-9, 1e-10
MOMENT_POWERS = tuple((a, b, c) for a in range(5) for b in range(5-a) for c in range(5-a-b))
REFERENCE_MOMENTS = np.array([factorial(a)*factorial(b)*factorial(c)/factorial(a+b+c+3)
                              for a, b, c in MOMENT_POWERS])


def require(condition, message):
    if not condition:
        raise ValueError(message)


def relative_close(actual, expected, label, tolerance=ENERGY_RTOL):
    """Compare against an explicit positive scale, with no denominator floor."""
    actual, expected = np.asarray(actual), np.asarray(expected)
    require(np.isfinite(actual).all() and np.isfinite(expected).all(), f"Nonfinite {label}.")
    scale = float(np.max(np.abs(expected)))
    difference = float(np.max(np.abs(actual - expected)))
    require(difference == 0 if scale == 0 else difference <= tolerance * scale,
            f"Independent {label} differs.")
    return difference / scale if scale else 0.


class Budget:
    def __init__(self, wall_seconds=120., maximum_bytes=2 * 2**30, reserve_bytes=512 * 2**20):
        self.started = monotonic()
        self.wall_seconds, self.maximum_bytes, self.reserve_bytes = wall_seconds, maximum_bytes, reserve_bytes
        self.peak_rss = 0

    def check(self):
        rss = psutil.Process().memory_info().rss
        self.peak_rss = max(self.peak_rss, rss)
        require(monotonic() - self.started <= self.wall_seconds, "Verifier wall budget exceeded.")
        require(rss <= self.maximum_bytes, "Verifier resident-memory budget exceeded.")
        require(psutil.virtual_memory().available >= self.reserve_bytes, "Verifier physical-memory reserve breached.")


def duffy_rule():
    """64 positive weights on the unit tetrahedron, exact through degree four."""
    nodes, weights = np.polynomial.legendre.leggauss(4)
    nodes, weights = (nodes + 1) / 2, weights / 2
    r, s, t = np.meshgrid(nodes, nodes, nodes, indexing="ij")
    wr, ws, wt = np.meshgrid(weights, weights, weights, indexing="ij")
    bary = np.column_stack((((1-r)*(1-s)*(1-t)).ravel(), r.ravel(),
                            ((1-r)*s).ravel(), ((1-r)*(1-s)*t).ravel()))
    return bary, (wr * ws * wt * (1-r)**2 * (1-s)).ravel()


QUADRATURE_BARY, QUADRATURE_WEIGHTS = duffy_rule()


def clip_one_plane(vertices, original_x, plane, keep_below=True):
    """Clip one barycentric tetrahedron with independent 1/2/3-inside cases.

    The two-inside and three-inside polytopes are triangular prisms, each
    partitioned by a fixed three-tetrahedron stencil. No hull or perturbation
    is used. Zero-volume pieces from exact face/edge/vertex cuts are discarded.
    """
    vertices, original_x = np.asarray(vertices, float), np.asarray(original_x, float)
    distances = vertices @ original_x - plane
    if not keep_below:
        distances = -distances
    inside, outside = np.flatnonzero(distances <= 0), np.flatnonzero(distances > 0)
    if len(inside) == 4:
        return [vertices]
    if not len(inside):
        return []

    def intersection(a, b):
        ratio = distances[a] / (distances[a] - distances[b])
        return vertices[a] + ratio * (vertices[b] - vertices[a])

    if len(inside) == 1:
        a = inside[0]
        pieces = [[vertices[a], *(intersection(a, b) for b in outside)]]
    elif len(inside) == 3:
        a, b, c = inside
        d = outside[0]
        aa, bb, cc = (intersection(i, d) for i in inside)
        pieces = [[vertices[a], vertices[b], vertices[c], cc],
                  [vertices[a], vertices[b], bb, cc], [vertices[a], aa, bb, cc]]
    else:
        a, b = inside
        c, d = outside
        ac, ad, bc, bd = intersection(a, c), intersection(a, d), intersection(b, c), intersection(b, d)
        pieces = [[vertices[a], ac, ad, bd], [vertices[a], ac, bc, bd],
                  [vertices[a], vertices[b], bc, bd]]
    return [piece for row in pieces if abs(np.linalg.det((piece := np.asarray(row))[1:, 1:]
                                                       - piece[0, 1:])) > 0]


def clip_slab(original_x, lower, upper):
    require(np.isfinite(original_x).all() and np.isfinite([lower, upper]).all() and lower < upper,
            "Invalid clipping interval or coordinates.")
    first = clip_one_plane(np.eye(4), original_x, upper, True)
    return [piece for tetra in first for piece in clip_one_plane(tetra, original_x, lower, False)]


def p2_values(bary):
    return np.column_stack([*(bary[:, i] * (2*bary[:, i]-1) for i in range(4)),
                            *(4*bary[:, i]*bary[:, j] for i, j in EDGES)])


def p2_derivatives(bary):
    """Derivatives with respect to the four independent barycentric symbols."""
    derivative = np.zeros((len(bary), 10, 4))
    for i in range(4):
        derivative[:, i, i] = 4*bary[:, i]-1
    for node, (i, j) in enumerate(EDGES, start=4):
        derivative[:, node, i], derivative[:, node, j] = 4*bary[:, j], 4*bary[:, i]
    return derivative


def monomial_moments(bary, weights):
    values = np.prod(bary[:, None, 1:] ** np.asarray(MOMENT_POWERS)[None, :, :], axis=2)
    return weights @ values


def material_tensor(material):
    c11, c12, c44, density = (float(material[key]) for key in
                              ("C11_Pa", "C12_Pa", "C44_Pa", "density_kg_m3"))
    require(np.isfinite([c11, c12, c44, density]).all() and density > 0
            and c44 > 0 and c11-c12 > 0 and c11+2*c12 > 0, "Invalid cubic material.")
    rotation = np.asarray(material["crystal_rotation"], float)
    require(rotation.shape == (3, 3) and np.isfinite(rotation).all()
            and np.allclose(rotation @ rotation.T, np.eye(3), atol=1e-12, rtol=0)
            and abs(np.linalg.det(rotation)-1) <= 1e-12, "Invalid crystal-to-laboratory rotation.")
    crystal = np.zeros((3, 3, 3, 3))
    for i in range(3):
        crystal[i, i, i, i] = c11
        for j in range(3):
            if i != j:
                crystal[i, i, j, j] = c12
                crystal[i, j, i, j] = crystal[i, j, j, i] = c44
    return density, np.einsum("ia,jb,kc,ld,abcd->ijkl", rotation, rotation, rotation, rotation, crystal)


def validate_layout(data, column=0):
    points, tets = np.asarray(data["points_m"]), np.asarray(data["tetrahedra"])
    coordinates, component = np.asarray(data["dof_coordinates_m"]), np.asarray(data["dof_component"])
    elements, displacement = np.asarray(data["element_dofs"]), np.asarray(data["displacements_per_sqrt_w"])
    require(np.asarray(data["element_order"]).item() == 2 and points.ndim == 2 and points.shape[1] == 3
            and tets.ndim == 2 and tets.shape[1] == 4 and len(tets) > 0
            and np.issubdtype(tets.dtype, np.integer) and tets.min() >= 0 and tets.max() < len(points),
            "Invalid full-P2 mesh topology.")
    require(coordinates.ndim == 2 and coordinates.shape[1] == 3 and component.shape == (len(coordinates),)
            and elements.shape == (30, len(tets)) and np.issubdtype(elements.dtype, np.integer)
            and elements.min() >= 0 and elements.max() < len(coordinates)
            and displacement.ndim == 2 and displacement.shape[0] == len(coordinates)
            and 0 <= column < displacement.shape[1], "Invalid P2 displacement/component layout.")
    require(np.isfinite(points).all() and np.isfinite(coordinates).all() and np.isfinite(displacement).all(),
            "Nonfinite full-P2 field.")
    scale = float(np.max(np.ptp(points, axis=0)))
    require(scale > 0, "Degenerate mesh extent.")
    pairs = ((0, 0), (1, 1), (2, 2), (3, 3), *EDGES)
    for node, (a, b) in enumerate(pairs):
        ids = elements[3*node:3*node+3]
        expected = (points[tets[:, a]] + points[tets[:, b]]) / 2
        require(np.array_equal(component[ids], np.broadcast_to(np.arange(3)[:, None], ids.shape))
                and np.all(np.abs(coordinates[ids] - expected[None, :, :]) <= 64*np.finfo(float).eps*scale),
                "P2 component coordinates or edge order differs.")
    require(np.array_equal(np.unique(elements), np.arange(len(coordinates))), "P2 layout has unreferenced DOFs.")
    return points, tets, elements, displacement[:, column]


def integrate_regions(data, material, frequency_hz, cuts, *, column=0, budget=None, batch_size=512):
    """Integrate each native mesh independently; no cross-mesh interpolation."""
    points, tets, elements, vector = validate_layout(data, column)
    density, tensor = material_tensor(material)
    cuts = np.asarray(cuts, float)
    require(cuts.ndim == 1 and len(cuts) >= 2 and np.isfinite(cuts).all() and np.all(np.diff(cuts) > 0)
            and np.isfinite(frequency_hz) and frequency_hz > 0, "Invalid energy frequency or region cuts.")
    require(points[:, 0].min() >= cuts[0] and points[:, 0].max() <= cuts[-1], "Regions do not cover the mesh.")
    accumulated = np.zeros((len(cuts)-1, 3))
    cut_count, maximum_moment_error = 0, 0.
    omega = 2*np.pi*frequency_hz
    reference_values, reference_derivatives = p2_values(QUADRATURE_BARY), p2_derivatives(QUADRATURE_BARY)
    for start in range(0, len(tets), batch_size):
        if budget:
            budget.check()
        ids = np.arange(start, min(start+batch_size, len(tets)))
        vertices = points[tets[ids]]
        jacobian = (vertices[:, 1:] - vertices[:, :1]).transpose(0, 2, 1)
        determinant = abs(np.linalg.det(jacobian))
        require(np.isfinite(determinant).all() and np.all(determinant > 0), "Degenerate energy tetrahedron.")
        inverse = np.linalg.inv(jacobian)
        grad_bary = np.concatenate((-inverse.sum(axis=1)[:, None, :], inverse), axis=1)
        nodal = vector[elements[:, ids].T].reshape(-1, 10, 3)
        lo, hi = vertices[:, :, 0].min(axis=1), vertices[:, :, 0].max(axis=1)
        cut_ids = np.flatnonzero(np.any((lo[:, None] < cuts[None, 1:-1])
                                       & (hi[:, None] > cuts[None, 1:-1]), axis=1))
        moment_sums = {int(index): np.zeros(len(MOMENT_POWERS)) for index in cut_ids}
        cut_count += len(cut_ids)
        for region, (lower, upper) in enumerate(zip(cuts[:-1], cuts[1:])):
            whole = np.flatnonzero((lo >= lower) & (hi <= upper))
            if len(whole):
                field = np.einsum("qn,bnc->bqc", reference_values, nodal[whole])
                gradient = np.einsum("qnl,blj,bnc->bqcj", reference_derivatives, grad_bary[whole], nodal[whole], optimize=True)
                strain = (gradient + gradient.transpose(0, 1, 3, 2)) / 2
                kinetic = density*omega**2/4 * np.sum(abs(field)**2, axis=2)
                elastic = np.einsum("bqij,ijkl,bqkl->bq", strain.conj(), tensor, strain).real / 4
                weights = determinant[whole, None] * QUADRATURE_WEIGHTS[None, :]
                accumulated[region] += [float(determinant[whole].sum()/6),
                                        float(np.sum(weights*kinetic)), float(np.sum(weights*elastic))]
            partial = np.flatnonzero((lo < upper) & (hi > lower) & ~((lo >= lower) & (hi <= upper)))
            for index in partial:
                for piece in clip_slab(vertices[index, :, 0], lower, upper):
                    local_det = abs(np.linalg.det(piece[1:, 1:] - piece[0, 1:]))
                    bary = QUADRATURE_BARY @ piece
                    if int(index) in moment_sums:
                        moment_sums[int(index)] += monomial_moments(bary, local_det*QUADRATURE_WEIGHTS)
                    field = p2_values(bary) @ nodal[index]
                    gradient = np.einsum("qnl,lj,nc->qcj", p2_derivatives(bary), grad_bary[index], nodal[index])
                    strain = (gradient + gradient.transpose(0, 2, 1)) / 2
                    kinetic = density*omega**2/4 * np.sum(abs(field)**2, axis=1)
                    elastic = np.einsum("qij,ijkl,qkl->q", strain.conj(), tensor, strain).real / 4
                    weights = determinant[index]*local_det*QUADRATURE_WEIGHTS
                    accumulated[region] += [determinant[index]*local_det/6,
                                            np.dot(weights, kinetic), np.dot(weights, elastic)]
        for values in moment_sums.values():
            error = float(max(abs(values-REFERENCE_MOMENTS)/REFERENCE_MOMENTS))
            maximum_moment_error = max(maximum_moment_error, error)
            require(error <= VOLUME_RTOL, "Independent clipped volume/degree-four moments fail parent closure.")
    require(np.isfinite(accumulated).all() and np.all(accumulated >= 0), "Nonpositive or nonfinite integrated energy.")
    totals = accumulated.sum(axis=0)
    require(totals[0] > 0 and totals[1]+totals[2] > 0, "Zero mesh volume or total stored energy.")
    return {"regions": accumulated, "totals": totals,
            "fractions": (accumulated[:, 1]+accumulated[:, 2])/(totals[1]+totals[2]),
            "cut_tetrahedra": int(cut_count), "maximum_cut_degree_four_moment_relative_error": maximum_moment_error}


def check_tetra10(path, data, column=0):
    """Read the actual VTU independently and require unscaled full-P2 content."""
    import meshio
    _, tets, elements, vector = validate_layout(data, column)
    coordinates, component = np.asarray(data["dof_coordinates_m"]), np.asarray(data["dof_component"])
    component_ids = [np.flatnonzero(component == i) for i in range(3)]
    require(all(len(ids) == len(component_ids[0]) for ids in component_ids), "VTU component counts differ.")
    for ids in component_ids[1:]:
        require(np.array_equal(coordinates[ids], coordinates[component_ids[0]]), "VTU component-node order differs.")
    scalar_map = np.full(len(coordinates), -1, dtype=np.int64)
    scalar_map[component_ids[0]] = np.arange(len(component_ids[0]))
    expected_cells = scalar_map[elements[::3]].T
    expected_field = np.column_stack([vector[ids] for ids in component_ids])
    viewed = meshio.read(path)
    require(len(viewed.cells) == 1 and viewed.cells[0].type == "tetra10"
            and np.array_equal(viewed.points, coordinates[component_ids[0]])
            and np.array_equal(viewed.cells[0].data, expected_cells), "VTU physical P2 points or tetra10 connectivity differs.")
    for key, values in (("displacement_real_m_per_sqrt_W", expected_field.real),
                        ("displacement_imag_m_per_sqrt_W", expected_field.imag)):
        require(key in viewed.point_data and np.array_equal(viewed.point_data[key], values),
                "VTU complex displacement was omitted, normalized, or phase shifted.")
    return {"points": len(viewed.points), "tetra10_cells": len(tets), "exact_complex_field_readback": True}


def file_sha(path, budget=None):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(4*2**20), b""):
            digest.update(block)
            if budget:
                budget.check()
    return digest.hexdigest()


def resolve_path(workspace, value):
    workspace, value = Path(workspace).resolve(), str(value).replace("\\", "/")
    candidate = Path(value)
    if not candidate.is_absolute() or not candidate.is_relative_to(workspace):
        if "/results/" in value:
            candidate = workspace / "results" / value.split("/results/", 1)[1]
        else:
            candidate = workspace / value
    candidate = candidate.resolve(strict=True)
    require(candidate.is_file() and candidate.is_relative_to(workspace), "An input path escapes the workspace.")
    return candidate


def checked_artifact(workspace, link, budget=None):
    path = resolve_path(workspace, link["path"])
    require(path.stat().st_size == link["bytes"] and file_sha(path, budget) == link["sha256"],
            f"Artifact bytes or SHA-256 differs: {path.name}.")
    return path


def read_json(path):
    require(Path(path).stat().st_size <= 4*2**20, "A bounded JSON receipt exceeds 4 MiB.")
    return json.loads(Path(path).read_bytes())


def matrix_energies(paths, data, frequency_hz, budget=None, column=0):
    """Load only one real saved matrix at a time, without assembly."""
    from scipy.sparse import load_npz
    vector = np.asarray(data["displacements_per_sqrt_w"])[:, column]
    result = {}
    for key, multiplier in (("K", .25), ("M", (2*np.pi*frequency_hz)**2/4)):
        if budget:
            budget.check()
        matrix = load_npz(paths[key])
        require(matrix.shape == (len(vector), len(vector)) and matrix.dtype == np.float64
                and np.isfinite(matrix.data).all(), "Saved real K/M matrix layout differs.")
        value = np.vdot(vector, matrix @ vector)
        del matrix
        require(value.real >= 0 and abs(value.imag) <= ENERGY_RTOL*abs(value.real), "Invalid saved-matrix quadratic energy.")
        result[key] = float(value.real*multiplier)
        if budget:
            budget.check()
    return result


def check_region_publication(expected, independent, cuts):
    rows = expected["regions"]
    require(len(rows) == 3, "All three published regions are required.")
    differences = {}
    for index, row in enumerate(rows):
        require(row["name"] == NAMES[index] and row["x_interval_m"] == list(cuts[index:index+2]),
                "The published physical region intervals differ.")
        actual = independent["regions"][index]
        differences[row["name"]] = {
            "volume_relative_difference": relative_close(row["volume_m3"], actual[0], "regional volume", VOLUME_RTOL),
            "kinetic_relative_difference": relative_close(row["kinetic_energy_J_per_W"], actual[1], "regional kinetic energy"),
            "elastic_relative_difference": relative_close(row["elastic_energy_J_per_W"], actual[2], "regional elastic energy")}
        relative_close(row["total_energy_J_per_W"], actual[1]+actual[2], "regional total energy")
        require(abs(row["total_energy_fraction"] - independent["fractions"][index]) <= ENERGY_RTOL,
                "Independent regional energy fraction differs.")
    totals = expected["totals"]
    for name, actual in zip(("volume_m3", "kinetic_energy_J_per_W", "elastic_energy_J_per_W"), independent["totals"]):
        relative_close(totals[name], actual, "global "+name, VOLUME_RTOL if name == "volume_m3" else ENERGY_RTOL)
    combined = independent["totals"][1:].sum()
    relative_close(totals["total_energy_J_per_W"], combined, "global total energy")
    relative_close(totals["energy_per_incident_power_s"], combined, "power-normalized stored energy")
    return differences


def check_integration_proof(proof):
    require(proof["quadrature_degree"] == 4 and proof["original_parent_P2_evaluated"] is True
            and proof["centroid_region_assignment"] is False and proof["geometry_perturbation_applied"] is False
            and proof["degree_four_monomials_per_cut_tetrahedron"] == len(MOMENT_POWERS),
            "Unsupported producer P2 quadrature or regional geometry proof.")
    for key in ("maximum_cut_moment_relative_closure", "analytic_rectangular_lead_relative_volume_errors",
                "kinetic_elastic_absolute_partition_difference_over_total_energy"):
        values = np.asarray(proof[key])
        require(np.isfinite(values).all() and np.all(values >= 0) and np.all(values <= VOLUME_RTOL),
                "The producer geometric/energy partition proof did not pass its declared gate.")


def validate_header(result, producer_sha256):
    require(result.get("status") == "saved_field_energy_compared_finite_increment"
            and result.get("schema_version") == 1 and all(result.get(k) == v for k, v in AUTHOR.items()),
            "Unsupported producer proof/status/schema.")
    require(result["producer"]["sha256"] == producer_sha256 and len(producer_sha256) == 64
            and all(char in "0123456789abcdef" for char in producer_sha256),
            "Producer SHA-256 does not match the externally supplied executed source.")
    scope = result["scope"]
    require(scope["frequency_hz"] == FREQUENCY and scope["incident_root"] == ROOT
            and scope["incident_column"] == COLUMN and scope["region_cuts_m"] == list(CUTS)
            and scope["native_incident_gauge_preserved"] is True,
            "Unsupported root, phase, frequency, or physical region scope.")
    require(all(scope[key] is False for key in ("new_solve_performed", "new_assembly_performed", "field_fitting_performed",
                "cross_mesh_interpolation_performed", "spatial_convergence_established", "frequency_convergence_established")),
            "Unsupported solve, interpolation, phase fit, or convergence claim.")
    require(result["predeclared_target"] == {"max_absolute_fraction_increment": .02, "comparison": "strict_less_than"},
            "The predeclared finite fraction target changed.")


def verify(workspace, receipt_path, producer_sha256, *, budget=None):
    budget = budget or Budget()
    workspace, receipt_path = Path(workspace).resolve(), Path(receipt_path).resolve()
    budget.check()
    result = read_json(receipt_path)
    validate_header(result, producer_sha256)
    checked_artifact(workspace, result["producer"], budget)
    inputs = result["inputs"]
    require(inputs["accepted_comparison"]["sha256"] == COMPARISON_SHA, "Accepted comparison identity changed.")
    comparison = read_json(checked_artifact(workspace, inputs["accepted_comparison"], budget))
    require(comparison["status"] == "compared_finite_inplane_increment" and comparison["frequency_hz"] == FREQUENCY
            and comparison["shared_basis"]["status"] == "identical_saved_basis", "The accepted comparison proof differs.")
    checked_artifact(workspace, result["material_source"], budget)
    measured, checks, matrix_inputs = {}, {}, {}
    for name, summary_role, replay_role in (("h100", "h100_8875_summary", "point_replay"),
                                             ("h80", "h80_8875_summary", "accepted_point")):
        require(inputs[name+"_replay"] == comparison["inputs"][replay_role]
                and inputs[name+"_replay"]["sha256"] == REPLAY_SHAS[name]
                and inputs[name+"_summary"] == comparison["inputs"][summary_role],
                "The accepted point/summary input mapping differs.")
        replay = read_json(checked_artifact(workspace, inputs[name+"_replay"], budget))
        summary_path = checked_artifact(workspace, inputs[name+"_summary"], budget)
        summary = read_json(summary_path)
        require(replay["status"] == "point_verified_within_saved_discretization"
                and replay["full_p2_field_identity_passed"] is True and replay["original_equations_passed"] is True
                and summary["frequency_hz"] == FREQUENCY and summary["incident_side"] == "left"
                and summary["incident_mode_indices"][COLUMN] == ROOT and summary["port_dofs"] == 675
                and summary["configuration"]["mesh"]["element_order"] == 2
                and summary["configuration"]["mesh"]["thickness_layers"] == 4
                and summary["checkpointed_scattering"]["phase_adjustment_applied"] is False
                and summary["checkpointed_scattering"]["reference_planes_m"] == [CUTS[0], CUTS[-1]]
                and replay["public_summary_sha256"] == inputs[name+"_summary"]["sha256"]
                and result["material_source"]["sha256"] == summary["implementation"]["source_sha256"]["src/phononic/material_AP_pnc.py"],
                "Accepted field proof, unit-power drive, or P2 discretization differs.")
        field_link = inputs[name+"_fields"]
        require(field_link["sha256"] == summary["output_sha256"]["selected_fields_AP_pnc.npz"],
                "The selected field is not bound to its accepted public summary.")
        field_path = checked_artifact(workspace, field_link, budget)
        require(field_path.parent == summary_path.parent, "Selected field and accepted summary directories differ.")
        keys = ("points_m", "tetrahedra", "dof_coordinates_m", "dof_component", "element_dofs", "element_order",
                "displacements_per_sqrt_w", "incident_mode_indices", "frequencies_hz", "incident_powers_w")
        with np.load(field_path, allow_pickle=False) as archive:
            data = {key: archive[key] for key in keys}
        require(np.array_equal(data["incident_mode_indices"], [1302, 1346, 1292, 1286, 1282])
                and np.array_equal(data["frequencies_hz"], np.full(5, FREQUENCY))
                and data["displacements_per_sqrt_w"].shape == (summary["ndof"], 5)
                and data["displacements_per_sqrt_w"].dtype == np.complex128
                and data["incident_powers_w"].shape == (5,) and np.isfinite(data["incident_powers_w"]).all()
                and abs(float(data["incident_powers_w"][COLUMN])-1.) <= 1e-7,
                "Archive frequency, incident root, or unit-power normalization differs.")
        # Retain exactly one field column after its accepted layout is checked.
        data["displacements_per_sqrt_w"] = data["displacements_per_sqrt_w"][:, COLUMN:COLUMN+1].copy()
        published = result["meshes"][name]
        require(published["frequency_hz"] == FREQUENCY and published["incident_root"] == ROOT
                and published.get("incident_column", published.get("column")) == COLUMN
                and published["incident_power_w"] == float(data["incident_powers_w"][COLUMN]),
                "Published field drive or incident normalization differs.")
        check_integration_proof(published["integration_checks"])
        independent = integrate_regions(data, summary["configuration"]["material"], FREQUENCY, CUTS, budget=budget)
        checks[name] = {"regional_differences": check_region_publication(published, independent, CUTS),
                        "independent_cut_tetrahedra": independent["cut_tetrahedra"],
                        "independent_maximum_degree_four_moment_relative_error": independent["maximum_cut_degree_four_moment_relative_error"]}
        # Analytic rectangular lead checks protect geometry independently of energy closure.
        for index in (0, 2):
            relative_close(independent["regions"][index, 0], 1.5e-6*1e-6*200e-9,
                           "analytic rectangular lead volume", VOLUME_RTOL)
        whole = integrate_regions(data, summary["configuration"]["material"], FREQUENCY, (CUTS[0], CUTS[-1]), budget=budget)
        for index, label in enumerate(("volume", "kinetic energy", "elastic energy")):
            relative_close(independent["totals"][index], whole["totals"][index],
                           "independent partition "+label, VOLUME_RTOL if index == 0 else ENERGY_RTOL)
        stage = read_json(checked_artifact(workspace, replay["phase_receipts"]["stage"], budget))
        paths = {key: checked_artifact(workspace, stage["array_files"][key], budget) for key in ("K", "M")}
        matrices = matrix_energies(paths, data, FREQUENCY, budget)
        relative_close(matrices["M"], independent["totals"][1], "global saved-M kinetic energy")
        relative_close(matrices["K"], independent["totals"][2], "global saved-K elastic energy")
        matrix_inputs[name] = {key: stage["array_files"][key] for key in ("K", "M")}
        checks[name]["saved_matrix_energies_J_per_W"] = {"kinetic": matrices["M"], "elastic": matrices["K"]}
        export = published["selected_field_export"]
        require(export["displacement_units"] == "m/sqrt(W)" and export["native_complex_gauge_preserved"] is True
                and export["visualization_amplitude_rescaled"] is False
                and export["edge_order"] == [list(edge) for edge in EDGES],
                "Unsupported field units, P2 mapping, normalization or phase metadata.")
        vtu_path = checked_artifact(workspace, published["selected_field_vtu"], budget)
        checks[name]["tetra10_readback"] = check_tetra10(vtu_path, data)
        measured[name] = {"fractions": independent["fractions"].tolist(), "total_energy_J_per_W": float(independent["totals"][1:].sum())}
        del data
        budget.check()
    increments = np.asarray(measured["h80"]["fractions"]) - measured["h100"]["fractions"]
    maximum = float(max(abs(increments)))
    published = result["comparison"]
    require(np.allclose(published["regional_fraction_increments"], increments, rtol=0, atol=ENERGY_RTOL)
            and abs(published["max_absolute_fraction_increment"]-maximum) <= ENERGY_RTOL
            and published["finite_fraction_target_passed"] is (maximum < .02),
            "Independent finite localization-fraction comparison differs.")
    relative_change = (measured["h80"]["total_energy_J_per_W"]-measured["h100"]["total_energy_J_per_W"])/measured["h100"]["total_energy_J_per_W"]
    require(np.isfinite(published["relative_total_energy_change"])
            and abs(published["relative_total_energy_change"]-relative_change) <= ENERGY_RTOL,
            "Independent signed relative total-energy diagnostic differs.")
    budget.check()
    return {**AUTHOR, "schema_version": 1, "status": "saved_field_energy_independently_verified",
            "receipt": {"path": str(receipt_path), "bytes": receipt_path.stat().st_size, "sha256": file_sha(receipt_path)},
            "executed_producer_sha256": producer_sha256, "verifier_sha256": file_sha(Path(__file__)),
            "material_source": result["material_source"],
            "numerical_environment": {"python": platform.python_version(), "platform": platform.platform(),
                "packages": {name: version(name) for name in ("numpy", "scipy", "meshio", "psutil")},
                "thread_environment": {key: os.environ.get(key) for key in
                    ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "BLIS_NUM_THREADS", "NUMEXPR_NUM_THREADS")}},
            "method": "Explicit P2 with 64-point Gauss-Duffy quadrature; case-based two-plane tetrahedron splitting; sequential saved real K/M quadratic forms; exact meshio tetra10 readback.",
            "regional_energy_relative_tolerance": ENERGY_RTOL, "volume_relative_tolerance": VOLUME_RTOL,
            "checks": checks, "matrix_inputs": matrix_inputs, "measurements": measured,
            "regional_fraction_increments": increments.tolist(), "max_absolute_fraction_increment": maximum,
            "finite_fraction_target_passed": maximum < .02, "elapsed_seconds": monotonic()-budget.started,
            "peak_sampled_rss_bytes": budget.peak_rss, "body_solve_performed": False, "body_assembly_performed": False,
            "spatial_convergence_established": False, "frequency_convergence_established": False,
            "conversion_error_cause_established": False}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--workspace", type=Path, default=Path(__file__).resolve().parents[1])
    parser.add_argument("--receipt", type=Path, required=True)
    parser.add_argument("--producer-sha256", required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    require(not args.output.exists(), "Verifier output already exists; scientific evidence will not be overwritten.")
    result = verify(args.workspace, args.receipt, args.producer_sha256)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("x", encoding="utf-8") as stream:
        json.dump(result, stream, indent=2, allow_nan=False)
        stream.write("\n")
    print(json.dumps({"status": result["status"], "output": str(args.output), "elapsed_seconds": result["elapsed_seconds"]}))


if __name__ == "__main__":
    main()
