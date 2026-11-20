"""One saved root-1302 P2 field-energy control. Author: Athul Prem | _AP_pnc.

No assembly, SAFE calculation, factorization, solve, field fit or interpolation.
The native complex incident gauge and displacement per sqrt(W) are preserved.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
from html import escape
import json
import os
from pathlib import Path
import platform
import sys
from time import monotonic

THREAD_KEYS = ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "BLIS_NUM_THREADS", "NUMEXPR_NUM_THREADS")
if __name__ == "__main__":
    os.environ.update(dict.fromkeys(THREAD_KEYS, "1"))

import meshio
import numpy as np
import psutil
import scipy
import skfem
from scipy.spatial import ConvexHull, QhullError
from skfem import ElementTetP2
from skfem.quadrature import get_quadrature_tet

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from phononic.material_AP_pnc import CubicMaterial
from phononic.report_style_AP_pnc import available_link, report_page, scroll_table

AUTHOR = {"author": "Athul Prem", "file_tag": "_AP_pnc"}
FREQUENCY, INCIDENT_ROOT, INCIDENT_COLUMN = 887_500_000, 1302, 0
CUTS = np.array([0., 1.5e-6, 15.15e-6, 16.65e-6])
REGION_NAMES = ("left_lead", "patterned_region", "right_lead")
EDGES = ((0, 1), (1, 2), (0, 2), (0, 3), (1, 3), (2, 3))
COMPARISON_PATH = "results/runs/device_frequency_inplane_8875_AP_pnc/comparison/frequency_inplane_comparison_AP_pnc.json"
COMPARISON_SHA = "107d1c62817349d0a6900650d10dd3826ee492ca4bc36abc8195ee3c3e7911c7"
REPLAY_SHAS = {"h100": "ac9d1ccd9d5346102d97d03b2dac7e47fc3cb319f99d67d8a1d33bba087fc2b9",
               "h80": "17bcad69b7d567331489d32a798f6755df292272473df78e9338ba259683fd34"}
FIELD_SHAS = {"h100": "b4b4e7363c488d83515df75a5eba1ffb053f0c01940829d8d270e3c76c7b22c7",
              "h80": "4006732336aac65989207fbf3c0572ca931d6919bbbd80827725137abdf854de"}
LIMIT = .02
RECORD_NAME = "field_energy_AP_pnc.json"
RECEIPT_NAME = "independent_verification_AP_pnc.json"
MOMENT_POWERS = tuple((a, b, c) for a in range(5) for b in range(5-a) for c in range(5-a-b))
LIMITATIONS = (
    "One finite global h100/h80 in-plane increment, including lead interiors; no spatial or frequency convergence is established.",
    "Regional energy fractions describe the total driven field, not the location or cause of weak-conversion error.",
    "No kinetic/elastic equipartition is imposed. Energy per incident power is not a measured delay, lifetime or Q.",
    "Previous failed scattering criteria, numerical-floor limits, fixed-property material assumptions and unvalidated experimental adequacy remain unresolved.",
)


def require(condition, message):
    if not condition:
        raise ValueError(message)


def sha(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(2**20), b""):
            h.update(block)
    return h.hexdigest()


def artifact(path, workspace=ROOT):
    path, workspace = Path(path).resolve(), Path(workspace).resolve()
    return {"path": path.relative_to(workspace).as_posix() if path.is_relative_to(workspace) else str(path),
            "bytes": path.stat().st_size, "sha256": sha(path)}


def checked_path(workspace, link, *, expected_sha=None, max_bytes=4*2**20):
    root = Path(workspace).resolve()
    path = (root / link["path"]).resolve(strict=True)
    require(path.is_relative_to(root) and path.is_file(), "A scientific input escapes the workspace.")
    require(path.stat().st_size == link["bytes"] and path.stat().st_size <= max_bytes,
            "A scientific input size changed or exceeds the bounded read limit.")
    require(link["sha256"] == (expected_sha or link["sha256"]) and sha(path) == link["sha256"],
            "A scientific input SHA-256 differs from accepted evidence.")
    return path


def read_json(path):
    require(Path(path).stat().st_size <= 4*2**20, "JSON input exceeds 4 MiB.")
    return json.loads(Path(path).read_bytes())


def quadrature():
    q, w = get_quadrature_tet(4)
    return np.column_stack((1-q.sum(axis=0), q.T)), w


def p2_shape(barycentric):
    """Return scalar P2 shapes and parent-reference derivatives, skfem order."""
    xyz = np.asarray(barycentric)[:, 1:].T
    values = [ElementTetP2().lbasis(xyz, i) for i in range(10)]
    return np.stack([v[0] for v in values], axis=1), np.stack([v[1].T for v in values], axis=1)


def moments(barycentric, weights):
    xyz = np.asarray(barycentric)[:, 1:]
    return np.array([np.dot(weights, np.prod(xyz ** power, axis=1)) for power in MOMENT_POWERS])


def clip_parent_slab(vertices_x, lower, upper):
    """Partition the exact affine tetra/slab intersection in parent coordinates.

    All intersection vertices lie on original edges. ConvexHull triangulates
    boundary facets; coning them to an interior point partitions the convex
    volume. No QJ/joggling or displaced classification plane is permitted.
    Lower-dimensional intersections have exactly zero volume and are omitted.
    """
    x = np.asarray(vertices_x, dtype=float)
    require(x.shape == (4,) and np.isfinite(x).all() and np.isfinite([lower, upper]).all() and lower < upper,
            "A tetrahedron slab requires finite coordinates and increasing bounds.")
    if x.max() <= lower or x.min() >= upper:
        return np.empty((0, 4, 4))
    identity = np.eye(4)
    if x.min() >= lower and x.max() <= upper:
        return identity[None, :, :]
    vertices = [identity[i] for i in range(4) if lower <= x[i] <= upper]
    for i, j in EDGES:
        for cut in (lower, upper):
            if min(x[i], x[j]) < cut < max(x[i], x[j]):
                fraction = (cut-x[i]) / (x[j]-x[i])
                vertices.append((1-fraction)*identity[i] + fraction*identity[j])
    bary = np.unique(np.asarray(vertices), axis=0)
    require(len(bary) >= 4, "A positive-volume clipped tetrahedron lacks four vertices.")
    if len(bary) == 4:
        return bary[None, :, :]
    # An invertible change of coordinates improves a thin, but real, sliver's
    # hull conditioning without moving any vertex or physical cutting plane.
    centered = bary[:, 1:]-bary[:, 1:].mean(axis=0)
    _, singular, axes = np.linalg.svd(centered, full_matrices=False)
    require(singular[-1] > 0, "A positive-volume clipped polytope is numerically rank deficient.")
    conditioned = (centered@axes.T)/singular
    try:
        hull = ConvexHull(conditioned)
    except QhullError as error:
        raise ValueError("Clipped tetrahedron has unresolved geometric degeneracy; no perturbation was applied.") from error
    center = bary.mean(axis=0)
    subtets = np.stack([np.vstack((center, bary[face])) for face in hull.simplices])
    determinants = np.abs(np.linalg.det(subtets[:, 1:, 1:] - subtets[:, :1, 1:]))
    # Exact zero-volume cones may occur on redundant coplanar facets. A real
    # sliver is retained, rather than removed using an arbitrary size cutoff.
    return subtets[determinants > 0]


def mapped_rule(subtets, barycentric, weights):
    if not len(subtets):
        return np.empty((0, 4)), np.empty(0)
    determinant = np.abs(np.linalg.det(subtets[:, 1:, 1:] - subtets[:, :1, 1:]))
    return np.einsum("qv,svb->sqb", barycentric, subtets).reshape(-1, 4), (determinant[:, None]*weights).ravel()


def validate_field(data):
    """Check complete P2 component, vertex and explicit six-edge ordering."""
    p, t = data["points_m"], data["tetrahedra"]
    coordinates, component, edofs, u = (data[k] for k in ("dof_coordinates_m", "dof_component", "element_dofs", "u"))
    require(p.ndim == 2 and p.shape[1] == 3 and t.ndim == 2 and t.shape[1] == 4 and len(t), "Invalid affine tetra mesh layout.")
    require(np.issubdtype(t.dtype, np.integer) and t.min() >= 0 and t.max() < len(p), "Invalid tetra vertex indices.")
    require(coordinates.shape == (len(u), 3) and component.shape == u.shape and u.ndim == 1
            and np.iscomplexobj(u) and all(np.isfinite(a).all() for a in (p, coordinates, u)), "Invalid finite complex P2 field layout.")
    require(np.issubdtype(edofs.dtype, np.integer) and edofs.shape == (30, len(t))
            and edofs.min() >= 0 and edofs.max() < len(u), "Invalid complete vector-P2 element map.")
    require(set(np.unique(component)) == {0, 1, 2}, "P2 vector components must be x/y/z.")
    local = edofs.T.reshape(-1, 10, 3)
    require(np.all(component[local] == np.arange(3)), "P2 component ordering differs from the explicit scalar-node map.")
    scalar_ids = np.flatnonzero(component == 0)
    require(len(u) == 3*len(scalar_ids) and len(np.unique(edofs)) == len(u), "The P2 map omits field DOFs.")
    scalar_coordinates = coordinates[scalar_ids]
    require(len(np.unique(scalar_coordinates, axis=0)) == len(scalar_ids), "Duplicate scalar P2 coordinates are unresolved.")
    tolerance = max(np.ptp(p, axis=0).max(), np.finfo(float).tiny)*2e-13
    for start in range(0, len(t), 256):
        stop = start+256
        vertices = p[t[start:stop]]
        expected = np.concatenate((vertices, np.stack([(vertices[:, i]+vertices[:, j])/2 for i, j in EDGES], axis=1)), axis=1)
        require(np.allclose(coordinates[local[start:stop]], expected[:, :, None, :], rtol=0, atol=tolerance),
                "P2 coordinates differ from original vertices or explicit edge midpoints.")
    nodal = data["nodal_dofs"]
    require(np.issubdtype(nodal.dtype, np.integer) and nodal.shape == (3, len(p))
            and nodal.min() >= 0 and nodal.max() < len(u)
            and np.all(component[nodal] == np.arange(3)[:, None])
            and np.allclose(coordinates[nodal], p[None, :, :], rtol=0, atol=tolerance), "Original nodal DOF map differs.")
    scalar_map = np.full(len(u), -1, dtype=np.int64)
    scalar_map[scalar_ids] = np.arange(len(scalar_ids))
    connectivity = scalar_map[local[:, :, 0]]
    values = np.empty((len(scalar_ids), 3), dtype=complex)
    for c in range(3):
        values[connectivity.ravel(), c] = u[local[:, :, c].ravel()]
    require(np.isfinite(values).all(), "Full scalar P2 export has missing vector values.")
    return local, scalar_coordinates, connectivity, values


def density_integral(element_u, inverse_jacobian, barycentric, weights, determinant, material, frequency_hz):
    """Return volume, cycle-averaged kinetic and elastic energies (J/W)."""
    shape, derivative = p2_shape(barycentric)
    uq = np.einsum("qn,enc->eqc", shape, element_u)
    gradient = np.einsum("qnk,ekj->eqnj", derivative, inverse_jacobian)
    du = np.einsum("enc,eqnj->eqcj", element_u, gradient)
    strain = (du+du.swapaxes(-1, -2))*.5
    kinetic = material.rho*(2*np.pi*frequency_hz)**2*.25*np.sum(abs(uq)**2, axis=-1)
    elastic_complex = .25*np.einsum("eqij,ijkl,eqkl->eq", strain.conj(), material.stiffness_tensor(), strain)
    require(np.all(abs(elastic_complex.imag) <= 1e-12*abs(elastic_complex.real)),
            "Elastic energy has a non-negligible imaginary part.")
    result = np.column_stack((np.full(len(element_u), weights.sum()), kinetic@weights, elastic_complex.real@weights))*determinant[:, None]
    require(np.isfinite(result).all() and np.all(result >= 0), "Integrated physical energy or volume is negative/nonfinite.")
    return result


def integrate_regions(data, material, frequency_hz, cuts=CUTS, batch_elements=256, guard=None):
    """Integrate parent P2 fields, clipping every crossed tetrahedron exactly."""
    require(isinstance(batch_elements, int) and 1 <= batch_elements <= 512, "Batch size must be between 1 and 512.")
    cuts = np.asarray(cuts, dtype=float)
    require(cuts.shape == (4,) and np.isfinite(cuts).all() and np.all(np.diff(cuts) > 0)
            and np.isfinite(frequency_hz) and frequency_hz > 0, "Invalid physical partition or frequency.")
    p, t = data["points_m"], data["tetrahedra"]
    local, _, _, _ = validate_field(data)
    tolerance = (cuts[-1]-cuts[0])*2e-13
    require(p[:, 0].min() >= cuts[0]-tolerance and p[:, 0].max() <= cuts[-1]+tolerance, "Mesh extends outside region bounds.")
    q, w = quadrature()
    reference_moments = moments(q, w)
    region_totals, direct_total = np.zeros((3, 3)), np.zeros(3)
    maximum_moment_error, cut_count = 0., 0
    for start in range(0, len(t), batch_elements):
        if guard:
            guard()
        vertices = p[t[start:start+batch_elements]]
        jac = (vertices[:, 1:]-vertices[:, :1]).swapaxes(1, 2)
        determinant = abs(np.linalg.det(jac))
        require(np.isfinite(determinant).all() and np.all(determinant > 0), "An original tetrahedron is degenerate.")
        inverse = np.linalg.inv(jac)
        ue = data["u"][local[start:start+batch_elements]]
        exact = density_integral(ue, inverse, q, w, determinant, material, frequency_hz)
        direct_total += exact.sum(axis=0)
        xmin, xmax = vertices[:, :, 0].min(axis=1), vertices[:, :, 0].max(axis=1)
        uncut = np.zeros(len(vertices), dtype=bool)
        for region, (lower, upper) in enumerate(zip(cuts[:-1], cuts[1:])):
            mask = (xmin >= lower) & (xmax <= upper) & ~uncut
            region_totals[region] += exact[mask].sum(axis=0)
            uncut |= mask
        for i in np.flatnonzero(~uncut):
            cut_count += 1
            closure = np.zeros(len(MOMENT_POWERS))
            for region, (lower, upper) in enumerate(zip(cuts[:-1], cuts[1:])):
                rq, rw = mapped_rule(clip_parent_slab(vertices[i, :, 0], lower, upper), q, w)
                if len(rw):
                    closure += moments(rq, rw)
                    region_totals[region] += density_integral(ue[i:i+1], inverse[i:i+1], rq, rw, determinant[i:i+1], material, frequency_hz)[0]
            error = float(np.max(abs(closure-reference_moments)/reference_moments))
            maximum_moment_error = max(maximum_moment_error, error)
            require(error <= 1e-10, "Clipped volume/degree-four polynomial moments fail parent partition closure.")
    difference = abs(region_totals.sum(axis=0)-direct_total)
    require(np.all(difference[direct_total == 0] == 0), "A zero parent integral has a nonzero regional partition.")
    partition_error = np.divide(difference, abs(direct_total), out=np.zeros(3), where=direct_total != 0)
    total_energy = float(direct_total[1:].sum())
    require(total_energy > 0, "Total stored energy is nonpositive.")
    energy_closure_over_total = difference[1:]/total_energy
    require(partition_error[0] <= 1e-10 and np.max(energy_closure_over_total) <= 1e-10
            and np.all(region_totals[:, 1:].sum(axis=1) > 0), "Regional volume/energy partition fails closure or has nonpositive regional energy.")
    regions = [{"name": name, "x_interval_m": cuts[i:i+2].tolist(), "volume_m3": float(row[0]),
                "kinetic_energy_J_per_W": float(row[1]), "elastic_energy_J_per_W": float(row[2]),
                "total_energy_J_per_W": float(row[1:].sum()), "total_energy_fraction": float(row[1:].sum()/total_energy)}
               for i, (name, row) in enumerate(zip(REGION_NAMES, region_totals))]
    return {"regions": regions,
            "totals": {"volume_m3": float(direct_total[0]), "kinetic_energy_J_per_W": float(direct_total[1]),
                       "elastic_energy_J_per_W": float(direct_total[2]), "total_energy_J_per_W": total_energy,
                       "energy_per_incident_power_s": total_energy},
            "integration_checks": {"quadrature_degree": 4, "original_parent_P2_evaluated": True,
                                   "centroid_region_assignment": False, "geometry_perturbation_applied": False,
                                   "invertible_hull_conditioning": "centered SVD axes/scales; original barycentric vertices retained",
                                   "cut_tetrahedra": cut_count, "degree_four_monomials_per_cut_tetrahedron": len(MOMENT_POWERS),
                                   "maximum_cut_moment_relative_closure": maximum_moment_error,
                                   "kinetic_elastic_absolute_partition_difference_over_total_energy": energy_closure_over_total.tolist(),
                                   "volume_kinetic_elastic_relative_partition_closure": partition_error.tolist()}}


def write_selected_vtu(path, data):
    _, points, cells, values = validate_field(data)
    meshio.write_points_cells(path, points, [("tetra10", cells)],
                              point_data={"displacement_real_m_per_sqrt_W": values.real,
                                          "displacement_imag_m_per_sqrt_W": values.imag},
                              binary=True, compression="zlib")
    return {"scalar_P2_nodes": len(points), "tetra10_cells": len(cells),
            "edge_order": [list(edge) for edge in EDGES], "native_complex_gauge_preserved": True,
            "visualization_amplitude_rescaled": False, "displacement_units": "m/sqrt(W)"}


def material_from_config(config):
    return CubicMaterial(C11=config["C11_Pa"], C12=config["C12_Pa"], C44=config["C44_Pa"],
                         rho=config["density_kg_m3"], rotation=config["crystal_rotation"])


def load_selected_field(path, summary):
    with np.load(path, allow_pickle=False) as saved:
        require(saved["element_order"].item() == 2 and np.array_equal(saved["incident_mode_indices"], [1302, 1346, 1292, 1286, 1282])
                and np.array_equal(saved["frequencies_hz"], np.full(5, FREQUENCY)), "Saved field order, roots or frequency differ.")
        field = saved["displacements_per_sqrt_w"]
        require(field.dtype == np.complex128 and field.shape == (summary["ndof"], 5), "The archive lacks all accepted complex P2 fields.")
        powers = saved["incident_powers_w"]
        require(powers.shape == (5,) and np.isfinite(powers).all(), "The saved five incident powers have an invalid layout.")
        power = float(powers[INCIDENT_COLUMN])
        require(np.isfinite(power) and abs(power-1) <= 1e-7, "The selected saved incident drive is not unit-power normalized.")
        data = {key: saved[key] for key in ("points_m", "tetrahedra", "dof_coordinates_m", "dof_component", "element_dofs", "nodal_dofs")}
        data["u"] = field[:, INCIDENT_COLUMN].copy()
    return data, power


def load_inputs(workspace, comparison_path):
    root, comparison_path = Path(workspace).resolve(), Path(comparison_path).resolve(strict=True)
    require(comparison_path.is_relative_to(root) and comparison_path.stat().st_size <= 4*2**20
            and sha(comparison_path) == COMPARISON_SHA, "The frozen accepted comparison differs.")
    comparison = read_json(comparison_path)
    require(comparison["status"] == "compared_finite_inplane_increment" and comparison["frequency_hz"] == FREQUENCY
            and comparison["shared_basis"]["status"] == "identical_saved_basis", "Accepted comparison scope differs.")
    inputs, sources = {"accepted_comparison": artifact(comparison_path, root)}, {}
    for mesh, expected_dofs, resolution in (("h100", 307611, 100), ("h80", 426627, 80)):
        replay_role = "point_replay" if mesh == "h100" else "accepted_point"
        rp = checked_path(root, comparison["inputs"][replay_role], expected_sha=REPLAY_SHAS[mesh])
        sp = checked_path(root, comparison["inputs"][mesh+"_8875_summary"])
        replay, summary = read_json(rp), read_json(sp)
        require(replay["status"] == "point_verified_within_saved_discretization" and replay["problem_dofs"] == expected_dofs
                and replay["frequency_hz"] == FREQUENCY and replay["full_p2_field_identity_passed"] is True
                and replay["public_summary_sha256"] == sha(sp), "Accepted field replay does not bind this mesh/summary.")
        require(summary["ndof"] == expected_dofs and summary["frequency_hz"] == FREQUENCY
                and summary["incident_side"] == "left" and summary["incident_mode_indices"] == [1302, 1346, 1292, 1286, 1282]
                and summary["configuration"]["mesh"]["mesh_size_nm"] == resolution
                and summary["configuration"]["mesh"]["thickness_layers"] == 4
                and summary["checkpointed_scattering"]["phase_adjustment_applied"] is False
                and summary["checkpointed_scattering"]["reference_planes_m"] == [0., 1.665e-5], "Saved selected response scope differs.")
        fp = sp.parent / "selected_fields_AP_pnc.npz"
        require(summary["output_sha256"][fp.name] == FIELD_SHAS[mesh], "The accepted selected-field hash differs.")
        checked_path(root, {"path": str(fp), "bytes": fp.stat().st_size, "sha256": FIELD_SHAS[mesh]}, max_bytes=160*2**20)
        for role, path in (("replay", rp), ("summary", sp), ("fields", fp)):
            inputs[mesh+"_"+role] = artifact(path, root)
        sources[mesh] = (summary, fp)
    a, b = (sources[m][0]["configuration"] for m in ("h100", "h80"))
    require(all(a[k] == b[k] for k in ("geometry", "material", "analysis")), "Physical configurations differ between saved fields.")
    require(a["geometry"]["length_nm"] == 16650 and a["geometry"]["width_nm"] == 1000
            and a["geometry"]["thickness_nm"] == 200, "The frozen physical dimensions differ.")
    material_source = ROOT/"src/phononic/material_AP_pnc.py"
    require(all(sha(material_source) == sources[m][0]["implementation"]["source_sha256"]["src/phononic/material_AP_pnc.py"]
                for m in ("h100", "h80")), "Current material tensor source differs from the accepted archived source.")
    return inputs, sources


def independent_receipt(output, record):
    """Identify the independent verification of these exact regional-integral bytes."""
    path = output.parent / RECEIPT_NAME
    source = output / RECORD_NAME
    try:
        if not path.is_file() or not source.is_file():
            return None
        proof = json.loads(path.read_bytes())
        payload = source.read_bytes()
        binding = proof["receipt"]
        if (proof["status"] == "saved_field_energy_independently_verified"
                and binding["bytes"] == len(payload)
                and binding["sha256"] == hashlib.sha256(payload).hexdigest()
                and proof["executed_producer_sha256"] == record["producer"]["sha256"]
                and proof["material_source"] == record["material_source"]
                and json.loads(payload) == record):
            return path
    except (OSError, ValueError, KeyError, TypeError):
        pass
    return None


def write_report(output, record):
    output = Path(output)
    rows = []
    for mesh, value in record["meshes"].items():
        for row in value["regions"]:
            rows.append({"mesh": mesh, **{k: v for k, v in row.items() if k != "x_interval_m"},
                         "x_lower_m": row["x_interval_m"][0], "x_upper_m": row["x_interval_m"][1]})
    csv_path = output / "regional_energy_AP_pnc.csv"
    with csv_path.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    cells = "".join("<tr>"+"".join(f"<td>{escape(str(row[k]))}</td>" for k in rows[0])+"</tr>" for row in rows)
    header = "".join(f'<th scope="col">{escape(k)}</th>' for k in rows[0])
    table = scroll_table(f"<table><thead><tr>{header}</tr></thead><tbody>{cells}</tbody></table>",
                         "Regional energy integrals")
    bars = "".join(f'<p>{mesh}: '+"".join(f'<span title="{r["name"]}: {r["total_energy_fraction"]:.8g}" style="display:inline-block;background:{color};width:{r["total_energy_fraction"]*80:.8g}%;height:1.2rem"></span>' for r, color in zip(value["regions"], ("#547cb3", "#da9944", "#64a780")))+"</p>" for mesh, value in record["meshes"].items())
    workspace = Path(record["workspace"])
    field_links = []
    for mesh, value in record["meshes"].items():
        selected = output / Path(value["selected_field_vtu"]["path"]).name
        original = workspace / record["inputs"][mesh+"_fields"]["path"]
        field_links.append(available_link(output, selected, f"{mesh} root 1302 full quadratic field (VTU)"))
        field_links.append(available_link(output, original, f"{mesh} original all-drive full-P2 field (NPZ)",
                                          unavailable="original workspace archive required"))
    links = "".join(f"<li>{link}</li>" for link in field_links)
    outputs = " · ".join((available_link(output, output / RECORD_NAME, "Regional integrals and provenance (JSON)", download=True),
                          available_link(output, csv_path, "Regional integrals (CSV)", download=True)))
    receipt = independent_receipt(output, record)
    verification = (available_link(output, receipt, "Independent numerical verification receipt (JSON)") if receipt else
                    '<span class="availability">Independent numerical verification receipt (unavailable for this report)</span>')
    compact = available_link(output, ROOT / "verification/field_energy_8875_AP_pnc.json",
                             "Compact verification record for the archived comparison (JSON)")
    result = record["comparison"]
    status = "Passed" if result["finite_fraction_target_passed"] else "Failed"
    scope = ("One native left-incident root 1302 field per mesh at the original x = 0 and 16.65 µm planes. "
             "Regional fractions use total cycle-averaged kinetic plus elastic energy.")
    body = f'''
<p><strong>Finite fraction target: {status}.</strong> Maximum fraction increment {result["max_absolute_fraction_increment"]:.12g}, with target max |F80−F100| &lt; 0.02.</p>
<p class="sources">{outputs}</p>
<p class="sources">{verification} · {compact}</p>
<p>Displacements retain m/√W and their original complex phase. Integrated energies have units J/W = s (numerically joules for a 1 W drive); no second division by stored unit-drive power is applied. E/P is a diagnostic, not delay or Q.</p>
{bars}<p>Bars: left lead (blue), patterned region (orange), right lead (green).</p>
{table}
<h2>Field files and scope</h2><ul>{links}</ul><ul>{''.join('<li>'+escape(s)+'</li>' for s in LIMITATIONS)}</ul>'''
    html = report_page("Saved regional field energy at 887.5 MHz", "field_energy_8875", body, scope=scope,
                       navigation=available_link(output, ROOT / "verification/README.md", "Verification records"))
    (output/"report_AP_pnc.html").write_text(html, encoding="utf-8")


def run(workspace, comparison_path, output, batch_elements=256):
    start, output, workspace = monotonic(), Path(output).resolve(), Path(workspace).resolve()
    require(not output.exists(), "Output already exists; preserve scientific results in a fresh directory.")
    require(output.is_relative_to(workspace), "Output must remain inside the workspace.")
    process, peak_rss = psutil.Process(), 0
    def guard():
        nonlocal peak_rss
        peak_rss = max(peak_rss, process.memory_info().rss)
        require(monotonic()-start < 90 and peak_rss <= 2**30 and psutil.virtual_memory().available >= 512*2**20,
                "Saved-field postprocessing exceeded its 90 s / 1 GiB / 512 MiB reserve admission.")
    guard()
    inputs, sources = load_inputs(workspace, comparison_path)
    guard()
    output.mkdir(parents=True)
    record = {**AUTHOR, "schema_version": 1, "status": "saved_field_energy_compared_finite_increment", "workspace": str(workspace),
              "research_question": "Are the saved total-field regional energy fractions stable under the h100/h80 in-plane increment at 887.5 MHz?",
              "scope": {"frequency_hz": FREQUENCY, "incident_root": INCIDENT_ROOT, "incident_column": INCIDENT_COLUMN,
                        "region_cuts_m": CUTS.tolist(), "native_incident_gauge_preserved": True, "new_solve_performed": False,
                        "new_assembly_performed": False, "field_fitting_performed": False, "cross_mesh_interpolation_performed": False,
                        "spatial_convergence_established": False, "frequency_convergence_established": False},
              "inputs": inputs, "producer": artifact(__file__, workspace),
              "material_source": artifact(ROOT/"src/phononic/material_AP_pnc.py", workspace),
              "numerical_environment": {"python": platform.python_version(), "platform": platform.platform(),
                                        "numpy": np.__version__, "scipy": scipy.__version__, "scikit-fem": skfem.__version__, "meshio": meshio.__version__,
                                        "thread_environment": {k: os.environ.get(k) for k in THREAD_KEYS}},
              "predeclared_target": {"max_absolute_fraction_increment": LIMIT, "comparison": "strict_less_than"},
              "meshes": {}, "limitations": list(LIMITATIONS)}
    for mesh, (summary, field_path) in sources.items():
        guard()
        data, power = load_selected_field(field_path, summary)
        value = integrate_regions(data, material_from_config(summary["configuration"]["material"]), FREQUENCY, guard=guard, batch_elements=batch_elements)
        guard()
        width, thickness = summary["configuration"]["geometry"]["width_nm"]*1e-9, summary["configuration"]["geometry"]["thickness_nm"]*1e-9
        analytic_leads = np.array([CUTS[1]-CUTS[0], CUTS[3]-CUTS[2]])*width*thickness
        lead_error = abs(np.array([value["regions"][i]["volume_m3"] for i in (0, 2)])-analytic_leads)/analytic_leads
        require(np.max(lead_error) <= 1e-10, "Clipped left/right lead volumes differ from the physical rectangular leads.")
        value["integration_checks"]["analytic_rectangular_lead_relative_volume_errors"] = lead_error.tolist()
        vtu = output/(mesh+"_root1302_P2_AP_pnc.vtu")
        export = write_selected_vtu(vtu, data)
        value.update(frequency_hz=FREQUENCY, incident_root=INCIDENT_ROOT, incident_column=INCIDENT_COLUMN, incident_power_w=power,
                     selected_field_vtu=artifact(vtu, workspace), selected_field_export=export)
        record["meshes"][mesh] = value
        del data
        guard()
    a, b = (record["meshes"][mesh] for mesh in ("h100", "h80"))
    increments = [y["total_energy_fraction"]-x["total_energy_fraction"] for x, y in zip(a["regions"], b["regions"])]
    maximum = max(abs(i) for i in increments)
    record["comparison"] = {"regional_fraction_increments": increments, "max_absolute_fraction_increment": maximum,
                            "finite_fraction_target_passed": maximum < LIMIT,
                            "relative_total_energy_change": (b["totals"]["total_energy_J_per_W"]-a["totals"]["total_energy_J_per_W"])/a["totals"]["total_energy_J_per_W"]}
    guard()
    record["resources"] = {"elapsed_s": monotonic()-start, "sampled_peak_rss_bytes": peak_rss, "rss_limit_bytes": 2**30,
                           "wall_limit_s": 90, "available_memory_reserve_bytes": 512*2**20, "output_limit_bytes": 200*2**20,
                           "batch_elements": batch_elements}
    (output / RECORD_NAME).write_text(json.dumps(record, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    write_report(output, record)
    guard()
    require(sum(p.stat().st_size for p in output.iterdir()) <= 200*2**20, "Completed output exceeds the admitted 200 MiB ceiling.")
    return record


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--workspace", type=Path, default=ROOT)
    parser.add_argument("--comparison", type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--batch-elements", type=int, default=256)
    args = parser.parse_args(argv)
    record = run(args.workspace, args.comparison or args.workspace/COMPARISON_PATH, args.output, args.batch_elements)
    print(json.dumps({"status": record["status"], **record["comparison"]}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
