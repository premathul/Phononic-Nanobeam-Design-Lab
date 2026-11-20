"""Compute one saved-frequency SAFE point for the 150 nm uniform lead.

Athul Prem | _AP_pnc. Reuses the archived paired-device cross-section layout.
"""

from __future__ import annotations

import argparse
import os
from pathlib import Path
import sys
from types import SimpleNamespace

THREADS = dict.fromkeys(("OPENBLAS_NUM_THREADS", "OMP_NUM_THREADS", "MKL_NUM_THREADS", "BLIS_NUM_THREADS"), "1")
if __name__ == "__main__":
    os.environ.update(THREADS)

import numpy as np
from skfem import Basis, ElementTetP2, ElementTriP2, ElementVector, MeshTet, MeshTri

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT), str(ROOT / "src")]
from phononic import port_modes_AP_pnc as safe
from phononic import workflow_AP_pnc as workflow
from phononic.resumable_scattering_AP_pnc import AUTHOR, _exclusive, _link, _numerical_identity, _read, _verified
from tools import map_scattering_AP_pnc as native
from tools import verify_port_count_interval_AP_pnc as verify

RUNS = ROOT / "results/runs"
STUDY = RUNS / "physical_thickness_section_AP_pnc"
DEVICE = RUNS / "physical_thickness_device_AP_pnc"
SAMPLES = {995_000_000: "995MHz", 997_500_000: "997p5MHz", 1_000_000_000: "1000MHz"}
OPERATORS = ("K0", "M", "Q", "G")
WIDTH_NM, THICKNESS_NM, PORT_DOFS = 1000, 150, 135


def matched_layout():
    """Require common normalized topology and lab material from both device endpoints."""
    sections, materials, links = [], [], {}
    for thickness, label in ((100, "thin100"), (200, "thick200")):
        endpoint = {}
        for frequency, tag in SAMPLES.items():
            directory = DEVICE / f"{label}_central400_rim48_{tag.replace('1000MHz', '1GHz')}"
            state = native.read_map_checkpoint(directory)
            config = state["inputs"]["configuration"]
            geometry, mesh = config["geometry"], config["mesh"]
            verify.require(any(point["frequency_GHz"] == frequency / 1e9 and point["status"] == "accepted"
                               for point in state["points"])
                           and (geometry["width_nm"], geometry["thickness_nm"], mesh["element_order"],
                                mesh["thickness_layers"]) == (WIDTH_NM, thickness, 2, 2),
                           f"Unexpected port configuration or unreadable accepted point in {directory}.")
            materials.append(config["material"])
            pin = state["shared"]["artifacts"]["port_section"]
            section_file = _verified(pin, directory)
            endpoint[tag] = _link(directory / native.CHECKPOINT, ROOT)
            if frequency == 995_000_000:
                section_hash = pin["sha256"]
            verify.require(pin["sha256"] == section_hash, f"The {thickness} nm section changed by frequency.")
        links[f"{thickness}_nm"] = {"section": _link(section_file, ROOT), "frequency_checkpoints": endpoint}
        with np.load(section_file, allow_pickle=False) as arrays:
            points, triangles, mass = (arrays[key] for key in ("points_m", "triangles", "M"))
        verify.require(points.shape == (2, 15) and triangles.shape == (3, 16)
                       and mass.shape == (PORT_DOFS, PORT_DOFS)
                       and tuple(np.unique(row).size for row in points) == (5, 3),
                       "Expected the native 4-by-2 quadratic section.")
        np.testing.assert_allclose(np.ptp(points, axis=1), np.array([WIDTH_NM, thickness]) * 1e-9,
                                   rtol=1e-14, atol=0)
        sections.append((points, triangles))
    verify.require(all(material == materials[0] for material in materials), "The endpoint lab materials differ.")
    (thin, thin_triangles), (thick, triangles) = sections
    np.testing.assert_array_equal(thin_triangles, triangles)
    np.testing.assert_allclose(thin, thick * np.array([[1.0], [0.5]]), rtol=1e-14, atol=0)
    thick[1] *= THICKNESS_NM / 200
    return thick, triangles, materials[0], links


def assemble(yz, triangles, material):
    # As in the archived 100 nm control, one prism slab gives assemble_port
    # congruent P2 end DOF maps. No volume operator or response is assembled.
    n = yz.shape[1]
    points = np.vstack((np.repeat([0., np.ptp(yz[0])], n), np.tile(yz, 2)))
    a, b, c = np.sort(triangles, axis=0)
    tetrahedra = np.column_stack((np.stack((a, b, c, c + n)), np.stack((a, b, b + n, c + n)),
                                   np.stack((a, a + n, b + n, c + n))))
    mesh = MeshTet(points, tetrahedra).oriented()
    layout = Basis(mesh, ElementVector(ElementTetP2()), intorder=4)
    ge = workflow.CubicMaterial(C11=material["C11_Pa"], C12=material["C12_Pa"], C44=material["C44_Pa"],
                                rho=material["density_kg_m3"], rotation=material["crystal_rotation"])
    return safe.assemble_port(SimpleNamespace(mesh=mesh, basis=layout), ge, max_dofs=PORT_DOFS)


def save_arrays(path, **arrays):
    temporary = path.with_suffix(".npz.tmp")
    with temporary.open("wb") as stream:
        np.savez_compressed(stream, **arrays)
    temporary.replace(path)


def shared_section(source):
    yz, triangles, material, links = matched_layout()
    inputs = {**AUTHOR, "schema_version": 1, "physical_width_nm": WIDTH_NM, "physical_thickness_nm": THICKNESS_NM,
              "material": material, "element_order": 2, "width_intervals": 4, "thickness_layers": 2,
              "section_triangles": 16, "port_dofs": PORT_DOFS, "frequencies_hz": list(SAMPLES),
              "lead_axis_lab": "+x", "lateral_boundary": "traction_free",
              "wave_convention": "exp(i*k*x - i*omega*t)", "native_endpoint_sections": links,
              "driver": _link(Path(__file__).resolve(), ROOT),
              "independent_replay": _link(Path(verify.__file__).resolve(), ROOT), "threads": THREADS}
    identity = workflow.digest({**inputs, "numerical_identity": _numerical_identity(source)})
    data, receipt = STUDY / "mid150_section_AP_pnc.npz", STUDY / "mid150_section_AP_pnc.json"
    if receipt.exists():
        saved = _read(receipt)
        verify.require(saved["inputs_sha256"] == identity and _verified(saved["section_arrays"], STUDY) == data,
                       "Saved 150 nm section inputs differ.")
    resources = None
    if not data.exists():
        with workflow.measured_resources() as resources:
            port = assemble(yz, triangles, material)
            np.testing.assert_array_equal(port.basis.mesh.p, yz)
            np.testing.assert_array_equal(port.basis.mesh.t, triangles)
            verify.require(port.basis.N == PORT_DOFS, "Unexpected midpoint section DOF count.")
            width_m, thickness_m = np.ptp(yz, axis=1)
            save_arrays(data, **{key: getattr(port, key) for key in OPERATORS}, points_m=yz, triangles=triangles,
                        left_dofs=port.left_dofs, right_dofs=port.right_dofs, width_m=width_m, thickness_m=thickness_m,
                        reference_speed_m_s=port.reference_speed_m_s, inputs_sha256=identity)
    with np.load(data, allow_pickle=False) as arrays:
        verify.require(str(arrays["inputs_sha256"]) == identity, "Unbound midpoint section arrays.")
        np.testing.assert_array_equal(arrays["points_m"], yz)
        np.testing.assert_array_equal(arrays["triangles"], triangles)
        basis = Basis(MeshTri(yz, triangles), ElementVector(ElementTriP2(), dim=3), intorder=4)
        ops = {key: arrays[key] for key in OPERATORS}
        port = safe.PortSystem(None, basis, arrays["left_dofs"], arrays["right_dofs"],
                               reference_speed_m_s=float(arrays["reference_speed_m_s"]), **ops)
    if not receipt.exists():
        workflow.write_json(receipt, {**inputs, "inputs_sha256": identity, "source": source,
                                      "section_arrays": _link(data, STUDY), "assembly_resources": resources})
    return port, ops, _link(data, STUDY), _link(receipt, STUDY)


def run(frequency):
    with _exclusive(STUDY / ".mid150_section_AP_pnc.lock"):
        source = workflow.source_identity()
        port, ops, section_link, metadata_link = shared_section(source)
        stem = f"mid150_{SAMPLES[frequency]}"
        data, receipt = STUDY / f"{stem}_modes_AP_pnc.npz", STUDY / f"{stem}_AP_pnc.json"
        record = {**AUTHOR, "schema_version": 1, "frequency_hz": frequency, "section_arrays": section_link,
                  "section_metadata": metadata_link}
        if receipt.exists():
            saved = _read(receipt)
            verify.require(all(saved[key] == value for key, value in record.items()), f"Saved inputs differ for {stem}.")
            verify.require(saved["status"] == "accepted", f"Earlier failure retained in {receipt}.")
            verify.require(_verified(saved["mode_arrays"], STUDY) == data, f"Saved mode archive differs for {stem}.")
            print(f"{stem}: accepted, reused saved point")
            return
        width_m = float(np.ptp(port.basis.mesh.p[0]))
        with workflow.measured_resources() as resources:
            try:
                if not data.exists():
                    modes = safe.solve_port_modes(port, frequency)
                    save_arrays(data, **{key: getattr(modes, key) for key in verify.MODE_FIELDS},
                                section_sha256=section_link["sha256"])
                with np.load(data, allow_pickle=False) as arrays:
                    record["mode_arrays"] = _link(data, STUDY)
                    verify.require(str(arrays["section_sha256"]) == section_link["sha256"], "Unbound midpoint modes.")
                    metrics = verify.replay_modes(ops, arrays, frequency, length_m=width_m,
                                                  reference_speed_m_s=port.reference_speed_m_s)
                    k = arrays["wavenumbers_rad_m"]
                    closed = k[~arrays["propagating"]]
                    record.update(status="accepted", metrics=metrics, cutoff_status="solver_resolved_at_sample",
                                  minimum_abs_k_times_width=float(abs(k).min()) * width_m,
                                  minimum_closed_abs_imag_k_times_width=float(abs(closed.imag).min()) * width_m)
            except (RuntimeError, ValueError, np.linalg.LinAlgError) as exc:
                record.update(status="failed", error=str(exc))
        workflow.write_json(receipt, {**record, "source_git_commit": source["git_commit"], "resources": resources})
        verify.require(record["status"] == "accepted", f"{stem} failed: {record.get('error')}")
        print(f"{stem}: {metrics['discrete_roots']} roots, "
              f"{metrics['positive_open_channels']}/{metrics['negative_open_channels']} open, "
              f"independent residual {metrics['maximum_eigen_backward_residual']:.3e}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("frequency_hz", type=int, choices=SAMPLES)
    run(parser.parse_args().frequency_hz)
