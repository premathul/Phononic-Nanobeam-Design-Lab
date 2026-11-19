"""Nested central mesh and native two-hole P2 check. Author: Athul Prem | _AP_pnc."""

import contextlib
from copy import deepcopy
from importlib.util import find_spec
import io
import json
from pathlib import Path
import sys
from tempfile import TemporaryDirectory
import unittest
from unittest.mock import patch

import numpy as np
from scipy.sparse import load_npz

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "tools"))
sys.path.insert(0, str(ROOT / "src"))

import map_scattering_AP_pnc as mapper
from phononic import workflow_AP_pnc as workflow


CUTS_NM = (400, 1400)
TARGETS_NM = (140, 105)
FREQUENCY_GHZ = 0.1


def two_hole_region():
    config = json.loads((ROOT / "configs/perforated_layered_reference_AP_pnc.json").read_bytes())
    config["name"] = "two_hole_central_refinement_AP_pnc"
    config["mesh"].update(mesh_size_nm=200, port_size_nm=200, thickness_layers=1,
                          circle_resolution=16, lead_mesh_size_nm=150, patterned_x_interval_nm=list(CUTS_NM))
    config["analysis"]["frequency_GHz"] = FREQUENCY_GHZ
    config["resources"].update(max_dofs=7000, max_port_dofs=63)
    return config


def coordinate_cells(points):
    return {tuple(sorted(map(tuple, cell))) for cell in points}


def exterior_and_central_floor(mesh):
    points_nm = mesh.p.T * 1e9
    floor = mesh.facets[:, mesh.boundary_facets()].T
    floor = floor[np.all(points_nm[floor, 2] == points_nm[:, 2].min(), axis=1)]
    x = points_nm[floor, 0]
    central = np.all((x >= CUTS_NM[0] - 1e-10) & (x <= CUTS_NM[1] + 1e-10), axis=1)
    return points_nm[floor[central], :2], points_nm[floor[~central], :2]


def area(triangles):
    return abs(np.linalg.det(triangles[:, 1:] - triangles[:, :1])) / 2


def bottom_edges(triangles):
    edges = [tuple(sorted((tuple(a), tuple(b)))) for triangle in triangles
             for a, b in zip(triangle, np.roll(triangle, -1, axis=0), strict=True)]
    unique, counts = np.unique(edges, axis=0, return_counts=True)
    return unique, counts


class CentralRefinementConfiguration(unittest.TestCase):
    def test_opt_in_validates_size_and_regional_mesh(self):
        config = two_hole_region()
        self.assertNotIn("central_max_edge_nm", workflow.resolve_config(config)["mesh"])
        for invalid_size in (0, -1, True, None, "100", float("nan"), float("inf")):
            invalid = deepcopy(config)
            invalid["mesh"]["central_max_edge_nm"] = invalid_size
            with self.subTest(size=invalid_size), self.assertRaisesRegex(ValueError, "central_max_edge_nm"):
                workflow.resolve_config(invalid)
        for kind in ("layered", "gmsh"):
            invalid = deepcopy(config)
            invalid["mesh"].update(kind=kind, central_max_edge_nm=105)
            if kind == "layered":
                del invalid["mesh"]["lead_mesh_size_nm"]
                del invalid["mesh"]["patterned_x_interval_nm"]
            with self.subTest(kind=kind), self.assertRaises(ValueError):
                workflow.resolve_config(invalid)


@unittest.skipUnless(find_spec("gmsh"), "requires gmsh")
class CentralRefinementWorkflow(unittest.TestCase):
    def assert_nested(self, parent, child):
        # Map every child to its geometric parent, independently of node IDs
        # or the reported refinement diagnostics.
        child_area = area(child)
        self.assertTrue(np.all(child_area > 0))
        membership = np.zeros(len(child), dtype=int)
        for triangle, parent_area in zip(parent, area(parent), strict=True):
            barycentric = np.linalg.solve((triangle[1:] - triangle[0]).T,
                                          (child - triangle[0]).reshape(-1, 2).T).T.reshape(-1, 3, 2)
            inside = np.all((barycentric >= -1e-10), axis=(1, 2))
            inside &= np.all(barycentric.sum(axis=2) <= 1 + 1e-10, axis=1)
            membership += inside
            self.assertAlmostEqual(child_area[inside].sum() / parent_area, 1, places=10)
        np.testing.assert_array_equal(membership, np.ones(len(child), dtype=int))

    def test_nested_footprint_preserved_traces_and_native_map(self):
        with TemporaryDirectory() as temporary:
            root = Path(temporary)
            config = two_hole_region()
            config_path = root / "two_hole_AP_pnc.json"
            workflow.write_json(config_path, config)
            cache = root / "mesh-check"
            parent = workflow.preflight_mesh(config, cache)
            parent_directory = Path(parent["output_directory"])
            parent_record = {p.name: (p.read_bytes(), p.stat().st_mtime_ns) for p in parent_directory.iterdir()}

            records = [parent]
            for target in TARGETS_NM:
                with contextlib.redirect_stdout(io.StringIO()):
                    self.assertEqual(workflow.main(["--config", str(config_path), "--output-root", str(cache),
                                                    "--mesh-only", "--central-max-edge-nm", str(target)]), 0)
                config["mesh"]["central_max_edge_nm"] = target
                record = workflow.preflight_mesh(config, cache)
                self.assertTrue(record["reused"])
                self.assertEqual(record["mesh_inputs"]["settings"]["central_max_edge_nm"], target)
                records.append(record)
            self.assertEqual(len({record["mesh_hash"] for record in records}), len(records))
            meshes = [workflow.load_checkpoint(Path(record["output_directory"]), record["mesh_hash"])[0]
                      for record in records]
            largest_edges_nm = self.check_geometry(meshes)
            self.assertEqual([record["end_face_dofs"] for record in records], [{"xleft": 63, "xright": 63}] * 3)

            workflow.write_json(config_path, config)
            output = root / "map"
            stdout = io.StringIO()
            stderr = io.StringIO()
            with (patch.object(workflow, "build_mesh", side_effect=AssertionError("checked mesh rebuilt")),
                  contextlib.redirect_stdout(stdout), contextlib.redirect_stderr(stderr)):
                status = mapper.main(["--config", str(config_path), "--frequencies-ghz", str(FREQUENCY_GHZ),
                                      "--output", str(output), "--max-new-points", "1", "--mesh-cache-root", str(cache)])
            self.assertEqual(status, 0, stderr.getvalue())
            saved = mapper.read_saved_map(output)
            point, = saved["points"]
            self.assertEqual(point["status"], "accepted")
            self.assertEqual(saved["shared"]["mesh_hash"], records[-1]["mesh_hash"])
            self.assertEqual(saved["shared"]["reference_planes_x_m"], [0, config["geometry"]["length_nm"] * 1e-9])
            summary_path = output / point["accepted"]["summary"]["path"]
            summary = json.loads(summary_path.read_bytes())
            physical = self.check_scattering(saved["shared"], summary_path.parent, summary, output)

            accepted_paths = (summary_path, output / mapper.CHECKPOINT, output / "map_AP_pnc.json")
            accepted_files = [(path.read_bytes(), path.stat().st_mtime_ns) for path in accepted_paths]
            config["mesh"]["central_max_edge_nm"] = TARGETS_NM[0]
            with self.assertRaisesRegex(ValueError, "checkpoint differs"):
                mapper.map_scattering(config, [FREQUENCY_GHZ], output, max_new_points=1)
            self.assertEqual([(path.read_bytes(), path.stat().st_mtime_ns) for path in accepted_paths], accepted_files)
            self.assertEqual({p.name: (p.read_bytes(), p.stat().st_mtime_ns) for p in parent_directory.iterdir()}, parent_record)
            print("Two-hole central refinement: " + json.dumps({
                "target_nm": [None, *TARGETS_NM], "maximum_edge_at_least_200nm_from_cuts_nm": largest_edges_nm,
                "tetrahedra": [record["mesh_tetrahedra"] for record in records],
                "ndof": [record["ndof"] for record in records], "frequency_ghz": FREQUENCY_GHZ,
                "channels": summary["num_incident_channels"], **physical,
                "assembly_resources": saved["shared"]["assembly_resources"],
                "scattering_resources": summary["scattering_resources"],
            }))

    def check_geometry(self, meshes):
        footprints = [exterior_and_central_floor(mesh) for mesh in meshes]
        for (parent, _), (child, _) in zip(footprints, footprints[1:]):
            self.assertGreater(len(child), len(parent))
            self.assert_nested(parent, child)

        def protected(mesh):
            x = mesh.p[0, mesh.t]
            start, stop = np.array(CUTS_NM) * 1e-9
            lead = (x.max(axis=0) <= start) | (x.min(axis=0) >= stop)
            center = (x.min(axis=0) >= start) & (x.max(axis=0) <= stop)
            self.assertTrue(np.all(lead ^ center))
            cut_facets = [np.flatnonzero(np.all(mesh.p[0, mesh.facets] == cut, axis=0)) for cut in (start, stop)]
            for facets in cut_facets:
                self.assertGreater(len(facets), 0)
                self.assertTrue(np.all(mesh.f2t[:, facets] >= 0))
            protected_faces = (*cut_facets, mesh.boundaries["xleft"], mesh.boundaries["xright"])
            return (coordinate_cells(mesh.p.T[mesh.t.T[lead]]),
                    [coordinate_cells(mesh.p.T[mesh.facets[:, ids].T]) for ids in protected_faces])

        base_protected = protected(meshes[0])
        base_outer_floor = coordinate_cells(footprints[0][1])
        rims = None
        largest_edges = []
        for index, (mesh, (central, outer)) in enumerate(zip(meshes, footprints, strict=True)):
            self.assertTrue(np.all(mesh.mapping().detA > 0))
            if index:
                self.assertLessEqual(set(map(tuple, meshes[index - 1].p.T)), set(map(tuple, mesh.p.T)))
                self.assertEqual(protected(mesh), base_protected)
                self.assertEqual(coordinate_cells(outer), base_outer_floor)
            edges, counts = bottom_edges(central)
            self.assertTrue(np.all((counts == 1) | (counts == 2)))
            boundary = edges[counts == 1]
            wall = np.any(np.all(np.isclose(boundary[:, :, 1, None], [-300, 300], rtol=0, atol=1e-10), axis=1), axis=1)
            cut = np.any(np.all(np.isclose(boundary[:, :, 0, None], CUTS_NM, rtol=0, atol=1e-10), axis=1), axis=1)
            exposed_rims = coordinate_cells(boundary[~(wall | cut)])
            if rims is None:
                rims = exposed_rims
                self.assertEqual(len(rims), 32)
            self.assertEqual(exposed_rims, rims)
            free_faces_nm = mesh.p.T[mesh.facets[:, mesh.boundaries["free"]].T] * 1e9
            planar = np.all(free_faces_nm[:, :, 2] == free_faces_nm[:, :1, 2], axis=1)
            sidewalls = np.any(np.all(np.isclose(free_faces_nm[:, :, 1, None], [-300, 300],
                                               rtol=0, atol=1e-10), axis=1), axis=1)
            # An inconsistent extrusion diagonal would leave a spurious interior free face.
            for face in free_faces_nm[~(planar | sidewalls)]:
                self.assertIn(tuple(sorted(set(map(tuple, face[:, :2])))), rims)
            # Both cut transitions have reached the requested interior edge limit here.
            core = edges[np.all((edges[:, :, 0] >= 600 - 1e-10) & (edges[:, :, 0] <= 1200 + 1e-10), axis=1)]
            largest_edges.append(float(np.linalg.norm(core[:, 1] - core[:, 0], axis=1).max()))
        for previous, current, target in zip(largest_edges, largest_edges[1:], TARGETS_NM):
            self.assertLess(current, previous)
            self.assertLessEqual(current, target + 1e-9)
        return largest_edges

    def check_scattering(self, shared, point_path, summary, output):
        self.assertEqual(summary["hole_count"], 2)
        self.assertGreater(summary["num_incident_channels"], 1)
        self.assertLess(summary["maximum_relative_port_residual"], 1e-8)
        self.assertTrue(summary["power_balance_passed"])
        self.assertIsNone(summary["matched_reference_passed"])
        self.assertFalse(summary["spatial_convergence_established"])
        equations = summary["checkpointed_scattering"]["original_equations"]
        relative = max(row["relative_residual"] for row in equations["channels"])
        backward = max(row["relative_backward_residual"] for row in equations["channels"])
        self.assertLess(relative, 1e-9)
        self.assertLess(backward, 1e-9)

        # Replay a coherent left-incident block from the saved body operators,
        # total fields and SAFE traces, using outward nodal work.
        stiffness, mass = (load_npz(output / shared["artifacts"][key]["path"]) for key in ("stiffness", "mass"))
        omega = 2 * np.pi * FREQUENCY_GHZ * 1e9
        with np.load(point_path / "selected_fields_AP_pnc.npz", allow_pickle=False) as fields:
            roots = fields["incident_mode_indices"]
            left, right = fields["port_left_dofs"], fields["port_right_dofs"]
            incident_u = fields["port_displacements"][:, roots]
            incident_force = -fields["port_axial_forces"][:, roots]
            total_u = fields["displacements_per_sqrt_w"]
            total_force = (stiffness - omega**2 * mass) @ total_u
            weights = np.arange(1, len(roots) + 1) + 1j * np.arange(len(roots), 0, -1)
            weights /= np.linalg.norm(weights)
            field = total_u @ weights
            force = total_force @ weights
            left_u, left_force = incident_u @ weights, incident_force @ weights
            incident_power = -(omega / 2) * np.vdot(left_u, left_force).imag
            direct_r = (omega / 2) * np.vdot(field[left] - left_u, force[left] - left_force).imag
            direct_t = (omega / 2) * np.vdot(field[right], force[right]).imag
            reflection, transmission = fields["reflection_amplitudes"], fields["transmission_amplitudes"]
            modal_r, modal_t = (np.linalg.norm(block @ weights)**2 for block in (reflection, transmission))
            defect = reflection.conj().T @ reflection + transmission.conj().T @ transmission - np.eye(len(roots))
            coherent_norm = float(np.max(abs(np.linalg.eigvalsh(defect))))
            coherent_error = float(max(abs(direct_r - modal_r), abs(direct_t - modal_t), abs(direct_r + direct_t - 1)))
        self.assertAlmostEqual(incident_power, 1, delta=1e-7)
        self.assertLess(coherent_norm, 0.01)
        self.assertLess(coherent_error, 0.01)
        return {"relative_residual": relative, "backward_residual": backward,
                "maximum_relative_port_residual": summary["maximum_relative_port_residual"],
                "maximum_modal_flux_difference": summary["maximum_modal_flux_difference"],
                "coherent_body_force_power_error": coherent_error, "coherent_defect_spectral_norm": coherent_norm}


if __name__ == "__main__":
    unittest.main()
