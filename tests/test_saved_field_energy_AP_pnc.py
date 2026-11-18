"""Portable exact-integration/export checks. Author: Athul Prem | _AP_pnc."""
from __future__ import annotations

import json
import math
from pathlib import Path
import tempfile
import sys
import unittest
from unittest.mock import patch

import meshio
import numpy as np
from skfem import Basis, ElementTetP2, ElementVector, MeshTet

sys.path.insert(0, str(Path(__file__).resolve().parents[1]/"src"))

from phononic.elasticity_AP_pnc import assemble_elasticity
from phononic.material_AP_pnc import CubicMaterial
from tools import analyze_saved_field_energy_AP_pnc as energy


def make_field(length=2e-6, field=None):
    mesh = MeshTet.init_tensor(np.array([0., length/2, length]), np.array([-.35e-6, .35e-6]), np.array([-.15e-6, .15e-6])).oriented()
    basis = Basis(mesh, ElementVector(ElementTetP2()), intorder=4)
    component = np.empty(basis.N, dtype=np.int8)
    for c, ids in enumerate(basis.split_indices()):
        component[ids] = c
    vector = (field or (lambda x: np.column_stack((1+x[:, 0]/length, 2j+0*x[:, 0], x[:, 1]/length))))(basis.doflocs.T)
    u = vector[np.arange(basis.N), component].astype(complex)
    return {"points_m": mesh.p.T, "tetrahedra": mesh.t.T, "dof_coordinates_m": basis.doflocs.T,
            "dof_component": component, "element_dofs": basis.element_dofs, "nodal_dofs": basis.nodal_dofs, "u": u}, mesh


class ClippingTests(unittest.TestCase):
    def check_moments(self, x, cuts):
        q, w = energy.quadrature()
        actual = np.zeros(len(energy.MOMENT_POWERS))
        for a, b in zip(cuts[:-1], cuts[1:]):
            rq, rw = energy.mapped_rule(energy.clip_parent_slab(x, a, b), q, w)
            if len(rw):
                actual += energy.moments(rq, rw)
        expected = np.array([math.prod(math.factorial(n) for n in power)/math.factorial(sum(power)+3)
                             for power in energy.MOMENT_POWERS])
        np.testing.assert_allclose(actual, expected, rtol=5e-12, atol=2e-16)

    def test_all_degree_four_moments_for_one_three_two_two_and_two_planes(self):
        for x, cuts in (([0, 1, 1, 1], [0, .2, .7, 1]), ([0, 0, 1, 1], [0, .3, .8, 1]),
                        ([0, .2, .6, 1], [0, .2, .6, 1]), ([0, 0, 0, 1], [0, .1, .8, 1])):
            with self.subTest(x=x):
                self.check_moments(x, cuts)

    def test_roundoff_near_vertex_and_face_cuts_preserve_real_slivers(self):
        for epsilon in (np.finfo(float).eps, 1e-14, 1e-10):
            for x in ([0, 0, 1, 1], [0, 1, 1, 1], [0, .1, .6, 1]):
                with self.subTest(epsilon=epsilon, x=x):
                    self.check_moments(x, [0, epsilon, 1-epsilon, 1])

    def test_tangent_zero_measure_intersections_are_empty(self):
        self.assertEqual(energy.clip_parent_slab([0, 1, 1, 1], -1, 0).shape, (0, 4, 4))
        self.assertEqual(energy.clip_parent_slab([0, 0, 0, 1], 1, 2).shape, (0, 4, 4))
        with self.assertRaisesRegex(ValueError, "increasing"):
            energy.clip_parent_slab([0, 0, 0, 1], 1, 1)

    def test_hull_failure_never_joggles_or_discards_geometry(self):
        with patch.object(energy, "ConvexHull", side_effect=energy.QhullError("injected")):
            with self.assertRaisesRegex(ValueError, "no perturbation"):
                energy.clip_parent_slab([0, 0, 1, 1], .2, .8)


class FieldEnergyTests(unittest.TestCase):
    cuts = np.array([0, .47e-6, 1.27e-6, 2e-6])
    frequency = 887_500_000

    def test_complex_quadratic_analytic_slab_energies(self):
        amplitude, length = (1+2j)*1e-9, 2e-6
        data, _ = make_field(field=lambda x: np.column_stack((amplitude*(x[:, 0]/length)**2, 0*x[:, 0], 0*x[:, 0])))
        material = CubicMaterial()
        result = energy.integrate_regions(data, material, self.frequency, self.cuts, batch_elements=2)
        area = .7e-6*.3e-6
        for row, a, b in zip(result["regions"], self.cuts[:-1], self.cuts[1:]):
            kinetic = material.rho*(2*np.pi*self.frequency)**2*abs(amplitude)**2*area*(b**5-a**5)/(20*length**4)
            elastic = material.C11*abs(amplitude)**2*area*(b**3-a**3)/(3*length**4)
            np.testing.assert_allclose(row["kinetic_energy_J_per_W"], kinetic, rtol=5e-13)
            np.testing.assert_allclose(row["elastic_energy_J_per_W"], elastic, rtol=5e-13)
            np.testing.assert_allclose(row["volume_m3"], area*(b-a), rtol=5e-13)
        self.assertGreater(result["integration_checks"]["cut_tetrahedra"], 0)
        self.assertEqual(result["totals"]["energy_per_incident_power_s"], result["totals"]["total_energy_J_per_W"])

    def test_total_energy_matches_independent_rotated_stiffness_and_mass(self):
        length = 2e-6
        def field(x):
            a, b, c = (x[:, i]/length for i in range(3))
            return 1e-9*np.column_stack((1+a+a*b+1j*c*c, b*b+2j*a*c, c+a*a-1j*b))
        data, mesh = make_field(field=field)
        theta = .37
        rotation = np.array([[np.cos(theta), -np.sin(theta), 0], [np.sin(theta), np.cos(theta), 0], [0, 0, 1]])
        material = CubicMaterial(rotation=rotation)
        system = assemble_elasticity(mesh, material)
        result = energy.integrate_regions(data, material, self.frequency, self.cuts)
        u = data["u"]
        expected = [np.vdot(u, system.M@u).real*(2*np.pi*self.frequency)**2/4, np.vdot(u, system.K@u).real/4]
        np.testing.assert_allclose([result["totals"]["kinetic_energy_J_per_W"], result["totals"]["elastic_energy_J_per_W"]], expected, rtol=5e-13)

    def test_constant_field_has_analytic_volume_fractions_without_equipartition(self):
        data, _ = make_field(field=lambda x: np.tile([1e-9+2j*1e-9, 0, 0], (len(x), 1)))
        result = energy.integrate_regions(data, CubicMaterial(), self.frequency, self.cuts)
        np.testing.assert_allclose([r["total_energy_fraction"] for r in result["regions"]], np.diff(self.cuts)/2e-6, rtol=1e-13)
        self.assertLess(result["totals"]["elastic_energy_J_per_W"], result["totals"]["kinetic_energy_J_per_W"]*1e-25)

    def test_phase_and_amplitude_preserve_fractions_and_quadratic_scaling(self):
        data, _ = make_field()
        first = energy.integrate_regions(data, CubicMaterial(), self.frequency, self.cuts)
        data["u"] *= 2.3*np.exp(.77j)
        second = energy.integrate_regions(data, CubicMaterial(), self.frequency, self.cuts, batch_elements=1)
        np.testing.assert_allclose([r["total_energy_fraction"] for r in first["regions"]], [r["total_energy_fraction"] for r in second["regions"]], atol=2e-15)
        np.testing.assert_allclose(second["totals"]["total_energy_J_per_W"], first["totals"]["total_energy_J_per_W"]*2.3**2, rtol=3e-14)

    def test_wrong_edge_components_nonfinite_and_degenerate_inputs_fail(self):
        for change in ("edges", "components", "nonfinite", "degenerate"):
            data, _ = make_field()
            if change == "edges":
                data["element_dofs"] = data["element_dofs"].copy()
                data["element_dofs"][[12, 15]] = data["element_dofs"][[15, 12]]
            elif change == "components":
                data["dof_component"][0] = 1
            elif change == "nonfinite":
                data["u"][0] = np.nan
            else:
                data["points_m"] = data["points_m"].copy()
                data["points_m"][:, 2] = 0
                data["dof_coordinates_m"] = data["dof_coordinates_m"].copy()
                data["dof_coordinates_m"][:, 2] = 0
            with self.subTest(change=change), self.assertRaises(ValueError):
                energy.integrate_regions(data, CubicMaterial(), self.frequency, self.cuts)

    def test_tetra10_export_keeps_all_nodes_components_and_native_complex_values(self):
        data, _ = make_field()
        _, points, cells, values = energy.validate_field(data)
        with tempfile.TemporaryDirectory() as td:
            path = Path(td)/"selected_AP_pnc.vtu"
            metadata = energy.write_selected_vtu(path, data)
            result = meshio.read(path)
        self.assertEqual(metadata["scalar_P2_nodes"], len(data["u"])//3)
        self.assertGreater(len(points), len(data["points_m"]))
        self.assertFalse(metadata["visualization_amplitude_rescaled"])
        np.testing.assert_array_equal(result.points, points)
        np.testing.assert_array_equal(result.cells_dict["tetra10"], cells)
        np.testing.assert_array_equal(result.point_data["displacement_real_m_per_sqrt_W"]+1j*result.point_data["displacement_imag_m_per_sqrt_W"], values)


class EvidenceTests(unittest.TestCase):
    def test_selected_archive_requires_exact_root_frequency_complete_P2_and_unit_power(self):
        data, _ = make_field()
        original = {k: v for k, v in data.items() if k != "u"}
        original.update(element_order=2, incident_mode_indices=np.array([1302, 1346, 1292, 1286, 1282]),
                        frequencies_hz=np.full(5, energy.FREQUENCY), incident_powers_w=np.ones(5),
                        displacements_per_sqrt_w=np.column_stack([data["u"]]*5))
        changes = (None, ("incident_mode_indices", np.array([999, 1346, 1292, 1286, 1282])),
                   ("frequencies_hz", np.full(5, 1e9)), ("element_order", 1),
                   ("incident_powers_w", np.array([2., 1, 1, 1, 1])),
                   ("displacements_per_sqrt_w", original["displacements_per_sqrt_w"][:, :4]))
        for change in changes:
            with self.subTest(change=change[0] if change else None), tempfile.TemporaryDirectory() as td:
                path = Path(td)/"fields_AP_pnc.npz"
                np.savez(path, **(original if change is None else {**original, change[0]: change[1]}))
                if change:
                    with self.assertRaises(ValueError):
                        energy.load_selected_field(path, {"ndof": len(data["u"])})
                else:
                    loaded, power = energy.load_selected_field(path, {"ndof": len(data["u"])})
                    self.assertEqual(power, 1.)
                    np.testing.assert_array_equal(loaded["u"], data["u"])

    def fixture(self, root, changed_summary=None):
        def write(name, value):
            path = root/name
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text(json.dumps(value), encoding="utf-8")
            return path, energy.artifact(path, root)
        config = {"geometry": {"length_nm": 16650, "width_nm": 1000, "thickness_nm": 200},
                  "material": {}, "analysis": {"incident_side": "left"}}
        comparison = {"status": "compared_finite_inplane_increment", "frequency_hz": energy.FREQUENCY,
                      "shared_basis": {"status": "identical_saved_basis"}, "inputs": {}}
        fields, replays = {}, {}
        for mesh, ndof, resolution in (("h100", 307611, 100), ("h80", 426627, 80)):
            path = root/mesh/"selected_fields_AP_pnc.npz"
            path.parent.mkdir()
            path.write_bytes(b"small header-only fixture")
            fields[mesh] = energy.sha(path)
            summary = {"ndof": ndof, "frequency_hz": energy.FREQUENCY, "incident_side": "left",
                       "incident_mode_indices": [1302, 1346, 1292, 1286, 1282],
                       "configuration": {**config, "mesh": {"mesh_size_nm": resolution, "thickness_layers": 4}},
                       "checkpointed_scattering": {"phase_adjustment_applied": False, "reference_planes_m": [0., 1.665e-5]},
                       "implementation": {"source_sha256": {"src/phononic/material_AP_pnc.py": energy.sha(energy.ROOT/"src/phononic/material_AP_pnc.py")}},
                       "output_sha256": {path.name: fields[mesh]}}
            if changed_summary and mesh == "h100":
                changed_summary(summary)
            sp, summary_link = write(mesh+"/summary_AP_pnc.json", summary)
            replay = {"status": "point_verified_within_saved_discretization", "problem_dofs": ndof,
                      "frequency_hz": energy.FREQUENCY, "full_p2_field_identity_passed": True,
                      "public_summary_sha256": energy.sha(sp)}
            _, replay_link = write(mesh+"/replay_AP_pnc.json", replay)
            replays[mesh] = replay_link["sha256"]
            comparison["inputs"]["point_replay" if mesh == "h100" else "accepted_point"] = replay_link
            comparison["inputs"][mesh+"_8875_summary"] = summary_link
        cp, comparison_link = write("comparison_AP_pnc.json", comparison)
        return cp, comparison_link["sha256"], fields, replays

    def test_small_metadata_seam_accepts_exact_fields_and_rejects_rehashed_scope_or_material_change(self):
        changes = (None, lambda s: s["incident_mode_indices"].__setitem__(0, 999),
                   lambda s: s["checkpointed_scattering"].__setitem__("phase_adjustment_applied", True),
                   lambda s: s["implementation"]["source_sha256"].__setitem__("src/phononic/material_AP_pnc.py", "0"*64))
        for change in changes:
            with self.subTest(change=change), tempfile.TemporaryDirectory() as td:
                root = Path(td)
                cp, frozen_sha, fields, replays = self.fixture(root, change)
                with patch.object(energy, "COMPARISON_SHA", frozen_sha), patch.object(energy, "FIELD_SHAS", fields), patch.object(energy, "REPLAY_SHAS", replays):
                    if change:
                        with self.assertRaises(ValueError):
                            energy.load_inputs(root, cp)
                    else:
                        inputs, sources = energy.load_inputs(root, cp)
                        self.assertEqual(set(sources), {"h100", "h80"})
                        self.assertEqual(inputs["h100_fields"]["sha256"], fields["h100"])
                        # Even same-size archive corruption is rejected before np.load.
                        (root/"h100/selected_fields_AP_pnc.npz").write_bytes(b"changed header-only input")
                        with self.assertRaisesRegex(ValueError, "SHA"):
                            energy.load_inputs(root, cp)

    def test_small_end_to_end_workflow_writes_native_fields_and_finite_receipt(self):
        data, _ = make_field(length=energy.CUTS[-1])
        material = {"C11_Pa": 128.5e9, "C12_Pa": 48.3e9, "C44_Pa": 66.8e9, "density_kg_m3": 5323, "crystal_rotation": np.eye(3).tolist()}
        summary = {"configuration": {"material": material, "geometry": {"width_nm": 700., "thickness_nm": 300.}}}
        with tempfile.TemporaryDirectory() as td:
            root, output = Path(td), Path(td)/"output"
            inputs = {m+"_fields": {"path": m+"/original.npz"} for m in ("h100", "h80")}
            sources = {m: (summary, root/m/"original.npz") for m in ("h100", "h80")}
            with patch.object(energy, "load_inputs", return_value=(inputs, sources)), patch.object(energy, "load_selected_field", side_effect=lambda *_: ({k: v.copy() for k, v in data.items()}, 1.0)):
                result = energy.run(root, root/"comparison.json", output, batch_elements=2)
            self.assertTrue(result["comparison"]["finite_fraction_target_passed"])
            self.assertEqual(result["comparison"]["max_absolute_fraction_increment"], 0.)
            self.assertEqual(len(list(output.iterdir())), 5)
            self.assertFalse(result["scope"]["new_solve_performed"])
            self.assertEqual(result["numerical_environment"]["scikit-fem"], energy.skfem.__version__)
            for mesh in ("h100", "h80"):
                self.assertTrue((output/(mesh+"_root1302_P2_AP_pnc.vtu")).is_file())
            self.assertEqual(json.loads((output/"field_energy_AP_pnc.json").read_bytes())["status"], result["status"])

    def test_hash_binding_and_workspace_boundary(self):
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            path = root/"input_AP_pnc.json"
            path.write_text("{}", encoding="utf-8")
            link = energy.artifact(path, root)
            self.assertEqual(energy.checked_path(root, link), path)
            with self.assertRaisesRegex(ValueError, "SHA"):
                energy.checked_path(root, link, expected_sha="0"*64)
            path.write_text("[]", encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "SHA"):
                energy.checked_path(root, link)
            with self.assertRaisesRegex(ValueError, "escapes"):
                energy.checked_path(root/"child", {**link, "path": str(path)})

    def test_existing_output_is_never_overwritten(self):
        with tempfile.TemporaryDirectory() as td:
            with self.assertRaisesRegex(ValueError, "already exists"):
                energy.run(Path(td), Path(td)/"comparison.json", Path(td))

    def test_report_retains_units_limits_and_native_field_links(self):
        data, _ = make_field()
        result = energy.integrate_regions(data, CubicMaterial(), 887_500_000, FieldEnergyTests.cuts)
        result["selected_field_vtu"] = {"path": "h100_root1302_P2_AP_pnc.vtu"}
        with tempfile.TemporaryDirectory() as td:
            out = Path(td)
            record = {"workspace": td, "meshes": {"h100": result, "h80": result},
                      "inputs": {m+"_fields": {"path": "original.npz"} for m in ("h100", "h80")},
                      "comparison": {"max_absolute_fraction_increment": 0., "finite_fraction_target_passed": True}}
            energy.write_report(out, record)
            html = (out/"report_AP_pnc.html").read_text(encoding="utf-8")
            self.assertIn("J/W = s", html)
            self.assertIn("no second division", html)
            self.assertIn("not delay or Q", html)
            self.assertIn("full quadratic field", html)
            self.assertIn("no spatial or frequency convergence", html)
            self.assertEqual(len((out/"regional_energy_AP_pnc.csv").read_text().splitlines()), 7)


if __name__ == "__main__":
    unittest.main()
