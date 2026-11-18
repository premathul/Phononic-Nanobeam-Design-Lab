"""Portable independent P2/energy/cut verification. Athul Prem | _AP_pnc."""

from copy import deepcopy
from fractions import Fraction
from itertools import permutations
from math import comb, factorial
from pathlib import Path
from tempfile import TemporaryDirectory
import unittest

import meshio
import numpy as np
from scipy.sparse import diags, save_npz

from tools import verify_saved_field_energy_AP_pnc as verify


MATERIAL = {"C11_Pa": 5., "C12_Pa": 2., "C44_Pa": 1.5,
            "density_kg_m3": 3., "crystal_rotation": np.eye(3).tolist()}


def fixture(vertices=None, function=None):
    vertices = np.array([[0., 0., 0.], [1., 0., 0.], [0., 1., 0.], [0., 0., 1.]]) if vertices is None else np.asarray(vertices)
    nodes = np.vstack((vertices, [(vertices[a]+vertices[b])/2 for a, b in verify.EDGES]))
    if function is None:
        function = lambda p: np.column_stack(((1+1j)*p[:, 0]**2, (2-1j)*p[:, 1]**2, (.5+2j)*p[:, 2]**2))
    values = function(nodes).reshape(-1)
    return {"points_m": vertices, "tetrahedra": np.arange(4).reshape(1, 4),
            "dof_coordinates_m": np.repeat(nodes, 3, axis=0), "dof_component": np.tile(np.arange(3), len(nodes)),
            "element_dofs": np.arange(30).reshape(30, 1), "element_order": np.array(2),
            "displacements_per_sqrt_w": values[:, None]}


def simplex_slab_moment(a, b, c, lower, upper):
    """Analytic x^a y^b z^c integral; no geometry or quadrature implementation."""
    exponent = b+c+2
    lower, upper = Fraction(str(lower)), Fraction(str(upper))
    return float(Fraction(factorial(b)*factorial(c), factorial(exponent)) * sum(
        (-1)**k*comb(exponent, k)*(upper**(a+k+1)-lower**(a+k+1))/(a+k+1)
        for k in range(exponent+1)))


def clipped_moment(vertices, lower, upper, powers):
    answer = 0.
    determinant = abs(np.linalg.det((vertices[1:]-vertices[0]).T))
    for piece in verify.clip_slab(vertices[:, 0], lower, upper):
        local_determinant = abs(np.linalg.det(piece[1:, 1:]-piece[0, 1:]))
        points = verify.QUADRATURE_BARY @ piece @ vertices
        answer += determinant*local_determinant*np.dot(verify.QUADRATURE_WEIGHTS,
                                                      np.prod(points**np.asarray(powers), axis=1))
    return answer


def publication(independent, cuts):
    totals = independent["totals"]
    return {"regions": [{"name": verify.NAMES[i], "x_interval_m": list(cuts[i:i+2]),
                         "volume_m3": row[0], "kinetic_energy_J_per_W": row[1], "elastic_energy_J_per_W": row[2],
                         "total_energy_J_per_W": row[1]+row[2], "total_energy_fraction": independent["fractions"][i]}
                        for i, row in enumerate(independent["regions"])],
            "totals": {"volume_m3": totals[0], "kinetic_energy_J_per_W": totals[1], "elastic_energy_J_per_W": totals[2],
                       "total_energy_J_per_W": totals[1]+totals[2], "energy_per_incident_power_s": totals[1]+totals[2]}}


class IndependentSavedFieldEnergyTests(unittest.TestCase):
    def test_duffy_rule_all_degree_four_barycentric_monomials(self):
        bary, weights = verify.duffy_rule()
        self.assertEqual(len(weights), 64)
        self.assertTrue(np.all(weights > 0))
        for a in range(5):
            for b in range(5-a):
                for c in range(5-a-b):
                    for d in range(5-a-b-c):
                        powers = [a, b, c, d]
                        exact = np.prod([factorial(i) for i in powers])/factorial(sum(powers)+3)
                        self.assertAlmostEqual(np.dot(weights, np.prod(bary**powers, axis=1))/exact, 1., places=13)

    def test_case_split_matches_analytic_degree_four_slab_moments(self):
        vertices = fixture()["points_m"]
        # Permutations alter 1/2/3-inside indexing and orientation without altering the domain.
        for order in permutations(range(4)):
            for lower, upper in ((0., .2), (.2, .7), (.7, 1.), (0., 1.)):
                for a in range(5):
                    for b in range(5-a):
                        for c in range(5-a-b):
                            actual = clipped_moment(vertices[list(order)], lower, upper, (a, b, c))
                            expected = simplex_slab_moment(a, b, c, lower, upper)
                            self.assertAlmostEqual(actual/expected, 1., delta=3e-12)

    def test_two_inside_splits_and_exact_face_edge_vertex_cuts(self):
        # Rotated unit-simplex x coordinate is its old x+y coordinate. A middle
        # plane intersects four edges, exercising the two-inside prism case.
        matrix = np.array([[1., 1., 0.], [0., 1., 0.], [0., 0., 1.]])
        vertices = fixture()["points_m"] @ matrix.T
        for order in permutations(range(4)):
            points = vertices[list(order)]
            whole = clipped_moment(points, -1., 2., (0, 0, 0))
            for cut in (0., 1e-10, .25, .5, .9, 1.-1e-10, 1.):
                below = clipped_moment(points, -1., cut, (0, 0, 0))
                above = clipped_moment(points, cut, 2., (0, 0, 0))
                # Barycentric marginal for x+y is Beta(2,2): CDF=3x^2-2x^3.
                exact_below = (3*cut**2-2*cut**3)/6
                self.assertAlmostEqual(below, exact_below, delta=5e-15)
                self.assertAlmostEqual(below+above, whole, delta=5e-15)
        for lower, upper in ((0., 1.), (-1., 0.), (1., 2.)):
            actual = clipped_moment(vertices, lower, upper, (0, 0, 0))
            self.assertAlmostEqual(actual, 1/6 if lower == 0 else 0., delta=5e-15)

    def test_complex_quadratic_kinetic_and_elastic_energies_are_analytic(self):
        data, cuts, frequency = fixture(), (0., .2, .7, 1.), 1/np.pi
        measured = verify.integrate_regions(data, MATERIAL, frequency, cuts)
        for row, lower, upper in zip(measured["regions"], cuts[:-1], cuts[1:]):
            moment = lambda a, b, c: simplex_slab_moment(a, b, c, lower, upper)
            kinetic = 3*(2*moment(4, 0, 0)+5*moment(0, 4, 0)+4.25*moment(0, 0, 4))
            elastic = 5*(2*moment(2, 0, 0)+5*moment(0, 2, 0)+4.25*moment(0, 0, 2))
            elastic += 4*(moment(1, 1, 0)+2.5*moment(1, 0, 1)-moment(0, 1, 1))
            np.testing.assert_allclose(row, [moment(0, 0, 0), kinetic, elastic], rtol=2e-12, atol=0)
        self.assertAlmostEqual(sum(measured["fractions"]), 1., places=14)
        self.assertNotAlmostEqual(measured["totals"][1], measured["totals"][2], places=3)

    def test_rigid_complex_translation_has_zero_elastic_energy_without_floor(self):
        data = fixture(function=lambda p: np.tile([1+2j, 3-1j, .5j], (len(p), 1)))
        measured = verify.integrate_regions(data, MATERIAL, 1/np.pi, (0., .3, .6, 1.))
        np.testing.assert_allclose(measured["regions"][:, 1], 3*(5+10+.25)*measured["regions"][:, 0], rtol=1e-13)
        self.assertLess(measured["totals"][2], 1e-28)
        verify.relative_close(0., 0., "zero")
        with self.assertRaisesRegex(ValueError, "differs"):
            verify.relative_close(1e-200, 0., "zero")

    def test_scaled_physical_units_and_complex_gauge_preserve_energy_fractions(self):
        original = fixture()
        cuts = (0., .2, .7, 1.)
        first = verify.integrate_regions(original, MATERIAL, 1/np.pi, cuts)
        scaled = deepcopy(original)
        scale, phase = 1e-6, np.exp(.71j)
        scaled["points_m"] *= scale
        scaled["dof_coordinates_m"] *= scale
        scaled["displacements_per_sqrt_w"] *= phase
        second = verify.integrate_regions(scaled, MATERIAL, 1/np.pi, np.asarray(cuts)*scale)
        np.testing.assert_allclose(second["regions"][:, 0], first["regions"][:, 0]*scale**3, rtol=1e-13)
        np.testing.assert_allclose(second["regions"][:, 1], first["regions"][:, 1]*scale**3, rtol=1e-13)
        np.testing.assert_allclose(second["regions"][:, 2], first["regions"][:, 2]*scale, rtol=1e-13)

    def test_region_publication_rejects_redistribution_despite_identical_global_totals(self):
        cuts = (0., .2, .7, 1.)
        independent = verify.integrate_regions(fixture(), MATERIAL, 1/np.pi, cuts)
        saved = publication(independent, cuts)
        verify.check_region_publication(saved, independent, cuts)
        changed = deepcopy(saved)
        changed["regions"][0]["kinetic_energy_J_per_W"] += .01
        changed["regions"][1]["kinetic_energy_J_per_W"] -= .01
        with self.assertRaisesRegex(ValueError, "regional kinetic energy"):
            verify.check_region_publication(changed, independent, cuts)
        changed = deepcopy(saved)
        changed["totals"]["energy_per_incident_power_s"] *= .5
        with self.assertRaisesRegex(ValueError, "power-normalized"):
            verify.check_region_publication(changed, independent, cuts)

    def test_tetra10_readback_requires_all_complex_edge_values_and_native_gauge(self):
        data = fixture()
        points = data["dof_coordinates_m"][::3]
        values = data["displacements_per_sqrt_w"][:, 0].reshape(-1, 3)
        cells = np.arange(10).reshape(1, 10)
        with TemporaryDirectory() as temporary:
            path = Path(temporary)/"selected.vtu"
            def write(vector=values, topology=cells):
                meshio.write_points_cells(path, points, [("tetra10", topology)],
                    point_data={"displacement_real_m_per_sqrt_W": vector.real,
                                "displacement_imag_m_per_sqrt_W": vector.imag})
            write()
            self.assertTrue(verify.check_tetra10(path, data)["exact_complex_field_readback"])
            for changed in (values*2, values*np.exp(.1j)):
                write(changed)
                with self.assertRaisesRegex(ValueError, "displacement"):
                    verify.check_tetra10(path, data)
            changed = values.copy()
            changed[4, 0] += .1j
            write(changed)
            with self.assertRaisesRegex(ValueError, "displacement"):
                verify.check_tetra10(path, data)
            reordered = cells.copy()
            reordered[:, [4, 5]] = reordered[:, [5, 4]]
            write(topology=reordered)
            with self.assertRaisesRegex(ValueError, "connectivity"):
                verify.check_tetra10(path, data)

    def test_saved_matrix_quadratic_energy_uses_complex_field_and_quarter_factors(self):
        data, frequency = fixture(), 1/np.pi
        vector = data["displacements_per_sqrt_w"][:, 0]
        kdiag, mdiag = np.arange(1., 31.), np.arange(31., 61.)
        with TemporaryDirectory() as temporary:
            paths = {key: Path(temporary)/(key+".npz") for key in ("K", "M")}
            save_npz(paths["K"], diags(kdiag))
            save_npz(paths["M"], diags(mdiag))
            result = verify.matrix_energies(paths, data, frequency)
            self.assertAlmostEqual(result["K"], sum(kdiag*abs(vector)**2)/4)
            self.assertAlmostEqual(result["M"], sum(mdiag*abs(vector)**2))

    def test_bad_layout_and_material_are_rejected(self):
        data = fixture()
        changed = deepcopy(data)
        changed["element_dofs"][[12, 15]] = changed["element_dofs"][[15, 12]]
        with self.assertRaisesRegex(ValueError, "coordinates or edge order"):
            verify.validate_layout(changed)
        changed = deepcopy(data)
        changed["displacements_per_sqrt_w"][0, 0] = np.nan
        with self.assertRaisesRegex(ValueError, "Nonfinite"):
            verify.validate_layout(changed)
        material = deepcopy(MATERIAL)
        material["crystal_rotation"][0][0] = -1
        with self.assertRaisesRegex(ValueError, "rotation"):
            verify.material_tensor(material)

    def test_header_binds_external_source_and_rejects_scope_or_claim_changes(self):
        record = {**verify.AUTHOR, "schema_version": 1, "status": "saved_field_energy_compared_finite_increment",
                  "producer": {"sha256": "a"*64},
                  "scope": {"frequency_hz": verify.FREQUENCY, "incident_root": verify.ROOT, "incident_column": 0,
                            "region_cuts_m": list(verify.CUTS), "native_incident_gauge_preserved": True,
                            **dict.fromkeys(("new_solve_performed", "new_assembly_performed", "field_fitting_performed",
                                             "cross_mesh_interpolation_performed", "spatial_convergence_established",
                                             "frequency_convergence_established"), False)},
                  "predeclared_target": {"max_absolute_fraction_increment": .02, "comparison": "strict_less_than"}}
        verify.validate_header(record, "a"*64)
        with self.assertRaisesRegex(ValueError, "externally"):
            verify.validate_header(record, "b"*64)
        for key, value in (("incident_root", 1282), ("native_incident_gauge_preserved", False),
                           ("field_fitting_performed", True), ("spatial_convergence_established", True)):
            changed = deepcopy(record)
            changed["scope"][key] = value
            with self.assertRaises(ValueError):
                verify.validate_header(changed, "a"*64)

    def test_producer_geometry_proof_rejects_centroids_perturbation_or_failed_moments(self):
        proof = {"quadrature_degree": 4, "original_parent_P2_evaluated": True,
                 "centroid_region_assignment": False, "geometry_perturbation_applied": False,
                 "degree_four_monomials_per_cut_tetrahedron": 35,
                 "maximum_cut_moment_relative_closure": 1e-14,
                 "analytic_rectangular_lead_relative_volume_errors": [2e-15, 3e-15],
                 "kinetic_elastic_absolute_partition_difference_over_total_energy": [1e-14, 1e-14]}
        verify.check_integration_proof(proof)
        for key, value in (("centroid_region_assignment", True), ("geometry_perturbation_applied", True),
                           ("maximum_cut_moment_relative_closure", 2e-10),
                           ("analytic_rectangular_lead_relative_volume_errors", [np.nan, 0.])):
            changed = deepcopy(proof)
            changed[key] = value
            with self.assertRaises(ValueError):
                verify.check_integration_proof(changed)


if __name__ == "__main__":
    unittest.main()
