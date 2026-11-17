"""Analytic mass-metric polarization checks. Athul Prem | _AP_pnc."""

import csv
import json
from pathlib import Path
from tempfile import TemporaryDirectory
from unittest import TestCase

import numpy as np

from tools import diagnose_port_polarization_AP_pnc as diagnosis


def rectangle_fixture():
    """Exact bilinear consistent mass on a W=6, t=2 rectangle, no assembly.

    Linear translations/rotation and u_x=y are represented exactly. The
    non-diagonal mass distinguishes these tests from lumped nodal averaging.
    """
    width, thickness, density = 6.0, 2.0, 7.0
    scalar_coordinates = np.array([[-3., -1.], [-3., 1.], [3., -1.], [3., 1.]])
    coordinates = np.repeat(scalar_coordinates, 3, axis=0)
    components = np.arange(12).reshape(4, 3).T
    interval_mass = np.array([[2., 1.], [1., 2.]]) / 6
    scalar_mass = density * width * thickness * np.kron(interval_mass, interval_mass)
    matrix = np.zeros((12, 12))
    for ids in components:
        matrix[np.ix_(ids, ids)] = scalar_mass
    templates = np.zeros((12, 4))
    for c, ids in enumerate(components):
        templates[ids, c] = 1
    templates[components[1], 3] = -scalar_coordinates[:, 1]
    templates[components[2], 3] = scalar_coordinates[:, 0]
    # Independent continuum integrals: rho*A and rho*A*(W²+t²)/12.
    norms = np.array([84., 84., 84., 280.])
    orthonormal = templates / np.sqrt(norms)
    higher = np.zeros(12)
    higher[components[0]] = scalar_coordinates[:, 0]
    higher /= np.sqrt(252.)  # rho*A*W²/12.
    return coordinates, components, matrix, orthonormal, higher


def evaluate(fields, coordinates=None, components=None, matrix=None):
    c, ids, mass, _, _ = rectangle_fixture()
    return diagnosis.diagnose_modes(coordinates_yz_m=c if coordinates is None else coordinates,
        component_dofs_xyz=ids if components is None else components,
        mass_kg_per_m=mass if matrix is None else matrix, displacements=np.asarray(fields))


class PolarizationMetricTests(TestCase):
    def test_exact_translations_rotation_and_unexplained_content(self):
        _, _, _, rigid, higher = rectangle_fixture()
        result = evaluate(np.column_stack((rigid, higher)))
        np.testing.assert_allclose(result["mass_centroid_yz_m"], [0, 0], atol=1e-15)
        self.assertAlmostEqual(result["mass_per_length_kg_m"], 84.)
        np.testing.assert_allclose(result["normalized_template_gram_real"], np.eye(4), atol=1e-15)
        np.testing.assert_allclose(result["normalized_template_gram_imag"], 0, atol=1e-15)
        expected_fractions = [[1, 0, 0], [0, 1, 0], [0, 0, 1], [0, .1, .9], [1, 0, 0]]
        for i, row in enumerate(result["modes"]):
            np.testing.assert_allclose(row["component_fractions_xyz"], expected_fractions[i], atol=1e-14)
            np.testing.assert_allclose(row["template_scores_tx_ty_tz_rx"], np.eye(5, 4)[i], atol=1e-14)
            self.assertAlmostEqual(row["displacement_mass_norm"], 1.)
            self.assertAlmostEqual(row["template_remainder_fraction"], float(i == 4))
        self.assertEqual(result["modes"][4]["descriptive_largest_template"], "tied")

    def test_known_complex_mixture_has_analytic_partition(self):
        _, _, _, rigid, higher = rectangle_fixture()
        weights = np.array([np.sqrt(.1), 1j * np.sqrt(.2), -np.sqrt(.3), np.sqrt(.15)])
        field = rigid @ weights + 1j * np.sqrt(.25) * higher
        row = evaluate(field[:, None])["modes"][0]
        np.testing.assert_allclose(row["template_scores_tx_ty_tz_rx"], [.1, .2, .3, .15], atol=1e-14)
        np.testing.assert_allclose(row["component_fractions_xyz"], [.35, .215, .435], atol=1e-14)
        self.assertAlmostEqual(row["template_projection_fraction"], .75)
        self.assertAlmostEqual(row["template_remainder_fraction"], .25)
        self.assertEqual(row["descriptive_largest_template"], "Tz")

    def test_phase_scaling_mode_permutation_and_coordinate_origin_invariance(self):
        coordinates, _, _, rigid, higher = rectangle_fixture()
        fields = np.column_stack((rigid[:, 0] + .3j * higher,
                                  rigid[:, 3] - .2 * rigid[:, 2], rigid[:, 1] + .4j * rigid[:, 0]))
        base = evaluate(fields)
        order = [2, 0, 1]
        scales = np.array([2e-7, .5e8, 3.])
        altered = fields[:, order] * (scales * np.exp(1j * np.array([.3, 1.2, -2.4])))
        shifted = evaluate(altered, coordinates=coordinates + [11.3, -21.7])
        np.testing.assert_allclose(shifted["mass_centroid_yz_m"], [11.3, -21.7], atol=1e-14)
        for j, i in enumerate(order):
            for key in ("component_fractions_xyz", "template_scores_tx_ty_tz_rx",
                        "template_projection_fraction", "template_remainder_fraction"):
                np.testing.assert_allclose(shifted["modes"][j][key], base["modes"][i][key], atol=1e-14)
            self.assertAlmostEqual(shifted["modes"][j]["displacement_mass_norm"]
                                   / (base["modes"][i]["displacement_mass_norm"] * scales[j]**2), 1.)

    def test_global_dof_reordering_preserves_component_ordered_values(self):
        coordinates, components, matrix, rigid, _ = rectangle_fixture()
        order = np.array([8, 0, 11, 2, 5, 3, 9, 1, 6, 10, 4, 7])
        inverse = np.argsort(order)
        base = evaluate(rigid)
        candidate = evaluate(rigid[order], coordinates=coordinates[order], components=inverse[components],
                             matrix=matrix[np.ix_(order, order)])
        for a, b in zip(base["modes"], candidate["modes"], strict=True):
            np.testing.assert_allclose(a["template_scores_tx_ty_tz_rx"], b["template_scores_tx_ty_tz_rx"], atol=1e-14)

    def test_inconsistent_component_maps_mass_and_nonphysical_fields_rejected(self):
        coordinates, components, matrix, rigid, _ = rectangle_fixture()
        bad_components = components.copy()
        bad_components[1, [0, 1]] = bad_components[1, [1, 0]]
        bad_mass = matrix.copy()
        bad_mass[0, 1] = bad_mass[1, 0] = 1
        asymmetric = matrix.copy()
        asymmetric[0, 3] += 1
        for kwargs in ({"components": bad_components}, {"matrix": bad_mass}, {"matrix": asymmetric},
                       {"matrix": -matrix}, {"coordinates": coordinates * np.nan}):
            with self.subTest(kwargs=list(kwargs)):
                with self.assertRaises(ValueError):
                    evaluate(rigid, **kwargs)
        for field in (np.zeros((12, 1)), rigid * np.nan):
            with self.assertRaises(ValueError):
                evaluate(field)


class PolarizationProvenanceTests(TestCase):
    def test_changed_input_identity_and_duplicate_role_rejected(self):
        with TemporaryDirectory() as temporary:
            path = Path(temporary) / "input.json"
            path.write_text('{"accepted":true}\n', encoding="utf-8")
            digest = diagnosis.file_sha(path)
            inputs = []
            diagnosis.checked_input(inputs, "proof", path, digest, path.stat().st_size)
            with self.assertRaises(ValueError):
                diagnosis.checked_input(inputs, "proof", path, digest)
            path.write_text('{"accepted":false}\n', encoding="utf-8")
            with self.assertRaises(ValueError):
                diagnosis.checked_input([], "proof", path, digest)

    def test_response_join_checks_signed_root_order_frequency_and_saved_values(self):
        modes = {"roots": np.array([11, 8, 5, 2]), "reverse_roots": np.array([12, 9, 6, 3])}
        summary = {"frequency_hz": 100_000_000, "incident_side": "left", "incident_mode_indices": [11, 8, 5, 2],
                   "num_incident_channels": 4, "power_balance_passed": True}
        with TemporaryDirectory() as temporary:
            path = Path(temporary) / "scattering.csv"
            def write_rows(order=(1, 2, 3, 4), frequency=.1):
                with path.open("w", newline="", encoding="utf-8") as stream:
                    writer = csv.DictWriter(stream, fieldnames=["incident_mode_order", "frequency_GHz", "reflection", "transmission"])
                    writer.writeheader()
                    for i in order:
                        writer.writerow({"incident_mode_order": i, "frequency_GHz": frequency,
                                         "reflection": i / 100, "transmission": 1 - i / 100})
            write_rows()
            result = diagnosis.read_response_rows(path, summary, modes)
            self.assertEqual([row["root"] for row in result], [11, 8, 5, 2])
            self.assertEqual([row["reverse_root"] for row in result], [12, 9, 6, 3])
            comparison = {"status": "compared", "mesh_pair_passed": True, "phase_adjustment_applied": False,
                "spatial_convergence_established": False, "frequency_convergence_established": False,
                "rows": [{"quantity": quantity, "reference_incident_root": row["root"],
                          "candidate_incident_root": row["root"], "reference_power": row[key], "candidate_power": row[key]}
                         for quantity, key in (("reflection", "R"), ("transmission", "T")) for row in result]}
            diagnosis.check_comparison_rows(comparison, result, result)
            comparison["rows"][0]["candidate_incident_root"] = 8
            with self.assertRaises(ValueError):
                diagnosis.check_comparison_rows(comparison, result, result)
            for order, frequency in (((2, 1, 3, 4), .1), ((1, 2, 3, 4), 1.)):
                write_rows(order, frequency)
                with self.assertRaises(ValueError):
                    diagnosis.read_response_rows(path, summary, modes)

    def test_publication_is_new_only_and_keeps_raw_metrics(self):
        _, _, _, rigid, _ = rectangle_fixture()
        row = {**evaluate(rigid[:, :1])["modes"][0], "root": 7, "reverse_root": 8,
               "wavenumber_rad_m": 100., "signed_power_w": 1., "R": .02, "T": .98}
        document = {"responses": {"fixture": {"section": "synthetic", "rows": [row]}}}
        with TemporaryDirectory() as temporary:
            path = diagnosis.publish(document, temporary)
            saved = json.loads(path.read_bytes())
            self.assertEqual(saved["responses"]["fixture"]["rows"][0], row)
            self.assertEqual(saved["output_sha256"][diagnosis.CSV_NAME], diagnosis.file_sha(Path(temporary) / diagnosis.CSV_NAME))
            old = path.read_bytes()
            with self.assertRaises(ValueError):
                diagnosis.publish(document, temporary)
            self.assertEqual(path.read_bytes(), old)
