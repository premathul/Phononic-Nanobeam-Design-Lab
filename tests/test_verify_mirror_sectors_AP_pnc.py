"""Independent mirror-content acceptance checks. Athul Prem | _AP_pnc."""

from copy import deepcopy
import json
from pathlib import Path
from tempfile import TemporaryDirectory
import unittest
from unittest.mock import patch

import numpy as np

from tools import verify_mirror_sectors_AP_pnc as verify


def fixture(diagonal=0, function=None):
    vertices = np.array([[-5e-7, -1e-7], [5e-7, -1e-7], [5e-7, 1e-7], [-5e-7, 1e-7]])
    triangles = np.array([[0, 1, 2], [0, 2, 3]]) if diagonal == 0 else np.array([[0, 1, 3], [1, 2, 3]])
    edges = sorted({tuple(sorted((a, b))) for tri in triangles for a, b in ((tri[0], tri[1]), (tri[1], tri[2]), (tri[0], tri[2]))})
    nodes = np.vstack((vertices, [(vertices[a] + vertices[b]) / 2 for a, b in edges]))
    components = np.arange(3 * len(nodes)).reshape(-1, 3).T
    fields = np.zeros((3 * len(nodes), 4), dtype=complex)
    if function is None:
        for c in range(3):
            fields[components[c], c] = 1
        fields[components[1], 3] = -nodes[:, 1]
        fields[components[2], 3] = nodes[:, 0]
    else:
        sampled = np.asarray(function(nodes))
        fields = np.empty((3 * len(nodes), sampled.shape[2]), dtype=complex)
        for c in range(3):
            fields[components[c]] = sampled[:, c]
    return dict(vertices_yz_m=vertices, triangles=triangles,
                coordinates_yz_m=np.repeat(nodes, 3, axis=0), component_dofs_xyz=components,
                displacements=fields, density_kg_m3=5323.)


class IndependentMirrorTests(unittest.TestCase):
    def test_rigid_templates_have_polar_vector_sectors(self):
        grid, rows, _ = verify.integrate_mirrors(**fixture())
        self.assertEqual(grid["refinement_triangles"], 4)
        self.assertEqual(grid["quadrature_points"], 36)
        expected = np.eye(4)[[0, 2, 1, 3]]
        np.testing.assert_allclose([r["weights_pp_pm_mp_mm"] for r in rows], expected, atol=2e-14)
        # A scalar reflection would incorrectly put Ty and Tz in the ++ sector.
        self.assertEqual(rows[1]["descriptive_largest_sector"], [-1, 1])
        self.assertEqual(rows[2]["descriptive_largest_sector"], [1, -1])

    def test_complex_mixture_is_linear_without_conjugation(self):
        args = fixture()
        u = args["displacements"][:, :3] @ np.array([1, 2j, 3 * np.exp(.3j)])
        args["displacements"] = u[:, None]
        _, rows, _ = verify.integrate_mirrors(**args)
        np.testing.assert_allclose(rows[0]["weights_pp_pm_mp_mm"], np.array([1, 9, 4, 0]) / 14, atol=2e-14)
        args["displacements"] *= 3 * np.exp(1.7j)
        _, rotated, _ = verify.integrate_mirrors(**args)
        for key in ("weights_pp_pm_mp_mm", "mirror_correlations_y_z_yz_real", "mirror_correlations_y_z_yz_imag"):
            np.testing.assert_allclose(rows[0][key], rotated[0][key], atol=2e-14)

    def test_non_global_p2_across_flipped_diagonal_has_known_integrals(self):
        def field(p):
            value = np.zeros((len(p), 3, 1))
            value[:, 0, 0] = abs(p[:, 0] / 5e-7 - p[:, 1] / 1e-7)
            return value
        _, rows, section = verify.integrate_mirrors(**fixture(function=field))
        np.testing.assert_allclose(rows[0]["mirror_correlations_y_z_yz_real"], [.5, .5, 1], atol=2e-14)
        np.testing.assert_allclose(rows[0]["weights_pp_pm_mp_mm"], [.75, 0, 0, .25], atol=2e-14)
        p = np.array([[-2e-7, 7e-8], [4e-7, -5e-8], [0., 0.]])
        np.testing.assert_allclose(verify.evaluate_p2(section, p), field(p), atol=2e-14)
        # The opposite diagonal changes the piecewise shape but not this result.
        def reflected_field(p):
            value = np.zeros((len(p), 3, 1))
            value[:, 0, 0] = abs(p[:, 0] / 5e-7 + p[:, 1] / 1e-7)
            return value
        _, opposite, _ = verify.integrate_mirrors(**fixture(1, reflected_field))
        np.testing.assert_allclose(rows[0]["weights_pp_pm_mp_mm"], opposite[0]["weights_pp_pm_mp_mm"], atol=2e-14)

    def test_rejects_noncentered_incomplete_and_noncartesian_geometry(self):
        for mutation in ("center", "sub_geometry_gate_asymmetry", "missing", "duplicate", "interior_vertex"):
            args = fixture()
            if mutation == "center":
                args["vertices_yz_m"][:, 0] += 1e-10
            elif mutation == "sub_geometry_gate_asymmetry":
                args["vertices_yz_m"][:, 0] += 1e-19
            elif mutation == "missing":
                args["triangles"] = args["triangles"][:1]
            elif mutation == "duplicate":
                args["triangles"][1] = args["triangles"][0]
            else:
                args["vertices_yz_m"][0, 1] += 1e-9
            with self.subTest(mutation=mutation), self.assertRaises(ValueError):
                verify.integrate_mirrors(**args)

    def test_rejects_corrupt_component_map_zero_and_nonfinite_fields(self):
        for mutation in ("component", "zero", "nan"):
            args = fixture()
            if mutation == "component":
                args["component_dofs_xyz"][0, 0] = args["component_dofs_xyz"][0, 1]
            elif mutation == "zero":
                args["displacements"][:, 0] = 0
            else:
                args["displacements"][0, 0] = np.nan
            with self.subTest(mutation=mutation), self.assertRaises(ValueError):
                verify.integrate_mirrors(**args)

    def test_roundoff_grid_line_deviations_preserve_raw_coordinates(self):
        args = fixture()
        vertices = args["vertices_yz_m"]
        # Two values on the same intended y-grid line remain distinct in raw
        # coordinates. The independent topology grouping must not snap them.
        delta = np.spacing(5e-7)
        vertices[0, 0] += delta
        vertices[3, 0] -= delta
        original = vertices.copy()
        edges = sorted({tuple(sorted((a, b))) for tri in args["triangles"]
                        for a, b in ((tri[0], tri[1]), (tri[1], tri[2]), (tri[0], tri[2]))})
        nodes = np.vstack((vertices, [(vertices[a] + vertices[b]) / 2 for a, b in edges]))
        args["coordinates_yz_m"] = np.repeat(nodes, 3, axis=0)
        args["displacements"][args["component_dofs_xyz"][1], 3] = -nodes[:, 1]
        args["displacements"][args["component_dofs_xyz"][2], 3] = nodes[:, 0]
        grid, rows, section = verify.integrate_mirrors(**args)
        self.assertGreater(grid["maximum_grid_coordinate_residual_m"], 0)
        self.assertGreater(grid["maximum_axis_reflection_mismatch_m"][0], 0)
        np.testing.assert_array_equal(section["vertices"], original)
        np.testing.assert_allclose([r["weights_pp_pm_mp_mm"] for r in rows], np.eye(4)[[0, 2, 1, 3]], atol=2e-14)

    def test_projector_publication_rejects_tampered_weights_and_correlations(self):
        _, rows, _ = verify.integrate_mirrors(**fixture())
        for key in ("weights_pp_pm_mp_mm", "mirror_correlations_y_z_yz_real", "normalized_projector_gram_real"):
            bad = deepcopy(rows[0])
            if key == "normalized_projector_gram_real":
                bad[key][0][1] += .1
            else:
                bad[key][0] += .1
            with self.subTest(key=key), self.assertRaises(ValueError):
                verify.compare_mode(bad, rows[0], "synthetic root")

    def test_raw_midline_samples_reject_view_gauge_and_coordinate_changes(self):
        def field(p):
            values = np.zeros((len(p), 3, 2), dtype=complex)
            values[:, 2, 0] = 1 + 2j * p[:, 0] / 5e-7
            values[:, 2, 1] = (p[:, 0] / 5e-7)**2 + .4j
            return values
        _, _, section = verify.integrate_mirrors(**fixture(function=field))
        y = np.linspace(-5e-7, 5e-7, 129)
        values = field(np.column_stack((y, np.zeros(129))))[:, 2]
        profile = dict(coordinates_y_m=y.tolist(), z_m=0, units="m/sqrt(W)", normalization_applied=False,
                       phase_rotation_applied=False,
                       modes=[dict(root=root, uz_real=values[:, i].real.tolist(), uz_imag=values[:, i].imag.tolist())
                              for i, root in enumerate((11, 13))])
        self.assertLess(verify.verify_profiles(profile, section, [11, 13]), 1e-13)
        bad = deepcopy(profile)
        bad["modes"][0]["uz_real"][23] += .01
        with self.assertRaisesRegex(ValueError, "raw complex"):
            verify.verify_profiles(bad, section, [11, 13])
        bad = deepcopy(profile)
        bad["coordinates_y_m"][20] += 1e-10
        with self.assertRaisesRegex(ValueError, "coordinates"):
            verify.verify_profiles(bad, section, [11, 13])

    def test_zero_midline_is_valid_with_full_field_rms_normalization(self):
        def field(p):
            values = np.zeros((len(p), 3, 2), dtype=complex)
            values[:, 2, 0] = p[:, 1] / 1e-7
            values[:, 2, 1] = 1j * p[:, 1] / 1e-7
            return values
        _, _, section = verify.integrate_mirrors(**fixture(function=field))
        profile = dict(coordinates_y_m=np.linspace(-5e-7, 5e-7, 129).tolist(), z_m=0,
                       units="m/sqrt(W)", normalization_applied=False, phase_rotation_applied=False,
                       modes=[dict(root=root, uz_real=[0.] * 129, uz_imag=[0.] * 129) for root in (11, 13)])
        self.assertLess(verify.verify_profiles(profile, section, [11, 13]), 1e-13)

    def test_exact_role_binding_rejects_swapped_and_duplicate_artifacts(self):
        with TemporaryDirectory() as temporary:
            root = Path(temporary).resolve()
            for name in ("one.json", "two.json"):
                (root / name).write_text(json.dumps({"name": name}), encoding="utf-8")
            pins = {name: (filename, verify.sha256(root / filename))
                    for name, filename in (("one_role", "one.json"), ("two_role", "two.json"))}
            document = {"inputs": [dict(role=role, path=name, sha256=digest, bytes=(root / name).stat().st_size)
                                   for role, (name, digest) in pins.items()]}
            with patch.dict(verify.ROLE_PINS, pins, clear=True):
                self.assertEqual(set(verify.verified_inputs(document, root)), set(pins))
                bad = deepcopy(document)
                bad["inputs"][0].update({key: document["inputs"][1][key] for key in ("path", "sha256", "bytes")})
                with self.assertRaisesRegex(ValueError, "role identity"):
                    verify.verified_inputs(bad, root)
                bad = deepcopy(document)
                bad["inputs"][1] = bad["inputs"][0]
                with self.assertRaisesRegex(ValueError, "duplicate"):
                    verify.verified_inputs(bad, root)


if __name__ == "__main__":
    unittest.main()
