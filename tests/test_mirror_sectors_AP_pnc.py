"""Small mirror-projector and immutable-role checks. Athul Prem | _AP_pnc."""

import copy
import csv
import hashlib
import json
from pathlib import Path
from tempfile import TemporaryDirectory
from unittest import TestCase
from unittest.mock import patch

import numpy as np
from skfem import Basis, ElementTriP2, ElementVector, MeshTri

from tools import diagnose_mirror_sectors_AP_pnc as mirror


def fixture(diagonal=0, cells=(2, 2)):
    ny, nz = cells
    vertices = np.array([(y, z) for y in np.linspace(-1., 1., ny + 1)
                         for z in np.linspace(-1., 1., nz + 1)])
    triangles = []
    for y in range(ny):
        for z in range(nz):
            a, b = y * (nz + 1) + z, (y + 1) * (nz + 1) + z
            c, d = b + 1, a + 1
            triangles.extend(((a, b, c), (a, c, d)) if diagonal == 0
                             else ((a, b, d), (b, c, d)))
    mesh = MeshTri(vertices.T, np.asarray(triangles).T)
    basis = Basis(mesh, ElementVector(ElementTriP2(), dim=3), intorder=4)
    return {"vertices_yz_m": mesh.p.T, "triangles": mesh.t.T,
            "coordinates_yz_m": basis.doflocs.T, "component_dofs_xyz": np.asarray(basis.split_indices()),
            "shape_cells_yz": cells, "density_kg_m3": 1.}


def templates(data):
    components = data["component_dofs_xyz"]
    values = np.zeros((len(data["coordinates_yz_m"]), 4), complex)
    values[components[0], 0] = 1
    values[components[2], 1] = 1
    values[components[1], 2] = 1
    values[components[1], 3] = -data["coordinates_yz_m"][components[1], 1]
    values[components[2], 3] = data["coordinates_yz_m"][components[2], 0]
    return values


class MirrorSectorsTests(TestCase):
    def test_four_known_full_polar_vector_sectors(self):
        data = fixture()
        result, _ = mirror.measure_sectors(**data, displacements=templates(data))
        np.testing.assert_allclose([row["weights_pp_pm_mp_mm"] for row in result["modes"]], np.eye(4), atol=2e-14)
        self.assertEqual([row["descriptive_largest_sector"] for row in result["modes"]], [list(x) for x in mirror.SECTORS])
        self.assertEqual(result["grid"]["quadrature_points"], 96)
        self.assertAlmostEqual(result["grid"]["area_m2"], 4.)
        for row in result["modes"]:
            np.testing.assert_allclose(row["weights_pp_pm_mp_mm"], row["correlation_weights_pp_pm_mp_mm"], atol=2e-14)

    def test_complex_mixture_gauge_scale_and_no_conjugation(self):
        data = fixture()
        pure = templates(data)
        pure[:, 3] *= np.sqrt(1.5)  # All four templates now have equal mass norm.
        coefficients = np.array([1 + 2j, -3j, 2 - .5j, -1 + 4j])
        field = pure @ coefficients
        expected = abs(coefficients)**2 / np.sum(abs(coefficients)**2)
        result, _ = mirror.measure_sectors(**data, displacements=np.column_stack((field, field * 3e-7 * np.exp(.73j))))
        for row in result["modes"]:
            np.testing.assert_allclose(row["weights_pp_pm_mp_mm"], expected, atol=2e-14)
            self.assertLess(row["maximum_orthogonality_residual"], 1e-14)
        self.assertAlmostEqual(result["modes"][1]["displacement_mass_norm"]
                               / result["modes"][0]["displacement_mass_norm"], 9e-14, delta=1e-26)

    def test_opposite_diagonals_reproduce_global_quadratic(self):
        results = []
        for diagonal in (0, 1):
            data = fixture(diagonal)
            values = templates(data) @ np.array([1j, 2, -.5j, .3 + .1j])
            ids = data["component_dofs_xyz"][0]
            y, z = data["coordinates_yz_m"][ids].T
            values[ids] += (1 - .7j) * y * z
            result, _ = mirror.measure_sectors(**data, displacements=values[:, None])
            results.append(result["modes"][0])
        for key in ("weights_pp_pm_mp_mm", "mirror_correlations_y_z_yz_real", "displacement_mass_norm"):
            np.testing.assert_allclose(results[0][key], results[1][key], rtol=2e-14, atol=2e-14)

    def test_non_global_p2_field_requires_both_diagonals(self):
        data = fixture(0, cells=(1, 1))
        ids = data["component_dofs_xyz"][0]
        y, z = data["coordinates_yz_m"][ids].T
        values = np.zeros((len(data["coordinates_yz_m"]), 1), complex)
        values[ids, 0] = abs(y - z)
        result, _ = mirror.measure_sectors(**data, displacements=values)
        row = result["modes"][0]
        np.testing.assert_allclose(row["mirror_correlations_y_z_yz_real"], [.5, .5, 1], atol=2e-14)
        np.testing.assert_allclose(row["weights_pp_pm_mp_mm"], [.75, 0, 0, .25], atol=2e-14)
        self.assertAlmostEqual(row["displacement_mass_norm"], 8 / 3, places=13)

    def test_malformed_asymmetric_nonuniform_grids_reject(self):
        original = fixture()
        variants = []
        data = copy.deepcopy(original)
        data["vertices_yz_m"][0, 0] += .001
        variants.append(data)
        data = copy.deepcopy(original)
        data["vertices_yz_m"][:, 0] += .01
        variants.append(data)
        data = copy.deepcopy(original)
        data["triangles"][0] = data["triangles"][1]
        variants.append(data)
        data = copy.deepcopy(original)
        data["triangles"] = data["triangles"][:-1]
        variants.append(data)
        data = copy.deepcopy(original)
        data["triangles"][0] = [0, 1, len(data["vertices_yz_m"]) - 1]
        variants.append(data)
        for data in variants:
            with self.subTest(vertices=data["vertices_yz_m"].tolist()):
                with self.assertRaises(ValueError):
                    mirror.measure_sectors(**data, displacements=templates(original))

    def test_roundoff_coordinates_are_retained_and_reported(self):
        data = fixture(cells=(1, 1))
        data["vertices_yz_m"][0, 0] += 2 * np.finfo(float).eps
        before = data["vertices_yz_m"].copy()
        _, _, grid = mirror.validate_grid(data["vertices_yz_m"], data["triangles"], data["shape_cells_yz"])
        np.testing.assert_array_equal(data["vertices_yz_m"], before)
        self.assertGreater(grid["maximum_axis_reflection_mismatch_m"][0], 0)
        self.assertLess(grid["maximum_axis_reflection_mismatch_m"][0], 64 * np.finfo(float).eps * 2)

    def test_component_order_and_zero_field_reject(self):
        data = fixture()
        values = templates(data)
        changed = copy.deepcopy(data)
        changed["component_dofs_xyz"] = changed["component_dofs_xyz"][[1, 0, 2]]
        for args, field in ((changed, values), (data, np.zeros_like(values))):
            with self.assertRaises(ValueError):
                mirror.measure_sectors(**args, displacements=field)

    def test_exact_roles_and_actual_signed_participants_reject_aliases(self):
        mirror.require_participants("z4", mirror.PARTICIPANTS["z4"], mirror.SIGNED_ROLES)
        for roots, roles in ((mirror.PARTICIPANTS["z6"], mirror.SIGNED_ROLES),
                             (mirror.PARTICIPANTS["z4"][::-1], mirror.SIGNED_ROLES),
                             (mirror.PARTICIPANTS["z4"], mirror.SIGNED_ROLES[::-1])):
            with self.assertRaises(ValueError):
                mirror.require_participants("z4", roots, roles)
        with TemporaryDirectory() as directory:
            root = Path(directory)
            (root / "a").write_text("accepted", encoding="utf-8")
            (root / "b").write_text("accepted", encoding="utf-8")
            digest = hashlib.sha256(b"accepted").hexdigest()
            with patch.object(mirror, "ROLE_PINS", {"section": ("a", digest), "receipt": ("b", "0" * 64)}):
                inputs = []
                self.assertEqual(mirror.bind_input(root, "section", inputs), (root / "a").resolve())
                with self.assertRaises(ValueError):
                    mirror.bind_input(root, "section", inputs)
                with self.assertRaises(ValueError):
                    mirror.bind_input(root, "receipt", [])

    def test_profiles_allow_zero_and_preserve_complex_gauge(self):
        data = fixture()
        scalar = mirror.section_basis(data["vertices_yz_m"], data["triangles"],
                                      data["coordinates_yz_m"], data["component_dofs_xyz"])
        # Profile samples use physical nanometre-scale coordinates; a constant
        # field in this larger synthetic domain has the same exact sample.
        values = np.zeros((len(data["coordinates_yz_m"]), 2), complex)
        values[data["component_dofs_xyz"][2], 1] = 2 - 3j
        profile = mirror.forward_profiles(scalar, values, data["component_dofs_xyz"], (11, 12))
        np.testing.assert_allclose(profile["modes"][0]["uz_real"], 0, atol=1e-14)
        np.testing.assert_allclose(profile["modes"][1]["uz_real"], 2, atol=1e-14)
        np.testing.assert_allclose(profile["modes"][1]["uz_imag"], -3, atol=1e-14)
        self.assertFalse(profile["phase_rotation_applied"])
        self.assertEqual(len(profile["coordinates_y_m"]), 129)

    def test_fresh_publication_has_eight_csv_rows_and_preserves_existing(self):
        result, _ = mirror.measure_sectors(**fixture(), displacements=templates(fixture()))
        document = {"sections": {}}
        for name in ("z4", "z6"):
            value = copy.deepcopy(result)
            for i, row in enumerate(value["modes"]):
                row.update(root=mirror.PARTICIPANTS[name][i], signed_role=mirror.SIGNED_ROLES[i],
                           wavenumber_rad_m=float(i + 1), signed_power_w=1 if i < 2 else -1)
            document["sections"][name] = value
        with TemporaryDirectory() as directory:
            output = Path(directory) / "fresh"
            target = mirror.publish(document, output)
            saved = json.loads(target.read_text(encoding="utf-8"))
            self.assertEqual(saved["output_sha256"][mirror.CSV_NAME], mirror.file_sha(output / mirror.CSV_NAME))
            with (output / mirror.CSV_NAME).open(newline="", encoding="utf-8") as stream:
                self.assertEqual(len(list(csv.DictReader(stream))), 8)
            before = target.read_bytes()
            with self.assertRaises(ValueError):
                mirror.publish(document, output)
            self.assertEqual(target.read_bytes(), before)
