"""Explicit-quadrature polarization checks. Athul Prem | _AP_pnc."""

from copy import deepcopy
import csv
from math import factorial
import json
from pathlib import Path
from tempfile import TemporaryDirectory
import unittest
from unittest.mock import patch

import numpy as np

from tools import verify_port_polarization_AP_pnc as verify


def rectangle_fixture(shift=(0.0, 0.0)):
    """Two triangles and their independent vertex/midpoint P2 layout."""
    width, thickness = 1e-6, 2e-7
    vertices = np.asarray([[0, 0], [width, 0], [width, thickness], [0, thickness]])
    cells = np.asarray([[0, 1, 2], [0, 2, 3]])
    scalar_nodes = vertices.tolist()
    for a, b in ((0, 1), (1, 2), (0, 2), (2, 3), (0, 3)):
        scalar_nodes.append(((vertices[a] + vertices[b]) / 2).tolist())
    scalar_nodes = np.asarray(scalar_nodes)
    components = np.arange(3 * len(scalar_nodes)).reshape(-1, 3).T
    values = np.zeros((3 * len(scalar_nodes), 6), dtype=complex)
    for component in range(3):
        values[components[component], component] = 1
    values[components[1], 3] = -(scalar_nodes[:, 1] - thickness / 2)
    values[components[2], 3] = scalar_nodes[:, 0] - width / 2
    values[:, 4] = values[:, 0] + 2j * values[:, 1]
    values[components[0], 5] = scalar_nodes[:, 0] - width / 2
    return {"vertices_yz_m": vertices + shift, "triangles": cells,
            "coordinates_yz_m": np.repeat(scalar_nodes + shift, 3, axis=0),
            "component_dofs_xyz": components, "displacements": values,
            "density_kg_m3": 5323.0}


class IndependentPortPolarizationTests(unittest.TestCase):
    def test_duffy_rule_integrates_all_degree_four_monomials(self):
        barycentric, weights, _ = verify.triangle_quadrature()
        for a in range(5):
            for b in range(5 - a):
                for c in range(5 - a - b):
                    actual = np.sum(weights * barycentric[:, 0]**a * barycentric[:, 1]**b
                                    * barycentric[:, 2]**c)
                    exact = factorial(a) * factorial(b) * factorial(c) / factorial(a + b + c + 2)
                    self.assertAlmostEqual(actual / exact, 1, places=14)

    def test_translations_rotation_mixed_and_unprojected_field(self):
        result = verify.integrate_section(**rectangle_fixture())
        self.assertAlmostEqual(result["area_m2"] / 2e-13, 1, places=14)
        self.assertAlmostEqual(result["mass_per_length_kg_m"] / 1.0646e-9, 1, places=14)
        np.testing.assert_allclose(result["mass_centroid_yz_m"], [5e-7, 1e-7], rtol=1e-14, atol=0)
        np.testing.assert_allclose(result["normalized_template_gram_real"], np.eye(4), rtol=0, atol=1e-14)
        for index in range(3):
            mode = result["modes"][index]
            np.testing.assert_allclose(mode["component_fractions_xyz"], np.eye(3)[index], atol=1e-14)
            np.testing.assert_allclose(mode["template_scores_tx_ty_tz_rx"], np.eye(4)[index], atol=1e-14)
            self.assertAlmostEqual(mode["displacement_mass_norm"] / 1.0646e-9, 1, places=14)
        rotation = result["modes"][3]
        np.testing.assert_allclose(rotation["component_fractions_xyz"], [0, 1 / 26, 25 / 26], atol=1e-14)
        np.testing.assert_allclose(rotation["template_scores_tx_ty_tz_rx"], [0, 0, 0, 1], atol=1e-14)
        exact_rotation_mass = 1.0646e-9 * ((1e-6)**2 + (2e-7)**2) / 12
        self.assertAlmostEqual(rotation["displacement_mass_norm"] / exact_rotation_mass, 1, places=14)
        np.testing.assert_allclose(result["modes"][4]["template_scores_tx_ty_tz_rx"], [.2, .8, 0, 0], atol=1e-14)
        self.assertAlmostEqual(result["modes"][4]["template_remainder_fraction"], 0, places=14)
        self.assertAlmostEqual(result["modes"][5]["template_remainder_fraction"], 1, places=14)

    def test_gauge_scaling_and_origin_do_not_change_scores(self):
        original = rectangle_fixture()
        expected = verify.integrate_section(**original)
        shifted = rectangle_fixture(shift=(7e-6, -9e-6))
        factors = np.asarray([2j, -3, .2 + .7j, -17j, -2 + 3j, .01j])
        shifted["displacements"] *= factors
        actual = verify.integrate_section(**shifted)
        for index, (a, b) in enumerate(zip(actual["modes"], expected["modes"])):
            np.testing.assert_allclose(a["component_fractions_xyz"], b["component_fractions_xyz"], atol=1e-13)
            np.testing.assert_allclose(a["template_scores_tx_ty_tz_rx"], b["template_scores_tx_ty_tz_rx"], atol=1e-13)
            self.assertAlmostEqual(a["displacement_mass_norm"] / b["displacement_mass_norm"], abs(factors[index])**2,
                                   delta=1e-10 * abs(factors[index])**2)

    def test_rejects_component_and_coordinate_map_tampering(self):
        fixture = rectangle_fixture()
        fixture["component_dofs_xyz"][0, 0], fixture["component_dofs_xyz"][1, 1] = (
            fixture["component_dofs_xyz"][1, 1], fixture["component_dofs_xyz"][0, 0])
        with self.assertRaisesRegex(ValueError, "coordinate map"):
            verify.integrate_section(**fixture)
        fixture = rectangle_fixture()
        fixture["coordinates_yz_m"][0, 0] += 1e-8
        with self.assertRaisesRegex(ValueError, "coordinate map"):
            verify.integrate_section(**fixture)
        fixture = rectangle_fixture()
        fixture["component_dofs_xyz"][0, 0] = fixture["component_dofs_xyz"][1, 0]
        with self.assertRaisesRegex(ValueError, "component map"):
            verify.integrate_section(**fixture)

    def test_rejects_score_gram_and_remainder_tampering(self):
        expected = verify.integrate_section(**rectangle_fixture())
        for field in ("component_fractions_xyz", "template_scores_tx_ty_tz_rx", "template_remainder_fraction"):
            bad = deepcopy(expected)
            if isinstance(bad["modes"][0][field], list):
                bad["modes"][0][field][0] += 1e-5
            else:
                bad["modes"][0][field] += 1e-5
            with self.assertRaisesRegex(ValueError, field):
                verify.compare_metrics(bad, expected)
        bad = deepcopy(expected)
        bad["normalized_template_gram_real"][0][0] += 1e-5
        with self.assertRaisesRegex(ValueError, "Gram|gram"):
            verify.compare_metrics(bad, expected)

    def test_independent_interpolation_recovers_permuted_complex_mode_matching(self):
        source, target = rectangle_fixture(), rectangle_fixture()
        source["displacements"] = source["displacements"][:, :4]
        permutation = [2, 0, 3, 1]
        target["displacements"] = target["displacements"][:, permutation] * [1j, -2, .3j, -4j]
        matched, overlaps = verify.independent_mode_match(source, target)
        np.testing.assert_array_equal(matched, np.argsort(permutation))
        np.testing.assert_allclose(overlaps, 1, atol=1e-14)
        target["displacements"][:, 1] = target["displacements"][:, 0]
        with self.assertRaisesRegex(ValueError, "ambiguous"):
            verify.independent_mode_match(source, target)

    def test_input_manifest_must_match_accepted_provenance(self):
        self.assertTrue(all(len(value) == 64 and set(value) <= set("0123456789abcdef")
                            for value in (*verify.LEDGER_HASHES.values(), *verify.PINNED_METADATA_HASHES.values())))
        with TemporaryDirectory() as directory:
            root = Path(directory).resolve()
            source, ledger = root / "source.csv", root / "ledger.json"
            source.write_text("accepted saved response\n", encoding="utf-8")
            ledger.write_text(json.dumps({"accepted": {"sha256": verify.sha256(source)}}), encoding="utf-8")
            document = {"inputs": [{"role": "response", "path": "source.csv", "bytes": source.stat().st_size,
                                    "sha256": verify.sha256(source)}]}
            with patch.dict(verify.LEDGER_HASHES, {"ledger.json": verify.sha256(ledger)}, clear=True):
                self.assertEqual(verify.verified_inputs(document, root)["response"], source)
                source.write_text("unaccepted new response\n", encoding="utf-8")
                document["inputs"][0].update(bytes=source.stat().st_size, sha256=verify.sha256(source))
                with self.assertRaisesRegex(ValueError, "accepted identity"):
                    verify.verified_inputs(document, root)

    def test_response_join_rejects_wrong_root_and_nonfinite_power(self):
        base = verify.integrate_section(**rectangle_fixture())["modes"][:4]
        sections = {name: {"modes": [dict(mode, root=root) for mode, root in zip(base, roots)]}
                    for name, roots in (("z4", [1348, 1346, 1344, 1294]),
                                        ("z6", [1948, 1916, 1940, 1873]))}
        responses, roles, records = {}, {}, {}
        with TemporaryDirectory() as directory:
            root = Path(directory).resolve()
            for name, section in (("h100_z4", "z4"), ("h80_z4", "z4"), ("h80_z6", "z6")):
                path = root / f"{name}.csv"
                with path.open("w", newline="", encoding="utf-8") as stream:
                    writer = csv.writer(stream)
                    writer.writerow(("incident_mode_order", "frequency_GHz", "reflection", "transmission"))
                    writer.writerows((index + 1, .1, .02, .98) for index in range(4))
                roles[f"{name}_scattering"] = path
                output_sha = {"scattering_AP_pnc.csv": verify.sha256(path)}
                summary = {"frequency_hz": 1e8, "incident_side": "left", "num_incident_channels": 4,
                           "incident_mode_indices": [mode["root"] for mode in sections[section]["modes"]],
                           "power_balance_passed": True, "run_hash": name, "output_sha256": output_sha}
                summary_path = root / f"{name}_summary.json"
                summary_path.write_text(json.dumps(summary), encoding="utf-8")
                roles[f"{name}_summary"] = summary_path
                records[name] = {"run_hash": name, "output_sha256": output_sha}
                responses[name] = {"run_hash": name, "rows": [dict(mode, R=.02, T=.98, reverse_root=mode["root"] + 1)
                                                                for mode in sections[section]["modes"]]}
            self.assertEqual(verify.verify_response_rows(responses, sections, roles, records)["h80_z6"]["checked_rows"], 4)
            for field, value in (("root", 1948), ("reverse_root", 4), ("R", float("nan")), ("R", .03)):
                altered = deepcopy(responses)
                altered["h100_z4"]["rows"][0][field] = value
                with self.assertRaisesRegex(ValueError, "signed-root/response"):
                    verify.verify_response_rows(altered, sections, roles, records)
            summary = json.loads(roles["h100_z4_summary"].read_bytes())
            summary["incident_mode_indices"] = summary["incident_mode_indices"][::-1]
            roles["h100_z4_summary"].write_text(json.dumps(summary), encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "summary/root/run/CSV"):
                verify.verify_response_rows(responses, sections, roles, records)

    def test_accepted_artifacts_cannot_be_swapped_between_scientific_roles(self):
        with TemporaryDirectory() as directory:
            root, roles = Path(directory).resolve(), {}

            def write(role, value, relative=None):
                path = root / (relative or f"{role}.json")
                path.parent.mkdir(parents=True, exist_ok=True)
                path.write_text(json.dumps(value), encoding="utf-8")
                roles[role] = path
                return {"sha256": verify.sha256(path), "bytes": path.stat().st_size}

            links = {role: write(role, {"unique_archive": role})
                     for role in ("z4_section", "z4_modes", "z6_section", "z6_modes")}
            z4_safe = write("z4_safe_receipt", {"safe_section": links["z4_section"], "safe_modes": links["z4_modes"]})
            write("z6_safe_receipt", {"unique_receipt": "z6"})
            records = {}
            for point in ("h100_z4", "h80_z4", "h80_z6"):
                csv_link = write(f"{point}_scattering", {"unique_csv": point})
                summary_link = write(f"{point}_summary", {"unique_summary": point})
                records[point] = {"run_hash": point, "summary_sha256": summary_link["sha256"],
                                  "output_sha256": {"scattering_AP_pnc.csv": csv_link["sha256"]}}
            write("inplane_comparison", {"reference": records["h100_z4"], "candidate": records["h80_z4"]})
            write("thickness_comparison", {"reference": records["h80_z4"], "candidate": records["h80_z6"]})
            ledgers = {"low_band_inplane_verification": {"independent_replay": {"accepted_shared_safe": z4_safe}},
                       "low_band_thickness_verification": {},
                       "low_band_device_verification": {"scientific_inputs": {"accepted_safe_modes": links["z6_modes"]}},
                       "port_inventory_verification": {"archived_section": {"mode_archive": links["z6_section"]}}}
            ledger_paths = dict(zip(ledgers, verify.LEDGER_HASHES))
            for role, content in ledgers.items():
                write(role, content, ledger_paths[role])
            ledger_pins = {name: verify.sha256(root / name) for name in ledger_paths.values()}
            metadata_pins = {role: verify.sha256(roles[role]) for role in verify.PINNED_METADATA_HASHES}
            with (patch.dict(verify.LEDGER_HASHES, ledger_pins, clear=True),
                  patch.dict(verify.PINNED_METADATA_HASHES, metadata_pins, clear=True)):
                self.assertEqual(verify.bind_scientific_roles(roles, root), records)
                swapped = dict(roles)
                swapped["h100_z4_scattering"], swapped["h80_z4_scattering"] = (
                    swapped["h80_z4_scattering"], swapped["h100_z4_scattering"])
                with self.assertRaisesRegex(ValueError, "scientific role identity"):
                    verify.bind_scientific_roles(swapped, root)


if __name__ == "__main__":
    unittest.main()
